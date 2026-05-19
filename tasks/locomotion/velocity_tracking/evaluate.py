"""Velocity-tracking evaluator.

Adds four pieces of locomotion-specific behaviour on top of
:class:`evaluators.base.BaseEvaluator`:

    1. Per-episode commanded vs. achieved velocity error (XY + yaw).
    2. Mean foot air time over the episode.
    3. Camera FOLLOW: cfg.task.cameras[*].eye / .lookat are interpreted as
       OFFSETS from the robot base each frame, so the video tracks the
       walking robot instead of letting it stride out of frame. This is the
       override that makes the locomotion videos watchable.
    4. Velocity arrows in the rendered MP4 — GREEN = commanded (xy lin),
       BLUE = current (xy lin). Direct port of IsaacLab
       ``UniformVelocityCommand._debug_vis_callback`` /
       ``_resolve_xy_velocity_to_arrow`` (the visualization the official
       Isaac-Velocity-Flat-G1-v0 task draws). Math, anchor offset, default
       marker scale, and the scale * norm * 3 amplification all mirror
       IsaacLab exactly so the videos look like the upstream debug viewer.

For the first iteration we deliberately do NOT pin a fixed command schedule
in eval — the env keeps resampling commands every ``cmd_resampling_time_s``
so the recorded video shows the policy reacting to multiple commands. To pin
one command for a stress test, set ``cfg.task.eval.fixed_command: [vx, vy, wz]``.
"""
from __future__ import annotations

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)

from evaluators.base import BaseEvaluator
from evaluators.utils import grab_frame

# Vertical offset (m) applied to the arrows above the robot's base. Matches
# IsaacLab's UniformVelocityCommand._debug_vis_callback exactly (line 191 of
# isaaclab/envs/mdp/commands/velocity_command.py). For G1 with pelvis at
# ~0.76 m world, this puts the arrows at ~1.26 m — chest/throat level.
_ARROW_Z_OFFSET = 0.5

# Arrow base scale (sx, sy, sz). Per-frame, sx is multiplied by
# ``norm(xy_vel) * 3.0`` so a stationary command renders a zero-length
# arrow and a 1 m/s command renders a 1.5 m arrow. Matches the override
# in UniformVelocityCommandCfg (commands_cfg.py lines 94-95).
_ARROW_BASE_SCALE = (0.5, 0.5, 0.5)


class VelocityTrackingEvaluator(BaseEvaluator):

    def __init__(self, cfg):
        super().__init__(cfg)
        eval_block = getattr(cfg.task, "eval", None) or {}
        # Optional: clamp the command to a fixed (vx, vy, wz) for the entire eval.
        self._fixed_command = eval_block.get("fixed_command", None)
        # Lazy-built on the first frame after the stage is live.
        self._goal_vel_vis: VisualizationMarkers | None = None
        self._cur_vel_vis: VisualizationMarkers | None = None
        self._reset_episode_stats()

    def _reset_episode_stats(self) -> None:
        self._lin_err_sum = 0.0
        self._ang_err_sum = 0.0
        self._airtime_sum = 0.0
        self._n_steps = 0

    def on_episode_start(self, ep: int) -> None:
        self._reset_episode_stats()
        # If the user pinned a command, write it once and we'll re-pin every
        # step to defeat the env's resampling timer.
        if self._fixed_command is not None:
            cmd = torch.tensor(
                self._fixed_command, dtype=torch.float32, device=self.unwrapped.device
            )
            self.unwrapped._commands_dict["base_velocity"][:] = cmd

    def on_step(self, step: int, info: dict) -> None:
        env = self.unwrapped

        # Re-pin fixed command if requested (overrides env's auto-resample).
        if self._fixed_command is not None:
            cmd = torch.tensor(
                self._fixed_command, dtype=torch.float32, device=env.device
            )
            env._commands_dict["base_velocity"][:] = cmd

        cmd = env._commands_dict["base_velocity"][0]              # (3,)
        ach = env.robot.data.root_lin_vel_b[0]                     # (3,)
        ang = env.robot.data.root_ang_vel_b[0]                     # (3,)
        self._lin_err_sum += float(((cmd[:2] - ach[:2]) ** 2).sum().sqrt().item())
        self._ang_err_sum += float(abs(cmd[2].item() - ang[2].item()))

        air_time = env.contact_sensor.data.current_air_time
        if air_time is not None and air_time.numel() > 0:
            self._airtime_sum += float(air_time[0, env._feet_sensor_ids].mean().item())

        self._n_steps += 1

    def episode_summary(self, ep: int, steps: int, total_reward: float) -> str:
        n = max(self._n_steps, 1)
        return (
            f"  ep{ep:03d} | steps={steps:4d} | reward={total_reward:7.2f} "
            f"| lin_err={self._lin_err_sum / n:.3f} m/s "
            f"| ang_err={self._ang_err_sum / n:.3f} rad/s "
            f"| mean_airtime={self._airtime_sum / n:.3f} s"
        )

    # ------------------------------------------------------------------
    # Arrow visualization (port of IsaacLab UniformVelocityCommand debug viz)
    # ------------------------------------------------------------------
    def _ensure_velocity_markers(self) -> None:
        """First-frame lazy-init of the goal/current velocity arrows.

        Markers can't be created until the USD stage is live, which is
        guaranteed by the time ``_record_frame`` first runs (after
        ``BaseEvaluator._prime_renderer``).
        """
        if self._goal_vel_vis is not None:
            return
        goal_cfg = GREEN_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Eval/velocity_goal"
        )
        cur_cfg = BLUE_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Eval/velocity_current"
        )
        # Match the override IsaacLab's UniformVelocityCommandCfg performs on
        # both cfgs (commands_cfg.py:94-95).
        goal_cfg.markers["arrow"].scale = _ARROW_BASE_SCALE
        cur_cfg.markers["arrow"].scale = _ARROW_BASE_SCALE
        self._goal_vel_vis = VisualizationMarkers(goal_cfg)
        self._cur_vel_vis = VisualizationMarkers(cur_cfg)
        self._goal_vel_vis.set_visibility(True)
        self._cur_vel_vis.set_visibility(True)

    def _resolve_xy_velocity_to_arrow(
        self,
        xy_velocity: torch.Tensor,
        visualizer: VisualizationMarkers,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Port of UniformVelocityCommand._resolve_xy_velocity_to_arrow.

        Returns ``(scales, world_quats)`` for ``VisualizationMarkers.visualize``.
        ``xy_velocity`` is in the robot's base frame (both the commanded
        vel and ``root_lin_vel_b`` are base-frame); the arrow is yawed by
        ``atan2(vy, vx)`` in the base frame and then rotated into world by
        ``base_quat_w``.
        """
        env = self.unwrapped
        default_scale = visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(
            default_scale, device=env.device
        ).repeat(xy_velocity.shape[0], 1)
        # Stretch along arrow's x by ``norm * 3``. Zero velocity → zero-length.
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = env.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat

    def _update_velocity_markers(self) -> None:
        """Place both arrows for the current frame."""
        self._ensure_velocity_markers()
        env = self.unwrapped
        base_pos_w = env.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += _ARROW_Z_OFFSET

        goal_scale, goal_quat = self._resolve_xy_velocity_to_arrow(
            env._commands_dict["base_velocity"][:, :2], self._goal_vel_vis,
        )
        cur_scale, cur_quat = self._resolve_xy_velocity_to_arrow(
            env.robot.data.root_lin_vel_b[:, :2], self._cur_vel_vis,
        )
        self._goal_vel_vis.visualize(base_pos_w, goal_quat, goal_scale)
        self._cur_vel_vis.visualize(base_pos_w, cur_quat, cur_scale)

    def _record_frame(self, writers: dict) -> None:
        """Tracking-camera variant: cam.eye / cam.lookat are robot-relative.

        Each frame, the camera is re-anchored to the robot's current world
        XY position (Z is kept absolute so the camera doesn't bob with the
        torso). This is what keeps the policy in view as it walks across
        the plane. The velocity arrows are also updated each frame so the
        viewer sees both the commanded heading (green) and current xy
        velocity (blue) in world frame above the robot.
        """
        env = self.unwrapped
        base_pos = env.robot.data.root_pos_w[0].detach().cpu().numpy()
        # Anchor in XY only — leave Z to cfg.eye / cfg.lookat absolute heights.
        anchor = np.array([base_pos[0], base_pos[1], 0.0], dtype=np.float64)

        # Arrows must be updated BEFORE any render passes pick them up.
        self._update_velocity_markers()

        if self.use_raytracing:
            for _ in range(3):
                self.env.render()
        for cam_name, cam_cfg in self.cameras.items():
            eye    = (np.array(list(cam_cfg.eye),    dtype=np.float64) + anchor).tolist()
            lookat = (np.array(list(cam_cfg.lookat), dtype=np.float64) + anchor).tolist()
            self.sim.set_camera_view(eye=eye, target=lookat)
            writers[cam_name].append_data(grab_frame(self.env))
