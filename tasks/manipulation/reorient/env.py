# Reorient task -- DirectRLEnv
from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from assets.robots.g1_cfg import ACTION_SCALE, ACTUATED_JOINTS

from .env_cfg import (
    CUBOID_SIZE,
    LEFT_FINGERTIP_BODIES,
    LEFT_PALM_BODY,
    NUM_ARMS,
    NUM_FINGERTIPS_PER_HAND,
    RIGHT_FINGERTIP_BODIES,
    RIGHT_PALM_BODY,
    ReorientEnvCfg,
)
from . import events as event_fn
from . import observations as obs_fn
from . import rewards as rew_fn
from . import terminations as term_fn


class ReorientEnv(DirectRLEnv):
    cfg: ReorientEnvCfg

    def __init__(
        self,
        cfg: ReorientEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # -- Asset handles --
        self.robot: Articulation = self.scene["robot"]
        self.cuboid: RigidObject = self.scene["cuboid"]
        self.goal: RigidObject = self.scene["goal"]

        # -- Actuated joint IDs (28 total) --
        self.actuated_joint_ids, resolved_names = self.robot.find_joints(
            ACTUATED_JOINTS, preserve_order=True
        )
        assert len(self.actuated_joint_ids) == 28, (
            f"expected 28 actuated joints, got {len(self.actuated_joint_ids)}: "
            f"{resolved_names}"
        )

        # -- Per-joint action scale (delta control) --
        self.action_scale = torch.tensor(
            ACTION_SCALE, dtype=torch.float32, device=self.device
        )

        # -- Body IDs --
        self.left_palm_body_id = self.robot.find_bodies(LEFT_PALM_BODY)[0][0]
        self.right_palm_body_id = self.robot.find_bodies(RIGHT_PALM_BODY)[0][0]

        left_fingertip_ids = [
            self.robot.find_bodies(name)[0][0] for name in LEFT_FINGERTIP_BODIES
        ]
        right_fingertip_ids = [
            self.robot.find_bodies(name)[0][0] for name in RIGHT_FINGERTIP_BODIES
        ]
        self.fingertip_body_ids = torch.tensor(
            [left_fingertip_ids, right_fingertip_ids],
            dtype=torch.long,
            device=self.device,
        )

        # -- Target volume tensors --
        self.target_volume_origin = torch.tensor(
            self.cfg.target_volume_origin, dtype=torch.float32, device=self.device
        )
        self.target_volume_min = torch.tensor(
            [e[0] for e in self.cfg.target_volume_extent],
            dtype=torch.float32,
            device=self.device,
        )
        self.target_volume_max = torch.tensor(
            [e[1] for e in self.cfg.target_volume_extent],
            dtype=torch.float32,
            device=self.device,
        )

        # -- Delta-control joint target buffer --
        self.joint_targets = self.robot.data.default_joint_pos[
            :, self.actuated_joint_ids
        ].clone()

        # -- Task state buffers --
        self.lifted_object = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.near_goal_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.successes = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.closest_fingertip_dist = -torch.ones(
            self.num_envs,
            NUM_ARMS,
            NUM_FINGERTIPS_PER_HAND,
            dtype=torch.float32,
            device=self.device,
        )
        self.closest_keypoint_max_dist = -torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        # -- Goal-reset queue and per-step near-goal flag --
        self.reset_goal_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.near_goal = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # -- Per-step task state (updated in observations.compute_task_state) --
        self.object_init_pos_w = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self.palm_center_pos = torch.zeros(
            self.num_envs, NUM_ARMS, 3, dtype=torch.float32, device=self.device
        )
        self.fingertip_pos = torch.zeros(
            self.num_envs,
            NUM_ARMS,
            NUM_FINGERTIPS_PER_HAND,
            3,
            dtype=torch.float32,
            device=self.device,
        )
        self.curr_fingertip_distances = torch.zeros(
            self.num_envs,
            NUM_ARMS,
            NUM_FINGERTIPS_PER_HAND,
            dtype=torch.float32,
            device=self.device,
        )
        # -- Keypoint offsets (configurable count via cfg.num_keypoints) --
        # 8: all cube corners, the symmetric SO(3) default and what we train
        #    and evaluate against.
        # 4: tetrahedral subset (donor SAPG convention, ablation only).
        self.num_keypoints = int(self.cfg.num_keypoints)
        _kp_corners = self._build_keypoint_corners(self.num_keypoints).to(self.device)
        self.keypoint_offsets = _kp_corners * (
            CUBOID_SIZE[0] * self.cfg.keypoint_scale / 2.0
        )

        self.obj_keypoint_pos = torch.zeros(
            self.num_envs, self.num_keypoints, 3, dtype=torch.float32, device=self.device
        )
        self.goal_keypoint_pos = torch.zeros(
            self.num_envs, self.num_keypoints, 3, dtype=torch.float32, device=self.device
        )
        self.keypoints_max_dist = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        # -- Curriculum state --
        self._frame_count = 0
        self._last_curriculum_update = 0
        self._prev_episode_successes = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._initial_success_tolerance = self.cfg.success_tolerance
        self._print_reach_envelope_once()

    # -----------------------------------------------------------------
    # Keypoint corners
    # -----------------------------------------------------------------
    @staticmethod
    def _build_keypoint_corners(num_keypoints: int) -> torch.Tensor:
        """Return ``(K, 3)`` unit-cube corner offsets in ±1 form.

        K=8: all 8 corners, the symmetric default for SO(3) targets.
        K=4: tetrahedral diagonal subset (the SAPG donor convention).
        """
        if num_keypoints == 8:
            corners = torch.tensor(
                [
                    [+1, +1, +1], [+1, +1, -1], [+1, -1, +1], [+1, -1, -1],
                    [-1, +1, +1], [-1, +1, -1], [-1, -1, +1], [-1, -1, -1],
                ],
                dtype=torch.float32,
            )
        elif num_keypoints == 4:
            corners = torch.tensor(
                [
                    [+1, +1, +1], [+1, +1, -1],
                    [-1, -1, +1], [-1, -1, -1],
                ],
                dtype=torch.float32,
            )
        else:
            raise ValueError(
                f"num_keypoints must be 4 or 8, got {num_keypoints}"
            )
        return corners

    # -----------------------------------------------------------------
    # Step pipeline
    # -----------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        goal_reset_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if goal_reset_ids.numel() > 0:
            event_fn.reset_goal_only(self, goal_reset_ids)
            self.reset_goal_buf[goal_reset_ids] = False

        self.actions = actions.clamp(-1.0, 1.0)
        self.joint_targets = self.joint_targets + self.actions * self.action_scale
        lo = self.robot.data.soft_joint_pos_limits[:, self.actuated_joint_ids, 0]
        hi = self.robot.data.soft_joint_pos_limits[:, self.actuated_joint_ids, 1]
        self.joint_targets = self.joint_targets.clamp(lo, hi)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(
            self.joint_targets,
            joint_ids=self.actuated_joint_ids,
        )

    def _get_observations(self) -> dict:
        obs_fn.compute_task_state(self)
        return {"policy": obs_fn.get_observations(self)}

    def _get_rewards(self) -> torch.Tensor:
        return rew_fn.compute_reward(self)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # No need to call compute_task_state again -- _get_observations ran it
        # earlier in the same step and no physics has happened since, so
        # env.keypoints_max_dist is already current.
        keypoint_success_tol = self.cfg.success_tolerance * self.cfg.keypoint_scale
        self.near_goal = self.keypoints_max_dist <= keypoint_success_tol
        self.near_goal_steps = self.near_goal_steps + self.near_goal.int()
        is_success = self.near_goal_steps >= self.cfg.success_steps

        self.successes = self.successes + is_success.float()
        self.reset_goal_buf = self.reset_goal_buf | is_success

        self.near_goal_steps = torch.where(
            is_success, torch.zeros_like(self.near_goal_steps), self.near_goal_steps
        )
        self.closest_keypoint_max_dist = torch.where(
            is_success,
            -torch.ones_like(self.closest_keypoint_max_dist),
            self.closest_keypoint_max_dist,
        )

        # ``task_episode_success_per_env`` is consumed by the trainer's
        # episode tracker (gates "did this episode succeed?"). Episodes end on
        # drop / timeout / max_consecutive_successes, NOT on a single hit of
        # the goal — so reporting per-step ``is_success`` here would be near
        # zero at episode termination. Use the cumulative ``self.successes``
        # which counts all goal hits within the current (still-running)
        # episode; the trainer reads it on the step the env terminates,
        # right before ``_reset_idx`` zeros the buffer.
        self.extras["task_episode_success_per_env"] = (self.successes > 0).float()

        self._update_curriculum()

        return term_fn.compute_dones(self)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        # Save episode successes before clearing for curriculum
        self._prev_episode_successes[env_ids] = self.successes[env_ids]
        event_fn.reset_robot(self, env_ids)
        event_fn.reset_objects(self, env_ids)
        event_fn.reset_buffers(self, env_ids)

    # -----------------------------------------------------------------
    # Curriculum
    # -----------------------------------------------------------------
    def _update_curriculum(self):
        """Smooth exponential decay of success_tolerance.
        No jumps, no triggers. Pure function of training progress.
        Warmup: tolerance stays at initial value for first 10k iterations.
        Then exponentially decays to target over next 80k iterations."""
        if not self.cfg.enable_curriculum:
            return
        self._frame_count += 1

        warmup = self.cfg.curriculum_warmup_frames
        total = self.cfg.curriculum_total_frames

        if self._frame_count < warmup:
            return

        progress = min((self._frame_count - warmup) / max(total - warmup, 1), 1.0)

        start_tol = self._initial_success_tolerance
        target_tol = self.cfg.target_success_tolerance

        import math
        new_tol = start_tol * (target_tol / start_tol) ** progress
        self.cfg.success_tolerance = new_tol

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------
    def _print_reach_envelope_once(self) -> None:
        left_palm_w = self.robot.data.body_pos_w[0, self.left_palm_body_id, :]
        right_palm_w = self.robot.data.body_pos_w[0, self.right_palm_body_id, :]
        robot_root_w = self.robot.data.root_pos_w[0, :]

        origin = self.target_volume_origin
        extent_min = self.target_volume_min
        extent_max = self.target_volume_max
        tv_min = origin + extent_min
        tv_max = origin + extent_max

        corners = torch.tensor(
            [
                [tv_min[0], tv_min[1], tv_min[2]],
                [tv_min[0], tv_min[1], tv_max[2]],
                [tv_min[0], tv_max[1], tv_min[2]],
                [tv_min[0], tv_max[1], tv_max[2]],
                [tv_max[0], tv_min[1], tv_min[2]],
                [tv_max[0], tv_min[1], tv_max[2]],
                [tv_max[0], tv_max[1], tv_min[2]],
                [tv_max[0], tv_max[1], tv_max[2]],
            ],
            device=self.device,
        )

        left_dists = torch.norm(corners - left_palm_w.unsqueeze(0), dim=-1)
        right_dists = torch.norm(corners - right_palm_w.unsqueeze(0), dim=-1)

        print("=" * 72)
        print("REORIENT -- REACH ENVELOPE CHECK (env 0)")
        print("=" * 72)
        print(f"  robot root pos:   {robot_root_w.tolist()}")
        print(f"  left  palm pos:   {left_palm_w.tolist()}")
        print(f"  right palm pos:   {right_palm_w.tolist()}")
        print(f"  table top z:      {self.cfg.table_top_z:.3f}")
        print(f"  cuboid spawn:     {self.cfg.cuboid_spawn_pos}")
        print(f"  target vol min:   {tv_min.tolist()}")
        print(f"  target vol max:   {tv_max.tolist()}")
        print(
            f"  left  palm -> TV corners:  "
            f"min={left_dists.min().item():.3f}  max={left_dists.max().item():.3f}"
        )
        print(
            f"  right palm -> TV corners:  "
            f"min={right_dists.min().item():.3f}  max={right_dists.max().item():.3f}"
        )
        print("  (G1 arm reach is ~0.65 m from shoulder; both palms should")
        print("   have max distance < ~0.70 m to the furthest corner)")
        print("=" * 72)