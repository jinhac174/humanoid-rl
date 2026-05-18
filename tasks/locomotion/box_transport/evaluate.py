"""Box-transport evaluator.

Adds on top of :class:`evaluators.base.BaseEvaluator`:

    1. Per-episode milestone counters (bimanual_touched, lifted, placed
       success-frames) printed in the one-line episode summary.
    2. Camera FOLLOW: cfg.task.cameras[*].eye/.lookat are interpreted as
       OFFSETS from the robot base each frame (Z absolute), so the video
       stays framed on the robot+box as the agent walks.

We deliberately do NOT pin a fixed command schedule — the env auto-derives
the velocity command from the task phase each step, so the policy reacts
to the live scene.
"""
from __future__ import annotations

import numpy as np

from evaluators.base import BaseEvaluator
from evaluators.utils import grab_frame


class BoxTransportEvaluator(BaseEvaluator):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._reset_episode_stats()

    def _reset_episode_stats(self) -> None:
        self._touched = False
        self._lifted  = False
        self._max_success_steps = 0
        self._final_dist_to_target = float("nan")

    def on_episode_start(self, ep: int) -> None:
        self._reset_episode_stats()

    def on_step(self, step: int, info: dict) -> None:
        env = self.unwrapped
        # Read the latched milestone flags from env 0.
        self._touched = bool(env._bimanual_contact_achieved[0].item())
        self._lifted  = bool(env._lift_achieved[0].item())
        ss = int(env._success_step_count[0].item())
        if ss > self._max_success_steps:
            self._max_success_steps = ss
        # Track final distance to target (xy) for the summary line.
        box_xy = env.box.data.root_pos_w[0, :2]
        tgt_xy = env._target_pos_w[0, :2]
        self._final_dist_to_target = float((box_xy - tgt_xy).norm().item())

    def episode_summary(self, ep: int, steps: int, total_reward: float) -> str:
        return (
            f"  ep{ep:03d} | steps={steps:4d} | reward={total_reward:8.2f} "
            f"| touched={int(self._touched)} lifted={int(self._lifted)} "
            f"| max_place_steps={self._max_success_steps:3d} "
            f"| final_dist_to_target={self._final_dist_to_target:.3f} m"
        )

    def _record_frame(self, writers: dict) -> None:
        """Tracking-camera variant — same pattern as velocity_tracking."""
        env = self.unwrapped
        base_pos = env.robot.data.root_pos_w[0].detach().cpu().numpy()
        anchor = np.array([base_pos[0], base_pos[1], 0.0], dtype=np.float64)

        if self.use_raytracing:
            for _ in range(3):
                self.env.render()
        for cam_name, cam_cfg in self.cameras.items():
            eye    = (np.array(list(cam_cfg.eye),    dtype=np.float64) + anchor).tolist()
            lookat = (np.array(list(cam_cfg.lookat), dtype=np.float64) + anchor).tolist()
            self.sim.set_camera_view(eye=eye, target=lookat)
            writers[cam_name].append_data(grab_frame(self.env))
