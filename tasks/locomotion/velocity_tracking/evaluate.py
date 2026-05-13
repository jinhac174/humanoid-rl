"""Velocity-tracking evaluator.

Adds gait-relevant metrics on top of :class:`evaluators.base.BaseEvaluator`:

    * Per-episode commanded vs. achieved velocity error (XY + yaw).
    * Mean foot air time over the episode.

For the first iteration we deliberately do NOT pin a fixed command schedule
in eval — the env continues to resample commands every
``cmd_resampling_time_s`` so the recorded video shows the policy reacting to
multiple commands. To pin a single command for a stress test, set
``cfg.task.eval.fixed_command: [vx, vy, wz]`` (consumed below if present).
"""
from __future__ import annotations

import torch

from evaluators.base import BaseEvaluator


class VelocityTrackingEvaluator(BaseEvaluator):

    def __init__(self, cfg):
        super().__init__(cfg)
        eval_block = getattr(cfg.task, "eval", None) or {}
        # Optional: clamp the command to a fixed (vx, vy, wz) for the entire eval.
        self._fixed_command = eval_block.get("fixed_command", None)
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
