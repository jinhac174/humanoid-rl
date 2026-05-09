"""Reorient-task evaluator.

Adds two pieces of task logic on top of :class:`evaluators.base.BaseEvaluator`:

    1. Goal-respawn cooldown. Reorient counts a "success" each time the policy
       holds the cube near the target pose for ``success_steps`` frames, and
       the env queues a fresh goal pose on the next ``_pre_physics_step``.
       Without a cooldown the goal cube can strobe when the policy hovers
       near the boundary; we suppress further respawn triggers for
       ``hold_frames`` after each accepted success so the video reads clearly.

    2. ``max_consecutive_successes`` is effectively disabled at eval. Training
       resets episodes after 50 successes, but for video we want the policy
       to keep going for the full ``episode_length_s`` window. This is set
       via the task yaml's ``eval:`` block (consumed by ``apply_eval_overrides``).
"""
from __future__ import annotations

from evaluators.base import BaseEvaluator


class ReorientEvaluator(BaseEvaluator):
    """Reorient eval with success counting + goal-respawn cooldown."""

    def __init__(self, cfg):
        super().__init__(cfg)
        # Cache hold_frames once; the task yaml's eval block carries it.
        eval_block = getattr(cfg.task, "eval", None) or {}
        self._hold_frames = int(eval_block.get("hold_frames", 90))
        self._current_ep = 0
        self._successes = 0
        self._respawn_cooldown = 0

    def on_episode_start(self, ep: int) -> None:
        self._current_ep = ep
        self._successes = 0
        self._respawn_cooldown = 0

    def on_step(self, step: int, info: dict) -> None:
        unwrapped = self.unwrapped
        # ``reset_goal_buf`` is set by env._get_dones the moment the keypoint
        # max distance dips under tolerance for ``success_steps`` frames. The
        # env consumes this flag on the NEXT _pre_physics_step to resample the
        # goal. We let the first such trigger after a cooldown propagate
        # (count the success, start a new cooldown) and clear any further
        # triggers fired during the cooldown window.
        if unwrapped.reset_goal_buf[0].item():
            if self._respawn_cooldown == 0:
                self._successes += 1
                self._respawn_cooldown = self._hold_frames
                print(
                    f"    [ep{self._current_ep:03d}] respawn "
                    f"#{self._successes} at step {step}"
                )
            else:
                # Cooldown active — drop this respawn trigger; env will skip it.
                unwrapped.reset_goal_buf[0] = False

        if self._respawn_cooldown > 0:
            self._respawn_cooldown -= 1

    def episode_summary(self, ep: int, steps: int, total_reward: float) -> str:
        return (
            f"  ep{ep:03d} | steps={steps:4d} | reward={total_reward:.2f} "
            f"| successes={self._successes}"
        )
