"""
Termination conditions for the reorient task -- Phase C.

Donor (`_compute_resets` in allegro_kuka_two_arms.py) has three reset triggers:

    drop          object_z < drop_threshold (donor uses 0.1, we use a
                  configurable cfg.object_drop_z derived from table_top_z)
    max_consec    successes >= max_consecutive_successes (50 in donor)
    timeout       progress_buf >= max_episode_length - 1

The donor merges all three into a single `reset_buf`. IsaacLab DirectRLEnv
splits them into (terminated, time_out) where terminated covers "real"
terminations (drop, max-consec) and time_out covers episode-length expiry.
This split lets the value bootstrapping in PPO treat truncations correctly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .env import ReorientEnv


def compute_dones(env: "ReorientEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (terminated, time_out), each shape (num_envs,) bool.

    `terminated` = the agent failed (dropped) or succeeded enough times to
                   trigger a reset (max_consecutive_successes).
    `time_out`   = the episode length expired without a terminal event.
    """
    # ── Drop ─────────────────────────────────────────────────────────────────
    object_z = env.cuboid.data.root_pos_w[:, 2]
    dropped = object_z < env.cfg.object_drop_z

    # ── Max consecutive successes ────────────────────────────────────────────
    # env.successes is updated in env._get_dones BEFORE this function is called.
    max_consec = env.successes >= env.cfg.max_consecutive_successes

    # ── Timeout ──────────────────────────────────────────────────────────────
    time_out = env.episode_length_buf >= env.max_episode_length - 1

    terminated = dropped | max_consec
    return terminated, time_out