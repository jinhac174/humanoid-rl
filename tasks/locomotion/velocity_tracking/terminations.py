"""Termination conditions.

Two reasons an episode ends:

    terminated  — illegal contact on torso (the robot crashed). The reward
                  applies the ``pen_termination`` (-200 by default) on this
                  step via ``mdp.is_terminated``.

    time_out    — episode_length_buf hit max_episode_length-1 without
                  termination. Truncation, no penalty.

Returned as ``(terminated, time_out)`` to match DirectRLEnv's convention.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .env import VelocityTrackingEnv


def compute_dones(env: "VelocityTrackingEnv") -> tuple[torch.Tensor, torch.Tensor]:
    # Illegal contact on torso: any history-frame's force magnitude above
    # the threshold counts. The contact sensor stores forces at
    # (N, history, num_bodies, 3). The standard "max over history" is what
    # IsaacLab's mdp.illegal_contact uses.
    forces_history = env.contact_sensor.data.net_forces_w_history  # (N, T, B, 3)
    if forces_history is None or forces_history.numel() == 0:
        # First step before sensor populated — no termination.
        terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        torso_forces = forces_history[:, :, env._torso_sensor_ids, :]   # (N, T, len, 3)
        torso_force_mag = torch.norm(torso_forces, dim=-1)              # (N, T, len)
        max_force = torso_force_mag.amax(dim=(1, 2))                    # (N,)
        terminated = max_force > env.cfg.illegal_contact_threshold

    time_out = env.episode_length_buf >= env.max_episode_length - 1

    return terminated, time_out
