"""Termination conditions.

Two reasons an episode ends:

    terminated  — the robot fell. Detected by THREE OR'd signals so it fires
                  regardless of *how* the robot goes down:

                    1. base height  : root_pos_w.z below ``termination_base_height``
                                      (bulletproof — no contact sensor needed;
                                      a pelvis below ~0.4 m is unambiguously down)
                    2. orientation  : projected_gravity_b.z above
                                      ``termination_gravity_z`` (upright ≈ -1.0,
                                      tipped past ~horizontal ≈ 0). Catches
                                      tip-overs that keep the base high.
                    3. torso contact: torso_link contact force over
                                      ``illegal_contact_threshold`` (kept as an
                                      extra signal; on its own it MISSES
                                      crawl / push-up poses where the chest is
                                      held off the floor by the arms — which is
                                      exactly the failure mode that let the
                                      policy learn to crawl instead of walk).

                  The reward applies ``pen_termination`` (-200) on this step
                  via ``mdp.is_terminated``.

    time_out    — episode_length_buf hit max_episode_length-1 without a
                  terminal event. Truncation, no penalty.

Returned as ``(terminated, time_out)`` to match DirectRLEnv's convention.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .env import VelocityTrackingEnv


def compute_dones(env: "VelocityTrackingEnv") -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.robot
    cfg = env.cfg

    # ── 1. Base too low ─────────────────────────────────────────────────
    # Pure kinematic check, independent of contact sensors. The most robust
    # "it fell" signal: if the pelvis is below the threshold, the robot is
    # on the ground no matter how it landed.
    base_height = robot.data.root_pos_w[:, 2]
    fell_low = base_height < cfg.termination_base_height

    # ── 2. Tipped over ──────────────────────────────────────────────────
    # projected_gravity_b is the gravity unit vector in the robot's base
    # frame: ~[0,0,-1] when upright, z-component rising toward 0 as it tips
    # toward horizontal, +1 fully inverted. Terminate once it's no longer
    # clearly "down". Catches tip-overs that happen to keep the base high.
    proj_grav_z = robot.data.projected_gravity_b[:, 2]
    fell_tilt = proj_grav_z > cfg.termination_gravity_z

    # ── 3. Illegal torso contact (extra signal) ─────────────────────────
    forces_history = env.contact_sensor.data.net_forces_w_history  # (N, T, B, 3)
    if forces_history is None or forces_history.numel() == 0:
        torso_contact = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    else:
        torso_forces = forces_history[:, :, env._torso_sensor_ids, :]   # (N,T,len,3)
        max_force = torch.norm(torso_forces, dim=-1).amax(dim=(1, 2))    # (N,)
        torso_contact = max_force > cfg.illegal_contact_threshold

    terminated = fell_low | fell_tilt | torso_contact

    time_out = env.episode_length_buf >= env.max_episode_length - 1

    return terminated, time_out
