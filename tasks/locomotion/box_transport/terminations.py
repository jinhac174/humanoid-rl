"""Episode termination logic.

Three reasons an episode ends:

    terminated  — the robot fell OR the box was dropped on the floor OR the
                  box was successfully placed on the target table. The first
                  two fire the ``pen_termination`` / ``pen_drop`` penalties;
                  the third is a positive terminal event (no penalty, and
                  ``rew_place_bonus`` fires for several frames before
                  termination via the ``_success_counter``).

    time_out    — episode_length_buf hit max_episode_length - 1. Truncation,
                  no penalty.

Robot-fell detection mirrors the locomotion env (base_height + gravity_z +
torso contact, OR-combined). Box drop is a simple z-below-threshold check.
Success is "box within ``rew_place_distance_tol`` of target for
``success_steps_required`` consecutive control steps."

The success counter lives on ``env._success_step_count`` and is updated each
step here so the reward module can read it without duplicating the math.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .env import BoxTransportEnv


def compute_dones(env: "BoxTransportEnv") -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.robot
    box = env.box
    cfg = env.cfg

    # ── 1. Robot fell — same 3-signal OR check as locomotion ──────────────
    base_height = robot.data.root_pos_w[:, 2]
    fell_low = base_height < cfg.termination_base_height

    proj_grav_z = robot.data.projected_gravity_b[:, 2]
    fell_tilt = proj_grav_z > cfg.termination_gravity_z

    forces_history = env.contact_sensor.data.net_forces_w_history
    if forces_history is None or forces_history.numel() == 0:
        torso_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        torso_forces = forces_history[:, :, env._torso_sensor_ids, :]
        max_force = torch.norm(torso_forces, dim=-1).amax(dim=(1, 2))
        torso_contact = max_force > cfg.illegal_contact_threshold

    fell = fell_low | fell_tilt | torso_contact

    # ── 2. Box dropped on floor ────────────────────────────────────────
    box_dropped = box.data.root_pos_w[:, 2] < cfg.box_drop_z

    # ── 3. Success counter — box on target table within tol ────────────
    box_xy = box.data.root_pos_w[:, :2]
    tgt_xy = env._target_pos_w[:, :2]
    box_z = box.data.root_pos_w[:, 2]
    near_target_xy = (box_xy - tgt_xy).norm(dim=-1) < cfg.rew_place_distance_tol
    near_target_z  = box_z < (env._target_z + 0.10)         # within 10 cm vertical
    placed_now = near_target_xy & near_target_z

    env._success_step_count = torch.where(
        placed_now,
        env._success_step_count + 1,
        torch.zeros_like(env._success_step_count),
    )
    success = env._success_step_count >= cfg.success_steps_required

    # ── Compose ────────────────────────────────────────────────────────
    terminated = fell | box_dropped | success
    time_out = env.episode_length_buf >= env.max_episode_length - 1

    # Cache flags so the reward module can read them without re-computing.
    env._fell_buf       = fell
    env._box_dropped_buf = box_dropped
    env._success_buf    = success

    return terminated, time_out
