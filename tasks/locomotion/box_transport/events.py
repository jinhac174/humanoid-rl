"""Reset and per-step events.

Four things happen here:

    reset_robot(env, env_ids)
        Spawn the robot near the world origin (small xy/yaw noise) with the
        default light-squat joint pose.

    reset_box(env, env_ids)
        Drop the box onto the start table's top with light xy randomization
        within ``cfg.reset_box_xy_range``.

    reset_target(env, env_ids)
        Sample a target xy on the target-table top within
        ``cfg.reset_target_xy_half``. Stored on ``env._target_pos_w``.

    update_autocmd(env)
        Derive the locomotion velocity command (lin_x, lin_y, ang_z) every
        step from the current task phase. When the box isn't lifted, point
        the command toward the box; once lifted, point it toward the target
        xy. Magnitude is clipped to ``cfg.autocmd_lin_vel_max`` and falls to
        zero within ``cfg.autocmd_stop_distance`` of the goal. ang_z is
        P-control on the heading error toward the goal direction.

        Called from ``BoxTransportEnv._pre_physics_step``.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_rotate_inverse

if TYPE_CHECKING:
    from .env import BoxTransportEnv


def _uniform(num: int, lo: float, hi: float, device: torch.device) -> torch.Tensor:
    return torch.rand(num, device=device) * (hi - lo) + lo


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _get_yaw(env: "BoxTransportEnv", env_ids: torch.Tensor) -> torch.Tensor:
    """Robot yaw in world frame from wxyz quaternion."""
    q = env.robot.data.root_quat_w[env_ids]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# -----------------------------------------------------------------------------
# Resets
# -----------------------------------------------------------------------------
def reset_robot(env: "BoxTransportEnv", env_ids: torch.Tensor) -> None:
    """Reset root pose + joint state with small uniform noise."""
    cfg = env.cfg
    n = env_ids.numel()
    device = env.device

    default_root_state = env.robot.data.default_root_state[env_ids].clone()
    dx = _uniform(n, *cfg.reset_pose_x_range, device=device)
    dy = _uniform(n, *cfg.reset_pose_y_range, device=device)
    yaw = _uniform(n, *cfg.reset_pose_yaw_range, device=device)

    default_root_state[:, 0] += dx
    default_root_state[:, 1] += dy
    half = 0.5 * yaw
    qz = torch.sin(half)
    qw = torch.cos(half)
    yaw_quat = torch.stack(
        [qw, torch.zeros_like(qw), torch.zeros_like(qw), qz], dim=-1
    )
    dq = default_root_state[:, 3:7]
    new_quat = torch.zeros_like(dq)
    new_quat[:, 0] = yaw_quat[:, 0] * dq[:, 0] - yaw_quat[:, 1] * dq[:, 1] \
                   - yaw_quat[:, 2] * dq[:, 2] - yaw_quat[:, 3] * dq[:, 3]
    new_quat[:, 1] = yaw_quat[:, 0] * dq[:, 1] + yaw_quat[:, 1] * dq[:, 0] \
                   + yaw_quat[:, 2] * dq[:, 3] - yaw_quat[:, 3] * dq[:, 2]
    new_quat[:, 2] = yaw_quat[:, 0] * dq[:, 2] - yaw_quat[:, 1] * dq[:, 3] \
                   + yaw_quat[:, 2] * dq[:, 0] + yaw_quat[:, 3] * dq[:, 1]
    new_quat[:, 3] = yaw_quat[:, 0] * dq[:, 3] + yaw_quat[:, 1] * dq[:, 2] \
                   - yaw_quat[:, 2] * dq[:, 1] + yaw_quat[:, 3] * dq[:, 0]
    default_root_state[:, 3:7] = new_quat

    default_root_state[:, :3] += env.scene.env_origins[env_ids]

    env.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
    env.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

    # Joint state at default (no scaling/noise for now — keep the warm-started
    # policy's expected initial state).
    default_jpos = env.robot.data.default_joint_pos[env_ids].clone()
    default_jvel = env.robot.data.default_joint_vel[env_ids].clone()
    env.robot.write_joint_state_to_sim(default_jpos, default_jvel, env_ids=env_ids)


def reset_box(env: "BoxTransportEnv", env_ids: torch.Tensor) -> None:
    """Spawn the box on the start-table top with xy randomization."""
    cfg = env.cfg
    n = env_ids.numel()
    device = env.device
    half = float(cfg.reset_box_xy_range)

    # Build the per-env state explicitly (matches the reorient task's pattern,
    # avoids any ambiguity about default_root_state's frame for rigid objects).
    spawn_local = env._box_spawn_pos_local                      # (3,)
    env_origins = env.scene.env_origins[env_ids]                # (n, 3)
    pos_noise = torch.zeros(n, 3, device=device)
    pos_noise[:, 0] = _uniform(n, -half, half, device=device)
    pos_noise[:, 1] = _uniform(n, -half, half, device=device)
    pos_w = spawn_local.unsqueeze(0) + pos_noise + env_origins   # (n, 3)

    # Identity orientation (wxyz = 1, 0, 0, 0) and zero velocity.
    state = torch.zeros(n, 13, device=device)
    state[:, 0:3] = pos_w
    state[:, 3] = 1.0    # qw = 1
    env.box.write_root_pose_to_sim(state[:, 0:7], env_ids=env_ids)
    env.box.write_root_velocity_to_sim(state[:, 7:13], env_ids=env_ids)


def reset_target(env: "BoxTransportEnv", env_ids: torch.Tensor) -> None:
    """Sample target xy on the target-table top, store in env._target_pos_w."""
    cfg = env.cfg
    n = env_ids.numel()
    device = env.device
    hx, hy = cfg.reset_target_xy_half

    dx = _uniform(n, -hx, hx, device=device)
    dy = _uniform(n, -hy, hy, device=device)
    tgt = torch.zeros(n, 3, device=device)
    tgt[:, 0] = env._target_table_center_w[env_ids, 0] + dx
    tgt[:, 1] = env._target_table_center_w[env_ids, 1] + dy
    tgt[:, 2] = env._target_z

    env._target_pos_w[env_ids] = tgt


# -----------------------------------------------------------------------------
# Per-step autocmd derivation
# -----------------------------------------------------------------------------
def update_autocmd(env: "BoxTransportEnv") -> None:
    """Recompute velocity_commands[*] from the current task phase.

    Heuristic:
        * If ``box.z <= BOX_LIFT_Z`` (not yet lifted), aim at the box xy.
        * Otherwise, aim at the target xy.

    The aim is converted to a base-frame (lin_x, lin_y) command, scaled by
    distance and clipped to ``cfg.autocmd_lin_vel_max``. ang_z is derived
    from heading error via P-control (matches locomotion's heading_command
    flow). Within ``cfg.autocmd_stop_distance`` of the goal, linear cmd
    falls to zero.

    Writes directly into ``env._commands_dict["base_velocity"]``.
    """
    cfg = env.cfg
    device = env.device

    # Active goal xy per env: box xy if not lifted, target xy if lifted.
    box_pos_w = env.box.data.root_pos_w
    lifted = (box_pos_w[:, 2] > env._box_lift_z)            # (N,)
    goal_xy = torch.where(
        lifted.unsqueeze(-1),
        env._target_pos_w[:, :2],
        box_pos_w[:, :2],
    )

    robot_pos_w = env.robot.data.root_pos_w
    delta_w = torch.zeros_like(robot_pos_w)
    delta_w[:, :2] = goal_xy - robot_pos_w[:, :2]

    # Convert (delta_x, delta_y, 0) into the robot's base frame.
    robot_quat_w = env.robot.data.root_quat_w
    delta_b = quat_rotate_inverse(robot_quat_w, delta_w)    # (N, 3)

    # Distance (xy norm) and unit direction in base frame.
    dist = delta_b[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (N, 1)
    dir_b = delta_b[:, :2] / dist                            # (N, 2)

    # Magnitude ramps from 0 at stop_distance up to max_vel.
    stop = float(cfg.autocmd_stop_distance)
    mag = (dist.squeeze(-1) - stop).clamp(min=0.0).clamp(max=float(cfg.autocmd_lin_vel_max))
    lin = dir_b * mag.unsqueeze(-1)                          # (N, 2)

    # Heading: angle to the goal in world frame; ang_z = P * wrap_to_pi(err).
    heading_target = torch.atan2(delta_w[:, 1], delta_w[:, 0])
    heading_current = _get_yaw(env, torch.arange(env.num_envs, device=device))
    err = _wrap_to_pi(heading_target - heading_current)
    ang_z = (cfg.autocmd_heading_stiffness * err).clamp(
        -float(cfg.autocmd_ang_vel_max), float(cfg.autocmd_ang_vel_max),
    )

    env._commands_dict["base_velocity"][:, 0] = lin[:, 0]
    env._commands_dict["base_velocity"][:, 1] = lin[:, 1]
    env._commands_dict["base_velocity"][:, 2] = ang_z
