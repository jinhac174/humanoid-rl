"""Reset and interval events.

Three things happen here:

    sample_velocity_commands(env, env_ids)
        Resample the (lin_x, lin_y, ang_z) velocity command for the given
        envs from cfg ranges. A small fraction (cfg.cmd_standing_prob) get
        an all-zero "stand still" command. Called from
        :class:`VelocityTrackingEnv._pre_physics_step` when an env's
        resample timer expires, and from ``_reset_idx`` on episode reset.

    reset_robot(env, env_ids)
        Place the root at default pose + uniform XY/yaw noise; reset joint
        positions to ``default * uniform(scale_range)`` and joint velocities
        to ``uniform(vel_range)``.

    push_robot(env, env_ids)
        (Optional, ``cfg.push_enabled``.) Add a uniform XY perturbation to
        the base linear velocity. Mimics the IsaacLab interval push event.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .env import VelocityTrackingEnv


def _uniform(num: int, lo: float, hi: float, device: torch.device) -> torch.Tensor:
    return torch.rand(num, device=device) * (hi - lo) + lo


def sample_velocity_commands(
    env: "VelocityTrackingEnv", env_ids: torch.Tensor | None,
) -> None:
    """Resample (lin_x, lin_y, ang_z). ``env_ids=None`` means all envs."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if env_ids.numel() == 0:
        return

    cfg = env.cfg
    n = env_ids.numel()
    device = env.device

    lin_x = _uniform(n, *cfg.cmd_lin_vel_x_range, device=device)
    lin_y = _uniform(n, *cfg.cmd_lin_vel_y_range, device=device)
    ang_z = _uniform(n, *cfg.cmd_ang_vel_z_range, device=device)

    # A fraction of envs get a "stand still" command (zero everything).
    if cfg.cmd_standing_prob > 0.0:
        stand_mask = torch.rand(n, device=device) < cfg.cmd_standing_prob
        lin_x = torch.where(stand_mask, torch.zeros_like(lin_x), lin_x)
        lin_y = torch.where(stand_mask, torch.zeros_like(lin_y), lin_y)
        ang_z = torch.where(stand_mask, torch.zeros_like(ang_z), ang_z)

    env._commands_dict["base_velocity"][env_ids] = torch.stack(
        [lin_x, lin_y, ang_z], dim=-1
    )


def reset_robot(env: "VelocityTrackingEnv", env_ids: torch.Tensor) -> None:
    """Reset root pose + joint state with uniform noise."""
    cfg = env.cfg
    n = env_ids.numel()
    device = env.device

    # ── Root pose ────────────────────────────────────────────────────────
    default_root_state = env.robot.data.default_root_state[env_ids].clone()
    # Add per-env XY/yaw noise, then translate to per-env origins.
    dx = _uniform(n, *cfg.reset_pose_x_range, device=device)
    dy = _uniform(n, *cfg.reset_pose_y_range, device=device)
    yaw = _uniform(n, *cfg.reset_pose_yaw_range, device=device)

    default_root_state[:, 0] += dx
    default_root_state[:, 1] += dy
    # Compose yaw rotation onto the default quaternion (default is upright).
    half = 0.5 * yaw
    qz = torch.sin(half)
    qw = torch.cos(half)
    yaw_quat = torch.stack(
        [qw, torch.zeros_like(qw), torch.zeros_like(qw), qz], dim=-1
    )
    # Hamilton product: out = yaw_quat * default_quat.
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

    # Translate to env origins.
    default_root_state[:, :3] += env.scene.env_origins[env_ids]

    # Velocity stays at default (zeros from default_root_state).
    env.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
    env.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

    # ── Joint state ──────────────────────────────────────────────────────
    default_jpos = env.robot.data.default_joint_pos[env_ids].clone()
    default_jvel = env.robot.data.default_joint_vel[env_ids].clone()

    # Random scale in [scale_lo, scale_hi] applied per-env, per-joint.
    scale_lo, scale_hi = cfg.reset_joint_pos_scale_range
    scale = _uniform(n * default_jpos.shape[1], scale_lo, scale_hi, device=device)
    scale = scale.view(n, default_jpos.shape[1])
    joint_pos = default_jpos * scale

    vel_lo, vel_hi = cfg.reset_joint_vel_range
    if vel_hi == vel_lo == 0.0:
        joint_vel = default_jvel
    else:
        joint_vel = _uniform(
            n * default_jvel.shape[1], vel_lo, vel_hi, device=device,
        ).view(n, default_jvel.shape[1])

    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def push_robot(env: "VelocityTrackingEnv", env_ids: torch.Tensor) -> None:
    """Add a uniform XY velocity kick to the base. No-op when push_enabled=False."""
    cfg = env.cfg
    n = env_ids.numel()
    device = env.device

    root_vel = env.robot.data.root_lin_vel_w[env_ids].clone()
    root_vel[:, 0] += _uniform(n, *cfg.push_velocity_xy_range, device=device)
    root_vel[:, 1] += _uniform(n, *cfg.push_velocity_xy_range, device=device)

    # Re-pack into the (pos, quat, lin_vel, ang_vel) state format expected by
    # write_root_velocity_to_sim.
    full_vel = torch.cat(
        [root_vel, env.robot.data.root_ang_vel_w[env_ids]], dim=-1,
    )
    env.robot.write_root_velocity_to_sim(full_vel, env_ids=env_ids)
