"""Observation assembly — flat 141-dim policy obs.

Layout (matches IsaacLab Isaac-Velocity-Flat-G1-v0, in the same order):

    [0:3]     base_lin_vel     (root frame, body twist)
    [3:6]     base_ang_vel     (root frame)
    [6:9]     projected_gravity (gravity in root frame, ~[0,0,-1] when upright)
    [9:12]    velocity_commands (lin_x, lin_y, ang_z)
    [12:55]   joint_pos_rel    (joint_pos - default_joint_pos)  43 DoF
    [55:98]   joint_vel_rel    (joint_vel - default_joint_vel)  43 DoF
    [98:141]  last_action      (action sent on the previous step)  43 DoF

Optional uniform observation noise is applied per term (matches the noise
parameters from the IsaacLab task). Disable via ``cfg.obs_noise_enabled=False``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .env import VelocityTrackingEnv


def _uniform_noise(x: torch.Tensor, half_range: float) -> torch.Tensor:
    """Add uniform noise in ``[-half_range, +half_range]`` per element."""
    return x + (2.0 * torch.rand_like(x) - 1.0) * half_range


def get_observations(env: "VelocityTrackingEnv") -> torch.Tensor:
    """Assemble the 141-dim policy observation."""
    robot = env.robot
    cfg = env.cfg

    # Body-frame twists (IsaacLab convention: root_lin_vel_b / root_ang_vel_b
    # are pre-rotated into the robot root frame).
    base_lin_vel = robot.data.root_lin_vel_b              # (N, 3)
    base_ang_vel = robot.data.root_ang_vel_b              # (N, 3)
    projected_gravity = robot.data.projected_gravity_b    # (N, 3)

    velocity_commands = env._commands_dict["base_velocity"]  # (N, 3)

    joint_ids = env._actuated_joint_ids
    joint_pos_rel = (
        robot.data.joint_pos[:, joint_ids]
        - robot.data.default_joint_pos[:, joint_ids]
    )                                                    # (N, 43)
    joint_vel_rel = (
        robot.data.joint_vel[:, joint_ids]
        - robot.data.default_joint_vel[:, joint_ids]
    )                                                    # (N, 43)

    last_action = env._actions                            # (N, 43)

    if cfg.obs_noise_enabled:
        base_lin_vel      = _uniform_noise(base_lin_vel, cfg.obs_noise_base_lin_vel)
        base_ang_vel      = _uniform_noise(base_ang_vel, cfg.obs_noise_base_ang_vel)
        projected_gravity = _uniform_noise(projected_gravity, cfg.obs_noise_projected_grav)
        joint_pos_rel     = _uniform_noise(joint_pos_rel, cfg.obs_noise_joint_pos)
        joint_vel_rel     = _uniform_noise(joint_vel_rel, cfg.obs_noise_joint_vel)

    return torch.cat(
        [
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            velocity_commands,
            joint_pos_rel,
            joint_vel_rel,
            last_action,
        ],
        dim=-1,
    )  # (N, 141)
