"""
Reset events for the reorient task.

Adapted from SAPG donor (allegro_kuka_two_arms.py) reset logic:
- Robot DoFs: uniform noise around defaults (arm and finger ranges separate)
- Cuboid: position noise +-XY, +-Z + full random SO(3) rotation
- Goal: random pose inside target volume with full random SO(3)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import math

import torch

if TYPE_CHECKING:
    from .env import ReorientEnv


def _random_quat_wxyz(num: int, device: torch.device) -> torch.Tensor:
    """
    Uniformly random unit quaternion in (w, x, y, z) convention.

    Ported from allegro_kuka_two_arms.py::get_random_quat (Marsaglia's method).
    """
    uvw = torch.rand(num, 3, device=device)
    two_pi = 2.0 * math.pi
    sqrt_1_mu = torch.sqrt(1.0 - uvw[:, 0])
    sqrt_mu = torch.sqrt(uvw[:, 0])
    q_w = sqrt_1_mu * torch.sin(two_pi * uvw[:, 1])
    q_x = sqrt_1_mu * torch.cos(two_pi * uvw[:, 1])
    q_y = sqrt_mu * torch.sin(two_pi * uvw[:, 2])
    q_z = sqrt_mu * torch.cos(two_pi * uvw[:, 2])
    # Stack as (w, x, y, z) -- IsaacLab convention
    return torch.stack([q_w, q_x, q_y, q_z], dim=-1)


def reset_robot(env: "ReorientEnv", env_ids: torch.Tensor) -> None:
    """Reset joints to default + noise; velocities to noise around zero.

    Donor applies different noise magnitudes for arm vs finger joints,
    and adds velocity noise to all DoFs. Leg joints (locked in G1_FIXED_CFG)
    are kept at their defaults with zero velocity.
    """
    n = env_ids.numel()
    cfg = env.cfg

    # Start from default pose for ALL joints (including locked legs)
    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    # Build per-actuated-joint noise coefficients
    # actuated_joint_ids: [0:7] left arm, [7:14] right arm,
    #                     [14:21] left hand, [21:28] right hand
    num_actuated = len(env.actuated_joint_ids)
    noise_coeff = torch.zeros(num_actuated, device=env.device)
    noise_coeff[0:7] = cfg.reset_dof_pos_noise_arm
    noise_coeff[7:14] = cfg.reset_dof_pos_noise_arm
    noise_coeff[14:21] = cfg.reset_dof_pos_noise_finger
    noise_coeff[21:28] = cfg.reset_dof_pos_noise_finger

    # Position noise: uniform in [-coeff, +coeff] per joint
    pos_noise = (2.0 * torch.rand(n, num_actuated, device=env.device) - 1.0) * noise_coeff
    joint_pos[:, env.actuated_joint_ids] += pos_noise

    # Velocity noise: uniform in [-vel_noise, +vel_noise] on actuated joints
    vel_noise = (2.0 * torch.rand(n, num_actuated, device=env.device) - 1.0) * cfg.reset_dof_vel_noise
    joint_vel[:, env.actuated_joint_ids] = vel_noise

    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # Fixed-base root sync
    default_root_state = env.robot.data.default_root_state[env_ids].clone()
    default_root_state[:, :3] += env.scene.env_origins[env_ids]
    env.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
    env.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

    # Reset delta-control targets to the noisy starting positions
    env.joint_targets[env_ids] = joint_pos[:, env.actuated_joint_ids]


def _sample_goal_pose(
    env: "ReorientEnv", env_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a uniform random pose inside the target volume for each env in
    `env_ids`. Returns (goal_pos_w, goal_quat_w) of shape (num, 3) and (num, 4).
    """
    num = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    rand01 = torch.rand(num, 3, device=env.device)
    tv_min = env.target_volume_min.unsqueeze(0)
    tv_max = env.target_volume_max.unsqueeze(0)
    goal_rel = tv_min + rand01 * (tv_max - tv_min)
    goal_pos_w = env.target_volume_origin.unsqueeze(0) + goal_rel + env_origins
    goal_quat_w = _random_quat_wxyz(num, device=env.device)
    return goal_pos_w, goal_quat_w


def reset_objects(env: "ReorientEnv", env_ids: torch.Tensor) -> None:
    """
    Reset cuboid to spawn pose with donor-level noise, and goal to a random
    pose inside the target volume.

    Donor noise: position +-0.1m XY, +-0.02m Z; full random SO(3) orientation.
    """
    num_reset = env_ids.numel()
    cfg = env.cfg
    env_origins = env.scene.env_origins[env_ids]

    # -- Cuboid ---------------------------------------------------------------
    cuboid_spawn = torch.tensor(
        cfg.cuboid_spawn_pos, dtype=torch.float32, device=env.device
    )
    # Position noise: donor uses +-resetPositionNoiseX/Y/Z
    pos_noise = torch.zeros(num_reset, 3, device=env.device)
    pos_noise[:, 0] = (2.0 * torch.rand(num_reset, device=env.device) - 1.0) * cfg.reset_position_noise_x
    pos_noise[:, 1] = (2.0 * torch.rand(num_reset, device=env.device) - 1.0) * cfg.reset_position_noise_y
    pos_noise[:, 2] = (2.0 * torch.rand(num_reset, device=env.device) - 1.0) * cfg.reset_position_noise_z
    cuboid_pos_w = cuboid_spawn.unsqueeze(0) + pos_noise + env_origins

    # Rotation: full random SO(3) when reset_rotation_noise >= 1.0
    if cfg.reset_rotation_noise >= 1.0:
        cuboid_quat_w = _random_quat_wxyz(num_reset, device=env.device)
    else:
        # Partial rotation noise: random yaw scaled by noise factor
        yaw = (torch.rand(num_reset, device=env.device) - 0.5) * (
            2.0 * math.pi * cfg.reset_rotation_noise
        )
        half = 0.5 * yaw
        cuboid_quat_w = torch.stack(
            [torch.cos(half), torch.zeros_like(half),
             torch.zeros_like(half), torch.sin(half)],
            dim=-1,
        )

    cuboid_state = torch.zeros(num_reset, 13, device=env.device)
    cuboid_state[:, 0:3] = cuboid_pos_w
    cuboid_state[:, 3:7] = cuboid_quat_w
    env.cuboid.write_root_pose_to_sim(cuboid_state[:, 0:7], env_ids=env_ids)
    env.cuboid.write_root_velocity_to_sim(cuboid_state[:, 7:13], env_ids=env_ids)

    # Record init pos for lift calculation
    env.object_init_pos_w[env_ids] = cuboid_pos_w

    # -- Goal -----------------------------------------------------------------
    goal_pos_w, goal_quat_w = _sample_goal_pose(env, env_ids)
    goal_state = torch.zeros(num_reset, 13, device=env.device)
    goal_state[:, 0:3] = goal_pos_w
    goal_state[:, 3:7] = goal_quat_w
    env.goal.write_root_pose_to_sim(goal_state[:, 0:7], env_ids=env_ids)
    env.goal.write_root_velocity_to_sim(goal_state[:, 7:13], env_ids=env_ids)


def reset_goal_only(env: "ReorientEnv", env_ids: torch.Tensor) -> None:
    """
    Resample only the goal pose (NOT the cuboid, NOT the robot).
    Called from _pre_physics_step on the step after a success.
    """
    num = env_ids.numel()
    goal_pos_w, goal_quat_w = _sample_goal_pose(env, env_ids)

    goal_state = torch.zeros(num, 13, device=env.device)
    goal_state[:, 0:3] = goal_pos_w
    goal_state[:, 3:7] = goal_quat_w
    env.goal.write_root_pose_to_sim(goal_state[:, 0:7], env_ids=env_ids)
    env.goal.write_root_velocity_to_sim(goal_state[:, 7:13], env_ids=env_ids)


def reset_buffers(env: "ReorientEnv", env_ids: torch.Tensor) -> None:
    """Clear all task-specific state tensors for the given envs."""
    env.lifted_object[env_ids] = False
    env.near_goal_steps[env_ids] = 0
    env.successes[env_ids] = 0.0
    env.reset_goal_buf[env_ids] = False
    env.near_goal[env_ids] = False
    # Sentinel -1 triggers lazy init on first obs computation (see donor code).
    env.closest_fingertip_dist[env_ids] = -1.0
    env.closest_keypoint_max_dist[env_ids] = -1.0