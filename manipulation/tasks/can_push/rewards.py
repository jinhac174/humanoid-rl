import torch
from isaaclab.envs import DirectRLEnv


def can_in_target(env: DirectRLEnv) -> torch.Tensor:
    """1.0 if can center is within target radius (XY only)."""
    xy_dist = torch.norm(
        env.can.data.root_pos_w[:, :2] - env.target_pos_w[:, :2], dim=-1
    )
    return (xy_dist < env.cfg.success_radius).float()


def reward_approach(env: DirectRLEnv) -> torch.Tensor:
    """Left palm approaches can."""
    left_palm = env.robot.data.body_pos_w[:, env.left_palm_idx, :]
    can_pos   = env.can.data.root_pos_w
    full_dist = torch.norm(left_palm - can_pos, dim=-1)
    xy_dist   = torch.norm(left_palm[:, :2] - can_pos[:, :2], dim=-1)
    z_offset  = left_palm[:, 2] - can_pos[:, 2]

    reach_progress = torch.clamp(
        env.prev_left_dist - full_dist, min=0.0, max=0.02
    ) / 0.02

    return (
        0.45 * torch.exp(-8.0  * xy_dist)
        + 0.20 * torch.exp(-18.0 * torch.abs(z_offset - 0.04))
        + 0.35 * reach_progress
    )


def reward_push(env: DirectRLEnv) -> torch.Tensor:
    """Can moves toward target."""
    xy_dist = torch.norm(
        env.can.data.root_pos_w[:, :2] - env.target_pos_w[:, :2], dim=-1
    )
    push_progress = torch.clamp(
        env.prev_can_to_target - xy_dist, min=0.0, max=0.02
    ) / 0.02

    return (
        0.60 * torch.exp(-2.0 * xy_dist)
        + 0.40 * push_progress
    )


def reward_success(env: DirectRLEnv) -> torch.Tensor:
    return can_in_target(env)


def penalty_drop(env: DirectRLEnv, drop_threshold: float = 0.08) -> torch.Tensor:
    dropped = env.can.data.root_pos_w[:, 2] < (env.can_spawn_z - drop_threshold)
    return dropped.float()


def penalty_right_idle(env: DirectRLEnv) -> torch.Tensor:
    """Keep right arm parked -- always active for push task."""
    joint_pos = env.robot.data.joint_pos[:, env.actuated_joint_ids]
    joint_vel = env.robot.data.joint_vel[:, env.actuated_joint_ids]
    right_pos     = joint_pos[:, env.right_arm_slice]
    right_nominal = env.nominal_joint_pos[:, env.right_arm_slice]
    right_vel     = joint_vel[:, env.right_arm_slice]
    deviation = (right_pos - right_nominal).pow(2).mean(dim=-1)
    velocity  = right_vel.pow(2).mean(dim=-1)
    return deviation + 0.3 * velocity


def penalty_joint_limits(env: DirectRLEnv) -> torch.Tensor:
    joint_pos   = env.robot.data.joint_pos[:, env.actuated_joint_ids]
    joint_range = (env.actuated_joint_upper - env.actuated_joint_lower).clamp(min=1e-6)
    margin      = 0.10 * joint_range
    lower_viol  = torch.relu((env.actuated_joint_lower + margin) - joint_pos) / margin
    upper_viol  = torch.relu(joint_pos - (env.actuated_joint_upper - margin)) / margin
    return (lower_viol + upper_viol).mean(dim=-1)


def penalty_action_rate(env: DirectRLEnv) -> torch.Tensor:
    return (env.actions - env.prev_actions).pow(2).mean(dim=-1)


def penalty_joint_vel(env: DirectRLEnv) -> torch.Tensor:
    joint_vel = env.robot.data.joint_vel[:, env.actuated_joint_ids]
    return joint_vel.pow(2).mean(dim=-1)