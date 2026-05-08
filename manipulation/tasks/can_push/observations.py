import torch
from isaaclab.envs import DirectRLEnv


def joint_pos(env: DirectRLEnv, joint_ids: list[int]) -> torch.Tensor:
    return env.robot.data.joint_pos[:, joint_ids]


def joint_vel(env: DirectRLEnv, joint_ids: list[int]) -> torch.Tensor:
    return env.robot.data.joint_vel[:, joint_ids]


def target_error(env: DirectRLEnv, joint_ids: list[int]) -> torch.Tensor:
    return env.target_joint_pos - env.robot.data.joint_pos[:, joint_ids]


def can_pos_relative(env: DirectRLEnv) -> torch.Tensor:
    return env.can.data.root_pos_w - env.robot.data.root_pos_w


def can_lin_vel(env: DirectRLEnv) -> torch.Tensor:
    return env.can.data.root_lin_vel_w


def target_pos_relative(env: DirectRLEnv) -> torch.Tensor:
    # target_pos_w is a fixed constant tensor set in env.__init__
    # add env offsets since target_pos_w is already in world frame
    return env.target_pos_w - env.robot.data.root_pos_w


def left_palm_pos_relative(env: DirectRLEnv) -> torch.Tensor:
    return (
        env.robot.data.body_pos_w[:, env.left_palm_idx, :]
        - env.robot.data.root_pos_w
    )


def right_palm_pos_relative(env: DirectRLEnv) -> torch.Tensor:
    return (
        env.robot.data.body_pos_w[:, env.right_palm_idx, :]
        - env.robot.data.root_pos_w
    )


def can_to_left_palm(env: DirectRLEnv) -> torch.Tensor:
    return (
        env.robot.data.body_pos_w[:, env.left_palm_idx, :]
        - env.can.data.root_pos_w
    )


def can_to_right_palm(env: DirectRLEnv) -> torch.Tensor:
    return (
        env.robot.data.body_pos_w[:, env.right_palm_idx, :]
        - env.can.data.root_pos_w
    )


def can_to_target(env: DirectRLEnv) -> torch.Tensor:
    return env.target_pos_w - env.can.data.root_pos_w


def get_obs(env: DirectRLEnv, joint_ids: list[int]) -> torch.Tensor:
    """
    108-dim observation vector.

    [0:28]    joint_pos
    [28:56]   joint_vel
    [56:84]   target_error
    [84:87]   can_pos          -- relative to robot root
    [87:90]   can_lin_vel      -- world frame
    [90:93]   target_pos       -- relative to robot root (fixed)
    [93:96]   left_palm_pos    -- relative to robot root
    [96:99]   right_palm_pos   -- relative to robot root
    [99:102]  can_to_left_palm
    [102:105] can_to_right_palm
    [105:108] can_to_target
    """
    return torch.cat([
        joint_pos(env, joint_ids),
        joint_vel(env, joint_ids),
        target_error(env, joint_ids),
        can_pos_relative(env),
        can_lin_vel(env),
        target_pos_relative(env),
        left_palm_pos_relative(env),
        right_palm_pos_relative(env),
        can_to_left_palm(env),
        can_to_right_palm(env),
        can_to_target(env),
    ], dim=-1)