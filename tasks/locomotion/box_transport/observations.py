"""Observation assembly — 164-dim policy obs.

Layout (the first 141 dims are an exact mirror of the velocity-tracking obs
so a locomotion checkpoint can be warm-started):

    [0:3]      base_lin_vel       (root frame)
    [3:6]      base_ang_vel       (root frame)
    [6:9]      projected_gravity  (root frame, ~[0,0,-1] when upright)
    [9:12]     velocity_commands  (auto-derived from task phase — toward box if
                                   not lifted, toward target if lifted)
    [12:55]    joint_pos_rel      (joint_pos - default_joint_pos)  43 DoF
    [55:98]    joint_vel_rel      (joint_vel - default_joint_vel)  43 DoF
    [98:141]   last_action        (action sent on the previous step)  43 DoF

    [141:144]  box_pos_rel        (box - robot root, in robot base frame)
    [144:148]  box_quat           (world quaternion, wxyz)
    [148:151]  box_lin_vel_b      (box lin vel, robot base frame)
    [151:154]  target_pos_rel     (target - robot root, in robot base frame)
    [154:157]  l_palm_to_box      (box - left_palm,  robot base frame)
    [157:160]  r_palm_to_box      (box - right_palm, robot base frame)
    [160:163]  box_to_target      (target - box, world frame)
    [163]      lifted_flag        (1.0 if box.z > BOX_LIFT_Z this step)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_rotate_inverse

if TYPE_CHECKING:
    from .env import BoxTransportEnv


def _uniform_noise(x: torch.Tensor, half_range: float) -> torch.Tensor:
    return x + (2.0 * torch.rand_like(x) - 1.0) * half_range


def get_observations(env: "BoxTransportEnv") -> torch.Tensor:
    robot = env.robot
    box = env.box
    cfg = env.cfg

    # ── Locomotion proprioception (mirror of velocity-tracking) ──────────
    base_lin_vel = robot.data.root_lin_vel_b
    base_ang_vel = robot.data.root_ang_vel_b
    projected_gravity = robot.data.projected_gravity_b
    velocity_commands = env._commands_dict["base_velocity"]   # auto-derived

    joint_ids = env._actuated_joint_ids
    joint_pos_rel = (
        robot.data.joint_pos[:, joint_ids]
        - robot.data.default_joint_pos[:, joint_ids]
    )
    joint_vel_rel = (
        robot.data.joint_vel[:, joint_ids]
        - robot.data.default_joint_vel[:, joint_ids]
    )
    last_action = env._actions

    # ── Box-relative / target-relative ────────────────────────────────
    robot_pos_w  = robot.data.root_pos_w                # (N, 3)
    robot_quat_w = robot.data.root_quat_w               # (N, 4) wxyz
    box_pos_w    = box.data.root_pos_w                  # (N, 3)
    box_quat_w   = box.data.root_quat_w                 # (N, 4) wxyz
    box_lin_w    = box.data.root_lin_vel_w              # (N, 3)
    target_pos_w = env._target_pos_w                    # (N, 3)
    l_palm_w     = robot.data.body_pos_w[:, env._left_palm_body_id]   # (N, 3)
    r_palm_w     = robot.data.body_pos_w[:, env._right_palm_body_id]  # (N, 3)

    box_pos_rel    = quat_rotate_inverse(robot_quat_w, box_pos_w - robot_pos_w)
    box_lin_vel_b  = quat_rotate_inverse(robot_quat_w, box_lin_w)
    target_pos_rel = quat_rotate_inverse(robot_quat_w, target_pos_w - robot_pos_w)
    l_palm_to_box  = quat_rotate_inverse(robot_quat_w, box_pos_w - l_palm_w)
    r_palm_to_box  = quat_rotate_inverse(robot_quat_w, box_pos_w - r_palm_w)
    # box_to_target left in world frame: it's a goal direction independent of
    # the robot's current heading.
    box_to_target  = target_pos_w - box_pos_w

    lifted_flag = (box_pos_w[:, 2:3] > env._box_lift_z).float()  # (N, 1)

    # ── Optional uniform noise ───────────────────────────────────────────
    if cfg.obs_noise_enabled:
        base_lin_vel      = _uniform_noise(base_lin_vel, cfg.obs_noise_base_lin_vel)
        base_ang_vel      = _uniform_noise(base_ang_vel, cfg.obs_noise_base_ang_vel)
        projected_gravity = _uniform_noise(projected_gravity, cfg.obs_noise_projected_grav)
        joint_pos_rel     = _uniform_noise(joint_pos_rel, cfg.obs_noise_joint_pos)
        joint_vel_rel     = _uniform_noise(joint_vel_rel, cfg.obs_noise_joint_vel)
        box_pos_rel       = _uniform_noise(box_pos_rel, cfg.obs_noise_box_pos)

    return torch.cat(
        [
            base_lin_vel, base_ang_vel, projected_gravity, velocity_commands,
            joint_pos_rel, joint_vel_rel, last_action,
            box_pos_rel, box_quat_w, box_lin_vel_b,
            target_pos_rel, l_palm_to_box, r_palm_to_box, box_to_target,
            lifted_flag,
        ],
        dim=-1,
    )  # (N, 164)
