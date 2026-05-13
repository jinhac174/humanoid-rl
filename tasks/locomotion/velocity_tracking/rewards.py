"""Reward orchestration.

We import IsaacLab's tested reward implementations (both the core ``mdp``
helpers and the locomotion-specific ones) and call them with the pre-resolved
``SceneEntityCfg`` objects on ``env._scfg``. This keeps the per-step math
identical to the upstream Isaac-Velocity-Flat-G1-v0 task; only the weights
and termination logic are ours.

Order of operations matters for one detail: ``mdp.is_terminated`` reads
``env.termination_manager.terminated``. Our ``_TerminationShim`` exposes
``env._terminated_buf`` which ``env._get_dones`` populates BEFORE
``env._get_rewards`` runs (DirectRLEnv calls dones → rewards → reset → obs).

All terms are also written to ``env.extras`` for wandb logging, both raw
(pre-weight) and scaled.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp

if TYPE_CHECKING:
    from .env import VelocityTrackingEnv


def compute_reward(env: "VelocityTrackingEnv") -> torch.Tensor:
    cfg = env.cfg
    s = env._scfg

    # ── Tracking (positive) ──────────────────────────────────────────────
    track_lin_xy = mdp.track_lin_vel_xy_exp(
        env, std=cfg.rew_track_lin_vel_std,
        command_name="base_velocity", asset_cfg=s["robot_all"],
    )
    # G1RoughEnvCfg uses the yaw-frame variant of lin-vel tracking and the
    # world-frame variant of ang-vel-z tracking.
    track_lin_xy_yaw = locomotion_mdp.rewards.track_lin_vel_xy_yaw_frame_exp(
        env, std=cfg.rew_track_lin_vel_std,
        command_name="base_velocity", asset_cfg=s["robot_all"],
    )
    track_ang_z = locomotion_mdp.rewards.track_ang_vel_z_world_exp(
        env, std=cfg.rew_track_ang_vel_std,
        command_name="base_velocity", asset_cfg=s["robot_all"],
    )
    feet_air_time = locomotion_mdp.rewards.feet_air_time_positive_biped(
        env, command_name="base_velocity",
        threshold=cfg.rew_feet_air_time_threshold,
        sensor_cfg=s["feet_contact"],
    )

    # ── Penalties ────────────────────────────────────────────────────────
    # Termination penalty fires only when env.reset_terminated is True (i.e.
    # illegal contact, NOT timeout) — see _TerminationShim.
    term_penalty = mdp.is_terminated(env)
    lin_vel_z    = mdp.lin_vel_z_l2(env, asset_cfg=s["robot_all"])
    ang_vel_xy   = mdp.ang_vel_xy_l2(env, asset_cfg=s["robot_all"])
    flat_orient  = mdp.flat_orientation_l2(env, asset_cfg=s["robot_all"])
    action_rate  = mdp.action_rate_l2(env)
    dof_torques  = mdp.joint_torques_l2(env, asset_cfg=s["legs_for_acc_torque"])
    dof_acc      = mdp.joint_acc_l2(env, asset_cfg=s["legs_for_acc_torque"])
    pos_limits_ankle = mdp.joint_pos_limits(env, asset_cfg=s["ankles"])
    feet_slide   = locomotion_mdp.rewards.feet_slide(
        env, sensor_cfg=s["feet_contact"], asset_cfg=s["feet_robot"],
    )
    dev_hip   = mdp.joint_deviation_l1(env, asset_cfg=s["hip_yaw_roll"])
    dev_arms  = mdp.joint_deviation_l1(env, asset_cfg=s["arms"])
    dev_hands = mdp.joint_deviation_l1(env, asset_cfg=s["hands"])
    dev_waist = mdp.joint_deviation_l1(env, asset_cfg=s["waist"])

    # ── Weighted sum ─────────────────────────────────────────────────────
    reward = (
        + cfg.rew_track_lin_vel_xy * track_lin_xy_yaw
        + cfg.rew_track_ang_vel_z  * track_ang_z
        + cfg.rew_feet_air_time    * feet_air_time
        - cfg.pen_termination      * term_penalty
        - cfg.pen_lin_vel_z        * lin_vel_z
        - cfg.pen_ang_vel_xy       * ang_vel_xy
        - cfg.pen_flat_orientation * flat_orient
        - cfg.pen_action_rate      * action_rate
        - cfg.pen_dof_torques      * dof_torques
        - cfg.pen_dof_acc          * dof_acc
        - cfg.pen_dof_pos_limits   * pos_limits_ankle
        - cfg.pen_feet_slide       * feet_slide
        - cfg.pen_joint_dev_hip    * dev_hip
        - cfg.pen_joint_dev_arms   * dev_arms
        - cfg.pen_joint_dev_hands  * dev_hands
        - cfg.pen_joint_dev_waist  * dev_waist
    )

    # ── Logging (read by trainer's iter_loggable_items) ──────────────────
    # Trainer flattens info["log"] OR top-level extras keys named "reward/*"
    # / "task/*". We use the top-level convention (matches reorient).
    extras = env.extras
    extras["reward/track_lin_vel_xy"]    = track_lin_xy_yaw
    extras["reward/track_lin_vel_xy_w"]  = mdp.track_lin_vel_xy_exp(
        env, std=cfg.rew_track_lin_vel_std, command_name="base_velocity",
        asset_cfg=s["robot_all"],
    )  # world-frame variant for diagnostics
    extras["reward/track_ang_vel_z"]     = track_ang_z
    extras["reward/feet_air_time"]       = feet_air_time
    extras["reward/feet_slide"]          = feet_slide
    extras["reward/lin_vel_z"]           = lin_vel_z
    extras["reward/ang_vel_xy"]          = ang_vel_xy
    extras["reward/flat_orientation"]    = flat_orient
    extras["reward/action_rate"]         = action_rate
    extras["reward/dof_torques"]         = dof_torques
    extras["reward/dof_acc"]             = dof_acc
    extras["reward/dof_pos_limits_ankle"] = pos_limits_ankle
    extras["reward/joint_dev_hip"]       = dev_hip
    extras["reward/joint_dev_arms"]      = dev_arms
    extras["reward/joint_dev_hands"]     = dev_hands
    extras["reward/joint_dev_waist"]     = dev_waist
    extras["reward/termination"]         = term_penalty
    extras["reward/total"]               = reward

    # Task diagnostics — what the agent is actually doing.
    extras["task/cmd_lin_vel_x"] = env._commands_dict["base_velocity"][:, 0]
    extras["task/cmd_lin_vel_y"] = env._commands_dict["base_velocity"][:, 1]
    extras["task/cmd_ang_vel_z"] = env._commands_dict["base_velocity"][:, 2]
    extras["task/base_height_z"] = env.robot.data.root_pos_w[:, 2]
    extras["task/base_lin_vel_x_b"] = env.robot.data.root_lin_vel_b[:, 0]

    return reward
