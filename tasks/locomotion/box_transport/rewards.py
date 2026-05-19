"""Reward orchestration — phase-1 velocity-tracking reward + manipulation milestones.

Architecture (see env.py and events.update_autocmd):

    * The HIGH-LEVEL controller (``events.update_autocmd``) writes a velocity
      command into ``env._commands_dict["base_velocity"]`` every step, aimed
      at the box (if not lifted) or the target (if lifted), with magnitude
      tapering to zero within ``cfg.autocmd_stop_distance`` of the goal.

    * The LOW-LEVEL policy was warm-started from a velocity-tracking
      checkpoint. We give it the SAME tracking reward it was trained on:

          + track_lin_vel_xy_yaw_frame_exp   (yaw-frame xy velocity match)
          + track_ang_vel_z_world_exp        (yaw-rate match)
          + feet_air_time_positive_biped     (stepping incentive, only when cmd≠0)

      The "walk-to-box" / "walk-to-target" behaviour emerges from the policy
      tracking the auto-derived command — no separate dense distance shaping.
      When the policy is within stop_distance of the active goal, command
      magnitude is zero, so the tracking reward pays the policy for being
      STILL — exactly when it needs to grasp / release.

    * Manipulation milestones layer on top:

          + bimanual_contact  (one-shot, both palms within GRIP_DISTANCE)
          + lift              (one-shot, box.z first crosses BOX_LIFT_Z)
          + place_bonus       (continuous, box xy within tol of target AND
                               on table)
          − drop              (one-shot, box hits the floor; from terminations)
          − termination       (-pen_termination, only when robot fell;
                               read from env._fell_buf, NOT generic
                               mdp.is_terminated, so box-drop / success
                               don't double-fire)

    * Locomotion regularizers kept at ~0.5× weight (see env_cfg pen_* fields).

One-shot milestone bonuses are gated by per-env latched flags
(``env._bimanual_contact_achieved``, ``env._lift_achieved``) so each fires
once per episode and can't be farmed. The flags are reset in
``BoxTransportEnv._reset_idx``. The flags also use CURRENT box state to
detect the transition — a dropped-and-re-lifted box does not double-fire
the lift bonus, but the policy is also not penalised for needing a second
attempt.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp

if TYPE_CHECKING:
    from .env import BoxTransportEnv


def compute_reward(env: "BoxTransportEnv") -> torch.Tensor:
    cfg = env.cfg
    s = env._scfg

    # ── Locomotion tracking (against auto-derived command) ──────────────
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

    # ── Geometry shortcuts ────────────────────────────────────────────
    box_pos_w    = env.box.data.root_pos_w
    target_pos_w = env._target_pos_w
    l_palm_w = env.robot.data.body_pos_w[:, env._left_palm_body_id]
    r_palm_w = env.robot.data.body_pos_w[:, env._right_palm_body_id]

    # ── Manipulation milestone 1 — bimanual contact (one-shot) ─────────
    l_palm_dist = (l_palm_w - box_pos_w).norm(dim=-1)
    r_palm_dist = (r_palm_w - box_pos_w).norm(dim=-1)
    grip = float(env._grip_distance)
    both_palms_close = (l_palm_dist < grip) & (r_palm_dist < grip)
    bimanual_event = both_palms_close & (~env._bimanual_contact_achieved)
    env._bimanual_contact_achieved = env._bimanual_contact_achieved | both_palms_close
    bimanual_contact = bimanual_event.float()

    # ── Manipulation milestone 2 — lift (one-shot) ────────────────────
    lifted_now = (box_pos_w[:, 2] > env._box_lift_z)
    lift_event = lifted_now & (~env._lift_achieved)
    env._lift_achieved = env._lift_achieved | lifted_now
    lift = lift_event.float()

    # ── Manipulation milestone 3 — place bonus (continuous) ───────────
    # Box xy near the per-env target AND z below "well above the table".
    box_xy_err = (box_pos_w[:, :2] - target_pos_w[:, :2]).norm(dim=-1)
    near_xy = box_xy_err < cfg.rew_place_distance_tol
    near_z  = box_pos_w[:, 2] < (env._target_z + 0.10)
    place_bonus = (near_xy & near_z).float()

    # ── Sparse penalties (one-shot via terminations) ───────────────────
    drop       = env._box_dropped_buf.float()
    term_robot = env._fell_buf.float()

    # ── Locomotion regularizers (kept, reduced weights via cfg) ──────
    lin_vel_z    = mdp.lin_vel_z_l2(env, asset_cfg=s["robot_all"])
    ang_vel_xy   = mdp.ang_vel_xy_l2(env, asset_cfg=s["robot_all"])
    flat_orient  = mdp.flat_orientation_l2(env, asset_cfg=s["robot_all"])
    action_rate  = mdp.action_rate_l2(env)
    dof_torques  = mdp.joint_torques_l2(env, asset_cfg=s["hips_and_knees"])
    dof_acc      = mdp.joint_acc_l2(env, asset_cfg=s["hips_and_knees"])
    pos_limits_ankle = mdp.joint_pos_limits(env, asset_cfg=s["ankles"])
    feet_slide   = locomotion_mdp.rewards.feet_slide(
        env, sensor_cfg=s["feet_contact"], asset_cfg=s["feet_robot"],
    )
    dev_hip   = mdp.joint_deviation_l1(env, asset_cfg=s["hip_yaw_roll"])
    dev_arms  = mdp.joint_deviation_l1(env, asset_cfg=s["arms"])
    dev_hands = mdp.joint_deviation_l1(env, asset_cfg=s["hands"])
    dev_waist = mdp.joint_deviation_l1(env, asset_cfg=s["waist"])

    # ── Weighted sum ──────────────────────────────────────────────────
    reward = (
        + cfg.rew_track_lin_vel_xy * track_lin_xy_yaw
        + cfg.rew_track_ang_vel_z  * track_ang_z
        + cfg.rew_feet_air_time    * feet_air_time
        + cfg.rew_bimanual_contact * bimanual_contact
        + cfg.rew_lift             * lift
        + cfg.rew_place_bonus      * place_bonus
        - cfg.pen_drop             * drop
        - cfg.pen_termination      * term_robot
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

    # ── Logging to extras (trainer flattens these to wandb) ────────────
    extras = env.extras
    extras["reward/track_lin_vel_xy"]    = track_lin_xy_yaw
    extras["reward/track_ang_vel_z"]     = track_ang_z
    extras["reward/feet_air_time"]       = feet_air_time
    extras["reward/bimanual_contact"]    = bimanual_contact
    extras["reward/lift"]                = lift
    extras["reward/place_bonus"]         = place_bonus
    extras["reward/drop"]                = drop
    extras["reward/termination"]         = term_robot
    extras["reward/lin_vel_z"]           = lin_vel_z
    extras["reward/ang_vel_xy"]          = ang_vel_xy
    extras["reward/flat_orientation"]    = flat_orient
    extras["reward/action_rate"]         = action_rate
    extras["reward/dof_torques"]         = dof_torques
    extras["reward/dof_acc"]             = dof_acc
    extras["reward/dof_pos_limits_ankle"] = pos_limits_ankle
    extras["reward/feet_slide"]          = feet_slide
    extras["reward/joint_dev_hip"]       = dev_hip
    extras["reward/joint_dev_arms"]      = dev_arms
    extras["reward/joint_dev_hands"]     = dev_hands
    extras["reward/joint_dev_waist"]     = dev_waist
    extras["reward/total"]               = reward

    # Task diagnostics — milestone progression and geometry.
    extras["task/dist_to_box"]      = (env.robot.data.root_pos_w[:, :2] - box_pos_w[:, :2]).norm(dim=-1)
    extras["task/dist_to_target"]   = (env.robot.data.root_pos_w[:, :2] - target_pos_w[:, :2]).norm(dim=-1)
    extras["task/box_height_z"]     = box_pos_w[:, 2]
    extras["task/box_xy_err"]       = box_xy_err
    extras["task/cmd_lin_x"]        = env._commands_dict["base_velocity"][:, 0]
    extras["task/cmd_lin_y"]        = env._commands_dict["base_velocity"][:, 1]
    extras["task/cmd_ang_z"]        = env._commands_dict["base_velocity"][:, 2]
    extras["task/lifted_now"]       = lifted_now.float()
    extras["task/lifted"]           = env._lift_achieved.float()
    extras["task/bimanual_touched"] = env._bimanual_contact_achieved.float()
    extras["task/success_steps"]    = env._success_step_count.float()

    return reward
