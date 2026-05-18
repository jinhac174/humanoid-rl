"""Reward orchestration — milestone-based sparse-shaped design.

User-confirmed structure:

    + walk_to_box        (dense, low weight; only when not lifted)
    + bimanual_contact   (one-shot: both palms within GRIP_DISTANCE of box)
    + box_lifted         (one-shot: box.z first exceeds BOX_LIFT_Z)
    + walk_to_target     (dense; gated on lifted)
    + place_bonus        (continuous while box stably on target)
    - drop_penalty       (one-shot when box hits floor; via terminations.compute_dones)
    - termination        (-pen_termination when robot fell)
    + locomotion regularizers (kept, reduced — see env_cfg pen_* weights)

The one-shot milestone bonuses are gated by per-env latched flags
(``env._bimanual_contact_achieved``, ``env._lift_achieved``) so each fires
once per episode and isn't re-collected. Both flags are reset in
``BoxTransportEnv._reset_idx``.

Order of ops: ``_get_dones`` (terminations.py) runs first within DirectRLEnv,
so ``env._fell_buf`` / ``env._box_dropped_buf`` / ``env._success_buf`` are
fresh by the time this function reads them.
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

    # ── Geometry shortcuts ────────────────────────────────────────────
    box_pos_w  = env.box.data.root_pos_w
    robot_pos_w = env.robot.data.root_pos_w
    target_pos_w = env._target_pos_w
    l_palm_w = env.robot.data.body_pos_w[:, env._left_palm_body_id]
    r_palm_w = env.robot.data.body_pos_w[:, env._right_palm_body_id]

    # ── Manipulation: phase 1 — walk to box ────────────────────────────
    dist_to_box = (robot_pos_w[:, :2] - box_pos_w[:, :2]).norm(dim=-1)
    # Smaller = better. Use exp shaping with cfg-std for a smooth gradient.
    walk_to_box = torch.exp(-(dist_to_box ** 2) / (cfg.rew_walk_to_box_std ** 2))

    # Only pay this while NOT lifted (don't reward standing next to the box
    # forever after the lift milestone).
    lifted_now = (box_pos_w[:, 2] > env._box_lift_z)
    walk_to_box = torch.where(lifted_now, torch.zeros_like(walk_to_box), walk_to_box)

    # ── Manipulation: phase 1.5 — bimanual contact (one-shot) ──────────
    l_palm_dist = (l_palm_w - box_pos_w).norm(dim=-1)
    r_palm_dist = (r_palm_w - box_pos_w).norm(dim=-1)
    grip = float(env._grip_distance)
    both_palms_close = (l_palm_dist < grip) & (r_palm_dist < grip)
    # First-time achievement: fires ONLY on the step the flag transitions
    # from False → True.
    bimanual_event = both_palms_close & (~env._bimanual_contact_achieved)
    env._bimanual_contact_achieved = env._bimanual_contact_achieved | both_palms_close
    bimanual_contact = bimanual_event.float()

    # ── Manipulation: phase 2 — lift (one-shot) ───────────────────────
    lift_event = lifted_now & (~env._lift_achieved)
    env._lift_achieved = env._lift_achieved | lifted_now
    lift = lift_event.float()

    # ── Manipulation: phase 3 — walk to target (gated on lifted) ──────
    dist_to_target = (robot_pos_w[:, :2] - target_pos_w[:, :2]).norm(dim=-1)
    walk_to_target_raw = torch.exp(-(dist_to_target ** 2) / (cfg.rew_walk_to_target_std ** 2))
    walk_to_target = torch.where(
        env._lift_achieved, walk_to_target_raw, torch.zeros_like(walk_to_target_raw),
    )

    # ── Manipulation: phase 4 — place bonus ─────────────────────────────
    # Continuous while box xy within tol of target and on the target table.
    box_xy_err = (box_pos_w[:, :2] - target_pos_w[:, :2]).norm(dim=-1)
    near_xy = box_xy_err < cfg.rew_place_distance_tol
    near_z  = box_pos_w[:, 2] < (env._target_z + 0.10)
    placed = (near_xy & near_z).float()
    place_bonus = placed

    # ── Penalties (sparse milestones) ───────────────────────────────────
    drop = env._box_dropped_buf.float()
    term_robot = env._fell_buf.float()

    # ── Locomotion regularizers (reduced weights via cfg) ──────────────
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

    # ── Weighted sum ─────────────────────────────────────────────────
    reward = (
        + cfg.rew_walk_to_box      * walk_to_box
        + cfg.rew_bimanual_contact * bimanual_contact
        + cfg.rew_lift             * lift
        + cfg.rew_walk_to_target   * walk_to_target
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

    # ── Logging to extras (wandb-flattened by the trainer) ─────────────
    extras = env.extras
    extras["reward/walk_to_box"]      = walk_to_box
    extras["reward/bimanual_contact"] = bimanual_contact
    extras["reward/lift"]             = lift
    extras["reward/walk_to_target"]   = walk_to_target
    extras["reward/place_bonus"]      = place_bonus
    extras["reward/drop"]             = drop
    extras["reward/termination"]      = term_robot
    extras["reward/lin_vel_z"]        = lin_vel_z
    extras["reward/ang_vel_xy"]       = ang_vel_xy
    extras["reward/flat_orientation"] = flat_orient
    extras["reward/action_rate"]      = action_rate
    extras["reward/dof_torques"]      = dof_torques
    extras["reward/dof_acc"]          = dof_acc
    extras["reward/dof_pos_limits_ankle"] = pos_limits_ankle
    extras["reward/feet_slide"]       = feet_slide
    extras["reward/joint_dev_hip"]    = dev_hip
    extras["reward/joint_dev_arms"]   = dev_arms
    extras["reward/joint_dev_hands"]  = dev_hands
    extras["reward/joint_dev_waist"]  = dev_waist
    extras["reward/total"]            = reward

    # Task diagnostics — milestone progression and geometry.
    extras["task/dist_to_box"]      = dist_to_box
    extras["task/dist_to_target"]   = dist_to_target
    extras["task/box_height_z"]     = box_pos_w[:, 2]
    extras["task/box_xy_err"]       = box_xy_err
    extras["task/lifted"]           = env._lift_achieved.float()
    extras["task/bimanual_touched"] = env._bimanual_contact_achieved.float()
    extras["task/success_steps"]    = env._success_step_count.float()

    return reward
