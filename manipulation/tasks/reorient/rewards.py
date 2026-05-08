"""
Reward terms for the reorient task -- Phase C.

Direct port of SAPG's `compute_kuka_reward` and helpers from
`isaacgymenvs/tasks/allegro_kuka/allegro_kuka_two_arms.py`. The two-arms
variant has exactly five terms and NO action penalties, NO hand-too-far
reset, and NO drop penalty (drop only triggers an episode reset; the donor
sets fall_penalty=0 in the YAML).

    fingertip_delta_rew  potential-based shaping on fingertip→object distance
                         (always active, sum over both hands' 3 fingertips each)
    lifting_rew          continuous reward for raising the cube above its start
                         (clipped to [0, 0.5], stops once lifted)
    lift_bonus_rew       one-shot bonus on the rising edge when z_lift crosses
                         lifting_bonus_threshold
    keypoint_rew         potential-based shaping on max keypoint distance to
                         goal (gated by lifted_object)
    bonus_rew            success bonus, fires whenever near_goal is True

Final reward (donor formula):

    reward = distance_delta_rew_scale * fingertip_delta_rew
           + lifting_rew_scale * lifting_rew
           + lift_bonus_rew
           + keypoint_rew_scale * keypoint_rew
           + bonus_rew

Side effects (mutate env state, kept verbatim from donor):
    _lifting_reward          updates env.lifted_object (latch)
    _distance_delta_rewards  updates env.closest_fingertip_dist (running min)
    _keypoint_reward         updates env.closest_keypoint_max_dist (running min)

Read-only inputs (computed in observations.compute_task_state, called from
env._get_dones BEFORE this module runs):
    env.curr_fingertip_distances
    env.keypoints_max_dist
    env.cuboid.data.root_pos_w
    env.object_init_pos_w
    env.near_goal       (set in env._get_dones from keypoints_max_dist)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .env import ReorientEnv


def _lifting_reward(env: "ReorientEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Continuous lifting shaping + one-shot lift bonus on rising edge.

    Returns (lifting_rew, lift_bonus_rew), both shape (N,). Updates
    env.lifted_object as a side effect.
    """
    object_pos_z = env.cuboid.data.root_pos_w[:, 2]
    z_lift = 0.05 + object_pos_z - env.object_init_pos_w[:, 2]
    lifting_rew = torch.clip(z_lift, 0.0, 0.5)

    # Latch update + rising-edge detection. The "& ~prev_latch" check is
    # the whole reason lifted_object MUST be updated here, not in
    # compute_task_state -- we need the pre-update value to detect the edge.
    lifted_now = (z_lift > env.cfg.lifting_bonus_threshold) | env.lifted_object
    just_lifted = lifted_now & ~env.lifted_object
    lift_bonus_rew = env.cfg.lifting_bonus * just_lifted.float()

    # Stop continuous shaping once latched (donor: "lifting_rew *= ~lifted_object").
    lifting_rew = lifting_rew * (~lifted_now).float()

    env.lifted_object = lifted_now
    return lifting_rew, lift_bonus_rew


def _distance_delta_rewards(env: "ReorientEnv") -> torch.Tensor:
    """
    Potential-based shaping on fingertip→object distance.

    For each fingertip, reward the amount by which the fingertip got closer to
    the object than its best-ever distance this episode. Sum over both arms'
    fingertips.

    Note: the two-arms donor does NOT gate this by `~lifted_object` (the
    single-arm version does, but two-arms keeps shaping on so the second
    hand stays close to the object during reorientation).

    Side effect: updates env.closest_fingertip_dist to the running min.
    """
    deltas = env.closest_fingertip_dist - env.curr_fingertip_distances  # (N, 2, 3)

    # Update running min BEFORE clipping the deltas. This ordering matters:
    # the reward is the pre-clip delta (so regression → negative → clipped to
    # 0), but the running min should always track the actual smallest distance.
    env.closest_fingertip_dist = torch.minimum(
        env.closest_fingertip_dist, env.curr_fingertip_distances
    )

    deltas = torch.clip(deltas, 0.0, 10.0)  # only reward progress, not regression
    fingertip_delta_rew = deltas.sum(dim=-1).sum(dim=-1)  # sum over tips, then arms
    return fingertip_delta_rew


def _keypoint_reward(
    env: "ReorientEnv", lifted_object: torch.Tensor
) -> torch.Tensor:
    """
    Potential-based shaping on max keypoint distance, gated by lifted_object.

    Same shape as _distance_delta_rewards but on keypoints_max_dist (the L∞
    distance over the 4 keypoints between object and goal). The gating means
    the agent gets zero keypoint reward until it has lifted the cube -- this
    blocks the degenerate "slide on table" strategy.

    Side effect: updates env.closest_keypoint_max_dist to the running min.
    """
    deltas = env.closest_keypoint_max_dist - env.keypoints_max_dist  # (N,)
    env.closest_keypoint_max_dist = torch.minimum(
        env.closest_keypoint_max_dist, env.keypoints_max_dist
    )
    deltas = torch.clip(deltas, 0.0, 100.0)
    keypoint_rew = deltas * lifted_object.float()
    return keypoint_rew


def compute_reward(env: "ReorientEnv") -> torch.Tensor:
    """
    Five-term donor reward. All scale constants come from env.cfg, which mirrors
    AllegroKuka.yaml exactly:
        distance_delta_rew_scale = 50.0
        lifting_rew_scale        = 20.0
        lifting_bonus            = 300.0
        keypoint_rew_scale       = 200.0
        reach_goal_bonus         = 1000.0
        success_steps            = 1
    """
    # Order matters: _lifting_reward updates env.lifted_object, and the keypoint
    # reward gate reads the just-updated value (so the agent earns keypoint
    # reward on the very same step it crosses the lift threshold).
    lifting_rew, lift_bonus_rew = _lifting_reward(env)
    fingertip_delta_rew = _distance_delta_rewards(env)
    keypoint_rew = _keypoint_reward(env, env.lifted_object)

    bonus_rew = env.near_goal.float() * (
        env.cfg.reach_goal_bonus / env.cfg.success_steps
    )

    reward = (
        env.cfg.distance_delta_rew_scale * fingertip_delta_rew
        + env.cfg.lifting_rew_scale * lifting_rew
        + lift_bonus_rew
        + env.cfg.keypoint_rew_scale * keypoint_rew
        + bonus_rew
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    # PPOTrainer reads env.extras after every env.step, takes .mean() of any
    # tensor value, accumulates across rollout steps, and flushes to wandb.
    # The contract is FLAT keys (nested dicts are silently skipped). We pass
    # tensors of shape (N,) or larger; the trainer means over all dims.
    #
    # Naming: "reward/<term>_raw" = unscaled donor term (matches the donor's
    # "_unscaled" suffix). "reward/<term>" = scaled by the donor coefficient,
    # which is what actually contributes to the total. lift_bonus and bonus
    # are already at-scale in the donor (no separate _raw versions).
    extras = env.extras
    extras["reward/fingertip_delta_raw"] = fingertip_delta_rew
    extras["reward/fingertip_delta"] = (
        env.cfg.distance_delta_rew_scale * fingertip_delta_rew
    )
    extras["reward/lifting_raw"] = lifting_rew
    extras["reward/lifting"] = env.cfg.lifting_rew_scale * lifting_rew
    extras["reward/lift_bonus"] = lift_bonus_rew
    extras["reward/keypoint_raw"] = keypoint_rew
    extras["reward/keypoint"] = env.cfg.keypoint_rew_scale * keypoint_rew
    extras["reward/bonus"] = bonus_rew
    extras["reward/total"] = reward

    # Task diagnostics -- what's actually happening in the scene each step.
    extras["task/lifted_frac"] = env.lifted_object.float()
    extras["task/near_goal_frac"] = env.near_goal.float()
    extras["task/successes_cum"] = env.successes
    extras["task/keypoints_max_dist"] = env.keypoints_max_dist
    extras["task/fingertip_dist"] = env.curr_fingertip_distances
    extras["task/object_z"] = env.cuboid.data.root_pos_w[:, 2]

    return reward