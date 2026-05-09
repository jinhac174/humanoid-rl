"""
Observations for the reorient task.

Split into two functions:

    compute_task_state(env)
        Updates all per-step task state tensors on `env`. Called from
        ReorientEnv._get_observations and ReorientEnv._get_dones so the
        state is fresh for both rewards and termination checks. Populates:
            env.palm_center_pos          (N, 2, 3)
            env.fingertip_pos            (N, 2, 3, 3)
            env.curr_fingertip_distances (N, 2, 3)
            env.closest_fingertip_dist   (N, 2, 3)   -- lazy init from -1
            env.obj_keypoint_pos         (N, K, 3)   -- K = env.num_keypoints
            env.goal_keypoint_pos        (N, K, 3)
            env.keypoints_max_dist       (N,)
            env.closest_keypoint_max_dist (N,)       -- lazy init from -1

        env.lifted_object is updated in rewards._lifting_reward (rising-edge
        detection) -- not here.

    get_observations(env)
        Reads the task state and assembles the (96 + 3*K)-dim policy obs.
        With the default K=8 keypoints the obs is 120-dim:

            [0:28]    joint_pos (actuated, raw radians)
            [28:56]   joint_vel (actuated, raw)
            [56:59]   object_pos  (robot-relative)
            [59:63]   object_quat (world frame, wxyz)
            [63:66]   goal_pos    (robot-relative)
            [66:70]   goal_quat   (world frame, wxyz)
            [70:73]   left_palm_pos  (robot-relative)
            [73:76]   right_palm_pos (robot-relative)
            [76:85]   left fingertip pos × 3 (robot-relative)
            [85:94]   right fingertip pos × 3 (robot-relative)
            [94 : 94+3K]   K keypoint deltas obj→goal (world frame)
            [94+3K : 95+3K] lifted_object as float
            [95+3K : 96+3K] near_goal_steps / success_steps, clamped to [0, 1]
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from .env import ReorientEnv


def compute_task_state(env: "ReorientEnv") -> None:
    """Update all per-step task state tensors from the current sim state."""
    num_envs = env.num_envs
    num_keypoints = env.num_keypoints  # 8 by default; see ReorientEnvCfg.num_keypoints

    # ── Object and goal poses (world frame) ──────────────────────────────────
    object_pos = env.cuboid.data.root_pos_w        # (N, 3)
    object_quat = env.cuboid.data.root_quat_w      # (N, 4) wxyz
    goal_pos = env.goal.data.root_pos_w            # (N, 3)
    goal_quat = env.goal.data.root_quat_w          # (N, 4)

    # ── Palm centers ─────────────────────────────────────────────────────────
    # G1 Dex3 palm link frames sit at the approximate palm center already, so
    # no offset is applied in Phase B. If the donor fingertip/palm offsets
    # matter later we'll measure them from the USD in Phase D.
    left_palm = env.robot.data.body_pos_w[:, env.left_palm_body_id, :]    # (N, 3)
    right_palm = env.robot.data.body_pos_w[:, env.right_palm_body_id, :]  # (N, 3)
    env.palm_center_pos = torch.stack([left_palm, right_palm], dim=1)     # (N, 2, 3)

    # ── Fingertip positions ──────────────────────────────────────────────────
    # fingertip_body_ids is shape (2, 3) = (arms, tips_per_hand). Flatten to
    # gather all 6 at once, then reshape back.
    flat_tip_ids = env.fingertip_body_ids.reshape(-1)                     # (6,)
    env.fingertip_pos = env.robot.data.body_pos_w[:, flat_tip_ids, :].view(
        num_envs, 2, 3, 3
    )                                                                     # (N, 2, 3, 3)

    # ── Fingertip → object distances ─────────────────────────────────────────
    obj_pos_b = object_pos.unsqueeze(1).unsqueeze(1)                      # (N, 1, 1, 3)
    env.curr_fingertip_distances = torch.norm(
        env.fingertip_pos - obj_pos_b, dim=-1
    )                                                                     # (N, 2, 3)

    # Lazy init: on the first step after reset, closest_fingertip_dist is -1;
    # replace with the current distances so "best-ever" starts at "now".
    lazy_mask = env.closest_fingertip_dist < 0
    env.closest_fingertip_dist = torch.where(
        lazy_mask, env.curr_fingertip_distances, env.closest_fingertip_dist
    )

    # ── Object and goal keypoints ────────────────────────────────────────────
    # obj_kp[i] = object_pos + quat_apply(object_quat, keypoint_offsets[i])
    # Broadcast: quat (N, 4, 4) × offsets (N, 4, 3) → (N, 4, 3)
    kp_offsets_batched = env.keypoint_offsets.unsqueeze(0).expand(
        num_envs, -1, -1
    )                                                                     # (N, 4, 3)
    obj_quat_b = object_quat.unsqueeze(1).expand(-1, num_keypoints, -1)   # (N, 4, 4)
    goal_quat_b = goal_quat.unsqueeze(1).expand(-1, num_keypoints, -1)    # (N, 4, 4)

    env.obj_keypoint_pos = object_pos.unsqueeze(1) + quat_apply(
        obj_quat_b, kp_offsets_batched
    )                                                                     # (N, 4, 3)
    env.goal_keypoint_pos = goal_pos.unsqueeze(1) + quat_apply(
        goal_quat_b, kp_offsets_batched
    )                                                                     # (N, 4, 3)

    # Max (L∞) keypoint distance between object and goal
    keypoint_deltas = env.obj_keypoint_pos - env.goal_keypoint_pos        # (N, 4, 3)
    keypoint_l2 = torch.norm(keypoint_deltas, dim=-1)                     # (N, 4)
    env.keypoints_max_dist = keypoint_l2.max(dim=-1).values               # (N,)

    # Lazy init same as closest_fingertip_dist.
    env.closest_keypoint_max_dist = torch.where(
        env.closest_keypoint_max_dist < 0,
        env.keypoints_max_dist,
        env.closest_keypoint_max_dist,
    )

    # NOTE: lifted_object is updated in rewards._lifting_reward, not here.
    # The donor needs the PRE-update value to detect the rising edge
    # (just_lifted_above_threshold = lifted_object_now & ~lifted_object_prev)
    # and fire the one-shot lift bonus. If we update the latch here, the
    # reward function sees a post-update value and the rising edge never fires.


def get_observations(env: "ReorientEnv") -> torch.Tensor:
    """Assemble the 108-dim policy observation from already-computed task state."""
    robot = env.robot
    robot_root_pos = robot.data.root_pos_w                          # (N, 3)

    joint_pos = robot.data.joint_pos[:, env.actuated_joint_ids]     # (N, 28)
    joint_vel = robot.data.joint_vel[:, env.actuated_joint_ids]     # (N, 28)

    object_pos_rel = env.cuboid.data.root_pos_w - robot_root_pos    # (N, 3)
    object_quat = env.cuboid.data.root_quat_w                       # (N, 4)
    goal_pos_rel = env.goal.data.root_pos_w - robot_root_pos        # (N, 3)
    goal_quat = env.goal.data.root_quat_w                           # (N, 4)

    palm_rel = env.palm_center_pos - robot_root_pos.unsqueeze(1)    # (N, 2, 3)
    fingertip_rel = env.fingertip_pos - robot_root_pos.unsqueeze(1).unsqueeze(1)
    # fingertip_rel shape (N, 2, 3, 3) → flatten tail to 18

    K = env.num_keypoints
    keypoints_rel_goal = env.obj_keypoint_pos - env.goal_keypoint_pos  # (N, K, 3)

    lifted = env.lifted_object.float().unsqueeze(-1)                    # (N, 1)
    near_goal_norm = (
        env.near_goal_steps.float() / max(1, env.cfg.success_steps)
    ).clamp(0.0, 1.0).unsqueeze(-1)                                     # (N, 1)

    return torch.cat(
        [
            joint_pos,                                 # 28
            joint_vel,                                 # 28
            object_pos_rel,                            # 3
            object_quat,                               # 4
            goal_pos_rel,                              # 3
            goal_quat,                                 # 4
            palm_rel.reshape(env.num_envs, 6),         # 6
            fingertip_rel.reshape(env.num_envs, 18),   # 18
            keypoints_rel_goal.reshape(env.num_envs, 3 * K),  # 3*K
            lifted,                                    # 1
            near_goal_norm,                            # 1
        ],
        dim=-1,
    )  # total = 96 + 3*K (default K=8 → 120)