import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor

from manipulation.robots.g1 import (
    ACTUATED_JOINTS,
    LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS,
    LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS,
    LEFT_PALM_BODY, RIGHT_PALM_BODY,
    ACTION_SCALE,
)
from manipulation.tasks.can_push.env_cfg import CanPushEnvCfg
from manipulation.tasks.can_push import observations as obs_fn
from manipulation.tasks.can_push import rewards as rew_fn
from manipulation.tasks.can_push import terminations as term_fn
from manipulation.tasks.can_push import events as event_fn


class CanPushEnv(DirectRLEnv):

    cfg: CanPushEnvCfg

    def __init__(self, cfg: CanPushEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ── Body indices ──────────────────────────────────────────────────────
        self.left_palm_idx  = self.robot.find_bodies(LEFT_PALM_BODY)[0][0]
        self.right_palm_idx = self.robot.find_bodies(RIGHT_PALM_BODY)[0][0]

        # ── Joint ids and slices ──────────────────────────────────────────────
        self.left_arm_joint_ids   = list(self.robot.find_joints(LEFT_ARM_JOINTS)[0])
        self.right_arm_joint_ids  = list(self.robot.find_joints(RIGHT_ARM_JOINTS)[0])
        self.left_hand_joint_ids  = list(self.robot.find_joints(LEFT_HAND_JOINTS)[0])
        self.right_hand_joint_ids = list(self.robot.find_joints(RIGHT_HAND_JOINTS)[0])
        self.actuated_joint_ids   = list(self.robot.find_joints(ACTUATED_JOINTS)[0])

        n_la = len(self.left_arm_joint_ids)
        n_ra = len(self.right_arm_joint_ids)
        n_lh = len(self.left_hand_joint_ids)
        n_rh = len(self.right_hand_joint_ids)
        act_dim = n_la + n_ra + n_lh + n_rh

        s = 0
        self.left_arm_slice   = slice(s, s + n_la); s += n_la
        self.right_arm_slice  = slice(s, s + n_ra); s += n_ra
        self.left_hand_slice  = slice(s, s + n_lh); s += n_lh
        self.right_hand_slice = slice(s, s + n_rh)

        # ── Action scale ──────────────────────────────────────────────────────
        self.action_scale = torch.tensor(
            ACTION_SCALE, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # ── Joint limits ──────────────────────────────────────────────────────
        joint_limits = getattr(self.robot.data, "soft_joint_pos_limits", None)
        if joint_limits is None:
            joint_limits = self.robot.data.joint_pos_limits
        if joint_limits.dim() == 3:
            lower = joint_limits[0, self.actuated_joint_ids, 0]
            upper = joint_limits[0, self.actuated_joint_ids, 1]
        else:
            lower = joint_limits[self.actuated_joint_ids, 0]
            upper = joint_limits[self.actuated_joint_ids, 1]
        self.actuated_joint_lower = lower.to(self.device).unsqueeze(0)
        self.actuated_joint_upper = upper.to(self.device).unsqueeze(0)

        # ── Target position -- fixed, read from scene config ───────────────────
        # Target is a static visual USD. It never moves so we store its
        # world position as a constant tensor. No runtime asset interaction needed.
        tp = cfg.scene.target.init_state.pos
        self.target_pos_w = torch.tensor(
            [tp[0], tp[1], tp[2]], dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(cfg.scene.num_envs, 1)  # (num_envs, 3)

        # ── Per-env buffers ───────────────────────────────────────────────────
        num_envs = cfg.scene.num_envs
        self.nominal_joint_pos  = torch.zeros(num_envs, act_dim, device=self.device)
        self.target_joint_pos   = torch.zeros(num_envs, act_dim, device=self.device)
        self.actions            = torch.zeros(num_envs, act_dim, device=self.device)
        self.prev_actions       = torch.zeros(num_envs, act_dim, device=self.device)
        self.can_spawn_z        = torch.zeros(num_envs, device=self.device)
        self.prev_left_dist     = torch.zeros(num_envs, device=self.device)
        self.prev_can_to_target = torch.zeros(num_envs, device=self.device)

    # ── Scene ─────────────────────────────────────────────────────────────────

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.scene.robot)
        self.can   = RigidObject(self.cfg.scene.can)
        # target is spawned by InteractiveScene automatically via AssetBaseCfg
        # we do not instantiate it here -- position is stored as a constant tensor

        self.left_hand_contact  = ContactSensor(self.cfg.scene.left_hand_contact)
        self.right_hand_contact = ContactSensor(self.cfg.scene.right_hand_contact)

        self.scene.articulations["robot"]        = self.robot
        self.scene.rigid_objects["can"]          = self.can
        self.scene.sensors["left_hand_contact"]  = self.left_hand_contact
        self.scene.sensors["right_hand_contact"] = self.right_hand_contact

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        event_fn.reset_robot(self, env_ids)
        event_fn.reset_can(self, env_ids)
        event_fn.reset_buffers(self, env_ids)
        
    # ── Observations ──────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        return {"policy": obs_fn.get_obs(self, self.actuated_joint_ids)}

    # ── Actions ───────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-1.0, 1.0)
        delta = self.actions * self.action_scale
        self.target_joint_pos = torch.max(
            torch.min(self.target_joint_pos + delta, self.actuated_joint_upper),
            self.actuated_joint_lower,
        )

    def _apply_action(self):
        self.robot.set_joint_position_target(
            self.target_joint_pos,
            joint_ids=self.actuated_joint_ids,
        )

    # ── Rewards ───────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg

        r_approach = rew_fn.reward_approach(self)
        r_push     = rew_fn.reward_push(self)
        r_success  = rew_fn.reward_success(self)
        p_drop        = rew_fn.penalty_drop(self)
        p_right_idle  = rew_fn.penalty_right_idle(self)
        p_jlimits     = rew_fn.penalty_joint_limits(self)
        p_action_rate = rew_fn.penalty_action_rate(self)
        p_joint_vel   = rew_fn.penalty_joint_vel(self)

        total = (
            cfg.reward_approach_weight  * r_approach
            + cfg.reward_push_weight    * r_push
            + cfg.reward_success_weight * r_success
            - cfg.penalty_drop_weight          * p_drop
            - cfg.penalty_right_idle_weight    * p_right_idle
            - cfg.penalty_joint_limits_weight  * p_jlimits
            - cfg.penalty_action_rate_weight   * p_action_rate
            - cfg.penalty_joint_vel_weight     * p_joint_vel
        )

        # update progress buffers after reward computation
        left_palm = self.robot.data.body_pos_w[:, self.left_palm_idx, :]
        can_pos   = self.can.data.root_pos_w

        self.prev_left_dist[:]      = torch.norm(left_palm - can_pos, dim=-1).detach()
        self.prev_can_to_target[:]  = torch.norm(
            can_pos[:, :2] - self.target_pos_w[:, :2], dim=-1
        ).detach()
        self.prev_actions[:] = self.actions.detach()

        self.extras.setdefault("log", {})
        self.extras["log"].update({
            "reward/approach":    r_approach.mean().item(),
            "reward/push":        r_push.mean().item(),
            "reward/success":     r_success.mean().item(),
            "penalty/drop":       p_drop.mean().item(),
            "penalty/right_idle": p_right_idle.mean().item(),
            "penalty/jlimits":    p_jlimits.mean().item(),
            "penalty/action_rate":p_action_rate.mean().item(),
            "penalty/joint_vel":  p_joint_vel.mean().item(),
            "state/success_rate": r_success.mean().item(),
        })

        return total

    # ── Terminations ──────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        success   = term_fn.termination_success(self)
        dropped   = term_fn.termination_can_dropped(self)
        timed_out = term_fn.termination_timeout(self)
        terminated = success | dropped

        self.extras.setdefault("log", {})
        self.extras["log"].update({
            "done/success_rate":   success.float().mean().item(),
            "done/drop_rate":      dropped.float().mean().item(),
            "done/timeout_rate":   timed_out.float().mean().item(),
            "done/terminated_rate":terminated.float().mean().item(),
        })

        # Per-env success indicator for trainer episode stats. Flat key (not
        # under extras["log"]) so the trainer reads it as a tensor. On steps
        # where an env terminates via drop, success=False → 0.0; via success,
        # success=True → 1.0. Excluded from rollout mean aggregation.
        self.extras["task_episode_success_per_env"] = success.float()

        return terminated, timed_out