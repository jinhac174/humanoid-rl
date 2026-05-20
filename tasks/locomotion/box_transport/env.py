"""Box-transport DirectRLEnv.

Loco-manipulation: G1 walks to a box on the start table, grips it
bimanually, walks to the target table, and places it. Same 43-DoF action
space and proprioception layout as ``Velocity-Tracking`` so a locomotion
checkpoint can be warm-started (the extra observation dims for box/target/
palm tensors come AFTER the 141 locomotion dims — see ``observations.py``).

The env owns:

* ``self._commands_dict["base_velocity"]``  — (N, 3) per-env velocity
                                              command derived each step
                                              from the task phase (toward
                                              box if not lifted, toward
                                              target if lifted). Plays
                                              the role of the locomotion
                                              env's resampled velocity
                                              command so the warm-started
                                              policy keeps seeing the
                                              same kind of input.
* ``self._actions``, ``self._prev_actions`` — last + previous action
                                              (action-rate penalty).
* ``self._joint_targets``                   — most recent target sent
                                              into the implicit actuators.
* ``self._target_pos_w``                    — (N, 3) target xy on target
                                              table top, sampled at
                                              reset.
* Milestone latches (``_bimanual_contact_achieved``, ``_lift_achieved``),
  a success counter (``_success_step_count``), and termination cause
  buffers (``_fell_buf``, ``_box_dropped_buf``, ``_success_buf``) which
  ``terminations.py`` populates and ``rewards.py`` reads.
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from assets.robots.g1_cfg import LOCOMOTION_ACTUATED_JOINTS

from . import events as event_fn
from . import observations as obs_fn
from . import rewards as rew_fn
from . import terminations as term_fn
from .env_cfg import (
    BOX_LIFT_Z,
    BOX_REST_Z,
    BOX_SPAWN_POS,
    GRIP_DISTANCE,
    LEFT_FOOT_BODY,
    LEFT_PALM_BODY,
    RIGHT_FOOT_BODY,
    RIGHT_PALM_BODY,
    TARGET_ANCHOR_POS,
    TARGET_Z,
    TORSO_BODY,
    BoxTransportEnvCfg,
)


# ----- Manager shims (IsaacLab mdp.* helpers expect a manager-based env) ------
# ``command_manager`` — needed by the phase-1 velocity-tracking reward
# helpers (track_lin_vel_*, track_ang_vel_*, feet_air_time_positive_biped),
# which we re-use here so the policy gets the same locomotion-shaped reward
# signal it was warm-started on.
# ``action_manager`` — needed by mdp.action_rate_l2 (locomotion regularizer).
class _CommandShim:
    """Returns whatever tensor lives at ``env._commands_dict[name]``."""
    def __init__(self, env: "BoxTransportEnv"):
        self._env = env
    def get_command(self, name: str) -> torch.Tensor:
        return self._env._commands_dict[name]


class _ActionShim:
    def __init__(self, env: "BoxTransportEnv"):
        self._env = env
    @property
    def action(self) -> torch.Tensor:
        return self._env._actions
    @property
    def prev_action(self) -> torch.Tensor:
        return self._env._prev_actions
    @property
    def total_action_dim(self) -> int:
        return int(self._env._actions.shape[1])


# ----- Env -------------------------------------------------------------------
class BoxTransportEnv(DirectRLEnv):
    cfg: BoxTransportEnvCfg

    def __init__(
        self,
        cfg: BoxTransportEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # -- Asset handles --
        self.robot: Articulation = self.scene["robot"]
        self.box:   RigidObject  = self.scene["box"]
        self.contact_sensor: ContactSensor = self.scene["contact_forces"]

        # -- Actuated joints (43 DoF, locomotion order: legs|waist|arms|hands) --
        self._actuated_joint_ids, _ = self.robot.find_joints(
            LOCOMOTION_ACTUATED_JOINTS, preserve_order=True,
        )
        assert len(self._actuated_joint_ids) == 43

        self._action_scale = float(cfg.action_scale)

        # -- Body ids (palms, torso) — articulation body index space, used
        #    to read body_pos_w / body_quat_w. --
        self._left_palm_body_id  = self.robot.find_bodies(LEFT_PALM_BODY)[0][0]
        self._right_palm_body_id = self.robot.find_bodies(RIGHT_PALM_BODY)[0][0]
        self._torso_body_id      = self.robot.find_bodies(TORSO_BODY)[0][0]

        feet_sensor_ids, _ = self.contact_sensor.find_bodies(
            [LEFT_FOOT_BODY, RIGHT_FOOT_BODY],
        )
        self._feet_sensor_ids = torch.tensor(
            feet_sensor_ids, dtype=torch.long, device=self.device,
        )
        torso_sensor_ids, _ = self.contact_sensor.find_bodies(TORSO_BODY)
        self._torso_sensor_ids = torch.tensor(
            torso_sensor_ids, dtype=torch.long, device=self.device,
        )

        # -- Control buffers --
        self._actions      = torch.zeros(self.num_envs, 43, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 43, device=self.device)
        self._joint_targets = self.robot.data.default_joint_pos[
            :, self._actuated_joint_ids
        ].clone()

        # -- Auto-derived velocity command (filled by events.update_autocmd) --
        self._commands_dict = {
            "base_velocity": torch.zeros(self.num_envs, 3, device=self.device),
        }

        # -- Per-env target pose (filled by events.reset_target at episode start) --
        # target_anchor_w is the world-frame anchor near the front edge of
        # each env's target table (env_origins + TARGET_ANCHOR_POS); the
        # per-episode target is sampled in a small box around it.
        env_origins = self.scene.env_origins                         # (N, 3)
        self._target_anchor_w = env_origins + torch.tensor(
            TARGET_ANCHOR_POS, device=self.device, dtype=env_origins.dtype,
        )
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_z = float(TARGET_Z)
        self._box_lift_z = float(BOX_LIFT_Z)
        self._box_rest_z = float(BOX_REST_Z)
        self._grip_distance = float(GRIP_DISTANCE)
        # Used by events.reset_box (env-local box spawn position).
        self._box_spawn_pos_local = torch.tensor(
            BOX_SPAWN_POS, dtype=torch.float32, device=self.device,
        )

        # -- Milestone latches (reset in _reset_idx) --
        self._bimanual_contact_achieved = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )
        self._lift_achieved = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )
        # -- Success counter — consecutive frames the box is on target --
        self._success_step_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
        )
        # -- Cause-of-termination buffers (filled by terminations.compute_dones) --
        self._fell_buf        = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._box_dropped_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success_buf     = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # -- Manager shims (velocity-tracking rewards + action-rate use these) --
        self.command_manager = _CommandShim(self)
        self.action_manager  = _ActionShim(self)

        # -- Pre-resolved SceneEntityCfg objects (mdp.* helpers want these) --
        self._scfg = self._build_scene_entity_cfgs()

        # -- Sample initial targets so reset_idx isn't strictly required for env 0
        #    to have a valid target on the very first obs read. --
        event_fn.reset_target(self, torch.arange(self.num_envs, device=self.device))

    # ------------------------------------------------------------------
    def _build_scene_entity_cfgs(self) -> dict:
        cfgs = {
            "robot_all": SceneEntityCfg("robot", joint_names=".*"),
            "hip_yaw_roll": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"],
            ),
            "ankles": SceneEntityCfg(
                "robot",
                joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            ),
            "arms": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",   ".*_elbow_joint",
                    ".*_wrist_roll_joint",     ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
            ),
            "hands": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hand_index_0_joint",  ".*_hand_index_1_joint",
                    ".*_hand_middle_0_joint", ".*_hand_middle_1_joint",
                    ".*_hand_thumb_0_joint",  ".*_hand_thumb_1_joint",
                    ".*_hand_thumb_2_joint",
                ],
            ),
            "waist": SceneEntityCfg("robot", joint_names="waist_.*_joint"),
            "hips_and_knees": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_.*_joint", ".*_knee_joint"],
            ),
            "feet_contact": SceneEntityCfg(
                "contact_forces", body_names=[LEFT_FOOT_BODY, RIGHT_FOOT_BODY],
            ),
            "feet_robot": SceneEntityCfg(
                "robot", body_names=[LEFT_FOOT_BODY, RIGHT_FOOT_BODY],
            ),
        }
        for c in cfgs.values():
            c.resolve(self.scene)
        return cfgs

    # ------------------------------------------------------------------
    # Step pipeline
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Auto-derive the velocity command from the task phase BEFORE the obs
        # / reward step reads it.
        event_fn.update_autocmd(self)

        self._prev_actions = self._actions.clone()
        self._actions = actions.clamp(-1.0, 1.0)

        # IsaacLab convention: target = action * scale + default_pos.
        default_pos = self.robot.data.default_joint_pos[:, self._actuated_joint_ids]
        self._joint_targets = self._actions * self._action_scale + default_pos
        lo = self.robot.data.soft_joint_pos_limits[:, self._actuated_joint_ids, 0]
        hi = self.robot.data.soft_joint_pos_limits[:, self._actuated_joint_ids, 1]
        self._joint_targets = self._joint_targets.clamp(lo, hi)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(
            self._joint_targets, joint_ids=self._actuated_joint_ids,
        )

    def _get_observations(self) -> dict:
        return {"policy": obs_fn.get_observations(self)}

    def _get_rewards(self) -> torch.Tensor:
        return rew_fn.compute_reward(self)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return term_fn.compute_dones(self)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        event_fn.reset_robot(self, env_ids)
        event_fn.reset_box(self, env_ids)
        event_fn.reset_target(self, env_ids)
        # Clear control + task buffers for these envs.
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._bimanual_contact_achieved[env_ids] = False
        self._lift_achieved[env_ids] = False
        self._success_step_count[env_ids] = 0
