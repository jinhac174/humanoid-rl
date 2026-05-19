"""Velocity-tracking DirectRLEnv — direct-style port of the IsaacLab
``Isaac-Velocity-Flat-G1-v0`` task.

The env owns:

* ``self._commands``        — (N, 3) lin_x, lin_y, ang_z velocity command
                              per env, resampled every ``cmd_resampling_time_s``.
* ``self._actions``,
  ``self._prev_actions``    — last applied action and the one before it
                              (for action-rate penalty + obs's last_action term).
* ``self._joint_targets``   — most recent target position commanded into the
                              implicit actuators.
* ``self._push_timer``      — countdown for the optional periodic push event.

It also exposes three tiny "manager" shims on itself so the IsaacLab
``mdp`` reward / observation functions (which were written against
``ManagerBasedRLEnv``) work unchanged. See ``rewards.py`` for the orchestration
and ``observations.py`` for the observation assembly.
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from assets.robots.g1_cfg import LOCOMOTION_ACTUATED_JOINTS

from . import events as event_fn
from . import observations as obs_fn
from . import rewards as rew_fn
from . import terminations as term_fn
from .env_cfg import (
    LEFT_FOOT_BODY,
    RIGHT_FOOT_BODY,
    TORSO_BODY,
    VelocityTrackingEnvCfg,
)


# ----- Manager shims -----------------------------------------------------------
# IsaacLab's ``mdp.*`` reward and observation helpers expect a ManagerBasedRLEnv
# with a ``command_manager``, ``action_manager``, and ``termination_manager``.
# DirectRLEnv has none of these. Instead of forking each helper, we expose
# minimal duck-typed shims on the env object so the helpers' attribute lookups
# resolve correctly.

class _CommandShim:
    """Returns whatever tensor lives at ``env._commands_dict[name]``."""
    def __init__(self, env: "VelocityTrackingEnv"):
        self._env = env
    def get_command(self, name: str) -> torch.Tensor:
        return self._env._commands_dict[name]


class _ActionShim:
    """Holds ``action`` (current step) and ``prev_action`` (one step ago).

    Also exposes ``total_action_dim`` so ``isaaclab_rl.rsl_rl.RslRlVecEnvWrapper``
    (which assumes a manager-based env) accepts a DirectRLEnv with this shim
    in place. The wrapper queries the attribute once at init.
    """
    def __init__(self, env: "VelocityTrackingEnv"):
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


class _TerminationShim:
    """Exposes ``terminated`` (bool tensor) for ``mdp.is_terminated`` reward."""
    def __init__(self, env: "VelocityTrackingEnv"):
        self._env = env
    @property
    def terminated(self) -> torch.Tensor:
        # Match ManagerBasedRLEnv: ``terminated`` excludes time-out, so it's
        # ``reset_terminated`` (truncations don't count as task termination).
        return self._env._terminated_buf


# ----- Env ---------------------------------------------------------------------
class VelocityTrackingEnv(DirectRLEnv):
    cfg: VelocityTrackingEnvCfg

    def __init__(
        self,
        cfg: VelocityTrackingEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # -- Asset handles --
        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene["contact_forces"]

        # -- Actuated joints (43 DoF, locomotion order: legs|waist|arms|hands) --
        self._actuated_joint_ids, _ = self.robot.find_joints(
            LOCOMOTION_ACTUATED_JOINTS, preserve_order=True
        )
        assert len(self._actuated_joint_ids) == 43, (
            f"expected 43 actuated joints, got {len(self._actuated_joint_ids)}"
        )
        # IsaacLab convention: target = action * cfg.action_scale + default_pos.
        # Flat scalar (0.5 rad by default) applied uniformly to every joint,
        # matching ``mdp.JointPositionActionCfg(scale=0.5, use_default_offset=True)``
        # in IsaacLab's reference velocity task. We deliberately DO NOT use
        # the per-joint LOCOMOTION_ACTION_SCALE here -- that derivation is for
        # delta-PD control (``target += action * scale``) which is what the
        # reorient manipulation task uses; on top of an offset-from-default
        # action it would crush leg authority by half-to-thirteenth.
        self._action_scale = float(cfg.action_scale)

        # -- Body ids (feet for slide/air-time, torso for illegal contact) --
        self._feet_body_ids = torch.tensor(
            [
                self.robot.find_bodies(LEFT_FOOT_BODY)[0][0],
                self.robot.find_bodies(RIGHT_FOOT_BODY)[0][0],
            ],
            dtype=torch.long, device=self.device,
        )
        # Contact-sensor body indices live in a separate index space (the
        # sensor's own body list). Resolve them via the sensor's API.
        feet_sensor_ids, _ = self.contact_sensor.find_bodies(
            [LEFT_FOOT_BODY, RIGHT_FOOT_BODY]
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

        # -- Velocity commands --
        # ``base_velocity``: (lin_x, lin_y, ang_z) per env, exposed via the
        # _CommandShim under name "base_velocity" for IsaacLab's mdp rewards.
        # ``heading``: target yaw per env when cmd_heading_command=True. ang_z
        # is then re-derived each step in _pre_physics_step via P-control on
        # (target_heading - current_heading) — same as IsaacLab's
        # UniformVelocityCommand(heading_command=True).
        self._commands_dict = {
            "base_velocity": torch.zeros(self.num_envs, 3, device=self.device),
            "heading":       torch.zeros(self.num_envs, device=self.device),
        }
        # Per-env timer that triggers a fresh resample.
        self._cmd_resample_steps = max(
            int(self.cfg.cmd_resampling_time_s / self.step_dt), 1
        )
        self._cmd_steps_until_resample = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )

        # -- Push event --
        self._push_step_interval = max(
            int(self.cfg.push_interval_s / self.step_dt), 1
        )
        self._push_steps_remaining = torch.full(
            (self.num_envs,), self._push_step_interval,
            dtype=torch.int32, device=self.device,
        )

        # -- Termination buffer (used by reward shim and dones) --
        self._terminated_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # -- Manager shims (duck-typed surfaces for IsaacLab mdp helpers) --
        self.command_manager     = _CommandShim(self)
        self.action_manager      = _ActionShim(self)
        self.termination_manager = _TerminationShim(self)

        # -- Pre-resolve SceneEntityCfg objects once (joint_ids/body_ids are
        # filled by .resolve(scene)). These are passed into mdp.* functions in
        # rewards.py and observations.py so we don't pay the lookup per step.
        self._scfg = self._build_scene_entity_cfgs()

        # Sample initial commands and stagger their resample timers so they
        # don't all flip on the same global step.
        event_fn.sample_velocity_commands(self, env_ids=None)
        self._cmd_steps_until_resample = torch.randint(
            0, self._cmd_resample_steps,
            (self.num_envs,), dtype=torch.int32, device=self.device,
        )

    # -------------------------------------------------------------------
    # SceneEntityCfg pre-resolution
    # -------------------------------------------------------------------
    def _build_scene_entity_cfgs(self) -> dict:
        """Build the joint/body cfg objects that ``mdp.*`` helpers expect.

        ``SceneEntityCfg.resolve(scene)`` populates ``.joint_ids`` and
        ``.body_ids`` from the joint/body name patterns. We do this once at
        ``__init__`` time so the per-step reward calls are pure index lookups.
        """
        cfgs = {
            # Whole robot.
            "robot_all": SceneEntityCfg("robot", joint_names=".*"),
            # Joint groups for joint_deviation_l1 penalties.
            "hip_yaw_roll":   SceneEntityCfg(
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
            # Matches IsaacLab G1Flat: dof_torques_l2 and dof_acc_l2 are
            # applied to hips + knees only (NOT ankles). Penalizing ankle
            # torques over-suppresses the small ankle commands needed to
            # stabilize a stride.
            "hips_and_knees": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_.*_joint", ".*_knee_joint"],
            ),
            # Contact sensor cfgs.
            "feet_contact":  SceneEntityCfg(
                "contact_forces", body_names=[LEFT_FOOT_BODY, RIGHT_FOOT_BODY]
            ),
            "torso_contact": SceneEntityCfg(
                "contact_forces", body_names=TORSO_BODY,
            ),
            "feet_robot": SceneEntityCfg(
                "robot", body_names=[LEFT_FOOT_BODY, RIGHT_FOOT_BODY]
            ),
        }
        for c in cfgs.values():
            c.resolve(self.scene)
        return cfgs

    # -------------------------------------------------------------------
    # Step pipeline
    # -------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Heading-based ang_z (IsaacLab parity). When cmd_heading_command is
        # True, this rewrites _commands_dict["base_velocity"][:, 2] based on
        # the current heading error toward the per-env target heading. Runs
        # BEFORE obs/reward so they see a consistent derived ang_z.
        event_fn.update_heading_command_ang_z(self)

        # Track previous action for action-rate reward and last_action obs.
        self._prev_actions = self._actions.clone()
        self._actions = actions.clamp(-1.0, 1.0)

        # IsaacLab convention: target = action * cfg.action_scale + default_pos.
        # Flat scale across joints (matches mdp.JointPositionActionCfg(scale=0.5,
        # use_default_offset=True) in the reference task).
        default_pos = self.robot.data.default_joint_pos[:, self._actuated_joint_ids]
        self._joint_targets = self._actions * self._action_scale + default_pos
        lo = self.robot.data.soft_joint_pos_limits[:, self._actuated_joint_ids, 0]
        hi = self.robot.data.soft_joint_pos_limits[:, self._actuated_joint_ids, 1]
        self._joint_targets = self._joint_targets.clamp(lo, hi)

        # Tick the per-env timers; resample expired commands.
        self._cmd_steps_until_resample -= 1
        resample_ids = (self._cmd_steps_until_resample <= 0).nonzero(as_tuple=False).squeeze(-1)
        if resample_ids.numel() > 0:
            event_fn.sample_velocity_commands(self, resample_ids)
            self._cmd_steps_until_resample[resample_ids] = self._cmd_resample_steps

        # Periodic push (only if enabled).
        if self.cfg.push_enabled:
            self._push_steps_remaining -= 1
            push_ids = (self._push_steps_remaining <= 0).nonzero(as_tuple=False).squeeze(-1)
            if push_ids.numel() > 0:
                event_fn.push_robot(self, push_ids)
                self._push_steps_remaining[push_ids] = self._push_step_interval

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(
            self._joint_targets, joint_ids=self._actuated_joint_ids,
        )

    def _get_observations(self) -> dict:
        return {"policy": obs_fn.get_observations(self)}

    def _get_rewards(self) -> torch.Tensor:
        return rew_fn.compute_reward(self)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = term_fn.compute_dones(self)
        # Cache for the reward shim's ``termination_manager.terminated``.
        self._terminated_buf = terminated
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        event_fn.reset_robot(self, env_ids)
        # Reset task buffers for these envs.
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._push_steps_remaining[env_ids] = self._push_step_interval
        # Resample a fresh command on reset and stagger its timer.
        event_fn.sample_velocity_commands(self, env_ids)
        self._cmd_steps_until_resample[env_ids] = self._cmd_resample_steps
