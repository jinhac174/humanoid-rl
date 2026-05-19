"""Velocity-tracking env config (flat ground, G1 Dex3, 43-DoF).

This is the direct-style port of ``Isaac-Velocity-Flat-G1-v0``. Differences
from the IsaacLab manager-based reference:

    * Robot: our ``G1_FREE_CFG`` (43 DoF Dex3) vs IsaacLab's ``G1_MINIMAL_CFG``
      (23 DoF, Inspire hand).
    * Terrain: flat plane (rough terrain + height scanner are deferred to a
      v2; once flat works we add them).
    * No managers — observations/rewards/terminations/events are computed
      directly inside :class:`VelocityTrackingEnv`.

Reward weights, command ranges, push intervals, etc. are exposed as fields
on this cfg so they can be overridden from ``configs/task/velocity_tracking.yaml``.
"""
from __future__ import annotations

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from gymnasium.spaces import Box

from assets.robots.g1_cfg import G1_FREE_CFG


# --- Body name constants --------------------------------------------------
ROOT_BODY = "pelvis"
TORSO_BODY = "torso_link"
LEFT_FOOT_BODY = "left_ankle_roll_link"
RIGHT_FOOT_BODY = "right_ankle_roll_link"
ILLEGAL_CONTACT_BODIES = ("torso_link", "pelvis", ".*_hip_.*_link", ".*_knee_link")


# --- Scene ---------------------------------------------------------------
@configclass
class VelocityTrackingSceneCfg(InteractiveSceneCfg):
    """G1 free-base + flat plane + contact sensors + sky light."""

    # Flat ground (we add the rough generator later if needed).
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Robot (free-base, squat init pose, all 43 DoF actuated).
    robot = G1_FREE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Contact sensors over every body — used for foot air-time reward and
    # illegal-contact termination on the torso.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # Lights.
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(1.0, 1.0, 1.0)),
    )


# --- Env cfg -------------------------------------------------------------
@configclass
class VelocityTrackingEnvCfg(DirectRLEnvCfg):
    scene: VelocityTrackingSceneCfg = VelocityTrackingSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 200,                       # 200 Hz physics
        render_interval=4,                # render every 4 phys steps (= once per control step)
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(gpu_max_rigid_patch_count=10 * 2 ** 15),
    )

    decimation: int = 4                   # 200 Hz / 4 = 50 Hz control
    episode_length_s: float = 20.0

    # ── Spaces ───────────────────────────────────────────────────────────
    # Obs (concatenated): base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
    #                   + velocity_commands(3) + joint_pos_rel(43) + joint_vel_rel(43)
    #                   + last_action(43) = 141
    observation_space = Box(low=-np.inf, high=np.inf, shape=(141,))
    state_space = 0
    action_space = Box(low=-1.0, high=1.0, shape=(43,))

    # ── Action ──────────────────────────────────────────────────────────
    # target_pos = action * action_scale + default_pos   (IsaacLab convention)
    action_scale: float = 0.5

    # ── Velocity commands (UniformVelocityCommand-equivalent) ──────────
    cmd_lin_vel_x_range:  tuple = (0.0, 1.0)        # match IsaacLab G1Flat
    cmd_lin_vel_y_range:  tuple = (-0.5, 0.5)
    cmd_ang_vel_z_range:  tuple = (-1.0, 1.0)
    cmd_resampling_time_s: float = 10.0
    cmd_standing_prob:    float = 0.02              # fraction sampled with all-zero command
    # Heading-based ang_z control (IsaacLab convention): instead of sampling
    # ang_z directly, sample a target heading and derive ang_z each step via
    # P-control on the heading error. Produces a smoother yaw signal.
    cmd_heading_command:   bool  = True
    cmd_heading_stiffness: float = 0.5

    # ── Reset noise ────────────────────────────────────────────────────
    reset_pose_x_range:  tuple = (-0.5, 0.5)
    reset_pose_y_range:  tuple = (-0.5, 0.5)
    reset_pose_yaw_range: tuple = (-3.14, 3.14)
    reset_joint_pos_scale_range: tuple = (1.0, 1.0)  # multiplier on default_joint_pos
    reset_joint_vel_range: tuple = (0.0, 0.0)

    # ── Push (interval event, randomly perturb base velocity) ──────────
    push_enabled:           bool  = False           # off for first version
    push_interval_s:        float = 12.0
    push_velocity_xy_range: tuple = (-0.5, 0.5)

    # ── Reward weights (G1RoughEnvCfg defaults, lightly cleaned) ───────
    # Tracking (positive)
    rew_track_lin_vel_xy:  float = 1.0
    rew_track_lin_vel_std: float = 0.5              # exponential kernel std
    rew_track_ang_vel_z:   float = 2.0
    rew_track_ang_vel_std: float = 0.5
    rew_feet_air_time:     float = 0.25
    rew_feet_air_time_threshold: float = 0.4

    # Penalties (negative weights -- expressed as positive magnitudes here,
    # signs applied in rewards.py for clarity)
    pen_termination:       float = 200.0
    pen_lin_vel_z:         float = 0.0              # disabled in G1RoughEnvCfg
    pen_ang_vel_xy:        float = 0.05
    pen_flat_orientation:  float = 1.0
    pen_action_rate:       float = 0.005
    pen_dof_torques:       float = 1.5e-7
    pen_dof_acc:           float = 1.25e-7
    pen_dof_pos_limits:    float = 1.0              # only applied to ankles
    pen_feet_slide:        float = 0.1
    pen_joint_dev_hip:     float = 0.1
    pen_joint_dev_arms:    float = 0.1
    pen_joint_dev_hands:   float = 0.05
    pen_joint_dev_waist:   float = 0.1

    # ── Termination ────────────────────────────────────────────────────
    # The episode terminates (and the -200 pen_termination fires) when ANY
    # of these trip. See terminations.py. base_height + gravity_z are the
    # robust catch-alls; torso contact alone misses crawl/push-up falls.
    termination_base_height:   float = 0.4   # pelvis z (m) below this = fell.
                                             # Spawn z is 0.76; squat-walk stays
                                             # well above 0.4, fully-down ~0.15.
    termination_gravity_z:     float = -0.4  # projected_gravity_b.z above this
                                             # = tipped past ~66deg from vertical.
                                             # Upright = -1.0, horizontal = 0.0.
    illegal_contact_threshold: float = 1.0   # torso_link |force| (N) above which
                                             # we terminate (extra signal).

    # ── Observation noise (Uniform, half-range per term) ──────────────
    obs_noise_base_lin_vel:    float = 0.1
    obs_noise_base_ang_vel:    float = 0.2
    obs_noise_projected_grav:  float = 0.05
    obs_noise_joint_pos:       float = 0.01
    obs_noise_joint_vel:       float = 1.5
    obs_noise_enabled:         bool  = True

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # Match IsaacLab's velocity_env_cfg: tick the contact sensor every
        # physics step so feet_air_time and illegal-contact detection don't
        # alias to control-step boundaries.
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
