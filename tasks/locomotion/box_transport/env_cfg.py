"""Box-transport env config.

Builds on the velocity-tracking scene (free-base G1, flat ground, contact
sensors) with two tables and a graspable box. Same 43-DoF action space as
locomotion so a locomotion checkpoint can be warm-started.

Assets are primitive ``CuboidCfg`` (table + box) — same pattern as the
reorient task, no Nucleus dependency. Swap to USD assets later if needed.

Reward / observation / termination knobs are exposed as fields here so they
can be overridden from ``configs/task/box_transport.yaml``.
"""
from __future__ import annotations

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from gymnasium.spaces import Box

from assets.robots.g1_cfg import G1_FREE_CFG


# --- Body name constants (shared with the locomotion env) ----------------
ROOT_BODY = "pelvis"
TORSO_BODY = "torso_link"
LEFT_FOOT_BODY = "left_ankle_roll_link"
RIGHT_FOOT_BODY = "right_ankle_roll_link"
LEFT_PALM_BODY = "left_hand_palm_link"
RIGHT_PALM_BODY = "right_hand_palm_link"


# --- Scene layout constants (meters, world frame) ------------------------
# Robot spawns at the world origin (0,0,*) facing +x. L-shaped task: walk
# +x to the start table, grab the box, turn 90° and walk -y to the target.
TABLE_SIZE = (1.0, 1.0, 0.55)            # (sx, sy, sz). Height lowered 0.78 → 0.55
                                         # so the box sits at a comfortable
                                         # grab height for the G1 (pelvis 0.74).
TABLE_TOP_Z = TABLE_SIZE[2]              # 0.55 m

START_TABLE_POS  = (2.5,  0.0, TABLE_SIZE[2] / 2)   # 2.5 m forward of spawn
TARGET_TABLE_POS = (2.5, -2.0, TABLE_SIZE[2] / 2)   # 2.0 m to the RIGHT of start
                                                    # (pulled in from 3.0 — shorter
                                                    # carry). Off-axis so the
                                                    # autocmd never points through
                                                    # the start table.

BOX_SIZE = (0.24, 0.24, 0.24)            # 24 cm cube (was 30) — easier bimanual grip
BOX_MASS = 0.8                           # kg (was 1.5) — scaled down with the box

# Box sits toward the FRONT (robot-facing, smaller-x) edge of the start
# table so the robot can reach it without standing inside the footprint.
BOX_SPAWN_POS = (
    START_TABLE_POS[0] - 0.30,             # x = 2.20  (box front face 0.08 m in)
    START_TABLE_POS[1],
    TABLE_TOP_Z + BOX_SIZE[2] / 2 + 0.01,  # 1 cm above the table surface
)

# Target: sampled near the FRONT (robot-facing, +y) edge of the target
# table — same reasoning, the robot reaches over from in front.
TARGET_ANCHOR_POS = (
    TARGET_TABLE_POS[0],
    TARGET_TABLE_POS[1] + TABLE_SIZE[1] / 2 - 0.30,   # y = -1.80
    TABLE_TOP_Z + BOX_SIZE[2] / 2 + 0.001,
)
# Uniform half-range sampled around TARGET_ANCHOR_POS each reset.
TARGET_XY_HALF_RANGE = (0.12, 0.10)
TARGET_Z = TABLE_TOP_Z + BOX_SIZE[2] / 2 + 0.001  # tiny epsilon over the surface

# Drop threshold: episode ends when box.z falls below this (hit floor).
# Box CENTER resting on the floor is at BOX_SIZE/2 = 0.12; 0.20 fires once
# the box is clearly off any table and almost on the floor.
BOX_DROP_Z = 0.20

# Lift threshold: box center must rise 10 cm above its table-rest height
# (TABLE_TOP_Z + BOX_SIZE/2 = 0.67) to count as lifted — otherwise the
# flag would fire at spawn before the policy has done anything.
BOX_LIFT_Z = TABLE_TOP_Z + BOX_SIZE[2] / 2 + 0.10   # = 0.77 m

# Bimanual-contact gating: both palms within this distance from the box
# centre to count as "gripping".
GRIP_DISTANCE = 0.20                     # m


# --- Scene --------------------------------------------------------------
@configclass
class BoxTransportSceneCfg(InteractiveSceneCfg):
    """G1 free-base + flat plane + two tables + graspable box."""

    # Flat ground (matches locomotion).
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

    # Robot (free-base, light-squat init pose, 43 DoF actuated).
    robot = G1_FREE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Two tables — kinematic, primitive cuboid. Wood-ish tan colour.
    start_table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StartTable",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.70, 0.58, 0.40)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=START_TABLE_POS),
    )

    target_table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetTable",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            # Slightly different colour so it's distinguishable in videos.
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.50, 0.60)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TARGET_TABLE_POS),
    )

    # The box the robot must transport.
    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            # Cardboard-ish brown.
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.60, 0.45, 0.30)),
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=1.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_SPAWN_POS),
    )

    # Contact sensors over every body — used for foot air-time, illegal-contact
    # termination, and palm/forearm-to-box contact detection.
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
class BoxTransportEnvCfg(DirectRLEnvCfg):
    scene: BoxTransportSceneCfg = BoxTransportSceneCfg(
        num_envs=4096,
        env_spacing=8.0,        # wider than locomotion: tables + walk corridor
    )

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(gpu_max_rigid_patch_count=10 * 2 ** 15),
    )

    decimation: int = 4
    episode_length_s: float = 30.0       # 50 Hz * 30 s = 1500 control steps

    # ── Spaces ───────────────────────────────────────────────────────────
    # Obs layout (164 dims):
    #   [0:141]    locomotion proprioception (kept identical for warm-start;
    #              the velocity_commands slot is auto-derived from the task
    #              phase — see env._pre_physics_step).
    #   [141:144]  box_pos_rel    (box - robot root, in robot base frame)
    #   [144:148]  box_quat       (world; orientation isn't base-frame critical)
    #   [148:151]  box_lin_vel_b  (box linear vel, robot base frame)
    #   [151:154]  target_pos_rel (target - robot root, in robot base frame)
    #   [154:157]  l_palm_to_box  (box - left_palm, robot base frame)
    #   [157:160]  r_palm_to_box  (box - right_palm, robot base frame)
    #   [160:163]  box_to_target  (target - box, world frame)
    #   [163]      lifted_flag    (1 if box.z > BOX_LIFT_Z this step)
    observation_space = Box(low=-np.inf, high=np.inf, shape=(164,))
    state_space = 0
    action_space = Box(low=-1.0, high=1.0, shape=(43,))

    # ── Action ──────────────────────────────────────────────────────────
    action_scale: float = 0.5             # same as locomotion (offset-from-default)

    # ── Velocity command derivation ─────────────────────────────────────
    # The locomotion obs slot for velocity_commands is filled by an auto-
    # derived signal: when not lifted, point toward the box; when lifted,
    # point toward the target. Magnitude in [0, max_vel_x]; ang_z via P-control
    # on heading error toward the active goal (same as locomotion's heading
    # command). This keeps the warm-started policy fed with the same kind of
    # input it was trained on.
    autocmd_lin_vel_max:   float = 1.0
    autocmd_ang_vel_max:   float = 1.0
    autocmd_heading_stiffness: float = 0.5
    # Distance to the active goal at which command magnitude reaches zero.
    # Box / target now sit near the front edge of their tables, so 0.50 m
    # puts the pelvis ~0.3 m in front of the table face — close enough that
    # extended arms reach the box, far enough that the body stays clear.
    autocmd_stop_distance: float = 0.50

    # ── Reset noise (robot spawn) ──────────────────────────────────────
    reset_pose_x_range:  tuple = (-0.20, 0.20)
    reset_pose_y_range:  tuple = (-0.20, 0.20)
    reset_pose_yaw_range: tuple = (-0.3, 0.3)   # mostly forward-facing
    reset_joint_pos_scale_range: tuple = (1.0, 1.0)
    reset_joint_vel_range: tuple = (0.0, 0.0)

    # Box-spawn xy randomization (uniform half-range). Small — the box is
    # near the table edge, so large noise would push it off.
    reset_box_xy_range: float = 0.05
    # Target xy randomization (uniform half-range around TARGET_ANCHOR_POS).
    reset_target_xy_half: tuple = TARGET_XY_HALF_RANGE

    # ── Reward weights ───────────────────────────────────────────────────
    # Architecture:
    #   * Navigation = phase-1 velocity-tracking reward against the
    #     auto-derived command from ``events.update_autocmd``.
    #   * Arm guidance = DENSE reach shaping (rew_reach) — palm-to-box
    #     distance kernel, always on, so the arms have a continuous
    #     gradient toward the box (and toward keeping it gripped during
    #     the carry). Without this the only arm signal was the sparse
    #     bimanual_contact bonus, which is too sparse to discover.
    #   * Manipulation milestones = bimanual_contact (one-shot) →
    #     lift (one-shot) → place_bonus (continuous on target) → drop pen.

    # Locomotion tracking (mirror of velocity_tracking.yaml defaults).
    rew_track_lin_vel_xy:       float = 1.0
    rew_track_lin_vel_std:      float = 0.5
    rew_track_ang_vel_z:        float = 1.0
    rew_track_ang_vel_std:      float = 0.5
    rew_feet_air_time:          float = 0.75
    rew_feet_air_time_threshold:float = 0.4

    # Dense arm-reach shaping. reward = exp(-mean_palm_to_box_dist² / std²),
    # always on. Naturally ≈0 while the robot is still walking in (palms
    # far from box) and ramps up as the arms approach — also rewards
    # keeping the palms on the box through the carry.
    rew_reach:                  float = 1.5
    rew_reach_std:              float = 0.5

    # Manipulation milestones.
    rew_bimanual_contact:   float = 10.0     # sparse: one-shot bonus on first frame both palms within GRIP_DISTANCE
    rew_lift:               float = 100.0    # sparse: one-shot bonus on first frame box.z > BOX_LIFT_Z
    rew_place_bonus:        float = 500.0    # continuous while box xy within rew_place_distance_tol of target AND on table
    rew_place_distance_tol: float = 0.15

    # Negative — discourage dropping.
    pen_drop:               float = 100.0    # one-shot when box hits the floor

    # Locomotion regularizers — RESTORED to phase-1 (velocity_tracking)
    # values. The earlier 0.5× reduction degraded gait quality (weird /
    # jumpy gaits in the from-scratch runs). Matching phase 1 also keeps
    # the warm-started policy inside the reward landscape it was trained
    # on. Arm / hand deviation stays very low — those joints must move
    # freely for the manipulation.
    pen_termination:        float = 200.0    # robot fell
    pen_lin_vel_z:          float = 0.2
    pen_ang_vel_xy:         float = 0.05
    pen_flat_orientation:   float = 1.0      # pelvis upright
    pen_action_rate:        float = 0.005
    pen_dof_torques:        float = 2.0e-6
    pen_dof_acc:            float = 1.0e-7
    pen_dof_pos_limits:     float = 1.0
    pen_feet_slide:         float = 0.1
    pen_joint_dev_hip:      float = 0.1
    pen_joint_dev_arms:     float = 0.02     # very low — arms must reach
    pen_joint_dev_hands:    float = 0.01
    # POSTURE: the 0.3 waist-deviation penalty did NOT fix the backward
    # lean. Bumped to 0.6 AND backed by a direct torso-upright term
    # (pen_torso_upright) — gravity projected into the torso_link frame,
    # which catches a torso lean even when the pelvis itself is level.
    pen_joint_dev_waist:    float = 0.6
    pen_torso_upright:      float = 1.0

    # ── Termination ──────────────────────────────────────────────────────
    # Robot fell (same 3-signal check as locomotion).
    termination_base_height:   float = 0.4
    termination_gravity_z:     float = -0.4
    illegal_contact_threshold: float = 1.0
    # Box on the floor. See BOX_DROP_Z comment — box center below ~0.20
    # means it is essentially at floor level (rests at 0.15).
    box_drop_z:                float = BOX_DROP_Z
    # Success: box on target table within tolerance for N consecutive steps.
    success_steps_required:    int   = 25      # 0.5 s at 50 Hz

    # ── Observation noise ─────────────────────────────────────────────
    obs_noise_base_lin_vel:    float = 0.1
    obs_noise_base_ang_vel:    float = 0.2
    obs_noise_projected_grav:  float = 0.05
    obs_noise_joint_pos:       float = 0.01
    obs_noise_joint_vel:       float = 1.5
    obs_noise_box_pos:         float = 0.01
    obs_noise_enabled:         bool  = True

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # Match locomotion: tick the contact sensor every physics step so
        # foot-air-time / illegal-contact detection don't alias.
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
