"""G1 humanoid robot configurations.

This module is split into three sections:

    1. SHARED — motor specs, joint name lists, action scales, body name
       constants. Used by every task domain (manipulation today, locomotion
       later).

    2. MANIPULATION — ``G1_FIXED_CFG``. Root pinned (``fix_root_link=True``)
       and legs+waist held still by high-stiffness implicit actuators
       (``_LEGS_PINNED_ACTUATORS``). Only the upper body (arms, wrists, hands)
       is actively controlled by the policy. ``ACTUATED_JOINTS`` and
       ``ACTION_SCALE`` describe the 28-DoF upper-body action space and are
       what fixed-base manipulation tasks consume.

    3. LOCOMOTION — placeholder. When locomotion tasks land, add a
       ``G1_FREE_CFG`` here using ``_LEGS_ACTIVE_ACTUATORS`` (real motor
       physics on legs+waist instead of the manipulation pin) and
       ``fix_root_link=False``. Expose locomotion-side ``ACTUATED_JOINTS``
       / ``ACTION_SCALE`` symbols alongside the manipulation ones rather
       than overwriting them — the two modes share the same USD asset.

The same USD (``g1_dex3.usd``) is used for both modes; the difference is
purely how the cfg wires actuators and root-link fixing.
"""
import math

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from utils.paths import ASSET_ROOT


# =============================================================================
# 1. SHARED — joint names, motor specs, asset path
# =============================================================================

# ── Asset path ────────────────────────────────────────────────────────────────
G1_USD_PATH = str(ASSET_ROOT / "robots" / "g1" / "usd" / "g1_dex3.usd")

# ── Actuator physics (Unitree G1 motor specs: 5020 + 4010) ───────────────────
# Formula: K = I * w^2,   C = 2 * zeta * I * w
# w = 10 Hz * 2pi,   zeta = 2.0
_I_5020 = 0.003609725
_I_4010 = 0.00425
_W      = 10 * 2.0 * math.pi
_Z      = 2.0

STIFFNESS_5020 = _I_5020 * _W ** 2
STIFFNESS_4010 = _I_4010 * _W ** 2
DAMPING_5020   = 2.0 * _Z * _I_5020 * _W
DAMPING_4010   = 2.0 * _Z * _I_4010 * _W

STIFFNESS_FINGER = 0.5
STIFFNESS_THUMB0 = 2.0
DAMPING_FINGER   = 0.1
DAMPING_THUMB0   = 0.1

# ── Action scales (delta PD: 0.25 * effort_limit / stiffness) ────────────────
SCALE_ARM    = 0.25 * 25.0 / STIFFNESS_5020
SCALE_WRIST  = 0.25 *  5.0 / STIFFNESS_4010
SCALE_FINGER = 0.25 *  1.4 / STIFFNESS_FINGER
SCALE_THUMB0 = 0.25 *  2.45 / STIFFNESS_THUMB0

# ── Joint name lists ─────────────────────────────────────────────────────────
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LEFT_HAND_JOINTS = [
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
]

RIGHT_HAND_JOINTS = [
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

# Legs and waist are listed here for completeness; manipulation tasks do NOT
# include these in ACTUATED_JOINTS (they are pinned by _LEGS_PINNED_ACTUATORS).
# Locomotion will consume them when it lands.
LEFT_LEG_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]
RIGHT_LEG_JOINTS = [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
WAIST_JOINTS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

# ── Body name constants ──────────────────────────────────────────────────────
LEFT_PALM_BODY   = "left_hand_palm_link"
RIGHT_PALM_BODY  = "right_hand_palm_link"
LEFT_WRIST_BODY  = "left_wrist_yaw_link"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"


# =============================================================================
# 2. MANIPULATION — fixed-base, upper-body action space
# =============================================================================

# Action vector order (28 DoF):
#   [0:7]    left arm        [7:14]  right arm
#   [14:21]  left hand       [21:28] right hand
ACTUATED_JOINTS = (
    LEFT_ARM_JOINTS
    + RIGHT_ARM_JOINTS
    + LEFT_HAND_JOINTS
    + RIGHT_HAND_JOINTS
)

ACTION_SCALE = (
    [SCALE_ARM, SCALE_ARM, SCALE_ARM, SCALE_ARM, SCALE_ARM, SCALE_WRIST, SCALE_WRIST]
    + [SCALE_ARM, SCALE_ARM, SCALE_ARM, SCALE_ARM, SCALE_ARM, SCALE_WRIST, SCALE_WRIST]
    + [SCALE_FINGER, SCALE_FINGER, SCALE_FINGER, SCALE_FINGER, SCALE_THUMB0, SCALE_FINGER, SCALE_FINGER]
    + [SCALE_FINGER, SCALE_FINGER, SCALE_FINGER, SCALE_FINGER, SCALE_THUMB0, SCALE_FINGER, SCALE_FINGER]
)

# ── Upper-body actuator groups (real motor physics) ──────────────────────────
_UPPERBODY_ACTUATORS = {
    "arm_joints": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
        ],
        stiffness=STIFFNESS_5020,
        damping=DAMPING_5020,
        effort_limit=25.0,
        velocity_limit=37.0,
    ),
    "wrist_joints": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
        ],
        stiffness=STIFFNESS_4010,
        damping=DAMPING_4010,
        effort_limit=5.0,
        velocity_limit=22.0,
    ),
    "hand_joints": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hand_index_0_joint",
            ".*_hand_index_1_joint",
            ".*_hand_middle_0_joint",
            ".*_hand_middle_1_joint",
            ".*_hand_thumb_1_joint",
            ".*_hand_thumb_2_joint",
        ],
        stiffness=STIFFNESS_FINGER,
        damping=DAMPING_FINGER,
        effort_limit=1.4,
        velocity_limit=12.0,
    ),
    "thumb0_joint": ImplicitActuatorCfg(
        joint_names_expr=[".*_hand_thumb_0_joint"],
        stiffness=STIFFNESS_THUMB0,
        damping=DAMPING_THUMB0,
        effort_limit=2.45,
        velocity_limit=3.14,
    ),
}

# ── Legs+waist pinned (manipulation only) ────────────────────────────────────
# High stiffness keeps these joints at their default values without making them
# part of the policy's action space. Locomotion replaces this with real motor
# physics — see _LEGS_ACTIVE_ACTUATORS placeholder in section 3.
_LEGS_PINNED_ACTUATORS = {
    "locked_joints": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_pitch_joint",
            ".*_hip_roll_joint",
            ".*_hip_yaw_joint",
            ".*_knee_joint",
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
            "waist_.*_joint",
        ],
        stiffness=500.0,
        damping=50.0,
        effort_limit=500.0,
    ),
}

# ── Init joint poses (manipulation: arms ready, legs/waist at zero) ──────────
_INIT_JOINT_POS_MANIPULATION = {
    ".*_hip_pitch_joint":         0.0,
    ".*_hip_roll_joint":          0.0,
    ".*_hip_yaw_joint":           0.0,
    ".*_knee_joint":              0.0,
    ".*_ankle_pitch_joint":       0.0,
    ".*_ankle_roll_joint":        0.0,
    "waist_.*_joint":             0.0,
    "left_shoulder_pitch_joint":  -0.9,
    "left_shoulder_roll_joint":   0.4,
    "right_shoulder_pitch_joint": -0.9,
    "right_shoulder_roll_joint":  0.4,
    ".*_elbow_joint":             0.6,
    ".*_wrist_roll_joint":        0.0,
    ".*_wrist_pitch_joint":       0.0,
    ".*_wrist_yaw_joint":         0.0,
    ".*_hand_.*":                 0.0,
}

# ── Manipulation cfg: fix_root_link=True + pinned legs + actuated upper body ─
G1_FIXED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=_INIT_JOINT_POS_MANIPULATION,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={**_UPPERBODY_ACTUATORS, **_LEGS_PINNED_ACTUATORS},
)


# =============================================================================
# 3. LOCOMOTION — placeholder (not yet implemented)
# =============================================================================
#
# When locomotion tasks land, add the following symbols here (do NOT touch
# the manipulation symbols above — both modes share the file):
#
#   _LEGS_ACTIVE_ACTUATORS = {
#       "leg_joints": ImplicitActuatorCfg(
#           joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"],
#           stiffness=<motor_specs>, damping=<motor_specs>, effort_limit=<...>,
#       ),
#       "waist_joints": ImplicitActuatorCfg(
#           joint_names_expr=["waist_.*_joint"],
#           stiffness=<motor_specs>, ..., effort_limit=<...>,
#       ),
#   }
#
#   LOCOMOTION_ACTUATED_JOINTS = (
#       LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS + WAIST_JOINTS
#       + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
#       + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS
#   )
#   LOCOMOTION_ACTION_SCALE = [...]   # per-joint scales matching the order above
#
#   G1_FREE_CFG = ArticulationCfg(
#       spawn=...   # same USD, fix_root_link=False
#       actuators={**_UPPERBODY_ACTUATORS, **_LEGS_ACTIVE_ACTUATORS},
#   )
