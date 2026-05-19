"""G1 humanoid robot configurations.

Two ``ArticulationCfg`` are exported:

    G1_FIXED_CFG   — fixed-base manipulation. Root pinned to the world
                      (``fix_root_link=True``); the policy still actuates only
                      the upper-body 28 DoF (``ACTUATED_JOINTS`` /
                      ``ACTION_SCALE``). Legs / feet / waist are present in
                      the cfg with real motor physics, so they hold their
                      default pose stiffly under their own actuators.

    G1_FREE_CFG    — locomotion. ``fix_root_link=False`` and a slightly
                      squatted init pose. The policy actuates all 43 DoF
                      (``LOCOMOTION_ACTUATED_JOINTS`` /
                      ``LOCOMOTION_ACTION_SCALE``).

Motor armature / stiffness / damping values match the calibrated Unitree G1
specs used by the upstream `humanoid_isaac` codebase:

    K = I * w^2,   C = 2 * zeta * I * w,   w = 10 Hz, zeta = 2.0

    5020   :  arms (shoulder/elbow/wrist_roll), feet (×2), waist roll/pitch (×2)
    7520_14:  hip_yaw, hip_pitch, knee, waist_yaw
    7520_22:  hip_roll
    4010   :  wrist_pitch, wrist_yaw

Per-joint effort / velocity limits and the squat init pose are also taken
from there (we reproduce only what we need; we do NOT wrap actuators in
their ``DelayedImplicitActuator`` since manipulation+locomotion in this
project don't currently model actuation delay).
"""
import math

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from hrl_utils.paths import ASSET_ROOT


# =============================================================================
# 1. SHARED — joint names, motor specs, asset path
# =============================================================================

# ── Asset path ────────────────────────────────────────────────────────────────
G1_USD_PATH = str(ASSET_ROOT / "robots" / "g1" / "usd" / "g1_dex3.usd")

# ── Motor specs (Unitree G1) ─────────────────────────────────────────────────
ARMATURE_5020    = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010    = 0.00425

NATURAL_FREQ  = 10.0 * 2.0 * math.pi   # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020    = ARMATURE_5020    * NATURAL_FREQ ** 2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ ** 2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ ** 2
STIFFNESS_4010    = ARMATURE_4010    * NATURAL_FREQ ** 2

DAMPING_5020      = 2.0 * DAMPING_RATIO * ARMATURE_5020    * NATURAL_FREQ
DAMPING_7520_14   = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22   = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010      = 2.0 * DAMPING_RATIO * ARMATURE_4010    * NATURAL_FREQ

# Hand fingers (Dex3): the simple uniform model that matches the live
# humanoid_isaac G1_DEX3_CFG (the ``OLD`` cfg uses the H_K_MCP / H_K_IP
# series instead — kept as an alternative tuning path, not used here).
STIFFNESS_FINGER = 0.5
STIFFNESS_THUMB0 = 2.0
DAMPING_FINGER   = 0.1
DAMPING_THUMB0   = 0.1

# ── Per-joint effort / velocity limits ───────────────────────────────────────
EFFORT = {
    # legs
    "left_hip_yaw_joint":     88.0,  "right_hip_yaw_joint":    88.0,
    "left_hip_roll_joint":   139.0,  "right_hip_roll_joint":  139.0,
    "left_hip_pitch_joint":   88.0,  "right_hip_pitch_joint":  88.0,
    "left_knee_joint":       139.0,  "right_knee_joint":      139.0,
    # feet
    "left_ankle_pitch_joint": 50.0,  "right_ankle_pitch_joint": 50.0,
    "left_ankle_roll_joint":  50.0,  "right_ankle_roll_joint":  50.0,
    # waist
    "waist_yaw_joint":   88.0,
    "waist_roll_joint":  50.0,
    "waist_pitch_joint": 50.0,
    # arms (5020) + wrists (4010)
    "left_shoulder_pitch_joint":  25.0, "right_shoulder_pitch_joint": 25.0,
    "left_shoulder_roll_joint":   25.0, "right_shoulder_roll_joint":  25.0,
    "left_shoulder_yaw_joint":    25.0, "right_shoulder_yaw_joint":   25.0,
    "left_elbow_joint":           25.0, "right_elbow_joint":          25.0,
    "left_wrist_roll_joint":      25.0, "right_wrist_roll_joint":     25.0,
    "left_wrist_pitch_joint":      5.0, "right_wrist_pitch_joint":     5.0,
    "left_wrist_yaw_joint":        5.0, "right_wrist_yaw_joint":       5.0,
    # hands (Dex3)
    "left_hand_index_0_joint":  1.4,  "right_hand_index_0_joint":  1.4,
    "left_hand_index_1_joint":  1.4,  "right_hand_index_1_joint":  1.4,
    "left_hand_middle_0_joint": 1.4,  "right_hand_middle_0_joint": 1.4,
    "left_hand_middle_1_joint": 1.4,  "right_hand_middle_1_joint": 1.4,
    "left_hand_thumb_0_joint":  2.45, "right_hand_thumb_0_joint":  2.45,
    "left_hand_thumb_1_joint":  1.4,  "right_hand_thumb_1_joint":  1.4,
    "left_hand_thumb_2_joint":  1.4,  "right_hand_thumb_2_joint":  1.4,
}

VELOCITY = {
    "left_hip_yaw_joint":     32.0,  "right_hip_yaw_joint":    32.0,
    "left_hip_roll_joint":    20.0,  "right_hip_roll_joint":   20.0,
    "left_hip_pitch_joint":   32.0,  "right_hip_pitch_joint":  32.0,
    "left_knee_joint":        20.0,  "right_knee_joint":       20.0,
    "left_ankle_pitch_joint": 37.0,  "right_ankle_pitch_joint": 37.0,
    "left_ankle_roll_joint":  37.0,  "right_ankle_roll_joint":  37.0,
    "waist_yaw_joint":   32.0,
    "waist_roll_joint":  37.0,
    "waist_pitch_joint": 37.0,
    "left_shoulder_pitch_joint": 37.0, "right_shoulder_pitch_joint": 37.0,
    "left_shoulder_roll_joint":  37.0, "right_shoulder_roll_joint":  37.0,
    "left_shoulder_yaw_joint":   37.0, "right_shoulder_yaw_joint":   37.0,
    "left_elbow_joint":          37.0, "right_elbow_joint":          37.0,
    "left_wrist_roll_joint":     37.0, "right_wrist_roll_joint":     37.0,
    "left_wrist_pitch_joint":    22.0, "right_wrist_pitch_joint":    22.0,
    "left_wrist_yaw_joint":      22.0, "right_wrist_yaw_joint":      22.0,
    "left_hand_index_0_joint":  12.0, "right_hand_index_0_joint":  12.0,
    "left_hand_index_1_joint":  12.0, "right_hand_index_1_joint":  12.0,
    "left_hand_middle_0_joint": 12.0, "right_hand_middle_0_joint": 12.0,
    "left_hand_middle_1_joint": 12.0, "right_hand_middle_1_joint": 12.0,
    "left_hand_thumb_0_joint":   3.14,"right_hand_thumb_0_joint":   3.14,
    "left_hand_thumb_1_joint":  12.0, "right_hand_thumb_1_joint":  12.0,
    "left_hand_thumb_2_joint":  12.0, "right_hand_thumb_2_joint":  12.0,
}

# Stiffness lookup by joint name (used to compute action scales). The motor
# group a joint belongs to picks its (K, C, armature). Feet and waist
# roll/pitch are driven by 2× a 5020 motor (matches humanoid_isaac).
STIFFNESS = {
    # legs
    "left_hip_yaw_joint":     STIFFNESS_7520_14, "right_hip_yaw_joint":    STIFFNESS_7520_14,
    "left_hip_roll_joint":    STIFFNESS_7520_22, "right_hip_roll_joint":   STIFFNESS_7520_22,
    "left_hip_pitch_joint":   STIFFNESS_7520_14, "right_hip_pitch_joint":  STIFFNESS_7520_14,
    "left_knee_joint":        STIFFNESS_7520_22, "right_knee_joint":       STIFFNESS_7520_22,
    # feet
    "left_ankle_pitch_joint": 2.0 * STIFFNESS_5020, "right_ankle_pitch_joint": 2.0 * STIFFNESS_5020,
    "left_ankle_roll_joint":  2.0 * STIFFNESS_5020, "right_ankle_roll_joint":  2.0 * STIFFNESS_5020,
    # waist
    "waist_yaw_joint":   STIFFNESS_7520_14,
    "waist_roll_joint":  2.0 * STIFFNESS_5020,
    "waist_pitch_joint": 2.0 * STIFFNESS_5020,
    # arms / wrists
    "left_shoulder_pitch_joint": STIFFNESS_5020, "right_shoulder_pitch_joint": STIFFNESS_5020,
    "left_shoulder_roll_joint":  STIFFNESS_5020, "right_shoulder_roll_joint":  STIFFNESS_5020,
    "left_shoulder_yaw_joint":   STIFFNESS_5020, "right_shoulder_yaw_joint":   STIFFNESS_5020,
    "left_elbow_joint":          STIFFNESS_5020, "right_elbow_joint":          STIFFNESS_5020,
    "left_wrist_roll_joint":     STIFFNESS_5020, "right_wrist_roll_joint":     STIFFNESS_5020,
    "left_wrist_pitch_joint":    STIFFNESS_4010, "right_wrist_pitch_joint":    STIFFNESS_4010,
    "left_wrist_yaw_joint":      STIFFNESS_4010, "right_wrist_yaw_joint":      STIFFNESS_4010,
    # hands
    "left_hand_index_0_joint":  STIFFNESS_FINGER, "right_hand_index_0_joint":  STIFFNESS_FINGER,
    "left_hand_index_1_joint":  STIFFNESS_FINGER, "right_hand_index_1_joint":  STIFFNESS_FINGER,
    "left_hand_middle_0_joint": STIFFNESS_FINGER, "right_hand_middle_0_joint": STIFFNESS_FINGER,
    "left_hand_middle_1_joint": STIFFNESS_FINGER, "right_hand_middle_1_joint": STIFFNESS_FINGER,
    "left_hand_thumb_0_joint":  STIFFNESS_THUMB0, "right_hand_thumb_0_joint":  STIFFNESS_THUMB0,
    "left_hand_thumb_1_joint":  STIFFNESS_FINGER, "right_hand_thumb_1_joint":  STIFFNESS_FINGER,
    "left_hand_thumb_2_joint":  STIFFNESS_FINGER, "right_hand_thumb_2_joint":  STIFFNESS_FINGER,
}

# ── Joint name lists ─────────────────────────────────────────────────────────
LEFT_LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
]
RIGHT_LEG_JOINTS = [
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
WAIST_JOINTS = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
LEFT_HAND_JOINTS = [
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
]
RIGHT_HAND_JOINTS = [
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
]

# ── Body name constants ──────────────────────────────────────────────────────
LEFT_PALM_BODY   = "left_hand_palm_link"
RIGHT_PALM_BODY  = "right_hand_palm_link"
LEFT_WRIST_BODY  = "left_wrist_yaw_link"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"


# =============================================================================
# 2. ACTUATOR GROUPS — defined once, reused by both fix-base and free-base cfgs
# =============================================================================

_LEGS_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=[
        ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint",
        ".*_knee_joint",
    ],
    effort_limit_sim={
        ".*_hip_yaw_joint":   88.0,
        ".*_hip_roll_joint": 139.0,
        ".*_hip_pitch_joint": 88.0,
        ".*_knee_joint":     139.0,
    },
    velocity_limit_sim={
        ".*_hip_yaw_joint":   32.0,
        ".*_hip_roll_joint":  20.0,
        ".*_hip_pitch_joint": 32.0,
        ".*_knee_joint":      20.0,
    },
    stiffness={
        ".*_hip_pitch_joint": STIFFNESS_7520_14,
        ".*_hip_roll_joint":  STIFFNESS_7520_22,
        ".*_hip_yaw_joint":   STIFFNESS_7520_14,
        ".*_knee_joint":      STIFFNESS_7520_22,
    },
    damping={
        ".*_hip_pitch_joint": DAMPING_7520_14,
        ".*_hip_roll_joint":  DAMPING_7520_22,
        ".*_hip_yaw_joint":   DAMPING_7520_14,
        ".*_knee_joint":      DAMPING_7520_22,
    },
    armature={
        ".*_hip_pitch_joint": ARMATURE_7520_14,
        ".*_hip_roll_joint":  ARMATURE_7520_22,
        ".*_hip_yaw_joint":   ARMATURE_7520_14,
        ".*_knee_joint":      ARMATURE_7520_22,
    },
)

_FEET_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
    effort_limit_sim=50.0,
    velocity_limit_sim=37.0,
    stiffness=2.0 * STIFFNESS_5020,
    damping=2.0 * DAMPING_5020,
    armature=2.0 * ARMATURE_5020,
)

_WAIST_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
    effort_limit_sim=50.0,
    velocity_limit_sim=37.0,
    stiffness=2.0 * STIFFNESS_5020,
    damping=2.0 * DAMPING_5020,
    armature=2.0 * ARMATURE_5020,
)

_WAIST_YAW_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=["waist_yaw_joint"],
    effort_limit_sim=88.0,
    velocity_limit_sim=32.0,
    stiffness=STIFFNESS_7520_14,
    damping=DAMPING_7520_14,
    armature=ARMATURE_7520_14,
)

_ARMS_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=[
        ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",   ".*_elbow_joint",
        ".*_wrist_roll_joint",     ".*_wrist_pitch_joint", ".*_wrist_yaw_joint",
    ],
    effort_limit_sim={
        ".*_shoulder_pitch_joint": 25.0,
        ".*_shoulder_roll_joint":  25.0,
        ".*_shoulder_yaw_joint":   25.0,
        ".*_elbow_joint":          25.0,
        ".*_wrist_roll_joint":     25.0,
        ".*_wrist_pitch_joint":     5.0,
        ".*_wrist_yaw_joint":       5.0,
    },
    velocity_limit_sim={
        ".*_shoulder_pitch_joint": 37.0,
        ".*_shoulder_roll_joint":  37.0,
        ".*_shoulder_yaw_joint":   37.0,
        ".*_elbow_joint":          37.0,
        ".*_wrist_roll_joint":     37.0,
        ".*_wrist_pitch_joint":    22.0,
        ".*_wrist_yaw_joint":      22.0,
    },
    stiffness={
        ".*_shoulder_pitch_joint": STIFFNESS_5020,
        ".*_shoulder_roll_joint":  STIFFNESS_5020,
        ".*_shoulder_yaw_joint":   STIFFNESS_5020,
        ".*_elbow_joint":          STIFFNESS_5020,
        ".*_wrist_roll_joint":     STIFFNESS_5020,
        ".*_wrist_pitch_joint":    STIFFNESS_4010,
        ".*_wrist_yaw_joint":      STIFFNESS_4010,
    },
    damping={
        ".*_shoulder_pitch_joint": DAMPING_5020,
        ".*_shoulder_roll_joint":  DAMPING_5020,
        ".*_shoulder_yaw_joint":   DAMPING_5020,
        ".*_elbow_joint":          DAMPING_5020,
        ".*_wrist_roll_joint":     DAMPING_5020,
        ".*_wrist_pitch_joint":    DAMPING_4010,
        ".*_wrist_yaw_joint":      DAMPING_4010,
    },
    armature={
        ".*_shoulder_pitch_joint": ARMATURE_5020,
        ".*_shoulder_roll_joint":  ARMATURE_5020,
        ".*_shoulder_yaw_joint":   ARMATURE_5020,
        ".*_elbow_joint":          ARMATURE_5020,
        ".*_wrist_roll_joint":     ARMATURE_5020,
        ".*_wrist_pitch_joint":    ARMATURE_4010,
        ".*_wrist_yaw_joint":      ARMATURE_4010,
    },
)

_HANDS_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=[
        ".*_hand_index_0_joint",  ".*_hand_index_1_joint",
        ".*_hand_middle_0_joint", ".*_hand_middle_1_joint",
        ".*_hand_thumb_0_joint",  ".*_hand_thumb_1_joint", ".*_hand_thumb_2_joint",
    ],
    effort_limit_sim={
        ".*_hand_index_0_joint":  1.4,
        ".*_hand_middle_0_joint": 1.4,
        ".*_hand_thumb_0_joint":  2.45,
        ".*_hand_index_1_joint":  1.4,
        ".*_hand_middle_1_joint": 1.4,
        ".*_hand_thumb_1_joint":  1.4,
        ".*_hand_thumb_2_joint":  1.4,
    },
    velocity_limit_sim={
        ".*_hand_index_0_joint":  12.0,
        ".*_hand_middle_0_joint": 12.0,
        ".*_hand_thumb_0_joint":   3.14,
        ".*_hand_index_1_joint":  12.0,
        ".*_hand_middle_1_joint": 12.0,
        ".*_hand_thumb_1_joint":  12.0,
        ".*_hand_thumb_2_joint":  12.0,
    },
    stiffness={
        ".*_hand_index_0_joint":  STIFFNESS_FINGER,
        ".*_hand_middle_0_joint": STIFFNESS_FINGER,
        ".*_hand_thumb_0_joint":  STIFFNESS_THUMB0,
        ".*_hand_index_1_joint":  STIFFNESS_FINGER,
        ".*_hand_middle_1_joint": STIFFNESS_FINGER,
        ".*_hand_thumb_1_joint":  STIFFNESS_FINGER,
        ".*_hand_thumb_2_joint":  STIFFNESS_FINGER,
    },
    damping={
        ".*_hand_index_0_joint":  DAMPING_FINGER,
        ".*_hand_middle_0_joint": DAMPING_FINGER,
        ".*_hand_thumb_0_joint":  DAMPING_THUMB0,
        ".*_hand_index_1_joint":  DAMPING_FINGER,
        ".*_hand_middle_1_joint": DAMPING_FINGER,
        ".*_hand_thumb_1_joint":  DAMPING_FINGER,
        ".*_hand_thumb_2_joint":  DAMPING_FINGER,
    },
)

_ALL_ACTUATORS = {
    "legs":      _LEGS_ACTUATOR,
    "feet":      _FEET_ACTUATOR,
    "waist":     _WAIST_ACTUATOR,
    "waist_yaw": _WAIST_YAW_ACTUATOR,
    "arms":      _ARMS_ACTUATOR,
    "hands":     _HANDS_ACTUATOR,
}


# =============================================================================
# 3. ACTION SCALES — delta-PD: scale * effort_limit / stiffness, scale=0.25
# =============================================================================

ACTION_SCALE_COEF = 0.25


def _action_scale(joint_name: str) -> float:
    return ACTION_SCALE_COEF * EFFORT[joint_name] / STIFFNESS[joint_name]


# Manipulation: 28 upper-body DoF. Order is left arm | right arm | left hand | right hand.
ACTUATED_JOINTS = (
    LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS
)
ACTION_SCALE = [_action_scale(j) for j in ACTUATED_JOINTS]
assert len(ACTUATED_JOINTS) == 28

# Locomotion: 43 DoF. Order is legs | waist | arms | hands.
LOCOMOTION_ACTUATED_JOINTS = (
    LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS
    + WAIST_JOINTS
    + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
    + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS
)
LOCOMOTION_ACTION_SCALE = [_action_scale(j) for j in LOCOMOTION_ACTUATED_JOINTS]
assert len(LOCOMOTION_ACTUATED_JOINTS) == 43


# =============================================================================
# 4. INIT POSES
# =============================================================================

# Manipulation pose: arms in a comfortable hover ready to manipulate; legs and
# waist neutral (root is pinned, so the legs hold whatever the actuators say).
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

# Locomotion init pose: matches IsaacLab G1_MINIMAL_CFG. Light squat (NOT the
# deeper humanoid_isaac stance), arms slightly forward. The lighter stance
# reduces the actuator load needed just to stand and starts the policy closer
# to a natural walking pose.
#   IsaacLab values:  hip_pitch=-0.20  knee=0.42  ankle_pitch=-0.23  spawn_z=0.74
_INIT_JOINT_POS_LOCOMOTION = {
    ".*_hip_pitch_joint":           -0.20,
    ".*_hip_roll_joint":             0.0,
    ".*_hip_yaw_joint":              0.0,
    ".*_knee_joint":                 0.42,
    ".*_ankle_pitch_joint":         -0.23,
    ".*_ankle_roll_joint":           0.0,
    "waist_.*_joint":                0.0,
    "left_shoulder_pitch_joint":     0.35,
    "left_shoulder_roll_joint":      0.16,
    "right_shoulder_pitch_joint":    0.35,
    "right_shoulder_roll_joint":    -0.16,
    ".*_shoulder_yaw_joint":         0.0,
    ".*_elbow_joint":                0.87,
    ".*_wrist_roll_joint":           0.0,
    ".*_wrist_pitch_joint":          0.0,
    ".*_wrist_yaw_joint":            0.0,
    ".*_hand_.*":                    0.0,
}


# =============================================================================
# 5. ARTICULATION CFGS
# =============================================================================

# Shared spawn (rigid_props). fix_root_link is set per-cfg below.
_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
)


def _spawn(fix_root_link: bool) -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(
        usd_path=G1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=_RIGID_PROPS,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=fix_root_link,
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    )


# ── Manipulation: fix_root_link=True, arms-ready pose ─────────────────────────
G1_FIXED_CFG = ArticulationCfg(
    spawn=_spawn(fix_root_link=True),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=_INIT_JOINT_POS_MANIPULATION,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators=_ALL_ACTUATORS,
)

# ── Locomotion: fix_root_link=False, light-squat init pose ─────────────────
G1_FREE_CFG = ArticulationCfg(
    spawn=_spawn(fix_root_link=False),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),     # matches IsaacLab G1_MINIMAL_CFG
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=_INIT_JOINT_POS_LOCOMOTION,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators=_ALL_ACTUATORS,
)
