import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from gymnasium.spaces import Box

from manipulation.robots.g1 import G1_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from manipulation.utils.paths import ASSET_ROOT
_SCENES_DIR  = ASSET_ROOT / "scenes"
_OBJECTS_DIR = ASSET_ROOT / "objects"

# Fixed-base config for can_push robot spawn
_G1_CANPUSH_CFG = G1_CFG.replace(
    spawn=G1_CFG.spawn.replace(
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=G1_CFG.init_state.replace(
        pos=(2.34882, -0.63841, 0.80127),
        rot=(0.7071, 0.0, 0.0, 0.7071),
    ),
)


@configclass
class CanPushSceneCfg(InteractiveSceneCfg):

    robot = _G1_CANPUSH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    left_hand_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    right_hand_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    kitchen: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Kitchen",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_SCENES_DIR / "kitchen.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    can: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Can",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_OBJECTS_DIR / "can.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.145, -0.2979, 0.7690)),
    )

    target: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_OBJECTS_DIR / "target.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.6206, -0.2387, 0.741838)),
    )


@configclass
class CanPushEnvCfg(DirectRLEnvCfg):

    scene: CanPushSceneCfg = CanPushSceneCfg(
        num_envs=4096,
        env_spacing=5.0,
    )

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 60,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    decimation: int = 2
    episode_length_s: float = 15.0
    success_radius: float = 0.5

    observation_space = Box(low=-np.inf, high=np.inf, shape=(108,))
    action_space      = Box(low=-1.0,    high=1.0,    shape=(28,))

    can_spawn_x_range: tuple = (-0.093, 0.093)
    can_spawn_y_range: tuple = (-0.082, 0.082)

    # reward weights -- defaults match configs/task/can_push.yaml
    reward_approach_weight:  float = 3.0
    reward_push_weight:      float = 5.0
    reward_success_weight:   float = 20.0

    # penalty weights
    penalty_drop_weight:          float = 10.0
    penalty_right_idle_weight:    float = 0.50
    penalty_joint_limits_weight:  float = 0.20
    penalty_action_rate_weight:   float = 0.05
    penalty_joint_vel_weight:     float = 0.01