"""
Verify the multi-color cube USD by spawning it and saving a screenshot.
Run:
    ~/IsaacLab/isaaclab.sh -p scripts/verify_cube.py
"""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import torch
import numpy as np
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg

CUBE_USD = str(Path("assets/objects/cube_multicolor.usd").resolve())
GOAL_USD = str(Path("assets/objects/cube_multicolor_goal.usd").resolve())


@configclass
class VerifySceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground plane + two colored cubes."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CUBE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )

    goal = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        spawn=sim_utils.UsdFileCfg(
            usd_path=GOAL_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.5)),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1/60)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = VerifySceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    # Step a few times to let things settle
    for _ in range(30):
        sim.step()

    # Set camera and render
    sim.set_camera_view(
        eye=[0.8, -0.8, 0.8],
        target=[0.15, 0.0, 0.5],
    )
    for _ in range(10):
        sim.step()

    # Try to capture a frame via the replicator/viewport
    try:
        from omni.isaac.core.utils.viewports import set_camera_view
        import omni.replicator.core as rep

        rp = rep.create.render_product("/OmniverseKit_Persp", (1280, 720))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([rp])

        for _ in range(5):
            sim.step()
        rep.orchestrator.step(rt_subframes=4)

        data = rgb.get_data()
        if data is not None:
            from PIL import Image
            img = Image.fromarray(data[:, :, :3])
            out_path = Path("assets/objects/cube_verify.png")
            img.save(str(out_path))
            print(f"[verify] screenshot saved: {out_path}")
        else:
            print("[verify] render returned None, but USD files were loaded successfully")
    except Exception as e:
        print(f"[verify] screenshot failed ({e}), but USD files were loaded successfully")
        print("[verify] you can inspect them visually via eval.py with a trained checkpoint")

    print("[verify] cube_multicolor.usd loaded without errors")
    simulation_app.close()


if __name__ == "__main__":
    main()