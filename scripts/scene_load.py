"""
Universal scene loader — headless, saves one frame per camera.

Usage:
    ~/IsaacLab/isaaclab.sh -p scripts/scene_load.py task=reorient
"""
# Put the project root on sys.path so `from algos...`, `from hrl_utils.paths...`,
# `from assets.robots.g1_cfg...`, etc. resolve regardless of which python
# launches us (kit python rewrites PYTHONPATH, so an env-var-only approach
# isn't reliable).
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import importlib
import numpy as np
import imageio.v2 as imageio
import hydra
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from isaaclab.app import AppLauncher


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):

    app_launcher   = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import tasks

    save_dir = Path(cfg.log_root) / "debug" / "scene" / cfg.task.log_name
    save_dir.mkdir(parents=True, exist_ok=True)

    SCRIPT_KEYS = {
        "gym_id", "log_name",
        "env_cfg_module", "env_cfg_class",
        "evaluator_module", "evaluator_class",
        "cameras", "viewer",
        "eval",
    }

    cameras = cfg.task.cameras
    first_cam = next(iter(cameras.values()))

    module      = importlib.import_module(cfg.task.env_cfg_module)
    EnvCfgClass = getattr(module, cfg.task.env_cfg_class)
    env_cfg     = EnvCfgClass()
    env_cfg.scene.num_envs = 1

    task_dict = OmegaConf.to_container(cfg.task, resolve=True)
    for key, val in task_dict.items():
        if key in SCRIPT_KEYS:
            continue
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, val)

    env_cfg.viewer.resolution  = (cfg.task.viewer.resolution[0],
                                  cfg.task.viewer.resolution[1])
    env_cfg.viewer.env_index   = 0
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye         = tuple(first_cam.eye)
    env_cfg.viewer.lookat      = tuple(first_cam.lookat)

    env = gym.make(cfg.task.gym_id, cfg=env_cfg, render_mode="rgb_array")
    env.reset()

    sim = env.unwrapped.sim

    # warm up + prime the renderer with a dummy render
    for _ in range(50):
        sim.step()
    env.render()  # prime — discard this frame
    for _ in range(10):
        sim.step()

    # ── Capture each camera ───────────────────────────────────────────────────
    for cam_name, cam_cfg in cameras.items():
        sim.set_camera_view(
            eye=list(cam_cfg.eye),
            target=list(cam_cfg.lookat),
        )
        for _ in range(20):
            sim.step()
        env.render()  # discard — let renderer settle
        for _ in range(5):
            sim.step()

        frame = env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 4:
            frame = frame[0]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        mean_val = frame.mean()
        print(f"[scene_load] {cam_name} — mean pixel: {mean_val:.1f}")

        out_path = save_dir / f"{cam_name}.png"
        imageio.imwrite(str(out_path), frame)
        print(f"[scene_load] saved: {out_path}")

    env.close()
    simulation_app.close()
    print("[scene_load] done")


if __name__ == "__main__":
    main()