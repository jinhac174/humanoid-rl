"""Debug respawn capture — headless, no policy, periodic resets, mp4 output.

Useful when you want to *see* the spawn pose / sim settling without training
anything. Steps the env with a zero action (so the actuators just track the
default joint pose), records one mp4 per camera, and forces a fresh reset
every ``respawn_seconds``.

Usage::

    ~/IsaacLab/isaaclab.sh -p scripts/debug_respawn.py task=velocity_tracking
    ~/IsaacLab/isaaclab.sh -p scripts/debug_respawn.py task=velocity_tracking \\
        respawn_seconds=5 total_seconds=30 video_fps=30

Outputs land at ``$log_root/debug/respawn/<task>/<camera>.mp4`` (one mp4
per camera defined in the task yaml). Workflow separation: ``outputs/debug/``
is the bucket for visualization stuff; training runs stay at
``outputs/<task>/<algo>/run_NN/``.
"""
# Put the project root on sys.path so the same import patterns the other
# scripts use ('from algos...', 'from hrl_utils.paths...', etc.) resolve.
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import importlib
from pathlib import Path

import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from isaaclab.app import AppLauncher


# Hydra-only fields that shouldn't be pushed onto env_cfg.
_TASK_YAML_SCRIPT_KEYS = {
    "gym_id", "log_name",
    "env_cfg_module", "env_cfg_class",
    "evaluator_module", "evaluator_class",
    "wandb_project", "wandb_tag",
    "cameras", "viewer",
    "eval",
}


def _grab_frame(env) -> np.ndarray:
    frame = env.render()
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import tasks  # noqa: F401   triggers gym.register for all tasks

    # ── Knobs (overridable from CLI) ──────────────────────────────────────
    # video_fps defaults to None -> use the env's control rate (1/step_dt) so
    # mp4 plays back at 1.0x real time. Override via CLI to slow down or speed
    # up (e.g., video_fps=15 plays at 0.3x, video_fps=120 at 2.4x for K=50Hz).
    respawn_seconds = float(cfg.get("respawn_seconds", 5.0))
    total_seconds   = float(cfg.get("total_seconds", 30.0))
    video_fps_cli   = cfg.get("video_fps", None)
    width           = int(cfg.get("video_width", 1280))
    height          = int(cfg.get("video_height", 720))

    save_dir = Path(cfg.log_root) / "debug" / "respawn" / cfg.task.log_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Build env_cfg (num_envs=1, push task yaml fields) ─────────────────
    module = importlib.import_module(cfg.task.env_cfg_module)
    EnvCfgClass = getattr(module, cfg.task.env_cfg_class)
    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = 1

    task_dict = OmegaConf.to_container(cfg.task, resolve=True)
    for key, val in task_dict.items():
        if key in _TASK_YAML_SCRIPT_KEYS:
            continue
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, val)

    cameras = cfg.task.cameras
    first_cam = next(iter(cameras.values()))
    env_cfg.viewer.resolution = (width, height)
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = tuple(first_cam.eye)
    env_cfg.viewer.lookat = tuple(first_cam.lookat)

    env = gym.make(cfg.task.gym_id, cfg=env_cfg, render_mode="rgb_array")
    sim = env.unwrapped.sim

    action_dim = env.unwrapped.single_action_space.shape[0]
    device = env.unwrapped.device
    zero_action = torch.zeros(1, action_dim, device=device)

    step_dt = float(env.unwrapped.step_dt)
    respawn_steps = max(int(round(respawn_seconds / step_dt)), 1)
    total_steps   = max(int(round(total_seconds   / step_dt)), 1)

    # Default video_fps to control rate so playback is 1.0x real-time.
    video_fps = int(video_fps_cli) if video_fps_cli is not None else int(round(1.0 / step_dt))
    playback_rate = video_fps * step_dt

    print(f"[debug_respawn] task:         {cfg.task.log_name}")
    print(f"[debug_respawn] output:       {save_dir}")
    print(f"[debug_respawn] step_dt:      {step_dt*1000:.1f} ms (control {1.0/step_dt:.0f} Hz)")
    print(f"[debug_respawn] video_fps:    {video_fps} -> playback {playback_rate:.2f}x real time")
    print(f"[debug_respawn] respawn:      every {respawn_seconds}s = {respawn_steps} steps")
    print(f"[debug_respawn] total:        {total_seconds}s = {total_steps} steps")
    print(f"[debug_respawn] cameras:      {list(cameras.keys())}")

    # ── Prime renderer (textures/cameras settle in the first few steps) ──
    env.reset()
    for _ in range(50):
        sim.step()
    env.render()
    for _ in range(10):
        sim.step()

    # ── Open one writer per camera ────────────────────────────────────────
    writers = {}
    for cam_name in cameras:
        path = save_dir / f"{cam_name}.mp4"
        writers[cam_name] = imageio.get_writer(str(path), fps=video_fps)

    def record_frame():
        for cam_name, cam_cfg in cameras.items():
            sim.set_camera_view(
                eye=list(cam_cfg.eye), target=list(cam_cfg.lookat),
            )
            writers[cam_name].append_data(_grab_frame(env))

    # ── Step loop with periodic respawn ───────────────────────────────────
    record_frame()
    step = 0
    next_respawn = respawn_steps
    while step < total_steps:
        env.step(zero_action)
        step += 1
        record_frame()
        if step >= next_respawn:
            print(f"[debug_respawn] respawn at step {step} ({step*step_dt:.1f}s)")
            env.reset()
            next_respawn = step + respawn_steps

    for w in writers.values():
        w.close()

    for cam_name in cameras:
        print(f"[debug_respawn] saved: {save_dir / f'{cam_name}.mp4'}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
