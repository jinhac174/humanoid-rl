"""Base evaluator.

Symmetric to ``algos/<algo>/trainer.py``: each task module exports an
evaluator class that owns its eval loop. ``BaseEvaluator`` here covers the
generic flow (build env, load policy, run episodes, write per-camera videos)
and exposes hooks that task subclasses override for task-specific behaviour
(success counting, goal respawn cooldown, eval-time env_cfg overrides, etc.).

A task that needs no overrides can omit ``evaluator_module`` /
``evaluator_class`` from its yaml — ``scripts/eval.py`` will fall back to
this class.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch
from omegaconf import DictConfig, OmegaConf

from evaluators.utils import grab_frame, load_policy_from_checkpoint, setup_raytracing


# Hydra-only fields that are not env_cfg attributes; never push these into
# the env_cfg via the generic copy loop.
_TASK_YAML_SCRIPT_KEYS = {
    "gym_id",
    "log_name",
    "env_cfg_module",
    "env_cfg_class",
    "evaluator_module",
    "evaluator_class",
    "wandb_project",
    "wandb_tag",
    "cameras",
    "viewer",
    "eval",  # eval-only block; consumed by apply_eval_overrides
}


class BaseEvaluator:
    """Generic evaluation loop. Override hooks in subclasses for task logic.

    Lifecycle (called from ``scripts/eval.py``):

        evaluator = EvaluatorClass(cfg)
        evaluator.setup_renderer()           # before the env is built
        evaluator.build_env()
        evaluator.load_policy()
        evaluator.run()                      # episode loop + video writers
        evaluator.close()

    Hooks a task evaluator typically overrides:

        apply_eval_overrides(env_cfg)        push task yaml's ``eval:`` block
        on_episode_start(ep)                 zero per-episode counters
        on_step(step, info)                  count successes, drive cooldowns
        episode_summary(ep, steps, reward)   one-line per-episode stdout
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        ckpt = Path(cfg.checkpoint).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        self.checkpoint_path: Path = ckpt

        # Mirror the training run's directory layout:
        #   outputs/<task>/<algo>/run_XX/eval/<ckpt_stem>/
        self.eval_dir: Path = ckpt.parent.parent / "eval" / ckpt.stem
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        # Filled in by setup_*/build_*/load_*
        self.env = None
        self.unwrapped = None
        self.sim = None
        self.network = None
        self.get_action = None
        self.is_sapg = False
        self.cameras = None
        self.use_raytracing: bool = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_renderer(self) -> None:
        """Configure path tracing if requested. Must run before env build."""
        if getattr(self.cfg, "raytraced", False):
            spp = int(getattr(self.cfg, "spp", 64))
            self.use_raytracing = setup_raytracing(spp)
            if self.use_raytracing:
                print(f"[eval] ray-tracing ENABLED (spp={spp})")
        else:
            print("[eval] rasterization mode (use raytraced=true for publication quality)")

    def build_env(self) -> None:
        """Construct the gym env at num_envs=1, applying eval overrides."""
        import gymnasium as gym
        # Side-effect: registers gym ids for every task in this codebase.
        import tasks  # noqa: F401

        cfg = self.cfg

        module = importlib.import_module(cfg.task.env_cfg_module)
        EnvCfgClass = getattr(module, cfg.task.env_cfg_class)
        env_cfg = EnvCfgClass()
        env_cfg.scene.num_envs = 1

        # Push every yaml field that names an env_cfg attribute. Same pattern
        # train.py uses; keep them in sync.
        task_dict = OmegaConf.to_container(cfg.task, resolve=True)
        for key, val in task_dict.items():
            if key in _TASK_YAML_SCRIPT_KEYS:
                continue
            if hasattr(env_cfg, key):
                setattr(env_cfg, key, val)

        # Eval-time overrides come AFTER the generic copy so they win.
        self.apply_eval_overrides(env_cfg)

        # Camera placement for the rendered viewer (drives the first cam frame).
        cameras = cfg.task.cameras
        first_cam = next(iter(cameras.values()))
        env_cfg.viewer.resolution = (cfg.video_width, cfg.video_height)
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye = tuple(first_cam.eye)
        env_cfg.viewer.lookat = tuple(first_cam.lookat)

        self.env = gym.make(cfg.task.gym_id, cfg=env_cfg, render_mode="rgb_array")
        self.unwrapped = self.env.unwrapped
        self.sim = self.unwrapped.sim
        self.cameras = cameras

    def apply_eval_overrides(self, env_cfg: Any) -> None:
        """Push the task yaml's ``eval:`` block onto env_cfg.

        Default: copy every key in ``cfg.task.eval`` whose name is an
        env_cfg attribute. Tasks with structural eval changes can override.
        """
        eval_block = getattr(self.cfg.task, "eval", None)
        if eval_block is None:
            return
        for key, val in OmegaConf.to_container(eval_block, resolve=True).items():
            if hasattr(env_cfg, key):
                setattr(env_cfg, key, val)

    def load_policy(self) -> None:
        """Auto-detect PPO vs SAPG/EPO from the checkpoint and rebuild the net."""
        obs_dim = self.unwrapped.single_observation_space["policy"].shape[0]
        action_dim = self.unwrapped.single_action_space.shape[0]
        device = self.unwrapped.device
        self.network, self.get_action, self.is_sapg = load_policy_from_checkpoint(
            self.checkpoint_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            deterministic=bool(self.cfg.deterministic),
        )

    # ------------------------------------------------------------------
    # Episode hooks (override in subclasses)
    # ------------------------------------------------------------------

    def on_episode_start(self, ep: int) -> None:
        """Called at the start of each episode, after env.reset."""

    def on_step(self, step: int, info: dict) -> None:
        """Called after every env.step (before the frame is recorded)."""

    def episode_summary(self, ep: int, steps: int, total_reward: float) -> str:
        """One-line per-episode summary printed to stdout."""
        return f"  ep{ep:03d} | steps={steps:4d} | reward={total_reward:.2f}"

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _open_writers(self, ep: int) -> dict:
        writers = {}
        for cam_name in self.cameras:
            path = self.eval_dir / f"{cam_name}_ep{ep:03d}.mp4"
            writers[cam_name] = imageio.get_writer(str(path), fps=self.cfg.video_fps)
        return writers

    def _record_frame(self, writers: dict) -> None:
        if self.use_raytracing:
            # Path tracing needs a few extra render passes to converge.
            for _ in range(3):
                self.env.render()
        for cam_name, cam_cfg in self.cameras.items():
            self.sim.set_camera_view(eye=list(cam_cfg.eye), target=list(cam_cfg.lookat))
            frame = grab_frame(self.env)
            writers[cam_name].append_data(frame)

    def _prime_renderer(self) -> None:
        """Warm up the renderer once after the first reset.

        The first few simulated steps and first render() are unreliable
        (textures not loaded, cameras uninitialised). Throw them away so the
        recorded videos start clean.
        """
        prime_steps = 100 if self.use_raytracing else 50
        for _ in range(prime_steps):
            self.sim.step()
        self.env.render()
        for _ in range(10):
            self.sim.step()

    def run(self) -> None:
        """Episode loop. Each episode writes one mp4 per camera."""
        cfg = self.cfg
        env = self.env
        unwrapped = self.unwrapped

        print(f"[eval] checkpoint:  {self.checkpoint_path}")
        print(f"[eval] output:      {self.eval_dir}")
        print(f"[eval] resolution:  {cfg.video_width}x{cfg.video_height}")

        env.reset(seed=0)
        self._prime_renderer()

        for ep in range(cfg.num_episodes):
            obs_dict, _ = env.reset(seed=ep)
            obs_raw = obs_dict["policy"]
            self.on_episode_start(ep)

            writers = self._open_writers(ep)
            self._record_frame(writers)

            max_steps = int(unwrapped.max_episode_length)
            total_reward = 0.0
            step = 0

            while step < max_steps:
                with torch.no_grad():
                    action = self.get_action(obs_raw)
                obs_dict, reward, terminated, timed_out, info = env.step(action)
                obs_raw = obs_dict["policy"]
                total_reward += reward[0].item()

                self.on_step(step, info)
                self._record_frame(writers)

                step += 1
                if (terminated | timed_out)[0].item():
                    break

            for w in writers.values():
                w.close()
            print(self.episode_summary(ep, step, total_reward))

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
