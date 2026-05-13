# Put the project root on sys.path so `from algos...`, `from hrl_utils.paths...`,
# `from assets.robots.g1_cfg...`, etc. resolve regardless of which python
# launches us (kit python rewrites PYTHONPATH, so an env-var-only approach
# isn't reliable).
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import re
import importlib
import torch
import wandb
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from isaaclab.app import AppLauncher


def get_next_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_ids = []
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"run_(\d+)", p.name)
        if m:
            run_ids.append(int(m.group(1)))
    next_id = 0 if not run_ids else max(run_ids) + 1
    run_dir = base_dir / f"run_{next_id:02d}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):

    app_launcher   = AppLauncher(headless=cfg.headless)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import tasks

    torch.manual_seed(cfg.seed)

    # ── Run directory ──────────────────────────────────────────────────────
    base_dir = Path(cfg.log_root) / cfg.task.log_name / cfg.algo.name
    run_dir  = get_next_run_dir(base_dir)
    for d in ["checkpoints", "hydra", "wandb", "eval"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)
    print(f"[train] run dir: {run_dir}")

    # ── Save config ────────────────────────────────────────────────────────
    OmegaConf.save(cfg, run_dir / "hydra" / "config_resolved.yaml", resolve=True)
    with open(run_dir / "hydra" / "overrides.txt", "w") as f:
        f.write("\n".join(HydraConfig.get().overrides.task))

    # ── W&B ───────────────────────────────────────────────────────────────
    wandb.init(
        project  = cfg.wandb.project,
        name     = f"{cfg.algo.name}_{cfg.task.log_name}_{run_dir.name}",
        group    = f"{cfg.task.log_name}_{cfg.algo.name}",
        job_type = "train",
        tags     = [
            cfg.task.log_name,
            cfg.algo.name,
            f"seed{cfg.seed}",
            f"n{cfg.num_envs}",
        ],
        notes    = cfg.wandb.get("notes", ""),
        dir      = str(run_dir / "wandb"),
        mode     = cfg.wandb.mode,
        config   = OmegaConf.to_container(cfg, resolve=True),
    )

    # ── Build env_cfg ──────────────────────────────────────────────────────
    module      = importlib.import_module(cfg.task.env_cfg_module)
    EnvCfgClass = getattr(module, cfg.task.env_cfg_class)
    env_cfg     = EnvCfgClass()

    # push num_envs
    env_cfg.scene.num_envs = cfg.num_envs

    # push all task yaml fields that exist on env_cfg
    task_dict = OmegaConf.to_container(cfg.task, resolve=True)
    # Hydra-only fields that don't belong on env_cfg.
    SCRIPT_KEYS = {
        "gym_id", "log_name",
        "env_cfg_module", "env_cfg_class",
        "evaluator_module", "evaluator_class",
        "cameras", "viewer",
        "eval",     # eval-time overrides; consumed by scripts/eval.py only
    }
    for key, val in task_dict.items():
        if key in SCRIPT_KEYS:
            continue
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, val)

    env = gym.make(cfg.task.gym_id, cfg=env_cfg)

    # ── Trainer ───────────────────────────────────────────────────────────
    from algos import TRAINER_REGISTRY
    trainer = TRAINER_REGISTRY[cfg.algo.name](env=env, cfg=cfg, run_dir=run_dir)
    trainer.run()

    # ── Cleanup ───────────────────────────────────────────────────────────
    wandb.finish()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()