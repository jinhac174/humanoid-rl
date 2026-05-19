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


def next_run_index(base_dir: Path, algo: str) -> int:
    """Return the next free NN under ``base_dir`` for runs of this algo.

    Scans for sibling directories matching ``<algo>_NN`` and returns
    ``max(NN) + 1`` (or 0 if none exist). Pure read-only.
    """
    if not base_dir.exists():
        return 0
    pat = re.compile(rf"^{re.escape(algo)}_(\d+)$")
    ids = []
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if m:
            ids.append(int(m.group(1)))
    return 0 if not ids else max(ids) + 1


def build_run_names(cfg: DictConfig, base_dir: Path) -> tuple[str, str]:
    """Return ``(disk_name, wandb_name)``.

    Disk name is shell-friendly (lowercase, underscore) — e.g. ``ppo_00``.
    Wandb name is display-friendly (uppercase, pipe-separator) — e.g.
    ``PPO | 00``. NN is independently incremented per ``<task>/<algo>``
    directory: ``outputs/velocity_tracking/ppo/ppo_00/`` and
    ``outputs/reorient/ppo/ppo_00/`` are unrelated.

    Optional ``+run_tag=<short_tag>`` CLI override is appended to both:
    ``ppo_00_lr3e4`` / ``PPO | 00 | lr3e4``.
    """
    idx = next_run_index(base_dir, cfg.algo.name)
    nn = f"{idx:02d}"
    disk_name  = f"{cfg.algo.name}_{nn}"
    wandb_name = f"{cfg.algo.name.upper()} | {nn}"
    run_tag = cfg.get("run_tag", None)
    if run_tag:
        disk_name  = f"{disk_name}_{run_tag}"
        wandb_name = f"{wandb_name} | {run_tag}"
    return disk_name, wandb_name


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):

    app_launcher   = AppLauncher(headless=cfg.headless)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import tasks

    torch.manual_seed(cfg.seed)

    # ── Run directory ──────────────────────────────────────────────────────
    # outputs/<task>/<algo>/<algo>_NN/    e.g. outputs/velocity_tracking/ppo/ppo_00
    # Wandb run name is the same NN, but display-formatted ("PPO | 00").
    # NN is per-(task, algo); ppo_00 in different tasks aren't related.
    base_dir = Path(cfg.log_root) / cfg.task.log_name / cfg.algo.name
    base_dir.mkdir(parents=True, exist_ok=True)
    disk_name, wandb_name = build_run_names(cfg, base_dir)
    run_dir = base_dir / disk_name
    run_dir.mkdir(parents=False, exist_ok=False)
    for d in ["checkpoints", "hydra", "wandb", "eval"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)
    print(f"[train] run dir:   {run_dir}")
    print(f"[train] wandb run: {wandb_name}")

    # ── Save config ────────────────────────────────────────────────────────
    OmegaConf.save(cfg, run_dir / "hydra" / "config_resolved.yaml", resolve=True)
    with open(run_dir / "hydra" / "overrides.txt", "w") as f:
        f.write("\n".join(HydraConfig.get().overrides.task))

    # ── W&B ───────────────────────────────────────────────────────────────
    # Per-task wandb project (declared in configs/task/<task>.yaml). Falls
    # back to the global cfg.wandb.project if the task didn't set one.
    wandb_project = (
        getattr(cfg.task, "wandb_project", None) or cfg.wandb.project
    )
    # Tags are two chips for filtering in the wandb UI: the task label and
    # the algo. Everything else (num_envs, seed, num_blocks, run_tag) lives
    # in wandb.config via OmegaConf.to_container below and is queryable
    # there — no need to clutter the tag list.
    task_tag = getattr(cfg.task, "wandb_tag", None) or cfg.task.log_name
    tags = [task_tag, cfg.algo.name]

    # rsl_rl's OnPolicyRunner manages its own wandb session when
    # logger="wandb" in its config — initializing wandb here would conflict.
    # The trainer reads cfg.task.wandb_project + run_dir.name and passes them
    # through to rsl_rl so the run still lands in the right project.
    if cfg.algo.name == "rsl_rl_ppo":
        print(f"[train] rsl_rl_ppo: wandb managed by rsl_rl "
              f"(project={wandb_project}, run={wandb_name})")
    else:
        wandb.init(
            project  = wandb_project,
            name     = wandb_name,
            group    = f"{cfg.task.log_name}_{cfg.algo.name}",
            job_type = "train",
            tags     = tags,
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
        "wandb_project", "wandb_tag",
        "max_iterations",   # task-level training budget; pushed onto cfg.algo below
        "cameras", "viewer",
        "eval",     # eval-time overrides; consumed by scripts/eval.py only
    }
    for key, val in task_dict.items():
        if key in SCRIPT_KEYS:
            continue
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, val)

    # Task-level training budget overrides the algo's fallback. ``max_iterations``
    # is a function of task difficulty + per-iter sample count, not of the
    # specific RL algorithm, so it lives on the task yaml. If a task omits it
    # we fall back to the algo's default and warn.
    task_max_iter = task_dict.get("max_iterations", None)
    if task_max_iter is not None:
        cfg.algo.max_iterations = int(task_max_iter)
        print(f"[train] max_iterations = {task_max_iter} (from task yaml)")
    else:
        print(f"[train] WARNING: task '{cfg.task.log_name}' yaml has no "
              f"max_iterations; falling back to algo default "
              f"{cfg.algo.max_iterations}")

    env = gym.make(cfg.task.gym_id, cfg=env_cfg)

    # ── Trainer ───────────────────────────────────────────────────────────
    from algos import TRAINER_REGISTRY
    trainer = TRAINER_REGISTRY[cfg.algo.name](env=env, cfg=cfg, run_dir=run_dir)

    # Optional warm-start: load a phase-1 (or any task with a 141-d
    # locomotion-prefix obs) checkpoint into the freshly built trainer.
    # Handles obs-dim mismatch by zero-filling the new columns of the
    # first weight matrix. See algos/warm_start.py for details.
    if cfg.get("checkpoint", None):
        if cfg.algo.name == "rsl_rl_ppo":
            print("[train] WARNING: warm-start not supported for rsl_rl_ppo "
                  "(trainer manages its own state); ignoring cfg.checkpoint")
        else:
            from algos.warm_start import warm_start_load
            warm_start_load(trainer, cfg.checkpoint)

    trainer.run()

    # ── Cleanup ───────────────────────────────────────────────────────────
    if wandb.run is not None:    # closes whichever run is active (ours or rsl_rl's)
        wandb.finish()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()