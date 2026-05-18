"""rsl_rl PPO trainer adapter.

rsl_rl is the PPO implementation IsaacLab uses for its official reference
locomotion tasks. We wrap rsl_rl's ``OnPolicyRunner`` so our
``scripts/train.py`` can dispatch it the same way it dispatches our custom
PPO / SAPG / EPO.

We go through IsaacLab's @configclass schemas (``RslRlOnPolicyRunnerCfg`` +
``RslRlPpoActorCriticCfg`` + ``RslRlPpoAlgorithmCfg``) and call IsaacLab's
``handle_deprecated_rsl_rl_cfg`` before producing the dict for the runner.
That handler translates the legacy ``policy`` block into the new
``actor`` / ``critic`` blocks required by rsl-rl >= 4.0, so the trainer
keeps working across rsl_rl versions.
"""
from __future__ import annotations

import importlib.metadata as _md
from pathlib import Path

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner


class RslRlPPOTrainer:
    """Same constructor signature as PPOTrainer / SAPGTrainer / EPOTrainer."""

    def __init__(self, env, cfg, run_dir):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.device = env.unwrapped.device

        # rsl_rl needs its own env wrapper.
        clip_actions = cfg.algo.get("clip_actions", None)
        self.env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

        a = cfg.algo
        wandb_project = (
            cfg.task.get("wandb_project", None) or cfg.wandb.project
        )

        # Build the IsaacLab @configclass instance.
        agent_cfg = RslRlOnPolicyRunnerCfg(
            seed=int(cfg.seed),
            device=str(self.device),
            num_steps_per_env=int(a.num_steps_per_env),
            max_iterations=int(a.max_iterations),
            empirical_normalization=bool(a.empirical_normalization),
            obs_groups={"actor": ["policy"], "critic": ["policy"]},
            clip_actions=clip_actions,
            save_interval=int(a.save_interval),
            experiment_name=str(cfg.task.log_name),
            run_name=self.run_dir.name,
            logger=str(a.get("logger", "wandb")),
            wandb_project=str(wandb_project),
            policy=RslRlPpoActorCriticCfg(
                init_noise_std=float(a.init_noise_std),
                actor_obs_normalization=bool(a.empirical_normalization),
                critic_obs_normalization=bool(a.empirical_normalization),
                actor_hidden_dims=list(a.actor_hidden_dims),
                critic_hidden_dims=list(a.critic_hidden_dims),
                activation=str(a.activation),
            ),
            algorithm=RslRlPpoAlgorithmCfg(
                value_loss_coef=float(a.value_loss_coef),
                use_clipped_value_loss=bool(a.use_clipped_value_loss),
                clip_param=float(a.clip_param),
                entropy_coef=float(a.entropy_coef),
                num_learning_epochs=int(a.num_learning_epochs),
                num_mini_batches=int(a.num_mini_batches),
                learning_rate=float(a.learning_rate),
                schedule=str(a.schedule),
                gamma=float(a.gamma),
                lam=float(a.lam),
                desired_kl=float(a.desired_kl),
                max_grad_norm=float(a.max_grad_norm),
            ),
        )

        # Translate the legacy 'policy' block into 'actor'/'critic' for newer
        # rsl-rl. With rsl-rl < 4.0 this is a no-op.
        try:
            installed_version = _md.version("rsl-rl-lib")
        except _md.PackageNotFoundError:
            installed_version = "0.0.0"
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # rsl_rl writes model_<iter>.pt + tensorboard files into log_dir.
        # Use our standard checkpoints/ subdir so eval.py paths work
        # without modification.
        log_dir = self.run_dir / "checkpoints"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.runner = OnPolicyRunner(
            self.env,
            agent_cfg.to_dict(),
            log_dir=str(log_dir),
            device=str(self.device),
        )

    def run(self, start_iteration: int = 0):
        self.runner.learn(
            num_learning_iterations=int(self.cfg.algo.max_iterations),
            init_at_random_ep_len=True,
        )
