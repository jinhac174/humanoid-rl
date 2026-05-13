import re
import time
import torch
import wandb
from pathlib import Path
from omegaconf import OmegaConf

from hrl_utils.logging import iter_loggable_items

_EXCLUDE_FROM_ROLLOUT_AGG = {"task_episode_success_per_env"}


class PPOTrainer:

    def __init__(self, env, cfg, run_dir: Path):
        self.env     = env
        self.cfg     = cfg
        self.run_dir = run_dir
        self.device  = env.unwrapped.device

        obs_dim    = env.unwrapped.single_observation_space["policy"].shape[0]
        action_dim = env.unwrapped.single_action_space.shape[0]
        num_envs   = cfg.num_envs

        from algos.ppo.ppo import PPO
        self.agent = PPO(
            obs_dim    = obs_dim,
            action_dim = action_dim,
            num_envs   = num_envs,
            cfg        = cfg.algo,
            device     = self.device,
        )

        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._episode_return   = torch.zeros(num_envs, device=self.device)
        self._episode_length   = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._last_episode_return  = torch.zeros(num_envs, device=self.device)
        self._last_episode_length  = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._last_episode_success = torch.zeros(num_envs, device=self.device)
        self._completions_this_iter = 0

    # -- Checkpoint ------------------------------------------------------------

    def save_checkpoint(self, iteration: int):
        path = self.ckpt_dir / f"model_{iteration}.pt"
        ckpt = {
            "model":     self.agent.network.state_dict(),
            "optimizer": self.agent.optimizer.state_dict(),
            "obs_mean":  self.agent.obs_mean.cpu(),
            "obs_var":   self.agent.obs_var.cpu(),
            "obs_count": self.agent.obs_count.cpu(),
            "iteration": iteration,
        }
        if self.agent.normalize_value:
            ckpt["value_mean_std"] = self.agent.value_mean_std.state_dict()
        torch.save(ckpt, path)
        return path

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.network.load_state_dict(ckpt["model"])
        self.agent.optimizer.load_state_dict(ckpt["optimizer"])
        self.agent.obs_mean  = ckpt["obs_mean"].to(self.device)
        self.agent.obs_var   = ckpt["obs_var"].to(self.device)
        self.agent.obs_count = ckpt["obs_count"].to(self.device)
        if self.agent.normalize_value and "value_mean_std" in ckpt:
            self.agent.value_mean_std.load_state_dict(ckpt["value_mean_std"])
        return ckpt.get("iteration", 0)

    # -- Training loop ---------------------------------------------------------

    def run(self, start_iteration=0):
        cfg   = self.cfg
        agent = self.agent
        env   = self.env

        obs_dict, _ = env.reset()
        obs = agent.normalize_obs(obs_dict["policy"], update_stats=True)

        max_iter = cfg.algo.max_iterations

        for iteration in range(start_iteration, max_iter):

            rollout_log_sums  = {}
            rollout_log_count = 0

            t0 = time.time()
            for _ in range(cfg.algo.num_steps_per_env):
                actions, log_probs, values = agent.collect_step(obs)

                obs_dict, rewards, terminated, timed_out, info = env.step(actions)
                next_obs = obs_dict["policy"]

                for k, v in iter_loggable_items(info):
                    if k in _EXCLUDE_FROM_ROLLOUT_AGG:
                        continue
                    if isinstance(v, torch.Tensor):
                        rollout_log_sums[k] = rollout_log_sums.get(k, 0.0) + float(v.mean())
                    elif isinstance(v, float):
                        rollout_log_sums[k] = rollout_log_sums.get(k, 0.0) + v
                rollout_log_count += 1

                # -- Value bootstrap (rl_games convention) ---------------------
                # At truncation steps, add gamma * V(s_t) to reward so the GAE
                # bootstrap isn't zeroed by the done mask.
                rewards = rewards * agent.reward_scale
                if agent.value_bootstrap:
                    bootstrap = cfg.algo.gamma * values.clamp(-1e4, 1e4) * timed_out.float()
                    rewards = rewards + bootstrap

                next_obs = agent.normalize_obs(next_obs, update_stats=True)
                dones    = (terminated | timed_out).float()

                # -- Episode tracking ------------------------------------------
                self._episode_return += rewards
                self._episode_length += 1
                done_bool = (terminated | timed_out)
                if done_bool.any():
                    finished = done_bool
                    self._last_episode_return[finished] = self._episode_return[finished]
                    self._last_episode_length[finished] = self._episode_length[finished].float()
                    if "task_episode_success_per_env" in info:
                        success_per_env = info["task_episode_success_per_env"]
                        self._last_episode_success[finished] = success_per_env[finished].float()
                    else:
                        self._last_episode_success[finished] = terminated[finished].float()
                    self._episode_return[finished] = 0.0
                    self._episode_length[finished] = 0
                    self._completions_this_iter += int(finished.sum().item())

                agent.insert(obs, actions, rewards, dones, values, log_probs)
                obs = next_obs

            rollout_elapsed = max(time.time() - t0, 1e-9)
            sim_steps = cfg.num_envs * cfg.algo.num_steps_per_env
            sps = sim_steps / rollout_elapsed

            step_reward_mean    = agent.buffer.rewards.mean().item()
            rollout_return_mean = agent.buffer.rewards.sum(dim=0).mean().item()

            agent.compute_returns(obs)
            losses = agent.update()

            log_info = {}
            if rollout_log_count > 0:
                log_info = {k: v / rollout_log_count for k, v in rollout_log_sums.items()}

            metrics = {
                "rollout/step_reward_mean": step_reward_mean,
                "rollout/return_mean":      rollout_return_mean,
                "train/iteration":          iteration,
                **losses,
                **log_info,
            }
            metrics.update({
                "episode/return_mean":  self._last_episode_return.mean().item(),
                "episode/length_mean":  self._last_episode_length.mean().item(),
                "episode/success_rate": self._last_episode_success.mean().item(),
                "episode/completions":  self._completions_this_iter,
                "sim/sps":             sps,
                "sim/rollout_seconds": rollout_elapsed,
            })
            self._completions_this_iter = 0

            wandb.log(metrics, step=iteration)

            if iteration % 100 == 0:
                print(
                    f"[{iteration}/{max_iter}] "
                    f"ret={rollout_return_mean:.2f} "
                    f"rew={step_reward_mean:.4f} "
                    f"loss={losses['loss/total']:.4f}"
                )

            if iteration % 500 == 0 and iteration > 0:
                path = self.save_checkpoint(iteration)
                print(f"[train] checkpoint: {path}")

        path = self.save_checkpoint(max_iter)
        print(f"[train] final model: {path}")