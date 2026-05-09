"""
SAPG algorithm -- Split and Aggregate Policy Gradients.

Implements the four SAPG mechanisms on top of PPO:
    A. Block setup -- partition envs, assign coefficients
    B. Obs augmentation -- append block coefficient to observations
    C. Batch augmentation -- roll coefficients across blocks, recompute values
    D. Per-block entropy + coef_cond sigma in the loss
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from algos.sapg.buffer import SAPGRolloutBuffer
from algos.sapg.network import SAPGActorCritic
from algos.sapg.utils import filter_leader, swap_and_flatten01
from utils.logging import explained_variance
from utils.normalization import RunningMeanStd


class SAPG:
    """SAPG algorithm with leader-follower batch augmentation."""

    def __init__(self, obs_dim, action_dim, num_envs, cfg, device):
        self.cfg = cfg
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_envs = num_envs

        # -- Block setup (Mechanism A) -----------------------------------------
        self.num_blocks = cfg.num_blocks
        self.block_size = num_envs // self.num_blocks
        assert num_envs % self.num_blocks == 0, (
            f"num_envs ({num_envs}) must be divisible by num_blocks ({self.num_blocks})"
        )

        env_block_ids = torch.arange(self.num_blocks, device=device).repeat_interleave(
            self.block_size
        )
        self.env_block_ids = env_block_ids

        self.block_coefs = torch.linspace(50.0, 0.0, self.num_blocks, device=device)
        self.coef_embd = self.block_coefs[env_block_ids].unsqueeze(1)

        entropy_per_block = torch.linspace(
            0.5, 0.0, self.num_blocks, device=device
        ) * cfg.entropy_coef_scale
        self.entropy_coef_per_env = entropy_per_block[env_block_ids]

        self.block_ids_unique = self.block_coefs

        # -- Network -----------------------------------------------------------
        self.network = SAPGActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=list(cfg.hidden_dims),
            activation=cfg.activation,
            block_ids=self.block_ids_unique,
            extra_param_size=cfg.extra_param_size,
        ).to(device)

        self.optimizer = Adam(self.network.parameters(), lr=cfg.learning_rate)

        self.lr_schedule  = getattr(cfg, "lr_schedule",  "fixed")
        self.kl_threshold = getattr(cfg, "kl_threshold", 0.016)
        self.lr_min       = getattr(cfg, "lr_min",       1.0e-6)
        self.lr_max       = getattr(cfg, "lr_max",       1.0e-2)
        self.current_lr   = float(cfg.learning_rate)

        self.buffer = SAPGRolloutBuffer(
            num_steps=cfg.num_steps_per_env,
            num_envs=num_envs,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )

        self.reward_scale = getattr(cfg, "reward_scale", 1.0)
        self.value_coef = getattr(cfg, "value_coef", 4.0)
        self.bounds_loss_coef = getattr(cfg, "bounds_loss_coef", 0.0001)
        self.off_policy_ratio = getattr(cfg, "off_policy_ratio", 1.0)
        self.horizon_length = cfg.num_steps_per_env

        # -- Value normalization -----------------------------------------------
        self.normalize_value = getattr(cfg, "normalize_value", False)
        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((1,)).to(device)

        self.value_bootstrap = getattr(cfg, "value_bootstrap", False)

        # -- Welford tracks ONLY the 108 raw dims -----------------------------
        self.obs_mean = torch.zeros(obs_dim, device=device)
        self.obs_var = torch.ones(obs_dim, device=device)
        self.obs_count = torch.tensor(1.0, device=device)

    # -- Obs handling ----------------------------------------------------------

    def build_network_input(self, obs_norm_108: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs_norm_108, self.coef_embd], dim=1)

    def _update_obs_stats(self, obs):
        batch = obs.reshape(-1, obs.shape[-1]).detach()
        n = batch.shape[0]
        new_count = self.obs_count + n
        new_mean = (self.obs_count * self.obs_mean + batch.sum(0)) / new_count
        delta_old = batch - self.obs_mean
        delta_new = batch - new_mean
        new_var = (self.obs_var * self.obs_count + (delta_old * delta_new).sum(0)) / new_count
        self.obs_mean = new_mean
        self.obs_var = new_var.clamp(min=1e-6)
        self.obs_count = new_count

    def normalize_obs(self, obs, update_stats=False):
        obs = obs.clamp(-100.0, 100.0)
        if update_stats:
            self._update_obs_stats(obs)
        return ((obs - self.obs_mean) / (self.obs_var.sqrt() + 1e-8)).clamp(-10.0, 10.0)

    def _update_lr_adaptive(self, mean_kl: float):
        if self.lr_schedule != "adaptive":
            return
        if mean_kl > 2.0 * self.kl_threshold:
            self.current_lr = max(self.current_lr / 1.5, self.lr_min)
        elif mean_kl < 0.5 * self.kl_threshold:
            self.current_lr = min(self.current_lr * 1.5, self.lr_max)
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.current_lr

    def _denorm_value(self, v: torch.Tensor) -> torch.Tensor:
        """Denormalize a value tensor if normalize_value is enabled."""
        if self.normalize_value:
            return self.value_mean_std.denormalize(v.unsqueeze(-1)).squeeze(-1)
        return v

    # -- Rollout collection ----------------------------------------------------

    @torch.no_grad()
    def collect_step(self, obs_aug_norm):
        action, log_prob, _, value, mu, sigma = self.network.get_action_and_value(
            obs_aug_norm
        )
        value = self._denorm_value(value)
        return action, log_prob, value, mu, sigma

    def insert(self, obs, actions, rewards, dones, values, log_probs, mus, sigmas):
        self.buffer.insert(obs, actions, rewards, dones, values, log_probs, mus, sigmas)

    @torch.no_grad()
    def compute_returns(self, last_obs_raw_108):
        last_norm = self.normalize_obs(last_obs_raw_108, update_stats=False)
        last_input = self.build_network_input(last_norm)
        last_value = self.network.get_value(last_input)
        last_value = self._denorm_value(last_value)
        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.cfg.gamma,
            lam=self.cfg.lam,
        )

    # -- Batch augmentation (Mechanism C) --------------------------------------

    @torch.no_grad()
    def augment_batch(self, batch_dict: dict) -> dict:
        """SAPG batch augmentation.

        All values/returns in the batch are in REAL reward scale (denormalized).
        Value normalization happens later in update(), before the epoch loop.
        """
        num_blocks = self.num_blocks
        block_size = self.block_size
        horizon = self.horizon_length
        orig_len = len(batch_dict["obses"])

        num_repeat = min(num_blocks, int(self.off_policy_ratio) + 1)
        repeat_idxs = [0] + list(
            np.random.choice(range(1, num_blocks), num_repeat - 1, replace=False)
        )

        new_batch = {}

        coef_embd_flat = self.coef_embd.repeat_interleave(horizon, dim=0)
        raw_108 = batch_dict["obses"]
        raw_108_norm = self.normalize_obs(raw_108, update_stats=False)

        copies = []
        for k in repeat_idxs:
            rolled_coef = torch.roll(coef_embd_flat, block_size * horizon * k, dims=0)
            copies.append(torch.cat([raw_108_norm, rolled_coef], dim=1))
        obses_cat = torch.cat(copies, dim=0)

        mask = torch.zeros(len(obses_cat), dtype=torch.bool, device=self.device)
        mask[orig_len:] = True

        obses_cat = filter_leader(obses_cat, orig_len, repeat_idxs, num_blocks)
        mask = filter_leader(mask, orig_len, repeat_idxs, num_blocks)
        new_batch["obses"] = obses_cat
        new_batch["off_policy_mask"] = mask
        self._last_off_policy_frac = float(mask.float().mean().item())

        for key, val in batch_dict.items():
            if key in ("obses", "values", "returns", "rewards", "dones"):
                continue
            val_cat = torch.cat([val] * len(repeat_idxs), dim=0)
            val_cat = filter_leader(val_cat, orig_len, repeat_idxs, num_blocks)
            new_batch[key] = val_cat

        # -- Recompute values + TD(0) returns for follower copies -------------
        returns_list = [batch_dict["returns"]]
        values_list = [batch_dict["values"]]

        mb_rewards_2d = self.buffer.rewards
        mb_dones_2d = self.buffer.dones
        buf_obs_raw = self.buffer.obs

        for r_k in repeat_idxs[1:]:
            flat_raw = buf_obs_raw.reshape(horizon * self.num_envs, -1)
            flat_norm = self.normalize_obs(flat_raw, update_stats=False)
            rolled_coef = torch.roll(
                self.coef_embd, block_size * r_k, dims=0
            )
            rolled_coef_flat = rolled_coef.repeat(horizon, 1)
            net_input = torch.cat([flat_norm, rolled_coef_flat], dim=1)

            new_values = []
            chunk = 8192
            for i in range(0, len(net_input), chunk):
                v = self.network.get_value(net_input[i : i + chunk])
                # Denormalize: augment_batch works in real reward scale
                v = self._denorm_value(v)
                new_values.append(v)
            new_values = torch.cat(new_values, dim=0)

            last_raw = buf_obs_raw[-1]
            last_norm = self.normalize_obs(last_raw, update_stats=False)
            last_input = torch.cat([last_norm, rolled_coef], dim=1)
            last_values = self.network.get_value(last_input)
            last_values = self._denorm_value(last_values)

            new_values_2d = new_values.view(horizon, self.num_envs)
            all_values = torch.cat(
                [new_values_2d, last_values.unsqueeze(0)], dim=0
            )
            td0_returns = mb_rewards_2d + self.cfg.gamma * all_values[1:] * (
                1 - mb_dones_2d
            )

            def flatten_tn(x):
                return x.transpose(0, 1).reshape(horizon * self.num_envs)

            returns_list.append(flatten_tn(td0_returns))
            values_list.append(flatten_tn(new_values_2d))

        new_batch["returns"] = torch.cat(returns_list, dim=0)
        new_batch["values"] = torch.cat(values_list, dim=0)
        new_batch["returns"] = filter_leader(
            new_batch["returns"], orig_len, repeat_idxs, num_blocks
        )
        new_batch["values"] = filter_leader(
            new_batch["values"], orig_len, repeat_idxs, num_blocks
        )

        return new_batch

    # -- Per-block entropy coefficient lookup ----------------------------------

    def _get_per_block_entropy_coef(self, obs_batch: torch.Tensor) -> torch.Tensor:
        coef_vals = obs_batch[:, self.obs_dim]
        ec_candidates = self.entropy_coef_per_env[:: self.block_size]
        ec_ids = self.block_ids_unique
        diffs = (coef_vals.unsqueeze(1) - ec_ids.unsqueeze(0)).abs()
        indices = diffs.argmin(dim=1)
        return ec_candidates[indices]

    # -- Update ----------------------------------------------------------------

    def update(self, augmented_batch: dict) -> dict:
        # -- Pre-normalization diagnostics (real reward scale, on-policy only) -
        with torch.no_grad():
            flat_values  = self.buffer.values.reshape(-1)
            flat_returns = self.buffer.returns.reshape(-1)
            ev         = explained_variance(flat_values, flat_returns)
            value_mean = flat_values.mean().item()
            value_std  = flat_values.std().item()
            sigma_mean = self.network.sigma.exp().mean().item()

            per_env_return = self.buffer.rewards.sum(dim=0)
            per_block_return = torch.zeros(self.num_blocks, device=self.device)
            block_counts    = torch.zeros(self.num_blocks, device=self.device)
            per_block_return.scatter_add_(0, self.env_block_ids, per_env_return)
            block_counts.scatter_add_(
                0, self.env_block_ids, torch.ones_like(per_env_return)
            )
            per_block_return = per_block_return / block_counts.clamp(min=1)

            per_block_sigma = self.network.sigma.exp().mean(dim=-1)
            per_block_extra_norm = self.network.extra_params.norm(dim=-1)
            off_policy_frac = getattr(self, "_last_off_policy_frac", float("nan"))

        # -- Normalize value targets in the augmented batch --------------------
        if self.normalize_value:
            # Update stats on real-scale returns from the ON-POLICY buffer
            self.value_mean_std.update(self.buffer.returns.reshape(-1, 1))
            # Normalize the AUGMENTED batch values/returns in-place
            augmented_batch["values"] = self.value_mean_std.normalize(
                augmented_batch["values"].unsqueeze(-1)
            ).squeeze(-1)
            augmented_batch["returns"] = self.value_mean_std.normalize(
                augmented_batch["returns"].unsqueeze(-1)
            ).squeeze(-1)

        total_loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0
        approx_kl_sum = 0.0
        clip_frac_sum = 0.0
        num_updates = 0

        for _ in range(self.cfg.num_learning_epochs):
            epoch_kl_sum  = 0.0
            epoch_updates = 0

            for mb in self.buffer.get_batches_from_flat(
                augmented_batch, self.cfg.num_mini_batches
            ):
                obs = mb["obs"]
                actions = mb["actions"]
                old_log_probs = mb["old_log_probs"]
                advantages = mb["advantages"]
                returns = mb["returns"]
                old_values = mb["old_values"]
                off_policy_mask = mb["off_policy_mask"]

                _, new_log_probs, entropy, new_values, mu, sigma = (
                    self.network.get_action_and_value(obs, action=actions)
                )

                ratio = (new_log_probs - old_log_probs).exp()

                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean()
                    clip_frac = (torch.abs(ratio - 1.0) > self.cfg.clip_param).float().mean()
                approx_kl_sum += approx_kl.item()
                clip_frac_sum += clip_frac.item()
                epoch_kl_sum  += approx_kl.item()
                epoch_updates += 1

                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param
                ) * advantages
                loss_policy = -torch.min(surr1, surr2).mean()

                values_clipped = old_values + torch.clamp(
                    new_values - old_values,
                    -self.cfg.clip_param,
                    self.cfg.clip_param,
                )
                loss_v_unclipped = (new_values - returns).pow(2)
                loss_v_clipped = (values_clipped - returns).pow(2)
                loss_value = torch.max(loss_v_unclipped, loss_v_clipped).mean()

                per_sample_ec = self._get_per_block_entropy_coef(obs)
                loss_entropy = -(per_sample_ec * entropy).mean()

                soft_bound = 1.1
                mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                loss_bounds = (mu_loss_low + mu_loss_high).sum(dim=-1).mean()

                loss = (
                    loss_policy
                    + self.value_coef * loss_value
                    + loss_entropy
                    + self.bounds_loss_coef * loss_bounds
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                policy_loss += loss_policy.item()
                value_loss += loss_value.item()
                entropy_loss += loss_entropy.item()
                num_updates += 1

            if epoch_updates > 0:
                self._update_lr_adaptive(epoch_kl_sum / epoch_updates)

        self.buffer.reset()

        n = max(num_updates, 1)
        return {
            "loss/total":   total_loss   / n,
            "loss/policy":  policy_loss  / n,
            "loss/value":   value_loss   / n,
            "loss/entropy": entropy_loss / n,
            "policy/approx_kl":         approx_kl_sum / n,
            "policy/clip_fraction":     clip_frac_sum / n,
            "policy/sigma_mean":        sigma_mean,
            "value/explained_variance": ev,
            "value/mean":               value_mean,
            "value/std":                value_std,
            "train/learning_rate":      self.current_lr,
            "sapg/off_policy_fraction": off_policy_frac,
            **{f"sapg/return_block_{i}":    per_block_return[i].item()    for i in range(self.num_blocks)},
            **{f"sapg/sigma_block_{i}":     per_block_sigma[i].item()     for i in range(self.num_blocks)},
            **{f"sapg/extra_norm_block_{i}": per_block_extra_norm[i].item() for i in range(self.num_blocks)},
        }