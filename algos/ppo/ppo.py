import torch
import torch.nn as nn
from torch.optim import Adam

from algos.ppo.buffer import RolloutBuffer
from algos.ppo.network import ActorCritic
from hrl_utils.logging import explained_variance
from hrl_utils.normalization import RunningMeanStd


class PPO:
    def __init__(self, obs_dim, action_dim, num_envs, cfg, device):
        self.cfg    = cfg
        self.device = device

        self.network = ActorCritic(
            obs_dim            = obs_dim,
            action_dim         = action_dim,
            shared             = getattr(cfg, "shared", False),
            hidden_dims        = list(getattr(cfg, "hidden_dims", [1024, 1024, 512, 512])),
            actor_hidden_dims  = list(getattr(cfg, "actor_hidden_dims", [512, 256, 128])),
            critic_hidden_dims = list(getattr(cfg, "critic_hidden_dims", [512, 256, 128])),
            activation         = cfg.activation,
            init_noise_std     = cfg.init_noise_std,
            use_tanh           = getattr(cfg, "use_tanh", False),
        ).to(device)

        self.optimizer = Adam(self.network.parameters(), lr=cfg.learning_rate)

        self.lr_schedule  = getattr(cfg, "lr_schedule",  "fixed")
        self.kl_threshold = getattr(cfg, "kl_threshold", 0.016)
        self.lr_min       = getattr(cfg, "lr_min",       1.0e-6)
        self.lr_max       = getattr(cfg, "lr_max",       1.0e-2)
        self.current_lr   = float(cfg.learning_rate)

        self.buffer = RolloutBuffer(
            num_steps  = cfg.num_steps_per_env,
            num_envs   = num_envs,
            obs_dim    = obs_dim,
            action_dim = action_dim,
            device     = device,
        )

        self.reward_scale = getattr(cfg, "reward_scale", 1.0)
        self.value_coef = getattr(cfg, "value_coef", 0.5)
        self.bounds_loss_coef = getattr(cfg, "bounds_loss_coef", 0.0001)

        # -- Value normalization (rl_games: normalize_value) -------------------
        # Critic outputs normalized values; we denormalize before storing in
        # the buffer so GAE works in real reward scale. Before the update we
        # re-normalize stored values/returns so the loss is in normalized space.
        self.normalize_value = getattr(cfg, "normalize_value", False)
        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((1,)).to(device)

        # -- Value bootstrap (rl_games: value_bootstrap) -----------------------
        # At truncation (timed_out), add gamma * V(s_t) to the reward so the
        # GAE bootstrap isn't masked by the done flag.
        self.value_bootstrap = getattr(cfg, "value_bootstrap", False)

        # -- Online obs normalizer (Welford) -----------------------------------
        self.obs_mean  = torch.zeros(obs_dim, device=device)
        self.obs_var   = torch.ones(obs_dim, device=device)
        self.obs_count = torch.tensor(1.0, device=device)

    # -- Obs normalisation -----------------------------------------------------
    def _update_obs_stats(self, obs):
        batch          = obs.reshape(-1, obs.shape[-1]).detach()
        n              = batch.shape[0]
        new_count      = self.obs_count + n
        new_mean       = (self.obs_count * self.obs_mean + batch.sum(0)) / new_count
        delta_old      = batch - self.obs_mean
        delta_new      = batch - new_mean
        new_var        = (self.obs_var * self.obs_count + (delta_old * delta_new).sum(0)) / new_count
        self.obs_mean  = new_mean
        self.obs_var   = new_var.clamp(min=1e-6)
        self.obs_count = new_count

    def normalize_obs(self, obs, update_stats=False):
        obs = obs.clamp(-100.0, 100.0)
        if update_stats:
            self._update_obs_stats(obs)
        return ((obs - self.obs_mean) / (self.obs_var.sqrt() + 1e-8)).clamp(-10.0, 10.0)

    # -- Rollout ---------------------------------------------------------------
    @torch.no_grad()
    def collect_step(self, obs):
        action, log_prob, _, value, _ = self.network.get_action_and_value(obs)
        # Denormalize value so the buffer stores real-scale values for GAE
        if self.normalize_value:
            value = self.value_mean_std.denormalize(value.unsqueeze(-1)).squeeze(-1)
        return action, log_prob, value

    def insert(self, obs, actions, rewards, dones, values, log_probs):
        self.buffer.insert(obs, actions, rewards, dones, values, log_probs)

    @torch.no_grad()
    def compute_returns(self, last_obs):
        last_value = self.network.get_value(last_obs)
        if self.normalize_value:
            last_value = self.value_mean_std.denormalize(
                last_value.unsqueeze(-1)
            ).squeeze(-1)
        self.buffer.compute_returns_and_advantages(
            last_value = last_value,
            gamma = self.cfg.gamma,
            lam   = self.cfg.lam,
        )

    def _update_lr_adaptive(self, mean_kl: float):
        if self.lr_schedule != "adaptive":
            return
        if mean_kl > 2.0 * self.kl_threshold:
            self.current_lr = max(self.current_lr / 1.5, self.lr_min)
        elif mean_kl < 0.5 * self.kl_threshold:
            self.current_lr = min(self.current_lr * 1.5, self.lr_max)
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.current_lr

    # -- Update ----------------------------------------------------------------
    def update(self):
        # -- Pre-normalization diagnostics (real reward scale) -----------------
        with torch.no_grad():
            flat_values  = self.buffer.values.reshape(-1)
            flat_returns = self.buffer.returns.reshape(-1)
            ev         = explained_variance(flat_values, flat_returns)
            value_mean = flat_values.mean().item()
            value_std  = flat_values.std().item()
            sigma_mean = self.network.log_std.exp().mean().item()

        # -- Normalize value targets (rl_games convention) ---------------------
        # Update running stats on real-scale returns, then normalize both
        # values and returns in-place before the PPO epoch loop.
        if self.normalize_value:
            self.value_mean_std.update(self.buffer.returns.reshape(-1, 1))
            self.buffer.values = self.value_mean_std.normalize(
                self.buffer.values.unsqueeze(-1)
            ).squeeze(-1)
            self.buffer.returns = self.value_mean_std.normalize(
                self.buffer.returns.unsqueeze(-1)
            ).squeeze(-1)

        total_loss    = 0.0
        policy_loss   = 0.0
        value_loss    = 0.0
        entropy_loss  = 0.0
        bounds_loss_sum = 0.0
        approx_kl_sum = 0.0
        clip_frac_sum = 0.0
        num_updates   = 0

        for _ in range(self.cfg.num_learning_epochs):
            epoch_kl_sum  = 0.0
            epoch_updates = 0

            for obs, actions, old_log_probs, advantages, returns, old_values in self.buffer.get_batches(self.cfg.num_mini_batches):
                _, new_log_probs, entropy, new_values, mu = (
                    self.network.get_action_and_value(obs, action=actions)
                )

                # -- Policy loss -----------------------------------------------
                ratio = (new_log_probs - old_log_probs).exp()

                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean()
                    clip_frac = (torch.abs(ratio - 1.0) > self.cfg.clip_param).float().mean()
                approx_kl_sum += approx_kl.item()
                clip_frac_sum += clip_frac.item()
                epoch_kl_sum  += approx_kl.item()
                epoch_updates += 1

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * advantages
                loss_policy = -torch.min(surr1, surr2).mean()

                # -- Value loss (clipped, no 0.5 factor -- donor convention) ----
                values_clipped = old_values + torch.clamp(
                    new_values - old_values, -self.cfg.clip_param, self.cfg.clip_param
                )
                loss_v_unclipped = (new_values - returns).pow(2)
                loss_v_clipped   = (values_clipped - returns).pow(2)
                loss_value = torch.max(loss_v_unclipped, loss_v_clipped).mean()

                # -- Entropy loss ----------------------------------------------
                loss_entropy = -entropy.mean()

                # -- Bounds loss -----------------------------------------------
                soft_bound = 1.1
                mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                mu_loss_low  = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                loss_bounds = (mu_loss_low + mu_loss_high).sum(dim=-1).mean()

                loss = (
                    loss_policy
                    + self.value_coef * loss_value
                    + self.cfg.entropy_coef * loss_entropy
                    + self.bounds_loss_coef * loss_bounds
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss    += loss.item()
                policy_loss   += loss_policy.item()
                value_loss    += loss_value.item()
                entropy_loss  += loss_entropy.item()
                bounds_loss_sum += loss_bounds.item()
                num_updates   += 1

            if epoch_updates > 0:
                self._update_lr_adaptive(epoch_kl_sum / epoch_updates)

        self.buffer.reset()

        return {
            "loss/total":   total_loss    / num_updates,
            "loss/policy":  policy_loss   / num_updates,
            "loss/value":   value_loss    / num_updates,
            "loss/entropy": entropy_loss  / num_updates,
            "loss/bounds":  bounds_loss_sum / num_updates,
            "policy/approx_kl":         approx_kl_sum / num_updates,
            "policy/clip_fraction":     clip_frac_sum / num_updates,
            "policy/sigma_mean":        sigma_mean,
            "value/explained_variance": ev,
            "value/mean":               value_mean,
            "value/std":                value_std,
            "train/learning_rate":      self.current_lr,
        }