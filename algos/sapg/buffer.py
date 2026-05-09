"""
SAPG rollout buffer.

Extends the PPO RolloutBuffer with storage for per-step mus and sigmas,
which SAPG needs for KL divergence computation and importance sampling.
"""
import torch


class SAPGRolloutBuffer:
    """On-policy rollout buffer with mu/sigma storage for SAPG."""

    def __init__(self, num_steps, num_envs, obs_dim, action_dim, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # obs_dim here is the RAW env obs dim (108), NOT augmented.
        # The block coefficient is appended after normalization, right before
        # the network call -- it never enters the buffer or Welford stats.
        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.mus = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.sigmas = torch.zeros(num_steps, num_envs, action_dim, device=device)

        # Computed after rollout
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)
        self.step = 0

    def insert(self, obs, actions, rewards, dones, values, log_probs, mus, sigmas):
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.log_probs[self.step] = log_probs
        self.mus[self.step] = mus
        self.sigmas[self.step] = sigmas
        self.step += 1

    def compute_returns_and_advantages(self, last_value, gamma, lam):
        """GAE-Lambda for on-policy data."""
        last_gae = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_not_done - self.values[t]
            last_gae = delta + gamma * lam * next_not_done * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = last_gae + self.values[t]

    def get_flat_batch(self):
        """Return all data as flat (T*N, ...) tensors in a dict.

        This is the raw on-policy batch before SAPG augmentation.
        """
        T, N = self.num_steps, self.num_envs

        def sf(x):
            """swap_and_flatten01: (T,N,...) → (T*N,...)"""
            return x.transpose(0, 1).reshape(T * N, *x.shape[2:])

        return {
            "obses": sf(self.obs),
            "actions": sf(self.actions),
            "rewards": sf(self.rewards),         # kept for augmentation
            "dones": sf(self.dones),
            "values": sf(self.values),
            "log_probs": sf(self.log_probs),
            "returns": sf(self.returns),
            "mus": sf(self.mus),
            "sigmas": sf(self.sigmas),
        }

    def get_batches_from_flat(self, flat_dict, num_mini_batches):
        """Yield minibatches from an (augmented) flat batch dict.

        Handles advantage normalization and random shuffling.
        """
        obses = flat_dict["obses"]
        total = obses.shape[0]
        batch_size = total // num_mini_batches

        returns = flat_dict["returns"]
        values = flat_dict["values"]
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = torch.randperm(total, device=self.device)
        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs": obses[idx],
                "actions": flat_dict["actions"][idx],
                "old_log_probs": flat_dict["log_probs"][idx],
                "advantages": advantages[idx],
                "returns": returns[idx],
                "old_values": values[idx],
                "old_mus": flat_dict["mus"][idx],
                "old_sigmas": flat_dict["sigmas"][idx],
                "off_policy_mask": flat_dict.get("off_policy_mask", torch.zeros(total, dtype=torch.bool, device=self.device))[idx],
            }

    def reset(self):
        self.step = 0