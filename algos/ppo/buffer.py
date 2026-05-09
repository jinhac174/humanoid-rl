import torch


class RolloutBuffer:

    def __init__(self, num_steps, num_envs, obs_dim, action_dim, device):
        self.num_steps  = num_steps
        self.num_envs   = num_envs
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.device     = device

        self.obs        = torch.zeros(num_steps, num_envs, obs_dim,    device=device)
        self.actions    = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.rewards    = torch.zeros(num_steps, num_envs,             device=device)
        self.dones      = torch.zeros(num_steps, num_envs,             device=device)
        self.values     = torch.zeros(num_steps, num_envs,             device=device)
        self.log_probs  = torch.zeros(num_steps, num_envs,             device=device)
        self.advantages = torch.zeros(num_steps, num_envs,             device=device)
        self.returns    = torch.zeros(num_steps, num_envs,             device=device)
        self.step = 0

    def insert(self, obs, actions, rewards, dones, values, log_probs):
        self.obs[self.step]       = obs
        self.actions[self.step]   = actions
        self.rewards[self.step]   = rewards
        self.dones[self.step]     = dones
        self.values[self.step]    = values
        self.log_probs[self.step] = log_probs
        self.step += 1

    def compute_returns_and_advantages(self, last_value, gamma, lam):
        """
        GAE-Lambda advantage estimation.
        FIX: next_not_done uses dones[t] not dones[t+1].
        dones[t] marks that the episode ended at step t --
        the bootstrap must be cut at t, not t+1.
        """
        last_gae = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            next_not_done = 1.0 - self.dones[t]
            delta         = self.rewards[t] + gamma * next_value * next_not_done - self.values[t]
            last_gae      = delta + gamma * lam * next_not_done * last_gae

            self.advantages[t] = last_gae
            self.returns[t]    = last_gae + self.values[t]

    def get_batches(self, num_mini_batches):
        total      = self.num_steps * self.num_envs
        indices    = torch.randperm(total, device=self.device)
        batch_size = total // num_mini_batches

        obs_flat        = self.obs.view(total, self.obs_dim)
        actions_flat    = self.actions.view(total, self.action_dim)
        log_probs_flat  = self.log_probs.view(total)
        values_flat     = self.values.view(total)
        returns_flat    = self.returns.view(total)
        advantages_flat = self.advantages.view(total)
        advantages_flat = (advantages_flat - advantages_flat.mean()) / \
                          (advantages_flat.std() + 1e-8)

        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            yield (
                obs_flat[idx],
                actions_flat[idx],
                log_probs_flat[idx],
                advantages_flat[idx],
                returns_flat[idx],
                values_flat[idx],
            )

    def reset(self):
        self.step = 0