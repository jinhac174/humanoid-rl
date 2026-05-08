"""
Running mean/std normalization for value targets.

Mirrors rl_games' RunningMeanStd used with normalize_value=True.
The critic learns to predict values in normalized space; we denormalize
at rollout-collection time so GAE works in real reward scale, then
re-normalize stored values/returns before the PPO update.
"""
import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    """Online running mean/std using Welford's algorithm.

    Stored as nn.Module so state persists in state_dict() for
    checkpoint save/load.
    """

    def __init__(self, shape: tuple, epsilon: float = 1e-5):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """Update running stats with a batch of values. x shape: (batch, *shape)."""
        x = x.to(torch.float64)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.running_mean
        tot_count = self.count + batch_count
        new_mean = self.running_mean + delta * batch_count / tot_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        self.running_mean = new_mean
        self.running_var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Transform x from real scale to normalized scale."""
        return (x - self.running_mean.float()) / (self.running_var.float().sqrt() + 1e-5)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Transform x from normalized scale back to real scale."""
        return x * (self.running_var.float().sqrt() + 1e-5) + self.running_mean.float()