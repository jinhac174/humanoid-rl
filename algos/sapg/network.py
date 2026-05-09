"""
SAPG Actor-Critic network.

Extends the shared-trunk architecture with two SAPG mechanisms:

1. **extra_param** mode: The 1D block coefficient appended to the obs is
   replaced by a learned (num_blocks, param_size) embedding before the MLP.
   This lets the network learn rich per-block representations rather than
   relying on a raw scalar.

2. **coef_cond** sigma: Instead of a single learned log_std (action_dim,),
   SAPG maintains (num_blocks, action_dim) -- each block has its own
   exploration noise level, learned end-to-end.

Both use the appended block coefficient as a lookup key: the network reads
the last element of obs, matches it against known block IDs, and uses the
index to select the correct embedding / sigma row.
"""
import math
import torch
import torch.nn as nn
from torch.distributions import Normal


class SAPGActorCritic(nn.Module):
    """Shared-trunk actor-critic with SAPG block conditioning."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple = (1024, 1024, 512, 512),
        activation: str = "elu",
        block_ids: torch.Tensor = None,
        extra_param_size: int = 32,
    ):
        """
        Args:
            obs_dim: Raw env observation dim (e.g. 108). The network will
                     receive obs_dim + 1 (with appended block coef) but the
                     MLP input is obs_dim + extra_param_size after embedding
                     substitution.
            action_dim: Action space dimension.
            hidden_dims: MLP layer sizes.
            activation: Activation function name.
            block_ids: 1D tensor of unique per-block coefficient values,
                       shape (num_blocks,). These are the linspace(50, 0, K)
                       values from SAPG block setup.
            extra_param_size: Dimension of the learned per-block embedding.
        """
        super().__init__()

        assert block_ids is not None, "block_ids required for SAPG network"
        num_blocks = len(block_ids)

        # Store block IDs as buffer (not parameter -- they're constant lookup keys)
        self.register_buffer("block_ids", block_ids.float())
        self.pid_idx = obs_dim  # position in augmented obs where block coef sits
        self.num_blocks = num_blocks

        # Learned per-block embedding: replaces the 1D coef with a rich vector
        self.extra_params = nn.Parameter(
            torch.randn(num_blocks, extra_param_size) * 0.01
        )

        # Per-block sigma (coef_cond): each block learns its own action noise
        self.sigma = nn.Parameter(
            torch.zeros(num_blocks, action_dim)
        )

        # MLP input: obs_dim (raw env obs) + extra_param_size (learned embed)
        mlp_input = obs_dim + extra_param_size
        act_fn = nn.ELU if activation == "elu" else nn.ReLU

        trunk_layers = []
        prev = mlp_input
        for h in hidden_dims:
            trunk_layers.extend([nn.Linear(prev, h), act_fn()])
            prev = h
        self.trunk = nn.Sequential(*trunk_layers)

        self.actor_head = nn.Linear(prev, action_dim)
        self.critic_head = nn.Linear(prev, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def _lookup_block_idx(self, obs_augmented: torch.Tensor) -> torch.Tensor:
        """Extract block indices from the augmented obs.

        The block coefficient is the last element of the obs (at position
        self.pid_idx). We match it against self.block_ids to get indices.

        Returns:
            block_idx: (batch,) long tensor of block indices.
        """
        coef = obs_augmented[:, self.pid_idx]  # (batch,)
        # Exact float matching -- same as donor's argmax approach
        diffs = (coef.unsqueeze(1) - self.block_ids.unsqueeze(0)).abs()
        return diffs.argmin(dim=1)  # (batch,)

    def _preprocess_obs(self, obs_augmented: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace the appended 1D coef with the learned embedding.

        Args:
            obs_augmented: (batch, obs_dim + 1) -- raw obs + block coef

        Returns:
            obs_for_mlp: (batch, obs_dim + extra_param_size)
            block_idx: (batch,) long tensor
        """
        block_idx = self._lookup_block_idx(obs_augmented)
        raw_obs = obs_augmented[:, : self.pid_idx]  # (batch, obs_dim)
        embed = self.extra_params[block_idx]  # (batch, extra_param_size)
        return torch.cat([raw_obs, embed], dim=1), block_idx

    def forward(self, obs_augmented: torch.Tensor):
        """Full forward pass.

        Args:
            obs_augmented: (batch, obs_dim + 1)

        Returns:
            mu: (batch, action_dim)
            sigma: (batch, action_dim)
            value: (batch,)
        """
        obs_mlp, block_idx = self._preprocess_obs(obs_augmented)
        features = self.trunk(obs_mlp)
        mu = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        sigma = self.sigma[block_idx].exp()  # per-block sigma, stored as log
        return mu, sigma, value

    def get_action_and_value(self, obs_augmented, action=None):
        """Sample or evaluate action, return log_prob, entropy, value, mu, sigma.

        Used during both rollout collection and PPO update.
        """
        mu, sigma, value = self.forward(obs_augmented)
        dist = Normal(mu, sigma)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy, value, mu, sigma

    def get_value(self, obs_augmented):
        """Value-only forward (no action head). Used for bootstrapping."""
        obs_mlp, _ = self._preprocess_obs(obs_augmented)
        features = self.trunk(obs_mlp)
        return self.critic_head(features).squeeze(-1)