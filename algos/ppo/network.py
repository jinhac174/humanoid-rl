import math
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Supports two modes controlled by `shared`:

        shared=True  (donor default):
            Single trunk [obs → hidden_dims → features], then two 1-layer
            heads (actor_head → action_dim, critic_head → 1). The trunk runs
            once per forward pass, so actor and critic share all hidden layers.
            This matches rl_games `separate: False`.

        shared=False:
            Separate actor and critic MLPs with independently configured
            hidden_dims. No shared parameters.

    Other donor-matching flags:
        use_tanh=False  (donor: mu_activation=None -- raw output, env clamps)
        init_noise_std=1.0  (donor: sigma_init val=0 → log_std=0 → std=1)
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        shared=True,
        hidden_dims=(1024, 1024, 512, 512),
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=1.0,
        use_tanh=False,
    ):
        super().__init__()
        self.shared = shared
        self.use_tanh = use_tanh

        act_fn = nn.ELU if activation == "elu" else nn.ReLU

        if shared:
            trunk_layers = []
            prev = obs_dim
            for h in hidden_dims:
                trunk_layers.extend([nn.Linear(prev, h), act_fn()])
                prev = h
            self.trunk = nn.Sequential(*trunk_layers)
            self.actor_head = nn.Linear(prev, action_dim)
            self.critic_head = nn.Linear(prev, 1)
        else:
            actor_layers = []
            prev = obs_dim
            for h in actor_hidden_dims:
                actor_layers.extend([nn.Linear(prev, h), act_fn()])
                prev = h
            actor_layers.append(nn.Linear(prev, action_dim))
            self.actor = nn.Sequential(*actor_layers)

            critic_layers = []
            prev = obs_dim
            for h in critic_hidden_dims:
                critic_layers.extend([nn.Linear(prev, h), act_fn()])
                prev = h
            critic_layers.append(nn.Linear(prev, 1))
            self.critic = nn.Sequential(*critic_layers)

        # Learned log-std -- observation-independent, matches donor's fixed_sigma
        self.log_std = nn.Parameter(
            torch.full((action_dim,), math.log(init_noise_std))
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Small gain on actor output → policy starts near zero actions
        if self.shared:
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
            nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        else:
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
            nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def get_value(self, obs):
        if self.shared:
            return self.critic_head(self.trunk(obs)).squeeze(-1)
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        if self.shared:
            features = self.trunk(obs)
            mean = self.actor_head(features)
            value = self.critic_head(features).squeeze(-1)
        else:
            mean = self.actor(obs)
            value = self.critic(obs).squeeze(-1)

        if self.use_tanh:
            mean = torch.tanh(mean)

        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy, value, mean