"""
EPO algorithm = SAPG + Genetic Algorithm on latent embeddings.

Inherits all SAPG mechanisms:
    - Block setup (num_blocks policies)
    - Leader-follower aggregation via importance sampling
    - Per-block entropy coefficient
    - coef_cond sigma, extra_params learned embeddings

Adds:
    - Per-block fitness tracking from episode returns
    - Periodic genetic algorithm on extra_params:
        * Trigger: max(f) - min(f) > gamma * median(f)  (paper Eq. 3)
        * Selection: top (K-2) non-master blocks as elites
        * Crossover: average two random elites
        * Mutation: add N(0, sigma^2) noise

Master block (index 0) is never mutated -- it's our final evaluation policy
and is already being updated via gradient descent with off-policy aggregation.
"""
from __future__ import annotations

import torch

from manipulation.algos.sapg.sapg import SAPG
from manipulation.algos.epo.ga import (
    FitnessTracker,
    should_trigger_ga,
    apply_ga,
)


class EPO(SAPG):
    """EPO = SAPG + Genetic Algorithm on extra_params embeddings.

    All PPO/SAPG logic inherited unchanged. We override nothing in the
    update loop. The trainer calls record_episode_completion() and
    maybe_apply_ga() at the right points in the outer loop.
    """

    def __init__(self, obs_dim, action_dim, num_envs, cfg, device):
        super().__init__(obs_dim, action_dim, num_envs, cfg, device)

        # -- GA hyperparameters --
        self.ga_trigger_gamma = getattr(cfg, "ga_trigger_gamma", 0.3)
        self.mutation_sigma = getattr(cfg, "mutation_sigma", 0.1)
        self.fitness_window = getattr(cfg, "fitness_window", 32)
        self.ga_min_interval = getattr(cfg, "ga_min_interval", 10)
        self.ga_warmup_iterations = getattr(cfg, "ga_warmup_iterations", 50)

        # num_elites: -1 means "use paper default K-2"
        cfg_elites = getattr(cfg, "num_elites", -1)
        if cfg_elites < 0:
            # K = num_blocks (population size including master)
            # Non-master count = num_blocks - 1
            # Paper x = K - 2 means keep all but 2
            # In our semantics: non-master elites to keep
            self.num_elites = max(self.num_blocks - 2 - 1, 2)
            # -1 extra because we exclude master from the "population" for GA
        else:
            self.num_elites = cfg_elites

        # -- Fitness tracker --
        self.fitness_tracker = FitnessTracker(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            window=self.fitness_window,
            device=device,
        )

        # -- GA scheduling state --
        self._iter_count = 0
        self._last_ga_iter = -1

        # -- GA logging state (read once per iteration by trainer) --
        self._last_ga_info = {
            "epo/ga_applied": 0,
            "epo/ga_triggered": 0,
            "epo/fitness_max": 0.0,
            "epo/fitness_min": 0.0,
            "epo/fitness_median": 0.0,
            "epo/fitness_spread": 0.0,
            "epo/num_mutated": 0,
        }

    def record_episode_completions(
        self,
        env_ids_done: torch.Tensor,
        episode_returns: torch.Tensor,
    ):
        """Called by trainer each step there are finished envs.

        env_ids_done:   (M,) indices of envs that completed this step
        episode_returns: (M,) corresponding episodic returns (real scale)
        """
        self.fitness_tracker.update_from_completions(
            env_ids_done, episode_returns, self.env_block_ids
        )

    def maybe_apply_ga(self) -> dict:
        """Called by trainer once per iteration, after the PPO update.

        Returns a dict of GA diagnostics for logging.
        """
        self._iter_count += 1

        info = {
            "epo/ga_applied": 0,
            "epo/ga_triggered": 0,
            "epo/fitness_max": 0.0,
            "epo/fitness_min": 0.0,
            "epo/fitness_median": 0.0,
            "epo/fitness_spread": 0.0,
            "epo/num_mutated": 0,
        }

        # Always compute and log fitness
        fitness = self.fitness_tracker.get_fitness()  # (num_blocks,)
        non_master = fitness[1:]
        info["epo/fitness_max"] = float(non_master.max().item())
        info["epo/fitness_min"] = float(non_master.min().item())
        info["epo/fitness_median"] = float(non_master.median().item())
        info["epo/fitness_spread"] = info["epo/fitness_max"] - info["epo/fitness_min"]
        for i in range(self.num_blocks):
            info[f"epo/fitness_block_{i}"] = float(fitness[i].item())

        # Gate: warmup
        if self._iter_count < self.ga_warmup_iterations:
            self._last_ga_info = info
            return info

        # Gate: min interval since last GA
        if self._last_ga_iter >= 0:
            if self._iter_count - self._last_ga_iter < self.ga_min_interval:
                self._last_ga_info = info
                return info

        # Gate: paper Eq. 3 trigger
        triggered = should_trigger_ga(
            fitness, self.ga_trigger_gamma, non_master_only=True
        )
        info["epo/ga_triggered"] = int(triggered)

        if not triggered:
            self._last_ga_info = info
            return info

        # Apply GA to extra_params (network has a Parameter of shape (num_blocks, embed_dim))
        extra_params = self.network.extra_params.data  # (num_blocks, embed_dim)
        _, mutated_blocks = apply_ga(
            extra_params,
            fitness,
            num_elites=self.num_elites,
            mutation_sigma=self.mutation_sigma,
        )

        # Reset fitness history for mutated blocks so next round's selection
        # uses fresh data (old fitness belonged to the replaced embedding)
        for b in mutated_blocks:
            self.fitness_tracker.reset_block(b)

        info["epo/ga_applied"] = 1
        info["epo/num_mutated"] = len(mutated_blocks)
        self._last_ga_iter = self._iter_count
        self._last_ga_info = info
        return info