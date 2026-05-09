"""
Genetic algorithm operations for EPO (Evolutionary Policy Optimization).

Operates on per-block latent embeddings (extra_params in SAPG):
    - Selection: rank blocks by fitness, keep top-x as elites
    - Crossover: average two random elite embeddings
    - Mutation: add Gaussian noise

Fitness = moving average of episodic return, per block.

Paper: Algorithm 1, Equation 3 (trigger condition).

Block semantics in our port:
    - num_blocks parallel policies (the 'population' K in the paper)
    - Block 0 = master (leader), receives on-policy + off-policy (SAPG)
    - Blocks 1..K-1 = followers, on-policy only
    - Each block has a learnable latent embedding of size extra_param_size

EPO extends SAPG by letting the GA periodically modify the embeddings,
on top of the gradient updates. The master (block 0) is always preserved
(it's the one we care about at evaluation time).
"""
from __future__ import annotations

import torch


class FitnessTracker:
    """Per-block running-average fitness tracker using a cyclic buffer.

    Updates:
        - Each env completing an episode contributes its return to its block.
        - Per-block fitness = mean over the last `window` completed episodes.

    Paper uses single-episode fitness; we use a moving average for variance
    reduction (standard practice when fitness is noisy).
    """

    def __init__(self, num_blocks: int, block_size: int, window: int, device):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.window = window
        self.device = device

        # Cyclic buffer of recent episode returns per block
        # Shape: (num_blocks, window); initial NaN so we know which slots are filled
        self.buffer = torch.full(
            (num_blocks, window), float("nan"), device=device, dtype=torch.float32
        )
        # Next write index per block (cyclic)
        self.write_idx = torch.zeros(num_blocks, dtype=torch.long, device=device)

    def update_from_completions(
        self, env_ids_completed: torch.Tensor, returns_completed: torch.Tensor,
        env_block_ids: torch.Tensor,
    ):
        """Record episode returns for envs that just completed.

        env_ids_completed: (M,) indices of envs that finished this step
        returns_completed: (M,) corresponding episode returns
        env_block_ids:     (num_envs,) each env's block assignment
        """
        if env_ids_completed.numel() == 0:
            return

        block_ids = env_block_ids[env_ids_completed]  # (M,)
        # Loop per block because writes go to different cyclic positions
        for b in range(self.num_blocks):
            mask = block_ids == b
            if not mask.any():
                continue
            rets = returns_completed[mask]
            for r in rets:
                idx = self.write_idx[b].item()
                self.buffer[b, idx] = r
                self.write_idx[b] = (self.write_idx[b] + 1) % self.window

    def get_fitness(self) -> torch.Tensor:
        """Return per-block fitness: mean of non-NaN entries.

        Returns: (num_blocks,) tensor. Blocks with no completed episodes
        get 0.0 (treated as "no fitness info yet").
        """
        fitness = torch.zeros(self.num_blocks, device=self.device)
        for b in range(self.num_blocks):
            vals = self.buffer[b]
            mask = ~torch.isnan(vals)
            if mask.any():
                fitness[b] = vals[mask].mean()
        return fitness

    def reset_block(self, block_idx: int):
        """Wipe fitness history for a block (after crossover/mutation).

        Call this after mutating a block's embedding so stale fitness
        doesn't bias the next selection round.
        """
        self.buffer[block_idx] = float("nan")
        self.write_idx[block_idx] = 0


def should_trigger_ga(
    fitness: torch.Tensor,
    ga_trigger_gamma: float,
    non_master_only: bool = True,
) -> bool:
    """Check EPO paper's Equation 3:
        max(f) - min(f) > gamma * median(f)

    When non_master_only=True, computes over blocks 1..K-1 (excludes master),
    matching paper line 9: "for k = 2, ..., K do fk <- Evaluate(...)"
    """
    if non_master_only:
        f = fitness[1:]  # exclude master (block 0)
    else:
        f = fitness

    # Skip if any block has zero fitness (not yet populated) -- policies
    # haven't differentiated enough yet.
    if torch.any(f == 0.0):
        return False

    median_f = f.median()
    # Guard against median near zero (early training)
    if median_f.abs() < 1e-6:
        return False

    spread = f.max() - f.min()
    return bool((spread > ga_trigger_gamma * median_f.abs()).item())


@torch.no_grad()
def apply_ga(
    extra_params: torch.Tensor,
    fitness: torch.Tensor,
    num_elites: int,
    mutation_sigma: float,
) -> tuple[torch.Tensor, list[int]]:
    """Apply one GA step to the latent embeddings. Master (block 0) is preserved.

    extra_params: (num_blocks, embed_dim) -- latent embeddings, modified in place.
    fitness:      (num_blocks,) per-block fitness scores.
    num_elites:   number of non-master blocks to preserve unchanged.
                  Remaining non-master blocks are replaced by
                  crossover+mutation of two random elites.
    mutation_sigma: std of Gaussian noise added to crossover children.

    Returns:
        (modified extra_params, list of block indices that were mutated)

    Note on master handling: Paper Algorithm 1 loops k=2..K (iterates over
    non-masters). The master (k=1, our block 0) is preserved and trained by
    gradient descent + off-policy aggregation. We follow the same convention:
    never touch extra_params[0].
    """
    num_blocks, embed_dim = extra_params.shape
    num_non_master = num_blocks - 1  # paper's "K-1" from block 1 to K-1 (0-indexed: 1..num_blocks-1)

    if num_elites < 2:
        # Need at least 2 elites to do crossover; clamp.
        num_elites = 2
    if num_elites > num_non_master:
        num_elites = num_non_master

    # Rank non-master blocks by fitness (descending)
    non_master_fitness = fitness[1:]  # (num_non_master,)
    # argsort descending
    ranked = torch.argsort(non_master_fitness, descending=True)  # values are 0..num_non_master-1
    # Convert to global block indices (shift by +1 since block 0 is master)
    elite_global_ids = (ranked[:num_elites] + 1).tolist()
    replaced_global_ids = (ranked[num_elites:] + 1).tolist()

    # Generate crossover+mutation children for each replaced slot
    mutated_blocks = []
    device = extra_params.device
    for child_block_id in replaced_global_ids:
        # Pick two distinct random elites
        if len(elite_global_ids) >= 2:
            i, j = torch.randperm(len(elite_global_ids), device=device)[:2].tolist()
            parent_i = elite_global_ids[i]
            parent_j = elite_global_ids[j]
        else:
            # Degenerate: only one elite. Use it for both parents.
            parent_i = parent_j = elite_global_ids[0]

        # Crossover: simple average (paper Section 4.1)
        child = 0.5 * (extra_params[parent_i] + extra_params[parent_j])

        # Mutation: add Gaussian noise
        noise = torch.randn_like(child) * mutation_sigma
        child = child + noise

        extra_params[child_block_id] = child
        mutated_blocks.append(child_block_id)

    return extra_params, mutated_blocks