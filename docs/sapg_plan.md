# SAPG Port Plan — Donor Dissection & Implementation Map

## 1. What `--sapg` Turns On (experiment_args.py)

```python
args.expl_type = 'mixed_expl_learn_param'
args.use_others_experience = 'lf'       # leader-follower filtering
args.sigma = 'coef_cond'                # per-block learned sigma
```

Plus these mandatory hyperparameters:
- `num_expl_coef_blocks` → number of SAPG blocks (e.g. 6)
- `expl_coef_block_size = num_envs // num_expl_coef_blocks`
- `expl_reward_type = 'entropy'` → per-block entropy coefficient
- `expl_reward_coef_scale` → scales the entropy coefficient range
- `off_policy_ratio = 1.0` → how many extra blocks to sample

## 2. The Four SAPG Mechanisms

### Mechanism A — Block Setup (a2c_common.py __init__)

Environments are partitioned into `num_blocks` groups of `block_size` each.

```
num_blocks = num_envs // block_size
env_ids[i] = i // block_size          # which block env i belongs to

# Per-block scalar coefficient (used as block identifier in obs)
embedding_genvec = linspace(50.0, 0.0, num_blocks)  # block 0 = 50, last = 0
intr_reward_coef_embd = embedding_genvec[env_ids].reshape(-1, 1)  # (num_envs, 1)

# Per-block entropy coefficient (for loss, not reward buffer)
intr_reward_coef = linspace(0.5, 0.0, num_blocks)[env_ids] * scale
# Block 0 (leader) has HIGHEST entropy → most exploration
# Last block has ZERO entropy → pure exploitation
```

Block 0 is the **leader** (evaluation policy). All other blocks are **followers**.
The leader gets all its own data + filtered data from followers.

### Mechanism B — Obs Augmentation (a2c_common.py env_step, env_reset)

Every observation gets the 1D block coefficient appended:

```python
# In env_step() and env_reset():
obs = env.step(actions)                          # shape (N, 108)
obs = cat([obs, intr_reward_coef_embd], dim=1)   # shape (N, 109)
```

The network sees 109-dim input. But in `extra_param` mode, the raw 1D coef
is REPLACED by a learned 32-dim embedding (see Mechanism D), so the MLP
actually receives `108 + 32 = 140` dims.

### Mechanism C — Batch Augmentation (a2c_common.py augment_batch_for_mixed_expl)

This is the heart of SAPG. After collecting a rollout:

1. **Sample repeat indices**: `[0] + random_sample(1..num_blocks-1, off_policy_ratio)`
   With `off_policy_ratio=1`, this is `[0, k]` — the leader plus one random follower.

2. **Duplicate the rollout** for each repeat index `k`:
   - Take the full rollout data
   - ROLL the coefficient embedding by `block_size * k` positions
   - This means each env's data now has a DIFFERENT block's coefficient appended
   - Mark rolled data as `off_policy_mask = True`

3. **Recompute values** for rolled copies using current value network with new obs
   (because the value function is conditioned on the block coefficient)

4. **Compute 1-step TD returns** for rolled copies:
   `R_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1})`
   NOT GAE — just TD(0). Only the original on-policy data uses GAE.

5. **Leader-follower filtering** (`filter_leader`):
   - For repeat_idx=0 (leader): keep ALL data (full batch)
   - For repeat_idx=k (follower): keep ONLY block k-1's slice
   This means the leader trains on everything, followers contribute one block each.

6. **Shuffle** the augmented batch (preserving sequence structure for RNN).

The augmented batch is then used for normal PPO update, except with per-block entropy.

### Mechanism D — Network Conditioning (network_builder.py A2CBuilder.Network)

Two changes to the network:

**extra_param mode** (replaces raw coefficient with learned embedding):
```python
# __init__:
self.param_ids = unique_block_coefficients          # (num_blocks,)
self.extra_params = nn.Parameter(randn(num_blocks, 32))  # learned per-block embedding
self.pid_idx = 108  # where the coef starts in obs

# forward:
idxs = argmax(obs[:, pid_idx] == param_ids)         # find which block
obs = cat([obs[:, :pid_idx], extra_params[idxs]])    # (N, 108+32=140)
# Then feed through normal MLP
```

**coef_cond sigma** (per-block learned action noise):
```python
# __init__:
self.sigma = nn.Parameter(zeros(num_blocks, action_dim))  # NOT (action_dim,)

# forward:
idxs = argmax(obs[:, sigma_id_idx] == sigma_ids)
sigma = sigma_act(self.sigma[idxs])                  # (N, action_dim)
```

Each block learns its OWN exploration noise. Combined with per-block entropy,
this creates a population of policies with diverse exploration strategies
sharing a single set of MLP weights.

### Mechanism E — Per-Block Entropy in Loss (a2c_continuous.py calc_gradients)

```python
# Instead of a single entropy_coef:
ec_candidates = intr_reward_coef[::block_size]           # one per block
ec_identifiers = intr_reward_coef_embd[::block_size, 0]  # block IDs
ec_indices = argmax(obs_batch[:, -1] == ec_identifiers)   # which block each sample belongs to
entropy_coef = ec_candidates[ec_indices]                  # (batch,) per-sample entropy coef

# Loss uses per-sample entropy_coef instead of scalar:
loss = a_loss + critic_coef * c_loss - (entropy_coef * entropy).mean() + bounds_loss
```

## 3. What Changes vs Our PPO

| Component | PPO (current) | SAPG (needed) | Action |
|---|---|---|---|
| Config | `configs/algo/ppo.yaml` | Need SAPG-specific fields | **New file** `sapg.yaml` |
| Network | Shared trunk, single sigma | extra_param lookup + per-block sigma | **New file** |
| Buffer | obs, actions, rewards, dones, values, log_probs | + mus, sigmas (for KL/IS), off_policy_mask | **New file** |
| Obs handling | Pass through as-is | Append block coef embedding before network | **New logic** |
| Rollout | Simple collect → GAE | Collect → GAE (on-policy) + batch augment → TD(0) (off-policy) | **New logic** |
| Update | Single entropy_coef | Per-block entropy_coef from obs | **Modified loss** |
| Trainer | PPOTrainer | Different loop: augment batch between rollout and update | **New file** |

### What can be reused from PPO:
- Welford obs normalization (identical)
- GAE computation (identical for on-policy portion)
- Clipped value loss (identical)
- Clipped surrogate loss (identical)
- Gradient clipping (identical)
- Checkpoint save/load structure (extend for SAPG state)

### What must be new:
- Block coefficient setup + obs augmentation
- Batch augmentation (augment_batch_for_mixed_expl)
- filter_leader utility
- extra_param network mode
- coef_cond sigma
- Per-block entropy in loss
- Value recomputation for off-policy data
- TD(0) returns for off-policy data

## 4. File Map

```
configs/algo/sapg.yaml                     # SAPG config (fork of ppo.yaml + SAPG fields)
manipulation/algos/sapg/__init__.py        # Register SAPGTrainer
manipulation/algos/sapg/utils.py           # filter_leader, swap_and_flatten01
manipulation/algos/sapg/buffer.py          # SAPGRolloutBuffer (extends RolloutBuffer)
manipulation/algos/sapg/network.py         # SAPGActorCritic (extra_param + coef_cond sigma)
manipulation/algos/sapg/sapg.py            # SAPG class (block setup, obs augment, batch augment, update)
manipulation/algos/sapg/trainer.py         # SAPGTrainer (modified training loop)
manipulation/algos/__init__.py             # Add 'sapg' to TRAINER_REGISTRY
```

## 5. Implementation Order

1. **utils.py** — Pure functions, no dependencies, testable standalone
2. **buffer.py** — Extends RolloutBuffer with mus/sigmas/off_policy_mask
3. **network.py** — SAPGActorCritic with extra_param + coef_cond sigma
4. **sapg.py** — Core algorithm class (depends on buffer, network, utils)
5. **sapg.yaml** — Config (all PPO fields + SAPG additions)
6. **trainer.py** — Training loop (depends on sapg.py)
7. **__init__.py** — Registration

## 6. Key Implementation Details

### Block coefficient values
```python
num_blocks = num_envs // block_size
# Donor uses linspace(50.0, 0.0, num_blocks) as block identifiers
# These are NOT meaningful numbers — they're just unique IDs that the
# network learns to embed via extra_params. The specific values 50→0
# are arbitrary; what matters is they're unique and float-comparable.
block_ids = torch.linspace(50.0, 0.0, num_blocks)  # (num_blocks,)
```

### Entropy coefficient schedule
```python
# Per-block entropy: block 0 (leader) has most entropy, last has none
entropy_per_block = torch.linspace(0.5, 0.0, num_blocks) * entropy_coef_scale
```

### Obs flow in rollout
```
env produces: obs (N, 108)
SAPG appends: obs = cat([obs, block_coef_embd], dim=1)  → (N, 109)
Normalize:    obs_normalized = welford_normalize(obs)     → (N, 109)
Network sees: obs[:, :108] + extra_params[block_lookup]   → (N, 140)
```

### Batch augmentation flow
```
1. Collect rollout: T steps × N envs → batch_dict with obs (T*N, 109)
2. Sample repeat_idxs = [0, random_follower_k]
3. For each k in repeat_idxs:
   a. Copy full batch
   b. Roll the coef_embd column by block_size * k
   c. If k > 0: recompute values with rolled obs, compute TD(0) returns
4. Concatenate all copies
5. Apply filter_leader: leader keeps all, each follower keeps its block only
6. Shuffle (preserving seq structure if RNN, but we don't use RNN)
7. Feed to PPO update with per-block entropy coefficients
```

### TD(0) returns for off-policy data
```python
# NOT GAE — just 1-step bootstrap:
returns_offpol = rewards + gamma * V(next_obs_with_new_coef) * (1 - done)
```
This is critical: off-policy data uses a SIMPLER return estimator because
GAE with stale off-policy data would introduce too much bias.

## 7. Donor Hyperparameters for Two-Arms Reorientation

From the SAPG paper and repo defaults:
- `num_blocks`: 6 (paper uses 6 for allegro_kuka tasks)
- `entropy_coef_scale`: task-dependent (try 1.0 initially)
- `off_policy_ratio`: 1.0
- `extra_param_size`: 32
- All PPO params: same as AllegroKukaPPO.yaml (already locked in our ppo.yaml)
- LSTM: the paper uses LSTM for allegro_kuka. Our port does NOT use LSTM (Phase D consideration).

## 8. What We Explicitly Defer

- LSTM support (Phase D/E — our obs is already 108-dim, no temporal structure yet)
- RND intrinsic reward (donor supports it but uses entropy for two-arms)
- Multi-GPU / distributed training
- PBT (Population Based Training)
- `mixed_expl_disjoint` mode (only `mixed_expl_learn_param` is SAPG)