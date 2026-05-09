# Humanoid RL — G1

Reinforcement learning on the Unitree G1 humanoid using IsaacLab. Manipulation
tasks (fixed-base) live under `manipulation/`; locomotion will land under
`locomotion/`. Algorithms (`algos/`), evaluator base (`evaluators/`),
robot configs (`robots/`), and shared utilities (`utils/`) sit at the top
level so both task domains share them.

## Environment

- **Robot**: Unitree G1 with Dex3 hands. Manipulation runs fixed-base with
  legs+waist pinned via high-stiffness implicit actuators (only the upper
  body is in the action space, 28 DoF). Locomotion will switch to
  free-floating root with the legs+waist as real actuators.
- **Simulator**: IsaacLab 2.3.2 / Isaac Sim 4.5
- **Cluster**: Yonsei HPC, SLURM, Singularity SIF container
- **Algorithms**: PPO (custom, with GAE / value clipping / online obs
  normalization), SAPG (per-block conditioning + leader-follower batch
  augmentation), EPO (SAPG + genetic algorithm on latent embeddings)

## Project Structure

```
humanoid-rl/
├── algos/                              # Shared trainers + agents
│   ├── ppo/
│   │   ├── buffer.py                   # RolloutBuffer with GAE
│   │   ├── network.py                  # ActorCritic MLP
│   │   ├── ppo.py                      # PPO update + obs normalization
│   │   └── trainer.py                  # Training loop, checkpointing, W&B
│   ├── sapg/                           # SAPG = PPO + per-block conditioning
│   └── epo/                            # EPO = SAPG + genetic algorithm
├── evaluators/                         # Shared eval framework
│   ├── base.py                         # BaseEvaluator (default eval loop)
│   └── utils.py                        # Frame capture, ckpt → policy reconstruction
├── robots/
│   └── g1.py                           # G1_FIXED_CFG (manipulation), G1_FREE_CFG (locomotion: TODO)
├── utils/
│   ├── paths.py                        # PROJECT_ROOT, ASSET_ROOT
│   ├── logging.py                      # explained_variance, iter_loggable_items
│   └── normalization.py                # RunningMeanStd for value targets
├── manipulation/
│   └── tasks/
│       ├── __init__.py                 # Registers each task's gym id
│       └── reorient/
│           ├── env_cfg.py              # Scene + env config, reward weights
│           ├── env.py                  # DirectRLEnv subclass, wires components
│           ├── observations.py         # 96 + 3*K policy obs (default K=8 → 120-d)
│           ├── rewards.py              # Donor SAPG five-term reward
│           ├── terminations.py         # drop / max-consec / timeout
│           ├── events.py               # Reset logic
│           └── evaluate.py             # ReorientEvaluator (success counting + respawn cooldown)
├── configs/
│   ├── train.yaml                      # Top-level train config
│   ├── eval.yaml                       # Generic eval params (video size, fps, raytraced)
│   ├── task/reorient.yaml              # Reorient hyperparams + cameras + eval-time overrides
│   └── algo/{ppo,sapg,epo}.yaml        # Per-algo hyperparams
├── scripts/
│   ├── train.py                        # Hydra entry: builds env_cfg + dispatches trainer
│   ├── eval.py                         # Hydra entry: builds env + dispatches evaluator
│   └── scene_load.py                   # Headless screenshot per camera
├── tools/
│   ├── cmd.txt                         # Quick-reference command list
│   ├── commands.sh                     # Common shell snippets
│   └── pycache_remove.sh               # Clear __pycache__ recursively
├── pyproject.toml                      # Editable package install (algos, evaluators, robots, utils, manipulation, locomotion)
└── assets/
    ├── robots/g1/                      # G1 USD + meshes + URDF
    └── objects/cube_multicolor*.usd    # Reorient task cube (physics + visual goal)
```

## Setup

```bash
# First-time scratch setup (creates the outputs path the symlink points at)
mkdir -p /scratch2/danielc174/humanoid-rl/outputs

# (Optional) migrate prior runs from the old folder name. Pick ONE:
#   mv  /scratch2/danielc174/humanoid-manipulation /scratch2/danielc174/humanoid-rl
#   ln -s /scratch2/danielc174/humanoid-manipulation/outputs/* \
#         /scratch2/danielc174/humanoid-rl/outputs/

# Install as editable package into the IsaacLab Python env (preferred)
~/IsaacLab/isaaclab.sh -p -m pip install -e .

# Fallback: if pip install into IsaacLab Python isn't possible, set PYTHONPATH instead
export PYTHONPATH=~/projects/humanoid-rl:$PYTHONPATH
```

## Run Commands

All scripts are task- and algo-agnostic; specify both via Hydra overrides.

```bash
# Visualize scene (saves PNGs to outputs/scene_load/<task>/)
~/IsaacLab/isaaclab.sh -p scripts/scene_load.py task=reorient

# Train PPO (debug: wandb disabled by default)
~/IsaacLab/isaaclab.sh -p scripts/train.py task=reorient

# Train PPO with wandb online
~/IsaacLab/isaaclab.sh -p scripts/train.py task=reorient wandb.mode=online

# Train SAPG
~/IsaacLab/isaaclab.sh -p scripts/train.py task=reorient algo=sapg wandb.mode=online

# Train EPO with 8 blocks
~/IsaacLab/isaaclab.sh -p scripts/train.py task=reorient algo=epo algo.num_blocks=8 wandb.mode=online

# Eval (records mp4 per camera under <run_dir>/eval/<ckpt_stem>/)
~/IsaacLab/isaaclab.sh -p scripts/eval.py task=reorient \
    checkpoint=outputs/reorient/sapg/run_00/checkpoints/model_21000.pt
```

## Adding a New Task

1. Create `<domain>/tasks/<task_name>/` with `env_cfg.py`, `env.py`,
   `observations.py`, `rewards.py`, `terminations.py`, `events.py`,
   `__init__.py` (registers the gym id).
2. (Optional) add `evaluate.py` with a `<Task>Evaluator(BaseEvaluator)`
   subclass for task-specific eval logic. Tasks without it fall back to
   `BaseEvaluator`.
3. Add `configs/task/<task_name>.yaml` with:
   - `gym_id`, `log_name`
   - `env_cfg_module`, `env_cfg_class`
   - (optional) `evaluator_module`, `evaluator_class`
   - `cameras:` map for video rendering
   - `viewer:` defaults
   - (optional) `eval:` block of fields the evaluator pushes onto env_cfg
4. Register in `<domain>/tasks/__init__.py`: `from . import <task_name>`.
5. Run: `train.py task=<task_name>`

## Adding Locomotion Later

When locomotion lands, it slots in alongside manipulation:

1. Create `locomotion/tasks/<task>/` mirroring manipulation's structure.
2. In `robots/g1.py` (section 3 placeholder), add `G1_FREE_CFG` with
   `fix_root_link=False` and the active-leg actuator group, plus
   `LOCOMOTION_ACTUATED_JOINTS` / `LOCOMOTION_ACTION_SCALE`.
3. Locomotion tasks consume those new symbols; manipulation keeps using
   `G1_FIXED_CFG` and the upper-body `ACTUATED_JOINTS`.
4. The same algorithms in `algos/` and the same evaluator base in
   `evaluators/` work for both — no fork.

## Reorient Task

**Goal**: bimanually reorient a 5 cm cube (mass 0.1 kg, multi-color faces)
on a 1×1 m table to a randomly sampled target pose.

**Observation (120-d, default K=8 corners)**: joint pos+vel of the 28
actuated joints, robot-relative object & goal pose, palm + fingertip
positions, K = 8 cube corners' delta vectors to the goal, lifted flag,
near-goal progress.

**Reward (donor AllegroKuka five-term)**: fingertip→object distance
shaping, lifting shaping + one-shot lift bonus, keypoint→goal shaping
(gated by lifted), success bonus on each near-goal hit.

**Termination**: drop (object below ``object_drop_z``), max consecutive
successes (50), timeout (10 s).

**Eval-time overrides** (in `configs/task/reorient.yaml::eval`):
tighter ``success_tolerance`` (0.04), longer ``success_steps`` (40 frames
hold), 20 s episodes, ``max_consecutive_successes`` raised so the policy
keeps reorienting through the whole video, ``hold_frames`` cooldown
between accepted goal respawns to keep the rendered cube from strobing.

## Key Design Notes

- **Delta PD control**: actions are joint position deltas, accumulated into
  target, clamped to soft joint limits.
- **Per-joint action scales**: derived from motor physics
  (`0.25 * effort_limit / stiffness`).
- **8-corner keypoints by default**: symmetric over SO(3), trains and
  evaluates with the same observation shape. Toggle via `NUM_KEYPOINTS` in
  `manipulation/tasks/reorient/env_cfg.py`.
- **Robot-relative positions**: positions in the obs are subtracted from
  the robot root so multi-env grid offsets vanish.
- **GAE**: `next_not_done = 1 - dones[t]` (not `dones[t+1]`).
- **Value normalization**: rl_games convention — critic outputs in
  normalized space, denormalized for buffer storage / GAE, renormalized
  for the loss.
- **Value bootstrap**: at truncation, `gamma * V(s_t)` is added to the
  reward to keep GAE sane (rl_games approximation).
- **Reward weights**: stored as fields on `env_cfg`, pushed from task yaml
  by `scripts/train.py` (via `setattr` for any field name that matches).
