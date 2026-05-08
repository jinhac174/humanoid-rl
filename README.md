# Humanoid Manipulation

Reinforcement learning for G1 humanoid robot manipulation tasks using IsaacLab.

## Environment

- **Robot**: Unitree G1 with Dex3 hands (fixed-base)
- **Simulator**: IsaacLab 2.3.2 / Isaac Sim 4.5
- **Cluster**: Yonsei HPC, SLURM, Singularity SIF container
- **Algorithms**: PPO (custom, with GAE / value clipping / online obs normalization), SAPG (per-block entropy conditioning); EPO planned

## Project Structure

```
humanoid-manipulation/
├── manipulation/
│   ├── robots/
│   │   └── g1.py                  # G1_CFG, G1_FIXED_CFG, actuator physics, action scales
│   ├── tasks/
│   │   ├── can_push/              # Push can into target region
│   │   │   ├── env_cfg.py         # Scene + env config, reward/penalty weight fields
│   │   │   ├── env.py             # DirectRLEnv subclass, wires all components
│   │   │   ├── observations.py    # 108-dim obs vector
│   │   │   ├── rewards.py         # Reward + penalty functions
│   │   │   ├── terminations.py    # Success, drop, timeout
│   │   │   └── events.py          # Reset logic
│   │   └── reorient/              # Reorient object to target pose
│   │       └── (same structure as can_push)
│   ├── algos/
│   │   ├── ppo/
│   │   │   ├── buffer.py          # RolloutBuffer with GAE
│   │   │   ├── network.py         # ActorCritic MLP
│   │   │   ├── ppo.py             # PPO update + obs normalization
│   │   │   └── trainer.py         # Training loop, checkpointing, W&B logging
│   │   └── sapg/
│   │       ├── network.py         # ActorCritic with per-block conditioning
│   │       ├── sapg.py            # SAPG update (per-block entropy, bounds loss)
│   │       ├── buffer.py          # Extended rollout buffer
│   │       ├── trainer.py         # SAPG training loop
│   │       └── utils.py           # Helpers
│   └── utils/
│       └── paths.py               # Path resolution utilities
├── configs/
│   ├── train.yaml                 # Top-level train config
│   ├── eval.yaml                  # Top-level eval config
│   ├── task/
│   │   ├── can_push.yaml          # Task params, reward weights, cameras
│   │   └── reorient.yaml          # Task params, reward weights, cameras
│   └── algo/
│       ├── ppo.yaml               # PPO hyperparameters (base)
│       └── sapg.yaml              # SAPG-specific overrides (inherits ppo.yaml)
├── scripts/
│   ├── train.py                   # Universal train entry point
│   ├── eval.py                    # Universal eval entry point (edit CHECKPOINT_PATH)
│   ├── scene_load.py              # Headless scene capture (saves PNG per camera)
│   └── create_insert_assets.py    # Asset generation helper
├── tools/
│   ├── commands.sh                # Common shell invocations
│   └── cmd.txt                    # Quick-reference command list
├── docs/
│   └── sapg_plan.md               # SAPG implementation notes
└── assets/
    ├── robots/g1/usd/g1_dex3.usd
    ├── scenes/kitchen.usd
    └── objects/can.usd, target.usd
```

## Setup

```bash
# Outputs symlink (run once)
mkdir -p /scratch2/danielc174/humanoid-manipulation/outputs
ln -s /scratch2/danielc174/humanoid-manipulation/outputs ~/projects/humanoid-manipulation/outputs

# Recommended: install as editable package into the IsaacLab Python env
~/IsaacLab/isaaclab.sh -p -m pip install -e .

# Fallback: if pip install into IsaacLab Python isn't possible, set PYTHONPATH instead
# (add to SLURM script or .bashrc)
export PYTHONPATH=~/projects/humanoid-manipulation:$PYTHONPATH
```

## Run Commands

All scripts are task-agnostic. Specify task on the command line.

```bash
# Visualize scene (headless, saves PNGs to outputs/scene_load/can_push/)
~/IsaacLab/isaaclab.sh -p scripts/scene_load.py task=can_push

# Train PPO (disable wandb for debug runs)
~/IsaacLab/isaaclab.sh -p scripts/train.py task=can_push wandb.mode=disabled

# Train PPO with wandb
~/IsaacLab/isaaclab.sh -p scripts/train.py task=can_push

# Train SAPG
~/IsaacLab/isaaclab.sh -p scripts/train.py task=reorient algo=sapg

# Eval (edit CHECKPOINT_PATH at top of eval.py first)
~/IsaacLab/isaaclab.sh -p scripts/eval.py task=can_push
```

## Adding a New Task

1. Create `manipulation/tasks/<task_name>/` with `env_cfg.py`, `env.py`, `observations.py`, `rewards.py`, `terminations.py`, `events.py`, `__init__.py`
2. Add `configs/task/<task_name>.yaml` with `gym_id`, `log_name`, `env_cfg_module`, `env_cfg_class`, reward weights, cameras (no `configs/robot/` directory — robot config lives in `manipulation/robots/g1.py`)
3. Register in `manipulation/tasks/__init__.py`: `from . import <task_name>`
4. Run: `train.py task=<task_name>`

## Observation Space (108-dim)

| Slice | Content |
|---|---|
| [0:28] | joint positions |
| [28:56] | joint velocities |
| [56:84] | target_error (target_joint_pos - joint_pos) |
| [84:87] | can position (robot-relative) |
| [87:90] | can linear velocity |
| [90:93] | target position (robot-relative, fixed) |
| [93:96] | left palm position (robot-relative) |
| [96:99] | right palm position (robot-relative) |
| [99:102] | vector: can → left palm |
| [102:105] | vector: can → right palm |
| [105:108] | vector: can → target |

The table above is specific to `can_push`. The `reorient` task has its own 108-dim layout; see `manipulation/tasks/reorient/observations.py`.

## can_push Task

**Goal**: Left arm pushes can into circular target region (radius=0.5m).

**Rewards**: approach (left palm → can) + push (can → target) + success bonus

**Penalties**: drop, right arm idle deviation, joint limits, action rate, joint velocity

**Termination**: success | can dropped | timeout (15s)

**Robot spawn**: `(2.34882, -0.63841, 0.80127)`, 90° Z rotation

**Can spawn**: `(2.145, -0.2979, 0.7690)` ± XY randomization + random yaw

**Target**: fixed at `(2.6206, -0.2387, 0.741838)`

## Key Design Notes

- **Delta PD control**: actions are joint position deltas, accumulated into target, clamped to joint limits
- **Per-joint action scales**: derived from motor physics (`0.25 * effort_limit / stiffness`)
- **Robot-relative positions**: all positions subtracted from robot root to eliminate multi-env grid offset
- **Target is static visual**: no physics, position stored as fixed constant tensor `env.target_pos_w`
- **Reward weights**: defined in task yaml, pushed to `env_cfg` at runtime by `train.py`
- **GAE fix**: `next_not_done = 1 - dones[t]` (not `dones[t+1]`)
- **Value clipping**: prevents value loss explosions
- **Entropy coef**: 0.025 (prevents entropy collapse)