#!/usr/bin/env bash
# Common shell snippets. Source or copy lines as needed.

# Train (set seed= and adjust num_envs / algo / wandb.mode as needed)
~/IsaacLab/isaaclab.sh -p scripts/train.py task=reorient algo=ppo headless=true wandb.mode=online seed=42

# Eval (edit checkpoint path)
~/IsaacLab/isaaclab.sh -p scripts/eval.py task=reorient \
  checkpoint=outputs/reorient/ppo/run_00/checkpoints/model_70000.pt

# Wipe outputs (DESTRUCTIVE)
rm -rf /scratch2/danielc174/humanoid-rl/outputs
mkdir -p /scratch2/danielc174/humanoid-rl/outputs
