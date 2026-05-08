~/IsaacLab/isaaclab.sh -p scripts/train.py agent=ppo headless=true wandb.mode=online seed=

~/IsaacLab/isaaclab.sh -p scripts/eval.py headless=true

rm -rf /scratch2/danielc174/humanoid-manipulation/outputs
mkdir -p /scratch2/danielc174/humanoid-manipulation/outputs