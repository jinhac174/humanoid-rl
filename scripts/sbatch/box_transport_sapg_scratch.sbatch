#!/bin/bash
#SBATCH --job-name=bt_sapg_sc
#SBATCH --partition=suma_rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=/scratch2/danielc174/humanoid-rl/slurm/%x_%j.out
#SBATCH --error=/scratch2/danielc174/humanoid-rl/slurm/%x_%j.err

export ALGO=sapg
export MODE=scratch
source /home/danielc174/projects/humanoid-rl/scripts/sbatch/_box_transport_train.tpl.sh
