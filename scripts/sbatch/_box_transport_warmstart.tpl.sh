#!/bin/bash
# Template body for box_transport warm-started sbatch runs.
# Sourced by per-algo wrappers via `ALGO=<name> source _box_transport_warmstart.tpl.sh`.
#
# Required env var:  ALGO   ∈ {ppo, sapg, epo}
# Optional env var:  RUN_TAG (default: "warmstart")  — appended to the run-dir
#                    name to distinguish from from-scratch runs.

set -euo pipefail

: "${ALGO:?must be set: ppo|sapg|epo}"
: "${RUN_TAG:=warmstart}"

IMAGE=/home/danielc174/IsaacSim/isaacsim.sif
ISAACLAB=/home/danielc174/IsaacLab
PROJECT=/home/danielc174/projects/humanoid-rl
CACHE=/home/danielc174/isaac_cache
SCRATCH=/scratch2/danielc174

CKPT="/scratch2/danielc174/humanoid-rl/outputs/velocity_tracking/${ALGO}/${ALGO}_00/checkpoints/model_1500.pt"

mkdir -p "$CACHE"/{kit,ov,pip,glcache,computecache,logs,data,documents}

echo "[sbatch] node=${SLURMD_NODENAME:-?}  job=${SLURM_JOB_ID:-?}"
echo "[sbatch] algo=$ALGO  run_tag=$RUN_TAG"
echo "[sbatch] warm-start ckpt: $CKPT"
[ -f "$CKPT" ] || { echo "[sbatch] CKPT NOT FOUND"; exit 1; }

singularity exec --cleanenv --writable-tmpfs --nv \
  --bind "$ISAACLAB":/IsaacLab \
  --bind "$PROJECT":"$PROJECT" \
  --bind "$SCRATCH":"$SCRATCH" \
  --bind "$CACHE"/kit:/isaac-sim/kit/cache \
  --bind "$CACHE"/ov:/root/.cache/ov \
  --bind "$CACHE"/pip:/root/.cache/pip \
  --bind "$CACHE"/glcache:/root/.cache/nvidia/GLCache \
  --bind "$CACHE"/computecache:/root/.nv/ComputeCache \
  --bind "$CACHE"/logs:/root/.nvidia-omniverse/logs \
  --bind "$CACHE"/data:/root/.local/share/ov/data \
  --bind "$CACHE"/documents:/root/Documents \
  --pwd "$PROJECT" \
  --env ACCEPT_EULA=Y \
  --env PRIVACY_CONSENT=Y \
  --env HEADLESS=1 \
  --env GIT_PYTHON_REFRESH=quiet \
  "$IMAGE" \
  /IsaacLab/isaaclab.sh -p scripts/train.py \
    task=box_transport \
    "algo=$ALGO" \
    wandb.mode=online \
    "checkpoint=$CKPT" \
    "+run_tag=$RUN_TAG"
