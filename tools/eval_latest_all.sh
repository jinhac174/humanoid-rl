#!/usr/bin/env bash
# Eval the latest checkpoint for each algo (PPO, SAPG, EPO, rsl_rl_ppo) of a
# given task. Default: 1080p, forward-walk fixed command (so videos always
# attempt motion instead of randomly sampling "stand still" segments).
#
# Usage:
#   tools/eval_latest_all.sh                                # velocity_tracking, 1080p, forward
#   tools/eval_latest_all.sh velocity_tracking 1280 720     # lower res
#   tools/eval_latest_all.sh velocity_tracking 1920 1080 random
#                                                           # random commands (no fixed)
#   tools/eval_latest_all.sh reorient 1920 1080 random      # reorient task
#
# To run in background and check back later:
#   nohup tools/eval_latest_all.sh > eval_latest.log 2>&1 &
#   tail -f eval_latest.log    # watch progress
set -euo pipefail

TASK="${1:-velocity_tracking}"
W="${2:-1920}"
H="${3:-1080}"
CMD_MODE="${4:-forward}"    # "forward" pins +x walking; "random" lets env resample

ROOT="outputs/${TASK}"

# Build the optional Hydra override for fixed_command.
extra_args=()
if [ "${CMD_MODE}" = "forward" ] && [ "${TASK}" = "velocity_tracking" ]; then
    extra_args+=('++task.eval.fixed_command=[1.0,0.0,0.0]')
fi

latest_ckpt() {
    local algo="$1"
    find "${ROOT}/${algo}" -name "model_*.pt" 2>/dev/null \
        | awk -F'model_|.pt' '{print $0, $2+0}' \
        | sort -k2 -n \
        | tail -1 \
        | awk '{print $1}'
}

echo "[eval_latest_all] task=${TASK} res=${W}x${H} cmd=${CMD_MODE}"

for algo in ppo sapg epo rsl_rl_ppo; do
    ck="$(latest_ckpt "$algo")"
    if [ -z "$ck" ]; then
        echo "[eval_latest_all] no checkpoints for ${algo}, skipping"
        continue
    fi
    echo ""
    echo "============================================================"
    echo "[eval_latest_all] ${algo}: ${ck}"
    echo "============================================================"
    ~/IsaacLab/isaaclab.sh -p scripts/eval.py \
        task="${TASK}" \
        checkpoint="${ck}" \
        video_width="${W}" video_height="${H}" \
        num_episodes=1 \
        "${extra_args[@]}"
done

echo ""
echo "[eval_latest_all] DONE. videos at:"
echo "  ${ROOT}/<algo>/<algo>_NN/eval/model_<iter>/{follow,side}.mp4"
