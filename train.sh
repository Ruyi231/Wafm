#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/lizhenhao/Project/openpi}"
cd "$PROJECT_DIR"
source env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

EXP_NAME="${EXP_NAME:-capacity_match_seed42}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-0}"
COMPUTE_NORM_STATS="${COMPUTE_NORM_STATS:-0}"
DRY_RUN="${DRY_RUN:-0}"
NORM_STATS_PATH="${NORM_STATS_PATH:-/nfs/lizhenhao/huggingface/lerobot/libero_full/norm_stats.json}"

run_cmd() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf "+"
        printf " %q" "$@"
        printf "\n"
    else
        "$@"
    fi
}

if [[ "$OVERWRITE" == "1" && "$RESUME" == "1" ]]; then
    echo "OVERWRITE=1 and RESUME=1 cannot be enabled together." >&2
    exit 2
fi

if [[ $# -gt 0 ]]; then
    CONFIGS=("$@")
else
    CONFIGS=(
        "libero_full_raw_film_m384"
        "libero_full_raw_3xfilm_m128"
    )
fi

if [[ "$COMPUTE_NORM_STATS" != "1" && ! -f "$NORM_STATS_PATH" ]]; then
    echo "Normalization statistics not found: $NORM_STATS_PATH" >&2
    echo "Set COMPUTE_NORM_STATS=1 to generate them before training." >&2
    exit 3
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "WANDB_MODE=$WANDB_MODE"
echo "EXP_NAME=$EXP_NAME"
echo "Configs: ${CONFIGS[*]}"

for config_name in "${CONFIGS[@]}"; do
    echo "Starting config: $config_name"

    if [[ "$COMPUTE_NORM_STATS" == "1" ]]; then
        run_cmd uv run scripts/compute_norm_stats.py --config-name "$config_name"
    fi

    train_args=(
        scripts/train.py
        "$config_name"
        "--exp-name=$EXP_NAME"
    )

    if [[ "$OVERWRITE" == "1" ]]; then
        train_args+=(--overwrite)
    elif [[ "$RESUME" == "1" ]]; then
        train_args+=(--resume)
    fi

    run_cmd uv run "${train_args[@]}"
done
