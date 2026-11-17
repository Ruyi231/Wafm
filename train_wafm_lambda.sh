#!/usr/bin/env bash
# Single GPU, one seed: lambda=0.03 followed by lambda=0.3. The existing 0.1 run is reused.
# GPU_ID=0 bash train_wafm_lambda.sh
# DRY_RUN=1 bash train_wafm_lambda.sh
# GPU_ID=0 bash train_wafm_lambda.sh 0.3  # run only the second configuration
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -gt 1 ]]; then
    echo "Usage: bash train_wafm_lambda.sh [all|0.03|0.3]" >&2
    exit 2
fi
case "${1:-all}" in
    all) suffixes=(0p03 0p3) ;;
    0.03) suffixes=(0p03) ;;
    0.3) suffixes=(0p3) ;;
    *) echo "Unknown lambda: $1. Choose all, 0.03, or 0.3." >&2; exit 2 ;;
esac

seed="${SEED:-42}"
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
gpu="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
if [[ "$gpu" == *,* || "$gpu" == *[[:space:]]* ]] || \
   [[ ! "$gpu" =~ ^[0-9]+$ && "$gpu" != GPU-* && "$gpu" != MIG-* ]]; then
    echo "Select exactly one GPU with GPU_ID or CUDA_VISIBLE_DEVICES: $gpu" >&2
    exit 2
fi

dry_run="${DRY_RUN:-0}"
if [[ "$dry_run" != 1 ]]; then source "${TRAIN_ENV_FILE:-./env.sh}"; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
printf 'Serial WAFM lambda sweep: GPU=%s; seed=%s\n' "$gpu" "$seed"

# Check the entire queue before starting the first expensive training run.
if [[ "$dry_run" != 1 ]]; then
    for suffix in "${suffixes[@]}"; do
        config="libero_full_wafm_l2_nogate_loss_lambda_$suffix"
        exp_name="${EXP_PREFIX:-$config}_seed${seed}"
        log_file="${LOG_DIR:-logs/wafm_lambda}/$config/$exp_name.log"
        if [[ -e "$log_file" || -e "checkpoints/$config/$exp_name" ]]; then
            echo "Run or log already exists: $exp_name. Choose a new EXP_PREFIX." >&2
            exit 3
        fi
    done
fi

for suffix in "${suffixes[@]}"; do
    config="libero_full_wafm_l2_nogate_loss_lambda_$suffix"
    exp_name="${EXP_PREFIX:-$config}_seed${seed}"
    cmd=(uv run scripts/train.py "$config" "--exp-name=$exp_name" "--seed=$seed")
    printf 'Command:'; printf ' %q' "${cmd[@]}"; printf '\n'
    if [[ "$dry_run" == 1 ]]; then continue; fi
    log_dir="${LOG_DIR:-logs/wafm_lambda}/$config"
    mkdir -p "$log_dir"
    snapshot="$log_dir/$exp_name.config.txt"
    uv run python -c 'import dataclasses,sys; from openpi.training.config import get_config; c=dataclasses.replace(get_config(sys.argv[1]),seed=int(sys.argv[2]),exp_name=sys.argv[3]); print(dataclasses.asdict(c))' \
        "$config" "$seed" "$exp_name" > "$snapshot"
    printf '\ntraining_seed=%s\nexp_name=%s\nGPU=%s\n' "$seed" "$exp_name" "$gpu" >> "$snapshot"
    git rev-parse HEAD >> "$snapshot"
    git diff -- src/openpi/models src/openpi/training >> "$snapshot"
    # Foreground pipeline + pipefail: finish this run before starting the next;
    # any training or logging failure stops the queue.
    "${cmd[@]}" 2>&1 | tee "$log_dir/$exp_name.log"
    printf 'Completed: %s (seed %s)\n' "$config" "$seed"
done
