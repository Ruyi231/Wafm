#!/usr/bin/env bash
# Default: one seed (42), Raw-3xM128+loss then Raw-M384+loss, on ONE GPU.
# Examples: GPU_ID=0 bash train_shared_loss.sh
#           DRY_RUN=1 bash train_shared_loss.sh
#           GPU_ID=0 SEED=0 bash train_shared_loss.sh raw_loss
set -euo pipefail
cd "$(dirname "$0")"

config_for_variant() {
    case "$1" in
        raw_3x_loss) echo libero_full_raw_3xfilm_m128_wavloss ;;
        raw_3x_noloss) echo libero_full_raw_3xfilm_m128 ;;
        raw_wide_loss) echo libero_full_raw_film_m384_wavloss ;;
        raw_wide_noloss) echo libero_full_raw_film_m384 ;;
        wfm_loss) echo libero_full_wfm_l2_shared_loss ;;
        wfm_noloss) echo libero_full_wfm_l2_shared_noloss ;;
        *) echo "Unknown variant: $1" >&2; return 2 ;;
    esac
}

selection="${1:-raw_loss}"
if [[ $# -gt 0 ]]; then shift; fi
case "$selection" in
    raw_loss) variants=(raw_3x_loss raw_wide_loss) ;;
    *) config_for_variant "$selection" > /dev/null; variants=("$selection") ;;
esac
if [[ $# -eq 0 ]]; then set -- "${SEED:-42}"; fi
# Explicit positional seeds remain supported, but never run implicitly.
for seed in "$@"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
done

gpu="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
if [[ "$gpu" == *,* || "$gpu" == *[[:space:]]* ]] || \
   [[ ! "$gpu" =~ ^[0-9]+$ && "$gpu" != GPU-* && "$gpu" != MIG-* ]]; then
    echo "Select exactly one GPU with GPU_ID or CUDA_VISIBLE_DEVICES: $gpu" >&2
    exit 2
fi
dry_run="${DRY_RUN:-0}"
if [[ "$dry_run" != 1 ]]; then source "${TRAIN_ENV_FILE:-./env.sh}"; fi
# Enforce the selected single GPU even if the environment script changes it.
export CUDA_VISIBLE_DEVICES="$gpu"
printf 'Serial training: GPU=%s; seeds=%s\n' "$gpu" "$*"

# Check every planned run before spending GPU time on the first one.
if [[ "$dry_run" != 1 ]]; then
    for variant in "${variants[@]}"; do
        config="$(config_for_variant "$variant")"
        for seed in "$@"; do
            exp_name="${EXP_PREFIX:-$config}_seed${seed}"
            log_file="${LOG_DIR:-logs/shared_loss}/$config/$exp_name.log"
            if [[ -e "$log_file" || -e "checkpoints/$config/$exp_name" ]]; then
                echo "Run or log already exists: $exp_name. Choose a new EXP_PREFIX." >&2
                exit 3
            fi
        done
    done
fi

for variant in "${variants[@]}"; do
    config="$(config_for_variant "$variant")"
    for seed in "$@"; do
        exp_name="${EXP_PREFIX:-$config}_seed${seed}"
        cmd=(uv run scripts/train.py "$config" "--exp-name=$exp_name" "--seed=$seed")
        printf 'Command:'; printf ' %q' "${cmd[@]}"; printf '\n'
        if [[ "$dry_run" == 1 ]]; then continue; fi
        log_dir="${LOG_DIR:-logs/shared_loss}/$config"
        log_file="$log_dir/$exp_name.log"
        mkdir -p "$log_dir"
        snapshot="$log_dir/$exp_name.config.txt"
        uv run python -c 'import dataclasses,sys; from openpi.training.config import get_config; c=dataclasses.replace(get_config(sys.argv[1]),seed=int(sys.argv[2]),exp_name=sys.argv[3]); print(dataclasses.asdict(c))' \
            "$config" "$seed" "$exp_name" > "$snapshot"
        printf '\ntraining_seed=%s\nexp_name=%s\nGPU=%s\n' "$seed" "$exp_name" "$gpu" >> "$snapshot"
        git rev-parse HEAD >> "$snapshot"
        git diff -- src/openpi/models src/openpi/training >> "$snapshot"
        # Foreground pipeline: wait for training AND tee to exit. With pipefail,
        # any failed training stops the queue instead of starting the next run.
        "${cmd[@]}" 2>&1 | tee "$log_file"
        printf 'Completed: %s (seed %s)\n' "$config" "$seed"
    done
done
