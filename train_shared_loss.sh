#!/usr/bin/env bash
# Examples: bash train_shared_loss.sh raw_3x_loss 0 1
#           DRY_RUN=1 bash train_shared_loss.sh wfm_loss 0 1
set -euo pipefail
cd "$(dirname "$0")"
variant="${1:-raw_3x_loss}"
if [[ $# -gt 0 ]]; then shift; fi
case "$variant" in
    raw_3x_loss) config=libero_full_raw_3xfilm_m128_wavloss ;;
    raw_3x_noloss) config=libero_full_raw_3xfilm_m128 ;;
    raw_wide_loss) config=libero_full_raw_film_m384_wavloss ;;
    raw_wide_noloss) config=libero_full_raw_film_m384 ;;
    wfm_loss) config=libero_full_wfm_l2_shared_loss ;;
    wfm_noloss) config=libero_full_wfm_l2_shared_noloss ;;
    *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac
if [[ $# -eq 0 ]]; then set -- 0 1; fi
# Validate all seeds before starting any training.
for seed in "$@"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
done
if [[ "${DRY_RUN:-0}" != 1 ]]; then source "${TRAIN_ENV_FILE:-./env.sh}"; fi
for seed in "$@"; do
    exp_name="${EXP_PREFIX:-$config}_seed${seed}"
    cmd=(uv run scripts/train.py "$config" "--exp-name=$exp_name" "--seed=$seed")
    printf 'Command:'; printf ' %q' "${cmd[@]}"; printf '\n'
    if [[ "${DRY_RUN:-0}" == 1 ]]; then continue; fi
    log_dir="${LOG_DIR:-logs/shared_loss}/$config"
    log_file="$log_dir/$exp_name.log"
    if [[ -e "$log_file" || -e "checkpoints/$config/$exp_name" ]]; then
        echo "Run or log already exists: $exp_name. Choose a new EXP_PREFIX." >&2
        exit 3
    fi
    mkdir -p "$log_dir"
    # Store the resolved protocol and git revision, including uncommitted changes.
    snapshot="$log_dir/$exp_name.config.txt"
    uv run python -c 'import dataclasses,sys; from openpi.training.config import get_config; c=dataclasses.replace(get_config(sys.argv[1]),seed=int(sys.argv[2]),exp_name=sys.argv[3]); print(dataclasses.asdict(c))' \
        "$config" "$seed" "$exp_name" > "$snapshot"
    printf '\ntraining_seed=%s\nexp_name=%s\n' "$seed" "$exp_name" >> "$snapshot"
    git rev-parse HEAD >> "$snapshot"
    git diff -- src/openpi/models src/openpi/training >> "$snapshot"
    "${cmd[@]}" 2>&1 | tee "$log_file"
done
