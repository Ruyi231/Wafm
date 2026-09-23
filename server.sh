#!/usr/bin/env bash
set -e

PROJECT_DIR="/nfs/lizhenhao/Project/work1"
cd "$PROJECT_DIR"

. env.sh

# Select an idle GPU. GPU 5 was almost full and could terminate the policy
# process during model loading or inference, which appears as a simulator
# WebSocket disconnect. Override this at launch time when GPU allocation changes.
SERVER_GPU="${SERVER_GPU:-0}"
export CUDA_VISIBLE_DEVICES="$SERVER_GPU"

# Keep the serving endpoint and checkpoint explicit, matching the stable
# launcher in /nfs/lizhenhao/Project/openpi/server.sh.
SERVER_PORT="${SERVER_PORT:-42004}"
POLICY_CONFIG="${POLICY_CONFIG:-plan1_subband_l2_norm_recon}"
POLICY_DIR="${POLICY_DIR:-$PROJECT_DIR/checkpoints/plan1_subband_l2_norm_recon/stage1_recon_bs128_seed42/29999}"

for required_path in \
    "$POLICY_DIR/_CHECKPOINT_METADATA" \
    "$POLICY_DIR/params" \
    "$POLICY_DIR/assets"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Incomplete checkpoint; missing: $required_path" >&2
        exit 1
    fi
done

export PYTHONPATH="src:."
export PYTHONUNBUFFERED=1

echo "Starting OpenPI WebSocket policy server"
echo "  config:     $POLICY_CONFIG"
echo "  checkpoint: $POLICY_DIR"
echo "  GPU:        $SERVER_GPU"
echo "  port:       $SERVER_PORT"

if [[ "${SERVER_DRY_RUN:-0}" == "1" ]]; then
    echo "Dry run complete; server was not started."
    exit 0
fi

exec uv run --offline scripts/serve_policy.py \
    --port "$SERVER_PORT" \
    policy:checkpoint \
    --policy.config "$POLICY_CONFIG" \
    --policy.dir "$POLICY_DIR"
