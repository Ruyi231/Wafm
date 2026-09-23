#!/usr/bin/env bash
set -eo pipefail

cd /nfs/lizhenhao/Project/work1
source env.sh

export CUDA_VISIBLE_DEVICES=0,2
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1

mkdir -p logs
uv run --offline scripts/train.py plan1_subband_l2_norm \
  --exp-name stage1_bs128_seed42_live \
  --batch-size 128 \
  --fsdp-devices 2 \
  --no-wandb-enabled \
  --no-overwrite \
  --no-resume \
  2>&1 | tee logs/plan1_subband_l2_norm_stage1_bs128_live.log
