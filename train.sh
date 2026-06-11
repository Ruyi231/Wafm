cd /nfs/lizhenhao/Project/openpi

. env.sh

export CUDA_VISIBLE_DEVICES=2,3
export WANDB_MODE=offline


XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_wasfm_replace_finetune --exp-name=pi05_libero_wasfm_replace_finetune --overwrite
