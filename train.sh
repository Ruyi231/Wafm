cd /nfs/lizhenhao/Project/openpi

. env.sh

export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE=offline


XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py libero_full_raw_film_m384 --exp-name=libero_full_raw_film_m384 --overwrite