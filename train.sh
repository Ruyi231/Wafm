cd /nfs/lizhenhao/Project/openpi

. env.sh

export CUDA_VISIBLE_DEVICES=6,7
export WANDB_MODE=offline


XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py wafm_l2_nogate_loss_calvin_ABC_D_with_state --exp-name=wafm_l2_nogate_loss_calvin_ABC_D_with_state --overwrite
