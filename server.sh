cd /nfs/lizhenhao/Project/openpi

. env.sh
# 此处选择空闲的GPU
export CUDA_VISIBLE_DEVICES=0

# libero
policy_config="wafm_l2_nogate_loss_calvin_ABC_D_with_state"
policy_dir="/nfs/lizhenhao/Project/openpi/checkpoints/wafm_l2_nogate_loss_calvin_ABC_D_with_state/wafm_l2_nogate_loss_calvin_ABC_D_with_state/29999"



uv run scripts/serve_policy.py \
    --port 42001\
    policy:checkpoint \
    --policy.config $policy_config \
    --policy.dir $policy_dir
