cd /nfs/lizhenhao/Project/openpi

. env.sh
# 此处选择空闲的GPU
export CUDA_VISIBLE_DEVICES=3

# libero
policy_config="libero_full_raw_3xfilm_m128"
policy_dir="/nfs/lizhenhao/Project/openpi/checkpoints/libero_full_raw_3xfilm_m128/libero_full_raw_3xfilm_m128/29999"



uv run scripts/serve_policy.py \
    --port 42003\
    policy:checkpoint \
    --policy.config $policy_config \
    --policy.dir $policy_dir
