# WAFM λ 敏感性实验（LIBERO Full）

直接继承原来的最优配置 `libero_full_wafm_l2_nogate_loss`，只修改
`model.lambda_wavelet_flow_loss`。训练目标仍为原版 WAFM 的
`loss_action + λ * loss_wavelet_flow`，其中小波损失监督 head 内部预测的子带速度。

| λ | 配置名 | 本次串行脚本 |
| --- | --- | --- |
| 0.03 | `libero_full_wafm_l2_nogate_loss_lambda_0p03` | 第一组 |
| 0.1 | `libero_full_wafm_l2_nogate_loss` | 使用已有实验 |
| 0.3 | `libero_full_wafm_l2_nogate_loss_lambda_0p3` | 第二组 |

两组继承相同的数据及归一化设置、π0.5 预训练权重、L2、M128、无 gate、
action horizon=10、batch size=128、30,000 步、学习率 5e-5、EMA=0.999。
使用原配置的同一份归一化统计；数据和预训练权重路径也沿用原配置。

在服务器仓库目录运行：

```bash
git switch decrete_band
git pull --ff-only origin decrete_band

# 先查看将执行的命令，不加载训练环境、不创建日志。
DRY_RUN=1 GPU_ID=0 bash train_wafm_lambda.sh

# 单张 H200，先完成 λ=0.03，再开始 λ=0.3；默认只跑 seed=42。
GPU_ID=0 bash train_wafm_lambda.sh
```

`GPU_ID` 优先于已有的 `CUDA_VISIBLE_DEVICES`，两者都未设置时使用 GPU 0。
脚本拒绝多卡列表，任一训练失败立即停止。每组日志和配置快照保存在
`logs/wafm_lambda/<配置名>/`，checkpoint 保存在
`checkpoints/<配置名>/<配置名>_seed42/`。脚本检查已有日志和 checkpoint，
避免覆盖已有结果。

λ=0.1 原实验默认 seed=42，本轮两组也默认 seed=42。若比较其他已有种子，
可显式设置 `SEED=0`，并让三个 λ 使用相同的训练种子和评测协议。
需要单独启动某一组时：

```bash
GPU_ID=0 bash train_wafm_lambda.sh 0.03
GPU_ID=0 bash train_wafm_lambda.sh 0.3
```

这两组配置属于 `decrete_band` 的原版 WAFM；此前 Raw/WFM 共享输出损失对照
位于 `exp/capacity-matched-raw-heads` 分支，两个实验使用不同的小波损失定义，
汇总结果时应分别记录。
