# Raw/WFM 共享小波损失对照

此改动直接基于 `exp/capacity-matched-raw-heads`：沿用已有 `action_velocity_head`、Raw-FiLM实现及full-LIBERO训练协议。原linear/raw配置默认 `lambda_wavelet_flow_loss=0.0`，已有训练与推理行为保持不变。

## 下一步先跑什么

在训练服务器checkout这个分支，并保留你实际的训练数据、基座和归一化资源路径。先核对原Raw配置与已经训练的WFM seed0/seed1具有相同数据、状态输入、batch、初始化基座、优化器、EMA与checkpoint选择规则。

```bash
git switch exp/capacity-matched-raw-heads
git pull --ff-only

# 只打印串行计划，不启动训练、不加载env.sh
DRY_RUN=1 bash train_shared_loss.sh

# 单卡串行：先Raw-3×M128+loss，成功结束后再Raw-M384+loss
# 两项均只跑默认seed=42；将0换成H200的实际GPU编号
GPU_ID=0 bash train_shared_loss.sh

# 若要与已经完成的seed0实验配对，显式指定SEED=0
GPU_ID=0 SEED=0 bash train_shared_loss.sh

# 单独补某一组无loss对照，同样只跑一个默认seed
GPU_ID=0 bash train_shared_loss.sh raw_3x_noloss
```

脚本从项目目录运行，默认读取 `env.sh`，可通过 `TRAIN_ENV_FILE`指定其他环境脚本。默认选择 `raw_loss` 队列（先3×M128，后M384），默认只跑seed42，与分支现有默认seed一致；不会自动扩展到seed0/1。单seed先作为探索性对照，不能据此判断跨训练种子的稳定性；与已有seed0/1结果比较时需注明seed不同，或显式设 `SEED=0`/`SEED=1`。

GPU选择顺序为 `GPU_ID`、已有 `CUDA_VISIBLE_DEVICES`、默认0；只接受一张卡的编号或GPU/MIG UUID，拒绝多GPU列表。前一训练及其日志进程退出后才启动下一项，训练失败立即停止整个队列。开跑前会检查全部计划任务，已有日志或checkpoint时拒绝运行，不传 `--overwrite`或`--resume`。

每组使用独立实验名和日志。`EXP_PREFIX`可指定新的运行名称；训练协议、实际seed、GPU和git版本保存到日志旁的config文件中。不要把一个seed的checkpoint当另一个seed的初始化。后续若明确需要多seed，仍可显式传入，如 `bash train_shared_loss.sh raw_loss 0 1`，它们也会严格串行。

| 启动参数 | 训练配置 | 预测头 | λ |
|---|---|---|---:|
| `raw_loss`（默认队列） | 先`raw_3x_loss`，再`raw_wide_loss` | 两种Raw头依次训练 | .1 |
| `raw_3x_noloss` | `libero_full_raw_3xfilm_m128` | Raw-3×M128 | 0 |
| `raw_3x_loss` | `libero_full_raw_3xfilm_m128_wavloss` | Raw-3×M128 | .1 |
| `raw_wide_noloss` | `libero_full_raw_film_m384` | Raw-M384 | 0 |
| `raw_wide_loss` | `libero_full_raw_film_m384_wavloss` | Raw-M384 | .1 |
| `wfm_noloss` | `libero_full_wfm_l2_shared_noloss` | WFM-L2，无门控 | 0 |
| `wfm_loss` | `libero_full_wfm_l2_shared_loss` | WFM-L2，无门控 | .1 |

新增Raw+loss配置直接克隆对应的旧Raw配置。新增WFM配置克隆Raw-3×M128的训练协议，仅改变预测头与损失开关。Raw和WFM有效head在action_dim=32、token_dim=1024、M128时均为812,256参数。没有额外门控或门控正则。

## 辅助损失的含义

训练目标仍是 `u_t = noise - actions`。对最终 `[B,T,A]` 速度预测和目标做相同的Haar变换：

\[
L = \operatorname{mean}(v_t-u_t)^2
  +\lambda\sum_b\operatorname{mean}\bigl(W_b(Pv_t)-W_b(Pu_t)\bigr)^2.
\]

其中`P`为右侧末值复制填充；L2、T=10时补至12步，A2/D2/D1长度为3/3/6。每个子带分别取MSE均值再求和，不是所有系数的统一MSE。辅助损失在FP32计算，保留梯度，不做stop-gradient。λ=0时完全跳过辅助计算，保留原MSE路径。采样仍只使用最终动作速度，不计算训练辅助损失。

现有Raw头的输入和预测仍在原始时域；小波变换只出现在其训练损失。新 `wavelet_film` 预测头对动作和条件特征分别分解，各带使用与Raw同一FiLM单元，IDWT后裁剪至原T步，再计算同一个输出损失。

## 你已经训练的WFM结果

如果旧WFM直接监督内部子带，T=10的填充/裁剪会导致该目标与“最终输出重新DWT”的目标不完全相同。旧结果可作为原方案，与新增Raw+loss先作比较，但不能改名为严格共享输出损失的WFM格。

若要完整严格2×2比较，再运行：

```bash
GPU_ID=0 bash train_shared_loss.sh wfm_noloss
GPU_ID=0 bash train_shared_loss.sh wfm_loss
```

若已有λ=0的WFM结果，需确认协议及实现一致后才复用。原始线性base不能替代容量匹配Raw无loss。已有WFM分支的参数命名不同，旧WFM checkpoint继续使用原代码和配置评估，不要用本分支的新WFM配置直接加载旧WFM权重。

## 评估与测试

评估服务必须使用训练时对应的配置名，例如：

```bash
uv run scripts/serve_policy.py --port 42003 policy:checkpoint \
  --policy.config libero_full_raw_3xfilm_m128_wavloss \
  --policy.dir checkpoints/libero_full_raw_3xfilm_m128_wavloss/libero_full_raw_3xfilm_m128_wavloss_seed42/29999
```

检查点目录以实际生成结果为准。保持suite、初态、评估seed及每任务rollout数相同；不要把训练seed当成评估seed。新配置仅支持JAX `scripts/train.py`；本次未接入 `train_pytorch.py`。

在完整训练环境中测试：

```bash
JAX_PLATFORMS=cpu uv run pytest src/openpi/models/raw_velocity_head_test.py \
  src/openpi/models/wavelet_flow_loss_test.py \
  src/openpi/models/wavelet_velocity_head_test.py
```

正式长跑前，各组选真实batch做一次训练、checkpoint保存和重新加载推理检查。训练成功不等于已验证性能，本次代码修改不包含新训练结果。
