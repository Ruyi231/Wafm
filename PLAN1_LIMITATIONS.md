# Plan 1：NH-WaFM 限制与未完成验证

_本文件只记录当前代码与已执行测试能够支持的结论；截至 2026-07-28。_

---

## 📋 结论边界

当前交付已经完成真正子带 flow、子带统计归一化、轻量层次耦合、两种 action token 条件、子带 ODE 和 checkpoint 部分加载，并通过定向单测与 400 步合成 overfit。

当前交付没有执行 LIBERO、CALVIN 或 RoboTwin 的完整训练和评测，也没有多随机种子任务成功率。因此不能声称 NH-WaFM 优于原始 \(\pi_{0.5}\) 或 legacy WaFM。

| 结论 | 是否可声称 | 依据 |
| --- | --- | --- |
| Haar 与归一化 roundtrip 正确 | 可以，限已测 shape | 定向单测 |
| 速度反归一化不添加 mean | 可以 | 显式实现与单测 |
| 1/5/10 步 dummy sampling 可 JIT 运行 | 可以 | 定向 model 测试 |
| 固定合成批次可 overfit | 可以 | 400 步结果 JSON |
| 完整 Pi0 可在真实数据上收敛 | 不可以 | 未执行 |
| 任务成功率提升 | 不可以 | 未执行 benchmark |
| 推理延迟可接受 | 不可以 | 未做 wall-clock/GPU 基准 |

## ⚠️ 实验与数据限制

### Overfit 只覆盖子带 head

400 步实验训练的是 `NormalizedHierarchicalWaveletFlowHead`，输入为固定合成动作、固定随机 action token 和固定 timestep。它没有训练 PaliGemma、完整 action expert、真实 observation pipeline 或真实数据 loader。

该实验的意义是发现明显的 bridge、shape、梯度或 IDWT 错误。它不是端到端 imitation learning 结果，也不能替代真实动作终点误差或任务成功率。

### 大规模实验未执行

以下结果均未产生：

- LIBERO 任务成功率、动作终点误差、jerk 和夹爪切换误差
- CALVIN 单任务指标
- RoboTwin 或真实精细操作指标
- 两个及以上随机种子的均值、方差和置信区间
- 原始 \(\pi_{0.5}\)、legacy WaFM 与 NH-WaFM 的同数据、同预算公平比较
- level 1/2/3 的训练与推理成本比较

[`PLAN1_EXPERIMENT_RESULTS.csv`](PLAN1_EXPERIMENT_RESULTS.csv) 的 benchmark 字段应保持空值或显式 `pending`，直到真实评测完成。不得把合成 overfit loss 写入 success-rate 列。

### 统计文件与数据严格绑定

wavelet mean/std 依赖：

- OpenPI `norm_stats` 的具体内容
- 数据集与数据 transform
- action horizon 与 action dimension
- Haar levels 与 edge-padding 规则

更换任一项后都应重新计算统计。统计脚本会拒绝未加载动作 `norm_stats` 的 data config，并优先写入模型配置的 `wavelet_norm_stats_path`；模型加载器会继续校验 levels、horizon 和 action dimension。但这些检查仍无法证明数据集内容本身与 `source_config` 名称一致，服务器记录必须保存数据版本与统计文件 SHA-256。

## ⚙️ 配置与运行限制

### JSON 实验矩阵仍不是运行时配置

[`configs/plan1_experiments.json`](configs/plan1_experiments.json) 是可审计参数矩阵，不会被 `openpi.training.config.cli()` 自动读取。阶段 1 的四个 LIBERO 条目已经手工注册为命名 `TrainConfig`；`debug_nh_wafm` 仍只使用 fake data 与 dummy 模型。

阶段 1 服务器验证前仍必须：

1. 在 `assets/pi05_libero/physical-intelligence/libero` 生成真实动作 `norm_stats`
2. 用同一归一化数据生成 `wavelet_norm_stats_l2.json`
3. 运行新增配置测试、debug 训练、checkpoint resume 和短训练门槛
4. 训练和 `serve_policy.py` 使用同一个配置名

如果只在训练 CLI 临时覆盖模型字段，而评测仍用 `pi05_libero`，服务端会按原始 \(\pi_{0.5}\) 架构创建模型，导致 checkpoint tree 不匹配。

阶段 2 的 hierarchical/Band Query 与阶段 3 的 level 1/2/3 条目仍未注册，这是有意的顺序门控，不是已经可启动的实验。

### 缺少完整 benchmark 驱动

仓库包含 LIBERO client。当前 checkout 没有与之等价的完整 CALVIN benchmark driver，也没有现成 RoboTwin experiment。已有 CALVIN policy adapter 或 server config 不等于完成可复现 benchmark。

### JAX 路径已实现，PyTorch 路径未移植

NH-WaFM 修改位于 JAX/Flax NNX 的 `Pi0`。`scripts/train_pytorch.py` 和 PyTorch 模型没有对应的子带 flow bridge、层次 head 或子带 ODE 实现。JAX NH-WaFM checkpoint 也不能据此假设可直接转换为具备相同行为的 PyTorch policy。

## 🧪 测试覆盖限制

### 已执行范围

实际执行的是 wavelet、normalization、checkpoint 和 dummy Pi0 的定向测试，共得到：

- 16 个 wavelet/head/normalization case 通过
- 4 个 NH-WaFM 非有限配置校验 case 通过
- 4 个 checkpoint case 通过
- 4 个 NH-WaFM model case 通过
- 1 个 legacy mode case 通过

这些结果覆盖了用户要求的核心测试名称，但不是整个仓库的全量回归。

### 尚未执行范围

- 全仓库 `pytest`
- 真实 LeRobot/RLDS data loader 端到端 smoke
- `debug_nh_wafm` 的完整 `train_step` 与 checkpoint save smoke；本次尝试在导入阶段因隔离环境缺少 `lerobot` 而停止
- 多 GPU、FSDP、checkpoint save/resume 的真实训练循环
- CUDA 上的 compile time、峰值显存和 step time
- 真实 batch size 128/256 的 forward/backward
- 实际 policy server 与 LIBERO client 的 WebSocket 联调
- 长时间 ODE 采样的数值漂移和多 seed NaN 检查

因此当前“JIT 稳定”仅指已测 dummy shape 与 `num_steps=1/5/10`，不能外推到所有设备、batch size 和数据配置。

## 🔬 方法学限制

### Shared noise 是非 canonical 消融

canonical `wavelet_shared_noise=false` 在每个标准化子带直接采样独立 \(N(0,I)\)。`true` 分支先在完整 padded horizon 上采样一个物理动作域高斯张量，再执行正交 Haar DWT，并用训练数据 band stats 做状态归一化。这样避免了先按原 horizon 采样再 edge-replicate 噪声所引入的补齐相关性。

shared 分支保留一个跨 band 对应的动作域噪声样本，但归一化后的边际分布通常不再是标准 \(N(0,I)\)，因为它使用真实动作子带的 mean/std。它只能作为噪声先验消融，不能与 canonical 独立子带结果混合。

### 层次条件是预测值，不是真实 coarse target

细带 head 使用预测的 \(A_L\) 和前一个更粗 detail 作为条件。训练时没有 teacher forcing coarse target。这样避免训练/推理条件不一致，但误差可能从粗带向细带传播。`wavelet_detach_coarse_condition` 只能控制梯度路径，不能消除数值误差传播。

### Band query 仍是轻量单头注意力

band query 只从当前 action token 提取条件，没有长期历史或世界状态建模。每个 band 最多使用 4 个可学习 query，再静态映射到目标 band 时间长度。增加 levels 会增加新的 band 模块；极短 band 跨越 1–4 个 query 的边界时，改变 horizon 也可能改变 query 参数 shape，二者都需要重新检查 checkpoint 兼容。

### Cross-band consistency 尚未验证收益

跨频带项先反归一化为物理速度，再对每个 band 的全部系数平方求和，并比较预测/目标的 energy fraction。平方和按系数数量自然计权，并可由正交 Haar 的 Parseval 关系解释；但它尚未在真实数据上验证。它可能与直接 band MSE 重复，或在总能量很低的样本上受分母 epsilon 影响。默认关闭是当前最安全的实验起点。

### Action reconstruction loss 不是独立信息

Haar 变换在未裁剪的正交情形下保留能量，因此均匀 band MSE 与动作域 MSE 存在强相关。padding、不同 band 权重和标准差缩放会使两者不完全等价，但真实收益必须通过消融验证。该损失默认关闭。

### Euler 求解器未做精度/成本比较

推理保留显式 Euler，尚未比较更高阶求解器、不同 step 数的动作质量或延迟。已测 `num_steps=1/5/10` 只证明代码可运行，不说明哪个步数最优。

## 💾 checkpoint 限制

部分加载仅允许缺失路径包含 `lora` 或 `wavelet_flow_head`。这保证旧 \(\pi_{0.5}\) 权重可用于新训练初始化，但不保证随机初始化的新 head 在零步训练时产生可用动作。

其他限制包括：

- legacy 与 NH-WaFM 的 `wavelet_flow_head` 内部树不同，必须用匹配配置评测
- levels、conditioning mode 或 bottleneck dimension 改变时，head shape 可能不兼容
- 旧完整 optimizer state 不能无损迁移到新 head
- Orbax resume 应用于相同架构，不应用于旧架构到 NH-WaFM 的迁移
- extra 参数默认可 warning 后删除；需要严格审计时应设置 `remove_extra_params=false`
- 当前 checkpoint `assets` 回调只保存 OpenPI 动作 `norm_stats`，不会复制独立的 wavelet stats JSON；迁移 checkpoint 时必须同时保留配置所指向的统计文件及其 SHA-256

日志能防止静默跳过，但无法替代一次实际 checkpoint load 与 policy inference 验证。当前定向测试使用的是小型参数树，不是真实多 GB checkpoint。

## 🛑 止损状态

### 已解除的代码级止损

固定合成批次的所有子带 loss 和 IDWT 动作速度 MSE 均显著下降：

- `loss_band_A`: 2.701014 → 0.006562
- `loss_band_D1`: 2.598769 → 0.028478
- `loss_band_D2`: 2.181447 → 0.003566
- `action_idwt_mse`: 1.261985 → 0.005659

因此“子带 head 完全无法 overfit”这一代码级止损条件未触发。

### 尚未解除的研究级止损

以下条件仍无数据判断：

- 频带归一化是否让真实数据各 band loss 更稳定下降
- 训练 loss 改善是否转化为动作域终点误差改善
- 两个以上 seed 是否稳定优于原始 \(\pi_{0.5}\)
- 推理耗时增加是否有明确任务收益
- Band Query 或层次耦合是否带来独立收益

在这些条件验证前，不应继续加入复杂 gate、长期历史、阶段标签或世界模型。

## 🎯 建议的下一验证顺序

1. 已完成：为同一 LIBERO 数据注册四个阶段 1 命名配置
2. 在服务器计算 level 2 统计，校验 metadata 和抽样 roundtrip
3. 每个配置先跑短训练并检查所有 band loss、energy 和 gradient norm
4. 用同一 checkpoint step 比较 action-domain validation error
5. 只在阶段 1 稳定后注册并运行 independent/hierarchical/Band Query
6. 只在结构收益稳定后注册并比较 level 1/2/3
7. 最后运行至少两个 seed 的 LIBERO；CALVIN/RoboTwin 在补齐 driver 后再开始

每一步都应把真实命令、commit、seed、checkpoint step 和原始指标写入 [`PLAN1_EXPERIMENT_RESULTS.csv`](PLAN1_EXPERIMENT_RESULTS.csv)。未执行的行保持 `pending`，负结果也应保留。
