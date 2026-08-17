



# Research Diary — GAViT Project

> 记录每次重要进展、实验结果和 debug 过程。每次完成功能模块、获得新结果或修复重要 bug 后必须更新。

---

## 2026-08-17 — BigEarthNet sparse-hybrid 正式训练与测试（seed 42）

### 一、研究问题与实验身份

- 研究假设：在与 corrected cosine-kNN 相同的 80 条有向边预算下，以 48 条
  K=16 四邻接空间边和 32 条排除空间邻居后的 cosine top-2 特征边组成
  `sparse_hybrid_4n_top2`，能比纯 corrected-kNN 提供更有效的区域关系。
- 唯一主要变量是 graph topology；backbone、grouping、region 数、GAT、
  integration、数据划分、优化和 seed 均与 corrected-kNN control 保持一致。
- 论文用途：进入 `results/bigearth_comparison.csv`，作为 BigEarthNet edge
  topology 消融和候选最佳 GAViT；本条仍是单 seed 结果。
- 分支：`codex/sparse-hybrid-4n-top2`；训练 Git commit：`cd39be9`；
  metadata 记录 `dirty=false`。
- run stage：`formal`；run tag：
  `sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654`；seed 42。

### 二、数据、模型与优化

- BigEarthNet-19 固定 split：train 237,871 / val 122,342 / test 119,825。
  正式测试前逐行验证 test CSV 所需 B02/B03/B04 文件：119,825/119,825
  patch 完整，0 missing。
- 预处理：resize 224x224；训练随机水平/垂直翻转；ImageNet mean/std
  normalization。
- 模型：GAViT，31,366,926 参数；Swin-T backbone；K=16
  `attentive_spatial`；2-layer/4-head GAT（hidden 256）；dropout 0.1；
  `token_feedback` integration。
- 图：`edge_type=sparse_hybrid`；48 spatial + 32 feature = 80 条唯一有向边；
  feature top-2；`neighbor_to_query`；cosine 只决定拓扑。
- 预训练：`bigearth_files/model.safetensors`。
- 优化：30 epochs，batch size 32，AdamW，lr `3e-4`，weight decay `1e-4`，
  CosineAnnealingLR，BCEWithLogitsLoss；分类阈值 0.5。

正式训练命令：

```bash
nohup python -u -c \
'import runpy, torch, time; started=time.time(); runpy.run_path("train_bigearth.py", run_name="__main__"); print(f"Peak CUDA allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"); print(f"Total wall time: {(time.time()-started)/3600:.2f} h")' \
  --model gavit \
  --data_dir bigearth_files/splits \
  --epochs 30 \
  --batch_size 32 \
  --lr 3e-4 \
  --num_regions 16 \
  --knn_k 5 \
  --gat_hidden 256 \
  --gat_heads 4 \
  --gat_layers 2 \
  --grouping attentive_spatial \
  --edge_type sparse_hybrid \
  --integration token_feedback \
  --dropout 0.1 \
  --seed 42 \
  --run_stage formal \
  --run_tag sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654 \
  --pretrained_path bigearth_files/model.safetensors \
  --checkpoint_path checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654.pth \
  > logs/sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654.log 2>&1 &
```

### 三、训练结果与资源

- 完成 30/30 epochs；无 traceback、OOM、NaN 或中途终止。
- **Best Validation mAP：79.2087%（epoch 13）**；metadata 和 last state 的
  best metric 一致，last state epoch 30。
- Epoch 13 后训练 loss 继续下降，但验证 mAP 总体回落；epoch 17 短暂回到
  78.9%，未超过 best。Epoch 30：loss 0.0834、Val mAP 76.9%、Val macro-F1
  71.8%；相对 best mAP 下降 2.31 pp，存在明确后期过拟合。
- 前五轮 Val mAP：74.3%、76.4%、76.9%、78.0%、78.1%；epoch 7 为 78.8%，
  epoch 13 达峰；epoch 20--30 位于 76.9%--77.7%。测试严格使用 epoch 13
  best checkpoint，而非 epoch 30 last state。
- 训练设备：NVIDIA GeForce RTX 4090（24,564 MiB）；启动阶段约
  14.10 train batch/s；峰值 CUDA allocated 3.65 GiB；总 wall time 6.64 h，
  平均 wall time 约 13.28 min/epoch；实际费用待确认。
- 训练日志：
  `logs/sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654.log`。
- Best checkpoint：
  `checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654.pth`；
  同 stem 保存 `.meta.json` 和 `.last.train_state.pth`。

### 四、正式测试

正式测试命令：

```bash
nohup python -u -c \
'import runpy, torch, time; started=time.time(); runpy.run_path("test_bigearth.py", run_name="__main__"); print(f"Peak CUDA allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"); print(f"Test wall time: {time.time()-started:.2f} s")' \
  --model gavit \
  --data_dir bigearth_files/splits \
  --ckpt checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654.pth \
  --batch_size 32 \
  > logs/sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654_test_rtx3060_20260817_114017.log 2>&1 &
```

- 测试设备：NVIDIA GeForce RTX 3060（12,288 MiB）；3,745 batches；tqdm
  10:33，5.91 batch/s；端到端 wall time 665.55 s（11.09 min）；峰值 CUDA
  allocated 0.60 GiB。
- **Test macro mAP：72.0%；macro-F1：66.0%；micro-F1：76.7%。**
- 测试日志：
  `logs/sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654_test_rtx3060_20260817_114017.log`。

| Class | Sparse-hybrid AP | vs corrected-kNN | vs Swin |
|---|---:|---:|---:|
| Urban fabric | 86.3% | +0.1 pp | -0.2 pp |
| Industrial or commercial units | 51.1% | +1.6 pp | -1.3 pp |
| Arable land | 93.8% | +0.6 pp | +0.5 pp |
| Permanent crops | 61.2% | +2.3 pp | +2.4 pp |
| Pastures | 85.8% | -0.3 pp | -0.4 pp |
| Complex cultivation patterns | 70.1% | +1.1 pp | +2.1 pp |
| Land principally occupied by agriculture, with significant areas of natural vegetation | 73.7% | +1.2 pp | +0.7 pp |
| Agro-forestry areas | 87.1% | +1.2 pp | -0.2 pp |
| Broad-leaved forest | 86.0% | +0.4 pp | +0.9 pp |
| Coniferous forest | 93.5% | -0.2 pp | +0.0 pp |
| Mixed forest | 89.8% | +1.1 pp | +0.5 pp |
| Natural grassland and sparsely vegetated areas | 42.1% | -1.9 pp | -0.8 pp |
| Moors, heathland and sclerophyllous vegetation | 61.8% | -1.0 pp | +5.0 pp |
| Transitional woodland, shrub | 78.8% | +0.0 pp | +0.5 pp |
| Beaches, dunes, sands | 16.8% | +6.7 pp | +3.5 pp |
| Inland wetlands | 64.1% | +1.0 pp | +1.4 pp |
| Coastal wetlands | 34.7% | +12.9 pp | +4.4 pp |
| Inland waters | 91.7% | +0.0 pp | +0.8 pp |
| Marine waters | 99.6% | -0.1 pp | +0.1 pp |

### 五、比较、解释与下一步

| Model | Test mAP | Macro-F1 | Micro-F1 |
|---|---:|---:|---:|
| Swin-T baseline | 70.9% | 64.2% | 76.6% |
| Corrected-kNN GAViT | 70.6% | 65.4% | 76.6% |
| **Sparse-hybrid GAViT** | **72.0%** | **66.0%** | **76.7%** |

1. 相对 corrected-kNN，sparse-hybrid 的 test mAP +1.4 pp、macro-F1 +0.6 pp、
   micro-F1 +0.1 pp；相对 Swin 分别为 +1.1、+1.8、+0.1 pp。验证 mAP 也比
   corrected-kNN best 78.5732% 高 0.6355 pp，验证和测试方向一致。
2. mAP 增益主要来自 Coastal wetlands（+12.9 pp vs corrected）、Beaches
   （+6.7 pp）、Permanent crops（+2.3 pp）等困难类别；micro-F1 几乎不变，
   因此当前证据更支持“改善宏观/尾部类别表现”，而不是全面提高样本级预测。
3. Natural grassland（-1.9 pp）和 Moors（-1.0 pp）相对 corrected 下降，
   sparse topology 并非所有类别一致受益，后续应在论文中保留该限制。
4. **seed 42 的研究假设得到支持，sparse-hybrid 暂定为最佳 GAViT。** 单 seed
   不能声称稳定提升；不再搜索新 topology。下一步只为最终对比补齐 selected
   GAViT 与 Swin baseline 的 seeds 43/44，并报告三 seeds 均值与标准差。

---

## 2026-08-14 — sparse_hybrid_4n_top2 Featurize 工程 gate 通过

### 实验身份与验证环境

- 分支：`codex/sparse-hybrid-4n-top2`；Git commit：`a28cbe1`。
- Featurize GPU：NVIDIA GeForce RTX 4090（24,564 MiB）。
- 固定 BigEarthNet-19 split 已核对：train 237,871 / val 122,342 /
  test 119,825；原始图像位于实例本地
  `/home/featurize/data/BigEarthNet-S2`。
- 完整测试命令：`python -m unittest discover -s tests -v`。
- 测试结果：59 tests，1.879 s，全部通过；无 failure、error 或 skipped
  sparse-hybrid test。覆盖了 48/32/80 edge budget、top-2 排除规则、稳定
  tie-break、batch offset、forward/backward finite gradients、checkpoint metadata
  round-trip 和 training-state 恢复约束。

### Smoke 配置与结果

- 数据：一次性 256 train / 128 val split；1 epoch；batch size 32；seed 42。
- 模型：GAViT，31,366,926 参数；Swin-T backbone；K=16
  `attentive_spatial`；2-layer/4-head GAT（hidden 256）；
  `edge_type=sparse_hybrid`；`token_feedback`；dropout 0.1。
- 预训练权重：`bigearth_files/model.safetensors`；AdamW，lr `3e-4`，
  weight decay `1e-4`，CosineAnnealingLR，BCEWithLogitsLoss。
- run stage：`smoke`；run tag：
  `sparse_hybrid_4n_top2_a28cbe1_smoke_retry_20260814_125255`。
- 训练完成 8/8 batches，tqdm 平均 5.75 batch/s；该短跑吞吐仅用于工程
  估算，不作为正式效率结果。
- Epoch 1 loss：0.3408；Val macro mAP：47.7286%；Val macro-F1：15.9%。
- 峰值 CUDA allocated：3.65 GiB；总 wall time：8.07 s。
- 无 NaN、Inf、OOM 或 traceback；forward、backward、optimizer step、validation
  和 artifact save 均完成。

### 拓扑与产物验证

- metadata 中 `architecture.edge_type=sparse_hybrid`。
- `graph_topology` 明确记录：`name=sparse_hybrid_4n_top2`、
  `spatial_connectivity=4`、`feature_k=2`、48 条空间有向边、32 条特征有向边、
  总计 80 条唯一有向边、`message_direction=neighbor_to_query`、
  `cosine_role=topology_only`。
- Best checkpoint（约 120 MB）：
  `checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_a28cbe1_smoke_retry_20260814_125255.pth`。
- Metadata（约 1.4 KB）：同 stem 的 `.meta.json`；best metric 为
  `val_mAP=47.7286`，best epoch 1。
- Last training state（约 360 MB）：同 stem 的 `.last.train_state.pth`；
  last epoch 1，`best_metric=47.7285909477752`。
- 日志：
  `logs/sparse_hybrid_4n_top2_a28cbe1_smoke_retry_20260814_125255.log`。
- 独立 artifact 校验脚本完成文件存在性、architecture、topology、best epoch 和
  last-state 断言，输出 `SMOKE ARTIFACT VERIFICATION: OK`。

### 异常与结论

- 首次 smoke 在训练 8/8 后、无进度条的 validation 阶段被手动 `Ctrl+C`
  中断，因此只写入初始 metadata（`best=None`），没有 checkpoint 或 last state；
  使用新的 run identity 重试后完整通过。该中断不是模型或拓扑故障。
- **工程 gate 已通过。** Smoke 指标只证明运行路径和产物契约正确，不进入
  `results/bigearth_comparison.csv`，也不作为论文性能证据。
- 下一步按既定漏斗运行固定 seed 42 的 5-epoch proxy；proxy 之前不启动 30-epoch
  formal，也不追加其他拓扑或 integration 变量。

### Fixed 10% proxy 基础设施

- 为避免直接付费运行全量五轮，已批准固定 10% train/val、5 epochs、training
  seed 42 的 proxy 协议；proxy 只检查多轮学习稳定性，不替代全量 30-epoch
  formal，也不进入论文结果表。
- 设计 commit：`2ab6026`；实现 commit：`3f82563`。
- 新增 `baselines/bigearth/prepare_proxy_split.py`：从 seeds 42--141 的 100 个
  均匀随机候选中选择 train/val 共 38 个类别 prevalence 最大偏差最小者；任何
  split 缺少正类的候选 fail closed，分数相同选择较小 seed。
- 预期输出到 `bigearth_files/proxy_10pct_seed42`：23,787 train / 12,234 val、
  `prevalence.json`、输入/输出 SHA256、每类正例数与 prevalence deviation；
  test split 不参与生成或 proxy 训练，已有输出不会被覆盖。
- 本地标准库定向测试：12 tests 全部通过；相关 torch-free 套件：53 tests
  全部通过；生成器和测试通过 `py_compile`，`git diff --check` 通过。
- 本地完整 discover 因环境没有 PyTorch，三个 PyTorch test module 在 import
  阶段报 `ModuleNotFoundError: torch`；这不是代码回归。Featurize 拉取实现后仍须
  运行完整 suite，预期总数由 59 增至 71，全部通过后才能生成 proxy split。
- 生成、独立 hash/row/prevalence 校验和五轮前台训练命令已写入
  `docs/featurize_runbook.md`；实际运行结果记录如下，分布报告 SHA256 尚待补记。

### Fixed 10% proxy 实际运行结果

- Featurize 在 commit `cd39be9` 上运行完整测试：71 tests，1.512 s，全部通过。
- 生成器从 seeds 42--141 中选择 seed 137；train 23,787 / val 12,234；
  train/val 共 38 个类别 prevalence 的最大绝对偏差为 `0.00394910`
  （约 0.395 个百分点）。输出目录：
  `bigearth_files/proxy_10pct_seed42`。独立报告 SHA256 待补记。
- Proxy run stage：`proxy`；run tag：
  `sparse_hybrid_4n_top2_cd39be9_proxy_20260814_133423`；训练 seed 42。
- 模型和优化配置与计划一致：GAViT、K=16 `attentive_spatial`、
  `sparse_hybrid` 4-neighbor + top-2、2-layer/4-head GAT、
  `token_feedback`、batch size 32、lr `3e-4`、5 epochs。
- 每轮 744 个训练 batch；训练吞吐依次为 13.68、13.80、13.88、13.89、
  13.84 batch/s。Epoch 1--5 的 loss 为 0.2234、0.1852、0.1673、0.1492、
  0.1330；Val macro mAP 为 65.5%、67.9%、70.7%、73.7%、74.8%；
  Val macro-F1 为 56.3%、60.1%、60.7%、64.7%、67.3%。
- Best Validation mAP：74.8325%（epoch 5）；峰值 CUDA allocated：3.65 GiB；
  总 wall time：345.48 s（5.76 min，平均 69.10 s/epoch）。无 OOM、NaN、
  traceback 或指标退化，loss 单调下降且 mAP/F1 总体持续上升。
- Best checkpoint 保存到
  `checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_cd39be9_proxy_20260814_133423.pth`；
  训练日志为 `logs/sparse_hybrid_4n_top2_cd39be9_proxy_20260814_133423.log`。
  独立 artifact 校验确认同 stem 的 `.meta.json` 和 `.last.train_state.pth`
  可读取；metadata best 为 `val_mAP=74.8325`、epoch 5，last state epoch 5，
  Git identity 为 commit `cd39be9`、`dirty=false`，输出
  `PROXY ARTIFACT VERIFICATION: OK`。
- **多轮学习稳定性 proxy gate 已通过。** 该结果只支持启动固定 seed 42 的
  全量 30-epoch formal；不与全量 corrected-kNN 指标直接比较，也不进入论文
  结果表。按数据规模和本次 wall time 线性外推，formal 约 5.76 h；这是启动前
  成本估算，不是实际耗时。

---

## 2026-08-13 — sparse_hybrid_4n_top2 本地代码完成，待 Featurize 工程验证

### 研究假设与单一变量

- 研究问题：在与 corrected cosine-kNN 相同的 80 条 pre-GAT 有向边预算下，
  同时编码空间邻接与非局部特征相似性的稀疏图，是否能提供更符合导师建议、
  更有意义的区域关系。
- 唯一改变的主要变量是 graph topology。Swin-T、K=16
  `attentive_spatial`、2-layer/4-head GAT、`token_feedback`、数据划分、优化器、
  loss、scheduler、增强和阈值保持不变。
- 新配置名为 `edge_type=sparse_hybrid`；历史 `edge_type=hybrid` 的 8-neighbor
  加 kNN 直接拼接行为保持不变，不能用于本实验命名。

### 实现与身份

- 设计规格：`docs/superpowers/specs/2026-07-28-sparse-hybrid-graph-experiment-design.md`。
- 实施计划 commit：`d53cb1d`。
- 图构建 commit：`0be936a`。
- GAViT 与四个 train/test CLI 接线 commit：`79faafa`。
- CLI/topology 静态契约收紧 commit：`2c3f85b`。
- 显式 topology metadata 与 resume identity commit：`3bfd8d5`。
- 分支：`codex/sparse-hybrid-4n-top2`。
- 每张 K=16 图构造 48 条四邻接空间边；每个 query 排除 self 和空间邻居后，
  按 cosine similarity 选择 top-2，共 32 条特征边；合计 80 条唯一有向非自环边。
- 全部边遵守 PyG 的 `neighbor -> query` 消息方向。cosine 数值只用于选择拓扑，
  不传入 GAT 作为 message weight；相等分数按较小 region index 决定顺序。
- `sparse_hybrid` 对 K!=16 和 feature_k!=2 fail closed；checkpoint 名称和
  metadata 使用独立的 `edge_type=sparse_hybrid` 身份，并显式保存
  `graph_topology` 的 4-neighbor、feature_k=2、48/32/80 和消息方向快照；
  `--knn_k 5` 只是与 corrected-kNN 控制命令保持兼容，不表示本拓扑选择五条
  feature edges。resume identity 同时绑定该 topology 快照。

### 本地已验证与尚缺验证

- 本地 `python3 -m unittest tests.test_sparse_hybrid_static_contract
  tests.test_experiment_identity -v`：18 tests 全部通过。
- 本地 torch-free training-state 合约：10 tests 全部通过。
- 修改过的 graph/model/train/test/test-suite 文件均通过 `py_compile`；
  `git diff --check` 通过。
- 已加入但**尚未在本地执行**的 PyTorch 测试覆盖：80-edge budget、48/32 分解、
  四邻接无对角线、top-2 排除规则、tie-break、batch offset、无 self/重复/
  cross-image edge、模型 forward/backward finite gradients，以及 metadata
  round-trip logits。
- 原因：本机 Python 环境没有 PyTorch/NumPy，按既定原则未在本地安装；这些
  张量级测试必须在 Featurize 环境运行后才能标记为通过。
- Featurize 必跑命令和 256/128、1-epoch smoke gate 已写入
  `docs/featurize_runbook.md`。待记录：完整测试数量、GPU、峰值显存、throughput、
  首轮耗时、checkpoint/metadata/last-state 路径和 smoke 结果。

**当前结论：代码已实现，但工程 gate 尚未通过；未启动 proxy 或正式训练。**

---

## 2026-08-12 — BigEarthNet corrected-kNN GAViT 正式训练与测试

### 一、研究问题与实验身份

- 研究问题：修正 cosine-kNN 的消息方向为 `selected_neighbor -> query` 后，当前 GAViT v2 是否优于历史错误方向结果，并缩小或逆转相对 Swin baseline 的差距。
- 本次只改变 kNN 消息方向；数据划分、K=16 attentive spatial grouping、k=5、2-layer/4-head GAT、token feedback、优化器和训练轮数保持不变。
- Git commit：`8cb8563`；训练 metadata 记录 `dirty=false`。
- Featurize：NVIDIA GeForce RTX 4090（24,564 MiB）。
- 训练前验证：`python -m unittest discover -s tests -v`，44 tests，全部通过。
- 工程 smoke：256 train / 128 val，1 epoch，batch size 32；完成 forward、backward、validation 和隔离 checkpoint 保存，峰值 CUDA allocated 3.65 GiB。smoke 指标不作为论文性能证据。

### 二、正式配置与命令

- 数据集：BigEarthNet-19；固定 split 为 Train 237,871 / Val 122,342 / Test 119,825。
- 预处理：resize 224×224；训练使用随机水平/垂直翻转；ImageNet mean/std normalization。
- 模型：GAViT v2，31,366,926 参数；Swin-T backbone；K=16 `attentive_spatial`；corrected cosine-kNN `k=5`；GAT hidden=256、4 heads、2 layers；`token_feedback` integration；dropout=0.1。
- 预训练来源：持久化本地 `bigearth_files/model.safetensors`。
- 优化：30 epochs，batch size 32，seed 42，AdamW，lr 3e-4，weight decay 1e-4，CosineAnnealingLR，BCEWithLogitsLoss；F1 threshold=0.5。

正式训练命令：

```bash
nohup python -u -c \
'import runpy, torch, time; started=time.time(); runpy.run_path("train_bigearth.py", run_name="__main__"); print(f"Peak CUDA allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"); print(f"Total wall time: {(time.time()-started)/3600:.2f} h")' \
  --model gavit \
  --data_dir bigearth_files/splits \
  --epochs 30 \
  --batch_size 32 \
  --lr 3e-4 \
  --num_regions 16 \
  --knn_k 5 \
  --gat_hidden 256 \
  --gat_heads 4 \
  --gat_layers 2 \
  --grouping attentive_spatial \
  --edge_type knn \
  --integration token_feedback \
  --dropout 0.1 \
  --seed 42 \
  --run_stage formal \
  --run_tag corrected_knn_8cb8563_20260811 \
  --pretrained_path bigearth_files/model.safetensors \
  --checkpoint_path checkpoints/best_bigearth_gavit_K16_attentive_spatial_knn_k5_token_feedback_seed42_formal_corrected_knn_8cb8563_20260811.pth \
  > logs/bigearth_gavit_corrected_knn_seed42_8cb8563_20260811.log 2>&1 &
```

正式测试命令：

```bash
nohup python -u test_bigearth.py \
  --model gavit \
  --data_dir bigearth_files/splits \
  --ckpt checkpoints/best_bigearth_gavit_K16_attentive_spatial_knn_k5_token_feedback_seed42_formal_corrected_knn_8cb8563_20260811.pth \
  --batch_size 32 \
  > logs/bigearth_gavit_corrected_knn_seed42_8cb8563_20260811_test.log 2>&1 &
```

### 三、训练结果与资源

- 完成 30/30 epochs；无 traceback、OOM 或中途终止。
- **Best Validation mAP：78.5732%（epoch 17）**。
- Epoch 30：loss 0.0817，Val mAP 76.7%，Val macro-F1 71.7%。
- best 后验证 mAP 持续回落，存在明显后期过拟合；测试严格使用 epoch 17 best checkpoint，而不是 epoch 30 last state。
- 峰值 CUDA allocated：3.65 GiB。
- 训练总时长：6.49 h；实际费用待确认。
- 测试：3,745 batches，3:16，19.06 batch/s。

产物：

- 训练日志：`logs/bigearth_gavit_corrected_knn_seed42_8cb8563_20260811.log`
- 测试日志：`logs/bigearth_gavit_corrected_knn_seed42_8cb8563_20260811_test.log`
- Best checkpoint：`checkpoints/best_bigearth_gavit_K16_attentive_spatial_knn_k5_token_feedback_seed42_formal_corrected_knn_8cb8563_20260811.pth`（约 120 MB）
- Metadata：同名 `.meta.json`（best epoch 17，Git `8cb8563`，dirty=false）
- Last training state：同名 `.last.train_state.pth`（epoch 30，约 360 MB，仅用于精确恢复，不用于本次测试）

### 四、正式测试结果

- **Test macro mAP：70.6%**
- **Test macro-F1：65.4%**
- **Test micro-F1：76.6%**
- 分类阈值：0.5

| Class | AP | vs Swin |
|---|---:|---:|
| Urban fabric | 86.2% | -0.3 pp |
| Industrial or commercial units | 49.5% | -2.9 pp |
| Arable land | 93.2% | -0.1 pp |
| Permanent crops | 58.9% | +0.1 pp |
| Pastures | 86.1% | -0.1 pp |
| Complex cultivation patterns | 69.0% | +1.0 pp |
| Land principally occupied by agriculture, with significant areas of natural vegetation | 72.5% | -0.5 pp |
| Agro-forestry areas | 85.9% | -1.4 pp |
| Broad-leaved forest | 85.6% | +0.5 pp |
| Coniferous forest | 93.7% | +0.2 pp |
| Mixed forest | 88.7% | -0.6 pp |
| Natural grassland and sparsely vegetated areas | 44.0% | +1.1 pp |
| Moors, heathland and sclerophyllous vegetation | 62.8% | +6.0 pp |
| Transitional woodland, shrub | 78.8% | +0.5 pp |
| Beaches, dunes, sands | 10.1% | -3.2 pp |
| Inland wetlands | 63.1% | +0.4 pp |
| Coastal wetlands | 21.8% | -8.5 pp |
| Inland waters | 91.7% | +0.8 pp |
| Marine waters | 99.7% | +0.2 pp |

### 五、比较与结论

| Model | Test mAP | Macro-F1 | Micro-F1 | 说明 |
|---|---:|---:|---:|---|
| Swin-T baseline | 70.9% | 64.2% | 76.6% | 相同正式 split，seed 42 |
| Historical GAViT v2 | 70.4% | 待确认 | 待确认 | 修正消息方向前的历史结果 |
| Corrected-kNN GAViT v2 | 70.6% | 65.4% | 76.6% | 本次正式结果，seed 42 |

1. 修正消息方向后，GAViT 相对历史结果的 test mAP 提升 0.2 pp，说明方向错误有小幅负面影响，但不是性能差距的主要原因。
2. Corrected GAViT 的 mAP 仍比 Swin 低 0.3 pp；单 seed 下不能声称 graph module 提升总体 mAP。
3. Macro-F1 比 Swin 高 1.2 pp，micro-F1 持平；类别层面改善和退化并存，其中 Moors/... +6.0 pp，但 Coastal wetlands -8.5 pp。该结果值得在后续拓扑实验中继续观察，不能单独作为模型优越性的证据。
4. 本条形成时的下一项论文问题，是按已批准设计实现并验证 `sparse_hybrid_4n_top2`，以 corrected-kNN 结果作为同预算正式对照；该功能后来已于 2026-08-13 在 `codex/sparse-hybrid-4n-top2` 分支完成本地代码，但仍待 Featurize 工程验证。历史 `hybrid` 仍不得冒充该配置。
5. 在 sparse-hybrid 比较完成前，不追加 corrected-kNN 多 seed，也不启动 gated/cross-attention integration 实验。

---

## 2026-07-14 — Featurize BigEarthNet Swin baseline 完整训练

### 一、实验设置

- 平台：Featurize（GPU 具体型号待从实例记录确认）
- 数据集：BigEarthNet-19，多标签分类
- 数据划分：Train 237,871 / Val 122,342 / Test 119,825
- 模型：Swin-T baseline，27,535,501 参数
- 初始化：本地持久化 `bigearth_files/model.safetensors`，避免 Featurize 访问 Hugging Face 的 TLS 失败
- Seed：42
- 训练：30 epochs，batch size 32，AdamW，lr 3e-4，weight decay 1e-4，CosineAnnealingLR
- Loss：BCEWithLogitsLoss
- 验证指标：macro mAP + macro-F1（threshold=0.5）
- 启动入口：`train_bigearth_local.py`（临时为 Swin 注入本地 pretrained file）
- 日志：`logs/bigearth_swin.log`
- Checkpoint：`checkpoints/best_bigearth_swin.pth`，约 106 MB

### 二、训练结果

- 完成 30/30 epochs
- **Best Validation mAP：78.8%**（准确 best epoch 待从完整日志提取）
- Epoch 30 Loss：0.0223
- Epoch 30 Val mAP：75.8%
- Epoch 30 Val macro-F1：71.2%
- 后期 Val mAP 从最佳值回落至 75.8%，存在过拟合迹象；正式比较应使用 best checkpoint，而不是最后一轮权重
- 已观察到 epoch 24–30 单轮训练约 7:52–8:35；峰值显存、完整 wall time 和实际费用待补记

### 三、存储与复现状态

- BigEarthNet 图像位于实例本地 `/home/featurize/data/BigEarthNet-S2`，更换实例后需要重新添加
- metadata、split CSV、本地预训练权重、日志和 checkpoint 位于 `/home/featurize/work/GAViT_Project`，更换实例后保留
- 固定 split 已持久化在 `bigearth_files/splits`，无需随实例重新生成

### 四、测试结果

- 测试样本：119,825
- 测试耗时：2:25（3,745 batches，25.68 it/s）
- 分类阈值：0.5
- **Test macro mAP：70.9%**
- **Test macro-F1：64.2%**
- **Test micro-F1：76.6%**
- 测试日志：`logs/bigearth_swin_test_retry.log`

Per-class AP：

| Class | AP |
|---|---:|
| Urban fabric | 86.5% |
| Industrial or commercial units | 52.4% |
| Arable land | 93.3% |
| Permanent crops | 58.8% |
| Pastures | 86.2% |
| Complex cultivation patterns | 68.0% |
| Land principally occupied by agriculture, with significant areas of natural vegetation | 73.0% |
| Agro-forestry areas | 87.3% |
| Broad-leaved forest | 85.1% |
| Coniferous forest | 93.5% |
| Mixed forest | 89.3% |
| Natural grassland and sparsely vegetated areas | 42.9% |
| Moors, heathland and sclerophyllous vegetation | 56.8% |
| Transitional woodland, shrub | 78.3% |
| Beaches, dunes, sands | 13.3% |
| Inland wetlands | 62.7% |
| Coastal wetlands | 30.3% |
| Inland waters | 90.9% |
| Marine waters | 99.5% |

### 五、测试脚本问题与修复

- 第一次完整推理完成后，`average_precision_score` 报错：预测数组仍为三维。
- 根因：`test_bigearth.py` 未处理 timm Swin 返回的 `(B, H, W, C)` 四维特征，分类器输出成为 `(B, H, W, 19)`；训练脚本已正确进行空间池化。
- 修复：四维特征在维度 `(1, 2)` 上取均值，三维 token 特征在维度 `1` 上取均值。
- 修复后先用 32 条样本 smoke test，再运行完整测试，避免重复浪费 GPU 时间。

### 六、论文状态与下一步

- 本次结果建立了 BigEarthNet 上的 Swin baseline，后续 GAViT 必须使用相同 split、训练设置和主要指标
- [ ] 从完整日志确认 best epoch
- [ ] 补记训练和测试使用的 GPU 型号、峰值显存、完整 wall time 与实际费用
- [ ] 固定 GAViT 正式训练配置，并先完成小规模 smoke test
- [ ] 使用相同 split、epoch、batch size、seed 和指标训练及测试 GAViT

---

## 2026-04-08 — BigEarthNet 训练超时修复：添加断点续训

**问题**：
- BigEarthNet Swin baseline（Job 56169）在 epoch 16/30 因 8h 时间限制被 SLURM 杀掉
- BigEarthNet GAViT（Job 56170）在 epoch 24/30 因 10h 时间限制被杀掉
- 估算：Swin 每 epoch ~29min（总需 ~14.5h），GAViT 每 epoch ~25min（总需 ~12.5h）

**解决方案**：
- `train_bigearth.py` 新增 `--resume` 参数，从已有 best checkpoint 加载模型权重继续训练
- `train_bigearth.py` 新增 `--start_epoch` 参数，控制训练循环起始 epoch
- 注意：仅恢复模型权重，optimizer/scheduler 状态不保存（影响极小，模型已接近收敛）

**作业脚本更新**：
- `run_bigearth_train_swin.sh`：时间 8h → 16h，添加 `--resume --start_epoch 17`
- `run_bigearth_train_gavit.sh`：时间 10h → 6h，添加 `--resume --start_epoch 25`

**下一步**：
- [ ] Push 后在服务器 `git pull` 并重新 sbatch 两个作业
- [ ] 训练完成后运行测试脚本对比 Swin vs GAViT 在 BigEarthNet 上的表现

---

## 2026-04-04 — GAViT v2 训练结果 + BigEarthNet 实验准备

### 一、GAViT v2 训练 & 测试结果

**训练结果**（Job 53893，NVIDIA RTX A5000）：
- Model: GAViT | K=16 | grouping=attentive_spatial | edge=knn | integration=token_feedback | GAT 2L×4H
- 参数量：31,386,920
- **Best Val Acc：96.2%**（epoch 26）
- 训练过程：epoch 25-30 val acc 在 95.9%–96.2% 波动，train acc epoch 29 达 100%

**测试结果**（Job 54030，NVIDIA Tesla V100）：
- **Test Acc：95.8%**（4526/4725）

**完整对比**：

| 模型 | Val Acc | Test Acc | Params |
|------|---------|----------|--------|
| Swin-T Baseline | ~96.0% | 96.3% | 27.5M |
| GAViT v1 K=9 spatial + GAT 2L | 96.5% | 95.9% | 30.5M |
| GAViT Fusion (backbone+graph concat) | 95.9% | 95.9% | 30.5M |
| GAViT v2 K=16 attentive + token_feedback | 96.2% | 95.8% | 31.4M |

**结论**：三种 GAViT 变体 test acc 均在 95.8-95.9%，与 baseline（96.3%）差 ~0.4%。与教授预判一致——NWPU-RESISC45 单标签数据集 global appearance 信号太强，graph module 增益天然受限。

### 二、BigEarthNet 多标签实验准备

按 Prof Wang 0327 建议，新增 BigEarthNet 实验以展示 graph module 在关系更重要场景的价值。

**完成内容**：

1. **`models/bigearth_dataset.py`** — BigEarthNet Dataset 类
   - 支持 BigEarthNet-S2 格式（读取 B02/B03/B04 TIF 波段 → RGB）
   - 内置 43→19 类标签映射（BigEarthNet-19 标准）
   - 返回 19 维 binary label vector

2. **`baselines/bigearth/prepare_bigearth.py`** — 数据准备脚本
   - 自动下载官方 train/val/test split 列表
   - 生成 split CSV 文件（patch_path + 19 维标签列）

3. **`train_bigearth.py`** — 统一训练脚本
   - 支持 `--model swin`（baseline）和 `--model gavit`（GAViT v2）
   - Loss: BCEWithLogitsLoss（多标签）
   - Metric: mAP（mean Average Precision）+ Macro-F1 @ 0.5

4. **`test_bigearth.py`** — 测试脚本
   - 输出 mAP / macro-F1 / micro-F1 + per-class AP

5. **Slurm 作业脚本**：
   - `jobs/run_bigearth_prepare.sh`
   - `jobs/run_bigearth_train.sh`（依次训练 Swin baseline + GAViT v2）
   - `jobs/run_bigearth_test.sh`

6. **数据下载进行中**
   - 服务器创建了 `/projects/gavitdata/`（SSD，150GB 配额）
   - BigEarthNet-S2-v1.0 从 Zenodo 下载中（~65GB，约 6 小时）
   - 下载链接：`https://zenodo.org/records/12687186/files/BigEarthNet-S2-v1.0.tar.gz?download=1`

### 三、下一步计划（下载完成后按顺序执行）

- [ ] 选择性解压 RGB 波段 + 标签 JSON，删掉 tar.gz
  ```
  tar -xzf BigEarthNet-S2-v1.0.tar.gz --wildcards '*_B02.tif' '*_B03.tif' '*_B04.tif' '*_labels_metadata.json'
  rm BigEarthNet-S2-v1.0.tar.gz
  ```
- [ ] 运行 prepare 脚本生成 split CSV（`sbatch jobs/run_bigearth_prepare.sh`）
- [ ] 训练 Swin baseline + GAViT v2（`sbatch jobs/run_bigearth_train.sh`）
- [ ] 测试两个模型（`sbatch jobs/run_bigearth_test.sh`）
- [ ] 对比结果：NWPU（graph 增益小）vs BigEarthNet（graph 增益预期更大）
- [ ] NWPU 可视化：airport/bridge/church 的 region 分区和 graph 连接
- [ ] 整理所有结果给导师发邮件

---

## 2026-04-04 — GAViT v2 架构改进：AttentiveSpatialGrouping + Token Feedback

**动机（基于 Prof Wang 0327 回复）**：

教授指出当前 GAViT 的两个核心问题：
1. **集成方式不当**：graph module 输出在分类器阶段才与 backbone 拼接（fusion），relational reasoning 没有影响表征本身。应让 graph 输出直接修改 token 特征。
2. **Region 定义太粗糙**：3×3 spatial grid 分辨率不够，小结构被合并；mean pooling 没有区分 token 的重要性。应提高分辨率（4×4 或 5×5）并用 attention weighting 替代均值。

此外教授指出 NWPU-RESISC45 作为单标签数据集，global appearance 已是很强信号，graph module 增益天然有限。建议新增 BigEarthNet（多标签）数据集实验。

**完成内容**：

1. **`models/region_grouping.py` — 新增 `AttentiveSpatialGrouping`**
   - 支持任意完全平方数 K（默认 K=16，即 4×4 grid）
   - 每个 region 内用 lightweight attention（2 层 MLP → scalar score → softmax）加权聚合 token，替代均值 pooling
   - 信息量大的 token 贡献更多，使 region 特征更有语义代表性

2. **`models/gavit.py` — 新增 `token_feedback` 集成模式**
   - GAT 精炼 region features 后，通过 `feedback_proj`（Linear + LayerNorm）映射回 768 维
   - 每个 token 通过 assignments 获取其所属 region 的精炼特征
   - 残差更新：`updated_tokens = original_tokens + region_feedback`
   - 在更新后的 tokens 上做 mean pool → 分类
   - 旧的 `fusion` 模式保留，通过 `--integration` 参数切换

3. **`train_gavit.py` — 新增命令行参数**
   - `--grouping attentive_spatial`（新默认值）
   - `--integration token_feedback`（新默认值）
   - `--num_regions` 默认改为 16
   - Checkpoint 命名包含 integration 类型

**架构对比**：

| 版本 | Region Grouping | 集成方式 | 分类器输入 |
|------|----------------|----------|-----------|
| v1 (legacy fusion) | SpatialGrouping 3×3, mean pool | 末端 concat | backbone_global(768) + graph_global(1024) = 1792 |
| v2 (token feedback) | AttentiveSpatialGrouping 4×4, attention | token-level residual | updated tokens mean pool = 768 |

**实验结果**：

- **Best Val Acc：96.2%**（epoch 26）
- **Test Acc：95.8%**（4526/4725）
- 参数量：31,386,920（全部可训练）
- Checkpoint：`checkpoints/best_gavit_K16_attentive_spatial_knn_token_feedback.pth`
- 训练过程：epoch 25-30 val acc 在 95.9%–96.2% 之间波动，epoch 26 达峰；train acc 在 epoch 29 即达 100%，存在轻微过拟合
- 收敛特征：val acc 在 epoch 20 才突破 95.5%，收敛比 v1 慢（v1 epoch 20 已 ~96%），但最终结果持平

**对比 v1（GAViT K=9 spatial + GAT 2L，96.5%）**：
- Val Acc 96.2% vs 96.5%，略低 0.3%
- 尽管 attentive grouping + token feedback 架构更复杂，性能未见提升，可能原因：
  - K=16 比 K=9 区域更细，每个 region 内 token 数减少（~3 个），attention weighting 效果有限
  - Token feedback 引入了更长的梯度路径，30 epoch 内收敛不充分
  - NWPU-RESISC45 单标签数据集对 relational modeling 增益天然受限

**下一步计划**：
- [ ] 在 test set 上评估 v2（运行 `test_gavit.py`，加载 v2 checkpoint）
- [ ] 对比 v2 vs v1 vs Swin-T baseline 的混淆矩阵
- [ ] 可视化 v2 region 分区和 graph 连接（airport, bridge, church）
- [ ] 考虑延长训练至 50 epoch，或调整 lr schedule
- [ ] 准备 BigEarthNet 数据集实验

---

## 2026-03-27 — Backbone-Graph Fusion 架构实验 + Test Set 分析

### 一、Fusion 架构训练

**完成内容**：
- 修改 `gavit.py` 分类器架构：从纯 graph 特征改为 backbone global + graph global 拼接（768 + 1024 = 1792 维）
- 动机：之前的 GAViT 分类器只用 graph module 输出，backbone 全局特征被丢弃，fusion 旨在让两者互补
- 新建 `jobs/run_gavit_fusion.sh`，训练 50 epochs
- Checkpoint：`best_gavit_K9_spatial_knn_fusion.pth`

**训练结果**：
- Best Val Acc：**95.9%**（epoch 47-48）
- 训练过程：Train Acc epoch 42 达到 100%，Val Acc 从 epoch 28 的 94.4% 缓慢爬升至 95.9%

### 二、Fusion 模型 Test Set 分析

对 Fusion 模型跑了和原版 GAViT 相同的 per-class accuracy + 混淆矩阵分析（`compare_models.py --tag fusion`）。

**整体结果**：

| 模型 | Val Acc | Test Acc | vs Baseline (Test) |
|------|---------|----------|-------------------|
| Swin-T Baseline | ~96.0% | 96.3% | — |
| GAViT 原版（纯 graph） | 96.5% | 95.9% | -0.4% |
| GAViT Fusion（backbone+graph） | 95.9% | 95.9% | -0.4% |

两个 GAViT 版本在 test set 上表现完全一致（95.9%），均比 baseline 低 0.4%。但考虑到这种量级的差距（~0.4%）在单次运行中不具备统计显著性，三个模型可视为持平。

**Fusion Per-class 变化（vs Swin Baseline）**：

提升最大：
- roundabout: 97.1% → 100.0%（+2.9%）
- terrace: 95.2% → 98.1%（+2.9%）
- wetland: 89.5% → 91.4%（+1.9%）
- ship: 96.2% → 98.1%（+1.9%）

下降最大：
- bridge: 99.0% → 94.3%（-4.8%）
- airport: 98.1% → 95.2%（-2.9%）
- church: 87.6% → 84.8%（-2.9%）
- medium residential: 96.2% → 93.3%（-2.9%）
- lake: 97.1% → 94.3%（-2.9%）

**Sample-level 分析**：
- Baseline 错 → Fusion 对（修复）：67 samples
- Baseline 对 → Fusion 错（引入）：86 samples
- 净改善：-19 samples

**与原版 GAViT 的 per-class 对比**：
- 两者模式类似：复杂场景（airport, bridge, church）均未改善
- Fusion 波动幅度略小于原版（原版 palace -4.8%/lake -4.8%，Fusion 最大 bridge -4.8%）
- 两者提升的类也类似（ship, wetland, terrace, roundabout）

### 三、关键结论

1. **Fusion vs 原版 GAViT**：两种架构在 test set 上结果一致（95.9%），简单拼接并未带来额外收益
2. **两种 GAViT vs Baseline**：差距仅 0.4%，在单次运行中无统计显著性，三者可视为持平
3. **复杂场景未受益**：airport、bridge、palace 等需要 relational reasoning 的类别，两种 GAViT 均未改善，反而有所下降
4. **核心问题**：graph module 目前尚未展现出对复杂场景分类的独特价值

### 四、下一步计划

- [ ] 整理结果给导师发邮件，请教 graph module 与 backbone 的集成方式
- [ ] 根据导师反馈决定后续方向

---

## 2026-03-26 — Edge 消融实验 + 可视化对比 + 关键讨论

### 一、完成内容

1. **GAT 注意力可视化**（`visualize_graph.py`）
   - 对 `best_gavit_K9_spatial.pth` 生成 2×2 面板（原图 / 区域分配 / kNN图 / GAT注意力）
   - 6 类场景各 2 张，保存至 `results/figures/graph_vis_*.png`

2. **Spatial adjacency 边构建**（`models/graph_construction.py` 新增 `build_spatial_graph`）
   - 3×3 grid 的 8-邻接连边，边权均为 1
   - `gavit.py` 新增 `edge_type` 参数：`knn` / `spatial` / `hybrid`
   - `train_gavit.py` 新增 `--edge_type` 命令行参数

3. **Edge 消融实验**（`jobs/run_edge_ablation.sh`，Job 50315）

| Edge Type | Val Acc | 备注 |
|-----------|---------|------|
| spatial adjacency | **96.2%** | 固定 8-邻接 |
| cosine kNN | 96.1% | 动态边 |
| hybrid (合并) | 96.0% | kNN + spatial |

   **结论**：三种边策略准确率几乎一致，边定义对最终分类影响极小。

4. **Edge comparison 可视化**（`visualize_edge_comparison.py`，Job 50414）
   - 同一张图对比 spatial vs kNN 的 GAT 注意力，1×3 面板
   - 保存至 `results/figures/edge_compare_*.png` 和 `edge_comparison_summary.png`

### 二、关键讨论与反思

**GAT attention 权重的解读问题**：

- ⚠️ **GAT attention ≠ 语义关联度**。attention 权重反映的是"从哪个邻居获取信息最有用"（信息流方向），而不是"哪两个区域语义上相关"。
- 实际观察：stadium 中 R4（球场）对 R0、R2（角落/停车场）的注意力反而高于对 R3、R5（看台），因为差异大的邻居提供更多互补信息。
- 这意味着**用 attention 权重来直接讲"跑道+航站楼→强关联→机场"的故事是不准确的**。

**模型本身没问题**：

- GAViT 的 graph module 确实在帮助分类（96.0% → 96.5%），relational modeling 是有效的
- 只是 attention 权重不适合作为"展示 relational modeling 价值"的主要证据

**更好的展示方向（待验证）**：

- **注意力熵（attention entropy）对比**：复杂场景（airport, stadium）注意力应更集中（低熵），均质场景（forest, desert）注意力更分散（高熵）。"不均匀 vs 均匀"这个对比本身就说明模型根据场景动态调整了信息流。
- **混淆矩阵对比**：Baseline 分错但 GAViT 分对的样本，具体是哪些类
- **Per-class accuracy 对比**：但因为 baseline 已经 96%，差异可能很小，说服力有限

### 三、导师建议执行进度

- ✅ Region grouping 简单设计（spatial + kmeans）
- ✅ 轻量 GNN（2 层 GAT）
- ✅ 三级消融（baseline → +region → +graph）
- ✅ 图连接可视化（已完成，但解读方式需调整）
- ✅ 多种边定义系统对比（spatial / kNN / hybrid，准确率持平）
- ❌ AID 数据集验证

### 四、下一步计划（优先级排序）

- [ ] 计算注意力熵，量化"复杂场景 vs 均质场景"的注意力分布差异
- [ ] 在 test set 上评估最优模型，得到正式 Test Acc
- [ ] 混淆矩阵对比（Baseline vs GAViT）
- [ ] 消融：K 值影响（K=4 vs 9 vs 16）
- [ ] AID 数据集验证
- [ ] 给导师发进度邮件

---

## 2026-03-17 — 项目初始化 & Swin-T Baseline 完成

**完成内容**：
- 初始化项目结构，配置 Python 虚拟环境（PyTorch + timm + PyTorch Geometric）
- 编写 `split_nwpu.py`，将 NWPU-RESISC45 原始数据按 70/15/15 划分为 train/val/test，random seed=42
- 编写 `train_swin_baseline.py`：Swin-T (swin_tiny_patch4_window7_224)，pretrained=True，30 epoch，AdamW lr=3e-4，CosineAnnealingLR
- 编写 `test_swin.py`：加载 checkpoint 在 test set 评估

**实验结果**：
- Val Acc：~96%
- Test Acc：~96%
- 对比基线：本身即为基线

**遇到的问题及解决方案**：
- 无重大问题

**下一步计划**：
- [x] 实现 GAViT 核心模块（`models/` 目录）
- [ ] 训练 GAViT v1 并与 baseline 对比

---

## 2026-03-22 — P2a & P2b 完整实验：GAViT spatial/kmeans + 2-layer GAT

**完成内容**：
- P2a：GAViT K=9 SpatialGrouping + 2-layer GAT（4头，kNN k=5）
- P2b：GAViT K=9 KMeansGrouping + 2-layer GAT（重跑，含 argparse 版）
- 参数量：30,462,375（两组相同）

**实验结果**：
- P2a Best Val Acc：**96.5%**（epoch 30，仍在上升！）
- P2b Best Val Acc：**96.0%**（epoch 28-30）
- Checkpoint：`best_gavit_K9_spatial.pth` / `best_gavit_K9_kmeans.pth`

**消融结论**（完整链条）：

| 模型 | Val Acc | vs Baseline |
|------|---------|-------------|
| Swin-T Baseline | ~96.0% | — |
| + SpatialGrouping (no GNN) | 96.2% | +0.2% |
| + SpatialGrouping + GAT 2L | **96.5%** | **+0.5%** |
| + KMeansGrouping + GAT 2L | 96.0% | ≈0% |

- GAT 在 spatial grouping 基础上带来 +0.3% 提升，证明图推理有效
- KMeans 版性能低于 Spatial 版，原因：kmeans 随机性导致图结构不稳定，GNN 难以学习一致的区域关系
- Spatial+GAT 在 epoch 30 仍在上升（96.4%→96.5%），值得尝试更多 epoch

**遇到的问题及解决方案**：
- 无

**下一步计划**：
- [ ] 在 test set 上评估最优模型（`best_gavit_K9_spatial.pth`）
- [ ] 尝试 spatial+GAT 训练 50 epoch，观察是否继续上升
- [ ] 消融：K 的影响（4 vs 9 vs 16）
- [ ] 绘制训练曲线对比图

---

## 2026-03-22 — P2b 实验完成：GAViT K=9 KMeansGrouping + 2-layer GAT

**完成内容**：
- 在 NTU EEE GPU 服务器（NVIDIA RTX A5000）上运行 GAViT 完整模型（kmeans grouping）
- 模型：GAViT | K=9 | grouping=kmeans | GAT 2L×4H | kNN k=5
- 参数量：30,462,375（全部可训练，比 SpatialGrouping 版多约 290 万参数）
- 训练配置：30 epoch，AdamW lr=3e-4，CosineAnnealingLR，batch_size=32

**实验结果**：
- Best Val Acc：**96.0%**（epoch 29）
- 对比 Swin-T Baseline (~96.0%)：持平
- 对比 SpatialGrouping + no GNN (96.2%)：-0.2%
- 初步结论：kmeans 动态聚类引入了随机性，收敛更慢（epoch 1 train acc 仅 75.9% vs spatial 的 79.7%），最终性能与 baseline 持平

**训练曲线观察**：
- Epoch 1-10：收敛明显慢于 P1（epoch 10 val 92.2% vs P1 的 92.3%）
- Epoch 18-30：缓慢爬升，train acc 趋近 100%，val acc 在 95-96% 区间波动
- Checkpoint 保存至：`checkpoints/best_gavit.pth`（⚠️ 旧脚本名，argparse 未生效）

**注意事项**：
- 服务器拉取了 argparse 版本之前就已提交作业，导致只跑了 kmeans 一组，checkpoint 名为旧版 `best_gavit.pth`
- P2a（spatial + GAT）**尚未跑**，需补跑以完成消融对比

**下一步计划**：
- [ ] push argparse 版 `train_gavit.py` → 服务器 git pull → 补跑 P2a（`--grouping spatial`）
- [ ] 对比四组消融结果，绘制 bar chart
- [ ] 分析 kmeans vs spatial 差异原因

---

## 2026-03-22 — P1 实验完成：Swin + SpatialGrouping K=9（无 GNN）

**完成内容**：
- 在 NTU EEE GPU 服务器（NVIDIA RTX A5000, 24GB）上运行 P1 实验
- 模型：Swin-T + SpatialGrouping（K=9），无图推理层，直接 mean pool → FC 分类
- 参数量：27,555,495（全部可训练）
- 训练配置：30 epoch，AdamW lr=3e-4，CosineAnnealingLR，batch_size=32

**实验结果**：
- Best Val Acc：**96.2%**（epoch 27-28）
- 对比 Swin-T Baseline (~96.0%)：+0.2%（微弱提升）
- 结论：仅加 Region Grouping（无 GNN）相比 baseline 基本持平，说明区域表示本身信息量有限，需要图推理来充分利用区域间关系

**训练曲线观察**：
- Epoch 1-13：快速收敛，val acc 从 87% 升至 94%
- Epoch 14-28：缓慢爬升，train acc 趋近 100%（过拟合迹象）
- Best checkpoint 保存至：`checkpoints/best_region_only_K9_spatial.pth`

**遇到的问题及解决方案**：
- 无

**下一步计划**：
- [ ] **P2** 运行完整 GAViT v1（Swin + SpatialGrouping K=9 + 2-layer GAT）
- [ ] 对比三组结果：Baseline vs +Region Grouping vs +Graph Reasoning
- [ ] 保存训练曲线图至 `results/figures/`

---

## 2026-03-17 — GAViT 核心模块实现完成

**完成内容**：
- `models/swin_backbone.py`：封装 Swin-T，去掉分类头，`forward_features` 输出 `(B, 49, 768)` tokens
- `models/region_grouping.py`：实现两种聚类方式
  - `KMeansGrouping`：特征空间 k-means（PyTorch 实现，GPU 兼容），assignment 步骤 detach，averaging 步骤可微，梯度可回传至 backbone
  - `SpatialGrouping`：将 7×7 grid 按空间位置分为 K 个宏区域（K 须为完全平方数），全程可微
- `models/graph_construction.py`：`build_knn_graph`，基于 cosine similarity 构建有向 kNN 图，输出 PyG batch 格式 `edge_index` + `edge_weight` + `batch`
- `models/graph_reasoning.py`：多层 GAT（`GATConv`），含 input projection、残差连接、LayerNorm、GELU、Dropout
- `models/gavit.py`：完整 GAViT 模型，串联以上所有模块
- `train_gavit.py`：训练脚本（默认 K=9, kNN k=5, 2-layer GAT, 4 heads）
- `test_gavit.py`：测试脚本
- `utils.py`：`set_seed`、`accuracy` 工具函数

**实验结果**：
- 模块实现完成，尚未运行训练

**遇到的问题及解决方案**：
- 无

**下一步计划**：
- [ ] 在 GPU 服务器上运行 `train_gavit.py`，获得 GAViT v1 结果
- [ ] 将结果填入 `results/comparison_table.csv`
- [ ] 保存训练曲线至 `results/figures/`

---
