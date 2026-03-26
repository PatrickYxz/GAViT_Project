# Research Diary — GAViT Project

> 记录每次重要进展、实验结果和 debug 过程。每次完成功能模块、获得新结果或修复重要 bug 后必须更新。

---

## 2026-03-26 — GAT 注意力可视化完成 & 下一阶段规划

**完成内容**：
- 新增 `visualize_graph.py`：对训练好的 GAViT 模型提取 GAT 注意力权重，生成 2×2 可视化面板（原图 / 区域分配 / kNN 图 / GAT 注意力图）
- 在服务器运行可视化脚本，对 6 类场景（airport, stadium, harbor, dense_residential, forest, desert）各 2 张图生成可视化结果
- 可视化结果保存至 `results/figures/graph_vis_*.png`

**可视化观察**：
- 复杂场景（stadium, harbor）的注意力分布有一定结构性，但与均质场景（forest, desert）的差异不够显著
- 当前仅使用 cosine kNN 边构建策略，注意力区分度有限
- 结论：可视化初步可用，但需要改进边构建策略以增强解释性

**导师建议回顾（对照执行情况）**：
- ✅ Region grouping 简单设计（spatial + kmeans）
- ✅ 轻量 GNN（2 层 GAT）
- ✅ 三级消融（baseline → +region → +graph）
- ✅ 图连接可视化
- ⚠️ 多种边定义系统对比（仅完成 cosine kNN，未做 spatial adjacency）
- ❌ AID 数据集验证

**下一步计划**：
- [ ] 实现 spatial adjacency 边构建策略，与 cosine kNN 做消融对比
- [ ] 在 test set 上评估最优模型，得到正式 Test Acc
- [ ] 消融：K 值影响（K=4 vs 9 vs 16）
- [ ] 在 AID 数据集上验证泛化性
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
