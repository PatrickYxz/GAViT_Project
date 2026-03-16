# CLAUDE.md — GAViT Project Context

> 每次启动 Claude Code 会话时，请首先阅读此文件，了解项目背景、当前进度和操作规范。

---

## 一、项目概述

**项目全称**：Graph-Augmented Vision Transformers for Remote Sensing Image Scene Understanding（GAViT）

**核心目标**：在 Swin Transformer 骨干网络之上，引入动态场景图模块，通过建模图像区域间的空间与语义关系，提升遥感图像场景分类性能。

The core idea: Swin Transformer processes image patches independently via attention, but does not explicitly model semantic relationships between image regions. GAViT adds a graph reasoning layer on top of the transformer to capture inter-region relationships.

**指导教授**：Wang Lipo 教授，NTU EEE

**数据集**：
- 主数据集：NWPU-RESISC45（45类，每类700张，256×256）
  - Split: 70% train / 15% val / 15% test (random seed 42)
  - Data path: `datasets/NWPU-RESISC45_split/{train,val,test}/{class_name}/`
- 备选：AID（30类，10,000张，用于泛化性验证）

**技术路线（三阶段）**：
1. Patch Embedding：Swin Transformer 提取视觉 token（49 tokens, 768-dim）
2. Region Grouping：将 token 聚类为语义区域（k-means 或空间分组）
3. Dynamic Graph：以区域为节点，边编码空间/特征相似性；Graph Attention 层进行区域间关系推理

---

## 二、架构流程

```
Input Image (224×224 RGB)
    │
    ▼
┌─────────────────────────────┐
│  Swin Transformer Backbone  │  (Swin-T, pretrained, from timm)
│  4×4 patch embed → 4 stages │
│  56×56 → 28×28 → 14×14 → 7×7
└─────────────┬───────────────┘
              │ Output: 49 tokens, each 768-dim
              ▼
┌─────────────────────────────┐
│  Region Grouping Module     │  (k-means or spatial grid clustering)
│  49 tokens → K region nodes │  K is a hyperparameter (e.g., 9)
│  Clustering in feature space│
└─────────────┬───────────────┘
              │ Output: K region-level feature vectors
              ▼
┌─────────────────────────────┐
│  Dynamic Graph Construction │
│  - Nodes = region features  │
│  - Edges = cosine kNN similarity
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Graph Neural Network Layers│  (GAT, 1–2 layers)
│  Refine region features via │
│  neighbor aggregation       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Graph Pooling → Classifier │  (mean pool over nodes → FC → 45 classes)
└─────────────────────────────┘
```

---

## 三、代码结构（当前实际状态）

```
GAViT_Project/
├── CLAUDE.md                        ← 本文件
├── datasets/
│   └── NWPU-RESISC45_split/        ← train / val / test（7:1.5:1.5）
├── checkpoints/
│   └── best_swin.pth               ← Swin-T baseline 最优权重
├── baselines/
│   └── swin_baseline/
│       ├── split_nwpu.py           ← 数据集划分脚本（已完成）
│       ├── train_swin_baseline.py  ← Swin-T baseline 训练脚本（已完成）
│       └── test_swin.py            ← baseline 测试脚本（已完成）
├── models/
│   ├── __init__.py
│   ├── swin_backbone.py            ← Swin-T 特征提取（无分类头，输出 49 tokens）
│   ├── region_grouping.py          ← KMeansGrouping + SpatialGrouping
│   ├── graph_construction.py       ← cosine kNN 图构建，输出 PyG edge_index
│   ├── graph_reasoning.py          ← 多层 GAT（残差 + LayerNorm）
│   └── gavit.py                    ← 完整 GAViT 模型
├── train_gavit.py                  ← GAViT 训练脚本
├── test_gavit.py                   ← GAViT 测试脚本
├── utils.py                        ← set_seed, accuracy 等工具函数
├── results/
│   └── figures/                    ← 训练曲线、混淆矩阵、图可视化等
└── research_diary.md               ← 研究日记（每次完成功能或 debug 后更新）
```

---

## 四、关键技术参数

- **Framework:** PyTorch + timm (Swin-T) + PyTorch Geometric (GNN)
- **Backbone:** `swin_tiny_patch4_window7_224`, pretrained=True
- **Swin-T output:** 49 tokens (7×7 grid), 768-dim each
- **Training config:** batch_size=32, lr=3e-4, AdamW, CosineAnnealingLR, 30 epochs
- **GAViT defaults:** K=9 regions, kNN k=5, 2-layer GAT (4 heads, 256 hidden)
- **GPU:** NTU School of EEE GPU server (CUDA)
- **Image preprocessing:** Resize 224×224, normalize with ImageNet mean/std

---

## 五、当前实验结果（必须记录，勿删）

| 模型 | 数据集 | Val Acc | Test Acc | 备注 |
|------|--------|---------|----------|------|
| Swin-T Baseline | NWPU-RESISC45 | ~96% | ~96% | 30 epoch, AdamW, lr=3e-4, CosineAnnealing |

> **规则**：每次新实验完成后，将结果追加到此表格，注明超参数与实验条件。图像和混淆矩阵保存至 `results/figures/`。

---

## 六、待完成实验（按优先级）

- [x] **Baseline:** Swin-T only（~96%，已完成）
- [x] **GAViT 核心模块实现**（models/ 目录，已完成）
- [ ] **P1** Swin-T + Region Grouping only（无 GNN）← **当前任务**，先跑 spatial grouping（K=9），再跑 kmeans
- [ ] **P2** 训练 GAViT v1，与 baseline 对比（K=9, kNN, 2-layer GAT）
- [ ] **P2** 消融：K 的影响（4 vs 9 vs 16）
- [ ] **P2** 消融：边构建策略（cosine similarity vs spatial adjacency vs kNN）
- [ ] **P2** 消融：GNN 层数（1 vs 2 层，GAT vs GCN）
- [ ] **P3** 在 AID 数据集上验证泛化性

---

## 七、研究日记规范

**文件位置**：`research_diary.md`（项目根目录）

每次完成以下任意一项后，必须更新日记：
- 完成一个新功能模块
- 完成一次 debug
- 获得新的实验结果
- 重要的代码重构

**日记格式模板**：

```markdown
## YYYY-MM-DD — [简短标题]

**完成内容**：
- 具体做了什么

**实验结果**（如有）：
- 指标：xxx%
- 对比基线变化：+/- x%

**遇到的问题及解决方案**：
- 问题：...
- 解决：...

**下一步计划**：
- [ ] 待办事项1
- [ ] 待办事项2
```

---

## 八、必须保存的图表与数据

以下内容属于论文必需材料，每次生成后立即保存：

1. **训练曲线**：`results/figures/loss_acc_curve_{model_name}.png`
2. **混淆矩阵**：`results/figures/confusion_matrix_{model_name}.png`
3. **对比结果表**：更新 `results/comparison_table.csv`
4. **消融实验表**：`results/ablation_study.csv`
5. **图可视化**：`results/figures/graph_vis_{epoch}.png`（GAViT 训练后）
6. **注意力热图**：`results/figures/attention_map_{sample}.png`

> **规则**：文件名必须包含模型名称或日期，方便追踪版本。

---

## 九、本地 ↔ 服务器同步工作流

**代码同步通过 Git 完成，禁止手动上传文件。**

远程仓库：`https://github.com/PatrickYxz/GAViT_Project.git`

### 本地改完代码后（Windows）

```bash
git add 具体文件        # 不要用 git add -A，避免误提交敏感文件
git commit -m "[类型] 简短描述"
git push origin main
```

### 服务器端拉取并提交作业

```bash
cd ~/GAViT_Project
git pull
sbatch jobs/run_region_only.sh   # 或其他作业
```

### 服务器端首次克隆（仅一次）

```bash
cd ~
git clone https://github.com/PatrickYxz/GAViT_Project.git
```

### 监控作业

```bash
squeue                          # 查看作业状态
tail -f logs/作业名_JOBID.out   # 实时跟踪输出
scancel JOBID                   # 取消作业
```

---

## 十、GitHub commit 规范

**每次完成以下操作后，必须推送到 GitHub**：
- 完成新功能模块
- 成功运行新实验并记录结果
- 更新 research_diary.md
- 修复重要 bug

**commit 类型前缀**：
- `[feat]` — 新功能（如：新增 region grouping 模块）
- `[exp]` — 实验结果（如：GAViT v1 val acc 96.8%）
- `[fix]` — Bug 修复
- `[doc]` — 文档/日记更新
- `[refactor]` — 代码重构

**示例**：
```
[exp] Swin-T baseline — val acc 96.2%, test acc 95.9%
[feat] 新增 k-means region grouping 模块
[fix] 修复 graph_builder 中 edge weight 归一化错误
[doc] 更新 research_diary 2026-03-17
```

---

## 十、代码规范

- Python 3.8+，框架：PyTorch + timm + PyTorch Geometric
- 超参数统一在脚本顶部 `CONFIG` 区域定义，方便复现
- 每个新模块写完后附带简单的单元测试（`if __name__ == "__main__":` 块）
- 使用 `tqdm` 显示进度条
- Device 处理：`"cuda" if torch.cuda.is_available() else "cpu"`
- 随机种子固定为 42（通过 `utils.set_seed(42)`）
- 所有实验结果保留小数点后一位（如 96.2%，而非 96.1973%）
- 每 epoch 清晰打印 train loss / train acc / val acc
- 按 val accuracy 保存最优 checkpoint

---

## 十一、来自导师的注意事项

- 每两周向教授发送进度邮件，附图表结果（不需要发代码，只发结果表格/图像）
- 实验结果**绝对不可以造假**，所有结果须可复现
- 消融实验是必须的：至少比较 (i) Swin baseline (ii) +region grouping (iii) +graph reasoning
- 结果差异需要进行统计显著性分析（Wilcoxon test）
- 每次运行若结果有随机性，需运行 5–10 次取均值 ± 标准差
- Compare with published methods in tables

---

*最后更新：见 research_diary.md*
