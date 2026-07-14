# GAViT 项目协作规则

## 项目目标

- 本项目是遥感场景分类论文项目，核心问题是：相比 Swin baseline，区域分组和图推理能否带来稳定、可解释且值得额外复杂度的提升。
- NWPU-RESISC45 用于单标签场景分类验证；BigEarthNet-19 用于多标签场景分类验证。
- 不以单次最高指标为唯一目标。实验必须能够支撑模型对比、消融、效率分析或失败原因讨论。

## 开始工作前

- 先阅读 `research_diary.md` 的最新条目，再检查 `results/comparison_table.csv`、`results/ablation_study.csv` 和相关日志。
- 先确认现有 checkpoint、数据划分和日志是否已经回答问题，不重复已完成或已证明无效的实验。
- 涉及 Featurize 时先阅读 `docs/featurize_runbook.md`。
- 只把当前代码、当前日志和新验证结果当作现状依据；旧日记中的计划不代表已完成。

## 节省训练成本

- 不为了“试试看”启动完整训练。每次完整训练必须对应一个明确的论文问题。
- 训练前写清：研究假设、相比已有实验改变的主要变量、结果将进入的表格/图片/结论，以及是否存在更便宜的验证方法。
- 优先使用已有 checkpoint 做测试、可视化、per-class 分析和离线统计；这些能够回答的问题不重新训练。
- 先用代码检查、小样本或单 epoch smoke test 排除路径、标签、显存、依赖和模型前向问题。
- 记录吞吐、每轮耗时和预计总时长，用于比较显卡与配置成本。

## 实验设计规则

- 一次实验只改变一个主要变量，保证消融结果可以解释。
- 候选配置先使用同一个固定 seed 训练一次；明显无提升或不能补充论文结论的配置不重复。
- 最终只对 Swin baseline 和选定的最佳 GAViT 各运行三个 seeds，报告均值和标准差。
- 差距很小时先检查日志、方差和评估流程，再决定是否值得追加训练。
- 负面结果同样写入 `research_diary.md`，防止未来重复付费验证。
- 不因结果“不好看”而无理由重跑，不在训练后临时改变主要指标或挑选规则。

## 训练前检查

- 数据文件存在，train/val/test 数量与已记录划分一致，标签维度和类型正确。
- 模型能在目标设备完成前向传播，预训练权重从预期来源加载。
- 小规模训练能完成 forward、backward、validation 和 checkpoint 保存。
- smoke checkpoint 使用独立文件名，不覆盖正式实验 checkpoint。
- 正式命令、Git commit、seed、输出日志和 checkpoint 名称在启动前确定。
- 后台训练启动后同时检查进程、GPU 显存和日志中的首个训练 batch，不能只看到 PID 就判断成功。

## 论文实验记录

每次正式实验都在 `research_diary.md` 记录：

- 实验身份：日期、实验名、Git commit、完整命令和 seed。
- 数据：数据集版本、划分数量、预处理和数据增强。
- 模型：架构、参数量、预训练来源，以及 grouping、region 数量、edge type、GAT 和 integration 配置。
- 优化：epochs、batch size、optimizer、learning rate、weight decay、scheduler、loss 和分类阈值。
- 资源：GPU 型号、峰值显存、吞吐、每轮耗时、总时长和可获得时的实际费用。
- 输出：最佳 epoch、最终 epoch、训练/验证/测试指标、日志路径、checkpoint 路径和训练曲线数据。
- 解释：相对 baseline 的变化、假设是否成立、异常现象，以及对应的论文表格、图片或讨论段落。

未经验证的指标必须明确标注“待测试”或“待确认”，不能根据日志片段推测。

## 指标要求

- NWPU 单标签分类：Accuracy、macro-F1、per-class accuracy 和混淆矩阵。
- BigEarthNet 多标签分类：mAP 为主要指标，同时记录 macro-F1、micro-F1、per-class AP 和分类阈值。
- 最终核心对比报告三个 seeds 的均值与标准差，并同时报告参数量、推理速度和显存开销。
- 保存最佳验证 checkpoint，同时记录最佳轮和最终轮，避免隐藏后期过拟合。

## 代码与 Featurize 分工

- 本地仓库负责正式代码修改、测试、文档和 Git 历史；Featurize 负责 GPU 训练和评估。
- 代码、日志、checkpoint、metadata、split CSV 和预训练权重放在 Featurize 的持久化 `work` 目录；大型原始数据放实例本地 `data` 目录。
- 避免在 Featurize 直接修改正式训练代码。紧急 hot patch 必须记录，并尽快同步回本地仓库。
- 不覆盖用户未提交的本地或服务器修改，不使用破坏性 Git 命令清理工作区。

## 完成标准

- 代码改动经过与风险相称的测试，文档与实际命令一致。
- 训练任务只有在日志显示模型、设备和 batch 正常运行后才算启动成功。
- 实验只有在日志、最佳 checkpoint、配置和论文所需指标均已保存后才算完成。
- 完成实验后立即更新 `research_diary.md` 和对应结果表；不要依赖聊天记录保存关键事实。
