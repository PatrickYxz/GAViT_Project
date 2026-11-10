# DFAGCN正式seed42：同划分对比范围复核

Current user summary relayed by parent task; no local raw numeric return was present at this reviewer read. Parent must bind the values to archived raw return and identity checks before marking the formal row audited.

DFAGCN：OA86.53174603174604%，macro-F1 86.45890874660422%，OA与21806/25200整数正确数一致（这里只做算术复核）。

| 对照 | OA% | macro-F1% | 对照−DFAGCN OA pp | 对照−DFAGCN F1 pp | 净多对 |
|---|---:|---:|---:|---:|---:|
| swin | 93.932540 | 93.909145 | +7.400794 | +7.450237 | 1865 |
| fixed | 94.103175 | 94.089933 | +7.571429 | +7.631025 | 1908 |
| soft | 93.869048 | 93.854987 | +7.337302 | +7.396078 | 1849 |

同manifest、同train/val/refit/test样本及信息隔离原则；不能称所有训练细节统一。DFAGCN的validation CE选轮与Swin系列validation OA选轮不同，骨干、权重、输入、增强、优化和硬件也不同。可报告完整方法单seed实测差值，不能作单图模块因果或稳定优越性声明。

原论文NWPU20% OA89.29%仅作文献参考：十次随机划分；本项目固定manifest训练seed42，原Table VII均值/SD单元格未直接核到。不能据两数差距称数值复现成功或失败。

## 候选汇报段落

本次新增了DFAGCN独立实现的NWPU-RESISC45 seed42正式结果：在固定20%训练池、80%测试划分上，测试OA为86.532%，macro-F1为86.459%。同一清单、同一训练seed的Swin-T、固定4×4 GAViT及软分配GAViT测试OA分别为93.933%、94.103%和93.869%；固定GAViT相对该DFAGCN实现高7.571个百分点。这里比较的是本项目披露配置下的完整方法：DFAGCN采用冻结torchvision VGG16、三支GCN与LIBLINEAR，预训练、预处理、优化和选轮规则与Swin系列不同，且本次使用RTX3080，既有seed42三模型使用RTX2080Ti。因此，这一结果支持同一划分上的单seed方法比较，不能归因于某个图模块，也不能称为原作者实现或原论文分数的数值复现。GAViT相对自身Swin基线的稳定增益仍未建立：固定版seed42仅提高0.171个百分点，seed43下降0.091个百分点。

## 便宜离线下一步

- Use existing test metrics/confusion matrix to independently recompute25200 samples,21806 correct,45 supports560 and macro-F1; retain full per-class accuracy/F1.
- Describe largest per-class deficits and largest confusion directions from existing outputs, without attributing causes from class names.
- If existing predictions.csv files can be read, align exact unique image paths, labels, class order and manifest; compute both-correct, corrections, introduced-errors and both-wrong against seed42 Swin/fixed/soft, with net differences1865/1908/1849 as consistency checks.
- Summarize saved extraction/calibration/refit/test times and per-stage memory separately, preserving GPU/setup identity; separate cached classifier timing from full-pipeline timing.
- Retain this recipe, negative/positive findings and exact provenance. Do not tune to the test or rerun solely to match paper89.29.

## 当前证据来源

- `.worktrees/sparse-hybrid-4n-top2/results/nwpu_two_seed_comparison.csv`
- `.worktrees/sparse-hybrid-4n-top2/results/nwpu_seed42_test_comparison.md`
- `.worktrees/sparse-hybrid-4n-top2/results/nwpu_seed42_closeout.md`
- `.worktrees/sparse-hybrid-4n-top2/results/nwpu_seed42_calibration_comparison.md`
- `.worktrees/sparse-hybrid-4n-top2/results/nwpu_two_seed_summary.md`
- `docs/external_baselines_20261027/protocol_and_acceptance.md`
- `docs/external_baselines_20261027/formal_3080_runner_20261030.md`
- `docs/external_baselines_20261027/pipeline_design_20261029.md`
- `docs/external_baselines_20261027/literature_comparison.md`
- `configs/dfagcn_independent_v1.json`
