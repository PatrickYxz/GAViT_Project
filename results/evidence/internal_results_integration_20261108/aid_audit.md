# AID内部结果、效率与消融证据包

日期：2026-11-08。本次仅新增本审计JSON/MD，无模型执行、联网、SSH、Git、论文或旧表改写。

## 结果与证据层级

论文内部重复主表建议用原预定且共同协议的42/43/44。三次两模型共六阶段均报告complete/exit0；seed44是完整有效负向重复，不删除、不因下降重跑。42原始5000预测/30类/900混淆单元已本地复算；43/44为用户回传collector与逐类摘要，没有本地原始预测/完整混淆或checkpoint字节。

| Seed | Swin OA% | Fixed OA% | OA差pp | Swin macro-F1% | Fixed macro-F1% | 原件状态 |
|---|---:|---:|---:|---:|---:|---|
| 42 | 96.34 | 96.94 | +0.60 | 96.22088514124282 | 96.7811271561119 | 42原预测已回收 |
| 43 | 96.86 | 97.16 | +0.30 | 96.68420610630325 | 96.95587595593263 | 回传摘要 |
| 44 | 96.84 | 96.08 | -0.76 | 96.66111935641514 | 95.87462595609628 | 回传摘要 |
| 46 | 96.56 | 97.12 | +0.56 | 缺失 | 缺失 | 仅确认OA |
| 47 | 95.72 | 96.08 | +0.36 | 缺失 | 缺失 | 仅确认OA |

共同协议：4000train/1000val选择、5000pool fresh refit、5000test，split42/val4242。deb2b45/17源hash、RTX2080Ti、Swin-T、共享RGB224增强、AdamW3e-4/WD1e-4/batch32/30轮Cosine。refit轮数Swin/Fixed：42为21/20，43为25/29，44为25/21，T_max仍30。43/44来源归属仍是回传摘要，不升级为原文件字节独立核验。

## 标准库Decimal重算

均值±样本SD（n-1）；差值先同seed配对再汇总，打印前不舍入。JSON保留40位工作精度下的Decimal字符串；超出原分数精度的尾数不增加实测精度。

| 范围 | 指标 | n | Swin | Fixed | Fixed−Swin pp |
|---|---|---:|---:|---:|---:|
| 原预定42/43/44 | oa | 3 | 96.6800000000 ± 0.2946183973 | 96.7266666667 ± 0.5707305260 | 0.0466666667 ± 0.7145161533 |
| 原预定42/43/44 | macro_f1 | 3 | 96.5220702013 ± 0.2610892181 | 96.5372096894 ± 0.5804284525 | 0.0151394881 ± 0.7090697705 |
| 指定汇报42/43/46/47 | oa | 4 | 96.3700000000 ± 0.4829078587 | 96.8250000000 ± 0.5057996968 | 0.4550000000 ± 0.1473091986 |
| 指定子集仅42/43有F1 | macro_f1 | 2 | 96.4525456238 ± 0.3276173963 | 96.8685015560 ± 0.1235660614 | 0.4159559322 ± 0.2040513349 |
| 全历史42/43/44/46/47 | oa | 5 | 96.4640000000 ± 0.4680598252 | 96.6760000000 ± 0.5503453461 | 0.2120000000 ± 0.5581397674 |
| 仅历史46/47 | oa | 2 | 96.1400000000 ± 0.5939696962 | 96.6000000000 ± 0.7353910524 | 0.4600000000 ± 0.1414213562 |

原三seed OA差+0.60/+0.30/−0.76pp，两正一负；macro-F1配对均值约+0.01513949pp。近零差不等于统计等效，也未证明稳定提升。SD为同一冻结划分的训练重复；同5000图不是15000张独立图片。

指定42/43/46/47是观察到已有结果后指定的汇报子集，排除了负向44且混合46/47历史4090回传；可保留进度展示，不能替代预定三seed主证据或推稳定性。全5seed只宜作混合条件历史附录。46/47仅OA，F1、逐类/原始预测/混淆、原始run、checkpoint身份/UUID/软件、best/final和资源仍缺；不能借本次seed42原件填补。AID soft这些seed均未运行。

## 样本与错误证据

seed42重新验证5000唯一同序路径/真值、完整900混淆单元、refit/train_plus_val/test身份与clean deb2b45。Swin4817正确，Fixed4847；原预测配对双方都对4754、Fixed纠正93/新增63、都错90，净+30。43回传纠正62/新增47、净+15；44回传60/98、净−38。后两次计数不是本地原预测重算。

seed44四阶段106个epoch/loss/lr/val四元组和三seed两重点类混淆行已做历史核验：无缺轮，最早best和CosineT_max30一致；21轮refit是选定预算，不是中断。Commercial净+23/+3/−21、RailwayStation+2/+3/−10、Park+7/−1/−5，显示类别收益随seed变化。44的Commercial/Station净多错21/10；行边际差不能直接指认同图迁移、graph因果或过拟合根因。

## 同卡模型前向效率

RTX2080Ti，FP32，无AMP/TF32；预驻留GPU的ImageNet归一化合成RGB224。完整模型含Python构图与设备执行，排除读盘/decode/resize/normalize/H2D、加载、预热、argmax和指标，不是完整数据集吞吐。每配置独立进程，20预热+5组×50次perf_counter/CUDA同步；顺序Swinb1→GAViTb1→GAViTb32→Swinb32。SD只对5组均值，不是250单次、训练seed或置信区间。

| Model | Batch | Params | ms/批 ± 组均值SD | 图/s | allocated MiB | reserved MiB |
|---|---:|---:|---:|---:|---:|---:|
| swin | 1 | 27543960 | 14.891226000 ± 0.596856840 | 67.153637988 | 133.267578125 | 158.000000000 |
| gavit | 1 | 31375385 | 20.015683056 ± 0.167939418 | 49.960823081 | 147.755371094 | 180.000000000 |
| gavit | 32 | 31375385 | 66.884989000 ± 0.629308621 | 478.433210178 | 609.353027344 | 962.000000000 |
| swin | 32 | 27543960 | 56.648041896 ± 0.261797555 | 564.891546627 | 594.482421875 | 942.000000000 |

多3,831,425参数（13.9102184290%）；batch1延迟+34.4125934023%，batch32吞吐−15.3052983293%，allocated增加14.87060546875MiB（2.5014373717%）。batch32摊销Swin1.77025130925/Fixed2.09015590625ms/图不替代单请求延迟。allocated/reserved不相加；warmup后模型+输入+输出/中间量峰值，排除加载/预检，不是NVML整卡。峰值每配置单测，无误差条。总wall88.260854984秒含其他步骤，不作吞吐分母。

证据为回传collector完整摘要，四阶段exit0且权重SHA匹配42原test。当前本地无250原耗时/温度/时钟/功耗或完整阶段run.log，median/P95仅能引用原报告，不能重算或排除顺序/温漂。首个较慢Swinb1组未删除。没有各模块耗时或跨设备速度/实际费用证据。

## 已测消融与空计划

| 对象 | 状态与边界 |
|---|---|
| AID full fixed4x4 graph vs Swin | completed：Main42/43/44 OA/F1;46/47 OA-only supplement. Tests full added pathway, not individual module contributions. |
| AID soft assignment | not_run：No42/43/44/46/47 soft outcome; do not create a third row. |
| AID region-only/no-GNN | no_current_formal_evidence：Historical NWPU region-only validation does not fill this AID cell. |
| AID edge recipe/GAT type/depth | no_current_formal_evidence：80-edge forward checks are architecture identity, not a trained edge ablation. |
| AID K4/K9/K16 | no_current_formal_evidence：No current AID trained K4/K9 result; early K9/K16 changed grouping and integration simultaneously. |
| AID feedback scaling/removal | no_current_formal_evidence：Do not transfer NWPU val or BEN selected-case interventions. |
| Legacy NWPU region-only/graph chain | historical_validation_report：2026-03-22 diary: approximately96.0 baseline,96.2 K9 region-only,96.5 spatial+GAT,96.0 kmeans+GAT. Old70/15/15/architecture; not modern20%pool test nor AID. Logs/checkpoints not freshly audited. |
| Legacy NWPU edges | historical_validation_report：2026-03-26 diary: spatial96.2,cosineKNN96.1,hybrid96.0;3x3/8-neighbour historical recipe, not current4n+top2 component result. |
| Root ablation_study.csv | mostly_empty_old_plan：Nonbaseline metric fields are blank. Root comparison_table.csv has old NWPU validation values, not an AID formal ablation. |
| NWPU feedback alpha | completed_reported_diagnostic：1260 calibration-val;fixed/soft alpha0,.25,.5,.75,1;alpha<1 all lowerOA/F1,alpha0 net-9 each. Jointly trained weights; not independently trained no-GNN/Swin and not AID/test. |
| BEN K9/K16 | completed_cross_batch_reference：119825test seed42;K9mAP71.08 vs historicalK16~72.00 and4same-image captures. Historical hardware/software/rounding differences; not AID or a clean multiseed K-only ablation. |

根ablation_study.csv非baseline结果列为空；comparison_table.csv是旧NWPU验证值。旧3月no-GNN/edge日记确有验证数字，但原日志/权重未在本审计重新验收，且不是AID或现代20%pool refit test。不能沿用“证明图有效”之类旧因果措辞。NWPU反馈alpha和BEN K9/K16是真实既有诊断/参考，但保留其数据集、val/case、跨批次硬件/精度限制，不用填AID消融。

## 可直接写入英文论文的段落

### Repeated AID evaluation

On the prespecified AID training seeds 42, 43 and 44, Swin-T achieves 96.6800 ± 0.2946% OA and fixed-grid GAViT achieves 96.7267 ± 0.5707%, where the dispersions are sample standard deviations across three training repetitions on one frozen partition. The paired OA difference is +0.0467 ± 0.7145 percentage points. Macro-F1 is 96.5221 ± 0.2611% for Swin-T and 96.5372 ± 0.5804% for GAViT, with a paired difference of +0.0151 ± 0.7091 points. Gains of +0.60 and +0.30 points in seeds 42 and 43 are offset by a −0.76-point result in seed 44. This valid negative run is retained. These observations do not demonstrate a stable aggregate benefit, and the near-zero difference is not an equivalence-test result.

### Reporting subset and provenance

A later progress-report subset contains seeds 42, 43, 46 and 47. Its four positive OA differences are reported separately because the subset was selected after existing results were available and excludes the valid negative seed 44. Seeds 46 and 47 have only user-confirmed historical OA values: their original evaluation records and macro-F1 scores are unavailable. This subset therefore does not replace the prespecified three-seed comparison or provide a four-seed macro-F1 estimate. Original seed 42 test predictions have since been recovered and independently reconstructed locally; the seed 43 and 44 metrics retain the provenance of returned collector summaries.

### Computational cost

Using the fixed seed 42 refit checkpoints, a same-device FP32 benchmark on an RTX 2080 Ti measures complete model forwards on GPU-resident synthetic 224×224 inputs. After 20 warm-up passes, five groups of 50 synchronised forwards give mean batch1 latencies of 14.8912 ms for Swin-T and 20.0157 ms for fixed-grid GAViT. Batch32 throughput decreases from 564.8915 to 478.4332 images/s. The parameter count increases from 27,543,960 to 31,375,385 (+13.91%), while batch32 peak allocated CUDA memory increases from 594.4824 to 609.3530 MiB (+2.50%). These measurements exclude image reading, preprocessing and transfer, and do not isolate the cost of grouping, graph attention or feedback. Together with the small, seed-dependent accuracy differences, the current evidence does not establish stable accuracy benefits sufficient to justify the measured additional computation.

### Component scope

The AID experiment compares the complete fixed regional graph pathway with the shared Swin-T baseline; it does not separately identify the contribution of region pooling, graph edges, graph attention or token feedback. No formal AID region-only, edge-type, graph-depth, region-count or feedback-scaling ablation is reported. Available feedback perturbations and K9/K16 reference comparisons concern other datasets and retain their own validation, checkpoint and execution boundaries.

## 精确来源索引

同目录aid_audit.json提供40位Decimal统计、每seed可用性、原预测/效率重算结果与下列源文件SHA。

- `research_diary.md` — `f442faf0f791a3bd1c03c6addc0d8baabccf78eefc1da0b2454652797f5856c5`
- `docs/supervisor_progress_report_20261012.md` — `335ada4d942a8e84be8f85aebe3f1b511cad9eff34fb84eb7c266bcc22153c7a`
- `thesis/ntu_gavit/README.md` — `f1f75da89efc8816f85369e5ce54993603fb267e098c91b334be5f5a39474cdf`
- `results/seed46_seed47_reconstruction_20261104.json` — `6bec50cec48c6f69e22100bf7cac441728645bcb0694d6213c548c83ce3ea12d`
- `results/supervisor_progress_data_20261012.json` — `7a825fdc1a26de05d1e181ec5d0767b3c2101859fbdb8cc968058cab9c842bc2`
- `results/aid_seed42_three_model_paired_analysis_20261106.json` — `486f2da4412b03c97e8ffe332929299f2a8d6d0ea646c499fa8e3fc3b5f47877`
- `results/comparison_table.csv` — `54efc5572b6271e9019bd5546951ae84f9f06bf23fc7ff58021036b419de562f`
- `results/ablation_study.csv` — `33182d7a7802e11c0580e66b7736c1d679af45ba4399b47369b582d8e9bd2011`
- `docs/ben_grid3_grid4_seed42_20261020.md` — `c2ea90cc294be1528497f0eb24e1460d63872e68b0cc09433382f9352767033c`
- `.worktrees/sparse-hybrid-4n-top2/results/aid_multiseed_summary.json` — `2418d5841daa98e3e77f51afe37e295969d22bd8f2f950e2746ee5e32510623a`
- `.worktrees/sparse-hybrid-4n-top2/results/aid_multiseed_summary.md` — `2eb6bce94e5d11cdf76ad303c98f2ad6117a419ca378dec20d0d5f1febc1ae43`
- `.worktrees/sparse-hybrid-4n-top2/results/aid_seed42_comparison.md` — `7c1a8bd54c5f2acffad5c364b02d85b8d3eceb6bb3f1631e16fcbcb4a8ed6ae1`
- `.worktrees/sparse-hybrid-4n-top2/results/aid_seed43_comparison.md` — `27b32ab56e1f0eff3427ee0de5ad319993063c7f635e874573b63e46ba9f7d93`
- `.worktrees/sparse-hybrid-4n-top2/results/aid_seed44_comparison.md` — `8e74bb5cc8158bde2e86216834323aacaef9e6c836ed206ba244ff014ee29b69`
- `.worktrees/sparse-hybrid-4n-top2/results/aid_seed44_curve_confusion_analysis.md` — `b122f7380e0a2638e6bfd88ae2cfad47b0e09a00b799eee3951f3b00acb618d1`
- `.worktrees/sparse-hybrid-4n-top2/results/aid_multiseed_class_stability.md` — `d1dad560744c8005f2ccd7963f333dda819a1da4eb377d128c6614f70f1fa216`
- `.worktrees/sparse-hybrid-4n-top2/results/aid_efficiency_summary.json` — `c600f8776b37716aa83f4aa49a11c7775cb373c166ece06f069c556e33a057c2`
- `.worktrees/sparse-hybrid-4n-top2/results/aid_efficiency_summary.md` — `e81a2bd3645878e1f9dcb2b27bdd1807e9b350f2559d04209d9024c528205b6d`
- `.worktrees/sparse-hybrid-4n-top2/results/nwpu_feedback_sensitivity_seed42.md` — `540ceafa5adfd6270c440816c1a596e8f73b16e7ed0236adfc6a369d8e5741b0`
- `.worktrees/sparse-hybrid-4n-top2/results/evidence/aid_efficiency_20260922_01/summary.json` — `66adec56144c9b6899a0e925988dbbeaa212fc02cc9f621ffe7599f0ff6de42a`
- `.worktrees/sparse-hybrid-4n-top2/results/evidence/aid_efficiency_20260922_01/provenance.json` — `b49ffb2377cbd3f47151ba99af5b41c20ce31160f63d8ceb8b576ea1448b7791`
- `.worktrees/sparse-hybrid-4n-top2/results/evidence/aid_seed43_batch_20260921/summary.json` — `7052e1c9be066691ebee48617899a15c69068edb33ffd0149385227c7dbe1716`
- `.worktrees/sparse-hybrid-4n-top2/results/evidence/aid_seed43_batch_20260921/provenance.json` — `68be5f5ae7afb88309db0662330ed73221cf100a9ee4cee3d35472ae3e112f2e`
- `.worktrees/sparse-hybrid-4n-top2/results/evidence/aid_seed44_batch_20260922/summary.json` — `6197932e8d3a8152b582b3566f7e6689b0eb3461c2581f3bed5b0085a02b6e0f`
- `.worktrees/sparse-hybrid-4n-top2/results/evidence/aid_seed44_batch_20260922/provenance.json` — `c7056da884de9b79c2887c126bb7758f1951a9f68621022c7574aced08b49935`
- `results/evidence/external_results_integration_20261106/aid_core_raw/swin/predictions.csv` — `54ba93da7cc0a3e4b4a5591df359b596224cf64957670e53f20e6da80e2b6fd3`
- `results/evidence/external_results_integration_20261106/aid_core_raw/swin/per_class.csv` — `f3639baf424a944171025f060d2e667c7da217df85c4ab9dc185e1bf1b6a07e7`
- `results/evidence/external_results_integration_20261106/aid_core_raw/swin/metrics.json` — `67267884cc7a22db850ac77432781410c5a6fd43f2c2bfa193e183413b8fe309`
- `results/evidence/external_results_integration_20261106/aid_core_raw/swin/run.json` — `6a25d3f18979731de6711f8ebb2f4287228b215c7d3e1789c23b9556acf006a2`
- `results/evidence/external_results_integration_20261106/aid_core_raw/swin/run.exit` — `bde294368bfed77c2cddf8cec271d398aee9cdbab3b26e1059281bd33adb0120`
- `results/evidence/external_results_integration_20261106/aid_core_raw/gavit/predictions.csv` — `2923a519ffe76a88f6a24b574bba23f5e4a58a6435aabf23195a24a140ee0fe0`
- `results/evidence/external_results_integration_20261106/aid_core_raw/gavit/per_class.csv` — `5f28f5e0e4f1a45542ca6e73640c427acb54fb5cd6ae5b3283b7990ef5c726cc`
- `results/evidence/external_results_integration_20261106/aid_core_raw/gavit/metrics.json` — `26144e735387ca95a31c06847987b0bc8b05a81a99c3cd6c5129d91f0541449a`
- `results/evidence/external_results_integration_20261106/aid_core_raw/gavit/run.json` — `614f6beacf735954934e0ab5de149a21769a58ef077285d8c80a53697c206ccd`
- `results/evidence/external_results_integration_20261106/aid_core_raw/gavit/run.exit` — `bde294368bfed77c2cddf8cec271d398aee9cdbab3b26e1059281bd33adb0120`
