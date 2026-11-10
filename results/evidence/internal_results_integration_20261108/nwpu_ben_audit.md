# NWPU与BigEarthNet内部结果证据审计

审计日期：2026-11-08。仅本地只读资料核对及标准库算术；33项一致性检查通过。未运行模型、训练、GPU、SSH或联网，未改论文/旧CSV。

## 证据分层与论文表结构

- A：已归档机器生成的用户回传summary/log并有身份/退出核对；不等于本地下载全部checkpoint/预测后重新评估。
- H：历史打印控制结果，通常只有一位小数，训练终止/身份差异必须随表。
- R：用户确认的46/47历史主指标，原平台和记录已丢失。F1、AP、完整配置、选轮、UUID、checkpoint不可补造。
- N：未运行/没有完成结果。

建议正文先列A/H可归档身份的逐seed结果，R另列历史回传表或带明确R列的补充表。可同时保留用户指定42/43/46/47汇报子集，但不能称完整预设seed或统一条件重复；BEN44/45必须在全历史表保留。

## NWPU：现行冻结协议，45类

manifest `ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`；5040/1260校准30轮，选最早最佳OA预算，从预训练fresh refit6300，test25200。42设备2080Ti，43为4090，故不是纯seed方差。46/47相同配置/数量仅为用户确认，没有原件SHA再核验。

| seed | 模型 | OA % | macro-F1 % | 正确数 | refit轮 | 证据 |
|---|---|---:|---:|---:|---:|---|
| 42 | swin | 93.93253968 | 93.90914547 | 23671 | 29 | A |
| 42 | fixed | 94.1031746 | 94.08993346 | 23714 | 29 | A |
| 42 | soft | 93.86904762 | 93.85498722 | 23655 | 28 | A |
| 43 | swin | 93.85714286 | 93.83452989 | 23652 | 23 | A |
| 43 | fixed | 93.76587302 | 93.74351253 | 23629 | 26 | A |
| 43 | soft | 93.62698413 | 93.58782117 | 23594 | 29 | A |
| 46 | swin | 93.76 | — | — | — | R |
| 46 | fixed | 93.96 | — | — | — | R |
| 46 | soft | 93.54 | — | — | — | R |
| 47 | swin | 93.68 | — | — | — | R |
| 47 | fixed | 94.19 | — | — | — | R |
| 47 | soft | 93.55 | — | — | — | R |

seed43固定−Swin OA为−0.09126984pp，soft−Swin为−0.23015873pp。全部现有42/43/46/47中soft均低于Swin及固定版。不能用seed42+0.1706pp掩去43负结果，也不外推为所有可学习分组均无效。

seed42服务器配对摘要已归档：fixed相对Swin纠正586/新增543（净+43）；soft相对Swin570/586（净−16）；soft相对fixed538/597（净−59）。这是不同训练模型的正确性配对，不是反馈/分组单因素干预。原完整25200条CSV未在本地，不能声称本地从全部原始向量复算。

早期K9约96.5%验证/95.9%日记test、K16 95.8%(4526/4725)的split、pooling和feedback不同；只做历史方法演进，不进现行25200表，不能当纯K消融。旧日记中“持平/天然上限/证明图有效”等解释不是当前科学结论。

## BigEarthNet-19：逐seed全部历史

RGB、19标签；237871/122342/119825 train/val/test。按最佳验证mAP checkpoint测试，不做NWPU/AID式train+val fresh refit。F1阈值0.5；AP/mAP是排序指标，不以0.5阈值计算。H行来源只打印一位小数；展示8位不意味着更多原始精度，JSON保留precision标签。

| seed | 模型 | mAP % | macro-F1 % | micro-F1 % | best/final轮 | 证据 |
|---|---|---:|---:|---:|---|---|
| 42 | swin | 70.9 | 64.2 | 76.6 | —/30 | H |
| 42 | fixed | 72 | 66 | 76.7 | 13/30 | H |
| 42 | soft | 70.17457057 | 64.04828509 | 76.79995865 | 10/30 | A |
| 43 | swin | 71.1 | 64.1 | 76 | 6/30 | H |
| 43 | fixed | 71.7 | 65.7 | 76.8 | 8/21 | H |
| 43 | soft | 72.86757085 | 66.67389162 | 76.90597118 | 8/30 | A |
| 44 | swin | 71.4 | 65.7 | 76.7 | 9/— | H |
| 44 | fixed | 70.2 | 65.2 | 77.1 | 17/26 | H |
| 44 | soft | — | — | — | —/— | N |
| 45 | swin | 71.67692954 | 65.08421202 | 76.73226436 | 6/30 | A |
| 45 | fixed | 70.25823624 | 64.29547698 | 76.70245685 | 10/30 | A |
| 45 | soft | 70.83691876 | 64.86482284 | 76.38335431 | 11/30 | A |
| 46 | swin | 71 | — | — | —/— | R |
| 46 | fixed | 71.8 | — | — | —/— | R |
| 46 | soft | 71.4 | — | — | —/— | R |
| 47 | swin | 70.6 | — | — | —/— | R |
| 47 | fixed | 71.1 | — | — | —/— | R |
| 47 | soft | 70.3 | — | — | —/— | R |

重要身份：

- fixed43只保存到21轮，best8；43 Swin和soft为30轮。fixed训练4090/test3060，新soft43当前4090/ed480e6，与旧bb9f1cf控制不是完全同条件。
- fixed44测试70.2/65.2/77.1保持；后期metadata best17/78.5171与旧日记/根CSV best19/77.9冲突。Swin44至少观察到27轮，最终train.exit缺失；不能补写30轮或成功退出。
- 45三个模型full30；Swin/fixed deb2b45，soft ed480e6且不同物理2080Ti UUID，仍保留跨批次限制。
- 新soft42：10-17完整30轮/test119825，best10，best验证78.7500为metadata四位小数，末轮验证75.70723466，test70.1745705651681/64.04828509185678/76.7999586513447。scheduler修复/worker SHA保留，不能拿早期smoke指标代替。
- soft43：旧preflight的training_started=false确实不能升级为训练完成；但后续10-12原始回传及资源补录已给出full30/test119825/三exit0/best8。原回传SHA806d5c...cb4580与validation一致，故现在应列正式完成72.86757085064232，而非仍列“待训练”。
- soft44无完成结果；46/47无F1、19类AP或原始run，全部留空。

## 统计范围与负面结果

- nwpu_archived_42_43_OA：swin 93.89484127±0.05331361 (n=2,seeds=42,43)；fixed 93.93452381±0.23850824 (n=2,seeds=42,43)；soft 93.74801587±0.17116474 (n=2,seeds=42,43)。
- nwpu_all_42_43_46_47_OA_mixed：swin 93.80742063±0.11047143 (n=4,seeds=42,43,46,47)；fixed 94.0047619±0.18535528 (n=4,seeds=42,43,46,47)；soft 93.64650794±0.15336537 (n=4,seeds=42,43,46,47)。
- ben_predefined_historical42_43_44_45_mAP：swin 71.26923238±0.34072926 (n=4,seeds=42,43,44,45)；fixed 71.03955906±0.94409636 (n=4,seeds=42,43,44,45)；soft 71.29302006±1.40324051 (n=3,seeds=42,43,45)。
- ben_three_model_common_archived42_43_45_mAP：swin 71.22564318±0.40341608 (n=3,seeds=42,43,45)；fixed 71.31941208±0.93116627 (n=3,seeds=42,43,45)；soft 71.29302006±1.40324051 (n=3,seeds=42,43,45)。
- ben_all_controls42_43_44_45_46_47_mAP_mixed：swin 71.11282159±0.37996351 (n=6,seeds=42,43,44,45,46,47)；fixed 71.17637271±0.79291481 (n=6,seeds=42,43,44,45,46,47)；soft 71.11581203±1.09301052 (n=5,seeds=42,43,45,46,47)。
- ben_common42_43_45_46_47_mAP_mixed：swin 71.05538591±0.39461868 (n=5,seeds=42,43,45,46,47)；fixed 71.37164725±0.70703697 (n=5,seeds=42,43,45,46,47)；soft 71.11581203±1.09301052 (n=5,seeds=42,43,45,46,47)。
- ben_user_selected42_43_46_47_mAP_mixed：swin 70.9±0.21602469 (n=4,seeds=42,43,46,47)；fixed 71.65±0.38729833 (n=4,seeds=42,43,46,47)；soft 71.18553535±1.24919456 (n=4,seeds=42,43,46,47)。

这些均为描述性统计，SD使用n−1。BEN完整历史42/43/44/45的fixed−Swin平均mAP为−0.22967332pp（2正2负），加46/47后的六seed约+0.06355112pp（4正2负），不能只选正seed。指定42/43/46/47的+0.75pp是事后汇报子集且混合R证据，不是稳定收益证明。soft共同归档42/43/45在Swin上有一正两负；所有共同五seed42/43/45/46/47与六seedSwin均值不能直接相减。

F1汇总必须自带seed：NWPU42/43（三模型n2）；BEN Swin/fixed42/43/44/45(n4)，soft42/43/45(n3)。绝不填入46/47 F1再称四seed完整指标。分阶段时间含验证/IO/工件检查，RSS常不含workers；不同卡/训练终止不作同条件速度或成本结论。

## 19类AP与图诊断可以证明什么

BEN是19类AP，不能混用AID30类的per-class OA/F1。42/43均有Swin/fixed打印AP与soft精确AP，保留全部19类；AP只是类别排序表现，不是类别显著性或语义区域质量。下面区分打印差值符号与考虑舍入后的可靠方向：

| seed / candidate→Swin | 打印正/负/零 | 舍入范围内升/降/未定 |
|---|---|---|
| 42_fixed | 13/5/1 | 12/5/2 |
| 42_soft | 10/9/0 | 10/8/1 |
| 43_fixed | 10/8/1 | 10/4/5 |
| 43_soft | 11/8/0 | 11/7/1 |

精确AP与一位小数Swin比较的差约±0.05pp；两边均打印一位小数约±0.10pp。均值19个已舍入AP不能升级为重新计算完整119825张概率排序mAP。soft42海岸湿地约−19.64pp、沙滩−4.43pp、草地+6.24pp；soft43对应约+9.81/+12.47/−1.32pp，方向反转应保留，不能写跨seed一致改善。

受训K9/K16（seed42）：K9 test mAP71.07533208、macro-F1 64.48830571、micro-F1 76.68872316；历史K16为72.0/66.0/76.7（1位小数）。约+0.924668/+1.511694/+0.011277pp，其中micro-F1差在旧舍入误差内。K9 best10/final30、K16 best13/final30，但训练物理卡/软件/精度批次不同，不是隔离K的公平因果消融。

同四张预定val row97389/14680/28450/4439共76项判断：63双方对、9双方错、3 K16-only纠正、1 K9-only纠正（即K16新增漏判）。8次真实前向捕获、37文件hash和488条pre-GAT边已有返回审计；K9每图24+18=42边，K16 48+32=80边。图边不等于GAT注意力，颜色是区域ID；固定分区没有学习地物边界，区域内汇总权重高集中也不是语义质量分数。四例不估计总体获益率。

另一个BEN seed44反馈zero诊断不能混同K9/K16：同一已训练权重将反馈投影输出置零，对24张固定诊断图中全部4个阈值翻转做图，normal反馈2次纠正误报、2次新增漏报。可描述整支反馈对这些案例的影响，不能当重新训练的无图baseline、总体mAP收益或某条边因果贡献。

NWPU90例fixed/soft图册用seed42校准best权重：82双方对、3 soft-only对、0 fixed-only对、5双方错，39.09%token主归属改变。既不是Swin对照，也不是25200张refit test总体；与正式soft净−59并不矛盾。

## 优先引用文件

- `research_diary.md`
- `docs/supervisor_progress_report_20261012.md`
- `results/multiseed_test_results.csv`
- `results/seed46_seed47_reconstruction_20261104.json`
- `results/ben_four_seed_three_model_20261017.json`
- `results/ben_seed42_seed43_vs_swin_20261013.json`
- `results/ben_grid3_grid4_seed42_20261020.json`
- `results/evidence/bigearth_soft42_complete_20261017/summary.json`
- `results/evidence/bigearth_soft42_complete_20261017/validation.json`
- `.worktrees/sparse-hybrid-4n-top2/results/nwpu_two_seed_comparison.csv`
- `.worktrees/sparse-hybrid-4n-top2/results/nwpu_seed43_test_summary.json`
- `.worktrees/sparse-hybrid-4n-top2/results/nwpu_seed42_closeout.md`
- `.worktrees/sparse-hybrid-4n-top2/results/evidence/nwpu_refit_test42_20260929/test_pairs_server_summary.json`
- `.worktrees/sparse-hybrid-4n-top2/results/bigearth_comparison.csv`
- `.worktrees/sparse-hybrid-4n-top2/results/bigearth_four_seed_summary.json`
- `.worktrees/sparse-hybrid-4n-top2/results/bigearth_soft43_test_summary.json`
- `.worktrees/sparse-hybrid-4n-top2/results/evidence/bigearth_soft43_complete_20261012/reported_output.txt`
- `.worktrees/sparse-hybrid-4n-top2/results/evidence/bigearth_soft43_complete_20261012/reported_summary_output.txt`
- `.worktrees/sparse-hybrid-4n-top2/results/evidence/bigearth_soft43_complete_20261012/resource_validation.json`
- `.worktrees/sparse-hybrid-4n-top2/results/evidence/seed44_reconciliation_20260914.json`
- `docs/ben_gridpair42_visual_analysis_20261020.md`
- `.worktrees/sparse-hybrid-4n-top2/results/seed44_feedback_figure_caption.md`
