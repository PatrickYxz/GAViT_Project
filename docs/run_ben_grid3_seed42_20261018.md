# BEN固定3×3 seed42补充实验

2026-10-20最终补充：已取回原训练summary，K9 best10/final30、val best79.1052%、实际RTX4090 UUID和资源已归档；两套受训模型的同图可视化、真实图边及预测分析已完成，见[正式图稿与分析](ben_gridpair42_visual_analysis_20261020.md)。以下保留准备、运行及分阶段回传过程，不再启动训练或捕获。

2026-10-18准备，用户已授权“那就补吧”，执行一个K9正式实验。2026-10-19收到PID3599及RGB扫描；2026-10-20已收到总exit_code=0及119825张/19类test指标，mAP71.07533208222321%、macro-F1 64.48830571218534%、micro-F1 76.68872316423352%。[新回传](../results/evidence/ben_grid3_seed42_metrics_20261020/user_return.txt)及[3×3/4×4对照](ben_grid3_grid4_seed42_20261020.md)保存；完整summary尚待回传，best/final、实际GPU UUID与资源/门禁记录仍待归档。

## 研究问题与范围

2026-10-20用户重申：本次补跑的主要用途是补齐**受训3×3与4×4模型的同图可视化对照**。需要在同一批BEN图像上展示区域分配、真实前向产生的空间边/特征边及分类预测，观察细化区域后哪些局部关系和预测发生变化。训练完成只是取得可视化所需权重，不能把收到性能指标当作这项可视化任务已经完成。

test mAP是性能对照的主要指标，macro-F1、micro-F1、19类AP、阈值0.5、资源和完整预测为辅助记录；这些结果为图像案例提供总体背景。独立的3×3／4×4消融表保留，不替换既有42/43/46/47三模型表，也不根据结果选择保留哪些seed。

只新增K9/seed42的30轮训练，复用历史K16/seed42作为参考，不重训Swin、软分配或4×4。既有权重与本地/用户回传库存未发现可复用的K9结果，继续离线测试或画几何覆盖图无法补出受训K9分数。

历史参考为cd39be9固定16区域模型，best13/final30，test打印mAP72.0%、macro-F1 66.0%、micro-F1 76.7%。其训练4090/测试3060，新K9在当前4090；历史软件/精度记录不齐，物理GPU也不同。因此此项是单seed、跨批次的历史参考比较，不能作为严格同环境因果结论或多seed稳定优势证据。旧mAP只有一位小数，显示72.00%不会增加精度。

## 固定协议

- 数据：原BEN-19 RGB、19类映射与237871/122342/119825划分；原split SHA、预处理、增强及ImageNet权重SHA不变。逐行检查标签、重复路径、全部RGB文件和原TIF读取路径；短测另取96train/64val，不覆盖正式划分。
- 模型：从完整提交cd39be91a013bcf98099808c451300f87d3898b7导出独立源码快照，attention pooling、token_feedback、GAT hidden256/4头/2层、dropout0.1保持。固定3×3区域，四邻接＋排除自身/空间邻居后的cosine top2，方向neighbor→query。K9为24+18=42条唯一有向pre-GAT边；K16为48+32=80。节点/边数随K变化，GAT内部自环不计入这些数值。
- 优化：seed42、epochs30、batch32、workers4、AdamW lr3e-4/wd1e-4、CosineAnnealingLR T_max30、BCEWithLogitsLoss；按val mAP严格改善选best，再完整test，不按test选epoch。
- 设备：一张RTX4090，记录UUID/driver/软件；FP32/noAMP、matmulTF32=false/cuDNN TF32=true、8线程。与已完成soft42的软件和精度检查一致，GPU UUID可不同。历史验证/测试pin_memory=false的主存修复用于新快照，训练loader保持原设置；此为IO差异，如实记录。
- 代码身份：原工作区不改。独立快照包含13个源码文件、原始/修改后SHA、基础Git提交及补丁SHA，Git provenance明确标为modified_frozen_snapshot/dirty=true。正式门禁校验整个快照，不能把它称作未修改的cd39be9。K16图张量要求与原版逐值完全一致，并在同一K16模型上换回原图函数核对logits差值≤1e-5；CUDA scatter归约的舍入差不要求逐位为零。

## 运行方式

本次已收到总退出0和正式测试指标。先读取已有summary补齐权重/训练元信息，再用已有两套权重完成同图可视化捕获；这两步属于不同完成条件。下面的安装命令仅保留为首次部署记录。

打开[安装并后台运行的单行命令](../results/evidence/ben_grid3_seed42_prepare_20261018/install_and_launch_command.txt)，整行复制到Featurize终端。它从服务器现有Git对象导出代码到新目录，不pull、不覆盖原项目，不安装依赖或联网下载。需要现有持久化权重、split、旧seed45数据门禁脚本、soft42 preflight报告，以及已恢复的实例本地BEN原始数据。

输出固定为`/home/featurize/work/ben_grid3_seed42_20261018_01`。安装或启动遇到已有目录/日志会停止并保留，不删除或改名绕过。后台流程顺序为：

1. 核验代码/预训练/划分/历史对照、4090身份、已有K9库存、全量RGB文件；执行真实K9/K16图张量及完整前向检查。
2. K9独立96/64一轮短测：forward、backward、首个optimizer step、validation、checkpoint、fresh-model reload误差≤1e-5，并捕获实际42边。
3. 门禁全部通过后自动进入K9完整30轮；每轮保存loss、val mAP/F1、学习率、耗时，best与last状态分开。
4. 验证选best后完整119825×19 test；保存三个总体指标、19类AP、预测、checkpoint绑定、资源和退出码。

任何一步异常即停止后续阶段；不会因为负面结果自动重跑。PID只表示分派；短测也会输出首批标记，只有`START train`之后出现`TRAIN_FIRST_BATCH_OK K=9 device=cuda`并核对GPU进程/显存，才确认正式训练首批启动。

查看进度（单行）：

```bash
tail -n 35 /home/featurize/work/ben_grid3_seed42_20261018_01/run.log
```

持续跟随日志：

```bash
tail -f /home/featurize/work/ben_grid3_seed42_20261018_01/run.log
```

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

结束后读取：

```bash
cat /home/featurize/work/ben_grid3_seed42_20261018_01/run.exit /home/featurize/work/ben_grid3_seed42_20261018_01/summary.json
```

完成需要preflight/smoke/train/test与总退出均0、summary complete、final30、best与history一致、完整test及同一checkpoint SHA。运行中或失败时不填性能表。原始日志/状态保留，返回摘要后再同步本地结果与研究日记。

## 训练后的可视化交付

2026-10-20：实际捕获工具及[单行CPU运行包](run_ben_gridpair42_visual_20261020.md)已准备，通过本地相关检查；服务器捕获待执行。它自动核对并打包已有summary，不需要先单独粘贴summary才能开始这一步。

先完成3×3／4×4，再继续既定的Swin／固定4×4／软分配三模型对照。3×3／4×4图稿按同一输入排列：原图、3×3区域覆盖与真实图边、4×4区域覆盖与真实图边，下方列真实标签及两模型对应类别概率（阈值0.5）。空间边与特征边分别显示，箭头采用实际neighbor→query方向；区域内汇总权重如需展示，须来自各自受训模型的真实前向。

第一批可复用已有几何图的4张归档验证输入97389/14680/28450/4439，全部保留，便于直接补上同图捕获。它们是历史诊断子集，需使用新K9/42和历史K16/42重新前向，旧seed44分数和图边不能冒充本次结果，也不能把这4例当作正式test收益率。后续若选收益/退化案例，须用配对预测与明确选样规则，保留未改善案例。

捕获使用同一输入张量、预处理、设备和精度，严格加载两套权重，记录checkpoint/source/input SHA；对比开启/关闭捕获的logits，确认观察钩子未改变输出。现有BEN反馈捕获工具硬编码K16/80边，需要适配K9/42边后才能用于本项；训练短测里的首图边检查也不能替代这批图像的捕获。当前仅完成几何图和性能回传，可视化真实前向、配对预测及新图稿仍待完成。固定区域边界由位置规则确定，受训后要补看的内容主要是区域汇总权重、图边和预测变化。

## 验证与成本

[准备记录](../results/evidence/ben_grid3_seed42_prepare_20261018/preparation_validation.json)区分本地标准库检查和待服务器执行的真实张量/CUDA检查。新增11项检查通过，另有43项来源/载荷/安装fixture核对；独立复核的数值容差问题已修复。完整unittest执行190项，因缺torch产生8项错误、缺pytest产生13项错误，全部名称及原因保存在[完整日志](../results/evidence/ben_grid3_seed42_prepare_20261018/full_unittest.log)与准备记录中，不能称全套测试通过。本地也没有timm/PyG，真实模型、CUDA与一轮短测尚未运行；服务器门禁失败时不会开始正式训练。

以上为2026-10-18准备时的本地验证范围；2026-10-20回传的总退出0表明队列报告成功走完各门禁和训练/测试，但实际门禁数值与设备详情仍需已有summary归档。未新增本地或远端推理来“复算”F1，当前只独立复算19类AP均值与差值；实际资源和费用仍未知。

按此前4090完整BEN任务约5h46m～6h22m估计，本次预留6～8小时用于准备、短测、训练和测试；以用户此前提供的1.67元/小时仅作条件预算，约10～14元，当前实例价格及实际费用以平台为准。首轮耗时出来后才能缩小估计范围，不把短测的96张吞吐直接外推。
