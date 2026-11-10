# Seed44 固定案例图反馈诊断

2026-09-19：original运行 `pair44_feedback_k_7ehbne` 已完成，RTX2080 Ti、24张、exit0，最大缓存差约0.000001，反馈纠正/破坏为2/2。输入核对与数值干预结束，当前不需要重跑下方命令；下一步检查已保存的原图案例与实际图边，见[最新原图结果](../results/seed44_original_feedback_diagnosis.md)。

2026-09-18最新：服务器 `pair44_feedback_qnjm_rb7` 完整日志已回传，RTX3060/gallery/24张图、24例内部数值和拓扑检查通过、exit0。原缓存最大概率差0.889688，24项关注分数中4项翻转；该gallery运行只作工程与JPEG案例诊断，不能代替原RGB输入分析。见[最新结果与原图恢复步骤](../results/seed44_feedback_diagnosis.md)。以下gallery启动说明保留供追溯，当前不重复运行，应先确认原图/压缩包状态。

## 这次要回答什么

在同一张固定验证案例、同一份 GAViT44 权重上，将图反馈设为零，查看预测分数和阈值0.5下的对错是否改变。实际区域与图边从正常前向捕获，用于论文案例分析。这能检查图反馈在具体案例上的作用，不能证明测试集AP下降的原因，也不能替代重新训练的无GNN消融或Swin baseline。

用户已重启实例，本次默认使用完整验证诊断中保存在 `work` 的 JPEG 案例图，避免解压原始数据。JPEG经过压缩，与原RGB评估输入不同；正常前向分数也可能变化，因此缓存分数、JPEG正常分数、屏蔽反馈分数分别保留。原来的成败分组只用于固定选样，不当作JPEG的新成败结果。

## 当前步骤：复用已部署脚本核对24张原图

用户已回传 `/home/featurize/data/BigEarthNet-S2` 目录存在。它不证明所有波段齐全，先不解压、不提取ZIP；直接使用已部署脚本的original模式：

```bash
/usr/bin/time -v python -u /home/featurize/work/pair44_feedback_qnjm_rb7/diagnose_bigearth_feedback.py --input-mode original
```

这是24张固定案例的前台诊断，保持当前终端打开。省略 `--out` 会自动创建新的 `/home/featurize/work/pair44_feedback_随机后缀/`，开头打印 `OUTPUT`，不会覆盖gallery产物。该入口先检查原清单对应的所有所选RGB文件及代码身份；缺文件时在加载模型前停止。原图存在且可解码后，再执行同权重、同模型的正常/屏蔽/恢复前向，并对每张图全部19类要求正常预测与原RGB缓存概率差≤1e-4。

结果以 `Input: original; unique images: 24.`、`FEEDBACK COMPLETE`、`Exit status: 0` 为完成证据；成功报告会列出最大缓存差，完整身份/输入哈希、逐张JSON/PT/PNG/SVG和报告保存在打印的新目录。该前台调用不经过原gallery后台包装器，因此不产生 `DIAGNOSTIC_EXIT` 或外层 `diagnostic.exit`，也不自动写 `diagnostic.log`；终端输出可以直接回传，数据与报告文件仍保存。

若显示缺文件或 `Original-image prediction differs from full-val cache`，保留新目录的 `failure.json` 和报错，先定位缺失输入或剩余环境差异。不扩大容差或改成gallery来跳过检查。当前仅准备命令，尚无original运行返回。

## 服务器操作：复制文本，不上传工具包

打开本地 [可粘贴命令](paste_bigearth_feedback_gallery.sh)，全选复制到服务器正常Shell提示符后执行。若仍在 `dquote>`，先 Ctrl+C 退出。复制从 `python - <<PY` 到最后单独一行 `PY` 的全部文本，不要复制终端提示符或Markdown围栏。

命令会在 `/home/featurize/work/pair44_feedback_随机后缀/` 保存脚本、后台执行日志与退出码；脚本的SHA256会在启动前核对。无需向服务器传文件、Git拉取或修改正式模型。脚本使用已有工具：

- 案例来源：`/home/featurize/work/pair44_fullval_8Tvk9d/diagnosis`。
- 正式仓库：`/home/featurize/work/GAViT_Project`。
- 既有配对工具：`/home/featurize/work/gavit-pair-diagnostic-20260915/tools`。
- 权重：来源 `run.json` 中记录的 GAViT checkpoint；校验侧车metadata、模型配置和SHA256。

粘贴后会打印 `tail -f /home/featurize/work/pair44_feedback_.../diagnostic.log`。运行打印出的整行查看进度；保留绝对结果路径，重新连接也能使用。PID只说明后台进程已创建，看到 `DEVICE` 和首个 `CASE ... GATES PASS` 才说明GPU诊断已开始。

成功以 `FEEDBACK COMPLETE`、`DIAGNOSTIC_EXIT 0` 和 `diagnosis/run.json` 的 `status=complete` 为准。把日志末尾从 `# GAViT fixed-case feedback probe` 开始的报告及两个完成标记复制回来即可。若出现Traceback，回传最后的错误文本；保留目录，不要修改源数据或跳过失败案例。案例图丢失时脚本会停止，不自动解压或改变来源。

## 固定选择与检查

从原 `cases.csv` 中，对类别11/14/16的“真实标签0/1 × 两者对/两者错/Swin独对/GAViT独对”各取已经排好序的第一条，最多24个类别案例，同一图片只计算一次。空分组不补选，不根据干预效果挑图。此集合包含刻意选择的错误/成功案例，不能计算总体AP、F1或错误率。

每张图片均运行以下四项数值核验，绝对与相对容差均为1e-5：

1. 无hook的正常前向，与捕获实际中间量的前向一致。
2. 捕获的token、区域反馈和assignment经原classifier重建，与正常前向一致。
3. 仅将整个 `feedback_proj` 的输出置零，与原模型backbone token均值经过原classifier一致。置零点在投影与LayerNorm之后，不残留其偏置。
4. 移除屏蔽hook后，恢复原始输出；所有hook均在finally中移除。

捕获图输入必须有49个token、16个非空区域、80条无重复且无自环的有向边，其中48条四邻域空间边、每个区域2条额外特征入边。图像上的箭头是实际 `neighbor -> query` 边；GAT内部自环单独记录，图中不画。余弦分数只用于选边，不当作GAT attention；反馈L2也不代表区域重要性。

正常和屏蔽反馈的预测会分别与真实标签比较。`Feedback corrects` 表示保留反馈时对、屏蔽时错；`Feedback breaks` 反之；`Both correct/wrong` 是两种前向均对/错。统计仅针对入选类别案例。零反馈是共同训练后模型的一次推理干预，不能证明独立训练的无图模型会更好或更差。

## 保存的证据

`diagnosis` 内保存来源文件/代码/输入图像哈希、选样清单、加载权重身份、软件与设备信息；每张图片的实际输入PNG、区域与图边SVG、中间张量PT、19类正常/屏蔽分数JSON；以及 `cases.html`、`summary.json`、`report.md`、`run.json`。失败保留 `failure.json`。外层保留执行脚本、`diagnostic.log` 和 `diagnostic.exit`。正式test及原诊断文件不改写。

`--prepare-only` 可在无Torch/无原始数据环境检查gallery路径、来源代码、标签和选样，不加载权重或核验GPU。原始RGB仍在时可单独指定 `--input-mode original`，额外要求正常预测与原缓存概率差不超过1e-4；本次粘贴命令固定gallery，不自动切换。

本地11项新诊断测试和16项原配对诊断回归测试通过；本地没有Torch。服务器已返回正常完成标记，按该脚本执行流程意味着逐张数值与拓扑检查未触发异常；本地仍未收到实际图张量、逐项误差和GPU资源记录，不将其表述为已独立复核。
