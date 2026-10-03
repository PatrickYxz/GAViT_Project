# NWPU seed42 有界 FFN 候选实验 — 2026-10-03

状态：方案已批准；本页固定研究问题、参数和完成标准。新 FFN 版本尚未提交或发布，服务器新版本目录、完整 commit、实际 GPU 短测和正式训练均待确认。本页不提供指向不存在 release 的启动命令。

## 研究问题与本次范围

假设：在 mean pooling 前加入同一轻量非线性残差 FFN，是否改善 soft GAViT 对 token 与图反馈的利用，并使其校准表现超过同样增加 FFN 的 Swin。标准 FFN 本身不是方法创新；有收益也不等于已确定原模型的失败原因。

唯一主要变量是 `token_refinement: none → residual_ffn_v1`。构成 Swin/soft GAViT × 无 FFN/有 FFN 的四格比较；无 FFN 两格复用已完成的 seed42 校准，不重训。两种模型都增加同一 FFN，避免把一般新增容量的收益直接归于图推理。

本批仅新增两个 NWPU seed42、各 30 epoch 的 calibration：Swin+FFN 与 soft GAViT+FFN。正式训练前各做一次 1 epoch、每类 train8/val2 的工程 smoke（总计 360/90）。不增加 fixed+FFN，不搜索宽度/初始化/学习率，不自动 refit、test、扩 seed、跑 BigEarthNet 或失败后重试。

旧 checkpoint 的反馈幅度诊断已完成，不能回答重新训练 FFN 的收益；更便宜的本地结构/梯度/身份测试和上述 GPU smoke 先排除工程问题。正式结果进入独立 NWPU FFN 校准四格表及成本、曲线、配对错误分析；不改写既有 test 主表或混入旧 70/15/15 协议约 96% 的结果。

## 冻结的 FFN 与模型配置

- 公式：`y = x + fc2(GELU(fc1(LayerNorm(x))))`，作用于每个 token；768 → 192 → 768，两层 Linear 均有 bias。
- LayerNorm `eps=1e-5`；GELU 使用精确形式；FFN 内无 dropout。原分类头和图模块的 dropout 保持原配方。
- 初始化：独立 CPU RNG seed42003，fc1 Xavier uniform、bias0；fc2 weight/bias 全0；不消耗原模型或训练数据的随机数流。两模型使用相同初始 FFN 权重。
- Swin：backbone tokens → FFN → mean pool → 原分类头。soft GAViT：原 token feedback → FFN → mean pool → 原分类头；不改反馈幅度。
- GAViT 保持 `soft_prototype`、K16、128 维共享投影、局部 Chebyshev 半径1、温度0.1、一次分配，`sparse_hybrid`、GAT hidden256/4heads/2layers、`token_feedback`；每图48空间边+32特征边。
- FFN 新增参数 297,408；预计总参数 Swin 27,852,903、soft GAViT 31,784,168，须以新模型实际计数核对并保存。

## 复用证据与冻结输入

旧结果见 [校准比较](../results/nwpu_seed42_calibration_comparison.md)、[用户回传结构化证据](../results/evidence/nwpu_calibration42_20260928/server_return.json) 和 [只读诊断](../results/nwpu_seed42_cached_diagnosis.md)。这些是已归档的服务器回传；不声称本地独立下载权重或复跑前向。

| 旧对照 | 最佳 val OA (%) | 同 checkpoint macro-F1 (%) | 最佳轮 | 正确数/1260 |
|---|---:|---:|---:|---:|
| Swin | 94.2063492063492 | 94.171552 | 29 | 1187 |
| soft GAViT | 93.73015873015873 | 93.765721 | 28 | 1181 |

- 旧发布版本：`ed480e60fea03ed786fdba70c144de5e815c34f5`；服务器旧代码目录 `/home/featurize/work/GAViT_SOFT_20260927`，该版本不含新 FFN 开关。
- 旧产物根目录：`/home/featurize/work/nwpu_ablation42_20260928_01`，复用其 `swin`、`soft` 下的 run.json、history.json、validation_predictions.csv、validation_metrics.json 与 best.pth。
- 旧 Swin best SHA256：`bbfd4e472739126276e79014f944f911d7b0abbc6f863bb291d799845b9cd0ea`。
- 旧 soft best SHA256：`674da671e1df1618add93cd18e791e0a81fdfb77ce0dc13478eb963bfa29cc12`。
- manifest：`/home/featurize/work/nwpu_protocol_20260927_02.json`；SHA256 `ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`；train5040/val1260/test25200，总31500。同类重复像素组不跨 split，不重新 prepare。
- 预训练：`/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors`；SHA256 `f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`。
- 数据根目录：`/home/featurize/data/NWPU-RESISC45`。持久 ZIP 为 `/home/featurize/work/NWPU-RESISC45_a826b44.zip`，SHA256 `beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53`。
- 配方：RGB224、原共同增强和归一化、CrossEntropyLoss、batch32、workers2、seed42、AdamW lr3e-4/wd1e-4、betas0.9/0.999、eps1e-8、CosineAnnealingLR T_max30/eta_min0；FP32、无 AMP、两项 TF32 关闭、8 CPU 线程。
- 旧硬件/运行时已确认 RTX2080Ti、torch2.2.2、CUDA12.1。新运行实测 GPU 型号/UUID、驱动、软件、精度和线程配置后保存；不能用旧记录冒充当前硬件，同型号不同物理卡也应注明。

## 发布与执行门禁

1. 本地实现和相关回归测试完成后，单独发布包含 FFN 的干净版本，记录完整 commit 和源码哈希。新发布 commit：**待填**；新服务器干净代码目录：**待填**。保留旧服务器代码与所有旧产物，不在脏目录强行 pull、覆盖修改或临时改正式模型。
2. 新版本必须能从保存的 architecture/metadata 重建 FFN，校验 checkpoint hash 并通过重载。关闭开关的旧 checkpoint 保持兼容；非法数据集、seed、phase 或不匹配配置应拒绝。完成这些检查后再交付用户可复制的单行命令，无需上传工具包。
3. 先读现有输入和输出，排除本批已完成/活动/失败产物；已有同身份结果先审计复用。核对清单、权重 SHA、旧对照身份和完整性、GPU 空闲及依赖，不无条件重装或重下载。
4. `work` 保存代码、manifest、权重、日志和成果；`data` 是实例临时图像盘。仅在清单图片确实缺失时验证持久 ZIP 后恢复缺失图片，保留已有文件；完整图片仍须由入口逐图解码/哈希检查。不要因为换实例自动重新生成划分。
5. 顺序执行 Swin+FFN smoke → soft GAViT+FFN smoke；每项输出独立。全部 smoke 与成本记录通过并写入日记后，再执行两项 30 epoch calibration。每阶段失败即停止，保留失败日志和产物，不覆盖目录或自动重试。

拟定输出根目录 `/home/featurize/work/nwpu_ffn42_20261003_01`，子目录 `swin_ffn_smoke`、`soft_ffn_smoke`、`swin_ffn`、`soft_ffn`；目前仅为计划路径，启动前确认不存在并冻结。日志文件同名放在持久化 work，独立保存各阶段退出码和总摘要。

## 精确入口参数提纲（不是可执行发布命令）

入口是新发布版本的 `run_scene.py train`；新版本路径确认前禁止改用旧 release 执行。下面是待封装的参数集合，故意不填虚构的 Python 脚本路径。

```text
共同参数：
--token-refinement residual_ffn_v1 --seed 42
--manifest /home/featurize/work/nwpu_protocol_20260927_02.json
--data-root /home/featurize/data/NWPU-RESISC45
--pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors
--device cuda --batch-size 32 --workers 2

Swin smoke：--model swin --phase smoke --train-per-class 8 --val-per-class 2
soft smoke：--model gavit --grouping soft_prototype --phase smoke --train-per-class 8 --val-per-class 2
Swin 校准：--model swin --phase calibrate
soft 校准：--model gavit --grouping soft_prototype --phase calibrate
各阶段另加 --output 对应独立子目录（最终绝对路径待发布时冻结）。
```

smoke 固定1轮，calibrate 固定30轮，由入口按 phase 确定；不添加不存在的 `--epochs` 参数。正式命令不得包含 `--random-init`、`--calibration`、refit 或 evaluate/test。初始化使用共同 ImageNet 预训练，不从旧校准 best 接续训练。

## 运行证据与成本

- 启动成功需同时看到进程、实际 GPU 显存、模型/设备与首个训练 batch；PID 或目录存在不足以确认。可用 `nvidia-smi` 与独立 launch.log 核查，`Ctrl+C` 退出 tail 不等于停止训练。
- smoke 必须完成 forward、backward、optimizer step、360/90 train/val、checkpoint 保存及重载，无 NaN/Inf/OOM；soft 实际构造48+32边。验证 FFN 身份、相同初始化、有限梯度/参数更新，并保存入口要求的 soft 探针。
- 每项须有进程 exit0、`run.exit` 为 `exit_code=0`、run.json `status=complete`，且 phase/seed/数量/架构/哈希相符。正式 history 到 epoch30，最佳与最终轮均保留；smoke checkpoint 不进入性能比较表。
- 保存 GPU/软件、参数量、峰值 allocated/reserved 显存、纯训练吞吐、train/val/存盘时间和总墙钟。吞吐与阶段计时先明确是否含 IO/验证；没有公平纯前向测量时，不声称推理提速。
- 旧 Swin+soft 两次校准全过程合计 `1394.24341203 + 1483.651305341 = 2877.894717371` 秒，约47分58秒，仅为旧版历史成本，不是 FFN 新版预算或测得耗时。
- 新预算依据同机两个 GPU smoke 的训练/验证耗时、预检/存盘开销测量后估算，并用首个正式完整 epoch 更新。保存实测值及估算口径；小时价格未确认时不填写实际费用，不按局部进度条声称训练完成。

## 结果记录与继续条件

每模型按最高 val OA 选 best，平分取先达到者；macro-F1、per-class accuracy、混淆矩阵和逐样本预测都来自这个 checkpoint，不单独挑 macro-F1 最佳轮。保存 best/last 权重、配置/源码身份、history、验证 metrics/predictions/per-class、资源、日志及退出状态。

四格表必须同时列旧 Swin、旧 soft、新 Swin+FFN、新 soft+FFN。按同1260条路径配对，报告双方正确/纠正/新增错误/双方错误、净正确数、OA与macro-F1差值及全部逐类变化；同时比较 soft+FFN 对旧 soft、对 Swin+FFN 和对旧 Swin，并报告各自 FFN 增量及成本。

只有新 soft+FFN 的 val OA **同时严格大于** `94.2063492063492%`（旧 Swin）、新 Swin+FFN 的 val OA、`93.73015873015873%`（旧 soft），才进入“是否值得下一批多 seed”的讨论；相等不通过。仍须结合同 checkpoint macro-F1、配对损益与实测成本判断，过线不自动授权任何后续训练。

任何一项未超过、只追平、收益仅体现为共同 FFN 容量增加、或成本不值得，都保留为负面/混合结果并收尾；不因分数不好临时改选轮、改主指标、调参或重跑。单 seed、经过选轮的 val 收益不能证明稳定性、显著性、语义区域更合理或 FFN 解决了因果根因；不利用已有 test 分数挑配置。

完成回传后立即更新 `research_diary.md` 和本批独立校准结果表，明确哪些内容是实际服务器回传、哪些经过本地复算、哪些未下载独立验证。新模型正式 test、三个 seeds 的均值/标准差与公平推理效率仍是后续独立事项。
