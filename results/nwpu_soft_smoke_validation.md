# NWPU soft_prototype GPU短测 — 2026-09-28

运行`/home/featurize/work/nwpu_soft_smoke42_20260927_02`的训练、验证及checkpoint重建完成：用户回传exit0和`run.json.status=complete`。本地核对20个源码哈希、配置、清单/预训练身份、样本数量、GPU和真实图边，与发布版本一致。[原始终端摘录](evidence/nwpu_soft_smoke42_20260927_02/terminal_return.txt)、[提取JSON](evidence/nwpu_soft_smoke42_20260927_02/parsed_run.json)、[核对记录](evidence/nwpu_soft_smoke42_20260927_02/validation.json)。没有下载原始checkpoint或冒充直接连接服务器。

| 项目 | 回传及核对结果 |
|---|---|
| Code | ed480e6，dirty=false；20个运行源码SHA256匹配发布候选及当前源文件 |
| 模型 | GAViT / soft_prototype / sparse_hybrid / token_feedback，45类、16区域 |
| 软分配 | local_prototype_cosine_v1，128维、半径1、温度0.1、1次分配；完整配置匹配 |
| 数据 | 同一冻结NWPU manifest；360训练 / 90验证；1轮，seed42 |
| 配方 | CE、AdamW3e-4/wd1e-4、Cosine T_max30、batch32、workers2、FP32无AMP、TF32=false |
| 图构造 | actual_forward_checked=true；每图48空间+32特征=80条输入边 |
| 硬件/环境 | RTX2080 Ti；Python3.11.8 / torch2.2.2 / CUDA12.1 / cuDNN8902 / timm1.0.27 / PyG2.8.0；8线程 |
| 参数量 | 31,486,760；全部可训练 |
| 最佳/最终epoch | 1 / 1 |
| 小样本val OA | 40.0%，仅90张/1轮工程结果，不进入正式比较或消融精度表 |
| Checkpoint SHA256（报告值） | 41c595b7d877fef3d85ac9f2ff6792b6ff866bb7504ceeaca7ce2bccb939982c |
| 保存重建max logit diff | 3.5762786865234375e-7，通过入口≤1e-5门槛 |
| 总墙钟 | 131.20525253秒；含预检、训练/验证、探针、存盘及重建，不是纯训练单轮时间 |
| PyTorch峰值allocated/reserved | 3,931,566,592 / 4,336,910,336 bytes = 3.6616 / 4.0391 GiB |
| 主进程峰值RSS | 1,885,908,992 bytes = 1.7564 GiB；不代表所有loader子进程内存总和 |
| 探针摘要 | val固定8图，last_epoch1，累计1.590042928秒 |

与已确认身份一致：manifest SHA256`ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`；预训练SHA256`f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`。原_01缺图失败及目录碰撞保留，不计作另一训练结果。_02成功表明该次运行的数据校验和CUDA训练/重载链路完成。

## 已回答与后续任务

本次回答了新模型能否用真实NWPU图片、本地ImageNet权重和CUDA双worker完成训练、验证及保存重载。它不回答相对Swin/固定网格的收益，也不能用131秒直接乘30估算正式训练成本（正式样本更多且此数含一次性检查和重建）。

**09-28后续摘要已回传**：[结构化记录](evidence/nwpu_soft_smoke42_20260927_02/probe_summary_user_return.json)。纯train3.235914936秒、111.2513793图/秒（360/耗时复算一致）、val0.525494186秒、loss3.579246097。epoch0→1平均熵0.692803→1.014223 nats，固定输入平均绝对S变化0→0.03925493；原型低范数数均0，最小范数0.556113→0.661625；8图反馈/token范数比均值1.038349→0.759035。四份快照存在标志均true。

上述有限值与变化没有显示明显数值错误，熵上升说明平均分配更分散，不单独判定为改善或退化。非零S变化可来自骨干、注意力及投影共同变化，不能代替分配专属梯度验证；低范数为0不能证明原型语义不塌缩。快照文件存在也不等于张量归一化/身份已独立校验。

**09-28两项服务器检查已回传通过**：[终端结果](evidence/nwpu_soft_smoke42_20260927_02/final_checks_server_return.txt)。S行和误差2.3841858e-7、W列和误差2.9802322e-7，均低于1e-5；S变化0.03925493与先前摘要一致，最小区域质量1.00067246。检查命令同时核对epoch/样本身份、mask外零值、张量有限非负、best/final别名一致。CUDA梯度检查在权重和原输入SHA256匹配后，使用原探针前2图及仅依赖S的随机线性目标；投影/注意力/骨干的梯度绝对值和分别6396.8946/9804.4772/596998.5822，有限非零，图推理/反馈投影/分类头梯度为空。原始输出是用户终端回传，并非本地直接读取服务器张量。

NWPU预定工程检查收尾：真实数据训练/验证、保存重建、实际图边、快照及分配可微路径均获得证据。梯度和区域质量的量级不是优化质量或分类收益指标；1轮小样本不证明稳定提升。下一步按[160张原图存在检查](../docs/check_bigearth_soft_data.md)推进BigEarth96/64的多标签入口短测；该短测和正式三模型实验仍待执行，草稿PR不自动转为就绪或合并。

以下只读摘要命令已执行成功，保留为记录，无需重复：

```bash
python -c 'import json; from pathlib import Path; p=Path("/home/featurize/work/nwpu_soft_smoke42_20260927_02"); q=p/"assignment_probe"; h=json.loads((p/"history.json").read_text()); d=[json.loads(s) for s in (q/"assignment_diagnostics.jsonl").read_text().splitlines()]; print(json.dumps({"history":[{k:r[k] for k in ("epoch","train_loss","train_seconds","train_images_per_second","validation_seconds")} for r in h],"probe":[{k:r[k] for k in ("epoch","entropy_mean","mean_abs_delta_S","low_norm_prototypes","prototype_norm_min","feedback_to_token_ratio")} for r in d],"snapshots":{n:(q/n).is_file() for n in ("epoch_000.pt","epoch_001.pt","best.pt","final.pt")}},indent=2))'
```

该命令只读取JSON和检查文件存在；不等同于加载并校验PT张量。不因40%短测OA而改超参或重选seed。
