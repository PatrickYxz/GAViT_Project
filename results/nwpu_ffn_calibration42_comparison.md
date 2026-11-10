# NWPU seed42 FFN 校准结果 — 2026-10-05 收尾

**状态：两组30轮训练及约定的缓存收尾完成。FFN未通过冻结的val OA继续条件，保留负面结果，停止该候选后续训练。无需重复索取本批摘要或补跑。**

研究问题：在mean pooling前加入同一个297408参数残差FFN，是否改善soft GAViT并超过同样加FFN的Swin。旧Swin/soft校准结果复用，四模型都是同1260张val；本轮没有新增test或多seed结果。

## 四格校准表

最高val OA选best，平分取首次达到的epoch；下表macro-F1均来自同一个best。

| 模型 | best OA (%) | 同best macro-F1 (%) | 正确数/1260 | 最佳轮 | 末轮OA (%) | 末5轮OA均值 (%) |
|---|---:|---:|---:|---:|---:|---:|
| Swin | 94.206349 | 94.171552 | 1187 | 29 | 94.206349 | 94.000000 |
| Swin + FFN | 94.047619 | 94.044703 | 1185 | 29 | 94.047619 | 93.809524 |
| soft GAViT | 93.730159 | 93.765721 | 1181 | 28 | 93.492063 | 93.587302 |
| soft GAViT + FFN | 93.174603 | 93.145677 | 1174 | 22 | 92.936508 | 92.936508 |

四模型run/exit/counts/30行history、首次最高OA选轮、预测1260唯一同路径同标签、45类各28张支持，均通过服务器返回命令的检查。OA与macro-F1从保存的best预测重新计算，与validation_metrics及所选history行一致。服务器重新哈希两个新best文件，结果与上一轮报告相同：

- Swin+FFN：`bec792b9c51aab2c6d09e99edbc6c6533e488eb0e0477f1921f0985138fb9bf1`。
- soft+FFN：`a926e6bd23b459029accb8e1d2e9e4bae5f397ecd345cec8e90777214e2e87ed`。

## 配对损益

前者相对后者；纠正指参照错误而目标正确，新增指参照正确而目标错误。

| 比较 | OA差(pp) | macro-F1差(pp) | 双方正确 | 纠正 | 新增错误 | 双方错误 | 净正确数 | 类别升/降/平 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| soft GAViT + FFN − soft GAViT | -0.555556 | -0.620044 | 1147 | 27 | 34 | 52 | -7 | 12/15/18 |
| soft GAViT + FFN − Swin + FFN | -0.873016 | -0.899026 | 1146 | 28 | 39 | 47 | -11 | 8/15/22 |
| soft GAViT + FFN − Swin | -1.031746 | -1.025875 | 1150 | 24 | 37 | 49 | -13 | 10/20/15 |
| Swin + FFN − Swin | -0.158730 | -0.126849 | 1157 | 28 | 30 | 45 | -2 | 12/14/19 |

每组四格合计1260，边际正确数、纠正−新增、逐类净和均已在本地复核。全部非零逐类正确数差保存在[缓存回传JSON](evidence/nwpu_ffn_calibration42_20261005/cached_closeout_server_summary.json)的`pairs.*.class_net`；未列类别为0。三模型间逐类差值的可加性也通过检查。

soft+FFN相对原soft有12类改善、15类退步、18类不变。最大下降是church少4张；forest、freeway、mountain、thermal_power_station各多2张。每类仅28张val，一张对应3.571429pp，不能把局部类别变化解释为稳定的语义规律。

## 结果含义与停止决定

1. 新soft OA93.174603%同时低于旧Swin94.206349%、旧soft93.730159%及新Swin+FFN94.047619%，三项严格超过条件全部失败。macro-F1方向一致，不能通过临时更换主指标挽回候选。
2. FFN并非完全没有改变预测：soft纠正27张，同时新增34张错误，净少对7张；Swin纠正28、新增30，净少对2张。当前问题是总体收益未覆盖新增错误，不能据此定位具体架构失败原因。
3. soft+FFN第22轮best到末轮仅少对3张（−0.238095pp）；原soft第28轮best到末轮也少3张。本次比较已经使用正确的best，不能仅以最佳轮提前就认定过拟合，或以换回best为补救。末5轮均值同样未改善。
4. 结论限本次seed42和冻结配方。不能称FFN一般无效、稳定退步、统计显著或已确定根因；不因负面结果追加epoch、seed、调参、refit/test或BigEarth训练。

可用于论文的表述：在固定seed42与NWPU-RESISC45校准协议下，对Swin与soft GAViT加入相同的池化前残差FFN后，OA分别下降0.1587和0.5556个百分点，macro-F1分别下降0.1268和0.6200个百分点。因此，该配置未支持通过额外token非线性变换改善soft GAViT相对基线表现的假设。

## 参数、计时与设备

| 模型 | 参数量 | 末轮train loss | 累计训练环节秒数 | 累计验证环节秒数 |
|---|---:|---:|---:|---:|
| Swin | 27,555,495 | 0.001907911 | 901.440769 | 115.985504 |
| Swin + FFN | 27,852,903 | 0.001694292 | 836.361268 | 85.025076 |
| soft GAViT | 31,486,760 | 0.002012969 | 1001.800290 | 118.622921 |
| soft GAViT + FFN | 31,784,168 | 0.002503329 | 1018.461295 | 116.455914 |

两模型参数增量均为297408，与冻结设计相符。新Swin/soft+FFN整个run耗时分别1160.389391/1341.166717秒，合计2501.556108431秒（41分41.556秒）；该合计不含外层启动预检/阶段间隙。allocated显存峰值3.564785/3.666964GiB，保存重载差0/2.384185791015625e-7。计时包含各自数据读取/训练环节，跨次运行的缓存、IO及物理卡状态不能控制；不据此声称公平推理加速。实际租价未提供。

此次新两组hardware完全相同：RTX2080Ti、CUDA12.1、cuDNN8902、Python3.11.8、torch2.2.2、torchvision0.17.2、timm1.0.27、torch-geometric2.8.0、numpy1.26.4、pillow10.3.0、8线程；code_git为058f1d8且dirty=false。未返回GPU UUID/驱动，不能断言物理卡与旧实验完全相同。新配置完整SHA与FP32/noAMP/TF32-off等预期见[冻结协议](../docs/run_nwpu_ffn_trial_20261003.md)，本次紧凑命令检查四组manifest/pretrained/training相等，没有重新输出所有值或重验完整源码树。

## 证据和完成边界

- [用户原文](evidence/nwpu_ffn_calibration42_20261005/cached_closeout_server_return.txt)按字节归档，含9条结果和`COMPACT_CLOSEOUT_PASS`；文本中命令AST与交付命令一致。
- [结构化回传](evidence/nwpu_ffn_calibration42_20261005/cached_closeout_server_summary.json)、[本地一致性核对](evidence/nwpu_ffn_calibration42_20261005/cached_closeout_validation.json)、[完整四格JSON](nwpu_ffn_calibration42_summary.json)已更新。独立只读审阅没有发现算术或身份矛盾。
- 服务器完成缓存指标/选轮/路径/两个新best字节哈希的检查，本地复核回传及原摘要的一致性；没有本地下载预测/权重或重跑GPU。旧best文件本轮未重哈希。
- PT探针逐元素复算、last/训练状态文件字节审计、本地完整history镜像不在此精简收尾范围，不声称做过，也不再为已明确的停止决定重复索取。
- 本轮有界候选已收尾，不等于论文全部多seed、外部方法和效率要求完成。原始服务器曲线/逐类文件/checkpoint继续保留，历史旧协议comparison_table.csv/ablation_study.csv不混写；本批以本页为准。
