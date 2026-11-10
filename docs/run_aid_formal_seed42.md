# AID seed42正式实验执行记录

09-21回传更新：两模型校准、refit、最终test均完成，双方显式exit0。5000test OA为Swin96.34%/GAViT96.94%，macro-F1为96.22088514124282%/96.7811271561119%；身份匹配。下方所有GPU命令仅为运行历史，不重复执行。当前工作是[读取已有预测与逐类表](analyze_aid_seed42.md)。

2026-09-21。E00及两模型校准/refit/test已通过；完整逐类和逐样本分析材料仍待从服务器读取。工程短测见[验收表](../results/aid_smoke_validation.md)，最新实测见[阶段结果](../results/aid_seed42_comparison.md)，保持服务器代码版本不变。

## 研究问题与固定身份

- 假设：在冻结AID划分和训练配方下，固定4×4区域图推理能提升场景分类表现，且额外成本值得。单个seed42用于首轮配对证据，不作稳定性结论；候选保留依据验证集和成本，不根据test重跑或换seed。
- 配对主要变量：Swin-T vs K16 attentive_spatial/sparse_hybrid/token_feedback GAViT；其余保持相同。Swin参数27543960，GAViT31375385。两模型正式校准与refit已完成，尚无test结果；两模型短测已完成更便宜的路径/模型/存盘验证。refit使用各自预先约定的验证选轮规则，21/20轮的差异由该规则产生。
- 产物：`results/aid_seed42_comparison.md`的分阶段结果，以及后续正式AID对比表、逐类正确率/混淆矩阵和成本讨论。主指标OA，次指标macro-F1，保持各自最佳和最终epoch。短测OA不进入正式结果。
- Git `deb2b45`，正式运行入口要求干净工作树；seed42；CUDA / batch32 / workers2 / float32 / TF32关闭 / AdamW lr0.0003 wd0.0001 / CE / CosineAnnealingLR T_max30。RGB224、ImageNet归一化及两模型共享增强沿用已记录的training身份。
- 数据清单：`/home/featurize/work/aid_protocol_20260920_03.json`，SHA256 `30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22`。AID schema2 / scene-pool-group-v2，外层split_seed42、内部val_seed4242，4000/1000/5000；相同像素按组隔离，1组Industrial重复均train。数据根目录`/home/featurize/data/AID/AID`。
- 预训练：`/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors`，SHA256 `f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`。正式实验从该预训练文件重新初始化，不接着smoke checkpoint训练。

## 执行顺序

1. Swin校准：4000 train / 1000 val / 30轮，按val OA最高选择epoch，同分最早；test不参与选择。新输出`aid_swin_calibrate42_20260920_01`。
2. Swin完成并验收后，GAViT同配方校准30轮，使用独立输出目录；不在同一GPU并发运行两者。
3. 两模型分别从同一预训练重新初始化，在完整5000张训练池上训练各自校准选定的轮数；Cosine周期仍30，不从校准权重续训。最终使用该固定预算的末轮权重。
4. 选择冻结后，各自评估5000张test，保存OA、macro-F1、逐类正确率/支持数、混淆矩阵和预测。不得把40/10/50的校准checkpoint误写成50%训练结果。

## Swin校准历史命令（已完成）

```bash
nohup python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py train --model swin --phase calibrate --seed 42 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --output /home/featurize/work/aid_swin_calibrate42_20260920_01 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --device cuda --batch-size 32 --workers 2 >> /home/featurize/work/aid_swin_calibrate42_20260920_01.launch.log 2>&1 < /dev/null &
```

查看启动日志（含入口导入错误；采用追加避免重复尝试覆盖旧日志）：

```bash
tail -f /home/featurize/work/aid_swin_calibrate42_20260920_01.launch.log
```

前面会再次校验全量图片。须看到`MODEL swin ... PHASE calibrate; train=4000 val=1000`、实际GPU、`TRAIN FIRST BATCH OK`以及`EPOCH 1/30`才算进入预期训练。可在另一终端用`pgrep -af run_scene.py`及`nvidia-smi`核查进程与显存；只出现后台PID不算启动成功。Ctrl+C只停止tail，后台训练继续。

首个完整epoch后读取`history.json`的train_seconds、validation_seconds、train_images_per_second及本轮日志，用全量实测估余下29轮时间；smoke 3秒训练不能当正式每轮耗时。总预算须包括两次30轮校准、各自选定轮数的5000训练池重训、测试/存盘/扫描时间。实际租价未提供，费用待确认。

结束验收：

```bash
cat /home/featurize/work/aid_swin_calibrate42_20260920_01/run.exit /home/featurize/work/aid_swin_calibrate42_20260920_01/run.json
```

要求EPOCH 30/30、SCENE CALIBRATE COMPLETE、status complete、exit_code=0、checkpoint重载差≤1e-5及最佳/最终epoch均记录。运行目录有run.log、config、selected_samples、history、best/last checkpoint及metadata等。入口拒绝覆盖已有目录；失败重试采用新编号并保留失败证据。当前入口不提供自动续训。未验收前不自动排队其他模型、seed或test。

## GAViT校准历史命令（已完成）

保持同一实例与配方，首次在新输出目录运行：

```bash
nohup python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py train --model gavit --phase calibrate --seed 42 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --output /home/featurize/work/aid_gavit_calibrate42_20260920_01 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --device cuda --batch-size 32 --workers 2 >> /home/featurize/work/aid_gavit_calibrate42_20260920_01.launch.log 2>&1 < /dev/null &
```

```bash
tail -f /home/featurize/work/aid_gavit_calibrate42_20260920_01.launch.log
```

启动须看到MODEL gavit / PHASE calibrate / train=4000 val=1000和实际RTX2080Ti；TRAIN FIRST BATCH OK内应有actual_forward_checked=True、48空间边+32特征边。首轮EPOCH 1/30后可回传进度并保持后台继续；若查看GPU占用，停止tail后执行nvidia-smi不会停止训练。只有后台PID不算启动成功。

完成后要求EPOCH 30/30、SCENE CALIBRATE COMPLETE，并回传：

```bash
cat /home/featurize/work/aid_gavit_calibrate42_20260920_01/run.exit /home/featurize/work/aid_gavit_calibrate42_20260920_01/run.json
```

以上核对已通过：Swin选21轮、GAViT选20轮。两模型test仍待最终拟合之后进行。

## Swin最终拟合历史命令（已完成）

目的：按既定50%训练池/50%测试协议，用全部5000张训练池拟合最终模型。校准已用于选择训练预算；refit重新加载同一ImageNet预训练，合并原train与val，不继续使用校准权重训练。预算自动从服务器原始校准报告读取，Cosine T_max仍30，不新增超参选择或测试集反馈。已有短测及配对校准已排除基本工程问题，无需再做一次完整校准。

已完成本地只读身份核对，发布入口validate_calibration对Swin/GAViT分别返回21/20轮；没有执行GPU重训。服务器入口仍会核查原始报告、校准checkpoint、metadata、退出码、代码及全量数据。保留原校准目录，不修改run.json或代码；本地规范化证据不替换服务器文件。

先运行Swin，验收后再运行GAViT，不并发排队：

```bash
nohup python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py train --model swin --phase refit --seed 42 --calibration /home/featurize/work/aid_swin_calibrate42_20260920_01 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --output /home/featurize/work/aid_swin_refit42_20260920_01 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --device cuda --batch-size 32 --workers 2 >> /home/featurize/work/aid_swin_refit42_20260920_01.launch.log 2>&1 < /dev/null &
```

```bash
tail -f /home/featurize/work/aid_swin_refit42_20260920_01.launch.log
```

预检后须看到MODEL swin、PHASE refit、train=5000 val=0、实际GPU、TRAIN FIRST BATCH OK及EPOCH 1/21。val_OA=None正常，因为原验证集已并入训练池，不再承担验证选择。首轮history记录全训练池吞吐和训练秒数；用实测轮耗时×剩余20轮，另加扫描/存盘估余下时长。此前30轮校准全程21分21秒（Swin）与22分47秒（GAViT）只作为资源参照，不能直接当作refit预计时长或推理benchmark；费用待提供。

结束后读取：

```bash
cat /home/featurize/work/aid_swin_refit42_20260920_01/run.exit /home/featurize/work/aid_swin_refit42_20260920_01/run.json
```

验收SCENE REFIT COMPLETE、exit0/status complete、final_epoch21、train5000/val0、来源身份正确及重载差≤1e-5；最终checkpoint应为last.pth，best_epoch/best_val为空正常。新目录已存在时保留证据并先查状态，不覆盖或直接重跑。当前不自动启动test。

## GAViT最终拟合历史命令（已完成）

Swin已验收通过。研究问题与冻结配方保持不变：完成GAViT在相同5000张训练池上的最终拟合，以便随后与Swin比较保留测试集表现。已通过GAViT smoke、完整校准及Swin refit，无需新增校准或改变超参。预算由GAViT自己的校准best epoch20自动读取，仍从同一ImageNet预训练重新初始化。

```bash
nohup python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py train --model gavit --phase refit --seed 42 --calibration /home/featurize/work/aid_gavit_calibrate42_20260920_01 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --output /home/featurize/work/aid_gavit_refit42_20260920_01 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --device cuda --batch-size 32 --workers 2 >> /home/featurize/work/aid_gavit_refit42_20260920_01.launch.log 2>&1 < /dev/null &
```

```bash
tail -f /home/featurize/work/aid_gavit_refit42_20260920_01.launch.log
```

启动检查MODEL gavit / PHASE refit / train=5000 val=0、实际GPU、TRAIN FIRST BATCH OK及EPOCH 1/20；图捕获actual_forward_checked=True且48空间边+32特征边。val_OA=None符合设计。实际吞吐/每轮时间由首轮history记录；Swin refit全程14分53秒仅作参照，GAViT总时长按首轮实测与剩余轮数估计，费用待租价。

出现EPOCH 20/20及SCENE REFIT COMPLETE后读取：

```bash
cat /home/featurize/work/aid_gavit_refit42_20260920_01/run.exit /home/featurize/work/aid_gavit_refit42_20260920_01/run.json
```

验收exit0/status complete、final_epoch20、train5000/val0、校准来源及配对身份正确、实际图48+32边、last.pth重载差≤1e-5。best_epoch/best_val为空正常；其余验收条件同Swin。两模型各用固定预算末轮last.pth进行最终test，保存OA、macro-F1、逐类指标/混淆矩阵/逐样本预测，更新正式对比表。测试结果尚未产生。

## Swin最终测试历史命令（已完成，exit0）

研究问题是两模型在同一保留测试集上的单seed泛化差异。模型、轮数和权重已冻结，不再训练或用test调整选择。复用已有last.pth是回答问题的直接方法；预检会校验原图、manifest、checkpoint/metadata、来源完成状态和源码身份。相同batch32/workers2、CUDA、float32、TF32关闭、确定性评估变换；样本应5000。实际GPU和环境由评估报告记录。

先测试Swin，输出使用新的独立目录：

```bash
nohup python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py evaluate --checkpoint /home/featurize/work/aid_swin_refit42_20260920_01/last.pth --split test --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --output /home/featurize/work/aid_swin_test42_20260921_01 --device cuda --batch-size 32 --workers 2 >> /home/featurize/work/aid_swin_test42_20260921_01.launch.log 2>&1 < /dev/null &
```

```bash
tail -f /home/featurize/work/aid_swin_test42_20260921_01.launch.log
```

入口先进行全量10000张图片完整性扫描，最终只评估5000张test；扫描数不是测试样本数。当前_predict没有逐batch打印，扫描后加载权重/推理期间短暂无日志正常，不要求不存在的训练FIRST BATCH提示；最终以完成标志、样本数和退出报告验收。若需核实正在运行，可用pgrep -af run_scene.py与nvidia-smi，不能仅凭PID或无新日志断言成功/失败。

出现SCENE EVALUATION COMPLETE后，Ctrl+C退出tail，运行以下只读摘要命令并回传：

```bash
python -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); print((p/"run.exit").read_text()); print((p/"run.json").read_text()); m=json.loads((p/"metrics.json").read_text()); print(json.dumps({k:m[k] for k in ("samples","oa_percent","macro_f1_percent","dataset","split","training_phase","fit_scope","manifest_sha256","checkpoint_sha256")},indent=2))' /home/featurize/work/aid_swin_test42_20260921_01
```

要求status complete、exit_code0、samples5000、split test、training_phase refit、fit_scope train_plus_val、同manifest SHA，checkpoint SHA为`fc04ddd58e233bc7291b6a5e54a2c3e9efc45c0584a709aa28a0eba496178018`。OA与macro-F1已按百分数保存；不再次乘100。完整metrics.json含30类指标和混淆矩阵（行真值/列预测），另有predictions.csv/per_class.csv，保留供后续配对统计；摘要不替代完整产物。

## GAViT最终测试历史命令（已完成，exit0）

Swin指标与身份已核对，缺失的显式退出码在下方一并补读；不因此重跑测试。继续已冻结的GAViT测试计划，不根据Swin分数调整GAViT配置或权重：

```bash
nohup python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py evaluate --checkpoint /home/featurize/work/aid_gavit_refit42_20260920_01/last.pth --split test --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --output /home/featurize/work/aid_gavit_test42_20260921_01 --device cuda --batch-size 32 --workers 2 >> /home/featurize/work/aid_gavit_test42_20260921_01.launch.log 2>&1 < /dev/null &
```

```bash
tail -f /home/featurize/work/aid_gavit_test42_20260921_01.launch.log
```

出现SCENE EVALUATION COMPLETE后退出tail并运行下方摘要命令，一并读取Swin退出码：

```bash
python -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); s=pathlib.Path(sys.argv[2]); print("Swin:",(s/"run.exit").read_text()); print("GAViT:",(p/"run.exit").read_text()); print((p/"run.json").read_text()); m=json.loads((p/"metrics.json").read_text()); print(json.dumps({k:m[k] for k in ("samples","oa_percent","macro_f1_percent","dataset","split","training_phase","fit_scope","manifest_sha256","checkpoint_sha256")},indent=2))' /home/featurize/work/aid_gavit_test42_20260921_01 /home/featurize/work/aid_swin_test42_20260921_01
```

GAViT预期checkpoint SHA为`a023563e7c4566436660a2d9bc778a9adb3e9b5a64f39f821ce81e909c913c7a`，5000test/refit/train_plus_val、同manifest/代码/评估环境，明确exit0及OA/macro-F1。测试全量10000张文件扫描后只评估5000张；无逐batch日志。当前未代为启动GAViT测试。两份test结果验收后再完成正式比较和解释；推理延迟/吞吐与峰值显存需独立标准化测量，当前evaluate未记录这些资源统计，不由训练wall time推算。完整metrics.json、per_class.csv、predictions.csv均保留供后续分析。
