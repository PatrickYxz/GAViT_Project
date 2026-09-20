# AID 接入与服务器短测

更新：2026-09-20。用于AID正式实验前的工程验收（E00）。本地统一入口已实现；用户确认服务器尚未准备 AID。真实数据预检、ImageNet权重加载与CUDA短测仍待执行，不能把本地合成图片测试写成AID实验结果。

## 1. 先准备数据和代码

AID应为30类、10000张600×600 RGB图像。数据来源及下载入口见[AID官方项目页](https://captain-whu.github.io/AID/)。如果平台数据集列表提供该数据，可直接在服务器添加；本次未确认平台是否收录，也未替用户下载数据。

示例假设解压后直接呈现 `/home/featurize/data/AID/Airport/*.jpg` 等类别目录。若多嵌套了一层AID，应把`--data-root`指向实际包含30个类别文件夹的那层。**这些是路径示例，不是已核实的服务器目录。** 不需要为这一步恢复BigEarthNet原始TIF。

代码分支为 `codex/aid-entry-20260920`，基于 `8f2ed07671ed50387998a46a2d2167a7a4dfc2ed`。在服务器用一个全新目录拉取，保留原有BigEarthNet实验目录及其修改。下面的同步命令在分支推送成功后执行；目标目录已存在时Git会拒绝，不删除目录强行重试。

```bash
git clone --single-branch --branch codex/aid-entry-20260920 https://github.com/PatrickYxz/GAViT_Project.git /home/featurize/work/GAViT_AID_20260920
```

```bash
git -C /home/featurize/work/GAViT_AID_20260920 log -1 --oneline
git -C /home/featurize/work/GAViT_AID_20260920 status --short --branch
python /home/featurize/work/GAViT_AID_20260920/run_scene.py --help
```

数据、预训练路径确认及代码同步后检查：

```bash
ls -ld /home/featurize/data/AID
ls -lh /home/featurize/work/GAViT_AID_20260920/run_scene.py /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors
python -c 'import torch, torchvision, timm, torch_geometric; print(torch.__version__, torchvision.__version__, timm.__version__, torch_geometric.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

`model.safetensors`沿用先前本地ImageNet预训练文件的路径示例；不是训练好的BigEarthNet分类checkpoint。入口只从指定本地文件加载，两模型记录同一文件SHA256，不自动联网下载。若该文件不存在，先确认现有预训练来源和实际路径。

## 2. 生成一次固定划分

```bash
python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py prepare --dataset AID --data-root /home/featurize/data/AID --output /home/featurize/work/aid_protocol_20260920_01.json
```

此步骤检查原始尺寸/RGB、30类名称与总数、文件可读性、逐文件哈希和解码像素重复。只保存引用清单，不复制图片、不改变原文件；重复、缺失、损坏或不匹配会停止，不能靠删样本强行通过。

外层训练池50%、其余test；池内20%用于val，预期train4000/val1000/test5000，最终以输出清单为准。分层按类取整、路径排序，split seed42和内部val seed4242与训练seed分开记录。换实例后保留清单，仅恢复相同图像并指向新的根目录。

完成标志是`MANIFEST COMPLETE`。已有输出不会覆盖；失败后保留证据，后续新运行换新的编号。

## 3. 先Swin短测，成功后再GAViT

两模型使用同一清单和同一训练配方。每类最多8张训练、2张验证，完整AID预计240/60张，1 epoch；test不参与训练或预测。代码只读取test文件以做完整性检查。

启动Swin（整行复制，无需创建Shell变量）：

```bash
nohup python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py train --model swin --phase smoke --manifest /home/featurize/work/aid_protocol_20260920_01.json --data-root /home/featurize/data/AID --output /home/featurize/work/aid_swin_smoke_20260920_01 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --device cuda --batch-size 32 --workers 2 > /dev/null 2>&1 < /dev/null &
```

```bash
tail -f /home/featurize/work/aid_swin_smoke_20260920_01/run.log
```

看到完成标记后按Ctrl+C停止tail，不会终止已完成的任务，再核对：

```bash
cat /home/featurize/work/aid_swin_smoke_20260920_01/run.exit
cat /home/featurize/work/aid_swin_smoke_20260920_01/run.json
```

Swin通过后，单独启动GAViT：

```bash
nohup python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py train --model gavit --phase smoke --manifest /home/featurize/work/aid_protocol_20260920_01.json --data-root /home/featurize/data/AID --output /home/featurize/work/aid_gavit_smoke_20260920_01 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --device cuda --batch-size 32 --workers 2 > /dev/null 2>&1 < /dev/null &
```

```bash
tail -f /home/featurize/work/aid_gavit_smoke_20260920_01/run.log
```

```bash
cat /home/featurize/work/aid_gavit_smoke_20260920_01/run.exit
cat /home/featurize/work/aid_gavit_smoke_20260920_01/run.json
```

日志由入口写在新输出目录中，重连后可以直接读，不依赖旧Shell变量。已有输出目录会拒绝重新运行，重试更换`_01`为新编号。若目录/日志根本没有生成，可能是入口导入前就失败；用相同参数前台运行检查导入错误，不把PID当作启动成功。

## 4. 验收与结果位置

必须同时满足：

- 日志确认模型、实际CUDA设备、数据集与240/60样本；`TRAIN FIRST BATCH OK`和`EPOCH 1/1`均出现。
- GAViT实际前向捕获每图80条边：48条四邻域空间边、32条非邻接特征边，各节点2条特征入边。
- `best.pth`、`last.pth`、对应metadata及最后训练状态已写出；保存前预测与重建模型预测差不超过1e-5。
- 日志有`SCENE SMOKE COMPLETE`，`run.json`为`status: complete`，`run.exit`为`exit_code=0`。断电/强制结束可能没有exit文件，不能当成功。

每个目录还保存`config.json`、`selected_samples.json`、`history.json`、`validation_metrics.json`、`validation_per_class.csv`、`validation_predictions.csv`。OA、macro-F1、逐类正确率/支持数、混淆矩阵（行是真值、列是预测）均保留完整精度。代码根目录Git身份、相关源文件哈希、权重哈希、软件/设备与实际配置随结果保存。

吞吐来自本次训练循环；显存峰值包含保存后模型重建，主机RSS只覆盖主进程。它们用于发现资源问题，不能当作最终公平效率benchmark。当前没有自动续训入口；中断后保留状态与日志，再判断处理方式，不覆盖运行目录。

## 5. 后续正式阶段的边界

`train --phase calibrate`在内部train上运行30轮，按val OA选best，同分取最早；`--phase refit --calibration <完成的校准目录>`从同一预训练文件重新开始，在整个训练池上训练已选轮数，Cosine周期仍为30。它要求源报告、退出码、模型/seed/协议/代码/训练配置和最佳checkpoint身份一致；不接着校准checkpoint训练。

正式阶段要求真实数据、CUDA、batch32、本地预训练以及干净Git。暂不在本指南提供正式后台启动命令，须先验收上述两次真实GPU短测。这个限制不妨碍准备数据和本地检查。

评估由`evaluate --checkpoint ... --manifest ... --data-root ... --output ... --split val|test`完成，只接受成功完成运行的选定checkpoint，并检查当前相关源码哈希与checkpoint记录相同；smoke不能测test，refit不能把已经见过的val当独立验证集。校准checkpoint的test结果只代表40/10/50流程；真正用满50%训练池的是refit末轮权重，不能混写。

当前本地CPU测试使用明确标为`synthetic`的合成图片和随机权重；另外验证了本地权重文件能给两模型加载相同backbone张量。均不代表实际ImageNet文件、真实AID或CUDA已验收。测试依赖装在独立临时环境，未改用户项目环境。

本地Mac双worker检查未通过：沙箱内出现OpenMP共享内存错误，获准在沙箱外重试后180秒超时，后一次根因未确定。已成功完成的是workers=0的两模型30类CPU命令行流程；不据此推断Linux服务器双worker正常或异常。服务器短测须实际验证上述`--workers 2`，如失败先读日志，不静默修改正式配置。
