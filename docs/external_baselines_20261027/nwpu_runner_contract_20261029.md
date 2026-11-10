# 真实NWPU小样本运行器契约

此入口为下一次Featurize实例准备；2026-10-29用户明确告知实例尚未开启。本轮只实现与离线验证，不把历史路径存在、资源清单或函数测试写成真实GPU训练完成。

## 固定输入与范围

- manifest默认`/home/featurize/work/nwpu_protocol_20260927_02.json`，完整SHA为`ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`；代码不提供改写此固定身份的CLI开关。
- 全manifest必须为45类31500行、5040/1260/25200，逐类112/28/560，路径/标签/文件SHA合法。按原清单路径顺序取每类8张train和2张val，共360/90，pool仅两者并集450。
- 数据默认`/home/featurize/data/NWPU-RESISC45`。若所选图片缺失或SHA不符，允许从持久ZIP`/home/featurize/work/NWPU-RESISC45_a826b44.zip`恢复本次450张。
- ZIP完整SHA必须为`beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53`。只接受直接类路径或`NWPU-RESISC45/`前缀，不执行extractall；拒绝路径越界、软链接、重复/歧义成员，核验每张恢复字节SHA，不解压test图片、不覆盖原数据。
- VGG是已披露独立实现的torchvision ImageNet VGG16文件，文件名`vgg16-397923af.pth`。要求交付metadata或显式参数给出完整64位SHA，并核到实际文件；不把8位前缀当完整校验。
- 配方保持7×7、k10、GCN256×2、FC1024→512、seed42、一轮、batch8、C1。官方LIBLINEAR需事先按交付源码在目标平台编译。

## 两种动作

`inspect`只用标准库，报告实际manifest/权重/数据候选、包位置/版本、LIBLINEAR、nvidia-smi GPU UUID及compute processes；不安装、不下载、不建训练目录、不训练。ZIP存在时标记待完整hash/recovery；Python包只定位元数据，import和CUDA实际运行留给run。`ready`只表示运行前资源清单齐备，不能替代首个实际batch或完成产物。

`run --gpu-uuid GPU-实际UUID`再次预检，要求该物理UUID为RTX4090且无既有compute process。进程内设CUDA_VISIBLE_DEVICES为同一UUID，再初始化CUDA，FP32、AMP/TF32关闭，8CPU线程。输出目录必须新建。

载入已核验VGG，在CUDA上仅提取pool450一次，记录真实首VGG batch、PID、设备/形状/显存，再派生train/val缓存。只走calibrate和fresh subset refit，不调用evaluate/test。GCN首个真实训练batch和epoch日志来自训练入口；调用函数前不据此声称已启动成功。失败时保存failure/traceback/exit，可能已请求训练但尚无完成证据时training_started为null，使用实际batch日志判断。

成功需完整一轮calibration/refit history、best/final/model与线性产物、450样本subset身份和重载误差≤1e-6。summary保存GPU UUID、软件、精度、峰值allocated/reserved、权重载入与特征耗时/吞吐、校准/refit/总耗时、原manifest/权重/源码/配方/二进制SHA；run.exit=0后才输出DFAGCN_NWPU_SMOKE_COMPLETE。

## 便携交付接口

运行器路径`tools/run_dfagcn_nwpu_smoke.py`。公共参数：`--manifest --data-root --archive --weights --weights-metadata --weights-sha256 --recipe --train-binary --predict-binary --output --cpu-threads --extract-batch-size`；run另强制`--gpu-uuid`。

默认权重先查交付目录weights/，否则查交付目录上一层的同名文件，方便pyz和权重一起上传work。metadata默认交付目录`bundled_weights_provenance.json`，完整SHA键为`sha256`；显式SHA与metadata同时出现时必须相同。二进制默认`liblinear/bin/train`与predict。默认输出`runs/nwpu_smoke_01`，重跑需新名字。

pyz bootstrap可通过`DFAGCN_DELIVERY_ENTRY`传原始pyz路径，inspect生成的run_command便使用该入口。bootstrap负责代码解包/交付清单/原生编译，运行器不隐式获取任何输入。

## 离线验证边界

9项标准库测试覆盖精确取样、坏manifest、ZIP完整/单图SHA、仅恢复所选、已有输出拒绝、路径/歧义拒绝、GPU UUID/型号/占用拒绝、完整权重身份、inspect无训练写入和run必填UUID。GPU信息用外部I/O边界fixture验证门禁，不宣称本机CUDA运行。真实GPU smoke等待用户开启实例后执行。
