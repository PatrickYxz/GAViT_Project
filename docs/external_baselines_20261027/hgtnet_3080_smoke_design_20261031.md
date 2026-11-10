# HGTNet RTX3080 real NWPU smoke

用户已同意上一轮下一步：准备3080运行包并做真实NWPU小样本工程门禁。本轮交付代码/本地验证/便携包，不直接操作服务器或租GPU。沿用已通过的独立模型结构；不重开模型选择或额外批准循环。

## 决策与边界

采用独立小zipapp而非服务器手改/整仓库依赖安装。复用既有NWPU只读helper（tools/run_dfagcn_nwpu_smoke.py）的manifest、360/90固定选样和受SHA约束的分阶段ZIP恢复；不会运行其DFAGCN训练入口。只打包所需HGTNet源码和数据helper，无LIBLINEAR/旧DF模型或在线下载。新权重单独上传351654207B官方候选Swin-B；不使用旧VGG或Swin-T文件。

只做一个全参数训练epoch，seed42、batch2、Adam lr1e-4、CE，Resize224 bilinear/RGB/ImageNet normalize，无数据增强；验证90、test0。此优化设置为一次性工程检查，不是正式论文recipe。模型和配置保持上一轮af6ef4...身份；不冻结骨干、不AMP/TF32、不自动降batch/改结构。真实GPU峰值和耗时必须运行后取得。

预检inspect仅stdlib：完整manifest/配置/权重SHA与文件，选中图片hash或完整ZIP存在、依赖metadata、空闲10GiB3080、work至少2GiB/scratch至少1GiB（同卷合并），独立新output/scratch路径。若图片缺但ZIP存在，说明pending full hash/recovery而非文件未上传；run才完整hash ZIP并仅恢复450图。唯一合格空闲卡自动选择，生成绑定此次UUID的单行run命令，不锁旧UUID；run必须明确UUID且实际导入后核对CUDA12.1/cuDNN8902、Torch2.2.2(+cu121)/vision0.17.2(+cu121)、NumPy1.26.4/Pillow10.3.0与物理身份。

模型/代码/log/选样/预训练报告/历史/最终summary/checkpoint放work；恢复图片放新scratch下，不覆盖旧文件。run没有test入口、正式训练、自动重试、epoch恢复或清理历史。首批只有forward/backward/step成功才写实际training_started/progress，不能凭PID或调用函数宣称成功。

## 文件与接口

- external_baselines/hgtnet/nwpu_data.py：stdlib数据检查，重用旧helper；prepare_inputs(manifest_path, expected_sha256=MANIFEST_SHA)返回class_names、selected(train/val/pool)、counts和manifest_sha256。validate_selected/recover_selected透传。make_dataset(root,records)延迟导入Torch/PIL，返回真实RGB归一化224 tensor和int label；不随机增强。helper原文件不修改。
- external_baselines/hgtnet/resources.py：stdlib inventory_gpu()；select_gpu(requested_uuid,inventory)只接受空闲唯一10GiB RTX3080，返回gpu dict；dependency_inventory() + check_runtime(runtime,gpu_uuid)。runtime字段沿用旧GPU探针（torch/torchvision/numpy/pillow/cuda/cudnn/gpu.uuid/device/amp/tf32等）。capacity(output,scratch)与require_new_paths保证分卷预算和路径不冲突。inspect用metadata，run必须验证真实imports与设备。
- external_baselines/hgtnet/nwpu_engine.py：Torch训练/验证与checkpoint数值回读，支持generic模型/小CPU fixture做真实本地行为测试；CLI只允许绑定CUDA运行。保存真实数据metadata，不误用旧checkpoint固定synthetic=true身份。完成所有450样本、validation、checkpoint数值回读、finite及身份检查后才可summary completed=true。
- tools/run_hgtnet_nwpu_smoke.py：stdlib inspect/status入口、lazy CUDA run、短摘要、run.exit/progress、单行命令。保持旧核心九个HGT源码不变，增加新文件。
- tools/package_hgtnet_nwpu_smoke.py：独占构建/安装manifest校验zipapp；metadata/权重文件不打包大blob。actions inspect/run/status；由同名目录安装，源码更改/外来目录拒绝覆盖。

默认包/home/featurize/work/hgtnet_nwpu_smoke_3080_20261031.pyz，默认安装同名无后缀目录，output runs/nwpu_smoke_seed42_01，scratch /tmp/hgtnet_nwpu_smoke_3080_20261031_seed42_01，权重work/swin_b-68c6b09e.pth。manifest、ZIP沿用既有固定路径。

## 验证与验收

先行为测试RED→GREEN，真实CPU小图片读取/训练/validation/保存/回读，真实full HGTNet使用已有候选权重做小fixture接线检查（不称真实NWPU）。mock只在NVIDIA与磁盘/远端边界；CLI预检在禁止ML imports进程仍可运行，实际pyz解包、manifest与源码一致、防覆盖和status not_started。独立审查后运行全部相关DF/HGT tests，比较旧52文件哈希和上一轮6模型源码/配置/权重保持。更新日记与可复制README。无目标3080回传前，只称“运行包已本地验证”，不称GPU门禁通过或有HGT准确率。
