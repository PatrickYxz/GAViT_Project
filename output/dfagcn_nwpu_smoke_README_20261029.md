# DFAGCN真实NWPU小样本：离线交付

2026-10-30更新：本包真实NWPU360/90/450工程检查已回传exit0和completed=true，交付身份核对通过，重载误差0，RTX4090总耗时26.26秒。[结果汇总](../results/dfagcn_nwpu_smoke_20261030.md)已保存，本批无需重跑；下方上传/预检步骤保留为操作记录。该检查没有运行真实test，不产生正式比较分数。

## 上传两个文件

实例开启后选择RTX4090，将下列文件上传到`/home/featurize/work/`，保留文件名：

- [dfagcn_nwpu_smoke_20261029.pyz](dfagcn_nwpu_smoke_20261029.pyz)：134156字节，含冻结代码、配方、完整权重哈希和官方LIBLINEAR源码；SHA256为`f7ff678be35517aca7edc3a181862a56d739cc0be1a7496a5c63dac4b269daca`。
- [vgg16-397923af.pth](../checkpoints/dfagcn_pretrained_20261029/vgg16-397923af.pth)：553433881字节，约528MiB；SHA256为`397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`。

这份VGG权重来自torchvision官方来源；实际CPU核对FC6与三个卷积层相对标准torchvision实现最大误差均0。它是已披露独立版本的预训练选择，不证明原MatConvNet等价。

## 先编译，再检查

每条都是独立一行，无Shell变量。第一次运行会在同名目录解出并校验代码，不修改旧GAViT目录。

```sh
python /home/featurize/work/dfagcn_nwpu_smoke_20261029.pyz build-linear
```

```sh
python /home/featurize/work/dfagcn_nwpu_smoke_20261029.pyz inspect
```

build-linear只在当前平台编译包内官方LIBLINEAR v250源码。inspect只检查资源，不下载、恢复数据或启动训练；它检查权重完整SHA、原manifest、数据或ZIP、依赖位置/版本、实际RTX4090 UUID及已有GPU进程。

只有`ready: true`时才会打印`run_command`。将该值的完整一行复制到同一终端；它已绑定本次实际GPU UUID。不要沿用历史GPU UUID。

如果`blockers`非空，先按输出补齐对应输入，再运行inspect。不要因此重新生成数据划分。输出已存在时保留目录，先检查已有运行，不覆盖。

## 数据与运行范围

历史默认路径如下，是否仍存在以新实例inspect结果为准：

- manifest：`/home/featurize/work/nwpu_protocol_20260927_02.json`，固定SHA `ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`。
- 原图：`/home/featurize/data/NWPU-RESISC45`。
- 持久ZIP：`/home/featurize/work/NWPU-RESISC45_a826b44.zip`，固定SHA `beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53`。

若实例本地原图已清空，run会先验证ZIP完整SHA，再仅恢复所选450张，不必先解压31500张。随后在CUDA提取pool450一次，派生train360/val90缓存，完成一轮校准与fresh subset refit及LIBLINEAR。默认配方为7×7、k10、GCN256×2、MLP1024→512、batch8、seed42、FP32、关闭AMP/TF32。

看到`DFAGCN_NWPU_DEVICE`后，实际提取batch会打印`DFAGCN_NWPU_VGG_BATCH`，实际训练batch会打印`DFAGCN_TRAIN_BATCH`。这些连同GPU进程才是执行证据；资源预检不等于前向或训练已通过。

## 完成后回传

成功须打印`DFAGCN_NWPU_SMOKE_COMPLETE`，保存`run.exit`的exit_code0、summary、完整一轮calibration/refit history、checkpoint/线性模型、重载误差及GPU/耗时/峰值显存。

```sh
python -c 'import json,sys; print(json.dumps([json.load(open(sys.argv[1])),{k:v for k,v in json.load(open(sys.argv[2])).items() if k in sys.argv[3:]}],indent=2))' /home/featurize/work/dfagcn_nwpu_smoke_20261029/runs/nwpu_smoke_01/run.exit /home/featurize/work/dfagcn_nwpu_smoke_20261029/runs/nwpu_smoke_01/summary.json completed training_started counts physical_gpu_uuid runtime manifest_sha256 weights_sha256 pipeline_source_sha256 recipe_sha256 selected_epoch selected_C reload_max_abs_error cuda_peak_allocated_bytes cuda_peak_reserved_bytes pooled_feature_extraction_seconds calibration_seconds refit_seconds wall_seconds
```

此命令只打印退出码及关键字段，省略混淆矩阵，避免终端输出截断。把输出贴回聊天后，再核验结果和实际资源。失败会保留failure/traceback/exit；不要覆盖原目录或直接扩大到正式训练。

## 本地验证

105项DFAGCN包/交付测试通过；实际pyz解包、代码复用/变更拒绝、官方源码编译、权重完整校验和预检均已运行。本机预检正确停在缺服务器manifest和RTX4090，未创建训练目录。新测试含已初始化CUDA上下文拒绝及运行时配方变更拒绝。

尚未验证Linux编译、真实NWPU、目标CUDA或正式性能。全旧仓库发现检查204项仍有9failure/9error，缺pytest/timm/torch_geometric；[详细证据](../results/evidence/dfagcn_nwpu_prepare_20261029/validation.json)单独保留，不能称全仓库通过。
