# HGTNet RTX3090 小样本检查

用户本次租到RTX3090：UUID `GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1`，总24576MiB/回传空闲24123MiB/驱动610.57.04。旧包仅接受RTX3080/10240MiB，不能直接用于本次卡。已生成独立3090包，旧包和旧工件保留。

## 只换小运行包

将本地 `output/hgtnet_nwpu_smoke_3090_20261102.pyz` 上传到：

`/home/featurize/work/hgtnet_nwpu_smoke_3090_20261102.pyz`

包52719字节，SHA256 `33e6f0a9ab449b314ba62b6961389db2d7464039c598b7775f7fef881bda19ff`。

若原Swin-B权重已上传至 `/home/featurize/work/swin_b-68c6b09e.pth`，不必重传。否则仍上传同一份351654207字节文件，完整SHA256 `68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893`；本地在 `checkpoints/hgtnet_pretrained_20261031/swin_b-68c6b09e.pth`。不改用旧VGG/Swin-T文件，不安装或在线获取依赖。

## 先预检

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_nwpu_smoke_3090_20261102.pyz inspect
```

本包manifest绑定3090，自动识别当前唯一空闲3090/24576MiB，输出当前UUID绑定的 `run_command`。用户本次UUID来自实际NVIDIA回传，程序仍重新查询，不将它硬编码以免下一实例变化。

`ready: true` 时直接复制输出的 `run_command` 执行；`ready: false` 时把完整输出贴回。检查使用stdlib，不启动Torch或训练；run实际导入后仍验证Python/Torch/vision/CUDA12.1/cuDNN8902、精度、显卡可见性/名称/显存和UUID等条件。版本检查沿用已有规则，不因换卡放宽。

## 内容保持一致

- 完整独立HGTNet、原Swin-B候选初始化、原manifest/ZIP和选样规则。
- seed42、batch2、train360/val90/test0、一轮180steps、Adam1e-4/CE、RGB224 ImageNet normalize。
- 全参数更新、无增强、AMP与两种TF32关闭。显存增加不自动加大batch或改训练设置。
- 训练/验证/保存/独立模型数值回读通过后才complete；本次仍是工程门禁，不是正式实验或论文准确率。

代码、日志和结果保存于新目录：

`/home/featurize/work/hgtnet_nwpu_smoke_3090_20261102/runs/nwpu_smoke_seed42_01`

ZIP恢复图片保存于新实例本地目录：

`/tmp/hgtnet_nwpu_smoke_3090_20261102_rtx3090_seed42_01/images`

work2GiB/scratch1GiB预算、防覆盖、不自动重跑规则沿用。已有manifest `/home/featurize/work/nwpu_protocol_20260927_02.json` 和ZIP `/home/featurize/work/NWPU-RESISC45_a826b44.zip`直接复用；缺原图时run完整SHA校验后只恢复450张train/val，test0。

## 完成后短摘要

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_nwpu_smoke_3090_20261102.pyz status
```

须同时 `state: complete`、`completed: true`、`exit_code: 0`，核对180steps、360/90/test0、checkpoint回读误差≤1e-6。首批日志只在实际forward/backward/step成功后打印。实际3090显存/耗时/工程结果仍待本次服务器运行，不继承3080或CPU的测量结果。

本地显卡/入口/打包39项检查通过，旧包及模型/数据/引擎/配置/权重SHA保持。全量数值回归因本机旧临时环境已缺Torch/NumPy未跑通，完整失败记录保存在 `results/evidence/hgtnet_3090_prepare_20261102/README.md`；这不代表目标服务器缺少依赖，须以新包预检和实际运行判断。
