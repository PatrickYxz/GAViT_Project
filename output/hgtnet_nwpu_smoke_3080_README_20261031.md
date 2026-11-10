# HGTNet：3080真实NWPU小样本检查

本次运行是工程检查：完整独立HGTNet、固定seed42、train360/val90/test0、batch2、1 epoch、Adam lr1e-4、FP32，无增强/AMP/TF32。它验证实际图片、GPU前后向、validation、checkpoint保存与数值回读；不用于论文准确率比较，不接正式训练。

## 上传两个文件

上传至 `/home/featurize/work/`，文件名保持原样：

1. 运行包：`hgtnet_nwpu_smoke_3080_20261031.pyz`（52,238字节）。SHA256 `e798372589f83359f6d33d3c78d48ea1a82a760310b23f26aa0d6e689835ce78`。
2. 新权重：`swin_b-68c6b09e.pth`（351,654,207字节）。本地位于 `/Users/patrick/Projects/GAViT_Project/checkpoints/hgtnet_pretrained_20261031/swin_b-68c6b09e.pth`。

权重完整SHA256：`68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893`。这是官方torchvision Swin-B候选初始化，不能用之前的VGG或Swin-T权重替代，也不是经过核验的作者HGTNet权重。包不在线下载或安装依赖。

原有两个文件直接复用：

- `/home/featurize/work/nwpu_protocol_20260927_02.json`，SHA `ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`。
- `/home/featurize/work/NWPU-RESISC45_a826b44.zip`，SHA `beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53`。已有完整450张选定图片时可直接使用，缺图时run只恢复选定train/val图片到实例本地盘。

## 先运行这一行

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_nwpu_smoke_3080_20261031.pyz inspect
```

预检仅使用标准库，检查完整权重SHA、清单、数据来源、依赖版本、当前显卡和空间，不启动训练。它选择当前唯一空闲10GiB RTX3080，并生成绑定本次UUID的 `run_command`，不锁之前的实例UUID。

- `ready: true`：复制输出中的完整 `run_command` 执行。
- `ready: false`：查看 `blockers` 与对应 `*_error`；把完整预检输出贴回即可定位，不要盲目重装Torch或重传已有文件。
- `data_source: persistent_zip_pending_full_hash_and_selected_recovery`：表示已找到ZIP，完整hash/450图恢复由run执行；不表示上传失败，也不表示数据完整性已确认。
- Torch metadata支持`2.2.2`或`2.2.2+cu121`，vision支持`0.17.2`或`0.17.2+cu121`；run仍严格检查实际CUDA12.1/cuDNN8902及已记录软件条件。

## 运行和回传

首次以生成的命令前台运行。看到 `HGTNET_SMOKE_DEVICE` 后，还须看到 `first_optimizer_step`：它只在真实forward/backward/参数更新成功后打印，并保存首批梯度和各模块更新证据。程序会完成全部360张训练、90张验证与独立模型checkpoint回读。

结束标记为 `HGTNET_NWPU_SMOKE_COMPLETE`。随后运行这一行，回传短摘要：

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_nwpu_smoke_3080_20261031.pyz status
```

必须同时有 `state: complete`、`completed: true`、`exit_code: 0`，并核对train360/val90/test0、180个optimizer steps、epoch1、checkpoint回读误差≤1e-6。单看PID、首批或某个进度百分比都不是完成证据。出现错误会保留真实已开始状态，不自动改batch或重跑。

## 文件和空间

安装源码、run.log、preflight/runtime/selected inputs、候选加载报告、binding、first_step、history、val predictions、checkpoint及summary/run.exit在持久化目录：

`/home/featurize/work/hgtnet_nwpu_smoke_3080_20261031/runs/nwpu_smoke_seed42_01`

ZIP恢复目录在实例本地：`/tmp/hgtnet_nwpu_smoke_3080_20261031_seed42_01/images`。实例归还后该目录会丢失；已保存的work结果保留。

work预留2GiB、scratch预留1GiB，同卷时合并3GiB。checkpoint约299MB且只有一份（本次best/final均为epoch1），不创建旧DFAGCN的全量特征缓存/SVM文本。已有output或scratch拒绝覆盖；无自动删除和续训。实际耗时/显存由服务器本轮记录，当前本地检查不能估算租卡费用。

本地验证与来源边界见 `results/evidence/hgtnet_3080_smoke_prepare_20261031/README.md`。目标3080与真实NWPU检查当前仍待执行。

本地137项HGTNet与148项DFAGCN相关测试共285项通过，实际最终pyz预检在禁止ML导入时通过，完整HGTNet生成JPEG夹具的训练/validation/独立模型回读误差0；这些不代替目标GPU门禁。
