# HGTNet × AID 只读资产检查包

本包只检查下一阶段 smoke 所需输入，不包含训练入口，不启动训练或 CUDA 计算，不安装依赖、不下载、不重新划分数据、不写服务器文件。需要服务器现有 Python 3.9+；仅使用标准库。

交付文件：`hgtnet_aid_inputs_20261104.pyz`，6520 字节，SHA256：

```text
1cd2ab967cdbc386b294b397c6804bf06c5796c01332c709e44646f98d03f036
```

把包上传到 `/home/featurize/work/` 后，复制这一整行执行即可；JSON 直接显示在终端：

```bash
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_inputs_20261104.pyz inspect --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --weights /home/featurize/work/swin_b-68c6b09e.pth
```

若 AID 已解压到其他目录，只替换 `--data-root` 后的绝对路径。该目录下应直接有 Airport、Industrial 等 30 个类别目录。`--manifest` 和 `--weights` 也可指向原文件的新绝对位置；预期 SHA 固定，不提供更换协议或跳过哈希的选项。路径及其父目录不能使用符号链接。

检查原 manifest 的 SHA、schema2、AID 30 类、固定 group-v2 协议/seed42与4242、4000 train/1000 val/5000 test、标签/路径/字节及像素摘要字段、像素组不跨类或 split；pool5000 由 train+val 得出。AID 各类数量不同，工具按每类实际数量核验原协议比例。不会生成新 split，也不会解码图像或重新计算像素 SHA。

按原 manifest 的路径顺序，每类选8张 train、2张 val，共240/60；只读取这300张图片做文件字节 SHA。test5000只核验 manifest 元数据，不打开 test 图片。Industrial 同像素组保留在原 split，不自行去重。

对候选 Swin-B 权重做完整流式 SHA（预期 `68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893`）；它是独立实现的部分初始化候选，不能替换成已训练的 NWPU 45 类 checkpoint。工具只通过 `nvidia-smi` 查询当前实际 GPU，通过 distribution metadata 查询依赖版本，并报告磁盘容量。不会猜 GPU UUID、原 AID ZIP 名称/来源或未知 ZIP SHA。

缺文件、坏 SHA、坏元数据、图片缺失或路径不安全都会在 JSON 中明确报告，其余盘点继续。完成盘点的退出码为0，**退出码0不表示资产通过**；非法命令行返回非零。主要字段：

- `status`：`INPUT_CHECK_INCOMPLETE` 表示资产仍缺失或未通过；`INPUT_ASSETS_VERIFIED_SMOKE_PREPARATION_PENDING` 仅表示上述输入资产通过，仍需实现并验证 smoke。
- `manifest` / `weights`：查看 `status`、`errors` / `error`、实际与预期 SHA；图片缺失/不符见 `manifest.image_checks`。
- `gpu`、`package_versions`、`disk`：保留实际可见状态；磁盘路径不存在时明确标记测量的是哪个已存在父目录。
- `training_started:false`、`training_entry_available:false`：本阶段没有训练启动或训练入口。

下一步只需回传这次命令显示的完整 JSON 文本，或保存后的同内容小 JSON 文件，便于确认原 manifest、300张候选图片、初始权重与当前实例资源。无需回传 test 图片、模型大文件或 AID 原 ZIP。资产缺失时先定位原件，不下载猜测版本或重新生成 split。

本地验证：13项标准库 unittest 通过；实际最终 zipapp 在禁止非标准库导入与禁止文件修改的 guard 下，以本机缺资产路径完成 JSON 盘点、退出0；原 NWPU 正式包 SHA 保持 `b2bc9077cf37e75c83fbb72e7c4047d2fb9ede04db8fb4e4a56344500ff572e1`。尚未在目标服务器检查真实 AID 或运行 GPU smoke，不能据本地夹具宣称数据或训练已通过。
