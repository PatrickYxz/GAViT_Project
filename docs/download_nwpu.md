# NWPU-RESISC45 服务器直接下载

2026-09-27。Featurize目录未搜到RESISC45；用户提供的VHR-10/RSOD包不适用于本项目的场景分类。此方案不需要电脑中转，不依赖新训练代码部署。

**服务器回传更新（09-27）**：ZIP已下载427389445/427389445字节（wget报告75秒），SHA256校验`OK`；解压后`NWPU_READY: classes=45 images=31500 each_class=700 /home/featurize/data/NWPU-RESISC45`。随后group策略prepare完成逐图解码/哈希与划分，train5040/val1260/test25200；8组同类重复均在单个split内，31500文件保留、31492个唯一像素图像。固定`nwpu_protocol_20260927_02.json`供后续三模型共用，详见[执行说明](run_soft_assignment_smoke.md)。以下下载命令保留为历史及原图丢失后的恢复说明，当前无需重复下载、解压或prepare。

## 来源与当前核验

- [TorchGeo 0.6.1 RESISC45源码](https://docs.torchgeo.org/en/v0.6.1/_modules/torchgeo/datasets/resisc45.html)给出固定版本ZIP、目录`NWPU-RESISC45`和MD5 `75206b2e16446591afa88e2628744886`。
- 其`torchgeo/resisc45`仓库地址当前重定向至`isaaccorley/resisc45`；固定revision为`a826b44d938a883185f11ebe3d512d38b464312f`。[发布者说明](https://huggingface.co/datasets/isaaccorley/resisc45/blob/main/README.md)称只将原RAR转换为ZIP，不修改图像。
- 当前对该固定revision的HF tree API读取成功：`NWPU-RESISC45.zip`为427,389,445字节，LFS SHA256为`beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53`。[公开文件校验页](https://huggingface.co/datasets/isaaccorley/resisc45/blob/main/NWPU-RESISC45.zip)相同。
- 初次准备时，镜像`hf-mirror.net`上该revision的HEAD本地联网核验返回200、`application/zip`、相同Content-Length；随后用户服务器下载及哈希成功，见上方回传更新。本地没有下载完整427MB或核验所有图像。
- 只取原始图片ZIP，不使用TorchGeo附带的train/val/test清单。项目后续仍以自己的manifest/seed协议生成划分，不能把外部划分混入既定三模型比较。

## 逐行执行

1. 下载到持久化work。固定新文件名，支持断点续传，沿用此前AID成功使用的镜像和禁用代理方式。

```bash
wget --no-proxy -c --timeout=30 --tries=3 -O /home/featurize/work/NWPU-RESISC45_a826b44.zip https://hf-mirror.net/datasets/isaaccorley/resisc45/resolve/a826b44d938a883185f11ebe3d512d38b464312f/NWPU-RESISC45.zip
```

2. 校验一致后才解压到实例data；`-n`保留已存在文件。

```bash
printf '%s\n' 'beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53  /home/featurize/work/NWPU-RESISC45_a826b44.zip' | sha256sum -c - && unzip -n -q /home/featurize/work/NWPU-RESISC45_a826b44.zip -d /home/featurize/data
```

3. 检查解压层级、45类和每类700张图片。

```bash
python -c 'from pathlib import Path; p=Path("/home/featurize/data/NWPU-RESISC45"); c={d.name:len(list(d.glob("*.jpg"))) for d in p.iterdir() if d.is_dir()}; assert len(c)==45 and set(c.values())=={700},c; print("NWPU_READY: classes=45 images=31500 each_class=700",p)'
```

将第2行的`OK`和第3行的`NWPU_READY`输出回传。它们只证明压缩包身份和解压后的数量；接下来prepare仍需逐图解码/文件及像素哈希/重复检查，不先开训。若校验或解压失败保留文件和输出，不重新生成split绕过。

这三行没有使用需跨会话保存的shell变量，没有安装依赖或下载配套脚本。以后实例原图消失，可复用work中的已校验ZIP执行第2、3行，无需重新下载。GPU和新代码部署状态仍需单独核对。
