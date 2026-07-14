# Featurize Runbook for GAViT

本手册记录 GAViT 在 Featurize 上运行 BigEarthNet 实验的已验证流程。项目原则和论文记录要求见根目录 `AGENTS.md`，实验结果见 `research_diary.md`。

## Storage Model

Featurize 有两类用途不同的目录：

| 内容 | 路径 | 生命周期 |
|---|---|---|
| 代码、日志、checkpoint | `/home/featurize/work/GAViT_Project` | 更换或归还实例后保留 |
| metadata、split CSV、预训练权重 | `/home/featurize/work/GAViT_Project/bigearth_files` | 更换或归还实例后保留 |
| BigEarthNet 原始图像 | `/home/featurize/data/BigEarthNet-S2` | 实例归还后删除 |

`work` 是网络同步盘，适合保存小文件和成果，但大量随机读取较慢。原始 BigEarthNet 必须放在实例本地 `data` 目录，不能移动到 `work`。

## New Instance Checklist

新实例开始工作时按顺序检查：

```bash
nvidia-smi

cd /home/featurize/work/GAViT_Project
git status --short --branch

ls -lh bigearth_files/metadata.parquet
ls -lh bigearth_files/model.safetensors
wc -l bigearth_files/splits/*.csv
ls -lh checkpoints logs

ls -ld /home/featurize/data/BigEarthNet-S2
```

预期 split 行数包含表头：

```text
237872 train.csv
122343 val.csv
119826 test.csv
```

如果最后一条命令找不到 BigEarthNet，说明实例本地数据已清空，需要重新添加平台数据集。不要因此重新生成 metadata 或 split CSV。

## BigEarthNet Data

在 Featurize 工作区点击左下角 Featurize 图标，选择“添加数据集”，搜索 `BigEarthNet`，下载并解压 `BigEarthNet-S2.zip`。通过底部“后台任务”等待解压完成。

不要使用网络挂载来训练；数百万个 TIF 文件需要实例本地随机读取性能。解压后的结构为：

```text
/home/featurize/data/BigEarthNet-S2/
  S2A_MSIL2A_..._T33UUP/
    S2A_MSIL2A_..._37_88/
      *_B02.tif
      *_B03.tif
      *_B04.tif
      ...
```

平台版本只包含 TIF，不包含旧版 `*_labels_metadata.json`。已验证文件数量为：

```text
6593856 tif files
```

解压后检查：

```bash
find /home/featurize/data/BigEarthNet-S2 -type f -name '*_B04.tif' | head
find /home/featurize/data/BigEarthNet-S2 -type f | wc -l
```

## Persistent Metadata and Splits

BigEarthNet v2 标签来自独立 metadata：

```text
https://zenodo.org/records/10891137/files/metadata.parquet?download=1
```

当前持久化文件应已存在：

```text
/home/featurize/work/GAViT_Project/bigearth_files/metadata.parquet
/home/featurize/work/GAViT_Project/bigearth_files/splits/train.csv
/home/featurize/work/GAViT_Project/bigearth_files/splits/val.csv
/home/featurize/work/GAViT_Project/bigearth_files/splits/test.csv
```

只有这些文件丢失时才重新生成：

```bash
cd /home/featurize/work/GAViT_Project
mkdir -p bigearth_files

wget -O bigearth_files/metadata.parquet \
  'https://zenodo.org/records/10891137/files/metadata.parquet?download=1'

python baselines/bigearth/prepare_bigearth.py \
  --data_dir /home/featurize/data/BigEarthNet-S2 \
  --metadata_parquet bigearth_files/metadata.parquet \
  --out_dir bigearth_files/splits
```

已验证输出：

```text
metadata rows: 480038
train: 237871
val: 122342
test: 119825
missing_patch: 0
missing_bands: 0
unknown_split: 0
```

如果 `missing_patch` 或 `missing_bands` 非零，先确认数据集是否仍在解压；不要使用不完整 split 开始训练。

## Python Dependencies

用户级 Python 包安装在 `/home/featurize/work/.local`，通常会随同步盘保留。新实例先验证，不要无条件重装：

```bash
python - <<'PY'
import numpy
import pandas
import pyarrow
import rasterio
import sklearn
import timm
import torch
import torch_geometric
print('All imports OK')
PY
```

缺少包时再安装：

```bash
pip install --user \
  timm torch-geometric scikit-learn pandas pyarrow tqdm \
  matplotlib seaborn rasterio pillow 'httpx[socks]'
```

安装 `httpx[socks]` 可以补齐 SOCKS 客户端依赖，但不能保证 Featurize 能稳定访问 Hugging Face。

## Local Swin Pretrained Weights

Featurize 访问 Hugging Face 时，`httpx` 和 `requests` 都出现过 `SSL UNEXPECTED_EOF` 或 `Connection reset by peer`。不要把正式训练依赖于在线下载。

持久化权重路径：

```text
/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors
```

验证文件和本地加载：

```bash
cd /home/featurize/work/GAViT_Project
ls -lh bigearth_files/model.safetensors

python - <<'PY'
import timm

path = 'bigearth_files/model.safetensors'
model = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=True,
    num_classes=0,
    pretrained_cfg_overlay={'file': path},
)
print('Local Swin weights loaded OK')
print('Parameters:', sum(p.numel() for p in model.parameters()))
PY
```

已验证 backbone 参数量为 `27519354`。

当前服务器使用未跟踪的 `train_bigearth_local.py` 作为临时入口，在 Swin `timm.create_model` 调用中加入：

```python
pretrained_cfg_overlay={"file": "bigearth_files/model.safetensors"}
```

如果临时入口不存在，先复制而不修改 Git 跟踪文件：

```bash
cp train_bigearth.py train_bigearth_local.py
sed -i \
  's|pretrained=True, num_classes=0|pretrained=True, num_classes=0, pretrained_cfg_overlay={"file": "bigearth_files/model.safetensors"}|' \
  train_bigearth_local.py
python -m py_compile train_bigearth_local.py
```

这个临时入口只覆盖 Swin baseline。正式代码后续应增加显式本地预训练路径参数，并同步支持 GAViT backbone；在完成该代码改动前，不要假设 GAViT 已经离线加载权重。

## Preflight Smoke Tests

先确认 dataset 已包含 list-to-NumPy 修复：

```bash
grep -n 'self.labels' models/bigearth_dataset.py
```

必须看到：

```python
self.labels = np.asarray(labels, dtype=np.float32)
```

数据读取测试：

```bash
python - <<'PY'
import csv
import numpy as np
from torchvision import transforms
from models.bigearth_dataset import BigEarthNetDataset

patches, labels = [], []
with open('bigearth_files/splits/train.csv') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        patches.append(row['patch_path'])
        labels.append([float(row[f'label_{j}']) for j in range(19)])
        if i == 3:
            break

dataset = BigEarthNetDataset(
    patches,
    np.asarray(labels, dtype=np.float32),
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
)
image, label = dataset[0]
print('image:', image.shape)
print('label:', label.shape, 'positive:', label.sum().item())
PY
```

建立一次性小 split：

```bash
mkdir -p /home/featurize/data/BigEarthNet-RGB_split_smoke
head -n 257 bigearth_files/splits/train.csv \
  > /home/featurize/data/BigEarthNet-RGB_split_smoke/train.csv
head -n 129 bigearth_files/splits/val.csv \
  > /home/featurize/data/BigEarthNet-RGB_split_smoke/val.csv
```

使用本地权重入口运行单 epoch：

```bash
python train_bigearth_local.py \
  --model swin \
  --data_dir /home/featurize/data/BigEarthNet-RGB_split_smoke \
  --epochs 1 \
  --batch_size 16
```

smoke test 必须完成训练、验证和 checkpoint 保存。开始正式训练前保护 smoke checkpoint：

```bash
mv checkpoints/best_bigearth_swin.pth \
  checkpoints/best_bigearth_swin_smoke.pth
```

如果正式 checkpoint 已存在，不要运行这条 `mv`；先确认文件时间和实验身份，避免覆盖有效结果。

## Full Training

Featurize 不是 SLURM 集群，不使用 `sbatch`。经过 smoke test 后，Swin baseline 的已验证后台命令为：

```bash
cd /home/featurize/work/GAViT_Project
mkdir -p logs

nohup python -u train_bigearth_local.py \
  --model swin \
  --data_dir /home/featurize/work/GAViT_Project/bigearth_files/splits \
  --epochs 30 \
  --batch_size 32 \
  > logs/bigearth_swin.log 2>&1 &
```

启动后记录 shell 返回的 PID。不要仅凭后台 PID 判断训练成功。

## Monitoring and Recovery

查看日志和进程：

```bash
tail -n 50 logs/bigearth_swin.log
tail -f logs/bigearth_swin.log
ps -p PID -o pid,etime,stat,%cpu,%mem,cmd
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

训练成功启动需要同时满足：

- 进程仍存在。
- Python 进程占用 GPU 显存。
- 日志出现模型、`Device: cuda` 和首个 epoch batch。

`Ctrl+C` 只退出 `tail -f`。关闭浏览器或终端不会停止 `nohup` 任务；停止或归还实例会终止训练。

训练结束后检查：

```bash
tail -n 80 logs/bigearth_swin.log
ls -lh checkpoints/best_bigearth_swin.pth
grep -E 'Epoch \[|Best Validation' logs/bigearth_swin.log | tail -n 15
```

`--resume` 只恢复模型权重，不恢复 optimizer 或 scheduler 状态。使用前先从日志确定中断 epoch，并在研究日志中记录恢复边界：

```bash
python train_bigearth.py --resume --start_epoch N [其他原始参数]
```

## Evaluation

换实例后先重新添加并完整解压 BigEarthNet。测试脚本以 `pretrained=False` 创建结构，再加载训练 checkpoint，因此不需要在线下载 ImageNet 权重：

```bash
cd /home/featurize/work/GAViT_Project

python test_bigearth.py \
  --model swin \
  --data_dir /home/featurize/work/GAViT_Project/bigearth_files/splits \
  --ckpt checkpoints/best_bigearth_swin.pth \
  --batch_size 32
```

正式记录 test mAP、macro-F1、micro-F1、per-class AP、阈值、GPU、运行时间和 checkpoint。测试结果必须写入 `research_diary.md` 和对应结果表。

## Known Failure Modes

### Git TLS 或本地修改阻止 pull

Featurize 网络可能出现 `gnutls_handshake`。重试前先检查工作区：

```bash
git status --short
```

远程临时修改阻止 `git pull` 时，只 stash 明确的临时文件：

```bash
git stash push -m 'featurize temporary patch' path/to/file.py
git pull
```

不要 stash、覆盖或删除来源不明的修改。

### Hugging Face TLS 失败

增加 timeout、安装 SOCKS 或把 `huggingface_hub` 从 1.x 降到 0.x 都未解决实例 TLS 中断。使用持久化本地 `model.safetensors`，不要继续重复付费排查同一下载路径。

### 浏览器上传卡住

上传权重时若百分比长时间不变，在终端检查服务器文件大小：

```bash
ls -lh bigearth_files/model.safetensors
```

如果大小停止增长，终止任务并保留残片，再使用 Chrome 重传。完整文件已验证约为 `109M`，且必须通过 `timm` 本地加载测试；不能只凭文件名判断完整。

### 新实例缺少数据

`work` 中的代码和成果仍在，但 `/home/featurize/data` 会清空。重新添加平台 BigEarthNet 数据集即可；不要重新制作已持久化的 metadata 和 splits。

### checkpoint 被覆盖

smoke test 和正式训练默认使用相同 checkpoint 名。启动前查看文件时间、大小和日志，并给 smoke checkpoint 改名。不要覆盖已经验证的 `best_bigearth_swin.pth`。
