# RTX 3080：重启 GAViT44 正式训练（含 pin_memory 修复）

2026-09-09。背景：2080 Ti 上 GAViT44 在 epoch1 训练段结束后、首次验证期间被内核 OOM killer 静默杀死（诊断链与单变量实验证实根因为验证 loader `pin_memory=True` 的锁页主机内存池在 24GB 实例上耗尽 RAM；修复提交 `36345f7d5ffccf714992120abe8e4f6b52d438c7`，仅将 `train_bigearth.py`/`test_bigearth.py` 的评估 loader 改为 `pin_memory=False`，纯资源改动，数据/模型/指标语义不变）。全量验证诊断（122342 图）在修复后 PASS 且内存平稳。

在 Featurize 终端整段复制执行。启动前会先 `git pull --ff-only` 拉取修复提交；preflight 校验新 commit、GPU、预训练权重与全部 RGB 文件。训练体写入独立目录，退出码落盘 `train.exit`，主机可用内存每 30 秒记录到 `memavail.log`。

```bash
bash <<'BASH'
set -euo pipefail
cd /home/featurize/work/GAViT_Project
git pull --ff-only
gavit_run=/home/featurize/work/gavit44_3080_20260909
mkdir "$gavit_run"

python -u - <<'PY' 2>&1 | tee "$gavit_run/preflight.log"
import csv, hashlib, importlib.metadata, subprocess, sys
from pathlib import Path
import torch

def require(ok, message):
    if not ok:
        raise SystemExit(message)

def digest(filename):
    value = hashlib.sha256()
    with Path(filename).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], text=True).strip()
require(commit == '36345f7d5ffccf714992120abe8e4f6b52d438c7' and not dirty,
        'STOP: code identity changed; expected the pin_memory hotfix commit; do not reset or overwrite it.')
busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
require(not busy, 'STOP: another GPU process is running.')
require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'STOP: one working CUDA device is required.')
require(torch.cuda.get_device_name(0).endswith('RTX 3080'), 'STOP: GPU is not RTX 3080.')
torch.zeros(1, device='cuda').add_(1)
torch.cuda.synchronize()
print('Commit:', commit, 'Python:', sys.version)
print('GPU:', torch.cuda.get_device_name(0), 'CUDA runtime:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version())
print('Packages:', {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'torch-geometric', 'numpy', 'scikit-learn', 'rasterio')})
print('Pretrained SHA256:', digest('bigearth_files/model.safetensors'))
seen = set()
header = ['patch_path', *[f'label_{i}' for i in range(19)]]
for split, expected in (('train', 237871), ('val', 122342), ('test', 119825)):
    filename = Path('bigearth_files/splits') / f'{split}.csv'
    count = 0
    print('Checking RGB files:', split, flush=True)
    with filename.open(newline='') as handle:
        reader = csv.reader(handle)
        require(next(reader, None) == header, f'STOP: invalid header in {filename}')
        for row in reader:
            require(len(row) == 20 and row[0] and row[0] not in seen, f'STOP: invalid/duplicate sample in {filename}')
            require(all(x in ('0', '1', '0.0', '1.0') for x in row[1:]), 'STOP: invalid labels.')
            seen.add(row[0])
            patch = Path(row[0])
            require(patch.is_absolute(), f'STOP: expected absolute patch path: {patch}')
            for band in ('B02', 'B03', 'B04'):
                image = patch / f'{patch.name}_{band}.tif'
                require(image.is_file(), f'STOP: missing {image}; finish extracting the dataset first.')
            count += 1
            if count % 50000 == 0:
                print(split, count, flush=True)
    require(count == expected, f'STOP: {split} count={count}, expected={expected}')
    print(split, count, 'OK; SHA256:', digest(filename), flush=True)
print('Preflight passed. Starting the formal run, not a smoke.', flush=True)
PY

cat > /tmp/gavit44_train_3080.sh <<'TRAINER'
cd /home/featurize/work/GAViT_Project || exit 1
python -u -c '
import resource, runpy, sys, time, torch
if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3080":
    raise SystemExit("STOP: expected RTX 3080 CUDA device; no CPU fallback.")
print("Trainer arguments:", sys.argv[1:], flush=True)
started = time.perf_counter()
torch.cuda.reset_peak_memory_stats()
try:
    runpy.run_path("train_bigearth.py", run_name="__main__")
finally:
    print("Training wall hours:", (time.perf_counter() - started) / 3600, flush=True)
    print("Peak CUDA allocated GiB:", torch.cuda.max_memory_allocated() / 2**30, flush=True)
    print("Peak CUDA reserved GiB:", torch.cuda.max_memory_reserved() / 2**30, flush=True)
    print("Main-process peak RSS KiB (not whole-job RAM):", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)
    print("Precision:", torch.get_default_dtype(), torch.get_float32_matmul_precision(), "matmul TF32:", torch.backends.cuda.matmul.allow_tf32, "cuDNN TF32:", torch.backends.cudnn.allow_tf32, "deterministic:", torch.backends.cudnn.deterministic, "benchmark:", torch.backends.cudnn.benchmark, flush=True)
' \
  --model gavit --data_dir bigearth_files/splits \
  --epochs 30 --batch_size 32 --lr 3e-4 --seed 44 \
  --num_regions 16 --grouping attentive_spatial --edge_type sparse_hybrid \
  --knn_k 5 --gat_hidden 256 --gat_heads 4 --gat_layers 2 \
  --integration token_feedback --dropout 0.1 \
  --run_stage formal --run_tag rtx3080_seed44_20260909 \
  --pretrained_path bigearth_files/model.safetensors \
  --checkpoint_path "$gavit_run/gavit.pth"
TRAINER
sed -i "s|\$gavit_run|$gavit_run|" /tmp/gavit44_train_3080.sh

nohup bash -c 'bash /tmp/gavit44_train_3080.sh; echo "train_exit=$?" > /home/featurize/work/gavit44_3080_20260909/train.exit' > "$gavit_run/train.log" 2>&1 < /dev/null &
gavit_pid=$!
printf '%s\n' "$gavit_pid" > "$gavit_run/train.pid"

nohup bash -c 'while :; do printf "%s avail=%sMiB\n" "$(date +%F_%T)" "$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)"; sleep 30; done' > "$gavit_run/memavail.log" 2>&1 < /dev/null &
mon_pid=$!
printf '%s\n' "$mon_pid" > "$gavit_run/mem.pid"

printf 'Launch submitted. train PID=%s, mem-monitor PID=%s\nOutput=%s\n' "$gavit_pid" "$mon_pid" "$gavit_run"
sleep 5
ps -p "$gavit_pid" -o pid,etime,args || true
nvidia-smi
tail -n 30 "$gavit_run/train.log"
BASH
```

`Launch submitted` 只表示后台命令已提交，不代表启动成功。必须同时确认进程仍在、GPU 有显存占用、日志出现 GAViT 正确配置及首个训练 batch。首次导入/读 CSV 可能超过 5 秒，若日志仍在初始化，稍后查看同一个日志，不重复启动。

查看后续日志：`tail -n 40 /home/featurize/work/gavit44_3080_20260909/train.log`。训练结束后检查 `cat .../train.exit`（应为 `train_exit=0`；文件不存在则说明进程被硬杀，`memavail.log` 可查死亡前内存轨迹）和 `kill $(cat .../mem.pid)` 停止内存监控。

输出包括 `preflight.log`、`train.log`、`train.exit`、`train.pid`、`memavail.log`、`mem.pid`、best `gavit.pth`、`gavit.meta.json` 和 `gavit.last.train_state.pth`。只有训练、验证及这些产物经过检查，才可判断完成。测试 best checkpoint 另行执行；本命令不自动测试或启动后续模型。

预计每轮约 35~40 分钟（2080 Ti 训练段 29.5 分钟 + 3080 上验证实测约 7 分钟），30 轮约 18~20 小时；跨硬件差异（2080 Ti 失败的 epoch1 不计入任何结果）与 pin_memory 修复均需在论文/记录中披露。主进程 RSS 不能代替包含 DataLoader 子进程的全作业内存；中断时保留产物，尤其不能把发生错误的运行报告为完成。
