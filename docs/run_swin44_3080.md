# RTX 3080：Swin44 正式训练（seed44 配对，GAViT44 已完成）

2026-09-10。背景：GAViT44 已在同一块 RTX 3080 上完成（best epoch 19，Val mAP 77.9%；Test mAP 70.2%，见 research_diary 2026-09-10 条目与 results/bigearth_comparison.csv）。按配对原则，Swin44 在**同一块 3080** 上执行，超参与 Swin43 一致（30 epochs、batch32、lr 3e-4、AdamW、CosineAnnealingLR、seed 44）。评估 loader 的 pin_memory OOM 修复已包含在代码中（commit `36345f7`，Swin 验证同样受益）。

在 Featurize 终端整段复制执行。流程：`git pull --ff-only` 后 `git checkout` 到记录 commit `a23a5663e7bef9749143e9bb91932edaea2884bd`（detached HEAD；其后的 docs-only 提交不影响代码一致性）；preflight 校验 commit、GPU、预训练权重与全部 RGB 文件。训练体写入独立目录，退出码落盘 `train.exit`，主机可用内存每 30 秒记录到 `memavail.log`。

```bash
bash <<'BASH'
set -euo pipefail
cd /home/featurize/work/GAViT_Project
git pull --ff-only
git checkout a23a5663e7bef9749143e9bb91932edaea2884bd
swin_run=/home/featurize/work/swin44_3080_20260910
mkdir "$swin_run"

python -u - <<'PY' 2>&1 | tee "$swin_run/preflight.log"
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
require(commit == 'a23a5663e7bef9749143e9bb91932edaea2884bd' and not dirty,
        'STOP: code identity changed; expected the recorded seed44 commit; do not reset or overwrite it.')
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

cat > /tmp/swin44_train_3080.sh <<'TRAINER'
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
  --model swin --data_dir bigearth_files/splits \
  --epochs 30 --batch_size 32 --lr 3e-4 --seed 44 \
  --run_stage formal --run_tag rtx3080_seed44_20260910 \
  --pretrained_path bigearth_files/model.safetensors \
  --checkpoint_path "$swin_run/swin.pth"
TRAINER
sed -i "s|\$swin_run|$swin_run|" /tmp/swin44_train_3080.sh

nohup bash -c 'bash /tmp/swin44_train_3080.sh; echo "train_exit=$?" > /home/featurize/work/swin44_3080_20260910/train.exit' > "$swin_run/train.log" 2>&1 < /dev/null &
swin_pid=$!
printf '%s\n' "$swin_pid" > "$swin_run/train.pid"

nohup bash -c 'while :; do printf "%s avail=%sMiB\n" "$(date +%F_%T)" "$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)"; sleep 30; done' > "$swin_run/memavail.log" 2>&1 < /dev/null &
mon_pid=$!
printf '%s\n' "$mon_pid" > "$swin_run/mem.pid"

printf 'Launch submitted. train PID=%s, mem-monitor PID=%s\nOutput=%s\n' "$swin_pid" "$mon_pid" "$swin_run"
sleep 5
ps -p "$swin_pid" -o pid,etime,args || true
nvidia-smi
tail -n 30 "$swin_run/train.log"
BASH
```

`Launch submitted` 只表示后台命令已提交，不代表启动成功。必须同时确认进程仍在、GPU 有显存占用、日志出现 Swin 配置及首个训练 batch。首次导入/读 CSV 可能超过 5 秒，若日志仍在初始化，稍后查看同一个日志，不重复启动。

查看后续日志：`tail -n 40 /home/featurize/work/swin44_3080_20260910/train.log`。训练结束后检查 `cat .../train.exit`（应为 `train_exit=0`；文件不存在则说明进程被硬杀，`memavail.log` 可查死亡前内存轨迹）和 `kill $(cat .../mem.pid)` 停止内存监控。

输出包括 `preflight.log`、`train.log`、`train.exit`、`train.pid`、`memavail.log`、`mem.pid`、best `swin.pth`、`swin.meta.json` 和 `swin.last.train_state.pth`。只有训练、验证及这些产物经过检查，才可判断完成。测试 best checkpoint 另行执行；本命令不自动测试。

预计耗时：Swin43 在 4090 上 30 轮 5.22 小时（~10.4 分钟/轮）；3080 约为 4090 的一半算力，估计每轮 15~18 分钟（含验证），30 轮约 8~10 小时。Swin 无 GAT 分支，显存与内存压力均小于 GAViT。训练硬件 RTX 3080 与 GAViT44 相同，满足同 seed 同硬件配对；Swin43（4090）与 Swin44（3080）的跨硬件差异仅需在资源披露中说明，指标可比性不受影响。
