# 2080 Ti：直接启动 GAViT seed44

2026-09-08。用户已回传：2080 Ti 22528MiB、无GPU进程、commit bb9f1cf、Git无改动输出、权重109M，train/val/test CSV行数正确。目录存在尚不能证明解压完整，下面在同一启动流程中检查全部RGB文件；不执行独立smoke、不改变训练入口、不自动启动Swin或seed45。

在 Featurize 终端整段复制执行。无需上传旧工具包。输出目录固定且使用 `mkdir` 拒绝覆盖；失败时保留目录，先检查原因，不删除旧产物或盲目重跑。

```bash
bash <<'BASH'
set -euo pipefail
cd /home/featurize/work/GAViT_Project
gavit_run=/home/featurize/work/gavit44_2080ti_20260908
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
require(commit == 'bb9f1cff463c996bc5f72695af538898c91490eb' and not dirty, 'STOP: code identity changed; do not reset or overwrite it.')
busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
require(not busy, 'STOP: another GPU process is running.')
require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'STOP: one working CUDA device is required.')
require(torch.cuda.get_device_name(0).endswith('RTX 2080 Ti'), 'STOP: GPU is not RTX 2080 Ti.')
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

nohup python -u -c '
import resource, runpy, sys, time, torch
if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 2080 Ti":
    raise SystemExit("STOP: expected RTX 2080 Ti CUDA device; no CPU fallback.")
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
  --run_stage formal --run_tag rtx2080ti_seed44_20260908 \
  --pretrained_path bigearth_files/model.safetensors \
  --checkpoint_path "$gavit_run/gavit.pth" \
  > "$gavit_run/train.log" 2>&1 < /dev/null &
gavit_pid=$!
printf '%s\n' "$gavit_pid" > "$gavit_run/train.pid"
printf 'Launch submitted. PID=%s\nOutput=%s\n' "$gavit_pid" "$gavit_run"
sleep 5
ps -p "$gavit_pid" -o pid,etime,args || true
nvidia-smi
tail -n 30 "$gavit_run/train.log"
BASH
```

`Launch submitted`只表示后台命令已提交，不代表启动成功。必须同时确认进程仍在、GPU有显存占用、日志出现GAViT正确配置及首个训练batch。首次导入/读CSV可能超过5秒，若日志仍在初始化，稍后查看同一个日志，不重复启动。

查看后续日志：`tail -n 40 /home/featurize/work/gavit44_2080ti_20260908/train.log`。实时查看可用 `tail -f`；此时Ctrl+C只退出日志查看，不终止后台训练。不要归还实例。

输出包括 `preflight.log`、`train.log`、`train.pid`、best `gavit.pth`、`gavit.meta.json` 和 `gavit.last.train_state.pth`。只有训练、验证及这些产物经过检查，才可判断完成。测试best checkpoint另行执行；本命令不自动测试或启动后续模型。

主进程RSS不能代替包含DataLoader子进程的全作业内存；当前简化启动不声称获得完整RAM峰值。中断时保留产物，尤其不能把发生错误的运行报告为完成。新硬件的实际速度、精度行为及费用仍待运行记录。
