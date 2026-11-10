# NWPU soft smoke：保留失败记录后重试

2026-09-28。用户回传的launch日志通过`>>`追加，包含先前缺图和最新输出目录碰撞。已核对发布代码：`_run_output`在读取数据前执行`mkdir(exist_ok=False)`；因此最新`FileExistsError`这次调用没有再次检查图片或开始训练，不能据追加日志断言解压后仍然缺图。

保留`nwpu_soft_smoke42_20260927_01`及其日志。此次只改运行输出为`_02`，模型、seed、清单、优化和数据路径不变，无需重新prepare、下载或修改代码。以下是给用户在服务器执行的命令；尚未得到文件恢复或重试成功的返回。

先Ctrl+C退出tail，然后核对既有清单中的全部路径及新目录可用性：

```bash
python -c 'import json; from pathlib import Path; root=Path("/home/featurize/data/NWPU-RESISC45"); rows=json.loads(Path("/home/featurize/work/nwpu_protocol_20260927_02.json").read_text())["records"]; missing=[r["path"] for r in rows if not (root/r["path"]).is_file()]; assert len(rows)==31500,len(rows); assert not missing,missing[:8]; assert not Path("/home/featurize/work/nwpu_soft_smoke42_20260927_02").exists(),"Output _02 already exists"; print("FILES_READY: 31500/31500; OUTPUT_02_FREE")'
```

只有出现`FILES_READY: 31500/31500; OUTPUT_02_FREE`才继续。该检查只验证路径存在；训练入口仍会逐图解码与哈希校验，不用此检查替代正式预检。

```bash
nohup python -u /home/featurize/work/GAViT_SOFT_20260927/run_scene.py train --model gavit --grouping soft_prototype --phase smoke --seed 42 --train-per-class 8 --val-per-class 2 --manifest /home/featurize/work/nwpu_protocol_20260927_02.json --data-root /home/featurize/data/NWPU-RESISC45 --output /home/featurize/work/nwpu_soft_smoke42_20260927_02 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --device cuda --batch-size 32 --workers 2 >> /home/featurize/work/nwpu_soft_smoke42_20260927_02.launch.log 2>&1 < /dev/null &
tail -f /home/featurize/work/nwpu_soft_smoke42_20260927_02.launch.log
```

首batch时在另一终端检查`nvidia-smi`，确认进程和显存。收到`SCENE SMOKE COMPLETE`后Ctrl+C退出tail并回传：

```bash
cat /home/featurize/work/nwpu_soft_smoke42_20260927_02/run.exit /home/featurize/work/nwpu_soft_smoke42_20260927_02/run.json
```

成功标准仍为exit0、status complete、360/90数量、实际48+32图边、重载差≤1e-5及探针产物。若路径检查或新运行失败，回传本次输出并保留目录，不删失败记录重用同一名称。
