import base64
import hashlib
from pathlib import Path
import shlex
import zlib

ROOT = Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2')
source = (ROOT / 'tools/benchmark_aid.py').read_bytes()
digest = hashlib.sha256(source).hexdigest()
payload = base64.b64encode(zlib.compress(source, 9)).decode()
output = '/home/featurize/work/aid_efficiency_20260922_01'
bootstrap = ('import base64,hashlib,pathlib,subprocess,sys,zlib; '
             f's=zlib.decompress(base64.b64decode("{payload}")); '
             f'assert hashlib.sha256(s).hexdigest()=="{digest}","Probe checksum mismatch"; '
             f'p=pathlib.Path("{output}"); p.mkdir(parents=True,exist_ok=False); '
             'f=p/"benchmark_aid.py"; f.write_bytes(s); '
             'log=(p/"run.log").open("xb"); '
             'job=subprocess.Popen([sys.executable,"-u",str(f),"--output",str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); '
             'print("PID:",job.pid,"\\nOutput:",p,"\\nWatch: tail -f",p/"run.log")')
command = 'python -c ' + shlex.quote(bootstrap)
(ROOT / 'docs/paste_aid_efficiency.sh').write_text(command + '\n')
guide = f'''# E06：已有 AID 权重的推理效率测量

状态：脚本及本地验证已准备；服务器 CUDA 测量待执行、待回传。

目的：三seed的 OA 平均差仅 +0.0467 pp，GAViT 多 3831425 个参数（+13.91%）。本次补充实际前向延迟、吞吐与显存，判断当前实现的收益是否值得额外开销。不训练、不重新测试准确率、不需要解压 AID。

## 测量口径

- 固定采用已完成的 **seed42 refit** 配对权重（预先约定的第一个seed，无效率结果挑选）；Swin/GAViT参数为27543960/31375385。读取各自 `aid_{{model}}_refit42_20260920_01/last.pth` 及 metadata/run.json/run.exit。
- 校验干净训练版本 `deb2b45`、全部源码哈希、冻结 manifest/预训练身份及实际 checkpoint 字节SHA256。原始图片和预训练文件本身不参与推理，无需解压或联网。
- 输入为固定随机生成的 RGB 张量，经 ImageNet 归一化，形状 B×3×224×224；同batch下两模型逐字节相同，提前放进显存。测量包含完整 Swin、GAViT 分组、图构建、GAT 与反馈前向。**这是合成输入上的模型计算基准，不是完整数据流水线或测试集吞吐。** 原图读取、解码、缩放、归一化、CPU→GPU传输、argmax与指标计算均不计时；完整处理链耗时仍待单独测量。
- FP32、无 AMP/TF32，eval + inference_mode，8 CPU线程，cuDNN deterministic=true/benchmark=false。图边检查仅在预检前向执行，计时前移除hook。
- 四个独立子进程依次测 `Swin b1 → GAViT b1 → GAViT b32 → Swin b32`；串行且每步退出后才启动下一步。该顺序预先固定，反转两个batch的模型顺序，但不能完全消除温度/时钟漂移。记录GPU UUID、驱动、时钟、温度和功耗；前后及每组重复检查是否有其他GPU计算进程，发现则停止。这些快照不能保证捕获所有瞬时干扰。
- 每项清除加载/预检的缓存后预热20次，测5组×50次。每次前向后同步CUDA，以主机时钟记录该请求完成的耗时，包含Python调度与同步开销。保留250次原始值、均值、中位数、P95、5组均值及样本SD。吞吐=总图片数/总测量秒数；batch32的摊销每图耗时不当成batch1请求延迟。
- 显存峰值在预热后重置统计，包含模型、驻留输入、当前输出和前向临时张量；报告 allocated/reserved 字节，另保留基线。排除权重加载、图捕获及预热峰值；reserved包含稳定推理缓存。不是nvidia-smi整卡显存，也不与含训练/重载的旧峰值直接比较。
- 本次无新模型/候选或参数调整，输出用于E06模型效率表。重复计时用于观察测量波动，不等于跨seed准确率重复。若实例GPU不同于原RTX2080Ti，记录实际设备，只比较本次同卡结果。

## 1. 直接粘贴这一整行启动

无需上传工具包、git pull、安装依赖或修改正式训练目录。下面一行仅把[可读脚本](../tools/benchmark_aid.py)解压写入新的 `work` 结果目录，验证SHA256，再启动脱离终端的后台任务。内嵌压缩文本是为了减少复制长度，不执行网络下载。已有同名目录时拒绝覆盖；重试要换整个输出目录后缀。

脚本 SHA256：`{digest}`。GPU应无其他训练/推理任务。

```bash
{command}
```

后台启动显示PID只表示进程创建。看到 `FORWARD PASS` 和第一组 `repeat` 才确认模型已在CUDA上运行。共4项、每项5组；预计数分钟，具体取决于本次实例，不能用本地CPU验证估算速度。

## 2. 看进度

```bash
tail -f {output}/run.log
```

看到 `AID EFFICIENCY COMPLETE` 后按Ctrl+C退出tail（不会终止后台进程）。如果出现异常、任务退出或没有完成标记，先回传日志，不重新启动相同目录。

## 3. 完成后一次性回传

```bash
cat {output}/run.exit {output}/summary.json
```

验收要求：`exit_code=0`、摘要 `status=complete`、四阶段全部成功、GPU/源码/输入口径匹配。结果保存为 `stages/*_b*.json`（原始250次计时、图核查、全部身份和遥测）及对应 `.exit`。失败时总目录保留 `failure.json`/`run.exit` 与已完成阶段，不生成成功摘要。

若未完成，用这一行回传原因：

```bash
tail -n 60 {output}/run.log
```

## 本地验证边界

验证记录见[本地报告](../results/aid_efficiency_local_validation.md)。异步计时同步、吞吐换算、失败停止、不可覆盖、GPU竞争和配对身份检查已做本地测试；两模型在CPU上用随机权重完成真实前向/存盘重载，GAViT实际捕获48+32条边。**尚未访问服务器、验证其当前文件、运行CUDA计时或取得任何效率数值。** 正式seed42权重及设备会由服务器命令再次核验。
'''
(ROOT / 'docs/run_aid_efficiency.md').write_text(guide)
print({'script_bytes': len(source), 'command_chars': len(command), 'sha256': digest})
