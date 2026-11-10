"""Build and validate the copy/paste launcher without connecting to a server."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
tool = ROOT/'tools/diagnose_nwpu_feedback.py'
helper = ROOT/'tools/visualize_nwpu_regions.py'
b = tool.read_bytes()
sha = hashlib.sha256(b).hexdigest()
helper_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
assert helper_sha == '35ed63ce1ceda120fe9b5e78b157032c5e89d6b351f0caf67e05eef19d4080ba'
payload = base64.b64encode(zlib.compress(b,9)).decode()
server = '/home/featurize/work'
script_path = f'{server}/nwpu_feedback_tool_20261002_{sha[:12]}.py'
out_path = f'{server}/nwpu_feedback42_20261002_01'
log_path = out_path+'.launch.log'
helper_path = f'{server}/nwpu_region_tool_20260929_35ed63ce1ced.py'
required = [helper_path, f'{server}/NWPU-RESISC45_a826b44.zip',
            f'{server}/nwpu_protocol_20260927_02.json', f'{server}/GAViT_SOFT_20260927/models/gavit.py']
required += [f'{server}/nwpu_ablation42_20260928_01/{name}/{filename}'
             for name in ('fixed','soft') for filename in ('run.json','run.exit','best.pth','validation_predictions.csv')]
j = json.dumps
code = '; '.join([
    'import base64,hashlib,os,subprocess,sys,zlib', 'from pathlib import Path',
    f'b=zlib.decompress(base64.b64decode({j(payload)}))',
    f'assert hashlib.sha256(b).hexdigest()=={j(sha)}, "Tool hash mismatch"',
    f'p=Path({j(script_path)})', f'o=Path({j(out_path)})', f'l=Path({j(log_path)})',
    'assert not o.exists() and not l.exists(), "Output/log exists; preserve and inspect before retrying"',
    f'required={j(required)}',
    'missing=[s for s in required if not Path(s).is_file()]',
    'assert not missing, "Missing persistent inputs: "+str(missing)',
    f'assert hashlib.sha256(Path({j(helper_path)}).read_bytes()).hexdigest()=={j(helper_sha)}, "Helper hash mismatch"',
    'assert not p.exists() or p.read_bytes()==b, "Existing tool differs"',
    'p.exists() or p.open("xb").write(b)', 'f=l.open("x")',
    'q=subprocess.Popen([sys.executable,"-u",str(p),"--output",str(o),"--device","cuda"],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"})',
    'f.close()', 'print("PID:",q.pid)', 'print("OUTPUT:",o)', 'print("PROGRESS: tail -f",l)',
    'print("Completion requires NWPU FEEDBACK SENSITIVITY COMPLETE and exit_code=0")'])
assert "'" not in code and '\n' not in code
command = "python -c '"+code+"'"
assert shlex.split(command) == ['python','-c',code]
compile(code,'delivery','exec')
for shell in ('bash','zsh'):
    subprocess.run([shell,'-n'],input=command+'\n',text=True,check=True)
# Execute the exact launch code with redirected paths and a fake process boundary.
# Verifies delivery/fail-fast behavior; it never runs the server/GPU diagnostic.
with tempfile.TemporaryDirectory(prefix='feedback-delivery-') as td:
    root = Path(td)
    local_code = code.replace(server,td)
    for path in required:
        p = Path(path.replace(server,td));p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(helper.read_bytes() if path==helper_path else b'fixture')
    with patch('subprocess.Popen',return_value=SimpleNamespace(pid=12345)) as popen:
        exec(compile(local_code,'fixture_delivery','exec'),{})
        assert popen.call_count==1
        args,kwargs=popen.call_args
        assert args[0][-2:]==['--device','cuda'] and kwargs['start_new_session']
        assert Path(script_path.replace(server,td)).read_bytes()==b
        log=Path(log_path.replace(server,td));assert log.is_file()
        before=log.read_bytes()
        try: exec(compile(local_code,'duplicate_delivery','exec'),{})
        except AssertionError as e: assert 'Output/log exists' in str(e)
        else: raise AssertionError('Duplicate invocation accepted')
        assert popen.call_count==1 and log.read_bytes()==before
    # A fresh identity with a missing required checkpoint must fail before launch.
    missing_code=local_code.replace('nwpu_feedback42_20261002_01','nwpu_feedback42_missing')
    Path(required[-2].replace(server,td)).unlink()
    with patch('subprocess.Popen') as popen:
        try: exec(compile(missing_code,'missing_delivery','exec'),{})
        except AssertionError as e: assert 'Missing persistent inputs' in str(e)
        else: raise AssertionError('Missing checkpoint accepted')
        assert not popen.called

(ROOT/'docs/launch_nwpu_feedback_20261002.txt').write_text(command+'\n')
doc=f'''# NWPU 反馈强度诊断：复制执行

目的：判断已训练的图反馈减弱后，验证集预测如何变化，为是否值得研究反馈接口提供依据。本次不训练，不使用 test，不自动挑选系数或启动下一轮实验。服务器尚未启动，结果待回传。

固定 NWPU seed42 校准阶段的 fixed/soft best 权重与原 1260 张验证图，分别计算 α=0、0.25、0.5、0.75、1。α=1 必须复现原缓存概率及完整验证 OA（fixed 93.015873%、soft 93.730159%）；否则停止并保留错误。校准训练数量5040、验证1260；不使用6300图 refit 权重。

需要 work 中保留之前的 NWPU ZIP、清单、两份校准目录、冻结 ed480e6 仓库及原可视化脚本。直接按清单读取原图或 ZIP 成员，并核对文件和像素哈希；无需恢复 BigEarthNet，也无需解压 NWPU。启动前检查文件，后台再核对源码/权重/清单。若某文件缺失，只回传报错，不重复运行或覆盖旧输出。

只跑两次1260图的推理，各系数共用同一份骨干/图计算；额外只重算分类头。不是一次14小时训练。实际总耗时受模型读取和磁盘影响，本次尚无服务器计时，先看首个 CAPTURE 进度；不要因 LOAD 暂时无输出重复启动。

## 1. 启动（下面是完整一行）

也可打开 [纯命令文件](launch_nwpu_feedback_20261002.txt) 全选复制。不要复制服务器提示符。

```bash
{command}
```

## 2. 查看进度

```bash
tail -f {log_path}
```

完成标记为 `NWPU FEEDBACK SENSITIVITY COMPLETE`，随后按 Ctrl+C 只退出 tail。

## 3. 完成后回传这一条的输出

```bash
cat {out_path}/run.exit {out_path}/summary.json
```

若未完成或报错，回传：

```bash
tail -n 45 {log_path}
```

## 固定范围与解读规则

- 研究假设：冻结权重下，降低图反馈幅度可能改变准确率、类别损益与损失。唯一干预是反馈乘数，模型分别内部比较。
- 产出：完整五档 OA、macro-F1、交叉熵、相对α=1的纠正/新增错误，逐类统计、混淆矩阵和配对预测。`fixed_cache.pt`/`soft_cache.pt`保存平均token、平均反馈、分类头及logits，之后可离线复算，不必重复骨干前向。
- 成功须有全部1260图、缓存概率一致、α=1重建及OA一致、CPU保存重载复算通过、summary complete及exit_code=0。不能只依据PID。
- 若多个较低α改善OA且损失、逐类损益支持一致趋势，才形成“可研究反馈幅度”的后续假设；仍需另行设计训练对照。仅某一α单点最好、仅损失降低或仅soft改善，都不足以认定根因。
- α=0是在联合训练好的模型中移除反馈，不等于独立训练的无图模型或Swin。所有系数为观察扰动，不写入正式test主表、不宣称稳定提升。无改善也保留，不扩大系数搜索或自动重训。
- 该结果只对应NWPU校准权重，不能直接解释BigEarthNet差距；不能证明软分配语义正确、图反馈是唯一根因、或新门控训练有效。
- 资源记录用于此次诊断执行审计，不当作模型公平效率比较。后续未自动获准新增训练或跨seed搜索。

独立诊断SHA256：`{sha}`；复用helper：`{helper_sha}`。正式模型代码、已有权重和实验表保持原身份。

本地检查见[验收记录](../results/nwpu_feedback_sensitivity_local_validation.md)。本地CPU合成输入检查与服务器真实权重/原图CUDA运行分开记录。
'''
(ROOT/'docs/run_nwpu_feedback_sensitivity_20261002.md').write_text(doc)
report=dict(status='passed', tool_sha256=sha, helper_sha256=helper_sha,
            payload_roundtrip=zlib.decompress(base64.b64decode(payload))==b,
            command_characters=len(command),shell_checks=['bash -n','zsh -n'],
            delivery_fixture='exact launcher with temporary paths and fake Popen',
            duplicate_preserved=True,missing_checkpoint_blocks_launch=True,
            scope='Local delivery verification only; no server operation or GPU launch')
(HERE/'delivery_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
