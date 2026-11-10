import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT = Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2')
RELEASE = Path('/Users/patrick/.codex/worktrees/aid-entry/GAViT_Project')
SOURCE = Path('/Users/patrick/.codex/attachments/80f9d7ab-5376-4c07-85c9-18ab36e6a04a/已粘贴的文本.txt')
raw = SOURCE.read_bytes()
text = raw.decode()
assert text.splitlines().count('exit_code=0') == 1
start = text.index('{')
data, end = json.JSONDecoder().raw_decode(text[start:])
assert text[start + end:].strip() == '(base) ➜ ~'
spec = importlib.util.spec_from_file_location('benchmark_aid', ROOT / 'tools/benchmark_aid.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
assert data['status'] == 'complete'
assert data['protocol'] == probe.PROTOCOL
assert data['checkpoint_sha256'] == probe.WEIGHTS
assert data['probe_sha256'] == hashlib.sha256((ROOT / 'tools/benchmark_aid.py').read_bytes()).hexdigest()
assert data['stage_exit_codes'] == {f'{m}_b{b}': 0 for m, b in probe.STAGES}
identity = data['common_identity']
assert identity['manifest_sha256'] == probe.MANIFEST_SHA
assert identity['pretrained_sha256'] == probe.PRETRAINED_SHA
assert identity['code']['git'] == {'commit': 'deb2b45', 'dirty': False}
hashes = identity['code']['files_sha256']
assert len(hashes) == 17
assert all(hashlib.sha256((RELEASE / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
previous = json.loads((ROOT / 'results/evidence/aid_swin_refit42_20260920_01/run.json').read_text())
for key in ('device_name', 'cuda_runtime', 'cudnn', 'python', 'torch_num_threads', 'packages'):
    assert data['hardware'][key] == previous['hardware'][key]
assert data['graph_capture'] == {'swin': {}, 'gavit': {'edges_per_image': 80, 'spatial_edges': 48,
    'feature_edges': 32, 'actual_forward_checked': True}}
rows = {(r['model'], r['batch_size']): r for r in data['results']}
assert len(data['results']) == len(rows) == 4 and set(rows) == set(probe.STAGES)
derived_rows = []
for (model, batch), r in rows.items():
    t = r['timing']
    means = t['repeat_batch_ms_means']
    assert r['parameters'] == probe.PARAMETERS[model]
    assert len(means) == 5 and all(math.isfinite(v) and v > 0 for v in means)
    checks = {'batch_ms_mean': statistics.mean(means), 'repeat_mean_sample_sd_ms': statistics.stdev(means),
        'repeat_mean_min_ms': min(means), 'repeat_mean_max_ms': max(means),
        'amortized_ms_per_image': statistics.mean(means) / batch,
        'images_per_second': batch * 1000 / statistics.mean(means)}
    for key, expected in checks.items():
        assert math.isclose(t[key], expected, rel_tol=1e-10, abs_tol=1e-9), (model, batch, key)
    assert 0 < t['batch_ms_median'] <= t['batch_ms_p95']
    assert type(r['peak_allocated_bytes']) is int and type(r['peak_reserved_bytes']) is int
    assert r['parameters'] * 4 <= r['peak_allocated_bytes'] <= r['peak_reserved_bytes']
    derived_rows.append({**r, 'peak_allocated_mib': r['peak_allocated_bytes'] / 2**20,
                        'peak_reserved_mib': r['peak_reserved_bytes'] / 2**20})
timed_seconds = sum(r['timing']['batch_ms_mean'] * 250 / 1000 for r in rows.values())
assert data['elapsed_seconds'] >= timed_seconds
comparisons = {}
for batch in (1, 32):
    a, b = rows['swin', batch], rows['gavit', batch]
    comparisons[str(batch)] = {
        'mean_batch_latency_increase_ms': b['timing']['batch_ms_mean'] - a['timing']['batch_ms_mean'],
        'mean_batch_latency_change_percent': 100 * (b['timing']['batch_ms_mean'] / a['timing']['batch_ms_mean'] - 1),
        'throughput_change_percent': 100 * (b['timing']['images_per_second'] / a['timing']['images_per_second'] - 1),
        'allocated_memory_increase_mib': (b['peak_allocated_bytes'] - a['peak_allocated_bytes']) / 2**20,
        'allocated_memory_change_percent': 100 * (b['peak_allocated_bytes'] / a['peak_allocated_bytes'] - 1),
        'reserved_memory_increase_mib': (b['peak_reserved_bytes'] - a['peak_reserved_bytes']) / 2**20,
        'reserved_memory_change_percent': 100 * (b['peak_reserved_bytes'] / a['peak_reserved_bytes'] - 1),
    }
multi_path = ROOT / 'results/aid_multiseed_summary.json'
multi = json.loads(multi_path.read_text())
original_primary = {k: copy.deepcopy(v) for k, v in multi.items() if k not in ('efficiency', 'posthoc_review')}
analysis = {'status': 'reported_model_only_benchmark_verified', 'dataset': 'AID', 'checkpoint_seed': 42,
    'server_output': '/home/featurize/work/aid_efficiency_20260922_01',
    'scope': data['scope'], 'rows': derived_rows, 'gavit_relative_to_swin': comparisons,
    'elapsed_seconds': data['elapsed_seconds'], 'measured_forward_seconds_derived': timed_seconds,
    'hardware': data['hardware'], 'gpu_uuid': data['gpu_uuid'], 'driver_version': data['driver_version'],
    'checkpoint_sha256': data['checkpoint_sha256'], 'probe_sha256': data['probe_sha256'],
    'paired_accuracy_difference_mean_pp_n3': multi['paired_delta_statistics']['delta_oa_pp']['mean_pp'],
    'parameter_increase_percent': multi['efficiency']['parameter_increase_percent'],
    'raw_timings_and_telemetry_locally_received': False,
    'full_input_pipeline_measured': False,
    'evidence': 'evidence/aid_efficiency_20260922_01/summary.json'}
evidence = ROOT / 'results/evidence/aid_efficiency_20260922_01'
evidence.mkdir(parents=True, exist_ok=False)
(evidence / 'reported_output.txt').write_bytes(raw)
def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
write(evidence / 'summary.json', data)
write(evidence / 'provenance.json', {'source': str(SOURCE), 'source_sha256': hashlib.sha256(raw).hexdigest(),
    'archive_sha256': hashlib.sha256((evidence / 'reported_output.txt').read_bytes()).hexdigest(),
    'normalization': 'Extract one JSON object; preserve original pasted command, exit_code and shell prompt separately.',
    'not_received': ['original summary.json file bytes', 'original run.exit file bytes', 'run.log',
                     'stages/*.json raw timings and telemetry', 'checkpoint bytes'],
    'completion_basis': 'reported exit_code=0, complete summary, four zero stage exits; script requires successful stage validation before summary'})
write(evidence / 'validation.json', {'status': 'passed', 'summary_exit_and_four_stages': True,
    'protocol_checkpoint_probe_identities': True, 'local_release_source_hashes_matched': len(hashes),
    'hardware_matches_seed42_except_device_alias_and_unrecorded_original_driver': True,
    'graph_capture': True, 'mean_sd_min_max_throughput_amortization_recomputed': True,
    'memory_ordering_and_mib_conversion': True, 'median_p95_recomputed_from_raw_samples': False,
    'raw_per_stage_identity_input_hash_telemetry_independently_checked': False,
    'measured_forward_seconds_derived': timed_seconds, 'reported_total_elapsed_seconds': data['elapsed_seconds']})
write(ROOT / 'results/aid_efficiency_summary.json', analysis)
multi['efficiency'].update(standardized_inference_speed_and_memory='reported_model_only_benchmark_verified',
    report='aid_efficiency_summary.md', evidence=analysis['evidence'], checkpoint_seed=42,
    scope=data['scope'], full_input_pipeline_measured=False,
    gavit_relative_to_swin=copy.deepcopy(comparisons))
multi['posthoc_review']['next_step'] = 'E06 model-only benchmark returned and verified; full input pipeline and independent research tasks remain pending'
assert {k: v for k, v in multi.items() if k not in ('efficiency', 'posthoc_review')} == original_primary
write(multi_path, multi)
report = '''# AID seed42：Swin / GAViT 模型推理效率

2026-09-22回传，服务器目录 `/home/featurize/work/aid_efficiency_20260922_01`。总退出码0、摘要complete、四阶段退出码均0；本地摘要算术与身份核对通过。总wall time **88.26秒**，不把包含加载/预检/预热的总时长当吞吐分母。

当前固定4×4区域图版本在三seed上OA均值只增加 **0.0467 pp**；本次同卡测量中单张延迟增加 **34.41%**，batch32吞吐下降 **15.31%**。额外显存较小，但当前证据尚未显示额外计算开销对应稳定的整体准确率收益。这是该实现、协议、数据划分与设备下的结论，不证明所有图方法无效，也不分解构图/GAT/反馈各自耗时。

## 论文效率表

设备 **RTX2080Ti**，FP32，无AMP/TF32；输入为固定合成RGB归一化张量，224×224，预先驻留GPU。完整模型前向计时包含Python图构建和设备执行；不含文件读取、图像处理、CPU→GPU传输、argmax或指标计算。**不是测试集吞吐或完整应用延迟。**

| 指标 | Swin | GAViT | GAViT相对变化 |
|---|---:|---:|---:|
| 参数量（M，十进制） | 27.544 | 31.375 | +13.91% |
| batch1平均请求延迟（ms） | 14.891 | 20.016 | +34.41% |
| batch1吞吐（图/秒） | 67.154 | 49.961 | −25.60% |
| batch32平均批次耗时（ms） | 56.648 | 66.885 | +18.07% |
| batch32吞吐（图/秒） | 564.892 | 478.433 | −15.31% |
| batch1峰值allocated（MiB） | 133.268 | 147.755 | +10.87%（+14.488 MiB） |
| batch32峰值allocated（MiB） | 594.482 | 609.353 | +2.50%（+14.871 MiB） |
| batch1峰值reserved（MiB） | 158.000 | 180.000 | +13.92%（+22 MiB） |
| batch32峰值reserved（MiB） | 942.000 | 962.000 | +2.12%（+20 MiB） |

MiB = 2^20字节；原始字节及全部精度见[派生JSON](aid_efficiency_summary.json)。allocated为PyTorch活跃分配，reserved为分配器保留，两者不相加，也不当作nvidia-smi整卡占用。范围为预热后的模型+输入+当前输出/临时张量，排除checkpoint加载与预检峰值。batch32摊销每图耗时1.770/2.090 ms，不是batch1独立请求延迟。参数精确数27543960/31375385，增加3831425。

## 重复计时与局限

每项一个独立进程，20次预热后5组×50次同步主机计时，保留全部组次；顺序固定为Swin b1、GAViT b1、GAViT b32、Swin b32。表中SD是**5组均值**的样本SD，不是250次单次延迟SD，更不是训练seed方差或置信区间。

| 配置 | 平均耗时±组均值SD（ms/批） | 5组均值范围（ms/批） | 回传中位数 / P95（ms/批） |
|---|---:|---:|---:|
| Swin b1 | 14.891 ± 0.597 | 14.512–15.919 | 14.058 / 19.162 |
| GAViT b1 | 20.016 ± 0.168 | 19.738–20.157 | 19.514 / 20.574 |
| Swin b32 | 56.648 ± 0.262 | 56.327–57.001 | 56.768 / 58.707 |
| GAViT b32 | 66.885 ± 0.629 | 66.163–67.534 | 66.827 / 68.678 |

Swin b1首组均值15.919 ms较后面高，全部预定数据保留，没有事后删组。每个batch下GAViT的5组均值均高于Swin范围，支持本次顺序测量的方向；不能外推为不同GPU/输入/精度下的固定倍率。峰值内存单次测量，没有构造误差条。没有追加训练或重跑来选择更好数字。

## 身份与验收范围

- 固定seed42最终refit权重，与原test使用的hash匹配；Swin21轮、GAViT20轮。manifest/预训练来源与冻结协议一致，源码17文件哈希逐一匹配本地发布版本deb2b45，回传Git为clean。
- 当前报告CUDA12.1、cuDNN8902、Python3.11.8、torch2.2.2、torchvision0.17.2、timm1.0.27、PyG2.8.0、numpy1.26.4、Pillow10.3.0、8线程；驱动610.57.04。设备UUID和精确身份在回传摘要。驱动与UUID来自本次报告，不推断与训练实例物理设备相同。
- GAViT报告真实前向核查48空间边+32特征边；probe SHA与本地已验证脚本完全相同。脚本在成功汇总前要求四项hardware/GPU UUID/驱动/源码/协议相同、同batch输入hash相同，且任一子进程失败即停止。相关逐阶段原始JSON仍在服务器，本地未逐份重查。
- 本地重新计算4项×5组均值的总体均值、样本SD、min/max、吞吐与摊销耗时；总计时约39.610秒小于完整wall time88.261秒，内存数量关系与转换一致。
- 本次收到的是终端粘贴摘要和exit0，不是服务器原文件字节副本。原始250次耗时、GPU温度/时钟/功耗遥测、完整run.log及checkpoint字节未本地取得；不能独立重算中位数/P95或断言不存在温漂/瞬时干扰。无需为此次摘要分析重复运行。

原文与提取JSON见[证据目录摘要](evidence/aid_efficiency_20260922_01/summary.json)，[来源记录](evidence/aid_efficiency_20260922_01/provenance.json)及[核验结果](evidence/aid_efficiency_20260922_01/validation.json)。原文保留首尾shell提示符；只在派生JSON中去除。

## 可用于论文的表述

在AID固定数据划分的三次训练中，Swin与固定4×4区域图GAViT的OA分别为96.6800±0.2946%和96.7267±0.5707%，配对均值增益为0.0467个百分点。采用预先固定的seed42最终权重，在RTX2080Ti上以FP32和驻留GPU的224×224合成输入测量完整模型前向，GAViT相较Swin的单张平均延迟由14.89增加至20.02 ms，batch32吞吐由564.89降至478.43图/秒，峰值已分配显存由594.48增加至609.35 MiB。因此当前实现尚未显示足以支撑额外计算开销的稳定整体精度收益。

状态：**E06模型前向基准的回传与摘要核验完成**；包含原图读取、处理及传输的完整流程仍待测。E02外部方法、E05分组候选等独立任务仍未完成，不将本结果当作整个论文实验完成。三seed准确率和既有负面seed44保留不变。
'''
(ROOT / 'results/aid_efficiency_summary.md').write_text(report)
(evidence / 'analyze_return.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'validation': 'passed', 'elapsed_seconds': data['elapsed_seconds'], 'comparisons': comparisons,
                  'primary_accuracy_fields_preserved': True}, indent=2))
