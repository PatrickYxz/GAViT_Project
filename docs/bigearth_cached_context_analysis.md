# 完整验证缓存：按水体标签分组

09-17状态：此分析已在服务器完成并回传，当前无需重复执行。整体与分组结果见[完整分析](../results/seed44_fullval_context_analysis.md)，下方命令保留为复现记录。

使用已完成的`pair44_fullval_8Tvk9d`预测缓存，不加载模型、RGB图片或CUDA，不改变权重、阈值、划分。目的是检查GAViT与Swin的差异是否集中在特定共现环境，不能仅凭分组相关性作因果解释。

下面直接复用服务器已经部署的`bigearth_pair_core.py`计算AP和0.5阈值的TP/FN/FP。启动行不含Shell引号；完整复制到最后单独一行`PY`。这是终端内联分析，无须上传文件。

```bash
python - /home/featurize/work/pair44_fullval_8Tvk9d/diagnosis /home/featurize/work/gavit-pair-diagnostic-20260915/tools 122342 <<PY
import csv, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from bigearth_pair_core import binary_metrics
p = Path(sys.argv[1])
run = json.loads((p / "run.json").read_text())
assert run["status"] == "complete", "Run has not completed"
fields = ["row_index", "label_14", "label_16", "label_17", "label_18"]
fields += [f"{m}_score_{c}" for c in (14, 16) for m in ("swin", "gavit")]
with (p / "predictions.csv").open() as f:
    rows = [{k: float(r[k]) for k in fields} for r in csv.DictReader(f) if r["cohort"] == "uniform"]
assert len(rows) == len({r["row_index"] for r in rows}) == int(sys.argv[3]), "Wrong sample count or duplicate rows"
assert all(r[k] in (0, 1) for r in rows for k in ("label_14", "label_16", "label_17", "label_18"))
print("RUN", run["status"], "N", len(rows), "AP_CHECKS", run.get("ap_reference_checks"))
print("FULL_PER_CLASS_CSV")
print((p / "per_class.csv").read_text())
groups = {"all": None, "sea_only": (1, 0), "inland_only": (0, 1), "both": (1, 1), "neither": (0, 0)}
print("CONTEXT: class group N positive | Swin AP TP FN FP | GAViT AP TP FN FP")
for c in (14, 16):
    for name, key in groups.items():
        subset = [r for r in rows if key is None or (r["label_18"], r["label_17"]) == key]
        y = [int(r[f"label_{c}"]) for r in subset]
        parts = [f"{c} {name} {len(y)} {sum(y)}"]
        for model in ("swin", "gavit"):
            if not y:
                parts.append("NA 0 0 0")
                continue
            m, _ = binary_metrics(y, [r[f"{model}_score_{c}"] for r in subset])
            ap = "NA" if m["ap"] is None else format(100 * m["ap"], ".4f")
            parts.append(f"{ap} {m['tp']} {m['fn']} {m['fp']}")
        print(" | ".join(parts))
PY
```

回传打印的文本即可。编号14为沙滩/沙丘/沙地，16为海岸湿地；AP单位%，TP为命中的正例、FN为漏判正例、FP为误报负例。`all`是完整验证集，另外四组互斥且覆盖全部图片；分组使用真实水体标签，不是模型预测。`sea_only`表示有海洋标签、无内陆水体标签，其他组依此类推。

每组纳入该组全部正例和负例；没有正例或没有负例时AP显示NA。只在同一组内比较Swin/GAViT的AP，不把组间AP高低直接解释为背景造成的性能变化，因为各组类别比例不同。完整val中“沙滩+内陆水体”总共也只有15个正例，应持续保留小样本限制。

该命令在本地已用现有4096张均匀验证缓存验证：仅替换目录与期望行数，互斥组计数/正例和混淆计数之和等于all，all的AP/混淆计数与已复核per_class.csv一致。后续完整val服务器输出已回传并核对分组计数及指标一致性；本地小样本验证与服务器完整val结果分别留档。
