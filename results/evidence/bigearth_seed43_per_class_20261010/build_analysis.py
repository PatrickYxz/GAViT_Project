"""Rebuild descriptive class tables from archived, existing test results only."""
import csv
import hashlib
import io
import json
import re
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RAW = HERE / "server_return.txt"
PAIR = ROOT / "results/bigearth_seed42_seed44_class_stability.csv"
LAST = ROOT / "results/bigearth_seed45_per_class_comparison.csv"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue())


blocks = RAW.read_text().split("Test mAP   (macro): ")[1:]
assert len(blocks) == 2
models = []
for block in blocks:
    entries = [(name, Decimal(value)) for value, name in
               re.findall(r"^\s*([0-9]+\.[0-9]+)%  (.+)$", block, re.M)]
    assert len(entries) == len(dict(entries)) == 19
    assert all(0 <= value <= 100 for _, value in entries)
    metrics = [Decimal(re.match(r"([0-9.]+)%", block)[1])]
    metrics += [Decimal(value) for value in
                re.findall(r"Test F1\s+\((?:macro|micro)\): ([0-9.]+)%", block)]
    assert len(metrics) == 3
    ap = dict(entries)
    assert abs(sum(ap.values()) / 19 - metrics[0]) <= Decimal("0.1")
    models.append({"ap": ap, "metrics": metrics})
assert models[0]["metrics"] == list(map(Decimal, ["71.1", "64.1", "76.0"]))
assert models[1]["metrics"] == list(map(Decimal, ["71.7", "65.7", "76.8"]))
pair = read_csv(PAIR)
last_rows = read_csv(LAST)
last = {row["Class"]: row for row in last_rows}
classes = [row["Class"] for row in pair]
assert len(pair) == len(last_rows) == len(last) == len(set(classes)) == 19
assert classes == list(models[0]["ap"]) == list(models[1]["ap"])
assert set(classes) == set(last)

single, combined, summaries = [], [], []
for old in pair:
    name = old["Class"]
    s43, g43 = (model["ap"][name] for model in models)
    delta43 = g43 - s43
    single.append({"Class": name, "Swin_AP(%)": s43, "GAViT_AP(%)": g43,
                   "Delta_AP(pp)": delta43,
                   "Contribution_to_macro_mAP(pp)": delta43 / 19})
    row = {"Class": name}
    deltas = []
    for seed in [42, 43, 44, 45]:
        if seed == 43:
            swin, gavit = s43, g43
        elif seed == 45:
            swin = Decimal(last[name]["Swin_AP(%)"])
            gavit = Decimal(last[name]["GAViT_AP(%)"])
            assert abs(gavit - swin - Decimal(last[name]["Delta_AP(pp)"])) < Decimal("1e-12")
        else:
            swin, gavit = Decimal(old[f"Swin{seed}_AP(%)"]), Decimal(old[f"GAViT{seed}_AP(%)"])
            assert gavit - swin == Decimal(old[f"Seed{seed}_delta(pp)"])
        assert 0 <= swin <= 100 and 0 <= gavit <= 100
        delta = gavit - swin
        row.update({f"Swin{seed}_AP(%)": swin, f"GAViT{seed}_AP(%)": gavit,
                    f"Seed{seed}_delta(pp)": delta})
        deltas.append(delta)
    mean = sum(deltas) / 4
    sd = (sum((value - mean) ** 2 for value in deltas) / 3).sqrt()
    counts = {"Positive_Seeds": sum(value > 0 for value in deltas),
              "Negative_Seeds": sum(value < 0 for value in deltas),
              "Tied_Seeds": sum(value == 0 for value in deltas)}
    row.update(counts)
    row.update({"Mean_delta(pp)": mean, "Sample_SD_delta(pp)": sd})
    combined.append(row)
    summaries.append({"class": name, "deltas_pp": dict(zip(map(str, [42, 43, 44, 45]), map(str, deltas))),
                      **counts, "mean_delta_pp": str(mean), "sample_sd_delta_pp": str(sd)})

single_path = ROOT / "results/bigearth_seed43_per_class_comparison.csv"
combined_path = ROOT / "results/bigearth_four_seed_per_class_comparison.csv"
save_csv(single_path, list(single[0]), single)
save_csv(combined_path, list(combined[0]), combined)
counts43 = {"improved": sum(row["Delta_AP(pp)"] > 0 for row in single),
            "declined": sum(row["Delta_AP(pp)"] < 0 for row in single),
            "tied_at_printed_precision": sum(row["Delta_AP(pp)"] == 0 for row in single)}
assert counts43 == {"improved": 10, "declined": 8, "tied_at_printed_precision": 1}
assert not any(row["Positive_Seeds"] == 4 or row["Negative_Seeds"] == 4 for row in combined)
summary = {
    "date": "2026-10-10", "dataset": "BigEarthNet-19", "seeds": [42, 43, 44, 45],
    "new_evidence": "User-pasted excerpts from two existing seed43 test logs; no new inference or training.",
    "sources_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [RAW, PAIR, LAST]},
    "seed43": {"Swin_reported_metrics_percent": list(map(str, models[0]["metrics"])),
               "GAViT_reported_metrics_percent": list(map(str, models[1]["metrics"])),
               "reported_metric_deltas_pp": [str(g-s) for s,g in zip(models[0]["metrics"], models[1]["metrics"])],
               "class_counts": counts43,
               "mean_printed_AP_Swin_percent": str(sum(models[0]["ap"].values()) / 19),
               "mean_printed_AP_GAViT_percent": str(sum(models[1]["ap"].values()) / 19),
               "mean_printed_AP_delta_pp": str(sum(row["Delta_AP(pp)"] for row in single) / 19),
               "Swin_test_peak_allocated_GiB": "0.58", "Swin_test_wall_seconds": "234.20",
               "GAViT_test_timing_and_memory": "not present in returned excerpt"},
    "all_four_positive_classes": [], "all_four_negative_classes": [],
    "precision": "Seeds42/43/44 APs rounded to 0.1 percent; seed45 uses returned full precision. Ties mean printed ties only. Printed AP means do not replace reported aggregate mAP.",
    "limits": "Descriptive existing-run comparison; historical hardware/completion differences retained. No significance test, per-image prediction audit, scene-level mechanism or graph-causal conclusion.",
    "classes": summaries,
}
(HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

lines = ["# BigEarthNet固定分组：四seed逐类AP对照", "", "2026-10-10。补齐seed43现有测试日志中的19类AP；没有重新训练或推理。",
         "", "[原始回传](evidence/bigearth_seed43_per_class_20261010/server_return.txt)、[解析与来源SHA](evidence/bigearth_seed43_per_class_20261010/summary.json)、[seed43 CSV](bigearth_seed43_per_class_comparison.csv)、[四seed CSV](bigearth_four_seed_per_class_comparison.csv)、[聚合指标与运行边界](bigearth_four_seed_summary.md)。",
         "", "## seed43结果", "", "Swin/GAViT的mAP为71.1/71.7%，macro-F1为64.1/65.7%，micro-F1为76.0/76.8%；差值分别为+0.6/+1.6/+0.8 pp。逐类10升、8降、1打印持平。",
         "", "最大提升：Moors, heathland and sclerophyllous vegetation +4.2 pp；Permanent crops +3.3 pp；Inland wetlands +2.9 pp；Coastal wetlands +1.6 pp；Natural grassland and sparsely vegetated areas +1.3 pp。最大下降为Beaches, dunes, sands −2.7 pp，其余下降为−0.1至−0.4 pp。",
         "", "## 四seed方向", "", "差值均为GAViT−Swin，单位pp。均值和样本SD由各seed配对差计算，属于描述性汇总。",
         "", "| Class | seed42 | seed43 | seed44 | seed45 | 均值 ± SD | 升/降/打印持平 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for row in combined:
    cells = [row["Class"]] + [f'{row[f"Seed{seed}_delta(pp)"]:+.2f}' for seed in [42,43,44,45]]
    cells += [f'{row["Mean_delta(pp)"]:+.2f} ± {row["Sample_SD_delta(pp)"]:.2f}',
              f'{row["Positive_Seeds"]}/{row["Negative_Seeds"]}/{row["Tied_Seeds"]}']
    lines.append("| " + " | ".join(cells) + " |")
lines += ["", "没有类别四次全部提升或四次全部下降。此前只用42/44/45判断的Arable land、Inland waters三次正向和Natural grassland三次负向，在seed43分别出现负向、负向和正向，不能外推为四seed一致。",
          "", "农业与自然植被混合区、内陆湿地是3次提升、1次打印持平；自然草地、海滩沙丘沙地是1次提升、3次下降；海岸湿地是2次提升、2次下降，但后两次降幅很大，四次平均−7.42 pp、样本SD约" + f'{combined[16]["Sample_SD_delta(pp)"]:.2f}' + " pp。这些可作为失败分析线索，不能称为显著稳定的类别收益或退化。",
          "", "## 精度与证据边界", "", "seed42/43/44只保留一位小数AP，seed45使用完整返回精度；接近零的差值谨慎解释，打印持平不证明真实AP相等。seed43打印AP均值为71.110526/71.736842%，配对均值差为+0.626316 pp，与日志71.1/71.7%相容；正式聚合表继续使用日志报告值。",
          "", "Swin43回传含测试峰值allocated 0.58 GiB、wall 234.20 s；GAViT43片段未含计时/显存，不填补。此片段不重新证明权重身份、退出状态或逐图预测；沿用此前已记录的测试身份和边界。GAViT43仍是epoch8 best测试、训练到epoch21停止，不能写为30/30。跨seed硬件与历史训练完成条件不同，不能把所有波动唯一归因于随机seed。",
          "", "AP是整个类别的排序指标，不是某一张场景图像的准确率；没有取得逐图预测/分组/图反馈证据，无法仅凭本表解释哪个具体场景的图结构导致得失。", ""]
(ROOT / "results/bigearth_four_seed_class_analysis.md").write_text("\n".join(lines))
print(json.dumps({"seed43_counts": counts43, "classes": len(combined),
                  "all_four_positive": 0, "all_four_negative": 0,
                  "coastal_mean_and_sd_pp": [str(combined[16]["Mean_delta(pp)"]), str(combined[16]["Sample_SD_delta(pp)"])]}, ensure_ascii=False))
