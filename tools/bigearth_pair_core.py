"""Dependency-free selection and analysis for the paired validation diagnostic."""

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random


def write_csv(path, fields, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def iter_split(path, num_classes):
    """Validate labels while preserving the zero-based source CSV data row."""
    seen = set()
    columns = [f"label_{i}" for i in range(num_classes)]
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"patch_path", *columns}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing patch_path/label columns: {path}")
        for index, row in enumerate(reader):
            patch = (row.get("patch_path") or "").strip()
            if not patch or patch in seen:
                raise ValueError(f"Empty or duplicate patch_path at {path}:{index + 2}")
            seen.add(patch)
            try:
                values = [float(row[key]) for key in columns]
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid labels at {path}:{index + 2}") from exc
            if any(value not in (0.0, 1.0) for value in values):
                raise ValueError(f"Labels must be binary at {path}:{index + 2}")
            yield {"row_index": index, "patch_path": patch, "labels": [int(v) for v in values]}


def label_statistics(records, num_classes, focus_indices):
    positive = [0] * num_classes
    cooccurrence = {c: [0] * num_classes for c in focus_indices}
    n = 0
    for row in records:
        n += 1
        for c, value in enumerate(row["labels"]):
            positive[c] += value
        for focus in focus_indices:
            if row["labels"][focus]:
                for c, value in enumerate(row["labels"]):
                    cooccurrence[focus][c] += value
    if not n:
        raise ValueError("Empty split")
    return {"n": n, "positive": positive, "focus_cooccurrence": cooccurrence}


def select_samples(records, count, seed, focus_indices, min_focus_support):
    """Uniform sample first; label-based supplements never enter its metrics."""
    if count < 1 or min_focus_support < 0 or not records:
        raise ValueError("Nonempty records, positive sample count and nonnegative support required")
    rng = random.Random(seed)
    uniform = set(rng.sample(range(len(records)), min(count, len(records))))
    selected = {i: ["uniform"] for i in uniform}
    for c in focus_indices:
        for label in (0, 1):
            present = sum(records[i]["labels"][c] == label for i in selected)
            candidates = [i for i, r in enumerate(records) if i not in selected and r["labels"][c] == label]
            extra = rng.sample(candidates, min(max(min_focus_support - present, 0), len(candidates)))
            for i in extra:
                selected[i] = [f"focus_{c}_label_{label}"]
    return [dict(records[i], cohort="uniform" if i in uniform else "focus_supplement",
                 selection_reasons=selected[i]) for i in sorted(selected)]


def validate_predictions(manifest, predictions, num_classes):
    if len(manifest) != len(predictions):
        raise ValueError("Prediction count does not match manifest")
    for row, prediction in zip(manifest, predictions):
        if row["row_index"] != prediction["row_index"]:
            raise ValueError("Prediction row order does not match manifest")
        scores = prediction["scores"]
        if len(scores) != num_classes or any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
            raise ValueError(f"Invalid probability vector for row {row['row_index']}")


def binary_metrics(labels, scores, threshold=.5):
    """Noninterpolated AP, grouping tied scores as one PR threshold step."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("Labels/scores must have the same nonzero length")
    if any(y not in (0, 1) for y in labels) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError("Invalid labels/scores")
    positive = sum(labels)
    negative = len(labels) - positive
    tp = sum(y == 1 and p >= threshold for y, p in zip(labels, scores))
    fp = sum(y == 0 and p >= threshold for y, p in zip(labels, scores))
    fn, tn = positive - tp, negative - fp
    result = {"positive": positive, "negative": negative, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
              "precision": tp / (tp + fp) if tp + fp else 0.0,
              "recall": tp / positive if positive else 0.0,
              "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
              "ap": None, "rank_status": "ok"}
    if not positive or not negative:
        result["rank_status"] = "no_positive" if not positive else "no_negative"
        return result, []
    curve, found, total, ap = [], 0, 0, 0.0
    for score, group in itertools.groupby(sorted(zip(scores, labels), reverse=True), key=lambda pair: pair[0]):
        group_labels = [y for _, y in group]
        new_positive = sum(group_labels)
        found += new_positive
        total += len(group_labels)
        precision = found / total
        ap += (new_positive / positive) * precision
        curve.append({"threshold": score, "precision": precision, "recall": found / positive})
    result["ap"] = ap
    return result, curve


def case_bucket(label, swin_score, gavit_score, threshold=.5):
    sw_ok = (swin_score >= threshold) == bool(label)
    ga_ok = (gavit_score >= threshold) == bool(label)
    if sw_ok and ga_ok:
        return "both_correct"
    if sw_ok:
        return "swin_only_correct"
    if ga_ok:
        return "gavit_only_correct"
    return "both_wrong"


def write_analysis(output, manifest, swin, gavit, class_names, focus_indices, examples_per_bucket=2):
    output = Path(output)
    for predictions in (swin, gavit):
        validate_predictions(manifest, predictions, len(class_names))
    uniform = [i for i, row in enumerate(manifest) if row["cohort"] == "uniform"]
    uniform_set = set(uniform)
    if not uniform:
        raise ValueError("Uniform cohort is empty")
    metrics, pr_rows = [], []
    for c, name in enumerate(class_names):
        labels = [manifest[i]["labels"][c] for i in uniform]
        record = {"class_index": c, "class": name, "positive": sum(labels), "negative": len(labels) - sum(labels)}
        for model, predictions in (("swin", swin), ("gavit", gavit)):
            m, curve = binary_metrics(labels, [predictions[i]["scores"][c] for i in uniform])
            record["rank_status"] = m["rank_status"]
            for key in ("ap", "f1", "precision", "recall"):
                record[f"{model}_{key}_percent"] = None if m[key] is None else 100 * m[key]
            for key in ("tp", "fp", "fn", "tn"):
                record[f"{model}_{key}"] = m[key]
            pr_rows.extend(dict(class_index=c, model=model, **p) for p in curve)
        record["delta_ap_pp"] = (record["gavit_ap_percent"] - record["swin_ap_percent"]
                                 if record["rank_status"] == "ok" else None)
        metrics.append(record)
    write_csv(output / "per_class.csv", list(metrics[0]), metrics)
    write_csv(output / "pr_curves.csv", ["class_index", "model", "threshold", "precision", "recall"], pr_rows)

    prediction_fields = ["row_index", "patch_path", "cohort"]
    for prefix in ("label", "swin_score", "gavit_score"):
        prediction_fields.extend(f"{prefix}_{c}" for c in range(len(class_names)))

    def prediction_rows():
        for row, sw, ga in zip(manifest, swin, gavit):
            item = {key: row[key] for key in ("row_index", "patch_path", "cohort")}
            for prefix, values in (("label", row["labels"]), ("swin_score", sw["scores"]), ("gavit_score", ga["scores"])):
                item.update({f"{prefix}_{c}": value for c, value in enumerate(values)})
            yield item
    write_csv(output / "predictions.csv", prediction_fields, prediction_rows())

    cases, counts = [], []
    buckets = ("swin_only_correct", "gavit_only_correct", "both_correct", "both_wrong")
    for c in focus_indices:
        for bucket in buckets:
            for label in (0, 1):
                indices = [i for i, row in enumerate(manifest) if row["labels"][c] == label
                           and case_bucket(label, swin[i]["scores"][c], gavit[i]["scores"][c]) == bucket]
                # Stable selection within every outcome; do not pick only the largest score gaps.
                indices.sort(key=lambda i: hashlib.sha256(f"42:{c}:{bucket}:{label}:{manifest[i]['patch_path']}".encode()).digest())
                chosen = indices[:examples_per_bucket]
                counts.append({"class_index": c, "class": class_names[c], "bucket": bucket, "label": label,
                               "available": len(indices), "uniform_available": sum(i in uniform_set for i in indices), "selected": len(chosen)})
                for i in chosen:
                    row = manifest[i]
                    cases.append({"class_index": c, "class": class_names[c], "bucket": bucket, "label": label,
                                  "row_index": row["row_index"], "patch_path": row["patch_path"], "cohort": row["cohort"],
                                  "swin_score": swin[i]["scores"][c], "gavit_score": gavit[i]["scores"][c],
                                  "true_labels": json.dumps([name for name, y in zip(class_names, row["labels"]) if y], ensure_ascii=False),
                                  "image": f"images/row_{row['row_index']}.jpg"})
    write_csv(output / "case_counts.csv", ["class_index", "class", "bucket", "label", "available", "uniform_available", "selected"], counts)
    write_csv(output / "cases.csv", ["class_index", "class", "bucket", "label", "row_index", "patch_path", "cohort",
                                    "swin_score", "gavit_score", "true_labels", "image"], cases)
    summary = {"scope": "validation_subset_diagnostic_not_formal_test", "threshold": .5,
               "uniform_n": len(uniform), "supplement_n": len(manifest) - len(uniform),
               "evaluable_classes": sum(r["rank_status"] == "ok" for r in metrics),
               "class_names": class_names, "per_class": metrics, "case_selection_seed": 42,
               "case_selection": "hash order within each class/outcome/true-label stratum; scores do not rank examples",
               "notes": ["AP/PR use only the uniform validation cohort.",
                         "Supplemental samples support case inspection, not population metrics.",
                         "AP is unavailable when the uniform cohort lacks positives or negatives.",
                         "Precision/recall/F1 zero denominators return zero; inspect support counts."]}
    write_json(output / "summary.json", summary)
    lines = ["# Swin / GAViT 验证样本诊断", "", f"均匀验证样本：{len(uniform)}；独立补充样本：{summary['supplement_n']}。",
             "AP/PR只使用均匀验证样本，非正式测试集结果。成败分类阈值为0.5。", "",
             "| 类别 | 正例数 | Swin AP % | GAViT AP % | 差值 pp |", "|---|---:|---:|---:|---:|"]
    for c in focus_indices:
        r = metrics[c]
        values = ["N/A" if r[k] is None else f"{r[k]:.2f}" for k in ("swin_ap_percent", "gavit_ap_percent", "delta_ap_pp")]
        lines.append(f"| {r['class']} | {r['positive']} | {' | '.join(values)} |")
    lines.extend(["", "完整19类：per_class.csv；配对分数：predictions.csv；成败样本：cases.html / cases.csv。",
                  "样本计数：case_counts.csv；PR曲线数据：pr_curves.csv。",
                  "本次不绘制区域或图边；图反馈是否导致错误，需要后续真实前向诊断。"])
    with (output / "report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return summary
