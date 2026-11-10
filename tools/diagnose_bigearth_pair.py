#!/usr/bin/env python3
"""Compare existing Swin/GAViT checkpoints on a fixed validation subset.

Can run from an isolated tools bundle via --repo; never trains or writes weights.
--prepare-only validates split CSVs and creates the manifest without torch.
"""

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

if __package__:
    from . import bigearth_pair_core as core
else:
    import bigearth_pair_core as core


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_class_names(repo):
    tree = ast.parse((repo / "models/bigearth_dataset.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CLASSES_19" for t in node.targets):
            names = ast.literal_eval(node.value)
            if len(names) == 19 and len(set(names)) == 19 and all(isinstance(n, str) for n in names):
                return names
    raise ValueError("Repository must define 19 distinct CLASSES_19 names")


def prepare(args, output):
    names = read_class_names(args.repo)
    if any(c < 0 or c >= len(names) for c in args.focus_classes):
        raise ValueError("Focus class index is outside CLASSES_19")
    statistics, fingerprints, val = {}, {}, None
    for split in ("train", "val", "test"):
        path = args.data_dir / f"{split}.csv"
        fingerprints[split] = {"path": str(path), "sha256": sha256(path)}
        records = core.iter_split(path, len(names))
        if split == "val":
            val = list(records)
            records = val
        statistics[split] = core.label_statistics(records, len(names), args.focus_classes)
        print(f"Split {split}: {statistics[split]['n']:,} rows", flush=True)
    counts = [statistics[s]["n"] for s in ("train", "val", "test")]
    if args.expected_counts is not None and counts != args.expected_counts:
        raise ValueError(f"Split counts {counts} != expected {args.expected_counts}")
    manifest = core.select_samples(val, args.samples, args.seed, args.focus_classes, args.min_focus_support)
    core.write_json(output / "manifest.json", manifest)
    fields = ["row_index", "patch_path", "cohort", "selection_reasons"] + [f"label_{c}" for c in range(len(names))]
    rows = []
    for record in manifest:
        row = {k: record[k] for k in ("row_index", "patch_path", "cohort")}
        row["selection_reasons"] = ";".join(record["selection_reasons"])
        row.update({f"label_{c}": y for c, y in enumerate(record["labels"])})
        rows.append(row)
    core.write_csv(output / "manifest.csv", fields, rows)
    core.write_csv(output / "label_support.csv", ["split", "class_index", "class", "positive", "negative", "prevalence"], (
        {"split": split, "class_index": c, "class": name, "positive": stat["positive"][c],
         "negative": stat["n"] - stat["positive"][c], "prevalence": stat["positive"][c] / stat["n"]}
        for split, stat in statistics.items() for c, name in enumerate(names)))
    core.write_json(output / "label_statistics.json", statistics)
    git = {}
    for key, command in (("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--short"])):
        result = subprocess.run(["git", "-C", str(args.repo), *command], capture_output=True, text=True)
        git[key] = result.stdout.strip() if result.returncode == 0 else None
    sources = ["models/bigearth_dataset.py", "models/gavit.py", "models/region_grouping.py",
               "models/graph_construction.py", "models/graph_reasoning.py", "models/swin_backbone.py",
               "models/swin_features.py", "utils.py", "experiment_identity.py"]
    info = {"started_at": datetime.now(timezone.utc).isoformat(), "repo": str(args.repo), "git": git,
            "split_files": fingerprints, "split_counts": counts, "sample_seed": args.seed,
            "requested_uniform_n": args.samples, "min_focus_support": args.min_focus_support,
            "focus_classes": args.focus_classes, "class_names": names,
            "manifest_sha256": sha256(output / "manifest.json"),
            "source_sha256": {name: sha256(args.repo / name) for name in sources if (args.repo / name).is_file()},
            "diagnostic_sha256": {Path(__file__).name: sha256(__file__), Path(core.__file__).name: sha256(core.__file__)},
            "scope": "validation_subset_diagnostic_not_formal_test", "argv": sys.argv}
    core.write_json(output / "preflight.json", info)
    print(f"Manifest: {len(manifest)} images, including {sum(r['cohort'] == 'uniform' for r in manifest)} uniform", flush=True)
    return names, manifest, info


def build_model(name, checkpoint, device):
    import torch
    import timm
    from experiment_identity import load_checkpoint_metadata, validate_checkpoint_identity
    from models.gavit import GAViT
    from models.swin_features import pool_swin_features
    from utils import resolve_arch_config

    metadata = load_checkpoint_metadata(str(checkpoint))
    if metadata is None:
        raise ValueError(f"Checkpoint sidecar is required: {checkpoint}")
    validate_checkpoint_identity(metadata, expected_model=name, expected_dataset="BigEarthNet-19", expected_num_classes=19)
    weight_hash = sha256(checkpoint)
    if metadata.get("checkpoint_sha256") and metadata["checkpoint_sha256"] != weight_hash:
        raise ValueError(f"Checkpoint hash differs from its sidecar: {checkpoint}")
    if name == "swin":
        architecture = metadata["architecture"]
        backbone_name = architecture.get("backbone", "swin_tiny_patch4_window7_224")
        if backbone_name != "swin_tiny_patch4_window7_224":
            raise ValueError(f"Unsupported baseline backbone: {backbone_name}")

        class SwinBaseline(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
                self.classifier = torch.nn.Sequential(
                    torch.nn.LayerNorm(self.backbone.num_features),
                    torch.nn.Dropout(architecture.get("dropout", .1)),
                    torch.nn.Linear(self.backbone.num_features, 19))

            def forward(self, images):
                return self.classifier(pool_swin_features(self.backbone.forward_features(images)))

        model = SwinBaseline()
    else:
        defaults = {"num_classes": 19, "num_regions": 16, "knn_k": 5, "gat_hidden": 256,
                    "gat_heads": 4, "gat_layers": 2, "dropout": .1, "grouping": "attentive_spatial",
                    "edge_type": "sparse_hybrid", "integration": "token_feedback"}
        architecture, _ = resolve_arch_config({}, metadata, defaults)
        model = GAViT(**architecture, pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    print(f"Loaded {name}: {checkpoint}; architecture={architecture}", flush=True)
    return model, {"path": str(checkpoint), "sha256": weight_hash, "metadata": metadata,
                   "parameters": sum(p.numel() for p in model.parameters())}


def predict_pair(models, manifest, device, batch_size, workers, stage):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from models.bigearth_dataset import BigEarthNetDataset

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    labels = np.asarray([row["labels"] for row in manifest], dtype=np.float32)
    dataset = BigEarthNetDataset([row["patch_path"] for row in manifest], labels, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)
    predictions = {name: [] for name in models}
    start, processed = time.perf_counter(), 0
    with torch.inference_mode():
        for batch_index, (images, batch_labels) in enumerate(loader):
            batch_rows = manifest[processed:processed + len(images)]
            if not np.array_equal(batch_labels.numpy(), labels[processed:processed + len(images)]):
                raise ValueError("Dataset labels no longer match manifest row order")
            images = images.to(device)
            for name, model in models.items():
                logits = model(images)
                if logits.shape != (len(batch_rows), 19) or not torch.isfinite(logits).all().item():
                    raise ValueError(f"Invalid {name} logits in {stage}")
                scores = torch.sigmoid(logits).cpu().tolist()
                predictions[name].extend({"row_index": row["row_index"], "scores": score} for row, score in zip(batch_rows, scores))
            processed += len(batch_rows)
            if batch_index == 0 or processed == len(manifest) or (batch_index + 1) % 20 == 0:
                elapsed = time.perf_counter() - start
                rate = processed / elapsed
                print(f"{stage}: {processed}/{len(manifest)} images; paired throughput={rate:.2f} img/s; remaining~{(len(manifest)-processed)/rate:.0f}s", flush=True)
    for prediction in predictions.values():
        core.validate_predictions(manifest, prediction, 19)
    return predictions, {"images": processed, "wall_seconds": time.perf_counter() - start,
                         "batch_size": batch_size, "workers": workers, "pin_memory": False}


def verify_ap_with_sklearn(manifest, predictions):
    from sklearn.metrics import average_precision_score

    indices = [i for i, row in enumerate(manifest) if row["cohort"] == "uniform"]
    checked = 0
    for prediction in predictions.values():
        for c in range(19):
            labels = [manifest[i]["labels"][c] for i in indices]
            scores = [prediction[i]["scores"][c] for i in indices]
            metric, _ = core.binary_metrics(labels, scores)
            if metric["ap"] is not None:
                reference = average_precision_score(labels, scores)
                if abs(metric["ap"] - reference) > 1e-10:
                    raise ValueError(f"AP differs from sklearn for class {c}: {metric['ap']} vs {reference}")
                checked += 1
    return checked


def write_gallery(output):
    titles = {"swin_only_correct": "Swin正确 / GAViT错误", "gavit_only_correct": "GAViT正确 / Swin错误",
              "both_correct": "共同正确", "both_wrong": "共同错误"}
    with (output / "cases.csv").open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    cards = []
    for row in cases:
        cards.append(f"""<article><h2>{html.escape(row['class'])}</h2>
<p><strong>{titles[row['bucket']]}</strong> · 真实标签：{'存在' if row['label'] == '1' else '不存在'}</p>
<img src="{html.escape(row['image'], quote=True)}" alt="验证图像 {row['row_index']}">
<p>Swin {float(row['swin_score']):.3f} · GAViT {float(row['gavit_score']):.3f}</p>
<p class="labels">全部真实标签：{html.escape(row['true_labels'])}</p>
<small>CSV行号 {row['row_index']}（从0计） · {html.escape(row['cohort'])}</small></article>""")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swin / GAViT 验证样本对比</title><style>
body{font:16px/1.6 system-ui,sans-serif;margin:28px;background:#f5f7f9;color:#172a3a}h1{font-size:25px}h2{font-size:17px;margin:0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}article{padding:18px;background:white;border:1px solid #dbe2e8;border-radius:8px}
img{display:block;width:224px;height:224px;max-width:100%;object-fit:contain;margin:12px auto;background:#eee}p{margin:8px 0}.labels,small{font-size:13px;color:#536373;overflow-wrap:anywhere}
</style><h1>Swin / GAViT 验证样本对比</h1>
<p>相同验证图片、原有权重；成败阈值0.5。每类保留双向分歧、共同正确和共同错误，并区分真实正负标签。</p>
<p>样本在各组内按固定哈希顺序选择；补充样本仅供案例分析。可用/选中数量见 case_counts.csv，随机验证子集指标见 report.md。</p><main class="grid">"""
    with (output / "cases.html").open("x", encoding="utf-8") as handle:
        handle.write(page + "\n".join(cards) + "</main></html>\n")


def export_case_images(output):
    from PIL import Image
    from models.bigearth_dataset import read_rgb_patch

    (output / "images").mkdir()
    with (output / "cases.csv").open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    for row in {r["row_index"]: r for r in cases}.values():
        image = Image.fromarray(read_rgb_patch(row["patch_path"]))
        image.resize((224, 224), Image.Resampling.BILINEAR).save(output / row["image"], quality=92)
    write_gallery(output)


def validate_image_paths(manifest):
    for row in manifest:
        patch = Path(row["patch_path"])
        if any((patch / (patch.name + ext)).is_file() for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG")):
            continue
        for band in ("B04", "B03", "B02"):
            image = patch / f"{patch.name}_{band}.tif"
            if not image.is_file():
                raise FileNotFoundError(f"Selected image is missing: {image}")


def run_inference(args, output, names, manifest, info):
    # All tensor imports are delayed so preparation/help work without a GPU runtime.
    sys.path.insert(0, str(args.repo))
    os.chdir(args.repo)
    validate_image_paths(manifest)
    print("Selected image files: PASS", flush=True)
    import torch
    from models.bigearth_dataset import CLASSES_19
    from utils import set_seed

    if names != CLASSES_19:
        raise ValueError("Runtime class order differs from manifest")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; stop before inference")
    if not args.swin_ckpt or not args.gavit_ckpt:
        raise ValueError("Both --swin-ckpt and --gavit-ckpt are required for inference")
    set_seed(args.seed)
    info["environment"] = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                           "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
                           "packages": {p: importlib.metadata.version(p) for p in ("timm", "torchvision", "numpy", "scikit-learn", "torch-geometric")}}
    print(f"Device: {info['environment']['device']}; torch={torch.__version__}; CUDA={torch.version.cuda}", flush=True)
    models, info["checkpoints"] = {}, {}
    for name, checkpoint in (("swin", args.swin_ckpt), ("gavit", args.gavit_ckpt)):
        models[name], info["checkpoints"][name] = build_model(name, checkpoint, args.device)
    core.write_json(output / "loaded_models.json", {"environment": info["environment"], "checkpoints": info["checkpoints"]})
    smoke_manifest = manifest[:min(8, len(manifest))]
    smoke, info["smoke"] = predict_pair(models, smoke_manifest, args.device, 8, 0, "Smoke")
    core.write_json(output / "smoke_scores.json", smoke)
    print("SMOKE PASS: both models produced finite 19-class scores on the same images", flush=True)
    if args.smoke_only:
        return "smoke_passed_only"
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    predictions, info["inference"] = predict_pair(models, manifest, args.device, args.batch_size, args.workers, "Validation diagnostic")
    if args.device == "cuda":
        info["inference"]["paired_peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    for name, prediction in predictions.items():
        core.write_json(output / f"{name}_scores.json", prediction)
    info["ap_reference_checks"] = verify_ap_with_sklearn(manifest, predictions)
    core.write_analysis(output, manifest, predictions["swin"], predictions["gavit"], names, args.focus_classes, args.examples_per_bucket)
    export_case_images(output)
    return "complete"


def integer_list(value):
    try:
        return [int(v) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc


def finish_run(output, info):
    # Only write the completion marker after all report/packaging work succeeds.
    if info["status"] == "complete":
        with zipfile.ZipFile(output / "results.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file() and path.name != "results.zip":
                    archive.write(path, path.relative_to(output))
            archive.writestr("run.json", json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    core.write_json(output / "run.json", info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--swin-ckpt", type=Path)
    parser.add_argument("--gavit-ckpt", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="New directory; existing outputs are refused")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42, help="Validation sampling/inference seed, not training seed")
    parser.add_argument("--min-focus-support", type=int, default=64, help="Target each focus class/label support with separately tagged supplements")
    parser.add_argument("--focus-classes", type=integer_list, default=[11, 14, 16])
    parser.add_argument("--examples-per-bucket", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--expected-counts", type=integer_list, help="train,val,test row counts")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.batch_size, args.examples_per_bucket) < 1 or min(args.workers, args.min_focus_support) < 0:
        parser.error("Invalid sample/batch/example/support/worker counts")
    if not args.focus_classes or len(set(args.focus_classes)) != len(args.focus_classes):
        parser.error("Focus classes must be nonempty and distinct")
    if args.expected_counts is not None and len(args.expected_counts) != 3:
        parser.error("--expected-counts requires train,val,test")
    if args.prepare_only and args.smoke_only:
        parser.error("Choose either --prepare-only or --smoke-only")
    for key in ("repo", "data_dir", "swin_ckpt", "gavit_ckpt", "out"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    output = args.out
    output.mkdir(parents=True, exist_ok=False)
    try:
        names, manifest, info = prepare(args, output)
        status = "prepared_only" if args.prepare_only else run_inference(args, output, names, manifest, info)
        info.update(status=status, finished_at=datetime.now(timezone.utc).isoformat())
        finish_run(output, info)
        if status == "complete":
            print((output / "report.md").read_text(), flush=True)
        print(f"DIAGNOSTIC {status.upper()}: {output}", flush=True)
        return 0
    except Exception as exc:
        core.write_json(output / "failure.json", {"status": "failed", "error": str(exc), "type": type(exc).__name__})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
