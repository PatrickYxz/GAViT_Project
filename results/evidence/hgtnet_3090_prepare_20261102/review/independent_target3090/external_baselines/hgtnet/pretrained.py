"""Audited *candidate* Swin-B initialization for the independent HGTNet model.

No constructor or function downloads anything. These are torchvision ImageNet
weights, not verified original HGTNet weights. A schema audit is not a load.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import hashlib
from pathlib import Path
import re
from typing import Any

import torch
from torch import Tensor, nn


CANDIDATE_KIND = "torchvision.Swin_B_Weights.IMAGENET1K_V1"
CANDIDATE_URL = "https://download.pytorch.org/models/swin_b-68c6b09e.pth"
CANDIDATE_SHA256_PREFIX = "68c6b09e"
_STAGE_DEPTHS = (2, 2, 18)


class PretrainedValidationError(ValueError):
    """A preflight failed; ``report`` contains JSON-safe evidence when available."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report


def _source_for_target(target: str) -> str | None:
    stem = re.fullmatch(r"patch_embed\.(projection|norm)\.(weight|bias)", target)
    if stem:
        return f"features.0.{0 if stem[1] == 'projection' else 2}.{stem[2]}"
    block = re.fullmatch(r"stages\.([012])\.blocks\.(\d+)\.(.+)", target)
    if block and int(block[2]) < _STAGE_DEPTHS[int(block[1])]:
        return f"features.{1 + 2 * int(block[1])}.{block[2]}.{block[3]}"
    merge = re.fullmatch(r"stages\.([01])\.downsample\.(norm\.(?:weight|bias)|reduction\.weight)", target)
    if merge:
        return f"features.{2 + 2 * int(merge[1])}.{merge[2]}"
    return None


def _new_target(target: str) -> bool:
    return bool(re.fullmatch(
        r"(?:stages\.[012]\.da|stages\.2\.downsample|graph_transition[12]|graph_blocks[12]|head)\..+",
        target,
    ))


def _skipped_source(source: str) -> bool:
    return bool(re.fullmatch(r"(?:features\.[67]|norm|head)\..+", source)) and source in _official_source_schema()


def _tensor_mapping(state: Mapping[str, Tensor]) -> None:
    if not isinstance(state, Mapping):
        raise TypeError("Pretrained state must be a mapping of string keys to tensors")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise TypeError("Pretrained state must contain only string keys and tensors")


def audit_pretrained_state(
    model: nn.Module, state: Mapping[str, Tensor], *, validate_values: bool = True,
) -> dict[str, Any]:
    """Inspect target compatibility and all supplied values without changing tensors.

    Tiny modules with the documented names may exercise the mapper; only the file
    loader validates the complete official Swin-B source schema. All missing
    reusable targets, unexpected sources, shape/dtype/layout errors and nonfinite
    source values prevent application. Meta or disabled value checks are explicitly
    schema-only and cannot authorize a load.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(validate_values, bool):
        raise TypeError("validate_values must be bool")
    _tensor_mapping(state)
    target_state = model.state_dict()
    _tensor_mapping(target_state)
    report: dict[str, Any] = {
        "schema_version": "hgtnet-pretrained-audit-v1",
        "candidate_kind": CANDIDATE_KIND,
        "is_original_hgtnet_weights": False,
        "source_identity_verified": False,
        "mode": "values_validated",
        "compatible": [],
        "missing": [],
        "unexpected": [],
        "shape_mismatch": [],
        "dtype_mismatch": [],
        "nonfinite": [],
        "invalid": [],
        "intentionally_initialized": [],
        "source_intentionally_skipped": [],
    }
    expected_sources: dict[str, str] = {}
    for target in sorted(target_state):
        source = _source_for_target(target)
        if source is not None:
            expected_sources[source] = target
            if source not in state:
                report["missing"].append(target)
        elif _new_target(target):
            report["intentionally_initialized"].append(target)
        else:
            report["missing"].append(target)

    has_meta = any(value.is_meta for value in state.values()) or any(value.is_meta for value in target_state.values())
    if not validate_values or has_meta:
        report["mode"] = "schema_only"
    for source in sorted(state):
        value = state[source]
        target = expected_sources.get(source)
        if target is None:
            if _skipped_source(source):
                report["source_intentionally_skipped"].append(source)
            else:
                report["unexpected"].append(source)
        invalid_layout = value.layout != torch.strided or value.is_quantized
        if invalid_layout:
            report["invalid"].append({"source": source, "target": target, "reason": "unsupported_tensor_layout"})
        if validate_values and not value.is_meta and not invalid_layout:
            if not bool(torch.isfinite(value).all().item()):
                report["nonfinite"].append({"source": source, "target": target})
        if target is None:
            continue
        destination = target_state[target]
        shape_ok = value.shape == destination.shape
        dtype_ok = value.dtype == destination.dtype
        target_layout_ok = destination.layout == torch.strided and not destination.is_quantized
        if not shape_ok:
            report["shape_mismatch"].append({
                "source": source, "target": target,
                "source_shape": list(value.shape), "target_shape": list(destination.shape),
            })
        if not dtype_ok:
            report["dtype_mismatch"].append({
                "source": source, "target": target,
                "source_dtype": str(value.dtype), "target_dtype": str(destination.dtype),
            })
        if not target_layout_ok:
            report["invalid"].append({"source": source, "target": target, "reason": "unsupported_target_layout"})
        if shape_ok and dtype_ok and not invalid_layout and target_layout_ok:
            report["compatible"].append({"source": source, "target": target, "numel": value.numel()})

    compatible_numel = sum(item["numel"] for item in report["compatible"])
    target_numel = sum(value.numel() for value in target_state.values())
    source_numel = sum(value.numel() for value in state.values())
    report["coverage"] = {
        "target_tensors": len(target_state), "target_numel": target_numel,
        "source_tensors": len(state), "source_numel": source_numel,
        "compatible_tensors": len(report["compatible"]), "compatible_numel": compatible_numel,
        "compatible_target_numel_fraction": compatible_numel / target_numel if target_numel else 0.0,
        "compatible_source_numel_fraction": compatible_numel / source_numel if source_numel else 0.0,
        "intentionally_initialized_tensors": len(report["intentionally_initialized"]),
        "intentionally_initialized_numel": sum(target_state[key].numel() for key in report["intentionally_initialized"]),
    }
    report["can_apply"] = (
        report["mode"] == "values_validated" and bool(report["compatible"])
        and not any(report[key] for key in ("missing", "unexpected", "shape_mismatch", "dtype_mismatch", "nonfinite", "invalid"))
    )
    return report


def apply_pretrained_state(model: nn.Module, state: Mapping[str, Tensor]) -> dict[str, Any]:
    """Preflight the entire state, then copy only the explicitly compatible keys."""
    report = audit_pretrained_state(model, state, validate_values=True)
    if not report["can_apply"]:
        raise PretrainedValidationError("Candidate pretraining failed preflight; model was not changed", report)
    target_state = model.state_dict()
    # Device conversions, including possible allocation failures, happen before
    # any model mutation. clone prevents source/target storage aliasing.
    copied = {
        item["target"]: state[item["source"]].detach().to(device=target_state[item["target"]].device).clone()
        for item in report["compatible"]
    }
    # Audit has already proved the expected missing keys exactly equal the new
    # modules, with no unexpected keys or size/dtype mismatches.
    result = model.load_state_dict(copied, strict=False)
    if sorted(result.missing_keys) != report["intentionally_initialized"] or result.unexpected_keys:
        raise RuntimeError("Model load_state_dict hooks disagreed with the preflight schema")
    report["loaded"] = [dict(item) for item in report["compatible"]]
    report["coverage"].update(
        loaded_tensors=len(report["loaded"]),
        loaded_numel=report["coverage"]["compatible_numel"],
        loaded_target_numel_fraction=report["coverage"]["compatible_target_numel_fraction"],
    )
    report["mode"] = "applied"
    return report


@lru_cache(maxsize=1)
def _official_source_schema() -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    # No parameters or activations are allocated on CPU and weights=None never
    # downloads. This schema describes torchvision, not original HGTNet weights.
    from torchvision.models import swin_b
    with torch.device("meta"):
        model = swin_b(weights=None)
    return {key: (tuple(value.shape), value.dtype) for key, value in model.state_dict().items()}


def _validate_official_source(state: Mapping[str, Tensor]) -> dict[str, Any]:
    _tensor_mapping(state)
    schema = _official_source_schema()
    report: dict[str, Any] = {
        "missing": sorted(set(schema) - set(state)), "unexpected": sorted(set(state) - set(schema)),
        "shape_mismatch": [], "dtype_mismatch": [], "nonfinite": [], "invalid": [],
    }
    for key, value in state.items():
        if key in schema:
            shape, dtype = schema[key]
            if tuple(value.shape) != shape:
                report["shape_mismatch"].append({"source": key, "source_shape": list(value.shape), "expected_shape": list(shape)})
            if value.dtype != dtype:
                report["dtype_mismatch"].append({"source": key, "source_dtype": str(value.dtype), "expected_dtype": str(dtype)})
        if value.is_meta or value.layout != torch.strided or value.is_quantized:
            report["invalid"].append(key)
        elif not bool(torch.isfinite(value).all().item()):
            report["nonfinite"].append(key)
    if any(report.values()):
        raise PretrainedValidationError("Candidate file does not contain a complete finite official Swin-B source state", report)
    return {"complete_schema": True, "all_values_finite": True, "source_tensors": len(state),
            "source_numel": sum(value.numel() for value in state.values()), "schema_source": "installed torchvision.models.swin_b(weights=None) on meta"}


def load_pretrained_file(model: nn.Module, path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Hash and safely load an existing official Swin-B candidate, never download.

    ``expected_sha256`` is the full measured 64-character identity, also required
    to match the official URL filename's eight-character prefix. The complete
    source schema and every source tensor value are validated before application.
    """
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None:
        raise PretrainedValidationError("expected_sha256 must be a complete 64-character hexadecimal SHA256")
    expected_sha256 = expected_sha256.lower()
    if not expected_sha256.startswith(CANDIDATE_SHA256_PREFIX):
        raise PretrainedValidationError("Expected SHA256 must match the official Swin-B filename prefix")
    source_path = Path(path).expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise PretrainedValidationError("Candidate path must be a regular file")
    with source_path.open("rb") as stream:
        actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_sha != expected_sha256:
            raise PretrainedValidationError("Candidate file SHA256 does not match the complete expected identity")
        stream.seek(0)
        # Same open descriptor binds the verified bytes to the safely loaded
        # state. No arbitrary pickle globals and no device-origin side effects.
        state = torch.load(stream, map_location="cpu", weights_only=True)
    source_validation = _validate_official_source(state)
    report = apply_pretrained_state(model, state)
    report["source_identity_verified"] = True
    report["source_validation"] = source_validation
    report["source"] = {
        "kind": CANDIDATE_KIND, "url": CANDIDATE_URL,
        "path": str(source_path), "sha256": actual_sha, "bytes": source_path.stat().st_size,
        "is_original_hgtnet_weights": False,
    }
    return report
