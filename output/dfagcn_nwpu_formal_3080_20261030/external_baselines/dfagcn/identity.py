"""JSON-only cache identity; callers must verify the source files themselves.

This index contains provenance metadata, not tensors, labels or a model.  A key
match cannot prove that supplied hashes are true or that a fitted model avoided
test data.  It prevents accidental reuse across declared extraction identities.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath


SCHEMA = "dfagcn-feature-cache-v1"


def _sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def _json_copy(value):
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("JSON mapping keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
    check(value)
    return json.loads(json.dumps(value, allow_nan=False))


def _metadata(value, name):
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty JSON mapping")
    return _json_copy(dict(value))


def build_cache_identity(*, manifest_sha256, class_names, samples, weights_sha256,
                         preprocessing, feature_spec, source_sha256, stage,
                         fitted_on_sha256=None, training_config=None):
    """Build a detached identity.  Sample order and class index order matter.

    ``frozen_pretrained`` is reusable across model fitting phases.  Learned
    calibration/refit features additionally require their fitting-data digest
    and training configuration; neither is inferred from a filename.
    """
    if (not isinstance(class_names, Sequence) or isinstance(class_names, str)
            or not class_names or any(not isinstance(x, str) or not x.strip() for x in class_names)
            or len(set(class_names)) != len(class_names)):
        raise ValueError("class_names must be a nonempty ordered sequence of unique names")
    if not isinstance(samples, Sequence) or isinstance(samples, str) or not samples:
        raise ValueError("samples must be a nonempty ordered sequence")
    checked_samples = []
    seen = set()
    for sample in samples:
        if not isinstance(sample, Mapping) or set(sample) != {"path", "sha256"}:
            raise ValueError("each sample requires exactly path and sha256")
        path = sample["path"]
        if (not isinstance(path, str) or not path or "\\" in path
                or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
                or PurePosixPath(path).as_posix() != path or path == "." or path in seen):
            raise ValueError("sample paths must be unique normalized relative POSIX paths")
        seen.add(path)
        checked_samples.append({"path": path, "sha256": _sha(sample["sha256"], "sample sha256")})
    if stage not in {"frozen_pretrained", "calibration", "refit"}:
        raise ValueError("stage must be frozen_pretrained, calibration or refit")
    if stage == "frozen_pretrained":
        if fitted_on_sha256 is not None or training_config is not None:
            raise ValueError("frozen_pretrained identity cannot declare model fitting")
        fit_digest, training = None, None
    else:
        fit_digest = _sha(fitted_on_sha256, "fitted_on_sha256")
        training = _metadata(training_config, "training_config")
    return {"schema": SCHEMA,
            "manifest_sha256": _sha(manifest_sha256, "manifest_sha256"),
            "class_names": list(class_names), "samples": checked_samples,
            "weights_sha256": _sha(weights_sha256, "weights_sha256"),
            "preprocessing": _metadata(preprocessing, "preprocessing"),
            "feature_spec": _metadata(feature_spec, "feature_spec"),
            "source_sha256": _sha(source_sha256, "source_sha256"), "stage": stage,
            "fitted_on_sha256": fit_digest, "training_config": training}


def _checked(identity):
    if not isinstance(identity, Mapping) or identity.get("schema") != SCHEMA:
        raise ValueError("unsupported cache identity schema")
    args = dict(identity)
    args.pop("schema")
    return build_cache_identity(**args)


def cache_key(identity):
    canonical = json.dumps(_checked(identity), sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_cache_index(path, identity):
    """Exclusively create an identity index; never overwrite a prior cache."""
    checked = _checked(identity)
    record = {"cache_key": cache_key(checked), "identity": checked}
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def read_cache_index(path, *, expected_identity):
    """Reject altered metadata and any mismatch with caller's expected identity."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict) or set(record) != {"cache_key", "identity"}:
        raise ValueError("invalid cache index record")
    checked = _checked(record["identity"])
    actual_key = cache_key(checked)
    if record["cache_key"] != actual_key or cache_key(expected_identity) != actual_key:
        raise ValueError("cache index identity mismatch or metadata alteration")
    return checked
