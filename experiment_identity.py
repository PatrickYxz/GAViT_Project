"""Torch-free experiment identity and checkpoint metadata helpers."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import subprocess


VALID_RUN_STAGES = ("smoke", "proxy", "formal")
RUN_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
RESUME_IDENTITY_KEYS = (
    "model",
    "dataset",
    "architecture",
    "training",
    "execution",
    "data",
    "pretrained",
    "best",
    "checkpoint_sha256",
)


def _run_git(args: list[str], cwd: str | None) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_git_state(cwd: str | None = None) -> dict:
    """Return the current commit and whether tracked/untracked files differ."""
    commit = _run_git(["rev-parse", "--short", "HEAD"], cwd)
    status = _run_git(["status", "--porcelain", "--untracked-files=all"], cwd)
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def assert_clean_git_state(state: dict) -> None:
    """Require a known commit with no tracked or untracked changes."""
    if not state.get("commit") or state.get("dirty") is not False:
        raise RuntimeError(
            "Formal runs require a clean Git commit; "
            f"observed commit={state.get('commit')!r}, dirty={state.get('dirty')!r}"
        )


def validate_run_tag(run_tag: str) -> str:
    """Validate a filesystem-safe, human-readable experiment tag."""
    if not isinstance(run_tag, str) or RUN_TAG_PATTERN.fullmatch(run_tag) is None:
        raise ValueError(
            "run_tag must match [A-Za-z0-9][A-Za-z0-9._-]*"
        )
    return run_tag


def validate_run_stage(run_stage: str) -> str:
    """Validate the controlled experiment tier."""
    if run_stage not in VALID_RUN_STAGES:
        raise ValueError(
            f"run_stage must be one of {VALID_RUN_STAGES}, got {run_stage!r}"
        )
    return run_stage


def metadata_path_for(ckpt_path: str) -> str:
    """Return the JSON sidecar path associated with a checkpoint."""
    base, _ = os.path.splitext(ckpt_path)
    return base + ".meta.json"


def training_state_path_for(ckpt_path: str) -> str:
    """Return a last-epoch state path separate from the best checkpoint."""
    base, _ = os.path.splitext(ckpt_path)
    return base + ".last.train_state.pth"


def resume_identity_for(metadata: dict) -> dict:
    """Return the immutable experiment snapshot bound to a training state."""
    return {
        key: deepcopy(metadata[key])
        for key in RESUME_IDENTITY_KEYS
        if key in metadata
    }


def sha256_file(path: str) -> str:
    """Return a file SHA-256 without loading the whole artifact into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_fresh_output_paths(paths: list[str]) -> None:
    """Refuse to overwrite any artifact belonging to a fresh run."""
    collisions = [os.path.abspath(path) for path in paths if os.path.exists(path)]
    if collisions:
        raise FileExistsError(
            "Fresh run would overwrite existing artifacts:\n  "
            + "\n  ".join(collisions)
        )


def write_checkpoint_metadata(ckpt_path: str, metadata: dict) -> None:
    """Write checkpoint metadata atomically as a JSON sidecar."""
    meta_path = metadata_path_for(ckpt_path)
    parent = os.path.dirname(meta_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    serialized = deepcopy(metadata)
    serialized["checkpoint_path"] = ckpt_path
    serialized["saved_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )

    temp_path = meta_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=2, ensure_ascii=False)
    os.replace(temp_path, meta_path)


def load_checkpoint_metadata(ckpt_path: str) -> dict | None:
    """Load checkpoint metadata, returning None for a legacy checkpoint."""
    meta_path = metadata_path_for(ckpt_path)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as handle:
        return json.load(handle)


def validate_checkpoint_identity(
    metadata: dict,
    *,
    expected_model: str,
    expected_dataset: str,
    expected_num_classes: int,
) -> None:
    """Reject a checkpoint belonging to a different model or dataset."""
    architecture = metadata.get("architecture")
    actual_num_classes = (
        architecture.get("num_classes")
        if isinstance(architecture, dict)
        else None
    )
    expected = {
        "model": expected_model,
        "dataset": expected_dataset,
        "num_classes": expected_num_classes,
    }
    actual = {
        "model": metadata.get("model"),
        "dataset": metadata.get("dataset"),
        "num_classes": actual_num_classes,
    }
    mismatches = [
        f"{key}: checkpoint={actual[key]!r}, expected={expected[key]!r}"
        for key in expected
        if actual[key] != expected[key]
    ]
    if mismatches:
        raise ValueError(
            "Checkpoint identity mismatch:\n  " + "\n  ".join(mismatches)
        )


def validate_complete_architecture(
    metadata: dict, required_keys: tuple[str, ...]
) -> None:
    """Require every architecture key needed to rebuild a new checkpoint."""
    architecture = metadata.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("Checkpoint metadata is missing architecture")
    missing = [key for key in required_keys if key not in architecture]
    if missing:
        raise ValueError(
            "Checkpoint architecture is incomplete; missing: "
            + ", ".join(missing)
        )
