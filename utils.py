import os
import random
import numpy as np
import torch

from experiment_identity import (
    get_git_state,
    load_checkpoint_metadata,
    metadata_path_for,
    resume_identity_for,
    sha256_file,
    training_state_path_for,
    validate_checkpoint_identity,
    validate_complete_architecture,
    write_checkpoint_metadata,
)


TRAINING_STATE_FORMAT_VERSION = 1


def _atomic_torch_save(payload, path: str) -> None:
    """Write a torch artifact without destroying the previous valid file."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temp_path = path + ".tmp"
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy for a batch (returns float in [0, 100])."""
    preds = outputs.argmax(dim=1)
    return 100.0 * (preds == labels).sum().item() / labels.size(0)


def save_checkpoint(model: torch.nn.Module, ckpt_path: str, metadata: dict):
    """Save model state_dict plus a JSON sidecar describing the experiment.

    The sidecar records the full architecture config, training hyperparameters,
    seed, git commit and best-metric info so that evaluation scripts can
    reconstruct the exact same model without hand-duplicated CLI flags.
    """
    _atomic_torch_save(model.state_dict(), ckpt_path)
    metadata["checkpoint_sha256"] = sha256_file(ckpt_path)
    meta = dict(metadata)
    meta["git"] = get_git_state()
    write_checkpoint_metadata(ckpt_path, meta)


def save_training_state(
    model,
    optimizer,
    scheduler,
    path: str,
    *,
    epoch: int,
    best_metric: float,
    resume_identity: dict | None = None,
) -> None:
    """Save the exact last-epoch state required to resume training."""
    _atomic_torch_save(
        {
            "format_version": TRAINING_STATE_FORMAT_VERSION,
            "resume_identity": resume_identity,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )


def load_training_state(
    model,
    optimizer,
    scheduler,
    path: str,
    *,
    map_location: str,
    expected_identity: dict | None = None,
) -> tuple[int, float]:
    """Restore an exact last-epoch state and return its next epoch/metric."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Training state not found: {path}")

    state = torch.load(path, map_location=map_location, weights_only=True)
    if state.get("format_version") != TRAINING_STATE_FORMAT_VERSION:
        raise ValueError(
            "Unsupported or missing training-state format_version: "
            f"{state.get('format_version')!r}"
        )
    if (
        expected_identity is not None
        and state.get("resume_identity") != expected_identity
    ):
        raise ValueError(
            "Training-state experiment identity does not match checkpoint metadata"
        )
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["epoch"]) + 1, float(state["best_metric"])


def resolve_arch_config(cli_overrides: dict, metadata: dict | None,
                        defaults: dict) -> tuple[dict, str]:
    """Resolve the architecture config used to rebuild a saved model.

    Args:
        cli_overrides: key -> value explicitly passed on the CLI, or None if
            the user did not pass that flag.
        metadata: sidecar dict loaded by load_checkpoint_metadata, or None.
        defaults: fallback values used when neither CLI nor metadata provide
            a key.

    Returns:
        (config, source) where source is "metadata" or "cli".

    Raises:
        ValueError: if a CLI value conflicts with the checkpoint metadata.
    """
    if metadata is None:
        cfg = {
            k: (cli_overrides[k] if cli_overrides.get(k) is not None else v)
            for k, v in defaults.items()
        }
        return cfg, "cli"

    validate_complete_architecture(metadata, tuple(defaults))
    arch = metadata["architecture"]
    cfg, conflicts = {}, []
    for k, default in defaults.items():
        meta_val = arch.get(k, default)
        cli_val = cli_overrides.get(k)
        if cli_val is not None and cli_val != meta_val:
            conflicts.append(f"  {k}: CLI={cli_val!r} vs metadata={meta_val!r}")
        cfg[k] = meta_val
    if conflicts:
        raise ValueError(
            "CLI arguments conflict with checkpoint metadata "
            f"({metadata_path_for(metadata.get('checkpoint_path', '?'))}):\n"
            + "\n".join(conflicts)
            + "\nDrop the conflicting flags, or point --ckpt at the "
              "checkpoint they were trained with."
        )
    return cfg, "metadata"


def load_validated_training_state(
    model,
    optimizer,
    scheduler,
    ckpt_path: str,
    *,
    map_location: str,
    expected_model: str,
    expected_dataset: str,
    expected_num_classes: int,
    expected_architecture: dict,
) -> tuple[int, float]:
    """Validate resume identity, then restore the exact last-epoch state."""
    metadata = load_checkpoint_metadata(ckpt_path)
    if metadata is None:
        raise FileNotFoundError(
            f"Checkpoint metadata not found: {metadata_path_for(ckpt_path)}"
        )

    validate_checkpoint_identity(
        metadata,
        expected_model=expected_model,
        expected_dataset=expected_dataset,
        expected_num_classes=expected_num_classes,
    )
    resolve_arch_config(
        dict(expected_architecture),
        metadata,
        dict(expected_architecture),
    )
    expected_hash = metadata.get("checkpoint_sha256")
    if expected_hash is not None:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")
        actual_hash = sha256_file(ckpt_path)
        if actual_hash != expected_hash:
            raise ValueError(
                "Best checkpoint hash does not match checkpoint metadata: "
                f"actual={actual_hash}, expected={expected_hash}"
            )
    return load_training_state(
        model,
        optimizer,
        scheduler,
        training_state_path_for(ckpt_path),
        map_location=map_location,
        expected_identity=resume_identity_for(metadata),
    )


def save_epoch_artifacts(
    model,
    optimizer,
    scheduler,
    ckpt_path: str,
    metadata: dict,
    *,
    epoch: int,
    metric_name: str,
    metric_value: float,
    best_metric: float,
) -> tuple[float, bool]:
    """Keep the best model separate while always saving the last epoch."""
    improved = metric_value > best_metric
    if improved:
        best_metric = metric_value
        metadata["best"] = {
            "metric": metric_name,
            "value": round(metric_value, 4),
            "epoch": epoch,
        }
        save_checkpoint(model, ckpt_path, metadata)

    save_training_state(
        model,
        optimizer,
        scheduler,
        training_state_path_for(ckpt_path),
        epoch=epoch,
        best_metric=best_metric,
        resume_identity=resume_identity_for(metadata),
    )
    return best_metric, improved
