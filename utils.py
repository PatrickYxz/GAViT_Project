import random
import numpy as np
import torch

from experiment_identity import (
    get_git_state,
    load_checkpoint_metadata,
    metadata_path_for,
    validate_complete_architecture,
    write_checkpoint_metadata,
)


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
    torch.save(model.state_dict(), ckpt_path)
    meta = dict(metadata)
    meta["git"] = get_git_state()
    write_checkpoint_metadata(ckpt_path, meta)


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
