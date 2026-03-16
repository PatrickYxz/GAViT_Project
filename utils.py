import os
import random
import numpy as np
import torch


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
