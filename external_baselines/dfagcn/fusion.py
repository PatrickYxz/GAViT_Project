"""DFAGCN Eq. (6)-(7) ordered weighted concatenation, without normalization."""

import math
from numbers import Real

import torch
from torch import Tensor


def paper_coefficients(dataset: str) -> dict[str, float]:
    """Published fixed values, not values selected on this project's test set."""
    if dataset == "AID":
        return {"alpha": 0.4, "beta": 0.3, "gamma": 1.1}
    if dataset == "NWPU":
        return {"alpha": 0.7, "beta": 0.3, "gamma": 0.4}
    raise ValueError("published coefficients are available only for AID and NWPU")


def _coefficient(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def weighted_concat(conv5: Tensor, conv4: Tensor, conv3: Tensor, fc: Tensor,
                    *, alpha: float, beta: float, gamma: float) -> Tensor:
    """Concatenate [conv5, alpha*conv4, beta*conv3, gamma*fc] to B×5632.

    Conv inputs are already-refined 512-dimensional branch exports, not the
    unpooled VGG maps.  No branch architecture or scaling is inferred here.
    """
    alpha = _coefficient(alpha, "alpha")
    beta = _coefficient(beta, "beta")
    gamma = _coefficient(gamma, "gamma")
    parts = (conv5, conv4, conv3, fc)
    for name, part, width in zip(("conv5", "conv4", "conv3", "fc"), parts,
                                 (512, 512, 512, 4096)):
        if not isinstance(part, Tensor) or not part.is_floating_point():
            raise TypeError(f"{name} must be a real floating Tensor")
        if part.ndim != 2 or part.shape[0] < 1 or part.shape[1] != width:
            raise ValueError(f"{name} must have nonempty shape B×{width}")
        if part.dtype != conv5.dtype or part.device != conv5.device:
            raise ValueError("branches must share dtype and device")
        if part.shape[0] != conv5.shape[0]:
            raise ValueError("branches must share batch size")
        if not torch.isfinite(part).all():
            raise ValueError(f"{name} contains nonfinite features")
    output = torch.cat((conv5, alpha * conv4, beta * conv3, gamma * fc), dim=1)
    if not torch.isfinite(output).all():
        raise ValueError("weighted features overflowed")
    return output
