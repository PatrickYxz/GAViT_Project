"""Explicit recipe validation for the independent DFAGCN implementation.

The recipe is an executable project choice, not a transcription of Table I.
Missing fields never acquire undocumented architecture/optimization defaults.
"""

from __future__ import annotations

import copy
import json
import math
from numbers import Real
from pathlib import Path


_REQUIRED = {
    "schema", "implementation", "pooling_hw", "knn_k", "metric", "gcn_dims", "fc_dims",
    "dropout", "seed", "epochs", "batch_size", "learning_rate", "weight_decay", "lr_step",
    "lr_gamma", "fusion", "C_grid", "liblinear_solver", "liblinear_bias",
}


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _real(value, name, *, minimum=0.0, strict=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    if not math.isfinite(float(value)) or (value <= minimum if strict else value < minimum):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {relation} {minimum}")


def validate_recipe(recipe: dict) -> dict:
    """Return a detached validated recipe without filling missing values."""
    if not isinstance(recipe, dict):
        raise TypeError("recipe must be a JSON object")
    missing, unknown = _REQUIRED - set(recipe), set(recipe) - _REQUIRED - {"choices"}
    if missing or unknown:
        raise ValueError(f"recipe fields missing={sorted(missing)}, unknown={sorted(unknown)}")
    if recipe["schema"] != "dfagcn-independent-recipe-v1":
        raise ValueError("unsupported independent recipe schema")
    if recipe["implementation"] != "dfagcn_independent_v1":
        raise ValueError("unsupported independent implementation")
    pooling = recipe["pooling_hw"]
    if not isinstance(pooling, list) or len(pooling) != 2:
        raise ValueError("pooling_hw must be two explicit positive integers")
    for size in pooling:
        _integer(size, "pooling_hw")
    _integer(recipe["knn_k"], "knn_k")
    if recipe["knn_k"] >= pooling[0] * pooling[1]:
        raise ValueError("pooled node count must exceed knn_k")
    if recipe["metric"] not in ("euclidean", "cosine"):
        raise ValueError("metric must be euclidean or cosine")
    for name in ("gcn_dims", "fc_dims"):
        if not isinstance(recipe[name], list) or not recipe[name]:
            raise ValueError(f"{name} must contain explicit positive layer widths")
        for width in recipe[name]:
            _integer(width, name)
    if recipe["fc_dims"][-1] != 512:
        raise ValueError("the final FC/ReLU branch export must have 512 dimensions")
    _real(recipe["dropout"], "dropout")
    if recipe["dropout"] >= 1:
        raise ValueError("dropout must be less than one")
    _integer(recipe["seed"], "seed", minimum=0)
    if recipe["seed"] >= 2 ** 32:
        raise ValueError("seed must fit the NumPy/Python reproducibility range")
    for name in ("epochs", "batch_size", "lr_step"):
        _integer(recipe[name], name)
    _real(recipe["learning_rate"], "learning_rate", strict=True)
    _real(recipe["weight_decay"], "weight_decay")
    _real(recipe["lr_gamma"], "lr_gamma", strict=True)
    if recipe["lr_gamma"] > 1:
        raise ValueError("lr_gamma must not increase learning rate")
    fusion = recipe["fusion"]
    if not isinstance(fusion, dict) or set(fusion) != {"alpha", "beta", "gamma"}:
        raise ValueError("fusion must declare exactly alpha, beta and gamma")
    for name, value in fusion.items():
        _real(value, f"fusion.{name}")
    grid = recipe["C_grid"]
    if not isinstance(grid, list) or not grid:
        raise ValueError("C_grid must contain explicit positive values")
    for value in grid:
        _real(value, "C_grid", strict=True)
    if any(right <= left for left, right in zip(grid, grid[1:])):
        raise ValueError("C_grid must be unique and strictly increasing")
    if type(recipe["liblinear_solver"]) is not int or recipe["liblinear_solver"] != 2:
        raise ValueError("independent v1 explicitly uses LIBLINEAR solver 2")
    _real(recipe["liblinear_bias"], "liblinear_bias")
    if "choices" in recipe:
        choices = recipe["choices"]
        if not isinstance(choices, dict) or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in choices.items()
        ):
            raise ValueError("choices must map nonempty strings to explanatory strings")
    # The public file format is JSON: reject values which cannot be serialized.
    json.dumps(recipe, allow_nan=False)
    return copy.deepcopy(recipe)


def load_recipe(path: str | Path) -> dict:
    return validate_recipe(json.loads(Path(path).read_text(encoding="utf-8")))
