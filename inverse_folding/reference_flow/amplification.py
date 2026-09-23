"""Amplification functions ``g(h)`` for Phase C."""

from __future__ import annotations

import warnings
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def normalize_h_values(
    h_raw: np.ndarray | list[float],
    stats: dict[str, Any],
) -> np.ndarray:
    """Normalize ``h_raw`` using corpus-wide mean and std from B3."""
    values = np.asarray(h_raw, dtype=np.float32)
    mean = float(stats["corpus_h_raw_mean"])
    std = float(stats["corpus_h_raw_std"])
    if std <= 0.0:
        raise ValueError("corpus_h_raw_std must be positive")
    return ((values - mean) / std).astype(np.float32, copy=False)


def shuffle_h_values(
    h_values: np.ndarray | list[float],
    *,
    seed: int,
) -> np.ndarray:
    """Deterministically permute a per-protein h vector."""
    values = np.asarray(h_values, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(values))
    return values[order].astype(np.float32, copy=False)


def amplification_factor(
    h_values: np.ndarray | list[float],
    config: Any,
) -> np.ndarray:
    """Materialize ``g(h)`` from an amplification config mapping/dataclass."""
    values = np.asarray(h_values, dtype=np.float32)
    cfg = _to_mapping(config)
    form = str(cfg["form"])
    mu = float(cfg.get("mu", 0.0))
    g_max_cap = float(cfg.get("g_max_cap", 20.0))

    if form == "constant_one":
        g = np.ones_like(values, dtype=np.float32)
    elif form == "linear_clamp":
        c = float(cfg["c"])
        g = 1.0 + c * np.maximum(0.0, values - mu)
    elif form == "sigmoid":
        c = float(cfg["c"])
        kappa = float(cfg["kappa"])
        logits = np.clip(kappa * (values - mu), -80.0, 80.0)
        g = 1.0 + c / (1.0 + np.exp(-logits))
    elif form == "power":
        c = float(cfg["c"])
        p = float(cfg["p"])
        g = 1.0 + c * np.power(np.maximum(0.0, values - mu), p)
    else:
        raise ValueError(f"unsupported amplification form: {form}")

    g = np.maximum(g.astype(np.float32, copy=False), 1.0)
    if np.any(g > g_max_cap):
        warnings.warn(
            f"amplification reached g_max_cap={g_max_cap}; clipping tail values",
            stacklevel=2,
        )
        g = np.minimum(g, g_max_cap)
    return g.astype(np.float32, copy=False)


def _to_mapping(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return config
    if is_dataclass(config):
        return asdict(config)
    raise TypeError("config must be a dict or dataclass")
