"""Density-stratified subsampling for Tier 2 v2 selection (PLAN_DATA_SEL §12).

Pure functions that bin a candidate pool along a continuous density column
(e.g. ``coverage_fraction``) and draw a target-size subsample whose per-bin
counts follow a desired marginal:

  - ``sample_uniform_bins``  — equal counts per bin (used for the S1 coarse
    length-15 downsample to ≈5000).
  - ``sample_gaussian_bins`` — unimodal Gaussian over the density axis,
    centered at the median, modest peak, with a per-bin floor (used for the
    S3 final downsample to ≈3000).

Both respect per-bin availability (a thin bin is never over-drawn) and a
``min_per_bin`` floor, and redistribute any deficit/surplus so the realized
total matches ``n_target`` when the pool has the capacity. Selection within a
bin is deterministic given ``seed``. Each call also returns a realized
histogram so capping is logged, never silent.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _assign_bins(values: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Equal-width binning over [min, max]. Returns (bin_index, edges).

    The max value falls in the last bin; degenerate (all-equal) input collapses
    to a single populated bin.
    """
    lo = float(np.min(values))
    hi = float(np.max(values))
    edges = np.linspace(lo, hi, n_bins + 1)
    if hi <= lo:
        return np.zeros(len(values), dtype=int), edges
    idx = np.digitize(values, edges[1:-1], right=False)
    return np.clip(idx, 0, n_bins - 1).astype(int), edges


def _gaussian_weights(
    centers: np.ndarray,
    mu: float,
    sigma: Optional[float],
    peak_to_tail: float,
) -> np.ndarray:
    """Gaussian bin weights. If sigma is None, derive it so the farthest bin
    center sits at weight 1/peak_to_tail relative to the peak."""
    if sigma is None:
        d = max(mu - centers.min(), centers.max() - mu)
        if d <= 0 or peak_to_tail <= 1.0:
            return np.ones_like(centers)
        sigma = d / np.sqrt(2.0 * np.log(peak_to_tail))
    if sigma <= 0:
        return np.ones_like(centers)
    return np.exp(-0.5 * ((centers - mu) / sigma) ** 2)


def _allocate(
    weights: np.ndarray,
    available: np.ndarray,
    n_target: int,
    min_per_bin: int,
) -> np.ndarray:
    """Greedy water-filling: allocate counts proportional to ``weights``,
    capped by ``available``, floored at ``min(min_per_bin, available)``, with
    the running total driven to ``n_target`` when capacity allows."""
    available = available.astype(int)
    w = weights.astype(float)
    floor = np.minimum(min_per_bin, available)
    alloc = floor.copy()
    total = int(alloc.sum())

    if total > n_target:
        # Floors overshoot the target — trim from bins most over their fair
        # share (largest alloc/weight), never below floor.
        while total > n_target:
            mask = alloc > floor
            if not mask.any():
                break
            score = np.where(mask, alloc / np.maximum(w, 1e-12), -np.inf)
            i = int(np.argmax(score))
            alloc[i] -= 1
            total -= 1
        return alloc

    # Fill toward target, each unit going to the bin most under its fair share
    # (smallest alloc/weight) that still has spare capacity.
    while total < n_target:
        spare = available - alloc
        mask = spare > 0
        if not mask.any():
            break
        score = np.where(mask, alloc / np.maximum(w, 1e-12), np.inf)
        i = int(np.argmin(score))
        alloc[i] += 1
        total += 1
    return alloc


def _draw(
    df: pd.DataFrame,
    value_col: str,
    n_bins: int,
    weights_fn,
    n_target: int,
    min_per_bin: int,
    seed: int,
) -> Tuple[List, pd.DataFrame]:
    values = df[value_col].to_numpy(dtype=float)
    bin_idx, edges = _assign_bins(values, n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    available = np.array([int((bin_idx == b).sum()) for b in range(n_bins)])
    weights = weights_fn(centers)
    weights = np.where(available > 0, weights, 0.0)  # no weight on empty bins
    alloc = _allocate(weights, available, n_target, min_per_bin)

    rng = np.random.default_rng(seed)
    selected: List = []
    rows = []
    index = df.index.to_numpy()
    for b in range(n_bins):
        members = index[bin_idx == b]
        take = int(min(alloc[b], len(members)))
        if take >= len(members):
            chosen = members
        else:
            chosen = rng.choice(members, size=take, replace=False)
        selected.extend(sorted(chosen.tolist()))
        rows.append({
            "bin": b,
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "bin_center": float(centers[b]),
            "n_available": int(len(members)),
            "n_selected": int(take),
        })
    return selected, pd.DataFrame(rows)


def sample_uniform_bins(
    df: pd.DataFrame,
    value_col: str,
    n_target: int,
    n_bins: int = 20,
    min_per_bin: int = 0,
    seed: int = 42,
) -> Tuple[List, pd.DataFrame]:
    """Equal-count-per-bin subsample over ``value_col``.

    Returns (selected_index_labels, histogram_df).
    """
    return _draw(
        df, value_col, n_bins,
        weights_fn=lambda c: np.ones_like(c),
        n_target=n_target, min_per_bin=min_per_bin, seed=seed,
    )


def sample_gaussian_bins(
    df: pd.DataFrame,
    value_col: str,
    n_target: int,
    n_bins: int = 20,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    peak_to_tail: float = 3.0,
    min_per_bin: int = 40,
    seed: int = 42,
) -> Tuple[List, pd.DataFrame]:
    """Unimodal-Gaussian subsample over ``value_col``.

    ``mu`` defaults to the median of ``value_col``; ``sigma`` (if None) is
    derived from ``peak_to_tail`` (central:edge bin-weight ratio). A
    ``min_per_bin`` floor keeps every bin populated.

    Returns (selected_index_labels, histogram_df).
    """
    mu_val = float(df[value_col].median()) if mu is None else float(mu)
    return _draw(
        df, value_col, n_bins,
        weights_fn=lambda c: _gaussian_weights(c, mu_val, sigma, peak_to_tail),
        n_target=n_target, min_per_bin=min_per_bin, seed=seed,
    )
