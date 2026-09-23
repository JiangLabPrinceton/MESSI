"""Residue-level supervision helpers (HIMP2).

Builds binary residue coverage labels, ambiguous masks, far-background masks,
and central-region masks for a single chunk; enumerates residue-supervision
windows for ``window_mode=all``; and aggregates window-level logits into
per-residue scores via ``max`` or ``log_mean_exp``.

Coordinate convention: spans are 0-based half-open ``[start, end)`` and
positions are local to the chunk (``0..chunk_len``).
"""

from __future__ import annotations

import math

import numpy as np
import torch


# ── Residue label / mask construction ───────────────────────────────────────

def build_residue_labels(
    positives: list[dict],
    chunk_len: int,
    central_start: int = 0,
    central_end: int | None = None,
    near_positive_cfg: dict | None = None,
) -> dict:
    """Build per-residue masks for one chunk.

    Returned dict (all arrays of length ``chunk_len``):
      - ``label`` (np.uint8): 1 if covered by any positive span, else 0.
      - ``ambiguous_mask`` (np.bool_): residues that are not covered but are
        within ``near_gap_max`` of some positive interval — only populated
        when ``near_positive_cfg["enabled"]`` and
        ``near_positive_cfg["apply_to_residue_labels"]`` are both True.
      - ``far_bg_mask`` (np.bool_): residues that are neither covered nor
        ambiguous (i.e. eligible as residue ranking negatives).
      - ``central_mask`` (np.bool_): True on residues inside the chunk's
        trusted central region ``[central_start, central_end)``. The trainer
        derives these per-chunk so that protein N-/C-termini (first /
        last chunk seams) are not silently dropped from supervision.

    Args:
        positives: list of ``{"start_0b", "end_0b", "pep_len"}`` dicts in
            chunk-local coordinates.
        chunk_len: number of residues in this chunk.
        central_start: chunk-local start of the trusted central region
            (inclusive, 0-based).
        central_end: chunk-local end of the trusted central region
            (exclusive). ``None`` defaults to ``chunk_len`` (full chunk).
        near_positive_cfg: optional dict with keys ``enabled``,
            ``apply_to_residue_labels``, ``near_gap_max``. When both
            enable-flags are True, ambiguous residues are computed.
    """
    L = int(chunk_len)
    label = np.zeros(L, dtype=np.uint8)
    for p in positives:
        s, e = int(p["start_0b"]), int(p["end_0b"])
        s = max(0, s)
        e = min(L, e)
        if s < e:
            label[s:e] = 1

    ambiguous_mask = np.zeros(L, dtype=bool)
    apply_residue = (
        near_positive_cfg is not None
        and bool(near_positive_cfg.get("enabled", False))
        and bool(near_positive_cfg.get("apply_to_residue_labels", False))
    )
    if apply_residue and positives:
        near_gap_max = int(near_positive_cfg["near_gap_max"])
        # Ambiguous = NOT covered by positive AND within near_gap_max of
        # some positive interval (residue distance to interval).
        for p in positives:
            s, e = int(p["start_0b"]), int(p["end_0b"])
            # Left side: [max(0, s - near_gap_max), s)
            left_start = max(0, s - near_gap_max)
            left_end = max(0, s)
            if left_start < left_end:
                ambiguous_mask[left_start:left_end] = True
            # Right side: [e, min(L, e + near_gap_max))
            right_start = min(L, e)
            right_end = min(L, e + near_gap_max)
            if right_start < right_end:
                ambiguous_mask[right_start:right_end] = True
        # Subtract covered residues — covered residues are NOT ambiguous.
        ambiguous_mask &= label.astype(bool) == False  # noqa: E712

    far_bg_mask = (label == 0) & (~ambiguous_mask)

    central_mask = np.zeros(L, dtype=bool)
    cs = max(0, int(central_start))
    ce = L if central_end is None else max(0, min(L, int(central_end)))
    if cs < ce:
        central_mask[cs:ce] = True

    return {
        "label": label,
        "ambiguous_mask": ambiguous_mask,
        "far_bg_mask": far_bg_mask,
        "central_mask": central_mask,
    }


# ── Window enumeration (window_mode=all) ────────────────────────────────────

def enumerate_all_residue_windows(
    central_start: int,
    central_end: int,
    min_k: int,
    max_k: int,
    max_windows: int,
) -> list[tuple[int, int]]:
    """Enumerate every ``(start, end)`` window with ``end - start ∈ [min_k, max_k]``
    that fits inside ``[central_start, central_end)``.

    Raises ``ValueError`` if the enumeration would exceed ``max_windows``;
    ``window_mode=all`` must fail loud rather than silently truncate.
    """
    if max_k < min_k:
        raise ValueError(f"max_k ({max_k}) must be >= min_k ({min_k})")
    if central_end <= central_start:
        return []
    region = central_end - central_start
    # Quick total estimate: sum_{k=min_k..max_k} max(0, region - k + 1)
    total = 0
    per_k = []
    for k in range(min_k, max_k + 1):
        n = max(0, region - k + 1)
        per_k.append((k, n))
        total += n
    if total > max_windows:
        raise ValueError(
            f"residue.window_mode=all would enumerate {total} windows for "
            f"region={region} k∈[{min_k},{max_k}], exceeds "
            f"max_windows_per_chunk={max_windows}. Reduce region (chunking) or "
            f"increase the cap explicitly."
        )
    windows: list[tuple[int, int]] = []
    for k, n in per_k:
        for offset in range(n):
            s = central_start + offset
            windows.append((s, s + k))
    return windows


# ── Aggregation: window logits → residue scores ─────────────────────────────

VALID_AGGREGATIONS = frozenset({"max", "log_mean_exp"})
RESERVED_AGGREGATIONS = frozenset({"topk_mean"})


def aggregate_window_logits_to_residues(
    window_logits: torch.Tensor,
    windows: list[tuple[int, int]],
    chunk_len: int,
    mode: str = "log_mean_exp",
    params: dict | None = None,
) -> torch.Tensor:
    """Aggregate ``window_logits`` (one per window) into residue scores.

    Returns a ``[chunk_len]`` tensor; residues outside every window's coverage
    are filled with ``-inf`` (callers must mask before computing losses).

    Aggregation modes (v0):
      - ``max``: element-wise max over windows covering each residue. Gradient
        flows through the winning window only.
      - ``log_mean_exp``: ``(1/beta) * (logsumexp(beta * logits) - log(n))`` over
        windows covering each residue. ``beta`` defaults to 1.0 and is passed
        via ``params``.

    Reserved (raise ``NotImplementedError``):
      - ``topk_mean`` (HIMP3-future ablation).
    """
    if mode in RESERVED_AGGREGATIONS:
        raise NotImplementedError(
            f"Aggregation {mode!r} is reserved but not implemented in v0; "
            f"available: {sorted(VALID_AGGREGATIONS)}"
        )
    if mode not in VALID_AGGREGATIONS:
        raise ValueError(
            f"Unknown residue aggregation {mode!r}; available: "
            f"{sorted(VALID_AGGREGATIONS)} (reserved: {sorted(RESERVED_AGGREGATIONS)})"
        )

    L = int(chunk_len)
    W = int(window_logits.shape[0])
    device = window_logits.device

    if W == 0 or L == 0:
        return torch.full((L,), float("-inf"), device=device, dtype=window_logits.dtype)

    # Build coverage matrix M[W, L]: True if window w covers residue i.
    starts = torch.tensor([s for s, _ in windows], device=device)
    ends = torch.tensor([e for _, e in windows], device=device)
    res_idx = torch.arange(L, device=device).unsqueeze(0)         # [1, L]
    M = (res_idx >= starts.unsqueeze(1)) & (res_idx < ends.unsqueeze(1))  # [W, L]

    n_covered = M.sum(dim=0)  # [L]
    uncovered = n_covered == 0

    if mode == "max":
        # Mask uncovered window slots with -inf, then max-reduce.
        masked = window_logits.unsqueeze(1).expand(W, L).masked_fill(~M, float("-inf"))
        scores, _ = masked.max(dim=0)  # [L]
        return scores

    # log_mean_exp
    p = params or {}
    beta = float(p.get("beta", 1.0))
    if beta <= 0:
        raise ValueError(f"log_mean_exp beta must be > 0, got {beta}")

    # logsumexp over windows covering each residue, then subtract log(n_covered).
    # Use mask via -inf substitution so logsumexp ignores uncovered slots.
    scaled = (beta * window_logits).unsqueeze(1).expand(W, L)
    masked = scaled.masked_fill(~M, float("-inf"))
    lse = torch.logsumexp(masked, dim=0)  # [L]
    log_n = torch.log(n_covered.clamp(min=1).float())
    scores = (lse - log_n) / beta
    scores = torch.where(uncovered, torch.full_like(scores, float("-inf")), scores)
    return scores
