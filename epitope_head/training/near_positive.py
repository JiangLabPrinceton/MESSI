"""Span relation classification + negative-penalty schedules (HIMP1).

PLAN_EPI_IMP.md §5 HIMP1: re-classify candidate negatives against observed
positives, then map (relation, end_point_gap) to a penalty weight via a
configurable schedule. v0 ships three schedules: ``ignore``, ``linear_clamp``,
``sigmoid``. ``exponential`` and ``thresholded_smooth`` are reserved API names
that raise ``NotImplementedError`` if requested.

Coordinate convention: spans are 0-based half-open ``[start, end)`` and ``pep_len
= end - start`` is the peptide length used by the scorer.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Callable, Iterable


# ── Span relation enum ──────────────────────────────────────────────────────

class SpanRelation(str, Enum):
    """Relation of a candidate span to the set of observed positives."""

    EXACT_POSITIVE = "exact_positive"
    OVERLAP_NONEXACT = "overlap_nonexact"
    ADJACENT_OR_NEAR = "adjacent_or_near"
    FAR_DECOY = "far_decoy"
    COUNTERFACTUAL_DISRUPTED = "counterfactual_disrupted"


# ── Classifier ──────────────────────────────────────────────────────────────

def classify_span_relation(
    candidate: tuple[int, int],
    positives: Iterable[tuple[int, int]],
    near_gap_max: int,
) -> tuple[SpanRelation, int]:
    """Classify a candidate span against observed positives.

    Args:
        candidate: ``(start_0b, end_0b)`` of the candidate span.
        positives: iterable of ``(start_0b, end_0b)`` for observed positives.
        near_gap_max: end-point distance threshold separating
            ``adjacent_or_near`` from ``far_decoy``.

    Returns:
        ``(relation, end_point_gap)`` where ``end_point_gap`` is the minimum
        over positives of ``max(|cand_start - pos_start|, |cand_end - pos_end|)``
        for non-overlapping cases. For ``exact_positive`` and
        ``overlap_nonexact``, gap is reported as 0 (not used downstream).
    """
    pos_list = [(int(p[0]), int(p[1])) for p in positives]
    cand = (int(candidate[0]), int(candidate[1]))
    cs, ce = cand

    pos_set = {(p[0], p[1]) for p in pos_list}
    if cand in pos_set:
        return SpanRelation.EXACT_POSITIVE, 0

    has_overlap = False
    for ps, pe in pos_list:
        if max(cs, ps) < min(ce, pe):
            has_overlap = True
            break
    if has_overlap:
        return SpanRelation.OVERLAP_NONEXACT, 0

    min_gap = None
    for ps, pe in pos_list:
        gap = max(abs(cs - ps), abs(ce - pe))
        if min_gap is None or gap < min_gap:
            min_gap = gap

    # No positives: classify as far_decoy with gap=0 (caller decides).
    if min_gap is None:
        return SpanRelation.FAR_DECOY, 0

    if min_gap <= near_gap_max:
        return SpanRelation.ADJACENT_OR_NEAR, min_gap
    return SpanRelation.FAR_DECOY, min_gap


# ── Schedule registry ───────────────────────────────────────────────────────

def _schedule_ignore(gap: int, near_gap_max: int, **_kwargs) -> float:
    """Ignore schedule: zero penalty for adjacent_or_near and overlap_nonexact."""
    return 0.0


def _schedule_linear_clamp(
    gap: int, near_gap_max: int, *,
    w_low: float = 0.0, w_high: float = 1.0,
    **_kwargs,
) -> float:
    """Piecewise linear from ``w_low`` at gap=0 to ``w_high`` at gap=``near_gap_max``.

    Values outside ``[0, near_gap_max]`` are clamped to the corresponding
    endpoint. ``near_gap_max == 0`` short-circuits to ``w_high``.
    """
    if near_gap_max <= 0:
        return float(w_high)
    t = gap / near_gap_max
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return float(w_low + t * (w_high - w_low))


def _schedule_sigmoid(
    gap: int, near_gap_max: int, *,
    center: float | None = None, slope: float = 1.0,
    **_kwargs,
) -> float:
    """Logistic transition centered at ``center`` (default ``near_gap_max/2``)."""
    c = float(near_gap_max) / 2.0 if center is None else float(center)
    return 1.0 / (1.0 + math.exp(-float(slope) * (float(gap) - c)))


SCHEDULE_REGISTRY: dict[str, Callable[..., float]] = {
    "ignore": _schedule_ignore,
    "linear_clamp": _schedule_linear_clamp,
    "sigmoid": _schedule_sigmoid,
}

#: Schedule names reserved by the API but not implemented in v0. Calling
#: ``get_schedule`` with any of these raises ``NotImplementedError`` so misuse
#: surfaces immediately rather than silently degrading.
RESERVED_SCHEDULES: frozenset[str] = frozenset({"exponential", "thresholded_smooth"})


def get_schedule(name: str) -> Callable[..., float]:
    """Return the schedule function for ``name``.

    Raises ``NotImplementedError`` for reserved names and ``ValueError`` for
    unknown names — never silently falls back to ``ignore``.
    """
    if name in RESERVED_SCHEDULES:
        raise NotImplementedError(
            f"Schedule {name!r} is reserved but not implemented in v0; "
            f"available: {sorted(SCHEDULE_REGISTRY.keys())}"
        )
    if name not in SCHEDULE_REGISTRY:
        raise ValueError(
            f"Unknown schedule {name!r}; available: "
            f"{sorted(SCHEDULE_REGISTRY.keys())} (reserved: {sorted(RESERVED_SCHEDULES)})"
        )
    return SCHEDULE_REGISTRY[name]


# ── Weight computation ──────────────────────────────────────────────────────

def compute_negative_weight(
    relation: SpanRelation,
    end_point_gap: int,
    near_gap_max: int,
    schedule_name: str,
    schedule_params: dict | None = None,
) -> float:
    """Map ``(relation, gap)`` to a penalty weight in ``[0, 1]``.

    Fixed rules (independent of schedule choice):
      - ``exact_positive``: 0.0 (defensive — should never be sampled as neg).
      - ``counterfactual_disrupted``: 1.0 (mutation-derived pseudo-negative).
      - ``far_decoy``: 1.0.

    Schedule-driven rules:
      - ``adjacent_or_near``: ``schedule(end_point_gap, near_gap_max, **params)``
      - ``overlap_nonexact``: ``schedule(0, near_gap_max, **params)`` — mirrors
        ``adjacent_or_near`` at gap=0 by convention.
    """
    if relation == SpanRelation.EXACT_POSITIVE:
        return 0.0
    if relation == SpanRelation.COUNTERFACTUAL_DISRUPTED:
        return 1.0
    if relation == SpanRelation.FAR_DECOY:
        return 1.0

    schedule_fn = get_schedule(schedule_name)
    params = schedule_params or {}

    if relation == SpanRelation.ADJACENT_OR_NEAR:
        return float(schedule_fn(int(end_point_gap), int(near_gap_max), **params))
    if relation == SpanRelation.OVERLAP_NONEXACT:
        return float(schedule_fn(0, int(near_gap_max), **params))

    raise ValueError(f"Unknown relation: {relation!r}")
