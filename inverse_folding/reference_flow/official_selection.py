"""Pure deterministic selection laws for paper-facing RF products.

The search implementations deliberately retain richer state.  This module owns only the
post-search ordering used to turn a two-axis candidate pool into an up-to-N official panel:
peel exact Pareto layers, order within a layer by the sum of the two full-pool average ranks,
then use a caller-supplied content key.  It has no I/O and does not participate in search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TwoAxisParetoOrder:
    """One candidate's immutable position in the official two-axis order."""

    index: int
    pareto_layer: int
    first_rank: float
    second_rank: float
    rank_sum: float
    selection_rank: int


def _average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        value = values[order[cursor]]
        while end < len(order) and values[order[end]] == value:
            end += 1
        # Positions are one-based; ties receive their average rank.
        rank = ((cursor + 1) + end) / 2.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return tuple(ranks)


def two_axis_pareto_order(
    *,
    first: Sequence[float],
    second: Sequence[float],
    tie_keys: Sequence[str],
) -> tuple[TwoAxisParetoOrder, ...]:
    """Return the complete frozen Pareto-layer/rank-sum ordering.

    Both axes are minimized.  Ranks are computed once over the complete input pool, not
    recomputed after a layer is removed.  This is the law used by the frozen V2 strict-output
    panel and is intentionally independent of any requested final candidate count.
    """

    if len(first) != len(second) or len(first) != len(tie_keys):
        raise ValueError("first, second, and tie_keys must have the same length")
    first_values = tuple(float(value) for value in first)
    second_values = tuple(float(value) for value in second)
    if any(not math.isfinite(value) for value in (*first_values, *second_values)):
        raise ValueError("two-axis official selection requires finite objective values")
    if not first_values:
        return ()

    first_ranks = _average_ranks(first_values)
    second_ranks = _average_ranks(second_values)
    remaining = set(range(len(first_values)))
    ordered: list[TwoAxisParetoOrder] = []
    layer = 1
    while remaining:
        front = []
        for index in sorted(remaining):
            dominated = any(
                other != index
                and first_values[other] <= first_values[index]
                and second_values[other] <= second_values[index]
                and (
                    first_values[other] < first_values[index]
                    or second_values[other] < second_values[index]
                )
                for other in remaining
            )
            if not dominated:
                front.append(index)
        if not front:  # pragma: no cover - finite partial order always has a minimal point
            raise RuntimeError("two-axis Pareto peeling produced an empty layer")
        front.sort(
            key=lambda index: (
                first_ranks[index] + second_ranks[index],
                str(tie_keys[index]),
                index,
            )
        )
        for index in front:
            ordered.append(TwoAxisParetoOrder(
                index=index,
                pareto_layer=layer,
                first_rank=first_ranks[index],
                second_rank=second_ranks[index],
                rank_sum=first_ranks[index] + second_ranks[index],
                selection_rank=len(ordered) + 1,
            ))
        remaining.difference_update(front)
        layer += 1
    return tuple(ordered)
