"""Deterministic two-axis ordering used by official RF products."""

from __future__ import annotations

import pytest

from inverse_folding.reference_flow.official_selection import two_axis_pareto_order


def test_two_axis_pareto_order_peels_layers_then_uses_global_rank_sum_and_tie_key():
    ordered = two_axis_pareto_order(
        first=(0.0, 1.0, 3.0, 2.0, 4.0),
        second=(3.0, 1.0, 0.0, 2.0, 4.0),
        tie_keys=("a", "b", "c", "d", "e"),
    )

    assert [row.index for row in ordered] == [1, 0, 2, 3, 4]
    assert [row.pareto_layer for row in ordered] == [1, 1, 1, 2, 3]
    assert ordered[0].rank_sum < ordered[1].rank_sum


@pytest.mark.parametrize(
    "first,second,tie_keys,match",
    [
        ((1.0,), (), ("a",), "same length"),
        ((float("nan"),), (1.0,), ("a",), "finite"),
        ((1.0,), (1.0,), (), "same length"),
    ],
)
def test_two_axis_pareto_order_fails_closed_on_invalid_inputs(
    first, second, tie_keys, match,
):
    with pytest.raises(ValueError, match=match):
        two_axis_pareto_order(first=first, second=second, tie_keys=tie_keys)
