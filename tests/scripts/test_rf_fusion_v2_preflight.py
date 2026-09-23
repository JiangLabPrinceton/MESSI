"""V2F7: the strict V2 preflight and its budget projection (PLAN §3.4, §5.1).

The projection exists to refuse a run BEFORE its budget is spent, so the only property that matters
is that it describes the run that will actually happen.  A projection that agrees with itself but
not with the execution graph is a second cost model, and the launch gate would then be bounding a
run nobody performs -- permissively, because the graph the ladder really executes is the larger of
the two.

The reconciliation against a realized ladder lives in ``test_fusion_v2_ladder.py``, next to the
forward-pass counter that establishes what "realized" means.  What is checked here is the
arithmetic itself, component by component, so a discrepancy names which term is wrong.
"""

from __future__ import annotations

import pytest

from scripts.rf_fusion_v2_preflight import (
    V2PreflightError,
    assert_launch_feasible,
    project_v2_budget,
)

#: The same schedule ``test_fusion_v2_ladder`` runs, so the two suites describe one experiment.
N_STEPS = 100
C0, R1, C1, R2, C2 = 50, 40, 62, 52, 72
K0, K1 = 3, 2


def _config(**over):
    from inverse_folding.reference_flow.fusion_v2.config import load_v2_config
    from tests.inverse_folding.test_fusion_v2_config import _mapping

    payload = _mapping(**{
        "substrate.n_steps": N_STEPS,
        "schedule.depth_cap": 2,
        "schedule.min_lookahead_tail_steps": 10,
        "schedule.points": [
            {"depth": 0, "r_step": R1, "c_source_step": C0, "c_next_step": C1,
             "n_lookaheads": K0, "band_key": "len4_8"},
            {"depth": 1, "r_step": R2, "c_source_step": C1, "c_next_step": C2,
             "n_lookaheads": K1, "band_key": "len4_8"},
        ],
        "caps.max_logical_dfe": 10 ** 6,
        "caps.max_head_calls": 10 ** 6,
        "caps.max_definitive_refolds": 10 ** 6,
        **over,
    })
    return load_v2_config(payload)


# --------------------------------------------------------------------------------------------
# the projection describes the execution graph, term by term
# --------------------------------------------------------------------------------------------


def test_the_root_prefix_is_projected_once_for_the_whole_ladder():
    """PLAN §3.4: the nominal source prefix is charged once; retries are a separate reserve.

    Only depth 0 captures a root; every deeper rung inherits the previous rung's propagated state,
    so a projection that charged ``c_d`` per depth would bound a run with a prefix per rung that no
    forward pass corresponds to.
    """
    projection = project_v2_budget(_config(), n_proteins=1)
    assert projection.root_capture_logical_dfe == C0


@pytest.mark.parametrize("max_retries", [0, 2, 7])
def test_preflight_reserves_the_configured_root_recapture_ceiling(max_retries):
    projection = project_v2_budget(
        _config(**{"caps.max_retries": max_retries}), n_proteins=1)
    assert projection.root_capture_retry_reserve_logical_dfe == max_retries * C0
    assert projection.max_total_logical_dfe == (
        projection.total_logical_dfe + projection.root_capture_retry_reserve_logical_dfe)


def test_only_depth_zero_forks_a_source_pool():
    """Depth ``d>0``'s source pool IS depth ``d-1``'s descendant pool.

    Charging it again would double the dominant term of the whole cost model.
    """
    projection = project_v2_budget(_config(), n_proteins=1)
    assert projection.per_protein_screen_dfe == K0 * (N_STEPS - C0)


def test_every_depth_charges_the_descendant_pool_it_generates():
    """The defect this pins down: the DEEPEST rung's descendants were screened and never charged.

    The old projection reached a rung's descendant pool only through the NEXT rung's inherited
    screening -- and the last rung has no next, so its whole pool fell out of the budget.
    """
    projection = project_v2_budget(_config(), n_proteins=1)
    # depth 0 forks K1 descendants over the tail after c_1; depth 1 forks K1 over the tail after c_2
    assert projection.per_protein_descendant_screen_dfe == (
        K1 * (N_STEPS - C1) + K1 * (N_STEPS - C2))


def test_each_segment_costs_exactly_its_own_span():
    projection = project_v2_budget(_config(), n_proteins=1)
    assert projection.per_protein_segment_dfe == (C1 - R1) + (C2 - R2)


def test_the_per_protein_total_is_the_sum_of_its_own_components():
    projection = project_v2_budget(_config(), n_proteins=1)
    assert projection.per_protein_logical_dfe == (
        projection.root_capture_logical_dfe
        + projection.per_protein_screen_dfe
        + projection.per_protein_segment_dfe
        + projection.per_protein_descendant_screen_dfe
    )
    assert projection.total_logical_dfe == projection.per_protein_logical_dfe


def test_the_head_and_refold_projections_count_every_pool_that_is_scored():
    """One Head request and one structure attempt per generated endpoint.

    Counting only the declared per-depth breadth misses every descendant pool, which is most of the
    refolds -- and refolds are about 94% of the marginal cost of a cycle at the measured unit costs,
    so a refold cap projected from the smaller number cannot bound anything.
    """
    projection = project_v2_budget(_config(), n_proteins=1)
    expected = K0 + K1 + K1
    assert projection.total_head_calls == expected
    assert projection.total_definitive_refolds == expected


def test_the_projection_scales_with_the_cohort():
    one = project_v2_budget(_config(), n_proteins=1)
    seven = project_v2_budget(_config(), n_proteins=7)
    assert seven.total_logical_dfe == 7 * one.per_protein_logical_dfe
    assert seven.total_head_calls == 7 * (K0 + K1 + K1)


def test_execution_replicates_scale_every_conservative_launch_component():
    one = project_v2_budget(_config(), n_proteins=1)
    paired = project_v2_budget(_config(), n_proteins=1, execution_replicates=112)
    assert paired.execution_replicates == 112
    assert paired.per_protein_logical_dfe == 112 * one.per_protein_logical_dfe
    assert paired.total_head_calls == 112 * one.total_head_calls
    assert paired.total_definitive_refolds == 112 * one.total_definitive_refolds


# --------------------------------------------------------------------------------------------
# the gate fails closed
# --------------------------------------------------------------------------------------------


def test_a_schedule_that_breaches_its_declared_dfe_cap_is_refused():
    tight = _config(**{"caps.max_logical_dfe": 10})
    projection = project_v2_budget(tight, n_proteins=1)
    assert "max_logical_dfe" in projection.breached_caps
    with pytest.raises(V2PreflightError):
        assert_launch_feasible(projection)


def test_a_refold_cap_is_checked_against_every_pool_not_only_the_declared_breadth():
    """A cap that only the under-counted projection satisfies is the case worth naming."""
    borderline = _config(**{"caps.max_definitive_refolds": K0 + K1})
    projection = project_v2_budget(borderline, n_proteins=1)
    assert "max_definitive_refolds" in projection.breached_caps


def test_a_feasible_schedule_passes():
    """The gate must not be satisfiable by refusing everything."""
    assert assert_launch_feasible(project_v2_budget(_config(), n_proteins=1)) is None


def test_a_negative_cohort_is_a_typed_refusal():
    with pytest.raises(V2PreflightError):
        project_v2_budget(_config(), n_proteins=-1)
