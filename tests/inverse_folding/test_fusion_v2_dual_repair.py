"""Regression cover for the defects an adversarial review found in the FIRST wiring repair.

Each test below names a defect that a full green test suite coexisted with. That is the point: the
suite passed while three arms ran one objective, while every Dual ladder run crashed at its first
adopted donor, while the three Dual tables were built and dropped, and while the first real feedback
row could not be written at all. A test that only asserts constructibility passes in all four
states.
"""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

import scripts.rf_fusion_v2_artifacts as artifacts
from inverse_folding.reference_flow.fusion_v2_runtime import archive as arch

from .test_fusion_v2_dual_artifact_rows import _events, _real_decision
from .test_fusion_v2_dual_execution import _run, _runtime

pq = pytest.importorskip("pyarrow.parquet")


# -- B5: the feedback table must be writable at all ----------------------------------------------

def test_a_real_feedback_table_survives_the_bundle_writer(tmp_path):
    """The round trip that used to pass an EMPTY feedback table, so a declared-bool column holding
    a normalized float was never exercised: the first real write raised ArrowInvalid."""
    rows = artifacts.dual_feedback_evidence_rows(_events(), arm="joint")
    assert rows, "fixture: the decision must carry Dual evidence"
    artifacts.write_v2_bundle(
        tmp_path, manifest={"schema_version": "v2run-1"},
        tables={n: [] for n in artifacts.V2_TABLE_SCHEMAS},
        dual_tables={"dual_endpoint_evidence": [], "dual_terminal_summary": [],
                     "dual_feedback_evidence": rows})
    table = pq.read_table(pathlib.Path(tmp_path) / "dual_feedback_evidence.parquet")
    assert table.num_rows == len(rows)
    for column in ("reopen_new_hotspot_a", "reopen_worsening_b"):
        assert "bool" not in str(table.schema.field(column).type), column


# -- majors: the artifact must agree with the decision it reads off -------------------------------

def test_write_eligibility_is_read_against_the_margin_the_run_applied():
    decision = _real_decision()
    dual = decision.decision_evidence.dual
    rows = artifacts.dual_feedback_evidence_rows(_events(), arm="joint")
    scored = [r for r in rows if r["joint_contribution"] is not None]
    assert scored
    for row in scored:
        assert row["write_eligible"] == (row["joint_contribution"] > dual.write_epsilon), (
            "the table recomputed eligibility on 0.0 while the run filtered on the derived joint "
            "margin, so it contradicts the decision it claims to be reading off")


def test_the_applied_write_margin_is_recorded_at_all():
    dual = _real_decision().decision_evidence.dual
    assert dual.write_epsilon >= 0.0
    assert "write_epsilon" in dual.canonical_payload()


def test_selection_is_read_off_the_per_position_rows_not_a_missing_key():
    """`selected` used to be derived from an event key the shard never writes, so it was
    structurally always False and `selection_rank` always None."""
    rows = artifacts.dual_feedback_evidence_rows(_events(), arm="joint")
    assert any(row["selected"] for row in rows), "no position was ever marked selected"
    assert any(row["selection_rank"] is not None for row in rows)


def test_the_reopen_attribution_names_the_conjunct_that_ordered_the_row():
    """`winning_allele` used to report the first PRESENT conjunct, so a position ordered by
    residual burden was attributed to a new-hotspot value of exactly 0.0 and its tie-break."""
    from inverse_folding.reference_flow.fusion_v2 import dual_policy as dp
    from inverse_folding.reference_flow.fusion_v2.joint_objective import AlleleRole

    zero = dp.UnionConjunct(value=0.0, value_a=0.0, value_b=0.0, winning_allele=AlleleRole.A)
    real = dp.UnionConjunct(value=2.0, value_a=1.0, value_b=2.0, winning_allele=AlleleRole.B)
    row = dp.DualReopenEvidence(position=3, new_hotspot=zero, worsening=zero, residual_burden=real)
    assert row.ordering_conjunct == "residual_burden"
    assert row.winning_allele == "B", (
        "the row was ordered by role B's residual burden but the attribution named role A's "
        "tie-break on a zero")


def test_a_typed_stall_keeps_the_dual_evidence_the_cycle_paid_for():
    from .test_fusion_v2_dual_select import _authority
    from .test_fusion_v2_head_directed_policy import (
        SOURCE_SEQ, _calibration, _coords, _donor, _policy, _source,
    )
    from inverse_folding.reference_flow.fusion_v2 import policy as pol

    source = _source(SOURCE_SEQ)
    endpoint = _donor(source=source)
    # A tolerance above every contribution: the counterfactual batch RUNS (two Head batches are
    # paid for) and then nothing is eligible.
    stalled = _policy(dual=_authority(endpoint),
                      calibration=_calibration(epsilon=0.0, tolerance=0.0)).decide(
        source=source, endpoint=endpoint, coordinates=_coords())
    if isinstance(stalled, pol.PolicyRejection):
        assert stalled.decision_evidence.dual is not None, (
            "a typed stall discarded the per-allele evidence two Head batches produced; the "
            "artifact could not say WHY the joint law found nothing")
        assert stalled.decision_evidence.dual.contributions


# -- majors: role B's landscape is watched ---------------------------------------------------------

def test_role_b_hotspot_drift_is_measured_when_a_reference_is_declared():
    from .test_fusion_v2_dual_execution import _FakeHeadOracleB, _calibration, _Overlay
    from inverse_folding.reference_flow.fusion_v2.joint_objective import build_arm_objective
    from inverse_folding.reference_flow.fusion_v2_runtime.dual_runtime import DualRuntime

    probe = _runtime()
    _run(dual=probe)
    reference_b = next(iter(probe.score_b_by_endpoint.values()))

    watched = DualRuntime(
        overlay=_Overlay(), arm="joint", head_b=_FakeHeadOracleB(),
        objective=build_arm_objective(_calibration(), arm="joint"),
        safety_reference_score_b=reference_b, safety_reference_binding_id="ref:wt")
    _run(dual=watched)
    telemetry = [e.hotspot_b for e in watched.evidence_by_endpoint.values()]
    assert telemetry and all(t is not None for t in telemetry), (
        "role B's whole-landscape drift is the one safety statement Dual is allowed to make and "
        "it had no producer at all")
    rows = artifacts.dual_endpoint_evidence_rows(watched.evidence_by_endpoint, arm="joint")
    assert all(r["hotspot_b_positive_count"] is not None for r in rows)


def test_a_run_that_declares_no_role_b_reference_reports_absent_not_zero():
    runtime = _runtime()
    _run(dual=runtime)
    rows = artifacts.dual_endpoint_evidence_rows(runtime.evidence_by_endpoint, arm="joint")
    assert all(r["hotspot_b_max_increase"] is None for r in rows)


# -- majors: the ordering law survives a fork -----------------------------------------------------

def test_forking_an_archive_preserves_its_ordering_law():
    """Every matched arm forks the shared pre-feedback archive. A default fork silently reverted a
    joint-ordered archive to head_global_risk while the gate still compared J."""
    sentinel = lambda endpoint: (0.0, str(endpoint.endpoint_id))          # noqa: E731
    forked = arch.ExactArchive(rank_key=sentinel).fork_view()
    assert forked._rank_key is sentinel


# -- majors: the launch gate must price both Heads ------------------------------------------------

def test_the_budget_projection_doubles_the_head_calls_for_a_dual_run():
    from scripts.rf_fusion_v2_preflight import project_v2_budget
    from tests.inverse_folding.test_fusion_v2_config import _mapping
    from inverse_folding.reference_flow.fusion_v2.config import load_v2_config

    config = load_v2_config(_mapping())
    single = project_v2_budget(config, n_proteins=1, n_heads=1)
    dual = project_v2_budget(config, n_proteins=1, n_heads=2)
    assert dual.total_head_calls == 2 * single.total_head_calls, (
        "the launch gate certified a Dual run on a projection that omits role B entirely")


# -- the union screen, measured rather than assumed ------------------------------------------------

def test_a_position_only_role_b_improves_reaches_the_joint_batch_through_select():
    """The DUALF4 end-to-end assertion was `joint_set >= single_set`, which the union reducer
    guarantees whatever role B says -- and measured, the two sets were IDENTICAL, so "no A prefilter"
    was never exercised through `select()` at all. The reason is a property of that fixture: its
    donor improves under role A at EVERY masked position, so there is nothing left for role B to add.

    Here the donor carries 'R' at position 9, which role A rejects with ``no_improved_window``
    (measured), and role B's landscape improves the window that covers it. The union must therefore
    make position 9 legal, or an A-only prefilter is still deciding which positions get scored.
    """
    import dataclasses

    from inverse_folding.reference_flow.fusion.state import sequence_md5
    from inverse_folding.reference_flow.fusion_v2 import identity as ident
    from inverse_folding.reference_flow.fusion_v2 import policy as pol

    from . import _v2_fixtures as F
    from .test_fusion_v2_dual_select import EVAL_B, _authority, _ScriptedHeadB
    from .test_fusion_v2_head_directed_policy import (
        INCUMBENT_SEQ, MASKED, SOURCE_SEQ, WINDOW_K, _Result, _calibration, _coords, _donor,
        _policy, _score, _source,
    )

    #: 'R' at position 9 is measured to be non-improving under role A; A/C/D keep the other three
    #: masked positions legal so the two candidate sets differ in exactly one place.
    sequence = list(SOURCE_SEQ)
    for offset, position in enumerate(MASKED):
        sequence[position] = "ACDR"[offset]
    source = _source(SOURCE_SEQ)
    endpoint = _donor("".join(sequence), source=source)

    #: The window that covers position 9 under K=4 windows starting at every index.
    IMPROVING_WINDOW = 9 - WINDOW_K + 1

    def _flat_b(seq):
        """Role B: flat everywhere except the window covering position 9, where the DONOR is better.

        Role A's landscape is untouched, so any position this adds was added by role B alone.
        """
        z = [0.0] * (len(seq) - WINDOW_K + 1)
        if sequence_md5(seq) == sequence_md5(endpoint.sequence):
            z[IMPROVING_WINDOW] = -5.0
        windows = tuple(
            F.HeadWindow(start_0b=i, end_0b=i + WINDOW_K, k=WINDOW_K, z=value)
            for i, value in enumerate(z))
        return F.EndpointHeadScore(
            protein_id="5ZHV_B", sequence_md5=sequence_md5(seq), sequence_length=len(seq),
            allele=EVAL_B.allele, score_scale=EVAL_B.score_scale, windows=windows,
            residue_hotspot=(0.0,) * len(seq), global_risk=float(sum(z)))

    class _WindowedHeadB(_ScriptedHeadB):
        def score(self, requests):
            rows = []
            for request in requests:
                score = _flat_b(request.sequence)
                rows.append(_Result(
                    score=score,
                    binding=ident.HeadScoreBinding(
                        protein_id="5ZHV_B", sequence_md5=score.sequence_md5,
                        sequence_length=len(request.sequence),
                        window_grid_digest=ident.window_grid_digest(score.windows),
                        evaluator=EVAL_B),
                    sequence_md5=score.sequence_md5, global_risk=score.global_risk))
            return rows

    reference_b = _flat_b(INCUMBENT_SEQ)
    authority = dataclasses.replace(
        _authority(endpoint, head_b=_WindowedHeadB()),
        incumbent_score_b=reference_b, safety_reference_score_b=reference_b,
        donor_score_b_by_endpoint={str(endpoint.endpoint_id): _flat_b(endpoint.sequence)})
    authority = dataclasses.replace(authority, incumbent_joint_value=float(
        authority.objective.evaluate(
            raw_a=_score(INCUMBENT_SEQ).global_risk,
            raw_b=reference_b.global_risk).value))

    def screen(decision):
        rows = {r.position: r for r in decision.decision_evidence.write_candidates}
        return {p for p, r in rows.items() if r.legal}, rows

    single, single_rows = screen(_policy(calibration=_calibration(epsilon=0.0)).decide(
        source=source, endpoint=endpoint, coordinates=_coords()))
    assert single_rows[9].rejection_reason == "no_improved_window", (
        "fixture: role A must reject position 9, or this test cannot detect a prefilter")

    joint, _ = screen(_policy(dual=authority, calibration=_calibration(epsilon=0.0)).decide(
        source=source, endpoint=endpoint, coordinates=_coords()))
    assert 9 in joint - single, (
        "role B improved the window covering position 9 and role A did not, yet the position never "
        "entered the joint batch -- an A-only prefilter is still deciding what gets scored")


# -- the terminal summary must report what its column name says ------------------------------------

def test_last_improving_depth_is_the_last_depth_j_improved_not_the_depth_reached():
    """The column was filled with the ladder's `depth_reached`, which says how far the loop got and
    nothing about whether J ever moved: a run that reached depth 4 and improved only at depth 1
    published "4", so every per-depth improvement rate read off this table was the depth cap."""
    from scripts.rf_fusion_v2_cohort import _last_improving_depth

    class _E:
        def __init__(self, eid): self.endpoint_id = eid

    class _V:
        def __init__(self, value): self.risk = type("_R", (), {"value": value})()

    # depth 0 baseline -5.0; depth 1 improves to -7.0; depths 2 and 3 do not improve on it.
    by_depth = {0: [_E("a")], 1: [_E("b")], 2: [_E("c")], 3: [_E("d")]}
    evidence = {"a": _V(-5.0), "b": _V(-7.0), "c": _V(-6.0), "d": _V(-7.0)}
    assert _last_improving_depth(by_depth, evidence) == 1

    # a ladder that reached depth 3 and never improved reports NOTHING, not 3 and not 0.
    flat = {"a": _V(-5.0), "b": _V(-4.0), "c": _V(-4.0), "d": _V(-3.0)}
    assert _last_improving_depth(by_depth, flat) is None

    assert _last_improving_depth({}, {}) is None
