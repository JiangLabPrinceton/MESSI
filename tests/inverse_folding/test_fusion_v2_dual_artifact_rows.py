"""DUALF5: the Dual tables are PRODUCED, and they carry the per-allele numbers.

The schemas existed and nothing built a row for them.  Worse, the two places where the joint law's
per-allele detail lives -- the leave-one-out A/B contrasts and the union reducer's winner -- were
computed on every cycle and then collapsed to a single scalar before anything could record them.  A
joint run's feedback table would then have been indistinguishable from a single-Head run's.

These tests build the rows from a REAL cycle rather than from hand-made evidence, because the claim
under test is that the production path produces them.
"""

from __future__ import annotations

import pathlib

import pytest

import scripts.rf_fusion_v2_artifacts as artifacts

from .test_fusion_v2_dual_execution import _run, _runtime

pq = pytest.importorskip("pyarrow.parquet")


def _cycle():
    runtime = _runtime()
    outcome = _run(dual=runtime)
    return runtime, outcome


# -- endpoint evidence --------------------------------------------------------------------------

def test_one_endpoint_row_per_jointly_scored_endpoint():
    runtime, _ = _cycle()
    rows = artifacts.dual_endpoint_evidence_rows(runtime.evidence_by_endpoint, arm="joint")
    assert len(rows) == len(runtime.evidence_by_endpoint)
    assert {row["endpoint_id"] for row in rows} == set(runtime.evidence_by_endpoint)


def test_the_endpoint_row_carries_both_raw_risks_not_only_the_joint_value():
    """The return boundary is a front over two axes; it cannot be recovered from a scalar."""
    runtime, _ = _cycle()
    row = artifacts.dual_endpoint_evidence_rows(runtime.evidence_by_endpoint, arm="joint")[0]
    evidence = runtime.evidence_by_endpoint[row["endpoint_id"]]
    assert row["raw_risk_a"] == pytest.approx(evidence.a.raw_risk)
    assert row["raw_risk_b"] == pytest.approx(evidence.b.raw_risk)
    assert row["joint_risk"] == pytest.approx(evidence.risk.value)
    assert row["u_a"] == pytest.approx(evidence.risk.u_a)
    assert row["u_b"] == pytest.approx(evidence.risk.u_b)
    assert row["active_worst"] in ("A", "B")
    assert row["allele_a"] != row["allele_b"]


def test_a_density_the_run_did_not_measure_is_absent_rather_than_zero():
    runtime, _ = _cycle()
    row = artifacts.dual_endpoint_evidence_rows(runtime.evidence_by_endpoint, arm="joint")[0]
    assert row["raw_density_a"] is None and row["joint_density"] is None, (
        "a run that never built the density axis did not measure zero")


def test_the_rows_are_stamped_with_the_executing_arm():
    runtime, _ = _cycle()
    for arm in ("joint", "a_only", "b_only"):
        rows = artifacts.dual_endpoint_evidence_rows(runtime.evidence_by_endpoint, arm=arm)
        assert {row["arm"] for row in rows} == {arm}


# -- terminal summary ---------------------------------------------------------------------------

def test_the_terminal_summary_names_the_best_endpoint_under_the_joint_objective():
    runtime, _ = _cycle()
    rows = artifacts.dual_terminal_summary_rows(
        runtime.evidence_by_endpoint, arm="joint", protein_id="5ZHV_B", root_id="r0",
        n_definitive=1, typed_stop="depth_cap")
    assert len(rows) == 1
    best = min(runtime.evidence_by_endpoint,
               key=lambda k: (runtime.evidence_by_endpoint[k].risk.value, k))
    assert rows[0]["best_joint_risk_endpoint_id"] == best
    assert rows[0]["n_endpoints"] == len(runtime.evidence_by_endpoint)


def test_a_run_that_scored_nothing_jointly_emits_no_summary_row():
    assert artifacts.dual_terminal_summary_rows(
        {}, arm="joint", protein_id="P", root_id="r", n_definitive=0) == []


# -- the bundle ----------------------------------------------------------------------------------

def test_the_produced_rows_write_and_read_back_through_the_bundle(tmp_path):
    runtime, _ = _cycle()
    tables = {
        "dual_endpoint_evidence": artifacts.dual_endpoint_evidence_rows(
            runtime.evidence_by_endpoint, arm="joint"),
        "dual_feedback_evidence": [],
        "dual_terminal_summary": artifacts.dual_terminal_summary_rows(
            runtime.evidence_by_endpoint, arm="joint", protein_id="5ZHV_B", root_id="r0",
            n_definitive=1, typed_stop="depth_cap"),
    }
    artifacts.write_v2_bundle(
        tmp_path, manifest={"schema_version": "v2run-1"},
        tables={name: [] for name in artifacts.V2_TABLE_SCHEMAS}, dual_tables=tables)
    table = pq.read_table(pathlib.Path(tmp_path) / "dual_endpoint_evidence.parquet")
    assert table.num_rows == len(tables["dual_endpoint_evidence"])
    assert "double" in str(table.schema.field("raw_risk_a").type)


# -- feedback evidence: the per-allele detail that used to be collapsed ---------------------------

def _real_decision():
    """A decision from the REAL head-directed policy under a Dual authority."""
    from .test_fusion_v2_dual_select import _authority
    from .test_fusion_v2_head_directed_policy import (
        SOURCE_SEQ, _calibration, _coords, _donor, _policy, _source,
    )

    source = _source(SOURCE_SEQ)
    endpoint = _donor(source=source)
    policy = _policy(dual=_authority(endpoint), calibration=_calibration(epsilon=0.0))
    return policy.decide(source=source, endpoint=endpoint, coordinates=_coords())


def _events():
    decision = _real_decision()
    return [{"transition_id": "T:d0", "protein_id": "5ZHV_B",
             "policy_evidence": decision.decision_evidence, "decision": decision}]


def test_the_joint_leave_one_out_keeps_both_per_allele_contrasts():
    rows = artifacts.dual_feedback_evidence_rows(_events(), arm="joint")
    assert rows, "the decision carried no Dual evidence, so the joint law's numbers were discarded"
    scored = [r for r in rows if r["counterfactual_sequence_md5"]]
    assert scored, "no position carried a leave-one-out counterfactual"
    for row in scored:
        assert row["contribution_a"] is not None and row["contribution_b"] is not None
        assert row["raw_a"] is not None and row["raw_b"] is not None
        assert row["joint_contribution"] is not None


def test_the_two_alleles_contribute_different_amounts_at_the_same_position():
    """Guard: if the per-allele columns were equal everywhere they would prove nothing.

    Sign disagreement is a property of the landscape and not guaranteed by any fixture; that the
    two contrasts are DISTINCT numbers is the claim that the joint law kept both rather than
    writing one scalar twice.
    """
    rows = [r for r in artifacts.dual_feedback_evidence_rows(_events(), arm="joint")
            if r["contribution_a"] is not None]
    assert rows
    assert any(r["contribution_a"] != r["contribution_b"] for r in rows)
    # Deliberately NOT asserted: that the joint contribution differs from both per-allele values.
    # When one allele is clearly worst at both the donor and the counterfactual, the smooth maximum
    # saturates and Delta J equals that allele's own contrast exactly -- which is the objective
    # behaving correctly, not a collapsed record. What proves nothing was collapsed is that both
    # contrasts are present and distinct, above.


def test_the_union_reopen_records_which_allele_demanded_the_position():
    rows = artifacts.dual_feedback_evidence_rows(_events(), arm="joint")
    winners = {row["reopen_winning_allele"] for row in rows}
    assert winners - {""}, (
        "no reopen candidate named a winning allele; the union reducer's two sides were computed "
        "and thrown away, so a joint reopen table reads exactly like a single-Head one")


def test_a_legacy_decision_contributes_no_feedback_rows():
    from .test_fusion_v2_head_directed_policy import (
        SOURCE_SEQ, _calibration, _coords, _donor, _policy, _source,
    )

    source = _source(SOURCE_SEQ)
    decision = _policy(calibration=_calibration(epsilon=0.0)).decide(
        source=source, endpoint=_donor(source=source), coordinates=_coords())
    assert artifacts.dual_feedback_evidence_rows(
        [{"transition_id": "T", "protein_id": "P",
          "policy_evidence": decision.decision_evidence, "decision": decision}],
        arm="joint") == []
