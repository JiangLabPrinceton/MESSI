"""DUALF4/F5 cost contract: a Dual cycle issues TWO Head batches and must be charged for both.

The ledger's logical identity is ``(event_id, protein_id, arm, phase)`` with no allele dimension.
:mod:`dual_lookahead` already namespaces the *lookahead* batches by role for exactly that reason;
this file covers the other Head batch a Dual cycle issues -- the leave-one-out counterfactual --
which is real GPU work on the same scale.

Two separate claims are tested, because they fail differently:

* **ledger identity** -- two batches sharing one ``event_id`` merge first-wins, so the run charges
  one allele's calls and registers the second batch as a RETRY. That breaches ``max_retries`` and
  fails the cell, while simultaneously under-reporting ``max_head_calls``. Both directions are
  wrong and neither is visible in the artifact.
* **budget unit** -- the per-cycle cap is named ``max_counterfactual_head_calls_per_cycle`` and
  bounds real forwards. Under Dual one position costs TWO of them, so a cap checked against the
  position count authorizes twice the GPU work it declares.

The legacy single-Head spellings are pinned in the same file: this is a Dual-only change and a
Dual-off ledger must stay byte-identical.
"""

from __future__ import annotations

import pytest

from inverse_folding.reference_flow.fusion_v2 import policy as pol
from inverse_folding.reference_flow.fusion_v2_runtime import ledger as lg

from .test_fusion_v2_dual_select import _authority
from .test_fusion_v2_head_directed_policy import (
    MASKED, SOURCE_SEQ, _calibration, _coords, _donor, _policy, _source,
)

PREFIX = "5ZHV_B:d0:sel"


def _meter(tmp_path):
    return lg.CostMeter(
        journal=lg.AttemptJournal(tmp_path / "journal.jsonl"), protein_id="5ZHV_B",
        arm="fusion_v2", gpu_clock=lambda: 0.0)


def _run(tmp_path, *, dual, budget=64):
    """``budget`` is the CANDIDATE ceiling on whichever path is under test.

    Legacy reads it off the calibration; Dual reads it off the signed overlay's own
    ``max_counterfactual_sequences_per_cycle``, because the legacy field's name says Head calls and
    its gate has always counted positions.
    """
    import dataclasses

    source = _source(SOURCE_SEQ)
    endpoint = _donor(source=source)
    authority = None
    if dual:
        authority = dataclasses.replace(
            _authority(endpoint), max_counterfactual_sequences_per_cycle=budget)
    policy = _policy(
        dual=authority,
        calibration=_calibration(epsilon=0.0, counterfactual_budget=budget))
    meter = _meter(tmp_path)
    result = policy.decide(
        source=source, endpoint=endpoint, coordinates=_coords(),
        runtime=pol.PolicyRuntime(cost_meter=meter, event_prefix=PREFIX))
    # A cycle that stalls before the counterfactual batch never opens an attempt, so the journal
    # file may not exist at all -- an empty ledger, not a missing one.
    path = meter.journal.path
    return result, (lg.events_from_journal(path) if path.exists() else ())


def _counterfactual(events):
    return {e.event_id for e in events if ":counterfactual" in e.event_id}


# -- ledger identity ---------------------------------------------------------------------------

def test_the_two_dual_counterfactual_batches_are_two_logical_events(tmp_path):
    _, events = _run(tmp_path, dual=True)
    assert len(_counterfactual(events)) == 2, (
        "both Heads' counterfactual batches landed on one event_id; the ledger would merge them "
        "first-wins, charge one allele and call the other a retry")


def test_a_dual_cycle_charges_head_calls_for_both_alleles(tmp_path):
    _, dual_events = _run(tmp_path, dual=True)
    totals = lg.aggregate_v2_ledger(dual_events)
    single_totals = lg.aggregate_v2_ledger(_run(tmp_path / "single", dual=False)[1])
    assert totals["head_calls"] == 2 * single_totals["head_calls"], (
        f"a Dual cycle scored the same positions on two Heads but charged "
        f"{totals['head_calls']} against the single-Head {single_totals['head_calls']}")


def test_a_dual_cycle_does_not_manufacture_a_retry(tmp_path):
    _, events = _run(tmp_path, dual=True)
    assert lg.aggregate_v2_ledger(events)["n_retries"] == 0, (
        "the second allele's batch was recorded as a retry of the first; max_retries would fail "
        "the cell for an event that never failed")


def test_only_role_b_is_namespaced_and_role_a_keeps_the_legacy_id(tmp_path):
    """One convention at every stage: role A's id is the legacy one, Dual or not.

    So a Dual ledger differs from a legacy ledger by exactly the added role-B rows -- "what did the
    second Head cost" is one subtraction, and an ``a_only`` arm's cost rows join straight against a
    legacy run's.
    """
    _, events = _run(tmp_path, dual=True)
    assert _counterfactual(events) == {f"{PREFIX}:counterfactual", f"{PREFIX}:counterfactual:b"}


def test_dual_off_keeps_the_legacy_counterfactual_event_id(tmp_path):
    _, events = _run(tmp_path, dual=False)
    assert _counterfactual(events) == {f"{PREFIX}:counterfactual"}


# -- budget unit -------------------------------------------------------------------------------

def _budget_stalled(result) -> bool:
    return isinstance(result, pol.PolicyRejection) and result.reason.startswith(
        pol.StallReason.COUNTERFACTUAL_BUDGET_EXCEEDED.value)


def test_the_dual_candidate_domain_is_not_halved_by_the_second_head(tmp_path):
    """The per-cycle ceiling bounds CANDIDATE POSITIONS on both paths, never Head calls.

    An earlier version charged it in Head calls under Dual, which halved the editable candidate
    domain purely because a second instrument existed. The domain a cycle may consider is a
    property of the design; the logical Head-call budget is 2*C and is the preflight's job.
    """
    result, _ = _run(tmp_path, dual=True, budget=len(MASKED))
    assert not _budget_stalled(result), (
        "a Dual cycle with exactly enough candidate budget was refused; the second Head halved "
        "the editable domain")


def test_the_dual_ceiling_still_refuses_a_cycle_above_the_candidate_domain(tmp_path):
    """The gate must not be satisfiable by never refusing."""
    result, _ = _run(tmp_path, dual=True, budget=len(MASKED) - 1)
    assert _budget_stalled(result)
    assert "logical Head call" in result.reason, (
        "a Dual refusal must still say what the batch would have cost in Head calls")


def test_dual_off_keeps_the_legacy_position_budget(tmp_path):
    # The signed artifact declares this cap in editable positions per cycle; a legacy run at
    # exactly the position count must still pass.
    assert not _budget_stalled(_run(tmp_path, dual=False, budget=len(MASKED))[0])
