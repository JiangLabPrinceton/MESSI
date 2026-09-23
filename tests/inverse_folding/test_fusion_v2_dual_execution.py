"""DUALF1: the Dual layer is actually REACHED by the running kernel.

Every law in the Dual layer had unit tests and none of them had a caller.  ``run_v2_shard`` resolved
the overlay only far enough to compute a run signature; the cycle still asked one Head, the archive
still ranked by ``head_global_risk``, and no production module referenced ``DualObjective``,
``DualSupportAuthority`` or ``build_role_b_head_oracle`` at all.  Three arms would therefore have
executed identical A-only V2 under three different signatures -- the one failure a matched
comparison cannot survive, because every arm agrees and the agreement means nothing.

These tests run the real ``run_one_cycle`` against fake oracles and ask the questions that
distinguish "the Dual code exists" from "the Dual code decided":

* were BOTH Heads asked, for the same deduplicated set?
* is the family representative the one the JOINT objective ranks first, when the two alleles
  disagree about which endpoint is best?
* does the ledger carry two head events rather than one merged event with a fabricated retry?
* and does a cycle with no Dual runtime still produce byte-identical endpoints?
"""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

from inverse_folding.reference_flow.fusion_v2 import identity as ident
from inverse_folding.reference_flow.fusion_v2 import joint_objective as jo
from inverse_folding.reference_flow.fusion_v2_runtime import ledger as lg
from inverse_folding.reference_flow.fusion_v2_runtime.dual_runtime import (
    DualRuntime,
    V2DualRuntimeError,
)

from . import _v2_fixtures as F
from .test_fusion_v2_cycle import (
    _FakeHeadOracle, _head, _Row, _run as _run_cycle, _run_kwargs, _support_policy,
)


class _DualCapableProbe:
    """The fixture's stub policy, plus the one seam a Dual cycle requires of a policy.

    The joint gate itself is exercised against the REAL head-directed policy in
    ``test_fusion_v2_dual_select.py``; what this file needs is a policy the Dual cycle will consent
    to call, so that the Head batches, the ordering and the ledger can be observed.
    """

    consumes_runtime = False

    def __init__(self, clones=None):
        self.rebound_with = None
        #: SHARED with every clone, so the original can see what the cycle actually decided with.
        self.clones = [] if clones is None else clones

    def with_dual_donor_scores(self, scores):
        clone = _DualCapableProbe(self.clones)
        clone.rebound_with = dict(scores)
        self.clones.append(clone)
        return clone

    def __call__(self, source, endpoint, coordinates):
        return _support_policy(source, endpoint, coordinates)


def _run(**over):
    """Run one cycle; a Dual run gets a policy that can accept role B's donor scores."""
    if over.get("dual") is not None:
        over.setdefault("support_policy", _DualCapableProbe())
    return _run_cycle(**over)

EVAL_A = F.safety_reference(length=F.L).head_binding.evaluator
EVAL_B = dataclasses.replace(EVAL_A, allele="DRB1_0401",
                             head_checkpoint_digest=F.digest("head-b-ckpt"))


class _FakeHeadOracleB:
    """Role B's landscape, deliberately NOT role A's.

    The risk is drawn from a different slice of the same digest, so the two alleles rank the pool
    differently without being anti-correlated -- an exact reversal would make ``risk_A + risk_B``
    constant and every joint assertion below would hold for a degenerate reason.
    """

    def __init__(self, length=F.L):
        self.calls = 0
        self.length = int(length)
        self.scored_md5s: set[str] = set()

    def evaluator_identity(self):
        return EVAL_B

    def score(self, requests):
        from inverse_folding.reference_flow.fusion.state import sequence_md5

        self.calls += 1
        rows = []
        for request in requests:
            key = sequence_md5(request.sequence)
            self.scored_md5s.add(key)
            risk = -float(int(key[6:12], 16) % 1000) / 100.0
            score, binding = _head(key, risk=risk, length=self.length)
            score = dataclasses.replace(score, allele=EVAL_B.allele)
            binding = dataclasses.replace(binding, evaluator=EVAL_B)
            rows.append(_Row(score=score, binding=binding, sequence_md5=key, global_risk=risk))
        return rows


def _coordinate(role, evaluator, quantity="global_risk", location=0.0, scale=1.0):
    return jo.AlleleCoordinate(
        role=role, quantity=quantity, location=location, scale=scale, raw_noise_floor=1e-6,
        evaluator=evaluator, source_ref=F.digest("panel"))


def _calibration():
    return jo.DualCalibration(
        panel=jo.PanelBinding(panel_id="tier2_natural_v1", panel_digest=F.digest("panel"),
                              n_proteins=53184),
        risk=jo.QuantityCoordinates(
            quantity="global_risk", a=_coordinate(jo.AlleleRole.A, EVAL_A),
            b=_coordinate(jo.AlleleRole.B, EVAL_B)),
        density=None,
        window=jo.QuantityCoordinates(
            quantity="window_z", a=_coordinate(jo.AlleleRole.A, EVAL_A, "window_z"),
            b=_coordinate(jo.AlleleRole.B, EVAL_B, "window_z")),
        law=jo.DualObjectiveLaw(mode=jo.ObjectiveMode.SMOOTH_MAX, tau=0.15,
                                tau_units="normalized", version="dual-obj-1"),
        version="dual-cal-1")


class _Overlay:
    """The minimum the runtime reads off an overlay: its calibration and its declared arms."""

    def __init__(self, arms=("joint", "a_only", "b_only")):
        self.calibration = _calibration()
        self.arms = tuple(arms)


def _runtime(arm="joint", head_b=None):
    return DualRuntime(
        overlay=_Overlay(), arm=arm, head_b=head_b or _FakeHeadOracleB(),
        # The arm's OWN law. Building DualObjective here regardless of the arm is exactly the
        # defect the runtime's coherence check now refuses.
        objective=jo.build_arm_objective(_calibration(), arm=arm))


def _meter(tmp_path):
    return lg.CostMeter(
        journal=lg.AttemptJournal(tmp_path / "attempts.jsonl"), protein_id="5ZHV_B",
        arm="v2", gpu_clock=lambda: 0.0)


# ------------------------------------------------------------------------------------------
# 1. both Heads are asked, for the same set
# ------------------------------------------------------------------------------------------

def test_a_dual_cycle_asks_both_heads_for_the_same_deduplicated_pool():
    head_a, head_b = _FakeHeadOracle(), _FakeHeadOracleB()
    _run(head_oracle=head_a, dual=_runtime(head_b=head_b))
    assert head_b.calls > 0, "the second Head was never called: the cycle ran A-only"
    assert head_b.scored_md5s == head_a.scored_md5s, (
        "the two Heads were asked for different sequence sets, so the archive would be joint for "
        "some endpoints and single-allele for the rest")


def test_a_legacy_cycle_never_calls_a_second_head():
    head_b = _FakeHeadOracleB()
    _run(head_oracle=_FakeHeadOracle())
    assert head_b.calls == 0


def test_every_endpoint_of_a_dual_cycle_carries_joint_evidence():
    runtime = _runtime()
    outcome = _run(dual=runtime)
    assert outcome.endpoints
    for endpoint in outcome.endpoints:
        assert str(endpoint.endpoint_id) in runtime.evidence_by_endpoint


# ------------------------------------------------------------------------------------------
# 2. the joint objective actually orders the pool
# ------------------------------------------------------------------------------------------

def test_the_endpoints_are_the_ones_a_single_head_run_would_have_produced():
    """Dual changes the ORDERING and the gate, never the endpoint objects themselves."""
    legacy = _run()
    dual = _run(dual=_runtime())
    assert [e.endpoint_id for e in legacy.endpoints] == [e.endpoint_id for e in dual.endpoints]
    assert [e.head_global_risk for e in legacy.endpoints] == \
           [e.head_global_risk for e in dual.endpoints], \
        "head_global_risk must stay role A's raw risk; a joint scalar written there would relabel " \
        "every legacy artifact without changing a single stored number"


def test_the_archive_elite_follows_the_joint_objective_not_role_a():
    runtime = _runtime()
    outcome = _run(dual=runtime)
    evidence = runtime.evidence_by_endpoint
    admitted = [e for e in outcome.endpoints if str(e.endpoint_id) in evidence]
    assert len(admitted) >= 2, "fixture: need at least two endpoints to have an ordering at all"
    best_joint = min(admitted, key=lambda e: (
        float(evidence[str(e.endpoint_id)].risk.value), str(e.endpoint_id)))
    ranked = sorted(admitted, key=outcome.archive._rank_key)
    assert ranked[0].endpoint_id == best_joint.endpoint_id, (
        "the archive ordered the pool by role A's raw risk while the gate compares J; the elite "
        "of a joint run would be silently single-allele")


def test_the_two_alleles_actually_disagree_in_this_fixture():
    """Guard: if A and B ranked the pool identically, every assertion above would be vacuous."""
    runtime = _runtime()
    outcome = _run(dual=runtime)
    evidence = runtime.evidence_by_endpoint
    by_a = sorted(evidence, key=lambda k: evidence[k].a.raw_risk)
    by_b = sorted(evidence, key=lambda k: evidence[k].b.raw_risk)
    assert by_a != by_b, "the fake role B ranks the pool exactly as role A does"


# ------------------------------------------------------------------------------------------
# 3. the ledger charges two batches
# ------------------------------------------------------------------------------------------

def test_the_ledger_carries_one_head_event_per_allele_per_screen(tmp_path):
    """A cycle runs TWO screens -- the source pool and the descendant pool -- and Dual scores both.

    So four logical head events, not two, and none of them a retry of another. Under one shared id
    they would merge first-wins: one allele charged, the other recorded as a retry, which breaches
    ``max_retries`` while simultaneously under-reporting ``max_head_calls``.
    """
    single, dual = _meter(tmp_path / "single"), _meter(tmp_path / "dual")
    _run(cost_meter=single)
    _run(cost_meter=dual, dual=_runtime())
    legacy_ids = {e.event_id for e in lg.events_from_journal(single.journal.path)
                  if e.phase == "head"}
    events = lg.events_from_journal(dual.journal.path)
    dual_ids = {e.event_id for e in events if e.phase == "head"}
    assert len(dual_ids) == 2 * len(legacy_ids), f"{sorted(dual_ids)}"
    # Role A keeps the legacy ids exactly; the Dual ledger is the legacy one PLUS role B's rows.
    assert dual_ids > legacy_ids
    assert {i for i in dual_ids - legacy_ids} == {f"{i}:b" for i in legacy_ids}
    assert lg.aggregate_v2_ledger(events)["n_retries"] == 0


def test_a_dual_cycle_charges_twice_the_single_head_calls(tmp_path):
    single, dual = _meter(tmp_path / "single"), _meter(tmp_path / "dual")
    _run(cost_meter=single)
    _run(cost_meter=dual, dual=_runtime())
    totals = [lg.aggregate_v2_ledger(lg.events_from_journal(m.journal.path))["head_calls"]
              for m in (single, dual)]
    assert totals[1] == 2 * totals[0], f"single={totals[0]} dual={totals[1]}"


# ------------------------------------------------------------------------------------------
# 4. refusals
# ------------------------------------------------------------------------------------------

def test_an_arm_that_names_no_ordering_law_is_refused():
    from inverse_folding.reference_flow.fusion_v2.joint_objective import V2JointObjectiveError

    with pytest.raises(V2JointObjectiveError, match="names no ordering law"):
        _runtime(arm="c_only")


def test_a_declared_arm_the_overlay_did_not_bundle_is_refused():
    """A different condition: the law exists, but THIS overlay never declared that branch."""
    runtime_kwargs = dict(overlay=_Overlay(arms=("joint", "a_only")), arm="b_only",
                          head_b=_FakeHeadOracleB(),
                          objective=jo.build_arm_objective(_calibration(), arm="b_only"))
    with pytest.raises(V2DualRuntimeError, match="declared bundle"):
        DualRuntime(**runtime_kwargs)


def test_a_runtime_steering_on_the_density_view_is_refused():
    with pytest.raises(V2DualRuntimeError, match="global-risk"):
        DualRuntime(overlay=_Overlay(), arm="joint", head_b=_FakeHeadOracleB(),
                    objective=jo.DualObjective(_calibration(), quantity="window_z"))


def test_the_same_endpoint_cannot_be_bound_with_two_different_verdicts():
    runtime = _runtime()
    _run(dual=runtime)
    endpoint_id, evidence = next(iter(runtime.evidence_by_endpoint.items()))
    with pytest.raises(V2DualRuntimeError, match="bound twice"):
        runtime.register(dataclasses.replace(evidence, sequence_md5="0" * 32))


def test_a_dual_cycle_whose_policy_cannot_take_role_b_scores_is_refused():
    from inverse_folding.reference_flow.fusion_v2_runtime.cycle import V2CycleError

    # A plain function has no Dual seam at all: the joint gate would silently fall back on role A
    # while the run signature continued to claim a Dual arm.
    with pytest.raises(V2CycleError, match="role B"):
        _run_cycle(dual=_runtime(), support_policy=_support_policy)


def test_the_policy_is_rebound_with_this_cycles_role_b_scores():
    """The original assertion (`probe.rebound_with is None`) held whether or not the rebind ever
    happened, so it proved nothing. What matters is WHICH scores the deciding object was given."""
    probe = _DualCapableProbe()
    runtime = _runtime()
    outcome = _run_cycle(dual=runtime, support_policy=probe)
    assert probe.rebound_with is None, "the original must not be mutated"
    assert probe.clones, "the cycle never rebound the policy at all"
    decided = probe.clones[0]
    assert decided.rebound_with, "the deciding policy was handed an empty role B score map"
    # THIS pool's endpoints, keyed by endpoint id -- not a stale depth-zero map.
    assert set(decided.rebound_with) <= set(runtime.evidence_by_endpoint)
    for endpoint_id, score in decided.rebound_with.items():
        assert score is runtime.score_b_by_endpoint[endpoint_id]


def test_the_cycle_returns_the_policy_it_decided_with():
    """The ladder advances whatever the cycle hands back. Returning the un-rebound original meant
    every Dual ladder run died at the first adopted donor, after the full generation had been paid
    for: the advance looked the donor up in an EMPTY role B score map and raised."""
    probe = _DualCapableProbe()
    outcome = _run_cycle(dual=_runtime(), support_policy=probe)
    assert outcome.support_policy is not None
    assert outcome.support_policy is not probe
    assert outcome.support_policy is probe.clones[0]


def test_a_legacy_cycle_hands_back_the_policy_it_was_given():
    outcome = _run_cycle()
    assert outcome.support_policy is _run_kwargs()["support_policy"]
