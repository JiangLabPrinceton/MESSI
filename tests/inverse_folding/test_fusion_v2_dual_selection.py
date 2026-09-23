"""DUALF3: family selection, archive elite, incumbent and donor gate under ONE frozen objective."""

from __future__ import annotations

import dataclasses

import pytest

from inverse_folding.reference_flow.fusion_v2 import dual_evidence as de
from inverse_folding.reference_flow.fusion_v2 import reward as rw
from inverse_folding.reference_flow.fusion_v2_runtime import archive as arch
from inverse_folding.reference_flow.fusion_v2_runtime import dual_selection as ds

from . import _v2_fixtures as fx
from .test_fusion_v2_dual_evidence import EVAL_B, calibration, objective, result_a, score_b


def endpoint(tokens, **over):
    return fx.endpoint(tokens=tokens, **over)


def pool():
    """Two feasible endpoints. Role A prefers the FIRST; role B prefers the SECOND."""
    first = endpoint((10, 20, 12, 13, 14, 15))
    second = endpoint((11, 20, 12, 13, 14, 15))
    first = dataclasses.replace(first, head_global_risk=-9.0,
                                head_score=dataclasses.replace(first.head_score, global_risk=-9.0))
    second = dataclasses.replace(second, head_global_risk=-1.0,
                                 head_score=dataclasses.replace(second.head_score, global_risk=-1.0))
    obj = objective()
    evidence = {
        first.endpoint_id: de.bind_dual_evidence(
            endpoint=first, result_a=result_a(first), result_b=score_b(first, risk=+50.0),
            objective=obj),
        second.endpoint_id: de.bind_dual_evidence(
            endpoint=second, result_a=result_a(second), result_b=score_b(second, risk=-50.0),
            objective=obj),
    }
    return first, second, evidence, obj


def test_the_legacy_ordering_is_unchanged_when_no_objective_is_injected():
    first, second, _, _ = pool()
    assert arch.legacy_rank_key(first) < arch.legacy_rank_key(second)
    assert arch.select_family_representatives((second, first))[0].endpoint_id == first.endpoint_id
    box = arch.ExactArchive()
    box.admit(second, depth=0)
    box.admit(first, depth=0)
    assert box.elite().endpoint_id == first.endpoint_id


def test_the_injected_objective_changes_the_elite_and_the_family_representative_together():
    first, second, evidence, _ = pool()
    key = ds.dual_rank_key(evidence)
    # role B is catastrophic on `first` and excellent on `second`, so the worst-residual law
    # reverses the single-Head preference
    assert key(second) < key(first)
    assert arch.select_family_representatives((first, second), rank_key=key)[0].endpoint_id \
        == second.endpoint_id
    box = arch.ExactArchive(rank_key=key)
    box.admit(first, depth=0)
    box.admit(second, depth=0)
    assert box.elite().endpoint_id == second.endpoint_id
    # ...and the same pool under the legacy key still prefers `first`, so the difference is the law
    legacy = arch.ExactArchive()
    legacy.admit(first, depth=0)
    legacy.admit(second, depth=0)
    assert legacy.elite().endpoint_id == first.endpoint_id


def test_the_elite_is_independent_of_admission_order_under_the_injected_objective():
    first, second, evidence, _ = pool()
    key = ds.dual_rank_key(evidence)
    forward, backward = arch.ExactArchive(rank_key=key), arch.ExactArchive(rank_key=key)
    forward.admit(first, depth=0)
    forward.admit(second, depth=0)
    backward.admit(second, depth=0)
    backward.admit(first, depth=0)
    assert forward.elite().endpoint_id == backward.elite().endpoint_id


def test_an_endpoint_without_dual_evidence_is_refused_rather_than_ordered_by_the_raw_head():
    first, second, evidence, _ = pool()
    partial = {first.endpoint_id: evidence[first.endpoint_id]}
    key = ds.dual_rank_key(partial)
    with pytest.raises(de.V2DualEvidenceError):
        key(second)


def test_a_pool_carrying_two_calibrations_cannot_be_ordered_at_all():
    first, second, evidence, _ = pool()
    other = objective(cal=calibration(density=True))
    evidence[second.endpoint_id] = de.bind_dual_evidence(
        endpoint=second, result_a=result_a(second), result_b=score_b(second, risk=-50.0),
        objective=other)
    with pytest.raises(de.V2DualEvidenceError):
        ds.dual_rank_key(evidence)


def test_an_infeasible_endpoint_never_becomes_elite_however_good_its_joint_value():
    first, second, evidence, _ = pool()
    from inverse_folding.reference_flow.fusion_v2 import state as st
    unvalidated = dataclasses.replace(second, feasibility_level=st.FeasibilityLevel.UNVALIDATED,
                                      structure_outcome=None)
    key = ds.dual_rank_key({**evidence, unvalidated.endpoint_id: evidence[second.endpoint_id]})
    box = arch.ExactArchive(rank_key=key)
    box.admit(first, depth=0)
    box.admit(unvalidated, depth=0)
    assert box.elite().endpoint_id == first.endpoint_id


# ------------------------------------------------------------------------------------------
# the donor gate
# ------------------------------------------------------------------------------------------

def incumbent_of(ep):
    return rw.bind_incumbent_from_endpoint(
        endpoint=ep, lineage_id=ep.lineage.family_id, evaluator=ep.head_binding.evaluator,
        safety_reference_sequence_md5="e" * 32, accepted_at_depth=0)


def test_the_dual_off_donor_gate_is_unchanged():
    first, second, _, _ = pool()
    held = incumbent_of(second)          # legacy risk -1.0
    verdict = rw.donor_gate(donor=first, incumbent=held, epsilon_r=0.005,
                            epsilon_source_ref=fx.digest("eps"))
    assert verdict.passed
    assert verdict.margin == pytest.approx(-1.0 - (-9.0))
    assert verdict.joint is None


def test_the_dual_gate_compares_joint_values_while_the_raw_fields_keep_their_meaning():
    first, second, evidence, _ = pool()
    held = incumbent_of(second)
    joint = ds.joint_comparison_for(
        donor=first, incumbent_value=evidence[second.endpoint_id].risk.value,
        incumbent_objective_digest=evidence[second.endpoint_id].risk.objective_digest,
        evidence_by_endpoint=evidence, objective=objective())
    verdict = rw.donor_gate(donor=first, incumbent=held, epsilon_r=0.005,
                            epsilon_source_ref=fx.digest("eps"), joint=joint)
    # single-Head would have accepted `first`; the joint law refuses it, because role B is
    # catastrophic there
    assert not verdict.passed
    assert verdict.reason is rw.DonorGateReason.STALL_NO_BETTER_DONOR
    assert verdict.donor_global_risk == pytest.approx(-9.0)      # still role A's RAW risk
    assert verdict.margin == pytest.approx(joint.margin)


def test_a_donor_and_incumbent_from_two_calibrations_cannot_form_a_margin():
    first, second, evidence, _ = pool()
    with pytest.raises(de.V2DualEvidenceError):
        ds.joint_comparison_for(
            donor=first, incumbent_value=-1.0,
            incumbent_objective_digest=fx.digest("a-different-calibration"),
            evidence_by_endpoint=evidence, objective=objective())


def test_a_joint_comparison_requires_finite_values_and_a_real_digest():
    with pytest.raises(rw.V2RewardError):
        rw.JointComparison(donor_value=float("nan"), incumbent_value=0.0,
                           objective_digest=fx.digest("o"))
    with pytest.raises(Exception):
        rw.JointComparison(donor_value=0.0, incumbent_value=0.0, objective_digest="unset:obj")


def test_an_infeasible_donor_is_still_refused_under_the_joint_law():
    first, second, evidence, _ = pool()
    from inverse_folding.reference_flow.fusion_v2 import state as st
    held = incumbent_of(second)
    unvalidated = dataclasses.replace(first, feasibility_level=st.FeasibilityLevel.UNVALIDATED,
                                      structure_outcome=None)
    joint = rw.JointComparison(donor_value=-99.0, incumbent_value=0.0,
                               objective_digest=evidence[first.endpoint_id].calibration_digest)
    verdict = rw.donor_gate(donor=unvalidated, incumbent=held, epsilon_r=0.005,
                            epsilon_source_ref=fx.digest("eps"), joint=joint)
    assert not verdict.passed
    assert verdict.reason is rw.DonorGateReason.DONOR_NOT_DEFINITIVE
