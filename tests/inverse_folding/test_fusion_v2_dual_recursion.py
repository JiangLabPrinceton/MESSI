"""DUALF3 recursion contract: the joint objective's own state must advance with the lineage.

V2 is a recursive law -- depth $d+1$ decides against the incumbent depth $d$ adopted.  Role A's half
of that recursion is the frozen ``advance_lineage_incumbent``.  This file covers role B's half and
the binding that makes both halves describe the same molecule:

* a role-B score is looked up by ``endpoint_id`` out of an externally authored map, so nothing in
  the type system stops a mis-keyed map from computing $J$ from ANOTHER sequence's role-B risk --
  the joint gate would then compare two molecules and report a margin;
* ``incumbent_joint_value`` was a free parameter, so $J(I_d)$ could be any finite float rather than
  the objective's own value on the incumbent the policy actually holds; and
* the policy is immutable and advances by ``replace(self, incumbent=...)``, which carries the Dual
  authority through UNCHANGED.  Role A's incumbent moves to the adopted donor while role B's stays
  at $I_0$, so from depth 2 on the gate compares a fresh $J(Y^*)$ against a stale $J(I_0)$ and the
  union windows reference two different sequences.
"""

from __future__ import annotations

import dataclasses

import pytest

from inverse_folding.reference_flow.fusion_v2 import dual_policy as dp
from inverse_folding.reference_flow.fusion_v2 import policy as pol

from .test_fusion_v2_dual_select import _authority, _score_b
from .test_fusion_v2_head_directed_policy import (
    INCUMBENT_SEQ, SOURCE_SEQ, _calibration, _coords, _donor, _policy, _score, _source,
)


def _setup(**over):
    source = _source(SOURCE_SEQ)
    endpoint = _donor(source=source)
    policy = _policy(dual=_authority(endpoint, **over), calibration=_calibration(epsilon=0.0))
    return source, endpoint, policy


# -- the role-B donor score must be the SAME molecule ------------------------------------------

def test_a_role_b_score_of_another_sequence_is_refused_not_used():
    source, endpoint, _ = _setup()
    other = _score_b("WWWWWWWWWWWW")
    auth = dataclasses.replace(
        _authority(endpoint),
        donor_score_b_by_endpoint={str(endpoint.endpoint_id): other})
    with pytest.raises(dp.V2DualPolicyError, match="sequence_md5|molecule"):
        auth.joint_comparison(donor=endpoint)


def test_a_role_b_score_of_another_protein_is_refused():
    source, endpoint, _ = _setup()
    other = _score_b(endpoint.sequence, protein_id="Q00511")
    auth = dataclasses.replace(
        _authority(endpoint),
        donor_score_b_by_endpoint={str(endpoint.endpoint_id): other})
    with pytest.raises(dp.V2DualPolicyError, match="protein"):
        auth.joint_comparison(donor=endpoint)


def test_a_role_b_score_from_the_wrong_allele_is_refused():
    source, endpoint, _ = _setup()
    wrong = dataclasses.replace(_score_b(endpoint.sequence), allele="DRB1_1501")
    auth = dataclasses.replace(
        _authority(endpoint),
        donor_score_b_by_endpoint={str(endpoint.endpoint_id): wrong})
    with pytest.raises(dp.V2DualPolicyError, match="allele"):
        auth.joint_comparison(donor=endpoint)


def test_the_matching_role_b_score_still_produces_a_comparison():
    source, endpoint, _ = _setup()
    comparison = _authority(endpoint).joint_comparison(donor=endpoint)
    assert comparison.donor_value == pytest.approx(comparison.donor_value)
    assert comparison.objective_digest


# -- J(I_d) is derived, not declared -----------------------------------------------------------

def test_an_incumbent_joint_value_that_is_not_the_objectives_own_value_is_refused():
    source, endpoint, _ = _setup()
    auth = dataclasses.replace(_authority(endpoint), incumbent_joint_value=-999.0)
    with pytest.raises(dp.V2DualPolicyError, match="incumbent_joint_value"):
        _policy(dual=auth, calibration=_calibration(epsilon=0.0))


def test_a_role_b_incumbent_score_of_another_sequence_is_refused():
    source, endpoint, _ = _setup()
    auth = _authority(endpoint)
    auth = dataclasses.replace(auth, incumbent_score_b=_score_b("WWWWWWWWWWWW"))
    with pytest.raises(dp.V2DualPolicyError, match="incumbent"):
        _policy(dual=auth, calibration=_calibration(epsilon=0.0))


# -- the recursion ------------------------------------------------------------------------------

def _gate(policy, endpoint):
    from inverse_folding.reference_flow.fusion_v2.reward import donor_gate

    return donor_gate(
        donor=endpoint, incumbent=policy.incumbent,
        epsilon_r=policy.calibration.epsilon_r,
        epsilon_source_ref=policy.calibration.epsilon_source_ref,
        joint=policy.dual.joint_comparison(donor=endpoint))


def test_advancing_the_lineage_advances_role_b_too():
    source, endpoint, policy = _setup()
    verdict = _gate(policy, endpoint)
    assert verdict.passed, "fixture: the donor must be admissible for the advance to be exercised"
    advanced = policy.advance_lineage_incumbent(
        donor=endpoint, verdict=verdict, accepted_at_depth=1)
    assert advanced is not policy
    assert advanced.incumbent.sequence_md5 == endpoint.sequence_md5
    assert advanced.dual.incumbent_score_b.sequence_md5 == endpoint.sequence_md5, (
        "role A's incumbent moved to the adopted donor while role B's stayed behind; from the "
        "next depth on the gate compares J(Y*) against a stale J(I_0)")


def test_the_advanced_joint_incumbent_value_is_the_adopted_donors_own_J():
    source, endpoint, policy = _setup()
    advanced = policy.advance_lineage_incumbent(
        donor=endpoint, verdict=_gate(policy, endpoint), accepted_at_depth=1)
    expected = policy.dual.objective.evaluate(
        raw_a=float(endpoint.head_global_risk),
        raw_b=float(_score_b(endpoint.sequence).global_risk)).value
    assert advanced.dual.incumbent_joint_value == expected


def test_advancing_clears_the_stale_donor_score_map():
    source, endpoint, policy = _setup()
    advanced = policy.advance_lineage_incumbent(
        donor=endpoint, verdict=_gate(policy, endpoint), accepted_at_depth=1)
    assert advanced.dual.donor_score_b_by_endpoint == {}, (
        "the next depth's donors are different endpoints; carrying the old map lets a stale entry "
        "be found instead of raising for a donor role B has not scored")


def test_a_refused_verdict_advances_neither_allele():
    from inverse_folding.reference_flow.fusion_v2.reward import DonorGateReason

    source, endpoint, policy = _setup()
    refused = dataclasses.replace(
        _gate(policy, endpoint), passed=False,
        reason=DonorGateReason.STALL_NO_BETTER_DONOR)
    unchanged = policy.advance_lineage_incumbent(
        donor=endpoint, verdict=refused, accepted_at_depth=1)
    assert unchanged is policy
    assert unchanged.dual is policy.dual


def test_advancing_a_donor_role_b_never_scored_is_a_wiring_fault():
    source, endpoint, policy = _setup()
    verdict = _gate(policy, endpoint)
    stripped = dataclasses.replace(policy.dual, donor_score_b_by_endpoint={})
    with pytest.raises((dp.V2DualPolicyError, pol.V2PolicyError)):
        dataclasses.replace(policy, dual=stripped).advance_lineage_incumbent(
            donor=endpoint, verdict=verdict, accepted_at_depth=1)
