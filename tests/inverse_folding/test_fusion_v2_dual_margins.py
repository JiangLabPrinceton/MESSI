"""DUALF3/F4: the Dual gates must compare NORMALIZED quantities against NORMALIZED floors.

PLAN 2.1: the joint margins are derived, not declared --

    eps_joint = max_a eps_a_raw / s_a

-- because ``grad_u J`` is the softmax weight vector, so ``J`` is 1-Lipschitz in the supremum norm
on ``u``.  Each derived margin records the per-allele raw floor it came from.

The defect this file pins is a UNIT error, and unit errors are silent: the donor gate compared a
joint value (normalized) against ``epsilon_r`` (role A's raw scale), and the write filter compared
``Delta J`` (normalized) against ``local_contribution_tolerance`` (also raw).  Whenever
``s_a != 1`` both thresholds mean a different amount of evidence than the run declares -- too
permissive when ``s_a > 1``, too strict when ``s_a < 1`` -- while every stored number still looks
well-formed.
"""

from __future__ import annotations

import dataclasses

import pytest

from inverse_folding.reference_flow.fusion_v2 import joint_objective as jo
from inverse_folding.reference_flow.fusion_v2 import policy as pol

from . import _v2_fixtures as F
from .test_fusion_v2_dual_select import (
    EVAL_A, EVAL_B, _authority, _calibration_dual, _coordinate, _score_b,
)
from .test_fusion_v2_head_directed_policy import (
    SOURCE_SEQ, _calibration, _coords, _donor, _policy, _score, _source,
)

#: Deliberately different, and neither is 1.0: a shared unit scale would let a raw threshold and a
#: normalized one agree by accident, which is exactly the coincidence that hid this.
SCALE_A, SCALE_B = 2.0, 8.0
FLOOR_A, FLOOR_B = 0.02, 0.04
#: max(0.02/2, 0.04/8) = 0.01 -- role A binds. This is the ONE-measurement bound.
EXPECTED_JOINT = max(FLOOR_A / SCALE_A, FLOOR_B / SCALE_B)
#: What a GATE compares against. The donor gate and the write filter each compare a DIFFERENCE of
#: two joint values; each side carries its own drift, so the floor is twice the single-measurement
#: bound. Using the single bound is too permissive by exactly a factor of two, which is the
#: difference between a gate that admits instrument noise and one that does not.
EXPECTED_DECISION = 2.0 * EXPECTED_JOINT


def _scaled_calibration():
    base = _calibration_dual()
    return dataclasses.replace(
        base,
        risk=jo.QuantityCoordinates(
            quantity="global_risk",
            a=dataclasses.replace(
                base.risk.a, scale=SCALE_A, raw_noise_floor=FLOOR_A),
            b=dataclasses.replace(
                base.risk.b, scale=SCALE_B, raw_noise_floor=FLOOR_B)))


def _scaled_authority(endpoint):
    cal = _scaled_calibration()
    objective = jo.DualObjective(cal)
    incumbent_b = _score_b("KKKKKKKKKKKK")
    return dataclasses.replace(
        _authority(endpoint), objective=objective,
        incumbent_joint_value=objective.evaluate(
            raw_a=_score("KKKKKKKKKKKK").global_risk,
            raw_b=incumbent_b.global_risk).value)


def _setup(*, raw_epsilon=0.0, raw_tolerance=0.0):
    source = _source(SOURCE_SEQ)
    endpoint = _donor(source=source)
    policy = _policy(
        dual=_scaled_authority(endpoint),
        calibration=_calibration(epsilon=raw_epsilon, tolerance=raw_tolerance))
    return source, endpoint, policy


# -- the derived margin exists and is the propagated worst allele -------------------------------

def test_the_joint_margin_is_the_worse_allele_of_the_propagated_raw_floors():
    assert _scaled_calibration().risk.joint_margin == pytest.approx(EXPECTED_JOINT)


# -- the donor gate ------------------------------------------------------------------------------

def test_the_dual_donor_gate_compares_against_the_derived_joint_margin():
    _, endpoint, policy = _setup()
    comparison = policy.dual.joint_comparison(donor=endpoint)
    assert comparison.epsilon == pytest.approx(EXPECTED_DECISION), (
        "the joint comparison carries no derived margin, so the gate had to fall back on a raw "
        "threshold for a normalized quantity")


def test_the_joint_comparison_records_the_per_allele_raw_floors_it_came_from():
    _, endpoint, policy = _setup()
    comparison = policy.dual.joint_comparison(donor=endpoint)
    assert comparison.raw_floor_a == pytest.approx(FLOOR_A)
    assert comparison.raw_floor_b == pytest.approx(FLOOR_B)


def test_a_raw_epsilon_large_on_the_raw_scale_does_not_gate_the_normalized_comparison():
    from inverse_folding.reference_flow.fusion_v2.reward import donor_gate

    # 5.0 is enormous in raw risk units and would refuse every donor; it must not be the number a
    # NORMALIZED comparison is judged against.
    _, endpoint, policy = _setup(raw_epsilon=5.0)
    verdict = donor_gate(
        donor=endpoint, incumbent=policy.incumbent, epsilon_r=5.0,
        epsilon_source_ref=policy.calibration.epsilon_source_ref,
        joint=policy.dual.joint_comparison(donor=endpoint))
    assert verdict.epsilon_r == pytest.approx(EXPECTED_DECISION), (
        "the verdict reports the raw knob rather than the margin it actually compared")
    assert verdict.passed


def test_dual_off_still_compares_against_the_raw_epsilon():
    from inverse_folding.reference_flow.fusion_v2.reward import donor_gate

    source = _source(SOURCE_SEQ)
    endpoint = _donor(source=source)
    policy = _policy(calibration=_calibration(epsilon=5.0))
    verdict = donor_gate(
        donor=endpoint, incumbent=policy.incumbent, epsilon_r=5.0,
        epsilon_source_ref=policy.calibration.epsilon_source_ref)
    assert verdict.epsilon_r == 5.0
    assert not verdict.passed


# -- the write filter ---------------------------------------------------------------------------

def test_the_dual_write_filter_uses_the_derived_margin_not_the_raw_tolerance():
    # A raw tolerance far above every normalized contribution would reject every write if it were
    # applied to Delta J.
    _, endpoint, policy = _setup(raw_tolerance=5.0)
    result = policy.decide(source=_source(SOURCE_SEQ), endpoint=endpoint, coordinates=_coords())
    assert isinstance(result, pol.PolicyDecision), (
        f"every write was filtered out by a raw-scale tolerance applied to normalized Delta J: "
        f"{getattr(result, 'reason', result)!r}")
    assert result.write_from_endpoint


# -- the factor of two, pinned on its own --------------------------------------------------------

def test_the_decision_margin_is_twice_the_single_measurement_bound():
    """``d_a = e_a / s_a`` bounds ONE reading's drift; a gate compares two readings.

    Pinned separately from the values above so that collapsing the two quantities back into one --
    the state this was repaired from -- fails here with a message that names the reason, rather
    than only shifting a number in another test.
    """
    objective = jo.DualObjective(_scaled_calibration())
    assert objective.joint_margin == pytest.approx(EXPECTED_JOINT)
    assert objective.decision_margin == pytest.approx(2.0 * objective.joint_margin)


def test_each_single_allele_arm_doubles_its_own_floor_not_the_pair():
    scaled = _scaled_calibration()
    for arm, floor, scale in (("a_only", FLOOR_A, SCALE_A), ("b_only", FLOOR_B, SCALE_B)):
        objective = jo.build_arm_objective(scaled, arm=arm)
        assert objective.joint_margin == pytest.approx(floor / scale)
        assert objective.decision_margin == pytest.approx(2.0 * floor / scale)
