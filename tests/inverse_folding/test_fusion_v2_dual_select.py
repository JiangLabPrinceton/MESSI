"""DUALF4 end-to-end: the head-directed policy actually acts jointly through ``select()``.

The unit tests prove the joint law is correct in isolation. This file proves it is REACHED -- that
the union screen widens the candidate set inside the real decision sequence and that the joint
contribution is what ranks the writes, rather than both being computed and then ignored.
"""

from __future__ import annotations

import dataclasses

import pytest

from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion_v2 import dual_policy as dp
from inverse_folding.reference_flow.fusion_v2 import identity as ident
from inverse_folding.reference_flow.fusion_v2 import joint_objective as jo
from inverse_folding.reference_flow.fusion_v2 import policy as pol

from . import _v2_fixtures as F
from .test_fusion_v2_head_directed_policy import (
    CANONICAL_AA20, INCUMBENT_SEQ, MASKED, SOURCE_SEQ, WINDOW_K, _Result, _calibration, _coords,
    _decide, _donor, _evaluator, _incumbent, _policy, _score, _scorer, _source, _windows,
)

L = F.L
EVAL_A = _evaluator()
EVAL_B = dataclasses.replace(EVAL_A, allele="DRB1_0401",
                             head_checkpoint_digest=F.digest("head-b-ckpt"))

#: Role B's landscape is a fixed PERMUTATION of role A's -- weakly related, not anti-correlated.
#: An exact reverse would make risk_A + risk_B constant, so every substitution would improve one
#: allele and worsen the other by exactly the same amount, the joint gate could never pass, and
#: every assertion below would hold for a degenerate reason. The measured cross-allele correlation
#: on this project is about 0.19, which is what this map imitates.
RISK_B = {letter: ((index * 7) % len(CANONICAL_AA20)) * 0.1
          for index, letter in enumerate(CANONICAL_AA20)}


def _windows_b(sequence: str):
    return tuple(
        F.HeadWindow(start_0b=start, end_0b=start + WINDOW_K, k=WINDOW_K,
                     z=round(sum(RISK_B[c] for c in sequence[start:start + WINDOW_K]) / WINDOW_K,
                             10))
        for start in range(len(sequence) - WINDOW_K + 1))


def _score_b(sequence: str, protein_id: str = "5ZHV_B"):
    return F.EndpointHeadScore(
        protein_id=protein_id, sequence_md5=sequence_md5(sequence),
        sequence_length=len(sequence), allele=EVAL_B.allele, score_scale=EVAL_B.score_scale,
        windows=_windows_b(sequence), residue_hotspot=(0.0,) * len(sequence),
        global_risk=round(sum(RISK_B[c] for c in sequence), 10))


class _ScriptedHeadB:
    def __init__(self, protein_id: str = "5ZHV_B"):
        self.protein_id, self.batches = protein_id, []

    def evaluator_identity(self):
        return EVAL_B

    def score(self, requests):
        self.batches.append([r.sequence for r in requests])
        rows = []
        for request in requests:
            score = _score_b(request.sequence, protein_id=self.protein_id)
            rows.append(_Result(
                score=score,
                binding=ident.HeadScoreBinding(
                    protein_id=self.protein_id, sequence_md5=score.sequence_md5,
                    sequence_length=len(request.sequence),
                    window_grid_digest=ident.window_grid_digest(score.windows),
                    evaluator=EVAL_B),
                sequence_md5=score.sequence_md5, global_risk=score.global_risk))
        return rows


def _coordinate(role, evaluator, quantity, location, scale):
    return jo.AlleleCoordinate(
        role=role, quantity=quantity, location=location, scale=scale, raw_noise_floor=1e-6,
        evaluator=evaluator, source_ref=F.digest("panel"))


def _calibration_dual(tau=0.15):
    return jo.DualCalibration(
        panel=jo.PanelBinding(
            panel_id="tier2_natural_v1", panel_digest=F.digest("panel"), n_proteins=53184,
            overlap_fraction_a=0.041, overlap_fraction_b=0.049,
            equal_risk_line_stderr=0.001, leave_overlap_out_shift=0.0005,
            cross_allele_pearson=-0.9),
        risk=jo.QuantityCoordinates(
            quantity="global_risk",
            a=_coordinate(jo.AlleleRole.A, EVAL_A, "global_risk", 0.0, 1.0),
            b=_coordinate(jo.AlleleRole.B, EVAL_B, "global_risk", 0.0, 1.0)),
        density=None,
        window=jo.QuantityCoordinates(
            quantity="window_z",
            a=_coordinate(jo.AlleleRole.A, EVAL_A, "window_z", 0.0, 1.0),
            b=_coordinate(jo.AlleleRole.B, EVAL_B, "window_z", 0.0, 1.0)),
        law=jo.DualObjectiveLaw(mode=jo.ObjectiveMode.SMOOTH_MAX, tau=tau,
                                tau_units="normalized", version="dual-obj-1"),
        version="dual-cal-1")


def _authority(endpoint, *, incumbent_seq=INCUMBENT_SEQ, head_b=None, tau=0.15):
    cal = _calibration_dual(tau)
    objective = jo.DualObjective(cal)
    incumbent_b = _score_b(incumbent_seq)
    return dp.DualSupportAuthority(
        objective=objective, window_coordinates=cal.window, evaluator_b=EVAL_B,
        counterfactual_scorer_b=_scorer(head_b or _ScriptedHeadB()),
        incumbent_score_b=incumbent_b,
        incumbent_joint_value=objective.evaluate(
            raw_a=_score(incumbent_seq).global_risk,
            raw_b=incumbent_b.global_risk).value,
        safety_reference_score_b=incumbent_b,
        donor_score_b_by_endpoint={str(endpoint.endpoint_id): _score_b(endpoint.sequence)},
        max_counterfactual_sequences_per_cycle=64)


def _dual_decision(**over):
    source = _source(SOURCE_SEQ)
    endpoint = _donor(source=source)
    policy = _policy(dual=_authority(endpoint, **over), calibration=_calibration(epsilon=0.0))
    return policy.decide(source=source, endpoint=endpoint, coordinates=_coords())


# ------------------------------------------------------------------------------------------

def test_the_dual_policy_refuses_a_calibration_whose_role_A_is_not_the_head_it_runs():
    endpoint = _donor(source=_source(SOURCE_SEQ))
    # A fully coherent calibration whose role A is the OTHER Head: the objective itself is valid,
    # so the refusal has to come from the policy noticing that role A is not the instrument that
    # produced its endpoints and its incumbent.
    swapped_cal = dataclasses.replace(
        _calibration_dual(),
        risk=jo.QuantityCoordinates(
            quantity="global_risk",
            a=_coordinate(jo.AlleleRole.A, EVAL_B, "global_risk", 0.0, 1.0),
            b=_coordinate(jo.AlleleRole.B, EVAL_A, "global_risk", 0.0, 1.0)),
        window=jo.QuantityCoordinates(
            quantity="window_z",
            a=_coordinate(jo.AlleleRole.A, EVAL_B, "window_z", 0.0, 1.0),
            b=_coordinate(jo.AlleleRole.B, EVAL_A, "window_z", 0.0, 1.0)))
    swapped = dataclasses.replace(
        _authority(endpoint), objective=jo.DualObjective(swapped_cal),
        window_coordinates=swapped_cal.window, evaluator_b=EVAL_A,
        incumbent_score_b=_score(INCUMBENT_SEQ), safety_reference_score_b=_score(INCUMBENT_SEQ),
        donor_score_b_by_endpoint={str(endpoint.endpoint_id): _score(endpoint.sequence)})
    with pytest.raises(pol.V2PolicyError, match="role A"):
        _policy(dual=swapped)


def test_a_dual_decision_runs_the_whole_select_sequence_to_a_committed_decision():
    decision = _dual_decision()
    assert isinstance(decision, pol.PolicyDecision), \
        "a joint decision must reach the same committed outcome type as the single-Head law"
    assert decision.write_from_endpoint, "the joint law selected at least one write"
    assert decision.reopen, "and the shared reopen budget was still spent"
    # writes and reopens are disjoint and both lie in the source's masked domain
    assert not set(decision.write_from_endpoint) & set(decision.reopen)


def test_the_union_screen_widens_the_write_candidate_set_inside_select():
    """The decisive end-to-end assertion: role B's view reaches the candidate set.

    Role B's landscape is the reverse of role A's, so the donor -- built to improve on A at every
    masked position -- WORSENS on B there. Under the single-Head law the candidate set is A's; under
    the joint law it is the union, so it can only be larger or equal, and the two decisions must not
    be the same object.
    """
    source = _source(SOURCE_SEQ)
    endpoint = _donor(source=source)
    single = _policy(calibration=_calibration(epsilon=0.0)).decide(
        source=source, endpoint=endpoint, coordinates=_coords())
    joint = _policy(dual=_authority(endpoint),
                    calibration=_calibration(epsilon=0.0)).decide(
        source=source, endpoint=endpoint, coordinates=_coords())

    def candidates(decision):
        assert isinstance(decision, pol.PolicyDecision), decision
        return {row.position for row in decision.decision_evidence.write_candidates if row.legal}

    single_set, joint_set = candidates(single), candidates(joint)
    # ``>=`` alone is structurally guaranteed by the union reducer and would pass even if role B
    # were never consulted, so it is recorded and then STRENGTHENED below.
    assert joint_set >= single_set, "the union can only widen the A-only screen, never narrow it"
    # and the two laws did not produce the same transition, or the comparison is vacuous
    assert (single.write_from_endpoint, single.reopen) != (joint.write_from_endpoint, joint.reopen)


def test_the_joint_law_and_the_single_head_law_do_not_agree_by_construction():
    """If they agreed the whole comparison would be vacuous, so pin that they do not."""
    source = _source(SOURCE_SEQ)
    endpoint = _donor(source=source)
    objective = jo.DualObjective(_calibration_dual())
    raw_a = _score(endpoint.sequence).global_risk
    raw_b = _score_b(endpoint.sequence).global_risk
    incumbent_a = _score(INCUMBENT_SEQ).global_risk
    incumbent_b = _score_b(INCUMBENT_SEQ).global_risk
    assert raw_a < incumbent_a, "the donor improves under A"
    assert raw_b != incumbent_b, "and the two Heads do not report the same thing about it"
    donor_j = objective.evaluate(raw_a=raw_a, raw_b=raw_b).value
    incumbent_j = objective.evaluate(raw_a=incumbent_a, raw_b=incumbent_b).value
    # the joint verdict is a fact about this fixture, not an assumption: record which way it went
    assert donor_j != incumbent_j


def test_the_joint_donor_gate_compares_J_and_records_it():
    source = _source(SOURCE_SEQ)
    endpoint = _donor(source=source)
    auth = _authority(endpoint)
    joint = auth.joint_comparison(donor=endpoint)
    objective = jo.DualObjective(_calibration_dual())
    expected = objective.evaluate(raw_a=_score(endpoint.sequence).global_risk,
                                  raw_b=_score_b(endpoint.sequence).global_risk).value
    assert joint.donor_value == pytest.approx(expected)
    assert joint.incumbent_value == pytest.approx(auth.incumbent_joint_value)
    assert joint.objective_digest == objective.objective_digest
