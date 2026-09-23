"""DUALF4 pure layer: joint leave-one-out attribution and the symmetric union reopen reducer."""

from __future__ import annotations

import dataclasses

import pytest

from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion_v2 import dual_policy as dp
from inverse_folding.reference_flow.fusion_v2 import evidence as ev
from inverse_folding.reference_flow.fusion_v2 import identity as ident
from inverse_folding.reference_flow.fusion_v2 import joint_objective as jo

from . import _v2_fixtures as fx
from .test_fusion_v2_dual_evidence import EVAL_A, EVAL_B, coordinate

L = fx.L
DONOR = "ACDEFG"[:L]
INCUMBENT = "GFEDCA"[:L]


# ------------------------------------------------------------------------------------------
# calibration with a per-window coordinate pair
# ------------------------------------------------------------------------------------------

def window_pair(*, scale_a=2.0, scale_b=4.0, loc_a=0.0, loc_b=1.0):
    return jo.QuantityCoordinates(
        quantity="window_z",
        a=coordinate(jo.AlleleRole.A, EVAL_A, quantity="window_z", location=loc_a, scale=scale_a),
        b=coordinate(jo.AlleleRole.B, EVAL_B, quantity="window_z", location=loc_b, scale=scale_b))


def calibration(**over):
    kwargs = dict(
        panel=jo.PanelBinding(
            panel_id="tier2_natural_v1", panel_digest=fx.digest("panel"), n_proteins=53184,
            overlap_fraction_a=0.041, overlap_fraction_b=0.049,
            equal_risk_line_stderr=0.028, leave_overlap_out_shift=0.004,
            cross_allele_pearson=0.19),
        risk=jo.QuantityCoordinates(
            quantity="global_risk", a=coordinate(jo.AlleleRole.A, EVAL_A),
            b=coordinate(jo.AlleleRole.B, EVAL_B)),
        density=None, window=window_pair(),
        law=jo.DualObjectiveLaw(mode=jo.ObjectiveMode.SMOOTH_MAX, tau=0.15,
                                tau_units="normalized", version="dual-obj-1"),
        version="dual-cal-1")
    kwargs.update(over)
    return jo.DualCalibration(**kwargs)


# ------------------------------------------------------------------------------------------
# window evidence builders: one allele can see a position the other cannot
# ------------------------------------------------------------------------------------------

def score(evaluator, *, z_by_window, sequence, md5=None):
    windows = tuple(
        fx.HeadWindow(start_0b=start, end_0b=start + fx.WINDOW_K, k=fx.WINDOW_K, z=float(z))
        for start, z in enumerate(z_by_window))
    return fx.EndpointHeadScore(
        protein_id="5ZHV_B", sequence_md5=md5 or sequence_md5(sequence),
        sequence_length=len(sequence), allele=evaluator.allele,
        score_scale=evaluator.score_scale, windows=windows,
        residue_hotspot=(-0.1,) * len(sequence), global_risk=float(max(z_by_window)))


N_WINDOWS = L - fx.WINDOW_K + 1


def paired(donor_z_a, ref_z_a, donor_z_b, ref_z_b, **over):
    a = ev.build_window_evidence(
        donor_score=score(EVAL_A, z_by_window=donor_z_a, sequence=DONOR),
        reference_score=score(EVAL_A, z_by_window=ref_z_a, sequence=INCUMBENT),
        evaluator=EVAL_A, reference_label="lineage_incumbent")
    b = ev.build_window_evidence(
        donor_score=score(EVAL_B, z_by_window=donor_z_b, sequence=DONOR,
                          md5=sequence_md5(DONOR)),
        reference_score=score(EVAL_B, z_by_window=ref_z_b, sequence=INCUMBENT,
                              md5=sequence_md5(INCUMBENT)),
        evaluator=EVAL_B, reference_label="lineage_incumbent")
    kw = dict(a=a, b=b, coordinates=calibration().window)
    kw.update(over)
    return dp.PairedWindowEvidence(**kw)


# ------------------------------------------------------------------------------------------
# 1. the union screen
# ------------------------------------------------------------------------------------------

def test_a_position_only_head_B_finds_improved_still_reaches_the_joint_batch():
    # A sees no improvement anywhere; B improves window 0, which covers positions 0..K-1.
    view = paired(donor_z_a=[0.0] * N_WINDOWS, ref_z_a=[0.0] * N_WINDOWS,
                  donor_z_b=[-1.0] + [0.0] * (N_WINDOWS - 1), ref_z_b=[0.0] * N_WINDOWS)
    assert not view.a.improves_at(0)
    assert view.b.improves_at(0)
    assert view.improves_at(0), "an A-only prefilter would have dropped this position"


def test_the_union_screen_is_symmetric_under_allele_swap():
    forward = paired(donor_z_a=[0.0] * N_WINDOWS, ref_z_a=[0.0] * N_WINDOWS,
                     donor_z_b=[-1.0] + [0.0] * (N_WINDOWS - 1), ref_z_b=[0.0] * N_WINDOWS)
    swapped = paired(donor_z_a=[-1.0] + [0.0] * (N_WINDOWS - 1), ref_z_a=[0.0] * N_WINDOWS,
                     donor_z_b=[0.0] * N_WINDOWS, ref_z_b=[0.0] * N_WINDOWS)
    assert forward.improves_at(0) == swapped.improves_at(0) is True


def test_a_position_no_allele_covers_is_uncovered_rather_than_zero():
    view = paired(donor_z_a=[0.0] * N_WINDOWS, ref_z_a=[0.0] * N_WINDOWS,
                  donor_z_b=[0.0] * N_WINDOWS, ref_z_b=[0.0] * N_WINDOWS)
    outside = L + 5
    assert not view.covered(outside)
    assert view.residual_burden_at(outside) is None
    assert view.worsening_union(outside).value is None


# ------------------------------------------------------------------------------------------
# 2. the normalization the reducer applies
# ------------------------------------------------------------------------------------------

def test_a_worsening_conjunct_is_normalized_as_a_difference_not_as_a_level():
    # donor worse than reference by +2.0 under A only.
    view = paired(donor_z_a=[2.0] + [0.0] * (N_WINDOWS - 1), ref_z_a=[0.0] * N_WINDOWS,
                  donor_z_b=[0.0] * N_WINDOWS, ref_z_b=[0.0] * N_WINDOWS)
    got = view.worsening_union(0)
    # scale_a = 2.0, location_a = 0.0 -> a LEVEL map would also give 1.0 here, so use a coordinate
    # whose location is non-zero to tell them apart
    shifted = paired(donor_z_a=[2.0] + [0.0] * (N_WINDOWS - 1), ref_z_a=[0.0] * N_WINDOWS,
                     donor_z_b=[0.0] * N_WINDOWS, ref_z_b=[0.0] * N_WINDOWS,
                     coordinates=window_pair(loc_a=10.0))
    assert got.value == pytest.approx(2.0 / 2.0)
    assert shifted.worsening_union(0).value == pytest.approx(2.0 / 2.0), \
        "the location must cancel in a difference"


def test_a_residual_burden_conjunct_is_normalized_as_an_absolute_level():
    view = paired(donor_z_a=[3.0] + [0.0] * (N_WINDOWS - 1), ref_z_a=[0.0] * N_WINDOWS,
                  donor_z_b=[0.0] * N_WINDOWS, ref_z_b=[0.0] * N_WINDOWS,
                  coordinates=window_pair(loc_a=1.0, scale_a=2.0, loc_b=1.0, scale_b=4.0))
    # A: (3.0 - 1.0)/2.0 = 1.0 ; B: (0.0 - 1.0)/4.0 = -0.25 -> A wins
    got = view.residual_burden_union(0)
    assert got.value_a == pytest.approx(1.0)
    assert got.value_b == pytest.approx(-0.25)
    assert got.value == pytest.approx(1.0)
    assert got.winning_allele is jo.AlleleRole.A


def test_the_reduced_value_is_symmetric_under_allele_swap_and_only_the_label_is_not():
    view = paired(donor_z_a=[3.0] + [0.0] * (N_WINDOWS - 1), ref_z_a=[0.0] * N_WINDOWS,
                  donor_z_b=[0.0] * N_WINDOWS, ref_z_b=[0.0] * N_WINDOWS)
    left = view.residual_burden_union(0)
    mirrored = dp._reduce(left.value_b, left.value_a)
    assert mirrored.value == pytest.approx(left.value)
    assert mirrored.winning_allele is not left.winning_allele


def test_an_exact_tie_is_broken_deterministically_and_declared():
    tied = dp._reduce(1.0, 1.0)
    assert tied.value == 1.0
    assert tied.winning_allele is jo.AlleleRole.A
    assert dp._reduce(1.0, 1.0).winning_allele is dp._reduce(1.0, 1.0).winning_allele


def test_the_reducer_refuses_to_borrow_the_aggregate_risk_scale():
    with pytest.raises(dp.V2DualPolicyError):
        paired(donor_z_a=[0.0] * N_WINDOWS, ref_z_a=[0.0] * N_WINDOWS,
               donor_z_b=[0.0] * N_WINDOWS, ref_z_b=[0.0] * N_WINDOWS,
               coordinates=calibration().risk)


def test_one_allele_supplied_twice_is_refused():
    a = ev.build_window_evidence(
        donor_score=score(EVAL_A, z_by_window=[0.0] * N_WINDOWS, sequence=DONOR),
        reference_score=score(EVAL_A, z_by_window=[0.0] * N_WINDOWS, sequence=INCUMBENT),
        evaluator=EVAL_A, reference_label="lineage_incumbent")
    with pytest.raises(dp.V2DualPolicyError):
        dp.PairedWindowEvidence(a=a, b=a, coordinates=calibration().window)


# ------------------------------------------------------------------------------------------
# 3. the joint leave-one-out
# ------------------------------------------------------------------------------------------

class Scorer:
    """A Head that returns a declared risk per counterfactual sequence."""

    def __init__(self, evaluator, risk_by_md5, default):
        self.evaluator, self.risk_by_md5, self.default = evaluator, risk_by_md5, default
        self.calls = 0

    def __call__(self, protein_id, sequences):
        self.calls += 1
        out = []
        for seq in sequences:
            md5 = sequence_md5(seq)
            out.append(dataclasses.replace(
                score(self.evaluator, z_by_window=[0.0] * N_WINDOWS, sequence=seq, md5=md5),
                global_risk=float(self.risk_by_md5.get(md5, self.default))))
        return [_WithBinding(row, self.evaluator) for row in out]


@dataclasses.dataclass
class _WithBinding:
    score: object
    evaluator: ident.HeadEvaluatorIdentity

    def __post_init__(self):
        self.sequence_md5 = self.score.sequence_md5
        self.global_risk = self.score.global_risk
        self.binding = ident.HeadScoreBinding(
            protein_id=self.score.protein_id, sequence_md5=self.score.sequence_md5,
            sequence_length=self.score.sequence_length,
            window_grid_digest=ident.window_grid_digest(self.score.windows),
            evaluator=self.evaluator)


def joint(positions=(0,), risk_a=None, risk_b=None, donor_a=-5.0, donor_b=-5.0):
    grid = ident.window_grid_digest(
        score(EVAL_A, z_by_window=[0.0] * N_WINDOWS, sequence=DONOR).windows)
    obj = jo.DualObjective(calibration())
    return dp.score_joint_leave_one_out(
        scorer_a=Scorer(EVAL_A, risk_a or {}, donor_a + 1.0),
        scorer_b=Scorer(EVAL_B, risk_b or {}, donor_b + 1.0),
        protein_id="5ZHV_B", donor_sequence=DONOR, donor_raw_a=donor_a, donor_raw_b=donor_b,
        incumbent_sequence=INCUMBENT, positions=positions,
        evaluator_a=EVAL_A, evaluator_b=EVAL_B, window_grid_digest=grid, objective=obj)


def test_the_write_evidence_is_the_joint_delta_not_a_per_allele_contrast():
    """The load-bearing assertion of DUALF4: a_i is Delta J, not Delta R_A and not Delta R_B.

    Deliberately asymmetric -- role A improves on reverting while role B worsens -- so the three
    quantities are numerically distinct and an implementation that quietly reported either allele's
    own contrast would fail rather than coincide.
    """
    donor_a, donor_b = -5.0, -1.0
    md5 = sequence_md5(ev.counterfactual_sequence(DONOR, INCUMBENT, 0))
    got = joint(positions=(0,), donor_a=donor_a, donor_b=donor_b,
                risk_a={md5: -9.0}, risk_b={md5: +3.0})[0]
    obj = jo.DualObjective(calibration())
    expected = obj.evaluate(raw_a=got.raw_a, raw_b=got.raw_b).value \
        - obj.evaluate(raw_a=donor_a, raw_b=donor_b).value
    assert got.contribution == pytest.approx(expected)

    assert got.contribution_a == pytest.approx(-9.0 - donor_a)      # -4.0, A got better
    assert got.contribution_b == pytest.approx(+3.0 - donor_b)      # +4.0, B got worse
    # all three are genuinely different numbers, so the assertion above has content
    assert got.contribution != pytest.approx(got.contribution_a)
    assert got.contribution != pytest.approx(got.contribution_b)
    # and the joint value follows the WORSE allele: reverting made B worse, so a_i > 0
    assert got.contribution > 0.0


def test_both_per_allele_contrasts_are_retained_as_telemetry():
    got = joint(positions=(0, 1))
    for row in got.values():
        assert row.contribution_a == pytest.approx(row.raw_a - (-5.0))
        assert row.contribution_b == pytest.approx(row.raw_b - (-5.0))


def test_each_allele_pays_exactly_one_head_batch_over_the_same_reverted_bytes():
    obj = jo.DualObjective(calibration())
    grid = ident.window_grid_digest(
        score(EVAL_A, z_by_window=[0.0] * N_WINDOWS, sequence=DONOR).windows)
    sa, sb = Scorer(EVAL_A, {}, -4.0), Scorer(EVAL_B, {}, -4.0)
    out = dp.score_joint_leave_one_out(
        scorer_a=sa, scorer_b=sb, protein_id="5ZHV_B", donor_sequence=DONOR,
        donor_raw_a=-5.0, donor_raw_b=-5.0, incumbent_sequence=INCUMBENT, positions=(0, 1, 2),
        evaluator_a=EVAL_A, evaluator_b=EVAL_B, window_grid_digest=grid, objective=obj)
    assert sa.calls == 1 and sb.calls == 1
    expected = {ev.counterfactual_sequence(DONOR, INCUMBENT, p) for p in (0, 1, 2)}
    assert {row.counterfactual_sequence_md5 for row in out.values()} == \
        {sequence_md5(seq) for seq in expected}


def test_an_allele_that_scored_different_bytes_is_refused():
    class Wrong(Scorer):
        def __call__(self, protein_id, sequences):
            return super().__call__(protein_id, [INCUMBENT for _ in sequences])
    obj = jo.DualObjective(calibration())
    grid = ident.window_grid_digest(
        score(EVAL_A, z_by_window=[0.0] * N_WINDOWS, sequence=DONOR).windows)
    with pytest.raises(dp.V2DualPolicyError):
        dp.score_joint_leave_one_out(
            scorer_a=Scorer(EVAL_A, {}, -4.0), scorer_b=Wrong(EVAL_B, {}, -4.0),
            protein_id="5ZHV_B", donor_sequence=DONOR, donor_raw_a=-5.0, donor_raw_b=-5.0,
            incumbent_sequence=INCUMBENT, positions=(0,), evaluator_a=EVAL_A, evaluator_b=EVAL_B,
            window_grid_digest=grid, objective=obj)


def test_a_position_shortlisted_twice_is_refused():
    with pytest.raises(dp.V2DualPolicyError):
        joint(positions=(0, 0))


def test_the_same_head_bound_to_both_roles_is_refused():
    obj = jo.DualObjective(calibration())
    grid = ident.window_grid_digest(
        score(EVAL_A, z_by_window=[0.0] * N_WINDOWS, sequence=DONOR).windows)
    with pytest.raises(dp.V2DualPolicyError):
        dp.score_joint_leave_one_out(
            scorer_a=Scorer(EVAL_A, {}, -4.0), scorer_b=Scorer(EVAL_A, {}, -4.0),
            protein_id="5ZHV_B", donor_sequence=DONOR, donor_raw_a=-5.0, donor_raw_b=-5.0,
            incumbent_sequence=INCUMBENT, positions=(0,), evaluator_a=EVAL_A, evaluator_b=EVAL_A,
            window_grid_digest=grid, objective=obj)


def test_a_density_or_window_view_cannot_attribute_a_write():
    obj = jo.DualObjective(calibration(), quantity="window_z")
    grid = ident.window_grid_digest(
        score(EVAL_A, z_by_window=[0.0] * N_WINDOWS, sequence=DONOR).windows)
    with pytest.raises(dp.V2DualPolicyError):
        dp.score_joint_leave_one_out(
            scorer_a=Scorer(EVAL_A, {}, -4.0), scorer_b=Scorer(EVAL_B, {}, -4.0),
            protein_id="5ZHV_B", donor_sequence=DONOR, donor_raw_a=-5.0, donor_raw_b=-5.0,
            incumbent_sequence=INCUMBENT, positions=(0,), evaluator_a=EVAL_A, evaluator_b=EVAL_B,
            window_grid_digest=grid, objective=obj)


# ------------------------------------------------------------------------------------------
# 4. the injected support authority
# ------------------------------------------------------------------------------------------

def b_score(**over):
    base = score(EVAL_B, z_by_window=[0.0] * N_WINDOWS, sequence=INCUMBENT)
    return dataclasses.replace(base, **over)


def authority(**over):
    cal = calibration()
    kwargs = dict(
        objective=jo.DualObjective(cal),
        window_coordinates=cal.window,
        evaluator_b=EVAL_B,
        counterfactual_scorer_b=Scorer(EVAL_B, {}, -4.0),
        incumbent_score_b=b_score(),
        incumbent_joint_value=-1.0,
        safety_reference_score_b=b_score(),
        donor_score_b_by_endpoint={"endpoint:x": b_score()},
        max_counterfactual_sequences_per_cycle=64,
    )
    kwargs.update(over)
    return dp.DualSupportAuthority(**kwargs)


def donor_stub(endpoint_id="endpoint:x", **over):
    """Role A's view of one donor: the identity fields the role B score is proved against."""
    reference = b_score()
    kwargs = dict(
        endpoint_id=endpoint_id, protein_id=reference.protein_id,
        sequence_md5=reference.sequence_md5, sequence_length=reference.sequence_length,
        head_global_risk=-1.0,
        head_binding=ident.HeadScoreBinding(
            protein_id=reference.protein_id, sequence_md5=reference.sequence_md5,
            sequence_length=reference.sequence_length,
            window_grid_digest=ident.window_grid_digest(reference.windows), evaluator=EVAL_A))
    kwargs.update(over)
    return dataclasses.make_dataclass(
        "_Donor", [(name, object) for name in kwargs], frozen=True)(**kwargs)


def test_the_authority_accepts_a_coherent_dual_binding():
    auth = authority()
    assert auth.donor_score_b(donor_stub()).allele == EVAL_B.allele


def test_an_authority_steering_on_a_non_risk_view_is_refused():
    cal = calibration()
    with pytest.raises(dp.V2DualPolicyError):
        authority(objective=jo.DualObjective(cal, quantity="window_z"))


def test_an_authority_carrying_the_aggregate_scale_as_window_coordinates_is_refused():
    with pytest.raises(dp.V2DualPolicyError):
        authority(window_coordinates=calibration().risk)


def test_an_evaluator_that_is_not_role_B_is_refused():
    with pytest.raises(dp.V2DualPolicyError):
        authority(evaluator_b=EVAL_A)


def test_a_role_B_score_for_the_wrong_allele_is_refused():
    with pytest.raises(dp.V2DualPolicyError):
        authority(incumbent_score_b=score(EVAL_A, z_by_window=[0.0] * N_WINDOWS,
                                          sequence=INCUMBENT))
    with pytest.raises(dp.V2DualPolicyError):
        authority(safety_reference_score_b=score(EVAL_A, z_by_window=[0.0] * N_WINDOWS,
                                                 sequence=INCUMBENT))


def test_a_missing_incumbent_joint_value_cannot_default_to_zero():
    for bad in (None, float("nan"), float("inf"), True):
        with pytest.raises(dp.V2DualPolicyError):
            authority(incumbent_joint_value=bad)


def test_a_donor_only_one_head_has_seen_is_refused_rather_than_substituted():
    auth = authority()
    with pytest.raises(dp.V2DualPolicyError):
        auth.donor_score_b(donor_stub("endpoint:never-scored-by-B"))


def test_the_union_view_is_a_drop_in_for_the_single_allele_view():
    """Drift guard: the policy must be able to consume a union wherever it consumed one allele.

    Names AND signatures, because a union that answered `worsening_at` with a record instead of a
    float would force a second branch in select(), and a second branch is the thing most likely to
    drift out of agreement with the single-Head law Dual-off must remain equivalent to.
    """
    import inspect
    view = paired(donor_z_a=[0.0] * N_WINDOWS, ref_z_a=[0.0] * N_WINDOWS,
                  donor_z_b=[0.0] * N_WINDOWS, ref_z_b=[0.0] * N_WINDOWS)
    single = view.a
    consumed = ("covered", "improves_at", "min_delta_at", "worsening_at", "residual_burden_at",
                "evidence_digest", "donor_sequence_md5", "sequence_length")
    for name in consumed:
        assert hasattr(view, name), name
    for name in ("covered", "improves_at", "min_delta_at", "worsening_at", "residual_burden_at"):
        assert inspect.signature(getattr(view, name)) == inspect.signature(getattr(single, name)), \
            name
    # and the scalar answers have the same types as the single-allele ones
    assert isinstance(view.worsening_at(0), float)
    assert view.min_delta_at(0) is None or isinstance(view.min_delta_at(0), float)
    assert isinstance(view.evidence_digest, str) and view.evidence_digest != single.evidence_digest
