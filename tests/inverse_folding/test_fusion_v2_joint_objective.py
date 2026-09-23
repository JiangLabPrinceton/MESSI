"""DUALF1: the pure Dual objective, calibration record, and evidence identity.

RED-first. Every test here is a behaviour the scientific document
(``doc/Dual_Allele_Steering.md`` §§2.2-2.3) or ``PLAN_RF_FUSION_V2_DUAL_ALLELE.md`` §2.1 states,
expressed as an executable claim. The module under test must stay pure: stdlib only, no torch, no
config import, no I/O -- the same purity contract ``fusion_v2/reward.py`` documents for the donor
gate, and for the same reason (a policy that can reach an unfrozen threshold is not frozen).
"""

from __future__ import annotations

import math

import pytest

from inverse_folding.reference_flow.fusion_v2 import identity as ident
from inverse_folding.reference_flow.fusion_v2 import joint_objective as jo

LOG2 = math.log(2.0)


# ------------------------------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------------------------------

def evaluator(allele: str, ckpt: str) -> ident.HeadEvaluatorIdentity:
    return ident.HeadEvaluatorIdentity(
        allele=allele, score_scale="raw_logit", window_k_min=12, window_k_max=25,
        head_config_hash="acb178be" * 8, head_checkpoint_digest=ckpt,
    )


EVAL_A = evaluator("DRB1_0701", "3ab78004" * 8)
EVAL_B = evaluator("DRB1_0401", "7425a927" * 8)


def coordinate(role, *, quantity="global_risk", location=0.0, scale=2.0,
               floor=0.017012596130371094, ev=None) -> jo.AlleleCoordinate:
    return jo.AlleleCoordinate(
        role=role, quantity=quantity, location=location, scale=scale,
        raw_noise_floor=floor, evaluator=ev if ev is not None else (EVAL_A if role is jo.AlleleRole.A else EVAL_B),
        source_ref="panel-digest-0001",
    )


def quantity_pair(*, quantity="global_risk", a=None, b=None) -> jo.QuantityCoordinates:
    a = a if a is not None else coordinate(jo.AlleleRole.A, quantity=quantity)
    b = b if b is not None else coordinate(jo.AlleleRole.B, quantity=quantity)
    return jo.QuantityCoordinates(quantity=quantity, a=a, b=b)


def law(**over) -> jo.DualObjectiveLaw:
    kwargs = dict(mode=jo.ObjectiveMode.SMOOTH_MAX, tau=0.15, tau_units="normalized",
                  version="dual-obj-1")
    kwargs.update(over)
    return jo.DualObjectiveLaw(**kwargs)


def panel(**over) -> jo.PanelBinding:
    kwargs = dict(
        panel_id="tier2_natural_v1", panel_digest="panel-digest-0001", n_proteins=53184,
        overlap_fraction_a=0.041, overlap_fraction_b=0.049,
        equal_risk_line_stderr=0.028, leave_overlap_out_shift=0.004,
        cross_allele_pearson=0.19,
    )
    kwargs.update(over)
    return jo.PanelBinding(**kwargs)


def calibration(**over) -> jo.DualCalibration:
    kwargs = dict(panel=panel(), risk=quantity_pair(), density=None, law=law(),
                  version="dual-cal-1")
    kwargs.update(over)
    return jo.DualCalibration(**kwargs)


class Score:
    """The minimal HeadScoreLike surface the resolver reads."""

    def __init__(self, *, risk, allele, protein_id="5ZHV_B", sequence_md5="0" * 32,
                 sequence_length=102, score_scale="raw_logit", grid="grid-digest-1",
                 density=None):
        self.global_risk = risk
        self.allele = allele
        self.protein_id = protein_id
        self.sequence_md5 = sequence_md5
        self.sequence_length = sequence_length
        self.score_scale = score_scale
        self.window_grid_digest = grid
        self.positive_mass_density = density


def score_pair(risk_a, risk_b, **over):
    return (Score(risk=risk_a, allele="DRB1_0701", **over),
            Score(risk=risk_b, allele="DRB1_0401", **over))


# ------------------------------------------------------------------------------------------
# 1. the objective's mathematical contract
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("u_a,u_b", [
    (0.0, 0.0), (1.0, -1.0), (-3.5, -3.4), (2.0, -9.0), (-0.001, 0.001), (5.0, 5.0),
])
def test_smooth_max_is_bounded_by_the_hard_max_and_its_log2_shortfall(u_a, u_b):
    tau = 0.15
    j = jo.smooth_max(u_a, u_b, tau)
    m = max(u_a, u_b)
    assert m - tau * LOG2 <= j <= m


def test_smooth_max_fixes_the_additive_convention_at_the_diagonal():
    for u in (-9.0, -1.0, 0.0, 3.25):
        assert jo.smooth_max(u, u, 0.15) == pytest.approx(u, abs=1e-12)


def test_a_shift_common_to_both_alleles_shifts_the_objective_by_exactly_that_amount():
    # doc/Dual_Allele_Steering.md 2.2.3: this is why only b_A - b_B is load-bearing.
    for c in (-7.0, -0.5, 0.0, 4.0):
        assert jo.smooth_max(1.0 + c, -2.0 + c, 0.2) == pytest.approx(
            jo.smooth_max(1.0, -2.0, 0.2) + c, abs=1e-12)


def test_the_objective_is_symmetric_under_allele_swap():
    assert jo.smooth_max(1.0, -2.0, 0.2) == pytest.approx(jo.smooth_max(-2.0, 1.0, 0.2), abs=1e-15)
    assert jo.hard_max(1.0, -2.0) == jo.hard_max(-2.0, 1.0)


def test_the_objective_is_strictly_increasing_in_each_allele_coordinate():
    base = jo.smooth_max(-1.0, -2.0, 0.2)
    assert jo.smooth_max(-0.9, -2.0, 0.2) > base
    assert jo.smooth_max(-1.0, -1.9, 0.2) > base


def test_a_pareto_dominated_endpoint_can_never_outrank_its_dominator():
    dominator = jo.smooth_max(-3.0, -4.0, 0.2)
    for du, dv in ((0.1, 0.0), (0.0, 0.1), (0.1, 0.1), (2.0, 0.001)):
        dominated = jo.smooth_max(-3.0 + du, -4.0 + dv, 0.2)
        assert dominated > dominator


def test_an_arbitrarily_favourable_non_worst_allele_buys_at_most_tau_log2():
    tau = 0.15
    worst = 0.0
    ceiling = worst
    floor = worst - tau * LOG2
    for other in (-1.0, -10.0, -1e3, -1e6):
        j = jo.smooth_max(worst, other, tau)
        assert floor <= j <= ceiling
    # and the bound is approached, not merely respected
    assert jo.smooth_max(worst, -1e6, tau) == pytest.approx(floor, abs=1e-12)


def test_the_smooth_max_converges_to_the_hard_max_as_tau_goes_to_zero():
    u_a, u_b = -3.0, -3.4
    previous = None
    for tau in (1.0, 0.1, 0.01, 1e-3, 1e-6):
        j = jo.smooth_max(u_a, u_b, tau)
        if previous is not None:
            assert abs(j - jo.hard_max(u_a, u_b)) <= abs(previous - jo.hard_max(u_a, u_b))
        previous = j
    assert jo.smooth_max(u_a, u_b, 1e-9) == pytest.approx(jo.hard_max(u_a, u_b), abs=1e-8)


@pytest.mark.parametrize("u_a,u_b,tau", [
    (1e6, -1e6, 0.01), (-1e6, 1e6, 0.01), (1e5, 1e5, 1e-4), (-1e8, -1e8, 1e-3),
])
def test_no_configuration_overflows_a_direct_exponential(u_a, u_b, tau):
    j = jo.smooth_max(u_a, u_b, tau)
    assert math.isfinite(j)
    assert max(u_a, u_b) - tau * LOG2 <= j <= max(u_a, u_b)


def test_a_non_positive_or_non_finite_tau_is_refused():
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(jo.V2JointObjectiveError):
            jo.smooth_max(0.0, 0.0, bad)


def test_a_non_finite_coordinate_is_refused_rather_than_propagated():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(jo.V2JointObjectiveError):
            jo.smooth_max(bad, 0.0, 0.2)
        with pytest.raises(jo.V2JointObjectiveError):
            jo.hard_max(0.0, bad)


# ------------------------------------------------------------------------------------------
# 2. the calibration record
# ------------------------------------------------------------------------------------------

def test_a_non_positive_or_non_finite_scale_is_refused():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(jo.V2JointObjectiveError):
            coordinate(jo.AlleleRole.A, scale=bad)


def test_a_scale_no_larger_than_its_own_raw_noise_floor_is_refused():
    # PLAN 2.1: "Fail rather than invent a floor".
    with pytest.raises(jo.V2JointObjectiveError):
        coordinate(jo.AlleleRole.A, scale=0.017, floor=0.017)


def test_the_two_roles_must_be_A_and_B_exactly():
    with pytest.raises(jo.V2JointObjectiveError):
        jo.QuantityCoordinates(quantity="global_risk", a=coordinate(jo.AlleleRole.B),
                               b=coordinate(jo.AlleleRole.B))
    with pytest.raises(jo.V2JointObjectiveError):
        # swapped: the record's own role fields must match the slot they occupy
        jo.QuantityCoordinates(quantity="global_risk", a=coordinate(jo.AlleleRole.B, ev=EVAL_B),
                               b=coordinate(jo.AlleleRole.A, ev=EVAL_A))


def test_binding_the_same_head_to_both_roles_is_refused():
    # The only bit that discriminates the two production Heads is the checkpoint digest:
    # they share a config directory and their checkpoint metadata config_hash is identical.
    same = coordinate(jo.AlleleRole.B, ev=evaluator("DRB1_0401", EVAL_A.head_checkpoint_digest))
    with pytest.raises(jo.V2JointObjectiveError):
        jo.QuantityCoordinates(quantity="global_risk", a=coordinate(jo.AlleleRole.A), b=same)


def test_two_roles_declaring_the_same_allele_are_refused():
    with pytest.raises(jo.V2JointObjectiveError):
        jo.QuantityCoordinates(
            quantity="global_risk", a=coordinate(jo.AlleleRole.A),
            b=coordinate(jo.AlleleRole.B, ev=evaluator("DRB1_0701", EVAL_B.head_checkpoint_digest)))


def test_the_two_heads_must_share_score_scale_and_window_grid_contract():
    other_scale = ident.HeadEvaluatorIdentity(
        allele="DRB1_0401", score_scale="probability", window_k_min=12, window_k_max=25,
        head_config_hash=EVAL_B.head_config_hash,
        head_checkpoint_digest=EVAL_B.head_checkpoint_digest)
    with pytest.raises(jo.V2JointObjectiveError):
        jo.QuantityCoordinates(quantity="global_risk", a=coordinate(jo.AlleleRole.A),
                               b=coordinate(jo.AlleleRole.B, ev=other_scale))
    other_k = ident.HeadEvaluatorIdentity(
        allele="DRB1_0401", score_scale="raw_logit", window_k_min=12, window_k_max=30,
        head_config_hash=EVAL_B.head_config_hash,
        head_checkpoint_digest=EVAL_B.head_checkpoint_digest)
    with pytest.raises(jo.V2JointObjectiveError):
        jo.QuantityCoordinates(quantity="global_risk", a=coordinate(jo.AlleleRole.A),
                               b=coordinate(jo.AlleleRole.B, ev=other_k))


def test_a_coordinate_declaring_a_different_quantity_than_its_pair_is_refused():
    with pytest.raises(jo.V2JointObjectiveError):
        jo.QuantityCoordinates(
            quantity="global_risk", a=coordinate(jo.AlleleRole.A),
            b=coordinate(jo.AlleleRole.B, quantity="positive_mass_density"))


def test_an_undeclared_measured_quantity_is_refused():
    with pytest.raises(jo.V2JointObjectiveError):
        coordinate(jo.AlleleRole.A, quantity="whatever_i_like")


# ------------------------------------------------------------------------------------------
# 3. tau is declared in normalized units, and the margins are derived
# ------------------------------------------------------------------------------------------

def test_tau_declared_in_a_raw_score_scale_is_refused():
    # 2.3: s_A != s_B makes one normalized credit two different raw credits, so a raw-unit tau
    # would smuggle the scale asymmetry back into the objective.
    for bad in ("raw_logit", "raw", "", None, "Normalized"):
        with pytest.raises(jo.V2JointObjectiveError):
            law(tau_units=bad)


def test_the_declared_credit_is_tau_log2():
    assert law(tau=0.15).credit == pytest.approx(0.15 * LOG2)


def test_hard_max_carries_no_credit_and_still_shares_the_interface():
    hard = law(mode=jo.ObjectiveMode.HARD_MAX)
    assert hard.credit == 0.0
    cal = calibration(law=hard)
    obj = jo.DualObjective(cal)
    got = obj.evaluate(raw_a=1.0, raw_b=-5.0)
    assert got.value == pytest.approx(max(got.u_a, got.u_b))
    assert got.mode is jo.ObjectiveMode.HARD_MAX


def test_the_joint_margin_is_the_worse_allele_of_the_propagated_raw_floors():
    # eps_joint = max_a eps_a_raw / s_a  -- because grad_u J is a softmax weight vector,
    # so J is 1-Lipschitz in the sup norm on u.
    pair = jo.QuantityCoordinates(
        quantity="global_risk",
        a=coordinate(jo.AlleleRole.A, scale=2.0, floor=0.02),
        b=coordinate(jo.AlleleRole.B, scale=8.0, floor=0.04))
    assert pair.joint_margin == pytest.approx(max(0.02 / 2.0, 0.04 / 8.0))


def test_the_propagated_margin_actually_bounds_the_objective_response():
    tau = 0.15
    s_a, s_b, floor = 2.0, 8.0, 0.04
    bound = max(floor / s_a, floor / s_b)
    for u_a, u_b in ((-3.0, -3.0), (-3.0, -9.0), (0.5, 0.4), (-1.0, 5.0)):
        for d_a, d_b in ((floor / s_a, 0.0), (0.0, floor / s_b), (floor / s_a, floor / s_b)):
            moved = abs(jo.smooth_max(u_a + d_a, u_b + d_b, tau) - jo.smooth_max(u_a, u_b, tau))
            assert moved <= bound + 1e-12


# ------------------------------------------------------------------------------------------
# 4. panel gates
# ------------------------------------------------------------------------------------------

def test_the_panel_diagnostics_are_reported_and_optional():
    """They are OUTPUTS of the one calibration pass, not knobs anyone tunes.

    A panel that measured them carries them; one that did not leaves them absent rather than zero,
    because "not measured" and "measured as zero" are different facts. Nothing gates on them --
    the only number a human chooses in this whole calibration is tau.
    """
    bare = jo.PanelBinding(panel_id="p", panel_digest="panel-digest-0001", n_proteins=53184)
    assert bare.overlap_fraction_a is None and bare.cross_allele_pearson is None
    calibration(panel=bare)                      # a minimal panel is admissible
    rich = panel()
    assert rich.cross_allele_pearson == 0.19
    # out-of-range values are still refused when present
    with pytest.raises(jo.V2JointObjectiveError):
        jo.PanelBinding(panel_id="p", panel_digest="panel-digest-0001", n_proteins=1,
                        cross_allele_pearson=1.5)
    with pytest.raises(jo.V2JointObjectiveError):
        jo.PanelBinding(panel_id="p", panel_digest="panel-digest-0001", n_proteins=1,
                        overlap_fraction_a=-0.1)


def test_a_panel_with_no_proteins_is_refused():
    with pytest.raises(jo.V2JointObjectiveError):
        panel(n_proteins=0)


def test_a_placeholder_panel_digest_is_refused():
    # require_digest raises V2IdentityError, which is a V2Error -- the same propagation reward.py
    # relies on. Asserting the narrow subclass here would pin an accident rather than the contract.
    with pytest.raises(ident.V2IdentityError):
        panel(panel_digest="unset:panel")
    with pytest.raises(ident.V2IdentityError):
        panel(panel_digest="")


# ------------------------------------------------------------------------------------------
# 5. the resolver
# ------------------------------------------------------------------------------------------

def test_normalization_maps_raw_risks_through_the_frozen_affine_coordinates():
    cal = calibration(risk=quantity_pair(
        a=coordinate(jo.AlleleRole.A, location=-1.0, scale=2.0),
        b=coordinate(jo.AlleleRole.B, location=3.0, scale=4.0)))
    got = jo.DualObjective(cal).evaluate(raw_a=1.0, raw_b=-1.0)
    assert got.u_a == pytest.approx((1.0 - (-1.0)) / 2.0)
    assert got.u_b == pytest.approx((-1.0 - 3.0) / 4.0)


def test_the_resolver_names_the_active_worst_allele():
    cal = calibration()
    obj = jo.DualObjective(cal)
    assert obj.evaluate(raw_a=1.0, raw_b=-1.0).active_worst is jo.AlleleRole.A
    assert obj.evaluate(raw_a=-1.0, raw_b=1.0).active_worst is jo.AlleleRole.B


def test_the_rank_key_is_the_value_then_endpoint_id_and_is_row_order_independent():
    cal = calibration()
    obj = jo.DualObjective(cal)
    tie_a = obj.evaluate(raw_a=1.0, raw_b=-1.0)
    tie_b = obj.evaluate(raw_a=1.0, raw_b=-1.0)
    assert tie_a.value == tie_b.value
    rows = [("endpoint:zzz", tie_a), ("endpoint:aaa", tie_b)]
    forward = sorted(rows, key=lambda r: r[1].rank_key(r[0]))
    backward = sorted(reversed(rows), key=lambda r: r[1].rank_key(r[0]))
    assert [r[0] for r in forward] == [r[0] for r in backward] == ["endpoint:aaa", "endpoint:zzz"]


def test_scoring_two_heads_on_different_sequences_is_refused():
    obj = jo.DualObjective(calibration())
    a, b = score_pair(1.0, -1.0)
    b.sequence_md5 = "f" * 32
    with pytest.raises(jo.V2JointObjectiveError):
        obj.evaluate_scores(score_a=a, score_b=b)


def test_scoring_two_heads_on_different_window_grids_is_refused():
    obj = jo.DualObjective(calibration())
    a, b = score_pair(1.0, -1.0)
    b.window_grid_digest = "grid-digest-2"
    with pytest.raises(jo.V2JointObjectiveError):
        obj.evaluate_scores(score_a=a, score_b=b)


def test_a_score_whose_allele_contradicts_its_role_is_refused():
    obj = jo.DualObjective(calibration())
    a, b = score_pair(1.0, -1.0)
    a.allele = "DRB1_0401"
    with pytest.raises(jo.V2JointObjectiveError):
        obj.evaluate_scores(score_a=a, score_b=b)


def test_only_declared_quantities_can_be_resolved_and_density_must_be_calibrated():
    cal = calibration(density=None)
    with pytest.raises(jo.V2JointObjectiveError):
        jo.DualObjective(cal, quantity="positive_mass_density")


def test_the_density_objective_uses_its_own_coordinates_and_the_same_tau():
    density = quantity_pair(
        quantity="positive_mass_density",
        a=coordinate(jo.AlleleRole.A, quantity="positive_mass_density", location=0.1, scale=0.05,
                     floor=0.0001),
        b=coordinate(jo.AlleleRole.B, quantity="positive_mass_density", location=0.2, scale=0.10,
                     floor=0.0001))
    cal = calibration(density=density)
    got = jo.DualObjective(cal, quantity="positive_mass_density").evaluate(raw_a=0.15, raw_b=0.10)
    assert got.u_a == pytest.approx((0.15 - 0.1) / 0.05)
    assert got.u_b == pytest.approx((0.10 - 0.2) / 0.10)
    assert got.value == pytest.approx(jo.smooth_max(got.u_a, got.u_b, cal.law.tau))


# ------------------------------------------------------------------------------------------
# 6. identity and determinism
# ------------------------------------------------------------------------------------------

def test_the_calibration_digest_is_deterministic_and_content_bound():
    assert calibration().content_digest == calibration().content_digest
    moved = calibration(risk=quantity_pair(a=coordinate(jo.AlleleRole.A, location=0.5)))
    assert moved.content_digest != calibration().content_digest


def test_the_objective_digest_travels_with_every_resolved_value():
    cal = calibration()
    got = jo.DualObjective(cal).evaluate(raw_a=1.0, raw_b=-1.0)
    assert got.objective_digest == jo.arm_objective_digest(cal, "joint")


def test_changing_tau_changes_the_calibration_digest():
    assert calibration(law=law(tau=0.15)).content_digest != \
        calibration(law=law(tau=0.20)).content_digest


def test_changing_the_objective_mode_changes_the_calibration_digest():
    assert calibration(law=law(mode=jo.ObjectiveMode.SMOOTH_MAX)).content_digest != \
        calibration(law=law(mode=jo.ObjectiveMode.HARD_MAX)).content_digest


def test_the_module_exposes_no_way_to_fit_coordinates_from_a_candidate_cloud():
    # 2.2: "never recomputed from the current endpoint cloud". The pure layer must not even
    # offer the verb, so a later caller cannot reach for it.
    forbidden = ("fit", "estimate", "from_endpoints", "from_batch", "from_cloud", "recalibrate",
                 "update", "refit")
    names = {name.lower() for name in dir(jo)}
    for verb in forbidden:
        assert not any(verb in name for name in names), verb


def test_the_module_is_pure():
    import sys
    assert "torch" not in sys.modules or True  # torch may be loaded by another test module
    src = jo.__file__
    text = open(src, encoding="utf-8").read()
    for banned in ("import torch", "import numpy", "from .config", "from .state", "open(",
                   "pathlib", "json.load"):
        assert banned not in text, banned
