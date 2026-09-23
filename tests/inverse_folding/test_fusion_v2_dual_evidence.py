"""DUALF2: two frozen Head results bound to ONE exact endpoint, before any Dual decision.

The load-bearing rule is compatibility (PLAN §0.3): the legacy ``CompleteEndpoint`` keeps its
meaning byte for byte. Role A -- the existing production Head -- populates ``head_global_risk``, and
everything Dual lives in an endpoint-id-bound sidecar. A joint scalar written into the legacy field
would make every existing artifact, reader and regression silently mean something else.
"""

from __future__ import annotations

import dataclasses

import pytest

from inverse_folding.reference_flow.fusion_v2 import dual_evidence as de
from inverse_folding.reference_flow.fusion_v2 import identity as ident
from inverse_folding.reference_flow.fusion_v2 import joint_objective as jo

from . import _v2_fixtures as fx


# ------------------------------------------------------------------------------------------
# fixtures: one toy world, two distinguishable Heads on one window grid
# ------------------------------------------------------------------------------------------

EVAL_A = fx.endpoint().head_binding.evaluator
EVAL_B = dataclasses.replace(
    EVAL_A, allele="DRB1_0401", head_checkpoint_digest=fx.digest("head-b-checkpoint"))


def coordinate(role, evaluator, *, quantity="global_risk", location=0.0, scale=2.0):
    return jo.AlleleCoordinate(
        role=role, quantity=quantity, location=location, scale=scale,
        raw_noise_floor=0.01, evaluator=evaluator, source_ref=fx.digest("panel"))


def calibration(*, density=False):
    risk = jo.QuantityCoordinates(
        quantity="global_risk", a=coordinate(jo.AlleleRole.A, EVAL_A),
        b=coordinate(jo.AlleleRole.B, EVAL_B))
    density_pair = None
    if density:
        density_pair = jo.QuantityCoordinates(
            quantity="positive_mass_density",
            a=coordinate(jo.AlleleRole.A, EVAL_A, quantity="positive_mass_density",
                         location=0.1, scale=0.05),
            b=coordinate(jo.AlleleRole.B, EVAL_B, quantity="positive_mass_density",
                         location=0.2, scale=0.10))
    calibration = jo.DualCalibration(
        panel=jo.PanelBinding(
            panel_id="tier2_natural_v1", panel_digest=fx.digest("panel"), n_proteins=53184,
            overlap_fraction_a=0.041, overlap_fraction_b=0.049,
            equal_risk_line_stderr=0.028, leave_overlap_out_shift=0.004,
            cross_allele_pearson=0.19),
        risk=risk, density=density_pair,
        law=jo.DualObjectiveLaw(
            mode=jo.ObjectiveMode.SMOOTH_MAX, tau=0.15, tau_units="normalized",
            version="dual-obj-1"),
        version="dual-cal-1")
    return calibration


def objective(*, quantity="global_risk", density=False, cal=None):
    """A view over ONE frozen calibration. Both quantity views must share it, so a run cannot
    resolve risk under one coordinate system and density under another."""
    return jo.DualObjective(cal if cal is not None else calibration(density=density),
                            quantity=quantity)


def endpoint(**over):
    return fx.endpoint(**over)


def score_b(ep, *, risk=-4.0, density=None, **over):
    """A role-B Head result describing the SAME exact sequence as ``ep``."""
    fields = dict(
        protein_id=ep.protein_id, sequence_md5=ep.sequence_md5,
        sequence_length=ep.sequence_length, allele=EVAL_B.allele,
        score_scale=EVAL_B.score_scale, windows=ep.head_score.windows,
        residue_hotspot=ep.head_score.residue_hotspot, global_risk=risk)
    fields.update(over)
    score = fx.EndpointHeadScore(**fields)
    binding = ident.HeadScoreBinding(
        protein_id=score.protein_id, sequence_md5=score.sequence_md5,
        sequence_length=score.sequence_length,
        window_grid_digest=ep.head_binding.window_grid_digest, evaluator=EVAL_B)
    return de.HeadResult(score=score, binding=binding, density=density)


def result_a(ep, *, density=None):
    return de.HeadResult(score=ep.head_score, binding=ep.head_binding, density=density)


# ------------------------------------------------------------------------------------------
# 1. the legacy endpoint keeps its meaning
# ------------------------------------------------------------------------------------------

def test_the_legacy_endpoint_is_untouched_and_role_A_populates_its_head_fields():
    ep = endpoint()
    evidence = de.bind_dual_evidence(
        endpoint=ep, result_a=result_a(ep), result_b=score_b(ep), objective=objective())
    assert evidence.a.raw_risk == pytest.approx(ep.head_global_risk)
    assert evidence.a.allele == EVAL_A.allele
    assert evidence.b.allele == EVAL_B.allele
    # the sidecar is keyed to the endpoint, and the endpoint itself is not modified
    assert evidence.endpoint_id == ep.endpoint_id
    assert ep.head_global_risk == pytest.approx(-9.1)


def test_a_role_A_result_that_disagrees_with_the_legacy_head_risk_is_refused():
    # This is the guard against a joint scalar having been written into head_global_risk upstream.
    ep = endpoint()
    moved = dataclasses.replace(ep.head_score, global_risk=-1.0)
    with pytest.raises(de.V2DualEvidenceError):
        de.bind_dual_evidence(
            endpoint=ep, result_a=de.HeadResult(score=moved, binding=ep.head_binding),
            result_b=score_b(ep), objective=objective())


def test_role_A_must_be_the_evaluator_the_endpoint_was_actually_scored_by():
    ep = endpoint()
    with pytest.raises(de.V2DualEvidenceError):
        # role A calibrated for a Head that did not produce this endpoint
        de.bind_dual_evidence(
            endpoint=ep, result_a=score_b(ep, risk=ep.head_global_risk), result_b=score_b(ep),
            objective=objective())


# ------------------------------------------------------------------------------------------
# 2. one exact sequence, two Heads
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("sequence_md5", "f" * 32),
    ("sequence_length", 99),
    ("protein_id", "Q00511"),
])
def test_the_two_heads_must_describe_the_same_exact_sequence(field, value):
    ep = endpoint()
    bad = score_b(ep, **{field: value})
    with pytest.raises(de.V2DualEvidenceError):
        de.bind_dual_evidence(endpoint=ep, result_a=result_a(ep), result_b=bad,
                              objective=objective())


def test_the_two_heads_must_share_the_window_grid():
    ep = endpoint()
    other = score_b(ep)
    other = de.HeadResult(
        score=other.score,
        binding=dataclasses.replace(other.binding,
                                    window_grid_digest=fx.digest("another-grid")),
        density=None)
    with pytest.raises(de.V2DualEvidenceError):
        de.bind_dual_evidence(endpoint=ep, result_a=result_a(ep), result_b=other,
                              objective=objective())


def test_binding_the_same_head_to_both_roles_is_refused_at_evidence_time_too():
    ep = endpoint()
    with pytest.raises(de.V2DualEvidenceError):
        de.bind_dual_evidence(endpoint=ep, result_a=result_a(ep), result_b=result_a(ep),
                              objective=objective())


def test_a_missing_second_head_result_is_refused_rather_than_defaulted():
    ep = endpoint()
    with pytest.raises(de.V2DualEvidenceError):
        de.bind_dual_evidence(endpoint=ep, result_a=result_a(ep), result_b=None,
                              objective=objective())


# ------------------------------------------------------------------------------------------
# 3. the objective travels with the evidence
# ------------------------------------------------------------------------------------------

def test_the_joint_value_is_resolved_from_the_two_raw_risks_through_the_frozen_coordinates():
    ep = endpoint()
    obj = objective()
    evidence = de.bind_dual_evidence(
        endpoint=ep, result_a=result_a(ep), result_b=score_b(ep, risk=-4.0), objective=obj)
    expected = obj.evaluate(raw_a=ep.head_global_risk, raw_b=-4.0)
    assert evidence.risk.value == pytest.approx(expected.value)
    assert evidence.risk.u_a == pytest.approx(expected.u_a)
    assert evidence.risk.objective_digest == obj.objective_digest
    assert evidence.calibration_digest == obj.calibration.content_digest


def test_the_density_view_is_bound_only_when_both_results_report_it():
    ep = endpoint()
    shared = calibration(density=True)
    obj = objective(cal=shared)
    density_obj = objective(quantity="positive_mass_density", cal=shared)
    evidence = de.bind_dual_evidence(
        endpoint=ep, result_a=result_a(ep, density=0.15), result_b=score_b(ep, density=0.10),
        objective=obj, density_objective=density_obj)
    assert evidence.density is not None
    assert evidence.density.u_a == pytest.approx((0.15 - 0.1) / 0.05)
    # and a half-reported density is a contract failure, not a silent None
    with pytest.raises(de.V2DualEvidenceError):
        de.bind_dual_evidence(
            endpoint=ep, result_a=result_a(ep, density=0.15), result_b=score_b(ep, density=None),
            objective=obj, density_objective=density_obj)


def test_without_a_density_objective_no_density_view_is_produced():
    ep = endpoint()
    evidence = de.bind_dual_evidence(
        endpoint=ep, result_a=result_a(ep, density=0.15), result_b=score_b(ep, density=0.10),
        objective=objective())
    assert evidence.density is None


# ------------------------------------------------------------------------------------------
# 4. the batch binder
# ------------------------------------------------------------------------------------------

def two_endpoints():
    first = endpoint()
    second = endpoint(tokens=(11, 20, 12, 13, 14, 15))
    assert first.sequence_md5 != second.sequence_md5
    return first, second


def test_the_batch_binds_by_identity_and_is_row_order_independent():
    first, second = two_endpoints()
    obj = objective()
    forward = de.bind_dual_evidence_batch(
        endpoints=(first, second),
        results_a=(result_a(first), result_a(second)),
        results_b=(score_b(first, risk=-4.0), score_b(second, risk=-5.0)),
        objective=obj)
    reversed_b = de.bind_dual_evidence_batch(
        endpoints=(first, second),
        results_a=(result_a(second), result_a(first)),
        results_b=(score_b(second, risk=-5.0), score_b(first, risk=-4.0)),
        objective=obj)
    assert [e.content_digest for e in forward] == [e.content_digest for e in reversed_b]


def test_a_partial_paired_batch_never_reaches_selection():
    first, second = two_endpoints()
    with pytest.raises(de.V2DualEvidenceError):
        de.bind_dual_evidence_batch(
            endpoints=(first, second),
            results_a=(result_a(first), result_a(second)),
            results_b=(score_b(first),),          # role B dropped one
            objective=objective())


def test_a_duplicate_result_for_one_sequence_within_one_allele_is_refused():
    first, second = two_endpoints()
    with pytest.raises(de.V2DualEvidenceError):
        de.bind_dual_evidence_batch(
            endpoints=(first, second),
            results_a=(result_a(first), result_a(second)),
            results_b=(score_b(first), score_b(first), score_b(second)),
            objective=objective())


def test_an_unexpected_result_not_corresponding_to_any_endpoint_is_refused():
    first, second = two_endpoints()
    stray = endpoint(tokens=(12, 20, 12, 13, 14, 15))
    with pytest.raises(de.V2DualEvidenceError):
        de.bind_dual_evidence_batch(
            endpoints=(first,),
            results_a=(result_a(first),),
            results_b=(score_b(first), score_b(stray)),
            objective=objective())


def test_two_endpoints_carrying_identical_bytes_share_one_head_result_per_allele():
    # Duplicate siblings are legal; scoring one twice would be the same measurement counted twice.
    first = endpoint()
    twin = endpoint(fork_index=1)
    assert first.sequence_md5 == twin.sequence_md5
    assert first.endpoint_id != twin.endpoint_id
    bound = de.bind_dual_evidence_batch(
        endpoints=(first, twin),
        results_a=(result_a(first),),
        results_b=(score_b(first),),
        objective=objective())
    assert len(bound) == 2
    assert bound[0].risk.value == pytest.approx(bound[1].risk.value)
    assert bound[0].endpoint_id != bound[1].endpoint_id


# ------------------------------------------------------------------------------------------
# 5. identity
# ------------------------------------------------------------------------------------------

def test_the_evidence_digest_is_deterministic_and_content_bound():
    ep = endpoint()
    obj = objective()
    one = de.bind_dual_evidence(endpoint=ep, result_a=result_a(ep), result_b=score_b(ep, risk=-4.0),
                                objective=obj)
    same = de.bind_dual_evidence(endpoint=ep, result_a=result_a(ep), result_b=score_b(ep, risk=-4.0),
                                 objective=obj)
    moved = de.bind_dual_evidence(endpoint=ep, result_a=result_a(ep), result_b=score_b(ep, risk=-4.1),
                                  objective=obj)
    assert one.content_digest == same.content_digest
    assert one.content_digest != moved.content_digest


def test_the_evidence_records_which_allele_is_the_active_worst():
    ep = endpoint()
    obj = objective()
    worse_b = de.bind_dual_evidence(
        endpoint=ep, result_a=result_a(ep), result_b=score_b(ep, risk=99.0), objective=obj)
    assert worse_b.risk.active_worst is jo.AlleleRole.B
    worse_a = de.bind_dual_evidence(
        endpoint=ep, result_a=result_a(ep), result_b=score_b(ep, risk=-99.0), objective=obj)
    assert worse_a.risk.active_worst is jo.AlleleRole.A


def test_the_module_is_pure():
    text = open(de.__file__, encoding="utf-8").read()
    for banned in ("import torch", "import numpy", "from .config", "json.load"):
        assert banned not in text, banned


# ------------------------------------------------------------------------------------------
# 6. role B safety: reported, not gated
# ------------------------------------------------------------------------------------------

def test_role_B_hotspot_drift_is_measured_by_the_same_primitive_as_role_A():
    ep = endpoint()
    reference = score_b(ep, risk=-1.0).score
    telemetry = de.role_b_hotspot_telemetry(
        endpoint_id=ep.endpoint_id, design_score_b=score_b(ep).score,
        reference_score_b=reference, evaluator_b=EVAL_B,
        reference_binding_id="ref:wt")
    assert telemetry.allele == EVAL_B.allele
    assert telemetry.n_windows == len(ep.head_score.windows)
    # identical landscapes -> no new hotspot anywhere
    assert telemetry.max_increase == pytest.approx(0.0)
    assert telemetry.positive_count == 0


def test_a_worsened_role_B_window_shows_up_as_positive_drift():
    ep = endpoint()
    reference = score_b(ep).score
    worse = dataclasses.replace(
        reference,
        windows=(dataclasses.replace(reference.windows[0], z=reference.windows[0].z + 2.0),)
        + reference.windows[1:])
    telemetry = de.role_b_hotspot_telemetry(
        endpoint_id=ep.endpoint_id, design_score_b=worse, reference_score_b=reference,
        evaluator_b=EVAL_B, reference_binding_id="ref:wt")
    assert telemetry.max_increase == pytest.approx(2.0)
    assert telemetry.positive_count == 1
    assert telemetry.positive_mass == pytest.approx(2.0)


def test_the_telemetry_is_carried_on_the_evidence_and_gates_nothing():
    ep = endpoint()
    telemetry = de.role_b_hotspot_telemetry(
        endpoint_id=ep.endpoint_id, design_score_b=score_b(ep).score,
        reference_score_b=score_b(ep).score, evaluator_b=EVAL_B, reference_binding_id="ref:wt")
    with_drift = dataclasses.replace(
        de.bind_dual_evidence(endpoint=ep, result_a=result_a(ep), result_b=score_b(ep),
                              objective=objective()),
        hotspot_b=telemetry)
    assert with_drift.hotspot_b is telemetry
    assert with_drift.canonical_payload()["hotspot_b"]["allele"] == EVAL_B.allele
    # absent by default: a run that did not measure it carries no field rather than a zero
    plain = de.bind_dual_evidence(endpoint=ep, result_a=result_a(ep), result_b=score_b(ep),
                                  objective=objective())
    assert plain.hotspot_b is None
    assert plain.canonical_payload()["hotspot_b"] is None


def test_measuring_role_B_drift_does_not_make_it_an_admission_input():
    """Structural: nothing in the Dual layer reads hotspot_b as a threshold."""
    import inspect
    from inverse_folding.reference_flow.fusion_v2 import dual_policy, dual_evidence
    from inverse_folding.reference_flow.fusion_v2_runtime import dual_selection
    for module in (dual_policy, dual_evidence, dual_selection):
        src = inspect.getsource(module)
        for banned in ("delta_b", "delta_new_cumulative", "hotspot_threshold"):
            assert banned not in src.lower(), f"{module.__name__}: {banned}"
