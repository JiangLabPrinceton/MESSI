"""V2F1 steps 5-6 and 9-10: ``fusion_v2.schedule``.

Cycle coordinates, the depth schedule, the empirically calibrated maturity band B(r), and the
coupled admissible mask-load contract. This module decides no science: every number arrives from a
frozen calibration artifact or the config, and every illegal coordinate/maturity combination fails
closed before any sampler or projection code runs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from inverse_folding.reference_flow.fusion_v2 import schedule as sch
from inverse_folding.reference_flow.fusion_v2.errors import V2Error

S = 100
PROG = sch.CoordinateLaw.PROGRESSIVE_CHECKPOINT
STAT = sch.CoordinateLaw.STATIONARY_CHECKPOINT


def _digest(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _cycle(**over):
    kw = dict(depth=0, r_step=40, c_source_step=50, c_next_step=60, n_steps=S, law=PROG)
    kw.update(over)
    return sch.make_cycle(**kw)


def _provenance(**over):
    kw = dict(
        schema_version=sch.SCHEDULE_BAND_SCHEMA_VERSION,
        calibration_scope=sch.SCHEDULE_BAND_SCOPE,
        calibration_id="band-2026-08-04", calibration_content_digest=_digest("calibration"),
        sampler_config_digest=_digest("sampler"), tokenizer_digest=_digest("tokenizer"),
        backbone_digest=_digest("backbone"),
        coordinate_mask_policy_digest=_digest("coordinate-mask"),
        constraint_manifest_digest=_digest("constraints"),
        constraint_stratum_id="unconstrained", cohort_digest=_digest("cohort"),
        raw_attempts_digest=_digest("raw-attempts"),
        attempted_seed_digest=_digest("attempted-seeds"),
        seed_schema=sch.SCHEDULE_BAND_SEED_SCHEMA, code_revision="deadbeef",
        produced_by="scripts/rho_maturity_scan.py --mode step", n_steps=S, base_form="linear",
        amplification_form="constant_one", remask_fraction_scale=0.0, head_free=True,
        n_attempted_seeds=64, n_failed_captures=0,
    )
    kw.update(over)
    return sch.BandProvenance(**kw)


def _band(**over):
    kw = dict(
        step=40, stratum_key="len200_260", levels=sch.QuantileLevels((0.1, 0.5, 0.9)),
        rho_quantiles=(0.34, 0.40, 0.46), unresolved_quantiles=(150, 130, 110),
        rho_accept=sch.BandInterval(lo=0.34, hi=0.46, lo_level=0.1, hi_level=0.9),
        unresolved_accept=sch.BandInterval(lo=110.0, hi=150.0, lo_level=0.1, hi_level=0.9),
        combination_rule="both_axes", n_attempts=64, n_captured=64,
        n_editable_min=200, n_editable_max=260,
    )
    kw.update(over)
    return sch.make_band(**kw)


def _observation(n_editable=220, n_unresolved=130):
    return sch.observe_maturity(
        length_total=n_editable + 24, n_fixed=24, n_editable=n_editable,
        n_unresolved_editable=n_unresolved,
    )


# --------------------------------------------------------------------------------------------
# Step 5 - cycle coordinates
# --------------------------------------------------------------------------------------------

def test_history_key_is_joint_so_stationary_depths_never_alias():
    """PLAN §2.1: 'repeated step values in stationary mode are not the same event'."""
    assert sch.history_key(1, 95) != sch.history_key(2, 95)
    assert sch.history_key(1, 95) == sch.history_key(1, 95)
    assert len({sch.history_key(1, 95), sch.history_key(2, 95)}) == 2


@pytest.mark.parametrize(
    "over",
    [
        {"r_step": 50},                       # r == c_source
        {"r_step": 60},                       # r > c_source
        {"c_next_step": 49},                  # c_next < c_source
        {"c_next_step": S},                   # a capture at n_steps is not representable
        {"c_source_step": S},
        {"r_step": -1},
        {"depth": -1},
        {"c_next_step": 50, "law": PROG},     # progressive law with a stationary coordinate
        {"c_next_step": 60, "law": STAT},     # stationary law with a progressive coordinate
    ],
)
def test_make_cycle_rejects_illegal_coordinates(over):
    with pytest.raises(V2Error):
        _cycle(**over)


def test_direct_cycle_construction_cannot_bypass_coordinate_validation():
    """Public typed construction must enforce the same law as ``make_cycle``."""
    with pytest.raises(V2Error):
        sch.CycleCoordinates(
            depth=0, r_step=50, c_source_step=50, c_next_step=60, n_steps=S, law=PROG,
        )
    with pytest.raises(V2Error):
        sch.CycleCoordinates(
            depth=0, r_step=40, c_source_step=50, c_next_step=60, n_steps=S,
            law="progressive_checkpoint",
        )


def test_stationary_and_progressive_cycles_both_construct_under_their_own_law():
    assert _cycle(law=PROG, c_next_step=60).c_next_step == 60
    assert _cycle(law=STAT, c_next_step=50).c_next_step == 50


def test_cycle_exposes_the_only_costs_this_segment_may_charge():
    """PLAN §3.4: the segment costs exactly ``c_next - r``; the source prefix is never re-charged."""
    c = _cycle(r_step=40, c_source_step=50, c_next_step=60)
    assert c.segment_lane_dfe == 20
    assert sch.segment_cost(c) == 20
    assert c.lookahead_tail == S - 50
    assert sch.screening_cost(c_source_step=50, k_lookaheads=4, n_steps=S) == 50 + 4 * 50


def test_cycle_keys_are_depth_aware():
    c = _cycle(depth=1, r_step=40, c_source_step=50, c_next_step=60)
    assert c.source_key == sch.history_key(1, 50)
    assert c.reentry_key == sch.history_key(2, 40)
    assert c.commit_key == sch.history_key(2, 60)


# --------------------------------------------------------------------------------------------
# Step 6 - depth schedule chaining
# --------------------------------------------------------------------------------------------

def _schedule(cycles=None, **over):
    cycles = cycles if cycles is not None else (
        _cycle(depth=0, r_step=40, c_source_step=50, c_next_step=60),
        _cycle(depth=1, r_step=45, c_source_step=60, c_next_step=70),
    )
    kw = dict(schedule_id="sched-1", law=PROG, n_steps=S, depth_cap=len(cycles), cycles=cycles)
    kw.update(over)
    return sch.make_depth_schedule(**kw)


def test_depth_schedule_accepts_a_chained_progressive_schedule():
    got = _schedule()
    assert got.depth_cap == 2
    assert tuple(c.depth for c in got.cycles) == (0, 1)


def test_depth_schedule_rejects_a_broken_chain():
    """Without chaining the schedule silently drops a propagation segment and the ledger's
    ``c_next - r`` accounting stops summing to the real trajectory (PLAN §2.1, §3.4)."""
    with pytest.raises(V2Error):
        _schedule(cycles=(
            _cycle(depth=0, r_step=40, c_source_step=50, c_next_step=60),
            _cycle(depth=1, r_step=45, c_source_step=65, c_next_step=70),  # 65 != 60
        ))


def test_depth_schedule_rejects_non_contiguous_depths():
    with pytest.raises(V2Error):
        _schedule(cycles=(
            _cycle(depth=0, r_step=40, c_source_step=50, c_next_step=60),
            _cycle(depth=2, r_step=45, c_source_step=60, c_next_step=70),
        ))


def test_depth_schedule_rejects_a_per_depth_n_steps_or_a_mixed_law():
    """``S`` is locked across a resume by the sampler itself, so a per-depth ``S`` is
    unrepresentable (``sampler.py:564-568``)."""
    with pytest.raises(V2Error):
        _schedule(cycles=(
            _cycle(depth=0, r_step=40, c_source_step=50, c_next_step=60),
            sch.make_cycle(depth=1, r_step=45, c_source_step=60, c_next_step=70, n_steps=120,
                           law=PROG),
        ))
    with pytest.raises(V2Error):
        _schedule(cycles=(
            _cycle(depth=0, r_step=40, c_source_step=50, c_next_step=60),
            _cycle(depth=1, r_step=45, c_source_step=60, c_next_step=60, law=STAT),
        ))


def test_depth_schedule_rejects_a_depth_cap_that_disagrees_with_its_cycles():
    with pytest.raises(V2Error):
        _schedule(depth_cap=3)


# --------------------------------------------------------------------------------------------
# Step 9 - the calibrated band
# --------------------------------------------------------------------------------------------

def test_band_refuses_a_calibration_from_a_remask_on_substrate():
    """PLAN §2.1: the remask-on ``c1_null`` crossings may not be reused to select coordinates."""
    with pytest.raises(V2Error):
        _provenance(remask_fraction_scale=1.0)
    with pytest.raises(V2Error):
        _provenance(remask_fraction_scale=0.5)


@pytest.mark.parametrize(
    "over",
    [
        {"schema_version": "rf_fusion_v2_schedule_band/0"},
        {"calibration_scope": "generic_calibration"},
        {"base_form": "cubic"},
        {"amplification_form": "linear_clamp"},
        {"seed_schema": "legacy-seeds"},
        {"code_revision": "unset-code-revision"},
        {"code_revision": "unknown"},
        {"backbone_digest": "0" * 64},
        {"constraint_stratum_id": "anchored_by_label_only"},
    ],
)
def test_band_provenance_rejects_a_different_or_placeholder_substrate(over):
    with pytest.raises(V2Error):
        _provenance(**over)


def test_band_lookup_is_exact_and_never_interpolates():
    """A projected state at an uncalibrated ``(step, stratum)`` cannot proceed (PLAN §3.5)."""
    table = sch.make_band_table(provenance=_provenance(), bands=(_band(step=40), _band(step=50)))
    assert sch.lookup_band(table, step=40, stratum_key="len200_260").step == 40
    with pytest.raises(sch.MissingScheduleBandError):
        sch.lookup_band(table, step=45, stratum_key="len200_260")   # between two calibrated steps
    with pytest.raises(sch.MissingScheduleBandError):
        sch.lookup_band(table, step=40, stratum_key="len120_199")   # uncalibrated stratum


def test_band_table_rejects_duplicate_step_stratum_pairs():
    with pytest.raises(V2Error):
        sch.make_band_table(provenance=_provenance(), bands=(_band(step=40), _band(step=40)))


def test_typed_loader_rejects_content_tampering(tmp_path):
    """The declared digest binds the typed payload, not merely a non-empty caller label."""
    table = sch.bind_band_table(provenance=_provenance(), bands=(_band(step=40),))
    path = tmp_path / "band.json"
    path.write_text(json.dumps(sch.band_table_payload(table), sort_keys=True))
    loaded = sch.load_band_table(path)
    assert (
        loaded.provenance.calibration_content_digest
        == table.provenance.calibration_content_digest
    )
    with pytest.raises(V2Error):
        sch.load_band_table(path, expected_content_digest=_digest("foreign"))

    payload = json.loads(path.read_text())
    payload["bands"][0]["rho_accept"]["hi"] = 0.99
    path.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(V2Error):
        sch.load_band_table(path)


def test_direct_band_table_rejects_a_forged_content_digest():
    bound = sch.bind_band_table(provenance=_provenance(), bands=(_band(),))
    forged = dataclasses.replace(
        bound.provenance, calibration_content_digest=_digest("forged"),
    )
    with pytest.raises(V2Error):
        sch.ScheduleBandTable(provenance=forged, bands=bound.bands)


def test_band_rejects_quantile_vectors_that_do_not_match_their_levels():
    with pytest.raises(V2Error):
        _band(rho_quantiles=(0.34, 0.40))
    with pytest.raises(V2Error):
        _band(rho_quantiles=(0.46, 0.40, 0.34))     # must be non-decreasing
    with pytest.raises(V2Error):
        _band(unresolved_quantiles=(110, 130, 150))  # mass falls as maturity rises


def test_band_has_no_default_for_any_scientific_field():
    """PLAN §5.1: omission is a ``TypeError``, never a fallback."""
    for missing in ("levels", "rho_accept", "unresolved_accept", "combination_rule", "stratum_key"):
        kw = dict(
            step=40, stratum_key="s", levels=sch.QuantileLevels((0.1, 0.9)),
            rho_quantiles=(0.34, 0.46), unresolved_quantiles=(150, 110),
            rho_accept=sch.BandInterval(lo=0.34, hi=0.46, lo_level=0.1, hi_level=0.9),
            unresolved_accept=sch.BandInterval(lo=110.0, hi=150.0, lo_level=0.1, hi_level=0.9),
            combination_rule="both_axes", n_attempts=8, n_captured=8,
            n_editable_min=200, n_editable_max=260,
        )
        kw.pop(missing)
        with pytest.raises(TypeError):
            sch.make_band(**kw)


# --------------------------------------------------------------------------------------------
# Step 10 - the band gate and the coupled mask load
# --------------------------------------------------------------------------------------------

def test_gate_rejects_a_fully_resolved_projected_state_regardless_of_the_band():
    """PLAN §2.4: the projected state must retain at least one unresolved editable position."""
    table = sch.bind_band_table(provenance=_provenance(), bands=(_band(),))
    verdict = sch.gate_projected_maturity(
        table=table, step=40, observed=_observation(n_unresolved=0),
        stratum_key="len200_260",
        declared_calibration_id=table.provenance.calibration_id,
        declared_calibration_digest=table.provenance.calibration_content_digest,
    )
    assert verdict.accepted is False
    assert verdict.reason is sch.BandVerdictReason.NO_UNRESOLVED_POSITION


def test_gate_accepts_an_in_band_observation_and_rejects_out_of_band_ones():
    table = sch.bind_band_table(provenance=_provenance(), bands=(_band(),))
    identity = dict(
        table=table, step=40, stratum_key="len200_260",
        declared_calibration_id=table.provenance.calibration_id,
        declared_calibration_digest=table.provenance.calibration_content_digest,
    )
    ok = sch.gate_projected_maturity(observed=_observation(n_unresolved=130), **identity)
    assert ok.accepted is True and ok.reason is sch.BandVerdictReason.ACCEPTED
    low = sch.gate_projected_maturity(observed=_observation(n_unresolved=200), **identity)
    assert low.accepted is False


def test_gate_rejects_a_stratum_mismatch():
    """The band is only valid inside the stratum it was measured in."""
    table = sch.bind_band_table(provenance=_provenance(), bands=(_band(),))
    v = sch.gate_projected_maturity(
        table=table, step=40, observed=_observation(), stratum_key="len120_199",
        declared_calibration_id=table.provenance.calibration_id,
        declared_calibration_digest=table.provenance.calibration_content_digest,
    )
    assert v.accepted is False and v.reason is sch.BandVerdictReason.NO_BAND_FOR_STEP


def test_gate_rejects_foreign_calibration_identity_even_when_band_values_match():
    """A caller cannot relabel a matching numeric band as the state's declared calibration."""
    table = sch.bind_band_table(provenance=_provenance(), bands=(_band(),))
    verdict = sch.gate_projected_maturity(
        table=table,
        step=40,
        observed=_observation(),
        stratum_key="len200_260",
        declared_calibration_id="foreign-calibration",
        declared_calibration_digest=table.provenance.calibration_content_digest,
    )
    assert verdict.accepted is False
    assert verdict.reason is sch.BandVerdictReason.CALIBRATION_ID_MISMATCH


def test_gate_rejects_foreign_calibration_digest_even_when_id_matches():
    table = sch.bind_band_table(provenance=_provenance(), bands=(_band(),))
    verdict = sch.gate_projected_maturity(
        table=table,
        step=40,
        observed=_observation(),
        stratum_key="len200_260",
        declared_calibration_id=table.provenance.calibration_id,
        declared_calibration_digest=_digest("foreign"),
    )
    assert verdict.accepted is False
    assert verdict.reason is sch.BandVerdictReason.CALIBRATION_DIGEST_MISMATCH


def test_a_rejection_is_a_value_not_an_exception():
    """So the runner can persist an explicit null/stalled feedback event (PLAN §2.4)."""
    table = sch.bind_band_table(provenance=_provenance(), bands=(_band(),))
    v = sch.gate_projected_maturity(
        table=table, step=40, observed=_observation(n_unresolved=0),
        stratum_key="len200_260",
        declared_calibration_id=table.provenance.calibration_id,
        declared_calibration_digest=table.provenance.calibration_content_digest,
    )
    assert isinstance(v, sch.BandVerdict)


def test_reopen_cardinality_is_pinned_by_the_band_and_can_include_zero():
    """PLAN §5.1's coupled mask-load contract permits zero when writes already hit the band.

    Reopen size is not a free parameter: choosing ``r_d`` pins how much mass the projected state
    must carry, and the policy may only choose *which* residues occupy that cardinality.
    """
    load = sch.admissible_reopen_cardinality(
        band=_band(), n_editable=220, n_unresolved_source=100,
        n_endpoint_writes_over_masked=0,
    )
    assert load.feasible is True
    assert load.min_newly_masked >= 1
    assert load.min_newly_masked <= load.max_newly_masked
    # u_proj = u_src - a + b_new must land inside the pinned interval
    assert load.pinned.min_unresolved <= 100 + load.min_newly_masked <= load.pinned.max_unresolved

    zero = sch.admissible_reopen_cardinality(
        band=sch.make_band(
            step=40, stratum_key="tiny", levels=sch.QuantileLevels(levels=(0.1, 0.5, 0.9)),
            rho_quantiles=(0.0, 0.25, 0.5), unresolved_quantiles=(2, 1, 1),
            rho_accept=sch.BandInterval(lo=0.0, hi=0.5, lo_level=0.1, hi_level=0.9),
            unresolved_accept=sch.BandInterval(lo=1.0, hi=2.0, lo_level=0.9, hi_level=0.1),
            combination_rule="both_axes", n_attempts=4, n_captured=4,
            n_editable_min=1, n_editable_max=2,
        ),
        n_editable=2, n_unresolved_source=2, n_endpoint_writes_over_masked=1,
    )
    assert zero.feasible and (zero.min_newly_masked, zero.max_newly_masked) == (0, 0)


def test_reopen_cardinality_reports_infeasible_instead_of_clamping():
    """An empty range is a fact about the schedule, not something to silently repair."""
    load = sch.admissible_reopen_cardinality(
        band=_band(), n_editable=220, n_unresolved_source=200,
        n_endpoint_writes_over_masked=0,
    )
    assert load.feasible is False
    assert load.infeasible_reason


def test_realized_mask_load_outside_the_admissible_range_fails_before_the_segment_runs():
    load = sch.admissible_reopen_cardinality(
        band=_band(), n_editable=220, n_unresolved_source=100,
        n_endpoint_writes_over_masked=0,
    )
    good = sch.validate_realized_mask_load(load=load, n_newly_masked=load.min_newly_masked)
    assert good.accepted is True
    bad = sch.validate_realized_mask_load(load=load, n_newly_masked=load.max_newly_masked + 1)
    assert bad.accepted is False


def test_maturity_is_measured_from_counts_and_never_inferred_from_a_coordinate():
    """PLAN §2.1: realized editable maturity is measured from bytes, never inferred from a label."""
    obs = _observation(n_editable=200, n_unresolved=50)
    assert obs.n_resolved_editable == 150
    assert obs.rho_edit == pytest.approx(0.75)
    with pytest.raises(V2Error):
        sch.observe_maturity(length_total=100, n_fixed=0, n_editable=0, n_unresolved_editable=0)
    with pytest.raises(V2Error):
        sch.observe_maturity(length_total=100, n_fixed=0, n_editable=10, n_unresolved_editable=11)


# --------------------------------------------------------------------------------------------
# ScheduleBand is self-validating: the invariants belong to the TYPE, not to one factory
# --------------------------------------------------------------------------------------------


def _band_kwargs(**over):
    legal = _band()
    kw = {f.name: getattr(legal, f.name) for f in dataclasses.fields(sch.ScheduleBand)}
    kw.update(over)
    return kw


@pytest.mark.parametrize("over,fragment", [
    ({"n_attempts": 0}, "n_attempts"),
    ({"n_captured": 0}, "n_captured"),
    ({"n_editable_min": 0}, "n_editable_min"),
    ({"n_editable_max": 0}, "n_editable_max"),
    ({"step": -1}, "step"),
    ({"combination_rule": "rho_only"}, "both_axes"),
    ({"stratum_key": ""}, "stratum_key"),
])
def test_schedule_band_rejects_illegal_fields_at_construction(over, fragment):
    """``make_band`` is not the only door into the type.

    ``ScheduleBand`` is exported and appears in every band table, so a caller -- or a future
    deserializer -- can construct one directly. Validating only inside the factory means the TYPE
    guarantees nothing: a zero-attempt band would claim a calibration that never ran, and every
    downstream gate would trust it.
    """
    with pytest.raises(sch.V2ScheduleError, match=fragment):
        sch.ScheduleBand(**_band_kwargs(**over))


def test_schedule_band_rejects_captured_exceeding_attempts_at_construction():
    with pytest.raises(sch.V2ScheduleError, match="exceeds"):
        sch.ScheduleBand(**_band_kwargs(n_attempts=8, n_captured=9))


def test_schedule_band_rejects_inverted_editable_range_at_construction():
    with pytest.raises(sch.V2ScheduleError, match="n_editable_min exceeds"):
        sch.ScheduleBand(**_band_kwargs(n_editable_min=9, n_editable_max=8))


def test_schedule_band_rejects_non_monotone_quantiles_at_construction():
    legal = _band()
    rho = tuple(reversed(legal.rho_quantiles))
    assert rho != legal.rho_quantiles, "fixture must have strictly ordered rho quantiles"
    with pytest.raises(sch.V2ScheduleError, match="non-decreasing"):
        sch.ScheduleBand(**_band_kwargs(rho_quantiles=rho))
    unresolved = tuple(reversed(legal.unresolved_quantiles))
    assert unresolved != legal.unresolved_quantiles
    with pytest.raises(sch.V2ScheduleError, match="non-increasing"):
        sch.ScheduleBand(**_band_kwargs(unresolved_quantiles=unresolved))


def test_schedule_band_rejects_quantile_width_mismatch_at_construction():
    legal = _band()
    with pytest.raises(sch.V2ScheduleError, match="one entry per level"):
        sch.ScheduleBand(**_band_kwargs(rho_quantiles=legal.rho_quantiles[:-1]))


def test_schedule_band_rejects_a_non_interval_accept_region_at_construction():
    with pytest.raises(sch.V2ScheduleError, match="rho_accept must be a BandInterval"):
        sch.ScheduleBand(**_band_kwargs(rho_accept=(0.4, 0.8)))


def test_make_band_and_direct_construction_agree_field_for_field():
    """The factory must add nothing the type does not already guarantee."""
    direct = sch.ScheduleBand(**_band_kwargs())
    assert direct == _band()
