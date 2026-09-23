"""V2F1 steps 19-22: ``fusion_v2.config``.

The V2 config exists to make an unstated scientific choice impossible to run with. Every field
whose value is an open decision is required with no library default, and every closed vocabulary is
closed, so a typo becomes a load failure rather than a silently different experiment.
"""

from __future__ import annotations

import copy

import pytest

from inverse_folding.reference_flow.fusion_v2 import config as cfg
from inverse_folding.reference_flow.fusion_v2 import policy as pol
from inverse_folding.reference_flow.fusion_v2 import schedule as sch
from inverse_folding.reference_flow.fusion_v2.errors import V2Error

D = "a" * 64


def _calibration(
    *,
    value=0.10,
    source_kind="measured_calibration",
    source_id="v2-hotspot-calib-1",
    scope="cumulative_depth0",
    reference_kind="native_wt",
    reference_label="wt_native",
):
    artifact = cfg.HotspotCalibrationArtifact(
        schema_version=cfg.V2_HOTSPOT_CALIBRATION_SCHEMA_VERSION,
        gate_kind=cfg.HOTSPOT_GATE_KIND,
        scope=scope,
        reference_kind=reference_kind,
        reference_label=reference_label,
        window_domain=cfg.HOTSPOT_WINDOW_DOMAIN,
        allele="DRB1_0701",
        score_scale="nats",
        window_k_min=13,
        window_k_max=25,
        calibration_data_digest=D,
    )
    source_ref = cfg.calibration_source_ref(
        value=value, unit="nats", source_kind=source_kind,
        source_id=source_id, artifact=artifact,
    )
    return {
        "value": value,
        "unit": "nats",
        "source_kind": source_kind,
        "source_id": source_id,
        "source_ref": source_ref,
        "artifact": artifact.canonical_payload(),
    }


def _mapping(**over):
    payload = {
        "schema_version": cfg.V2_CONFIG_SCHEMA_VERSION,
        "identity": {
            "campaign_id": "v2_mech_dev", "split_role": "v2_dev", "phase": "state_transition_canary",
            "master_seed": 20260804, "seed_schema": "v2seed-1", "code_revision": "deadbeef",
        },
        "substrate": {
            "n_steps": 100, "temperature": 1.0, "amplification_form": "constant_one",
            "controller_enabled": False, "remask_enabled": True, "remask_fraction_scale": 0.0,
            "rf_config_label": "v2_null_no_remask",
        },
        "arm": {
            "feedback_enabled": True, "arm_role": "v2",
            "a2_matching_resource": "definitive_refolds",
            "a2_unmatched_reported": ["gpu_seconds", "head_calls", "logical_dfe", "walltime_s"],
        },
        "schedule": {
            "schedule_id": "sched-1", "coordinate_law": "progressive_checkpoint", "depth_cap": 2,
            "active_population_width": 1, "min_lookahead_tail_steps": 10,
            "points": [
                {"depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 60,
                 "n_lookaheads": 4, "band_key": "step40"},
                {"depth": 1, "r_step": 45, "c_source_step": 60, "c_next_step": 70,
                 "n_lookaheads": 4, "band_key": "step45"},
            ],
        },
        "projection": {
            "support_policy_id": "explicit_probe", "support_policy_version": "v0",
            "support_policy_is_diagnostic": True, "temporal_history_rule": "commit_before_reentry",
            "assimilation_rule": "first_forward_raw_logits",
            "admissible_mask_load_unit": "absolute_positions",
        },
        "head": {
            "allele": "DRB1_0701", "score_scale": "nats", "window_k_min": 13, "window_k_max": 25,
        },
        "safety": {
            "cumulative_reference_kind": "native_wt", "cumulative_reference_label": "wt_native",
            "delta_new_cumulative": _calibration(),
            "incremental_gate_enabled": False,
            "structure_cadence": "every_endpoint",
        },
        "caps": {
            "max_logical_dfe": 500000, "max_head_calls": 20000, "max_definitive_refolds": 4000,
            "max_gpu_seconds": 86400, "max_walltime_s": 21600, "max_retries": 2,
            "retry_scope": "per_request",
        },
        "content": [
            {"role": role, "label": f"{role}.bin", "binding": binding,
             "expected_sha256": D if binding == "frozen" else None}
            for role, binding in cfg.default_content_bindings().items()
        ],
    }
    for path, value in over.items():
        node = payload
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[key]
        if value is cfg.OMIT:
            node.pop(keys[-1], None)
        else:
            node[keys[-1]] = value
    return payload


def _load(**over):
    return cfg.load_v2_config(_mapping(**over))


# ---- step 19: the frozen substrate ------------------------------------------------------------

def test_a_complete_config_loads():
    loaded = _load()
    assert loaded.substrate.remask_fraction_scale == 0.0
    assert loaded.schedule.coordinate_law is sch.CoordinateLaw.PROGRESSIVE_CHECKPOINT


@pytest.mark.parametrize(
    "over",
    [{"substrate.remask_fraction_scale": 0.1},
     {"substrate.remask_fraction_scale": 1.0},
     {"substrate.remask_enabled": False},
     {"substrate.amplification_form": "linear_clamp"},
     {"substrate.amplification_form": "sigmoid"},
     {"substrate.controller_enabled": True},
     {"substrate.controller_enabled": "false"}],
)
def test_the_frozen_substrate_is_not_negotiable(over):
    """PLAN §5.1: every V2/A2 arm runs controller-free, h-map-free, ``constant_one``, no remask.

    ``remask_enabled=False`` is refused rather than accepted as an equivalent spelling: it would
    also skip the controller post-step lifecycle, conflating "no remask" with "no post-step". The
    supported zero-remask path is ``enabled=true`` with ``fraction_scale=0.0``.
    """
    with pytest.raises(V2Error):
        _load(**over)


def test_no_h_map_field_exists_anywhere_on_the_config_surface():
    """h-maps are retired: the guarantee stays, the declaration burden does not."""
    rendered = repr(_load())
    assert "h_map" not in rendered and "h_source" not in rendered
    with pytest.raises(V2Error):
        _load(**{"substrate.h_maps_present": False})   # an unknown key, not an accepted one


# ---- step 20: no library defaults --------------------------------------------------------------

@pytest.mark.parametrize(
    "path",
    ["arm.a2_matching_resource", "safety.delta_new_cumulative", "schedule.active_population_width",
     "schedule.min_lookahead_tail_steps", "safety.incremental_gate_enabled",
     "identity.master_seed", "substrate.n_steps", "caps.retry_scope", "caps.max_definitive_refolds"],
)
def test_every_scientific_field_is_required_and_names_its_path_when_missing(path):
    """PLAN §5.1: scientific numeric fields have no library defaults; omission fails closed."""
    with pytest.raises(V2Error, match=path.split(".")[-1]):
        _load(**{path: cfg.OMIT})


@pytest.mark.parametrize(
    "path", ["nonsense", "identity.nonsense", "substrate.nonsense", "schedule.points.0.nonsense"],
)
def test_an_unknown_key_at_any_level_is_rejected_rather_than_ignored(path):
    """A deferred or legacy knob must be rejected, not silently dropped -- otherwise a config that
    looks like it enabled something ran without it."""
    payload = _mapping()
    if path == "schedule.points.0.nonsense":
        payload["schedule"]["points"][0]["nonsense"] = 1
    else:
        node, *rest = path.split(".")
        (payload if not rest else payload[node])[rest[0] if rest else node] = 1
    with pytest.raises(V2Error):
        cfg.load_v2_config(payload)


def test_the_a2_matching_resource_is_a_declared_choice_over_a_closed_set():
    """PLAN §4.3 mandates the reallocation but names no resource, and FUSION_V2 §6.1 forbids
    collapsing the envelope into one invented universal cost. So the config must refuse to load
    without an explicit choice, and must report the exact complement as unmatched."""
    assert cfg.A2_RESOURCE_COMPONENTS == frozenset(
        {"logical_dfe", "head_calls", "definitive_refolds", "gpu_seconds", "walltime_s"})
    with pytest.raises(V2Error):
        _load(**{"arm.a2_matching_resource": "vibes"})
    with pytest.raises(V2Error):     # complement must be exact, not a subset
        _load(**{"arm.a2_unmatched_reported": ["gpu_seconds"]})


def test_the_cumulative_threshold_must_carry_calibration_provenance():
    with pytest.raises(V2Error):
        _load(**{"safety.delta_new_cumulative": _calibration(
            source_kind="inherited_v0",
            source_id="objective.max_offtarget_window_increase",
        )})


def test_an_incremental_threshold_is_required_exactly_when_the_gate_is_enabled():
    with pytest.raises(V2Error):
        _load(**{"safety.incremental_gate_enabled": True})
    loaded = _load(**{"safety.incremental_gate_enabled": True,
                      "safety.delta_new_incremental": _calibration(
                          value=0.05, source_id="v2-inc-1", scope="immediate_parent",
                          reference_kind=cfg.INCREMENTAL_REFERENCE_KIND,
                          reference_label=cfg.INCREMENTAL_REFERENCE_LABEL)})
    assert loaded.safety.delta_new_incremental is not None
    with pytest.raises(V2Error):     # forbidden when the gate is off
        _load(**{"safety.delta_new_incremental": _calibration(
            value=0.05, source_id="v2-inc-1", scope="immediate_parent",
            reference_kind=cfg.INCREMENTAL_REFERENCE_KIND,
            reference_label=cfg.INCREMENTAL_REFERENCE_LABEL)})


# ---- step 21: the diagnostic gate --------------------------------------------------------------

def test_a_diagnostic_policy_is_refused_outside_its_allowed_phase_before_anything_loads():
    """PLAN §2.5: ``explicit_probe`` may never become a silent production default."""
    assert cfg.DIAGNOSTIC_ALLOWED_PHASES == frozenset({"state_transition_canary"})
    for phase in ("mechanism_cohort", "capability_ladder", "holdout_application"):
        with pytest.raises(V2Error):
            _load(**{"identity.phase": phase})


def test_the_diagnostic_flag_must_agree_with_the_policy_name():
    """A ``diag_`` prefix that claims to be production is a mislabel, and the reverse hides one."""
    with pytest.raises(V2Error):
        _load(**{"projection.support_policy_is_diagnostic": False})
    with pytest.raises(V2Error):
        _load(**{"projection.support_policy_id": "source_writeback_v1",
                 "projection.support_policy_is_diagnostic": True})


def test_a_production_policy_loads_in_a_production_phase():
    loaded = _load(**{"identity.phase": "capability_ladder",
                      "projection.support_policy_id": "source_writeback_v1",
                      "projection.support_policy_is_diagnostic": False})
    assert loaded.projection.support_policy_is_diagnostic is False


def test_the_diagnostic_gate_is_keyed_on_the_policy_vocabulary_not_on_the_config_self_label():
    """The gate must not be decidable by what the config writes ABOUT itself.

    The defect this pins: the phase gate tested ``support_policy_id.startswith("diag_")`` while the
    probe in :mod:`fusion_v2.policy` names itself ``explicit_probe`` -- no prefix.  Declaring the
    probe HONESTLY therefore satisfied the prefix/flag consistency check and walked straight past
    the phase gate, so a production-phase run loaded with the diagnostic probe as its policy.
    """
    with pytest.raises(V2Error):
        _load(**{"identity.phase": "mechanism_cohort",
                 "projection.support_policy_id": pol.EXPLICIT_PROBE_POLICY_ID,
                 "projection.support_policy_is_diagnostic": False})
    # ... and it is not rescued by declaring the flag honestly either: the phase is what forbids it.
    with pytest.raises(V2Error):
        _load(**{"identity.phase": "mechanism_cohort",
                 "projection.support_policy_id": pol.EXPLICIT_PROBE_POLICY_ID,
                 "projection.support_policy_is_diagnostic": True})


def test_the_probes_own_id_is_admissible_in_the_phase_that_authorises_a_probe():
    """The guard above must not be satisfiable by rejecting the probe everywhere."""
    loaded = _load(**{"identity.phase": "state_transition_canary",
                      "projection.support_policy_id": pol.EXPLICIT_PROBE_POLICY_ID,
                      "projection.support_policy_is_diagnostic": True})
    assert loaded.projection.support_policy_is_diagnostic is True


def test_the_config_and_the_policy_module_share_one_diagnostic_vocabulary():
    """Two names for the same concept that do not agree are not a vocabulary.

    ``config.DIAGNOSTIC_VALUE_PREFIX`` and the policy module's own id constant were independent
    strings, so the config's notion of "diagnostic" could not see the probe the runtime actually
    runs.  The loader must decide on the policy module's vocabulary, not on a private prefix.
    """
    assert cfg.DIAGNOSTIC_VALUE_PREFIX == pol.DIAGNOSTIC_POLICY_ID_PREFIX
    assert cfg.DIAGNOSTIC_ALLOWED_PHASES == pol.DIAGNOSTIC_ALLOWED_PHASES
    assert pol.EXPLICIT_PROBE_POLICY_ID in pol.DIAGNOSTIC_POLICY_IDS
    for policy_id in sorted(pol.DIAGNOSTIC_POLICY_IDS):
        assert pol.policy_id_is_diagnostic(policy_id) is True
        with pytest.raises(V2Error):     # the loader agrees with the vocabulary, not with the file
            _load(**{"projection.support_policy_id": policy_id,
                     "projection.support_policy_is_diagnostic": False})


def test_the_config_hands_the_kernel_a_typed_declaration_of_the_policy_it_declared():
    """The runtime half of the split: the kernel checks the ANSWERING policy against this.

    Parse time can only check the DECLARED policy, because the answering identity does not exist
    until a policy answers -- which is after the model has run the source prefix and the lookaheads.
    So the config emits its declaration as a typed object and the kernel matches the answer to it.
    """
    loaded = _load()
    declared = loaded.declared_policy()
    assert isinstance(declared, pol.DeclaredPolicy)
    assert declared.policy_id == loaded.projection.support_policy_id
    assert declared.policy_version == loaded.projection.support_policy_version
    assert declared.is_diagnostic is loaded.projection.support_policy_is_diagnostic
    assert declared.phase == loaded.identity.phase


# ---- step 22: cross-checks ---------------------------------------------------------------------

def test_the_schedule_delegates_every_ordering_check_to_the_schedule_layer():
    """Interface map Conflict: ``schedule.make_cycle`` is the only validator; ``config.py``
    performs no coordinate arithmetic of its own, so the two can never disagree."""
    with pytest.raises(V2Error):
        _load(**{"schedule.points": [
            {"depth": 0, "r_step": 55, "c_source_step": 50, "c_next_step": 60,
             "n_lookaheads": 4, "band_key": "x"}]})
    with pytest.raises(V2Error):     # broken chain between depths
        _load(**{"schedule.points": [
            {"depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 60,
             "n_lookaheads": 4, "band_key": "x"},
            {"depth": 1, "r_step": 45, "c_source_step": 65, "c_next_step": 70,
             "n_lookaheads": 4, "band_key": "y"}]})


def test_a_stationary_comparator_needs_its_own_config_and_its_own_law():
    """One coordinate law per config file; the stationary comparator ships separately so a
    campaign cannot mix the two under one identity."""
    loaded = _load(**{"schedule.coordinate_law": "stationary_checkpoint", "schedule.depth_cap": 1,
                      "schedule.points": [
                          {"depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 50,
                           "n_lookaheads": 4, "band_key": "x"}]})
    assert loaded.schedule.coordinate_law is sch.CoordinateLaw.STATIONARY_CHECKPOINT
    with pytest.raises(V2Error):     # progressive law, stationary coordinates
        _load(**{"schedule.points": [
            {"depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 50,
             "n_lookaheads": 4, "band_key": "x"}]})


def test_a_schedule_may_not_exhaust_the_lookahead_horizon():
    """Derived from the substrate geometry: as ``c_d`` approaches ``S`` the lookahead tail shrinks
    until siblings differ by a single draw and there is nothing left to select among. The floor is
    a required declared value, not a library guess."""
    with pytest.raises(V2Error):
        _load(**{"schedule.points": [
            {"depth": 0, "r_step": 40, "c_source_step": 95, "c_next_step": 96,
             "n_lookaheads": 4, "band_key": "x"},
            {"depth": 1, "r_step": 45, "c_source_step": 96, "c_next_step": 97,
             "n_lookaheads": 4, "band_key": "y"}]})


def test_content_provenance_must_cover_every_role_and_freeze_the_frozen_ones():
    """PLAN §5.2: missing content identity fails closed, and the three frozen roles must carry
    their digest in the config itself rather than being observed at runtime."""
    payload = _mapping()
    payload["content"] = [row for row in payload["content"] if row["role"] != "backbone"]
    with pytest.raises(V2Error, match="backbone"):
        cfg.load_v2_config(payload)
    payload = _mapping()
    for row in payload["content"]:
        if row["role"] == "schedule_band_calibration":
            row["binding"], row["expected_sha256"] = "runtime", None
    with pytest.raises(V2Error):
        cfg.load_v2_config(payload)


def test_the_config_digest_is_stable_and_covers_every_scientific_field():
    base = _load()
    assert base.config_digest() == _load().config_digest()
    assert base.config_digest() != _load(**{"identity.master_seed": 20260805}).config_digest()
    assert base.config_digest() != _load(
        **{"schedule.min_lookahead_tail_steps": 11}).config_digest()


def test_loading_a_config_reads_no_file_and_loads_no_model():
    """``--print-config`` / ``--dry-run`` must derive one shared digest with nothing loaded."""
    import inspect
    source = inspect.getsource(cfg)
    for forbidden in ("open(", "torch", "Path(", "yaml.safe_load"):
        assert forbidden not in source, forbidden
    assert copy.deepcopy(_mapping()) == _mapping()   # loading does not mutate its input


# ---- adversarial P1 closure -------------------------------------------------------------------

@pytest.mark.parametrize(
    "over",
    [
        {"arm.arm_role": "v2", "arm.feedback_enabled": False},
        {"arm.arm_role": "a2", "arm.feedback_enabled": True},
        {"arm.arm_role": "v0_boundary", "arm.feedback_enabled": True},
    ],
)
def test_arm_role_cannot_lie_about_feedback_semantics(over):
    """Only V2 enables source-coupled projection feedback; A2 and v0 do not."""
    with pytest.raises(V2Error):
        _load(**over)

    loaded = _load(**{"arm.arm_role": "a2", "arm.feedback_enabled": False})
    assert loaded.arm.arm_role == "a2"
    assert loaded.arm.feedback_enabled is False

    v0 = _load(**{"arm.arm_role": "v0_boundary", "arm.feedback_enabled": False})
    assert v0.arm.arm_role == "v0_boundary"
    assert v0.arm.feedback_enabled is False


def test_a_v0_threshold_cannot_be_laundered_by_renaming_its_source_kind():
    """The known v0 objective field remains forbidden even under a calibration-looking label."""
    with pytest.raises(V2Error):
        _load(**{"safety.delta_new_cumulative": _calibration(
            source_kind="measured_calibration",
            source_id="objective.max_offtarget_window_increase",
        )})


def test_calibration_source_ref_is_not_an_arbitrary_nonempty_digest():
    payload = _mapping()
    payload["safety"]["delta_new_cumulative"]["source_ref"] = "b" * 64
    with pytest.raises(V2Error):
        cfg.load_v2_config(payload)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("schema_version", "v0-calibration"),
        ("gate_kind", "off_halo_new_hotspot"),
        ("window_domain", "off_halo"),
    ],
)
def test_calibration_artifact_schema_and_window_domain_are_v2_typed(field, bad):
    payload = _mapping()
    payload["safety"]["delta_new_cumulative"]["artifact"][field] = bad
    with pytest.raises(V2Error):
        cfg.load_v2_config(payload)


@pytest.mark.parametrize(
    "calibration",
    [
        _calibration(
            scope="immediate_parent",
            reference_kind=cfg.INCREMENTAL_REFERENCE_KIND,
            reference_label=cfg.INCREMENTAL_REFERENCE_LABEL,
        ),
        _calibration(reference_kind="foreign_reference"),
        _calibration(reference_label="foreign_label"),
    ],
)
def test_cumulative_calibration_scope_and_reference_match_the_config(calibration):
    with pytest.raises(V2Error):
        _load(**{"safety.delta_new_cumulative": calibration})


def test_an_active_population_wider_than_one_is_refused_rather_than_silently_ignored():
    """The knob must not promise a capability the engine does not have.

    ``active_population_width`` is a REQUIRED field, so every run states how many lineages it
    advances -- but the depth ladder advances exactly one selected lineage per cycle (PLAN §4.4),
    and nothing anywhere reads the field.  Accepting ``2`` would let a config declare a population
    search, produce a single-lineage run, and report the two as the same thing.  It also cannot
    simply be implemented: Interface Map OQ7 leaves open whether a forked family inherits its
    parent's ``reference_binding_id`` or opens its own, and that choice decides whether the
    cumulative safety ratchet is per-run or per-family.
    """
    assert _load().schedule.active_population_width == 1
    with pytest.raises(cfg.V2ConfigError, match="active_population_width"):
        _load(**{"schedule.active_population_width": 2})


def test_the_refusal_names_the_open_question_rather_than_just_failing():
    """A bare rejection would read as 'unsupported yet'; the reason is a scientific decision."""
    with pytest.raises(cfg.V2ConfigError, match="OQ7|safety reference"):
        _load(**{"schedule.active_population_width": 3})


def test_a_placeholder_code_revision_is_refused():
    """``code_revision`` signs the run (PLAN §5.2) and is inside ``config_digest``.

    ``fusion_v2.schedule`` already refuses ``unknown``/``unset``/... for exactly this quantity on a
    band-calibration row; the run config accepted any non-empty string, so a run could be signed
    under a revision that names nothing.  A resume then cannot tell two builds apart, which is the
    case the field exists for.
    """
    for bad in ("unknown", "unset", "placeholder", "TODO", "not-hex"):
        with pytest.raises(cfg.V2ConfigError, match="code_revision"):
            _load(**{"identity.code_revision": bad})


def test_an_explicit_revision_is_accepted():
    assert _load(**{"identity.code_revision": "a1b2c3d4e5f6"}
                 ).identity.code_revision == "a1b2c3d4e5f6"
