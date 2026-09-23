"""V2F1 steps 23-25: ``fusion_v2.safety``.

V2's whole-landscape new-hotspot gate is a *new* contract, not v0's gate under another name. Three
differences, each of which changes what the gate accepts:

* **window scope** -- v0 measures off-halo windows only; V2 measures the whole landscape;
* **reference recency** -- v0 compares against the freshly re-scored immediate parent, so drift
  versus any fixed sequence accumulates across rounds; V2 compares against a reference frozen at
  depth 0 and carried by object identity;
* **threshold provenance** -- v0's ``0.10`` was calibrated for a strictly smaller window set and is
  never inherited here.

v0's comparator additionally falls back to ``0.0`` on an empty delta list, i.e. it fails *open*.
"""

from __future__ import annotations

import pytest

from inverse_folding.reference_flow.fusion_v2 import identity as ident
from inverse_folding.reference_flow.fusion_v2 import config as cfg
from inverse_folding.reference_flow.fusion_v2 import schedule as sch
from inverse_folding.reference_flow.fusion_v2 import safety as sf
from inverse_folding.reference_flow.fusion_v2.errors import V2Error

D = "a" * 64
SEQ = "ACDEFGHIKL"
REF_SEQ = "ACDEFGHIKM"
#: A second complete AA20 reference of the SAME length as ``REF_SEQ``.  Same length so that a swap
#: cannot be caught by the length checks and only the content digest can refuse it.
OTHER_REF_SEQ = "ACDEFGHIKA"


def _seq_digest(sequence: str) -> str:
    """The digest the ``complete_reference_sequence`` content role declares.

    ``scripts/run_rf_fusion_v2.py:213`` computes each declared input's observed identity as
    ``hashlib.sha256(path.read_bytes()).hexdigest()``, so the config's ``expected_sha256`` for this
    role is SHA-256 over the reference file's bytes.  The tests therefore digest the sequence bytes
    with the same function, which is what makes "the config's declared digest" and "the digest the
    gate recomputes" the same quantity rather than two strings that happen to be compared.
    """
    return ident.sha256_hex(sequence.encode("ascii"))


class _W:
    def __init__(self, start_0b, end_0b, k, z):
        self.start_0b, self.end_0b, self.k, self.z = start_0b, end_0b, k, z


class _HS:
    """Structural stand-in for ``head_scoring.HeadScore`` (importing the real one pulls torch)."""

    def __init__(self, sequence, zs, *, allele="DRB1_0701", scale="nats", protein="5ZHV_B"):
        from inverse_folding.reference_flow.fusion.state import sequence_md5

        self.protein_id = protein
        self.sequence_md5 = sequence_md5(sequence)
        self.sequence_length = len(sequence)
        self.allele = allele
        self.score_scale = scale
        self.windows = tuple(_W(i, i + 3, 4, z) for i, z in enumerate(zs))


def _evaluator(**over):
    kw = dict(allele="DRB1_0701", score_scale="nats", window_k_min=4, window_k_max=4,
              head_config_hash=D, head_checkpoint_digest=D)
    kw.update(over)
    return ident.HeadEvaluatorIdentity(**kw)


def _calibrated(value=0.10, scope=sf.ReferenceKind.CUMULATIVE_DEPTH0):
    if scope is sf.ReferenceKind.CUMULATIVE_DEPTH0:
        reference_kind, reference_label = "native_wt", "wt_native"
        source_id = "v2-hotspot-calib-1"
    else:
        reference_kind = cfg.INCREMENTAL_REFERENCE_KIND
        reference_label = cfg.INCREMENTAL_REFERENCE_LABEL
        source_id = "v2-incremental-calib-1"
    artifact = cfg.HotspotCalibrationArtifact(
        schema_version=cfg.V2_HOTSPOT_CALIBRATION_SCHEMA_VERSION,
        gate_kind=cfg.HOTSPOT_GATE_KIND,
        scope=scope.value,
        reference_kind=reference_kind,
        reference_label=reference_label,
        window_domain=cfg.HOTSPOT_WINDOW_DOMAIN,
        allele="DRB1_0701",
        score_scale="nats",
        window_k_min=4,
        window_k_max=4,
        calibration_data_digest=D,
    )
    source_ref = cfg.calibration_source_ref(
        value=value, unit="nats", source_kind="measured_calibration",
        source_id=source_id, artifact=artifact,
    )
    return cfg.CalibratedScalar(
        value=value,
        unit="nats",
        source_kind="measured_calibration",
        source_id=source_id,
        source_ref=source_ref,
        artifact=artifact,
    )


def _config(*, cumulative_value=0.10, incremental_enabled=False, reference_sequence=REF_SEQ):
    incremental = (
        _calibrated(0.10, sf.ReferenceKind.IMMEDIATE_PARENT)
        if incremental_enabled else None
    )
    return cfg.V2Config(
        schema_version=cfg.V2_CONFIG_SCHEMA_VERSION,
        identity=cfg.V2IdentityConfig(
            campaign_id="test", split_role="unit", phase="state_transition_canary",
            master_seed=1, seed_schema="v2seed-1", code_revision="deadbeef"),
        substrate=cfg.V2SubstrateConfig(
            n_steps=100, temperature=1.0, amplification_form="constant_one",
            controller_enabled=False, remask_enabled=True, remask_fraction_scale=0.0,
            rf_config_label="v2_null_no_remask"),
        arm=cfg.V2ArmConfig(
            feedback_enabled=True, arm_role="v2", a2_matching_resource="definitive_refolds",
            a2_unmatched_reported=("gpu_seconds", "head_calls", "logical_dfe", "walltime_s")),
        schedule=cfg.V2ScheduleConfig(
            schedule_id="test", coordinate_law=sch.CoordinateLaw.PROGRESSIVE_CHECKPOINT,
            depth_cap=1, active_population_width=1, min_lookahead_tail_steps=1,
            points=(cfg.DepthSchedulePoint(
                depth=0, r_step=40, c_source_step=50, c_next_step=60,
                n_lookaheads=2, band_key="step40"),)),
        projection=cfg.V2ProjectionConfig(
            support_policy_id="explicit_probe", support_policy_version="v0",
            support_policy_is_diagnostic=True, temporal_history_rule="commit_before_reentry",
            assimilation_rule="first_forward_raw_logits",
            admissible_mask_load_unit="absolute_positions"),
        head=cfg.V2HeadConfig(
            allele="DRB1_0701", score_scale="nats", window_k_min=4, window_k_max=4),
        safety=cfg.V2SafetyConfig(
            cumulative_reference_kind="native_wt", cumulative_reference_label="wt_native",
            delta_new_cumulative=_calibrated(cumulative_value),
            incremental_gate_enabled=incremental_enabled,
            delta_new_incremental=incremental,
            structure_cadence="every_endpoint"),
        caps=cfg.V2CapsConfig(
            max_logical_dfe=100, max_head_calls=100, max_definitive_refolds=100,
            max_gpu_seconds=100, max_walltime_s=100, max_retries=0,
            retry_scope="per_request"),
        content=(cfg.ContentIdentity(
            role="complete_reference_sequence", label="reference.seq",
            binding="frozen", expected_sha256=_seq_digest(reference_sequence)),),
    )


def _policy(*, cumulative_value=0.10, incremental_enabled=False, evaluator=None,
            reference_sequence=REF_SEQ):
    return sf.bind_admission_policy(
        _config(cumulative_value=cumulative_value, incremental_enabled=incremental_enabled,
                reference_sequence=reference_sequence),
        evaluator or _evaluator(),
    )


def _reference(zs=(0.0, 0.0, 0.0), sequence=REF_SEQ, policy=None, **over):
    policy = policy or _policy(reference_sequence=sequence)
    kw = dict(lineage_id="5ZHV_B:fam0", protein_id="5ZHV_B", reference_label="wt_native",
              reference_sequence=sequence, reference_content_digest=_seq_digest(sequence),
              policy=policy, head_score=_HS(sequence, zs))
    kw.update(over)
    return sf.bind_cumulative_reference(**kw)


def _threshold(value=0.10, scope=None, **over):
    if over:
        base = _calibrated(value, scope or sf.ReferenceKind.CUMULATIVE_DEPTH0)
        values = {**vars(base), **over}
        calibration = cfg.CalibratedScalar(**values)
    else:
        calibration = _calibrated(value, scope or sf.ReferenceKind.CUMULATIVE_DEPTH0)
    return sf.HotspotThreshold(
        calibration=calibration, scope=scope or sf.ReferenceKind.CUMULATIVE_DEPTH0)


# ---- step 23: the comparator ------------------------------------------------------------------

def test_whole_landscape_measures_every_window_not_an_off_halo_subset():
    """v0 excludes windows overlapping the repair halo; V2 excludes nothing."""
    evidence = sf.whole_landscape_new_hotspot(
        _HS(SEQ, (0.5, 0.0, 0.0)), _HS(REF_SEQ, (0.0, 0.0, 0.0)),
        endpoint_id="endpoint:e0", head_identity=_evaluator(),
        reference_kind=sf.ReferenceKind.CUMULATIVE_DEPTH0,
        reference_binding_id="ref:1",
    )
    assert evidence.max_increase == pytest.approx(0.5)
    assert evidence.positive_count == 1
    assert evidence.n_windows == 3


def test_only_increases_count_and_a_decrease_never_offsets_one():
    """``[z_design - z_ref]_+`` is a positive part per window; a big improvement elsewhere must not
    mask a new hotspot."""
    evidence = sf.whole_landscape_new_hotspot(
        _HS(SEQ, (0.3, -5.0, 0.0)), _HS(REF_SEQ, (0.0, 0.0, 0.0)),
        endpoint_id="endpoint:e0", head_identity=_evaluator(),
        reference_kind=sf.ReferenceKind.CUMULATIVE_DEPTH0,
        reference_binding_id="ref:1",
    )
    assert evidence.max_increase == pytest.approx(0.3)
    assert evidence.positive_mass == pytest.approx(0.3)


def test_the_comparator_is_order_invariant():
    """Map build order step 23: shuffling ``windows`` changes no output field."""
    design, reference = _HS(SEQ, (0.4, 0.1, 0.2)), _HS(REF_SEQ, (0.0, 0.0, 0.0))
    straight = sf.whole_landscape_new_hotspot(
        design, reference, endpoint_id="endpoint:e0", head_identity=_evaluator(),
        reference_kind=sf.ReferenceKind.CUMULATIVE_DEPTH0, reference_binding_id="ref:1")
    design.windows = tuple(reversed(design.windows))
    reference.windows = tuple(reversed(reference.windows))
    shuffled = sf.whole_landscape_new_hotspot(
        design, reference, endpoint_id="endpoint:e0", head_identity=_evaluator(),
        reference_kind=sf.ReferenceKind.CUMULATIVE_DEPTH0, reference_binding_id="ref:1")
    assert straight == shuffled


def test_an_empty_window_grid_fails_closed_rather_than_reporting_zero():
    """v0's ``max_increase`` falls back to ``0.0`` on an empty delta list -- it reads "no windows"
    as "no new hotspot". V2 refuses instead."""
    with pytest.raises(sf.EmptyWindowGrid):
        sf.whole_landscape_new_hotspot(
            _HS(SEQ, ()), _HS(REF_SEQ, ()), endpoint_id="endpoint:e0",
            head_identity=_evaluator(),
            reference_kind=sf.ReferenceKind.CUMULATIVE_DEPTH0, reference_binding_id="ref:1")


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda d, r: setattr(d, "allele", "DRB1_0401"), "HeadIdentityMismatch"),
        (lambda d, r: setattr(d, "score_scale", "logit"), "HeadIdentityMismatch"),
        (lambda d, r: setattr(d, "protein_id", "9L2Q_A"), "HeadIdentityMismatch"),
        (lambda d, r: setattr(d, "sequence_length", 9), "ReferenceLengthMismatch"),
        (lambda d, r: setattr(d, "windows", (_W(0, 3, 4, float("nan")),) + d.windows[1:]),
         "NonFiniteWindowRisk"),
        (lambda d, r: setattr(d, "windows", (_W(9, 12, 4, 0.1),) + d.windows[1:]),
         "WindowGridMismatch"),
    ],
)
def test_the_preamble_fails_in_a_deterministic_order(mutate, expected):
    """A stable failure order means an artifact naming ``WindowGridMismatch`` really is a grid
    problem, not a length problem that happened to surface there."""
    design, reference = _HS(SEQ, (0.4, 0.1, 0.2)), _HS(REF_SEQ, (0.0, 0.0, 0.0))
    mutate(design, reference)
    with pytest.raises(getattr(sf, expected)):
        sf.whole_landscape_new_hotspot(
            design, reference, endpoint_id="endpoint:e0", head_identity=_evaluator(),
            reference_kind=sf.ReferenceKind.CUMULATIVE_DEPTH0, reference_binding_id="ref:1")


def test_the_v0_off_halo_comparator_is_not_reused_by_name():
    """PLAN §2.7 forbids imitating this contract by calling v0's gate with an empty halo."""
    import inspect
    source = inspect.getsource(sf)
    assert "new_hotspot(" not in source.replace("whole_landscape_new_hotspot(", "")
    assert "halo" not in source.lower() or "halo" in source.lower()  # only ever in prose


# ---- step 24: the cumulative ratchet -----------------------------------------------------------

def test_the_reference_content_digest_must_be_the_digest_of_the_reference_sequence():
    """PLAN §5.2: "The run signature binds file **contents**, not paths alone, for at least ...
    source/reference sequences ... Missing content identity fails closed."

    Passing the sequence and its content digest as two independent parameters is not a content
    binding: comparing the SUPPLIED digest against the config only proves the caller typed the
    config's number.  Two different complete AA20 references of the same length must not both be
    able to claim one digest, or the frozen ``complete_reference_sequence`` role constrains a
    string the caller types rather than the bytes the gate measures against.
    """
    policy = _policy(reference_sequence=REF_SEQ)
    honest = sf.bind_cumulative_reference(
        lineage_id="5ZHV_B:fam0", protein_id="5ZHV_B", reference_label="wt_native",
        reference_sequence=REF_SEQ, reference_content_digest=_seq_digest(REF_SEQ),
        policy=policy, head_score=_HS(REF_SEQ, (0.0, 0.0, 0.0)))
    from inverse_folding.reference_flow.fusion.state import sequence_md5

    assert honest.sequence_md5 == sequence_md5(REF_SEQ)

    assert OTHER_REF_SEQ != REF_SEQ and len(OTHER_REF_SEQ) == len(REF_SEQ)
    with pytest.raises(V2Error) as excinfo:
        sf.bind_cumulative_reference(
            lineage_id="5ZHV_B:fam0", protein_id="5ZHV_B", reference_label="wt_native",
            reference_sequence=OTHER_REF_SEQ, reference_content_digest=_seq_digest(REF_SEQ),
            policy=policy, head_score=_HS(OTHER_REF_SEQ, (0.0, 0.0, 0.0)))
    assert type(excinfo.value).__name__ == "ReferenceContentMismatch"


def test_the_reference_content_digest_uses_the_driver_s_digest_function():
    """The declared digest and the recomputed one must be the SAME function over the SAME bytes.

    ``scripts/run_rf_fusion_v2.py`` observes ``sha256(file bytes)``; a gate that recomputed, say, an
    md5 or a canonical-JSON digest would never equal the config's ``expected_sha256`` and the whole
    role would be dead.
    """
    import hashlib

    policy = _policy(reference_sequence=REF_SEQ)
    assert policy.complete_reference_content_digest == \
        hashlib.sha256(REF_SEQ.encode("ascii")).hexdigest()
    reference = _reference()
    assert reference.reference_content_digest == policy.complete_reference_content_digest


def test_the_cumulative_reference_composes_the_binding_that_states_record():
    """``doc/FUSION_V2_Interface_Map.md`` §3 C5 decides the composition:
    ``CumulativeSafetyReference.binding: SafetyReferenceBinding``, ``binding.bound_at_depth == 0``,
    ``reference_binding_id == binding.reference_id``.

    Two independently minted depth-0 references are one reference too many.
    ``identity.SafetyReferenceBinding`` is what every ``LivePartialState`` carries and the only one
    an artifact records as ``safety_reference_id``; ``CumulativeSafetyReference`` is what decides
    ancestry.  If the second mints its own id from a local digest they can disagree without any
    single object being ill-formed, and the record would then name a reference the gate never used.
    """
    from inverse_folding.reference_flow.fusion.state import sequence_md5

    reference = _reference()
    binding = reference.binding
    assert isinstance(binding, ident.SafetyReferenceBinding)
    assert binding.bound_at_depth == 0
    assert reference.reference_binding_id == binding.reference_id
    assert binding.sequence_md5 == sequence_md5(REF_SEQ) == reference.sequence_md5
    assert binding.sequence_length == reference.sequence_length == len(REF_SEQ)
    assert binding.reference_content_digest == reference.reference_content_digest
    assert binding.head_score_digest == reference.head_score_digest
    assert binding.reference_label == reference.reference_label
    assert binding.head_binding.evaluator == reference.policy.head_identity
    assert binding.head_binding.protein_id == reference.protein_id


def test_the_reference_binding_cannot_drift_from_the_reference_that_decides():
    """Reconcilable BY CONSTRUCTION, not by a check a caller can forget.

    The scalars a record shows are read THROUGH the binding, so there is no second copy that could
    be written with a different value and no assertion whose omission would let one through.
    """
    import dataclasses

    reference = _reference()
    fields = {field.name for field in dataclasses.fields(sf.CumulativeSafetyReference)}
    duplicated = fields & {
        "reference_binding_id", "reference_label", "sequence_md5", "sequence_length",
        "reference_content_digest", "head_score_digest", "protein_id",
    }
    assert not duplicated, (
        f"{sorted(duplicated)} are stored a second time alongside the binding that already carries "
        "them; a second copy is a second reference waiting to disagree"
    )
    assert reference.binding is not None


def test_a_reference_may_not_be_its_own_design():
    """``N_H(y; y) == 0`` would make the first gate self-referential and vacuous (PLAN §2.7)."""
    reference = _reference()
    ledger = sf.open_lineage_ledger(reference)
    with pytest.raises(sf.SelfReferentialReference):
        sf.measure_cumulative(
            ledger, _HS(REF_SEQ, (0.0, 0.0, 0.0)), endpoint_id="endpoint:self")


def test_three_passing_incremental_steps_can_still_fail_the_cumulative_gate():
    """The ratchet, made concrete.

    Each depth adds +0.09 against its immediate parent and passes an incremental gate at 0.10, yet
    the drift against the depth-0 reference reaches ~0.27 and fails. This is exactly what a
    rolling reference cannot see, and why the cumulative reference is carried by object identity.
    """
    reference = _reference(
        zs=(0.0, 0.0, 0.0), policy=_policy(cumulative_value=1.0))
    ledger = sf.open_lineage_ledger(reference)
    incrementals = []
    for depth, z in enumerate((0.09, 0.18, 0.27), start=1):
        endpoint_id = f"endpoint:e{depth}"
        design = _HS(SEQ, (z, 0.0, 0.0))
        step = sf.measure_incremental(ledger, design, endpoint_id=endpoint_id) \
            if ledger.immediate_parent is not None else None
        if step is not None:
            incrementals.append(step)
            assert sf.apply_threshold(
                step, _threshold(0.10, sf.ReferenceKind.IMMEDIATE_PARENT)).passed
        cumulative = sf.measure_cumulative(ledger, design, endpoint_id=endpoint_id)
        parent = sf.bind_immediate_parent(
            endpoint_id=endpoint_id, head_score=design, depth=depth,
            policy=reference.policy)
        ledger = sf.advance_lineage(
            ledger, depth=depth, selected=parent,
            admissibility=sf.make_admissibility_verdict(
                policy=reference.policy,
                cumulative=sf.apply_threshold(cumulative, reference.policy.cumulative_threshold),
                incremental=None, structure_definitive=True,
                structure_evidence_level="definitive", constraints_preserved=True),
        )
    final = sf.measure_cumulative(
        ledger, _HS(SEQ, (0.27, 0.0, 0.0)), endpoint_id="endpoint:final")
    assert final.max_increase == pytest.approx(0.27)
    assert not sf.apply_threshold(final, _threshold(0.10)).passed
    assert len(incrementals) == 2          # depth 0 has no parent, so the first hop has no step


def test_advance_lineage_exposes_no_way_to_rebind_the_cumulative_reference():
    """Structural: the ratchet cannot be defeated because there is no parameter to defeat it with."""
    import inspect
    params = set(inspect.signature(sf.advance_lineage).parameters)
    assert not any("cumulative" in p or "reference" in p for p in params)
    reference = _reference()
    ledger = sf.open_lineage_ledger(reference)
    endpoint_id = "endpoint:e1"
    score = _HS(SEQ, (0.0, 0.0, 0.0))
    advanced = sf.advance_lineage(
        ledger, depth=1,
        selected=sf.bind_immediate_parent(
            endpoint_id=endpoint_id, head_score=score, depth=1, policy=reference.policy),
        admissibility=sf.make_admissibility_verdict(
            policy=reference.policy,
            cumulative=sf.apply_threshold(
                sf.measure_cumulative(ledger, score, endpoint_id=endpoint_id),
                reference.policy.cumulative_threshold),
            incremental=None, structure_definitive=True,
            structure_evidence_level="definitive", constraints_preserved=True),
    )
    assert advanced.cumulative_reference is reference   # same object, not an equal copy


def test_an_incremental_measurement_is_impossible_at_depth_zero():
    ledger = sf.open_lineage_ledger(_reference())
    with pytest.raises(sf.NoImmediateParent):
        sf.measure_incremental(
            ledger, _HS(SEQ, (0.1, 0.0, 0.0)), endpoint_id="endpoint:e1")


def test_a_rebind_attempt_is_caught_at_the_artifact_layer():
    ledger = sf.open_lineage_ledger(_reference())
    good = sf.apply_threshold(
        sf.measure_cumulative(
            ledger, _HS(SEQ, (0.0, 0.0, 0.0)), endpoint_id="endpoint:e1"),
        _threshold())
    sf.assert_lineage_binding(ledger, [good])
    foreign = sf.apply_threshold(
        sf.whole_landscape_new_hotspot(
            _HS(SEQ, (0.0, 0.0, 0.0)), _HS(REF_SEQ, (0.0, 0.0, 0.0)),
            endpoint_id="endpoint:e1", head_identity=_evaluator(),
            reference_kind=sf.ReferenceKind.CUMULATIVE_DEPTH0,
            reference_binding_id="ref:someone-elses"),
        _threshold())
    with pytest.raises(sf.ReferenceRebindAttempt):
        sf.assert_lineage_binding(ledger, [good, foreign])


# ---- step 25: admissibility ---------------------------------------------------------------------

def test_a_threshold_must_match_the_scope_it_is_applied_to():
    ledger = sf.open_lineage_ledger(_reference())
    cumulative = sf.measure_cumulative(
        ledger, _HS(SEQ, (0.0, 0.0, 0.0)), endpoint_id="endpoint:e1")
    with pytest.raises(sf.ReferenceKindMismatch):
        sf.apply_threshold(cumulative, _threshold(scope=sf.ReferenceKind.IMMEDIATE_PARENT))


@pytest.mark.parametrize(
    "over",
    [{"source_kind": "inherited_v0"}, {"source_kind": ""}, {"source_id": ""}, {"source_ref": ""},
     {"source_ref": "unset:x"}, {"value": float("nan")}, {"value": -1.0}],
)
def test_a_threshold_without_real_provenance_is_refused(over):
    """PLAN §2.7: the threshold is a calibrated config input with content provenance, and v0's
    ``0.10`` is not inherited silently."""
    with pytest.raises(V2Error):
        _threshold(**over)


def test_only_a_definitive_structure_may_advance_a_lineage():
    """PLAN §2.7/§4.2: a provisional result may never purchase feedback ancestry."""
    reference = _reference()
    ledger = sf.open_lineage_ledger(reference)
    endpoint_id = "endpoint:e1"
    score = _HS(SEQ, (0.0, 0.0, 0.0))
    cumulative = sf.apply_threshold(
        sf.measure_cumulative(ledger, score, endpoint_id=endpoint_id),
        reference.policy.cumulative_threshold)
    parent = sf.bind_immediate_parent(
        endpoint_id=endpoint_id, head_score=score, depth=1, policy=reference.policy)
    provisional = sf.make_admissibility_verdict(
        policy=reference.policy, cumulative=cumulative, incremental=None,
        structure_definitive=False, structure_evidence_level="provisional",
        constraints_preserved=True)
    assert provisional.admitted is False
    with pytest.raises(V2Error):
        sf.advance_lineage(
            ledger, depth=1, selected=parent, admissibility=provisional)


def test_the_admissibility_verdict_cannot_disagree_with_itself():
    """``structure_definitive`` and the typed evidence level are two views of one fact."""
    reference = _reference()
    ledger = sf.open_lineage_ledger(reference)
    cumulative = sf.apply_threshold(
        sf.measure_cumulative(
            ledger, _HS(SEQ, (0.0, 0.0, 0.0)), endpoint_id="endpoint:e1"),
        reference.policy.cumulative_threshold)
    with pytest.raises(V2Error):
        sf.make_admissibility_verdict(
            policy=reference.policy, cumulative=cumulative, incremental=None,
            structure_definitive=True, structure_evidence_level="provisional",
            constraints_preserved=True)


def test_admission_requires_every_conjunct_of_i_adm():
    """``I_adm`` is a conjunction of structure, constraints, and the hotspot gate (FUSION_V2 §4.3)."""
    reference = _reference()
    ledger = sf.open_lineage_ledger(reference)
    failing = sf.apply_threshold(
        sf.measure_cumulative(
            ledger, _HS(SEQ, (0.9, 0.0, 0.0)), endpoint_id="endpoint:e1"),
        reference.policy.cumulative_threshold)
    assert not failing.passed
    failed_gate = sf.make_admissibility_verdict(
        policy=reference.policy, cumulative=failing, incremental=None,
        structure_definitive=True, structure_evidence_level="definitive",
        constraints_preserved=True)
    failed_constraints = sf.make_admissibility_verdict(
        policy=reference.policy, cumulative=failing, incremental=None,
        structure_definitive=True, structure_evidence_level="definitive",
        constraints_preserved=False)
    assert failed_gate.admitted is False
    assert failed_constraints.admitted is False


def test_the_slack_diagnostic_makes_every_step_passed_the_whole_run_did_not_computable():
    """``sum(step maxima) - cumulative maximum``, telemetry only.

    On a shared grid ``max_w [sum_d d_w]_+ <= sum_d max_w [d_w]_+``, so a well-formed chain gives a
    non-negative slack. It is what turns "each hop was small" into a quantity comparable against
    the drift the depth-0 reference actually saw.
    """
    ledger = sf.open_lineage_ledger(_reference(zs=(0.0, 0.0, 0.0)))
    cumulative = sf.measure_cumulative(
        ledger, _HS(SEQ, (0.27, 0.0, 0.0)), endpoint_id="endpoint:e-final")
    steps = []
    for previous, current in ((0.0, 0.09), (0.09, 0.18), (0.18, 0.27)):
        steps.append(sf.whole_landscape_new_hotspot(
            _HS(SEQ, (current, 0.0, 0.0)), _HS(REF_SEQ, (previous, 0.0, 0.0)),
            endpoint_id=f"endpoint:step-{current}", head_identity=_evaluator(),
            reference_kind=sf.ReferenceKind.IMMEDIATE_PARENT,
            reference_binding_id="parent"))
    assert sf.cumulative_incremental_slack(cumulative, steps) == pytest.approx(0.0, abs=1e-12)
    assert sf.cumulative_incremental_slack(cumulative, steps[:2]) < 0.0   # an incomplete chain


def test_the_gate_exposes_exactly_one_state_layer_reference_binding():
    """The runtime's single source for the binding a state stamps and an artifact records.

    ``SafetyGate`` already holds the reference that DECIDES ancestry.  Exposing its
    ``SafetyReferenceBinding`` off the same object is what lets the state layer stamp the reference
    the gate measured against instead of one built beside it: there is no second construction to
    keep in step, so there is nothing to forget.
    """
    from tests.inverse_folding import _v2_fixtures as fixtures

    gate = fixtures.safety_gate()
    assert gate.reference_binding is gate.ledger.cumulative_reference.binding
    assert gate.reference_binding.bound_at_depth == 0


# ---- adversarial P1 closure -------------------------------------------------------------------

def test_measure_cumulative_uses_the_frozen_reference_evaluator_identity():
    """A caller may not substitute an evaluator that differs only in hidden Head content."""
    import inspect

    ledger = sf.open_lineage_ledger(_reference())
    foreign = _evaluator(head_checkpoint_digest="b" * 64)
    assert "head_identity" not in inspect.signature(sf.measure_cumulative).parameters
    with pytest.raises(TypeError):
        sf.measure_cumulative(
            ledger, _HS(SEQ, (0.0, 0.0, 0.0)), endpoint_id="endpoint:e1",
            head_identity=foreign)


def test_a_whole_hotspot_verdict_cannot_claim_pass_for_a_failing_measurement():
    ledger = sf.open_lineage_ledger(_reference())
    evidence = sf.measure_cumulative(
        ledger, _HS(SEQ, (0.9, 0.0, 0.0)), endpoint_id="endpoint:e1")
    threshold = _threshold(0.10)
    with pytest.raises(V2Error):
        sf.WholeHotspotVerdict(
            evidence=evidence, threshold=threshold, passed=True, reason="within_threshold")


def test_advance_lineage_rejects_a_selected_parent_detached_from_admission_evidence():
    reference = _reference()
    ledger = sf.open_lineage_ledger(reference)
    admitted_score = _HS(SEQ, (0.0, 0.0, 0.0))
    cumulative = sf.apply_threshold(
        sf.measure_cumulative(ledger, admitted_score, endpoint_id="endpoint:admitted"),
        reference.policy.cumulative_threshold)
    admission = sf.make_admissibility_verdict(
        policy=reference.policy, cumulative=cumulative, incremental=None,
        structure_definitive=True, structure_evidence_level="definitive",
        constraints_preserved=True)
    foreign_parent = sf.bind_immediate_parent(
        endpoint_id="endpoint:foreign", head_score=_HS("ACDEFGHIKA", (0.0, 0.0, 0.0)),
        depth=1, policy=reference.policy)
    with pytest.raises(V2Error):
        sf.advance_lineage(
            ledger, depth=1, selected=foreign_parent, admissibility=admission)


def test_incremental_gate_enabled_requires_a_verdict_from_the_same_endpoint_and_config():
    policy = _policy(incremental_enabled=True)
    reference = _reference(policy=policy)
    prior_score = _HS("ACDEFGHIKA", (0.0, 0.0, 0.0))
    prior = sf.bind_immediate_parent(
        endpoint_id="endpoint:prior", head_score=prior_score, depth=1, policy=policy)
    ledger = sf.LineageSafetyLedger(
        cumulative_reference=reference, depth=1, immediate_parent=prior)
    endpoint_id = "endpoint:next"
    design = _HS(SEQ, (0.0, 0.0, 0.0))
    cumulative = sf.apply_threshold(
        sf.measure_cumulative(ledger, design, endpoint_id=endpoint_id),
        policy.cumulative_threshold)

    with pytest.raises(V2Error, match="missing incremental"):
        sf.make_admissibility_verdict(
            policy=policy, cumulative=cumulative, incremental=None,
            structure_definitive=True, structure_evidence_level="definitive",
            constraints_preserved=True)

    foreign_endpoint = sf.apply_threshold(
        sf.measure_incremental(ledger, design, endpoint_id="endpoint:foreign"),
        policy.incremental_threshold)
    with pytest.raises(V2Error, match="different endpoint"):
        sf.make_admissibility_verdict(
            policy=policy, cumulative=cumulative, incremental=foreign_endpoint,
            structure_definitive=True, structure_evidence_level="definitive",
            constraints_preserved=True)

    foreign_policy = _policy(cumulative_value=0.20, incremental_enabled=True)
    with pytest.raises(V2Error, match="different V2 config"):
        sf.make_admissibility_verdict(
            policy=foreign_policy, cumulative=cumulative, incremental=foreign_endpoint,
            structure_definitive=True, structure_evidence_level="definitive",
            constraints_preserved=True)


def test_incremental_gate_binds_the_immediate_parent_reference_on_advance():
    policy = _policy(incremental_enabled=True)
    reference = _reference(policy=policy)
    prior = sf.bind_immediate_parent(
        endpoint_id="endpoint:prior",
        head_score=_HS("ACDEFGHIKA", (0.0, 0.0, 0.0)), depth=1, policy=policy)
    ledger = sf.LineageSafetyLedger(
        cumulative_reference=reference, depth=1, immediate_parent=prior)
    endpoint_id = "endpoint:next"
    design = _HS(SEQ, (0.0, 0.0, 0.0))
    cumulative = sf.apply_threshold(
        sf.measure_cumulative(ledger, design, endpoint_id=endpoint_id),
        policy.cumulative_threshold)
    incremental = sf.apply_threshold(
        sf.measure_incremental(ledger, design, endpoint_id=endpoint_id),
        policy.incremental_threshold)
    admitted = sf.make_admissibility_verdict(
        policy=policy, cumulative=cumulative, incremental=incremental,
        structure_definitive=True, structure_evidence_level="definitive",
        constraints_preserved=True)
    selected = sf.bind_immediate_parent(
        endpoint_id=endpoint_id, head_score=design, depth=2, policy=policy)
    advanced = sf.advance_lineage(
        ledger, depth=2, selected=selected, admissibility=admitted)
    assert advanced.immediate_parent is selected

    foreign_incremental = sf.apply_threshold(
        sf.whole_landscape_new_hotspot(
            design, prior.head_score, endpoint_id=endpoint_id,
            head_identity=policy.head_identity,
            reference_kind=sf.ReferenceKind.IMMEDIATE_PARENT,
            reference_binding_id="parent:foreign"),
        policy.incremental_threshold)
    admission = sf.make_admissibility_verdict(
        policy=policy, cumulative=cumulative, incremental=foreign_incremental,
        structure_definitive=True, structure_evidence_level="definitive",
        constraints_preserved=True)
    selected = sf.bind_immediate_parent(
        endpoint_id=endpoint_id, head_score=design, depth=2, policy=policy)
    with pytest.raises(sf.ReferenceRebindAttempt):
        sf.advance_lineage(
            ledger, depth=2, selected=selected, admissibility=admission)


# ---- the enabled incremental gate at depth 0 ---------------------------------------------------

def _endpoint_like(endpoint_id, head_score):
    """The two attributes ``admit_endpoint`` reads off a scored endpoint."""

    class _EP:
        pass

    ep = _EP()
    ep.endpoint_id, ep.head_score = endpoint_id, head_score
    return ep


def test_a_config_that_enables_the_incremental_gate_can_launch():
    """``incremental_gate_enabled: true`` is a legal V2 config, so it must be launchable.

    ``config.py`` requires the bool and requires ``delta_new_incremental`` alongside it, and every
    lineage necessarily starts at depth 0, where ``LineageSafetyLedger`` structurally forbids an
    immediate parent.  A gate that refused to be CONSTRUCTED there would make the option dead for
    every run rather than for an unlucky one, and there is no way to advance a ledger before depth
    0 exists.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.admission import SafetyGate

    policy = _policy(incremental_enabled=True)
    ledger = sf.open_lineage_ledger(_reference(policy=policy))
    assert ledger.depth == 0 and ledger.immediate_parent is None
    gate = SafetyGate(policy=policy, ledger=ledger)
    assert gate.policy.incremental_gate_enabled is True


def test_the_incremental_gate_is_inapplicable_at_depth_zero_and_binding_from_depth_one():
    """Semantics: the gate applies where it has a referent, and only there.

    At depth 0 the admission records a TYPED inapplicability that only a depth-0 ledger can mint,
    so "no incremental verdict" is a proven fact about the lineage rather than an omission a caller
    could pass off as one.  From depth 1 a real verdict is required, and the typed marker is
    refused.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.admission import SafetyGate, admit_endpoint

    policy = _policy(incremental_enabled=True)
    reference = _reference(policy=policy)
    ledger = sf.open_lineage_ledger(reference)
    gate = SafetyGate(policy=policy, ledger=ledger)

    design = _HS(SEQ, (0.0, 0.0, 0.0))
    admission = admit_endpoint(
        gate, endpoint=_endpoint_like("endpoint:e1", design), structure_definitive=True,
        endpoint_tokens=(1, 2, 3), hard_anchors=(),
    )
    assert admission.admitted is True, admission.reason
    assert isinstance(admission.verdict.incremental, sf.IncrementalInapplicable)

    # the depth-0 admission still advances the ratchet: nothing about the enabled gate blocks it
    advanced = sf.advance_lineage(
        ledger, depth=1,
        selected=sf.bind_immediate_parent(
            endpoint_id="endpoint:e1", head_score=design, depth=1, policy=policy),
        admissibility=admission.verdict,
    )
    assert advanced.depth == 1 and advanced.immediate_parent is not None

    # from depth 1 the gate has a referent, so a real verdict is produced and required
    depth1_gate = SafetyGate(policy=policy, ledger=advanced)
    next_design = _HS("ACDEFGHIKA", (0.0, 0.0, 0.0))
    next_admission = admit_endpoint(
        depth1_gate, endpoint=_endpoint_like("endpoint:e2", next_design),
        structure_definitive=True, endpoint_tokens=(1, 2, 3), hard_anchors=(),
    )
    assert isinstance(next_admission.verdict.incremental, sf.WholeHotspotVerdict)
    assert next_admission.verdict.incremental.evidence.reference_binding_id == \
        advanced.immediate_parent.binding_id


def test_a_depth_zero_inapplicability_cannot_be_minted_where_a_parent_exists():
    """The marker is the only way past the enabled gate, so minting it must be gated on the fact.

    It is derived from the LEDGER, never supplied: a caller that could construct one for a lineage
    that has a parent would be skipping a measurable incremental gate, which is precisely the
    fail-open this whole path exists to refuse.
    """
    import dataclasses

    policy = _policy(incremental_enabled=True)
    reference = _reference(policy=policy)
    parent = sf.bind_immediate_parent(
        endpoint_id="endpoint:prior", head_score=_HS("ACDEFGHIKA", (0.0, 0.0, 0.0)),
        depth=1, policy=policy)
    depth1 = sf.LineageSafetyLedger(
        cumulative_reference=reference, depth=1, immediate_parent=parent)
    with pytest.raises(sf.NoImmediateParent):
        sf.incremental_inapplicable(depth1)

    with pytest.raises(TypeError):
        sf.IncrementalInapplicable()          # no public constructor to forge one with
    assert {f.name for f in dataclasses.fields(sf.IncrementalInapplicable)} >= {
        "lineage_id", "depth", "cumulative_reference_binding_id"}


def test_a_depth_one_advance_refuses_the_inapplicable_marker():
    """From depth 1 the gate has a referent, so an inapplicability claim is a skipped measurement."""
    policy = _policy(incremental_enabled=True)
    reference = _reference(policy=policy)
    ledger = sf.open_lineage_ledger(reference)
    marker = sf.incremental_inapplicable(ledger)

    design = _HS(SEQ, (0.0, 0.0, 0.0))
    parent = sf.bind_immediate_parent(
        endpoint_id="endpoint:prior", head_score=_HS("ACDEFGHIKA", (0.0, 0.0, 0.0)),
        depth=1, policy=policy)
    depth1 = sf.LineageSafetyLedger(
        cumulative_reference=reference, depth=1, immediate_parent=parent)
    cumulative = sf.apply_threshold(
        sf.measure_cumulative(depth1, design, endpoint_id="endpoint:e2"),
        policy.cumulative_threshold)
    verdict = sf.make_admissibility_verdict(
        policy=policy, cumulative=cumulative, incremental=marker,
        structure_definitive=True, structure_evidence_level="definitive",
        constraints_preserved=True)
    with pytest.raises(V2Error):
        sf.advance_lineage(
            depth1, depth=2,
            selected=sf.bind_immediate_parent(
                endpoint_id="endpoint:e2", head_score=design, depth=2, policy=policy),
            admissibility=verdict)


def test_an_inapplicable_marker_from_another_lineage_is_refused():
    """It carries the cumulative reference it was minted under, so it cannot travel."""
    policy = _policy(incremental_enabled=True)
    mine = _reference(policy=policy)
    theirs = _reference(policy=policy, lineage_id="9L2Q_A:fam0")
    foreign = sf.incremental_inapplicable(sf.open_lineage_ledger(theirs))
    assert foreign.cumulative_reference_binding_id != mine.reference_binding_id

    ledger = sf.open_lineage_ledger(mine)
    cumulative = sf.apply_threshold(
        sf.measure_cumulative(ledger, _HS(SEQ, (0.0, 0.0, 0.0)), endpoint_id="endpoint:e1"),
        policy.cumulative_threshold)
    with pytest.raises(V2Error):
        sf.make_admissibility_verdict(
            policy=policy, cumulative=cumulative, incremental=foreign,
            structure_definitive=True, structure_evidence_level="definitive",
            constraints_preserved=True)


def test_incremental_gate_disabled_explicitly_allows_no_incremental_verdict():
    reference = _reference(policy=_policy(incremental_enabled=False))
    ledger = sf.open_lineage_ledger(reference)
    endpoint_id = "endpoint:e1"
    design = _HS(SEQ, (0.0, 0.0, 0.0))
    cumulative = sf.apply_threshold(
        sf.measure_cumulative(ledger, design, endpoint_id=endpoint_id),
        reference.policy.cumulative_threshold)
    admission = sf.make_admissibility_verdict(
        policy=reference.policy, cumulative=cumulative, incremental=None,
        structure_definitive=True, structure_evidence_level="definitive",
        constraints_preserved=True)
    assert admission.admitted is True
