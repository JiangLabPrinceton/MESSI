"""Shared constructors for V2 tests.

Not a test module.  ``LivePartialState`` / ``CompleteEndpoint`` / ``ScheduleBandTable`` each carry
20-30 cross-validated fields, so building one by hand in every suite is both noisy and a reliable
source of tests that fail for the wrong reason.  These helpers build the minimal LEGAL instance and
take keyword overrides, so a test perturbs exactly the one field it is about.

The world here is deliberately tiny: length 6, position 0 a hard anchor, positions 1-3 resolved
before the re-entry boundary, position 4 resolved AT/AFTER it (future information), position 5
unresolved.  That is the smallest shape that exercises every branch of the four-way support
partition.
"""

from __future__ import annotations

import dataclasses

from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
from inverse_folding.reference_flow.fusion_v2 import identity as ident
from inverse_folding.reference_flow.fusion_v2 import policy as pol
from inverse_folding.reference_flow.fusion_v2 import schedule as sch
from inverse_folding.reference_flow.fusion_v2 import state as st

D = "a" * 64
E = "b" * 64
MASK = 32
AA = frozenset(range(4, 24))
L = 6

R_STEP, C_SOURCE, C_NEXT, N_STEPS = 40, 50, 60, 100
EARLY_COMMITS = {1: 10, 2: 11, 3: 12}
LATE_COMMIT = 45

CANONICAL_AA20 = "ACDEFGHIKLMNPQRSTVWY"
#: The token -> residue map the runtime decodes completions with.  Kept HERE, once, because the
#: sequence below is derived from it: a hand-written pair of "these tokens" and "this sequence"
#: drifts silently, and a drifted pair is exactly the failure the projection kernel must catch --
#: bytes written back that the Head never scored.
ALPHABET = {token: CANONICAL_AA20[index] for index, token in enumerate(sorted(AA))}

ENDPOINT_TOKENS = (10, 20, 12, 13, 14, 15)
ENDPOINT_SEQ = "".join(ALPHABET[token] for token in ENDPOINT_TOKENS)
SEQ_MD5 = sequence_md5(ENDPOINT_SEQ)

TXN = "5ZHV_B:fam0:txn:d0:r40-c60:" + D[:12]
BAND_ID = "band:v2:step40"
STRATUM = "len4_8"


def digest(label: str) -> str:
    """A distinct, non-placeholder digest; BandProvenance rejects repeated-character strings."""
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


@dataclasses.dataclass(frozen=True)
class HeadWindow:
    start_0b: int
    end_0b: int
    k: int
    z: float


#: The toy Head's window width.  Fixed at 4 rather than "the whole sequence" so ONE evaluator
#: identity -- whose declared k-domain is part of that identity -- covers every length this suite
#: uses.  A grid whose k tracked the protein length would need a different evaluator per protein,
#: and two evaluators produce two incomparable score scales.
WINDOW_K = 4


def windows(length: int = L, z: float = -0.7) -> tuple[HeadWindow, ...]:
    """The window grid this toy world's Head produces at a given length.

    Every Head score, every binding and the safety reference must agree on it: the whole-landscape
    comparator aligns a design against the reference window by window, so two different grids are
    not two views of one landscape -- they are two landscapes, and their difference measures
    nothing."""
    return tuple(
        HeadWindow(start_0b=start, end_0b=start + WINDOW_K, k=WINDOW_K, z=float(z))
        for start in range(int(length) - WINDOW_K + 1)
    )


#: The grid for the default length-6 world.
WINDOWS = windows()


@dataclasses.dataclass(frozen=True)
class EndpointHeadScore:
    protein_id: str
    sequence_md5: str
    sequence_length: int
    allele: str
    score_scale: str
    windows: tuple[HeadWindow, ...]
    residue_hotspot: tuple[float, ...] | None
    global_risk: float | None


def conditioning(**over):
    """The run's frozen content provenance (PLAN §5.2), carried by value on every state.

    The two roles the projection kernel checks against -- the projection policy spec and the
    schedule-band calibration -- default to the ones the shared policy and band-table fixtures
    actually produce, so the default fixture describes a COHERENT run rather than one whose
    provenance names objects it never uses.  Both stay overridable, because "the answering policy
    is not the declared policy" is itself something the suite has to be able to express.
    """
    roles = {role: D for role in ident.OWN_CONDITIONING_FIELDS}
    roles["projection_policy_spec"] = digest("pspec")
    roles["schedule_band_calibration"] = default_band_calibration_digest()
    roles.update(over)
    return ident.V2ConditioningIdentity(
        base=ident.make_v2_conditioning(
            dplm_checkpoint=D, tokenizer=D, backbone_row=D, coordinate_mask=D,
            entry_config=D, fixed_token_policy=D,
        ),
        **roles,
    )


def head_binding(md5="0" * 32, length=L, protein_id="5ZHV_B"):
    evaluator = ident.HeadEvaluatorIdentity(
        allele="DRB1_0701", score_scale="nats", window_k_min=WINDOW_K, window_k_max=WINDOW_K,
        head_config_hash=D, head_checkpoint_digest=D,
    )
    return ident.HeadScoreBinding(
        protein_id=protein_id, sequence_md5=md5, sequence_length=int(length),
        window_grid_digest=ident.window_grid_digest(windows(length)), evaluator=evaluator,
    )


def safety_reference(length=L, protein_id="5ZHV_B"):
    """The depth-0 reference identity every ``LivePartialState`` stamps.

    STILL HAND-WRITTEN, and therefore still capable of disagreeing with the reference
    ``safety_gate()`` actually measures against: this declares ``sequence_md5="0"*32`` and
    ``reference_content_digest=D`` for a reference the gate binds at real digests.

    It cannot yet be read off the gate.  ``LivePartialState.__post_init__`` (``state.py:645``)
    requires ``conditioning.complete_reference == safety_reference.reference_content_digest``, and
    :func:`conditioning` declares every role as the length-blind placeholder ``D`` while a real
    content digest is per-length (the suite runs L=6 and LADDER_L=24).  Reconciling the two needs
    :func:`conditioning` to become length-aware and :func:`source` / :func:`endpoint` and the
    ladder suite's ``_run_kwargs`` to thread that length -- helpers outside this pass's scope.
    """
    return ident.SafetyReferenceBinding(
        reference_id="ref:wt", reference_label="wt_native", sequence_md5="0" * 32,
        sequence_length=int(length), reference_content_digest=D,
        head_binding=head_binding(length=length, protein_id=protein_id),
        head_score_digest=D, bound_at_depth=0, source_kind="predeclared_external",
    )


def provenance(token, *, kind, ref=st.FeedbackOriginRef.NONE, depth=0, step=0, txn=None,
               logprob=-1.5, n_events=1):
    origin = st.OriginEvidence(
        origin_kind=kind, origin_ref=ref, commit=sch.history_key(depth, step), token=token,
        evidence_logprob=logprob, transition_id=txn, evidence_digest=D,
    )
    return st.PositionProvenance(first_origin=origin, last_origin=origin, n_origin_events=n_events)


def lineage(depth=0, **over):
    kw = dict(protein_id="5ZHV_B", root_id="5ZHV_B:v2:d0:r0", family_id="fam0", depth=depth,
              parent_state_id=None, parent_transition_id=None, origin_endpoint_id=None)
    kw.update(over)
    return ident.LineageRef(**kw)


def source(**over):
    tokens = [10, 11, 12, 13, 14, MASK]
    kind = [st.ActiveOriginKind.HARD_ANCHOR] + [st.ActiveOriginKind.DENOISER_SAMPLE] * 4 + [
        st.ActiveOriginKind.UNRESOLVED]
    commit = ([None]
              + [sch.history_key(0, EARLY_COMMITS[i]) for i in (1, 2, 3)]
              + [sch.history_key(0, LATE_COMMIT), None])
    score = [None, -0.5, -0.6, -0.7, -0.8, None]
    status = ([st.ActiveScoreStatus.NOT_RANKED_ANCHOR]
              + [st.ActiveScoreStatus.HISTORICAL_NATURAL] * 4
              + [st.ActiveScoreStatus.MASKED])
    prov = ([provenance(tokens[0], kind=st.ActiveOriginKind.HARD_ANCHOR)]
            + [provenance(tokens[i], kind=st.ActiveOriginKind.DENOISER_SAMPLE,
                          step=EARLY_COMMITS[i]) for i in (1, 2, 3)]
            + [provenance(tokens[4], kind=st.ActiveOriginKind.DENOISER_SAMPLE, step=LATE_COMMIT)]
            + [provenance(MASK, kind=st.ActiveOriginKind.UNRESOLVED)])

    kw = dict(
        schema_version=ident.V2_STATE_SCHEMA_VERSION, lineage=lineage(),
        sampler_step=C_SOURCE, n_steps=N_STEPS,
        tokens=tuple(tokens), mask_token_id=MASK, aa_token_ids=AA,
        hard_anchors=((0, 10),), editable_positions=(1, 2, 3, 4, 5),
        active_origin_kind_by_pos=tuple(kind),
        feedback_origin_ref_by_pos=tuple([st.FeedbackOriginRef.NONE] * L),
        active_commit_depth_step_by_pos=tuple(commit),
        origin_transition_id_by_pos=tuple([None] * L),
        active_sampler_score_by_pos=tuple(score),
        active_score_status_by_pos=tuple(status),
        provenance_by_pos=tuple(prov),
        expired_protection=(),
        replay=st.ReplayIdentity(mode="identity", rng_state={"pos": 7}, fork_seed=None,
                                 replay_state_hash=D),
        accumulated_lineage_dfe=C_SOURCE,
        conditioning=conditioning(), safety_reference=safety_reference(),
        cost_event_ids=("evt:root",),
    )
    kw.update(over)
    return st.LivePartialState(**kw)


def endpoint(tokens=ENDPOINT_TOKENS, **over):
    """One legal endpoint of the default source.

    ``lineage=`` may be overridden to build a FOREIGN endpoint: the helper re-derives the protein,
    the Head binding and the source state from that lineage, so the result is internally legal and
    the only thing wrong with it is that it belongs to a different lineage than the source it will
    be handed to.  A foreign endpoint that failed its own constructor would test nothing.
    """
    lin = over.pop("lineage", None) or lineage()
    protein = lin.protein_id
    # The sequence is DERIVED from the tokens.  A helper that let a caller change one without the
    # other would hand out an endpoint whose residues are not the residues its own token vector
    # names -- the exact object the projection kernel exists to refuse.
    seq = "".join(ALPHABET[int(token)] for token in tokens)
    seq_md5 = sequence_md5(seq)
    windows_ = WINDOWS
    binding = ident.HeadScoreBinding(
        protein_id=protein, sequence_md5=seq_md5, sequence_length=L,
        window_grid_digest=ident.window_grid_digest(windows_),
        evaluator=safety_reference().head_binding.evaluator,
    )
    score = EndpointHeadScore(
        protein_id=protein, sequence_md5=seq_md5, sequence_length=L,
        allele=binding.evaluator.allele, score_scale=binding.evaluator.score_scale,
        windows=windows_, residue_hotspot=(-0.1,) * L, global_risk=-9.1,
    )
    src = source(lineage=lin, safety_reference=safety_reference(protein_id=protein))
    evidence = tuple(
        st.EndpointPositionEvidence(
            token=tok, commit=sch.history_key(0, 60 + i), completion_logprob=-2.5 - i,
            inherited_from_source=(tok == src.tokens[i]),
        )
        for i, tok in enumerate(tokens)
    )
    kw = dict(
        schema_version=ident.V2_STATE_SCHEMA_VERSION, lineage=lin, protein_id=protein,
        sequence=seq, sequence_md5=seq_md5, sequence_length=L,
        source_state_id=src.state_id, source_state_content_digest=src.content_digest,
        fork_index=0, fork_seed=4242,
        replay=st.ReplayIdentity(mode="fork", rng_state=None, fork_seed=4242,
                                 replay_state_hash=E),
        endpoint_provenance_evidence_by_pos=evidence,
        head_binding=binding, head_score=score, head_global_risk=-9.1,
        feasibility_level=st.FeasibilityLevel.DEFINITIVE,
        structure_outcome=StructureOutcome(feasible=True, metrics={"scTM": 0.91}),
        cost_event_ids=("evt:fork0",),
    )
    kw.update(over)
    return st.CompleteEndpoint(**kw)


def coords(**over):
    kw = dict(depth=0, r_step=R_STEP, c_source_step=C_SOURCE, c_next_step=C_NEXT,
              n_steps=N_STEPS, law=sch.CoordinateLaw.PROGRESSIVE_CHECKPOINT)
    kw.update(over)
    return sch.make_cycle(**kw)


def band_provenance(**over):
    kw = dict(
        schema_version=sch.SCHEDULE_BAND_SCHEMA_VERSION,
        calibration_scope=sch.SCHEDULE_BAND_SCOPE, calibration_id=BAND_ID,
        calibration_content_digest=digest("calib"), sampler_config_digest=digest("sampler"),
        tokenizer_digest=digest("tok"), backbone_digest=digest("bb"),
        coordinate_mask_policy_digest=digest("cmask"), constraint_manifest_digest=digest("cman"),
        constraint_stratum_id="anchored", cohort_digest=digest("cohort"),
        raw_attempts_digest=digest("raw"), attempted_seed_digest=digest("seeds"),
        seed_schema=sch.SCHEDULE_BAND_SEED_SCHEMA, code_revision="deadbeef",
        produced_by="scripts/rho_maturity_scan.py --mode step", n_steps=N_STEPS,
        base_form="linear", amplification_form="constant_one", remask_fraction_scale=0.0,
        head_free=True, n_attempted_seeds=32, n_failed_captures=0,
    )
    kw.update(over)
    return sch.BandProvenance(**kw)


def band(**over):
    kw = dict(
        step=R_STEP, stratum_key=STRATUM, levels=sch.QuantileLevels(levels=(0.1, 0.5, 0.9)),
        rho_quantiles=(0.4, 0.6, 0.8), unresolved_quantiles=(4, 2, 1),
        rho_accept=sch.BandInterval(lo=0.4, hi=0.8, lo_level=0.1, hi_level=0.9),
        unresolved_accept=sch.BandInterval(lo=1.0, hi=4.0, lo_level=0.9, hi_level=0.1),
        combination_rule="both_axes", n_attempts=32, n_captured=32,
        n_editable_min=4, n_editable_max=8,
    )
    kw.update(over)
    return sch.make_band(**kw)


def band_table(bands_override=None, **over):
    """``bands_override`` varies the band CONTENT while keeping the provenance identity, which is
    how a test distinguishes a digest bound to the calibration's name from one bound to its bytes."""
    return sch.make_band_table(provenance=band_provenance(**over),
                               bands=bands_override or (band(),))


def default_band_calibration_digest():
    """``make_band_table`` rebinds the digest to the table's own content, so it is READ, not named."""
    return band_table().provenance.calibration_content_digest


def policy_identity(**over):
    kw = dict(policy_id=pol.EXPLICIT_PROBE_POLICY_ID, policy_version="v0",
              policy_config_digest=digest("pcfg"), policy_spec_digest=digest("pspec"),
              is_diagnostic_only=True)
    kw.update(over)
    return ident.ProjectionPolicyIdentity(**kw)


#: 1 -> endpoint write, 4 -> source-feedback injection, 2 -> reopen, 3 and 5 -> ordinary carry.
REF_SETS = dict(write_from_endpoint=(1,), inject_from_source_feedback=(4,), reopen=(2,),
                carry_from_source=(3, 5))


def decision(**over):
    reasons = {
        1: pol.SupportReason.IMPROVEMENT_ASSOCIATED,
        4: pol.SupportReason.FUTURE_SOURCE_IDENTITY,
        2: pol.SupportReason.UNCERTAIN,
        3: pol.SupportReason.TEMPORALLY_VALID,
        5: pol.SupportReason.INHERITED_MASK,
    }
    kw = dict(**REF_SETS, reason_by_pos=reasons, policy=policy_identity())
    kw.update(over)
    return pol.PolicyDecision(**kw)


def declared_policy(**over):
    """What the run DECLARED its support policy to be (PLAN §2.5).

    The shared fixture runs the diagnostic probe, so it declares the probe and the one phase that
    authorises one -- any other pairing is unconstructible by design.
    """
    kw = dict(policy_id=pol.EXPLICIT_PROBE_POLICY_ID, policy_version="v0", is_diagnostic=True,
              phase="state_transition_canary")
    kw.update(over)
    return pol.DeclaredPolicy(**kw)


def project(**over):
    """Run the V2F2 kernel and return the ProjectionOutcome."""
    from inverse_folding.reference_flow.fusion_v2 import projection as proj

    table = over.pop("band_table", None) or band_table()
    kw = dict(
        source=source(), endpoint=endpoint(), endpoint_tokens=ENDPOINT_TOKENS,
        alphabet=ALPHABET,
        decision=decision(), declared_policy=declared_policy(), coordinates=coords(), band_table=table, stratum_key=STRATUM,
        descendant_fork_seed=9001, origin_transition_id=TXN, declared_band_id=BAND_ID,
        declared_band_digest=table.provenance.calibration_content_digest,
    )
    kw.update(over)
    return proj.source_writeback(**kw)


def projected(**over):
    """The committed ProjectedPartialState the V2F3 segment runs forward."""
    outcome = project(**over)
    assert outcome.committed, outcome.detail
    return outcome.projected


# --------------------------------------------------------------------------------------------
# the whole-landscape safety admission gate
# --------------------------------------------------------------------------------------------
#
# The gate needs a real ``V2Config``: ``bind_admission_policy`` derives the threshold, the Head
# domain and the frozen reference-content digest from it, and refuses anything else.  That is the
# point -- an admission policy assembled from loose numbers would be a policy nothing calibrated.

#: A WT reference distinct from anything the toy denoiser emits.  ``measure_cumulative`` refuses to
#: measure a design against itself, so the reference must not collide with a generated sequence.
REFERENCE_SEQ_6 = "WWYYWW"


def reference_sequence(length: int = L) -> str:
    seq = (REFERENCE_SEQ_6 * (int(length) // len(REFERENCE_SEQ_6) + 1))[:int(length)]
    return seq


def v2_config(*, cumulative_value=0.10, incremental_enabled=False, length=L):
    from inverse_folding.reference_flow.fusion_v2 import config as cfg
    from inverse_folding.reference_flow.fusion_v2 import safety as sf_mod

    def _calibrated(value, scope):
        """A threshold bound to a typed calibration artifact, not to a provenance label.

        Reused verbatim from the shape ``config.CalibratedScalar`` enforces: the source ref is
        DERIVED from the value, unit, source and artifact, so a number cannot be edited without
        also editing something that names it.
        """
        cumulative = scope == "cumulative"
        artifact = cfg.HotspotCalibrationArtifact(
            schema_version=cfg.V2_HOTSPOT_CALIBRATION_SCHEMA_VERSION,
            gate_kind=cfg.HOTSPOT_GATE_KIND,
            scope=(sf_mod.ReferenceKind.CUMULATIVE_DEPTH0 if cumulative
                   else sf_mod.ReferenceKind.IMMEDIATE_PARENT).value,
            reference_kind="native_wt" if cumulative else cfg.INCREMENTAL_REFERENCE_KIND,
            reference_label="wt_native" if cumulative else cfg.INCREMENTAL_REFERENCE_LABEL,
            window_domain=cfg.HOTSPOT_WINDOW_DOMAIN,
            allele="DRB1_0701", score_scale="nats",
            window_k_min=WINDOW_K, window_k_max=WINDOW_K,
            calibration_data_digest=D,
        )
        source_id = "v2-hotspot-calib-1" if cumulative else "v2-incremental-calib-1"
        return cfg.CalibratedScalar(
            value=float(value), unit="nats", source_kind="measured_calibration",
            source_id=source_id,
            source_ref=cfg.calibration_source_ref(
                value=float(value), unit="nats", source_kind="measured_calibration",
                source_id=source_id, artifact=artifact),
            artifact=artifact,
        )

    from inverse_folding.reference_flow.fusion_v2 import schedule as sch2

    return cfg.V2Config(
        schema_version=cfg.V2_CONFIG_SCHEMA_VERSION,
        identity=cfg.V2IdentityConfig(
            campaign_id="v2-canary", split_role="dev", phase="state_transition_canary",
            master_seed=20260805, seed_schema="v2seed-1", code_revision="deadbeef"),
        substrate=cfg.V2SubstrateConfig(
            n_steps=N_STEPS, temperature=1.0, amplification_form="constant_one",
            controller_enabled=False, remask_enabled=True, remask_fraction_scale=0.0,
            rf_config_label="v2_null_no_remask"),
        arm=cfg.V2ArmConfig(
            feedback_enabled=True, arm_role="v2", a2_matching_resource="definitive_refolds",
            a2_unmatched_reported=("gpu_seconds", "head_calls", "logical_dfe", "walltime_s")),
        schedule=cfg.V2ScheduleConfig(
            schedule_id="test", coordinate_law=sch2.CoordinateLaw.PROGRESSIVE_CHECKPOINT,
            depth_cap=1, active_population_width=1, min_lookahead_tail_steps=1,
            points=(cfg.DepthSchedulePoint(
                depth=0, r_step=R_STEP, c_source_step=C_SOURCE, c_next_step=C_NEXT,
                n_lookaheads=2, band_key="step40"),)),
        projection=cfg.V2ProjectionConfig(
            support_policy_id="explicit_probe", support_policy_version="v0",
            support_policy_is_diagnostic=True, temporal_history_rule="commit_before_reentry",
            assimilation_rule="first_forward_raw_logits",
            admissible_mask_load_unit="absolute_positions"),
        head=cfg.V2HeadConfig(
            allele="DRB1_0701", score_scale="nats",
            window_k_min=WINDOW_K, window_k_max=WINDOW_K),
        safety=cfg.V2SafetyConfig(
            cumulative_reference_kind="native_wt", cumulative_reference_label="wt_native",
            delta_new_cumulative=_calibrated(cumulative_value, "cumulative"),
            incremental_gate_enabled=incremental_enabled,
            delta_new_incremental=(_calibrated(0.10, "incremental")
                                   if incremental_enabled else None),
            structure_cadence="every_endpoint"),
        caps=cfg.V2CapsConfig(
            max_logical_dfe=10 ** 6, max_head_calls=10 ** 6, max_definitive_refolds=10 ** 6,
            max_gpu_seconds=10 ** 6, max_walltime_s=10 ** 6, max_retries=0,
            retry_scope="per_request"),
        content=(cfg.ContentIdentity(
            role="complete_reference_sequence", label="reference.fasta",
            binding="frozen", expected_sha256=D),),
    )


def safety_gate(*, length=L, reference_z=-0.7, cumulative_value=0.10, protein_id="5ZHV_B",
                incremental_enabled=False):
    """A bound admission gate for the toy world.

    ``reference_z`` is the knob that decides whether the gate BITES.  The toy Head scores every
    design at z=-0.7, so a reference at -0.7 leaves ``N_H = 0`` and every design is admissible,
    while a reference well below it makes every design a new hotspot.  Both regimes are needed: one
    to show the cycle still runs, the other to show the gate can actually stop it.

    The config's frozen ``complete_reference_sequence`` digest is DERIVED from the reference bytes
    this gate is then bound to, not typed independently: ``bind_cumulative_reference`` recomputes it
    and refuses a mismatch, so a fixture that named an unrelated digest would describe a run whose
    declared reference content is not the content it measures against.
    """
    import dataclasses as _dc

    from inverse_folding.reference_flow.fusion_v2 import safety as sf
    from inverse_folding.reference_flow.fusion_v2_runtime.admission import SafetyGate

    sequence = reference_sequence(length)
    content_digest = sf.reference_sequence_content_digest(sequence)
    base = v2_config(cumulative_value=cumulative_value, length=length,
                     incremental_enabled=incremental_enabled)
    config = _dc.replace(base, content=tuple(
        _dc.replace(row, expected_sha256=content_digest)
        if row.role == "complete_reference_sequence" else row
        for row in base.content
    ))
    policy = sf.bind_admission_policy(config, head_binding(length=length).evaluator)
    reference = sf.bind_cumulative_reference(
        lineage_id=f"{protein_id}:fam0", protein_id=protein_id, reference_label="wt_native",
        reference_sequence=sequence, reference_content_digest=content_digest, policy=policy,
        head_score=EndpointHeadScore(
            protein_id=protein_id, sequence_md5=sequence_md5(sequence),
            sequence_length=int(length), allele=policy.head_identity.allele,
            score_scale=policy.head_identity.score_scale,
            windows=windows(length, z=reference_z),
            residue_hotspot=(-0.1,) * int(length), global_risk=-1.0,
        ),
    )
    return SafetyGate(policy=policy, ledger=sf.open_lineage_ledger(reference))
