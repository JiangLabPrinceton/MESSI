"""V2F2: the pure source-coupled projection kernel ``q_phi`` and its policy boundary.

PLAN §2.4 defines the transition ``P~_{d+1} ~ q_phi(. | y*_d, P*_d, S, C, pi_d, r_d)`` and §2.5
defines the policy boundary.  This suite is written against the nine TDD acceptance bullets of
PLAN task V2F2, and every test names the bullet it discharges.

The load-bearing idea: the kernel is FROZEN and the support-scoring policy is NOT.  A policy may
only hand back typed support sets and reason evidence; every invariant that protects the science
is enforced by the kernel regardless of what the policy asked for.  An invalid policy result must
therefore become a typed NULL EVENT, never an exception and never an invented state.
"""

from __future__ import annotations

import dataclasses

import pytest

from inverse_folding.reference_flow.fusion_v2 import identity as ident
from inverse_folding.reference_flow.fusion_v2 import policy as pol
from inverse_folding.reference_flow.fusion_v2 import projection as proj
from inverse_folding.reference_flow.fusion_v2 import schedule as sch
from inverse_folding.reference_flow.fusion_v2 import state as st
from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion_v2.errors import V2Error
from tests.inverse_folding import _v2_fixtures as F

D = "a" * 64
E = "b" * 64
MASK = 32
AA = frozenset(range(4, 24))
L = 6

#: The token -> residue map the runtime decodes completions with, and the endpoint sequence DERIVED
#: from it.  Writing the two independently is how a token vector and the sequence the Head scored
#: drift apart, which is exactly what the kernel now refuses.
CANONICAL_AA20 = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET = {token: CANONICAL_AA20[index] for index, token in enumerate(sorted(AA))}


@dataclasses.dataclass(frozen=True)
class _HeadWindow:
    start_0b: int
    end_0b: int
    k: int
    z: float


@dataclasses.dataclass(frozen=True)
class _EndpointHeadScore:
    protein_id: str
    sequence_md5: str
    sequence_length: int
    allele: str
    score_scale: str
    windows: tuple[_HeadWindow, ...]
    residue_hotspot: tuple[float, ...] | None
    global_risk: float | None

#: Cycle coordinates shared by the whole suite: re-entry at 40, source at 50, next capture at 60.
R_STEP, C_SOURCE, C_NEXT, N_STEPS = 40, 50, 60, 100

#: Source commit steps.  Positions 1-3 committed BEFORE r_step (ordinary carry is legal);
#: position 4 committed AT/AFTER r_step (future information -- ordinary carry must fail closed).
EARLY_COMMITS = {1: 10, 2: 11, 3: 12}
LATE_COMMIT = 45

TXN = "5ZHV_B:fam0:txn:d0:r40-c60:" + D[:12]


# --------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------


def _conditioning(**over):
    """The run's frozen content provenance (PLAN §5.2), carried by value on every state.

    ``projection_policy_spec`` and ``schedule_band_calibration`` are overridable because they are
    what the kernel now checks the answering policy and the supplied band table against.
    """
    roles = {role: D for role in ident.OWN_CONDITIONING_FIELDS}
    # The default fixture describes a COHERENT run: the frozen band calibration is the digest of
    # the table these tests actually hand the kernel.  Leaving the two unrelated would make every
    # ordinary projection fail on the provenance check rather than on what it is testing.
    roles["schedule_band_calibration"] = _default_band_calibration_digest()
    roles.update(over)
    return ident.V2ConditioningIdentity(
        base=ident.make_v2_conditioning(
            dplm_checkpoint=D, tokenizer=D, backbone_row=D, coordinate_mask=D,
            entry_config=D, fixed_token_policy=D,
        ),
        **roles,
    )


def _head_binding(md5="0" * 32):
    """A Head score binding.  It is bound to a SPECIFIC sequence: the endpoint's binding must
    describe the endpoint, while the depth-0 safety reference describes the reference."""
    evaluator = ident.HeadEvaluatorIdentity(
        allele="DRB1_0701", score_scale="nats", window_k_min=13, window_k_max=25,
        head_config_hash=D, head_checkpoint_digest=D,
    )
    return ident.HeadScoreBinding(
        protein_id="5ZHV_B", sequence_md5=md5, sequence_length=L,
        window_grid_digest=D, evaluator=evaluator,
    )


def _safety_reference():
    return ident.SafetyReferenceBinding(
        reference_id="ref:wt", reference_label="wt_native", sequence_md5="0" * 32,
        sequence_length=L, reference_content_digest=D, head_binding=_head_binding(),
        head_score_digest=D, bound_at_depth=0, source_kind="predeclared_external",
    )


def _provenance(token, *, kind, ref=st.FeedbackOriginRef.NONE, depth=0, step=0, txn=None,
                logprob=-1.5, n_events=1):
    origin = st.OriginEvidence(
        origin_kind=kind, origin_ref=ref, commit=sch.history_key(depth, step), token=token,
        evidence_logprob=logprob, transition_id=txn, evidence_digest=D,
    )
    return st.PositionProvenance(first_origin=origin, last_origin=origin, n_origin_events=n_events)


def _lineage(depth=0, **over):
    kw = dict(protein_id="5ZHV_B", root_id="5ZHV_B:v2:d0:r0", family_id="fam0", depth=depth,
              parent_state_id=None, parent_transition_id=None, origin_endpoint_id=None)
    kw.update(over)
    return ident.LineageRef(**kw)


def _source(**over):
    """The live source state at ``c_source_step``.

    Position 0 is a hard anchor, 1-3 are resolved with commits before ``r_step``, 4 is resolved
    with a commit at ``LATE_COMMIT`` (>= r_step), and 5 is unresolved.
    """
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
    prov = ([_provenance(tokens[0], kind=st.ActiveOriginKind.HARD_ANCHOR)]
            + [_provenance(tokens[i], kind=st.ActiveOriginKind.DENOISER_SAMPLE,
                           step=EARLY_COMMITS[i]) for i in (1, 2, 3)]
            + [_provenance(tokens[4], kind=st.ActiveOriginKind.DENOISER_SAMPLE, step=LATE_COMMIT)]
            + [_provenance(MASK, kind=st.ActiveOriginKind.UNRESOLVED)])

    kw = dict(
        schema_version=ident.V2_STATE_SCHEMA_VERSION,
        lineage=_lineage(),
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
        conditioning=_conditioning(),
        safety_reference=_safety_reference(),
        cost_event_ids=("evt:root",),
    )
    kw.update(over)
    return st.LivePartialState(**kw)


#: The endpoint's token at every position.  Position 1 differs from the source (10 -> 20), which
#: is what makes ``write_from_endpoint`` a real byte change rather than a relabelling.
ENDPOINT_TOKENS = (10, 20, 12, 13, 14, 15)
ENDPOINT_SEQ = "".join(ALPHABET[token] for token in ENDPOINT_TOKENS)
SEQ_MD5 = sequence_md5(ENDPOINT_SEQ)


def _endpoint(tokens=ENDPOINT_TOKENS, **over):
    # Derived, never written alongside: see the note on ENDPOINT_SEQ.
    sequence = "".join(ALPHABET[int(token)] for token in tokens)
    seq_md5 = sequence_md5(sequence)
    windows = (_HeadWindow(start_0b=0, end_0b=L, k=L, z=-0.7),)
    head_binding = ident.HeadScoreBinding(
        protein_id="5ZHV_B", sequence_md5=seq_md5, sequence_length=L,
        window_grid_digest=ident.window_grid_digest(windows),
        evaluator=_safety_reference().head_binding.evaluator,
    )
    head_score = _EndpointHeadScore(
        protein_id="5ZHV_B", sequence_md5=seq_md5, sequence_length=L,
        allele=head_binding.evaluator.allele, score_scale=head_binding.evaluator.score_scale,
        windows=windows, residue_hotspot=(-0.1,) * L, global_risk=-9.1,
    )
    source = _source()
    evidence = tuple(
        st.EndpointPositionEvidence(
            token=tok, commit=sch.history_key(0, 60 + i),
            # A terminal completion log-probability.  PLAN §2.4: this may never become an active
            # sampler score.  The positive control in this suite depends on that number existing.
            completion_logprob=-2.5 - i, inherited_from_source=(tok == source.tokens[i]),
        )
        for i, tok in enumerate(tokens)
    )
    kw = dict(
        schema_version=ident.V2_STATE_SCHEMA_VERSION,
        lineage=_lineage(),
        protein_id="5ZHV_B",
        sequence=sequence, sequence_md5=seq_md5, sequence_length=L,
        source_state_id=source.state_id,
        source_state_content_digest=source.content_digest,
        fork_index=0, fork_seed=4242,
        replay=st.ReplayIdentity(mode="fork", rng_state=None, fork_seed=4242,
                                 replay_state_hash=E),
        endpoint_provenance_evidence_by_pos=evidence,
        head_binding=head_binding,
        head_score=head_score,
        head_global_risk=-9.1,
        feasibility_level=st.FeasibilityLevel.DEFINITIVE,
        structure_outcome=StructureOutcome(feasible=True, metrics={"scTM": 0.91}),
        cost_event_ids=("evt:fork0",),
    )
    kw.update(over)
    return st.CompleteEndpoint(**kw)


def _coords(**over):
    kw = dict(depth=0, r_step=R_STEP, c_source_step=C_SOURCE, c_next_step=C_NEXT,
              n_steps=N_STEPS, law=sch.CoordinateLaw.PROGRESSIVE_CHECKPOINT)
    kw.update(over)
    return sch.make_cycle(**kw)


BAND_ID = "band:v2:step40"


def _digest(label: str) -> str:
    """A distinct, non-placeholder digest.  BandProvenance rejects repeated-character strings."""
    import hashlib
    return hashlib.sha256(label.encode()).hexdigest()


def _band_provenance(**over):
    kw = dict(
        schema_version=sch.SCHEDULE_BAND_SCHEMA_VERSION,
        calibration_scope=sch.SCHEDULE_BAND_SCOPE, calibration_id=BAND_ID,
        calibration_content_digest=_digest("calib"), sampler_config_digest=_digest("sampler"),
        tokenizer_digest=_digest("tok"), backbone_digest=_digest("bb"),
        coordinate_mask_policy_digest=_digest("cmask"),
        constraint_manifest_digest=_digest("cman"),
        constraint_stratum_id="anchored", cohort_digest=_digest("cohort"),
        raw_attempts_digest=_digest("raw"),
        attempted_seed_digest=_digest("seeds"), seed_schema=sch.SCHEDULE_BAND_SEED_SCHEMA,
        code_revision="deadbeef",
        produced_by="scripts/rho_maturity_scan.py --mode step", n_steps=N_STEPS,
        base_form="linear", amplification_form="constant_one", remask_fraction_scale=0.0,
        head_free=True, n_attempted_seeds=32, n_failed_captures=0,
    )
    kw.update(over)
    return sch.BandProvenance(**kw)


def _band(**over):
    kw = dict(
        step=R_STEP, stratum_key="len4_8",
        levels=sch.QuantileLevels(levels=(0.1, 0.5, 0.9)),
        rho_quantiles=(0.4, 0.6, 0.8), unresolved_quantiles=(4, 2, 1),
        rho_accept=sch.BandInterval(lo=0.4, hi=0.8, lo_level=0.1, hi_level=0.9),
        unresolved_accept=sch.BandInterval(lo=1.0, hi=4.0, lo_level=0.9, hi_level=0.1),
        combination_rule="both_axes", n_attempts=32, n_captured=32,
        n_editable_min=4, n_editable_max=8,
    )
    kw.update(over)
    return sch.make_band(**kw)


def _band_table(**over):
    return sch.make_band_table(provenance=_band_provenance(**over), bands=(_band(),))


def _default_band_calibration_digest():
    """``make_band_table`` rebinds the digest to the table's own content, so it is READ, not named."""
    return _band_table().provenance.calibration_content_digest


def _policy_identity(**over):
    kw = dict(policy_id=pol.EXPLICIT_PROBE_POLICY_ID, policy_version="v0", policy_config_digest=D,
              policy_spec_digest=D, is_diagnostic_only=True)
    kw.update(over)
    return ident.ProjectionPolicyIdentity(**kw)


def _declared_policy(**over):
    """What the RUN declared, as the config emits it -- the thing the answer is matched against.

    The default suite runs the diagnostic probe, so the declaration names the probe and the phase
    that authorises one.  A production declaration is built explicitly by the tests that need it.
    """
    kw = dict(policy_id=pol.EXPLICIT_PROBE_POLICY_ID, policy_version="v0", is_diagnostic=True,
              phase="state_transition_canary")
    kw.update(over)
    return pol.DeclaredPolicy(**kw)


#: The reference partition.  Disjoint and exhaustive over editable positions (1,2,3,4,5):
#:   1 -> endpoint write (source 10 -> endpoint 20, a real byte change)
#:   4 -> source-feedback injection (its commit at 45 is >= r_step, so it cannot carry)
#:   2 -> reopen (newly masks a previously RESOLVED position)
#:   3 -> ordinary carry (commit at 12 < r_step)
#:   5 -> ordinary carry of an inherited mask
REF_SETS = dict(
    write_from_endpoint=(1,),
    inject_from_source_feedback=(4,),
    reopen=(2,),
    carry_from_source=(3, 5),
)


def _decision(**over):
    reasons = {
        1: pol.SupportReason.IMPROVEMENT_ASSOCIATED,
        4: pol.SupportReason.FUTURE_SOURCE_IDENTITY,
        2: pol.SupportReason.UNCERTAIN,
        3: pol.SupportReason.TEMPORALLY_VALID,
        5: pol.SupportReason.INHERITED_MASK,
    }
    kw = dict(**REF_SETS, reason_by_pos=reasons, policy=_policy_identity())
    kw.update(over)
    return pol.PolicyDecision(**kw)


def _project_kwargs(**over):
    # make_band_table rebinds calibration_content_digest to the table's own content, so the
    # declared digest must be READ OFF the table rather than guessed.
    table = over.pop("band_table", None) or _band_table()
    kw = dict(
        source=_source(), endpoint=_endpoint(),
        endpoint_tokens=ENDPOINT_TOKENS,
        alphabet=ALPHABET, decision=_decision(), coordinates=_coords(), band_table=table,
        stratum_key="len4_8", descendant_fork_seed=9001, origin_transition_id=TXN,
        declared_band_id=BAND_ID,
        declared_band_digest=table.provenance.calibration_content_digest,
        declared_policy=_declared_policy(),
    )
    kw.update(over)
    return kw


def _project(**over):
    return proj.source_writeback(**_project_kwargs(**over))


# --------------------------------------------------------------------------------------------
# the kernel produces a legal projected state at all
# --------------------------------------------------------------------------------------------


def test_the_reference_projection_commits():
    out = _project()
    assert out.outcome is st.TransitionOutcome.COMMITTED
    assert out.committed is True
    assert isinstance(out.projected, st.ProjectedPartialState)
    assert out.projected.r_step == R_STEP
    assert out.projected.c_next_step == C_NEXT


def test_the_projected_state_advances_lineage_depth_by_exactly_one():
    out = _project()
    assert out.projected.lineage.depth == _source().lineage.depth + 1
    assert out.projected.lineage.parent_state_id == _source().state_id


def test_projection_is_pure_and_leaves_the_source_untouched():
    """PLAN §2.4: the kernel 'operates on a copy'."""
    source = _source()
    before = dataclasses.asdict(source)
    _project(source=source)
    assert dataclasses.asdict(source) == before


# --------------------------------------------------------------------------------------------
# bullet 1 - hard anchors never change or reopen
# --------------------------------------------------------------------------------------------


def test_hard_anchors_are_byte_identical_before_and_after_projection():
    out = _project()
    assert out.projected.hard_anchors == _source().hard_anchors
    for position, token in _source().hard_anchors:
        assert out.projected.tokens[position] == token


@pytest.mark.parametrize("action", sorted(REF_SETS))
def test_no_support_action_may_name_a_hard_anchor(action):
    """A policy that tries to act on position 0 must produce a typed null, not a state."""
    sets = dict(REF_SETS)
    sets[action] = tuple(sorted(set(sets[action]) | {0}))
    out = _project(decision=_decision(**sets))
    assert out.outcome is st.TransitionOutcome.NULL_INVALID_POLICY_RESULT
    assert out.projected is None


def test_an_anchor_can_never_be_temporarily_protected():
    out = _project()
    protected = {p.position for p in out.projected.active_temporary_protection}
    assert protected.isdisjoint({position for position, _ in _source().hard_anchors})


# --------------------------------------------------------------------------------------------
# bullet 2 - a source token committed before r_d carries byte/history/active score exactly
# --------------------------------------------------------------------------------------------


def test_an_ordinary_carry_preserves_byte_history_and_active_score_exactly():
    source, out = _source(), _project()
    position = 3
    assert out.projected.tokens[position] == source.tokens[position]
    assert out.projected.active_sampler_score_by_pos[position] == \
        source.active_sampler_score_by_pos[position]
    assert out.projected.active_score_status_by_pos[position] is \
        source.active_score_status_by_pos[position]
    assert out.projected.active_commit_depth_step_by_pos[position] == \
        source.active_commit_depth_step_by_pos[position]
    assert out.projected.active_origin_kind_by_pos[position] is \
        source.active_origin_kind_by_pos[position]
    assert out.projected.provenance_by_pos[position] == source.provenance_by_pos[position]


def test_a_carried_inherited_mask_stays_canonically_masked():
    out = _project()
    position = 5
    assert out.projected.tokens[position] == MASK
    assert out.projected.active_origin_kind_by_pos[position] is st.ActiveOriginKind.UNRESOLVED
    assert out.projected.active_score_status_by_pos[position] is st.ActiveScoreStatus.MASKED
    assert out.projected.active_sampler_score_by_pos[position] is None


def test_a_natural_carry_receives_no_implicit_protection():
    """PLAN §2.4: 'Natural source carries receive no implicit protection.'"""
    out = _project()
    protected = {p.position for p in out.projected.active_temporary_protection}
    assert protected.isdisjoint(REF_SETS["carry_from_source"])


# --------------------------------------------------------------------------------------------
# bullet 3 - a token committed at or after r_d fails ordinary carry
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("commit_step", [R_STEP, R_STEP + 1, C_SOURCE - 1])
def test_carrying_a_token_committed_at_or_after_reentry_fails_closed(commit_step):
    """Position 4 is the future-information position; asking to carry it must null the event."""
    source = _source()
    commits = list(source.active_commit_depth_step_by_pos)
    commits[4] = sch.history_key(0, commit_step)
    provenance = list(source.provenance_by_pos)
    provenance[4] = _provenance(source.tokens[4], kind=st.ActiveOriginKind.DENOISER_SAMPLE,
                                step=commit_step)
    mutated = _source(active_commit_depth_step_by_pos=tuple(commits),
                      provenance_by_pos=tuple(provenance))
    sets = dict(REF_SETS)
    sets["carry_from_source"] = tuple(sorted(set(sets["carry_from_source"]) | {4}))
    sets["inject_from_source_feedback"] = ()
    out = _project(source=mutated, decision=_decision(**sets))
    assert out.outcome is st.TransitionOutcome.NULL_INVALID_POLICY_RESULT
    # Pin WHICH layer rejected.  The V2F1 state layer independently refuses an illegal carry, so
    # asserting only the outcome cannot tell the kernel's own gate from that fallback -- and a
    # deleted kernel gate would stay green.
    assert "re-entry boundary" in out.detail


def test_a_token_committed_before_reentry_may_be_carried():
    """The complement of the previous test: the gate is not vacuously rejecting everything."""
    source = _source()
    commits = list(source.active_commit_depth_step_by_pos)
    commits[4] = sch.history_key(0, R_STEP - 1)
    provenance = list(source.provenance_by_pos)
    provenance[4] = _provenance(source.tokens[4], kind=st.ActiveOriginKind.DENOISER_SAMPLE,
                                step=R_STEP - 1)
    mutated = _source(active_commit_depth_step_by_pos=tuple(commits),
                      provenance_by_pos=tuple(provenance))
    sets = dict(REF_SETS)
    sets["carry_from_source"] = tuple(sorted(set(sets["carry_from_source"]) | {4}))
    sets["inject_from_source_feedback"] = ()
    out = _project(source=mutated, decision=_decision(**sets))
    assert out.outcome is st.TransitionOutcome.COMMITTED


def test_a_future_committed_token_is_legal_as_an_explicit_source_feedback_injection():
    """PLAN §2.4: it 'must be assigned to inject_from_source_feedback or reopen'."""
    out = _project()
    assert out.outcome is st.TransitionOutcome.COMMITTED
    assert 4 in out.projected.support.inject_from_source_feedback


# --------------------------------------------------------------------------------------------
# bullets 4 & 5 - injected identities keep immutable provenance and begin without a score
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("position", [1, 4])
def test_an_injected_identity_begins_pending_with_no_fabricated_active_score(position):
    out = _project()
    assert out.projected.active_score_status_by_pos[position] is \
        st.ActiveScoreStatus.PENDING_ASSIMILATION
    assert out.projected.active_sampler_score_by_pos[position] is None
    assert out.projected.active_origin_kind_by_pos[position] is \
        st.ActiveOriginKind.FEEDBACK_INJECTION


def test_the_endpoint_completion_logprob_never_becomes_an_active_sampler_score():
    """PLAN §2.4 bullet 5, the evidence-namespace firewall."""
    endpoint = _endpoint()
    terminal = {e.completion_logprob for e in endpoint.endpoint_provenance_evidence_by_pos}
    out = _project(endpoint=endpoint)
    active = {s for s in out.projected.active_sampler_score_by_pos if s is not None}
    assert active.isdisjoint(terminal)


def test_the_endpoint_completion_logprob_survives_as_immutable_provenance():
    out = _project()
    endpoint = _endpoint()
    evidence = out.projected.provenance_by_pos[1].last_origin
    assert evidence.evidence_logprob == \
        endpoint.endpoint_provenance_evidence_by_pos[1].completion_logprob


@pytest.mark.parametrize("position,expected_ref", [
    (1, st.FeedbackOriginRef.SELECTED_ENDPOINT),
    (4, st.FeedbackOriginRef.SOURCE_STATE),
])
def test_each_injection_action_pins_its_own_origin_ref(position, expected_ref):
    out = _project()
    assert out.projected.feedback_origin_ref_by_pos[position] is expected_ref


@pytest.mark.parametrize("position", [1, 4])
def test_every_injection_sets_its_active_commit_to_depth_plus_one_at_reentry(position):
    """PLAN §2.4: 'Every endpoint/source-feedback injection sets its active commit event to
    (depth+1, r_d)' -- this is what stops a stationary cycle from aliasing two events."""
    out = _project()
    assert out.projected.active_commit_depth_step_by_pos[position] == \
        sch.history_key(_source().lineage.depth + 1, R_STEP)


@pytest.mark.parametrize("position", [1, 4])
def test_an_injection_preserves_the_original_first_origin_and_counts_the_event(position):
    """PLAN §2.4: a recursively carried token must not lose its original ancestry."""
    source, out = _source(), _project()
    assert out.projected.provenance_by_pos[position].first_origin == \
        source.provenance_by_pos[position].first_origin
    assert out.projected.provenance_by_pos[position].n_origin_events == \
        source.provenance_by_pos[position].n_origin_events + 1


def test_the_later_source_evidence_of_an_injection_is_retained_only_as_provenance():
    """Position 4's commit at step 45 is future information: it may survive as evidence but must
    not remain the ACTIVE commit."""
    out = _project()
    active = out.projected.active_commit_depth_step_by_pos[4]
    assert active != sch.history_key(0, LATE_COMMIT)
    assert active == sch.history_key(1, R_STEP)


# --------------------------------------------------------------------------------------------
# bullet 6 - reopened state is canonical
# --------------------------------------------------------------------------------------------


def test_a_reopened_position_is_canonically_masked():
    out = _project()
    position = 2
    assert out.projected.tokens[position] == MASK
    assert out.projected.active_origin_kind_by_pos[position] is st.ActiveOriginKind.UNRESOLVED
    assert out.projected.feedback_origin_ref_by_pos[position] is st.FeedbackOriginRef.NONE
    assert out.projected.active_score_status_by_pos[position] is st.ActiveScoreStatus.MASKED
    assert out.projected.active_sampler_score_by_pos[position] is None
    assert out.projected.active_commit_depth_step_by_pos[position] is None
    assert out.projected.origin_transition_id_by_pos[position] is None


def test_reopen_must_newly_mask_a_previously_resolved_position():
    """Relabelling an already-masked position as 'reopen' newly masks nothing."""
    sets = dict(REF_SETS)
    sets["reopen"] = (5,)                                   # 5 is already masked in the source
    sets["carry_from_source"] = (2, 3)
    out = _project(decision=_decision(**sets))
    assert out.outcome is st.TransitionOutcome.NULL_INVALID_POLICY_RESULT
    # As above: ProjectedPartialState also enforces reopen <= source_resolved_positions, so the
    # detail is what proves the kernel rejected it before construction.
    assert "newly masks nothing" in out.detail


def test_reopen_may_be_empty_when_the_state_stays_partial():
    sets = dict(REF_SETS)
    sets["reopen"] = ()
    sets["carry_from_source"] = (2, 3, 5)
    out = _project(decision=_decision(**sets))
    assert out.outcome is st.TransitionOutcome.COMMITTED
    assert out.projected.support.reopen == ()
    assert out.projected.realized_maturity.n_unresolved_editable == 1


def test_q_phi_refuses_a_fully_resolved_state_owned_by_the_cycle_fallback():
    endpoint_tokens = (10, 11, 12, 13, 14, 15)
    decision = _decision(
        write_from_endpoint=(5,), inject_from_source_feedback=(4,), reopen=(),
        carry_from_source=(1, 2, 3),
        reason_by_pos={
            1: pol.SupportReason.TEMPORALLY_VALID,
            2: pol.SupportReason.TEMPORALLY_VALID,
            3: pol.SupportReason.TEMPORALLY_VALID,
            4: pol.SupportReason.FUTURE_SOURCE_IDENTITY,
            5: pol.SupportReason.IMPROVEMENT_ASSOCIATED,
        },
    )
    out = _project(
        endpoint=_endpoint(tokens=endpoint_tokens), endpoint_tokens=endpoint_tokens,
        decision=decision,
    )
    assert out.outcome is st.TransitionOutcome.NULL_INVALID_POLICY_RESULT
    assert out.projected is None
    assert "terminal best-lookahead fallback is owned by the cycle" in out.detail


def test_write_from_endpoint_may_not_be_empty():
    sets = dict(REF_SETS)
    sets["write_from_endpoint"] = ()
    sets["carry_from_source"] = (1, 3, 5)
    out = _project(decision=_decision(**sets))
    assert out.outcome is st.TransitionOutcome.NULL_INVALID_POLICY_RESULT


# --------------------------------------------------------------------------------------------
# the four-way partition
# --------------------------------------------------------------------------------------------


def test_the_four_sets_must_be_disjoint():
    sets = dict(REF_SETS)
    sets["reopen"] = (2, 3)                                 # 3 is already a carry
    out = _project(decision=_decision(**sets))
    assert out.outcome is st.TransitionOutcome.NULL_INVALID_POLICY_RESULT


def test_the_four_sets_must_be_exhaustive_over_editable_positions():
    sets = dict(REF_SETS)
    sets["carry_from_source"] = (3,)                        # drops position 5
    out = _project(decision=_decision(**sets))
    assert out.outcome is st.TransitionOutcome.NULL_INVALID_POLICY_RESULT


def test_a_support_set_may_not_name_a_position_outside_the_editable_domain():
    sets = dict(REF_SETS)
    sets["reopen"] = (2, 99)
    out = _project(decision=_decision(**sets))
    assert out.outcome is st.TransitionOutcome.NULL_INVALID_POLICY_RESULT


def test_source_feedback_injection_must_act_on_a_source_resolved_position():
    """An injection over a masked source position would resolve a mask that the mask-load
    arithmetic (u_proj = u_src - a + b_new) does not count, silently breaking the band."""
    sets = dict(REF_SETS)
    sets["inject_from_source_feedback"] = (4, 5)
    sets["carry_from_source"] = (3,)
    out = _project(decision=_decision(**sets))
    assert out.outcome is st.TransitionOutcome.NULL_INVALID_POLICY_RESULT
    # The per-position algebra would also refuse a FEEDBACK_INJECTION holding a mask token, so the
    # detail is what proves the kernel rejected it on the mask-load ground the PLAN names.
    assert "mask-load identity" in out.detail


# --------------------------------------------------------------------------------------------
# temporary protection is a separate typed relation with exclusive expiry
# --------------------------------------------------------------------------------------------


def test_the_protected_set_is_exactly_the_two_injection_actions():
    out = _project()
    protected = {p.position for p in out.projected.active_temporary_protection}
    assert protected == set(REF_SETS["write_from_endpoint"]) | \
        set(REF_SETS["inject_from_source_feedback"])


def test_both_injection_kinds_receive_the_same_exclusive_expiry():
    out = _project()
    for protection in out.projected.active_temporary_protection:
        assert protection.expiry_step == C_NEXT
        assert protection.granted_at_step == R_STEP
        assert protection.granted_at_depth == _source().lineage.depth + 1
        assert protection.granted_by_transition_id == TXN


@pytest.mark.parametrize("position,reason", [
    (1, st.ProtectionGrantReason.ENDPOINT_INJECTION),
    (4, st.ProtectionGrantReason.SOURCE_FEEDBACK_INJECTION),
])
def test_the_protection_reason_names_the_action_that_earned_it(position, reason):
    out = _project()
    by_position = {p.position: p for p in out.projected.active_temporary_protection}
    assert by_position[position].grant_reason is reason


# --------------------------------------------------------------------------------------------
# bullet 7 - same inputs/seed produce identical state regardless of row order
# --------------------------------------------------------------------------------------------


def test_the_projected_state_is_invariant_to_support_set_row_order():
    shuffled = _decision(
        write_from_endpoint=(1,),
        inject_from_source_feedback=(4,),
        reopen=(2,),
        carry_from_source=(5, 3),                            # reversed
    )
    assert _project().projected.content_digest == \
        _project(decision=shuffled).projected.content_digest


def test_the_projected_state_is_invariant_to_reason_mapping_order():
    reasons = dict(reversed(list(_decision().reason_by_pos.items())))
    assert _project().projected.content_digest == \
        _project(decision=_decision(reason_by_pos=reasons)).projected.content_digest


def test_the_same_inputs_produce_a_byte_identical_state_id():
    assert _project().projected.state_id == _project().projected.state_id


# --------------------------------------------------------------------------------------------
# bullets 8 & 9 - POSITIVE CONTROLS: the kernel must actually read source and endpoint
# --------------------------------------------------------------------------------------------


def test_changing_the_endpoint_byte_changes_the_projected_state():
    """Endpoint dependence with fixed support.  A kernel that ignored the endpoint would produce
    the same state for a different endpoint, and this test is what catches it."""
    other = tuple([*ENDPOINT_TOKENS[:1], 21, *ENDPOINT_TOKENS[2:]])
    baseline = _project()
    changed = _project(endpoint=_endpoint(tokens=other), endpoint_tokens=other)
    assert changed.projected.tokens[1] == 21
    assert changed.projected.content_digest != baseline.projected.content_digest


def test_changing_a_carried_source_byte_changes_the_projected_state():
    """Source dependence with fixed support."""
    source = _source()
    tokens = list(source.tokens)
    tokens[3] = 19
    provenance = list(source.provenance_by_pos)
    provenance[3] = _provenance(19, kind=st.ActiveOriginKind.DENOISER_SAMPLE,
                                step=EARLY_COMMITS[3])
    mutated = _source(tokens=tuple(tokens), provenance_by_pos=tuple(provenance))
    changed = _project(source=mutated)
    assert changed.projected.tokens[3] == 19
    assert changed.projected.content_digest != _project().projected.content_digest


def test_an_endpoint_blind_write_is_detectable_because_the_written_byte_is_the_endpoints():
    """The endpoint token at position 1 (20) differs from the source token (10).  A kernel that
    wrote the SOURCE byte at an endpoint-write position would leave 10 here."""
    out = _project()
    assert out.projected.tokens[1] == ENDPOINT_TOKENS[1]
    assert out.projected.tokens[1] != _source().tokens[1]


def test_the_written_bytes_agree_with_the_endpoint_at_every_write_position():
    """Delegates to the V2F1 cross-check, which is the machine-verifiable form of bullet 9."""
    out = _project()
    st.validate_endpoint_writeback(projected=out.projected, endpoint_tokens=ENDPOINT_TOKENS)


# --------------------------------------------------------------------------------------------
# the schedule-band / mask-load gate
# --------------------------------------------------------------------------------------------


def test_the_realized_mask_load_matches_the_coupled_identity():
    """u_proj = u_src - a + b_new: 1 unresolved source position, 0 endpoint writes over a mask,
    1 newly reopened -> 2 unresolved editable positions in the projection."""
    out = _project()
    unresolved = [p for p in out.projected.editable_positions
                  if out.projected.tokens[p] == MASK]
    assert len(unresolved) == 2


def test_a_projection_outside_the_unresolved_band_is_band_incompatible():
    narrow = _band(unresolved_accept=sch.BandInterval(lo=4, hi=5, lo_level=0.1, hi_level=0.9))
    table = sch.make_band_table(provenance=_band_provenance(), bands=(narrow,))
    # A narrower band is different CONTENT, so it is a different calibration digest; the run's
    # frozen conditioning has to name the table it is actually gated by, or the kernel refuses it
    # as foreign before the band verdict is ever reached.
    out = _project(
        band_table=table,
        source=_source(conditioning=_conditioning(
            schedule_band_calibration=table.provenance.calibration_content_digest)),
    )
    assert out.outcome is st.TransitionOutcome.NULL_BAND_INCOMPATIBLE
    assert out.projected is None


def test_a_band_declared_under_a_different_calibration_id_is_rejected():
    out = _project(declared_band_id="band:v2:someone-elses")
    assert out.outcome is st.TransitionOutcome.NULL_BAND_INCOMPATIBLE


def test_a_stratum_the_band_was_not_calibrated_for_is_rejected():
    out = _project(stratum_key="len200_260")
    assert out.outcome is st.TransitionOutcome.NULL_BAND_INCOMPATIBLE


def test_a_null_outcome_carries_the_band_verdict_that_caused_it():
    out = _project(stratum_key="len200_260")
    assert out.band_verdicts
    assert not any(v.accepted for v in out.band_verdicts)


# --------------------------------------------------------------------------------------------
# null transitions and reason telemetry
# --------------------------------------------------------------------------------------------


def test_a_rejected_policy_result_becomes_a_typed_null_and_not_an_exception():
    rejection = pol.PolicyRejection(
        reason="no improvement-associated identity survived the reference comparison",
        reason_by_pos={1: pol.SupportReason.UNCERTAIN},
        policy=_policy_identity(),
    )
    out = _project(decision=rejection)
    assert out.outcome is st.TransitionOutcome.NULL_INVALID_POLICY_RESULT
    assert out.projected is None


def test_a_null_outcome_preserves_the_per_position_reason_telemetry():
    """FeedbackTransition forbids a support partition on a null outcome, so the reasons that
    explain the rejection have no home on the event row -- the outcome must carry them."""
    rejection = pol.PolicyRejection(
        reason="policy declined", reason_by_pos={2: pol.SupportReason.UNCERTAIN},
        policy=_policy_identity(),
    )
    out = _project(decision=rejection)
    assert out.reason_by_pos[2] == pol.SupportReason.UNCERTAIN.value


def test_a_committed_outcome_carries_every_positions_reason():
    out = _project()
    assert set(out.reason_by_pos) == set(_source().editable_positions)
    assert out.projected.support.reason_by_pos[1] == \
        pol.SupportReason.IMPROVEMENT_ASSOCIATED.value


def test_the_outcome_names_the_policy_that_produced_it():
    assert _project().policy == _policy_identity()


def test_a_null_outcome_still_names_its_policy():
    rejection = pol.PolicyRejection(reason="declined", reason_by_pos={},
                                    policy=_policy_identity())
    assert _project(decision=rejection).policy == _policy_identity()


# --------------------------------------------------------------------------------------------
# exact archive non-mutation
# --------------------------------------------------------------------------------------------


def test_the_endpoint_record_is_byte_identical_after_projection():
    endpoint = _endpoint()
    before = endpoint.content_digest
    _project(endpoint=endpoint)
    assert endpoint.content_digest == before


def test_the_endpoint_tokens_must_agree_with_the_endpoint_record():
    """A caller that decodes the endpoint into a different token tuple must fail closed rather
    than silently writing bytes the archived endpoint never contained."""
    wrong = tuple([99, *ENDPOINT_TOKENS[1:]])
    with pytest.raises(V2Error):
        _project(endpoint_tokens=wrong)


# --------------------------------------------------------------------------------------------
# the policy boundary
# --------------------------------------------------------------------------------------------


def test_the_explicit_probe_policy_is_labelled_diagnostic_only():
    """PLAN §2.5: it 'must be labeled diagnostic-only and may not become a silent production
    default'."""
    probe = pol.ExplicitProbePolicy(**REF_SETS, reason_by_pos=_decision().reason_by_pos)
    assert probe.identity().is_diagnostic_only is True
    assert probe.identity().policy_id.startswith("explicit_probe")


def test_the_explicit_probe_policy_returns_the_sets_it_was_given():
    probe = pol.ExplicitProbePolicy(**REF_SETS, reason_by_pos=_decision().reason_by_pos)
    decision = probe.decide(source=_source(), endpoint=_endpoint(), coordinates=_coords())
    assert isinstance(decision, pol.PolicyDecision)
    assert decision.write_from_endpoint == REF_SETS["write_from_endpoint"]
    assert decision.reopen == REF_SETS["reopen"]


def test_a_policy_result_carries_raw_tuples_not_a_constructed_partition():
    """If the policy result were a SupportPartition it would RAISE on an empty reopen set, which
    would convert an invalid policy result into an exception instead of the typed null that
    PLAN §2.4 requires."""
    empty = pol.PolicyDecision(write_from_endpoint=(), inject_from_source_feedback=(),
                               reopen=(), carry_from_source=(1, 2, 3, 4, 5),
                               reason_by_pos={}, policy=_policy_identity())
    assert empty.reopen == ()


def test_the_policy_module_cannot_read_the_run_config():
    """PLAN §2.5 forbids any unfrozen threshold; a policy that cannot import the run config
    cannot read one out of it."""
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(pol.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.update(node.module.split("."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.update(alias.name.split("."))
    assert "config" not in imported


def test_the_projection_module_does_not_import_the_seed_module():
    """seeds.py has in-degree 0 inside the package: that is the structural form of PLAN §2.6's
    exclusion law.  The descendant fork seed arrives as a plain int from the runner."""
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(proj.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.update(node.module.split("."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.update(alias.name.split("."))
    assert "seeds" not in imported


def test_the_descendant_fork_seed_is_carried_through_verbatim():
    assert _project(descendant_fork_seed=777).projected.descendant_fork_seed == 777


# --------------------------------------------------------------------------------------------
# cost accounting
# --------------------------------------------------------------------------------------------


def test_projection_itself_costs_no_dfe_and_declares_the_segment_exactly():
    out = _project()
    assert out.projected.inherited_lineage_dfe == _source().accumulated_lineage_dfe
    assert out.projected.planned_segment_dfe == C_NEXT - R_STEP


# --------------------------------------------------------------------------------------------
# q_phi(P, y) is defined over a BOUND pair, not over any two objects of the right types
# --------------------------------------------------------------------------------------------


def test_the_kernel_refuses_an_endpoint_from_another_protein():
    """``q_phi(P, y)`` is only defined when ``y`` is an endpoint of ``P``'s own lineage.

    Nothing about a token vector says which protein produced it.  An endpoint from another protein
    of the same length type-checks, decodes, and writes back cleanly -- and the resulting state
    would carry that endpoint's id in its lineage while containing another protein's residues.
    """
    foreign = F.endpoint(protein_id="OTHER_A", lineage=F.lineage(protein_id="OTHER_A",
                                                                root_id="OTHER_A:v2:d0:r0"))
    with pytest.raises(proj.V2ProjectionError, match="protein"):
        F.project(endpoint=foreign)


def test_the_kernel_refuses_an_endpoint_from_another_family():
    foreign = F.endpoint(lineage=F.lineage(family_id="fam9"))
    with pytest.raises(proj.V2ProjectionError, match="family|root"):
        F.project(endpoint=foreign)


def test_the_kernel_refuses_an_endpoint_that_may_not_become_ancestry():
    """PLAN §4.2: only a definitively feasible endpoint may become feedback ancestry.

    Selection filters for this today, so the kernel is a second gate -- and it is the one that
    matters, because the kernel is where ancestry is actually CREATED.  A caller that ranks
    endpoints itself, or a future policy that reaches past selection, must not be able to install an
    unrefolded design as the parent of everything downstream.
    """
    unvalidated = F.endpoint(feasibility_level=st.FeasibilityLevel.UNVALIDATED,
                             structure_outcome=None)
    with pytest.raises(proj.V2ProjectionError, match="definitive|ancestry"):
        F.project(endpoint=unvalidated)


def test_the_kernel_refuses_an_endpoint_that_violates_a_hard_anchor():
    """A hard anchor is a permanent class; an endpoint that disagrees with one is not this
    protein's endpoint, whatever its lineage fields claim."""
    tokens = (11,) + F.ENDPOINT_TOKENS[1:]      # position 0 is the anchor, pinned at token 10
    assert F.source().hard_anchors == ((0, 10),)
    with pytest.raises(proj.V2ProjectionError, match="anchor"):
        F.project(endpoint=F.endpoint(tokens=tokens), endpoint_tokens=tokens)


def test_the_kernel_refuses_an_endpoint_scored_by_a_different_head():
    """Two Head evaluators produce two incomparable score scales.

    The endpoint was selected because its Head score ranked best; if that score came from a
    different evaluator than the one this lineage's safety reference is bound to, the ranking that
    justified the selection was never a comparison at all.
    """
    other_evaluator = dataclasses.replace(
        F.safety_reference().head_binding.evaluator, head_checkpoint_digest=F.digest("other-ckpt"))
    binding = dataclasses.replace(F.endpoint().head_binding, evaluator=other_evaluator)
    with pytest.raises(proj.V2ProjectionError, match="evaluator|Head"):
        F.project(endpoint=F.endpoint(head_binding=binding))


def test_the_kernel_refuses_tokens_that_do_not_decode_to_the_scored_sequence():
    """The bytes written back must be the bytes the Head actually saw.

    The archive holds ``y`` twice: as a residue string, which is what the Head scored, and as a
    per-position token vector, which is what the projection writes.  Nothing inside a torch-free
    package can decode one into the other, so the kernel does not guess -- it requires the caller to
    declare the alphabet it decoded with, and then checks the round trip.  Without that, V2 can
    write residues no Head ever evaluated while the provenance names an endpoint that was scored.
    """
    swapped = dict(F.ALPHABET)
    first, second = F.ENDPOINT_TOKENS[0], F.ENDPOINT_TOKENS[1]
    assert swapped[first] != swapped[second], "the probe must actually change the decode"
    swapped[first], swapped[second] = swapped[second], swapped[first]
    with pytest.raises(proj.V2ProjectionError, match="decode|sequence"):
        F.project(alphabet=swapped)


def test_a_bound_pair_still_projects():
    """The guard above must not be satisfiable only by rejecting everything."""
    assert F.project().committed


def test_the_bound_context_is_reported_on_the_outcome():
    """The binding is evidence, not just a check: an audit must be able to read what was bound."""
    outcome = F.project()
    assert outcome.context is not None
    assert outcome.context.endpoint.endpoint_id == F.endpoint().endpoint_id
    assert outcome.context.source.state_id == F.source().state_id


# --------------------------------------------------------------------------------------------
# the frozen config is the runtime authority, not merely a label on the manifest
# --------------------------------------------------------------------------------------------


def test_a_policy_whose_spec_is_not_the_runs_frozen_policy_is_refused():
    """``V2ConditioningIdentity.projection_policy_spec`` is the run's FROZEN projection policy
    (PLAN §5.2), carried by value on every state.  Nothing compared the policy that actually
    answered against it, so a run could commit transitions produced by a policy its own provenance
    row does not describe -- and the artifact would name the declared one.
    """
    foreign = _policy_identity(policy_spec_digest=proj.canonical_digest("some other policy spec"))
    with pytest.raises(proj.V2ProjectionError):
        _project(decision=_decision(policy=foreign))


def test_a_band_table_that_is_not_the_runs_frozen_calibration_is_refused():
    """``schedule_band_calibration`` is likewise frozen in the conditioning.

    ``gate_projected_maturity`` only checks the declared id/digest against the table it was HANDED,
    so a self-consistent foreign calibration passes every existing check: the maturity gate would
    then be evaluated against bands measured on a different cohort.
    """
    table = _band_table()
    with pytest.raises(proj.V2ProjectionError):
        _project(source=_source(conditioning=_conditioning(schedule_band_calibration=D)),
                 band_table=table,
                 declared_band_digest=table.provenance.calibration_content_digest)


def test_a_declining_policy_is_still_checked_against_the_frozen_spec():
    """A foreign policy's REFUSAL is not this run's refusal either.

    Returning its null as a typed outcome would record "the policy declined" in the feedback-event
    table under the declared policy's identity, when the declared policy was never asked.
    """
    foreign = _policy_identity(policy_spec_digest=proj.canonical_digest("some other policy spec"))
    with pytest.raises(proj.V2ProjectionError):
        _project(decision=pol.PolicyRejection(reason="no", policy=foreign))


def test_a_policy_answer_that_carries_no_identity_at_all_is_refused():
    """The binding above is only a gate if it cannot be opted out of.

    ``PolicyDecision.policy`` is optional, so a decision that simply omits it would skip the
    frozen-config check entirely -- an anonymous policy would be the one policy exempt from having
    to be the run's declared policy, and PLAN §5.3 would then have a committed feedback event whose
    policy columns are null.
    """
    with pytest.raises(proj.V2ProjectionError, match="identity"):
        _project(decision=_decision(policy=None))


def test_an_identified_policy_still_projects():
    """The guard must not be satisfiable by refusing every policy."""
    assert _project().committed


# --------------------------------------------------------------------------------------------
# the DIAGNOSTIC gate decides on the policy that ANSWERS, not on a string the config wrote
# --------------------------------------------------------------------------------------------


def test_a_diagnostic_policy_cannot_even_be_declared_in_a_production_phase():
    """PLAN §2.5: a diagnostic policy 'may not become a silent production default'.

    Making the illegal declaration unconstructible is what lets the kernel stay phase-free: if a
    diagnostic declaration cannot exist outside ``DIAGNOSTIC_ALLOWED_PHASES``, then an answer that
    MATCHES its run's declaration cannot be diagnostic in a production phase.
    """
    for phase in ("mechanism_cohort", "policy_qualification", "capability_ladder",
                  "holdout_application"):
        with pytest.raises(V2Error):
            _declared_policy(phase=phase)
    assert _declared_policy(phase="state_transition_canary").is_diagnostic is True


def test_the_kernel_refuses_a_diagnostic_policy_the_run_never_declared():
    """The reproduced defect, at the runtime end.

    The phase gate used to read ``config.projection.support_policy_id`` -- a string the config
    author writes ABOUT ITSELF -- and never inspected the ``ProjectionPolicyIdentity`` that
    actually produces the transition.  A ``mechanism_cohort`` run could therefore declare a
    production policy in its config while the diagnostic probe drove ``source_writeback``, and the
    transition committed.
    """
    production = pol.DeclaredPolicy(policy_id="source_writeback_v1", policy_version="v0",
                                    is_diagnostic=False, phase="mechanism_cohort")
    with pytest.raises(proj.V2ProjectionError, match="diagnostic|declared"):
        _project(declared_policy=production)


def test_the_kernel_refuses_a_policy_that_misreports_its_own_diagnostic_status():
    """``is_diagnostic_only`` is the policy's self-report and must agree with the vocabulary.

    Without this, a probe could keep its registered diagnostic id while declaring itself
    production, and every gate keyed on the flag would wave it through.
    """
    lying = _policy_identity(is_diagnostic_only=False)
    with pytest.raises(proj.V2ProjectionError, match="diagnostic"):
        _project(decision=_decision(policy=lying))


def test_the_kernel_refuses_a_policy_version_the_run_did_not_declare():
    with pytest.raises(proj.V2ProjectionError, match="version"):
        _project(decision=_decision(policy=_policy_identity(policy_version="v9")))


def test_a_refusal_is_also_checked_against_the_declared_policy():
    """A foreign policy's REFUSAL is not this run's refusal, for the declaration as for the spec."""
    production = pol.DeclaredPolicy(policy_id="source_writeback_v1", policy_version="v0",
                                    is_diagnostic=False, phase="mechanism_cohort")
    with pytest.raises(proj.V2ProjectionError):
        _project(decision=pol.PolicyRejection(reason="no", policy=_policy_identity()),
                 declared_policy=production)


def test_the_kernel_will_not_run_without_the_runs_declared_policy():
    """A gate that can be opted out of by omission is not a gate.

    Same argument the module already makes for an anonymous policy answer: the declaration is a
    REQUIRED input, exactly like ``declared_band_id``/``declared_band_digest``, so a caller cannot
    obtain a transition without naming the policy its run froze.
    """
    kwargs = _project_kwargs()
    kwargs.pop("declared_policy")
    with pytest.raises(TypeError):
        proj.source_writeback(**kwargs)


def test_the_declared_policy_still_admits_the_policy_that_actually_answered():
    """The guard must not be satisfiable by refusing every policy."""
    assert _project().committed


def test_a_refusal_must_also_name_the_policy_that_refused():
    """A refusal is a policy ANSWER, and PLAN §4.5 makes it a result rather than a non-event.

    ``feedback_event_rows`` records null events in full precisely so a bundle is not a record of
    the successes alone -- so an anonymous refusal would put "the policy declined" in that table
    under policy columns nothing can fill.
    """
    with pytest.raises(proj.V2ProjectionError, match="identity"):
        _project(decision=pol.PolicyRejection(reason="no", policy=None))


def test_the_kernel_binds_the_sampler_horizon_not_only_the_checkpoint():
    """``c_source_step`` was bound to the source and ``n_steps`` was not.

    ``S`` is the trajectory's horizon: every maturity coordinate, every tail length and the whole
    cost model are defined relative to it, so a projection built under a different ``S`` describes
    a different trajectory while naming this one's source.
    """
    with pytest.raises(proj.V2ProjectionError, match="n_steps"):
        _project(coordinates=_coords(n_steps=N_STEPS + 40))
