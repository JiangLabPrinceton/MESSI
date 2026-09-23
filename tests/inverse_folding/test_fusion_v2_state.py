"""V2F1 steps 11-12: ``fusion_v2.state`` per-position algebra and ``LivePartialState``.

The per-position vectors are the layer where PLAN §2.4's evidence-namespace separation either holds
or silently collapses: endpoint/source provenance is immutable evidence, and only an active-context
sampler score with a valid status may drive remask, ranking, or confidence.
"""

from __future__ import annotations

import dataclasses

import pytest

from inverse_folding.reference_flow.fusion_v2 import identity as ident
from inverse_folding.reference_flow.fusion_v2 import schedule as sch
from inverse_folding.reference_flow.fusion_v2 import state as st
from inverse_folding.reference_flow.fusion_v2.errors import V2Error

D = "a" * 64
E = "b" * 64
MASK = 32
AA = frozenset(range(4, 24))          # a stand-in canonical AA20 id block
L = 6                                  # positions 0..5; 0 is a hard anchor, 5 stays unresolved


def _conditioning():
    return ident.V2ConditioningIdentity(
        base=ident.make_v2_conditioning(
            dplm_checkpoint=D, tokenizer=D, backbone_row=D, coordinate_mask=D,
            entry_config=D, fixed_token_policy=D,
        ),
        **{role: D for role in ident.OWN_CONDITIONING_FIELDS},
    )


def _safety_reference():
    evaluator = ident.HeadEvaluatorIdentity(
        allele="DRB1_0701", score_scale="nats", window_k_min=13, window_k_max=25,
        head_config_hash=D, head_checkpoint_digest=D,
    )
    binding = ident.HeadScoreBinding(
        protein_id="5ZHV_B", sequence_md5="0" * 32, sequence_length=L,
        window_grid_digest=D, evaluator=evaluator,
    )
    return ident.SafetyReferenceBinding(
        reference_id="ref:wt", reference_label="wt_native", sequence_md5="0" * 32,
        sequence_length=L, reference_content_digest=D, head_binding=binding,
        head_score_digest=D, bound_at_depth=0, source_kind="predeclared_external",
    )


def _provenance(token, *, kind, ref=st.FeedbackOriginRef.NONE, depth=0, step=0, txn=None):
    origin = st.OriginEvidence(
        origin_kind=kind, origin_ref=ref, commit=sch.history_key(depth, step), token=token,
        evidence_logprob=-1.5, transition_id=txn, evidence_digest=D,
    )
    return st.PositionProvenance(first_origin=origin, last_origin=origin, n_origin_events=1)


def _live(**over):
    """A minimal legal live state: one anchor, four resolved, one unresolved."""
    tokens = [10, 11, 12, 13, 14, MASK]
    kind = [st.ActiveOriginKind.HARD_ANCHOR] + [st.ActiveOriginKind.DENOISER_SAMPLE] * 4 + [
        st.ActiveOriginKind.UNRESOLVED]
    ref = [st.FeedbackOriginRef.NONE] * L
    commit = [None] + [sch.history_key(0, 10 + i) for i in range(4)] + [None]
    txn = [None] * L
    score = [None] + [-0.5] * 4 + [None]
    status = ([st.ActiveScoreStatus.NOT_RANKED_ANCHOR]
              + [st.ActiveScoreStatus.HISTORICAL_NATURAL] * 4
              + [st.ActiveScoreStatus.MASKED])
    prov = [_provenance(tokens[0], kind=st.ActiveOriginKind.HARD_ANCHOR)] + [
        _provenance(tokens[i], kind=st.ActiveOriginKind.DENOISER_SAMPLE, step=10 + i - 1)
        for i in range(1, 5)
    ] + [_provenance(MASK, kind=st.ActiveOriginKind.UNRESOLVED)]

    kw = dict(
        schema_version=ident.V2_STATE_SCHEMA_VERSION,
        lineage=ident.LineageRef(protein_id="5ZHV_B", root_id="5ZHV_B:v2:d0:r0", family_id="fam0",
                                 depth=0, parent_state_id=None, parent_transition_id=None,
                                 origin_endpoint_id=None),
        sampler_step=50, n_steps=100,
        tokens=tuple(tokens), mask_token_id=MASK, aa_token_ids=AA,
        hard_anchors=((0, 10),), editable_positions=(1, 2, 3, 4, 5),
        active_origin_kind_by_pos=tuple(kind),
        feedback_origin_ref_by_pos=tuple(ref),
        active_commit_depth_step_by_pos=tuple(commit),
        origin_transition_id_by_pos=tuple(txn),
        active_sampler_score_by_pos=tuple(score),
        active_score_status_by_pos=tuple(status),
        provenance_by_pos=tuple(prov),
        expired_protection=(),
        replay=st.ReplayIdentity(mode="identity", rng_state={"pos": 7}, fork_seed=None,
                                 replay_state_hash=D),
        accumulated_lineage_dfe=50,
        conditioning=_conditioning(),
        safety_reference=_safety_reference(),
        cost_event_ids=("evt:root",),
    )
    kw.update(over)
    return st.LivePartialState(**kw)


# --------------------------------------------------------------------------------------------
# Step 11 - per-position algebra
# --------------------------------------------------------------------------------------------

def test_a_minimal_legal_live_state_constructs():
    live = _live()
    assert live.realized_maturity.n_unresolved_editable == 1
    assert live.realized_maturity.n_editable == 5
    assert live.t == pytest.approx(0.5)


@pytest.mark.parametrize(
    "field",
    ["active_origin_kind_by_pos", "feedback_origin_ref_by_pos",
     "active_commit_depth_step_by_pos", "origin_transition_id_by_pos",
     "active_sampler_score_by_pos", "active_score_status_by_pos", "provenance_by_pos"],
)
def test_every_per_position_vector_must_have_length_l(field):
    """``L == len(tokens)``, indexed by absolute residue position -- never over editable_positions."""
    short = _live().__getattribute__(field)[:-1]
    with pytest.raises(V2Error):
        _live(**{field: short})


def test_feedback_injection_must_name_the_transition_that_wrote_it():
    """§7.2 #7: origin/provenance tampering must not pass integrity checks.

    ``feedback_origin_ref_by_pos`` says which *class* of feedback wrote a position; the concrete
    endpoint or source id is reached through ``origin_transition_id_by_pos``. An injection with no
    transition id is an orphan: its ancestry is unreachable, which is exactly the
    "reconstruct the mapping from sequence alone" the PLAN forbids.
    """
    kind = list(_live().active_origin_kind_by_pos)
    ref = list(_live().feedback_origin_ref_by_pos)
    kind[2] = st.ActiveOriginKind.FEEDBACK_INJECTION
    ref[2] = st.FeedbackOriginRef.SELECTED_ENDPOINT
    with pytest.raises(V2Error):
        _live(active_origin_kind_by_pos=tuple(kind), feedback_origin_ref_by_pos=tuple(ref))


def test_a_naturally_sampled_position_may_not_claim_a_feedback_origin():
    """The origin/ref cross-product is exhaustive, not two independent labels."""
    ref = list(_live().feedback_origin_ref_by_pos)
    ref[2] = st.FeedbackOriginRef.SELECTED_ENDPOINT  # position 2 is DENOISER_SAMPLE
    with pytest.raises(V2Error):
        _live(feedback_origin_ref_by_pos=tuple(ref))


def test_provenance_evidence_may_never_become_an_active_sampler_score():
    """PLAN §2.4 / §7.2 #4: the two namespaces are structurally separate.

    ``OriginEvidence.evidence_logprob`` is the score an identity carried in its *originating*
    context. A masked position's active score must stay ``None`` however rich its provenance is.
    """
    live = _live()
    assert live.provenance_by_pos[5].last_origin.evidence_logprob is not None
    assert live.active_sampler_score_by_pos[5] is None
    score = list(live.active_sampler_score_by_pos)
    score[5] = -0.3                                    # a masked position with a score
    with pytest.raises(V2Error):
        _live(active_sampler_score_by_pos=tuple(score))


@pytest.mark.parametrize(
    ("index", "status"),
    [(0, st.ActiveScoreStatus.HISTORICAL_NATURAL),     # an anchor is never ranked
     (5, st.ActiveScoreStatus.HISTORICAL_NATURAL),     # a masked position has no history
     (2, st.ActiveScoreStatus.MASKED),                 # a resolved position is not masked
     (2, st.ActiveScoreStatus.NOT_RANKED_ANCHOR)],     # a non-anchor is not an anchor
)
def test_score_status_must_agree_with_the_token_and_the_anchor_set(index, status):
    st_vec = list(_live().active_score_status_by_pos)
    st_vec[index] = status
    with pytest.raises(V2Error):
        _live(active_score_status_by_pos=tuple(st_vec))


def test_hard_anchor_tokens_must_match_the_token_vector():
    with pytest.raises(V2Error):
        _live(hard_anchors=((0, 99),))


def test_fixed_and_editable_must_partition_every_position():
    """Mirrors the v0 invariant: ``fixed ∪ editable`` covers ``[0, L)`` and the two are disjoint."""
    with pytest.raises(V2Error):
        _live(editable_positions=(1, 2, 3, 4))          # position 5 belongs to neither
    with pytest.raises(V2Error):
        _live(editable_positions=(0, 1, 2, 3, 4, 5))    # position 0 is an anchor


def test_a_resolved_editable_token_must_be_canonical_aa20():
    """The AA20 firewall is enforceable without a tokenizer because the id set is carried."""
    tokens = list(_live().tokens)
    tokens[2] = 99                                       # not in aa_token_ids, not the mask id
    with pytest.raises(V2Error):
        _live(tokens=tuple(tokens))


@pytest.mark.parametrize("anchor_token", [MASK, 99])
def test_a_hard_anchor_must_be_a_canonical_aa20_token(anchor_token):
    """Safety law 2: matching a forged anchor declaration is not enough; the token is AA20."""
    tokens = list(_live().tokens)
    tokens[0] = anchor_token
    provenance = list(_live().provenance_by_pos)
    provenance[0] = _provenance(anchor_token, kind=st.ActiveOriginKind.HARD_ANCHOR)
    with pytest.raises(V2Error):
        _live(
            tokens=tuple(tokens), hard_anchors=((0, anchor_token),),
            provenance_by_pos=tuple(provenance),
        )


def test_active_position_state_must_match_its_last_provenance_atom():
    """Changing immutable provenance without changing active bytes must fail closed."""
    provenance = list(_live().provenance_by_pos)
    provenance[2] = _provenance(23, kind=st.ActiveOriginKind.DENOISER_SAMPLE, step=11)
    with pytest.raises(V2Error, match="provenance"):
        _live(provenance_by_pos=tuple(provenance))


# --------------------------------------------------------------------------------------------
# Step 12 - what makes a LivePartialState a *live* state
# --------------------------------------------------------------------------------------------

def test_a_pending_assimilation_position_is_unrepresentable_on_a_live_state():
    """PLAN §3.1: fail before any active-score consumer if a non-anchor token is still pending.

    This is structural difference #1 between the live and projected layers: assimilation happens
    during the segment, so a captured live state that still carries a pending token means the
    segment exposed a state whose remask ranking would read an undefined score.
    """
    status = list(_live().active_score_status_by_pos)
    status[2] = st.ActiveScoreStatus.PENDING_ASSIMILATION
    with pytest.raises(V2Error):
        _live(active_score_status_by_pos=tuple(status))


def test_a_live_state_carries_no_active_temporary_protection():
    """Structural difference #2: protection expires at the top of ``c_{d+1}``, *before* the
    captured state is exposed, so a live state carries only expired-protection telemetry."""
    assert "active_temporary_protection" not in st.LivePartialState.__dataclass_fields__
    assert "expired_protection" in st.LivePartialState.__dataclass_fields__


def test_a_live_state_must_retain_at_least_one_unresolved_editable_position():
    """PLAN V2F6: progressive checkpoints must reject exhausted horizons.

    A fully resolved descendant can emit no lookahead, so it is a terminal endpoint rather than a
    live partial state -- the same rule the sampler enforces for a pre-terminal root.
    """
    tokens = list(_live().tokens)
    tokens[5] = 14
    kind = list(_live().active_origin_kind_by_pos)
    kind[5] = st.ActiveOriginKind.DENOISER_SAMPLE
    commit = list(_live().active_commit_depth_step_by_pos)
    commit[5] = sch.history_key(0, 14)
    status = list(_live().active_score_status_by_pos)
    status[5] = st.ActiveScoreStatus.HISTORICAL_NATURAL
    score = list(_live().active_sampler_score_by_pos)
    score[5] = -0.4
    with pytest.raises(V2Error):
        _live(tokens=tuple(tokens), active_origin_kind_by_pos=tuple(kind),
              active_commit_depth_step_by_pos=tuple(commit),
              active_score_status_by_pos=tuple(status),
              active_sampler_score_by_pos=tuple(score))


def test_depth_is_not_duplicated_beside_lineage_depth():
    """Interface map Conflict 22: a duplicated field that is merely equality-checked adds a failure
    mode without adding information."""
    assert "depth" not in st.LivePartialState.__dataclass_fields__
    assert _live().lineage.depth == 0


def test_schema_version_must_match_the_module_constant():
    with pytest.raises(V2Error):
        _live(schema_version="v2state-0")


def test_state_id_must_bind_the_state_content_digest():
    """§7.2 #28: a row whose id survived an edit to its payload must not validate."""
    live = _live()
    ident.assert_id_binds_digest(live.state_id, live.content_digest)
    with pytest.raises(V2Error):
        _live(state_id=ident.make_live_state_id("5ZHV_B", "fam0", depth=0, step=50,
                                                content_digest="b" * 64))


def test_live_state_id_must_bind_its_full_lineage_coordinate():
    live = _live()
    foreign_coordinate = ident.make_live_state_id(
        "5ZHV_B", "fam0", depth=0, step=49, content_digest=live.content_digest,
    )
    with pytest.raises(V2Error, match="state_id"):
        _live(state_id=foreign_coordinate)


def test_maturity_is_derived_from_tokens_and_is_not_a_stored_field():
    """PLAN §2.1: measured from bytes, never trusted from a declared value."""
    assert "realized_maturity" not in st.LivePartialState.__dataclass_fields__
    assert isinstance(_live().realized_maturity, sch.MaturityRecord)


def test_content_digest_is_stable_and_covers_the_per_position_evidence():
    live = _live()
    assert live.content_digest == _live().content_digest
    kind = list(live.active_origin_kind_by_pos)
    kind[3] = st.ActiveOriginKind.UNRESOLVED
    commit = list(live.active_commit_depth_step_by_pos)
    commit[3] = None
    tokens = list(live.tokens)
    tokens[3] = MASK
    status = list(live.active_score_status_by_pos)
    status[3] = st.ActiveScoreStatus.MASKED
    score = list(live.active_sampler_score_by_pos)
    score[3] = None
    provenance = list(live.provenance_by_pos)
    provenance[3] = _provenance(MASK, kind=st.ActiveOriginKind.UNRESOLVED)
    other = _live(tokens=tuple(tokens), active_origin_kind_by_pos=tuple(kind),
                  active_commit_depth_step_by_pos=tuple(commit),
                  active_score_status_by_pos=tuple(status),
                  active_sampler_score_by_pos=tuple(score),
                  provenance_by_pos=tuple(provenance))
    assert other.content_digest != live.content_digest


def test_identity_replay_rng_payload_is_content_bound_even_if_the_claimed_hash_is_reused():
    """The raw identity RNG state and replay hash may not be two separable truths."""
    other_replay = st.ReplayIdentity(
        mode="identity", rng_state={"pos": 8}, fork_seed=None, replay_state_hash=D,
    )
    assert _live(replay=other_replay).content_digest != _live().content_digest


def test_expired_protection_is_typed_and_content_bound():
    """A forged row must fail, and a real extra expiry record must change live-state identity."""
    with pytest.raises(V2Error):
        _live(expired_protection=("forged",))

    extra = st.TemporaryProtection(
        position=1, expiry_step=50, granted_at_depth=0, granted_at_step=40,
        granted_by_transition_id=TXN, grant_reason=st.ProtectionGrantReason.ENDPOINT_INJECTION,
    )
    assert _live(expired_protection=(extra,)).content_digest != _live().content_digest


# ============================================================================================
# Steps 13-15: ProjectedPartialState, the temporal carry gate, and feedback-event validation
# ============================================================================================

R_STEP, C_SOURCE, C_NEXT = 40, 50, 60
EP_TOKEN, SRC_TOKEN = 20, 21
TXN = "5ZHV_B:fam0:txn:d0:r40-c60:" + D[:12]


def _projected(**over):
    """A minimal legal projected state, one position per support action.

    pos 0 anchor | 1 endpoint-write | 2 source-feedback injection | 3 reopen
    pos 4 natural carry (committed before r_step) | 5 inherited mask carried unwritten
    """
    tokens = [10, EP_TOKEN, SRC_TOKEN, MASK, 14, MASK]
    kind = [st.ActiveOriginKind.HARD_ANCHOR,
            st.ActiveOriginKind.FEEDBACK_INJECTION,
            st.ActiveOriginKind.FEEDBACK_INJECTION,
            st.ActiveOriginKind.UNRESOLVED,
            st.ActiveOriginKind.DENOISER_SAMPLE,
            st.ActiveOriginKind.UNRESOLVED]
    ref = [st.FeedbackOriginRef.NONE,
           st.FeedbackOriginRef.SELECTED_ENDPOINT,
           st.FeedbackOriginRef.SOURCE_STATE,
           st.FeedbackOriginRef.NONE,
           st.FeedbackOriginRef.NONE,
           st.FeedbackOriginRef.NONE]
    reentry = sch.history_key(1, R_STEP)
    commit = [None, reentry, reentry, None, sch.history_key(0, 30), None]
    txn = [None, TXN, TXN, None, None, None]
    score = [None, None, None, None, -0.5, None]
    status = [st.ActiveScoreStatus.NOT_RANKED_ANCHOR,
              st.ActiveScoreStatus.PENDING_ASSIMILATION,
              st.ActiveScoreStatus.PENDING_ASSIMILATION,
              st.ActiveScoreStatus.MASKED,
              st.ActiveScoreStatus.HISTORICAL_NATURAL,
              st.ActiveScoreStatus.MASKED]
    prov = [_provenance(10, kind=st.ActiveOriginKind.HARD_ANCHOR),
            _provenance(EP_TOKEN, kind=st.ActiveOriginKind.FEEDBACK_INJECTION,
                        ref=st.FeedbackOriginRef.SELECTED_ENDPOINT, depth=1, step=R_STEP, txn=TXN),
            _provenance(SRC_TOKEN, kind=st.ActiveOriginKind.FEEDBACK_INJECTION,
                        ref=st.FeedbackOriginRef.SOURCE_STATE, depth=1, step=R_STEP, txn=TXN),
            _provenance(MASK, kind=st.ActiveOriginKind.UNRESOLVED),
            _provenance(14, kind=st.ActiveOriginKind.DENOISER_SAMPLE, step=30),
            _provenance(MASK, kind=st.ActiveOriginKind.UNRESOLVED)]

    def _prot(position, reason):
        return st.TemporaryProtection(
            position=position, expiry_step=C_NEXT, granted_at_depth=1, granted_at_step=R_STEP,
            granted_by_transition_id=TXN, grant_reason=reason,
        )

    kw = dict(
        schema_version=ident.V2_STATE_SCHEMA_VERSION,
        lineage=ident.LineageRef(
            protein_id="5ZHV_B", root_id="5ZHV_B:v2:d0:r0", family_id="fam0", depth=1,
            parent_state_id=_live().state_id, parent_transition_id=TXN,
            origin_endpoint_id=ident.make_endpoint_id(_live().state_id, fork_index=0,
                                                      sequence_md5_hex=SEQ_MD5),
        ),
        r_step=R_STEP, c_next_step=C_NEXT, n_steps=100,
        coordinate_law=sch.CoordinateLaw.PROGRESSIVE_CHECKPOINT,
        tokens=tuple(tokens), mask_token_id=MASK, aa_token_ids=AA,
        hard_anchors=((0, 10),), editable_positions=(1, 2, 3, 4, 5),
        # positions resolved in the SOURCE state at c_d; position 5 was already masked there.
        # Needed to tell a genuine reopen from an inherited mask relabelled as one.
        source_resolved_positions=(1, 2, 3, 4),
        active_origin_kind_by_pos=tuple(kind),
        feedback_origin_ref_by_pos=tuple(ref),
        active_commit_depth_step_by_pos=tuple(commit),
        origin_transition_id_by_pos=tuple(txn),
        active_sampler_score_by_pos=tuple(score),
        active_score_status_by_pos=tuple(status),
        provenance_by_pos=tuple(prov),
        support=st.SupportPartition(
            write_from_endpoint=(1,), inject_from_source_feedback=(2,), reopen=(3,),
            carry_from_source=(4, 5),
            reason_by_pos={1: "improvement_associated", 2: "future_source_identity",
                           3: "uncertain", 4: "temporally_valid", 5: "inherited_mask"},
        ),
        active_temporary_protection=(_prot(1, st.ProtectionGrantReason.ENDPOINT_INJECTION),
                                     _prot(2, st.ProtectionGrantReason.SOURCE_FEEDBACK_INJECTION)),
        descendant_fork_seed=1234567890,
        origin_transition_id=TXN,
        declared_band_id="band-2026-08-04", declared_band_digest=D,
        inherited_lineage_dfe=170, planned_segment_dfe=C_NEXT - R_STEP,
        conditioning=_conditioning(), safety_reference=_safety_reference(),
        cost_event_ids=("evt:proj",),
    )
    kw.update(over)
    return st.ProjectedPartialState(**kw)


def test_a_minimal_legal_projected_state_constructs():
    proj = _projected()
    assert proj.realized_maturity.n_unresolved_editable == 2
    assert proj.planned_segment_dfe == 20
    assert ident.id_namespace(proj.state_id) == "proj"


def test_projected_state_id_must_bind_its_full_lineage_coordinate():
    projected = _projected()
    foreign_coordinate = ident.make_projected_state_id(
        "5ZHV_B", "fam0", depth=1, r_step=R_STEP - 1,
        content_digest=projected.content_digest,
    )
    with pytest.raises(V2Error, match="state_id"):
        _projected(state_id=foreign_coordinate)


def test_set_like_position_inputs_are_canonicalized_before_hashing():
    """Equivalent support geometry must not fork identity merely because input rows were reordered."""
    reordered = _projected(
        editable_positions=(5, 4, 3, 2, 1),
        source_resolved_positions=(4, 3, 2, 1),
        active_temporary_protection=tuple(reversed(_projected().active_temporary_protection)),
    )
    assert reordered.editable_positions == _projected().editable_positions
    assert reordered.source_resolved_positions == _projected().source_resolved_positions
    assert reordered.content_digest == _projected().content_digest


def test_projected_state_rejects_an_untyped_coordinate_law():
    """Direct construction may not bypass the schedule coordinate-law validator."""
    with pytest.raises(V2Error, match="coordinate_law"):
        _projected(coordinate_law=sch.CoordinateLaw.PROGRESSIVE_CHECKPOINT.value)


def test_projection_itself_costs_no_dfe_and_declares_the_segment_exactly():
    """PLAN §3.4: the segment costs exactly ``c_next - r``; projection adds 0."""
    with pytest.raises(V2Error):
        _projected(planned_segment_dfe=19)


# ---- step 13: the protection relation -------------------------------------------------------

def test_both_injection_kinds_receive_the_same_exclusive_expiry():
    """§7.2 #15: endpoint and source-feedback injections may not differ in expiry by accident.

    PLAN §2.4 assigns them one shared ``protection_expiry_step = c_{d+1}``. Divergent expiry would
    silently make one injection class survive the segment and the other not, which reads as a
    mechanism difference rather than as a wiring bug.
    """
    proj = _projected()
    assert {p.expiry_step for p in proj.active_temporary_protection} == {C_NEXT}
    assert {p.granted_at_step for p in proj.active_temporary_protection} == {R_STEP}
    bad = (proj.active_temporary_protection[0],
           st.TemporaryProtection(position=2, expiry_step=C_NEXT - 5, granted_at_depth=1,
                                  granted_at_step=R_STEP, granted_by_transition_id=TXN,
                                  grant_reason=st.ProtectionGrantReason.SOURCE_FEEDBACK_INJECTION))
    with pytest.raises(V2Error):
        _projected(active_temporary_protection=bad)


def test_the_protected_set_is_exactly_the_two_injection_actions():
    """§7.2 #14: protection must not leak. A carried or reopened position is not protected, and a
    protected position that is not an injection would be a permanent fixed token in disguise."""
    proj = _projected()
    assert tuple(p.position for p in proj.active_temporary_protection) == (1, 2)
    with pytest.raises(V2Error):     # protecting a carried position
        _projected(active_temporary_protection=proj.active_temporary_protection + (
            st.TemporaryProtection(position=4, expiry_step=C_NEXT, granted_at_depth=1,
                                   granted_at_step=R_STEP, granted_by_transition_id=TXN,
                                   grant_reason=st.ProtectionGrantReason.ENDPOINT_INJECTION),))
    with pytest.raises(V2Error):     # leaving an injection unprotected
        _projected(active_temporary_protection=proj.active_temporary_protection[:1])


def test_temporary_protection_reason_and_transition_match_the_injection_action():
    endpoint, source = _projected().active_temporary_protection
    wrong_reason = dataclasses.replace(
        endpoint, grant_reason=st.ProtectionGrantReason.SOURCE_FEEDBACK_INJECTION,
    )
    with pytest.raises(V2Error):
        _projected(active_temporary_protection=(wrong_reason, source))

    foreign_transition = dataclasses.replace(source, granted_by_transition_id="foreign:txn")
    with pytest.raises(V2Error):
        _projected(active_temporary_protection=(endpoint, foreign_transition))


def test_a_hard_anchor_can_never_be_temporarily_protected():
    """Anchors are a separate permanent class; mixing the two is how temporary protection turns
    into a permanent fixed token (PLAN §2.4)."""
    with pytest.raises(V2Error):
        _projected(active_temporary_protection=(
            st.TemporaryProtection(position=0, expiry_step=C_NEXT, granted_at_depth=1,
                                   granted_at_step=R_STEP, granted_by_transition_id=TXN,
                                   grant_reason=st.ProtectionGrantReason.ENDPOINT_INJECTION),))


def test_protection_is_half_open_over_the_segment():
    """Protected at step ``s`` exactly while ``granted_at_step <= s < expiry_step`` (PLAN §2.4)."""
    prot = _projected().active_temporary_protection[0]
    assert prot.protects_at(R_STEP) and prot.protects_at(C_NEXT - 1)
    assert not prot.protects_at(C_NEXT)
    assert not prot.protects_at(R_STEP - 1)


# ---- step 14: the temporal carry gate -------------------------------------------------------

@pytest.mark.parametrize(
    ("commit_step", "legal"),
    [(R_STEP - 1, True), (0, True), (R_STEP, False), (R_STEP + 1, False), (C_SOURCE, False)],
)
def test_ordinary_carry_is_legal_only_for_a_token_committed_before_reentry(commit_step, legal):
    """PLAN §2.4: a token committed at step >= r_d is future information at that boundary.

    Checkpoints are top-of-step pre-denoiser states, so a token committed *at* ``r_d`` was written
    by the very step being rolled back. It must be explicitly injected or reopened, never presented
    as naturally present before the boundary.
    """
    assert st.carry_is_temporally_legal(
        sch.history_key(0, commit_step), token=14, r_step=R_STEP, mask_token_id=MASK
    ) is legal


def test_an_inherited_mask_is_carried_not_reopened():
    """A source position that was already masked and is not written stays a carry.

    If an inherited mask could be labelled ``reopen``, a policy could satisfy PLAN §2.4's non-empty
    ``reopen`` requirement while newly masking nothing -- a fail-open the byte-level accounting
    would never notice.
    """
    assert st.carry_is_temporally_legal(None, token=MASK, r_step=R_STEP, mask_token_id=MASK)
    assert not st.carry_is_temporally_legal(None, token=14, r_step=R_STEP, mask_token_id=MASK)
    with pytest.raises(V2Error):
        _projected(support=st.SupportPartition(
            write_from_endpoint=(1,), inject_from_source_feedback=(2,), reopen=(3, 5),
            carry_from_source=(4,),
            reason_by_pos={1: "a", 2: "b", 3: "c", 4: "d", 5: "e"},
        ))


# ---- step 15: the four-way partition --------------------------------------------------------

def test_the_four_support_sets_are_disjoint_and_exhaustive_over_the_editable_domain():
    """§7.2 #2: projection must not omit an editable position from the partition."""
    with pytest.raises(V2Error):     # position 5 assigned to nothing
        _projected(support=st.SupportPartition(
            write_from_endpoint=(1,), inject_from_source_feedback=(2,), reopen=(3,),
            carry_from_source=(4,), reason_by_pos={1: "a", 2: "b", 3: "c", 4: "d"}))
    with pytest.raises(V2Error):     # position 1 assigned twice
        _projected(support=st.SupportPartition(
            write_from_endpoint=(1,), inject_from_source_feedback=(1, 2), reopen=(3,),
            carry_from_source=(4, 5), reason_by_pos={1: "a", 2: "b", 3: "c", 4: "d", 5: "e"}))


def test_a_hard_anchor_can_never_enter_the_partition():
    """§7.2 #2: projection opens an anchor. Anchors are outside the editable domain entirely."""
    with pytest.raises(V2Error):
        _projected(support=st.SupportPartition(
            write_from_endpoint=(0, 1), inject_from_source_feedback=(2,), reopen=(3,),
            carry_from_source=(4, 5),
            reason_by_pos={0: "x", 1: "a", 2: "b", 3: "c", 4: "d", 5: "e"}))


def test_write_is_required_but_reopen_may_be_empty():
    """A write-only transition is legal when the exact mask-load equation needs no reopen."""
    with pytest.raises(V2Error):
        st.SupportPartition(write_from_endpoint=(), inject_from_source_feedback=(2,), reopen=(3,),
                            carry_from_source=(1, 4, 5),
                            reason_by_pos={1: "a", 2: "b", 3: "c", 4: "d", 5: "e"})
    support = st.SupportPartition(
        write_from_endpoint=(1,), inject_from_source_feedback=(2,), reopen=(),
        carry_from_source=(3, 4, 5),
        reason_by_pos={1: "a", 2: "b", 3: "c", 4: "d", 5: "e"},
    )
    assert support.reopen == ()


def test_every_partitioned_position_carries_a_policy_reason():
    with pytest.raises(V2Error):
        st.SupportPartition(write_from_endpoint=(1,), inject_from_source_feedback=(2,), reopen=(3,),
                            carry_from_source=(4, 5), reason_by_pos={1: "a", 2: "b", 3: "c"})


def test_a_reopened_position_must_actually_be_masked_in_the_projected_state():
    tokens = list(_projected().tokens)
    tokens[3] = 14                                    # claims reopen but stays resolved
    kind = list(_projected().active_origin_kind_by_pos)
    kind[3] = st.ActiveOriginKind.DENOISER_SAMPLE
    commit = list(_projected().active_commit_depth_step_by_pos)
    commit[3] = sch.history_key(0, 31)
    status = list(_projected().active_score_status_by_pos)
    status[3] = st.ActiveScoreStatus.HISTORICAL_NATURAL
    score = list(_projected().active_sampler_score_by_pos)
    score[3] = -0.6
    with pytest.raises(V2Error):
        _projected(tokens=tuple(tokens), active_origin_kind_by_pos=tuple(kind),
                   active_commit_depth_step_by_pos=tuple(commit),
                   active_score_status_by_pos=tuple(status),
                   active_sampler_score_by_pos=tuple(score))


def test_an_endpoint_written_position_must_carry_the_endpoint_token():
    """§7.2 #6: endpoint-written token provenance must match the endpoint it claims."""
    proj = _projected()
    st.validate_endpoint_writeback(
        projected=proj, endpoint_tokens=(10, EP_TOKEN, 99, 99, 99, 99),
    )
    with pytest.raises(V2Error):
        st.validate_endpoint_writeback(
            projected=proj, endpoint_tokens=(10, EP_TOKEN + 1, 99, 99, 99, 99),
        )


@pytest.mark.parametrize(
    ("position", "wrong_ref"),
    [(1, st.FeedbackOriginRef.SOURCE_STATE),
     (2, st.FeedbackOriginRef.SELECTED_ENDPOINT)],
)
def test_support_action_must_match_active_and_immutable_feedback_provenance(position, wrong_ref):
    """Endpoint writes and source injections are not interchangeable labels."""
    refs = list(_projected().feedback_origin_ref_by_pos)
    refs[position] = wrong_ref
    provenance = list(_projected().provenance_by_pos)
    provenance[position] = _provenance(
        _projected().tokens[position], kind=st.ActiveOriginKind.FEEDBACK_INJECTION,
        ref=wrong_ref, depth=1, step=R_STEP, txn=TXN,
    )
    with pytest.raises(V2Error):
        _projected(
            feedback_origin_ref_by_pos=tuple(refs), provenance_by_pos=tuple(provenance),
        )


def test_every_injected_position_begins_pending_with_no_fabricated_score():
    """PLAN §2.4: injections begin pending, with no fabricated active score."""
    status = list(_projected().active_score_status_by_pos)
    score = list(_projected().active_sampler_score_by_pos)
    status[1] = st.ActiveScoreStatus.ASSIMILATED
    score[1] = -0.2                                    # a score invented at projection time
    with pytest.raises(V2Error):
        _projected(active_score_status_by_pos=tuple(status),
                   active_sampler_score_by_pos=tuple(score))


def test_an_injection_commits_at_the_reentry_key_not_its_source_step():
    """PLAN §2.4: every injection sets its active commit event to ``(depth+1, r_d)``.

    The later source or endpoint commit evidence survives as provenance only -- representing an
    injected identity as naturally present before the boundary is exactly the rollback-causality
    violation safety law 13 forbids.
    """
    proj = _projected()
    assert proj.active_commit_depth_step_by_pos[1] == sch.history_key(1, R_STEP)
    assert proj.provenance_by_pos[1].last_origin.commit == sch.history_key(1, R_STEP)
    commit = list(proj.active_commit_depth_step_by_pos)
    commit[1] = sch.history_key(0, C_SOURCE)           # claims it was there before re-entry
    with pytest.raises(V2Error):
        _projected(active_commit_depth_step_by_pos=tuple(commit))


def test_a_projected_state_carries_a_fork_seed_and_no_replay_stream():
    """Structural difference #3: a projected state is a causal intervention, never an identity
    replay, so there is no ``rng_state`` field for one to be smuggled through."""
    assert "replay" not in st.ProjectedPartialState.__dataclass_fields__
    assert "descendant_fork_seed" in st.ProjectedPartialState.__dataclass_fields__


def test_a_projected_state_stores_band_identity_but_never_a_tolerance():
    """PLAN §2.1: the band is input calibration with provenance, not a hardcoded tolerance."""
    fields = set(st.ProjectedPartialState.__dataclass_fields__)
    assert {"declared_band_id", "declared_band_digest"} <= fields
    assert not any("toleran" in f or "accept" in f for f in fields)


# ============================================================================================
# Steps 16-18: propagated capture, CompleteEndpoint, ArchiveEntry, FeedbackTransition
# ============================================================================================

from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome  # noqa: E402

from inverse_folding.reference_flow.fusion.state import sequence_md5  # noqa: E402

SEQ = "ACDEFG"          # canonical AA20, length L
SEQ_MD5 = sequence_md5(SEQ)


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


def _captured(**over):
    """The live state a propagated segment captures at c_next: pending gone, protection expired."""
    proj = _projected()
    tokens = [10, EP_TOKEN, SRC_TOKEN, 15, 14, 16]      # the segment resolved 3 and 5
    kind = [st.ActiveOriginKind.HARD_ANCHOR,
            st.ActiveOriginKind.FEEDBACK_INJECTION, st.ActiveOriginKind.FEEDBACK_INJECTION,
            st.ActiveOriginKind.DENOISER_SAMPLE, st.ActiveOriginKind.DENOISER_SAMPLE,
            st.ActiveOriginKind.DENOISER_SAMPLE]
    ref = list(proj.feedback_origin_ref_by_pos)
    commit = [None, sch.history_key(1, R_STEP), sch.history_key(1, R_STEP),
              sch.history_key(1, 55), sch.history_key(0, 30), sch.history_key(1, 58)]
    txn = [None, TXN, TXN, None, None, None]
    score = [None, -0.31, -0.42, -0.55, -0.5, -0.61]
    status = [st.ActiveScoreStatus.NOT_RANKED_ANCHOR,
              st.ActiveScoreStatus.ASSIMILATED, st.ActiveScoreStatus.ASSIMILATED,
              st.ActiveScoreStatus.HISTORICAL_NATURAL, st.ActiveScoreStatus.HISTORICAL_NATURAL,
              st.ActiveScoreStatus.HISTORICAL_NATURAL]
    provenance = list(proj.provenance_by_pos)
    provenance[3] = _provenance(
        15, kind=st.ActiveOriginKind.DENOISER_SAMPLE, depth=1, step=55,
    )
    propagated_lineage = ident.LineageRef(
        protein_id=proj.lineage.protein_id, root_id=proj.lineage.root_id,
        family_id=proj.lineage.family_id, depth=proj.lineage.depth,
        parent_state_id=proj.state_id,
        parent_transition_id=proj.lineage.parent_transition_id,
        origin_endpoint_id=proj.lineage.origin_endpoint_id,
    )
    kw = dict(
        schema_version=ident.V2_STATE_SCHEMA_VERSION, lineage=propagated_lineage,
        sampler_step=C_NEXT, n_steps=100,
        tokens=tuple(tokens), mask_token_id=MASK, aa_token_ids=AA,
        hard_anchors=((0, 10),), editable_positions=(1, 2, 3, 4, 5),
        active_origin_kind_by_pos=tuple(kind), feedback_origin_ref_by_pos=tuple(ref),
        active_commit_depth_step_by_pos=tuple(commit), origin_transition_id_by_pos=tuple(txn),
        active_sampler_score_by_pos=tuple(score), active_score_status_by_pos=tuple(status),
        provenance_by_pos=tuple(provenance),
        expired_protection=proj.active_temporary_protection,
        replay=st.ReplayIdentity(mode="fork", rng_state=None, fork_seed=1234567890,
                                 replay_state_hash=D),
        accumulated_lineage_dfe=190, conditioning=_conditioning(),
        safety_reference=_safety_reference(), cost_event_ids=("evt:seg",),
    )
    kw.update(over)
    # one unresolved position is required of a live state; keep 5 open unless told otherwise
    if "tokens" not in over:
        toks = list(kw["tokens"]); toks[5] = MASK; kw["tokens"] = tuple(toks)
        k = list(kw["active_origin_kind_by_pos"]); k[5] = st.ActiveOriginKind.UNRESOLVED
        kw["active_origin_kind_by_pos"] = tuple(k)
        c = list(kw["active_commit_depth_step_by_pos"]); c[5] = None
        kw["active_commit_depth_step_by_pos"] = tuple(c)
        sc = list(kw["active_sampler_score_by_pos"]); sc[5] = None
        kw["active_sampler_score_by_pos"] = tuple(sc)
        stt = list(kw["active_score_status_by_pos"]); stt[5] = st.ActiveScoreStatus.MASKED
        kw["active_score_status_by_pos"] = tuple(stt)
    return st.LivePartialState(**kw)


def _capture(**over):
    kw = dict(projected=_projected(), propagated=_captured(), background_remask_events=0)
    kw.update(over)
    return st.validate_propagated_capture(**kw)


def test_a_clean_propagated_capture_validates():
    _capture()


def test_propagated_capture_must_name_the_projected_state_as_its_parent_edge():
    """The committed descendant must retain the source -> projection -> capture graph explicitly."""
    assert _captured().lineage.parent_state_id == _projected().state_id


def test_capture_must_land_on_the_declared_checkpoint():
    with pytest.raises(V2Error):
        _capture(propagated=_captured(sampler_step=C_NEXT - 1))


def test_any_background_remask_event_in_a_v2_segment_is_fatal():
    """§7.2 #11. The frozen substrate has zero background remask, so only an explicit projection
    may add masks; a single event means the segment ran on the wrong substrate."""
    with pytest.raises(V2Error):
        _capture(background_remask_events=1)


def test_unresolved_mass_may_not_increase_across_the_segment():
    """Denoising only resolves; anything else means something re-masked behind the contract.

    The projected state enters the segment with two unresolved positions, so a capture carrying
    three can only have come from a mask the explicit projection did not write.
    """
    assert _projected().realized_maturity.n_unresolved_editable == 2
    with pytest.raises(V2Error):
        _capture(propagated=_captured(
            tokens=(10, EP_TOKEN, SRC_TOKEN, MASK, MASK, MASK),
            active_origin_kind_by_pos=(
                st.ActiveOriginKind.HARD_ANCHOR, st.ActiveOriginKind.FEEDBACK_INJECTION,
                st.ActiveOriginKind.FEEDBACK_INJECTION, st.ActiveOriginKind.UNRESOLVED,
                st.ActiveOriginKind.UNRESOLVED, st.ActiveOriginKind.UNRESOLVED),
            active_commit_depth_step_by_pos=(None, sch.history_key(1, R_STEP),
                                             sch.history_key(1, R_STEP), None, None, None),
            active_sampler_score_by_pos=(None, -0.31, -0.42, None, None, None),
            active_score_status_by_pos=(
                st.ActiveScoreStatus.NOT_RANKED_ANCHOR, st.ActiveScoreStatus.ASSIMILATED,
                st.ActiveScoreStatus.ASSIMILATED, st.ActiveScoreStatus.MASKED,
                st.ActiveScoreStatus.MASKED, st.ActiveScoreStatus.MASKED)))


def test_every_granted_protection_must_be_recorded_as_expired():
    """§7.2 #14: protection must not leak across depth. The captured state proves each interval
    ended by carrying it in ``expired_protection``."""
    with pytest.raises(V2Error):
        _capture(propagated=_captured(expired_protection=_projected()
                                      .active_temporary_protection[:1]))


def test_capture_binds_full_protection_objects_not_only_the_position_set():
    expired = list(_projected().active_temporary_protection)
    expired[0] = dataclasses.replace(
        expired[0], grant_reason=st.ProtectionGrantReason.SOURCE_FEEDBACK_INJECTION,
    )
    with pytest.raises(V2Error, match="protection"):
        _capture(propagated=_captured(expired_protection=tuple(expired)))


def test_capture_binds_conditioning_and_safety_reference_by_value():
    foreign_conditioning = dataclasses.replace(_conditioning(), head_config=E)
    with pytest.raises(V2Error, match="conditioning"):
        _capture(propagated=_captured(conditioning=foreign_conditioning))

    foreign_reference = dataclasses.replace(_safety_reference(), reference_content_digest=E)
    with pytest.raises(V2Error, match="safety"):
        _capture(propagated=_captured(safety_reference=foreign_reference))


def test_capture_binds_protein_family_and_descendant_fork_identity():
    foreign_lineage = ident.LineageRef(
        protein_id="OTHER_A", root_id="OTHER_A:v2:d0:r0", family_id="fam1", depth=1,
        parent_state_id=_live().state_id, parent_transition_id=TXN,
        origin_endpoint_id=_projected().lineage.origin_endpoint_id,
    )
    with pytest.raises(V2Error, match="lineage"):
        _capture(propagated=_captured(lineage=foreign_lineage))

    foreign_replay = st.ReplayIdentity(
        mode="fork", rng_state=None, fork_seed=1234567891, replay_state_hash=D,
    )
    with pytest.raises(V2Error, match="fork"):
        _capture(propagated=_captured(replay=foreign_replay))


def test_capture_binds_sampler_domain_and_accumulated_segment_cost():
    with pytest.raises(V2Error, match="AA20"):
        _capture(propagated=_captured(aa_token_ids=AA | {99}))
    with pytest.raises(V2Error, match="lineage DFE"):
        _capture(propagated=_captured(accumulated_lineage_dfe=191))


# ---- step 17: CompleteEndpoint ---------------------------------------------------------------

def _endpoint(**over):
    src = _live()
    windows = (_HeadWindow(start_0b=0, end_0b=L, k=L, z=-0.7),)
    head_binding = ident.HeadScoreBinding(
        protein_id="5ZHV_B", sequence_md5=SEQ_MD5, sequence_length=L,
        window_grid_digest=ident.window_grid_digest(windows),
        evaluator=_safety_reference().head_binding.evaluator,
    )
    head_score = _EndpointHeadScore(
        protein_id="5ZHV_B", sequence_md5=SEQ_MD5, sequence_length=L,
        allele=head_binding.evaluator.allele, score_scale=head_binding.evaluator.score_scale,
        windows=windows, residue_hotspot=(-0.1,) * L, global_risk=-9.1,
    )
    evidence = tuple(
        st.EndpointPositionEvidence(token=10 + i, commit=sch.history_key(0, 10 + i),
                                    completion_logprob=-0.4, inherited_from_source=i < 4)
        for i in range(L)
    )
    kw = dict(
        schema_version=ident.V2_STATE_SCHEMA_VERSION, lineage=_live().lineage,
        protein_id="5ZHV_B", sequence=SEQ, sequence_md5=SEQ_MD5, sequence_length=L,
        source_state_id=src.state_id, source_state_content_digest=src.content_digest,
        fork_index=0, fork_seed=99,
        replay=st.ReplayIdentity(mode="fork", rng_state=None, fork_seed=99, replay_state_hash=D),
        endpoint_provenance_evidence_by_pos=evidence,
        head_binding=head_binding,
        head_score=head_score,
        head_global_risk=-9.1,
        feasibility_level=st.FeasibilityLevel.DEFINITIVE,
        structure_outcome=StructureOutcome(feasible=True, metrics={"scTM": 0.91}),
        cost_event_ids=("evt:tail",),
    )
    kw.update(over)
    return st.CompleteEndpoint(**kw)


def test_a_definitive_endpoint_constructs_and_binds_its_source():
    ep = _endpoint()
    assert ep.feasibility_level is st.FeasibilityLevel.DEFINITIVE
    assert ident.id_namespace(ep.endpoint_id) == "endpoint"


def test_endpoint_carries_the_complete_head_evidence_from_the_frozen_interface():
    assert "head_score" in st.CompleteEndpoint.__dataclass_fields__


def test_endpoint_rejects_a_forged_id_or_foreign_source_digest():
    foreign_source = ident.make_live_state_id(
        "5ZHV_B", "fam0", depth=0, step=50, content_digest=E,
    )
    forged_endpoint = ident.make_endpoint_id(
        foreign_source, fork_index=0, sequence_md5_hex=SEQ_MD5,
    )
    with pytest.raises(V2Error, match="endpoint_id"):
        _endpoint(endpoint_id=forged_endpoint)
    with pytest.raises(V2Error, match="content digest"):
        _endpoint(source_state_content_digest=E)


@pytest.mark.parametrize("bad_sequence", ["ACDEF_", "ACDEFX", "acdefg", "ACDEF", ""])
def test_a_non_canonical_sequence_never_reaches_the_head_or_the_archive(bad_sequence):
    """§7.2 #17. The Head is fail-open on non-canonical input, so this guard is load-bearing."""
    with pytest.raises(Exception):
        _endpoint(sequence=bad_sequence)


def test_definitive_feasibility_requires_an_evaluated_structure():
    """§7.2 #20: a provisional structure result may not purchase feedback ancestry."""
    with pytest.raises(V2Error):
        _endpoint(structure_outcome=StructureOutcome.deferred("v0 will recheck"))
    with pytest.raises(V2Error):
        _endpoint(feasibility_level=st.FeasibilityLevel.PROVISIONAL,
                  structure_outcome=StructureOutcome(feasible=True))
    with pytest.raises(V2Error):
        _endpoint(structure_outcome=StructureOutcome(feasible=False))


def test_only_a_definitive_endpoint_may_become_feedback_ancestry():
    assert st.endpoint_may_become_ancestry(_endpoint())
    unvalidated = _endpoint(feasibility_level=st.FeasibilityLevel.UNVALIDATED,
                            structure_outcome=None)
    assert not st.endpoint_may_become_ancestry(unvalidated)


def test_endpoint_evidence_is_per_position_and_length_bound():
    with pytest.raises(V2Error):
        _endpoint(endpoint_provenance_evidence_by_pos=_endpoint()
                  .endpoint_provenance_evidence_by_pos[:-1])


def test_the_head_binding_must_describe_this_endpoints_bytes():
    other = ident.HeadScoreBinding(
        protein_id="5ZHV_B", sequence_md5=sequence_md5("ACDEFH"), sequence_length=L,
        window_grid_digest=D, evaluator=_safety_reference().head_binding.evaluator)
    with pytest.raises(V2Error):
        _endpoint(head_binding=other)


def test_endpoint_digest_binds_head_identity_and_full_structure_evidence():
    endpoint = _endpoint()
    other_evaluator = ident.HeadEvaluatorIdentity(
        allele="DRB1_0701", score_scale="nats", window_k_min=13, window_k_max=24,
        head_config_hash=D, head_checkpoint_digest=D,
    )
    other_binding = dataclasses.replace(endpoint.head_binding, evaluator=other_evaluator)
    assert _endpoint(head_binding=other_binding).content_digest != endpoint.content_digest

    other_windows = (
        dataclasses.replace(endpoint.head_score.windows[0], z=-0.6),
    )
    other_head_score = dataclasses.replace(endpoint.head_score, windows=other_windows)
    assert _endpoint(head_score=other_head_score).content_digest != endpoint.content_digest

    other_structure = StructureOutcome(feasible=True, metrics={"scTM": 0.81})
    assert _endpoint(structure_outcome=other_structure).content_digest != endpoint.content_digest


# ---- step 18: ArchiveEntry and FeedbackTransition ---------------------------------------------

def _entry(**over):
    ep = _endpoint()
    kw = dict(
        endpoint_id=ep.endpoint_id, endpoint_content_digest=ep.content_digest, lineage=ep.lineage,
        sequence_equivalence_key=SEQ_MD5, feasibility_level=st.FeasibilityLevel.DEFINITIVE,
        first_depth_seen=0, last_depth_seen=0,
        membership=st.ArchiveMembership(is_elite=True, elite_rank=0, is_diversity_frontier=False),
    )
    kw.update(over)
    return st.ArchiveEntry(**kw)


def test_the_archive_entry_is_the_endpoint_identity():
    entry = _entry()
    assert entry.entry_id == ident.make_archive_entry_id(entry.endpoint_id)
    assert entry.family_id == entry.lineage.family_id


@pytest.mark.parametrize(
    ("frm", "to"),
    [(st.FeasibilityLevel.DEFINITIVE, st.FeasibilityLevel.PROVISIONAL),
     (st.FeasibilityLevel.DEFINITIVE, st.FeasibilityLevel.UNVALIDATED),
     (st.FeasibilityLevel.PROVISIONAL, st.FeasibilityLevel.UNVALIDATED)],
)
def test_feasibility_may_advance_but_never_regress(frm, to):
    """PLAN §2.3 monotonicity is at the level of exact endpoint identity: a level may be upgraded
    in place, never downgraded."""
    entry = _entry(feasibility_level=frm)
    assert st.advance_archive_entry(entry, feasibility_level=st.FeasibilityLevel.DEFINITIVE,
                                    depth=1).last_depth_seen == 1
    with pytest.raises(V2Error):
        st.advance_archive_entry(entry, feasibility_level=to, depth=1)


def test_last_depth_seen_is_non_decreasing():
    with pytest.raises(V2Error):
        st.advance_archive_entry(_entry(first_depth_seen=2, last_depth_seen=2),
                                 feasibility_level=st.FeasibilityLevel.DEFINITIVE, depth=1)


def _transition(**over):
    proj, cap, ep = _projected(), _captured(), _endpoint()
    kw = dict(
        schema_version=ident.V2_STATE_SCHEMA_VERSION, lineage=cap.lineage, source_depth=0,
        coordinate_law=sch.CoordinateLaw.PROGRESSIVE_CHECKPOINT,
        r_step=R_STEP, c_step=C_SOURCE, c_next_step=C_NEXT,
        source_state_id=_live().state_id, source_state_content_digest=_live().content_digest,
        selected_endpoint_id=ep.endpoint_id, endpoint_sequence_md5=SEQ_MD5,
        endpoint_content_digest=ep.content_digest,
        policy=ident.ProjectionPolicyIdentity(
            policy_id="source_writeback_v1", policy_version="1", policy_config_digest=D,
            policy_spec_digest=D, is_diagnostic_only=False),
        support=proj.support, projected_state_id=proj.state_id,
        projected_state_content_digest=proj.content_digest,
        propagated_state_id=cap.state_id, propagated_state_content_digest=cap.content_digest,
        descendant_fork_seed=1234567890, fork_index=0, pair_id="5ZHV_B:d0:r40-c60:p0",
        matched_seed_context_digest=D,
        assimilation=st.AssimilationRecord(n_pending_at_reentry=2, n_assimilated=2,
                                           assimilation_temperature=1.0),
        segment_logical_dfe=C_NEXT - R_STEP, background_remask_events=0,
        cost_event_ids=("evt:seg",), parent_transition_id=None,
        outcome=st.TransitionOutcome.COMMITTED,
    )
    kw.update(over)
    return st.FeedbackTransition(**kw)


def test_a_committed_transition_records_every_edge():
    txn = _transition()
    assert txn.lineage.depth == txn.source_depth + 1
    assert ident.id_namespace(txn.transition_id) == "txn"


def test_transition_rejects_an_untyped_coordinate_law():
    """Transition telemetry must bind the same typed law as the projected state."""
    with pytest.raises(V2Error, match="coordinate_law"):
        _transition(coordinate_law=sch.CoordinateLaw.PROGRESSIVE_CHECKPOINT.value)


def test_transition_id_must_bind_its_full_lineage_coordinate():
    transition = _transition()
    foreign_coordinate = ident.make_transition_id(
        "5ZHV_B", "fam0", depth=0, r_step=R_STEP - 1, c_next_step=C_NEXT,
        content_digest=transition.content_digest,
    )
    with pytest.raises(V2Error, match="transition_id"):
        _transition(transition_id=foreign_coordinate)


@pytest.mark.parametrize(
    "field",
    ["endpoint_sequence_md5", "endpoint_content_digest",
     "projected_state_content_digest", "propagated_state_content_digest",
     "descendant_fork_seed", "fork_index", "pair_id", "matched_seed_context_digest"],
)
def test_committed_transition_requires_the_complete_identity_matrix(field):
    with pytest.raises(V2Error, match="committed"):
        _transition(**{field: None})


@pytest.mark.parametrize(
    ("id_field", "digest_field"),
    [("source_state_id", "source_state_content_digest"),
     ("projected_state_id", "projected_state_content_digest"),
     ("propagated_state_id", "propagated_state_content_digest")],
)
def test_transition_state_edges_bind_their_declared_content_digest(id_field, digest_field):
    with pytest.raises(V2Error, match="digest"):
        _transition(**{digest_field: E})


@pytest.mark.parametrize(
    ("id_field", "foreign_id"),
    [
        ("source_state_id", lambda txn: ident.make_live_state_id(
            "5ZHV_B", "fam0", depth=0, step=C_SOURCE - 1,
            content_digest=txn.source_state_content_digest,
        )),
        ("projected_state_id", lambda txn: ident.make_projected_state_id(
            "5ZHV_B", "fam0", depth=1, r_step=R_STEP - 1,
            content_digest=txn.projected_state_content_digest,
        )),
        ("propagated_state_id", lambda txn: ident.make_live_state_id(
            "5ZHV_B", "fam0", depth=1, step=C_NEXT - 1,
            content_digest=txn.propagated_state_content_digest,
        )),
    ],
)
def test_transition_state_edges_bind_their_full_lineage_coordinate(id_field, foreign_id):
    transition = _transition()
    with pytest.raises(V2Error, match=id_field):
        _transition(**{id_field: foreign_id(transition)})


def test_transition_endpoint_and_descendant_lineage_bind_the_selected_source_pair():
    foreign_source = ident.make_live_state_id(
        "5ZHV_B", "fam0", depth=0, step=50, content_digest=E,
    )
    foreign_endpoint = ident.make_endpoint_id(
        foreign_source, fork_index=0, sequence_md5_hex=SEQ_MD5,
    )
    with pytest.raises(V2Error, match="endpoint"):
        _transition(selected_endpoint_id=foreign_endpoint)

    foreign_lineage = dataclasses.replace(
        _projected().lineage,
        parent_state_id=foreign_source,
    )
    with pytest.raises(V2Error, match="lineage"):
        _transition(lineage=foreign_lineage)


def test_transition_digest_binds_policy_and_assimilation_evidence():
    transition = _transition()
    other_policy = dataclasses.replace(transition.policy, policy_config_digest=E)
    assert _transition(policy=other_policy).content_digest != transition.content_digest

    other_assimilation = st.AssimilationRecord(
        n_pending_at_reentry=2, n_assimilated=2, assimilation_temperature=2.0,
    )
    assert _transition(assimilation=other_assimilation).content_digest != transition.content_digest


def test_the_descendant_lineage_is_one_depth_below_the_source():
    with pytest.raises(V2Error):
        _transition(source_depth=1)


def test_the_segment_charges_exactly_c_next_minus_r():
    with pytest.raises(V2Error):
        _transition(segment_logical_dfe=(C_NEXT - R_STEP) + 1)


@pytest.mark.parametrize(
    "outcome",
    [st.TransitionOutcome.NULL_NO_ADMISSIBLE_ENDPOINT,
     st.TransitionOutcome.NULL_INVALID_POLICY_RESULT,
     st.TransitionOutcome.NULL_BAND_INCOMPATIBLE,
     st.TransitionOutcome.STALLED_NO_NOVEL_DESCENDANT],
)
def test_a_null_transition_may_not_claim_a_propagated_descendant(outcome):
    """PLAN §2.4: an invalid policy result creates an explicit null/stalled event rather than an
    invented state, and PLAN §4.5 requires the archive to survive it untouched."""
    with pytest.raises(V2Error):
        _transition(outcome=outcome)
    null = _transition(outcome=outcome, propagated_state_id=None,
                       propagated_state_content_digest=None, projected_state_id=None,
                       projected_state_content_digest=None, support=None,
                       selected_endpoint_id=None, endpoint_sequence_md5=None,
                       endpoint_content_digest=None, segment_logical_dfe=0,
                       assimilation=None, descendant_fork_seed=None, fork_index=None,
                       pair_id=None, matched_seed_context_digest=None)
    assert null.outcome is outcome


def test_a_null_transition_may_not_claim_matched_descendant_identity():
    with pytest.raises(V2Error, match="null/stalled"):
        _transition(
            outcome=st.TransitionOutcome.NULL_NO_ADMISSIBLE_ENDPOINT,
            propagated_state_id=None, propagated_state_content_digest=None,
            projected_state_id=None, projected_state_content_digest=None, support=None,
            selected_endpoint_id=None, endpoint_sequence_md5=None, endpoint_content_digest=None,
            segment_logical_dfe=0, assimilation=None, descendant_fork_seed=None, fork_index=None,
        )


def test_the_matched_seed_context_crosses_the_boundary_as_a_string():
    """``state.py`` never imports ``seeds.py``: the pair context arrives already digested, so no
    treatment content can travel back into a seed through the state layer."""
    import inspect
    assert "seeds" not in inspect.getsource(st)
