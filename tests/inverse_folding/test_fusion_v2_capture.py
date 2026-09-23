"""V2F3b: the library-owned depth-0 capture provider (PLAN task V2F3).

PLAN requires a provider that "reuses ``ContinuationRequest(at_step=c_0)``, hard-anchor,
conditioning, and replay primitives and adapts the fresh checkpoint into ``LivePartialState``; it
does not copy the nested V1 script closure."

That closure is ``root_generator`` inside ``build_entry_oracles``
(``scripts/rf_fusion_v1_oracles.py``), and it is tangled with model preparation, per-protein
backbone caches, and a Head scorer.  None of that belongs in a state adapter.  So the provider is
split: a PURE adapter that turns a checkpoint plus identity material into a ``LivePartialState``,
and a thin capture helper that obtains the checkpoint.  V2F7 later extracts the model-preparation
half into a shared factory that the V1 path delegates to byte-identically.

The acceptance bullet is "depth-0 capture has exact coordinate/DFE/anchor/provenance identity", so
each of those four is tested separately below.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from inverse_folding.reference_flow.config import (
    AmplificationConfig,
    HShuffleConfig,
    ReferenceFlowConfig,
    RemaskConfig,
    SamplerConfig,
    ScheduleConfig,
)
from inverse_folding.reference_flow.fusion_v2 import state as st
from inverse_folding.reference_flow.fusion_v2_runtime.capture import (
    V2CaptureError,
    adapt_checkpoint_to_live_state,
    capture_depth_zero,
)
from inverse_folding.reference_flow.sampler import PositionDependentDFMSampler
from tests.inverse_folding import _v2_fixtures as F

VOCAB = 33
C0 = 50


def _denoiser(x_t: torch.Tensor, t: float, struct) -> torch.Tensor:
    length = int(x_t.shape[0])
    logits = torch.full((length, VOCAB), float("-inf"), dtype=torch.float32)
    for position in range(length):
        for token in sorted(F.AA):
            logits[position, token] = float((position * 7 + token * 3 + int(x_t[position])) % 11)
    return logits


def _cfg(**over):
    kw = dict(n_steps=F.N_STEPS, seed=11, temperature=1.0)
    kw.update(over)
    return ReferenceFlowConfig(
        sampler=SamplerConfig(
            n_steps=kw["n_steps"], seed=kw["seed"], temperature=kw["temperature"],
            n_designs_per_protein=1,
            remask=RemaskConfig(enabled=True, fraction_scale=0.0),
        ),
        schedule=ScheduleConfig(base_form="linear"),
        amplification=AmplificationConfig(form="constant_one", h_source="h_processed"),
        h_shuffle=HShuffleConfig(enabled=False, seed=None),
    )


#: The constraint class, not an inference from the trajectory: PLAN §2.4 requires hard anchors to
#: be "identified from the constraint class, never inferred from an unmask step".
ANCHORS = {0: 10}


def _capture(**over):
    kw = dict(
        sampler=PositionDependentDFMSampler(mask_token_id=F.MASK, vocab_size=VOCAB),
        denoiser=_denoiser, config=_cfg(), sequence_length=F.L,
        h_values=np.zeros(F.L, dtype=np.float32), residue_token_ids=F.AA,
        at_step=C0, fixed_tokens=ANCHORS,
        lineage=F.lineage(), mask_token_id=F.MASK, aa_token_ids=F.AA,
        conditioning=F.conditioning(), safety_reference=F.safety_reference(),
        cost_event_ids=("evt:root",),
    )
    kw.update(over)
    return capture_depth_zero(**kw)


# --------------------------------------------------------------------------------------------
# it produces a legal live state at all
# --------------------------------------------------------------------------------------------


def test_the_provider_returns_a_live_partial_state():
    assert isinstance(_capture(), st.LivePartialState)


def test_the_captured_state_is_at_depth_zero_with_no_parent_edges():
    live = _capture()
    assert live.lineage.depth == 0
    assert live.lineage.parent_state_id is None
    assert live.lineage.parent_transition_id is None
    assert live.lineage.origin_endpoint_id is None


# --------------------------------------------------------------------------------------------
# exact COORDINATE identity
# --------------------------------------------------------------------------------------------


def test_the_captured_state_sits_on_the_requested_step():
    assert _capture().sampler_step == C0


@pytest.mark.parametrize("at_step", [30, 50, 70])
def test_the_capture_step_is_whatever_was_requested(at_step):
    assert _capture(at_step=at_step).sampler_step == at_step


def test_the_schedule_length_is_carried_through():
    assert _capture().n_steps == F.N_STEPS


# --------------------------------------------------------------------------------------------
# exact DFE identity
# --------------------------------------------------------------------------------------------


def test_the_accumulated_lineage_dfe_equals_the_paid_prefix():
    """A checkpoint at the top of step s has paid exactly s lane-DFE (the V1 checkpoint contract),
    and a depth-0 root has no earlier lineage to inherit."""
    assert _capture().accumulated_lineage_dfe == C0


@pytest.mark.parametrize("at_step", [30, 70])
def test_the_dfe_tracks_the_capture_step(at_step):
    assert _capture(at_step=at_step).accumulated_lineage_dfe == at_step


# --------------------------------------------------------------------------------------------
# exact ANCHOR identity
# --------------------------------------------------------------------------------------------


def test_hard_anchors_come_from_the_constraint_class():
    live = _capture()
    assert live.hard_anchors == tuple(sorted(ANCHORS.items()))


def test_an_anchor_holds_its_declared_token():
    live = _capture()
    for position, token in live.hard_anchors:
        assert live.tokens[position] == token


def test_the_editable_domain_is_the_complement_of_the_anchors():
    live = _capture()
    assert set(live.editable_positions) == set(range(F.L)) - set(ANCHORS)


def test_an_anchor_is_never_ranked():
    live = _capture()
    for position, _ in live.hard_anchors:
        assert live.active_origin_kind_by_pos[position] is st.ActiveOriginKind.HARD_ANCHOR
        assert live.active_score_status_by_pos[position] is st.ActiveScoreStatus.NOT_RANKED_ANCHOR
        assert live.active_sampler_score_by_pos[position] is None


def test_a_resolved_position_is_never_silently_promoted_to_an_anchor():
    """The dangerous inversion: inferring anchors from "has been unmasked" would turn every
    resolved position into a permanent constraint."""
    live = _capture()
    resolved = [i for i in live.editable_positions if live.tokens[i] != F.MASK]
    assert resolved                                        # the fixture really resolves some
    anchors = {p for p, _ in live.hard_anchors}
    assert anchors.isdisjoint(resolved)


# --------------------------------------------------------------------------------------------
# exact PROVENANCE identity
# --------------------------------------------------------------------------------------------


def test_every_position_carries_provenance_consistent_with_its_token():
    live = _capture()
    for position, provenance in enumerate(live.provenance_by_pos):
        assert provenance.last_origin.token == live.tokens[position]
        assert provenance.last_origin.origin_kind is live.active_origin_kind_by_pos[position]


def test_a_depth_zero_root_has_no_feedback_provenance():
    """Nothing has been injected yet, so no position may claim a feedback origin."""
    live = _capture()
    assert set(live.feedback_origin_ref_by_pos) == {st.FeedbackOriginRef.NONE}
    assert set(live.origin_transition_id_by_pos) == {None}


def test_a_resolved_editable_position_records_the_step_it_was_sampled_at():
    live = _capture()
    for position in live.editable_positions:
        if live.tokens[position] == F.MASK:
            continue
        commit = live.active_commit_depth_step_by_pos[position]
        assert commit is not None
        assert commit.depth == 0
        assert 0 <= commit.step < C0


def test_an_unresolved_position_is_canonically_masked():
    live = _capture()
    unresolved = [i for i in live.editable_positions if live.tokens[i] == F.MASK]
    assert unresolved                                      # the capture really is pre-terminal
    for position in unresolved:
        assert live.active_origin_kind_by_pos[position] is st.ActiveOriginKind.UNRESOLVED
        assert live.active_score_status_by_pos[position] is st.ActiveScoreStatus.MASKED
        assert live.active_sampler_score_by_pos[position] is None
        assert live.active_commit_depth_step_by_pos[position] is None


def test_a_resolved_editable_position_keeps_its_sampling_score():
    live = _capture()
    for position in live.editable_positions:
        if live.tokens[position] == F.MASK:
            continue
        assert live.active_score_status_by_pos[position] is \
            st.ActiveScoreStatus.HISTORICAL_NATURAL
        assert live.active_sampler_score_by_pos[position] is not None


def test_the_provenance_logprob_is_the_score_the_sampler_recorded():
    live = _capture()
    for position in live.editable_positions:
        if live.tokens[position] == F.MASK:
            continue
        assert live.provenance_by_pos[position].last_origin.evidence_logprob == \
            live.active_sampler_score_by_pos[position]


# --------------------------------------------------------------------------------------------
# replay identity
# --------------------------------------------------------------------------------------------


def test_the_root_replays_in_identity_mode_on_the_captured_rng_state():
    """A fresh root is an IDENTITY resume: replaying it must reproduce the uninterrupted suffix,
    which is only possible if the exact bit-generator state travelled with it."""
    live = _capture()
    assert live.replay.mode == "identity"
    assert live.replay.fork_seed is None
    assert live.replay.rng_state is not None


def test_the_replay_hash_binds_the_captured_state():
    """Two captures at different steps must not share a replay hash, or a tampered resume would
    pass the sampler's own integrity gate."""
    assert _capture(at_step=40).replay.replay_state_hash != \
        _capture(at_step=60).replay.replay_state_hash


# --------------------------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------------------------


def test_the_same_seed_and_step_reproduce_the_same_root():
    assert _capture().content_digest == _capture().content_digest


def test_a_different_sampler_seed_produces_a_different_root():
    """Positive control: a provider that ignored the config seed would collapse every root."""
    assert _capture(config=_cfg(seed=11)).content_digest != \
        _capture(config=_cfg(seed=12)).content_digest


# --------------------------------------------------------------------------------------------
# fail-closed contracts
# --------------------------------------------------------------------------------------------


def test_a_fully_resolved_capture_is_refused():
    """A root with no unresolved editable position is not pre-terminal: there is nothing left for
    a segment to denoise and no lookahead can fork from it.

    Either guard is acceptable.  In practice the V1 sampler's own ``InvalidPreterminalRootError``
    fires first, which is the better outcome -- it rejects inside the trajectory rather than after
    the adapter has built a state -- so the provider's check is a backstop for a checkpoint that
    arrived by some other route."""
    from inverse_folding.reference_flow.sampler import InvalidPreterminalRootError

    with pytest.raises((V2CaptureError, InvalidPreterminalRootError)):
        _capture(at_step=F.N_STEPS - 1)


def test_a_lineage_deeper_than_zero_is_refused():
    """A capture makes a ROOT.  A deeper state is produced by a projection, and accepting one here
    would let a descendant be manufactured with depth-0 provenance -- no parent transition, no
    origin endpoint, and an accumulated DFE that forgets everything its ancestors paid."""
    deep = F.lineage(depth=1, parent_state_id="live:5ZHV_B:d0:" + F.digest("p")[:12],
                     parent_transition_id=F.TXN,
                     origin_endpoint_id="endpoint:5ZHV_B:" + F.digest("e")[:12])
    with pytest.raises(V2CaptureError, match="depth"):
        _capture(lineage=deep)


def test_a_capture_step_outside_the_schedule_is_refused():
    with pytest.raises((V2CaptureError, ValueError)):
        _capture(at_step=F.N_STEPS)


def test_an_anchor_outside_the_sequence_is_refused():
    with pytest.raises((V2CaptureError, ValueError)):
        _capture(fixed_tokens={F.L + 3: 10})


def test_the_adapter_refuses_a_checkpoint_whose_length_disagrees_with_the_conditioning():
    """The pure adapter is reachable on its own, so it must not rely on the capture helper having
    already validated its inputs."""
    live = _capture()
    with pytest.raises(V2CaptureError):
        adapt_checkpoint_to_live_state(
            checkpoint=_capture_checkpoint(), lineage=F.lineage(), mask_token_id=F.MASK,
            aa_token_ids=F.AA, conditioning=F.conditioning(),
            safety_reference=F.safety_reference(), cost_event_ids=("evt:root",),
            sequence_length=live.n_steps,          # deliberately wrong
        )


def _capture_checkpoint():
    sampler = PositionDependentDFMSampler(mask_token_id=F.MASK, vocab_size=VOCAB)
    from inverse_folding.reference_flow.sampler import ContinuationRequest

    out = sampler.sample(
        sequence_length=F.L, h_values=np.zeros(F.L, dtype=np.float32), denoiser=_denoiser,
        config=_cfg(), controller=None, struct=None, fixed_tokens=ANCHORS,
        continuation=ContinuationRequest(at_step=C0, early_stop=True),
        residue_token_ids=F.AA,
    )
    return out.continuation_checkpoint


# --------------------------------------------------------------------------------------------
# the captured root is usable by the rest of V2
# --------------------------------------------------------------------------------------------


def test_a_captured_root_is_a_legal_source_for_the_projection_kernel():
    """The point of the provider: what it produces must be projectable.  A root that q_phi cannot
    consume would make the whole depth-0 entry path useless."""
    live = _capture()
    assert live.state_id.startswith("live:") or ":live:" in live.state_id
    assert live.realized_maturity.n_unresolved_editable >= 1
