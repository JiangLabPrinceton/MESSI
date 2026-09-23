"""V2F3b: the library-owned depth-0 capture provider (PLAN task V2F3).

PLAN requires a provider that "reuses ``ContinuationRequest(at_step=c_0)``, hard-anchor,
conditioning, and replay primitives and adapts the fresh checkpoint into ``LivePartialState``", and
is explicit that it "does not copy the nested V1 script closure".

That closure is ``root_generator`` inside ``build_entry_oracles``
(``scripts/rf_fusion_v1_oracles.py``).  It is tangled with backbone preparation, per-protein caches,
a denoiser factory, and a Head scorer -- none of which belongs in a state adapter.  So the work is
split in two here:

:func:`adapt_checkpoint_to_live_state`
    The PURE half.  Checkpoint plus identity material in, ``LivePartialState`` out.  No sampler, no
    model, no I/O.  This is the piece with real content, and it is independently testable.

:func:`capture_depth_zero`
    A thin helper that obtains the checkpoint through the V1 sampler's own
    ``ContinuationRequest(at_step=c_0)`` and hands it to the adapter.  Model preparation stays with
    the caller; PLAN V2F7 later extracts it into a shared factory that the V1 path delegates to.

The V2F3 acceptance bullet is "depth-0 capture has exact coordinate/DFE/anchor/provenance
identity", and each of those four is a separate obligation:

coordinate
    ``sampler_step`` is the requested ``c_0`` and ``n_steps`` is the schedule it was captured on.
DFE
    A checkpoint at the top of step ``s`` has paid exactly ``s`` lane-DFE, and a depth-0 root has
    no earlier lineage to inherit, so ``accumulated_lineage_dfe == paid_prefix_dfe``.
anchor
    Hard anchors come from the CONSTRAINT CLASS the caller declared, never inferred from an unmask
    step (PLAN §2.4).  Inferring them would silently promote every resolved position to a permanent
    constraint.
provenance
    Every position gets an ``OriginEvidence`` consistent with its own token, commit step, and
    sampling score -- and none of them may claim a feedback origin, because nothing has been
    injected at depth 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
import torch

from ..fusion_v2.errors import V2Error
from ..fusion_v2.identity import (
    V2_STATE_SCHEMA_VERSION,
    LineageRef,
    SafetyReferenceBinding,
    V2ConditioningIdentity,
    canonical_digest,
)
from ..fusion_v2.schedule import history_key
from ..fusion_v2.state import (
    ActiveOriginKind,
    ActiveScoreStatus,
    FeedbackOriginRef,
    LivePartialState,
    OriginEvidence,
    PositionProvenance,
    ReplayIdentity,
)
from ..sampler import (
    ContinuationCheckpoint,
    ContinuationRequest,
    InvalidPreterminalRootError,
    _replay_state_hash,
)

__all__ = [
    "V2CaptureError", "FullyResolvedRootError", "adapt_checkpoint_to_live_state",
    "capture_depth_zero",
]


class V2CaptureError(V2Error):
    """The captured root is not a legal depth-0 V2 state."""


class FullyResolvedRootError(V2CaptureError):
    """A retryable V2 capture miss: the requested checkpoint is already terminal."""


def adapt_checkpoint_to_live_state(
    *,
    checkpoint: ContinuationCheckpoint,
    lineage: LineageRef,
    mask_token_id: int,
    aa_token_ids: frozenset[int] | Sequence[int],
    conditioning: V2ConditioningIdentity,
    safety_reference: SafetyReferenceBinding,
    cost_event_ids: Sequence[str],
    sequence_length: int | None = None,
) -> LivePartialState:
    """Turn a V1 continuation checkpoint into a depth-0 :class:`LivePartialState`.

    Pure: no sampler, no model, no I/O.  ``sequence_length`` is optional and, when given, is
    checked against the checkpoint -- the adapter is reachable on its own, so it must not assume
    the capture helper already validated its inputs.
    """
    if not isinstance(checkpoint, ContinuationCheckpoint):
        raise V2CaptureError("checkpoint must be a ContinuationCheckpoint")
    if not isinstance(lineage, LineageRef):
        raise V2CaptureError("lineage must be a LineageRef")
    if lineage.depth != 0:
        raise V2CaptureError(
            f"a fresh root is depth 0, got depth={lineage.depth}; a deeper state is produced by a "
            "projection, not by a capture"
        )

    tokens = tuple(int(v) for v in checkpoint.x_t.detach().cpu().tolist())
    length = len(tokens)
    if sequence_length is not None and int(sequence_length) != length:
        raise V2CaptureError(
            f"sequence_length={sequence_length} disagrees with the checkpoint's {length} positions"
        )

    mask_token_id = int(mask_token_id)
    aa_token_ids = frozenset(int(t) for t in aa_token_ids)
    anchors = tuple(sorted((int(p), int(t)) for p, t in checkpoint.fixed_tokens))
    anchor_positions = {position for position, _ in anchors}
    editable = tuple(int(i) for i in checkpoint.editable_positions)

    scores = np.asarray(checkpoint.scores, dtype=np.float64)
    unmask = tuple(int(v) for v in checkpoint.unmask_step_by_pos)

    kinds: list[ActiveOriginKind] = []
    statuses: list[ActiveScoreStatus] = []
    active_scores: list[float | None] = []
    commits: list[Any] = []
    provenance: list[PositionProvenance] = []

    for position, token in enumerate(tokens):
        if position in anchor_positions:
            # An anchor is a permanent constraint: never ranked, never scored, never reopened.
            kind = ActiveOriginKind.HARD_ANCHOR
            status = ActiveScoreStatus.NOT_RANKED_ANCHOR
            score: float | None = None
            commit = None
        elif token == mask_token_id:
            kind = ActiveOriginKind.UNRESOLVED
            status = ActiveScoreStatus.MASKED
            score = None
            commit = None
        else:
            if token not in aa_token_ids:
                raise V2CaptureError(
                    f"editable position {position} holds token {token}, which is neither the mask "
                    "token nor a canonical residue; a captured root must be denoisable"
                )
            step = unmask[position]
            if step < 0:
                raise V2CaptureError(
                    f"position {position} is resolved but has no unmask step; its sampling history "
                    "is required to decide whether it may be carried across a re-entry boundary"
                )
            value = float(scores[position])
            if not np.isfinite(value):
                raise V2CaptureError(
                    f"position {position} is resolved but its sampler score is {value}; a resolved "
                    "natural token must carry the log-probability it was drawn with"
                )
            kind = ActiveOriginKind.DENOISER_SAMPLE
            status = ActiveScoreStatus.HISTORICAL_NATURAL
            score = value
            commit = history_key(0, int(step))

        kinds.append(kind)
        statuses.append(status)
        active_scores.append(score)
        commits.append(commit)
        origin = OriginEvidence(
            origin_kind=kind,
            origin_ref=FeedbackOriginRef.NONE,
            # An anchor and a still-masked position have no commit event of their own; the root's
            # own coordinate is the honest key for their evidence.
            commit=commit if commit is not None else history_key(0, int(checkpoint.step)),
            token=int(token),
            evidence_logprob=score,
            transition_id=None,
            evidence_digest=canonical_digest({
                "root": "v2:depth0",
                "protein_id": lineage.protein_id,
                "root_id": lineage.root_id,
                "step": int(checkpoint.step),
                "position": int(position),
                "token": int(token),
            }),
        )
        provenance.append(
            PositionProvenance(first_origin=origin, last_origin=origin, n_origin_events=1)
        )

    replay = ReplayIdentity(
        mode="identity",
        rng_state=checkpoint.rng_state,
        fork_seed=None,
        replay_state_hash=_replay_state_hash(
            checkpoint.x_t, checkpoint.scores, int(checkpoint.step), int(checkpoint.n_steps),
            anchors, editable, unmask, "identity", checkpoint.rng_state, None,
        ),
    )

    return LivePartialState(
        schema_version=V2_STATE_SCHEMA_VERSION,
        lineage=lineage,
        sampler_step=int(checkpoint.step),
        n_steps=int(checkpoint.n_steps),
        tokens=tokens,
        mask_token_id=mask_token_id,
        aa_token_ids=aa_token_ids,
        hard_anchors=anchors,
        editable_positions=editable,
        active_origin_kind_by_pos=tuple(kinds),
        feedback_origin_ref_by_pos=tuple([FeedbackOriginRef.NONE] * length),
        active_commit_depth_step_by_pos=tuple(commits),
        origin_transition_id_by_pos=tuple([None] * length),
        active_sampler_score_by_pos=tuple(active_scores),
        active_score_status_by_pos=tuple(statuses),
        provenance_by_pos=tuple(provenance),
        # Depth 0 has no ancestor segment, so nothing has expired.
        expired_protection=(),
        replay=replay,
        # A checkpoint at the top of step s has paid exactly s lane-DFE, and a fresh root inherits
        # nothing earlier (PLAN §3.4: the ledger must not charge a prefix twice).
        accumulated_lineage_dfe=int(checkpoint.paid_prefix_dfe),
        conditioning=conditioning,
        safety_reference=safety_reference,
        cost_event_ids=tuple(str(v) for v in cost_event_ids),
    )


def capture_depth_zero(
    *,
    sampler: Any,
    denoiser: Callable[[torch.Tensor, float, Any], torch.Tensor],
    config: Any,
    sequence_length: int,
    h_values: np.ndarray | Sequence[float],
    residue_token_ids: frozenset[int] | Sequence[int],
    at_step: int,
    fixed_tokens: Mapping[int, int] | None,
    lineage: LineageRef,
    mask_token_id: int,
    aa_token_ids: frozenset[int] | Sequence[int],
    conditioning: V2ConditioningIdentity,
    safety_reference: SafetyReferenceBinding,
    cost_event_ids: Sequence[str],
    struct: Any = None,
) -> LivePartialState:
    """Capture a fresh root at ``c_0`` and adapt it into a depth-0 :class:`LivePartialState`.

    Uses the V1 sampler's own ``ContinuationRequest(at_step=c_0)`` with ``early_stop=True``, so the
    root is the true pre-denoiser boundary state and no terminal residual completion happens.

    ``fixed_tokens`` is the CONSTRAINT CLASS.  It is passed through to the sampler and becomes the
    state's hard anchors; the adapter never infers an anchor from the trajectory.
    """
    if fixed_tokens and {int(p) for p in fixed_tokens} == set(range(int(sequence_length))):
        raise V2CaptureError("root has no editable positions")
    try:
        output = sampler.sample(
            sequence_length=int(sequence_length),
            h_values=h_values,
            denoiser=denoiser,
            config=config,
            struct=struct,
            controller=None,
            fixed_tokens=fixed_tokens,
            continuation=ContinuationRequest(at_step=int(at_step), early_stop=True),
            residue_token_ids=residue_token_ids,
        )
    except InvalidPreterminalRootError as exc:
        raise FullyResolvedRootError(str(exc)) from exc
    checkpoint = output.continuation_checkpoint
    if checkpoint is None:
        raise V2CaptureError(
            f"no checkpoint was captured at step {at_step}; the request never triggered"
        )
    if checkpoint.maturity.n_unresolved_editable < 1:
        raise FullyResolvedRootError(
            f"the root captured at step {at_step} has no unresolved editable position "
            "(rho_edit == 1.0); it is not pre-terminal, so no segment can denoise it and no "
            "lookahead can fork from it"
        )
    return adapt_checkpoint_to_live_state(
        checkpoint=checkpoint,
        lineage=lineage,
        mask_token_id=mask_token_id,
        aa_token_ids=aa_token_ids,
        conditioning=conditioning,
        safety_reference=safety_reference,
        cost_event_ids=cost_event_ids,
        sequence_length=int(sequence_length),
    )
