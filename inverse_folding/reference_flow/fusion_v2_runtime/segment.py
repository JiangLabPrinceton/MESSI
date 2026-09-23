"""V2F3: the projected-segment executor (PLAN §3.1, §3.3, §3.4).

Runs a :class:`ProjectedPartialState` from the re-entry step ``r_d`` forward to ``c_{d+1}`` and
captures the descendant checkpoint on the true pre-denoiser boundary.

**Why a separate executor rather than a branch in the V1 sampler.**  PLAN §3.1 offers "a V2-specific
sampler input **or segment API**", and forbids reaching the same effect by passing
``continuation_resume`` and ``continuation`` together: "A projected state has different identity and
accumulated-cost semantics and deserves a distinct input contract."  This module is that contract.
``inverse_folding/reference_flow/sampler.py`` is byte-identical to HEAD and stays that way, because
the V1 reference-flow results are published.

**What this module is allowed to own, and what it must not.**  It owns CONTROL FLOW only: where the
loop starts, where it stops, when pending injections are assimilated, when temporary protection
expires.  Every numerical primitive is IMPORTED from the V1 modules --
``positionwise_unmask_probabilities``, ``amplification_factor``, ``_sample_categorical``,
``_enforce_fixed_tokens``, ``_apply_reparam_remask``, ``_union_protected``, ``editable_maturity``,
``_replay_state_hash``.  Copying any of them would be the most dangerous edit in this codebase: a
V2 loop that drifted from the V1 loop would (a) stop A2 from being a matched control of the same
process, and (b) invalidate the ``B(r)`` band table, which was calibrated on the V1 loop.
``test_fusion_v2_segment`` pins this with an identity control: a segment with no injections and no
reopening must reproduce the V1 sampler byte for byte over the same step range.

**The one genuinely new semantic** is first-forward assimilation (PLAN §3.1): at ``r_d``, the raw
base logits of the first ordinary forward score every pending injected token into an active sampler
score *without resampling it*, at the frozen sampler temperature.  Until that happens the token has
no active score at all, and PLAN §3.3 requires it to be assimilated before its temporary protection
expires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from ..amplification import amplification_factor
from ..fusion_v2.errors import V2Error
from ..fusion_v2.state import (
    ActiveOriginKind,
    ActiveScoreStatus,
    ProjectedPartialState,
)
from ..sampler import (
    ContinuationCheckpoint,
    _apply_reparam_remask,
    _enforce_fixed_tokens,
    _replay_state_hash,
    _sample_categorical,
    _union_protected,
    editable_maturity,
)
from ..fusion_v2.schedule import history_key
from ..schedule import positionwise_unmask_probabilities
from .scoring import token_logprob

__all__ = [
    "adapt_segment_to_live_state",
    "V2SegmentError",
    "PendingAssimilationError",
    "BackgroundRemaskError",
    "SegmentOutcome",
    "AssimilationEvent",
    "run_projected_segment",
    "run_projected_segment_batch",
    "segment_is_terminal",
    "pending_positions",
]

#: The V1 sampler's sentinel for "this position has never been unmasked".
NEVER_UNMASKED = -1

#: The V1 sampler's pre-denoiser checkpoint phase label; reused so a V2 checkpoint is
#: indistinguishable from a V1 one to every downstream consumer.
CONTINUATION_PHASE = "pre_denoiser_after_previous_remask"


class V2SegmentError(V2Error):
    """A segment contract violation: a mistyped state, or a substrate that is not the frozen one."""


class PendingAssimilationError(V2SegmentError):
    """A committed non-anchor token was still pending when an active-score consumer ran.

    PLAN §3.1: "fail before any active-score consumer if a non-anchor committed token remains
    pending".  An unassimilated token has no active score, so letting it reach a ranking would
    either crash on a NaN or, worse, silently rank it against a fabricated value.
    """


class BackgroundRemaskError(V2SegmentError):
    """The segment masked a position that projection did not reopen.

    PLAN §3.1: "only an explicit projection may add masks".  A background remask inside a V2/A2
    propagation segment is a global reopen operator competing with ``q_phi`` for control of
    reopened identities (PLAN §7.2).
    """


@dataclass(frozen=True)
class AssimilationEvent:
    """One pending injected token turned into an active sampler score.

    ``score`` is explicitly the CURRENT token's log-probability under the first active re-entry
    forward.  PLAN §3.3 is emphatic that this "is not a retroactive claim about how that token was
    originally sampled" -- the endpoint's own completion log-probability stays in the immutable
    provenance namespace and never appears here.
    """

    position: int
    token: int
    step: int
    score: float
    origin_kind: ActiveOriginKind


@dataclass(frozen=True)
class SegmentOutcome:
    """What one propagation segment produced."""

    checkpoint: ContinuationCheckpoint
    assimilation: tuple[AssimilationEvent, ...]
    expired_protection: tuple[int, ...]
    logical_dfe: int
    #: Forward calls issued for this lane's GROUP.  In scalar mode the group is this lane alone, so
    #: it equals ``logical_dfe``; in batch mode one call served ``batch_size`` lanes and the count
    #: is the same number of CALLS, not lane-steps.  A ledger charges a group's device cost ONCE as
    #: ``physical_forwards`` and attributes ``physical_forwards / batch_size`` to each lane -- so
    #: summing this field across lanes would multiply the real GPU cost by the batch width.
    physical_forwards: int
    background_remask_events: int
    #: How many lanes shared each of those forwards.  1 in scalar mode.
    batch_size: int = 1
    #: PARENT-RESUME evidence: the replay hash of the projected state at ``r_d`` this segment ran
    #: from.  It is not the descendant's identity -- that is computed from the descendant's own
    #: fields in :func:`adapt_segment_to_live_state`, because the two states differ in tokens, in
    #: scores and in step.  Keeping one field for both would make ``replay_state_hash`` mean one
    #: thing at depth 0 and a different thing at every depth after it.
    resume_state_hash: str = ""


def pending_positions(projected: ProjectedPartialState) -> tuple[int, ...]:
    """Positions awaiting first-forward assimilation, ascending."""
    return tuple(
        index for index, status in enumerate(projected.active_score_status_by_pos)
        if status is ActiveScoreStatus.PENDING_ASSIMILATION
    )


def _assert_frozen_substrate(config: Any, projected: ProjectedPartialState,
                             controller: Any) -> None:
    """Refuse anything but the frozen V2 substrate, before any denoiser call (PLAN §1.5)."""
    if controller is not None:
        raise V2SegmentError(
            "a controller may not run in a V2 propagation segment; the frozen substrate is "
            "controller-free"
        )
    declared = int(config.sampler.n_steps)
    if declared != int(projected.n_steps):
        raise V2SegmentError(
            f"config n_steps={declared} disagrees with the projected state's "
            f"n_steps={projected.n_steps}; the schedule the band was calibrated on and the one "
            "being run must be the same"
        )
    scale = float(config.sampler.remask.fraction_scale)
    if scale != 0.0:
        raise V2SegmentError(
            f"remask.fraction_scale={scale} != 0.0; reparameterized remask is a global reopen "
            "operator that would compete with q_phi for control of reopened identities"
        )
    form = str(config.amplification.form)
    if form != "constant_one":
        raise V2SegmentError(
            f"amplification.form={form!r} != 'constant_one'; the frozen V2 substrate is unamplified"
        )
    if getattr(config, "h_shuffle", None) is not None and bool(config.h_shuffle.enabled):
        raise V2SegmentError("h-map shuffling is retired; the frozen V2 substrate is h-map-free")


@dataclass
class _LaneState:
    """Everything one propagation lane carries between steps.

    Both drivers below own CONTROL FLOW only -- when to call the denoiser and how many lanes one
    call serves.  Every per-lane semantic lives in :func:`_lane_consume_logits`, which exists once.
    That is what makes PLAN §3.1's "preserve scalar/batch semantic parity" a structural property
    rather than a coincidence two implementations happen to share.
    """

    projected: ProjectedPartialState
    length: int
    mask_token_id: int
    n_steps: int
    start_step: int
    stop_step: int
    temperature: float
    base_form: str
    g_values: np.ndarray
    fixed_tokens: dict
    fixed_positions: frozenset
    editable_positions: tuple
    aa_token_ids: frozenset
    x_t: torch.Tensor
    unmask_step_by_pos: list
    scores: np.ndarray
    rng: Any
    pending: set
    protected: set
    reopened_by_projection: set
    dt: float
    remask_enabled: bool
    remask_fraction_scale: float
    assimilation: list
    n_forward: int = 0
    checkpoint: ContinuationCheckpoint | None = None


def _lane_init(
    projected: ProjectedPartialState, *, config: Any, h_values, residue_token_ids, controller,
) -> _LaneState:
    if not isinstance(projected, ProjectedPartialState):
        raise V2SegmentError("projected must be a ProjectedPartialState")
    _assert_frozen_substrate(config, projected, controller)

    length = len(projected.tokens)
    h = np.asarray(h_values, dtype=np.float32)
    if h.shape != (length,):
        raise V2SegmentError(f"h_values shape {h.shape} does not match length {length}")

    fixed_tokens = {int(p): int(t) for p, t in projected.hard_anchors}
    return _LaneState(
        projected=projected,
        length=length,
        mask_token_id=int(projected.mask_token_id),
        n_steps=int(projected.n_steps),
        start_step=int(projected.r_step),
        stop_step=int(projected.c_next_step),
        temperature=float(config.sampler.temperature),
        base_form=str(config.schedule.base_form),
        g_values=amplification_factor(h, config.amplification),
        fixed_tokens=fixed_tokens,
        fixed_positions=frozenset(fixed_tokens),
        editable_positions=tuple(int(i) for i in projected.editable_positions),
        aa_token_ids=frozenset(int(t) for t in residue_token_ids),
        x_t=torch.tensor(projected.tokens, dtype=torch.long),
        unmask_step_by_pos=[
            NEVER_UNMASKED if commit is None else int(commit.step)
            for commit in projected.active_commit_depth_step_by_pos
        ],
        # A position with no active score is NaN, never 0.0 or -inf: both of those are plausible
        # log-probabilities that would silently enter a ranking (PLAN §2.4, no fabricated score).
        scores=np.array(
            [np.nan if value is None else float(value)
             for value in projected.active_sampler_score_by_pos],
            dtype=np.float64,
        ),
        rng=np.random.default_rng(int(projected.descendant_fork_seed)),
        pending=set(pending_positions(projected)),
        protected={p.position for p in projected.active_temporary_protection},
        reopened_by_projection={
            i for i in projected.editable_positions
            if projected.tokens[i] == int(projected.mask_token_id)
        },
        dt=1.0 / float(projected.n_steps),
        remask_enabled=bool(config.sampler.remask.enabled),
        remask_fraction_scale=float(config.sampler.remask.fraction_scale),
        assimilation=[],
    )


def _lane_capture_if_due(state: _LaneState, step: int, t: float) -> bool:
    """Take the descendant checkpoint at the TOP of ``c_{d+1}``, before that step's forward.

    That is the true pre-denoiser boundary the V1 checkpoint contract defines, and it makes the
    paid prefix DFE exactly what this segment charged.
    """
    if step != state.stop_step:
        return False
    if state.pending:
        raise PendingAssimilationError(
            f"positions {sorted(state.pending)} were never assimilated but the segment is "
            "capturing; an unscored token must not reach the descendant state"
        )
    maturity = editable_maturity(
        x_t=state.x_t, editable_positions=state.editable_positions,
        fixed_positions=state.fixed_positions, mask_token_id=state.mask_token_id,
        aa_token_ids=state.aa_token_ids,
    )
    state.checkpoint = ContinuationCheckpoint(
        snapshot_phase=CONTINUATION_PHASE, step=int(step), t=float(t),
        x_t=state.x_t.detach().clone(), scores=state.scores.copy(),
        unmask_step_by_pos=tuple(int(v) for v in state.unmask_step_by_pos),
        rng_state=state.rng.bit_generator.state,
        fixed_tokens=tuple(sorted((int(p), int(tok)) for p, tok in state.fixed_tokens.items())),
        editable_positions=state.editable_positions,
        paid_prefix_dfe=int(state.n_forward), n_steps=state.n_steps, maturity=maturity,
    )
    return True


def _lane_consume_logits(
    state: _LaneState, logits: torch.Tensor, *, step: int, t: float, sampler: Any,
) -> None:
    """One lane's entire reaction to one forward.  The only place segment semantics live."""
    if logits.shape != (state.length, sampler.vocab_size):
        raise V2SegmentError(
            f"denoiser must return logits with shape ({state.length}, {sampler.vocab_size})"
        )
    if torch.isnan(logits).any():
        raise V2SegmentError(f"NaN logits encountered at step={step} t={t:.6f}")
    last_logits = logits.detach().cpu()

    # ---- first-forward assimilation (PLAN §3.1) -----------------------------------------------
    # Score every pending injected token against THESE raw base logits, without resampling it.
    # It happens before any active-score consumer in this step -- the unmask draws below and the
    # remask lifecycle at the end both read ``scores``.
    if state.pending and step == state.start_step:
        ordered = sorted(state.pending)
        index = np.array(ordered, dtype=np.int64)
        tokens = np.array([int(state.x_t[i]) for i in ordered], dtype=np.int64)
        assimilated = token_logprob(last_logits[index], tokens, temperature=state.temperature)
        for position, token, value in zip(ordered, tokens, assimilated):
            state.scores[position] = float(value)
            state.assimilation.append(AssimilationEvent(
                position=int(position), token=int(token), step=int(step), score=float(value),
                origin_kind=state.projected.active_origin_kind_by_pos[position],
            ))
        state.pending.clear()

    if state.pending:
        raise PendingAssimilationError(
            f"positions {sorted(state.pending)} are still pending at step {step}; assimilation "
            "must happen on the first ordinary forward at r_d"
        )

    probs = positionwise_unmask_probabilities(
        t=t, dt=state.dt, g_values=state.g_values, base_form=state.base_form,
    )
    masked_positions = (state.x_t == state.mask_token_id).cpu().numpy()
    if masked_positions.any():
        draws = state.rng.random(state.length) < probs
        selected_positions = np.flatnonzero(masked_positions & draws)
        if selected_positions.size:
            selected_logits = last_logits[selected_positions] / state.temperature
            sampled_tokens, sampled_logp = _sample_categorical(selected_logits, state.rng)
            state.x_t[selected_positions] = sampled_tokens
            state.scores[selected_positions] = sampled_logp
            for position in selected_positions.tolist():
                if state.unmask_step_by_pos[position] < 0:
                    state.unmask_step_by_pos[position] = step
            _enforce_fixed_tokens(
                x_t=state.x_t, fixed_tokens=state.fixed_tokens,
                unmask_step_by_pos=state.unmask_step_by_pos, step=step,
            )

    if state.remask_enabled and step < state.n_steps - 1:
        # Temporary protection joins the permanent anchors in the protected set for the whole
        # segment.  PLAN §3.3: "Temporary positions remain part of the maturity denominator" --
        # they are protected from remask, not removed from the editable domain.
        #
        # NOTE this union is UNREACHABLE under the frozen V2 substrate: fraction_scale is 0.0, so
        # _apply_reparam_remask is a no-op and the protected set cannot change any outcome.  It is
        # kept because the substrate gate is a config check, not a type guarantee, and a future
        # authorized substrate would need it.  Mutation testing confirms no test can distinguish
        # its removal -- the load-bearing protection contract under this substrate is the
        # deterministic expiry below, which IS tested.
        before = (state.x_t == state.mask_token_id).cpu().numpy().copy()
        _apply_reparam_remask(
            x_t=state.x_t, scores=state.scores, unmask_step_by_pos=state.unmask_step_by_pos,
            mask_token_id=state.mask_token_id, step=step, n_steps=state.n_steps,
            rank_scores=None,
            protected_positions=_union_protected(frozenset(state.protected),
                                                 state.fixed_positions),
            cutoff_scale=state.remask_fraction_scale,
        )
        _enforce_fixed_tokens(
            x_t=state.x_t, fixed_tokens=state.fixed_tokens,
            unmask_step_by_pos=state.unmask_step_by_pos, step=step,
        )
        after = (state.x_t == state.mask_token_id).cpu().numpy()
        newly = int(np.count_nonzero(after & ~before))
        if newly:
            raise BackgroundRemaskError(
                f"{newly} position(s) were remasked at step {step}; only an explicit projection "
                "may add masks in a V2 propagation segment"
            )


def segment_is_terminal(outcome: SegmentOutcome, projected: ProjectedPartialState) -> bool:
    """True when the segment resolved every editable position.

    That is an ordinary trajectory outcome, not a fault: the result is a COMPLETE sequence, which is
    a terminal endpoint rather than a live partial state.  ``adapt_segment_to_live_state`` therefore
    cannot build a descendant from it, and PLAN §4.5 requires the caller to stop with a TYPED
    reason.  Discovering this by catching the state layer's exception would be wrong twice: the
    caller cannot tell that exception apart from real corruption, and a legitimate stop would be
    reported as a crash.
    """
    if not isinstance(outcome, SegmentOutcome):
        raise V2SegmentError("outcome must be a SegmentOutcome")
    mask_token_id = int(projected.mask_token_id)
    return not any(
        int(outcome.checkpoint.x_t[index]) == mask_token_id
        for index in projected.editable_positions
    )


def _lane_finish(state: _LaneState, *, batch_size: int) -> SegmentOutcome:
    if state.checkpoint is None:
        raise V2SegmentError(
            f"the segment never reached its declared next checkpoint at step {state.stop_step}"
        )
    projected = state.projected
    # Temporary protection is segment-local and expires deterministically at the captured
    # descendant checkpoint.  PLAN §3.3: "no implicit cross-depth protection is allowed" -- the
    # next explicit policy may independently protect or reopen the identity.
    still_masked_after = {
        i for i in state.editable_positions
        if int(state.checkpoint.x_t[i]) == state.mask_token_id
    }
    return SegmentOutcome(
        checkpoint=state.checkpoint,
        assimilation=tuple(state.assimilation),
        expired_protection=tuple(sorted(state.protected)),
        logical_dfe=state.stop_step - state.start_step,
        physical_forwards=state.n_forward,
        background_remask_events=len(still_masked_after - state.reopened_by_projection),
        batch_size=int(batch_size),
        resume_state_hash=_replay_state_hash(
            torch.tensor(projected.tokens, dtype=torch.long),
            np.array([np.nan if v is None else float(v)
                      for v in projected.active_sampler_score_by_pos], dtype=np.float64),
            state.start_step, state.n_steps, tuple(sorted(state.fixed_tokens.items())),
            state.editable_positions,
            tuple(NEVER_UNMASKED if c is None else int(c.step)
                  for c in projected.active_commit_depth_step_by_pos),
            "fork", None, int(projected.descendant_fork_seed),
        ),
    )


def run_projected_segment(
    *,
    projected: ProjectedPartialState,
    sampler: Any,
    denoiser: Callable[[torch.Tensor, float, Any], torch.Tensor],
    config: Any,
    h_values: np.ndarray | Sequence[float],
    residue_token_ids: Sequence[int] | frozenset[int],
    struct: Any = None,
    controller: Any = None,
) -> SegmentOutcome:
    """Propagate ``projected`` from ``r_d`` to ``c_{d+1}`` as a single lane.

    The loop mirrors ``PositionDependentDFMSampler.sample`` step for step, including the order and
    the count of RNG draws, so an injection-free segment is byte-identical to the V1 trajectory
    over the same range.  The RNG is a fresh stream from the descendant fork seed, exactly as the
    V1 sampler does for a ``fork`` resume.
    """
    state = _lane_init(projected, config=config, h_values=h_values,
                       residue_token_ids=residue_token_ids, controller=controller)
    for step in range(state.start_step, state.n_steps):
        t = step / float(state.n_steps)
        if _lane_capture_if_due(state, step, t):
            break
        logits = denoiser(state.x_t, t, struct)
        state.n_forward += 1
        _lane_consume_logits(state, logits, step=step, t=t, sampler=sampler)
    return _lane_finish(state, batch_size=1)


def run_projected_segment_batch(
    *,
    projected_lanes: Sequence[ProjectedPartialState],
    sampler: Any,
    batched_denoiser: Callable[
        [list[torch.Tensor], float, list[Any]], list[torch.Tensor]
    ],
    config: Any,
    h_values: np.ndarray | Sequence[float],
    residue_token_ids: Sequence[int] | frozenset[int],
    structs: Sequence[Any] | None = None,
    controller: Any = None,
) -> tuple[SegmentOutcome, ...]:
    """Propagate several projected lanes with ONE batched forward per step.

    Every per-lane semantic is :func:`_lane_consume_logits`, the same function the scalar driver
    calls, so PLAN §3.1's scalar/batch parity is structural: there is no second implementation to
    drift.  What differs is only how many lanes one forward serves.

    Lanes must share ``r_d``, ``c_{d+1}`` and ``n_steps``.  Grouping is the caller's job, exactly as
    it is for the V1 ``sample_batch`` resume path: lanes that do not run the same span cannot be
    served by one forward, and quietly running them anyway would either drop a lane mid-flight or
    charge it for steps it never took.
    """
    lanes = list(projected_lanes)
    if not lanes:
        return ()
    structs = list(structs) if structs is not None else [None] * len(lanes)
    if len(structs) != len(lanes):
        raise V2SegmentError(
            f"{len(structs)} struct entries for {len(lanes)} lanes")

    states = [
        _lane_init(lane, config=config, h_values=h_values,
                   residue_token_ids=residue_token_ids, controller=controller)
        for lane in lanes
    ]
    head = states[0]
    for index, state in enumerate(states[1:], start=1):
        for field, label in ((("start_step",), "r_d"), (("stop_step",), "c_{d+1}"),
                             (("n_steps",), "n_steps"), (("length",), "sequence length")):
            name = field[0]
            if getattr(state, name) != getattr(head, name):
                raise V2SegmentError(
                    f"batch lanes must share the same {label}: lane 0 has "
                    f"{getattr(head, name)}, lane {index} has {getattr(state, name)}; group lanes "
                    "before batching"
                )

    n_batched = 0
    for step in range(head.start_step, head.n_steps):
        t = step / float(head.n_steps)
        if _lane_capture_if_due(head, step, t):
            for state in states[1:]:
                _lane_capture_if_due(state, step, t)
            break
        logits_by_lane = batched_denoiser([state.x_t.clone() for state in states], t, structs)
        n_batched += 1
        if len(logits_by_lane) != len(states):
            raise V2SegmentError(
                f"batched_denoiser returned {len(logits_by_lane)} logit tensors for "
                f"{len(states)} lanes"
            )
        for state, logits in zip(states, logits_by_lane):
            state.n_forward = n_batched
            _lane_consume_logits(state, logits, step=step, t=t, sampler=sampler)

    return tuple(_lane_finish(state, batch_size=len(states)) for state in states)

def adapt_segment_to_live_state(
    *,
    projected: ProjectedPartialState,
    outcome: SegmentOutcome,
    conditioning: Any,
    safety_reference: Any,
    cost_event_ids: Sequence[str],
) -> Any:
    """Turn a completed segment into the descendant :class:`LivePartialState` at ``c_{d+1}``.

    PLAN §3.1 requires the segment to "emit a new replayable ``LivePartialState`` without terminal
    residual completion".  This is where the projection's provenance becomes the descendant's
    history:

    * an injected position keeps ``FEEDBACK_INJECTION`` and the transition that wrote it, but its
      status advances from ``pending_assimilation`` to ``assimilated`` -- the first forward gave it
      a real active score, so it is now rankable on the same scale as a natural token;
    * a position the SEGMENT resolved becomes an ordinary ``DENOISER_SAMPLE`` committed at the step
      it was drawn;
    * everything else carries its projected row unchanged.

    Every temporary protection is recorded as expired.  PLAN §3.3: protection is segment-local and
    "no implicit cross-depth protection is allowed", so the descendant inherits the expiry record,
    never the protection itself.
    """
    from ..fusion_v2.state import (
        LivePartialState,
        OriginEvidence,
        PositionProvenance,
        ReplayIdentity,
    )

    checkpoint = outcome.checkpoint
    tokens = tuple(int(v) for v in checkpoint.x_t.detach().cpu().tolist())
    scores_out = np.asarray(checkpoint.scores, dtype=np.float64)
    assimilated_at = {event.position: event for event in outcome.assimilation}

    kinds = list(projected.active_origin_kind_by_pos)
    refs = list(projected.feedback_origin_ref_by_pos)
    commits = list(projected.active_commit_depth_step_by_pos)
    txns = list(projected.origin_transition_id_by_pos)
    scores: list[float | None] = list(projected.active_sampler_score_by_pos)
    statuses = list(projected.active_score_status_by_pos)
    provenance = list(projected.provenance_by_pos)
    anchor_positions = {position for position, _ in projected.hard_anchors}

    for position, token in enumerate(tokens):
        if position in anchor_positions:
            continue
        if position in assimilated_at:
            # The injected identity now has a real active score from the first re-entry forward.
            event = assimilated_at[position]
            scores[position] = float(event.score)
            statuses[position] = ActiveScoreStatus.ASSIMILATED
            previous = provenance[position]
            last = previous.last_origin
            provenance[position] = PositionProvenance(
                first_origin=previous.first_origin,
                last_origin=OriginEvidence(
                    origin_kind=last.origin_kind, origin_ref=last.origin_ref,
                    commit=last.commit, token=int(token),
                    evidence_logprob=last.evidence_logprob,
                    transition_id=last.transition_id, evidence_digest=last.evidence_digest,
                ),
                n_origin_events=previous.n_origin_events,
            )
            continue
        if projected.tokens[position] == projected.mask_token_id and token != projected.mask_token_id:
            # The segment resolved a position the projection had reopened: an ordinary draw.
            step = int(checkpoint.unmask_step_by_pos[position])
            kinds[position] = ActiveOriginKind.DENOISER_SAMPLE
            refs[position] = projected.feedback_origin_ref_by_pos[position].__class__.NONE
            commits[position] = history_key(projected.lineage.depth, step)
            txns[position] = None
            scores[position] = float(scores_out[position])
            statuses[position] = ActiveScoreStatus.HISTORICAL_NATURAL
            origin = OriginEvidence(
                origin_kind=ActiveOriginKind.DENOISER_SAMPLE,
                origin_ref=projected.feedback_origin_ref_by_pos[position].__class__.NONE,
                commit=history_key(projected.lineage.depth, step), token=int(token),
                evidence_logprob=float(scores_out[position]), transition_id=None,
                evidence_digest=projected.content_digest,
            )
            previous = provenance[position]
            provenance[position] = PositionProvenance(
                first_origin=previous.first_origin, last_origin=origin,
                n_origin_events=previous.n_origin_events + 1,
            )

    from ..fusion_v2.identity import LineageRef

    # The descendant's replay hash must identify the DESCENDANT.  ``capture_depth_zero`` already
    # stores the hash of the state it captured, so a reader can verify a root by recomputing it; a
    # descendant that stored the parent's resume hash instead would use the same field for a
    # different quantity, and no consumer could tell which convention it was reading.
    #
    # It is computed here rather than in the segment because only here do the descendant's own
    # values exist: the checkpoint's score vector still carries NaN at an injected position -- the
    # sampler never redraws a token that is already unmasked -- while the descendant carries the
    # first-forward assimilation score there.
    capture_state_hash = _replay_state_hash(
        torch.tensor(tokens, dtype=torch.long),
        np.array([np.nan if v is None else float(v) for v in scores], dtype=np.float64),
        int(checkpoint.step), int(projected.n_steps),
        tuple((int(position), int(token)) for position, token in projected.hard_anchors),
        tuple(int(index) for index in projected.editable_positions),
        tuple(NEVER_UNMASKED if commit is None else int(commit.step) for commit in commits),
        "fork", None, int(projected.descendant_fork_seed),
    )

    return LivePartialState(
        schema_version=projected.schema_version,
        # The descendant's parent is the PROJECTED state, not the source: the projection is a real
        # state in the lineage graph, and naming the source here would erase the edge that records
        # what feedback actually did.
        lineage=LineageRef(
            protein_id=projected.lineage.protein_id, root_id=projected.lineage.root_id,
            family_id=projected.lineage.family_id, depth=projected.lineage.depth,
            parent_state_id=projected.state_id,
            parent_transition_id=projected.origin_transition_id,
            origin_endpoint_id=projected.lineage.origin_endpoint_id,
        ),
        sampler_step=int(checkpoint.step),
        n_steps=int(projected.n_steps),
        tokens=tokens,
        mask_token_id=int(projected.mask_token_id),
        aa_token_ids=projected.aa_token_ids,
        hard_anchors=projected.hard_anchors,
        editable_positions=projected.editable_positions,
        active_origin_kind_by_pos=tuple(kinds),
        feedback_origin_ref_by_pos=tuple(refs),
        active_commit_depth_step_by_pos=tuple(commits),
        origin_transition_id_by_pos=tuple(txns),
        active_sampler_score_by_pos=tuple(scores),
        active_score_status_by_pos=tuple(statuses),
        provenance_by_pos=tuple(provenance),
        # Protection is segment-local; the descendant inherits the EXPIRY RECORD, never the grant.
        expired_protection=projected.active_temporary_protection,
        replay=ReplayIdentity(
            mode="fork", rng_state=None, fork_seed=int(projected.descendant_fork_seed),
            replay_state_hash=capture_state_hash,
        ),
        accumulated_lineage_dfe=int(projected.inherited_lineage_dfe) + outcome.logical_dfe,
        conditioning=conditioning,
        safety_reference=safety_reference,
        cost_event_ids=tuple(str(v) for v in cost_event_ids),
    )
