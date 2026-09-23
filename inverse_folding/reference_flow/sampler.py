"""Sampler implementation for Phase C1 position-dependent DFM."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from .amplification import amplification_factor, shuffle_h_values
from .config import ReferenceFlowConfig
from .schedule import positionwise_unmask_probabilities


@dataclass(frozen=True)
class SamplerSnapshot:
    """Pre-controller (pre-D2) state captured at a chosen sampler step.

    Used by the SC-GR signal-direction diagnostic (PLAN_RF_SC_GR_signal_diag.md
    P1): ``x_t`` is the partial completion the denoiser saw at ``step``, and
    ``struct_logits`` is the RAW denoiser output BEFORE any controller / D2
    correction (so cheap-P reuses the same original structural logits). Capture
    is side-effect free -- enabling it never changes the sampled trajectory.
    """

    step: int
    t: float
    x_t: torch.Tensor
    struct_logits: torch.Tensor
    scores: np.ndarray
    unmask_step_by_pos: tuple[int, ...]


@dataclass(frozen=True)
class ResumeState:
    """Mid-trajectory state to resume a completion from (signal-diag P2 oracle Y).

    Continuing from a captured :class:`SamplerSnapshot`: the loop starts at
    ``start_step`` from ``x_t`` (instead of a fully-masked init), reusing the
    committed-token ``scores`` for remask ranking. Combine with ``fixed_tokens``
    to freeze a block ``A_B`` and ``controller=None`` for a controller-off
    oracle continuation to ``t=1``.
    """

    x_t: torch.Tensor
    scores: np.ndarray
    unmask_step_by_pos: Sequence[int]
    start_step: int


# --------------------------------------------------------------------------- #
# V1F1 pre-terminal continuation surface (PLAN_RF_REFINE_FUSION_V1 §2.3-2.5)
# --------------------------------------------------------------------------- #
CONTINUATION_PHASE = "pre_denoiser_after_previous_remask"


class MaturityNotReachedError(ValueError):
    """A requested pre-terminal ``rho_edit`` target was never crossed in-loop.

    Raised (never silently completed into a terminal root) so a no-crossing root-prefix
    attempt is an explicit, catchable outcome for the driver's coverage telemetry.
    """


class InvalidPreterminalRootError(ValueError):
    """A captured root has no unresolved editable residue (``rho_edit == 1.0``) or no editable
    domain at all -- it is not a valid pre-terminal partial root."""


class FixedTokenError(ValueError):
    """A hard-anchor fixed-token mapping is invalid: out-of-range index, an anchor equal to the
    mask token, a non-canonical (non-AA20) token, or two raw keys that normalize to the same
    position with different tokens (PLAN §2.3 conflicting/changed-anchor hard failure)."""


@dataclass(frozen=True)
class MaturityTelemetry:
    """Editable-maturity accounting at a root boundary (raw, recomputable telemetry).

    ``rho_edit`` is the resolved fraction over the EDITABLE domain (sampler residue domain
    minus permanent fixed anchors); ``rho_known_sequence_identity`` also credits anchors.
    Every count is a function of the CURRENT ``x_t`` only -- never unmask history.
    """

    length_total: int
    n_fixed: int
    n_editable: int
    n_resolved_editable: int
    n_unresolved_editable: int
    rho_edit: float
    rho_known_sequence_identity: float
    editable_mask: tuple[bool, ...]
    fixed_mask: tuple[bool, ...]
    unresolved_editable_mask: tuple[bool, ...]


@dataclass(frozen=True)
class ContinuationRequest:
    """Request a single pre-denoiser checkpoint, triggered by an integer ``at_step`` OR the
    first post-remask upward crossing of an editable-maturity target ``at_rho_edit``.
    ``early_stop`` returns the masked root without terminal residual completion; otherwise
    capture is side-effect free (the trajectory continues unchanged).
    """

    at_step: int | None = None
    at_rho_edit: float | None = None
    early_stop: bool = True

    def __post_init__(self) -> None:
        has_step = self.at_step is not None
        has_rho = self.at_rho_edit is not None
        if has_step == has_rho:
            raise ValueError(
                "ContinuationRequest requires exactly one of at_step / at_rho_edit"
            )
        if has_step and int(self.at_step) < 0:
            raise ValueError(f"at_step must be >= 0, got {self.at_step}")
        if has_rho and not (0.0 < float(self.at_rho_edit) <= 1.0):
            raise ValueError(f"at_rho_edit must be in (0, 1], got {self.at_rho_edit}")


@dataclass(frozen=True)
class ContinuationCheckpoint:
    """A pre-denoiser (post-previous-remask) partial root: the exact state to fork/replay a
    completion from. Distinct from the post-forward diagnostic ``SamplerSnapshot`` -- captured
    at the TOP of the step loop BEFORE the step's denoiser forward, so a checkpoint at step
    ``s`` has paid exactly ``s`` lane-DFE. ``rng_state`` is the numpy bit-generator state for
    exact identity replay (consumed by V1F2). The content/conditioning hash envelope is added
    by the torch-free partial-entry layer (V1F3), not here.
    """

    snapshot_phase: str
    step: int
    t: float
    x_t: torch.Tensor
    scores: np.ndarray
    unmask_step_by_pos: tuple[int, ...]
    rng_state: dict[str, Any]
    fixed_tokens: tuple[tuple[int, int], ...]
    editable_positions: tuple[int, ...]
    paid_prefix_dfe: int
    n_steps: int
    maturity: MaturityTelemetry


def _replay_state_hash(
    x_t: torch.Tensor,
    scores: np.ndarray,
    start_step: int,
    n_steps: int,
    fixed_tokens: Sequence[tuple[int, int]],
    editable_positions: Sequence[int],
    unmask_step_by_pos: Sequence[int],
    mode: str,
    rng_state: "dict[str, Any] | None",
    fork_seed: "int | None",
) -> str:
    """A deterministic integrity digest over the replay-critical fields of a continuation. A
    resume whose x_t / scores / step / n_steps / anchors / editable set / unmask history / replay
    stream do not match this digest is a tampered or mismatched state and must fail before any
    denoiser call (PLAN §2.4). The stream binding is mode-specific: an ``identity`` resume binds
    its exact ``rng_state`` (so swapping in a *different valid* RNG state -- which would otherwise
    pass restore yet silently produce a non-identity suffix -- is caught), a ``fork`` resume binds
    its ``fork_seed``."""
    stream: dict[str, Any] = {"mode": mode}
    if mode == "identity":
        stream["rng_state"] = json.loads(json.dumps(rng_state, sort_keys=True, default=str))
    elif mode == "fork":
        stream["fork_seed"] = None if fork_seed is None else int(fork_seed)
    payload = json.dumps(
        {
            "x_t": [int(v) for v in x_t.detach().cpu().tolist()],
            "scores": [float(v) for v in np.asarray(scores, dtype=np.float64).tolist()],
            "start_step": int(start_step),
            "n_steps": int(n_steps),
            "fixed_tokens": sorted([int(p), int(tok)] for p, tok in fixed_tokens),
            "editable_positions": sorted(int(v) for v in editable_positions),
            "unmask_step_by_pos": [int(v) for v in unmask_step_by_pos],
            "stream": stream,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ContinuationResume:
    """Self-contained resume of a :class:`ContinuationCheckpoint`.

    ``mode == "identity"`` restores the exact numpy bit-generator state so the resumed run
    reproduces the uninterrupted suffix byte-for-byte (exact replay). ``mode == "fork"`` starts
    a fresh RNG from ``fork_seed`` -- an independent continuation from the same root bytes.
    Identity replay and fork are the two disjoint APIs of PLAN §2.6; a fork seed never
    impersonates the identity stream. The resume carries ``n_steps``, ``fixed_tokens``,
    ``editable_positions``, and a ``state_hash`` so the completion is self-contained: anchors
    are re-protected without the caller re-passing them, an n_steps mismatch is rejected, and a
    tampered ``x_t`` / ``scores`` fails the hash before any denoiser call.
    """

    x_t: torch.Tensor
    scores: np.ndarray
    unmask_step_by_pos: tuple[int, ...]
    start_step: int
    n_steps: int
    fixed_tokens: tuple[tuple[int, int], ...]
    editable_positions: tuple[int, ...]
    mode: str
    state_hash: str
    rng_state: "dict[str, Any] | None" = None
    fork_seed: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("identity", "fork"):
            raise ValueError(f"mode must be 'identity' or 'fork', got {self.mode!r}")
        if self.mode == "identity" and self.rng_state is None:
            raise ValueError("identity resume requires rng_state")
        if self.mode == "fork" and self.fork_seed is None:
            raise ValueError("fork resume requires fork_seed")

    @classmethod
    def _from_checkpoint(
        cls, checkpoint: "ContinuationCheckpoint", *, mode: str, **kw: Any
    ) -> "ContinuationResume":
        return cls(
            x_t=checkpoint.x_t,
            scores=checkpoint.scores,
            unmask_step_by_pos=checkpoint.unmask_step_by_pos,
            start_step=checkpoint.step,
            n_steps=checkpoint.n_steps,
            fixed_tokens=checkpoint.fixed_tokens,
            editable_positions=checkpoint.editable_positions,
            mode=mode,
            state_hash=_replay_state_hash(
                checkpoint.x_t, checkpoint.scores, checkpoint.step, checkpoint.n_steps,
                checkpoint.fixed_tokens, checkpoint.editable_positions,
                checkpoint.unmask_step_by_pos, mode, kw.get("rng_state"), kw.get("fork_seed"),
            ),
            **kw,
        )

    @classmethod
    def identity(cls, checkpoint: "ContinuationCheckpoint") -> "ContinuationResume":
        """Exact-replay resume: restore the checkpoint's RNG stream."""
        return cls._from_checkpoint(checkpoint, mode="identity", rng_state=checkpoint.rng_state)

    @classmethod
    def fork(cls, checkpoint: "ContinuationCheckpoint", fork_seed: int) -> "ContinuationResume":
        """Independent continuation from the same root bytes under a frozen fork seed."""
        return cls._from_checkpoint(checkpoint, mode="fork", fork_seed=int(fork_seed))


def editable_maturity(
    *,
    x_t: torch.Tensor,
    editable_positions: "Sequence[int] | range",
    fixed_positions: "Sequence[int] | frozenset[int]",
    mask_token_id: int,
    aa_token_ids: "Sequence[int] | frozenset[int]",
) -> MaturityTelemetry:
    """Compute editable maturity from the CURRENT ``x_t`` (never unmask history).

    ``rho_edit`` is the AA20-resolved fraction over the editable domain: a resolved editable
    position is one whose current token is in ``aa_token_ids``. A non-mask editable token that
    is NOT canonical AA20 (unknown / padding / out-of-range) is a HARD failure -- it must never
    be silently counted as mature (PLAN §2.3; guards against malformed resume / fixed state).
    ``editable_positions`` is the explicit editable domain (sampler residue domain minus
    permanent fixed anchors); it must be disjoint from ``fixed_positions``. ``n_editable == 0``
    is a hard failure -- there is no maturity to define.
    """
    length_total = int(x_t.shape[0])
    editable = tuple(sorted({int(i) for i in editable_positions}))
    fixed = frozenset(int(i) for i in fixed_positions)
    aa_ids = frozenset(int(i) for i in aa_token_ids)
    mask_id = int(mask_token_id)
    if mask_id in aa_ids:
        raise ValueError("mask_token_id must not be a member of aa_token_ids")
    if set(editable) & fixed:
        raise ValueError("editable_positions and fixed_positions must be disjoint")
    n_editable = len(editable)
    if n_editable == 0:
        raise ValueError(
            "editable set is empty (n_editable == 0): no editable maturity is defined"
        )
    x = x_t.detach().cpu()
    resolved_positions = set()
    for i in editable:
        tok = int(x[i].item())
        if tok == mask_id:
            continue
        if tok not in aa_ids:
            raise ValueError(
                f"editable position {i} holds non-canonical token {tok} "
                "(neither mask nor AA20); refusing to count it as mature"
            )
        resolved_positions.add(i)
    resolved = frozenset(resolved_positions)
    n_resolved = len(resolved)
    n_unresolved = n_editable - n_resolved
    n_fixed = len(fixed)
    editable_set = frozenset(editable)
    editable_mask = tuple(i in editable_set for i in range(length_total))
    fixed_mask = tuple(i in fixed for i in range(length_total))
    unresolved_editable_mask = tuple(
        (i in editable_set) and (i not in resolved) for i in range(length_total)
    )
    rho_edit = n_resolved / n_editable
    rho_known = (n_fixed + n_resolved) / length_total if length_total else 0.0
    return MaturityTelemetry(
        length_total=length_total,
        n_fixed=n_fixed,
        n_editable=n_editable,
        n_resolved_editable=n_resolved,
        n_unresolved_editable=n_unresolved,
        rho_edit=float(rho_edit),
        rho_known_sequence_identity=float(rho_known),
        editable_mask=editable_mask,
        fixed_mask=fixed_mask,
        unresolved_editable_mask=unresolved_editable_mask,
    )


def _validate_preterminal_root(maturity: MaturityTelemetry) -> None:
    """A pre-terminal partial root must have an editable domain and at least one unresolved
    editable residue (``rho_edit < 1.0``); otherwise there is nothing left to steer."""
    if maturity.n_editable == 0:
        raise InvalidPreterminalRootError("root has no editable positions")
    if maturity.n_unresolved_editable == 0:
        raise InvalidPreterminalRootError(
            "pre-terminal root has no unresolved editable residue (rho_edit == 1.0); "
            "not a valid partial root"
        )


def _is_upward_crossing(prev_rho: float | None, cur_rho: float, target: float) -> bool:
    """True when maturity crosses ``target`` upward at this boundary (the first boundary at or
    above target). Evaluated only at pre-denoiser boundaries, so a within-step transient
    (post-sampling, pre-remask) maturity is never a crossing candidate."""
    return cur_rho >= target and (prev_rho is None or prev_rho < target)


def _normalize_and_validate_fixed_tokens(
    fixed_tokens: "Mapping[int, int] | None",
    sequence_length: int,
    mask_token_id: int,
    aa_token_ids: "frozenset[int] | Sequence[int] | None" = None,
) -> dict[int, int]:
    """Normalize hard-anchor keys to int positions and fail-fast on any invalid anchor.

    Rejects (PLAN §2.3 hard failures): an out-of-range index; an anchor equal to the mask
    token; an anchor outside the canonical AA set (when ``aa_token_ids`` is supplied); and two
    raw keys that normalize to the same position with different tokens (silent-overwrite /
    conflicting anchor). Returns the normalized ``{position: token}`` mapping.
    """
    if not fixed_tokens:
        return {}
    mask_id = int(mask_token_id)
    aa_ids = frozenset(int(i) for i in aa_token_ids) if aa_token_ids is not None else None
    resolved: dict[int, int] = {}
    for raw_key, raw_val in fixed_tokens.items():
        pos = int(raw_key)
        token = int(raw_val)
        if not (0 <= pos < sequence_length):
            raise FixedTokenError(
                f"fixed_tokens position {pos} out of range [0, {sequence_length})"
            )
        if token == mask_id:
            raise FixedTokenError(
                f"fixed_tokens position {pos} anchors the mask token {token}"
            )
        if aa_ids is not None and token not in aa_ids:
            raise FixedTokenError(
                f"fixed_tokens position {pos} anchors non-canonical token {token} "
                "(not in the AA20 token-id set)"
            )
        if pos in resolved and resolved[pos] != token:
            raise FixedTokenError(
                f"conflicting fixed_tokens for position {pos}: {resolved[pos]} vs {token} "
                "(keys normalize to the same position)"
            )
        resolved[pos] = token
    return resolved


@dataclass
class SamplerOutput:
    tokens: torch.Tensor
    unmask_step_by_pos: list[int]
    g_values: list[float]
    trajectory_rows: list[dict[str, Any]]
    # Positions held fixed by a hard-anchor constraint (uricase enzyme mode v0).
    # Distinct from ordinary residues that merely commit at step 0.
    fixed_positions: tuple[int, ...] = ()
    # Pre-controller snapshots captured at requested steps (signal-diag P1).
    # Empty unless ``snapshot_steps`` was passed to ``sample``.
    snapshots: tuple[SamplerSnapshot, ...] = ()
    # V1F1 pre-terminal continuation (PLAN_RF_REFINE_FUSION_V1 §2.5). Non-None only when a
    # ``continuation`` request fired; ``stopped_early`` marks an early-stop masked root.
    continuation_checkpoint: "ContinuationCheckpoint | None" = None
    stopped_early: bool = False
    # V1F2 exact-replay support: the numpy bit-generator state at the end of this call (deep
    # copied). Lets identity replay assert final-RNG equality against the uninterrupted run.
    final_rng_state: "dict[str, Any] | None" = None
    # V1F2 cost/replay reconciliation telemetry: the committed-token scores at the end, and the
    # logical denoiser-forward equivalents charged in THIS call (lane-DFE).
    final_scores: tuple[float, ...] = ()
    logical_dfe: int = 0


@dataclass(frozen=True)
class SamplerBatchLane:
    sequence_length: int
    h_values: np.ndarray | list[float]
    config: ReferenceFlowConfig
    struct: Any = None
    shuffle_seed: int | None = None
    controller: Any | None = None
    protein_id: str = ""
    design_idx: int = 0
    fixed_tokens: Mapping[int, int] | None = None
    # V1F2 resumed batch: per-lane exact-replay / fork continuation from a root checkpoint.
    # All lanes in one sample_batch call must share the resume start step (group upstream).
    resume: "ContinuationResume | None" = None


@dataclass
class _SamplerLaneState:
    lane: SamplerBatchLane
    g_values: np.ndarray
    rng: np.random.Generator
    x_t: torch.Tensor
    unmask_step_by_pos: list[int]
    scores: np.ndarray
    trajectory_rows: list[dict[str, Any]]
    last_logits: torch.Tensor | None
    remask_enabled: bool
    remask_fraction_scale: float
    fixed_positions: frozenset[int]
    start_step: int = 0
    # Resolved anchor map (from the lane or, for a resumed lane, from the resume) so anchors are
    # re-enforced from the right source even when lane.fixed_tokens is None on resume.
    fixed_tokens: "Mapping[int, int] | None" = None


def _resolve_fixed_positions(
    fixed_tokens: Mapping[int, int] | None, sequence_length: int
) -> frozenset[int]:
    """Validate and collect hard-anchor positions (fail-fast on out-of-range index)."""
    if not fixed_tokens:
        return frozenset()
    positions: set[int] = set()
    for idx in fixed_tokens:
        i = int(idx)
        if not (0 <= i < sequence_length):
            raise ValueError(
                f"fixed_tokens position {i} out of range [0, {sequence_length})"
            )
        positions.add(i)
    return frozenset(positions)


def _union_protected(
    post_protected: tuple[int, ...], fixed_positions: frozenset[int]
) -> tuple[int, ...]:
    """Union controller-protected positions with the permanent hard-anchor set.

    With no fixed positions this returns ``post_protected`` unchanged so the
    no-constraint path stays bit-equivalent to legacy behavior.
    """
    if not fixed_positions:
        return post_protected
    return tuple(sorted(set(post_protected) | fixed_positions))


def _enforce_fixed_tokens(
    *,
    x_t: torch.Tensor,
    fixed_tokens: Mapping[int, int] | None,
    unmask_step_by_pos: list[int],
    step: int,
) -> None:
    """Restore permanent hard-anchor tokens after any sampling/remask mutation."""
    if not fixed_tokens:
        return
    for fixed_idx, fixed_token_id in fixed_tokens.items():
        i = int(fixed_idx)
        x_t[i] = int(fixed_token_id)
        if unmask_step_by_pos[i] < 0:
            unmask_step_by_pos[i] = int(step)


class PositionDependentDFMSampler:
    """Sampling-only position-dependent discrete flow matcher."""

    def __init__(self, *, mask_token_id: int, vocab_size: int) -> None:
        self.mask_token_id = int(mask_token_id)
        self.vocab_size = int(vocab_size)

    def sample(
        self,
        *,
        sequence_length: int,
        h_values: np.ndarray | list[float],
        denoiser: Callable[[torch.Tensor, float, Any], torch.Tensor],
        config: ReferenceFlowConfig,
        struct: Any = None,
        save_trajectories: bool = False,
        shuffle_seed: int | None = None,
        controller: Any | None = None,
        protein_id: str = "",
        design_idx: int = 0,
        fixed_tokens: Mapping[int, int] | None = None,
        snapshot_steps: Sequence[int] | None = None,
        initial_state: ResumeState | None = None,
        continuation: "ContinuationRequest | None" = None,
        residue_token_ids: "Sequence[int] | frozenset[int] | None" = None,
        continuation_resume: "ContinuationResume | None" = None,
    ) -> SamplerOutput:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        h = np.asarray(h_values, dtype=np.float32)
        if h.shape != (sequence_length,):
            raise ValueError(
                f"h_values shape {h.shape} does not match sequence_length={sequence_length}"
            )
        if config.h_shuffle.enabled:
            effective_shuffle_seed = (
                int(config.h_shuffle.seed)
                if shuffle_seed is None
                else int(shuffle_seed)
            )
            h = shuffle_h_values(h, seed=effective_shuffle_seed)

        g_values = amplification_factor(h, config.amplification)
        rng = np.random.default_rng(int(config.sampler.seed))
        # V1F1 continuation preconditions + hard-anchor firewall (PLAN §2.2/§2.3). These run
        # for every call; with no fixed tokens and no continuation they are a no-op, so legacy
        # behavior stays byte-identical.
        if continuation is not None and initial_state is not None:
            raise ValueError(
                "continuation with initial_state is not supported in V1F1 "
                "(resume + continuation DFE semantics are deferred to V1F2)"
            )
        if continuation_resume is not None and (
            initial_state is not None or continuation is not None
        ):
            raise ValueError(
                "continuation_resume is mutually exclusive with initial_state and continuation"
            )
        if continuation_resume is not None:
            # Self-contained replay contract: the resume carries its own anchors, n_steps, and a
            # tamper digest, so anchors are re-protected without the caller re-passing them, an
            # n_steps mismatch is rejected, and a tampered state fails before any denoiser call.
            if fixed_tokens is not None:
                raise ValueError(
                    "continuation_resume is self-contained; do not also pass fixed_tokens"
                )
            if int(continuation_resume.n_steps) != int(config.sampler.n_steps):
                raise ValueError(
                    f"continuation_resume.n_steps {continuation_resume.n_steps} != config "
                    f"n_steps {int(config.sampler.n_steps)}"
                )
            expected_hash = _replay_state_hash(
                continuation_resume.x_t, continuation_resume.scores,
                continuation_resume.start_step, continuation_resume.n_steps,
                continuation_resume.fixed_tokens, continuation_resume.editable_positions,
                continuation_resume.unmask_step_by_pos, continuation_resume.mode,
                continuation_resume.rng_state, continuation_resume.fork_seed,
            )
            if expected_hash != continuation_resume.state_hash:
                raise ValueError(
                    "continuation_resume state_hash mismatch (tampered or inconsistent state)"
                )
            fixed_tokens = {int(p): int(tok) for p, tok in continuation_resume.fixed_tokens}
        aa_token_ids = (
            frozenset(int(i) for i in residue_token_ids)
            if residue_token_ids is not None
            else None
        )
        if continuation is not None and aa_token_ids is None:
            raise ValueError(
                "continuation requires residue_token_ids (the canonical AA20 token-id set) so "
                "maturity counts AA20 membership, not merely non-mask tokens"
            )
        fixed_tokens = _normalize_and_validate_fixed_tokens(
            fixed_tokens, sequence_length, self.mask_token_id, aa_token_ids
        )
        fixed_positions = _resolve_fixed_positions(fixed_tokens, sequence_length)
        # Resume from an exact continuation checkpoint (V1F2), a mid-trajectory diagnostic
        # snapshot (signal-diag P2 ResumeState), or start fresh. Default (all None) is
        # byte-for-byte the legacy init.
        if continuation_resume is not None:
            x_t = continuation_resume.x_t.detach().clone().to(torch.long)
            if x_t.shape != (sequence_length,):
                raise ValueError(
                    f"continuation_resume.x_t shape {tuple(x_t.shape)} != ({sequence_length},)"
                )
            unmask_step_by_pos = [int(v) for v in continuation_resume.unmask_step_by_pos]
            scores = np.asarray(continuation_resume.scores, dtype=np.float64).copy()
            if len(unmask_step_by_pos) != sequence_length or scores.shape != (
                sequence_length,
            ):
                raise ValueError("continuation_resume arrays must match sequence_length")
            start_step = int(continuation_resume.start_step)
            if continuation_resume.mode == "identity":
                # Restore the exact RNG stream -> reproduce the uninterrupted suffix.
                rng.bit_generator.state = copy.deepcopy(continuation_resume.rng_state)
            else:  # fork -> a fresh, independent stream from the frozen fork seed
                rng = np.random.default_rng(int(continuation_resume.fork_seed))
        elif initial_state is not None:
            x_t = initial_state.x_t.detach().clone().to(torch.long)
            if x_t.shape != (sequence_length,):
                raise ValueError(
                    f"initial_state.x_t shape {tuple(x_t.shape)} != ({sequence_length},)"
                )
            unmask_step_by_pos = [int(v) for v in initial_state.unmask_step_by_pos]
            scores = np.asarray(initial_state.scores, dtype=np.float64).copy()
            if len(unmask_step_by_pos) != sequence_length or scores.shape != (
                sequence_length,
            ):
                raise ValueError("initial_state arrays must match sequence_length")
            start_step = int(initial_state.start_step)
        else:
            x_t = torch.full((sequence_length,), self.mask_token_id, dtype=torch.long)
            unmask_step_by_pos = [-1] * sequence_length
            # log-prob of the currently committed token at each position; -inf for
            # positions still masked (or freshly remasked). Used by the optional
            # reparam refinement to pick the bottom-k committed positions.
            scores = np.full(sequence_length, -np.inf, dtype=np.float64)
            start_step = 0
        # Hard-anchor constraint (uricase enzyme mode v0) / frozen block A_B (signal-diag P2):
        # seed fixed tokens and mark them committed at the start step. An exact continuation
        # checkpoint already carries anchors with their real unmask history, so re-seeding would
        # corrupt that history -> skip it (anchors stay protected via ``fixed_positions`` and
        # are re-enforced idempotently each step by ``_enforce_fixed_tokens``).
        if continuation_resume is None:
            for fixed_idx, fixed_token_id in (fixed_tokens or {}).items():
                x_t[int(fixed_idx)] = int(fixed_token_id)
                unmask_step_by_pos[int(fixed_idx)] = start_step
            _enforce_fixed_tokens(
                x_t=x_t,
                fixed_tokens=fixed_tokens,
                unmask_step_by_pos=unmask_step_by_pos,
                step=start_step,
            )
        snapshot_step_set = {int(s) for s in (snapshot_steps or ())}
        snapshots: list[SamplerSnapshot] = []
        trajectory_rows: list[dict[str, Any]] = []
        n_steps = int(config.sampler.n_steps)
        dt = 1.0 / float(n_steps)
        last_logits: torch.Tensor | None = None
        remask_enabled = bool(config.sampler.remask.enabled)
        remask_fraction_scale = float(config.sampler.remask.fraction_scale)

        # V1F1 pre-terminal continuation state. ``editable_positions`` is the explicit
        # editable domain (residue domain minus permanent fixed anchors), never inferred from
        # the mask pattern or unmask history (PLAN §2.3). ``n_forward`` counts denoiser
        # forwards charged in THIS call, so a checkpoint taken at the TOP of step ``s`` (fresh
        # start) has paid exactly ``s`` lane-DFE.
        editable_positions = (
            tuple(i for i in range(sequence_length) if i not in fixed_positions)
            if continuation is not None
            else ()
        )
        continuation_cp: ContinuationCheckpoint | None = None
        prev_rho_edit: float | None = None
        max_rho_edit = 0.0
        n_forward = 0
        if continuation is not None and continuation.at_step is not None:
            at = int(continuation.at_step)
            if not (start_step <= at < n_steps):
                raise ValueError(
                    f"continuation.at_step {at} out of range [{start_step}, {n_steps})"
                )

        for step in range(start_step, n_steps):
            t = step / float(n_steps)
            # V1F1: pre-denoiser continuation checkpoint (post-previous-remask boundary).
            # Evaluated at the TOP of the step BEFORE this step's forward, so the captured
            # ``x_t`` is the exact state the denoiser is about to see and the paid prefix DFE
            # is ``n_forward``. Additive: when ``continuation is None`` this is byte-identical
            # to the legacy trajectory (no reads that mutate ``x_t`` / ``scores`` / ``rng``).
            if continuation is not None and continuation_cp is None:
                _maturity = editable_maturity(
                    x_t=x_t,
                    editable_positions=editable_positions,
                    fixed_positions=fixed_positions,
                    mask_token_id=self.mask_token_id,
                    aa_token_ids=aa_token_ids,
                )
                max_rho_edit = max(max_rho_edit, _maturity.rho_edit)
                if continuation.at_step is not None:
                    _triggered = step == int(continuation.at_step)
                else:
                    _triggered = _is_upward_crossing(
                        prev_rho_edit, _maturity.rho_edit, float(continuation.at_rho_edit)
                    )
                prev_rho_edit = _maturity.rho_edit
                if _triggered:
                    continuation_cp = ContinuationCheckpoint(
                        snapshot_phase=CONTINUATION_PHASE,
                        step=int(step),
                        t=float(t),
                        x_t=x_t.detach().clone(),
                        scores=scores.copy(),
                        unmask_step_by_pos=tuple(int(v) for v in unmask_step_by_pos),
                        rng_state=copy.deepcopy(rng.bit_generator.state),
                        fixed_tokens=tuple(
                            sorted((int(k), int(v)) for k, v in (fixed_tokens or {}).items())
                        ),
                        editable_positions=editable_positions,
                        paid_prefix_dfe=int(n_forward),
                        n_steps=int(n_steps),
                        maturity=_maturity,
                    )
                    if continuation.early_stop:
                        _validate_preterminal_root(_maturity)
                        return SamplerOutput(
                            tokens=x_t.detach().clone(),
                            unmask_step_by_pos=list(unmask_step_by_pos),
                            g_values=g_values.astype(np.float32, copy=False).tolist(),
                            trajectory_rows=trajectory_rows,
                            fixed_positions=tuple(sorted(fixed_positions)),
                            snapshots=tuple(snapshots),
                            continuation_checkpoint=continuation_cp,
                            stopped_early=True,
                            final_rng_state=copy.deepcopy(rng.bit_generator.state),
                            final_scores=tuple(float(s) for s in scores),
                            logical_dfe=int(n_forward),
                        )
            logits = denoiser(x_t.clone(), t, struct)
            n_forward += 1
            if logits.shape != (sequence_length, self.vocab_size):
                raise ValueError(
                    "denoiser must return logits with shape "
                    f"({sequence_length}, {self.vocab_size})"
                )
            if torch.isnan(logits).any():
                raise FloatingPointError(f"NaN logits encountered at step={step} t={t:.6f}")

            structural_logits = logits
            # Pre-controller snapshot (signal-diag P1): capture the RAW denoiser
            # logits + the partial completion the denoiser saw, BEFORE any
            # controller / D2 correction mutates them. Side-effect free.
            if step in snapshot_step_set:
                snapshots.append(
                    SamplerSnapshot(
                        step=int(step),
                        t=float(t),
                        x_t=x_t.detach().clone(),
                        struct_logits=structural_logits.detach().cpu().clone(),
                        scores=scores.copy(),
                        unmask_step_by_pos=tuple(int(v) for v in unmask_step_by_pos),
                    )
                )
            if controller is not None:
                # Local import to avoid pulling controller deps when unused.
                from .controller import PostSamplingContext, SamplerStepContext

                ctx = SamplerStepContext(
                    x_t=x_t.detach().clone(),
                    logits=logits,
                    scores=scores.copy(),
                    step=step,
                    t=t,
                    mask_token_id=self.mask_token_id,
                    protein_id=protein_id,
                    design_idx=design_idx,
                    sequence_length=sequence_length,
                )
                result = controller.step(ctx)
                logits = result.logits
            corrected_logits = logits

            last_logits = logits.detach().cpu()
            probs = positionwise_unmask_probabilities(
                t=t,
                dt=dt,
                g_values=g_values,
                base_form=config.schedule.base_form,
            )
            masked_positions = (x_t == self.mask_token_id).cpu().numpy()
            selected_positions = np.array([], dtype=np.int64)
            sampled_tokens_actual = np.array([], dtype=np.int64)
            sampled_tokens_uncorrected: np.ndarray | None = None
            if masked_positions.any():
                draws = rng.random(sequence_length) < probs
                selected_positions = np.flatnonzero(masked_positions & draws)
                if selected_positions.size:
                    # PLAN §"Sampler integration" 4: snapshot RNG state AFTER
                    # Bernoulli draws are fixed and BEFORE categorical sampling
                    # so the paired branch can replay token sampling without
                    # changing the set of selected positions.
                    saved_state = (
                        rng.bit_generator.state if controller is not None else None
                    )
                    selected_logits = last_logits[selected_positions] / float(
                        config.sampler.temperature
                    )
                    sampled_tokens, sampled_logp = _sample_categorical(
                        selected_logits, rng
                    )
                    x_t[selected_positions] = sampled_tokens
                    scores[selected_positions] = sampled_logp
                    for pos in selected_positions.tolist():
                        if unmask_step_by_pos[pos] < 0:
                            unmask_step_by_pos[pos] = step
                    sampled_tokens_actual = sampled_tokens.numpy().astype(
                        np.int64, copy=False
                    )
                    _enforce_fixed_tokens(
                        x_t=x_t,
                        fixed_tokens=fixed_tokens,
                        unmask_step_by_pos=unmask_step_by_pos,
                        step=step,
                    )
                    # PLAN §"Sampler integration" 6-7: paired uncorrected sample
                    # via an isolated RNG clone so the real RNG is unaffected.
                    # Skipped when corrected == structural (no information gain)
                    # or when D2 paired_uncorrected_sample is disabled.
                    if (
                        controller is not None
                        and saved_state is not None
                        and getattr(controller, "config", None) is not None
                        and getattr(controller.config, "d2", None) is not None
                        and bool(controller.config.d2.enabled)
                        and bool(controller.config.d2.paired_uncorrected_sample)
                        and corrected_logits is not structural_logits
                    ):
                        paired_rng = np.random.default_rng()
                        paired_rng.bit_generator.state = saved_state
                        structural_selected_logits = (
                            structural_logits.detach().cpu()[selected_positions]
                            / float(config.sampler.temperature)
                        )
                        sampled_uncorrected, _ = _sample_categorical(
                            structural_selected_logits, paired_rng
                        )
                        sampled_tokens_uncorrected = (
                            sampled_uncorrected.numpy().astype(np.int64, copy=False)
                        )

            # Post-sampling hook (PLAN §"Sampler integration" 9-10). Skipped
            # when no controller is bound or when remask is disabled, which
            # keeps controller=None bit-equivalent with pre-D2/D3 behavior.
            remask_count = 0
            post_rank_scores: np.ndarray | None = None
            post_protected: tuple[int, ...] = ()
            if controller is not None and remask_enabled and step < n_steps - 1:
                post_ctx = PostSamplingContext(
                    x_t=x_t.detach().clone(),
                    scores=scores.copy(),
                    structural_logits=structural_logits,
                    corrected_logits=corrected_logits,
                    selected_positions=selected_positions,
                    sampled_tokens_actual=sampled_tokens_actual,
                    sampled_tokens_uncorrected=sampled_tokens_uncorrected,
                    step=step,
                    t=t,
                    n_steps=n_steps,
                    mask_token_id=self.mask_token_id,
                    protein_id=protein_id,
                    design_idx=design_idx,
                    sequence_length=sequence_length,
                )
                post_result = controller.post_step(post_ctx)
                post_rank_scores = post_result.rank_scores
                post_protected = post_result.protected_positions
            if remask_enabled and step < n_steps - 1:
                remask_result = _apply_reparam_remask(
                    x_t=x_t,
                    scores=scores,
                    unmask_step_by_pos=unmask_step_by_pos,
                    mask_token_id=self.mask_token_id,
                    step=step,
                    n_steps=n_steps,
                    rank_scores=post_rank_scores,
                    protected_positions=_union_protected(post_protected, fixed_positions),
                    cutoff_scale=remask_fraction_scale,
                )
                remask_count = remask_result.count
                _enforce_fixed_tokens(
                    x_t=x_t,
                    fixed_tokens=fixed_tokens,
                    unmask_step_by_pos=unmask_step_by_pos,
                    step=step,
                )
                # D3 remask telemetry hook (PLAN §D3-14). Skipped when the
                # controller does not expose post_remask (keeps duck-typed
                # stub controllers in unit tests bit-equivalent).
                if controller is not None and remask_result.remasked_positions:
                    post_remask_fn = getattr(controller, "post_remask", None)
                    if callable(post_remask_fn):
                        post_remask_fn(
                            remasked_positions=remask_result.remasked_positions,
                            step=step,
                            t=t,
                        )

            if save_trajectories:
                trajectory_rows.append(
                    {
                        "step": step,
                        "t": t,
                        "unmasked_mask": (x_t != self.mask_token_id).tolist(),
                        "token_argmax": last_logits.argmax(dim=-1).tolist(),
                        "remasked_count": int(remask_count),
                    }
                )

        # V1F1: a requested pre-terminal maturity that never crossed is explicit, never
        # silently completed into a terminal root (PLAN §2.5) -- fail-closed for BOTH early-stop
        # and observe modes. A diagnostic that tolerates a missing crossing must ask for it via
        # an explicit separate semantic, not by defaulting early_stop=False.
        if (
            continuation is not None
            and continuation_cp is None
            and continuation.at_rho_edit is not None
        ):
            raise MaturityNotReachedError(
                f"continuation at_rho_edit={continuation.at_rho_edit} never crossed; "
                f"max pre-denoiser rho_edit reached was {max_rho_edit:.6f}"
            )

        residual = torch.nonzero(x_t == self.mask_token_id, as_tuple=False).flatten()
        if residual.numel():
            if last_logits is None:
                raise RuntimeError("sampler ended without any denoiser logits")
            sampled_tokens, sampled_logp = _sample_categorical(
                last_logits[residual] / float(config.sampler.temperature),
                rng,
            )
            x_t[residual] = sampled_tokens
            residual_idx = residual.tolist()
            scores[residual_idx] = sampled_logp
            for pos in residual_idx:
                if unmask_step_by_pos[pos] < 0:
                    unmask_step_by_pos[pos] = n_steps
        _enforce_fixed_tokens(
            x_t=x_t,
            fixed_tokens=fixed_tokens,
            unmask_step_by_pos=unmask_step_by_pos,
            step=n_steps,
        )

        if (x_t == self.mask_token_id).any():
            raise RuntimeError("sampler finished with mask tokens still present")

        return SamplerOutput(
            tokens=x_t,
            unmask_step_by_pos=unmask_step_by_pos,
            g_values=g_values.astype(np.float32, copy=False).tolist(),
            trajectory_rows=trajectory_rows,
            fixed_positions=tuple(sorted(fixed_positions)),
            snapshots=tuple(snapshots),
            continuation_checkpoint=continuation_cp,
            stopped_early=False,
            final_rng_state=copy.deepcopy(rng.bit_generator.state),
            final_scores=tuple(float(s) for s in scores),
            logical_dfe=int(n_forward),
        )

    def sample_batch(
        self,
        *,
        lanes: list[SamplerBatchLane],
        batched_denoiser: Callable[
            [list[torch.Tensor], float, list[Any]], list[torch.Tensor]
        ],
        save_trajectories: bool = False,
    ) -> list[SamplerOutput]:
        """Sample independent RF lanes with one batched denoiser call per step.

        Controller, RNG, sampling, remask, and telemetry state remain per-lane.
        The only shared operation is the structural denoiser forward.
        """

        if not lanes:
            return []
        n_steps = int(lanes[0].config.sampler.n_steps)
        if n_steps <= 0:
            raise ValueError("sampler.n_steps must be positive")
        for idx, lane in enumerate(lanes):
            if int(lane.config.sampler.n_steps) != n_steps:
                raise ValueError(
                    "sample_batch requires all lanes to use the same n_steps; "
                    f"lane 0 has {n_steps}, lane {idx} has "
                    f"{int(lane.config.sampler.n_steps)}"
                )

        states = [self._init_batch_lane_state(lane) for lane in lanes]
        dt = 1.0 / float(n_steps)
        # V1F2 resumed batch: all lanes must share the resume start step (the driver groups
        # lanes by start step before batching). Fresh lanes all start at 0.
        start_step = states[0].start_step
        if any(state.start_step != start_step for state in states):
            raise ValueError(
                "sample_batch requires all lanes to share the same resume start_step; "
                "group lanes by start step before batching"
            )

        for step in range(start_step, n_steps):
            t = step / float(n_steps)
            structural_logits_by_lane = batched_denoiser(
                [state.x_t.clone() for state in states],
                t,
                [state.lane.struct for state in states],
            )
            if len(structural_logits_by_lane) != len(states):
                raise ValueError(
                    f"batched_denoiser returned {len(structural_logits_by_lane)} "
                    f"logit tensors for {len(states)} lanes"
                )
            for state, structural_logits in zip(states, structural_logits_by_lane):
                self._sample_batch_lane_step(
                    state=state,
                    structural_logits=structural_logits,
                    step=step,
                    t=t,
                    dt=dt,
                    n_steps=n_steps,
                    save_trajectories=save_trajectories,
                )

        return [self._finalize_batch_lane_state(state, n_steps=n_steps) for state in states]

    def _init_batch_lane_state(self, lane: SamplerBatchLane) -> _SamplerLaneState:
        sequence_length = int(lane.sequence_length)
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        h = np.asarray(lane.h_values, dtype=np.float32)
        if h.shape != (sequence_length,):
            raise ValueError(
                f"h_values shape {h.shape} does not match sequence_length={sequence_length}"
            )
        if lane.config.h_shuffle.enabled:
            effective_shuffle_seed = (
                int(lane.config.h_shuffle.seed)
                if lane.shuffle_seed is None
                else int(lane.shuffle_seed)
            )
            h = shuffle_h_values(h, seed=effective_shuffle_seed)
        g_values = amplification_factor(h, lane.config.amplification)
        resume = lane.resume
        if resume is not None:
            # Self-contained resumed lane: validate n_steps + tamper hash; anchors come from the
            # resume (not lane.fixed_tokens), which must be absent.
            if lane.fixed_tokens is not None:
                raise ValueError(
                    "lane.resume is self-contained; do not also set lane.fixed_tokens"
                )
            if int(resume.n_steps) != int(lane.config.sampler.n_steps):
                raise ValueError(
                    f"lane.resume.n_steps {resume.n_steps} != config n_steps "
                    f"{int(lane.config.sampler.n_steps)}"
                )
            if _replay_state_hash(
                resume.x_t, resume.scores, resume.start_step, resume.n_steps,
                resume.fixed_tokens, resume.editable_positions, resume.unmask_step_by_pos,
                resume.mode, resume.rng_state, resume.fork_seed,
            ) != resume.state_hash:
                raise ValueError(
                    "lane.resume state_hash mismatch (tampered or inconsistent state)"
                )
            source_fixed: Mapping[int, int] | None = {
                int(p): int(tok) for p, tok in resume.fixed_tokens
            }
        else:
            source_fixed = lane.fixed_tokens
        fixed_tokens = _normalize_and_validate_fixed_tokens(
            source_fixed, sequence_length, self.mask_token_id, None
        )
        fixed_positions = _resolve_fixed_positions(fixed_tokens, sequence_length)
        if resume is not None:
            # restore root bytes; anchors carried in x_t (not re-seeded).
            x_t = resume.x_t.detach().clone().to(torch.long)
            if x_t.shape != (sequence_length,):
                raise ValueError(
                    f"lane.resume.x_t shape {tuple(x_t.shape)} != ({sequence_length},)"
                )
            unmask_step_by_pos = [int(v) for v in resume.unmask_step_by_pos]
            scores = np.asarray(resume.scores, dtype=np.float64).copy()
            if len(unmask_step_by_pos) != sequence_length or scores.shape != (
                sequence_length,
            ):
                raise ValueError("lane.resume arrays must match sequence_length")
            start_step = int(resume.start_step)
            if resume.mode == "identity":
                rng = np.random.default_rng()
                rng.bit_generator.state = copy.deepcopy(resume.rng_state)
            else:  # fork
                rng = np.random.default_rng(int(resume.fork_seed))
        else:
            x_t = torch.full((sequence_length,), self.mask_token_id, dtype=torch.long)
            unmask_step_by_pos = [-1] * sequence_length
            scores = np.full(sequence_length, -np.inf, dtype=np.float64)
            start_step = 0
            rng = np.random.default_rng(int(lane.config.sampler.seed))
            for fixed_idx, fixed_token_id in (fixed_tokens or {}).items():
                x_t[int(fixed_idx)] = int(fixed_token_id)
                unmask_step_by_pos[int(fixed_idx)] = 0
            _enforce_fixed_tokens(
                x_t=x_t,
                fixed_tokens=fixed_tokens,
                unmask_step_by_pos=unmask_step_by_pos,
                step=0,
            )
        return _SamplerLaneState(
            lane=lane,
            g_values=g_values,
            rng=rng,
            x_t=x_t,
            unmask_step_by_pos=unmask_step_by_pos,
            scores=scores,
            trajectory_rows=[],
            last_logits=None,
            remask_enabled=bool(lane.config.sampler.remask.enabled),
            remask_fraction_scale=float(lane.config.sampler.remask.fraction_scale),
            fixed_positions=fixed_positions,
            start_step=start_step,
            fixed_tokens=fixed_tokens,
        )

    def _sample_batch_lane_step(
        self,
        *,
        state: _SamplerLaneState,
        structural_logits: torch.Tensor,
        step: int,
        t: float,
        dt: float,
        n_steps: int,
        save_trajectories: bool,
    ) -> None:
        sequence_length = int(state.lane.sequence_length)
        if structural_logits.shape != (sequence_length, self.vocab_size):
            raise ValueError(
                "batched_denoiser must return per-lane logits with shape "
                f"({sequence_length}, {self.vocab_size}); got "
                f"{tuple(structural_logits.shape)}"
            )
        if torch.isnan(structural_logits).any():
            raise FloatingPointError(f"NaN logits encountered at step={step} t={t:.6f}")

        logits = structural_logits
        controller = state.lane.controller
        if controller is not None:
            from .controller import SamplerStepContext

            ctx = SamplerStepContext(
                x_t=state.x_t.detach().clone(),
                logits=logits,
                scores=state.scores.copy(),
                step=step,
                t=t,
                mask_token_id=self.mask_token_id,
                protein_id=state.lane.protein_id,
                design_idx=state.lane.design_idx,
                sequence_length=sequence_length,
            )
            result = controller.step(ctx)
            logits = result.logits
        corrected_logits = logits

        state.last_logits = logits.detach().cpu()
        probs = positionwise_unmask_probabilities(
            t=t,
            dt=dt,
            g_values=state.g_values,
            base_form=state.lane.config.schedule.base_form,
        )
        masked_positions = (state.x_t == self.mask_token_id).cpu().numpy()
        selected_positions = np.array([], dtype=np.int64)
        sampled_tokens_actual = np.array([], dtype=np.int64)
        sampled_tokens_uncorrected: np.ndarray | None = None
        if masked_positions.any():
            draws = state.rng.random(sequence_length) < probs
            selected_positions = np.flatnonzero(masked_positions & draws)
            if selected_positions.size:
                saved_state = (
                    state.rng.bit_generator.state if controller is not None else None
                )
                selected_logits = state.last_logits[selected_positions] / float(
                    state.lane.config.sampler.temperature
                )
                sampled_tokens, sampled_logp = _sample_categorical(
                    selected_logits, state.rng
                )
                state.x_t[selected_positions] = sampled_tokens
                state.scores[selected_positions] = sampled_logp
                for pos in selected_positions.tolist():
                    if state.unmask_step_by_pos[pos] < 0:
                        state.unmask_step_by_pos[pos] = step
                sampled_tokens_actual = sampled_tokens.numpy().astype(np.int64, copy=False)
                _enforce_fixed_tokens(
                    x_t=state.x_t,
                    fixed_tokens=state.fixed_tokens,
                    unmask_step_by_pos=state.unmask_step_by_pos,
                    step=step,
                )
                if (
                    controller is not None
                    and saved_state is not None
                    and getattr(controller, "config", None) is not None
                    and getattr(controller.config, "d2", None) is not None
                    and bool(controller.config.d2.enabled)
                    and bool(controller.config.d2.paired_uncorrected_sample)
                    and corrected_logits is not structural_logits
                ):
                    paired_rng = np.random.default_rng()
                    paired_rng.bit_generator.state = saved_state
                    structural_selected_logits = (
                        structural_logits.detach().cpu()[selected_positions]
                        / float(state.lane.config.sampler.temperature)
                    )
                    sampled_uncorrected, _ = _sample_categorical(
                        structural_selected_logits, paired_rng
                    )
                    sampled_tokens_uncorrected = sampled_uncorrected.numpy().astype(
                        np.int64, copy=False
                    )

        remask_count = 0
        post_rank_scores: np.ndarray | None = None
        post_protected: tuple[int, ...] = ()
        if controller is not None and state.remask_enabled and step < n_steps - 1:
            from .controller import PostSamplingContext

            post_ctx = PostSamplingContext(
                x_t=state.x_t.detach().clone(),
                scores=state.scores.copy(),
                structural_logits=structural_logits,
                corrected_logits=corrected_logits,
                selected_positions=selected_positions,
                sampled_tokens_actual=sampled_tokens_actual,
                sampled_tokens_uncorrected=sampled_tokens_uncorrected,
                step=step,
                t=t,
                n_steps=n_steps,
                mask_token_id=self.mask_token_id,
                protein_id=state.lane.protein_id,
                design_idx=state.lane.design_idx,
                sequence_length=sequence_length,
            )
            post_result = controller.post_step(post_ctx)
            post_rank_scores = post_result.rank_scores
            post_protected = post_result.protected_positions
        if state.remask_enabled and step < n_steps - 1:
            remask_result = _apply_reparam_remask(
                x_t=state.x_t,
                scores=state.scores,
                unmask_step_by_pos=state.unmask_step_by_pos,
                mask_token_id=self.mask_token_id,
                step=step,
                n_steps=n_steps,
                rank_scores=post_rank_scores,
                protected_positions=_union_protected(
                    post_protected, state.fixed_positions
                ),
                cutoff_scale=state.remask_fraction_scale,
            )
            remask_count = remask_result.count
            _enforce_fixed_tokens(
                x_t=state.x_t,
                fixed_tokens=state.fixed_tokens,
                unmask_step_by_pos=state.unmask_step_by_pos,
                step=step,
            )
            if controller is not None and remask_result.remasked_positions:
                post_remask_fn = getattr(controller, "post_remask", None)
                if callable(post_remask_fn):
                    post_remask_fn(
                        remasked_positions=remask_result.remasked_positions,
                        step=step,
                        t=t,
                    )

        if save_trajectories:
            state.trajectory_rows.append(
                {
                    "step": step,
                    "t": t,
                    "unmasked_mask": (state.x_t != self.mask_token_id).tolist(),
                    "token_argmax": state.last_logits.argmax(dim=-1).tolist(),
                    "remasked_count": int(remask_count),
                }
            )

    def _finalize_batch_lane_state(
        self,
        state: _SamplerLaneState,
        *,
        n_steps: int,
    ) -> SamplerOutput:
        residual = torch.nonzero(
            state.x_t == self.mask_token_id, as_tuple=False
        ).flatten()
        if residual.numel():
            if state.last_logits is None:
                raise RuntimeError("sampler ended without any denoiser logits")
            sampled_tokens, sampled_logp = _sample_categorical(
                state.last_logits[residual]
                / float(state.lane.config.sampler.temperature),
                state.rng,
            )
            state.x_t[residual] = sampled_tokens
            residual_idx = residual.tolist()
            state.scores[residual_idx] = sampled_logp
            for pos in residual_idx:
                if state.unmask_step_by_pos[pos] < 0:
                    state.unmask_step_by_pos[pos] = n_steps
        _enforce_fixed_tokens(
            x_t=state.x_t,
            fixed_tokens=state.fixed_tokens,
            unmask_step_by_pos=state.unmask_step_by_pos,
            step=n_steps,
        )

        if (state.x_t == self.mask_token_id).any():
            raise RuntimeError("sampler finished with mask tokens still present")

        return SamplerOutput(
            tokens=state.x_t,
            unmask_step_by_pos=state.unmask_step_by_pos,
            g_values=state.g_values.astype(np.float32, copy=False).tolist(),
            trajectory_rows=state.trajectory_rows,
            fixed_positions=tuple(sorted(state.fixed_positions)),
            final_rng_state=copy.deepcopy(state.rng.bit_generator.state),
            # Both fields default to a FALSY value, so omitting them yields an output that looks
            # complete and costs nothing. Every lane in one batch runs the same steps from the same
            # shared start (enforced in `sample_batch`), so this lane's logical lane-DFE is exactly
            # that span. A caller deriving physical cost from it -- which the V1 oracles do -- would
            # otherwise book real GPU time as zero work (PLAN §2.3:382, §5.2:954).
            final_scores=tuple(float(s) for s in state.scores),
            logical_dfe=int(n_steps - state.start_step),
        )


def _sample_categorical(
    logits: torch.Tensor, rng: np.random.Generator
) -> tuple[torch.Tensor, np.ndarray]:
    """Categorical sample plus per-row log-prob of the chosen token."""
    log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
    probs = np.exp(log_probs)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    sampled = np.array(
        [int(rng.choice(probs.shape[1], p=row)) for row in probs],
        dtype=np.int64,
    )
    chosen_logp = log_probs[np.arange(probs.shape[0]), sampled]
    return torch.from_numpy(sampled).to(torch.long), chosen_logp.astype(np.float64, copy=False)


@dataclass
class ReparamRemaskResult:
    """Number of positions remasked plus the actual position list for telemetry."""

    count: int
    remasked_positions: tuple[int, ...]


def _apply_reparam_remask(
    *,
    x_t: torch.Tensor,
    scores: np.ndarray,
    unmask_step_by_pos: list[int],
    mask_token_id: int,
    step: int,
    n_steps: int,
    rank_scores: np.ndarray | None = None,
    protected_positions: tuple[int, ...] = (),
    cutoff_scale: float = 1.0,
) -> ReparamRemaskResult:
    """Re-mask the lowest-confidence committed positions.

    Mirrors DPLM's ``reparam-uncond-deterministic-linear`` rule: at step ``s``
    of ``T`` (1-indexed for the rate), keep ``s/T`` of the committed positions
    and re-mask the bottom ``1 - s/T`` by score.

    ``rank_scores`` overrides ``scores[]`` as the per-residue ranking signal
    (D3 commit-score path). When ``None`` the function uses ``scores[]``
    exactly as before. ``protected_positions`` removes positions from the
    remask candidate pool (D3 grace / final freeze). ``cutoff_scale``
    (``sampler.remask.fraction_scale``) multiplies the cutoff length: ``1.0``
    is a no-op, ``0.0`` re-masks nothing. With all defaults the output is
    byte-equivalent to the pre-D2/D3 implementation.

    Returns the number of positions re-masked this step.
    """
    committed_mask = (x_t != mask_token_id).cpu().numpy()
    n_committed = int(committed_mask.sum())
    if n_committed == 0:
        return ReparamRemaskResult(count=0, remasked_positions=())
    rate = 1.0 - (step + 1) / float(n_steps)
    cutoff_len = int(n_committed * rate * cutoff_scale)
    if cutoff_len <= 0:
        return ReparamRemaskResult(count=0, remasked_positions=())
    committed_positions = np.flatnonzero(committed_mask)
    if protected_positions:
        protected_set = {int(p) for p in protected_positions}
        committed_positions = np.array(
            [p for p in committed_positions if int(p) not in protected_set],
            dtype=committed_positions.dtype,
        )
    if committed_positions.size == 0:
        return ReparamRemaskResult(count=0, remasked_positions=())
    ranking = scores if rank_scores is None else rank_scores
    ranking_at_committed = ranking[committed_positions]
    # Lowest cutoff_len scores → re-mask. ``argpartition`` for O(n).
    if cutoff_len >= committed_positions.size:
        bottom_positions = committed_positions
    else:
        partition_idx = np.argpartition(ranking_at_committed, cutoff_len)[:cutoff_len]
        bottom_positions = committed_positions[partition_idx]
    if bottom_positions.size == 0:
        return ReparamRemaskResult(count=0, remasked_positions=())
    x_t[bottom_positions] = mask_token_id
    scores[bottom_positions] = -np.inf
    for pos in bottom_positions.tolist():
        unmask_step_by_pos[pos] = -1
    return ReparamRemaskResult(
        count=int(bottom_positions.size),
        remasked_positions=tuple(int(p) for p in bottom_positions.tolist()),
    )
