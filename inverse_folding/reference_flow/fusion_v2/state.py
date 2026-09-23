"""The typed V2 state layers of PLAN §2.2, as frozen torch-free records.

The job of this module is to make the illegal cross-layer states of PLAN §2.4 unrepresentable
rather than merely discouraged. Two separations carry most of the weight:

**Evidence namespaces.** ``OriginEvidence`` is immutable provenance -- the score an identity carried
in its *own originating context* (an endpoint completion score, or a source's pre-intervention
active score). ``active_sampler_score_by_pos`` is the active-context ranking signal the current
re-entry trajectory is legally allowed to consume. They are different fields of different types, so
an endpoint score cannot become an active remask score by assignment (PLAN §2.4, safety law 14).

**Layers.** A ``LivePartialState`` and a ``ProjectedPartialState`` differ structurally, not by a
label: only the projected layer admits ``PENDING_ASSIMILATION`` and an active protection set, and
only the live layer carries a real replay stream.

Length invariant throughout: ``L == len(tokens)``. Every ``*_by_pos`` field is a tuple of exactly
``L`` entries indexed by **absolute residue position**, matching the sampler's ``x_t`` / ``scores``
/ ``unmask_step_by_pos``. It is never indexed over ``editable_positions``.

Purity: no torch, no numpy, no I/O. Imports ``errors``, ``identity``, ``schedule``, and frozen v0
primitives only.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import V2Error
from .identity import (
    SafetyReferenceBinding,
    HeadScoreBinding,
    HeadScoreLike,
    ProjectionPolicyIdentity,
    V2_STATE_SCHEMA_VERSION,
    V2ConditioningIdentity,
    LineageRef,
    assert_id_binds_digest,
    canonical_digest,
    make_live_state_id,
    make_archive_entry_id,
    make_endpoint_id,
    make_projected_state_id,
    make_transition_id,
    require_digest,
    require_id_namespace,
    window_grid_digest,
)
from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion.v1_alloc import validate_complete_aa20
from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome

from .schedule import CoordinateLaw, HistoryKey, MaturityRecord, observe_maturity

__all__ = [
    "V2StateError", "ActiveOriginKind", "FeedbackOriginRef", "ActiveScoreStatus",
    "OriginEvidence", "PositionProvenance", "ReplayIdentity", "LivePartialState",
    "SupportAction", "ProtectionGrantReason", "TemporaryProtection", "SupportPartition",
    "ProjectedPartialState", "carry_is_temporally_legal", "validate_endpoint_writeback",
    "FeasibilityLevel", "TransitionOutcome", "validate_propagated_capture",
    "EndpointPositionEvidence", "CompleteEndpoint", "endpoint_may_become_ancestry",
    "ArchiveMembership", "ArchiveEntry", "advance_archive_entry", "AssimilationRecord",
    "FeedbackTransition",
]


class V2StateError(V2Error):
    """A typed state record violated one of the PLAN §2.4 construction invariants."""


class ActiveOriginKind(str, enum.Enum):
    """How the position's *current* identity came to be (FUSION_V2 §3.1, verbatim)."""

    UNRESOLVED = "unresolved"
    HARD_ANCHOR = "hard_anchor"
    DENOISER_SAMPLE = "denoiser_sample"
    FEEDBACK_INJECTION = "feedback_injection"


class FeedbackOriginRef(str, enum.Enum):
    """Which *class* of feedback wrote the position (FUSION_V2 §3.1, verbatim).

    The concrete endpoint or source-state id is reached through ``origin_transition_id_by_pos`` and
    the transition row it names, so a length-``L`` vector never stores a full id string ``L`` times.
    """

    NONE = "none"
    SOURCE_STATE = "source_state"
    SELECTED_ENDPOINT = "selected_endpoint"


class ActiveScoreStatus(str, enum.Enum):
    """Whether the active score at this position may drive remask or ranking (PLAN §2.4)."""

    MASKED = "masked"
    HISTORICAL_NATURAL = "historical_natural"
    PENDING_ASSIMILATION = "pending_assimilation"
    ASSIMILATED = "assimilated"
    NOT_RANKED_ANCHOR = "not_ranked_anchor"


#: Statuses that carry no active score. Anything else must carry a finite float.
_SCORELESS_STATUSES = frozenset({
    ActiveScoreStatus.MASKED,
    ActiveScoreStatus.PENDING_ASSIMILATION,
    ActiveScoreStatus.NOT_RANKED_ANCHOR,
})

#: Which origin kinds may legally name a feedback class. A naturally sampled or unresolved position
#: claiming a feedback origin is an inconsistent cross-product, not two independent labels.
_FEEDBACK_ORIGIN_KINDS = frozenset({ActiveOriginKind.FEEDBACK_INJECTION})


def _require_index(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise V2StateError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise V2StateError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_text(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value:
        raise V2StateError(f"{name} must be a non-empty str, got {value!r}")
    return value


def _require_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V2StateError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        raise V2StateError(f"{name} must be finite, got {out!r}")
    return out


def _require_md5(value: Any, name: str) -> str:
    text = _require_text(value, name)
    if len(text) != 32 or any(c not in "0123456789abcdef" for c in text):
        raise V2StateError(f"{name} must be a 32-char lowercase md5 hex digest, got {text!r}")
    return text


def _canonical_json_value(value: Any, name: str) -> Any:
    """Return the same JSON-normalized representation used by sampler replay hashing."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError) as exc:
        raise V2StateError(f"{name} is not canonically serializable: {exc}") from exc


def _canonical_index_set(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    normalized = tuple(sorted(_require_index(value, name) for value in values))
    if len(set(normalized)) != len(normalized):
        raise V2StateError(f"{name} contains duplicates")
    return normalized


def _canonical_anchor_set(
    values: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    for entry in values:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise V2StateError("hard_anchors entries must be (position, token) pairs")
        normalized.append(
            (_require_index(entry[0], "anchor position"),
             _require_index(entry[1], "anchor token"))
        )
    normalized.sort()
    if len({position for position, _ in normalized}) != len(normalized):
        raise V2StateError("hard_anchors contains duplicate positions")
    return tuple(normalized)


def _lineage_payload(lineage: LineageRef) -> dict[str, Any]:
    return {
        "protein_id": lineage.protein_id,
        "root_id": lineage.root_id,
        "family_id": lineage.family_id,
        "depth": lineage.depth,
        "parent_state_id": lineage.parent_state_id,
        "parent_transition_id": lineage.parent_transition_id,
        "origin_endpoint_id": lineage.origin_endpoint_id,
    }


@dataclass(frozen=True)
class OriginEvidence:
    """One immutable provenance atom: where an identity came from, and what it scored *there*.

    ``evidence_logprob`` is deliberately named to be unassignable to an active score. It records the
    endpoint's own completion log-probability, or the source's pre-intervention active score --
    neither of which is a statement about the projected state's re-entry context.
    """

    origin_kind: ActiveOriginKind
    origin_ref: FeedbackOriginRef
    commit: HistoryKey
    token: int
    evidence_logprob: float | None
    transition_id: str | None
    evidence_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.origin_kind, ActiveOriginKind):
            raise V2StateError("origin_kind must be an ActiveOriginKind")
        if not isinstance(self.origin_ref, FeedbackOriginRef):
            raise V2StateError("origin_ref must be a FeedbackOriginRef")
        if not isinstance(self.commit, HistoryKey):
            raise V2StateError("commit must be a HistoryKey; (depth, step) is the only legal key")
        _require_index(self.token, "token")
        if self.evidence_logprob is not None:
            _require_finite(self.evidence_logprob, "evidence_logprob")
        if self.transition_id is not None:
            require_id_namespace(self.transition_id, "txn")
        require_digest(self.evidence_digest, "evidence_digest")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "origin_kind": self.origin_kind.value,
            "origin_ref": self.origin_ref.value,
            "commit": [self.commit.depth, self.commit.step],
            "token": self.token,
            "evidence_logprob": self.evidence_logprob,
            "transition_id": self.transition_id,
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True)
class PositionProvenance:
    """``(first_origin, last_origin)`` plus an event count -- not a chain.

    Cost is exactly two evidence records per position, constant in depth. The full chain stays
    reconstructible by walking the transition table, which PLAN §5.3 already requires to persist
    every event's support sets and origin assignments; a per-position chain would cost ``O(L·D)``
    per state and duplicate that table.

    ``first_origin`` is what preserves original ancestry when an endpoint-feedback token is
    recursively carried into the next source state (PLAN §2.4), and ``n_origin_events`` makes an
    elided middle visible rather than implying ``first -> last`` was a single hop.
    """

    first_origin: OriginEvidence
    last_origin: OriginEvidence
    n_origin_events: int

    def __post_init__(self) -> None:
        for name in ("first_origin", "last_origin"):
            if not isinstance(getattr(self, name), OriginEvidence):
                raise V2StateError(f"{name} must be an OriginEvidence")
        _require_index(self.n_origin_events, "n_origin_events", minimum=1)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "first": self.first_origin.canonical_payload(),
            "last": self.last_origin.canonical_payload(),
            "n_origin_events": self.n_origin_events,
        }


@dataclass(frozen=True)
class ReplayIdentity:
    """Torch-free mirror of ``ContinuationResume``'s two-mode replay contract.

    ``replay_state_hash`` is produced by the sampler layer and carried here. Identity replay also
    binds the normalized raw RNG payload into every owning state digest, so changing RNG bytes while
    retaining a stale claimed hash cannot preserve state identity.
    """

    mode: str
    rng_state: Mapping[str, Any] | None
    fork_seed: int | None
    replay_state_hash: str

    def __post_init__(self) -> None:
        if self.mode not in ("identity", "fork"):
            raise V2StateError(f"replay mode must be 'identity' or 'fork', got {self.mode!r}")
        if self.mode == "identity":
            if self.rng_state is None or self.fork_seed is not None:
                raise V2StateError(
                    "identity replay binds an rng_state and no fork_seed, so a fork seed can never "
                    "impersonate an identity stream"
                )
            if not isinstance(self.rng_state, Mapping):
                raise V2StateError("identity replay rng_state must be a mapping")
            object.__setattr__(
                self, "rng_state", _canonical_json_value(dict(self.rng_state), "rng_state")
            )
        else:
            if self.fork_seed is None or self.rng_state is not None:
                raise V2StateError("fork replay binds a fork_seed and no rng_state")
            _require_index(self.fork_seed, "fork_seed")
        require_digest(self.replay_state_hash, "replay_state_hash")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "rng_state": self.rng_state,
            "fork_seed": self.fork_seed,
            "replay_state_hash": self.replay_state_hash,
        }


def _validate_position_algebra(
    *,
    tokens: tuple[int, ...],
    mask_token_id: int,
    aa_token_ids: frozenset[int],
    hard_anchors: tuple[tuple[int, int], ...],
    editable_positions: tuple[int, ...],
    origin_kind: tuple[ActiveOriginKind, ...],
    origin_ref: tuple[FeedbackOriginRef, ...],
    commit: tuple[HistoryKey | None, ...],
    transition_ids: tuple[str | None, ...],
    scores: tuple[float | None, ...],
    statuses: tuple[ActiveScoreStatus, ...],
    provenance: tuple[PositionProvenance, ...],
) -> tuple[frozenset[int], frozenset[int]]:
    """Shared per-position validation for every layer that carries the vectors.

    Returns the anchor and editable position sets so the caller does not re-derive them.
    """
    length = len(tokens)
    if length < 1:
        raise V2StateError("a state must carry at least one position")
    if not isinstance(aa_token_ids, frozenset) or len(aa_token_ids) != 20:
        raise V2StateError(
            f"aa_token_ids must be the canonical AA20 id set, got {len(aa_token_ids)} entries"
        )
    for token_id in aa_token_ids:
        _require_index(token_id, "AA20 token id")
    if mask_token_id in aa_token_ids:
        raise V2StateError("mask_token_id may not be a member of the canonical AA20 id set")
    vectors = {
        "active_origin_kind_by_pos": origin_kind,
        "feedback_origin_ref_by_pos": origin_ref,
        "active_commit_depth_step_by_pos": commit,
        "origin_transition_id_by_pos": transition_ids,
        "active_sampler_score_by_pos": scores,
        "active_score_status_by_pos": statuses,
        "provenance_by_pos": provenance,
    }
    for name, vector in vectors.items():
        if len(vector) != length:
            raise V2StateError(
                f"{name} has {len(vector)} entries but the state carries {length} positions; every "
                "*_by_pos vector is indexed by absolute residue position"
            )

    anchor_map = {}
    for entry in hard_anchors:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise V2StateError("hard_anchors entries must be (position, token) pairs")
        position, token = entry
        _require_index(position, "anchor position")
        _require_index(token, "anchor token")
        if position >= length:
            raise V2StateError(f"anchor position {position} is outside the {length}-residue state")
        if position in anchor_map:
            raise V2StateError(f"duplicate hard anchor at position {position}")
        if tokens[position] != token:
            raise V2StateError(
                f"hard anchor at position {position} declares token {token} but the state carries "
                f"{tokens[position]}; anchors are immutable through every transition"
            )
        if token not in aa_token_ids:
            raise V2StateError(
                f"hard anchor at position {position} carries token {token}, which is not a "
                "canonical AA20 token"
            )
        anchor_map[position] = token
    anchors = frozenset(anchor_map)

    editable = frozenset(editable_positions)
    if len(editable) != len(editable_positions):
        raise V2StateError("editable_positions contains duplicates")
    for position in editable_positions:
        _require_index(position, "editable position")
        if position >= length:
            raise V2StateError(f"editable position {position} is outside the state")
    if anchors & editable:
        raise V2StateError(
            f"positions {sorted(anchors & editable)} are both anchored and editable; the two "
            "classes partition the residue domain"
        )
    if anchors | editable != frozenset(range(length)):
        missing = sorted(frozenset(range(length)) - (anchors | editable))
        raise V2StateError(
            f"positions {missing} belong to neither the anchor nor the editable class; "
            "fixed and editable must partition [0, L)"
        )

    for index in range(length):
        token, kind, ref = tokens[index], origin_kind[index], origin_ref[index]
        status, score, commit_key = statuses[index], scores[index], commit[index]
        _require_index(token, f"tokens[{index}]")
        if not isinstance(kind, ActiveOriginKind):
            raise V2StateError(f"active_origin_kind_by_pos[{index}] must be an ActiveOriginKind")
        if not isinstance(ref, FeedbackOriginRef):
            raise V2StateError(f"feedback_origin_ref_by_pos[{index}] must be a FeedbackOriginRef")
        if not isinstance(status, ActiveScoreStatus):
            raise V2StateError(f"active_score_status_by_pos[{index}] must be an ActiveScoreStatus")
        if not isinstance(provenance[index], PositionProvenance):
            raise V2StateError(f"provenance_by_pos[{index}] must be a PositionProvenance")

        resolved = token != mask_token_id
        if resolved and index not in anchors and token not in aa_token_ids:
            raise V2StateError(
                f"position {index} carries token {token}, which is neither the mask id nor a "
                "canonical AA20 id; the Head firewall is enforced on the id set the state carries"
            )

        # origin kind must agree with the token and the anchor set
        if index in anchors:
            expected_kind = ActiveOriginKind.HARD_ANCHOR
        elif not resolved:
            expected_kind = ActiveOriginKind.UNRESOLVED
        else:
            expected_kind = None  # DENOISER_SAMPLE or FEEDBACK_INJECTION are both legal
        if expected_kind is not None and kind is not expected_kind:
            raise V2StateError(
                f"position {index} is {expected_kind.value} by construction but declares "
                f"{kind.value}"
            )
        if expected_kind is None and kind in (
            ActiveOriginKind.HARD_ANCHOR, ActiveOriginKind.UNRESOLVED
        ):
            raise V2StateError(
                f"position {index} carries a resolved editable token but declares {kind.value}"
            )

        # a feedback origin class requires a feedback origin kind AND a transition to reach
        if ref is not FeedbackOriginRef.NONE and kind not in _FEEDBACK_ORIGIN_KINDS:
            raise V2StateError(
                f"position {index} declares feedback origin {ref.value!r} but origin kind "
                f"{kind.value!r}; only a feedback injection may name a feedback class"
            )
        if kind is ActiveOriginKind.FEEDBACK_INJECTION:
            if ref is FeedbackOriginRef.NONE:
                raise V2StateError(
                    f"position {index} is a feedback injection but names no feedback class"
                )
            if transition_ids[index] is None:
                raise V2StateError(
                    f"position {index} is a feedback injection with no origin_transition_id; its "
                    "ancestry would be unreachable, and reconstructing it from the token is exactly "
                    "what PLAN §2.4 forbids"
                )
        if transition_ids[index] is not None:
            require_id_namespace(transition_ids[index], "txn")

        # status must agree with the token and the anchor set
        if index in anchors:
            expected_status = ActiveScoreStatus.NOT_RANKED_ANCHOR
        elif not resolved:
            expected_status = ActiveScoreStatus.MASKED
        else:
            expected_status = None
        if expected_status is not None and status is not expected_status:
            raise V2StateError(
                f"position {index} must carry status {expected_status.value!r}, got "
                f"{status.value!r}"
            )
        if expected_status is None and status in (
            ActiveScoreStatus.MASKED, ActiveScoreStatus.NOT_RANKED_ANCHOR
        ):
            raise V2StateError(
                f"position {index} carries a resolved editable token but declares status "
                f"{status.value!r}"
            )

        # the active score namespace: present iff the status says it is rankable
        if status in _SCORELESS_STATUSES:
            if score is not None:
                raise V2StateError(
                    f"position {index} has status {status.value!r} but carries active score "
                    f"{score!r}; provenance evidence is never copied into the active score slot"
                )
        else:
            if score is None:
                raise V2StateError(
                    f"position {index} has status {status.value!r} and must carry a finite active "
                    "score"
                )
            _require_finite(score, f"active_sampler_score_by_pos[{index}]")

        # commit identity: present iff the position is a resolved non-anchor
        wants_commit = resolved and index not in anchors
        if wants_commit and commit_key is None:
            raise V2StateError(
                f"position {index} is resolved and editable but records no commit event"
            )
        if not wants_commit and commit_key is not None:
            raise V2StateError(
                f"position {index} is unresolved or anchored and must record no commit event"
            )
        if commit_key is not None and not isinstance(commit_key, HistoryKey):
            raise V2StateError(
                f"active_commit_depth_step_by_pos[{index}] must be a HistoryKey; (depth, step) is "
                "the only legal key, so stationary cycles cannot alias two events"
            )

        last = provenance[index].last_origin
        mismatches: list[str] = []
        if last.token != token:
            mismatches.append(f"token {last.token} != {token}")
        if last.origin_kind is not kind:
            mismatches.append(f"kind {last.origin_kind.value} != {kind.value}")
        if last.origin_ref is not ref:
            mismatches.append(f"ref {last.origin_ref.value} != {ref.value}")
        if commit_key is not None and last.commit != commit_key:
            mismatches.append(f"commit {last.commit} != {commit_key}")
        if last.transition_id != transition_ids[index]:
            mismatches.append(
                f"transition {last.transition_id!r} != {transition_ids[index]!r}"
            )
        if mismatches:
            raise V2StateError(
                f"provenance_by_pos[{index}].last_origin does not describe the active position: "
                + "; ".join(mismatches)
            )

    return anchors, editable


@dataclass(frozen=True)
class LivePartialState:
    """Layer 1 -- the committed live partial state ``P_d`` captured at ``c_d``.

    A live state is what a lookahead may fork from, so it must be fully consumable: every non-anchor
    committed token carries a valid active score, no protection is still in force, and at least one
    editable position remains unresolved.
    """

    schema_version: str
    lineage: LineageRef
    sampler_step: int
    n_steps: int
    tokens: tuple[int, ...]
    mask_token_id: int
    aa_token_ids: frozenset[int]
    hard_anchors: tuple[tuple[int, int], ...]
    editable_positions: tuple[int, ...]
    active_origin_kind_by_pos: tuple[ActiveOriginKind, ...]
    feedback_origin_ref_by_pos: tuple[FeedbackOriginRef, ...]
    active_commit_depth_step_by_pos: tuple[HistoryKey | None, ...]
    origin_transition_id_by_pos: tuple[str | None, ...]
    active_sampler_score_by_pos: tuple[float | None, ...]
    active_score_status_by_pos: tuple[ActiveScoreStatus, ...]
    provenance_by_pos: tuple[PositionProvenance, ...]
    expired_protection: tuple["TemporaryProtection", ...]
    replay: ReplayIdentity
    accumulated_lineage_dfe: int
    conditioning: V2ConditioningIdentity
    safety_reference: SafetyReferenceBinding
    cost_event_ids: tuple[str, ...]
    #: Derived from the state's own content when omitted. An identifier is a function of what it
    #: identifies, so hand-minting one is only meaningful in a tamper test.
    state_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "hard_anchors", _canonical_anchor_set(self.hard_anchors))
        object.__setattr__(
            self, "editable_positions",
            _canonical_index_set(self.editable_positions, "editable position"),
        )
        expired: list[TemporaryProtection] = []
        for protection in self.expired_protection:
            if not isinstance(protection, TemporaryProtection):
                raise V2StateError(
                    "expired_protection entries must be TemporaryProtection records"
                )
            expired.append(protection)
        expired.sort(key=lambda item: item.position)
        if len({item.position for item in expired}) != len(expired):
            raise V2StateError("expired_protection contains duplicate positions")
        object.__setattr__(self, "expired_protection", tuple(expired))

        if self.schema_version != V2_STATE_SCHEMA_VERSION:
            raise V2StateError(
                f"schema_version {self.schema_version!r} != {V2_STATE_SCHEMA_VERSION!r}; an old "
                "digest must never silently validate against a new projection"
            )
        if not isinstance(self.lineage, LineageRef):
            raise V2StateError("lineage must be a LineageRef")
        if not isinstance(self.conditioning, V2ConditioningIdentity):
            raise V2StateError("conditioning must be a V2ConditioningIdentity, stored by value")
        if not isinstance(self.safety_reference, SafetyReferenceBinding):
            raise V2StateError("safety_reference must be a SafetyReferenceBinding")
        if not isinstance(self.replay, ReplayIdentity):
            raise V2StateError("replay must be a ReplayIdentity")
        _require_index(self.n_steps, "n_steps", minimum=1)
        _require_index(self.sampler_step, "sampler_step")
        if self.sampler_step >= self.n_steps:
            raise V2StateError(
                f"sampler_step={self.sampler_step} must be < n_steps={self.n_steps}; a state at "
                "n_steps is a terminal endpoint, not a live partial state"
            )
        _require_index(self.accumulated_lineage_dfe, "accumulated_lineage_dfe")
        _require_index(self.mask_token_id, "mask_token_id")
        if not self.aa_token_ids:
            raise V2StateError("aa_token_ids must be a non-empty canonical AA20 id set")
        if self.mask_token_id in self.aa_token_ids:
            raise V2StateError("mask_token_id may not be a member of the canonical AA20 id set")
        for event_id in self.cost_event_ids:
            _require_text(event_id, "cost_event_id")
        anchors, editable = _validate_position_algebra(
            tokens=self.tokens,
            mask_token_id=self.mask_token_id,
            aa_token_ids=self.aa_token_ids,
            hard_anchors=self.hard_anchors,
            editable_positions=self.editable_positions,
            origin_kind=self.active_origin_kind_by_pos,
            origin_ref=self.feedback_origin_ref_by_pos,
            commit=self.active_commit_depth_step_by_pos,
            transition_ids=self.origin_transition_id_by_pos,
            scores=self.active_sampler_score_by_pos,
            statuses=self.active_score_status_by_pos,
            provenance=self.provenance_by_pos,
        )
        for protection in self.expired_protection:
            if protection.position not in editable:
                raise V2StateError(
                    f"expired protection position {protection.position} is not editable"
                )
            if protection.position in anchors:
                raise V2StateError(
                    f"hard anchor position {protection.position} cannot carry expired temporary "
                    "protection telemetry"
                )
            if protection.expiry_step != self.sampler_step:
                raise V2StateError(
                    f"expired protection at position {protection.position} ended at "
                    f"{protection.expiry_step}, not capture step {self.sampler_step}"
                )

        if self.safety_reference.sequence_length != len(self.tokens):
            raise V2StateError("safety reference length does not match the live state")
        if self.safety_reference.head_binding.protein_id != self.lineage.protein_id:
            raise V2StateError("safety reference protein does not match the live lineage")
        if self.conditioning.complete_reference != self.safety_reference.reference_content_digest:
            raise V2StateError(
                "conditioning complete-reference digest does not match the bound safety reference"
            )

        # Structural difference #1 from the projected layer: a live state admits no pending token.
        pending = [
            index for index, status in enumerate(self.active_score_status_by_pos)
            if status is ActiveScoreStatus.PENDING_ASSIMILATION
        ]
        if pending:
            raise V2StateError(
                f"positions {pending} are still pending assimilation on a committed live state; "
                "an injected token must be assimilated during the segment, before any active-score "
                "consumer can read it (PLAN §3.1, §3.3)"
            )

        maturity = self.realized_maturity
        if maturity.n_unresolved_editable < 1:
            raise V2StateError(
                "a live partial state must retain at least one unresolved editable position; a "
                "fully resolved state can emit no lookahead and is a terminal endpoint"
        )
        digest = self.content_digest
        expected_state_id = make_live_state_id(
            self.lineage.protein_id, self.lineage.family_id,
            depth=self.lineage.depth, step=self.sampler_step, content_digest=digest,
        )
        if self.state_id is None:
            object.__setattr__(self, "state_id", expected_state_id)
        elif self.state_id != expected_state_id:
            raise V2StateError(
                f"state_id {self.state_id!r} does not equal the lineage/step/content-derived "
                f"identity {expected_state_id!r}"
            )

    @property
    def realized_maturity(self) -> MaturityRecord:
        """Derived from ``tokens`` alone -- never from commit history, never stored."""
        n_editable = len(self.editable_positions)
        n_unresolved = sum(
            1 for position in self.editable_positions
            if self.tokens[position] == self.mask_token_id
        )
        return observe_maturity(
            length_total=len(self.tokens),
            n_fixed=len(self.hard_anchors),
            n_editable=n_editable,
            n_unresolved_editable=n_unresolved,
        )

    @property
    def t(self) -> float:
        """Diffusion time of the capture coordinate. Derived, never stored."""
        return self.sampler_step / float(self.n_steps)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": V2_STATE_SCHEMA_VERSION,
            "layer": "live",
            "lineage": _lineage_payload(self.lineage),
            "sampler_step": self.sampler_step,
            "n_steps": self.n_steps,
            "tokens": list(self.tokens),
            "mask_token_id": self.mask_token_id,
            "aa_token_ids": sorted(self.aa_token_ids),
            "hard_anchors": [list(pair) for pair in self.hard_anchors],
            "editable_positions": list(self.editable_positions),
            "active_origin_kind_by_pos": [k.value for k in self.active_origin_kind_by_pos],
            "feedback_origin_ref_by_pos": [r.value for r in self.feedback_origin_ref_by_pos],
            "active_commit_depth_step_by_pos": [
                None if key is None else [key.depth, key.step]
                for key in self.active_commit_depth_step_by_pos
            ],
            "origin_transition_id_by_pos": list(self.origin_transition_id_by_pos),
            "active_sampler_score_by_pos": list(self.active_sampler_score_by_pos),
            "active_score_status_by_pos": [s.value for s in self.active_score_status_by_pos],
            "provenance_by_pos": [p.canonical_payload() for p in self.provenance_by_pos],
            "expired_protection": [
                protection.canonical_payload() for protection in self.expired_protection
            ],
            "replay": self.replay.canonical_payload(),
            "accumulated_lineage_dfe": self.accumulated_lineage_dfe,
            "conditioning": self.conditioning.canonical_payload(),
            "safety_reference": self.safety_reference.canonical_payload(),
            "cost_event_ids": list(self.cost_event_ids),
        }

    @property
    def content_digest(self) -> str:
        return canonical_digest(self.canonical_payload())


# --------------------------------------------------------------------------------------------
# Layer 4 -- the projected partial state produced by q_phi at r_d
# --------------------------------------------------------------------------------------------

class SupportAction(str, enum.Enum):
    """The four-way editable partition of PLAN §2.4."""

    WRITE_FROM_ENDPOINT = "write_from_endpoint"
    INJECT_FROM_SOURCE_FEEDBACK = "inject_from_source_feedback"
    REOPEN = "reopen"
    CARRY_FROM_SOURCE = "carry_from_source"


class ProtectionGrantReason(str, enum.Enum):
    """Why a position holds a temporary protection. Only injections may."""

    ENDPOINT_INJECTION = "endpoint_injection"
    SOURCE_FEEDBACK_INJECTION = "source_feedback_injection"


#: The two support actions that inject an identity and therefore begin pending assimilation.
_INJECTION_ACTIONS = (SupportAction.WRITE_FROM_ENDPOINT, SupportAction.INJECT_FROM_SOURCE_FEEDBACK)


@dataclass(frozen=True)
class TemporaryProtection:
    """A segment-local protection interval over one **editable** position.

    Half-open by construction: the position is protected from remask at sampler step ``s`` exactly
    while ``granted_at_step <= s < expiry_step`` (PLAN §2.4). Its members stay in
    ``editable_positions`` and never enter ``hard_anchors``, which is what stops a temporary
    protection quietly becoming a permanent fixed token.
    """

    position: int
    expiry_step: int
    granted_at_depth: int
    granted_at_step: int
    granted_by_transition_id: str
    grant_reason: ProtectionGrantReason

    def __post_init__(self) -> None:
        _require_index(self.position, "position")
        _require_index(self.expiry_step, "expiry_step")
        _require_index(self.granted_at_depth, "granted_at_depth")
        _require_index(self.granted_at_step, "granted_at_step")
        require_id_namespace(self.granted_by_transition_id, "txn")
        if not isinstance(self.grant_reason, ProtectionGrantReason):
            raise V2StateError("grant_reason must be a ProtectionGrantReason")
        if self.granted_at_step >= self.expiry_step:
            raise V2StateError(
                f"protection at position {self.position} would expire at step {self.expiry_step} "
                f"before it is granted at {self.granted_at_step}; the interval is half-open and "
                "must be non-empty"
            )

    def protects_at(self, step: int) -> bool:
        return self.granted_at_step <= _require_index(step, "step") < self.expiry_step

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "expiry_step": self.expiry_step,
            "granted_at_depth": self.granted_at_depth,
            "granted_at_step": self.granted_at_step,
            "granted_by_transition_id": self.granted_by_transition_id,
            "grant_reason": self.grant_reason.value,
        }


@dataclass(frozen=True)
class SupportPartition:
    """The four sorted position tuples the policy produced, plus its per-position reason evidence.

    Lives on the state layer because the transition record persists it (PLAN §5.3); the projection
    kernel and the policy protocol only pass it through.
    """

    write_from_endpoint: tuple[int, ...]
    inject_from_source_feedback: tuple[int, ...]
    reopen: tuple[int, ...]
    carry_from_source: tuple[int, ...]
    reason_by_pos: Mapping[int, str]

    def __post_init__(self) -> None:
        for action in SupportAction:
            object.__setattr__(
                self, action.value,
                _canonical_index_set(getattr(self, action.value), f"{action.value} position"),
            )
        seen: set[int] = set()
        for action in SupportAction:
            positions = getattr(self, action.value)
            for position in positions:
                if position in seen:
                    raise V2StateError(
                        f"position {position} appears in more than one support action; the four "
                        "sets are disjoint (PLAN §2.4)"
                    )
                seen.add(position)
        if not self.write_from_endpoint:
            raise V2StateError(
                "write_from_endpoint is empty; a transition that adopts no endpoint identity is "
                "not a feedback event (PLAN §2.4)"
            )
        missing = sorted(seen - set(self.reason_by_pos))
        if missing:
            raise V2StateError(
                f"positions {missing} carry no policy reason; per-position reason evidence is "
                "required for every partitioned position (PLAN §2.5)"
            )
        extra = sorted(set(self.reason_by_pos) - seen)
        if extra:
            raise V2StateError(f"reason_by_pos names unpartitioned positions {extra}")

    @property
    def all_positions(self) -> frozenset[int]:
        return frozenset(
            self.write_from_endpoint + self.inject_from_source_feedback
            + self.reopen + self.carry_from_source
        )

    @property
    def injected_positions(self) -> frozenset[int]:
        return frozenset(self.write_from_endpoint + self.inject_from_source_feedback)

    def action_of(self, position: int) -> SupportAction:
        for action in SupportAction:
            if position in getattr(self, action.value):
                return action
        raise V2StateError(f"position {position} is not in the partition")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            action.value: list(getattr(self, action.value)) for action in SupportAction
        } | {"reason_by_pos": {str(k): v for k, v in sorted(self.reason_by_pos.items())}}


def carry_is_temporally_legal(
    commit: HistoryKey | None, *, token: int, r_step: int, mask_token_id: int,
) -> bool:
    """May this source position be carried forward as ordinary history at ``r_d``?

    Two cases are legal, and only two:

    * an **inherited mask** -- masked in the source and not written by any support action. It stays
      unresolved, so it introduces no future information.
    * a **temporally valid resolved token** -- committed strictly before ``r_d``.

    A token committed at ``r_d`` or later is future information at that boundary: checkpoints are
    top-of-step pre-denoiser states, so a commit *at* ``r_d`` was written by the very step being
    rolled back. It must be explicitly injected or reopened.

    An already-masked position is deliberately a carry rather than a reopen. If it could be
    labelled ``reopen``, a policy could claim a reopen action while newly masking nothing -- a
    fail-open that the byte-level mask-load accounting would never see.
    """
    _require_index(r_step, "r_step")
    _require_index(mask_token_id, "mask_token_id")
    if commit is None:
        return token == mask_token_id
    if not isinstance(commit, HistoryKey):
        raise V2StateError("commit must be a HistoryKey or None")
    return 0 <= commit.step < r_step


@dataclass(frozen=True)
class ProjectedPartialState:
    """Layer 4 -- the immediate output of ``q_phi`` at ``r_d``, before any propagation.

    Four structural differences from :class:`LivePartialState`, none of them a label:
    it admits ``PENDING_ASSIMILATION``; it carries an *active* protection set rather than expired
    telemetry; it carries a fork seed and no replay stream, because a projection is a causal
    intervention rather than an identity replay; and it carries the two-coordinate horizon
    ``(r_step, c_next_step)`` together with the partition that produced it.
    """

    schema_version: str
    lineage: LineageRef
    r_step: int
    c_next_step: int
    n_steps: int
    coordinate_law: CoordinateLaw
    tokens: tuple[int, ...]
    mask_token_id: int
    aa_token_ids: frozenset[int]
    hard_anchors: tuple[tuple[int, int], ...]
    editable_positions: tuple[int, ...]
    source_resolved_positions: tuple[int, ...]
    active_origin_kind_by_pos: tuple[ActiveOriginKind, ...]
    feedback_origin_ref_by_pos: tuple[FeedbackOriginRef, ...]
    active_commit_depth_step_by_pos: tuple[HistoryKey | None, ...]
    origin_transition_id_by_pos: tuple[str | None, ...]
    active_sampler_score_by_pos: tuple[float | None, ...]
    active_score_status_by_pos: tuple[ActiveScoreStatus, ...]
    provenance_by_pos: tuple[PositionProvenance, ...]
    support: SupportPartition
    active_temporary_protection: tuple[TemporaryProtection, ...]
    descendant_fork_seed: int
    origin_transition_id: str
    declared_band_id: str
    declared_band_digest: str
    inherited_lineage_dfe: int
    planned_segment_dfe: int
    conditioning: V2ConditioningIdentity
    safety_reference: SafetyReferenceBinding
    cost_event_ids: tuple[str, ...]
    state_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "hard_anchors", _canonical_anchor_set(self.hard_anchors))
        object.__setattr__(
            self, "editable_positions",
            _canonical_index_set(self.editable_positions, "editable position"),
        )
        object.__setattr__(
            self, "source_resolved_positions",
            _canonical_index_set(self.source_resolved_positions, "source resolved position"),
        )
        protections: list[TemporaryProtection] = []
        for protection in self.active_temporary_protection:
            if not isinstance(protection, TemporaryProtection):
                raise V2StateError(
                    "active_temporary_protection entries must be TemporaryProtection records"
                )
            protections.append(protection)
        protections.sort(key=lambda item: item.position)
        object.__setattr__(self, "active_temporary_protection", tuple(protections))

        if self.schema_version != V2_STATE_SCHEMA_VERSION:
            raise V2StateError(f"schema_version must be {V2_STATE_SCHEMA_VERSION!r}")
        if not isinstance(self.lineage, LineageRef):
            raise V2StateError("lineage must be a LineageRef")
        if not isinstance(self.support, SupportPartition):
            raise V2StateError("support must be a SupportPartition")
        if not isinstance(self.conditioning, V2ConditioningIdentity):
            raise V2StateError("conditioning must be a V2ConditioningIdentity")
        if not isinstance(self.safety_reference, SafetyReferenceBinding):
            raise V2StateError("safety_reference must be a SafetyReferenceBinding")
        if not isinstance(self.coordinate_law, CoordinateLaw):
            raise V2StateError("coordinate_law must be a CoordinateLaw")
        _require_index(self.n_steps, "n_steps", minimum=1)
        _require_index(self.r_step, "r_step")
        _require_index(self.c_next_step, "c_next_step")
        _require_index(self.inherited_lineage_dfe, "inherited_lineage_dfe")
        _require_index(self.descendant_fork_seed, "descendant_fork_seed")
        _require_index(self.mask_token_id, "mask_token_id")
        require_id_namespace(self.origin_transition_id, "txn")
        _require_text(self.declared_band_id, "declared_band_id")
        require_digest(self.declared_band_digest, "declared_band_digest")
        if self.r_step >= self.c_next_step:
            raise V2StateError(
                f"r_step={self.r_step} must precede c_next_step={self.c_next_step}"
            )
        if self.c_next_step >= self.n_steps:
            raise V2StateError(
                f"c_next_step={self.c_next_step} must be < n_steps={self.n_steps}"
            )
        if self.planned_segment_dfe != self.c_next_step - self.r_step:
            raise V2StateError(
                f"planned_segment_dfe={self.planned_segment_dfe} != c_next_step - r_step = "
                f"{self.c_next_step - self.r_step}; the segment charges exactly this and the "
                "projection itself adds none (PLAN §3.4)"
            )
        if self.mask_token_id in self.aa_token_ids:
            raise V2StateError("mask_token_id may not be a member of the canonical AA20 id set")
        if self.lineage.parent_transition_id != self.origin_transition_id:
            raise V2StateError(
                "projected lineage parent_transition_id does not match origin_transition_id"
            )
        if self.safety_reference.sequence_length != len(self.tokens):
            raise V2StateError("safety reference length does not match the projected state")
        if self.safety_reference.head_binding.protein_id != self.lineage.protein_id:
            raise V2StateError("safety reference protein does not match the projected lineage")
        if self.conditioning.complete_reference != self.safety_reference.reference_content_digest:
            raise V2StateError(
                "conditioning complete-reference digest does not match the bound safety reference"
            )

        anchors, editable = _validate_position_algebra(
            tokens=self.tokens,
            mask_token_id=self.mask_token_id,
            aa_token_ids=self.aa_token_ids,
            hard_anchors=self.hard_anchors,
            editable_positions=self.editable_positions,
            origin_kind=self.active_origin_kind_by_pos,
            origin_ref=self.feedback_origin_ref_by_pos,
            commit=self.active_commit_depth_step_by_pos,
            transition_ids=self.origin_transition_id_by_pos,
            scores=self.active_sampler_score_by_pos,
            statuses=self.active_score_status_by_pos,
            provenance=self.provenance_by_pos,
        )

        source_resolved = frozenset(self.source_resolved_positions)
        if source_resolved - editable:
            raise V2StateError(
                f"source_resolved_positions names non-editable positions "
                f"{sorted(source_resolved - editable)}"
            )

        # --- the partition must exactly tile the editable domain -----------------------------
        partitioned = self.support.all_positions
        if partitioned & anchors:
            raise V2StateError(
                f"support assigns hard anchors {sorted(partitioned & anchors)}; anchors are a "
                "separate permanent class outside the editable domain (PLAN §2.4)"
            )
        if partitioned != editable:
            missing = sorted(editable - partitioned)
            extra = sorted(partitioned - editable)
            raise V2StateError(
                f"the four support sets must be exhaustive over the editable domain; missing "
                f"{missing}, unexpected {extra}"
            )

        # --- reopen means NEWLY masked, never an inherited mask relabelled --------------------
        for position in self.support.reopen:
            if position not in source_resolved:
                raise V2StateError(
                    f"position {position} is labelled reopen but was already masked in the source "
                    "state; an inherited mask is a carry. Otherwise a policy could claim a reopen "
                    "action while newly masking nothing (PLAN §2.4)"
                )
            if self.tokens[position] != self.mask_token_id:
                raise V2StateError(
                    f"position {position} is labelled reopen but carries a resolved token"
                )

        # --- injections begin pending, commit at the re-entry key, and are protected ----------
        reentry_key = HistoryKey(depth=self.lineage.depth, step=self.r_step)
        expected_ref_by_action = {
            SupportAction.WRITE_FROM_ENDPOINT: FeedbackOriginRef.SELECTED_ENDPOINT,
            SupportAction.INJECT_FROM_SOURCE_FEEDBACK: FeedbackOriginRef.SOURCE_STATE,
        }
        for action, expected_ref in expected_ref_by_action.items():
            for position in getattr(self.support, action.value):
                if self.active_origin_kind_by_pos[position] is not ActiveOriginKind.FEEDBACK_INJECTION:
                    raise V2StateError(
                        f"{action.value} position {position} must be a feedback injection"
                    )
                if self.feedback_origin_ref_by_pos[position] is not expected_ref:
                    raise V2StateError(
                        f"{action.value} position {position} must name {expected_ref.value} "
                        "provenance"
                    )
                if self.origin_transition_id_by_pos[position] != self.origin_transition_id:
                    raise V2StateError(
                        f"{action.value} position {position} does not name this projection's "
                        "origin transition"
                    )
        for position in sorted(self.support.injected_positions):
            if self.active_score_status_by_pos[position] is not ActiveScoreStatus.PENDING_ASSIMILATION:
                raise V2StateError(
                    f"injected position {position} must begin pending_assimilation with no "
                    "fabricated active score; assimilation happens on the first active forward "
                    "(PLAN §2.4, §3.1)"
                )
            if self.active_commit_depth_step_by_pos[position] != reentry_key:
                raise V2StateError(
                    f"injected position {position} must commit at the re-entry key {reentry_key}, "
                    "not at its source or endpoint step; later evidence survives as provenance "
                    "only (safety law 13)"
                )
        for position in self.support.reopen:
            if (
                self.active_origin_kind_by_pos[position] is not ActiveOriginKind.UNRESOLVED
                or self.feedback_origin_ref_by_pos[position] is not FeedbackOriginRef.NONE
                or self.active_commit_depth_step_by_pos[position] is not None
                or self.origin_transition_id_by_pos[position] is not None
            ):
                raise V2StateError(
                    f"reopen position {position} must reset active origin, feedback reference, "
                    "commit, and transition identity"
                )
        for position in sorted(self.support.carry_from_source):
            if not carry_is_temporally_legal(
                self.active_commit_depth_step_by_pos[position],
                token=self.tokens[position], r_step=self.r_step,
                mask_token_id=self.mask_token_id,
            ):
                raise V2StateError(
                    f"position {position} is labelled carry_from_source but its identity is not "
                    f"temporally valid at r_step={self.r_step}; it must be explicitly injected or "
                    "reopened"
                )

        # --- the protection relation ---------------------------------------------------------
        protected = [p.position for p in self.active_temporary_protection]
        if len(set(protected)) != len(protected):
            raise V2StateError("a position holds more than one temporary protection")
        if frozenset(protected) != self.support.injected_positions:
            raise V2StateError(
                f"the protected set {sorted(protected)} must equal the injected set "
                f"{sorted(self.support.injected_positions)}; protecting a carried or reopened "
                "position would be a permanent fixed token in disguise, and leaving an injection "
                "unprotected would let it be remasked before assimilation (PLAN §2.4, §3.3)"
            )
        for protection in self.active_temporary_protection:
            action = self.support.action_of(protection.position)
            expected_reason = {
                SupportAction.WRITE_FROM_ENDPOINT: ProtectionGrantReason.ENDPOINT_INJECTION,
                SupportAction.INJECT_FROM_SOURCE_FEEDBACK:
                    ProtectionGrantReason.SOURCE_FEEDBACK_INJECTION,
            }[action]
            if protection.grant_reason is not expected_reason:
                raise V2StateError(
                    f"protection at position {protection.position} has reason "
                    f"{protection.grant_reason.value!r}, expected {expected_reason.value!r} for "
                    f"{action.value}"
                )
            if protection.granted_by_transition_id != self.origin_transition_id:
                raise V2StateError(
                    f"protection at position {protection.position} was granted by a different "
                    "transition"
                )
            if protection.granted_at_depth != self.lineage.depth:
                raise V2StateError(
                    f"protection at position {protection.position} was granted at depth "
                    f"{protection.granted_at_depth}, not projected depth {self.lineage.depth}"
                )
            if protection.position in anchors:
                raise V2StateError(
                    f"position {protection.position} is a hard anchor and can never hold a "
                    "temporary protection"
                )
            if protection.expiry_step != self.c_next_step:
                raise V2StateError(
                    f"protection at position {protection.position} expires at "
                    f"{protection.expiry_step}, not at c_next_step={self.c_next_step}; endpoint and "
                    "source-feedback injections share one exclusive expiry (PLAN §2.4)"
                )
            if protection.granted_at_step != self.r_step:
                raise V2StateError(
                    f"protection at position {protection.position} is granted at "
                    f"{protection.granted_at_step}, not at r_step={self.r_step}"
                )

        if self.realized_maturity.n_unresolved_editable < 1:
            raise V2StateError(
                "a projected state must contain at least one unresolved editable position "
                "(PLAN §2.4)"
            )

        digest = self.content_digest
        expected_state_id = make_projected_state_id(
            self.lineage.protein_id, self.lineage.family_id,
            depth=self.lineage.depth, r_step=self.r_step, content_digest=digest,
        )
        if self.state_id is None:
            object.__setattr__(self, "state_id", expected_state_id)
        elif self.state_id != expected_state_id:
            raise V2StateError(
                f"state_id {self.state_id!r} does not equal the lineage/re-entry/content-derived "
                f"identity {expected_state_id!r}"
            )

    @property
    def realized_maturity(self) -> MaturityRecord:
        n_unresolved = sum(
            1 for position in self.editable_positions
            if self.tokens[position] == self.mask_token_id
        )
        return observe_maturity(
            length_total=len(self.tokens), n_fixed=len(self.hard_anchors),
            n_editable=len(self.editable_positions), n_unresolved_editable=n_unresolved,
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": V2_STATE_SCHEMA_VERSION,
            "layer": "projected",
            "lineage": _lineage_payload(self.lineage),
            "r_step": self.r_step,
            "c_next_step": self.c_next_step,
            "n_steps": self.n_steps,
            "coordinate_law": self.coordinate_law.value,
            "tokens": list(self.tokens),
            "mask_token_id": self.mask_token_id,
            "aa_token_ids": sorted(self.aa_token_ids),
            "hard_anchors": [list(pair) for pair in self.hard_anchors],
            "editable_positions": list(self.editable_positions),
            "source_resolved_positions": list(self.source_resolved_positions),
            "active_origin_kind_by_pos": [k.value for k in self.active_origin_kind_by_pos],
            "feedback_origin_ref_by_pos": [r.value for r in self.feedback_origin_ref_by_pos],
            "active_commit_depth_step_by_pos": [
                None if key is None else [key.depth, key.step]
                for key in self.active_commit_depth_step_by_pos
            ],
            "origin_transition_id_by_pos": list(self.origin_transition_id_by_pos),
            "active_sampler_score_by_pos": list(self.active_sampler_score_by_pos),
            "active_score_status_by_pos": [s.value for s in self.active_score_status_by_pos],
            "provenance_by_pos": [p.canonical_payload() for p in self.provenance_by_pos],
            "support": self.support.canonical_payload(),
            "active_temporary_protection": [
                p.canonical_payload() for p in self.active_temporary_protection
            ],
            "descendant_fork_seed": self.descendant_fork_seed,
            "origin_transition_id": self.origin_transition_id,
            "declared_band_id": self.declared_band_id,
            "declared_band_digest": self.declared_band_digest,
            "inherited_lineage_dfe": self.inherited_lineage_dfe,
            "planned_segment_dfe": self.planned_segment_dfe,
            "conditioning": self.conditioning.canonical_payload(),
            "safety_reference": self.safety_reference.canonical_payload(),
            "cost_event_ids": list(self.cost_event_ids),
        }

    @property
    def content_digest(self) -> str:
        return canonical_digest(self.canonical_payload())


def validate_endpoint_writeback(
    *, projected: ProjectedPartialState, endpoint_tokens: tuple[int, ...],
) -> None:
    """Every ``write_from_endpoint`` position must carry the selected endpoint's own token.

    Without this the provenance would name an endpoint the bytes did not come from, which is the
    "endpoint-written token provenance does not match the endpoint" case of PLAN §7.2. The check is
    a caller-supplied comparison rather than a stored copy, so the endpoint stays byte-identical in
    the archive and is never duplicated onto the state.
    """
    if len(endpoint_tokens) != len(projected.tokens):
        raise V2StateError(
            f"endpoint carries {len(endpoint_tokens)} positions but the projected state carries "
            f"{len(projected.tokens)}"
        )
    for position in projected.support.write_from_endpoint:
        if projected.tokens[position] != endpoint_tokens[position]:
            raise V2StateError(
                f"position {position} is labelled write_from_endpoint but carries token "
                f"{projected.tokens[position]} while the endpoint carries "
                f"{endpoint_tokens[position]}"
            )


# --------------------------------------------------------------------------------------------
# Segment capture, and Layers 2/3/5
# --------------------------------------------------------------------------------------------

class FeasibilityLevel(str, enum.Enum):
    """How much structural evidence an endpoint carries (FUSION_V2 §3.3).

    Only ``DEFINITIVE`` authorizes feedback ancestry, the reported feasible frontier, or a final
    returned design. The tri-state exists so ``unvalidated`` (nothing run) and ``provisional``
    (a cheap prior only) can never be read as the definitive verdict.
    """

    UNVALIDATED = "unvalidated"
    PROVISIONAL = "provisional"
    DEFINITIVE = "definitive"


_FEASIBILITY_ORDER = {
    FeasibilityLevel.UNVALIDATED: 0,
    FeasibilityLevel.PROVISIONAL: 1,
    FeasibilityLevel.DEFINITIVE: 2,
}


class TransitionOutcome(str, enum.Enum):
    """Typed outcomes so a refusal is recorded rather than turned into an invented state."""

    COMMITTED = "committed"
    NULL_NO_ADMISSIBLE_ENDPOINT = "null_no_admissible_endpoint"
    NULL_INVALID_POLICY_RESULT = "null_invalid_policy_result"
    NULL_BAND_INCOMPATIBLE = "null_band_incompatible"
    TERMINAL_BEST_LOOKAHEAD = "terminal_best_lookahead"
    STALLED_NO_NOVEL_DESCENDANT = "stalled_no_novel_descendant"


def validate_propagated_capture(
    *, projected: "ProjectedPartialState", propagated: LivePartialState,
    background_remask_events: int,
) -> None:
    """Check that a propagation segment ended where and how it declared it would.

    This is the trust boundary before a captured row enters the denoiser/archive graph. It binds
    the full scientific context and verifies that propagation only resolves projected masks:

    * it stopped at the declared ``c_{d+1}`` on the true pre-denoiser boundary;
    * nothing is still pending assimilation (the live layer already forbids this, restated here so
      the failure names the segment rather than the record);
    * every granted protection is recorded as expired, so protection cannot leak across depth;
    * the segment recorded **zero** background-remask events -- the frozen V2 substrate has none,
      so a single event means the segment ran on the wrong substrate; and
    * unresolved mass did not increase. Denoising only resolves, so an increase means something
      re-masked outside the explicit projection.
    """
    if not isinstance(projected, ProjectedPartialState):
        raise V2StateError("projected must be a ProjectedPartialState")
    if not isinstance(propagated, LivePartialState):
        raise V2StateError("propagated must be a LivePartialState")
    _require_index(background_remask_events, "background_remask_events")

    if propagated.sampler_step != projected.c_next_step:
        raise V2StateError(
            f"capture landed at step {propagated.sampler_step}, not at the declared "
            f"c_next_step={projected.c_next_step}"
        )
    lineage_fields = ("protein_id", "root_id", "family_id", "depth",
                      "parent_transition_id", "origin_endpoint_id")
    lineage_mismatches = [
        name for name in lineage_fields
        if getattr(propagated.lineage, name) != getattr(projected.lineage, name)
    ]
    if lineage_mismatches:
        raise V2StateError(
            f"propagated lineage differs from projected lineage in {lineage_mismatches}"
        )
    if propagated.lineage.parent_state_id != projected.state_id:
        raise V2StateError(
            "propagated lineage parent_state_id must name the exact projected state"
        )
    if projected.lineage.parent_transition_id != projected.origin_transition_id:
        raise V2StateError(
            "projected lineage transition edge does not match the projection origin transition"
        )

    domain_fields = (
        "n_steps", "mask_token_id", "aa_token_ids", "hard_anchors", "editable_positions",
    )
    mismatched_domain = [
        name for name in domain_fields if getattr(propagated, name) != getattr(projected, name)
    ]
    if mismatched_domain:
        raise V2StateError(
            f"propagated sampler/AA20/anchor/editable domain differs in {mismatched_domain}"
        )
    if propagated.conditioning != projected.conditioning:
        raise V2StateError("propagated conditioning differs from the projected state")
    if propagated.safety_reference != projected.safety_reference:
        raise V2StateError("propagated safety reference differs from the projected state")
    if propagated.replay.mode != "fork" or (
        propagated.replay.fork_seed != projected.descendant_fork_seed
    ):
        raise V2StateError(
            "propagated replay fork does not match projected descendant_fork_seed"
        )
    expected_lineage_dfe = projected.inherited_lineage_dfe + projected.planned_segment_dfe
    if propagated.accumulated_lineage_dfe != expected_lineage_dfe:
        raise V2StateError(
            f"propagated lineage DFE {propagated.accumulated_lineage_dfe} != inherited plus "
            f"segment DFE {expected_lineage_dfe}"
        )
    if background_remask_events != 0:
        raise V2StateError(
            f"the segment recorded {background_remask_events} background-remask events; the frozen "
            "V2/A2 substrate performs none, so only an explicit projection may add masks "
            "(PLAN §3.1, §5.1)"
        )
    pending = [
        index for index, status in enumerate(propagated.active_score_status_by_pos)
        if status is ActiveScoreStatus.PENDING_ASSIMILATION
    ]
    if pending:
        raise V2StateError(
            f"positions {pending} left the segment still pending assimilation; every injected "
            "token must be assimilated on the first active forward (PLAN §3.1)"
        )
    if propagated.expired_protection != projected.active_temporary_protection:
        raise V2StateError(
            "expired protection records do not exactly match the full grants; position-only "
            "matching cannot bind expiry, transition, depth, or grant reason (PLAN §3.3)"
        )
    for protection in projected.active_temporary_protection:
        if propagated.tokens[protection.position] != projected.tokens[protection.position]:
            raise V2StateError(
                f"protected injected token at position {protection.position} changed during the "
                "propagation segment"
            )

    for position, projected_token in enumerate(projected.tokens):
        propagated_token = propagated.tokens[position]
        if projected_token != projected.mask_token_id:
            if propagated_token != projected_token:
                raise V2StateError(
                    f"resolved position {position} changed without explicit remask/projection"
                )
            invariant_fields = (
                "active_origin_kind_by_pos", "feedback_origin_ref_by_pos",
                "active_commit_depth_step_by_pos", "origin_transition_id_by_pos",
                "provenance_by_pos",
            )
            changed = [
                name for name in invariant_fields
                if getattr(propagated, name)[position] != getattr(projected, name)[position]
            ]
            if changed:
                raise V2StateError(
                    f"resolved position {position} changed active/provenance identity fields "
                    f"{changed} during propagation"
                )
            projected_status = projected.active_score_status_by_pos[position]
            if projected_status is ActiveScoreStatus.PENDING_ASSIMILATION:
                if propagated.active_score_status_by_pos[position] is not ActiveScoreStatus.ASSIMILATED:
                    raise V2StateError(
                        f"injected position {position} was not assimilated on propagation"
                    )
            elif (
                propagated.active_score_status_by_pos[position] is not projected_status
                or propagated.active_sampler_score_by_pos[position]
                != projected.active_sampler_score_by_pos[position]
            ):
                raise V2StateError(
                    f"natural/anchor position {position} changed active score state during "
                    "propagation"
                )
        elif propagated_token == propagated.mask_token_id:
            invariant_fields = (
                "active_origin_kind_by_pos", "feedback_origin_ref_by_pos",
                "active_commit_depth_step_by_pos", "origin_transition_id_by_pos",
                "active_sampler_score_by_pos", "active_score_status_by_pos", "provenance_by_pos",
            )
            if any(
                getattr(propagated, name)[position] != getattr(projected, name)[position]
                for name in invariant_fields
            ):
                raise V2StateError(
                    f"still-masked position {position} changed active/provenance state"
                )
        else:
            if (
                propagated.active_origin_kind_by_pos[position]
                is not ActiveOriginKind.DENOISER_SAMPLE
                or propagated.feedback_origin_ref_by_pos[position] is not FeedbackOriginRef.NONE
                or propagated.origin_transition_id_by_pos[position] is not None
                or propagated.active_score_status_by_pos[position]
                is not ActiveScoreStatus.HISTORICAL_NATURAL
            ):
                raise V2StateError(
                    f"newly resolved position {position} is not a natural denoiser sample"
                )
            commit = propagated.active_commit_depth_step_by_pos[position]
            if (
                commit is None
                or commit.depth != projected.lineage.depth
                or not projected.r_step <= commit.step < projected.c_next_step
            ):
                raise V2StateError(
                    f"newly resolved position {position} has an impossible segment commit {commit}"
                )
    before = projected.realized_maturity.n_unresolved_editable
    after = propagated.realized_maturity.n_unresolved_editable
    if after > before:
        raise V2StateError(
            f"unresolved editable mass rose from {before} to {after} across the segment; "
            "propagation only resolves positions"
        )


@dataclass(frozen=True)
class EndpointPositionEvidence:
    """Per-position completion evidence for one exact endpoint.

    Grounded in ``SamplerOutput.unmask_step_by_pos`` and ``SamplerOutput.final_scores``, both of
    which the V1 completer currently computes and discards. ``completion_logprob`` is endpoint
    evidence and is never an active sampler score.
    """

    token: int
    commit: HistoryKey
    completion_logprob: float | None
    inherited_from_source: bool

    def __post_init__(self) -> None:
        _require_index(self.token, "token")
        if not isinstance(self.commit, HistoryKey):
            raise V2StateError("commit must be a HistoryKey")
        if self.completion_logprob is not None:
            _require_finite(self.completion_logprob, "completion_logprob")
        if not isinstance(self.inherited_from_source, bool):
            raise V2StateError("inherited_from_source must be an explicit bool")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "commit": [self.commit.depth, self.commit.step],
            "completion_logprob": self.completion_logprob,
            "inherited_from_source": self.inherited_from_source,
        }


def _structure_outcome_payload(outcome: StructureOutcome | None) -> dict[str, Any] | None:
    if outcome is None:
        return None
    if not isinstance(outcome, StructureOutcome):
        raise V2StateError("structure_outcome must be a StructureOutcome or None")
    if not isinstance(outcome.model_executed, bool) or not isinstance(outcome.evaluated, bool):
        raise V2StateError("structure outcome boolean fields must be explicit bool values")
    _require_text(outcome.cache_status, "structure cache_status")
    walltime = _require_finite(outcome.walltime_s, "structure walltime_s")
    if walltime < 0.0:
        raise V2StateError("structure walltime_s must be non-negative")
    if outcome.failure_reason is not None:
        _require_text(outcome.failure_reason, "structure failure_reason")
    metrics = None
    if outcome.metrics is not None:
        if not isinstance(outcome.metrics, Mapping):
            raise V2StateError("structure metrics must be a mapping")
        metrics = {}
        for name, value in outcome.metrics.items():
            key = _require_text(name, "structure metric name")
            metrics[key] = _require_finite(value, f"structure metric {key}")
    return {
        "evaluated": outcome.evaluated,
        "feasible": outcome.feasible,
        "cache_status": outcome.cache_status,
        "model_executed": outcome.model_executed,
        "failure_reason": outcome.failure_reason,
        "walltime_s": walltime,
        "metrics": metrics,
    }


def _head_score_payload(
    head_score: HeadScoreLike | None, binding: HeadScoreBinding,
) -> dict[str, Any] | None:
    if head_score is None:
        return None
    if not isinstance(head_score, HeadScoreLike):
        raise V2StateError("head_score must satisfy HeadScoreLike or be None")
    expected = {
        "protein_id": binding.protein_id,
        "sequence_md5": binding.sequence_md5,
        "sequence_length": binding.sequence_length,
        "allele": binding.evaluator.allele,
        "score_scale": binding.evaluator.score_scale,
    }
    mismatched = [
        name for name, value in expected.items() if getattr(head_score, name, None) != value
    ]
    if mismatched:
        raise V2StateError(f"head_score disagrees with head_binding in {mismatched}")
    windows = tuple(head_score.windows)
    if window_grid_digest(windows) != binding.window_grid_digest:
        raise V2StateError("head_score window grid does not match head_binding")
    window_rows = []
    for window in sorted(windows, key=lambda row: (row.start_0b, row.end_0b, row.k)):
        window_rows.append({
            "start_0b": _require_index(window.start_0b, "Head window start_0b"),
            "end_0b": _require_index(window.end_0b, "Head window end_0b"),
            "k": _require_index(window.k, "Head window k", minimum=1),
            "z": _require_finite(window.z, "Head window z"),
        })
    residue_hotspot = getattr(head_score, "residue_hotspot", None)
    if residue_hotspot is not None:
        if len(residue_hotspot) != binding.sequence_length:
            raise V2StateError("head_score residue_hotspot length does not match the sequence")
        residue_hotspot = [
            _require_finite(value, "Head residue hotspot") for value in residue_hotspot
        ]
    global_risk = getattr(head_score, "global_risk", None)
    if global_risk is not None:
        global_risk = _require_finite(global_risk, "Head global risk")
    return {
        "protein_id": head_score.protein_id,
        "sequence_md5": head_score.sequence_md5,
        "sequence_length": head_score.sequence_length,
        "allele": head_score.allele,
        "score_scale": head_score.score_scale,
        "windows": window_rows,
        "residue_hotspot": residue_hotspot,
        "global_risk": global_risk,
    }


@dataclass(frozen=True)
class CompleteEndpoint:
    """Layer 2 -- one exact complete AA20 endpoint with its Head and structure evidence.

    No forward edge is stored. Descendant lineage is derived from the transition table, which keeps
    the endpoint byte-identical in the archive forever (PLAN §2.3).
    """

    schema_version: str
    lineage: LineageRef
    protein_id: str
    sequence: str
    sequence_md5: str
    sequence_length: int
    source_state_id: str
    source_state_content_digest: str
    fork_index: int
    fork_seed: int
    replay: ReplayIdentity
    endpoint_provenance_evidence_by_pos: tuple[EndpointPositionEvidence, ...]
    head_binding: HeadScoreBinding
    head_score: HeadScoreLike | None
    head_global_risk: float
    feasibility_level: FeasibilityLevel
    structure_outcome: StructureOutcome | None
    cost_event_ids: tuple[str, ...]
    endpoint_id: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != V2_STATE_SCHEMA_VERSION:
            raise V2StateError(f"schema_version must be {V2_STATE_SCHEMA_VERSION!r}")
        if not isinstance(self.lineage, LineageRef):
            raise V2StateError("lineage must be a LineageRef")
        if not isinstance(self.feasibility_level, FeasibilityLevel):
            raise V2StateError("feasibility_level must be a FeasibilityLevel")
        _require_text(self.protein_id, "protein_id")
        _require_index(self.sequence_length, "sequence_length", minimum=1)
        _require_index(self.fork_index, "fork_index")
        _require_index(self.fork_seed, "fork_seed")
        _require_finite(self.head_global_risk, "head_global_risk")
        require_digest(self.source_state_content_digest, "source_state_content_digest")
        require_id_namespace(self.source_state_id, "live")
        assert_id_binds_digest(self.source_state_id, self.source_state_content_digest)
        if not isinstance(self.replay, ReplayIdentity):
            raise V2StateError("replay must be a ReplayIdentity")
        if self.replay.mode != "fork" or self.replay.fork_seed != self.fork_seed:
            raise V2StateError(
                "endpoint replay must be a fork stream bound to the endpoint fork_seed"
            )
        if self.lineage.protein_id != self.protein_id:
            raise V2StateError("endpoint protein_id does not match its source lineage")
        expected_source_prefix = (
            f"{self.lineage.protein_id}:{self.lineage.family_id}:live:d{self.lineage.depth}:"
        )
        if not self.source_state_id.startswith(expected_source_prefix):
            raise V2StateError(
                "source_state_id does not match the endpoint protein/family/depth lineage"
            )

        # The Head is fail-open on non-canonical input, so this guard is load-bearing. Reused from
        # the V1 entry stack rather than re-derived: a private copy would be a second place a
        # masked or 'X' residue could reach the Head.
        validate_complete_aa20(self.sequence, self.sequence_length)
        if sequence_md5(self.sequence) != self.sequence_md5:
            raise V2StateError(
                "sequence_md5 does not digest the sequence it claims to identify"
            )
        if len(self.endpoint_provenance_evidence_by_pos) != self.sequence_length:
            raise V2StateError(
                f"endpoint evidence has {len(self.endpoint_provenance_evidence_by_pos)} entries "
                f"but the sequence carries {self.sequence_length} positions"
            )
        for evidence in self.endpoint_provenance_evidence_by_pos:
            if not isinstance(evidence, EndpointPositionEvidence):
                raise V2StateError("endpoint evidence entries must be EndpointPositionEvidence")
        if not isinstance(self.head_binding, HeadScoreBinding):
            raise V2StateError("head_binding must be a HeadScoreBinding")
        if self.head_binding.protein_id != self.protein_id:
            raise V2StateError("head_binding describes a different protein")
        if self.head_binding.sequence_md5 != self.sequence_md5:
            raise V2StateError(
                "head_binding describes a different sequence than this endpoint carries"
            )
        if self.head_binding.sequence_length != self.sequence_length:
            raise V2StateError("head_binding declares a different sequence length")
        head_payload = _head_score_payload(self.head_score, self.head_binding)
        if self.feasibility_level is FeasibilityLevel.DEFINITIVE and head_payload is None:
            raise V2StateError("a definitive endpoint requires complete Head evidence")
        if head_payload is not None and head_payload["global_risk"] != self.head_global_risk:
            raise V2StateError("head_global_risk does not match head_score.global_risk")

        outcome = self.structure_outcome
        structure_payload = _structure_outcome_payload(outcome)
        if self.feasibility_level is FeasibilityLevel.DEFINITIVE:
            if outcome is None or not getattr(outcome, "evaluated", False):
                raise V2StateError(
                    "a definitive endpoint requires an evaluated structure outcome; a deferred or "
                    "absent result may never purchase feedback ancestry (PLAN §2.7, §4.2)"
                )
            if getattr(outcome, "feasible", None) is not True:
                raise V2StateError(
                    "a definitive endpoint requires a passing structure verdict"
                )
        elif outcome is not None and getattr(outcome, "evaluated", False):
            raise V2StateError(
                f"feasibility_level is {self.feasibility_level.value!r} but the structure outcome "
                "was evaluated; an evaluated result must be recorded at its true level"
            )

        del head_payload, structure_payload
        expected_endpoint_id = make_endpoint_id(
            self.source_state_id, fork_index=self.fork_index,
            sequence_md5_hex=self.sequence_md5,
        )
        if self.endpoint_id is None:
            object.__setattr__(self, "endpoint_id", expected_endpoint_id)
        elif self.endpoint_id != expected_endpoint_id:
            raise V2StateError(
                f"endpoint_id {self.endpoint_id!r} does not equal the source/fork/sequence-derived "
                f"identity {expected_endpoint_id!r}"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": V2_STATE_SCHEMA_VERSION,
            "layer": "endpoint",
            "lineage": _lineage_payload(self.lineage),
            "protein_id": self.protein_id,
            "sequence": self.sequence,
            "sequence_md5": self.sequence_md5,
            "sequence_length": self.sequence_length,
            "source_state_id": self.source_state_id,
            "source_state_content_digest": self.source_state_content_digest,
            "fork_index": self.fork_index,
            "fork_seed": self.fork_seed,
            "replay": self.replay.canonical_payload(),
            "evidence": [e.canonical_payload() for e in self.endpoint_provenance_evidence_by_pos],
            "head_binding": self.head_binding.canonical_payload(),
            "head_score": _head_score_payload(self.head_score, self.head_binding),
            "head_global_risk": self.head_global_risk,
            "feasibility_level": self.feasibility_level.value,
            "structure_outcome": _structure_outcome_payload(self.structure_outcome),
            "cost_event_ids": list(self.cost_event_ids),
        }

    @property
    def content_digest(self) -> str:
        return canonical_digest(self.canonical_payload())


def endpoint_may_become_ancestry(endpoint: CompleteEndpoint) -> bool:
    """Only a definitively feasible endpoint may be selected for feedback (PLAN §2.7, §4.2)."""
    return endpoint.feasibility_level is FeasibilityLevel.DEFINITIVE


@dataclass(frozen=True)
class ArchiveMembership:
    is_elite: bool
    elite_rank: int | None
    is_diversity_frontier: bool

    def __post_init__(self) -> None:
        for name in ("is_elite", "is_diversity_frontier"):
            if not isinstance(getattr(self, name), bool):
                raise V2StateError(f"{name} must be an explicit bool")
        if self.is_elite and self.elite_rank is None:
            raise V2StateError("an elite member must carry a rank")
        if self.elite_rank is not None:
            _require_index(self.elite_rank, "elite_rank")


@dataclass(frozen=True)
class ArchiveEntry:
    """Layer 3 -- one archive row over an exact endpoint identity.

    The entry *is* the endpoint identity (PLAN §2.3), so no second content is hashed and no second
    key namespace is created. ``family_id`` is read from the lineage rather than duplicated.
    """

    endpoint_id: str
    endpoint_content_digest: str
    lineage: LineageRef
    sequence_equivalence_key: str
    feasibility_level: FeasibilityLevel
    first_depth_seen: int
    last_depth_seen: int
    membership: ArchiveMembership

    def __post_init__(self) -> None:
        require_id_namespace(self.endpoint_id, "endpoint")
        require_digest(self.endpoint_content_digest, "endpoint_content_digest")
        if not isinstance(self.lineage, LineageRef):
            raise V2StateError("lineage must be a LineageRef")
        if not isinstance(self.feasibility_level, FeasibilityLevel):
            raise V2StateError("feasibility_level must be a FeasibilityLevel")
        if not isinstance(self.membership, ArchiveMembership):
            raise V2StateError("membership must be an ArchiveMembership")
        _require_text(self.sequence_equivalence_key, "sequence_equivalence_key")
        _require_index(self.first_depth_seen, "first_depth_seen")
        _require_index(self.last_depth_seen, "last_depth_seen")
        if self.last_depth_seen < self.first_depth_seen:
            raise V2StateError("last_depth_seen precedes first_depth_seen")

    @property
    def entry_id(self) -> str:
        return make_archive_entry_id(self.endpoint_id)

    @property
    def family_id(self) -> str:
        return self.lineage.family_id


def advance_archive_entry(
    entry: ArchiveEntry, *, feasibility_level: FeasibilityLevel, depth: int,
) -> ArchiveEntry:
    """Upgrade an archive row in place-by-value, never downgrade it.

    PLAN §2.3's monotonicity is at the level of exact endpoint identity: an ``unvalidated`` row may
    become ``definitive`` once structure runs, but a later depth can never demote what was already
    established, and ``last_depth_seen`` only moves forward.
    """
    if not isinstance(feasibility_level, FeasibilityLevel):
        raise V2StateError("feasibility_level must be a FeasibilityLevel")
    _require_index(depth, "depth")
    if _FEASIBILITY_ORDER[feasibility_level] < _FEASIBILITY_ORDER[entry.feasibility_level]:
        raise V2StateError(
            f"cannot demote {entry.endpoint_id} from {entry.feasibility_level.value!r} to "
            f"{feasibility_level.value!r}; archive feasibility only advances (PLAN §2.3)"
        )
    if depth < entry.last_depth_seen:
        raise V2StateError(
            f"depth {depth} precedes last_depth_seen {entry.last_depth_seen}"
        )
    return ArchiveEntry(
        endpoint_id=entry.endpoint_id, endpoint_content_digest=entry.endpoint_content_digest,
        lineage=entry.lineage, sequence_equivalence_key=entry.sequence_equivalence_key,
        feasibility_level=feasibility_level, first_depth_seen=entry.first_depth_seen,
        last_depth_seen=depth, membership=entry.membership,
    )


@dataclass(frozen=True)
class AssimilationRecord:
    """How many injected tokens were pending at re-entry and how many the first forward scored."""

    n_pending_at_reentry: int
    n_assimilated: int
    assimilation_temperature: float

    def __post_init__(self) -> None:
        _require_index(self.n_pending_at_reentry, "n_pending_at_reentry")
        _require_index(self.n_assimilated, "n_assimilated")
        _require_finite(self.assimilation_temperature, "assimilation_temperature")
        if self.assimilation_temperature <= 0.0:
            raise V2StateError("assimilation_temperature must be positive")
        if self.n_assimilated != self.n_pending_at_reentry:
            raise V2StateError(
                f"{self.n_pending_at_reentry} tokens were pending at re-entry but "
                f"{self.n_assimilated} were assimilated; every injected non-anchor token must be "
                "assimilated before any active-score consumer reads it (PLAN §3.1)"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "n_pending_at_reentry": self.n_pending_at_reentry,
            "n_assimilated": self.n_assimilated,
            "assimilation_temperature": self.assimilation_temperature,
        }


@dataclass(frozen=True)
class FeedbackTransition:
    """Layer 5 -- the replayable feedback event ``G_d``.

    ``source_depth`` is the depth of the *source*; ``lineage`` is the *descendant's*, so
    ``lineage.depth == source_depth + 1``. A non-committed outcome carries no projected or
    propagated state: PLAN §2.4 requires an explicit null/stalled event rather than an invented one.

    ``matched_seed_context_digest`` crosses the boundary as a plain string, so this module never
    imports the seed layer and no treatment content can travel back into a matched seed.
    """

    schema_version: str
    lineage: LineageRef
    source_depth: int
    coordinate_law: CoordinateLaw
    r_step: int
    c_step: int
    c_next_step: int
    source_state_id: str
    source_state_content_digest: str
    selected_endpoint_id: str | None
    endpoint_sequence_md5: str | None
    endpoint_content_digest: str | None
    policy: ProjectionPolicyIdentity
    support: SupportPartition | None
    projected_state_id: str | None
    projected_state_content_digest: str | None
    propagated_state_id: str | None
    propagated_state_content_digest: str | None
    descendant_fork_seed: int | None
    fork_index: int | None
    pair_id: str | None
    matched_seed_context_digest: str | None
    assimilation: AssimilationRecord | None
    segment_logical_dfe: int
    background_remask_events: int
    cost_event_ids: tuple[str, ...]
    parent_transition_id: str | None
    outcome: TransitionOutcome
    transition_id: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != V2_STATE_SCHEMA_VERSION:
            raise V2StateError(f"schema_version must be {V2_STATE_SCHEMA_VERSION!r}")
        if not isinstance(self.lineage, LineageRef):
            raise V2StateError("lineage must be a LineageRef")
        if not isinstance(self.outcome, TransitionOutcome):
            raise V2StateError("outcome must be a TransitionOutcome")
        if not isinstance(self.policy, ProjectionPolicyIdentity):
            raise V2StateError("policy must be a ProjectionPolicyIdentity")
        if not isinstance(self.coordinate_law, CoordinateLaw):
            raise V2StateError("coordinate_law must be a CoordinateLaw")
        _require_index(self.source_depth, "source_depth")
        _require_index(self.r_step, "r_step")
        _require_index(self.c_step, "c_step")
        _require_index(self.c_next_step, "c_next_step")
        _require_index(self.segment_logical_dfe, "segment_logical_dfe")
        _require_index(self.background_remask_events, "background_remask_events")
        require_id_namespace(self.source_state_id, "live")
        require_digest(self.source_state_content_digest, "source_state_content_digest")
        expected_source_state_id = make_live_state_id(
            self.lineage.protein_id, self.lineage.family_id,
            depth=self.source_depth, step=self.c_step,
            content_digest=self.source_state_content_digest,
        )
        if self.source_state_id != expected_source_state_id:
            raise V2StateError(
                f"source_state_id {self.source_state_id!r} does not equal the exact "
                f"lineage/c-step/content-digest-derived identity {expected_source_state_id!r}"
            )
        for event_id in self.cost_event_ids:
            _require_text(event_id, "cost_event_id")
        if self.lineage.depth != self.source_depth + 1:
            raise V2StateError(
                f"descendant lineage depth {self.lineage.depth} != source_depth + 1 "
                f"({self.source_depth + 1}); the transition record carries the descendant's lineage"
            )
        if self.r_step >= self.c_step:
            raise V2StateError(f"r_step={self.r_step} must precede c_step={self.c_step}")
        if self.c_next_step < self.c_step:
            raise V2StateError("c_next_step precedes c_step")
        if self.background_remask_events != 0:
            raise V2StateError(
                "a V2/A2 transition records zero background-remask events by contract"
            )
        if self.source_depth == 0:
            if self.parent_transition_id is not None:
                raise V2StateError("a depth-0 source has no parent transition")
        else:
            if self.parent_transition_id is None:
                raise V2StateError("a recurrent source must name its parent transition")
            require_id_namespace(self.parent_transition_id, "txn")

        committed = self.outcome is TransitionOutcome.COMMITTED
        committed_fields = {
            "selected_endpoint_id": self.selected_endpoint_id,
            "endpoint_sequence_md5": self.endpoint_sequence_md5,
            "endpoint_content_digest": self.endpoint_content_digest,
            "support": self.support,
            "projected_state_id": self.projected_state_id,
            "projected_state_content_digest": self.projected_state_content_digest,
            "propagated_state_id": self.propagated_state_id,
            "propagated_state_content_digest": self.propagated_state_content_digest,
            "descendant_fork_seed": self.descendant_fork_seed,
            "fork_index": self.fork_index,
            "pair_id": self.pair_id,
            "matched_seed_context_digest": self.matched_seed_context_digest,
            "assimilation": self.assimilation,
        }
        if committed:
            missing = sorted(name for name, value in committed_fields.items() if value is None)
            if missing:
                raise V2StateError(f"a committed transition is missing {missing}")
            if self.segment_logical_dfe != self.c_next_step - self.r_step:
                raise V2StateError(
                    f"segment_logical_dfe={self.segment_logical_dfe} != c_next_step - r_step = "
                    f"{self.c_next_step - self.r_step} (PLAN §3.4)"
                )
            if not isinstance(self.support, SupportPartition):
                raise V2StateError("a committed transition support must be a SupportPartition")
            if not isinstance(self.assimilation, AssimilationRecord):
                raise V2StateError(
                    "a committed transition assimilation must be an AssimilationRecord"
                )
            require_id_namespace(self.projected_state_id, "proj")
            require_id_namespace(self.propagated_state_id, "live")
            require_id_namespace(self.selected_endpoint_id, "endpoint")
            require_digest(self.endpoint_content_digest, "endpoint_content_digest")
            require_digest(
                self.projected_state_content_digest, "projected_state_content_digest"
            )
            require_digest(
                self.propagated_state_content_digest, "propagated_state_content_digest"
            )
            require_digest(self.matched_seed_context_digest, "matched_seed_context_digest")
            _require_md5(self.endpoint_sequence_md5, "endpoint_sequence_md5")
            _require_index(self.descendant_fork_seed, "descendant_fork_seed")
            _require_index(self.fork_index, "fork_index")
            _require_text(self.pair_id, "pair_id")
            expected_projected_state_id = make_projected_state_id(
                self.lineage.protein_id, self.lineage.family_id,
                depth=self.lineage.depth, r_step=self.r_step,
                content_digest=self.projected_state_content_digest,
            )
            if self.projected_state_id != expected_projected_state_id:
                raise V2StateError(
                    f"projected_state_id {self.projected_state_id!r} does not equal the exact "
                    "lineage/re-entry/content-digest-derived identity "
                    f"{expected_projected_state_id!r}"
                )
            expected_propagated_state_id = make_live_state_id(
                self.lineage.protein_id, self.lineage.family_id,
                depth=self.lineage.depth, step=self.c_next_step,
                content_digest=self.propagated_state_content_digest,
            )
            if self.propagated_state_id != expected_propagated_state_id:
                raise V2StateError(
                    f"propagated_state_id {self.propagated_state_id!r} does not equal the exact "
                    "lineage/c-next/content-digest-derived identity "
                    f"{expected_propagated_state_id!r}"
                )
            expected_endpoint_id = make_endpoint_id(
                self.source_state_id, fork_index=self.fork_index,
                sequence_md5_hex=self.endpoint_sequence_md5,
            )
            if self.selected_endpoint_id != expected_endpoint_id:
                raise V2StateError(
                    "selected endpoint does not bind this transition's source, fork, and sequence"
                )
            if self.lineage.parent_state_id != self.projected_state_id:
                raise V2StateError(
                    "descendant lineage parent_state_id does not name the projected state"
                )
            if self.lineage.origin_endpoint_id != self.selected_endpoint_id:
                raise V2StateError(
                    "descendant lineage origin_endpoint_id does not name the selected endpoint"
                )
            if self.assimilation.n_pending_at_reentry != len(self.support.injected_positions):
                raise V2StateError(
                    "assimilation pending count does not match the injected support cardinality"
                )
        else:
            present = sorted(name for name, value in committed_fields.items() if value is not None)
            if present:
                raise V2StateError(
                    f"outcome {self.outcome.value!r} is a null/stalled event but still claims "
                    f"{present}; a refused transition records the refusal rather than an invented "
                    "state (PLAN §2.4)"
                )
            if self.segment_logical_dfe != 0:
                raise V2StateError("a null/stalled transition ran no segment and charges no DFE")

        digest = self.content_digest
        if self.transition_id is None:
            object.__setattr__(self, "transition_id", make_transition_id(
                self.lineage.protein_id, self.lineage.family_id, depth=self.source_depth,
                r_step=self.r_step, c_next_step=self.c_next_step, content_digest=digest,
            ))
        else:
            expected_transition_id = make_transition_id(
                self.lineage.protein_id, self.lineage.family_id, depth=self.source_depth,
                r_step=self.r_step, c_next_step=self.c_next_step, content_digest=digest,
            )
            if self.transition_id != expected_transition_id:
                raise V2StateError(
                    f"transition_id {self.transition_id!r} does not equal the exact "
                    f"lineage/coordinate/content-digest-derived identity {expected_transition_id!r}"
                )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": V2_STATE_SCHEMA_VERSION,
            "layer": "transition",
            "lineage": _lineage_payload(self.lineage),
            "source_depth": self.source_depth,
            "coordinate_law": self.coordinate_law.value,
            "r_step": self.r_step,
            "c_step": self.c_step,
            "c_next_step": self.c_next_step,
            "source_state_id": self.source_state_id,
            "source_state_content_digest": self.source_state_content_digest,
            "selected_endpoint_id": self.selected_endpoint_id,
            "endpoint_sequence_md5": self.endpoint_sequence_md5,
            "endpoint_content_digest": self.endpoint_content_digest,
            "policy": self.policy.canonical_payload(),
            "support": None if self.support is None else self.support.canonical_payload(),
            "projected_state_id": self.projected_state_id,
            "projected_state_content_digest": self.projected_state_content_digest,
            "propagated_state_id": self.propagated_state_id,
            "propagated_state_content_digest": self.propagated_state_content_digest,
            "descendant_fork_seed": self.descendant_fork_seed,
            "fork_index": self.fork_index,
            "pair_id": self.pair_id,
            "matched_seed_context_digest": self.matched_seed_context_digest,
            "assimilation": (
                None if self.assimilation is None else self.assimilation.canonical_payload()
            ),
            "segment_logical_dfe": self.segment_logical_dfe,
            "background_remask_events": self.background_remask_events,
            "cost_event_ids": list(self.cost_event_ids),
            "parent_transition_id": self.parent_transition_id,
            "outcome": self.outcome.value,
        }

    @property
    def content_digest(self) -> str:
        return canonical_digest(self.canonical_payload())
