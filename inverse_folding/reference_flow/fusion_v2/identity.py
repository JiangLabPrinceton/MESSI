"""V2 identifiers and content digests.

Every V2 identifier is minted from typed, pre-declared fields. Nothing here derives an identity
from Python ``hash()``, row order, sequence similarity, or a score coincidence -- PLAN §2.11's
"do not reconstruct the mapping from sequence alone" applied to the whole V2 state graph.

This is the dependency leaf of ``fusion_v2`` above ``errors``: it imports stdlib and frozen v0/V1
primitives only, never ``fusion_v2.state``, so ``state.py`` can import it without a cycle.

Digest policy (interface map Conflict 14): ``sequence_md5`` is the project-wide *sequence* identity
and the Head join key; SHA-256 is used for file bytes and for canonical-JSON record digests. A
second sequence digest would create two keys for one object.

h-maps are retired. ``make_v2_conditioning`` fixes the controller-free / h-map-free declaration
internally, so no V2 config surface, artifact, or call site carries an h-map field. The underlying
``ConditioningDigest`` guard still runs -- the guarantee is kept, the declaration burden is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from inverse_folding.reference_flow.fusion.v1_records import (
    ConditioningDigest,
    canonical_json_bytes,
    content_digest,
    make_root_id,
    sha256_hex,
)

from .errors import V2Error

__all__ = [
    "V2_STATE_SCHEMA_VERSION", "ID_NAMESPACES", "CONTENT_ROLE_TO_FIELD", "FROZEN_DIGEST_ROLES",
    "OWN_CONDITIONING_FIELDS", "BASE_CONDITIONING_FIELDS", "V2IdentityError",
    "canonical_json_bytes", "sha256_hex", "content_digest", "make_root_id", "canonical_digest",
    "require_digest", "LineageRef", "make_v2_conditioning", "V2ConditioningIdentity",
    "WindowRiskLike", "HeadScoreLike", "HeadEvaluatorIdentity", "HeadScoreBinding",
    "window_grid_digest", "SafetyReferenceBinding", "ProjectionPolicyIdentity",
    "make_live_state_id", "make_projected_state_id", "make_endpoint_id", "make_transition_id",
    "make_archive_entry_id", "id_namespace", "require_id_namespace",
    "require_id_namespace_in", "assert_id_binds_digest",
    "assert_no_id_collisions",
]

#: Bumped whenever any canonical payload projection changes, so an old digest can never silently
#: validate against a new schema. Mirrors ``v1_records.SCHEMA_VERSION``.
V2_STATE_SCHEMA_VERSION = "v2state-1"

#: The namespace token embedded in every minted ID, one per typed state layer (PLAN §2.2).
ID_NAMESPACES = frozenset({"live", "proj", "endpoint", "txn", "archive"})

#: Length of the content-digest suffix carried by an ID. Long enough to detect a payload edit,
#: short enough to keep IDs readable in a parquet column.
_ID_DIGEST_PREFIX = 12

#: The identifier delimiter. Leaf components may never contain it.
_ID_DELIMITER = ":"

#: The project's spelling of "this digest is absent", matched case-insensitively.
_PLACEHOLDER_PREFIX = "unset:"


class V2IdentityError(V2Error):
    """An identifier or content digest is missing, malformed, or bound to the wrong content."""


# --------------------------------------------------------------------------------------------
# Digests
# --------------------------------------------------------------------------------------------

def canonical_digest(payload: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical-JSON projection of ``payload``.

    The only digest constructor other ``fusion_v2`` modules may call, so every V2 record digest
    shares one encoder with V1. A second encoder with different ``sort_keys``/``separators`` would
    digest the same object to different bytes.
    """
    return sha256_hex(canonical_json_bytes(payload))


def require_digest(value: Any, name: str) -> str:
    """Return ``value`` if it is a real digest; raise otherwise.

    Fails closed on the empty string and on any ``unset:`` placeholder, applying the V1 law of
    ``v1_records.ConditioningDigest.__post_init__`` to every V2 digest field (PLAN §5.2, "Missing
    content identity fails closed"). Deliberately does not require 64 hex characters: a git
    revision is a legal identity here and is not a SHA-256.
    """
    if isinstance(value, bool) or not isinstance(value, str):
        raise V2IdentityError(f"digest {name!r} must be a str, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise V2IdentityError(f"digest {name!r} must be non-empty")
    if stripped.lower().startswith(_PLACEHOLDER_PREFIX):
        raise V2IdentityError(
            f"digest {name!r} is a placeholder ({value!r}); every digest must come from real "
            "content (PLAN §5.2)"
        )
    return value


def _require_id_token(value: Any, name: str) -> str:
    """A leaf component of an identifier: non-empty, delimiter-free, not a namespace token.

    Without this, ``(protein_id, family_id, depth, step, digest) -> id`` is not injective:
    ``5ZHV:B``/``fam0`` and ``5ZHV``/``B:fam0`` mint one identifier and merge two lineages in every
    downstream join. A component equal to ``archive`` could additionally inject the archive prefix
    that :func:`id_namespace` resolves positionally.
    """
    text = _require_text(value, name)
    if _ID_DELIMITER in text:
        raise V2IdentityError(
            f"{name} may not contain {_ID_DELIMITER!r} ({text!r}); it is the identifier delimiter "
            "and would make two distinct lineages mint one identifier"
        )
    if text in ID_NAMESPACES:
        raise V2IdentityError(
            f"{name} may not be the namespace token {text!r}; it would alias a state layer"
        )
    return text


def _require_text(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value:
        raise V2IdentityError(f"{name} must be a non-empty str, got {value!r}")
    return value


def _require_index(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise V2IdentityError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise V2IdentityError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_md5(value: Any, name: str) -> str:
    text = _require_text(value, name)
    if len(text) != 32 or any(c not in "0123456789abcdef" for c in text):
        raise V2IdentityError(f"{name} must be a 32-char lowercase md5 hex digest, got {text!r}")
    return text


# --------------------------------------------------------------------------------------------
# Lineage
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LineageRef:
    """Pure lineage identity carried by every V2 layer. No payload, no scores.

    ``family_id`` is a *declared* input. ``fusion_v2`` never derives it from sequence, row order,
    or score -- what declares a family is an open decision (interface map OQ7), and guessing it
    here would let sibling multiplicity buy ancestry mass (PLAN §2.3).
    """

    protein_id: str
    root_id: str
    family_id: str
    depth: int
    parent_state_id: str | None
    parent_transition_id: str | None
    origin_endpoint_id: str | None

    def __post_init__(self) -> None:
        _require_text(self.protein_id, "protein_id")
        _require_text(self.root_id, "root_id")
        _require_text(self.family_id, "family_id")
        _require_index(self.depth, "depth")
        edges = {
            "parent_state_id": self.parent_state_id,
            "parent_transition_id": self.parent_transition_id,
            "origin_endpoint_id": self.origin_endpoint_id,
        }
        if self.depth == 0:
            present = sorted(n for n, v in edges.items() if v is not None)
            if present:
                raise V2IdentityError(
                    f"depth-0 lineage must have no parent edges, got {present}; a depth-0 root "
                    "descends from nothing (PLAN §2.2)"
                )
            return
        missing = sorted(n for n, v in edges.items() if v is None)
        if missing:
            raise V2IdentityError(
                f"depth-{self.depth} lineage is missing parent edges {missing}; every state and "
                "event edge must be replayable without sequence guessing (PLAN §5.3)"
            )
        # Each edge is layer-typed. Without this any layer could occupy any slot and all three
        # edges could be one string, which would make the lineage graph unreplayable while still
        # looking well-formed. A parent state may be either a committed live state or the
        # projected state that preceded it; the map does not narrow that one further.
        require_id_namespace_in(self.parent_state_id, ("live", "proj"), "parent_state_id")
        require_id_namespace_in(self.parent_transition_id, ("txn",), "parent_transition_id")
        require_id_namespace_in(self.origin_endpoint_id, ("endpoint",), "origin_endpoint_id")


# --------------------------------------------------------------------------------------------
# Content provenance
# --------------------------------------------------------------------------------------------

#: Fields ``V2ConditioningIdentity`` owns directly (the ``base`` bundle supplies the rest).
OWN_CONDITIONING_FIELDS: tuple[str, ...] = (
    "cohort_table",
    "reference_sequences",
    "constraint_manifest",
    "rf_sampler_config",
    "head_config",
    "head_checkpoint",
    "structure_backend",
    "structure_config",
    "v0_structure_gate_config",
    "projection_policy_spec",
    "schedule_band_calibration",
    "complete_reference",
    "code_revision",
)

#: Digest fields supplied by the reused V1 ``ConditioningDigest``.
BASE_CONDITIONING_FIELDS: tuple[str, ...] = (
    "base.dplm_checkpoint",
    "base.tokenizer",
    "base.backbone_row",
    "base.coordinate_mask",
    "base.entry_config",
    "base.fixed_token_policy",
)

#: PLAN §5.2 content roles -> the identity field each one lands in. PLAN says "at least", so
#: additions are legal and omissions are not. There is deliberately no h-map role: V2 consumes no
#: h-map, so there is nothing to hash and no placeholder to fake.
CONTENT_ROLE_TO_FIELD: Mapping[str, str] = {
    "cohort_table": "cohort_table",
    "reference_sequences": "reference_sequences",
    "backbone": "base.backbone_row",
    "coordinate_mask": "base.coordinate_mask",
    "constraint_manifest": "constraint_manifest",
    "rf_sampler_config": "rf_sampler_config",
    "tokenizer": "base.tokenizer",
    "dplm_checkpoint": "base.dplm_checkpoint",
    "fixed_token_policy": "base.fixed_token_policy",
    "head_config": "head_config",
    "head_checkpoint": "head_checkpoint",
    "structure_backend": "structure_backend",
    "structure_config": "structure_config",
    "v0_structure_gate_config": "v0_structure_gate_config",
    "projection_policy_spec": "projection_policy_spec",
    "schedule_band_calibration": "schedule_band_calibration",
    "complete_reference_sequence": "complete_reference",
    "code_revision": "code_revision",
}

#: Roles whose digest must be declared in the config itself, because PLAN §2.1/§2.5/§2.7 all
#: require the content to be bound *before* runtime rather than observed during it.
FROZEN_DIGEST_ROLES = frozenset({
    "projection_policy_spec", "schedule_band_calibration", "complete_reference_sequence",
})


def make_v2_conditioning(
    *,
    dplm_checkpoint: str,
    tokenizer: str,
    backbone_row: str,
    coordinate_mask: str,
    entry_config: str,
    fixed_token_policy: str,
) -> ConditioningDigest:
    """Build the reused V1 conditioning bundle for the frozen V2 substrate.

    ``controller_enabled`` and ``h_maps_present`` are fixed to ``False`` here and are **not**
    parameters: h-maps are retired and the controller-free substrate is frozen (PLAN §5.1), so a
    caller has nothing to decide. Passing either name is a ``TypeError``, which is the point --
    the guarantee is structural, not a convention. ``ConditioningDigest.__post_init__`` still
    enforces it, so the assertion survives even if this helper is bypassed.

    ``entry_config`` carries the **V2** config digest. There is deliberately no second field for it
    on :class:`V2ConditioningIdentity`, so the two cannot drift apart.
    """
    return ConditioningDigest(
        dplm_checkpoint=require_digest(dplm_checkpoint, "dplm_checkpoint"),
        tokenizer=require_digest(tokenizer, "tokenizer"),
        backbone_row=require_digest(backbone_row, "backbone_row"),
        coordinate_mask=require_digest(coordinate_mask, "coordinate_mask"),
        entry_config=require_digest(entry_config, "entry_config"),
        fixed_token_policy=require_digest(fixed_token_policy, "fixed_token_policy"),
        controller_enabled=False,
        h_maps_present=False,
    )


@dataclass(frozen=True)
class V2ConditioningIdentity:
    """The PLAN §5.2 content-provenance bundle: all digests, no paths.

    Stored **by value** on every persisted state row rather than as a manifest pointer, because
    PLAN §5.4 requires resume to validate every fragment before aggregation and a pointer-only
    fragment cannot be validated standalone (interface map Conflict 12). Precedent:
    ``PartialRootPayload.conditioning`` is by value.
    """

    base: ConditioningDigest
    cohort_table: str
    reference_sequences: str
    constraint_manifest: str
    rf_sampler_config: str
    head_config: str
    head_checkpoint: str
    structure_backend: str
    structure_config: str
    v0_structure_gate_config: str
    projection_policy_spec: str
    schedule_band_calibration: str
    complete_reference: str
    code_revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.base, ConditioningDigest):
            raise V2IdentityError("base must be a ConditioningDigest")
        if self.base.controller_enabled or self.base.h_maps_present:
            raise V2IdentityError(
                "the frozen V2 substrate is controller-free and h-map-free (PLAN §5.1)"
            )
        for name in OWN_CONDITIONING_FIELDS:
            require_digest(getattr(self, name), name)

    def canonical_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"schema": V2_STATE_SCHEMA_VERSION}
        payload.update({name: getattr(self, name) for name in OWN_CONDITIONING_FIELDS})
        for dotted in BASE_CONDITIONING_FIELDS:
            payload[dotted] = getattr(self.base, dotted.split(".", 1)[1])
        return payload

    def digest(self) -> str:
        return canonical_digest(self.canonical_payload())


# --------------------------------------------------------------------------------------------
# Head evidence identity
# --------------------------------------------------------------------------------------------

@runtime_checkable
class WindowRiskLike(Protocol):
    """Structural view of ``head_scoring.WindowRiskRecord`` -- duck-typed so ``fusion_v2`` never
    imports the scorer, which imports torch."""

    start_0b: int
    end_0b: int
    k: int
    z: float


@runtime_checkable
class HeadScoreLike(Protocol):
    """Structural view of ``head_scoring.HeadScore``. Same torch-avoidance convention as
    ``fusion/state.py`` and ``fusion/objective.py``."""

    protein_id: str
    sequence_md5: str
    sequence_length: int
    allele: str
    score_scale: str
    windows: tuple[Any, ...]


@dataclass(frozen=True)
class HeadEvaluatorIdentity:
    """Identity of the Head *evaluator* -- what scored, not what was scored.

    Kept separate from :class:`HeadScoreBinding` so that "the two Head identities differ" is
    testable independently of "the two sequences differ" (PLAN §7.2). Mirrors the scorer's own
    required metadata keys.
    """

    allele: str
    score_scale: str
    window_k_min: int
    window_k_max: int
    head_config_hash: str
    head_checkpoint_digest: str

    def __post_init__(self) -> None:
        _require_text(self.allele, "allele")
        _require_text(self.score_scale, "score_scale")
        _require_index(self.window_k_min, "window_k_min", minimum=1)
        _require_index(self.window_k_max, "window_k_max", minimum=1)
        if self.window_k_max < self.window_k_min:
            raise V2IdentityError(
                f"window_k_max ({self.window_k_max}) < window_k_min ({self.window_k_min})"
            )
        require_digest(self.head_config_hash, "head_config_hash")
        require_digest(self.head_checkpoint_digest, "head_checkpoint_digest")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": V2_STATE_SCHEMA_VERSION,
            "allele": self.allele,
            "score_scale": self.score_scale,
            "window_k_min": self.window_k_min,
            "window_k_max": self.window_k_max,
            "head_config_hash": self.head_config_hash,
            "head_checkpoint_digest": self.head_checkpoint_digest,
        }

    def digest(self) -> str:
        return canonical_digest(self.canonical_payload())


def window_grid_digest(windows: Iterable[Any]) -> str:
    """Digest of the sorted ``(start_0b, end_0b, k)`` coordinate set of a Head score.

    Two Head scores may be compared only when this matches: ``objective.aligned_window_z`` raises
    unless the coordinate sets are identical, and the grid for a fixed k-range is determined by
    sequence length. Carrying the digest lets a mismatch be caught before the comparator runs.
    """
    coords = sorted(
        (_require_index(w.start_0b, "start_0b"), _require_index(w.end_0b, "end_0b"),
         _require_index(w.k, "k", minimum=1))
        for w in windows
    )
    if not coords:
        raise V2IdentityError("window grid is empty; a Head score must carry at least one window")
    if len(set(coords)) != len(coords):
        seen: set[tuple[int, int, int]] = set()
        dupes = sorted({c for c in coords if c in seen or seen.add(c)})  # type: ignore[func-returns-value]
        raise V2IdentityError(
            f"window grid carries duplicate coordinates {dupes}; the comparator this digest "
            "pre-checks keys a dict by coordinate and silently keeps the last row, so a duplicate "
            "would let row order decide which z survives and fabricate a per-window delta"
        )
    return canonical_digest({"schema": V2_STATE_SCHEMA_VERSION, "windows": coords})


@dataclass(frozen=True)
class HeadScoreBinding:
    """Identity of one *scored sequence*: which protein, which bytes, which grid, which evaluator."""

    protein_id: str
    sequence_md5: str
    sequence_length: int
    window_grid_digest: str
    evaluator: HeadEvaluatorIdentity

    def __post_init__(self) -> None:
        _require_text(self.protein_id, "protein_id")
        _require_md5(self.sequence_md5, "sequence_md5")
        _require_index(self.sequence_length, "sequence_length", minimum=1)
        require_digest(self.window_grid_digest, "window_grid_digest")
        if not isinstance(self.evaluator, HeadEvaluatorIdentity):
            raise V2IdentityError("evaluator must be a HeadEvaluatorIdentity")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": V2_STATE_SCHEMA_VERSION,
            "protein_id": self.protein_id,
            "sequence_md5": self.sequence_md5,
            "sequence_length": self.sequence_length,
            "window_grid_digest": self.window_grid_digest,
            "evaluator": self.evaluator.canonical_payload(),
        }

    def digest(self) -> str:
        return canonical_digest(self.canonical_payload())


@dataclass(frozen=True)
class SafetyReferenceBinding:
    """Identity of the immutable depth-0 whole-landscape safety reference ``ybar``.

    Content-bound before any depth-0 endpoint is scored and immutable thereafter (PLAN §2.7), so
    a lineage cannot re-base its reference each depth and thereby accept unbounded cumulative
    hotspot drift.

    ``sequence_length`` must equal the design's: ``objective.aligned_window_z`` raises on differing
    window-coordinate sets, and for a fixed k-range the grid is determined by length. A
    differing-length external reference is structurally inadmissible, whichever reference the open
    scientific decision eventually picks.
    """

    reference_id: str
    reference_label: str
    sequence_md5: str
    sequence_length: int
    reference_content_digest: str
    head_binding: HeadScoreBinding
    head_score_digest: str
    bound_at_depth: int
    source_kind: str

    def __post_init__(self) -> None:
        _require_text(self.reference_id, "reference_id")
        _require_text(self.reference_label, "reference_label")
        _require_text(self.source_kind, "source_kind")
        _require_md5(self.sequence_md5, "sequence_md5")
        _require_index(self.sequence_length, "sequence_length", minimum=1)
        require_digest(self.reference_content_digest, "reference_content_digest")
        require_digest(self.head_score_digest, "head_score_digest")
        if not isinstance(self.head_binding, HeadScoreBinding):
            raise V2IdentityError("head_binding must be a HeadScoreBinding")
        _require_index(self.bound_at_depth, "bound_at_depth")
        if self.bound_at_depth != 0:
            raise V2IdentityError(
                f"bound_at_depth must be 0, got {self.bound_at_depth!r}; the cumulative safety "
                "reference is bound once at depth 0 and never re-based (PLAN §2.7)"
            )
        if self.head_binding.sequence_md5 != self.sequence_md5:
            raise V2IdentityError(
                "head_binding.sequence_md5 does not match the reference sequence; the reference's "
                "Head evidence must be bound to the reference bytes"
            )
        if self.head_binding.sequence_length != self.sequence_length:
            raise V2IdentityError(
                "head_binding.sequence_length does not match the reference sequence_length"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": V2_STATE_SCHEMA_VERSION,
            "reference_id": self.reference_id,
            "reference_label": self.reference_label,
            "sequence_md5": self.sequence_md5,
            "sequence_length": self.sequence_length,
            "reference_content_digest": self.reference_content_digest,
            "head_binding": self.head_binding.canonical_payload(),
            "head_score_digest": self.head_score_digest,
            "bound_at_depth": self.bound_at_depth,
            "source_kind": self.source_kind,
        }

    def digest(self) -> str:
        return canonical_digest(self.canonical_payload())


@dataclass(frozen=True)
class ProjectionPolicyIdentity:
    """Identity of the feedback-support policy that produced a projection.

    ``is_diagnostic_only`` is a required explicit bool: ``explicit_probe`` is admissible for
    deterministic tests and the first state-transition diagnostic, and may never become a silent
    production default (PLAN §2.5). The phase gate that rejects it lives in ``config.py``.
    """

    policy_id: str
    policy_version: str
    policy_config_digest: str
    policy_spec_digest: str
    is_diagnostic_only: bool

    def __post_init__(self) -> None:
        _require_text(self.policy_id, "policy_id")
        _require_text(self.policy_version, "policy_version")
        require_digest(self.policy_config_digest, "policy_config_digest")
        require_digest(self.policy_spec_digest, "policy_spec_digest")
        if not isinstance(self.is_diagnostic_only, bool):
            raise V2IdentityError(
                "is_diagnostic_only must be an explicit bool; a coercible value is not a "
                "declaration (PLAN §2.5)"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": V2_STATE_SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_config_digest": self.policy_config_digest,
            "policy_spec_digest": self.policy_spec_digest,
            "is_diagnostic_only": self.is_diagnostic_only,
        }

    def digest(self) -> str:
        return canonical_digest(self.canonical_payload())


# --------------------------------------------------------------------------------------------
# Identifier minting
# --------------------------------------------------------------------------------------------

def _suffix(digest: str, name: str) -> str:
    """The 12-character content suffix spliced into an identifier.

    ``require_digest`` deliberately accepts non-SHA-256 identities -- a git revision is a legal
    conditioning digest -- so the constraint belongs here, on the ID grammar, not on the digest
    domain. A digest whose first 12 characters contain the delimiter would be parsed as extra
    segments: a projected-state ID minted from a ``live:``-prefixed digest reports namespace
    ``live`` and is accepted where a live-state ID is required. A digest shorter than 12 characters
    both weakens collision resistance and lets an unrelated short digest bind an ID minted from a
    long one.
    """
    value = require_digest(digest, name)
    if len(value) < _ID_DIGEST_PREFIX:
        raise V2IdentityError(
            f"{name} must be at least {_ID_DIGEST_PREFIX} characters to suffix an identifier, "
            f"got {len(value)} ({value!r})"
        )
    sliced = value[:_ID_DIGEST_PREFIX]
    if _ID_DELIMITER in sliced:
        raise V2IdentityError(
            f"{name} may not contain {_ID_DELIMITER!r} within its first {_ID_DIGEST_PREFIX} "
            f"characters ({value!r}); it would be parsed as an identifier segment and could alias "
            "a state-layer namespace"
        )
    return sliced


def make_live_state_id(
    protein_id: str, family_id: str, *, depth: int, step: int, content_digest: str
) -> str:
    """``{protein}:{family}:live:d{depth}:c{step}:{digest12}``.

    Depth and step are *both* embedded: under ``stationary_checkpoint`` recurrence two feedback
    depths legitimately share a sampler step, and they must not alias (PLAN §2.1, §2.4).
    """
    return (
        f"{_require_id_token(protein_id, 'protein_id')}:{_require_id_token(family_id, 'family_id')}"
        f":live:d{_require_index(depth, 'depth')}:c{_require_index(step, 'step')}"
        f":{_suffix(content_digest, 'content_digest')}"
    )


def make_projected_state_id(
    protein_id: str, family_id: str, *, depth: int, r_step: int, content_digest: str
) -> str:
    """``{protein}:{family}:proj:d{depth}:r{r_step}:{digest12}``."""
    return (
        f"{_require_id_token(protein_id, 'protein_id')}:{_require_id_token(family_id, 'family_id')}"
        f":proj:d{_require_index(depth, 'depth')}:r{_require_index(r_step, 'r_step')}"
        f":{_suffix(content_digest, 'content_digest')}"
    )


def make_endpoint_id(source_state_id: str, *, fork_index: int, sequence_md5_hex: str) -> str:
    """``{source_state_id}:endpoint:k{fork}:{md5_12}``.

    Keyed on the *source state* and the fork index, so two roots that converge to the same complete
    sequence remain distinct logical endpoints (PLAN §2.3).
    """
    return (
        f"{require_id_namespace(source_state_id, 'live')}"
        f":endpoint:k{_require_index(fork_index, 'fork_index')}"
        f":{_require_md5(sequence_md5_hex, 'sequence_md5_hex')[:_ID_DIGEST_PREFIX]}"
    )


def make_transition_id(
    protein_id: str, family_id: str, *, depth: int, r_step: int, c_next_step: int,
    content_digest: str,
) -> str:
    """``{protein}:{family}:txn:d{depth}:r{r}-c{c_next}:{digest12}``."""
    return (
        f"{_require_id_token(protein_id, 'protein_id')}:{_require_id_token(family_id, 'family_id')}"
        f":txn:d{_require_index(depth, 'depth')}"
        f":r{_require_index(r_step, 'r_step')}-c{_require_index(c_next_step, 'c_next_step')}"
        f":{_suffix(content_digest, 'content_digest')}"
    )


def make_archive_entry_id(endpoint_id: str) -> str:
    """``archive:{endpoint_id}``.

    The archive entry *is* the endpoint identity (PLAN §2.3), so no second content is hashed and no
    second key namespace is created.
    """
    return f"archive:{require_id_namespace(endpoint_id, 'endpoint')}"


def id_namespace(object_id: str) -> str:
    """Return the outermost namespace token of ``object_id``.

    An archive ID wraps an endpoint ID, which in turn embeds its source live-state ID, so the token
    is resolved outside-in: an ``archive:`` prefix wins, otherwise the rightmost token wins (the
    embedded parent's token is to its left).
    """
    _require_text(object_id, "object_id")
    if object_id.startswith("archive:"):
        return "archive"
    for part in reversed(object_id.split(":")):
        if part in ID_NAMESPACES and part != "archive":
            return part
    raise V2IdentityError(
        f"{object_id!r} carries no namespace token; expected one of {sorted(ID_NAMESPACES)}"
    )


def require_id_namespace(object_id: str, expected: str) -> str:
    """Return ``object_id`` if its namespace is ``expected``; raise otherwise.

    Stops a projected-state ID being accepted where a live-state ID belongs -- the five typed
    layers of PLAN §2.2 stay distinct at the identifier level, not only in the type system.
    """
    if expected not in ID_NAMESPACES:
        raise V2IdentityError(f"unknown namespace {expected!r}")
    actual = id_namespace(object_id)
    if actual != expected:
        raise V2IdentityError(
            f"expected a {expected!r} identifier, got a {actual!r} one ({object_id!r})"
        )
    return object_id


def require_id_namespace_in(
    object_id: Any, allowed: tuple[str, ...], name: str = "object_id"
) -> str:
    """Return ``object_id`` if its namespace is one of ``allowed``; raise otherwise.

    The multi-namespace form of :func:`require_id_namespace`, used where the contract legitimately
    admits more than one state layer.
    """
    _require_text(object_id, name)
    unknown = sorted(set(allowed) - ID_NAMESPACES)
    if unknown:
        raise V2IdentityError(f"unknown namespace(s) {unknown}")
    actual = id_namespace(object_id)
    if actual not in allowed:
        raise V2IdentityError(
            f"{name} must name one of {sorted(allowed)}, got a {actual!r} identifier "
            f"({object_id!r})"
        )
    return object_id


def assert_id_binds_digest(object_id: str, content_digest: str) -> None:
    """Fail unless ``object_id`` was minted for ``content_digest``.

    Recomputes the suffix, so a row whose ID survived an edit to its payload is caught before the
    row is trusted (PLAN §7.2 "origin, provenance, active-score/status, or active-history tampering
    passes integrity checks").
    """
    _require_text(object_id, "object_id")
    expected = _suffix(content_digest, "content_digest")
    actual = object_id.rsplit(":", 1)[-1]
    if actual != expected:
        raise V2IdentityError(
            f"{object_id!r} does not bind content digest {content_digest[:_ID_DIGEST_PREFIX]!r} "
            f"(carries {actual!r}); the identifier was minted for different content"
        )


def assert_no_id_collisions(ids: Mapping[str, str]) -> None:
    """Fail if two names map to the same identifier.

    Same law as ``v1_seeds.assert_no_seed_collisions``, retyped for string IDs, which the int-keyed
    original cannot express.
    """
    seen: dict[str, str] = {}
    for name, value in ids.items():
        _require_text(value, f"id[{name}]")
        if value in seen:
            raise V2IdentityError(
                f"identifier collision: {name!r} and {seen[value]!r} both map to {value!r}"
            )
        seen[value] = name
