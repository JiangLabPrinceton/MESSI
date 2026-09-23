"""V2 whole-landscape new-hotspot safety and the cumulative-reference ratchet.

This is a **new** contract, not v0's gate under another name. PLAN §2.7 forbids imitating it by
calling the v0 immediate-parent off-halo API with an empty halo, and three differences make that
prohibition load-bearing rather than stylistic:

* **window scope.** v0 measures only windows disjoint from the repair halo. V2 measures the whole
  landscape, so a hotspot created inside the region the policy just edited is visible.
* **reference recency.** v0 compares against the freshly re-scored immediate parent each round, so
  drift versus any fixed sequence accumulates unseen. V2 compares against a reference frozen at
  depth 0 and carried by *object identity* through every ``advance_lineage``, which is what makes
  the ratchet inescapable.
* **threshold provenance.** v0's ``objective.max_offtarget_window_increase`` (``0.10`` in the
  shipped configs) was calibrated for a strictly smaller window set. Reusing that number for a
  larger set would be a silently different gate, so it is never inherited here.

v0's comparator also falls back to ``0.0`` on an empty delta list -- it reports "no windows" as "no
new hotspot", i.e. it fails **open**. Every failure mode here fails closed instead, in a
deterministic order so an artifact naming a grid mismatch really is a grid problem.

The one primitive genuinely reused from v0 is the window-alignment law, imported through the public
aliases so the reuse is visible: two scores may be compared only when their coordinate sets are
identical.

Purity: no torch, no numpy, no I/O. Head scores arrive duck-typed.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Any, Sequence

from inverse_folding.reference_flow.fusion.objective import window_coord
from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion.v1_alloc import validate_complete_aa20

from .config import CalibratedScalar, V2Config
from .errors import V2Error
from .identity import (
    HeadEvaluatorIdentity,
    HeadScoreBinding,
    SafetyReferenceBinding,
    canonical_digest,
    require_digest,
    require_id_namespace,
    sha256_hex,
    window_grid_digest,
)

__all__ = [
    "V2SafetyError", "HeadIdentityMismatch", "ReferenceLengthMismatch", "WindowGridMismatch",
    "EmptyWindowGrid", "NonFiniteWindowRisk", "SelfReferentialReference", "ReferenceRebindAttempt",
    "ReferenceContentMismatch",
    "ReferenceKindMismatch", "UncalibratedThreshold", "NoImmediateParent", "LineageDepthSkew",
    "ReferenceKind", "WholeHotspotEvidence", "HotspotThreshold", "WholeHotspotVerdict",
    "IncrementalInapplicable", "incremental_inapplicable",
    "SafetyAdmissionPolicy",
    "CumulativeSafetyReference", "ImmediateParentReference", "LineageSafetyLedger",
    "AdmissibilityVerdict", "reference_sequence_content_digest",
    "whole_landscape_new_hotspot", "bind_cumulative_reference",
    "bind_admission_policy", "bind_immediate_parent", "make_admissibility_verdict",
    "open_lineage_ledger", "measure_cumulative", "measure_incremental", "apply_threshold",
    "advance_lineage", "assert_lineage_binding", "cumulative_incremental_slack",
]


class V2SafetyError(V2Error):
    """A whole-landscape safety contract was violated."""


class HeadIdentityMismatch(V2SafetyError):
    """Two scores were produced by different evaluators, or describe different proteins."""


class ReferenceLengthMismatch(V2SafetyError):
    """Design and reference are different lengths, so their window grids cannot align."""


class WindowGridMismatch(V2SafetyError):
    """The two coordinate sets differ; there is no exact alignment to compare."""


class EmptyWindowGrid(V2SafetyError):
    """A score carries no windows. Reporting 0.0 here would be a fail-open."""


class NonFiniteWindowRisk(V2SafetyError):
    """A window carries a non-finite z; a NaN would compare false against every threshold."""


class SelfReferentialReference(V2SafetyError):
    """A design was measured against itself; ``N_H(y; y) == 0`` is vacuous."""


class ReferenceRebindAttempt(V2SafetyError):
    """A lineage's cumulative verdicts do not all name its single frozen reference."""


class ReferenceContentMismatch(V2SafetyError):
    """The reference bytes are not the bytes the declared content digest identifies."""


class ReferenceKindMismatch(V2SafetyError):
    """A threshold calibrated for one reference scope was applied to the other."""


class UncalibratedThreshold(V2SafetyError):
    """A threshold arrived without real calibration provenance."""


class NoImmediateParent(V2SafetyError):
    """An incremental measurement was requested at depth 0, where no parent exists."""


class LineageDepthSkew(V2SafetyError):
    """A ledger advance skipped or repeated a depth."""


class ReferenceKind(str, enum.Enum):
    """Which reference an evidence record was measured against."""

    CUMULATIVE_DEPTH0 = "cumulative_depth0"
    IMMEDIATE_PARENT = "immediate_parent"


def _require_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V2SafetyError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise V2SafetyError(f"{name} must be finite, got {out!r}")
    return out


def _require_text(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise V2SafetyError(f"{name} must be a non-empty str, got {value!r}")
    return value


def _require_md5(value: Any, name: str) -> str:
    text = _require_text(value, name)
    if len(text) != 32 or any(char not in "0123456789abcdef" for char in text):
        raise HeadIdentityMismatch(f"{name} must be a 32-character lowercase md5 digest")
    return text


def _score_digest(score: Any, head_identity: HeadEvaluatorIdentity, label: str) -> str:
    """Validate and content-bind one exact Head score, including every window risk."""
    if getattr(score, "allele", None) != head_identity.allele:
        raise HeadIdentityMismatch(
            f"{label} was scored for allele {getattr(score, 'allele', None)!r}, evaluator declares "
            f"{head_identity.allele!r}"
        )
    if getattr(score, "score_scale", None) != head_identity.score_scale:
        raise HeadIdentityMismatch(
            f"{label} uses score scale {getattr(score, 'score_scale', None)!r}, evaluator declares "
            f"{head_identity.score_scale!r}"
        )
    protein_id = _require_text(getattr(score, "protein_id", None), f"{label} protein_id")
    sequence_digest = _require_md5(
        getattr(score, "sequence_md5", None), f"{label} sequence_md5")
    sequence_length = getattr(score, "sequence_length", None)
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) \
            or sequence_length < 1:
        raise ReferenceLengthMismatch(f"{label} sequence_length must be a positive int")
    windows = []
    seen = set()
    for window in tuple(getattr(score, "windows", ())):
        coord = window_coord(window)
        if coord in seen:
            raise WindowGridMismatch(f"{label} grid carries duplicate coordinate {coord}")
        seen.add(coord)
        if not head_identity.window_k_min <= coord[2] <= head_identity.window_k_max:
            raise HeadIdentityMismatch(
                f"{label} window k={coord[2]} is outside evaluator domain "
                f"[{head_identity.window_k_min}, {head_identity.window_k_max}]"
            )
        z = _require_finite(getattr(window, "z", None), f"{label} window {coord} z")
        windows.append((*coord, z))
    if not windows:
        raise EmptyWindowGrid(f"{label} Head score carries no windows")
    return canonical_digest({
        "protein_id": protein_id,
        "sequence_md5": sequence_digest,
        "sequence_length": sequence_length,
        "head_identity_digest": head_identity.digest(),
        "windows": sorted(windows),
    })


@dataclass(frozen=True, init=False)
class WholeHotspotEvidence:
    """A pure, endpoint-bound measurement of ``N_H_whole``."""

    endpoint_id: str
    max_increase: float
    positive_mass: float
    positive_count: int
    n_windows: int
    reference_kind: ReferenceKind
    reference_binding_id: str
    design_sequence_md5: str
    reference_sequence_md5: str
    head_identity_digest: str
    design_head_score_digest: str
    reference_head_score_digest: str

@dataclass(frozen=True)
class HotspotThreshold:
    """A V2 calibrated scalar narrowed to one runtime reference scope."""

    calibration: CalibratedScalar
    scope: ReferenceKind

    def __post_init__(self) -> None:
        if not isinstance(self.calibration, CalibratedScalar):
            raise UncalibratedThreshold("calibration must be a typed CalibratedScalar")
        if not isinstance(self.scope, ReferenceKind):
            raise UncalibratedThreshold("scope must be a ReferenceKind")
        if self.calibration.artifact.scope != self.scope.value:
            raise ReferenceKindMismatch(
                f"threshold artifact scope {self.calibration.artifact.scope!r} does not match "
                f"runtime scope {self.scope.value!r}"
            )

    @property
    def value(self) -> float:
        return float(self.calibration.value)

    @property
    def source_ref(self) -> str:
        return self.calibration.source_ref


@dataclass(frozen=True)
class WholeHotspotVerdict:
    """Evidence plus the threshold that judged it, and the resulting boolean."""

    evidence: WholeHotspotEvidence
    threshold: HotspotThreshold
    passed: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, WholeHotspotEvidence):
            raise V2SafetyError("evidence must be WholeHotspotEvidence")
        if not isinstance(self.threshold, HotspotThreshold):
            raise V2SafetyError("threshold must be HotspotThreshold")
        if self.threshold.scope is not self.evidence.reference_kind:
            raise ReferenceKindMismatch("threshold and evidence reference scopes differ")
        expected = self.evidence.max_increase <= self.threshold.value
        expected_reason = "within_threshold" if expected else "new_hotspot_exceeds_threshold"
        if self.passed is not expected or self.reason != expected_reason:
            raise V2SafetyError(
                "WholeHotspotVerdict disagrees with its measurement and threshold; passed and "
                "reason are derived facts"
            )


def _aligned_finite_z(design: Any, reference: Any) -> dict[tuple[int, int, int], tuple[float, float]]:
    """Align two window grids exactly, in a deterministic failure order.

    Order matters: a caller that sees ``WindowGridMismatch`` must be able to trust that the two
    scores really do have different coordinate sets, rather than a length problem surfacing late.
    """
    design_windows = tuple(design.windows)
    reference_windows = tuple(reference.windows)
    if not design_windows or not reference_windows:
        raise EmptyWindowGrid(
            "a Head score must carry at least one window; treating an empty grid as zero new "
            "hotspot would be a fail-open"
        )
    out: dict[tuple[int, int, int], tuple[float, float]] = {}
    design_map: dict[tuple[int, int, int], float] = {}
    for window in design_windows:
        coord = window_coord(window)
        z = window.z
        if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(float(z)):
            raise NonFiniteWindowRisk(f"design window {coord} carries non-finite z {z!r}")
        if coord in design_map:
            raise WindowGridMismatch(f"design grid carries duplicate coordinate {coord}")
        design_map[coord] = float(z)
    reference_map: dict[tuple[int, int, int], float] = {}
    for window in reference_windows:
        coord = window_coord(window)
        z = window.z
        if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(float(z)):
            raise NonFiniteWindowRisk(f"reference window {coord} carries non-finite z {z!r}")
        if coord in reference_map:
            raise WindowGridMismatch(f"reference grid carries duplicate coordinate {coord}")
        reference_map[coord] = float(z)
    if set(design_map) != set(reference_map):
        only_design = sorted(set(design_map) - set(reference_map))
        only_reference = sorted(set(reference_map) - set(design_map))
        raise WindowGridMismatch(
            f"window coordinate sets differ; design-only {only_design[:3]}, reference-only "
            f"{only_reference[:3]}"
        )
    for coord in sorted(design_map):
        out[coord] = (design_map[coord], reference_map[coord])
    return out


def whole_landscape_new_hotspot(
    design: Any,
    reference: Any,
    *,
    endpoint_id: str,
    head_identity: HeadEvaluatorIdentity,
    reference_kind: ReferenceKind,
    reference_binding_id: str,
) -> WholeHotspotEvidence:
    r"""``N_H^whole(y; ybar) = max_w [ z_w(y) - z_w(ybar) ]_+`` over **all** aligned windows.

    The positive part is taken per window, so a large improvement elsewhere can never offset a new
    hotspot -- the quantity is "how much worse did any window get", not a net.
    """
    if not isinstance(head_identity, HeadEvaluatorIdentity):
        raise HeadIdentityMismatch("head_identity must be a HeadEvaluatorIdentity")
    require_id_namespace(endpoint_id, "endpoint")
    if not isinstance(reference_kind, ReferenceKind):
        raise V2SafetyError("reference_kind must be a ReferenceKind")
    _require_text(reference_binding_id, "reference_binding_id")

    # 1. evaluator and protein identity
    for label, score in (("design", design), ("reference", reference)):
        if getattr(score, "allele", None) != head_identity.allele:
            raise HeadIdentityMismatch(
                f"{label} was scored for allele {getattr(score, 'allele', None)!r}, evaluator "
                f"declares {head_identity.allele!r}"
            )
        if getattr(score, "score_scale", None) != head_identity.score_scale:
            raise HeadIdentityMismatch(
                f"{label} uses score scale {getattr(score, 'score_scale', None)!r}, evaluator "
                f"declares {head_identity.score_scale!r}"
            )
    if getattr(design, "protein_id", None) != getattr(reference, "protein_id", None):
        raise HeadIdentityMismatch(
            f"design protein {getattr(design, 'protein_id', None)!r} != reference protein "
            f"{getattr(reference, 'protein_id', None)!r}"
        )
    # 2. sequence length
    if getattr(design, "sequence_length", None) != getattr(reference, "sequence_length", None):
        raise ReferenceLengthMismatch(
            f"design length {getattr(design, 'sequence_length', None)} != reference length "
            f"{getattr(reference, 'sequence_length', None)}; for a fixed k-range the window grid "
            "is determined by length, so a differing-length reference is inadmissible"
        )
    # 3/4/5. empty grid, finiteness, exact alignment
    aligned = _aligned_finite_z(design, reference)
    design_head_score_digest = _score_digest(design, head_identity, "design")
    reference_head_score_digest = _score_digest(reference, head_identity, "reference")

    deltas = [max(0.0, d - r) for d, r in aligned.values()]
    positives = [value for value in deltas if value > 0.0]
    evidence = object.__new__(WholeHotspotEvidence)
    values = {
        "endpoint_id": endpoint_id,
        "max_increase": max(deltas),
        "positive_mass": math.fsum(positives),
        "positive_count": len(positives),
        "n_windows": len(aligned),
        "reference_kind": reference_kind,
        "reference_binding_id": reference_binding_id,
        "design_sequence_md5": str(getattr(design, "sequence_md5", "")),
        "reference_sequence_md5": str(getattr(reference, "sequence_md5", "")),
        "head_identity_digest": head_identity.digest(),
        "design_head_score_digest": design_head_score_digest,
        "reference_head_score_digest": reference_head_score_digest,
    }
    for name, value in values.items():
        object.__setattr__(evidence, name, value)
    return evidence


@dataclass(frozen=True, init=False)
class SafetyAdmissionPolicy:
    """Runtime safety policy derived from one exact V2 config and frozen Head evaluator."""

    config_digest: str
    safety_config_digest: str
    cumulative_reference_kind: str
    cumulative_reference_label: str
    complete_reference_content_digest: str
    head_identity: HeadEvaluatorIdentity
    cumulative_threshold: HotspotThreshold
    incremental_gate_enabled: bool
    incremental_threshold: HotspotThreshold | None

def _safety_config_digest(config: V2Config) -> str:
    safety = config.safety
    return canonical_digest({
        "cumulative_reference_kind": safety.cumulative_reference_kind,
        "cumulative_reference_label": safety.cumulative_reference_label,
        "delta_new_cumulative": safety.delta_new_cumulative.canonical_payload(),
        "incremental_gate_enabled": safety.incremental_gate_enabled,
        "delta_new_incremental": (
            None if safety.delta_new_incremental is None
            else safety.delta_new_incremental.canonical_payload()
        ),
        "structure_cadence": safety.structure_cadence,
    })


def bind_admission_policy(
    config: V2Config, head_identity: HeadEvaluatorIdentity,
) -> SafetyAdmissionPolicy:
    """Bind admission to the complete V2 config, exact Head identity, and reference content."""
    if not isinstance(config, V2Config):
        raise V2SafetyError("config must be a V2Config")
    if not isinstance(head_identity, HeadEvaluatorIdentity):
        raise HeadIdentityMismatch("head_identity must be a HeadEvaluatorIdentity")
    expected_head = (
        config.head.allele,
        config.head.score_scale,
        config.head.window_k_min,
        config.head.window_k_max,
    )
    observed_head = (
        head_identity.allele,
        head_identity.score_scale,
        head_identity.window_k_min,
        head_identity.window_k_max,
    )
    if observed_head != expected_head:
        raise HeadIdentityMismatch(
            f"runtime Head domain {observed_head!r} does not match config {expected_head!r}"
        )
    reference_rows = [
        row for row in config.content if row.role == "complete_reference_sequence"
    ]
    if len(reference_rows) != 1 or reference_rows[0].binding != "frozen" \
            or reference_rows[0].expected_sha256 is None:
        raise V2SafetyError(
            "config must carry one frozen complete_reference_sequence content digest"
        )
    incremental_present = config.safety.delta_new_incremental is not None
    if config.safety.incremental_gate_enabled != incremental_present:
        raise V2SafetyError(
            "incremental_gate_enabled and delta_new_incremental disagree in V2 config"
        )
    cumulative_threshold = HotspotThreshold(
        calibration=config.safety.delta_new_cumulative,
        scope=ReferenceKind.CUMULATIVE_DEPTH0,
    )
    incremental_threshold = (
        HotspotThreshold(
            calibration=config.safety.delta_new_incremental,
            scope=ReferenceKind.IMMEDIATE_PARENT,
        )
        if config.safety.delta_new_incremental is not None else None
    )
    policy = object.__new__(SafetyAdmissionPolicy)
    values = {
        "config_digest": config.config_digest(),
        "safety_config_digest": _safety_config_digest(config),
        "cumulative_reference_kind": config.safety.cumulative_reference_kind,
        "cumulative_reference_label": config.safety.cumulative_reference_label,
        "complete_reference_content_digest": reference_rows[0].expected_sha256,
        "head_identity": head_identity,
        "cumulative_threshold": cumulative_threshold,
        "incremental_gate_enabled": config.safety.incremental_gate_enabled,
        "incremental_threshold": incremental_threshold,
    }
    for name, value in values.items():
        object.__setattr__(policy, name, value)
    return policy


@dataclass(frozen=True, init=False)
class CumulativeSafetyReference:
    """The immutable depth-0 reference ``ybar``: its identity, its policy, its exact Head evidence.

    Built only by :func:`bind_cumulative_reference`, carried by object identity thereafter.

    **Why the identity is composed rather than restated.** ``doc/FUSION_V2_Interface_Map.md`` §3 C5
    decides this composition -- ``binding: SafetyReferenceBinding`` with
    ``binding.bound_at_depth == 0`` and ``reference_binding_id == binding.reference_id`` -- and §2
    (Conflict 7) explains why the concept is split by layer at all: ``identity`` holds the pure
    identity a ``LivePartialState`` can carry without dragging a Head score object into
    ``state.py``, and this type adds the live score.

    That split is the whole reason the two must be ONE object here.
    ``identity.SafetyReferenceBinding`` is what every state stamps and the only thing an artifact
    records as the lineage's reference; this type is what ``measure_cumulative`` actually compares
    designs against. If each minted its own id from its own digest, the reference that DECIDES
    ancestry and the reference the record NAMES could differ while both looked well-formed, and no
    audit over the artifacts could tell.

    So every scalar an artifact might quote is read THROUGH ``binding``. They are properties, not
    fields: a second stored copy is a second reference waiting to disagree, and reconciling by
    assertion only works when someone remembers to assert.
    """

    lineage_id: str
    binding: SafetyReferenceBinding
    policy: SafetyAdmissionPolicy
    head_score: Any

    @property
    def reference_binding_id(self) -> str:
        return self.binding.reference_id

    @property
    def protein_id(self) -> str:
        return self.binding.head_binding.protein_id

    @property
    def reference_label(self) -> str:
        return self.binding.reference_label

    @property
    def sequence_md5(self) -> str:
        return self.binding.sequence_md5

    @property
    def sequence_length(self) -> int:
        return self.binding.sequence_length

    @property
    def reference_content_digest(self) -> str:
        return self.binding.reference_content_digest

    @property
    def head_score_digest(self) -> str:
        return self.binding.head_score_digest


@dataclass(frozen=True)
class ImmediateParentReference:
    """The rolling parent, held separately from the cumulative reference.

    Structurally incapable of replacing it: nothing in this type is a
    :class:`CumulativeSafetyReference`, and :func:`advance_lineage` accepts only this type for the
    parent slot.
    """

    binding_id: str
    endpoint_id: str
    head_score: Any
    sequence_md5: str
    head_score_digest: str
    head_identity_digest: str
    config_digest: str
    depth: int

    def __post_init__(self) -> None:
        _require_text(self.binding_id, "binding_id")
        _require_text(self.endpoint_id, "endpoint_id")
        _require_text(self.sequence_md5, "sequence_md5")
        require_digest(self.head_score_digest, "head_score_digest")
        require_digest(self.head_identity_digest, "head_identity_digest")
        require_digest(self.config_digest, "config_digest")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise V2SafetyError(f"depth must be a non-negative int, got {self.depth!r}")
        expected_binding = "parent:" + canonical_digest({
            "endpoint_id": self.endpoint_id,
            "sequence_md5": self.sequence_md5,
            "head_score_digest": self.head_score_digest,
            "head_identity_digest": self.head_identity_digest,
            "config_digest": self.config_digest,
            "depth": self.depth,
        })[:16]
        if self.binding_id != expected_binding:
            raise V2SafetyError("immediate-parent binding_id does not bind its endpoint evidence")


def reference_sequence_content_digest(sequence: str) -> str:
    """The content digest of a complete reference sequence: SHA-256 over its bytes.

    **Why this function and not another.** The ``complete_reference_sequence`` row of
    ``config.content`` is a PLAN §5.2 provenance row, and the driver establishes each declared
    input's observed identity as ``hashlib.sha256(path.read_bytes()).hexdigest()``
    (``scripts/run_rf_fusion_v2.py``, ``parse_declared_inputs``). For the config's declared digest
    and the digest recomputed here to be comparable at all they must be the same function over the
    same bytes, so this is SHA-256 over the sequence's ASCII bytes and the file declared for that
    role is the canonical sequence itself -- no header, no wrapper, no trailing newline. Any
    container format would put bytes in the file that are not in the sequence, and a pure module
    (no I/O, PLAN §2.7) could then never verify the config's number against anything.

    ``sequence`` is expected to have passed ``validate_complete_aa20`` already, so ASCII encoding
    is total; encoding is named explicitly rather than defaulted so the digested bytes are a stated
    fact and not a locale.
    """
    return sha256_hex(sequence.encode("ascii"))


def bind_cumulative_reference(
    *,
    lineage_id: str,
    protein_id: str,
    reference_label: str,
    reference_sequence: str,
    reference_content_digest: str,
    policy: SafetyAdmissionPolicy,
    head_score: Any,
) -> CumulativeSafetyReference:
    """The only constructor of the depth-0 safety reference.

    Content-bound before any depth-0 endpoint is scored (PLAN §2.7). The reference must be complete
    canonical AA20 -- the Head is fail-open on non-canonical input, so the guard is reused from the
    V1 entry stack rather than re-derived.

    ``reference_sequence`` and ``reference_content_digest`` arrive as two parameters, so the digest
    is recomputed FROM the sequence rather than merely compared against the config. Checking only
    that the supplied digest equals ``policy.complete_reference_content_digest`` proves that the
    caller typed the config's number; it says nothing about the bytes the comparator will then
    measure every design against. PLAN §5.2 binds the run signature to file **contents**, so
    without this recomputation two different complete AA20 references of the same length could both
    claim one frozen digest and the ``complete_reference_sequence`` role would constrain a string
    instead of a landscape.
    """
    _require_text(lineage_id, "lineage_id")
    _require_text(protein_id, "protein_id")
    _require_text(reference_label, "reference_label")
    require_digest(reference_content_digest, "reference_content_digest")
    if not isinstance(policy, SafetyAdmissionPolicy):
        raise V2SafetyError("policy must be a SafetyAdmissionPolicy")
    if reference_label != policy.cumulative_reference_label:
        raise ReferenceRebindAttempt(
            f"reference label {reference_label!r} does not match config-bound label "
            f"{policy.cumulative_reference_label!r}"
        )
    if reference_content_digest != policy.complete_reference_content_digest:
        raise ReferenceRebindAttempt(
            "reference content digest does not match the config-bound complete reference"
        )
    validate_complete_aa20(reference_sequence)
    observed_content_digest = reference_sequence_content_digest(reference_sequence)
    if observed_content_digest != reference_content_digest:
        raise ReferenceContentMismatch(
            "the reference sequence is not the content the declared digest identifies: the "
            f"supplied bytes digest to {observed_content_digest}, the run declares "
            f"{reference_content_digest}. PLAN §5.2 binds the run signature to file contents, so a "
            "digest that names other bytes is not this reference's identity"
        )

    md5 = sequence_md5(reference_sequence)
    head_score_digest = _score_digest(head_score, policy.head_identity, "reference")
    if getattr(head_score, "sequence_md5", None) != md5:
        raise HeadIdentityMismatch(
            "the reference Head score does not describe the reference sequence"
        )
    if getattr(head_score, "sequence_length", None) != len(reference_sequence):
        raise ReferenceLengthMismatch("the reference Head score declares a different length")
    if getattr(head_score, "protein_id", None) != protein_id:
        raise HeadIdentityMismatch("the reference Head score names a different protein")

    binding_id = "ref:" + canonical_digest({
        "lineage_id": lineage_id,
        "protein_id": protein_id,
        "reference_label": reference_label,
        "reference_kind": policy.cumulative_reference_kind,
        "sequence_md5": md5,
        "reference_content_digest": reference_content_digest,
        "head_identity": policy.head_identity.digest(),
        "head_score_digest": head_score_digest,
        "config_digest": policy.config_digest,
    })[:16]
    # The ONE depth-0 reference identity (Interface Map §3 C5). States stamp this object and
    # artifacts record its ``reference_id``; the ancestry decision below reads the same object, so
    # "the reference that decided" and "the reference on the record" cannot be two things.
    binding = SafetyReferenceBinding(
        reference_id=binding_id,
        reference_label=reference_label,
        sequence_md5=md5,
        sequence_length=len(reference_sequence),
        reference_content_digest=reference_content_digest,
        head_binding=HeadScoreBinding(
            protein_id=protein_id,
            sequence_md5=md5,
            sequence_length=len(reference_sequence),
            window_grid_digest=window_grid_digest(tuple(getattr(head_score, "windows", ()))),
            evaluator=policy.head_identity,
        ),
        head_score_digest=head_score_digest,
        bound_at_depth=0,
        source_kind=policy.cumulative_reference_kind,
    )
    reference = object.__new__(CumulativeSafetyReference)
    values = {
        "lineage_id": lineage_id,
        "binding": binding,
        "policy": policy,
        "head_score": head_score,
    }
    for name, value in values.items():
        object.__setattr__(reference, name, value)
    return reference


def bind_immediate_parent(
    *, endpoint_id: str, head_score: Any, depth: int, policy: SafetyAdmissionPolicy,
) -> ImmediateParentReference:
    """Bind the selected exact endpoint as the next step's immediate-parent reference."""
    if not isinstance(policy, SafetyAdmissionPolicy):
        raise V2SafetyError("policy must be a SafetyAdmissionPolicy")
    require_id_namespace(endpoint_id, "endpoint")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise LineageDepthSkew(f"immediate parent depth must be >= 1, got {depth!r}")
    score_digest = _score_digest(head_score, policy.head_identity, "selected parent")
    md5 = _require_text(getattr(head_score, "sequence_md5", None), "selected parent sequence_md5")
    payload = {
        "endpoint_id": endpoint_id,
        "sequence_md5": md5,
        "head_score_digest": score_digest,
        "head_identity_digest": policy.head_identity.digest(),
        "config_digest": policy.config_digest,
        "depth": depth,
    }
    return ImmediateParentReference(
        binding_id="parent:" + canonical_digest(payload)[:16],
        endpoint_id=endpoint_id,
        head_score=head_score,
        sequence_md5=md5,
        head_score_digest=score_digest,
        head_identity_digest=policy.head_identity.digest(),
        config_digest=policy.config_digest,
        depth=depth,
    )


@dataclass(frozen=True)
class LineageSafetyLedger:
    """One lineage's ratchet state.

    ``cumulative_reference`` is carried by object identity across every advance; there is no field,
    parameter, or method through which a different one can be installed.
    """

    cumulative_reference: CumulativeSafetyReference
    depth: int
    immediate_parent: ImmediateParentReference | None

    def __post_init__(self) -> None:
        if not isinstance(self.cumulative_reference, CumulativeSafetyReference):
            raise V2SafetyError("cumulative_reference must be a CumulativeSafetyReference")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise LineageDepthSkew(f"depth must be a non-negative int, got {self.depth!r}")
        if self.depth == 0 and self.immediate_parent is not None:
            raise LineageDepthSkew("depth 0 has no immediate parent")
        if self.depth > 0 and self.immediate_parent is None:
            raise LineageDepthSkew(f"depth {self.depth} must carry an immediate parent")
        if self.immediate_parent is not None:
            policy = self.cumulative_reference.policy
            if self.immediate_parent.config_digest != policy.config_digest:
                raise V2SafetyError("immediate parent is bound to a different V2 config")
            if self.immediate_parent.head_identity_digest != policy.head_identity.digest():
                raise HeadIdentityMismatch(
                    "immediate parent is bound to a different Head evaluator")


def open_lineage_ledger(reference: CumulativeSafetyReference) -> LineageSafetyLedger:
    """The unique depth-0 ledger for a lineage.

    Nothing can be measured or admitted without one, so the reference is provably bound first.
    """
    return LineageSafetyLedger(cumulative_reference=reference, depth=0, immediate_parent=None)


def measure_cumulative(
    ledger: LineageSafetyLedger, design: Any, *, endpoint_id: str,
) -> WholeHotspotEvidence:
    """Measure a design against the lineage's frozen depth-0 reference."""
    if not isinstance(ledger, LineageSafetyLedger):
        raise V2SafetyError("ledger must be a LineageSafetyLedger")
    reference = ledger.cumulative_reference
    current_reference_digest = _score_digest(
        reference.head_score, reference.policy.head_identity, "reference")
    if current_reference_digest != reference.head_score_digest:
        raise ReferenceRebindAttempt(
            "the frozen cumulative Head evidence was mutated after binding")
    if getattr(design, "sequence_md5", None) == reference.sequence_md5:
        raise SelfReferentialReference(
            "a design may not be measured against itself; N_H(y; y) == 0 would make the gate "
            "vacuous (PLAN §2.7)"
        )
    return whole_landscape_new_hotspot(
        design, reference.head_score, endpoint_id=endpoint_id,
        head_identity=reference.policy.head_identity,
        reference_kind=ReferenceKind.CUMULATIVE_DEPTH0,
        reference_binding_id=reference.reference_binding_id,
    )


def measure_incremental(
    ledger: LineageSafetyLedger, design: Any, *, endpoint_id: str,
) -> WholeHotspotEvidence:
    """Measure a design against the immediate parent -- step-local telemetry, never the ratchet."""
    if not isinstance(ledger, LineageSafetyLedger):
        raise V2SafetyError("ledger must be a LineageSafetyLedger")
    if ledger.immediate_parent is None:
        raise NoImmediateParent("no immediate parent exists at depth 0")
    parent = ledger.immediate_parent
    current_parent_digest = _score_digest(
        parent.head_score, ledger.cumulative_reference.policy.head_identity, "immediate parent")
    if current_parent_digest != parent.head_score_digest:
        raise V2SafetyError("the immediate-parent Head evidence was mutated after binding")
    return whole_landscape_new_hotspot(
        design, parent.head_score, endpoint_id=endpoint_id,
        head_identity=ledger.cumulative_reference.policy.head_identity,
        reference_kind=ReferenceKind.IMMEDIATE_PARENT,
        reference_binding_id=parent.binding_id,
    )


def apply_threshold(
    evidence: WholeHotspotEvidence, threshold: HotspotThreshold,
) -> WholeHotspotVerdict:
    """Judge evidence with a threshold calibrated for that same reference scope."""
    if not isinstance(evidence, WholeHotspotEvidence):
        raise V2SafetyError("evidence must be a WholeHotspotEvidence")
    if not isinstance(threshold, HotspotThreshold):
        raise V2SafetyError("threshold must be a HotspotThreshold")
    if threshold.scope is not evidence.reference_kind:
        raise ReferenceKindMismatch(
            f"threshold is calibrated for {threshold.scope.value!r} but the evidence was measured "
            f"against {evidence.reference_kind.value!r}; the two scopes measure different things"
        )
    passed = evidence.max_increase <= threshold.value
    return WholeHotspotVerdict(
        evidence=evidence, threshold=threshold, passed=passed,
        reason="within_threshold" if passed else "new_hotspot_exceeds_threshold",
    )


@dataclass(frozen=True, init=False)
class AdmissibilityVerdict:
    """Verified ``I_adm`` result; construct only through :func:`make_admissibility_verdict`."""

    admitted: bool
    policy: SafetyAdmissionPolicy
    endpoint_id: str
    endpoint_sequence_md5: str
    endpoint_head_score_digest: str
    cumulative: WholeHotspotVerdict
    incremental: WholeHotspotVerdict | None
    structure_definitive: bool
    structure_evidence_level: str
    constraints_preserved: bool



@dataclass(frozen=True)
class IncrementalInapplicable:
    """Proof that the incremental gate HAS NO REFERENT here, not that it was skipped.

    The incremental gate compares a design against its immediate parent.  A depth-0 lineage has no
    parent, so with ``incremental_gate_enabled`` the gate could be neither satisfied nor waived and
    every legal config carrying it was unlaunchable -- the ledger a depth-0 gate is built from
    always has ``immediate_parent=None``.

    "Inapplicable" is therefore recorded as a TYPED FACT rather than as a missing verdict.  It is
    minted only by :func:`incremental_inapplicable`, only from a ledger that really has no parent,
    and it carries the cumulative reference it was minted under so it cannot travel to another
    lineage.  A ``None`` in the same slot would be indistinguishable from an omission, which is the
    fail-open this path exists to refuse.
    """

    lineage_id: str
    depth: int
    cumulative_reference_binding_id: str


def incremental_inapplicable(ledger: "LineageSafetyLedger") -> IncrementalInapplicable:
    """Mint the marker, and only where the fact holds.

    Derived from the LEDGER and never supplied by a caller: one that could construct a marker for a
    lineage WITH a parent would be skipping a measurable incremental gate.
    """
    if not isinstance(ledger, LineageSafetyLedger):
        raise V2SafetyError("ledger must be a LineageSafetyLedger")
    if ledger.immediate_parent is not None:
        raise NoImmediateParent(
            f"this lineage has an immediate parent at depth {ledger.depth}, so the incremental "
            "gate has a referent and must be MEASURED; an inapplicability claim here would skip a "
            "gate that can be satisfied"
        )
    return IncrementalInapplicable(
        lineage_id=ledger.cumulative_reference.lineage_id,
        depth=int(ledger.depth),
        cumulative_reference_binding_id=ledger.cumulative_reference.reference_binding_id,
    )


def make_admissibility_verdict(
    *,
    policy: SafetyAdmissionPolicy,
    cumulative: WholeHotspotVerdict,
    incremental: WholeHotspotVerdict | IncrementalInapplicable | None,
    structure_definitive: bool,
    structure_evidence_level: str,
    constraints_preserved: bool,
) -> AdmissibilityVerdict:
    """Validate every binding and derive the admission conjunction fail closed."""
    if not isinstance(policy, SafetyAdmissionPolicy):
        raise V2SafetyError("policy must be a SafetyAdmissionPolicy")
    if not isinstance(cumulative, WholeHotspotVerdict):
        raise V2SafetyError("cumulative must be a WholeHotspotVerdict")
    if not isinstance(structure_definitive, bool) or not isinstance(constraints_preserved, bool):
        raise V2SafetyError("structure_definitive and constraints_preserved must be explicit bools")
    _require_text(structure_evidence_level, "structure_evidence_level")
    if structure_definitive != (structure_evidence_level == "definitive"):
        raise V2SafetyError(
            f"structure_definitive={structure_definitive} disagrees with "
            f"structure_evidence_level={structure_evidence_level!r}"
        )
    evidence = cumulative.evidence
    if evidence.reference_kind is not ReferenceKind.CUMULATIVE_DEPTH0:
        raise ReferenceKindMismatch("cumulative verdict must use the depth-0 reference")
    if cumulative.threshold != policy.cumulative_threshold:
        raise V2SafetyError("cumulative verdict uses a threshold from a different V2 config")
    if evidence.head_identity_digest != policy.head_identity.digest():
        raise HeadIdentityMismatch("cumulative evidence uses a different Head evaluator")

    if policy.incremental_gate_enabled:
        if incremental is None:
            raise V2SafetyError(
                "incremental gate is enabled in config, so a missing incremental verdict fails "
                "closed"
            )
        if isinstance(incremental, IncrementalInapplicable):
            # The gate has no referent for this lineage.  The marker is accepted only where the
            # cumulative reference it was minted under is THIS lineage's, so it cannot be carried
            # in from a run whose depth-0 really did lack a parent.
            if incremental.cumulative_reference_binding_id != evidence.reference_binding_id:
                raise ReferenceRebindAttempt(
                    "the inapplicability marker was minted under a different lineage's cumulative "
                    "reference; it certifies a fact about that lineage, not about this one"
                )
            if incremental.depth != 0:
                raise NoImmediateParent(
                    f"an inapplicability marker at depth {incremental.depth} claims a lineage with "
                    "no parent above depth 0, where one necessarily exists"
                )
        elif (policy.incremental_threshold is None
                or incremental.threshold != policy.incremental_threshold):
            raise V2SafetyError("incremental verdict uses a threshold from a different V2 config")
        if isinstance(incremental, WholeHotspotVerdict) \
                and incremental.evidence.reference_kind is not ReferenceKind.IMMEDIATE_PARENT:
            raise ReferenceKindMismatch("incremental verdict must use the immediate parent")
        if isinstance(incremental, WholeHotspotVerdict) \
                and incremental.evidence.head_identity_digest != policy.head_identity.digest():
            raise HeadIdentityMismatch("incremental evidence uses a different Head evaluator")
        for field in (("endpoint_id", "design_sequence_md5", "design_head_score_digest")
                      if isinstance(incremental, WholeHotspotVerdict) else ()):
            if getattr(incremental.evidence, field) != getattr(evidence, field):
                raise V2SafetyError(
                    f"incremental and cumulative verdicts describe different endpoint {field}"
                )
    elif incremental is not None:
        raise V2SafetyError(
            "incremental gate is disabled in config; incremental verdict is illegal")

    admitted = (
        structure_definitive
        and constraints_preserved
        and cumulative.passed
        # An inapplicability marker neither passes nor fails: there is nothing to compare against,
        # so it must not veto an otherwise admissible design (that would make an enabled gate a
        # blanket refusal at depth 0) and must not be counted as a pass either.
        and (incremental is None or isinstance(incremental, IncrementalInapplicable)
             or incremental.passed)
    )
    verdict = object.__new__(AdmissibilityVerdict)
    values = {
        "admitted": admitted,
        "policy": policy,
        "endpoint_id": evidence.endpoint_id,
        "endpoint_sequence_md5": evidence.design_sequence_md5,
        "endpoint_head_score_digest": evidence.design_head_score_digest,
        "cumulative": cumulative,
        "incremental": incremental,
        "structure_definitive": structure_definitive,
        "structure_evidence_level": structure_evidence_level,
        "constraints_preserved": constraints_preserved,
    }
    for name, value in values.items():
        object.__setattr__(verdict, name, value)
    return verdict


def advance_lineage(
    ledger: LineageSafetyLedger,
    *,
    depth: int,
    selected: ImmediateParentReference,
    admissibility: AdmissibilityVerdict,
) -> LineageSafetyLedger:
    """Advance one depth, keeping the **same** cumulative reference object.

    There is deliberately no parameter through which a new cumulative reference can enter. That is
    what makes the ratchet inescapable: a lineage cannot re-base its reference, so three steps that
    each add a small parent-relative increase still fail once their sum crosses the depth-0
    threshold.
    """
    if not isinstance(ledger, LineageSafetyLedger):
        raise V2SafetyError("ledger must be a LineageSafetyLedger")
    if not isinstance(selected, ImmediateParentReference):
        raise V2SafetyError("selected must be an ImmediateParentReference")
    if not isinstance(admissibility, AdmissibilityVerdict):
        raise V2SafetyError("admissibility must be an AdmissibilityVerdict")
    if depth != ledger.depth + 1:
        raise LineageDepthSkew(
            f"advance to depth {depth} from a depth-{ledger.depth} ledger skips or repeats a depth"
        )
    if not admissibility.admitted:
        raise V2SafetyError("only an admitted endpoint may advance a lineage")
    if not admissibility.structure_definitive:
        raise V2SafetyError(
            "feedback ancestry requires definitive structure feasibility; a provisional result may "
            "never purchase it (PLAN §2.7, §4.2)"
        )
    policy = ledger.cumulative_reference.policy
    if admissibility.policy != policy:
        raise V2SafetyError("admissibility was derived from a different V2 safety config")
    if (admissibility.cumulative.evidence.reference_binding_id
            != ledger.cumulative_reference.reference_binding_id):
        raise ReferenceRebindAttempt(
            "the admitting verdict was measured against a different reference than this lineage's"
        )
    if selected.config_digest != policy.config_digest:
        raise V2SafetyError("selected parent is bound to a different V2 config")
    if selected.head_identity_digest != policy.head_identity.digest():
        raise HeadIdentityMismatch("selected parent is bound to a different Head evaluator")
    selected_binding = (
        selected.endpoint_id,
        selected.sequence_md5,
        selected.head_score_digest,
    )
    admitted_binding = (
        admissibility.endpoint_id,
        admissibility.endpoint_sequence_md5,
        admissibility.endpoint_head_score_digest,
    )
    if selected_binding != admitted_binding:
        raise V2SafetyError(
            "selected parent does not match the exact endpoint and Head evidence that was admitted"
        )
    current_selected_digest = _score_digest(
        selected.head_score, policy.head_identity, "selected parent")
    if current_selected_digest != selected.head_score_digest:
        raise V2SafetyError("selected parent Head evidence was mutated after binding")
    if policy.incremental_gate_enabled:
        if admissibility.incremental is None:
            raise V2SafetyError(
                "enabled incremental gate requires an existing immediate parent and its verdict"
            )
        if ledger.immediate_parent is None:
            # Depth 0: the gate has no referent, so the ONLY admissible incremental record is the
            # typed inapplicability minted from this very ledger.  A measured verdict here would
            # have to name a parent that does not exist.
            if not isinstance(admissibility.incremental, IncrementalInapplicable):
                raise NoImmediateParent(
                    "a measured incremental verdict was supplied for a lineage with no immediate "
                    "parent; it cannot have been measured against one"
                )
            if (admissibility.incremental.cumulative_reference_binding_id
                    != ledger.cumulative_reference.reference_binding_id):
                raise ReferenceRebindAttempt(
                    "the inapplicability marker belongs to a different lineage"
                )
        elif isinstance(admissibility.incremental, IncrementalInapplicable):
            # A parent exists, so inapplicability is a SKIPPED measurement, not a fact.
            raise NoImmediateParent(
                "an inapplicability marker was supplied for a lineage that HAS an immediate "
                "parent, where the incremental gate is measurable and therefore binding"
            )
        elif (admissibility.incremental.evidence.reference_binding_id
                != ledger.immediate_parent.binding_id):
            raise ReferenceRebindAttempt(
                "incremental verdict was measured against a different immediate parent"
            )
    if selected.depth != depth:
        raise LineageDepthSkew(
            f"the selected parent declares depth {selected.depth}, not {depth}"
        )
    return LineageSafetyLedger(
        cumulative_reference=ledger.cumulative_reference, depth=depth, immediate_parent=selected,
    )


def assert_lineage_binding(
    ledger: LineageSafetyLedger, verdicts: Sequence[WholeHotspotVerdict],
) -> None:
    """Artifact-layer ratchet audit.

    Every cumulative verdict in a lineage's record set must name that lineage's single reference
    binding. A record set that names two is a rebind, however well-formed each row looks alone.
    """
    expected = ledger.cumulative_reference.reference_binding_id
    for verdict in verdicts:
        if verdict.evidence.reference_kind is not ReferenceKind.CUMULATIVE_DEPTH0:
            continue
        if verdict.evidence.reference_binding_id != expected:
            raise ReferenceRebindAttempt(
                f"a cumulative verdict names reference {verdict.evidence.reference_binding_id!r} "
                f"but the lineage is bound to {expected!r}; a re-based reference would accept "
                "unbounded cumulative drift"
            )


def cumulative_incremental_slack(
    cumulative: WholeHotspotEvidence, incrementals: Sequence[WholeHotspotEvidence],
) -> float:
    """``sum(step maxima) - cumulative maximum``. Telemetry only, never a gate input.

    On a shared grid ``max_w [sum_d delta_w,d]_+ <= sum_d max_w [delta_w,d]_+``, so a well-formed
    chain gives a non-negative slack. It is the computable form of "every step passed but the whole
    run did not".
    """
    return math.fsum(e.max_increase for e in incrementals) - cumulative.max_increase
