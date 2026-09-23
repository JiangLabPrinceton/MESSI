r"""DUALF2: two frozen Head results bound to one exact endpoint, as an immutable sidecar.

Scientific authority: ``doc/Dual_Allele_Steering.md`` §§2.1, 3.1. Implementation contract:
``PLAN_RF_FUSION_V2_DUAL_ALLELE.md`` §§1.3, 2.3, DUALF2.

The compatibility rule shapes everything here. A run with no Dual overlay must be byte-equivalent to
single-Head V2, so :class:`~fusion_v2.state.CompleteEndpoint` is not extended and not modified: role
A is the existing production Head, it populates ``head_global_risk`` exactly as before, and every
Dual quantity lives in a separate record keyed by ``endpoint_id``. Writing a joint scalar into the
legacy field -- the one shortcut this design exists to prevent -- would silently change what every
existing artifact, reader and regression means.

Two Head scores may be reduced only when they describe **one exact sequence on one window grid**.
The check is by identity, never by position: a reordered batch matched positionally produces
plausible numbers with no signal in them, and there is no downstream test that could tell.

**Purity.** stdlib plus :mod:`fusion_v2.identity` and :mod:`fusion_v2.joint_objective` only. No
torch, no config, no I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import V2Error
from .identity import HeadScoreBinding, canonical_digest
from .joint_objective import (
    AlleleRole,
    DualObjective,
    JointValue,
    V2JointObjectiveError,
)

__all__ = [
    "V2DualEvidenceError",
    "positive_mass_density",
    "role_b_hotspot_telemetry",
    "AlleleHotspotTelemetry",
    "HeadResult",
    "AlleleEndpointScore",
    "DualEndpointEvidence",
    "bind_dual_evidence",
    "bind_dual_evidence_batch",
]

#: The endpoint fields a Dual sidecar binds itself to. All four must agree across both Heads.
_SEQUENCE_IDENTITY_FIELDS = ("protein_id", "sequence_md5", "sequence_length")


class V2DualEvidenceError(V2Error):
    """A paired-Head binding or Dual endpoint-evidence contract was violated."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V2DualEvidenceError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise V2DualEvidenceError(f"{name} must be finite, got {out!r}")
    return out


def positive_mass_density(residue_hotspot: Sequence[float], sequence_length: int) -> float:
    r"""``sum_i max(z_i, 0) / L``: the length-normalized positive hotspot mass.

    The return-boundary front minimizes this alongside the complete-sequence risk, so the Dual layer
    needs it in the same shape for both alleles.

    Known duplication, recorded rather than hidden: ``reference_flow/refine.py`` computes the same
    quantity inline over a numpy hotspot array for the RF refinement path, and
    ``scripts/materialize_v2_archive_facade.py`` recomputes it from the stored score JSON. All three
    must agree; unifying them is a separate change to a module currently under concurrent edit, and
    silently introducing a fourth spelling would be worse than naming the duplication here.
    """
    length = sequence_length
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise V2DualEvidenceError(f"sequence_length must be a positive int, got {length!r}")
    values = tuple(residue_hotspot)
    if len(values) != length:
        raise V2DualEvidenceError(
            f"residue_hotspot has {len(values)} entries for a sequence of length {length}; the "
            "density is a per-residue mean and a mismatched vector would divide by the wrong L"
        )
    return math.fsum(max(_finite(value, "residue_hotspot entry"), 0.0) for value in values) / length


@dataclass(frozen=True)
class HeadResult:
    """One frozen Head's verdict on one exact sequence, with the binding that identifies it.

    ``density`` is the length-normalized positive hotspot mass. It is optional because it is read
    only by the return-boundary front and never by runtime steering, so a run that does not build
    that front never has to compute it.
    """

    score: Any
    binding: HeadScoreBinding
    density: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.binding, HeadScoreBinding):
            raise V2DualEvidenceError("binding must be a HeadScoreBinding")
        risk = getattr(self.score, "global_risk", None)
        if risk is None:
            raise V2DualEvidenceError("a Head result must carry global_risk")
        _finite(risk, "global_risk")
        for field in _SEQUENCE_IDENTITY_FIELDS:
            declared = getattr(self.score, field, None)
            if declared != getattr(self.binding, field):
                raise V2DualEvidenceError(
                    f"the Head score declares {field}={declared!r} but its binding declares "
                    f"{getattr(self.binding, field)!r}; a score and the binding that identifies it "
                    "must describe one object"
                )
        evaluator = self.binding.evaluator
        if getattr(self.score, "allele", None) != evaluator.allele:
            raise V2DualEvidenceError(
                f"the Head score declares allele {getattr(self.score, 'allele', None)!r}, its "
                f"evaluator declares {evaluator.allele!r}"
            )
        if getattr(self.score, "score_scale", None) != evaluator.score_scale:
            raise V2DualEvidenceError(
                f"the Head score declares score scale "
                f"{getattr(self.score, 'score_scale', None)!r}, its evaluator declares "
                f"{evaluator.score_scale!r}"
            )
        if self.density is not None:
            _finite(self.density, "density")

    @property
    def raw_risk(self) -> float:
        return float(getattr(self.score, "global_risk"))

    @property
    def allele(self) -> str:
        return self.binding.evaluator.allele

    @property
    def sequence_md5(self) -> str:
        return self.binding.sequence_md5


@dataclass(frozen=True)
class AlleleHotspotTelemetry:
    """Role B's whole-landscape hotspot drift against the frozen depth-0 reference.

    **Reported, not gated.** The single-Head cumulative gate is inherited unchanged and remains the
    only admission law; this record says how far the non-optimized allele's landscape moved, and
    nothing consumes it as a threshold.

    That is a deliberately narrower contract than a second conjunctive gate, for three reasons. On
    the frozen high-risk substrate the allele-A ceiling is already set to an effectively disabling
    value, so a second gate would have nothing to bind against. No per-protein ceiling exists for
    role B, and the only producer of one is generative -- it samples a completion population and
    runs the structure gate -- which the calibration stage explicitly excludes. And the reference
    binding that a second gate would need is a field of the live partial state, inside its canonical
    payload, so adding one would move every state id and therefore every endpoint id, breaking both
    the Dual-off equivalence guarantee and the byte-identical shared-root requirement the recursive
    comparison rests on.

    Measure first. A second gate is justified by observed role-B drift in this telemetry, not by
    the symmetry of the formula.
    """

    allele: str
    max_increase: float
    positive_mass: float
    positive_count: int
    n_windows: int
    reference_sequence_md5: str

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "allele": self.allele,
            "max_increase": self.max_increase,
            "positive_mass": self.positive_mass,
            "positive_count": self.positive_count,
            "n_windows": self.n_windows,
            "reference_sequence_md5": self.reference_sequence_md5,
        }


def role_b_hotspot_telemetry(
    *,
    endpoint_id: str,
    design_score_b: Any,
    reference_score_b: Any,
    evaluator_b: Any,
    reference_binding_id: str,
) -> AlleleHotspotTelemetry:
    """Role B's ``N_H^whole`` against the frozen reference, via the existing pure primitive.

    Reuses ``safety.whole_landscape_new_hotspot`` rather than re-deriving the window comparison, so
    the two alleles' drift is measured by one instrument. Imported locally to keep a Dual-off run
    from importing the Dual layer at all.
    """
    from .safety import ReferenceKind, whole_landscape_new_hotspot

    evidence = whole_landscape_new_hotspot(
        design_score_b, reference_score_b, endpoint_id=str(endpoint_id),
        head_identity=evaluator_b, reference_kind=ReferenceKind.CUMULATIVE_DEPTH0,
        reference_binding_id=str(reference_binding_id),
    )
    return AlleleHotspotTelemetry(
        allele=evaluator_b.allele, max_increase=float(evidence.max_increase),
        positive_mass=float(evidence.positive_mass),
        positive_count=int(evidence.positive_count), n_windows=int(evidence.n_windows),
        reference_sequence_md5=str(evidence.reference_sequence_md5),
    )


@dataclass(frozen=True)
class AlleleEndpointScore:
    """One allele's raw verdict on the endpoint, bound to the instrument that produced it."""

    role: AlleleRole
    allele: str
    raw_risk: float
    raw_density: float | None
    head_binding_digest: str

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "allele": self.allele,
            "raw_risk": self.raw_risk,
            "raw_density": self.raw_density,
            "head_binding_digest": self.head_binding_digest,
        }


@dataclass(frozen=True)
class DualEndpointEvidence:
    """The endpoint-id-bound Dual record. Immutable, and never a substitute for the endpoint."""

    endpoint_id: str
    protein_id: str
    sequence_md5: str
    sequence_length: int
    window_grid_digest: str
    a: AlleleEndpointScore
    b: AlleleEndpointScore
    risk: JointValue
    density: JointValue | None
    calibration_digest: str
    #: Role B's hotspot drift, when the run chose to measure it. Never an admission input.
    hotspot_b: AlleleHotspotTelemetry | None = None

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "protein_id": self.protein_id,
            "sequence_md5": self.sequence_md5,
            "sequence_length": self.sequence_length,
            "window_grid_digest": self.window_grid_digest,
            "a": self.a.canonical_payload(),
            "b": self.b.canonical_payload(),
            "risk": self.risk.canonical_payload(),
            "density": None if self.density is None else self.density.canonical_payload(),
            "calibration_digest": self.calibration_digest,
            "hotspot_b": None if self.hotspot_b is None else self.hotspot_b.canonical_payload(),
        }

    @property
    def content_digest(self) -> str:
        return canonical_digest(self.canonical_payload())


def _require_objective(objective: Any, quantity: str, name: str) -> DualObjective:
    if not isinstance(objective, DualObjective):
        raise V2DualEvidenceError(f"{name} must be a DualObjective")
    if objective.quantity != quantity:
        raise V2DualEvidenceError(
            f"{name} resolves {objective.quantity!r} but {quantity!r} was required"
        )
    return objective


def _require_result(result: Any, role: AlleleRole) -> HeadResult:
    if result is None:
        raise V2DualEvidenceError(
            f"no Head result was supplied for role {role.value}; a Dual decision taken on one "
            "allele with the other defaulted is a single-allele decision wearing a joint label"
        )
    if not isinstance(result, HeadResult):
        raise V2DualEvidenceError(f"the role {role.value} result must be a HeadResult")
    return result


def bind_dual_evidence(
    *,
    endpoint: Any,
    result_a: Any,
    result_b: Any,
    objective: DualObjective,
    density_objective: DualObjective | None = None,
) -> DualEndpointEvidence:
    """Attach both Heads' verdicts to one exact endpoint, or refuse.

    ``endpoint`` is the unmodified legacy :class:`CompleteEndpoint`. Role A must be the evaluator
    that actually produced it and must report the risk the endpoint already carries: that equality
    is what proves no joint scalar was written into ``head_global_risk`` upstream, which is the
    single failure mode that would make every legacy artifact mean something else without any
    downstream check noticing.
    """
    obj = _require_objective(objective, "global_risk", "objective")
    left = _require_result(result_a, AlleleRole.A)
    right = _require_result(result_b, AlleleRole.B)

    coordinates = obj.coordinates
    for role, result in ((AlleleRole.A, left), (AlleleRole.B, right)):
        expected = coordinates.coordinate(role).evaluator
        if result.binding.evaluator != expected:
            raise V2DualEvidenceError(
                f"the result supplied for role {role.value} was produced by evaluator "
                f"{result.binding.evaluator.allele!r}/"
                f"{result.binding.evaluator.head_checkpoint_digest[:12]} but that role is "
                f"calibrated for {expected.allele!r}/{expected.head_checkpoint_digest[:12]}; "
                "applying one allele's coordinate to the other allele's risk would relabel the "
                "objective without changing a single stored number"
            )

    endpoint_binding = getattr(endpoint, "head_binding", None)
    if not isinstance(endpoint_binding, HeadScoreBinding):
        raise V2DualEvidenceError("endpoint must carry a HeadScoreBinding")
    if endpoint_binding.evaluator != left.binding.evaluator:
        raise V2DualEvidenceError(
            "role A is not the evaluator this endpoint was scored by; the legacy Head fields would "
            "then describe an instrument the Dual record does not name"
        )

    legacy_risk = getattr(endpoint, "head_global_risk", None)
    if legacy_risk is None:
        raise V2DualEvidenceError("endpoint must carry head_global_risk")
    if float(legacy_risk) != left.raw_risk:
        raise V2DualEvidenceError(
            f"the endpoint carries head_global_risk={float(legacy_risk)!r} but the role A result "
            f"reports {left.raw_risk!r}; the legacy field must remain exactly the role A raw risk "
            "(PLAN §0.3: Dual must never overwrite head_global_risk with a joint scalar)"
        )

    for field in _SEQUENCE_IDENTITY_FIELDS:
        anchor = getattr(endpoint, field, None)
        for role, result in ((AlleleRole.A, left), (AlleleRole.B, right)):
            declared = getattr(result.binding, field)
            if declared != anchor:
                raise V2DualEvidenceError(
                    f"the role {role.value} result declares {field}={declared!r} but the endpoint "
                    f"declares {anchor!r}; a joint value is defined for two alleles on ONE exact "
                    "sequence, and pairing across sequences compares two molecules"
                )
    for role, result in ((AlleleRole.A, left), (AlleleRole.B, right)):
        if result.binding.window_grid_digest != endpoint_binding.window_grid_digest:
            raise V2DualEvidenceError(
                f"the role {role.value} result was scored on window grid "
                f"{result.binding.window_grid_digest} but the endpoint declares "
                f"{endpoint_binding.window_grid_digest}; two grids are two landscapes, and their "
                "difference measures nothing"
            )

    if left.binding.evaluator.head_checkpoint_digest == \
            right.binding.evaluator.head_checkpoint_digest:
        raise V2DualEvidenceError(
            "both roles carry the same Head checkpoint digest; one Head has been bound twice and "
            "the objective would reduce an allele against itself"
        )

    try:
        risk = obj.evaluate(raw_a=left.raw_risk, raw_b=right.raw_risk)
    except V2JointObjectiveError as exc:
        raise V2DualEvidenceError(f"the joint risk could not be resolved: {exc}") from exc

    density: JointValue | None = None
    if density_objective is not None:
        density_obj = _require_objective(
            density_objective, "positive_mass_density", "density_objective")
        if density_obj.calibration.content_digest != obj.calibration.content_digest:
            raise V2DualEvidenceError(
                "the density objective was resolved from a different calibration than the risk "
                "objective; one endpoint carries one frozen coordinate system"
            )
        if left.density is None or right.density is None:
            raise V2DualEvidenceError(
                "a density objective was supplied but one of the two Heads reported no density; a "
                "half-reported return-boundary axis would silently fall back to a single allele"
            )
        density = density_obj.evaluate(raw_a=left.density, raw_b=right.density)

    return DualEndpointEvidence(
        endpoint_id=str(getattr(endpoint, "endpoint_id")),
        protein_id=str(getattr(endpoint, "protein_id")),
        sequence_md5=str(getattr(endpoint, "sequence_md5")),
        sequence_length=int(getattr(endpoint, "sequence_length")),
        window_grid_digest=endpoint_binding.window_grid_digest,
        a=AlleleEndpointScore(
            role=AlleleRole.A, allele=left.allele, raw_risk=left.raw_risk,
            raw_density=left.density, head_binding_digest=left.binding.digest()),
        b=AlleleEndpointScore(
            role=AlleleRole.B, allele=right.allele, raw_risk=right.raw_risk,
            raw_density=right.density, head_binding_digest=right.binding.digest()),
        risk=risk, density=density,
        calibration_digest=obj.calibration.content_digest,
    )


def _index_by_sequence(results: Sequence[Any], role: AlleleRole) -> Mapping[str, HeadResult]:
    """One result per DISTINCT sequence, matched by identity rather than by position."""
    indexed: dict[str, HeadResult] = {}
    for result in results:
        typed = _require_result(result, role)
        md5 = typed.sequence_md5
        if md5 in indexed:
            raise V2DualEvidenceError(
                f"duplicate role {role.value} Head result for sequence {md5}; the Head is a "
                "function of the sequence, so scoring one twice is the same measurement counted "
                "twice and the ledger would charge for both"
            )
        indexed[md5] = typed
    return indexed


def bind_dual_evidence_batch(
    *,
    endpoints: Sequence[Any],
    results_a: Sequence[Any],
    results_b: Sequence[Any],
    objective: DualObjective,
    density_objective: DualObjective | None = None,
) -> tuple[DualEndpointEvidence, ...]:
    """Bind a whole screen's worth of endpoints, proving both Heads covered the same sequence set.

    Endpoints may legally share bytes -- two forks producing an identical sequence are duplicate
    siblings -- so the two result sets are indexed by sequence and one result serves every endpoint
    carrying it. What may not happen is one allele covering a sequence the other did not: a
    partially paired batch entering selection would let the joint objective be resolved for some
    endpoints and quietly skipped for others, and the surviving set would be selected on a mixture
    of two laws.
    """
    by_a = _index_by_sequence(results_a, AlleleRole.A)
    by_b = _index_by_sequence(results_b, AlleleRole.B)
    expected = {str(getattr(endpoint, "sequence_md5")) for endpoint in endpoints}

    for role, indexed in ((AlleleRole.A, by_a), (AlleleRole.B, by_b)):
        unexpected = set(indexed) - expected
        if unexpected:
            raise V2DualEvidenceError(
                f"role {role.value} returned result(s) for {sorted(unexpected)}, which correspond "
                "to no endpoint in this batch; the batch does not describe what it was asked to "
                "score"
            )
        missing = expected - set(indexed)
        if missing:
            raise V2DualEvidenceError(
                f"role {role.value} is missing Head result(s) for {sorted(missing)}; a partially "
                "paired batch would resolve the joint objective for some endpoints and skip it "
                "for others, and the surviving set would be selected under a mixture of two laws"
            )

    return tuple(
        bind_dual_evidence(
            endpoint=endpoint,
            result_a=by_a[str(getattr(endpoint, "sequence_md5"))],
            result_b=by_b[str(getattr(endpoint, "sequence_md5"))],
            objective=objective, density_objective=density_objective)
        for endpoint in endpoints
    )
