"""V2F5A: the lineage reward incumbent and the donor gate (PLAN §2.5, §3.1, A.2).

PLAN §2.5 binds three complete-sequence references that must never be interchangeable:

* the **immutable cumulative safety reference** ``ybar`` -- never updates, owned by
  :mod:`fusion_v2.safety`, and decides ADMISSION;
* the **lineage reward incumbent** ``I_d`` -- the best ACCEPTED exact feasible endpoint for the
  family, owned by this module, and decides whether a donor may open feedback at all; and
* the **current donor** ``Y*_d`` -- the endpoint proposed to supply feedback at this depth.

The distinction is the whole point of the V2F5A change.  S7 selected a donor by exact global Head
rank and then applied a Head-blind positional dose; ranking alone is not direction.  Here a donor
may open feedback only when it beats the lineage's own incumbent by more than a calibrated
tolerance, and a lineage that has no such donor STALLS rather than adopting the least-bad endpoint:
PLAN A.2, "It must not fall back to an arbitrary endpoint token".

**Purity.**  No config import (the policy imports this module, and PLAN §2.5 forbids a policy that
can reach an unfrozen threshold), no torch, no I/O.  ``epsilon_r`` arrives as a plain float that the
config layer has already bound to a typed calibration artifact, together with the digest that names
it, so the number carried here is always traceable to the artifact that measured it.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Any

from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion.v1_alloc import validate_complete_aa20

from .errors import V2Error
from .identity import (
    HeadEvaluatorIdentity,
    SafetyReferenceBinding,
    canonical_digest,
    require_digest,
)

__all__ = [
    "V2RewardError",
    "IncumbentIdentityError",
    "IncumbentAliasError",
    "DEPTH0_BOOTSTRAP_RULE",
    "DEPTH0_INCUMBENT_RULES",
    "INCUMBENT_UPDATE_LAWS",
    "LineageIncumbentKind",
    "LineageIncumbent",
    "DonorGateReason",
    "DonorGateVerdict",
    "JointComparison",
    "bind_incumbent_from_safety_reference",
    "bind_incumbent_from_endpoint",
    "donor_gate",
    "advance_incumbent",
]


class V2RewardError(V2Error):
    """A reward-reference contract was violated."""


class IncumbentIdentityError(V2RewardError):
    """The incumbent does not describe the molecule, evaluator or bytes it claims to."""


class IncumbentAliasError(V2RewardError):
    """Two of the three non-interchangeable references collapsed onto one, undeclared."""


#: How ``I_0`` is bound before any depth-0 endpoint outcome is inspected.  PLAN §3.1 leaves the
#: depth-zero binding an OPEN decision, so the run must DECLARE which rule it used; this module
#: implements the vocabulary and refuses anything outside it.  It has no default anywhere.
DEPTH0_BOOTSTRAP_RULE = "best_admissible_depth0"

DEPTH0_INCUMBENT_RULES = frozenset({
    #: ``I_0 = ybar``: the frozen complete reference the run already content-binds for the safety
    #: ratchet also serves as the depth-0 reward baseline.  It is frozen before any endpoint is
    #: scored by construction, which is exactly PLAN §3.1's requirement, and it makes the first
    #: donor gate read "the donor must beat what we started from".
    "cumulative_safety_reference",
    #: ``I_0 =`` an externally predeclared complete design, scored through the same frozen Head and
    #: frozen before the depth-0 pool exists.  Implemented here; a runtime that offers it must
    #: supply the sequence through its own content-bound input role.
    "predeclared_external_design",
    #: No logical reward incumbent exists at D0. The cycle's rank-zero generated endpoint is
    #: adopted as I1; WT remains only the immutable safety/local-attribution reference.
    DEPTH0_BOOTSTRAP_RULE,
})

#: How ``I_d`` moves.  One law, named so the artifact records which one ran.
INCUMBENT_UPDATE_LAWS = frozenset({"strict_improvement_by_epsilon"})


class LineageIncumbentKind(str, enum.Enum):
    """What the incumbent IS -- which decides whether it may alias the safety reference."""

    #: ``I_d`` is the immutable cumulative safety reference (depth-0 binding only).
    CUMULATIVE_SAFETY_REFERENCE = "cumulative_safety_reference"
    #: ``I_d`` is an externally predeclared complete design, not from this run's pool.
    PREDECLARED_EXTERNAL_DESIGN = "predeclared_external_design"
    #: ``I_d`` is an exact feasible endpoint this lineage accepted.
    ACCEPTED_ENDPOINT = "accepted_endpoint"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V2RewardError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise V2RewardError(f"{name} must be finite, got {out!r}")
    return out


def _text(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise V2RewardError(f"{name} must be a non-empty str, got {value!r}")
    return value


@dataclass(frozen=True)
class LineageIncumbent:
    """``I_d``: the complete sequence the next donor must beat, with its exact Head evidence.

    Carries the SEQUENCE, not only its digest.  The leave-one-out counterfactual reverts a donor
    identity to the incumbent's residue at that position, so the bytes are load-bearing evidence
    rather than provenance -- and binding them here, verified against the digest, is what stops a
    policy from reverting toward a sequence nobody signed.

    ``aliases_safety_reference`` is DERIVED and then checked against the declared kind, so the one
    legal collapse of two references (the depth-0 ``I_0 = ybar`` binding) is always explicit and an
    accidental one is refused.
    """

    kind: LineageIncumbentKind
    incumbent_id: str
    protein_id: str
    lineage_id: str
    sequence: str
    sequence_md5: str
    sequence_length: int
    head_score: Any
    head_global_risk: float
    head_identity_digest: str
    head_score_digest: str
    bound_at_depth: int
    accepted_at_depth: int
    safety_reference_sequence_md5: str
    source_endpoint_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, LineageIncumbentKind):
            raise V2RewardError("kind must be a LineageIncumbentKind")
        for name in ("incumbent_id", "protein_id", "lineage_id"):
            _text(getattr(self, name), name)
        validate_complete_aa20(self.sequence, self.sequence_length)
        if sequence_md5(self.sequence) != self.sequence_md5:
            raise IncumbentIdentityError(
                "the incumbent's sequence_md5 does not digest the bytes it is stored with; the "
                "leave-one-out counterfactual reverts toward these bytes, so a mismatched digest "
                "would revert toward a sequence the run never signed"
            )
        if getattr(self.head_score, "sequence_md5", None) != self.sequence_md5:
            raise IncumbentIdentityError(
                "the incumbent's Head score describes a different sequence than its own bytes"
            )
        if getattr(self.head_score, "protein_id", None) != self.protein_id:
            raise IncumbentIdentityError("the incumbent's Head score describes a different protein")
        _finite(self.head_global_risk, "head_global_risk")
        if _finite(getattr(self.head_score, "global_risk", None),
                   "incumbent head_score.global_risk") != float(self.head_global_risk):
            raise IncumbentIdentityError(
                "head_global_risk does not equal the incumbent's own Head score global_risk"
            )
        require_digest(self.head_identity_digest, "head_identity_digest")
        require_digest(self.head_score_digest, "head_score_digest")
        for name in ("bound_at_depth", "accepted_at_depth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise V2RewardError(f"{name} must be a non-negative int, got {value!r}")
        aliases = self.safety_reference_sequence_md5 == self.sequence_md5
        if self.kind is LineageIncumbentKind.CUMULATIVE_SAFETY_REFERENCE and not aliases:
            raise IncumbentAliasError(
                "the incumbent declares itself the cumulative safety reference but carries "
                "different bytes; the two references would be one name over two molecules"
            )
        if self.kind is not LineageIncumbentKind.CUMULATIVE_SAFETY_REFERENCE and aliases:
            raise IncumbentAliasError(
                f"an incumbent of kind {self.kind.value!r} carries the cumulative safety "
                "reference's own bytes; PLAN §2.5 makes the two references non-interchangeable, and "
                "an undeclared collapse would make 'better than the incumbent' mean 'better than "
                "the reference' without the run ever saying so"
            )
        if self.kind is LineageIncumbentKind.ACCEPTED_ENDPOINT and not self.source_endpoint_id:
            raise V2RewardError(
                "an accepted-endpoint incumbent must name the endpoint it was accepted from"
            )
        if self.kind is not LineageIncumbentKind.ACCEPTED_ENDPOINT and self.source_endpoint_id:
            raise V2RewardError(
                f"an incumbent of kind {self.kind.value!r} may not name a source endpoint"
            )

    @property
    def aliases_safety_reference(self) -> bool:
        return self.safety_reference_sequence_md5 == self.sequence_md5

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "incumbent_id": self.incumbent_id,
            "protein_id": self.protein_id,
            "lineage_id": self.lineage_id,
            "sequence_md5": self.sequence_md5,
            "sequence_length": self.sequence_length,
            "head_global_risk": self.head_global_risk,
            "head_identity_digest": self.head_identity_digest,
            "head_score_digest": self.head_score_digest,
            "bound_at_depth": self.bound_at_depth,
            "accepted_at_depth": self.accepted_at_depth,
            "source_endpoint_id": self.source_endpoint_id,
            "aliases_safety_reference": self.aliases_safety_reference,
        }

    @property
    def content_digest(self) -> str:
        return canonical_digest(self.canonical_payload())


def _incumbent_id(payload: dict[str, Any]) -> str:
    return "incumbent:" + canonical_digest(payload)[:16]


def bind_incumbent_from_safety_reference(
    *,
    reference: Any,
    reference_sequence: str,
    lineage_id: str,
    evaluator: HeadEvaluatorIdentity,
    rule: str,
) -> LineageIncumbent:
    """``I_0 = ybar``: bind the depth-0 incumbent to the frozen complete reference.

    ``reference`` is duck-typed on ``fusion_v2.safety.CumulativeSafetyReference`` -- this module
    stays out of the safety layer's import graph so the policy's transitive closure cannot reach the
    run config -- but its ``binding`` is type-checked, so the object really is a bound depth-0
    reference and not a look-alike.

    The bytes are supplied separately and verified against the binding's own digest: the safety
    reference stores identity, not sequence, and the counterfactual needs the residues.
    """
    if rule != "cumulative_safety_reference":
        raise V2RewardError(
            f"bind_incumbent_from_safety_reference implements the "
            f"'cumulative_safety_reference' depth-0 rule; the run declared {rule!r}"
        )
    binding = getattr(reference, "binding", None)
    if not isinstance(binding, SafetyReferenceBinding):
        raise IncumbentIdentityError(
            "reference must carry a SafetyReferenceBinding; an unbound object could name any "
            "molecule as the lineage's reward baseline"
        )
    if int(binding.bound_at_depth) != 0:
        raise IncumbentIdentityError(
            f"the safety reference was bound at depth {binding.bound_at_depth}, not 0; only the "
            "depth-0 reference is frozen before any endpoint outcome is inspected (PLAN §3.1)"
        )
    if sequence_md5(reference_sequence) != binding.sequence_md5:
        raise IncumbentIdentityError(
            "the supplied reference bytes do not digest to the safety reference's own sequence_md5"
        )
    head_score = getattr(reference, "head_score", None)
    payload = {
        "kind": LineageIncumbentKind.CUMULATIVE_SAFETY_REFERENCE.value,
        "lineage_id": lineage_id,
        "reference_binding_id": binding.reference_id,
        "sequence_md5": binding.sequence_md5,
        "head_identity_digest": evaluator.digest(),
    }
    return LineageIncumbent(
        kind=LineageIncumbentKind.CUMULATIVE_SAFETY_REFERENCE,
        incumbent_id=_incumbent_id(payload),
        protein_id=binding.head_binding.protein_id,
        lineage_id=lineage_id,
        sequence=reference_sequence,
        sequence_md5=binding.sequence_md5,
        sequence_length=int(binding.sequence_length),
        head_score=head_score,
        head_global_risk=_finite(getattr(head_score, "global_risk", None),
                                 "safety reference global_risk"),
        head_identity_digest=evaluator.digest(),
        head_score_digest=binding.head_score_digest,
        bound_at_depth=0,
        accepted_at_depth=0,
        safety_reference_sequence_md5=binding.sequence_md5,
    )


def bind_incumbent_from_endpoint(
    *,
    endpoint: Any,
    lineage_id: str,
    evaluator: HeadEvaluatorIdentity,
    safety_reference_sequence_md5: str,
    accepted_at_depth: int,
    kind: LineageIncumbentKind = LineageIncumbentKind.ACCEPTED_ENDPOINT,
) -> LineageIncumbent:
    """Bind ``I_d`` to an exact endpoint this lineage accepted (or a predeclared external design).

    Refuses an endpoint that is not definitively feasible: PLAN §4.2 forbids a non-definitive
    endpoint from becoming ancestry, and an incumbent is what ancestry is measured against.
    """
    from .state import FeasibilityLevel  # local: state imports nothing this module needs at import

    if getattr(endpoint, "feasibility_level", None) is not FeasibilityLevel.DEFINITIVE:
        raise V2RewardError(
            "only a definitively feasible endpoint may become the lineage reward incumbent; a "
            "provisional one would make every later donor gate a comparison against an unverified "
            "design (PLAN §4.2)"
        )
    binding = getattr(endpoint, "head_binding", None)
    if getattr(binding, "evaluator", None) != evaluator:
        raise IncumbentIdentityError(
            "the endpoint was scored by a different Head evaluator than the one the lineage runs; "
            "the donor gate would subtract two instruments"
        )
    payload = {
        "kind": kind.value,
        "lineage_id": lineage_id,
        "endpoint_id": endpoint.endpoint_id,
        "sequence_md5": endpoint.sequence_md5,
        "head_identity_digest": evaluator.digest(),
        "accepted_at_depth": int(accepted_at_depth),
    }
    return LineageIncumbent(
        kind=kind,
        incumbent_id=_incumbent_id(payload),
        protein_id=endpoint.protein_id,
        lineage_id=lineage_id,
        sequence=endpoint.sequence,
        sequence_md5=endpoint.sequence_md5,
        sequence_length=int(endpoint.sequence_length),
        head_score=endpoint.head_score,
        head_global_risk=float(endpoint.head_global_risk),
        head_identity_digest=evaluator.digest(),
        head_score_digest=canonical_digest({
            "sequence_md5": endpoint.sequence_md5,
            "head_identity_digest": evaluator.digest(),
            "global_risk": float(endpoint.head_global_risk),
        }),
        bound_at_depth=int(accepted_at_depth),
        accepted_at_depth=int(accepted_at_depth),
        safety_reference_sequence_md5=str(safety_reference_sequence_md5),
        source_endpoint_id=(str(endpoint.endpoint_id)
                            if kind is LineageIncumbentKind.ACCEPTED_ENDPOINT else None),
    )


class DonorGateReason(str, enum.Enum):
    """Why a donor may or may not open directional feedback."""

    ACCEPTED = "accepted"
    #: PLAN A.2's typed stall: no endpoint improves the incumbent by more than ``epsilon_R``.
    STALL_NO_BETTER_DONOR = "stall_no_better_donor"
    #: The donor IS the incumbent -- it cannot improve on itself.
    DONOR_IS_INCUMBENT = "donor_is_incumbent"
    #: The donor never reached definitive feasibility, so it may not become ancestry at all.
    DONOR_NOT_DEFINITIVE = "donor_not_definitive"


@dataclass(frozen=True)
class JointComparison:
    """The two joint objective values a Dual donor gate compares, and the law that produced them.

    Supplied by the caller rather than read off the endpoints, because the joint value lives in the
    Dual evidence sidecar and this module may not import it: ``reward`` is a leaf, and a leaf that
    reached into the sidecar could no longer be reasoned about without it.

    Both values must carry the SAME objective digest. Two values from two calibrations are two
    scales, and their difference is not a margin -- the same law this module already applies to two
    Head evaluator identities, applied one level up.
    """

    donor_value: float
    incumbent_value: float
    objective_digest: str
    #: PLAN 2.1: ``eps_joint = max_a eps_a_raw / s_a``, DERIVED from the objective's own geometry
    #: rather than declared.  Carried here because the gate compares NORMALIZED values, and the
    #: run config's raw-scale ``epsilon_r`` is a threshold on a different quantity: whenever
    #: ``s_a != 1`` the two mean different amounts of evidence.
    epsilon: float = 0.0
    #: The per-allele raw floors the margin above was propagated from, so a verdict records its
    #: provenance rather than only its result.
    raw_floor_a: float = 0.0
    raw_floor_b: float = 0.0

    def __post_init__(self) -> None:
        _finite(self.donor_value, "donor_value")
        _finite(self.incumbent_value, "incumbent_value")
        require_digest(self.objective_digest, "objective_digest")
        for name in ("epsilon", "raw_floor_a", "raw_floor_b"):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise V2RewardError(
                    f"{name} must be non-negative, got {value}; a negative floor would ADMIT a "
                    "donor that is worse than the incumbent"
                )

    @property
    def margin(self) -> float:
        return self.incumbent_value - self.donor_value

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "donor_value": self.donor_value,
            "incumbent_value": self.incumbent_value,
            "objective_digest": self.objective_digest,
            "epsilon": self.epsilon,
            "raw_floor_a": self.raw_floor_a,
            "raw_floor_b": self.raw_floor_b,
        }


@dataclass(frozen=True)
class DonorGateVerdict:
    """The donor gate's decision plus every number behind it.

    ``margin = R_H(I_d) - R_H(Y*_d)`` -- how much better the donor is.  Recorded even on a refusal,
    because "no donor improved" and "one donor improved by less than the calibrated noise floor" are
    different facts about a cohort and the artifact must be able to tell them apart.
    """

    passed: bool
    reason: DonorGateReason
    donor_endpoint_id: str
    donor_global_risk: float
    incumbent_id: str
    incumbent_global_risk: float
    epsilon_r: float
    epsilon_source_ref: str
    #: Present only on a Dual gate. The raw per-allele risks above keep their meaning either way:
    #: ``donor_global_risk`` is always role A's raw risk, never a joint scalar.
    joint: JointComparison | None = None

    @property
    def margin(self) -> float:
        """The quantity the gate actually compared."""
        if self.joint is not None:
            return self.joint.margin
        return self.incumbent_global_risk - self.donor_global_risk

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason.value,
            "donor_endpoint_id": self.donor_endpoint_id,
            "donor_global_risk": self.donor_global_risk,
            "incumbent_id": self.incumbent_id,
            "incumbent_global_risk": self.incumbent_global_risk,
            "epsilon_r": self.epsilon_r,
            "epsilon_source_ref": self.epsilon_source_ref,
            "margin": self.margin,
            # Omitted when absent for the same reason the decision evidence omits its Dual block: a
            # legacy verdict's payload is digested, and an added null key is a new digest.
            **({} if self.joint is None else {"joint": self.joint.canonical_payload()}),
        }


def donor_gate(
    *,
    donor: Any,
    incumbent: LineageIncumbent,
    epsilon_r: float,
    epsilon_source_ref: str,
    joint: JointComparison | None = None,
) -> DonorGateVerdict:
    r"""``R_H(Y*_d) < R_H(I_d) - epsilon_R`` (PLAN §2.5, A.2), as a typed verdict.

    Strict, and against the LINEAGE incumbent rather than against the sibling pool: an endpoint that
    merely ranks first among four stochastic completions has beaten its siblings, not the design the
    lineage already holds.  ``epsilon_R`` is the frozen Head's own repeatability/precision floor, so
    a "win" inside it is instrument noise -- the gate exists to keep noise from opening ancestry.

    Refusals are VALUES.  A lineage with no admissible donor must be able to record
    ``stall_no_better_donor`` and keep its incumbent (PLAN §4.5); raising here would turn an ordinary
    scientific outcome into a crash the cohort runner cannot aggregate.  Wiring mistakes -- a donor
    for another protein, another evaluator, another length -- still raise, because they are not
    outcomes.
    """
    if not isinstance(incumbent, LineageIncumbent):
        raise V2RewardError("incumbent must be a LineageIncumbent")
    epsilon = _finite(epsilon_r, "epsilon_r")
    if epsilon < 0.0:
        raise V2RewardError(
            f"epsilon_r must be non-negative, got {epsilon}; a negative tolerance would ADMIT a "
            "donor that is worse than the incumbent"
        )
    require_digest(epsilon_source_ref, "epsilon_source_ref")
    if donor.protein_id != incumbent.protein_id:
        raise IncumbentIdentityError(
            f"donor protein {donor.protein_id!r} != incumbent protein {incumbent.protein_id!r}"
        )
    if int(donor.sequence_length) != int(incumbent.sequence_length):
        raise IncumbentIdentityError(
            f"donor length {donor.sequence_length} != incumbent length "
            f"{incumbent.sequence_length}; the leave-one-out revert is undefined between them"
        )
    donor_evaluator = getattr(getattr(donor, "head_binding", None), "evaluator", None)
    if donor_evaluator is None or donor_evaluator.digest() != incumbent.head_identity_digest:
        raise IncumbentIdentityError(
            "the donor and the incumbent were scored by different Head evaluator identities; their "
            "risks are two instruments and their difference is not a margin"
        )

    if joint is not None and not isinstance(joint, JointComparison):
        raise V2RewardError("joint must be a JointComparison")

    # A joint gate compares NORMALIZED values, so the floor it compares against is the derived
    # joint margin, not the run config's raw-scale knob.  The verdict reports the number actually
    # used: a verdict that reported the raw knob would describe a comparison that never happened.
    compared_epsilon = epsilon if joint is None else float(joint.epsilon)

    def verdict(passed: bool, reason: DonorGateReason) -> DonorGateVerdict:
        return DonorGateVerdict(
            passed=passed, reason=reason, donor_endpoint_id=str(donor.endpoint_id),
            donor_global_risk=float(donor.head_global_risk), incumbent_id=incumbent.incumbent_id,
            incumbent_global_risk=float(incumbent.head_global_risk),
            epsilon_r=compared_epsilon,
            epsilon_source_ref=epsilon_source_ref, joint=joint,
        )

    from .state import FeasibilityLevel

    if getattr(donor, "feasibility_level", None) is not FeasibilityLevel.DEFINITIVE:
        return verdict(False, DonorGateReason.DONOR_NOT_DEFINITIVE)
    if donor.sequence_md5 == incumbent.sequence_md5:
        return verdict(False, DonorGateReason.DONOR_IS_INCUMBENT)
    # One law, two scalars. Which scalar is compared is the ONLY thing the Dual overlay changes
    # here; feasibility, self-comparison, protein, length and evaluator identity are unchanged, and
    # a lineage with no admissible donor still stalls rather than adopting the least-bad endpoint.
    if joint is None:
        challenger, held = float(donor.head_global_risk), float(incumbent.head_global_risk)
    else:
        challenger, held = joint.donor_value, joint.incumbent_value
    if challenger < held - compared_epsilon:
        return verdict(True, DonorGateReason.ACCEPTED)
    return verdict(False, DonorGateReason.STALL_NO_BETTER_DONOR)


def advance_incumbent(
    *,
    incumbent: LineageIncumbent,
    donor: Any,
    verdict: DonorGateVerdict,
    evaluator: HeadEvaluatorIdentity,
    accepted_at_depth: int,
    law: str,
) -> LineageIncumbent:
    """The incumbent update law: replace ``I_d`` only on a PASSING donor gate.

    "A worse endpoint must not overwrite the lineage reference" (PLAN §3.2 of the working note,
    §2.5 here), so a refused gate returns the SAME object -- by identity, not by copy, which is what
    makes "the incumbent did not move" checkable downstream rather than merely true today.
    """
    if law not in INCUMBENT_UPDATE_LAWS:
        raise V2RewardError(
            f"incumbent update law {law!r} is not declared; available: "
            f"{sorted(INCUMBENT_UPDATE_LAWS)}"
        )
    if not isinstance(verdict, DonorGateVerdict):
        raise V2RewardError("verdict must be the DonorGateVerdict this donor was gated by")
    if verdict.donor_endpoint_id != str(donor.endpoint_id):
        raise V2RewardError(
            f"the verdict names donor {verdict.donor_endpoint_id!r} but the donor supplied is "
            f"{donor.endpoint_id!r}; an incumbent advanced on another endpoint's verdict would "
            "record an improvement nothing measured"
        )
    if verdict.incumbent_id != incumbent.incumbent_id:
        raise V2RewardError(
            "the verdict was measured against a different incumbent than the one being advanced"
        )
    if not verdict.passed:
        return incumbent
    return bind_incumbent_from_endpoint(
        endpoint=donor, lineage_id=incumbent.lineage_id, evaluator=evaluator,
        safety_reference_sequence_md5=incumbent.safety_reference_sequence_md5,
        accepted_at_depth=int(accepted_at_depth),
    )
