"""DUALF3: one frozen objective serving family selection, archive elite, incumbent and donor gate.

Implementation contract: ``PLAN_RF_FUSION_V2_DUAL_ALLELE.md`` §2.2, DUALF3.

The four surfaces that decide which endpoint becomes ancestry are independently constructed and each
reached for ``head_global_risk`` directly. Making them agree is not a refactor for tidiness: a run
whose family representative is chosen under one law and whose donor gate is applied under another
selects on a mixture of two objectives, and no artifact would record that it had.

The ordering is INJECTED, never detected. A resolver that fell back to the raw single-Head risk when
Dual evidence was missing would produce a joint run whose elite is quietly single-allele, so a
missing piece of evidence is a refusal here.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..fusion_v2.dual_evidence import DualEndpointEvidence, V2DualEvidenceError
from ..fusion_v2.reward import JointComparison

__all__ = ["dual_rank_key", "joint_comparison_for"]


def dual_rank_key(
    evidence_by_endpoint: Mapping[str, DualEndpointEvidence],
    *,
    quantity: str = "risk",
) -> Callable[[Any], tuple[float, str]]:
    """Build the ``(J, endpoint_id)`` ordering over a pool that already carries Dual evidence.

    Same shape as the legacy ``(head_global_risk, endpoint_id)`` key, so every surface that consumes
    an ordering consumes this one unchanged; only the scalar differs. Endpoint identity closes exact
    ties, so the ordering is a function of the SET of endpoints and not of their arrival order.
    """
    if quantity not in ("risk", "density"):
        raise V2DualEvidenceError(
            f"quantity must be 'risk' or 'density', got {quantity!r}"
        )
    digests = {evidence.risk.objective_digest for evidence in evidence_by_endpoint.values()}
    if len(digests) > 1:
        raise V2DualEvidenceError(
            f"the pool carries {len(digests)} different objective/calibration digests; endpoints ordered "
            "under two calibrations are ordered under two objectives, and the comparison between "
            "them is not a margin"
        )

    def key(endpoint: Any) -> tuple[float, str]:
        endpoint_id = str(getattr(endpoint, "endpoint_id"))
        evidence = evidence_by_endpoint.get(endpoint_id)
        if evidence is None:
            raise V2DualEvidenceError(
                f"endpoint {endpoint_id} carries no Dual evidence, so it cannot be ordered under "
                "the joint objective. Falling back to the raw single-Head risk here would make a "
                "joint run's elite silently single-allele while every artifact continued to claim "
                "otherwise"
            )
        value = evidence.risk if quantity == "risk" else evidence.density
        if value is None:
            raise V2DualEvidenceError(
                f"endpoint {endpoint_id} carries no {quantity} view of the joint objective"
            )
        return (float(value.value), endpoint_id)

    return key


def joint_comparison_for(
    *,
    donor: Any,
    incumbent_value: float,
    incumbent_objective_digest: str,
    evidence_by_endpoint: Mapping[str, DualEndpointEvidence],
    objective: Any,
) -> JointComparison:
    """The joint donor/incumbent pair the donor gate compares, with its identity checked.

    The incumbent's joint value is supplied rather than looked up because at depth zero the
    incumbent may be the immutable safety reference rather than an endpoint of this run -- in which
    case it has no endpoint id and no sidecar row, and both Heads must have scored the reference
    sequence for a joint gate to mean anything at all. Requiring the caller to produce it is what
    keeps that requirement visible instead of letting a missing reference score default to zero.
    """
    endpoint_id = str(getattr(donor, "endpoint_id"))
    evidence = evidence_by_endpoint.get(endpoint_id)
    if evidence is None:
        raise V2DualEvidenceError(
            f"donor {endpoint_id} carries no Dual evidence; a joint donor gate cannot be applied "
            "to an endpoint only one Head has seen"
        )
    if not (evidence.risk.objective_digest == incumbent_objective_digest
            == objective.objective_digest):
        raise V2DualEvidenceError(
            f"the donor was scored under calibration {evidence.calibration_digest[:12]} and the "
            f"incumbent under {str(incumbent_objective_digest)[:12]}; mixing objective versions "
            "inside one lineage means the improvement the gate measures was never measured"
        )
    coordinates = objective.coordinates
    return JointComparison(
        donor_value=float(evidence.risk.value),
        incumbent_value=float(incumbent_value),
        objective_digest=evidence.risk.objective_digest,
        # The DERIVED floor, exactly as ``DualSupportAuthority.joint_comparison`` produces it.
        # Left at the 0.0 default, this exported helper reproduced the very unit defect the gate
        # was repaired for -- a normalized comparison judged against no floor at all -- inside a
        # function the tests bless.
        epsilon=float(objective.decision_margin),
        raw_floor_a=float(coordinates.a.raw_noise_floor),
        raw_floor_b=float(coordinates.b.raw_noise_floor),
    )
