"""The whole-landscape safety admission gate, wired into the ancestry decision (PLAN §2.7, §4.2).

:mod:`fusion_v2.safety` implements the science: a cumulative reference frozen at depth 0, an exact
window-aligned ``N_H^whole`` comparator, a calibrated threshold, and an admission conjunction that
fails closed.  This module is the only thing that connects it to a running cycle.

**Why that connection is load-bearing.**  Structural feasibility and immune safety answer different
questions.  A design can fold perfectly and still open a new epitope hotspot the reference never
had -- and because feedback ancestry is *recursive*, admitting one puts that hotspot into the
parent of everything downstream, where the cumulative ratchet can no longer see it as new.  So
ancestry eligibility is a CONJUNCTION, and a cycle that promoted on the structure verdict alone
would be running V2's mechanism with its safety half disconnected while still reporting the
"definitive" label that is supposed to mean both.

The gate is a required argument of the cycle, not an optional one.  An optional safety gate defaults
to "off", and a run with it off is indistinguishable in its outputs from a run with it on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..fusion_v2.errors import V2Error
from ..fusion_v2.identity import SafetyReferenceBinding
from ..fusion_v2.safety import (
    AdmissibilityVerdict,
    LineageSafetyLedger,
    SafetyAdmissionPolicy,
    SelfReferentialReference,
    apply_threshold,
    make_admissibility_verdict,
    measure_cumulative,
    incremental_inapplicable,
    measure_incremental,
)

__all__ = ["V2AdmissionError", "SafetyGate", "EndpointAdmission", "admit_endpoint"]


class V2AdmissionError(V2Error):
    """The admission gate was wired wrongly -- a caller mistake, not a design verdict."""


@dataclass(frozen=True)
class SafetyGate:
    """One lineage's admission policy and its ratchet state, bound together.

    The two must come from the same place: :func:`~fusion_v2.safety.open_lineage_ledger` builds the
    ledger from a reference that already carries its policy, so a gate assembled from a policy and
    an unrelated ledger would apply one run's threshold to another run's reference.
    """

    policy: SafetyAdmissionPolicy
    ledger: LineageSafetyLedger

    def __post_init__(self) -> None:
        if not isinstance(self.policy, SafetyAdmissionPolicy):
            raise V2AdmissionError("policy must be a SafetyAdmissionPolicy")
        if not isinstance(self.ledger, LineageSafetyLedger):
            raise V2AdmissionError("ledger must be a LineageSafetyLedger")
        if self.ledger.cumulative_reference.policy is not self.policy:
            raise V2AdmissionError(
                "the ledger's cumulative reference was bound under a DIFFERENT admission policy; "
                "carrying the reference by object identity is what makes the ratchet inescapable, "
                "and pairing it with another policy would re-base the threshold it was calibrated "
                "against"
            )
        # NOTE: an enabled incremental gate at depth 0 is NOT a misconfiguration.  This used to
        # raise, which made every legal config carrying ``incremental_gate_enabled: true``
        # unlaunchable -- the ledger a depth-0 gate is built from always has
        # ``immediate_parent=None``, and there is no way to advance it before depth 0 exists.  The
        # gate simply has no referent yet; ``admit_endpoint`` records that as a typed
        # ``IncrementalInapplicable`` fact and the gate becomes binding from depth 1.

    @property
    def reference_binding(self) -> SafetyReferenceBinding:
        """The depth-0 ``SafetyReferenceBinding`` a state stamps and an artifact records.

        Read off the reference that DECIDES ancestry rather than built beside it (Interface Map §3
        C5).  ``LivePartialState.safety_reference`` and ``CumulativeSafetyReference`` are one
        concept split by layer -- identity on the state, identity plus the live Head score here --
        so a second construction of the identity is a second reference that can disagree with the
        one every design was actually measured against, while both stay individually well-formed.
        """
        return self.ledger.cumulative_reference.binding


@dataclass(frozen=True)
class EndpointAdmission:
    """One endpoint's admission decision plus the evidence behind it."""

    endpoint_id: str
    admitted: bool
    verdict: AdmissibilityVerdict | None
    reason: str

    @property
    def structure_only(self) -> bool:
        """True when nothing but the structure verdict was available to decide on."""
        return self.verdict is None


def admit_endpoint(
    gate: SafetyGate,
    *,
    endpoint: Any,
    structure_definitive: bool,
    endpoint_tokens: Sequence[int],
    hard_anchors: Sequence[tuple[int, int]],
) -> EndpointAdmission:
    """Decide whether one scored endpoint may become feedback ancestry.

    ``constraints_preserved`` is MEASURED here from the endpoint's own tokens against the source's
    hard anchors rather than asserted by the caller: it is one of the three conjuncts, and a conjunct
    supplied as a constant is not a gate.
    """
    if not isinstance(gate, SafetyGate):
        raise V2AdmissionError("gate must be a SafetyGate")
    tokens = tuple(int(token) for token in endpoint_tokens)
    constraints_preserved = all(
        tokens[int(position)] == int(token) for position, token in hard_anchors
    )

    try:
        cumulative_evidence = measure_cumulative(
            gate.ledger, endpoint.head_score, endpoint_id=endpoint.endpoint_id,
        )
    except SelfReferentialReference as exc:
        # The design IS the frozen reference.  ``N_H(y; y) == 0`` would make the gate vacuous, so
        # the comparator refuses to compute it -- and refusing ANCESTRY is the fail-closed reading:
        # feeding the reference back as its own parent teaches the lineage nothing, and the one
        # thing that must never happen is an unmeasured design acquiring the definitive label.
        return EndpointAdmission(
            endpoint_id=endpoint.endpoint_id, admitted=False, verdict=None,
            reason=f"design is the cumulative reference itself, so the gate cannot measure it: "
                   f"{exc}",
        )
    cumulative = apply_threshold(cumulative_evidence, gate.policy.cumulative_threshold)

    incremental = None
    if gate.policy.incremental_gate_enabled:
        if gate.ledger.immediate_parent is None:
            # Depth 0 has nothing to compare against.  Recorded as a typed FACT minted from this
            # ledger rather than left as ``None``, which would be indistinguishable from a gate
            # that was simply skipped -- and the marker cannot be forged for a lineage that does
            # have a parent, so it can never launder a skipped measurement.
            incremental = incremental_inapplicable(gate.ledger)
        else:
            incremental = apply_threshold(
                measure_incremental(gate.ledger, endpoint.head_score,
                                    endpoint_id=endpoint.endpoint_id),
                gate.policy.incremental_threshold,
            )

    verdict = make_admissibility_verdict(
        policy=gate.policy, cumulative=cumulative, incremental=incremental,
        structure_definitive=bool(structure_definitive),
        structure_evidence_level="definitive" if structure_definitive else "unvalidated",
        constraints_preserved=bool(constraints_preserved),
    )
    if verdict.admitted:
        reason = "admitted"
    else:
        failed = []
        if not structure_definitive:
            failed.append("structure not definitively feasible")
        if not constraints_preserved:
            failed.append("hard anchors not preserved")
        if not cumulative.passed:
            failed.append(
                f"whole-landscape new hotspot {cumulative.evidence.max_increase:.4f} exceeds the "
                f"cumulative threshold {cumulative.threshold.value:.4f}")
        if incremental is not None and not incremental.passed:
            failed.append(
                f"incremental new hotspot {incremental.evidence.max_increase:.4f} exceeds "
                f"{incremental.threshold.value:.4f}")
        reason = "; ".join(failed)
    return EndpointAdmission(
        endpoint_id=endpoint.endpoint_id, admitted=verdict.admitted, verdict=verdict,
        reason=reason,
    )
