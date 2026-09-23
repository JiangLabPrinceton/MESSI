r"""DUALF4: the joint leave-one-out contribution and the symmetric union reopen reducer.

Scientific authority: ``doc/Dual_Allele_Steering.md`` §2.5. Implementation contract:
``PLAN_RF_FUSION_V2_DUAL_ALLELE.md`` §§2.3, 2.4, DUALF4.

Two things happen here and they are the whole of the Dual actuation claim.

**One objective from selection to actuation.** The donor is chosen under :math:`J`, so the write
evidence must be :math:`a_i = J(y^{(-i)}) - J(y)` and not a per-allele contrast. Selecting under one
law and projecting under another is precisely the "Dual donor selector with a single-allele
actuator" degradation that hypothesis C3 exists to detect, and no artifact would record that it had
happened.

**The action space is a union, the budget is not.** A position enters the candidate space when
EITHER allele supplies the declared evidence; the two alleles never receive independent budgets, and
the shared reopen cardinality is unchanged. The forbidden shape is "screen on A, then score jointly":
it silently discards positions useful only to B while every downstream number continues to look
joint.

**Why the reducer needs its own coordinates.** The three Head-derived reopen conjuncts are
per-WINDOW readings, while ``risk`` is calibrated on the log-mean-exp aggregate of those windows.
Those are two different distributions on one raw scale, so the aggregate's location and scale do not
transfer. Two of the conjuncts are also DIFFERENCES (``[max_w delta_w]_+``) and one is an absolute
LEVEL (``max_w z_w(donor)``); a difference is divided by the scale alone, because the location
cancels in any difference and re-applying it would turn a zero difference into a non-zero
coordinate.

**Purity.** stdlib plus the ``fusion_v2`` leaves only. No torch, no config, no I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .errors import V2Error
from .evidence import (
    AlignedWindowEvidence,
    LeaveOneOutContribution,
    V2EvidenceError,
    counterfactual_sequence,
    score_leave_one_out,
)
from .identity import HeadEvaluatorIdentity, canonical_digest
from .joint_objective import AlleleRole, DualObjective, QuantityCoordinates

__all__ = [
    "DualReopenEvidence",
    "DualDecisionEvidence",
    "V2DualPolicyError",
    "PairedWindowEvidence",
    "UnionConjunct",
    "JointContribution",
    "score_joint_leave_one_out",
    "DualSupportAuthority",
]


class V2DualPolicyError(V2Error):
    """A joint write, union reopen, or paired counterfactual contract was violated."""


@dataclass(frozen=True)
class UnionConjunct:
    """One reopen conjunct reduced across the two alleles, keeping both values and the winner.

    ``value`` is ``None`` when NEITHER allele covers the position. Absent evidence is not zero: "no
    window reaches here" and "this position carries zero burden" are opposite facts about the Head's
    reach, and collapsing them would let an unreachable position outrank a measured-clean one.
    """

    value: float | None
    value_a: float | None
    value_b: float | None
    winning_allele: AlleleRole | None

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "winning_allele": None if self.winning_allele is None else self.winning_allele.value,
        }


def _reduce(value_a: float | None, value_b: float | None) -> UnionConjunct:
    """Max over the alleles that have evidence; role A breaks an exact tie, declared.

    Symmetric under allele swap in the VALUE, which is what decides the ordering. Only the winner
    label is asymmetric, and only on an exact numerical tie -- so an allele-swap replay changes the
    label and nothing that ranks.
    """
    if value_a is None and value_b is None:
        return UnionConjunct(value=None, value_a=None, value_b=None, winning_allele=None)
    if value_b is None:
        return UnionConjunct(value=value_a, value_a=value_a, value_b=None,
                             winning_allele=AlleleRole.A)
    if value_a is None:
        return UnionConjunct(value=value_b, value_a=None, value_b=value_b,
                             winning_allele=AlleleRole.B)
    winner = AlleleRole.A if value_a >= value_b else AlleleRole.B
    return UnionConjunct(value=max(value_a, value_b), value_a=value_a, value_b=value_b,
                         winning_allele=winner)


@dataclass(frozen=True)
class PairedWindowEvidence:
    """The two alleles' aligned window evidence, exposed as ONE union view on a shared coordinate.

    Deliberately mirrors the accessor names of :class:`AlignedWindowEvidence` so the policy consumes
    a union exactly where it consumed a single allele, rather than growing a second branch that
    could drift.
    """

    a: AlignedWindowEvidence
    b: AlignedWindowEvidence
    coordinates: QuantityCoordinates

    def __post_init__(self) -> None:
        for slot, side in (("a", self.a), ("b", self.b)):
            if not isinstance(side, AlignedWindowEvidence):
                raise V2DualPolicyError(f"{slot} must be an AlignedWindowEvidence")
        if not isinstance(self.coordinates, QuantityCoordinates):
            raise V2DualPolicyError("coordinates must be a QuantityCoordinates")
        if self.coordinates.quantity != "window_z":
            raise V2DualPolicyError(
                f"the union reducer needs per-window coordinates, got "
                f"{self.coordinates.quantity!r}; the aggregate risk scale does not transfer to the "
                "windows it summarizes"
            )
        if self.a.donor_sequence_md5 != self.b.donor_sequence_md5:
            raise V2DualPolicyError(
                "the two alleles scored different donor sequences; a union over two molecules is "
                "not a union"
            )
        if self.a.sequence_length != self.b.sequence_length:
            raise V2DualPolicyError("the two alleles disagree on the donor length")
        if self.a.head_identity_digest == self.b.head_identity_digest:
            raise V2DualPolicyError(
                "both sides carry the same Head identity digest; one allele has been supplied twice"
            )

    # -- union screen ---------------------------------------------------------------------------

    def covered(self, position: int) -> bool:
        """Covered when EITHER allele's grid reaches the position."""
        return self.a.covered(position) or self.b.covered(position)

    def improves_at(self, position: int) -> bool:
        """PLAN §2.3's union screen: an improved window under A **or** under B qualifies.

        This is the single line that separates real Dual actuation from a Dual donor selector with a
        single-allele actuator. Requiring both, or screening on A first, discards exactly the
        positions that only the second Head can see.
        """
        return self.a.improves_at(position) or self.b.improves_at(position)

    def covered_by(self, position: int) -> tuple[bool, bool]:
        return (self.a.covered(position), self.b.covered(position))

    # -- normalized conjuncts -------------------------------------------------------------------

    def worsening_union(self, position: int) -> UnionConjunct:
        """``[max_w delta_w]_+`` per allele, as a DIFFERENCE in the shared coordinate."""
        return _reduce(
            self.coordinates.normalize_difference(AlleleRole.A, self.a.worsening_at(position))
            if self.a.covered(position) else None,
            self.coordinates.normalize_difference(AlleleRole.B, self.b.worsening_at(position))
            if self.b.covered(position) else None,
        )

    def residual_burden_union(self, position: int) -> UnionConjunct:
        """``max_w z_w(donor)`` per allele, as an absolute LEVEL in the shared coordinate."""
        left = self.a.residual_burden_at(position)
        right = self.b.residual_burden_at(position)
        return _reduce(
            None if left is None else self.coordinates.normalize_level(AlleleRole.A, left),
            None if right is None else self.coordinates.normalize_level(AlleleRole.B, right),
        )

    def min_delta_union(self, position: int) -> UnionConjunct:
        """The most-improved window per allele, normalized as a difference.

        Reduced by MIN rather than max -- more negative is more improvement -- so the reducer is
        applied to the negated values and the sign restored, keeping one comparison rule.
        """
        left = self.a.min_delta_at(position)
        right = self.b.min_delta_at(position)
        flipped = _reduce(
            None if left is None
            else -self.coordinates.normalize_difference(AlleleRole.A, left),
            None if right is None
            else -self.coordinates.normalize_difference(AlleleRole.B, right),
        )
        return UnionConjunct(
            value=None if flipped.value is None else -flipped.value,
            value_a=None if flipped.value_a is None else -flipped.value_a,
            value_b=None if flipped.value_b is None else -flipped.value_b,
            winning_allele=flipped.winning_allele,
        )

    # -- drop-in scalar accessors -----------------------------------------------------------
    #
    # Same names, same signatures and same absent-value semantics as AlignedWindowEvidence, so the
    # policy consumes a union exactly where it consumed one allele. A second code path in select()
    # is the thing most likely to drift out of agreement with the single-Head law it must remain
    # equivalent to when Dual is off, so there is not one.

    def worsening_at(self, position: int) -> float:
        value = self.worsening_union(position).value
        return 0.0 if value is None else float(value)

    def residual_burden_at(self, position: int) -> float | None:
        return self.residual_burden_union(position).value

    def min_delta_at(self, position: int) -> float | None:
        return self.min_delta_union(position).value

    @property
    def evidence_digest(self) -> str:
        """Content identity of the PAIR, so a stall record names both comparisons it saw."""
        return canonical_digest({
            "a": self.a.evidence_digest,
            "b": self.b.evidence_digest,
            "coordinates": self.coordinates.canonical_payload(),
        })

    @property
    def donor_sequence_md5(self) -> str:
        return self.a.donor_sequence_md5

    @property
    def sequence_length(self) -> int:
        return self.a.sequence_length


# ----------------------------------------------------------------------------------------------
# the joint leave-one-out
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class JointContribution:
    r"""``a_i = J(y^{(-i)}) - J(y)`` for one position, with both per-allele contrasts retained.

    The per-allele contrasts are derived telemetry, not a second controller state: they explain
    whether a selected identity is shared-beneficial, worst-allele-specific or conflicting, and that
    classification is a reading of the run rather than an input to it.
    """

    position: int
    donor_residue: str
    incumbent_residue: str
    counterfactual_sequence_md5: str
    raw_a: float
    raw_b: float
    contribution_a: float
    contribution_b: float
    joint_donor_value: float
    joint_counterfactual_value: float

    @property
    def contribution(self) -> float:
        return self.joint_counterfactual_value - self.joint_donor_value

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "donor_residue": self.donor_residue,
            "incumbent_residue": self.incumbent_residue,
            "counterfactual_sequence_md5": self.counterfactual_sequence_md5,
            "raw_a": self.raw_a,
            "raw_b": self.raw_b,
            "contribution_a": self.contribution_a,
            "contribution_b": self.contribution_b,
            "joint_donor_value": self.joint_donor_value,
            "joint_counterfactual_value": self.joint_counterfactual_value,
            "contribution": self.contribution,
        }


def score_joint_leave_one_out(
    *,
    scorer_a: Any,
    scorer_b: Any,
    protein_id: str,
    donor_sequence: str,
    donor_raw_a: float,
    donor_raw_b: float,
    incumbent_sequence: str,
    positions: Sequence[int],
    evaluator_a: HeadEvaluatorIdentity,
    evaluator_b: HeadEvaluatorIdentity,
    window_grid_digest: str,
    objective: DualObjective,
) -> dict[int, JointContribution]:
    """One counterfactual set, two frozen Heads, one joint contribution per position.

    The reverted sequences are built ONCE here and the identity of that set is then required to hold
    for both allele batches. Each allele still gets exactly one Head batch over exactly that set --
    which is the cost that matters -- and a batch that covered a different set is a refusal rather
    than a silently half-joint attribution.

    The revert target is the CURRENT LINEAGE INCUMBENT, unchanged from the single-Head law. Reverting
    toward the source or toward WT would measure a different counterfactual than the donor gate
    admitted the donor on.
    """
    if not isinstance(objective, DualObjective) or objective.quantity != "global_risk":
        raise V2DualPolicyError(
            "the joint write law is defined on the steering objective; a density or window view "
            "cannot attribute a write"
        )
    if evaluator_a.head_checkpoint_digest == evaluator_b.head_checkpoint_digest:
        raise V2DualPolicyError("both roles carry the same Head checkpoint digest")

    ordered = [int(position) for position in positions]
    if len(set(ordered)) != len(ordered):
        raise V2DualPolicyError("a position was shortlisted twice for the counterfactual batch")
    if not ordered:
        raise V2DualPolicyError("the joint counterfactual batch needs at least one position")

    # Built once. Each allele's batch must then cover exactly this set of bytes.
    try:
        expected_md5 = {
            position: _md5_of(counterfactual_sequence(donor_sequence, incumbent_sequence, position))
            for position in ordered
        }
    except V2EvidenceError as exc:
        raise V2DualPolicyError(f"the counterfactual set could not be built: {exc}") from exc

    per_allele: dict[AlleleRole, dict[int, LeaveOneOutContribution]] = {}
    for role, scorer, donor_raw, evaluator in (
        (AlleleRole.A, scorer_a, donor_raw_a, evaluator_a),
        (AlleleRole.B, scorer_b, donor_raw_b, evaluator_b),
    ):
        try:
            per_allele[role] = score_leave_one_out(
                scorer=scorer, protein_id=protein_id, donor_sequence=donor_sequence,
                donor_global_risk=float(donor_raw), incumbent_sequence=incumbent_sequence,
                positions=ordered, evaluator=evaluator,
                window_grid_digest=window_grid_digest)
        except V2EvidenceError as exc:
            raise V2DualPolicyError(
                f"the role {role.value} counterfactual batch could not be bound: {exc}") from exc

    for role, rows in per_allele.items():
        for position in ordered:
            row = rows.get(position)
            if row is None:
                raise V2DualPolicyError(
                    f"role {role.value} returned no counterfactual for position {position}; a "
                    "half-covered batch would attribute a joint write from one allele's evidence"
                )
            if row.counterfactual_sequence_md5 != expected_md5[position]:
                raise V2DualPolicyError(
                    f"role {role.value} scored a different counterfactual at position {position} "
                    f"({row.counterfactual_sequence_md5} vs {expected_md5[position]}); the two "
                    "alleles must judge the same reverted bytes or the difference is not an "
                    "attribution"
                )

    donor_joint = objective.evaluate(raw_a=donor_raw_a, raw_b=donor_raw_b)
    out: dict[int, JointContribution] = {}
    for position in ordered:
        left = per_allele[AlleleRole.A][position]
        right = per_allele[AlleleRole.B][position]
        counterfactual = objective.evaluate(
            raw_a=left.counterfactual_global_risk, raw_b=right.counterfactual_global_risk)
        out[position] = JointContribution(
            position=position,
            donor_residue=left.donor_residue,
            incumbent_residue=left.incumbent_residue,
            counterfactual_sequence_md5=expected_md5[position],
            raw_a=float(left.counterfactual_global_risk),
            raw_b=float(right.counterfactual_global_risk),
            contribution_a=float(left.contribution),
            contribution_b=float(right.contribution),
            joint_donor_value=float(donor_joint.value),
            joint_counterfactual_value=float(counterfactual.value),
        )
    return out


def _md5_of(sequence: str) -> str:
    from inverse_folding.reference_flow.fusion.state import sequence_md5

    return sequence_md5(sequence)


@dataclass(frozen=True)
class DualReopenEvidence:
    """One reopen candidate's per-allele conjuncts, before the union reducer collapsed them.

    The policy consumes only the reduced value, which is correct -- the union view is a drop-in for
    the single-allele one so ``select()`` needs no second branch. But the reduced value alone cannot
    answer the question the reopen artifact exists for: WHICH allele demanded this position. Without
    the two sides and the winner, a joint run's reopen table is indistinguishable from a single-Head
    run's.
    """

    position: int
    new_hotspot: "UnionConjunct"
    worsening: "UnionConjunct"
    residual_burden: "UnionConjunct"

    @property
    def ordering_conjunct(self) -> str:
        """Which conjunct actually ordered this position, under the frozen reopen priority.

        The SAME predicate the legacy ``_reopen_reason`` applies: new-hotspot and worsening order
        the row only when they are TRUTHY, while residual burden orders on mere presence. Selecting
        the first merely-present conjunct instead named ``new_hotspot`` for every covered position,
        including the ones whose new-hotspot value is exactly 0.0 -- so the attribution column
        reported role A's tie-break on a zero while role B had set the value that did the ordering.
        """
        if self.new_hotspot.value:
            return "new_hotspot"
        if self.worsening.value:
            return "worsening"
        if self.residual_burden.value is not None:
            return "residual_burden"
        return ""

    @property
    def winning_allele(self) -> str:
        """The allele that set the value of the conjunct that actually ordered this position."""
        name = self.ordering_conjunct
        if not name:
            return ""
        conjunct = getattr(self, name)
        return "" if conjunct.winning_allele is None else conjunct.winning_allele.value


@dataclass(frozen=True)
class DualDecisionEvidence:
    """Everything the joint law computed that the single-allele record has no field for.

    Attached to the decision evidence rather than folded into it: the legacy rows keep their legacy
    meaning byte for byte, and a Dual-off decision carries no such object at all.
    """

    arm: str
    objective_digest: str
    #: The floor ``a_i`` was actually filtered against on this cycle -- the arm objective's own
    #: derived margin. Recorded because the artifact recomputed eligibility against 0.0 and
    #: therefore contradicted the decision it claimed to be reading off.
    write_epsilon: float = 0.0
    contributions: tuple["JointContribution", ...] = ()
    reopen: tuple[DualReopenEvidence, ...] = ()

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "objective_digest": self.objective_digest,
            "write_epsilon": self.write_epsilon,
            "contributions": [row.canonical_payload() for row in self.contributions],
            "reopen": [
                {"position": row.position,
                 "new_hotspot": row.new_hotspot.canonical_payload(),
                 "worsening": row.worsening.canonical_payload(),
                 "residual_burden": row.residual_burden.canonical_payload(),
                 "ordering_conjunct": row.ordering_conjunct,
                 "winning_allele": row.winning_allele}
                for row in self.reopen
            ],
        }


# ----------------------------------------------------------------------------------------------
# what the head-directed policy needs in order to act jointly
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DualSupportAuthority:
    """The injected second half of a joint support decision.

    Everything role B contributes arrives here as a value, so the policy itself gains no second code
    path for "how do I find the other Head" -- it either holds an authority and acts jointly, or
    holds none and behaves exactly as the frozen single-Head law does. That is what makes Dual-off
    equivalence a structural property rather than a tested coincidence.

    ``incumbent_joint_value`` is supplied rather than derived because at depth zero the lineage
    incumbent may be the immutable safety reference rather than an endpoint of this run, in which
    case it has no sidecar row -- and both Heads must have scored that reference for a joint gate to
    mean anything. Requiring it here keeps that requirement visible instead of letting a missing
    reference score default to zero.
    """

    objective: DualObjective
    window_coordinates: QuantityCoordinates
    evaluator_b: HeadEvaluatorIdentity
    counterfactual_scorer_b: Any
    incumbent_score_b: Any
    incumbent_joint_value: float
    safety_reference_score_b: Any
    donor_score_b_by_endpoint: Mapping[str, Any]
    #: The per-cycle CANDIDATE ceiling, in editable positions, from the signed overlay. Never the
    #: legacy Head-call field: charging that in Head calls would halve the editable domain because
    #: a second Head exists. The logical Head-call budget is 2x this and is the preflight's job.
    max_counterfactual_sequences_per_cycle: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.objective, DualObjective) \
                or self.objective.quantity != "global_risk":
            raise V2DualPolicyError(
                "the support authority steers on the global-risk objective; a density or window "
                "view cannot decide a donor or a write"
            )
        if not isinstance(self.window_coordinates, QuantityCoordinates) \
                or self.window_coordinates.quantity != "window_z":
            raise V2DualPolicyError(
                "window_coordinates must be the per-window pair; the aggregate risk scale does not "
                "transfer to the windows it summarizes"
            )
        if not isinstance(self.evaluator_b, HeadEvaluatorIdentity):
            raise V2DualPolicyError("evaluator_b must be a HeadEvaluatorIdentity")
        role_b = self.objective.coordinates.coordinate(AlleleRole.B).evaluator
        if self.evaluator_b != role_b:
            raise V2DualPolicyError(
                f"evaluator_b is {self.evaluator_b.allele!r}/"
                f"{self.evaluator_b.head_checkpoint_digest[:12]} but role B is calibrated for "
                f"{role_b.allele!r}/{role_b.head_checkpoint_digest[:12]}"
            )
        if not callable(self.counterfactual_scorer_b):
            raise V2DualPolicyError("counterfactual_scorer_b must be callable")
        ceiling = self.max_counterfactual_sequences_per_cycle
        if isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling < 1:
            raise V2DualPolicyError(
                f"max_counterfactual_sequences_per_cycle must be a positive int, got {ceiling!r}; "
                "it is the candidate domain one Dual cycle may consider and has no defensible "
                "default -- a zero would stall every transition and a silent reuse of the legacy "
                "Head-call field would halve the domain"
            )
        for name in ("incumbent_score_b", "safety_reference_score_b"):
            score = getattr(self, name)
            if getattr(score, "allele", None) != self.evaluator_b.allele:
                raise V2DualPolicyError(
                    f"{name} was scored for allele {getattr(score, 'allele', None)!r}, role B "
                    f"declares {self.evaluator_b.allele!r}"
                )
        value = self.incumbent_joint_value
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            raise V2DualPolicyError(
                f"incumbent_joint_value must be a finite real number, got {value!r}"
            )
        if not isinstance(self.donor_score_b_by_endpoint, Mapping):
            raise V2DualPolicyError("donor_score_b_by_endpoint must be a mapping")

    def joint_comparison(self, *, donor: Any) -> "JointComparison":
        """The ``(donor J, incumbent J)`` pair the donor gate compares, bound to this calibration."""
        from .reward import JointComparison as _JointComparison

        raw_b = getattr(self.donor_score_b(donor), "global_risk", None)
        if raw_b is None:
            raise V2DualPolicyError("role B's donor score carries no global_risk")
        value = self.objective.evaluate(
            raw_a=float(getattr(donor, "head_global_risk")), raw_b=float(raw_b))
        coordinates = self.objective.coordinates
        return _JointComparison(
            donor_value=float(value.value),
            incumbent_value=float(self.incumbent_joint_value),
            objective_digest=self.objective.objective_digest,
            # PLAN 2.1: derived from the objective's own geometry, with the per-allele raw floors
            # it came from recorded alongside it.
            # The DECISION margin: 2 * (this arm's own propagated floor). The gate compares a
            # DIFFERENCE of two joint values, each carrying its own drift, so the single
            # measurement bound is too permissive by exactly a factor of two. And the arm's own
            # floor, not the pair's -- reading the pair would hold a single-allele arm to a
            # threshold set by an instrument it does not steer on.
            epsilon=float(self.objective.decision_margin),
            raw_floor_a=float(coordinates.a.raw_noise_floor),
            raw_floor_b=float(coordinates.b.raw_noise_floor),
        )

    def paired_windows(self, *, donor_score_a: Any, donor_score_b: Any, reference_score_a: Any,
                       reference_score_b: Any, evaluator_a: HeadEvaluatorIdentity,
                       reference_label: str) -> "PairedWindowEvidence":
        """The union window view for one donor against one reference, on both alleles."""
        from .evidence import build_window_evidence

        return PairedWindowEvidence(
            a=build_window_evidence(
                donor_score=donor_score_a, reference_score=reference_score_a,
                evaluator=evaluator_a, reference_label=reference_label),
            b=build_window_evidence(
                donor_score=donor_score_b, reference_score=reference_score_b,
                evaluator=self.evaluator_b, reference_label=reference_label),
            coordinates=self.window_coordinates,
        )

    def donor_score_b(self, donor: Any) -> Any:
        """Role B's score of one donor, looked up by endpoint id and PROVED to be that molecule.

        Never a fallback: a donor only one Head has seen cannot be judged jointly, and substituting
        role A's score would make the union view compare an allele against itself.

        The map is authored outside this object, so the key alone proves nothing. A mis-keyed entry
        would compute $J$ from another sequence's role-B risk and the gate would report a margin
        between two molecules -- a wrong number with no wrong-looking field anywhere in the record.
        """
        endpoint_id = str(getattr(donor, "endpoint_id", donor))
        score = self.donor_score_b_by_endpoint.get(endpoint_id)
        if score is None:
            raise V2DualPolicyError(
                f"role B has no score for donor {endpoint_id}; a joint support decision cannot be "
                "taken on an endpoint only one Head has seen"
            )
        self.verify_role_b_score(score, against=donor, what=f"donor {endpoint_id}")
        return score

    def verify_role_b_score(self, score: Any, *, against: Any, what: str) -> None:
        """Prove one role-B score describes the same molecule, allele and window grid as ``against``.

        ``against`` is any object carrying the legacy identity fields -- a ``CompleteEndpoint`` or a
        ``LineageIncumbent``. Both are role A's view; this is the one place the two views are
        forced to agree.
        """
        for field in ("protein_id", "sequence_md5", "sequence_length"):
            anchor = getattr(against, field, None)
            declared = getattr(score, field, None)
            if anchor is None:
                raise V2DualPolicyError(f"{what}: role A's record carries no {field}")
            if declared != anchor:
                raise V2DualPolicyError(
                    f"{what}: the role B score declares {field}={declared!r} but role A declares "
                    f"{anchor!r}; a joint value is defined for two alleles on ONE exact sequence, "
                    "and pairing across sequences compares two molecules"
                )
        allele = getattr(score, "allele", None)
        if allele != self.evaluator_b.allele:
            raise V2DualPolicyError(
                f"{what}: the role B score was produced for allele {allele!r} but role B is "
                f"{self.evaluator_b.allele!r}"
            )
        binding = getattr(against, "head_binding", None)
        expected_grid = getattr(binding, "window_grid_digest", None)
        windows = getattr(score, "windows", None)
        if expected_grid is not None and windows is not None:
            from .identity import window_grid_digest as _grid

            if _grid(windows) != expected_grid:
                raise V2DualPolicyError(
                    f"{what}: the role B score is on a different window grid than role A; the "
                    "union view would align two coordinate systems position by position"
                )

    def assert_incumbent(self, incumbent: Any) -> None:
        """Prove role B's half of the lineage incumbent is the same design, at the same value.

        ``incumbent_joint_value`` used to be a free parameter: any finite float was accepted as
        $J(I_d)$, so a stale or hand-set value would silently move the donor gate's threshold.
        Here it is checked against the objective's own value on the incumbent this policy holds.
        """
        self.verify_role_b_score(
            self.incumbent_score_b, against=incumbent, what="the lineage incumbent")
        raw_a = getattr(incumbent, "head_global_risk", None)
        if raw_a is None:
            raise V2DualPolicyError("the incumbent carries no head_global_risk")
        derived = self.objective.evaluate(
            raw_a=float(raw_a),
            raw_b=float(getattr(self.incumbent_score_b, "global_risk"))).value
        if float(self.incumbent_joint_value) != float(derived):
            raise V2DualPolicyError(
                f"incumbent_joint_value is {self.incumbent_joint_value!r} but this objective's own "
                f"value on the incumbent is {derived!r}; J(I_d) is derived from the two Heads' "
                "risks, not declared, or the donor gate compares against a threshold nothing "
                "measured"
            )

    def advanced(self, *, donor: Any) -> "DualSupportAuthority":
        """Role B's half of ``advance_lineage_incumbent``: adopt ``donor`` as the new incumbent.

        The policy is immutable and advances by replacing its incumbent, which carried this object
        through UNCHANGED -- role A moved to the adopted donor while role B stayed at $I_0$, so from
        the next depth on the gate compared a fresh $J(Y^*)$ against a stale $J(I_0)$ and the union
        windows referenced two different sequences.

        The donor score map is CLEARED rather than carried: the next depth's donors are different
        endpoints, and a surviving entry would let a stale score be found where the correct
        behaviour is to raise for a donor role B has not scored yet.
        """
        score_b = self.donor_score_b(donor)
        return replace(
            self,
            incumbent_score_b=score_b,
            incumbent_joint_value=float(self.objective.evaluate(
                raw_a=float(getattr(donor, "head_global_risk")),
                raw_b=float(getattr(score_b, "global_risk"))).value),
            donor_score_b_by_endpoint={},
        )
