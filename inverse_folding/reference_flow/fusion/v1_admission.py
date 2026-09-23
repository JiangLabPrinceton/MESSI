"""V1F5 torch-free admission mapping and cohort coverage (PLAN_RF_REFINE_FUSION_V1 §2.11, §5).

The v0 admission decision is unchanged (ordered design_idx scan, definitive structure gate,
first ``N`` feasible rows). This layer makes that decision auditable WITHOUT reconstructing it
from sequences: it records design_idx -> structure verdict -> slot for every attempt, keyed by
root/continuation identity so convergent roots (same sequence, different lineage) stay
distinguishable. The structure gate itself is injected (the real v0 structure cache is wired in
by the driver), so this module stays pure and testable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FacadeCandidate:
    """One ordered complete facade row offered for v0 admission. ``source_id`` is the root_id
    (pre-terminal arm) or the complete-trajectory source id (terminal arm); ``continuation_id`` /
    ``root_equivalence_hash`` are None for a complete-control row."""

    design_idx: int
    seed: int
    sequence: str
    sequence_md5: str
    source_id: str
    continuation_id: str | None
    root_equivalence_hash: str | None


@dataclass(frozen=True)
class StructureOutcome:
    """Result of the definitive structure evaluation for one facade candidate (§2.11/§3.3): the
    gate PLUS the evidence the ledger and admission table need. The real v0 structure path returns
    these; a bare ``bool`` is coerced to ``StructureOutcome(feasible=bool)``.

    ``evaluated`` separates "structure said yes" from "structure was never run". The V1-A entry
    stage DEFERS definitive structure to the unchanged v0 admission, so its gate returns
    :meth:`deferred` -- and a deferred outcome may not carry a feasibility verdict, because every
    downstream table would then read an unevaluated candidate as one that passed.
    """

    feasible: bool | None
    cache_status: str = "miss"  # "hit" | "miss"
    model_executed: bool = True
    failure_reason: str | None = None
    walltime_s: float = 0.0
    metrics: Mapping[str, float] | None = None
    evaluated: bool = True

    def __post_init__(self) -> None:
        if self.evaluated:
            if not isinstance(self.feasible, bool):
                raise ValueError(
                    "an evaluated structure outcome must state a boolean verdict, got "
                    f"{self.feasible!r}"
                )
        else:
            if self.feasible is not None:
                raise ValueError(
                    f"a deferred structure outcome must not claim feasibility (got "
                    f"{self.feasible!r}): nothing was evaluated"
                )
            if self.model_executed:
                raise ValueError("a deferred structure outcome cannot report a model execution")

    @classmethod
    def deferred(cls, reason: str, *, walltime_s: float = 0.0) -> "StructureOutcome":
        """The entry stage did not evaluate structure; v0 admission will."""
        return cls(
            feasible=None, cache_status="deferred", model_executed=False,
            failure_reason=reason, walltime_s=walltime_s, evaluated=False,
        )


def _coerce_structure_outcome(result) -> StructureOutcome:
    return result if isinstance(result, StructureOutcome) else StructureOutcome(feasible=bool(result))


@dataclass(frozen=True)
class AdmissionAttempt:
    attempt_order: int
    design_idx: int
    source_id: str
    continuation_id: str | None
    root_equivalence_hash: str | None
    sequence_md5: str
    #: None when structure was DEFERRED -- never a stand-in for "passed".
    structure_feasible: bool | None
    verdict: str  # "admitted" | "structure_infeasible" | "deferred_to_v0"
    slot_idx: int | None
    cache_status: str = "miss"
    model_executed: bool = True
    failure_reason: str | None = None
    walltime_s: float = 0.0
    # v0 assigns the concrete ParticleState id per admitted slot; the driver fills this from the v0
    # admission output (the pure audit only knows the slot index it maps to).
    particle_id: str | None = None


@dataclass(frozen=True)
class AdmissionResult:
    attempts: tuple[AdmissionAttempt, ...]
    n_admitted: int
    insufficient: bool
    reason: str | None
    #: Rows offered to v0 without an entry-stage structure verdict. These are NOT admissions.
    n_deferred: int = 0

    @property
    def admitted(self) -> tuple[AdmissionAttempt, ...]:
        return tuple(a for a in self.attempts if a.verdict == "admitted")

    @property
    def structure_deferred(self) -> bool:
        return self.n_deferred > 0

    def slot_of_source(self, source_id: str) -> int | None:
        for attempt in self.attempts:
            if attempt.source_id == source_id and attempt.slot_idx is not None:
                return attempt.slot_idx
        return None


def admit_facade(
    candidates: Sequence[FacadeCandidate],
    structure_feasible_fn: Callable[[FacadeCandidate], bool],
    n_population: int,
    attempt_cap: int,
) -> AdmissionResult:
    """Scan the facade in ``(design_idx, seed)`` order, apply the definitive structure gate, and
    admit the first ``n_population`` feasible rows into slots 0..N-1, up to a common
    ``attempt_cap``. Every attempt (feasible or not) is recorded; a shortfall is reported as
    ``insufficient_feasible_initial_population`` rather than hidden by duplicating a parent.

    When the gate DEFERS (the V1-A entry stage does: v0 rechecks definitive structure), no row is
    admitted and no slot is assigned -- the whole ``attempt_cap`` is simply offered downstream, and
    the result says ``structure_deferred_to_v0``. Calling that "admitted" would put a structural
    claim into every artifact that nothing ever computed.
    """
    if int(attempt_cap) < int(n_population):
        raise ValueError(
            f"attempt_cap {attempt_cap} must be >= n_population {n_population}"
        )
    ordered = sorted(candidates, key=lambda c: (int(c.design_idx), int(c.seed)))
    attempts: list[AdmissionAttempt] = []
    slot = 0
    n_deferred = 0
    for candidate in ordered:
        # A deferred scan fills no slot, so it runs to the attempt cap and offers those rows to v0.
        if len(attempts) >= int(attempt_cap) or (n_deferred == 0 and slot >= int(n_population)):
            break
        outcome = _coerce_structure_outcome(structure_feasible_fn(candidate))
        if outcome.evaluated and n_deferred:
            raise ValueError(
                "structure policy changed mid-facade: an evaluated verdict after a deferred one "
                "would make n_admitted a partial count that reads as a complete one"
            )
        if not outcome.evaluated:
            if attempts and n_deferred == 0:
                raise ValueError(
                    "structure policy changed mid-facade: a deferred row after an evaluated one "
                    "would leave part of the population structurally unchecked but reported"
                )
            n_deferred += 1
            verdict, feasible, slot_idx = "deferred_to_v0", None, None
        else:
            feasible = bool(outcome.feasible)
            slot_idx = None
            if feasible:
                slot_idx = slot
                slot += 1
            verdict = "admitted" if feasible else "structure_infeasible"
        attempts.append(
            AdmissionAttempt(
                attempt_order=len(attempts),
                design_idx=int(candidate.design_idx),
                source_id=candidate.source_id,
                continuation_id=candidate.continuation_id,
                root_equivalence_hash=candidate.root_equivalence_hash,
                sequence_md5=candidate.sequence_md5,
                structure_feasible=feasible,
                verdict=verdict,
                slot_idx=slot_idx,
                cache_status=outcome.cache_status,
                model_executed=outcome.model_executed,
                failure_reason=outcome.failure_reason if feasible is not True else None,
                walltime_s=outcome.walltime_s,
            )
        )
    if n_deferred:
        # Nothing failed; the decision moved downstream. Reporting "insufficient" here would
        # blame the entry stage for a population v0 has not even looked at yet.
        return AdmissionResult(tuple(attempts), 0, False, "structure_deferred_to_v0", n_deferred)
    insufficient = slot < int(n_population)
    reason = "insufficient_feasible_initial_population" if insufficient else None
    return AdmissionResult(tuple(attempts), slot, insufficient, reason, 0)


@dataclass(frozen=True)
class CoverageRow:
    protein_id: str
    status: str


def build_cohort_coverage(
    requested_ids: Sequence[str], status_by_protein: Mapping[str, str]
) -> tuple[CoverageRow, ...]:
    """Build coverage over the FROZEN requested cohort: a protein never processed is surfaced as
    ``missing`` (never silently dropped); a duplicate requested id or a status for a
    non-requested protein is a hard error (PLAN §5 cohort_coverage)."""
    requested = list(requested_ids)
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate protein id in the requested cohort")
    extra = set(status_by_protein) - set(requested)
    if extra:
        raise ValueError(f"status supplied for non-requested proteins: {sorted(extra)}")
    return tuple(
        CoverageRow(protein_id=pid, status=status_by_protein.get(pid, "missing"))
        for pid in requested
    )
