"""V1F4 torch-free allocation layer: Head firewall, root value, deterministic root policies,
facade ordering, and matched-compute cost arithmetic (PLAN_RF_REFINE_FUSION_V1 §2.8-2.11, §3).

Pure Python: no torch, no Head implementation, no NMP. The Head is passed in as a callable and
is ONLY ever handed validated, complete, canonical AA20 sequences -- the firewall lives here,
outside the scorer, so it cannot be bypassed by a fail-open predictor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable

from .state import sequence_md5
from .v1_seeds import derive_seed

CANONICAL_AA20 = "ACDEFGHIKLMNPQRSTVWY"
_AA20 = frozenset(CANONICAL_AA20)


class HeadInputError(ValueError):
    """A sequence offered to the Head was not complete canonical AA20, or the Head returned a
    missing / non-finite / mis-counted result (PLAN §2.8)."""


def validate_complete_aa20(sequence: str, expected_length: int | None = None) -> None:
    """Reject mask, X, gap, unknown, lowercase, empty, or wrong-length input BEFORE the Head is
    called. The Head is fail-open on non-canonical input, so this is the load-bearing guard."""
    if not isinstance(sequence, str) or not sequence:
        raise HeadInputError("sequence must be a non-empty string")
    if expected_length is not None and len(sequence) != expected_length:
        raise HeadInputError(
            f"sequence length {len(sequence)} != expected {expected_length}"
        )
    bad = sorted(set(sequence) - _AA20)
    if bad:
        raise HeadInputError(f"non-canonical residue(s) {bad}; only uppercase AA20 allowed")


@dataclass(frozen=True)
class Continuation:
    continuation_id: str
    root_equivalence_hash: str
    set_tag: str  # est | eval | final
    seed: int
    sequence: str
    replicate_index: int


@dataclass(frozen=True)
class ScoredContinuation:
    continuation: Continuation
    global_risk: float
    sequence_md5: str


@dataclass(frozen=True)
class HeadRecord:
    """An identity-bound Head result. The Head returns these (not a bare float list), so a
    reordered / missing / extra result is caught by ``sequence_md5`` identity rather than a
    silent positional zip (PLAN §2.8)."""

    sequence_md5: str
    global_risk: float


HeadFn = Callable[[Sequence[str]], Sequence[HeadRecord]]


def head_risk_by_md5(
    head_fn: HeadFn, sequences: Sequence[str], expected_length: int
) -> tuple[dict[str, str], dict[str, float]]:
    """The shared Head firewall: validate every sequence as complete AA20, physically deduplicate
    for scoring, and bind each returned :class:`HeadRecord` BY ``sequence_md5``.

    Returns ``(md5_by_sequence, risk_by_md5)``. Both P1 arms go through this one function, so the
    complete-AA20 guard and the identity binding cannot drift between them -- a second copy for the
    Terminal arm would be a second place for a fail-open predictor to slip through (PLAN §2.8).
    """
    for sequence in sequences:
        validate_complete_aa20(sequence, expected_length)
    unique_sequences = list(dict.fromkeys(sequences))
    md5_by_sequence = {seq: sequence_md5(seq) for seq in unique_sequences}
    requested_md5 = set(md5_by_sequence.values())
    records = head_fn(unique_sequences)
    risk_by_md5: dict[str, float] = {}
    for record in records:
        digest = record.sequence_md5
        if digest in risk_by_md5:
            raise HeadInputError(f"duplicate Head record for sequence_md5 {digest}")
        value = float(record.global_risk)
        if not math.isfinite(value):
            raise HeadInputError(f"non-finite Head risk {record.global_risk!r} for {digest}")
        risk_by_md5[digest] = value
    if set(risk_by_md5) != requested_md5:
        missing = sorted(requested_md5 - set(risk_by_md5))
        extra = sorted(set(risk_by_md5) - requested_md5)
        raise HeadInputError(
            f"Head records do not match requested sequences: missing={missing} extra={extra}"
        )
    return md5_by_sequence, risk_by_md5


def evaluate_continuations(
    head_fn: HeadFn,
    continuations: Sequence[Continuation],
    expected_length: int,
) -> tuple[ScoredContinuation, ...]:
    """Score pre-terminal continuations through :func:`head_risk_by_md5`. Duplicate logical
    continuations keep their multiplicity (PLAN §2.8)."""
    md5_by_sequence, risk_by_md5 = head_risk_by_md5(
        head_fn, [cont.sequence for cont in continuations], expected_length
    )
    return tuple(
        ScoredContinuation(
            continuation=cont,
            global_risk=risk_by_md5[md5_by_sequence[cont.sequence]],
            sequence_md5=md5_by_sequence[cont.sequence],
        )
        for cont in continuations
    )


@dataclass(frozen=True)
class RootValue:
    root_equivalence_hash: str
    value: float
    n_samples: int


def _require_homogeneous_set(
    scored: Sequence[ScoredContinuation], set_tag: str, what: str
) -> str:
    """Enforce at runtime that a scored group is exactly one estimator/eval/final set of one
    root with distinct replicates (est/eval/final isolation is a runtime law, not a comment)."""
    tags = {sc.continuation.set_tag for sc in scored}
    if tags != {set_tag}:
        raise ValueError(f"{what} requires all set_tag=={set_tag!r}, got {sorted(tags)}")
    roots = {sc.continuation.root_equivalence_hash for sc in scored}
    if len(roots) != 1:
        raise ValueError(f"{what} mixes roots: {sorted(roots)}")
    replicates = [sc.continuation.replicate_index for sc in scored]
    if len(set(replicates)) != len(replicates):
        raise ValueError(f"{what} has duplicate replicate_index: {replicates}")
    return next(iter(roots))


def root_value(est_scored: Sequence[ScoredContinuation], k_est: int) -> float:
    """Exact ``K_EST`` arithmetic mean of complete-sequence Head ``global_risk`` (lower is
    better). Requires exactly ``k_est`` valid ``est`` samples of ONE root with DISTINCT
    replicates -- eval/final endpoints, mixed roots, and duplicate replicates are rejected, and
    a failed continuation must never shrink the denominator (PLAN §2.9)."""
    if len(est_scored) != int(k_est):
        raise ValueError(
            f"root value requires exactly k_est={k_est} valid samples, got {len(est_scored)}"
        )
    _require_homogeneous_set(est_scored, "est", "root_value")
    return sum(sc.global_risk for sc in est_scored) / float(k_est)


def select_value_beam(root_values: Sequence[RootValue], n: int) -> tuple[str, ...]:
    """Deterministic continuation-value beam over DISTINCT roots: lowest mean value first, ties
    broken by ``root_equivalence_hash``. Returns the inherited partial-root hashes (never their
    scored endpoints). A duplicated ``root_equivalence_hash`` is a hard error -- one basin must
    never occupy two beam slots (PLAN §2.4 distinct-root beam)."""
    hashes = [rv.root_equivalence_hash for rv in root_values]
    if len(set(hashes)) != len(hashes):
        raise ValueError("select_value_beam requires distinct root_equivalence_hash values")
    ranked = sorted(root_values, key=lambda rv: (rv.value, rv.root_equivalence_hash))
    return tuple(rv.root_equivalence_hash for rv in ranked[: int(n)])


def select_random_membership(
    eligible_hashes: Sequence[str], n: int, membership_seed: int
) -> tuple[str, ...]:
    """Which roots the ``random_partial`` VIEW holds, ordered by a Head-INDEPENDENT content hash.

    ``membership_seed`` must be ``SeedContext.random_membership_seed()``: a single un-indexed draw
    that decides membership only. The random and selected views then read the SAME per-root
    evaluation table, so switching the policy label leaves every continuation seed byte-identical
    (PLAN §2.9-§2.10 shared-pool law). Ordering by the Head value here would make the "random"
    baseline value-aware and destroy the comparison.
    """
    ranked = sorted(
        eligible_hashes,
        key=lambda h: derive_seed("policy_membership", int(membership_seed), str(h)),
    )
    if int(n) > len(ranked):
        raise ValueError(
            f"random membership needs {n} roots but only {len(ranked)} are eligible; a shrunk "
            "random view is not compute-matched with the selected view"
        )
    return tuple(ranked[: int(n)])


@dataclass(frozen=True)
class TerminalCandidate:
    sequence: str
    sequence_md5: str
    global_risk: float
    source_id: str


@dataclass(frozen=True)
class FacadeRow:
    design_idx: int
    seed: int
    sequence: str
    sequence_md5: str
    source_id: str
    root_equivalence_hash: str | None


def build_terminal_facade(
    pool: Sequence[TerminalCandidate], seed_fn: Callable[[str], int], *, facade_cap: int
) -> tuple[FacadeRow, ...]:
    """TERMINAL arm facade (§3.2.1): rank the arm's ``M_T`` complete trajectories by exact
    terminal Head under the total order ``(global_risk, sequence_md5, source_id)`` and submit ONLY
    the top ``facade_cap`` (= the frozen common ``F_cap``) to initial refold admission, so both
    arms attempt at most ``F_cap`` initial refolds.

    A pool smaller than ``facade_cap`` is a HARD FAILURE, not a shrunken facade: ``M_T`` is derived
    as ``floor(C_reserved / S)`` from the reserved pre-terminal budget, so ``M_T < F_cap`` can only
    mean a DFE accounting / off-by-one error. Quietly submitting fewer rows would silently give the
    two arms different refold budgets, which is exactly the confound this design removes.
    """
    if not isinstance(facade_cap, int) or isinstance(facade_cap, bool) or facade_cap <= 0:
        raise ValueError(f"facade_cap must be a positive int, got {facade_cap!r}")
    candidates = list(pool)
    if len(candidates) < facade_cap:
        raise ValueError(
            f"terminal pool has {len(candidates)} trajectories < facade_cap={facade_cap}: "
            "M_T < F_cap is a DFE accounting error (§3.2.1), never a reason to shrink the facade"
        )
    source_ids = [c.source_id for c in candidates]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("terminal pool has a duplicated source_id: one trajectory, two slots")
    for candidate in candidates:
        if not math.isfinite(float(candidate.global_risk)):
            raise ValueError(
                f"terminal candidate {candidate.source_id!r} has a non-finite Head risk "
                f"{candidate.global_risk!r}; it cannot be ranked"
            )
    ranked = sorted(candidates, key=lambda c: (c.global_risk, c.sequence_md5, c.source_id))
    return tuple(
        FacadeRow(
            design_idx=rank, seed=int(seed_fn(candidate.source_id)),
            sequence=candidate.sequence, sequence_md5=candidate.sequence_md5,
            source_id=candidate.source_id, root_equivalence_hash=None,
        )
        for rank, candidate in enumerate(ranked[:facade_cap])
    )


@dataclass(frozen=True)
class CollapsedFacadeRow:
    """A facade row whose complete sequence duplicated a better-ranked row's. Recorded, never
    silently dropped: which distinct roots converged is real information about the search."""

    design_idx: int
    source_id: str
    root_equivalence_hash: str | None
    sequence_md5: str
    representative_source_id: str
    representative_design_idx: int


@dataclass(frozen=True)
class FacadeCollapse:
    rows: tuple[FacadeRow, ...]
    collapsed: tuple[CollapsedFacadeRow, ...]

    @property
    def n_collapsed(self) -> int:
        return len(self.collapsed)


def collapse_facade_by_sequence(facade: Sequence[FacadeRow]) -> FacadeCollapse:
    """Collapse convergent facade rows so the v0 handoff is SEQUENCE-UNIQUE per protein.

    Two distinct roots can produce the same complete sequence. The reason to collapse is ONE thing,
    and it is about the terminal search, not about cost or identity:

    **Duplicate round-0 particles buy ancestry mass.** ``doc/RF-Refine-Fusion.md`` states the v0
    rule that "multiplicity does not buy ancestry mass" -- child budgets are divided by a parent's
    offspring count precisely so a repeated state cannot claim extra measure. Two identical round-0
    particles defeat that: the basin holds 2 of N slots, 2/N of the initial weight, and twice the
    child budget, purely because two roots converged. That is a real distortion of the terminal
    search, and it is asymmetric between the arms if they converge at different rates -- which is
    why the collapse EVENT is persisted (``facade_collapse.parquet``) and a shortened facade gets
    its own status rather than being silently absorbed.

    Two justifications previously given here were WRONG and are recorded so they are not
    reintroduced: (1) "v0 refolds the identical sequence twice" -- it does not; ``StructureCache``
    keys on ``sequence_md5`` and the second lookup is a hit, so a duplicate costs no refold;
    (2) "the particle id becomes ambiguous" -- it does not; ``make_particle_id`` embeds ``slot_idx``
    and ``entry_source_id`` now travels per row, so attribution is exact WITH duplicates.

    NOTE: no authority document mandates either collapsing or preserving facade-level multiplicity.
    ``continuations.parquet`` keeps every logical duplicate (that IS mandated) and root-value
    aggregation restores all logical samples; this collapse happens strictly downstream of both.

    The best-ranked row wins (facade order is already the arm's selection order) and keeps its own
    seed and lineage; ``design_idx`` is re-densified so v0's ordered scan sees a contiguous rank.
    """
    kept: list[FacadeRow] = []
    collapsed: list[CollapsedFacadeRow] = []
    representative: dict[str, FacadeRow] = {}
    for row in sorted(facade, key=lambda r: int(r.design_idx)):
        prior = representative.get(row.sequence_md5)
        if prior is None:
            representative[row.sequence_md5] = row
            kept.append(row)
            continue
        collapsed.append(
            CollapsedFacadeRow(
                design_idx=int(row.design_idx), source_id=row.source_id,
                root_equivalence_hash=row.root_equivalence_hash,
                sequence_md5=row.sequence_md5,
                representative_source_id=prior.source_id,
                representative_design_idx=int(prior.design_idx),
            )
        )
    densified = tuple(
        FacadeRow(
            design_idx=rank, seed=row.seed, sequence=row.sequence, sequence_md5=row.sequence_md5,
            source_id=row.source_id, root_equivalence_hash=row.root_equivalence_hash,
        )
        for rank, row in enumerate(kept)
    )
    return FacadeCollapse(densified, tuple(collapsed))


@dataclass(frozen=True)
class FinalMaterialization:
    """One fresh ``final`` completion of a selected partial root. Typed so an ``est``/``eval``
    endpoint can never be placed into the PRE-TERMINAL P1 facade, and so the ACTUAL final seed is
    preserved (never re-derived)."""

    root_equivalence_hash: str
    continuation: Continuation  # must carry set_tag == "final"
    source_id: str


def build_preterminal_facade(
    ranked_root_hashes: Sequence[str],
    materializations: Mapping[str, FinalMaterialization],
) -> tuple[FacadeRow, ...]:
    """PRE-TERMINAL facade: order follows the ROOT-VALUE rank (``ranked_root_hashes``), never the
    materialized terminal Head (PLAN §2.11). Each root's materialization must be a fresh
    ``final`` continuation of THAT root; an ``est``/``eval`` endpoint fails fast. The facade
    seed is the continuation's own final seed, not a re-derived one."""
    rows: list[FacadeRow] = []
    for rank, root_hash in enumerate(ranked_root_hashes):
        materialization = materializations[root_hash]
        cont = materialization.continuation
        if cont.set_tag != "final":
            raise ValueError(
                f"pre-terminal facade requires a fresh final continuation, got set_tag="
                f"{cont.set_tag!r} for root {root_hash!r}"
            )
        if materialization.root_equivalence_hash != root_hash or (
            cont.root_equivalence_hash != root_hash
        ):
            raise ValueError(
                f"materialization root mismatch for {root_hash!r}: "
                f"{materialization.root_equivalence_hash!r}/{cont.root_equivalence_hash!r}"
            )
        rows.append(
            FacadeRow(
                design_idx=rank, seed=int(cont.seed), sequence=cont.sequence,
                sequence_md5=sequence_md5(cont.sequence), source_id=materialization.source_id,
                root_equivalence_hash=root_hash,
            )
        )
    return tuple(rows)


# --------------------------------------------------------------------------- #
# Matched-compute DFE arithmetic (PLAN §3.1-3.2)
# --------------------------------------------------------------------------- #
def preterminal_entry_dfe(
    prefix_dfe_by_attempt: Sequence[int],
    k_est: int,
    tail_dfe_by_root: Sequence[int],
    final_tail_dfe: Sequence[int],
) -> int:
    """Actual pre-terminal entry DFE: prefix attempts + K_EST * unique-root tails + final tails."""
    return (
        sum(int(d) for d in prefix_dfe_by_attempt)
        + int(k_est) * sum(int(d) for d in tail_dfe_by_root)
        + sum(int(d) for d in final_tail_dfe)
    )


def reserved_p1_preterminal_dfe(
    prefix_dfe_by_attempt: Sequence[int],
    k_est: int,
    tail_dfe_by_root: Sequence[int],
    f_cap: int,
) -> int:
    """Reserved P1 pre-terminal budget computed BEFORE outcomes: worst-case ``F_cap * r_max`` final
    tail with ``r_max = max_u(S - s_u)`` derived internally from ``tail_dfe_by_root`` (never a
    redundant caller parameter). This reservation is also what the TERMINAL arm's trajectory count
    is derived from (``M_T = floor(C_reserved / S)``, §3.2.1), so it must never depend on an
    outcome."""
    tails = [int(d) for d in tail_dfe_by_root]
    r_max = max(tails) if tails else 0
    return (
        sum(int(d) for d in prefix_dfe_by_attempt)
        + int(k_est) * sum(tails)
        + int(f_cap) * r_max
    )


def reserved_t0_dfe(
    prefix_dfe_by_attempt: Sequence[int],
    k_est: int,
    k_eval: int,
    tail_dfe_by_root: Sequence[int],
) -> int:
    """Reserved T0 root-allocation budget: prefix + (K_EST + K_EVAL) * unique-root tails. T0 runs
    no ``final`` materialization at all -- its structure spend is the Q_T0 subsample (PLAN §3.2)."""
    return sum(int(d) for d in prefix_dfe_by_attempt) + (int(k_est) + int(k_eval)) * sum(
        int(d) for d in tail_dfe_by_root
    )


# --------------------------------------------------------------------------- #
# T0 evaluation membership, structure subset, authorization (PLAN §2.9-2.10)
# --------------------------------------------------------------------------- #
#: The THREE V1-A T0 policy views. ``selected_partial`` and ``random_partial`` are membership views
#: over ONE shared pre-terminal root pool and ONE shared held-out evaluation table;
#: ``independent_full`` is the compute-matched full-trajectory control. Branch-and-materialize was
#: dropped from V1-A, so a fourth policy here would reserve budget for a comparison never run.
_T0_POLICIES = ("selected_partial", "random_partial", "independent_full")


def t0_reserved_structure_requests(q_t0: int) -> int:
    """T0's reserved structure budget: ``3 * Q_T0`` -- one Q_T0 subsample per policy view."""
    if not isinstance(q_t0, int) or isinstance(q_t0, bool) or q_t0 <= 0:
        raise ValueError(f"q_t0 must be a positive int, got {q_t0!r}")
    return len(_T0_POLICIES) * q_t0


def build_common_eval_table(
    eval_by_root: Mapping[str, Sequence[ScoredContinuation]], k_eval: int
) -> dict[str, tuple[ScoredContinuation, ...]]:
    """One common held-out ``K_EVAL`` table per root: exactly ``k_eval`` ``eval`` continuations
    of that root with distinct replicates. Selected and random policies are MEMBERSHIP VIEWS
    over this same table, never separate policy-specific draws (PLAN §2.9)."""
    table: dict[str, tuple[ScoredContinuation, ...]] = {}
    for root_hash, scored in eval_by_root.items():
        if len(scored) != int(k_eval):
            raise ValueError(
                f"eval table for {root_hash!r} needs exactly k_eval={k_eval}, got {len(scored)}"
            )
        member = _require_homogeneous_set(scored, "eval", "common_eval_table")
        if member != root_hash:
            raise ValueError(f"eval table key {root_hash!r} != endpoint root {member!r}")
        table[root_hash] = tuple(scored)
    return table


def eval_membership_view(
    common_eval_table: Mapping[str, tuple[ScoredContinuation, ...]],
    root_hashes: Sequence[str],
) -> dict[str, tuple[ScoredContinuation, ...]]:
    """A policy's view over the COMMON eval table (the exact same rows), never a fresh draw. A
    root not present in the common table is a hard error."""
    view: dict[str, tuple[ScoredContinuation, ...]] = {}
    for root_hash in root_hashes:
        if root_hash not in common_eval_table:
            raise ValueError(f"root {root_hash!r} absent from the common eval table")
        view[root_hash] = common_eval_table[root_hash]
    return view


def root_balanced_eval_endpoints(
    common_eval_table: Mapping[str, tuple[ScoredContinuation, ...]],
    root_hashes: Sequence[str],
    subsample_seed: int,
    content_id: Callable[[object], str],
    *,
    cache: dict | None = None,
) -> list[ScoredContinuation]:
    """Root-balanced structure endpoints: EXACTLY one held-out endpoint per held root (runbook
    §6.0 policy-faithful B*), chosen by a stable Head-INDEPENDENT content hash over that root's
    common ``K_EVAL`` rows.

    A ``cache`` keyed by root hash, shared across the selected and random calls, guarantees a root
    held by BOTH views contributes the IDENTICAL endpoint object -- the overlap-reuse the frozen B*
    law requires (§6.0:397). The choice never reads ``global_risk``, so it adds no second
    endpoint-selection stage to the partial policies; the partials are one-per-root, never
    terminal-Head-truncated.
    """
    if cache is None:
        cache = {}
    chosen: list[ScoredContinuation] = []
    for root_hash in root_hashes:
        if root_hash not in cache:
            if root_hash not in common_eval_table:
                raise ValueError(f"root {root_hash!r} absent from the common eval table")
            rows = common_eval_table[root_hash]
            if not rows:
                raise ValueError(f"root {root_hash!r} has no eval endpoints to balance over")
            cache[root_hash] = min(
                rows,
                key=lambda sc: derive_seed(
                    "t0_root_endpoint", int(subsample_seed), str(content_id(sc))
                ),
            )
        chosen.append(cache[root_hash])
    return chosen


def build_t0_structure_subset(
    endpoints_by_policy: Mapping[str, Sequence[object]],
    q_t0: int,
    subsample_seed: int,
    content_id: Callable[[object], str],
) -> dict[str, tuple[object, ...]]:
    """Exactly the three V1-A policy views, each subsampled to EXACTLY ``q_t0`` endpoints by a
    Head-INDEPENDENT content hash; a policy with fewer than ``q_t0`` eligible endpoints fails
    closed rather than shrinking a denominator (PLAN §2.10)."""
    if set(endpoints_by_policy) != set(_T0_POLICIES):
        raise ValueError(
            f"T0 structure subset requires exactly policies {sorted(_T0_POLICIES)}, "
            f"got {sorted(endpoints_by_policy)}"
        )
    subset: dict[str, tuple[object, ...]] = {}
    for policy in _T0_POLICIES:
        endpoints = list(endpoints_by_policy[policy])
        if len(endpoints) < int(q_t0):
            raise ValueError(
                f"policy {policy!r} has {len(endpoints)} eligible endpoints < Q_T0={q_t0}"
            )
        ranked = sorted(
            endpoints,
            key=lambda ep: derive_seed("t0_subset", int(subsample_seed), str(content_id(ep))),
        )
        subset[policy] = tuple(ranked[: int(q_t0)])
    return subset


def authorize_final_materialization(
    selected_root_hashes: Sequence[str], continuation: Continuation
) -> bool:
    """A P1 final parent is authorized only by a SELECTED partial ROOT via a fresh ``final``
    continuation -- never by an ``est``/``eval`` endpoint. This is what makes branch-and-materialize
    structurally impossible rather than merely unused (PLAN §2.9/§2.11)."""
    if continuation.set_tag != "final":
        raise ValueError(
            f"only a fresh final continuation materializes a P1 parent, got "
            f"set_tag={continuation.set_tag!r}"
        )
    if continuation.root_equivalence_hash not in set(selected_root_hashes):
        raise ValueError(
            f"continuation root {continuation.root_equivalence_hash!r} is not a selected root"
        )
    return True


def matched_full_trajectory_allocation(reserved_dfe: int, s_steps: int) -> tuple[int, int]:
    """The largest integer number of full ``S``-step trajectories whose DFE does not exceed the
    reserved cap, and the integer remainder (recorded, never handed to the pre-terminal arm). Used
    both for the T0 ``independent_full`` control and for the Terminal arm's ``M_T`` (§3.2/§3.2.1)."""
    if int(s_steps) <= 0:
        raise ValueError("s_steps must be positive")
    count = int(reserved_dfe) // int(s_steps)
    remainder = int(reserved_dfe) - count * int(s_steps)
    return count, remainder
