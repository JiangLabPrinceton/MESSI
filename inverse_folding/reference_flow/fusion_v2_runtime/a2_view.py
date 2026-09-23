"""V2F4-3: the A2 feedback-off view and matched extra-lookahead allocation (PLAN §2.3, §4.3).

A2 is **a view of the same run**, not a second run.

PLAN §2.3: "The archive state immediately before feedback is the prospective A2 view.  A2 and V2
may not regenerate separate initial roots, lookaheads, Head results, or structure results."
PLAN §4.3: A2 "is not a separate root-generation run and may not be reconstructed post hoc from
only selected rows."

This is the load-bearing property of the whole experimental design.  If A2 regenerated its own
roots, every reported difference would confound the feedback mechanism with root-sampling variance,
and no amount of downstream statistics could separate them.  So :func:`snapshot_a2_view` takes the
COMPLETE archive as it stands at the moment before feedback -- refusing a filtered subset outright
-- and freezes it, so post-feedback descendants cannot leak into the control.

The second half is matched compute.  After feedback V2 spends resource on a projection, a
propagation segment, and descendant lookaheads; A2 spends the SAME resource on additional exact
futures from the UNCHANGED source state, under declared disjoint seeds.

Which resource "the same" means is a REQUIRED CHOICE with no default.  The measured unit costs are
2.42 s per refold against at most 0.029 s per DFE, so one refold is roughly 81 DFE and structure is
about 94% of the marginal cost of a feedback cycle.  Matching on DFE and matching on refolds are
therefore materially different experiments, and picking one silently would hand one arm more
compute without anyone noticing.  The unmatched components and the unspent residual are both
reported for the same reason: an arm that quietly received more of something is not a control.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..fusion_v2.config import A2_RESOURCE_COMPONENTS
from ..fusion_v2.errors import V2Error
from ..fusion_v2.identity import canonical_digest, require_digest, require_id_namespace

__all__ = [
    "V2A2Error",
    "LEDGER_STAGES",
    "PRE_FEEDBACK_STAGES",
    "POST_FEEDBACK_STAGES",
    "RESOURCE_FIELDS",
    "ALLOCATION_STATUSES",
    "ResourceLedger",
    "A2View",
    "MatchedLookaheadBudget",
    "A2Allocation",
    "snapshot_a2_view",
    "matched_extra_lookahead_budget",
    "ledger_stage",
    "resource_ledger_from_events",
    "stage_ledgers",
    "a2_matched_allocation",
]

#: The five envelope components, in a stable order.  Asserted equal to the config's closed
#: vocabulary so a component added there and forgotten here fails at import rather than by
#: silently dropping a resource out of every reported vector.
RESOURCE_FIELDS = ("logical_dfe", "head_calls", "definitive_refolds", "gpu_seconds", "walltime_s")
assert set(RESOURCE_FIELDS) == set(A2_RESOURCE_COMPONENTS)


class V2A2Error(V2Error):
    """An A2-view or matched-allocation contract violation."""


@dataclass(frozen=True)
class ResourceLedger:
    """Cost along every component A2 might be matched on.

    Every component is required.  A ledger with a missing field would make the matching choice look
    available when the number behind it was never measured.

    A ledger is either a TOTAL -- integer counts of DFE, Head calls and refolds -- or a RATE, i.e.
    the per-lookahead unit cost produced by :meth:`per_unit`, whose counts are legitimately
    fractional.  Rounding a rate to an integer would change the matched budget, which is the one
    number the depth-versus-breadth contrast is computed from.
    """

    logical_dfe: int | float = 0
    head_calls: int | float = 0
    definitive_refolds: int | float = 0
    gpu_seconds: float = 0.0
    walltime_s: float = 0.0

    def component(self, name: str) -> float:
        if name not in A2_RESOURCE_COMPONENTS:
            raise V2A2Error(
                f"{name!r} is not in the closed matching_resource vocabulary "
                f"{sorted(A2_RESOURCE_COMPONENTS)}"
            )
        return float(getattr(self, name))

    def components(self) -> dict[str, float]:
        """The whole vector, for an artifact that may not collapse it.

        FUSION_V2 §6.1: the envelope "is not silently collapsed into one invented universal cost",
        and "the full vector remains attached to every point".
        """
        return {name: float(getattr(self, name)) for name in RESOURCE_FIELDS}

    def plus(self, other: "ResourceLedger") -> "ResourceLedger":
        if not isinstance(other, ResourceLedger):
            raise V2A2Error("only a ResourceLedger may be added to a ResourceLedger")
        return ResourceLedger(
            **{name: getattr(self, name) + getattr(other, name) for name in RESOURCE_FIELDS})

    def per_unit(self, n_lookaheads: int) -> "ResourceLedger":
        """This total spread over the lookaheads that produced it: the unit cost of one exact
        future.

        Refused at zero for the same reason :func:`matched_extra_lookahead_budget` refuses a
        non-positive unit cost: a division by nothing reports an unbounded budget, which reads as
        "A2 gets infinite compute" rather than "nothing was measured".
        """
        n = int(n_lookaheads)
        if n <= 0:
            raise V2A2Error(
                f"a per-lookahead unit cost needs a positive number of lookaheads, got {n}; "
                "dividing a measured total by zero futures would report an unbounded unit"
            )
        return ResourceLedger(**{name: float(getattr(self, name)) / n for name in RESOURCE_FIELDS})


@dataclass(frozen=True)
class A2View:
    """The exact archive state immediately before feedback, frozen.

    Holds ids and digests rather than endpoint objects: the endpoints themselves live in the one
    archive, and duplicating them here would create a second copy that could silently diverge from
    the rows A2 is supposed to be a view OF.
    """

    protein_id: str
    source_state_id: str
    depth: int
    endpoint_ids: tuple[str, ...]
    content_digest_by_endpoint: Mapping[str, str]
    pre_feedback_cost: ResourceLedger | None = None
    a2_view_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.protein_id, str) or not self.protein_id:
            raise V2A2Error("protein_id must be a non-empty str")
        require_id_namespace(self.source_state_id, "live")
        if not self.source_state_id.startswith(f"{self.protein_id}:"):
            raise V2A2Error(
                "source_state_id does not belong to the A2 view's protein_id"
            )
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise V2A2Error(f"depth must be a non-negative int, got {self.depth!r}")
        if len(set(self.endpoint_ids)) != len(self.endpoint_ids):
            raise V2A2Error("an A2 view may not contain duplicate endpoint ids")
        digest_by_endpoint = dict(self.content_digest_by_endpoint)
        if set(digest_by_endpoint) != set(self.endpoint_ids):
            raise V2A2Error(
                "content_digest_by_endpoint must bind exactly the A2 view's endpoint ids"
            )
        for endpoint_id, digest in digest_by_endpoint.items():
            require_id_namespace(endpoint_id, "endpoint")
            require_digest(digest, f"content_digest_by_endpoint[{endpoint_id!r}]")

        payload = {
            "protein_id": self.protein_id,
            "source_state_id": self.source_state_id,
            "depth": self.depth,
            "endpoint_ids": sorted(self.endpoint_ids),
            "content_digest_by_endpoint": digest_by_endpoint,
        }
        object.__setattr__(
            self,
            "a2_view_id",
            f"a2:{self.source_state_id}:d{self.depth}:{canonical_digest(payload)[:12]}",
        )


@dataclass(frozen=True)
class MatchedLookaheadBudget:
    """How much extra exact-future work A2 gets, and what was left unmatched."""

    matched_component: str
    matched_amount: float
    per_lookahead_amount: float
    n_extra_lookaheads: int
    residual: float
    unmatched_components: tuple[str, ...]
    extra_fork_seeds: tuple[int, ...] = field(default_factory=tuple)


def snapshot_a2_view(
    archive: Any,
    *,
    protein_id: str,
    source_state_id: str,
    depth: int,
    pre_feedback_cost: ResourceLedger | None = None,
    endpoint_ids: Sequence[str] | None = None,
) -> A2View:
    """Freeze the complete archive as the prospective A2 view.

    ``endpoint_ids`` exists only so a caller can ASSERT the set it expects; supplying a strict
    subset is refused, because PLAN §4.3 forbids reconstructing A2 "post hoc from only selected
    rows".  A view built from the rows that happened to look good is not a control.
    """
    endpoints = archive.endpoints()
    complete = tuple(endpoint.endpoint_id for endpoint in endpoints)
    if endpoint_ids is not None and tuple(endpoint_ids) != complete:
        raise V2A2Error(
            "the A2 view must be the COMPLETE pre-feedback archive; it may not be reconstructed "
            f"from a selected subset ({len(tuple(endpoint_ids))} of {len(complete)} rows given)"
        )
    return A2View(
        protein_id=protein_id,
        source_state_id=source_state_id,
        depth=int(depth),
        endpoint_ids=complete,
        content_digest_by_endpoint={
            endpoint.endpoint_id: endpoint.content_digest for endpoint in endpoints
        },
        pre_feedback_cost=pre_feedback_cost,
    )


def matched_extra_lookahead_budget(
    *,
    v2_post_feedback_cost: ResourceLedger,
    matching_resource: str,
    per_lookahead_cost: ResourceLedger,
    used_seeds: Sequence[int] = (),
    seed_base: int = 1,
) -> MatchedLookaheadBudget:
    """Convert V2's post-feedback spend into A2's extra exact futures.

    ``matching_resource`` has no default on purpose: PLAN leaves the choice open, structure
    dominates cost, and guessing would silently change what the comparison controls for.
    """
    if not isinstance(v2_post_feedback_cost, ResourceLedger):
        raise V2A2Error("v2_post_feedback_cost must be a ResourceLedger")
    if not isinstance(per_lookahead_cost, ResourceLedger):
        raise V2A2Error("per_lookahead_cost must be a ResourceLedger")
    # Redundant with ResourceLedger.component(), which validates the same vocabulary and raises
    # the same error type; kept because it names the PARAMETER rather than the ledger field, which
    # is the more useful diagnostic for a mis-declared config.  Mutation testing confirms removing
    # it changes no observable behaviour.
    if matching_resource not in A2_RESOURCE_COMPONENTS:
        raise V2A2Error(
            f"matching_resource {matching_resource!r} is not in the closed vocabulary "
            f"{sorted(A2_RESOURCE_COMPONENTS)}"
        )

    matched = v2_post_feedback_cost.component(matching_resource)
    per = per_lookahead_cost.component(matching_resource)
    if per <= 0.0:
        raise V2A2Error(
            f"per_lookahead_cost.{matching_resource} is {per}; a non-positive unit cost would "
            "report an unbounded budget, which reads as 'A2 gets infinite compute' rather than "
            "'the cost model is broken'"
        )

    n_extra = int(matched // per)
    residual = float(matched - n_extra * per)

    taken = {int(s) for s in used_seeds}
    seeds: list[int] = []
    candidate = int(seed_base)
    while len(seeds) < n_extra:
        if candidate not in taken:
            seeds.append(candidate)
            taken.add(candidate)
        candidate += 1

    return MatchedLookaheadBudget(
        matched_component=matching_resource,
        matched_amount=matched,
        per_lookahead_amount=per,
        n_extra_lookaheads=n_extra,
        residual=residual,
        # The exact complement, reported rather than ignored.
        unmatched_components=tuple(sorted(set(A2_RESOURCE_COMPONENTS) - {matching_resource})),
        extra_fork_seeds=tuple(seeds),
    )


# --------------------------------------------------------------------------------------------
# assembling the measured vector out of the compute journal
# --------------------------------------------------------------------------------------------
#
# The amounts above are not free parameters: every one of them is already measured, per attempt, in
# the journal PLAN §5.4 requires.  What was missing is the assembly -- nothing turned those rows
# into a :class:`ResourceLedger`, so the one number the whole A2 contrast rests on was reported as
# a literal zero.  These helpers are that assembly and nothing more; they measure, they do not
# decide.

#: Where in one rung's execution graph each ledger phase was charged.  ``prefix`` and
#: ``source_pool`` are before the feedback boundary, ``feedback`` and ``descendant_pool`` after it.
LEDGER_STAGES = ("prefix", "source_pool", "feedback", "descendant_pool")
PRE_FEEDBACK_STAGES = ("prefix", "source_pool")
POST_FEEDBACK_STAGES = ("feedback", "descendant_pool")

_STAGE_BY_PHASE = {
    "root_capture": "prefix",
    "screen": "source_pool",
    "projection": "feedback",
    "segment": "feedback",
    "descendant_screen": "descendant_pool",
}
#: Head and structure are charged on BOTH sides of the boundary, so the phase alone cannot place
#: them: the pool that requested them is named in the event id the runner builds
#: (``{transition}:source:...`` / ``{transition}:descendant:...``).  Attributing them by phase
#: alone would move the descendant pool's refolds -- the dominant term in everything V2 spends
#: after feedback -- onto the pre-feedback side, and the matched allocation would then be computed
#: from a total that excludes the very work it exists to reassign.
_POOLED_PHASES = frozenset({"head", "structure"})
_POOL_MARKERS = ((":descendant:", "descendant_pool"), (":source:", "source_pool"))


def ledger_stage(*, phase: str, event_id: str) -> str:
    """Which stage of a rung paid this ledger row.

    Unrecognized rows are REFUSED rather than binned into a default.  A fallback would move real
    spend to whichever side it favours, silently, and the two arms would then differ by an amount
    no reader could see.
    """
    stage = _STAGE_BY_PHASE.get(str(phase))
    if stage is not None:
        return stage
    if str(phase) in _POOLED_PHASES:
        for marker, pooled in _POOL_MARKERS:
            if marker in str(event_id):
                return pooled
        raise V2A2Error(
            f"{phase!r} event {event_id!r} names no pool, so it cannot be attributed to a stage: "
            f"one of {[marker for marker, _ in _POOL_MARKERS]} must appear in the event id"
        )
    raise V2A2Error(
        f"phase {phase!r} has no declared stage; every ledger phase must be attributable to one of "
        f"{list(LEDGER_STAGES)} before it can be reported as matched or unmatched compute"
    )


def _as_events(rows: Sequence[Any]) -> tuple:
    from .ledger import V2LedgerEvent

    return tuple(row if isinstance(row, V2LedgerEvent) else V2LedgerEvent(**dict(row))
                 for row in rows)


def resource_ledger_from_events(rows: Sequence[Any]) -> ResourceLedger:
    """The five-component envelope of a set of ledger rows.

    Aggregated through :func:`aggregate_v2_ledger` rather than by summing columns here, so this
    inherits the audited counting rules: logical work once per ``event_id``, physical work once per
    ``attempt_id``.  Summing naively would double-charge every row a resume re-read -- inflating
    exactly the totals a matched-compute claim is computed from.
    """
    from .ledger import aggregate_v2_ledger

    totals = aggregate_v2_ledger(_as_events(rows))
    return ResourceLedger(
        logical_dfe=int(totals["logical_dfe"]),
        head_calls=int(totals["head_calls"]),
        # The cap layer bounds ``max_definitive_refolds`` with this same field, so the two readings
        # of "a refold" cannot diverge.
        definitive_refolds=int(totals["structure_attempts"]),
        gpu_seconds=float(totals["gpu_seconds"]),
        walltime_s=float(totals["walltime_s"]),
    )


def stage_ledgers(rows: Sequence[Any]) -> dict[str, ResourceLedger]:
    """One measured envelope per stage.  Every stage is present, zero when nothing was charged.

    A missing key would make a stage that ran nothing indistinguishable from one whose rows were
    dropped on the way to the artifact.
    """
    grouped: dict[str, list] = {stage: [] for stage in LEDGER_STAGES}
    for event in _as_events(rows):
        grouped[ledger_stage(phase=event.phase, event_id=event.event_id)].append(event)
    return {stage: resource_ledger_from_events(events) for stage, events in grouped.items()}


# --------------------------------------------------------------------------------------------
# what an artifact is allowed to call "matched"
# --------------------------------------------------------------------------------------------

#: Closed set.  A free-text status would let "matched" be written by anything that felt like it.
ALLOCATION_STATUSES = (
    "matched",                        # every unit of the matched resource was reassigned to A2
    "nothing_to_match",               # V2 spent nothing after this view, so there is nothing owed
    "declared_but_not_executed",      # the amount is measured and A2 received none of it
    "partially_executed",             # A2 received some of it
    "over_allocated",                 # A2 received MORE than V2 spent
    "unit_cost_not_measured",         # no measured per-lookahead cost, so no budget is derivable
    "not_derivable_in_this_shard",    # this arm never ran the feedback stage that defines the match
)


@dataclass(frozen=True)
class A2Allocation:
    """What an A2 view was ACTUALLY given, beside what matching would have owed it.

    PLAN §5.3 requires the A2 view table to carry "any matched extra-lookahead allocation".  The
    dangerous form of that field is a bare count: a downstream reader that sees
    ``matched_extra_lookaheads`` beside a declared ``matching_resource`` concludes the arms were
    compute-matched, and a zero written by a run that never allocated anything reads exactly like a
    run that had nothing to allocate.  So the executed count is reported next to the measured
    vector, the owed count, and an explicit status -- and the matching resource is reported ONLY
    when the matching really happened.
    """

    matching_resource: str
    status: str
    matched: bool
    executed_extra_lookaheads: int
    owed_extra_lookaheads: int | None
    pre_feedback_cost: ResourceLedger
    v2_post_feedback_cost: ResourceLedger
    per_lookahead_cost: ResourceLedger
    unmatched_amount: float | None
    residual: float | None
    unmatched_components: tuple[str, ...]
    extra_fork_seeds: tuple[int, ...]
    detail: str

    @property
    def reported_matching_resource(self) -> str | None:
        """The resource this view may CLAIM to be matched on -- ``None`` unless it really is.

        The declared choice is not lost: it stays in the report below and in the run manifest.
        What is withheld is the claim, because a non-null matching resource beside a count is the
        pair a reader interprets as "these two arms were compute-matched".
        """
        return self.matching_resource if self.matched else None

    def report(self) -> dict:
        """The allocation record an artifact carries beside the count."""
        return {
            "matched_allocation_executed": bool(self.matched),
            "status": self.status,
            "declared_matching_resource": self.matching_resource,
            "declared_unmatched_components": list(self.unmatched_components),
            "executed_extra_lookaheads": int(self.executed_extra_lookaheads),
            "owed_extra_lookaheads": self.owed_extra_lookaheads,
            "unmatched_amount": self.unmatched_amount,
            "residual_below_one_lookahead": self.residual,
            "a2_pre_feedback_cost": self.pre_feedback_cost.components(),
            "v2_post_feedback_cost": self.v2_post_feedback_cost.components(),
            "per_lookahead_cost": self.per_lookahead_cost.components(),
            "owed_extra_fork_seeds": list(self.extra_fork_seeds),
            "detail": self.detail,
        }


def a2_matched_allocation(
    *,
    matching_resource: str,
    pre_feedback_cost: ResourceLedger,
    v2_post_feedback_cost: ResourceLedger,
    per_lookahead_cost: ResourceLedger,
    executed_extra_lookaheads: int,
    post_feedback_spend_observed: bool,
    used_seeds: Sequence[int] = (),
    seed_base: int = 1,
) -> A2Allocation:
    """Compare what A2 received against what matching on the declared resource would owe it.

    ``post_feedback_spend_observed`` is required and has no default.  An A2 shard measures zero
    post-feedback spend because its arm HAS no feedback stage -- the amount it is owed is defined by
    its paired V2 run, which that shard cannot see.  Reading its own zero as "nothing to match"
    would let the control arm certify a matching nobody ever computed, which is the same false
    claim this whole object exists to prevent.
    """
    for name, ledger in (("pre_feedback_cost", pre_feedback_cost),
                         ("v2_post_feedback_cost", v2_post_feedback_cost),
                         ("per_lookahead_cost", per_lookahead_cost)):
        if not isinstance(ledger, ResourceLedger):
            raise V2A2Error(f"{name} must be a ResourceLedger")
    if matching_resource not in A2_RESOURCE_COMPONENTS:
        raise V2A2Error(
            f"matching_resource {matching_resource!r} is not in the closed vocabulary "
            f"{sorted(A2_RESOURCE_COMPONENTS)}"
        )
    executed = int(executed_extra_lookaheads)
    complement = tuple(sorted(set(A2_RESOURCE_COMPONENTS) - {matching_resource}))
    common = dict(
        matching_resource=matching_resource, executed_extra_lookaheads=executed,
        pre_feedback_cost=pre_feedback_cost, v2_post_feedback_cost=v2_post_feedback_cost,
        per_lookahead_cost=per_lookahead_cost, unmatched_components=complement,
    )

    if not post_feedback_spend_observed:
        return A2Allocation(
            status="not_derivable_in_this_shard", matched=False, owed_extra_lookaheads=None,
            unmatched_amount=None, residual=None, extra_fork_seeds=(),
            detail=(
                "this shard never ran the feedback stage, so the post-feedback spend that defines "
                "the matched allocation is not observable here; the amount A2 is owed comes from "
                "the paired V2 run"
            ),
            **common,
        )

    matched_amount = v2_post_feedback_cost.component(matching_resource)
    per = per_lookahead_cost.component(matching_resource)
    if per <= 0.0:
        return A2Allocation(
            status="unit_cost_not_measured", matched=False, owed_extra_lookaheads=None,
            unmatched_amount=matched_amount, residual=None, extra_fork_seeds=(),
            detail=(
                f"no per-lookahead {matching_resource} cost was measured, so the {matched_amount} "
                "V2 spent after this view cannot be converted into extra exact futures"
            ),
            **common,
        )

    budget = matched_extra_lookahead_budget(
        v2_post_feedback_cost=v2_post_feedback_cost, matching_resource=matching_resource,
        per_lookahead_cost=per_lookahead_cost, used_seeds=used_seeds, seed_base=seed_base,
    )
    owed = budget.n_extra_lookaheads
    # Measured in the RESOURCE, not in whole lookaheads: an amount too small to buy a lookahead is
    # still an amount one arm received and the other did not.
    unmatched_amount = float(matched_amount - executed * per)
    if matched_amount == 0.0 and executed == 0:
        status, matched = "nothing_to_match", True
        detail = ("V2 spent no " + matching_resource + " after this view, so there is nothing to "
                  "reassign to A2")
    elif unmatched_amount == 0.0:
        status, matched = "matched", True
        detail = (f"A2 received {executed} extra exact future(s), matching the {matched_amount} "
                  f"{matching_resource} V2 spent after this view")
    elif unmatched_amount < 0.0:
        status, matched = "over_allocated", False
        detail = (f"A2 received {executed} extra exact future(s), which is "
                  f"{-unmatched_amount} {matching_resource} MORE than V2 spent after this view")
    elif executed == 0:
        status, matched = "declared_but_not_executed", False
        detail = (
            f"A2 received NO extra exact futures; V2 spent {matched_amount} {matching_resource} "
            f"after this view (a projection, a propagation segment and a descendant pool), which "
            f"would buy {owed} extra lookahead(s) at the measured unit cost of {per}.  These arms "
            "are not compute-matched."
        )
    else:
        status, matched = "partially_executed", False
        detail = (
            f"A2 received {executed} of the {owed} extra exact futures the {matched_amount} "
            f"{matching_resource} V2 spent after this view would buy; {unmatched_amount} "
            f"{matching_resource} remains unmatched"
        )
    return A2Allocation(
        status=status, matched=matched, owed_extra_lookaheads=owed,
        unmatched_amount=unmatched_amount, residual=budget.residual,
        extra_fork_seeds=budget.extra_fork_seeds, detail=detail, **common,
    )
