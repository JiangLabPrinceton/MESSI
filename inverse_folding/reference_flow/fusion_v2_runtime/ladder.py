"""V2F6: the general D>=1 depth ladder, stationary comparator, and typed stopping (PLAN §4.4-4.5).

Generalizes the verified one-cycle transition to depth ``D >= 1`` "without changing its semantics",
and runs one compact stationary comparator "through the same segment API".

Three design commitments, each blocking a specific way the generalization could become a different
mechanism:

**One loop, parameterized by a coordinate law.**  PLAN V2F6 requires "both laws call the same
segment executor".  A stationary comparator running through its own code would measure the second
implementation rather than the schedule, and the comparison would be meaningless.

**Breadth and depth are DECLARED, never allocated.**  PLAN's revision separates implementing the
recursion from authorizing real ``D>1`` runs, so nothing here needs the refold-dominated cost model:
:class:`DepthPlan` states how many lookaheads each depth gets and the engine obeys it.  An automatic
allocator would be a scientific decision smuggled into an implementation.

**Capability is not authorization.**  Production ``D>1`` ships behind an explicit flag that defaults
to OFF.  PLAN keeps the real ladder locked until a powered one-cycle source-transmission gate and
the FeedbackSupportPolicy directionality gate both pass; when they do, only the runbook and config
change -- not this module.

Stopping is typed throughout (PLAN §4.5), and every stop returns the best existing definitive
archive state.  A stop never destroys the archive or replaces its elite with a worse descendant.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import with_reference_flow_overrides
from ..fusion_v2.errors import V2Error
from ..fusion_v2.identity import canonical_digest
from ..fusion_v2.reward import DEPTH0_BOOTSTRAP_RULE
from ..fusion_v2.schedule import CoordinateLaw, make_cycle
from ..fusion_v2.seeds import (
    ROOT_CAPTURE_MAX_ATTEMPTS,
    V2_SEED_ENCODING_VERSION,
    FeedbackPairSeedContext,
    V2SeedContext,
)
from ..fusion_v2.state import CompleteEndpoint, TransitionOutcome
from .archive import ExactArchive
from ..fusion_v2.safety import advance_lineage, bind_immediate_parent
from .admission import SafetyGate
from .capture import FullyResolvedRootError, capture_depth_zero

__all__ = [
    "V2LadderError",
    "ProductionDepthDisabledError",
    "StoppingReason",
    "DepthPlan",
    "DepthRecord",
    "LadderOutcome",
    "run_depth_ladder",
]

#: Identifies the executor both laws run through.  Asserted equal across laws so a future refactor
#: that forked the implementation is caught rather than silently shipped.
SEGMENT_EXECUTOR_ID = "fusion_v2_runtime.segment.run_projected_segment/1"


class V2LadderError(V2Error):
    """A ladder or schedule contract violation."""


class ProductionDepthDisabledError(V2LadderError):
    """A production ``D>1`` run was requested without authorization.

    PLAN V2F6: production ``D>1`` "remains launch-disabled until both a powered real one-cycle
    source-transmission gate and the frozen FeedbackSupportPolicy one-cycle reward-directionality
    gate pass".  The capability is implemented; the authorization is a runbook decision.
    """


class StoppingReason(str, enum.Enum):
    """PLAN §4.5's typed stopping reasons.  An untyped stop cannot be aggregated over a cohort."""

    DEPTH_CAP = "depth_cap"
    NO_ADMISSIBLE_ENDPOINT = "no_admissible_endpoint"
    INVALID_PROJECTION = "invalid_projection"
    NO_NOVEL_DESCENDANT = "no_novel_descendant"
    FRONTIER_PLATEAU = "frontier_plateau"
    DIVERSITY_COLLAPSE = "diversity_collapse"
    OPERATIONAL_CEILING = "operational_ceiling"
    NO_PRETERMINAL_ROOT = "no_preterminal_root"
    TERMINAL_BEST_LOOKAHEAD = "terminal_best_lookahead"


@dataclass(frozen=True)
class DepthPlan:
    """A declared schedule: one ``(r_d, c_d, c_{d+1})`` triple and one breadth per depth.

    Validated at construction, before any model exists, so a malformed ladder fails at
    ``--dry-run`` rather than after paying for a prefix.
    """

    law: CoordinateLaw
    depth_cap: int
    cycles: tuple[tuple[int, int, int], ...]
    lookaheads_per_depth: tuple[int, ...]
    n_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.law, CoordinateLaw):
            raise V2LadderError("law must be a CoordinateLaw")
        cap = int(self.depth_cap)
        if cap < 1:
            raise V2LadderError(f"depth_cap must be >= 1, got {cap}")
        if len(self.cycles) != cap:
            raise V2LadderError(
                f"depth_cap={cap} but {len(self.cycles)} cycle(s) declared; the schedule must state "
                "one coordinate triple per depth"
            )
        if len(self.lookaheads_per_depth) != cap:
            raise V2LadderError(
                f"lookaheads_per_depth has {len(self.lookaheads_per_depth)} entries for "
                f"depth_cap={cap}; breadth is declared per depth, never inferred"
            )
        if any(int(k) < 1 for k in self.lookaheads_per_depth):
            raise V2LadderError("every depth needs at least one lookahead")

        for depth, (r_step, c_source, c_next) in enumerate(self.cycles):
            # Delegated so the ladder cannot drift from the coordinate law the rest of V2 enforces.
            make_cycle(depth=depth, r_step=int(r_step), c_source_step=int(c_source),
                       c_next_step=int(c_next), n_steps=int(self.n_steps), law=self.law)
            if depth and int(c_source) != int(self.cycles[depth - 1][2]):
                raise V2LadderError(
                    f"depth {depth} sources at step {c_source} but depth {depth - 1} ends at "
                    f"{self.cycles[depth - 1][2]}; the cycles must chain or a segment is skipped"
                )

    def lookaheads_at(self, depth: int) -> int:
        return int(self.lookaheads_per_depth[int(depth)])

    def coordinates_at(self, depth: int) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.cycles[int(depth)])          # type: ignore[return-value]


@dataclass(frozen=True)
class DepthRecord:
    """What one rung of the ladder produced."""

    depth: int
    cycle: Any
    source_fork_seeds: tuple[int, ...]
    descendant_fork_seeds: tuple[int, ...]
    propagation_seed: int
    n_selected_lineages: int
    substrate_digest: str


@dataclass(frozen=True)
class LadderOutcome:
    """The whole ladder, plus what survived it."""

    stopping_reason: StoppingReason
    depth_reached: int
    cycles: tuple[DepthRecord, ...]
    archive: ExactArchive
    best_definitive: CompleteEndpoint | None
    total_logical_dfe: int
    substrate_digest: str
    production_depth_authorized: bool
    #: A deliberately non-production D>1 launch.  Kept separate from production authorization so
    #: an exploratory result can never be relabelled as evidence that the mechanism gates passed.
    exploratory_depth_override: bool
    #: The depth-0 prefix the LADDER paid, captured here rather than inside the first cycle.  It is
    #: its own field because it is the one piece of work no ``DepthRecord`` owns, and folding it
    #: into a rung would put it back on a record that did not run it.
    root_capture_logical_dfe: int = 0
    #: The safety ratchet as the ladder actually left it.  Its depth equals ``depth_reached``, so a
    #: stopped ladder cannot claim a lineage it never advanced through.
    final_safety_ledger: Any = None
    segment_executor_id: str = SEGMENT_EXECUTOR_ID
    detail: str = ""
    root_capture_attempts_used: int = 0
    root_capture_seed: int | None = None
    root_capture_status: str = "not_attempted"
    root_capture_detail: str = ""
    root_capture_n_unresolved_editable: int | None = None
    root_capture_rho_edit: float | None = None


def _substrate_digest(config: Any) -> str:
    """Content identity of the frozen substrate every causal arm must share (PLAN V2F6)."""
    return canonical_digest({
        "n_steps": int(config.sampler.n_steps),
        "temperature": float(config.sampler.temperature),
        "base_form": str(config.schedule.base_form),
        "amplification_form": str(config.amplification.form),
        "remask_enabled": bool(config.sampler.remask.enabled),
        "remask_fraction_scale": float(config.sampler.remask.fraction_scale),
        "h_shuffle_enabled": bool(config.h_shuffle.enabled),
    })


def run_depth_ladder(
    *,
    plan: DepthPlan,
    campaign_id: str,
    split_role: str,
    master_seed: int,
    phase: str | None = None,
    allow_production_depth_gt_1: bool = False,
    exploratory_depth_override: bool = False,
    root_capture_max_retries: int = ROOT_CAPTURE_MAX_ATTEMPTS - 1,
    **cycle_kwargs: Any,
) -> LadderOutcome:
    """Run the declared ladder, one rung at a time, through the one segment executor.

    Each rung's propagated capture becomes the next rung's source, so the ladder is a chain rather
    than a set of independent runs.  One archive spans every depth: PLAN §2.3 makes it the run's
    memory, and a per-depth archive would lose the elite the moment a generation regressed.
    """
    from .cycle import run_one_cycle

    if not isinstance(plan, DepthPlan):
        raise V2LadderError("plan must be a DepthPlan")
    if (isinstance(root_capture_max_retries, bool)
            or not isinstance(root_capture_max_retries, int)
            or root_capture_max_retries < 0):
        raise V2LadderError(
            "root_capture_max_retries must be a non-negative int, got "
            f"{root_capture_max_retries!r}"
        )
    exploratory = bool(exploratory_depth_override)
    production = bool(allow_production_depth_gt_1)
    if exploratory and production:
        raise ProductionDepthDisabledError(
            "production_depth_authorized and exploratory_depth_override are mutually exclusive; "
            "an exploratory result may not carry a production authorization label"
        )
    if exploratory:
        if plan.depth_cap <= 1:
            raise ProductionDepthDisabledError(
                "exploratory_depth_override is only meaningful for D>1; depth-one execution is "
                "already allowed without an override"
            )
        if phase != "capability_ladder" or not str(split_role).startswith("exploratory"):
            raise ProductionDepthDisabledError(
                "exploratory D>1 requires phase='capability_ladder' and a split_role beginning "
                f"with 'exploratory'; got phase={phase!r}, split_role={split_role!r}"
            )
    if plan.depth_cap > 1 and not (production or exploratory):
        raise ProductionDepthDisabledError(
            f"depth_cap={plan.depth_cap} requires production D>1, which is launch-disabled: PLAN "
            "V2F6 unlocks it only after a powered one-cycle source-transmission gate and the "
            "FeedbackSupportPolicy directionality gate both pass.  The capability is implemented; "
            "pass allow_production_depth_gt_1=True for a capability test."
        )

    # Checked HERE rather than left to the first cycle, so a misconfigured arm fails before the
    # root prefix is paid for.  ``feedback_enabled`` is the run's arm identity and ``cost_meter``
    # is the only thing that can record work started but never finished; neither has a default
    # that is safe to be wrong about (see ``run_one_cycle``).
    for required in ("feedback_enabled", "cost_meter"):
        if required in cycle_kwargs:
            continue
        raise V2LadderError(
            f"{required} must be supplied explicitly; the ladder runs real oracles and pays a root "
            "prefix before the first cycle, so neither the arm identity nor the compute journal "
            "has a defensible default"
        )
    config = cycle_kwargs["config"]
    substrate = _substrate_digest(config)

    lineage = cycle_kwargs.pop("lineage")
    protein_id = lineage.protein_id
    #: Run-level identity for the ordinary namespaces.  Every seed below is derived from it, so a
    #: seed is a function of WHICH RUN this is -- campaign, split, master seed and protein -- and
    #: not of the order in which the loop happened to ask for one.
    run_seeds = V2SeedContext(
        seed_schema=V2_SEED_ENCODING_VERSION, campaign_id=campaign_id, split_role=split_role,
        master_seed=int(master_seed), protein_id=protein_id,
    )

    def _pair_seeds(depth: int, r_step: int, c_next: int, breadth: int) -> FeedbackPairSeedContext:
        """The matched-descendant identity for one rung.

        A propagation from ``r_d`` to ``c_{d+1}`` is exactly the object PLAN §2.6 pairs, so it draws
        from the ``matched_descendant`` namespace whether or not this particular run has a partner
        arm.  Using a private ladder namespace would mean a ladder rung and the A arm of a pair at
        the same coordinates ran different streams, and the ladder could never be read as one arm.
        """
        return FeedbackPairSeedContext(
            seed_schema=V2_SEED_ENCODING_VERSION, campaign_id=campaign_id, split_role=split_role,
            master_seed=int(master_seed), protein_id=protein_id, depth=int(depth),
            r_step=int(r_step), c_next_step=int(c_next), pair_ordinal=0,
            n_forks=1, n_descendant_lookaheads=int(breadth),
        )

    safety_gate = cycle_kwargs["safety_gate"]
    dual = cycle_kwargs.get("dual")
    admission_by_endpoint: dict[str, Any] = {}
    # PLAN DUALF3: the archive elite is maintained under the SAME objective the donor gate applies.
    # An archive still ordered by ``head_global_risk`` would make a joint run's elite silently
    # single-allele while every artifact continued to claim otherwise.
    archive = ExactArchive(rank_key=None if dual is None else dual.rank_key())
    records: list[DepthRecord] = []
    reason = StoppingReason.DEPTH_CAP
    detail = ""
    used_seeds: set[int] = set()

    def _claim(seeds: Sequence[int]) -> tuple[int, ...]:
        """Record seeds and refuse a repeat.

        A reused seed would regenerate an endpoint the archive already holds, so the extra work
        would measure nothing.  Every seed here is a hash of a distinct typed tuple, so a collision
        is not an expected event to route around -- it is evidence that two different rungs derived
        the same identity, and continuing would silently produce a ladder whose depths are not
        independent.
        """
        claimed = tuple(int(seed) for seed in seeds)
        repeated = sorted(used_seeds.intersection(claimed))
        if repeated or len(set(claimed)) != len(claimed):
            raise V2LadderError(
                f"seed collision: {repeated or sorted(claimed)} was derived twice; two rungs share "
                "a random stream, so the deeper one measures nothing new"
            )
        used_seeds.update(claimed)
        return claimed

    # The root is captured HERE rather than inside the first cycle, so its state id exists before
    # any seed is drawn: PLAN §2.6 binds a lookahead seed to the state it forks from, and a seed
    # cannot name a state that does not exist yet.  Every rung then receives an explicit source,
    # which also makes depth 0 and depth d>0 the same code path.
    #: The prefix the capture below actually runs.  No rung is charged for it (see CycleCost).
    root_prefix_dfe = int(plan.cycles[0][1])
    root_capture_dfe = 0
    root_capture_attempts = 0
    root_capture_seed = None
    root_capture_status = "fully_resolved"
    root_capture_detail = ""
    root_capture_n_unresolved = 0
    root_capture_rho = 1.0
    source_override = None
    cost_meter = cycle_kwargs["cost_meter"]
    # Journaled here for the same reason the cycle journals its own capture: this prefix is real
    # forward passes, and a process that dies inside it must leave a record that it was started.
    root_event = f"{protein_id}:{lineage.family_id}:ladder_root"
    for attempt_index in range(root_capture_max_retries + 1):
        seed = int(config.sampler.seed) if attempt_index == 0 else run_seeds.depth0_root_seed(
            checkpoint_step=root_prefix_dfe, root_index=attempt_index)
        seed = _claim((seed,))[0]
        root_capture_seed = seed
        attempt_config = config if attempt_index == 0 else with_reference_flow_overrides(
            config, seed=seed)
        event_id = root_event if attempt_index == 0 else f"{root_event}:retry{attempt_index}"
        root_capture_attempts += 1
        root_capture_dfe += root_prefix_dfe
        with cost_meter.attempt(
            event_id=event_id, phase="root_capture", request_kind="prefix_capture",
            request_digest=canonical_digest({
                "length": int(cycle_kwargs["sequence_length"]),
                "at_step": root_prefix_dfe, "seed": seed,
            }),
            logical_dfe=root_prefix_dfe,
        ) as receipt:
            try:
                source_override = capture_depth_zero(
                    sampler=cycle_kwargs["sampler"], denoiser=cycle_kwargs["denoiser"],
                    config=attempt_config,
                    sequence_length=int(cycle_kwargs["sequence_length"]),
                    h_values=cycle_kwargs["h_values"],
                    residue_token_ids=cycle_kwargs["residue_token_ids"], at_step=root_prefix_dfe,
                    fixed_tokens=cycle_kwargs["fixed_tokens"], lineage=lineage,
                    mask_token_id=int(cycle_kwargs["mask_token_id"]),
                    aa_token_ids=cycle_kwargs["aa_token_ids"],
                    conditioning=cycle_kwargs["conditioning"],
                    safety_reference=cycle_kwargs["safety_reference"],
                    cost_event_ids=("evt:root",), struct=cycle_kwargs.get("struct"),
                )
            except FullyResolvedRootError as exc:
                root_capture_detail = str(exc)
                receipt.observe(physical_forwards=root_prefix_dfe)
                continue
            receipt.observe(physical_forwards=root_prefix_dfe)
        root_capture_status = "captured"
        root_capture_detail = ""
        root_capture_n_unresolved = source_override.realized_maturity.n_unresolved_editable
        root_capture_rho = source_override.realized_maturity.rho_edit
        break

    if source_override is None:
        return LadderOutcome(
            stopping_reason=StoppingReason.NO_PRETERMINAL_ROOT, depth_reached=0, cycles=(),
            archive=archive, best_definitive=None, total_logical_dfe=root_capture_dfe,
            substrate_digest=substrate, production_depth_authorized=production,
            exploratory_depth_override=exploratory, root_capture_logical_dfe=root_capture_dfe,
            final_safety_ledger=safety_gate.ledger,
            detail=(f"no pre-terminal root after {root_capture_attempts} deterministic attempts: "
                    f"{root_capture_detail}"),
            root_capture_attempts_used=root_capture_attempts, root_capture_seed=root_capture_seed,
            root_capture_status=root_capture_status, root_capture_detail=root_capture_detail,
            root_capture_n_unresolved_editable=root_capture_n_unresolved,
            root_capture_rho_edit=root_capture_rho,
        )
    inherited_endpoints = None
    #: Every sequence-equivalence class the ladder has already produced.  PLAN §4.5 types
    #: ``no novel descendant`` as a stopping reason but does not define novelty; the predicate is
    #: chosen here to be the archive's own ``sequence_equivalence_key``, which is the equivalence
    #: the rest of V2 already collapses families on.
    seen_equivalence_keys: set[str] = set()
    for depth in range(plan.depth_cap):
        r_step, c_source, c_next = plan.coordinates_at(depth)
        breadth = plan.lookaheads_at(depth)
        next_breadth = plan.lookaheads_at(min(depth + 1, plan.depth_cap - 1))
        pair = _pair_seeds(depth, r_step, c_next, next_breadth)
        # Only depth 0 forks a source pool.  Every deeper rung INHERITS the previous rung's
        # descendants (see below), so drawing source seeds there would mint identities nothing uses
        # and make the record claim work the run never did.
        source_seeds = _claim(
            run_seeds.lookahead_seed(depth=0, source_state_id=source_override.state_id,
                                     lookahead_index=index)
            for index in range(breadth)
        ) if depth == 0 else ()
        descendant_seeds = _claim(
            pair.matched_descendant_lookahead_seed(0, index) for index in range(next_breadth))
        propagation_seed = _claim((pair.matched_descendant_seed(0),))[0]

        cycle = run_one_cycle(
            lineage=lineage, c_source_step=c_source, r_step=r_step, c_next_step=c_next,
            source_fork_seeds=source_seeds, descendant_fork_seeds=descendant_seeds,
            descendant_propagation_seed=propagation_seed,
            origin_transition_id=f"{lineage.protein_id}:{lineage.family_id}:txn:d{depth}:"
                                 f"r{r_step}-c{c_next}:" + canonical_digest({
                                     "depth": depth, "r": r_step, "c": c_next,
                                     "seed": propagation_seed})[:12],
            coordinate_law=plan.law,
            source_override=source_override,
            source_endpoints=inherited_endpoints,
            archive=archive,
            **cycle_kwargs,
        )

        records.append(DepthRecord(
            depth=depth, cycle=cycle, source_fork_seeds=source_seeds,
            descendant_fork_seeds=descendant_seeds, propagation_seed=propagation_seed,
            # One selected lineage per rung: PLAN §4.4's one-cycle law, and the reason duplicate
            # siblings cannot buy extra ancestry mass.
            n_selected_lineages=1 if cycle.selected_endpoint is not None else 0,
            substrate_digest=substrate,
        ))

        if cycle.outcome is TransitionOutcome.TERMINAL_BEST_LOOKAHEAD:
            reason, detail = StoppingReason.TERMINAL_BEST_LOOKAHEAD, cycle.detail
            break
        if cycle.outcome is TransitionOutcome.NULL_NO_ADMISSIBLE_ENDPOINT:
            reason, detail = StoppingReason.NO_ADMISSIBLE_ENDPOINT, cycle.detail
            break
        if cycle.outcome is TransitionOutcome.STALLED_NO_NOVEL_DESCENDANT:
            # The rung's segment ran to a complete sequence.  The ladder stops with the reason the
            # cycle already typed rather than re-deriving one, so the two layers cannot disagree.
            reason, detail = StoppingReason.NO_NOVEL_DESCENDANT, cycle.detail
            break
        if cycle.outcome is not TransitionOutcome.COMMITTED:
            reason, detail = StoppingReason.INVALID_PROJECTION, cycle.detail
            break
        if cycle.propagated is None:
            reason, detail = StoppingReason.NO_NOVEL_DESCENDANT, "no descendant capture"
            break

        # ---- advance the safety ratchet one depth (PLAN §2.7) --------------------------------
        # The cumulative reference is carried by OBJECT IDENTITY through advance_lineage, which has
        # no parameter through which a different one could enter: that is what makes the ratchet
        # inescapable.  What advances is the lineage's position and its immediate parent, so the
        # next rung's incremental gate has a reference at all.
        selected = cycle.selected_endpoint
        admission = next(
            (a for a in (cycle.admissions or ()) + (cycle.descendant_admissions or ())
             if a.endpoint_id == selected.endpoint_id and a.verdict is not None),
            admission_by_endpoint.get(selected.endpoint_id),
        )
        if admission is None or admission.verdict is None:
            reason, detail = (
                StoppingReason.NO_ADMISSIBLE_ENDPOINT,
                f"selected endpoint {selected.endpoint_id} carries no admissibility verdict, so "
                "the safety ratchet cannot record what the lineage descended through",
            )
            break
        safety_gate = SafetyGate(
            policy=safety_gate.policy,
            ledger=advance_lineage(
                safety_gate.ledger, depth=depth + 1,
                selected=bind_immediate_parent(
                    endpoint_id=selected.endpoint_id, head_score=selected.head_score,
                    depth=depth + 1, policy=safety_gate.policy),
                admissibility=admission.verdict,
            ),
        )
        cycle_kwargs["safety_gate"] = safety_gate
        for record in (cycle.admissions or ()) + (cycle.descendant_admissions or ()):
            admission_by_endpoint[record.endpoint_id] = record

        # ---- advance the reward incumbent one depth (PLAN §2.5 / V2F5A) --------------------
        # The safety ratchet above and the reward incumbent are deliberately different objects.
        # For HeadDirectedCappedPolicy the donor that entered this lineage becomes I_{d+1}; if the
        # policy object remained the factory-built depth-zero instance, every deeper rung would
        # compare against I0 and could adopt a donor that regresses from the current lineage.
        # The policy the CYCLE decided with, not the one this loop handed it.  Under Dual the
        # kernel rebinds the injected policy with the pool's role B donor scores, and advancing the
        # un-rebound original looked the adopted donor up in an empty map and raised -- so every
        # Dual ladder run died at the first adopted donor.  A legacy cycle returns the same object
        # it was given, so this is a no-op there.
        support_policy = cycle.support_policy or cycle_kwargs["support_policy"]
        advance_reward = getattr(support_policy, "advance_lineage_incumbent", None)
        if advance_reward is not None:
            verdict = getattr(cycle.policy_evidence, "donor_gate", None)
            bootstrap = (
                depth == 0
                and getattr(support_policy, "depth0_incumbent_rule", None)
                    == DEPTH0_BOOTSTRAP_RULE
                and getattr(cycle.policy_evidence, "reward_gate_kind", None)
                    == DEPTH0_BOOTSTRAP_RULE
            )
            if verdict is None and not bootstrap:
                raise V2LadderError(
                    "a committed incumbent-aware policy emitted no donor-gate verdict; the next "
                    "depth cannot know which reference its reward comparison advanced from"
                )
            cycle_kwargs["support_policy"] = advance_reward(
                donor=selected, verdict=verdict, accepted_at_depth=depth + 1)

        source_override = cycle.propagated
        # Depth d's descendant pool IS depth d+1's source pool: re-screening the same state would
        # charge its tail twice (PLAN §3.4) and mint colliding endpoint ids.
        inherited_endpoints = cycle.descendant_endpoints
        lineage = cycle.propagated.lineage
        if not inherited_endpoints:
            reason, detail = StoppingReason.NO_NOVEL_DESCENDANT, "no descendant endpoints"
            break

        # A NON-EMPTY pool is not automatically a novel one.  A run whose descendants only re-derive
        # sequences an earlier depth already produced advances no search: it would reach the depth
        # cap, report ``depth_cap``, and leave an archive whose row count is depth times breadth
        # over a handful of distinct designs.  Every per-depth rate read off that artifact would be
        # computed over duplicates, so depth would read as improvement where it was repetition.
        seen_equivalence_keys.update(
            endpoint.sequence_md5 for endpoint in cycle.endpoints)
        novel = [endpoint for endpoint in inherited_endpoints
                 if endpoint.sequence_md5 not in seen_equivalence_keys]
        if not novel:
            reason, detail = (
                StoppingReason.NO_NOVEL_DESCENDANT,
                f"all {len(inherited_endpoints)} descendant(s) repeat a sequence-equivalence class "
                f"an earlier depth already produced ({len(seen_equivalence_keys)} distinct so far)",
            )
            break
        seen_equivalence_keys.update(
            endpoint.sequence_md5 for endpoint in inherited_endpoints)

    depth_reached = sum(
        1 for record in records if record.cycle.outcome is TransitionOutcome.COMMITTED)
    # The ladder's total is the sum of what every rung REALIZED plus the root prefix the ladder
    # itself paid.  The previous form -- screening + segment per rung -- was neither: it charged an
    # inherited rung's prefix a second time (no forward pass corresponds to it) and never charged
    # the DEEPEST rung's descendant pool at all, because that pool only reached the total through
    # the NEXT rung's inherited screening, and the last rung has no next.
    total = root_capture_dfe + sum(record.cycle.cost.total_logical_dfe for record in records)

    return LadderOutcome(
        stopping_reason=reason, depth_reached=depth_reached, cycles=tuple(records),
        archive=archive,
        # PLAN §4.5: any stop returns the best existing definitive archive state.
        best_definitive=archive.elite(),
        total_logical_dfe=total, substrate_digest=substrate,
        production_depth_authorized=production,
        exploratory_depth_override=exploratory,
        final_safety_ledger=safety_gate.ledger, detail=detail,
        root_capture_logical_dfe=root_capture_dfe,
        root_capture_attempts_used=root_capture_attempts, root_capture_seed=root_capture_seed,
        root_capture_status=root_capture_status, root_capture_detail=root_capture_detail,
        root_capture_n_unresolved_editable=root_capture_n_unresolved,
        root_capture_rho_edit=root_capture_rho,
    )
