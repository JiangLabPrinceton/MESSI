"""V2F5-1: the one-cycle runner (PLAN task V2F5).

One complete feedback cycle, wired from the parts V2F3 and V2F4 built:

    source checkpoint  ->  K exact lookaheads  ->  Head scoring  ->  structure promotion
                       ->  A2 view (before feedback)
                       ->  definitive selection  ->  q_phi projection
                       ->  propagation segment  ->  descendant capture
                       ->  descendant lookaheads

**Scientific boundary, restated because it is easy to lose.**  PLAN V2F5: "local tests validate
wiring only.  Real feedback transmission remains a cluster mechanism gate and must not be inferred
from fake-oracle success."  A green cycle here means every state edge is reconstructible and every
identity is explicit.  It does not mean feedback transmits.

**Ordering is a contract, not a convenience.**  The A2 view is snapshotted BEFORE selection, because
PLAN §2.3 defines it as "the archive state immediately before feedback"; taking it later would let
post-feedback descendants leak into the control arm.  Structure promotion happens BEFORE selection,
because PLAN §4.2 forbids a non-definitive endpoint from becoming ancestry.  Every null outcome is
typed and returned as a value (PLAN §4.5) -- a cycle that cannot proceed is a result, not an error.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..fusion_v2.errors import V2Error
from ..fusion_v2.identity import canonical_digest
from ..fusion_v2.projection import source_writeback
from ..fusion_v2.schedule import CoordinateLaw, make_cycle
from ..fusion_v2.state import (
    CompleteEndpoint,
    FeasibilityLevel,
    LivePartialState,
    ProjectedPartialState,
    TransitionOutcome,
    validate_propagated_capture,
)
from .a2_view import A2View, snapshot_a2_view
from .admission import EndpointAdmission, admit_endpoint
from .archive import ExactArchive, select_family_representatives
from .capture import capture_depth_zero
from .lookahead import (
    OracleRequest,
    bind_head_scores,
    generate_lookaheads,
    oracle_request,
    oracle_request_for_endpoint,
)
from .segment import (
    adapt_segment_to_live_state,
    run_projected_segment,
    segment_is_terminal,
)

__all__ = ["V2CycleError", "CycleCost", "CycleOutcome", "run_one_cycle"]


class V2CycleError(V2Error):
    """A cycle wiring contract violation -- a caller mistake, not a scientific outcome."""


@dataclass(frozen=True)
class CycleCost:
    """REALIZED logical lane-DFE: what this cycle actually ran, decomposed per PLAN §3.4.

    Every field is zero unless the work behind it happened.  A projected cost computed up front and
    returned unchanged on every exit path is not an accounting -- it is the same number for a cycle
    that completed, a cycle that stopped at the A2 view, and a cycle that refused to project.  PLAN
    §4.3 matches the two arms on one declared resource read out of exactly these fields, so a
    control arm billed for a segment it never ran has been given less than the comparison claims.

    The four fields PARTITION the work and never overlap, so ``total_logical_dfe`` is the execution
    graph itself.  The prefix is deliberately separate from the screening tails: a ladder rung that
    inherits its source has already had both paid by the rung that ran them, and a field that
    bundled them could only be charged twice or dropped entirely.  ``screen_logical_dfe`` recovers
    PLAN §3.4's ``C_screen = c + K(S-c)`` from the two of them.
    """

    #: Charged ONLY by the cycle that captured the source itself.  A rung handed an explicit source
    #: pays nothing: the prefix is on the record of whoever ran it.
    source_prefix_logical_dfe: int = 0
    #: The lookahead TAILS actually generated -- ``sum(S - c)`` over realized completions.  Zero
    #: when the rung inherited its source pool.
    screening_logical_dfe: int = 0
    #: ``c_{d+1} - r_d``, and only once a propagation segment has actually run.
    segment_logical_dfe: int = 0
    #: The descendant tails actually generated.
    descendant_screening_logical_dfe: int = 0

    @property
    def screen_logical_dfe(self) -> int:
        """PLAN §3.4's ``C_screen = c + K(S-c)`` for the cycles that paid a prefix."""
        return self.source_prefix_logical_dfe + self.screening_logical_dfe

    @property
    def total_logical_dfe(self) -> int:
        return (self.source_prefix_logical_dfe + self.screening_logical_dfe
                + self.segment_logical_dfe + self.descendant_screening_logical_dfe)


@dataclass(frozen=True)
class CycleOutcome:
    """Every artefact of one cycle, linked by identity.

    A null outcome carries everything produced up to the point it stopped, so a cycle that found no
    admissible endpoint is still fully auditable.
    """

    outcome: TransitionOutcome
    source: LivePartialState
    endpoints: tuple[CompleteEndpoint, ...]
    a2_view: A2View
    archive: ExactArchive
    cost: CycleCost
    selected_endpoint: CompleteEndpoint | None = None
    projected: ProjectedPartialState | None = None
    segment: Any = None
    propagated: LivePartialState | None = None
    descendant_endpoints: tuple[CompleteEndpoint, ...] = ()
    #: One admission decision per scored endpoint, in scoring order.  A run must be able to say WHY
    #: an endpoint was or was not eligible, not merely that it was not.
    admissions: tuple[EndpointAdmission, ...] = ()
    descendant_admissions: tuple[EndpointAdmission, ...] = ()
    #: The identity carried by the policy DECISION this cycle acted on -- ``None`` when no policy
    #: was asked (A2 stops before feedback).  Carried here because it is otherwise discarded: the
    #: projected state records the support partition but not who chose it, and re-reading it off
    #: the policy object afterwards is a SECOND source that can disagree with the answer the kernel
    #: actually acted on.  PLAN §5.3 makes the policy a load-bearing field of every feedback event.
    policy_identity: Any = None
    #: The policy's own per-position decision record, when it produced one (PLAN §5.3, V2F5A:
    #: "per-position decision evidence for every accepted and rejected write/reopen candidate").
    #: ``None`` for every policy that emits only the four support sets -- the diagnostic probe has
    #: no candidate ranking to record.  Present on a STALL as well as on a decision: PLAN §2.5
    #: requires a typed stall to preserve its rejected-candidate counts rather than vanish.
    policy_evidence: Any = None
    #: The policy object this cycle ACTUALLY decided with.  Under Dual the kernel rebinds the
    #: injected policy with this pool's role B donor scores before calling it, so the object that
    #: decided is not the object the caller handed in -- and the ladder advances the lineage
    #: incumbent on whatever it holds.  Advancing the un-rebound original looked up the adopted
    #: donor in an EMPTY role B score map and raised, killing every Dual ladder run at the first
    #: adopted donor, after the full generation and both Head batches had been paid for.
    #: ``None`` when no policy was consulted (a null cycle, or the A2 view).
    support_policy: Any = None
    detail: str = ""

    @property
    def committed(self) -> bool:
        return self.outcome is TransitionOutcome.COMMITTED


def _score_and_promote(
    *, completions, head_oracle, structure_oracle, evaluator, window_grid_digest,
    archive: ExactArchive, depth: int, safety_gate, hard_anchors,
    cost_meter, event_prefix: str, screen_event_id: str, dual=None,
) -> tuple[tuple[CompleteEndpoint, ...], tuple[EndpointAdmission, ...]]:
    """Score a completion pool, record every row, then promote on the ADMISSION conjunction.

    Recording happens before promotion so that an endpoint which fails either gate is still on the
    record.  PLAN §4.1: no scored endpoint is discarded from raw evidence.

    Promotion to ``DEFINITIVE`` requires the whole conjunction of PLAN §2.7/§4.2 -- structural
    feasibility, preserved constraints, and the whole-landscape immune gate -- because that is what
    the label is defined to mean.  Promoting on structure alone would put a design that opens a new
    epitope hotspot into the parent of everything downstream, where the cumulative ratchet can no
    longer see the hotspot as new.
    """
    # The Head is a function of the SEQUENCE, so it is asked once per DISTINCT sequence.  Duplicate
    # siblings are legal (PLAN §4.4 governs what they may buy, not whether they exist), and scoring
    # one twice would be the same measurement counted twice -- which bind_head_scores rejects.
    #
    # What travels is a TYPED REQUEST carrying the residue string, with the digest alongside as the
    # cache and join key.  Handing the digest itself to the oracle type-checks and satisfies every
    # identity-keyed fake, but a real Head tokenizes what it is given and a real refold folds it --
    # so a digest-for-sequence substitution can only fail after a cluster allocation is paid for.
    seen: set[str] = set()
    requests: list[OracleRequest] = []
    for completion in completions:
        if completion.sequence_md5 in seen:
            continue
        seen.add(completion.sequence_md5)
        requests.append(oracle_request(completion))
    # Journaled BEFORE the call (PLAN §5.4): a Head that dies mid-batch must leave a record that
    # something was started, or its burn is indistinguishable from a batch that never ran.  A Head
    # call is charged as ``head_calls``, never as logical DFE -- it is not a DFM forward, and
    # counting it as one would inflate the very total A2 matching is computed on.
    with cost_meter.attempt(
        event_id=f"{event_prefix}:head", phase="head", request_kind="head_batch",
        request_digest=canonical_digest([request.sequence_md5 for request in requests]),
        head_calls=len(requests),
    ) as receipt:
        results = head_oracle.score(requests)
        receipt.observe(physical_forwards=0, head_calls=len(requests))
    #: PLAN §5.3 lists cost among the load-bearing fields of a complete endpoint.  Without these
    #: the endpoint table and the compute ledger are two unconnected artifacts: nothing can say
    #: which screen forked a design or which Head batch scored it.  Named per POOL rather than
    #: per endpoint because that is the granularity the journal actually charges at.
    cost_event_ids = (screen_event_id, f"{event_prefix}:head")
    if dual is None:
        endpoints = bind_head_scores(
            completions=completions, results=results, evaluator=evaluator,
            window_grid_digest=window_grid_digest, cost_event_ids=cost_event_ids)
    else:
        # Imported HERE, not at module scope: PLAN §3.1 requires that a run with no Dual overlay
        # never import the Dual layer, and a module-level import would make that false for every
        # legacy run.
        from ..fusion_v2.joint_objective import AlleleRole
        from .dual_lookahead import dual_stage_event_id

        # The SAME deduplicated request set, asked of the second frozen Head.  Role A's endpoints
        # are still built by the unchanged single-Head binder, so the endpoint objects a Dual run
        # produces are the ones a single-Head run would have produced; everything joint lands in
        # the sidecar the runtime registers.
        results_b = dual.score_pool(
            requests=requests, cost_meter=cost_meter, event_prefix=event_prefix, stage="head")
        endpoints = dual.bind_pool(
            completions=completions, results_a=results, results_b=results_b,
            window_grid_digest=window_grid_digest,
            cost_event_ids=cost_event_ids + (
                dual_stage_event_id(event_prefix, "head", AlleleRole.B),))
    for endpoint in endpoints:
        archive.admit(endpoint, depth=depth)
    promoted: list[CompleteEndpoint] = []
    admissions: list[EndpointAdmission] = []
    for endpoint in endpoints:
        request = oracle_request_for_endpoint(endpoint)
        with cost_meter.attempt(
            event_id=f"{event_prefix}:structure:{endpoint.endpoint_id}", phase="structure",
            request_kind="structure_gate", request_digest=request.sequence_md5,
            structure_attempts=1,
        ) as receipt:
            outcome = structure_oracle(request)
            receipt.observe(physical_forwards=0)
        structure_definitive = bool(getattr(outcome, "feasible", False)) and \
            bool(getattr(outcome, "evaluated", False))
        admission = admit_endpoint(
            safety_gate, endpoint=endpoint, structure_definitive=structure_definitive,
            endpoint_tokens=[e.token for e in endpoint.endpoint_provenance_evidence_by_pos],
            hard_anchors=hard_anchors,
        )
        admissions.append(admission)
        passed = admission.admitted
        # A failed verdict is recorded on the ARCHIVE ROW, never on the endpoint record.  The
        # state layer allows an evaluated structure outcome only on a DEFINITIVE endpoint, and
        # DEFINITIVE requires a PASSING verdict -- so "evaluated and failed" is bookkeeping about
        # a design, not a property of it.  The failure is therefore explicit in the ledger
        # (PLAN §4.2 requires it to stay explicit) while the endpoint stays unvalidated and
        # ineligible for ancestry.  A design that folds but fails the immune gate takes the same
        # route: its structure outcome is real and recorded, and it is still not ancestry.
        archive.promote(
            endpoint.endpoint_id,
            feasibility_level=(FeasibilityLevel.DEFINITIVE if passed
                               else FeasibilityLevel.UNVALIDATED),
            structure_outcome=outcome, depth=depth,
        )
        if not passed:
            promoted.append(endpoint)
            continue
        promoted.append(
            CompleteEndpoint(**{
                **{field: getattr(endpoint, field) for field in (
                    "schema_version", "lineage", "protein_id", "sequence", "sequence_md5",
                    "sequence_length", "source_state_id", "source_state_content_digest",
                    "fork_index", "fork_seed", "replay",
                    "endpoint_provenance_evidence_by_pos", "head_binding", "head_score",
                    "head_global_risk", "cost_event_ids")},
                "feasibility_level": FeasibilityLevel.DEFINITIVE,
                "structure_outcome": outcome,
            })
        )
    return tuple(promoted), tuple(admissions)


def run_one_cycle(
    *,
    sampler: Any,
    denoiser: Callable[[torch.Tensor, float, Any], torch.Tensor],
    config: Any,
    sequence_length: int,
    h_values: np.ndarray | Sequence[float],
    residue_token_ids: frozenset[int] | Sequence[int],
    alphabet: Mapping[int, str],
    fixed_tokens: Mapping[int, int] | None,
    lineage: Any,
    mask_token_id: int,
    aa_token_ids: frozenset[int] | Sequence[int],
    conditioning: Any,
    safety_reference: Any,
    c_source_step: int,
    r_step: int,
    c_next_step: int,
    source_fork_seeds: Sequence[int],
    descendant_fork_seeds: Sequence[int],
    descendant_propagation_seed: int,
    head_oracle: Any,
    structure_oracle: Callable[[str], Any],
    support_policy: Callable[..., Any],
    band_table: Any,
    stratum_key: str,
    declared_band_id: str,
    declared_band_digest: str,
    origin_transition_id: str,
    safety_gate: Any,
    #: The run's ``DeclaredPolicy`` (``V2Config.declared_policy``).  REQUIRED: the kernel matches
    #: the ANSWERING policy against it, which is the runtime half of PLAN V2F1's "diagnostic policy
    #: in a production phase fails before model calls".
    declared_policy: Any,
    #: REQUIRED, and deliberately not defaulted.  This is the run's arm identity: ``config.arm.
    #: feedback_enabled`` must reach the engine, not merely the manifest.  A default of ``True``
    #: meant an A2 config ran WITH feedback while its own artifact recorded ``feedback_enabled:
    #: false`` -- a treatment arm labelled as its own control, and nothing in the outputs could
    #: distinguish the two.  Same argument as ``safety_gate``: an optional gate defaults to off.
    feedback_enabled: bool,
    #: REQUIRED.  PLAN §5.4: "Oracle requests are journaled before execution and outcomes
    #: afterward."  An optional meter defaults to no journal, and the sentinel the whole rule
    #: exists for -- ``unknown_after_start`` -- can then never be produced by a real run.
    cost_meter: Any,
    coordinate_law: Any = None,
    struct: Any = None,
    #: The Dual runtime, or ``None``.  A run with no overlay passes ``None`` and every Dual line in
    #: this module is unreachable, which is what makes Dual-off equivalence structural rather than
    #: a tested coincidence (PLAN §3.1).
    dual: Any = None,
    source_override: Any = None,
    source_endpoints: Any = None,
    archive: Any = None,
    endpoint_rank: int = 0,
    source_intervention: tuple[str, Any] | None = None,
    required_support_budget: Any = None,
    #: A policy answer to USE INSTEAD OF asking the policy.  PLAN §2.6's second intervention is
    #: "hold endpoint y and support FIXED, change or ablate compatible source state P" -- and
    #: "fixed" cannot be achieved by re-deriving the support from the perturbed source and then
    #: checking its cardinality, because masking a resolved position necessarily moves it between
    #: support classes.  The kernel still validates the pinned partition against the perturbed
    #: source, so an override cannot smuggle in an illegal projection; what it removes is the
    #: policy's freedom to answer differently, which is exactly what the intervention holds fixed.
    support_override: Any = None,
) -> CycleOutcome:
    """Run one complete feedback cycle and return everything it produced.

    The last four arguments exist for the paired executor and default to the ordinary V2 behaviour:

    ``feedback_enabled=False``
        Stop after the A2 view.  This is what A2 means -- the same run with the feedback stage off,
        not a second run.
    ``endpoint_rank``
        Which of the ranked admissible endpoints to hand the policy; ``1`` is the endpoint-change
        view's B arm.
    ``source_intervention``
        ``("ablate", positions)`` or ``("shuffle", seed)``, applied to the captured source before
        anything reads it, so both arms still share the SAME prefix computation.
    ``required_support_budget``
        A :class:`SupportBudget` the policy's partition must match exactly.  This is where PLAN
        §2.5's "matched on ... support cardinality" stops being a hope: a policy that returns a
        different-shaped support yields a typed null naming the parity failure, rather than a
        contrast whose difference cannot be attributed.
    """
    source_seeds = [int(s) for s in source_fork_seeds]
    descendant_seeds = [int(s) for s in descendant_fork_seeds]
    overlap = set(source_seeds) & set(descendant_seeds)
    if overlap:
        raise V2CycleError(
            f"source and descendant fork seeds must be disjoint, shared: {sorted(overlap)}; "
            "reusing a source seed would regenerate an endpoint the archive already holds, so the "
            "descendant pool would measure nothing new"
        )

    coordinates = make_cycle(
        depth=int(lineage.depth), r_step=int(r_step), c_source_step=int(c_source_step),
        c_next_step=int(c_next_step), n_steps=int(config.sampler.n_steps),
        # The ladder declares the law; defaulting here would make a stationary comparator run
        # under progressive coordinates and fail for a reason unrelated to its schedule.
        law=coordinate_law or CoordinateLaw.PROGRESSIVE_CHECKPOINT,
    )

    if source_override is not None:
        # The ladder hands each rung the previous rung's propagated capture, so a deeper source is
        # a REAL descendant rather than a fresh root: re-capturing would break the chain and
        # reintroduce root-sampling variance at every depth.
        source = source_override
        source_prefix_dfe = 0
    else:
        # Charged HERE because this is where it was spent.  When the ladder captures depth 0 itself
        # it charges its own root capture, so the prefix stays on exactly one record.
        source_prefix_dfe = int(c_source_step)
        with cost_meter.attempt(
            event_id=f"{origin_transition_id}:root_capture", phase="root_capture",
            request_kind="prefix_capture", request_digest=canonical_digest(
                {"length": int(sequence_length), "at_step": int(c_source_step)}),
            logical_dfe=source_prefix_dfe,
        ) as receipt:
            source = capture_depth_zero(
                sampler=sampler, denoiser=denoiser, config=config,
                sequence_length=int(sequence_length), h_values=h_values,
                residue_token_ids=residue_token_ids, at_step=int(c_source_step),
                fixed_tokens=fixed_tokens, lineage=lineage, mask_token_id=int(mask_token_id),
                aa_token_ids=aa_token_ids, conditioning=conditioning,
                safety_reference=safety_reference, cost_event_ids=("evt:root",), struct=struct,
            )
            receipt.observe(physical_forwards=source_prefix_dfe)

    if source_intervention is not None:
        from .paired import ablate_source, shuffle_source

        kind, payload = source_intervention
        if kind == "ablate":
            targets = tuple(payload) or tuple(
                position for position in source.editable_positions
                if source.tokens[position] != source.mask_token_id)[:1]
            source = ablate_source(source, positions=targets)
        elif kind == "shuffle":
            source = shuffle_source(source, seed=int(payload))
        else:
            raise V2CycleError(f"unknown source_intervention {kind!r}")

    # The ladder passes ONE archive across every depth: PLAN §2.3 makes it the run's memory, and a
    # per-depth archive would lose the elite the moment a generation regressed.
    archive = ExactArchive() if archive is None else archive
    evaluator = safety_reference.head_binding.evaluator
    # The declared window grid comes from the FROZEN safety reference, not from a probe completion.
    #
    # Reading it off a probe cost a full extra tail of forward passes that no ledger line accounted
    # for -- at the measured unit costs the tail is the dominant term, so the reported screening
    # cost understated every cycle by one whole lookahead.  It was also the weaker check: a probe
    # only proves the batch agrees with itself, while the reference grid is the grid the
    # whole-landscape comparator must align designs against, so a Head that silently regridded
    # would still have passed.
    window_grid_digest = safety_reference.head_binding.window_grid_digest

    if source_endpoints is not None:
        # A ladder rung inherits the PREVIOUS rung's descendant pool as its own source pool.
        # Re-screening the same state would charge the tail twice (PLAN §3.4 forbids it) and would
        # mint colliding endpoint ids, since an endpoint id binds (source state, fork index,
        # sequence) and the source state is unchanged.
        endpoints = tuple(source_endpoints)
        admissions = ()
        screening_dfe = 0
    else:
        with cost_meter.attempt(
            event_id=f"{origin_transition_id}:screen", phase="screen",
            request_kind="lookahead_pool", request_digest=canonical_digest(source_seeds),
            logical_dfe=len(source_seeds) * (int(config.sampler.n_steps) - int(c_source_step)),
        ) as receipt:
            completions = generate_lookaheads(
                source=source, sampler=sampler, denoiser=denoiser, config=config,
                h_values=h_values, residue_token_ids=residue_token_ids, fork_seeds=source_seeds,
                alphabet=alphabet, struct=struct,
            )
            receipt.observe(physical_forwards=sum(
                int(completion.logical_dfe) for completion in completions))
        endpoints, admissions = _score_and_promote(
            completions=completions, head_oracle=head_oracle,
            structure_oracle=structure_oracle, evaluator=evaluator,
            window_grid_digest=window_grid_digest, archive=archive,
            depth=int(lineage.depth), safety_gate=safety_gate,
            hard_anchors=source.hard_anchors,
            cost_meter=cost_meter, event_prefix=f"{origin_transition_id}:source",
            screen_event_id=f"{origin_transition_id}:screen", dual=dual,
        )
        # Summed over the completions that were REALLY produced, not over the seeds that were
        # requested: a fork that never ran is not a fork that was paid for.
        screening_dfe = sum(int(completion.logical_dfe) for completion in completions)

    # PLAN §2.3: the A2 view is the archive state IMMEDIATELY BEFORE feedback.
    a2_view = snapshot_a2_view(
        archive,
        protein_id=source.lineage.protein_id,
        source_state_id=source.state_id,
        depth=int(lineage.depth),
    )

    def _cost(*, segment_logical_dfe: int = 0,
              descendant_screening_logical_dfe: int = 0) -> CycleCost:
        """The bill as of THIS exit path.  Later stages add their own or they are not charged."""
        return CycleCost(
            source_prefix_logical_dfe=source_prefix_dfe,
            screening_logical_dfe=screening_dfe,
            segment_logical_dfe=int(segment_logical_dfe),
            descendant_screening_logical_dfe=int(descendant_screening_logical_dfe),
        )

    def _null(outcome: TransitionOutcome, detail: str) -> CycleOutcome:
        return CycleOutcome(outcome=outcome, source=source, endpoints=endpoints,
                            a2_view=a2_view, archive=archive, cost=_cost(), detail=detail,
                            admissions=admissions)

    if not feedback_enabled:
        # A2: the run stops at the view.  Everything before this point is shared with V2 by
        # construction, which is what makes the two arms a matched comparison.
        return CycleOutcome(
            outcome=TransitionOutcome.COMMITTED, source=source, endpoints=endpoints,
            a2_view=a2_view, archive=archive, cost=_cost(), admissions=admissions,
            detail="feedback disabled: this is the A2 view of the same run",
        )

    # PLAN DUALF3: the family representative, the archive elite, the incumbent and the donor gate
    # must agree on ONE objective.  Selecting under role A while gating under J would select on a
    # mixture of two laws, and no artifact would record that it had.
    admissible = select_family_representatives(
        endpoints, rank_key=None if dual is None else dual.rank_key())
    if not admissible:
        return _null(
            TransitionOutcome.NULL_NO_ADMISSIBLE_ENDPOINT,
            "no endpoint reached definitive feasibility, so none may become feedback ancestry",
        )
    rank = int(endpoint_rank)
    if rank >= len(admissible):
        return _null(
            TransitionOutcome.NULL_NO_ADMISSIBLE_ENDPOINT,
            f"endpoint_rank={rank} but only {len(admissible)} admissible endpoint(s) exist",
        )
    selected = admissible[rank]

    if source.realized_maturity.n_editable == 1:
        return CycleOutcome(
            outcome=TransitionOutcome.TERMINAL_BEST_LOOKAHEAD, source=source,
            endpoints=endpoints, a2_view=a2_view, archive=archive, cost=_cost(),
            selected_endpoint=selected, admissions=admissions,
            detail=("the source has one editable position; return the best definitive Head-ranked "
                    "lookahead directly because no recursive reopen action exists"),
        )

    if dual is not None:
        # The policy is immutable and is rebuilt one depth at a time by the ladder, which CLEARS
        # the donor score map on every advance -- the next depth's donors are different endpoints.
        # Rebinding here is what supplies this cycle's pool, so a donor role B has not scored
        # raises instead of silently resolving to a stale entry.
        rebind = getattr(support_policy, "with_dual_donor_scores", None)
        if rebind is None:
            raise V2CycleError(
                "a Dual cycle was given a support policy that cannot accept role B's donor "
                "scores; the joint gate would fall back on role A alone while the run's signature "
                "continued to claim a Dual arm"
            )
        support_policy = rebind(dual.donor_scores(admissible))

    if support_override is not None:
        decision = support_override
    elif getattr(support_policy, "consumes_runtime", False):
        # PLAN §5.4: a policy that calls the frozen Head while deciding (V2F5A's leave-one-out
        # counterfactual) must journal that batch like any other oracle request.  The meter is
        # created per shard, long after the oracle factory built the policy, so it cannot be a
        # constructor field -- and an unjournaled Head batch would spend real GPU work against a
        # cap the run's own accounting cannot see.  Only policies that ASK for the runtime get it;
        # every other policy keeps the three-argument call shape the kernel has always used.
        from ..fusion_v2.policy import PolicyRuntime

        decision = support_policy(
            source, selected, coordinates,
            runtime=PolicyRuntime(cost_meter=cost_meter,
                                  event_prefix=f"{origin_transition_id}:policy",
                                  source_depth=int(lineage.depth), selected_rank=rank,
                                  selected_endpoint_id=str(selected.endpoint_id)),
        )
    else:
        decision = support_policy(source, selected, coordinates)
    policy_identity = getattr(decision, "policy", None)
    # The policy's own per-position decision record (PLAN §5.3, V2F5A).  Read off the DECISION the
    # kernel acted on rather than off the policy object afterwards: the second is a different
    # source that can disagree with the answer that actually ran.
    policy_evidence = getattr(decision, "decision_evidence", None)
    if required_support_budget is not None and hasattr(decision, "reopen"):
        from .paired import SupportBudget, V2PairedError, assert_support_parity

        realized = SupportBudget(
            n_write_from_endpoint=len(decision.write_from_endpoint),
            n_inject_from_source_feedback=len(decision.inject_from_source_feedback),
            n_reopen=len(decision.reopen),
            n_carry_from_source=len(decision.carry_from_source),
        )
        try:
            assert_support_parity([required_support_budget, realized])
        except V2PairedError as exc:
            return CycleOutcome(
                outcome=TransitionOutcome.NULL_INVALID_POLICY_RESULT, source=source,
                endpoints=endpoints, a2_view=a2_view, archive=archive, cost=_cost(),
                selected_endpoint=selected, detail=str(exc), admissions=admissions,
                policy_identity=policy_identity, policy_evidence=policy_evidence,
            )
    endpoint_tokens = tuple(
        evidence.token for evidence in selected.endpoint_provenance_evidence_by_pos
    )
    projection = source_writeback(
        source=source, endpoint=selected, endpoint_tokens=endpoint_tokens, alphabet=alphabet,
        decision=decision, declared_policy=declared_policy,
        coordinates=coordinates, band_table=band_table, stratum_key=stratum_key,
        descendant_fork_seed=int(descendant_propagation_seed),
        origin_transition_id=origin_transition_id, declared_band_id=declared_band_id,
        declared_band_digest=declared_band_digest,
    )
    if not projection.committed:
        return CycleOutcome(
            outcome=projection.outcome, source=source, endpoints=endpoints, a2_view=a2_view,
            archive=archive, cost=_cost(), selected_endpoint=selected, detail=projection.detail,
            admissions=admissions, policy_identity=projection.policy,
            policy_evidence=policy_evidence, support_policy=support_policy,
        )

    projected = projection.projected
    segment_dfe_planned = int(c_next_step) - int(r_step)
    with cost_meter.attempt(
        event_id=f"{origin_transition_id}:segment", phase="segment",
        request_kind="propagation_segment", request_digest=projected.state_id,
        logical_dfe=segment_dfe_planned,
    ) as receipt:
        segment = run_projected_segment(
            projected=projected, sampler=sampler, denoiser=denoiser, config=config,
            h_values=h_values, residue_token_ids=residue_token_ids, struct=struct,
        )
        receipt.observe(physical_forwards=segment_dfe_planned)
    # The segment ran, so it is charged from here on -- including on the terminal-stop path below,
    # where the work was really spent and only the descendant pool was never reached.
    segment_dfe = segment_dfe_planned
    if segment_is_terminal(segment, projected):
        # The segment resolved its last mask.  That is an ordinary trajectory outcome, not a fault:
        # the result is a COMPLETE sequence, i.e. a terminal endpoint rather than a live partial
        # state, so there is no descendant to capture and nothing further to fork.  PLAN §4.5 makes
        # this a typed stop; letting the state layer's exception escape would report a normal
        # outcome as a crash and a cohort runner could not tell it from real corruption.  The paid
        # segment stays on the record so the ledger still charges what was actually spent.
        return CycleOutcome(
            outcome=TransitionOutcome.STALLED_NO_NOVEL_DESCENDANT, source=source,
            endpoints=endpoints, a2_view=a2_view, archive=archive,
            cost=_cost(segment_logical_dfe=segment_dfe),
            selected_endpoint=selected, projected=projected, segment=segment,
            admissions=admissions, policy_identity=policy_identity,
            policy_evidence=policy_evidence, support_policy=support_policy,
            detail=(
                f"the propagation segment resolved every editable position by step "
                f"{c_next_step}; the capture is a terminal endpoint, not a live partial state, so "
                "no descendant lineage continues from it"
            ),
        )

    propagated = adapt_segment_to_live_state(
        projected=projected, outcome=segment, conditioning=conditioning,
        safety_reference=safety_reference, cost_event_ids=source.cost_event_ids,
    )
    validate_propagated_capture(
        projected=projected, propagated=propagated,
        background_remask_events=segment.background_remask_events,
    )

    with cost_meter.attempt(
        event_id=f"{origin_transition_id}:descendant_screen", phase="descendant_screen",
        request_kind="lookahead_pool", request_digest=canonical_digest(descendant_seeds),
        logical_dfe=len(descendant_seeds) * (int(config.sampler.n_steps) - int(c_next_step)),
    ) as receipt:
        descendant_completions = generate_lookaheads(
            source=propagated, sampler=sampler, denoiser=denoiser, config=config,
            h_values=h_values, residue_token_ids=residue_token_ids,
            fork_seeds=descendant_seeds, alphabet=alphabet, struct=struct,
        )
        receipt.observe(physical_forwards=sum(
            int(completion.logical_dfe) for completion in descendant_completions))
    descendants, descendant_admissions = _score_and_promote(
        completions=descendant_completions, head_oracle=head_oracle,
        structure_oracle=structure_oracle, evaluator=evaluator,
        window_grid_digest=window_grid_digest, archive=archive,
        depth=int(propagated.lineage.depth), safety_gate=safety_gate,
        hard_anchors=propagated.hard_anchors,
        cost_meter=cost_meter, event_prefix=f"{origin_transition_id}:descendant",
        screen_event_id=f"{origin_transition_id}:descendant_screen",
        # Role B must score the DESCENDANT pool too.  Depth d's descendants are depth d+1's source
        # pool, so a descendant with no joint evidence would be un-orderable at the next rung --
        # the joint rank key raises rather than falling back on role A, which is correct, so a
        # half-wired Dual run would die one depth in instead of quietly steering single-allele.
        dual=dual,
    )

    return CycleOutcome(
        outcome=TransitionOutcome.COMMITTED, source=source, endpoints=endpoints,
        a2_view=a2_view, archive=archive,
        cost=_cost(segment_logical_dfe=segment_dfe,
                   descendant_screening_logical_dfe=sum(
                       int(completion.logical_dfe) for completion in descendant_completions)),
        selected_endpoint=selected,
        projected=projected, segment=segment, propagated=propagated,
        descendant_endpoints=descendants, admissions=admissions,
        descendant_admissions=descendant_admissions, policy_identity=policy_identity,
        policy_evidence=policy_evidence, support_policy=support_policy,
    )
