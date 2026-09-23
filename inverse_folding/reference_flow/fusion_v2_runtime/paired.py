"""V2F5-2: paired feedback-transmission evidence (PLAN task V2F5, §2.5, §2.6).

A committed cycle proves the wiring runs.  A PAIRED cycle asks the falsifiable question: does the
projected state actually depend on the things the mechanism claims it depends on?

PLAN V2F5 requires four paired views -- feedback-off, endpoint-change, source-change/ablation, and
source-shuffle -- plus a reward-ordered endpoint pair and a policy-level source-off comparator
matched on endpoint, support cardinality, re-entry horizon, fork seeds, and resources.

**The pairing law is the sharp edge.**  PLAN §2.6, enforced by
``fusion_v2.seeds.assert_matched_pair_seeds``: each ``(pair_id, fork_index)`` group is exactly TWO
rows, sharing one ``intervention_kind``, one ``shared_match_identity``, one ``c_source_step``, and
IDENTICAL realized seeds, while carrying DISTINCT ``treatment_identity`` in the two arm slots.

So a "pair" is one contrast, not a family of arms.  Four required views means four pairs.  The
seeds are identical across the two arms by construction: they are re-derived from the context alone
inside :class:`MatchedPairSeedRecord`, and nothing on the treatment side can reach them.  That is
what makes the design a matched comparison rather than two runs that happen to be labelled.

The distinctness of ``treatment_identity`` is not pedantry either -- a control that silently equals
its treatment reads as a null effect rather than as a broken experiment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..fusion_v2.errors import V2Error
from ..fusion_v2.identity import canonical_digest
from ..fusion_v2.seeds import FeedbackPairSeedContext, MatchedPairSeedRecord
from ..fusion_v2.state import LivePartialState, TransitionOutcome

__all__ = [
    "V2PairedError",
    "NotYetFrozenError",
    "MECHANISM_VIEWS",
    "POLICY_QUALIFICATION_VIEWS",
    "ALL_VIEWS",
    "SUPPORT_LAW_VIEW",
    "ENDPOINT_RELATIONS",
    "POLICY_VARIANTS",
    "PolicyQualificationResult",
    "assert_write_reopen_parity",
    "run_policy_qualification_view",
    "InterventionAxes",
    "SupportBudget",
    "assert_support_parity",
    "MechanismViewResult",
    "PairedArmRun",
    "run_mechanism_views",
    "INTERVENTIONS",
    "PairedArm",
    "PairedContrast",
    "SourceDependenceVerdict",
    "build_paired_contrasts",
    "ablate_source",
    "shuffle_source",
    "source_dependence_verdict",
]


class V2PairedError(V2Error):
    """A paired-evidence contract violation."""


#: AXIS 1 -- what the feedback machinery is doing.  These four are the falsifiers PLAN V2F5 names,
#: and they are the ones that execute today.
MECHANISM_VIEWS: tuple[str, ...] = (
    #: A2: the same source, no projection at all.
    "feedback_off",
    #: Same source and support shape, a DIFFERENT selected endpoint.
    "endpoint_change",
    #: Same endpoint and support shape, a perturbed (ablated) source.
    "source_change",
    #: Same endpoint and support shape, the source's editable bytes permuted.
    "source_shuffle",
)

#: V2F5A's contrast, deliberately NOT one of the four above.  The four hold the support LAW fixed
#: and perturb what it is given; this one holds source, donor, cardinalities, horizon, band and
#: seeds fixed and changes the LAW itself.  It is kept off ``MECHANISM_VIEWS`` so the mechanism
#: cohort's default set is unchanged and so nothing can run it through ``run_mechanism_views``,
#: whose B arm re-uses one policy object -- the qualification's B arm needs a different policy AND a
#: different declared identity.
SUPPORT_LAW_VIEW = "support_law"
POLICY_QUALIFICATION_VIEWS: tuple[str, ...] = (SUPPORT_LAW_VIEW,)

#: Everything ``InterventionAxes`` will accept as a mechanism view.
ALL_VIEWS: tuple[str, ...] = MECHANISM_VIEWS + POLICY_QUALIFICATION_VIEWS

#: AXIS 2 -- which endpoint the arm was handed.  A reward-ordered pair is a claim about SELECTION,
#: not about the mechanism; keeping it on its own axis stops it from reading as a sibling of
#: feedback-off.  Its gate waits for the production policy freeze (PLAN §2.5).
ENDPOINT_RELATIONS: tuple[str, ...] = ("as_selected", "reward_ordered")

#: AXIS 3 -- what the policy was allowed to see.  The source-off comparator is a policy-level
#: control (PLAN §2.5), also gated on the freeze.
POLICY_VARIANTS: tuple[str, ...] = ("as_configured", "source_off")

#: Levels whose real gate is not authorized yet.  Declared so the interface exists; refused so a
#: placeholder can never be mistaken for evidence.
_NOT_YET_FROZEN = {
    "endpoint_relation": {"reward_ordered"},
    "policy_variant": {"source_off"},
}

#: Backwards-compatible flat view, for callers that only need the mechanism axis.
INTERVENTIONS: tuple[str, ...] = MECHANISM_VIEWS


class NotYetFrozenError(V2PairedError):
    """An axis level whose scientific gate is not authorized yet.

    PLAN §2.5: no policy rule is authorized "until it is explicitly frozen in config/runbook
    provenance", and the ``FeedbackSupportPolicySpec`` freeze comes AFTER real one-cycle
    transmission.  Running a placeholder version of one of these would produce a number that looks
    like evidence and is not.
    """


@dataclass(frozen=True)
class InterventionAxes:
    """One intervention as a point in the three-axis space."""

    mechanism_view: str
    endpoint_relation: str = "as_selected"
    policy_variant: str = "as_configured"

    def __post_init__(self) -> None:
        for name, allowed in (
            ("mechanism_view", ALL_VIEWS),
            ("endpoint_relation", ENDPOINT_RELATIONS),
            ("policy_variant", POLICY_VARIANTS),
        ):
            value = getattr(self, name)
            if value not in allowed:
                raise V2PairedError(
                    f"{name}={value!r} is not a declared level; allowed: {list(allowed)}"
                )

    @property
    def kind(self) -> str:
        """The canonical ``intervention_kind`` written to the seed record."""
        return (f"mechanism={self.mechanism_view}"
                f"|endpoint={self.endpoint_relation}"
                f"|policy={self.policy_variant}")

    def assert_executable(self) -> None:
        for name, blocked in _NOT_YET_FROZEN.items():
            value = getattr(self, name)
            if value in blocked:
                raise NotYetFrozenError(
                    f"{name}={value!r} is declared but its gate is not frozen yet; PLAN §2.5 "
                    "authorizes it only after the FeedbackSupportPolicySpec freeze, and running a "
                    "placeholder would yield a number that looks like evidence"
                )


@dataclass(frozen=True)
class SupportBudget:
    """The ACTION VECTOR of a support partition: how many positions each action touched.

    Two arms are a contrast only if their budgets are equal.  Otherwise they differ in how much was
    changed as well as in what, and a difference in outcome cannot be attributed to the
    intervention.  PLAN §2.5 requires the source-off comparator to be "matched on ... support
    cardinality"; this is the type that makes that checkable.
    """

    n_write_from_endpoint: int
    n_inject_from_source_feedback: int
    n_reopen: int
    n_carry_from_source: int

    @classmethod
    def of(cls, partition: Any) -> "SupportBudget":
        return cls(
            n_write_from_endpoint=len(partition.write_from_endpoint),
            n_inject_from_source_feedback=len(partition.inject_from_source_feedback),
            n_reopen=len(partition.reopen),
            n_carry_from_source=len(partition.carry_from_source),
        )


def assert_support_parity(budgets: Sequence[SupportBudget]) -> None:
    """Refuse a set of arms whose supports differ in shape."""
    distinct = {b for b in budgets}
    if len(distinct) > 1:
        raise V2PairedError(
            "support parity violated: the arms used different action vectors "
            f"{sorted(map(str, distinct))}; a contrast whose arms changed different NUMBERS of "
            "positions cannot attribute its difference to the intervention"
        )


def assert_write_reopen_parity(budgets: Sequence[SupportBudget]) -> None:
    """The parity a SUPPORT-LAW contrast can actually hold: equal write and reopen counts.

    Full action-vector parity is the wrong contract here, and demanding it would kill every arm.
    The two laws reopen DIFFERENT positions by construction -- that is the treatment -- and a
    reopened position is drawn from the source-resolved set, which the temporal law then splits into
    ``inject_from_source_feedback`` (committed at or after ``r_d``) and ``carry_from_source``
    (committed before it).  Reopening a late-committed position instead of an early-committed one
    therefore moves exactly one position between inject and carry while their SUM is fixed at
    ``|resolved| - m_reopen``.

    So the quantities PLAN §8.4 names -- "realized write/reopen cardinalities" -- are matched here,
    the total support size is matched as a consequence, and the inject/carry split is allowed to
    differ because it is downstream of the identity choice being tested.
    """
    pairs = {(b.n_write_from_endpoint, b.n_reopen) for b in budgets}
    if len(pairs) > 1:
        raise V2PairedError(
            "write/reopen parity violated: the arms realized different (writes, reopens) "
            f"{sorted(pairs)}; a support-law contrast whose arms applied different DOSES cannot "
            "attribute its difference to the support identity"
        )
    totals = {
        b.n_write_from_endpoint + b.n_inject_from_source_feedback + b.n_reopen
        + b.n_carry_from_source
        for b in budgets
    }
    if len(totals) > 1:
        raise V2PairedError(
            f"the arms partitioned different editable domains {sorted(totals)}; they are not two "
            "views of one source state"
        )


_ARM_SLOTS = ("arm_a", "arm_b")


@dataclass(frozen=True)
class PairedArm:
    """One arm of one contrast, with its realized seeds already bound."""

    intervention_kind: str
    arm_slot: str
    treatment_identity: str
    shared_match_identity: str
    fork_index: int
    r_step: int
    c_next_step: int
    c_source_step: int
    realized_propagation_seed: int
    realized_lookahead_seeds: tuple[int, ...]
    seed_record: MatchedPairSeedRecord


@dataclass(frozen=True)
class PairedContrast:
    """Exactly two arms: the treatment and its matched control."""

    intervention_kind: str
    axes: "InterventionAxes"
    arm_a: PairedArm
    arm_b: PairedArm

    @property
    def arms(self) -> tuple[PairedArm, PairedArm]:
        return (self.arm_a, self.arm_b)


def build_paired_contrasts(
    *,
    context: FeedbackPairSeedContext,
    fork_index: int,
    c_source_step: int,
    interventions: Sequence[str] = INTERVENTIONS,
) -> tuple[PairedContrast, ...]:
    """Build one matched contrast per intervention.

    Every arm of every contrast draws the SAME seeds -- they come from ``context`` and
    ``fork_index`` only.  Treatment appears exclusively in ``treatment_identity``, which no
    derivation function reads.  This is why the design can be validated at ``--dry-run``: the whole
    seed table is computable before a model exists.
    """
    if not isinstance(context, FeedbackPairSeedContext):
        raise V2PairedError("context must be a FeedbackPairSeedContext")
    fork = int(fork_index)
    if fork < 0:
        raise V2PairedError(f"fork_index must be >= 0, got {fork_index}")
    resolved: list[InterventionAxes] = []
    for item in interventions:
        if isinstance(item, InterventionAxes):
            resolved.append(item)
        elif item in MECHANISM_VIEWS:
            resolved.append(InterventionAxes(mechanism_view=item))
        else:
            raise V2PairedError(
                f"unknown intervention {item!r}; pass an InterventionAxes or one of "
                f"{list(MECHANISM_VIEWS)}"
            )

    contrasts: list[PairedContrast] = []
    for axes in resolved:
        kind = axes.kind
        shared = canonical_digest({
            "match": "v2f5-paired",
            "intervention_kind": kind,
            "pair_id": context.pair_id,
            "fork_index": fork,
            "r_step": int(context.r_step),
            "c_next_step": int(context.c_next_step),
            "c_source_step": int(c_source_step),
        })
        arms = []
        for slot, role in zip(_ARM_SLOTS, ("treatment", "control")):
            record = MatchedPairSeedRecord(
                context=context,
                fork_index=fork,
                arm_slot=slot,
                intervention_kind=kind,
                # The ONLY place treatment lives.  No derivation reads it.
                treatment_identity=f"{kind}:{role}",
                shared_match_identity=shared,
                c_source_step=int(c_source_step),
                realized_propagation_seed=context.matched_descendant_seed(fork),
                realized_lookahead_seeds=tuple(
                    context.matched_descendant_lookahead_seed(fork, index)
                    for index in range(int(context.n_descendant_lookaheads))
                ),
            )
            arms.append(PairedArm(
                intervention_kind=kind, arm_slot=slot,
                treatment_identity=record.treatment_identity,
                shared_match_identity=shared, fork_index=fork,
                r_step=int(context.r_step), c_next_step=int(context.c_next_step),
                c_source_step=int(c_source_step),
                realized_propagation_seed=record.realized_propagation_seed,
                realized_lookahead_seeds=record.realized_lookahead_seeds,
                seed_record=record,
            ))
        contrasts.append(PairedContrast(intervention_kind=kind, axes=axes,
                                        arm_a=arms[0], arm_b=arms[1]))
    return tuple(contrasts)


# --------------------------------------------------------------------------------------------
# source interventions
# --------------------------------------------------------------------------------------------


def _rebuilt(source: LivePartialState, tokens: Sequence[int]) -> LivePartialState:
    """Rebuild a source with new bytes, leaving every other field alone.

    The per-position provenance must follow the bytes or the state layer refuses the record -- the
    active row and its last origin have to agree.
    """
    from ..fusion_v2.state import OriginEvidence, PositionProvenance

    provenance = []
    for position, token in enumerate(tokens):
        previous = source.provenance_by_pos[position]
        last = previous.last_origin
        provenance.append(PositionProvenance(
            first_origin=previous.first_origin,
            last_origin=OriginEvidence(
                origin_kind=last.origin_kind, origin_ref=last.origin_ref, commit=last.commit,
                token=int(token), evidence_logprob=last.evidence_logprob,
                transition_id=last.transition_id, evidence_digest=last.evidence_digest,
            ),
            n_origin_events=previous.n_origin_events,
        ))
    import dataclasses

    return dataclasses.replace(
        source, tokens=tuple(int(t) for t in tokens), provenance_by_pos=tuple(provenance),
        state_id=None,
    )


def ablate_source(source: LivePartialState, *, positions: Sequence[int]) -> LivePartialState:
    """Mask the named editable positions: the source-change / ablation view.

    Refuses to touch a hard anchor.  Removing a constraint would be a DIFFERENT EXPERIMENT wearing
    a control's label, not an ablation of the source signal.
    """
    anchors = {position for position, _ in source.hard_anchors}
    targets = {int(p) for p in positions}
    hit = sorted(targets & anchors)
    if hit:
        raise V2PairedError(
            f"positions {hit} are hard anchors; ablating a constraint changes the experiment "
            "rather than the source signal"
        )
    outside = sorted(targets - set(source.editable_positions))
    if outside:
        raise V2PairedError(f"positions {outside} are outside the editable domain")

    from ..fusion_v2.state import (
        ActiveOriginKind,
        ActiveScoreStatus,
        FeedbackOriginRef,
        OriginEvidence,
        PositionProvenance,
    )

    tokens = list(source.tokens)
    kinds = list(source.active_origin_kind_by_pos)
    refs = list(source.feedback_origin_ref_by_pos)
    statuses = list(source.active_score_status_by_pos)
    scores = list(source.active_sampler_score_by_pos)
    commits = list(source.active_commit_depth_step_by_pos)
    txns = list(source.origin_transition_id_by_pos)
    provenance = list(source.provenance_by_pos)

    for position in sorted(targets):
        tokens[position] = source.mask_token_id
        kinds[position] = ActiveOriginKind.UNRESOLVED
        refs[position] = FeedbackOriginRef.NONE
        statuses[position] = ActiveScoreStatus.MASKED
        scores[position] = None
        commits[position] = None
        txns[position] = None
        # The provenance must follow the bytes: a masked position whose last origin still claims
        # denoiser_sample is exactly the inconsistency the per-position algebra exists to refuse.
        previous = provenance[position]
        provenance[position] = PositionProvenance(
            first_origin=previous.first_origin,
            last_origin=OriginEvidence(
                origin_kind=ActiveOriginKind.UNRESOLVED, origin_ref=FeedbackOriginRef.NONE,
                commit=previous.last_origin.commit, token=int(source.mask_token_id),
                evidence_logprob=None, transition_id=None,
                evidence_digest=previous.last_origin.evidence_digest,
            ),
            n_origin_events=previous.n_origin_events,
        )

    import dataclasses

    return dataclasses.replace(
        source, tokens=tuple(int(t) for t in tokens),
        active_origin_kind_by_pos=tuple(kinds),
        feedback_origin_ref_by_pos=tuple(refs),
        active_score_status_by_pos=tuple(statuses),
        active_sampler_score_by_pos=tuple(scores),
        active_commit_depth_step_by_pos=tuple(commits),
        origin_transition_id_by_pos=tuple(txns),
        provenance_by_pos=tuple(provenance), state_id=None,
    )


def shuffle_source(source: LivePartialState, *, seed: int) -> LivePartialState:
    """Permute the source's RESOLVED editable bytes: the source-shuffle view.

    The multiset of residues is preserved, so the shuffle changes WHERE the source's information
    sits without changing WHAT it contains.  A projection that is invariant to this is not reading
    positional source evidence at all.

    Anchors are untouched, and masked positions stay masked -- permuting a mask would change the
    mask load and confound the contrast with a different maturity.
    """
    import random

    anchors = {position for position, _ in source.hard_anchors}
    movable = [
        position for position in source.editable_positions
        if position not in anchors and source.tokens[position] != source.mask_token_id
    ]
    values = [source.tokens[position] for position in movable]
    random.Random(int(seed)).shuffle(values)
    tokens = list(source.tokens)
    for position, value in zip(movable, values):
        tokens[position] = value
    return _rebuilt(source, tokens)


# --------------------------------------------------------------------------------------------
# the positive control
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceDependenceVerdict:
    """Did perturbing the subject actually change the projection?"""

    subject: str
    depends_on_source: bool
    differing_fields: tuple[str, ...]
    detail: str


#: The projected fields a real dependence must show up in.  Deliberately excludes bookkeeping like
#: ``state_id`` (a digest of everything) so the verdict names WHAT changed, not merely THAT it did.
_COMPARED_FIELDS = (
    "tokens",
    "support",
    "active_origin_kind_by_pos",
    "feedback_origin_ref_by_pos",
    "active_sampler_score_by_pos",
    "active_score_status_by_pos",
    "source_resolved_positions",
)


def source_dependence_verdict(
    *, baseline: Any, perturbed: Any, subject: str = "source",
) -> SourceDependenceVerdict:
    """The deterministic dependence check PLAN V2F5 requires.

    Two projections that are byte-identical after the subject was perturbed prove the kernel never
    read it.  PLAN's acceptance names the failing case explicitly -- "a deliberately source-blind
    kernel fails the deterministic source-dependence check" -- so this returns a VERDICT rather than
    raising: a failed control is a finding to report, not an exception to swallow.
    """
    differing = tuple(
        name for name in _COMPARED_FIELDS
        if getattr(baseline, name, None) != getattr(perturbed, name, None)
    )
    if differing:
        return SourceDependenceVerdict(
            subject=subject, depends_on_source=True, differing_fields=differing,
            detail=f"the projection changed in {list(differing)} when the {subject} was perturbed",
        )
    return SourceDependenceVerdict(
        subject=subject, depends_on_source=False, differing_fields=(),
        detail=(
            f"the two projections are byte-identical in every compared field, so the kernel is "
            f"blind to the {subject}: any apparent feedback it transmits is an artefact"
        ),
    )


# --------------------------------------------------------------------------------------------
# the actual paired executor
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedArmRun:
    """One arm, actually executed."""

    arm_slot: str
    treatment_identity: str
    realized_propagation_seed: int
    cycle: Any
    support_budget: SupportBudget | None


@dataclass(frozen=True)
class MechanismViewResult:
    """Both arms of one mechanism view, plus what the contrast supports.

    ``verdict`` is ``None`` whenever ``parity_violation`` is set: a malformed contrast must not also
    report a dependence claim, or a broken experiment gets laundered into evidence.
    """

    axes: InterventionAxes
    arm_a: PairedArmRun
    arm_b: PairedArmRun
    verdict: SourceDependenceVerdict | None
    parity_violation: str = ""


def _fixed_support_ablation(arm_a_cycle):
    """Arm A's realized support, plus a source position it is safe to ablate under it.

    Masking a position necessarily changes its resolvedness, and three of the four support classes
    are defined over resolvedness: an injection over a mask breaks the coupled mask-load identity
    ``u_proj = u_src - a + b_new``, a reopen of an already-unresolved position newly masks nothing,
    and an endpoint write over a newly-masked position shifts the admissible reopen envelope.  Only
    ``carry_from_source`` stays legal once its position is masked, because carrying an inherited
    mask is an ordinary carry.  So a FIXED-SUPPORT ablation exists only on a carried, resolved
    position -- and on some coordinates there is none.

    Returns ``(None, ())`` in that case.  PLAN §2.6 says "change OR ablate", and ``source_shuffle``
    is the change form: it rewrites source BYTES while preserving resolvedness, so it satisfies
    "support fixed" unconditionally and carries intervention 2's evidence wherever ablation cannot.
    The caller reports the absence as its own typed reason rather than as a parity violation, which
    would read as a broken experiment instead of an inapplicable one.
    """
    from ..fusion_v2.policy import PolicyDecision, SupportReason

    projected = getattr(arm_a_cycle, "projected", None)
    if projected is None or arm_a_cycle.policy_identity is None:
        return None, ()
    support = projected.support
    source = arm_a_cycle.source
    resolved = {p for p in source.editable_positions if source.tokens[p] != source.mask_token_id}
    targets = tuple(p for p in support.carry_from_source if p in resolved)[:1]
    if not targets:
        # Nothing carried is resolved, so no byte can be removed without moving a position out of
        # a class the pinned partition depends on.  Reported as "no intervention available"
        # rather than silently ablating something else.
        return None, ()
    return PolicyDecision(
        write_from_endpoint=support.write_from_endpoint,
        inject_from_source_feedback=support.inject_from_source_feedback,
        reopen=support.reopen,
        carry_from_source=support.carry_from_source,
        reason_by_pos={int(k): SupportReason(v) for k, v in dict(support.reason_by_pos).items()},
        policy=arm_a_cycle.policy_identity,
    ), targets


def run_mechanism_views(
    *,
    context: FeedbackPairSeedContext,
    fork_index: int,
    c_source_step: int,
    r_step: int,
    c_next_step: int,
    interventions: Sequence[Any] | None = None,
    **cycle_kwargs: Any,
) -> tuple[MechanismViewResult, ...]:
    """Run both arms of each mechanism view under identical seeds.

    The arms share every seed by construction (they come from ``context`` and ``fork_index`` only),
    share the source prefix, and differ ONLY in the intervention.  Perturbation is applied to arm B;
    arm A is the untouched reference.

    ``feedback_off`` is special: its B arm runs with feedback disabled, so it has no projection at
    all -- that is what A2 means.
    """
    from .cycle import run_one_cycle

    axes_list = tuple(
        InterventionAxes(mechanism_view=view) for view in MECHANISM_VIEWS
    ) if interventions is None else tuple(
        item if isinstance(item, InterventionAxes) else InterventionAxes(mechanism_view=item)
        for item in interventions
    )
    for axes in axes_list:
        axes.assert_executable()
        if axes.mechanism_view in POLICY_QUALIFICATION_VIEWS:
            # This runner's B arm re-uses ONE policy object and one declared identity; the
            # support-law contrast needs a second of each, so running it here would silently
            # produce two arms under the same law and report a null that means nothing.
            raise V2PairedError(
                f"mechanism_view={axes.mechanism_view!r} is a policy-qualification contrast and "
                "must be run through run_policy_qualification_view, which builds the matched "
                "control policy and its own declared identity"
            )

    contrasts = build_paired_contrasts(
        context=context, fork_index=fork_index, c_source_step=c_source_step,
        interventions=axes_list,
    )

    # ---- ONE realized pre-feedback pool, shared by every arm of every view -------------------
    #
    # PLAN §2.3/§4.3: A2 is "the archive state immediately before feedback" of the SAME run.  A
    # per-arm re-execution is not that.  Identical seeds make two executions produce equal values
    # under a deterministic fake oracle, which is why the difference is invisible in a unit test --
    # but under the conditions the experiment actually runs, a Head on a GPU is not bit-reproducible
    # and a structure oracle is a cache with state, so the arms would be comparing against
    # different realized pools while reporting a matched contrast.  It is also where the cost goes:
    # the source prefix and its K lookaheads dominate a cycle and were being paid once per arm per
    # view.
    #
    # The shared stage IS the A2 arm: running the cycle with feedback disabled is, by definition,
    # the view before feedback.  Everything else forks off it.
    shared_kwargs = dict(
        cycle_kwargs, c_source_step=c_source_step, r_step=r_step, c_next_step=c_next_step,
        descendant_propagation_seed=contrasts[0].arm_a.realized_propagation_seed,
        descendant_fork_seeds=contrasts[0].arm_a.realized_lookahead_seeds,
    )
    shared = run_one_cycle(**dict(shared_kwargs, feedback_enabled=False))

    # Each ARM gets its own transition id, derived from the caller's.
    #
    # Arm A and arm B produce different projected states, different propagated states and
    # different descendants: they are two transitions, not one transition attempted twice.  The
    # ledger keys its event ids on the transition id and counts a repeat under one event as a
    # RETRY -- whose whole purpose is to stop one arm from silently drawing more compute than its
    # match.  Sharing the id across arms blinded exactly that guard: a matched two-view run reads
    # as 13 retries of a 2-retry cap when nothing was retried at all (measured, job 12098305).
    # Deriving per arm restores the guard's meaning, and makes per-position provenance name the
    # arm that wrote it.
    from ..fusion_v2.identity import canonical_digest, make_transition_id

    lineage = cycle_kwargs["lineage"]
    caller_transition_id = cycle_kwargs["origin_transition_id"]

    def _arm_transition_id(*, intervention_kind: str, arm_slot: str) -> str:
        return make_transition_id(
            str(lineage.protein_id), str(lineage.family_id), depth=int(context.depth),
            r_step=int(r_step), c_next_step=int(c_next_step),
            content_digest=canonical_digest({
                "caller_transition_id": str(caller_transition_id),
                "intervention_kind": intervention_kind, "arm_slot": arm_slot}),
        )

    results: list[MechanismViewResult] = []
    for contrast in contrasts:
        view = contrast.axes.mechanism_view
        base_kwargs = dict(
            shared_kwargs,
            origin_transition_id=_arm_transition_id(
                intervention_kind=contrast.intervention_kind, arm_slot=contrast.arm_a.arm_slot),
            descendant_propagation_seed=contrast.arm_a.realized_propagation_seed,
            descendant_fork_seeds=contrast.arm_a.realized_lookahead_seeds,
            # Every arm READS the one realized pool: the same source state object and the same
            # scored endpoint objects.  Only the archive is forked, because each arm still writes
            # its own descendants and those must not appear in the other arm's history.
            source_override=shared.source,
            source_endpoints=shared.endpoints,
            # Stated rather than defaulted.  Every arm BUT ``feedback_off`` is a treatment arm by
            # construction here -- the causal probes perturb what feedback is given, not whether it
            # happens -- and ``feedback_off`` is served by the shared stage above, which was run
            # with the flag off explicitly.
            feedback_enabled=True,
        )

        arm_a_cycle = run_one_cycle(
            **dict(base_kwargs, archive=shared.archive.fork_view()))

        # The runner ENFORCES parity: arm A's action vector becomes arm B's contract, so a policy
        # that cannot match it yields a typed null naming the parity failure instead of a contrast
        # whose difference cannot be attributed (PLAN §2.5, "matched on ... support cardinality").
        # feedback_off is exempt: its B arm has no projection at all, by definition.
        required = (SupportBudget.of(arm_a_cycle.projected.support)
                    if arm_a_cycle.projected is not None and view != "feedback_off" else None)
        # From here ``base_kwargs`` is arm B's, so it takes arm B's transition id.  Arm A has
        # already run above and keeps the id it ran under.
        base_kwargs = dict(base_kwargs, origin_transition_id=_arm_transition_id(
            intervention_kind=contrast.intervention_kind, arm_slot=contrast.arm_b.arm_slot),
            # Arm A has already mutated its own fork.  Arm B must start from the same shared
            # pre-feedback archive, not from Arm A's post-feedback history.
            archive=shared.archive.fork_view())
        if required is not None:
            base_kwargs = dict(base_kwargs, required_support_budget=required)

        no_ablation_target = False
        if view == "feedback_off":
            # A2 is not executed again: it IS the shared stage.  Returning the object the pool came
            # from is what makes it a VIEW rather than a second run, and it is the only spelling a
            # non-deterministic oracle cannot quietly break.
            arm_b_cycle = shared
        elif view == "endpoint_change":
            arm_b_cycle = run_one_cycle(**dict(base_kwargs, endpoint_rank=1))
        elif view == "source_change":
            # PLAN §2.6 intervention 2: hold the endpoint AND THE SUPPORT fixed, ablate the source.
            #
            # Two things make this arm answerable rather than self-defeating.  First the support is
            # PINNED to arm A's realized partition instead of re-derived, because re-deriving it
            # from an ablated source changes the action vector by construction and the parity check
            # then kills the arm every time -- leaving `source_shuffle` green and the one control
            # that separates trajectory-coupled feedback from source-forgetting v0-style reopen
            # silently empty (PLAN §8.3 reads its result as a CLASSIFICATION, so it has to have
            # one).  Second the ablation target is drawn from `carry_from_source`, which is the one
            # support class that stays legal once its position is masked: an injection over a mask
            # breaks the coupled mask-load identity and a reopen of an already-unresolved position
            # newly masks nothing, so ablating into either class would make the pinned partition
            # illegal rather than merely different.
            pinned, targets = _fixed_support_ablation(arm_a_cycle)
            if pinned is None:
                # No carried, resolved position: this coordinate admits no fixed-support ablation.
                # The arm STILL RUNS -- it is a real second execution and its rows belong in the
                # artifact -- but the support is re-derived, so the contrast is not the fixed-support
                # intervention PLAN §2.6 asks for.  Recorded as its own reason below, because
                # "the intervention does not apply here" and "the arms disagreed" call for opposite
                # operator actions and the bare parity message could not tell them apart.
                no_ablation_target = True
                arm_b_cycle = run_one_cycle(
                    **dict(base_kwargs, source_intervention=("ablate", ())))
            else:
                arm_b_cycle = run_one_cycle(**dict(
                    base_kwargs, source_intervention=("ablate", targets),
                    support_override=pinned, required_support_budget=None))
        elif view == "source_shuffle":
            arm_b_cycle = run_one_cycle(
                **dict(base_kwargs,
                       source_intervention=("shuffle", contrast.arm_b.realized_propagation_seed)))
        else:                                                  # pragma: no cover - closed set
            raise V2PairedError(f"unhandled mechanism view {view!r}")

        budgets = tuple(
            SupportBudget.of(cycle.projected.support) if cycle.projected is not None else None
            for cycle in (arm_a_cycle, arm_b_cycle)
        )
        violation = ""
        if no_ablation_target:
            violation = (
                "support was NOT held fixed: arm A carried no resolved position, so this "
                "coordinate admits no fixed-support ablation -- every other support class becomes "
                "illegal once its position is masked.  The arms therefore differ in support as "
                "well as in source, and the contrast cannot be attributed to the source alone.  "
                "PLAN §2.6's 'change or ablate' is carried here by the source_shuffle view, which "
                "rewrites source bytes while preserving resolvedness."
            )
        present = [b for b in budgets if b is not None]
        # A named cause is never overwritten by the generic parity message: "this coordinate admits
        # no fixed-support ablation" and "the arms disagreed" call for opposite operator actions.
        if view != "feedback_off" and not violation:
            if len(present) == 2:
                try:
                    assert_support_parity(present)
                except V2PairedError as exc:                    # pragma: no cover - runner gates it
                    violation = str(exc)
            elif arm_b_cycle.outcome is not TransitionOutcome.COMMITTED:
                # The runner refused arm B, which is how a parity failure surfaces now.
                violation = arm_b_cycle.detail

        verdict = None
        # The ``not violation`` guard is a BACKSTOP, unreachable while the runner enforces parity:
        # a violated arm is nulled by the runner, so its ``projected`` is already None and the
        # second condition alone suffices.  Mutation testing confirms removing it changes nothing
        # observable today.  It is kept because the runner check is opt-in per call, and a caller
        # that omits ``required_support_budget`` must still never get a verdict on a malformed
        # contrast.
        if not violation and arm_a_cycle.projected is not None \
                and arm_b_cycle.projected is not None:
            subject = "endpoint" if view == "endpoint_change" else "source"
            verdict = source_dependence_verdict(
                baseline=arm_a_cycle.projected, perturbed=arm_b_cycle.projected, subject=subject,
            )
        elif not violation and view == "feedback_off":
            verdict = SourceDependenceVerdict(
                subject="feedback", depends_on_source=arm_b_cycle.projected is None,
                differing_fields=("projected",),
                detail="the feedback-off arm produced no projection, as A2 requires",
            )

        results.append(MechanismViewResult(
            axes=contrast.axes,
            arm_a=PairedArmRun(
                arm_slot=contrast.arm_a.arm_slot,
                treatment_identity=contrast.arm_a.treatment_identity,
                realized_propagation_seed=contrast.arm_a.realized_propagation_seed,
                cycle=arm_a_cycle, support_budget=budgets[0],
            ),
            arm_b=PairedArmRun(
                arm_slot=contrast.arm_b.arm_slot,
                treatment_identity=contrast.arm_b.treatment_identity,
                realized_propagation_seed=contrast.arm_b.realized_propagation_seed,
                cycle=arm_b_cycle, support_budget=budgets[1],
            ),
            verdict=verdict, parity_violation=violation,
        ))
    return tuple(results)


# --------------------------------------------------------------------------------------------
# V2F5A: the one-cycle policy-qualification contrast
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyQualificationResult:
    """One matched Head-directed-vs-source-geometry contrast (PLAN §2.6, §8.4).

    ``verdict`` here is NOT a directionality claim.  A committed pair proves the two support laws
    ran under matched source, donor, cardinalities, band and seeds; whether the descendant Head
    distribution shifts is a COHORT statistic over many such pairs, frozen in the runbook before the
    cohort is inspected.  This type carries the evidence that the pair is admissible into that
    statistic, and nothing more.
    """

    axes: InterventionAxes
    treatment: PairedArmRun
    control: PairedArmRun
    realized_writes: int
    realized_reopens: int
    #: The claim the artifact must be able to PROVE, not merely assert: every treatment write
    #: carried positive frozen-Head contribution evidence and the control consulted none.
    treatment_writes_are_head_directed: bool
    control_is_head_blind: bool
    parity_violation: str = ""

    @property
    def contrastable(self) -> bool:
        return (not self.parity_violation
                and self.treatment.cycle.projected is not None
                and self.control.cycle.projected is not None)


def _support_law_evidence(cycle: Any) -> tuple[bool, bool]:
    """``(writes_are_head_directed, head_evidence_consulted)`` read off one arm's decision record.

    Read from the evidence the POLICY emitted rather than from the policy object, because the
    object could be re-inspected after the fact and the record is what actually ran.  A missing
    record answers ``(False, False)``: an arm that recorded nothing has proved nothing.
    """
    evidence = getattr(cycle, "policy_evidence", None)
    if evidence is None:
        return False, False
    consulted = bool(getattr(evidence, "head_evidence_consulted", False))
    selected = [row for row in getattr(evidence, "write_candidates", ()) if row.selected]
    directed = bool(selected) and all(
        row.contribution is not None and row.contribution > 0.0 for row in selected)
    return directed, consulted


def run_policy_qualification_view(
    *,
    context: FeedbackPairSeedContext,
    fork_index: int,
    c_source_step: int,
    r_step: int,
    c_next_step: int,
    control_policy_factory: Any,
    control_declared_policy: Any,
    **cycle_kwargs: Any,
) -> PolicyQualificationResult:
    """Run the Head-directed treatment and its source-geometry control off ONE realized pool.

    The scientific treatment is support IDENTITY (PLAN §2.6), so everything else is held by
    construction rather than by convention:

    * **one pre-feedback stage** -- source prefix, K lookaheads, Head batch and structure gate run
      ONCE with feedback disabled, and both arms read the same realized endpoints.  Re-executing per
      arm would compare against two different pools under a non-deterministic Head and a stateful
      refold cache, while reporting a matched contrast;
    * **one donor** -- both arms receive the same ``endpoint_rank``, and the control re-runs the
      same donor gate, so neither arm can run on endpoints the other refused;
    * **one seed draw** -- both arms take ``context``-derived seeds that no treatment identity can
      reach (PLAN §2.6's exclusion law); and
    * **one dose** -- the control is built from the treatment's REALIZED ``(m_write, m_reopen)``.

    ``control_policy_factory(required_writes, required_reopens)`` builds the control; it is injected
    rather than constructed here so this module keeps no dependency on the oracle stack.
    ``control_declared_policy`` is the run's DECLARED control identity (``V2Config.
    declared_control_policy()``): the kernel refuses an answering policy the run never declared, and
    fabricating a declaration here would let an arm run under an identity the config cannot name.
    """
    from .cycle import run_one_cycle

    if control_policy_factory is None or not callable(control_policy_factory):
        raise V2PairedError(
            "control_policy_factory must be callable(required_writes, required_reopens); without "
            "the matched control there is no comparison, only a single-arm run"
        )
    if control_declared_policy is None:
        raise V2PairedError(
            "control_declared_policy is required: the projection kernel refuses a policy the run "
            "did not declare, and a declaration invented here would name an identity the config "
            "cannot"
        )

    # ``policy_variant`` stays ``as_configured``: it describes what a single ARM's policy was
    # allowed to see, and here the two arms differ on exactly that -- so the difference belongs to
    # the pair's mechanism view, which both arms share, and not to a level that would have to be
    # two values at once.
    axes = InterventionAxes(mechanism_view=SUPPORT_LAW_VIEW)
    contrasts = build_paired_contrasts(
        context=context, fork_index=fork_index, c_source_step=c_source_step,
        interventions=(axes,),
    )
    contrast = contrasts[0]

    shared_kwargs = dict(
        cycle_kwargs, c_source_step=c_source_step, r_step=r_step, c_next_step=c_next_step,
        descendant_propagation_seed=contrast.arm_a.realized_propagation_seed,
        descendant_fork_seeds=contrast.arm_a.realized_lookahead_seeds,
    )
    shared = run_one_cycle(**dict(shared_kwargs, feedback_enabled=False))

    from ..fusion_v2.identity import canonical_digest, make_transition_id

    lineage = cycle_kwargs["lineage"]
    caller_transition_id = cycle_kwargs["origin_transition_id"]

    def _arm_transition_id(arm_slot: str) -> str:
        return make_transition_id(
            str(lineage.protein_id), str(lineage.family_id), depth=int(context.depth),
            r_step=int(r_step), c_next_step=int(c_next_step),
            content_digest=canonical_digest({
                "caller_transition_id": str(caller_transition_id),
                "intervention_kind": contrast.intervention_kind, "arm_slot": arm_slot}),
        )

    base_kwargs = dict(
        shared_kwargs, feedback_enabled=True,
        source_override=shared.source, source_endpoints=shared.endpoints,
        origin_transition_id=_arm_transition_id(contrast.arm_a.arm_slot),
    )
    treatment_cycle = run_one_cycle(
        **dict(base_kwargs, archive=shared.archive.fork_view()))

    projected = treatment_cycle.projected
    if projected is None:
        # The treatment stalled (no better donor, no positive local write, band infeasible...).
        # The control is NOT run: with no realized cardinalities there is nothing to match, and a
        # control run at invented counts would be a second treatment.  The stall is the result.
        return PolicyQualificationResult(
            axes=axes,
            treatment=PairedArmRun(
                arm_slot=contrast.arm_a.arm_slot,
                treatment_identity=contrast.arm_a.treatment_identity,
                realized_propagation_seed=contrast.arm_a.realized_propagation_seed,
                cycle=treatment_cycle, support_budget=None),
            control=PairedArmRun(
                arm_slot=contrast.arm_b.arm_slot,
                treatment_identity=contrast.arm_b.treatment_identity,
                realized_propagation_seed=contrast.arm_b.realized_propagation_seed,
                cycle=shared, support_budget=None),
            realized_writes=0, realized_reopens=0,
            treatment_writes_are_head_directed=False, control_is_head_blind=False,
            parity_violation=(
                "the treatment arm produced no projection, so there are no realized cardinalities "
                f"to match a control against: {treatment_cycle.detail}"
            ),
        )

    realized = SupportBudget.of(projected.support)
    control_policy = control_policy_factory(
        required_writes=realized.n_write_from_endpoint, required_reopens=realized.n_reopen)
    control_cycle = run_one_cycle(**dict(
        base_kwargs,
        origin_transition_id=_arm_transition_id(contrast.arm_b.arm_slot),
        # The treatment has already admitted its descendants.  The matched control starts from an
        # independent view of the shared pre-feedback archive so treatment history cannot leak.
        archive=shared.archive.fork_view(),
        support_policy=control_policy,
        declared_policy=control_declared_policy,
    ))

    budgets = tuple(
        SupportBudget.of(cycle.projected.support) if cycle.projected is not None else None
        for cycle in (treatment_cycle, control_cycle)
    )
    violation = ""
    present = [budget for budget in budgets if budget is not None]
    if len(present) == 2:
        try:
            assert_write_reopen_parity(present)
        except V2PairedError as exc:
            violation = str(exc)
    else:
        violation = (
            "the control arm produced no projection under the treatment's realized cardinalities: "
            f"{control_cycle.detail}"
        )

    treatment_directed, treatment_consulted = _support_law_evidence(treatment_cycle)
    _control_directed, control_consulted = _support_law_evidence(control_cycle)
    del _control_directed, treatment_consulted
    return PolicyQualificationResult(
        axes=axes,
        treatment=PairedArmRun(
            arm_slot=contrast.arm_a.arm_slot,
            treatment_identity=contrast.arm_a.treatment_identity,
            realized_propagation_seed=contrast.arm_a.realized_propagation_seed,
            cycle=treatment_cycle, support_budget=budgets[0]),
        control=PairedArmRun(
            arm_slot=contrast.arm_b.arm_slot,
            treatment_identity=contrast.arm_b.treatment_identity,
            realized_propagation_seed=contrast.arm_b.realized_propagation_seed,
            cycle=control_cycle, support_budget=budgets[1]),
        realized_writes=realized.n_write_from_endpoint,
        realized_reopens=realized.n_reopen,
        treatment_writes_are_head_directed=treatment_directed,
        control_is_head_blind=not control_consulted,
        parity_violation=violation,
    )
