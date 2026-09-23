"""V2F2: the pure source-coupled projection kernel ``q_phi`` (PLAN §2.4).

The defining transition is

    P~_{d+1} ~ q_phi( . | y*_d, P*_d, S, C, pi_d, r_d )

implemented here as ``source_writeback``: an auditable, side-effect-free state transition rather
than hidden sampler code.  It takes a live source state at ``c_d``, a selected complete endpoint,
and a typed policy result, and returns a :class:`ProjectionOutcome` that is either a committed
:class:`~fusion_v2.state.ProjectedPartialState` at ``r_d`` or a typed null event.

**The kernel is frozen; the policy is not.**  Every invariant that protects the science lives
here, so a wrong policy produces a null event and never an invented state.  The kernel enforces:

* the four-way editable partition is disjoint and exhaustive;
* hard anchors are a separate permanent class, untouched and unreopenable;
* the temporal-history gate ``0 <= s_i^active < r_d`` for ordinary carry;
* the dual evidence namespaces -- terminal endpoint / later-source evidence stays immutable
  provenance and never becomes an active sampler score;
* injections begin ``PENDING_ASSIMILATION`` with no fabricated score and an active commit pinned
  at ``(depth+1, r_d)``;
* reopened positions are canonically masked;
* temporary protection is a separate typed relation with exclusive expiry at ``c_{d+1}``;
* the schedule-band and coupled mask-load gates; and
* exact archive non-mutation -- the endpoint record is read, never written.

Two structural boundaries are asserted by the test suite rather than merely documented:

* this module does **not** import :mod:`fusion_v2.seeds`.  ``seeds`` has in-degree zero inside the
  package, which is the structural form of PLAN §2.6's exclusion law; the descendant fork seed
  arrives as a plain ``int`` from the runner, so no treatment-dependent digest can reach it.
* :mod:`fusion_v2.policy` does not import :mod:`fusion_v2.config`, so a policy cannot read an
  unfrozen threshold out of the run config (PLAN §2.5).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .errors import V2Error
from .identity import LineageRef, canonical_digest
from .policy import (
    DeclaredPolicy,
    PolicyDecision,
    PolicyRejection,
    PolicyResult,
    SupportReason,
    V2PolicyError,
)
from .schedule import (
    BandVerdict,
    CycleCoordinates,
    EmptyBandIntersectionError,
    MissingScheduleBandError,
    ScheduleBandTable,
    admissible_reopen_cardinality,
    gate_projected_maturity,
    history_key,
    lookup_band,
    observe_maturity,
    validate_realized_mask_load,
)
from . import state as st

__all__ = [
    "V2ProjectionError", "ProjectionContext", "ProjectionOutcome", "source_writeback",
]


class V2ProjectionError(V2Error):
    """A caller contract violation in the projection kernel.

    Reserved for inputs the *runner* got wrong (a mistyped state, endpoint tokens that disagree
    with the archived endpoint).  A wrong *policy* answer is never an exception -- it is a null
    :class:`ProjectionOutcome`.
    """


@dataclass(frozen=True)
class ProjectionContext:
    """The bound ``(P, y)`` pair one ``q_phi`` call is defined over.

    ``q_phi(. | y*_d, P*_d, ...)`` is only meaningful when ``y`` is an exact endpoint of a lineage
    COMPATIBLE with ``P`` and eligible to become ancestry.  Nothing about a token vector carries
    that: an endpoint from another protein, or one whose structure was never evaluated, type-checks
    and writes back cleanly.  The result would be a state whose lineage names an endpoint that did
    not produce its residues -- which is precisely the object V2's central claim is about.

    Binding is a TYPE rather than a list of checks at the call site because a check the caller can
    skip is not a contract.  ``source_writeback`` cannot proceed without constructing one.

    **What is deliberately NOT required: ``endpoint.source_state_id == source.state_id``.**  The
    causal probes in :mod:`fusion_v2_runtime.paired` exist to ask whether the kernel is blind to the
    source; they perturb ``P`` and hold ``y`` fixed, so demanding state-identity would make the one
    experiment that can falsify feedback transmission inexpressible.  Compatibility is therefore
    bound at lineage granularity -- protein, root, family, length, anchors, Head evaluator -- which
    is what makes the two objects describe the same design problem.  Conditioning is bound
    transitively through ``root_id``: both objects name the same root, and the root's conditioning
    identity is carried by value on the source state.
    """

    source: st.LivePartialState
    endpoint: st.CompleteEndpoint
    endpoint_tokens: tuple[int, ...]
    alphabet: Mapping[int, str]

    def __post_init__(self) -> None:
        if not isinstance(self.source, st.LivePartialState):
            raise V2ProjectionError("source must be a LivePartialState")
        if not isinstance(self.endpoint, st.CompleteEndpoint):
            raise V2ProjectionError("endpoint must be a CompleteEndpoint")
        source, endpoint = self.source, self.endpoint
        object.__setattr__(self, "endpoint_tokens",
                           tuple(int(token) for token in self.endpoint_tokens))
        tokens = self.endpoint_tokens

        # -- the two objects describe the same design problem ----------------------------------
        if endpoint.protein_id != source.lineage.protein_id:
            raise V2ProjectionError(
                f"endpoint protein {endpoint.protein_id!r} is not the source's protein "
                f"{source.lineage.protein_id!r}; q_phi(P, y) is undefined across proteins"
            )
        for field_name in ("root_id", "family_id"):
            mine = getattr(endpoint.lineage, field_name)
            theirs = getattr(source.lineage, field_name)
            if mine != theirs:
                raise V2ProjectionError(
                    f"endpoint {field_name} {mine!r} does not match the source's {theirs!r}; the "
                    "endpoint belongs to a different family and may not become this lineage's "
                    "ancestry"
                )
        if endpoint.sequence_length != len(source.tokens):
            raise V2ProjectionError(
                f"endpoint carries {endpoint.sequence_length} positions, source has "
                f"{len(source.tokens)}")
        if len(tokens) != len(source.tokens):
            raise V2ProjectionError(
                f"endpoint_tokens has length {len(tokens)}, source has {len(source.tokens)}")

        # -- the Head that ranked the endpoint is this lineage's Head ---------------------------
        reference_evaluator = source.safety_reference.head_binding.evaluator
        if endpoint.head_binding.evaluator != reference_evaluator:
            raise V2ProjectionError(
                "the endpoint was scored by a different Head evaluator than the one this lineage's "
                "safety reference is bound to; the ranking that selected it was never a comparison "
                "on one scale"
            )

        # -- only a definitively feasible endpoint may become ancestry (PLAN §4.2) --------------
        if not st.endpoint_may_become_ancestry(endpoint):
            raise V2ProjectionError(
                f"endpoint {endpoint.endpoint_id} is {endpoint.feasibility_level.value!r}, not "
                "definitive; the kernel is where ancestry is created, so an endpoint whose "
                "structure was never evaluated may not be projected back into the lineage"
            )

        # -- the endpoint agrees with the archive, the anchors, and the scored sequence ---------
        for index, evidence in enumerate(endpoint.endpoint_provenance_evidence_by_pos):
            if evidence.token != tokens[index]:
                raise V2ProjectionError(
                    f"endpoint_tokens[{index}]={tokens[index]} disagrees with the archived "
                    f"endpoint's own evidence token {evidence.token}; the archive is exact and the "
                    "kernel may not write a byte it does not contain"
                )
        for position, token in source.hard_anchors:
            if tokens[position] != token:
                raise V2ProjectionError(
                    f"endpoint carries token {tokens[position]} at hard anchor position {position} "
                    f"where the constraint set pins {token}; an endpoint that breaks an anchor is "
                    "not this design problem's endpoint, whatever its lineage fields claim"
                )
        try:
            decoded = "".join(self.alphabet[token] for token in tokens)
        except (KeyError, TypeError) as exc:
            raise V2ProjectionError(
                f"endpoint token {exc} has no residue in the declared alphabet; the kernel refuses "
                "to guess an ordering it was not given"
            ) from exc
        if decoded != endpoint.sequence:
            raise V2ProjectionError(
                f"endpoint_tokens decode to {decoded!r} under the declared alphabet but the Head "
                f"scored {endpoint.sequence!r}; writing these tokens back would put residues into "
                "the lineage that no Head ever evaluated, while the provenance named an endpoint "
                "that was scored"
            )


@dataclass(frozen=True)
class ProjectionOutcome:
    """The result of one ``q_phi`` call: a committed state or a typed null, plus telemetry.

    ``FeedbackTransition`` forbids a support partition on a non-committed outcome, so the
    per-position reasons that explain a rejection have nowhere to live on the persisted event row.
    They live here instead, which keeps a null event auditable without reopening a V2F1 type.
    """

    outcome: st.TransitionOutcome
    projected: st.ProjectedPartialState | None
    detail: str
    reason_by_pos: Mapping[int, str]
    band_verdicts: tuple[BandVerdict, ...]
    policy: object | None
    #: The bound pair this call was defined over.  Carried so an audit can read WHAT was bound, not
    #: merely trust that something was.
    context: ProjectionContext | None = None

    @property
    def committed(self) -> bool:
        return self.outcome is st.TransitionOutcome.COMMITTED

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, st.TransitionOutcome):
            raise V2ProjectionError("outcome must be a TransitionOutcome")
        if (self.projected is None) is (self.outcome is st.TransitionOutcome.COMMITTED):
            raise V2ProjectionError(
                "a COMMITTED outcome must carry a projected state and a null outcome must not; "
                "an invented state on a null event is exactly what PLAN §2.4 forbids"
            )
        object.__setattr__(self, "reason_by_pos",
                           {int(k): str(v) for k, v in sorted(dict(self.reason_by_pos).items())})
        object.__setattr__(self, "band_verdicts", tuple(self.band_verdicts))


def _null(outcome, detail, reasons, verdicts, policy, context=None) -> ProjectionOutcome:
    return ProjectionOutcome(outcome=outcome, projected=None, detail=detail,
                             reason_by_pos=reasons, band_verdicts=tuple(verdicts), policy=policy,
                             context=context)


def _reason_strings(reason_by_pos: Mapping[int, SupportReason]) -> dict[int, str]:
    return {int(k): (v.value if isinstance(v, SupportReason) else str(v))
            for k, v in sorted(dict(reason_by_pos).items())}


def source_writeback(
    *,
    source: st.LivePartialState,
    endpoint: st.CompleteEndpoint,
    endpoint_tokens: Sequence[int],
    alphabet: Mapping[int, str],
    decision: PolicyResult,
    declared_policy: DeclaredPolicy,
    coordinates: CycleCoordinates,
    band_table: ScheduleBandTable,
    stratum_key: str,
    descendant_fork_seed: int,
    origin_transition_id: str,
    declared_band_id: str,
    declared_band_digest: str,
) -> ProjectionOutcome:
    """Project ``source`` back to ``coordinates.r_step`` under a policy's support sets.

    ``endpoint_tokens`` is the caller-decoded integer form of ``endpoint.sequence``, and
    ``alphabet`` is the map it was decoded with.  The kernel refuses to GUESS an ordering --
    ``fusion_v2`` holds no character-to-token map -- but it does verify the one it is handed, so
    the residues written back are provably the residues the Head scored.

    ``stratum_key`` is threaded in from the cohort artifact.  It is deliberately NOT derived from
    ``DepthSchedulePoint.band_key`` -- the Interface Map states they are different keys.
    """
    # Constructing the context IS the compatibility check: every binding q_phi is defined over is
    # enforced in its __post_init__, so there is no path into the kernel that skips one.
    context = ProjectionContext(
        source=source, endpoint=endpoint, endpoint_tokens=endpoint_tokens, alphabet=alphabet,
    )
    endpoint_tokens = context.endpoint_tokens

    if not isinstance(coordinates, CycleCoordinates):
        raise V2ProjectionError("coordinates must be a CycleCoordinates")
    if isinstance(descendant_fork_seed, bool) or not isinstance(descendant_fork_seed, int):
        raise V2ProjectionError("descendant_fork_seed must be an int supplied by the runner")

    if coordinates.n_steps != source.n_steps:
        # ``S`` is the trajectory's horizon: every maturity coordinate, every lookahead tail and
        # the whole PLAN §3.4 cost model are defined relative to it.  Binding only the checkpoint
        # let a projection re-base ``S`` while naming this source, i.e. describe a different
        # trajectory under this one's provenance.
        raise V2ProjectionError(
            f"coordinates.n_steps={coordinates.n_steps} does not match the source state's "
            f"n_steps={source.n_steps}; the projection would re-base the sampler horizon"
        )
    if coordinates.c_source_step != source.sampler_step:
        raise V2ProjectionError(
            f"coordinates.c_source_step={coordinates.c_source_step} does not match the source "
            f"state's sampler_step={source.sampler_step}")

    policy_identity = getattr(decision, "policy", None)

    # ---- the run's FROZEN config is the runtime authority ------------------------------------
    # PLAN §5.2 freezes the projection policy spec and the schedule-band calibration as content
    # identities, and ``V2ConditioningIdentity`` carries both BY VALUE on every state.  Nothing
    # downstream compared them against what actually answered, so a run could commit transitions
    # produced by a policy -- or gated by a band table -- that its own provenance row does not
    # describe, while the manifest reported the declared ones.  ``gate_projected_maturity`` is not
    # this check: it compares the declared id/digest against the table it was HANDED, so a
    # self-consistent foreign calibration satisfies it completely.
    #
    # Checked BEFORE the rejection branch, because a foreign policy's REFUSAL is not this run's
    # refusal either: recording it as a typed null would put "the policy declined" in the
    # feedback-event table under a policy identity that was never asked.
    #
    # Raised rather than returned as a null: this is the runner wiring the wrong object, which is
    # exactly the boundary ``V2ProjectionError`` is documented to own.
    # ---- the ANSWERING policy is the one this run declared -------------------------------------
    # The parse-time half of PLAN V2F1's "diagnostic policy in a production phase fails before
    # model calls" lives in ``DeclaredPolicy``: such a declaration is unconstructible.  This is the
    # other half.  The identity of the policy that ANSWERS does not exist until it answers -- after
    # the prefix and the K lookaheads have run -- so the kernel cannot re-derive the phase gate; it
    # only has to establish that the answer IS the declared object.  Composed, a run cannot obtain
    # a diagnostic transition in a production phase.
    #
    # Required, not optional, for the same reason ``declared_band_id`` is: a gate that can be
    # opted out of by omission is not a gate.
    if not isinstance(declared_policy, DeclaredPolicy):
        raise V2ProjectionError(
            "declared_policy must be the run's DeclaredPolicy; without it the kernel cannot say "
            "whether the policy that answered is the policy the run froze"
        )
    try:
        declared_policy.assert_answered_by(policy_identity)
    except V2PolicyError as exc:
        raise V2ProjectionError(str(exc)) from exc

    frozen = source.conditioning
    declared_policy_spec = getattr(policy_identity, "policy_spec_digest", None)
    if declared_policy_spec is None:
        # ``PolicyDecision.policy`` is optional, so without this an anonymous answer would be the
        # one policy exempt from having to be the run's declared policy -- and PLAN §5.3 would end
        # up with a committed feedback event whose policy columns are null.  A gate that can be
        # opted out of by omission is not a gate.
        raise V2ProjectionError(
            "the policy answer carries no identity.  This covers a REFUSAL as well as a decision: "
            "PLAN §4.5 makes a refusal a result and PLAN §5.3 requires every feedback event -- "
            "committed or null -- to name the policy that produced it, so an unattributable "
            "'the policy declined' row is a claim about a policy nobody can identify"
        )
    if declared_policy_spec != frozen.projection_policy_spec:
        raise V2ProjectionError(
            f"the answering policy declares spec {declared_policy_spec} but this run's frozen "
            f"conditioning names {frozen.projection_policy_spec} as its projection policy; the "
            "transition would be produced by a policy the run's own provenance does not describe"
        )
    if declared_band_digest != frozen.schedule_band_calibration:
        raise V2ProjectionError(
            f"the supplied schedule-band calibration {declared_band_digest} is not this run's "
            f"frozen {frozen.schedule_band_calibration}; B(r_d) would be evaluated against bands "
            "measured on a different cohort, and the verdict would still read as accepted"
        )

    # ---- a declining policy is a value, not an exception -----------------------------------
    if isinstance(decision, PolicyRejection):
        return _null(st.TransitionOutcome.NULL_INVALID_POLICY_RESULT, decision.reason,
                     _reason_strings(decision.reason_by_pos), (), policy_identity, context)
    if not isinstance(decision, PolicyDecision):
        raise V2ProjectionError("decision must be a PolicyDecision or a PolicyRejection")

    reasons = _reason_strings(decision.reason_by_pos)
    reject = lambda detail: _null(  # noqa: E731 - a local alias keeps the checks readable
        st.TransitionOutcome.NULL_INVALID_POLICY_RESULT, detail, reasons, (), policy_identity,
        context)

    # ---- the four-way partition --------------------------------------------------------------
    editable = set(source.editable_positions)
    anchors = {position for position, _ in source.hard_anchors}
    claimed = decision.claimed_positions
    claimed_set = set(claimed)

    if len(claimed) != len(claimed_set):
        return reject("the four support sets overlap; they must be pairwise disjoint")
    stray = claimed_set - editable
    if stray:
        touched_anchor = sorted(stray & anchors)
        if touched_anchor:
            return reject(
                f"support names hard anchor position(s) {touched_anchor}; anchors are a permanent "
                "class identified from the constraint set and can never enter a support action")
        return reject(f"support names non-editable position(s) {sorted(stray)}")
    missing = editable - claimed_set
    if missing:
        return reject(f"support omits editable position(s) {sorted(missing)}; the partition must "
                      "be exhaustive over the editable domain")

    if not decision.write_from_endpoint:
        return reject("write_from_endpoint is empty; a non-null feedback event must write at "
                      "least one endpoint identity")
    source_resolved = {p for p in source.editable_positions if source.tokens[p] != source.mask_token_id}

    newly_masked = [p for p in decision.reopen if p in source_resolved]
    if len(newly_masked) != len(decision.reopen):
        stale = sorted(set(decision.reopen) - source_resolved)
        return reject(
            f"reopen names already-unresolved position(s) {stale}; relabelling an inherited mask "
            "as 'reopen' newly masks nothing and would satisfy the non-empty requirement without "
            "opening any identity")

    off_source = sorted(set(decision.inject_from_source_feedback) - source_resolved)
    if off_source:
        return reject(
            f"inject_from_source_feedback names source-unresolved position(s) {off_source}; an "
            "injection over a mask would resolve a position the mask-load identity "
            "u_proj = u_src - a + b_new does not count, silently breaking the band arithmetic")

    # ---- the temporal-history gate -----------------------------------------------------------
    for position in decision.carry_from_source:
        if not st.carry_is_temporally_legal(
            source.active_commit_depth_step_by_pos[position],
            token=source.tokens[position], r_step=coordinates.r_step,
            mask_token_id=source.mask_token_id,
        ):
            commit = source.active_commit_depth_step_by_pos[position]
            return reject(
                f"position {position} was committed at {commit} which is not strictly before the "
                f"re-entry boundary r_d={coordinates.r_step}; it is future information and must be "
                "assigned to inject_from_source_feedback or reopen")

    # ---- build the projected per-position vectors --------------------------------------------
    depth = source.lineage.depth + 1
    reentry_key = history_key(depth, coordinates.r_step)
    endpoint_digest = endpoint.content_digest
    source_digest = source.content_digest

    write_set = set(decision.write_from_endpoint)
    inject_set = set(decision.inject_from_source_feedback)
    reopen_set = set(decision.reopen)

    tokens = list(source.tokens)
    kinds = list(source.active_origin_kind_by_pos)
    refs = list(source.feedback_origin_ref_by_pos)
    commits = list(source.active_commit_depth_step_by_pos)
    txns = list(source.origin_transition_id_by_pos)
    scores = list(source.active_sampler_score_by_pos)
    statuses = list(source.active_score_status_by_pos)
    provenance = list(source.provenance_by_pos)
    protections: list[st.TemporaryProtection] = []

    def _inject(position, *, token, ref, evidence_logprob, evidence_digest, grant_reason):
        """Write an injected identity: pending, unscored, active commit pinned at (d+1, r_d)."""
        tokens[position] = token
        kinds[position] = st.ActiveOriginKind.FEEDBACK_INJECTION
        refs[position] = ref
        commits[position] = reentry_key
        txns[position] = origin_transition_id
        # PLAN §2.4: injections "begin pending, with no fabricated active score".
        scores[position] = None
        statuses[position] = st.ActiveScoreStatus.PENDING_ASSIMILATION
        previous = source.provenance_by_pos[position]
        provenance[position] = st.PositionProvenance(
            # The original ancestry survives: a recursively carried endpoint-feedback token must
            # not lose it merely by becoming part of the next source state (PLAN §2.4).
            first_origin=previous.first_origin,
            last_origin=st.OriginEvidence(
                origin_kind=st.ActiveOriginKind.FEEDBACK_INJECTION, origin_ref=ref,
                commit=reentry_key, token=token,
                # Immutable evidence namespace: this is the terminal endpoint / later-source
                # score, retained as provenance only.  It is never an active sampler score.
                evidence_logprob=evidence_logprob,
                transition_id=origin_transition_id, evidence_digest=evidence_digest,
            ),
            n_origin_events=previous.n_origin_events + 1,
        )
        protections.append(st.TemporaryProtection(
            position=position, expiry_step=coordinates.c_next_step, granted_at_depth=depth,
            granted_at_step=coordinates.r_step, granted_by_transition_id=origin_transition_id,
            grant_reason=grant_reason,
        ))

    for position in sorted(write_set):
        _inject(position,
                token=endpoint_tokens[position],
                ref=st.FeedbackOriginRef.SELECTED_ENDPOINT,
                evidence_logprob=endpoint.endpoint_provenance_evidence_by_pos[
                    position].completion_logprob,
                evidence_digest=endpoint_digest,
                grant_reason=st.ProtectionGrantReason.ENDPOINT_INJECTION)

    for position in sorted(inject_set):
        _inject(position,
                token=source.tokens[position],
                ref=st.FeedbackOriginRef.SOURCE_STATE,
                evidence_logprob=source.active_sampler_score_by_pos[position],
                evidence_digest=source_digest,
                grant_reason=st.ProtectionGrantReason.SOURCE_FEEDBACK_INJECTION)

    for position in sorted(reopen_set):
        tokens[position] = source.mask_token_id
        kinds[position] = st.ActiveOriginKind.UNRESOLVED
        refs[position] = st.FeedbackOriginRef.NONE
        commits[position] = None
        txns[position] = None
        scores[position] = None
        statuses[position] = st.ActiveScoreStatus.MASKED
        previous = source.provenance_by_pos[position]
        provenance[position] = st.PositionProvenance(
            first_origin=previous.first_origin,
            last_origin=st.OriginEvidence(
                origin_kind=st.ActiveOriginKind.UNRESOLVED, origin_ref=st.FeedbackOriginRef.NONE,
                commit=reentry_key, token=source.mask_token_id, evidence_logprob=None,
                transition_id=None, evidence_digest=source_digest,
            ),
            n_origin_events=previous.n_origin_events + 1,
        )

    # carry_from_source and hard anchors keep their source rows verbatim.

    # ---- the schedule-band gate ---------------------------------------------------------------
    n_editable = len(source.editable_positions)
    n_unresolved = sum(1 for p in source.editable_positions if tokens[p] == source.mask_token_id)
    if n_unresolved == 0:
        return reject(
            "the policy fully resolves the editable domain; q_phi produces live partial states, "
            "while the one-position terminal best-lookahead fallback is owned by the cycle"
        )

    observed = observe_maturity(
        length_total=len(tokens), n_fixed=len(tokens) - n_editable,
        n_editable=n_editable, n_unresolved_editable=n_unresolved,
    )
    verdict = gate_projected_maturity(
        table=band_table, step=coordinates.r_step, observed=observed, stratum_key=stratum_key,
        declared_calibration_id=declared_band_id, declared_calibration_digest=declared_band_digest,
    )
    if not verdict.accepted:
        return _null(st.TransitionOutcome.NULL_BAND_INCOMPATIBLE,
                     f"projected maturity is outside B(r_d): {verdict.reason.value} {verdict.detail}",
                     reasons, (verdict,), policy_identity, context)

    # ---- the coupled mask-load contract -------------------------------------------------------
    n_endpoint_writes_over_masked = sum(
        1 for p in write_set if source.tokens[p] == source.mask_token_id)
    try:
        load = admissible_reopen_cardinality(
            band=lookup_band(band_table, step=coordinates.r_step, stratum_key=stratum_key),
            n_editable=n_editable,
            n_unresolved_source=len(set(source.editable_positions) - source_resolved),
            n_endpoint_writes_over_masked=n_endpoint_writes_over_masked,
        )
    except (EmptyBandIntersectionError, MissingScheduleBandError) as exc:
        # An empty admissible envelope is a fact about the SCHEDULE, not about the policy.
        return _null(st.TransitionOutcome.NULL_BAND_INCOMPATIBLE, str(exc), reasons, (verdict,),
                     policy_identity, context)
    if not load.feasible:
        return _null(st.TransitionOutcome.NULL_BAND_INCOMPATIBLE,
                     load.infeasible_reason or "the admissible reopen envelope is empty",
                     reasons, (verdict,), policy_identity, context)

    load_verdict = validate_realized_mask_load(load=load, n_newly_masked=len(newly_masked))
    if not load_verdict.accepted:
        # A realized count outside a FEASIBLE envelope is a fact about the POLICY.
        return _null(st.TransitionOutcome.NULL_INVALID_POLICY_RESULT,
                     f"realized reopen cardinality {len(newly_masked)} is outside the admissible "
                     f"envelope [{load.min_newly_masked}, {load.max_newly_masked}]",
                     reasons, (verdict, load_verdict), policy_identity, context)

    # ---- construct the projected state ---------------------------------------------------------
    support = st.SupportPartition(
        write_from_endpoint=decision.write_from_endpoint,
        inject_from_source_feedback=decision.inject_from_source_feedback,
        reopen=decision.reopen,
        carry_from_source=decision.carry_from_source,
        reason_by_pos=reasons,
    )
    try:
        projected = st.ProjectedPartialState(
            schema_version=source.schema_version,
            lineage=LineageRef(
                protein_id=source.lineage.protein_id, root_id=source.lineage.root_id,
                family_id=source.lineage.family_id, depth=depth,
                parent_state_id=source.state_id, parent_transition_id=origin_transition_id,
                origin_endpoint_id=endpoint.endpoint_id,
            ),
            r_step=coordinates.r_step, c_next_step=coordinates.c_next_step,
            n_steps=coordinates.n_steps, coordinate_law=coordinates.law,
            tokens=tuple(tokens), mask_token_id=source.mask_token_id,
            aa_token_ids=source.aa_token_ids, hard_anchors=source.hard_anchors,
            editable_positions=source.editable_positions,
            source_resolved_positions=tuple(sorted(source_resolved)),
            active_origin_kind_by_pos=tuple(kinds),
            feedback_origin_ref_by_pos=tuple(refs),
            active_commit_depth_step_by_pos=tuple(commits),
            origin_transition_id_by_pos=tuple(txns),
            active_sampler_score_by_pos=tuple(scores),
            active_score_status_by_pos=tuple(statuses),
            provenance_by_pos=tuple(provenance),
            support=support,
            active_temporary_protection=tuple(sorted(protections, key=lambda p: p.position)),
            descendant_fork_seed=descendant_fork_seed,
            origin_transition_id=origin_transition_id,
            declared_band_id=declared_band_id, declared_band_digest=declared_band_digest,
            inherited_lineage_dfe=source.accumulated_lineage_dfe,
            planned_segment_dfe=coordinates.c_next_step - coordinates.r_step,
            conditioning=source.conditioning, safety_reference=source.safety_reference,
            cost_event_ids=source.cost_event_ids,
        )
    except V2Error as exc:
        # Fail CLOSED but observably: no invented state, and the construction error text survives
        # in the telemetry so an unanticipated policy shape is diagnosable rather than silent.
        return reject(f"the policy result does not yield a legal projected state: {exc}")

    st.validate_endpoint_writeback(projected=projected, endpoint_tokens=endpoint_tokens)

    return ProjectionOutcome(
        outcome=st.TransitionOutcome.COMMITTED, projected=projected,
        detail="", reason_by_pos=reasons,
        band_verdicts=(verdict, load_verdict), policy=policy_identity, context=context,
    )
