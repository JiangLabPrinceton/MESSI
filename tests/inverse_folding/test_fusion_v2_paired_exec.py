"""V2F5-2b: the three intervention axes, SupportBudget parity, and the paired executor.

Three changes over the skeleton, all of them about making a contrast MEAN something:

**Three axes instead of one flat list.**  ``mechanism_view`` (what the feedback machinery is doing),
``endpoint_relation`` (which endpoint the arm was handed), and ``policy_variant`` (what the policy
was allowed to see) are orthogonal.  Flattening them into one enum made "reward-ordered endpoint"
look like a sibling of "feedback off", which it is not: one is a claim about selection, the other
about the mechanism.  A point in the 3-axis space is one intervention.

**SupportBudget parity, enforced by the runner.**  Two arms are only a contrast if their supports
have the SAME ACTION VECTOR -- the same number of endpoint writes, source-feedback injections,
reopens, and carries.  Otherwise the arms differ in how much was changed as well as in what, and a
difference in outcome cannot be attributed.  PLAN §2.5 requires the source-off comparator to be
"matched on ... support cardinality"; the runner is where that becomes true rather than hoped for.

**An executor that actually runs both arms.**  The skeleton proves the seed law before any model
exists.  The executor proves the mechanism claim: it runs the two arms under identical seeds and
reports whether the projection depended on what the intervention perturbed.

Scope per the current plan: the four MECHANISM views execute now.  ``reward_ordered`` and
``source_off`` have interfaces in place but their real gates wait for the production policy freeze
(PLAN §2.5), so they are declared and refused rather than silently half-run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from inverse_folding.reference_flow.fusion_v2 import policy as pol
from inverse_folding.reference_flow.fusion_v2 import seeds as sd
from inverse_folding.reference_flow.fusion_v2 import state as st
from inverse_folding.reference_flow.fusion_v2_runtime.paired import (
    ENDPOINT_RELATIONS,
    MECHANISM_VIEWS,
    POLICY_VARIANTS,
    InterventionAxes,
    NotYetFrozenError,
    SupportBudget,
    V2PairedError,
    assert_support_parity,
    build_paired_contrasts,
    run_mechanism_views,
)
from tests.inverse_folding import _v2_fixtures as F
from tests.inverse_folding.test_fusion_v2_cycle import (
    ALPHABET,
    VOCAB,
    _FakeHeadOracle,
    _cfg,
    _denoiser,
    _module_meter,
    _structure_oracle,
    _support_policy,
    _wide_band_table,
)
from inverse_folding.reference_flow.sampler import PositionDependentDFMSampler


#: A MATURE source on purpose.  At c=50 this toy world resolves only about one editable position,
#: and a source intervention cannot bite on one position: ablating it leaves nothing to reopen, and
#: shuffling a single value is the identity permutation.  Both would report "no source dependence"
#: for a reason that has nothing to do with the kernel.
C0, R1, C1 = 80, 40, 90


def _pair_context(**over):
    kw = dict(
        seed_schema=sd.V2_SEED_ENCODING_VERSION, campaign_id="v2-canary", split_role="dev",
        master_seed=20260805, protein_id="5ZHV_B", depth=0, r_step=R1, c_next_step=C1,
        pair_ordinal=0, n_forks=2, n_descendant_lookaheads=2,
    )
    kw.update(over)
    return sd.FeedbackPairSeedContext(**kw)


# --------------------------------------------------------------------------------------------
# the three axes
# --------------------------------------------------------------------------------------------


def test_the_four_mechanism_views_are_the_ones_plan_names():
    assert set(MECHANISM_VIEWS) == {
        "feedback_off", "endpoint_change", "source_change", "source_shuffle"}


def test_the_endpoint_relation_axis_is_separate_from_the_mechanism_axis():
    """A reward-ordered endpoint pair is a claim about SELECTION, not about the mechanism; putting
    it in the mechanism enum made it look like a sibling of feedback-off."""
    assert "reward_ordered" in ENDPOINT_RELATIONS
    assert "reward_ordered" not in MECHANISM_VIEWS


def test_the_policy_variant_axis_is_separate_from_the_mechanism_axis():
    assert "source_off" in POLICY_VARIANTS
    assert "source_off" not in MECHANISM_VIEWS


def test_each_axis_has_an_explicit_neutral_level():
    """Without a neutral level, every contrast would be a compound intervention."""
    assert "as_selected" in ENDPOINT_RELATIONS
    assert "as_configured" in POLICY_VARIANTS


def test_an_intervention_is_a_point_in_the_three_axis_space():
    axes = InterventionAxes(mechanism_view="feedback_off", endpoint_relation="as_selected",
                            policy_variant="as_configured")
    assert axes.kind
    assert "feedback_off" in axes.kind


def test_two_different_points_have_different_kinds():
    a = InterventionAxes(mechanism_view="feedback_off", endpoint_relation="as_selected",
                         policy_variant="as_configured")
    b = InterventionAxes(mechanism_view="source_change", endpoint_relation="as_selected",
                         policy_variant="as_configured")
    assert a.kind != b.kind


@pytest.mark.parametrize("field,value", [
    ("mechanism_view", "vibes"), ("endpoint_relation", "vibes"), ("policy_variant", "vibes"),
])
def test_an_unknown_level_on_any_axis_is_refused(field, value):
    kw = dict(mechanism_view="feedback_off", endpoint_relation="as_selected",
              policy_variant="as_configured")
    kw[field] = value
    with pytest.raises(V2PairedError):
        InterventionAxes(**kw)


def test_the_contrast_builder_accepts_axes():
    contrasts = build_paired_contrasts(
        context=_pair_context(), fork_index=0, c_source_step=C0,
        interventions=tuple(
            InterventionAxes(mechanism_view=view, endpoint_relation="as_selected",
                             policy_variant="as_configured")
            for view in MECHANISM_VIEWS
        ),
    )
    assert len(contrasts) == len(MECHANISM_VIEWS)


# --------------------------------------------------------------------------------------------
# SupportBudget parity
# --------------------------------------------------------------------------------------------


def test_the_budget_is_the_action_vector_of_a_partition():
    partition = st.SupportPartition(
        write_from_endpoint=(1,), inject_from_source_feedback=(4,), reopen=(2,),
        carry_from_source=(3, 5), reason_by_pos={i: "x" for i in (1, 2, 3, 4, 5)},
    )
    budget = SupportBudget.of(partition)
    assert budget.n_write_from_endpoint == 1
    assert budget.n_inject_from_source_feedback == 1
    assert budget.n_reopen == 1
    assert budget.n_carry_from_source == 2


def test_two_partitions_with_the_same_shape_have_the_same_budget():
    """Parity is about SHAPE, not about which positions were chosen -- that is the whole point."""
    a = st.SupportPartition(write_from_endpoint=(1,), inject_from_source_feedback=(4,),
                            reopen=(2,), carry_from_source=(3, 5),
                            reason_by_pos={i: "x" for i in (1, 2, 3, 4, 5)})
    b = st.SupportPartition(write_from_endpoint=(2,), inject_from_source_feedback=(3,),
                            reopen=(1,), carry_from_source=(4, 5),
                            reason_by_pos={i: "x" for i in (1, 2, 3, 4, 5)})
    assert SupportBudget.of(a) == SupportBudget.of(b)


def test_parity_holds_for_identical_action_vectors():
    budget = SupportBudget(n_write_from_endpoint=1, n_inject_from_source_feedback=1,
                           n_reopen=1, n_carry_from_source=2)
    assert_support_parity([budget, budget])


def test_parity_fails_when_one_arm_reopens_more():
    """Two arms that reopened different numbers of positions differ in HOW MUCH was changed as well
    as in WHAT, so a difference in outcome cannot be attributed to the intervention."""
    a = SupportBudget(n_write_from_endpoint=1, n_inject_from_source_feedback=1,
                      n_reopen=1, n_carry_from_source=2)
    b = SupportBudget(n_write_from_endpoint=1, n_inject_from_source_feedback=1,
                      n_reopen=2, n_carry_from_source=1)
    with pytest.raises(V2PairedError, match="parity|reopen"):
        assert_support_parity([a, b])


@pytest.mark.parametrize("field", [
    "n_write_from_endpoint", "n_inject_from_source_feedback", "n_reopen", "n_carry_from_source",
])
def test_parity_fails_on_any_component_of_the_action_vector(field):
    base = SupportBudget(n_write_from_endpoint=1, n_inject_from_source_feedback=1,
                         n_reopen=1, n_carry_from_source=2)
    import dataclasses

    other = dataclasses.replace(base, **{field: getattr(base, field) + 1})
    with pytest.raises(V2PairedError):
        assert_support_parity([base, other])


def test_parity_of_a_single_arm_is_trivially_satisfied():
    budget = SupportBudget(1, 1, 1, 2)
    assert_support_parity([budget])


# --------------------------------------------------------------------------------------------
# the runner enforces parity
# --------------------------------------------------------------------------------------------


def _exec(**over):
    table = _wide_band_table()
    kw = dict(
        context=_pair_context(), fork_index=0,
        sampler=PositionDependentDFMSampler(mask_token_id=F.MASK, vocab_size=VOCAB),
        denoiser=_denoiser, config=_cfg(), sequence_length=F.L,
        h_values=np.zeros(F.L, dtype=np.float32), residue_token_ids=F.AA,
        alphabet=ALPHABET, fixed_tokens={0: 10}, lineage=F.lineage(),
        mask_token_id=F.MASK, aa_token_ids=F.AA, conditioning=F.conditioning(
            # The kernel refuses a band table the run's own frozen provenance does not
            # name, so the fixture binds the two instead of leaving them unrelated.
            schedule_band_calibration=table.provenance.calibration_content_digest),
        safety_reference=F.safety_reference(),
        c_source_step=C0, r_step=R1, c_next_step=C1,
        source_fork_seeds=(5001, 5002, 5003),
        head_oracle=_FakeHeadOracle(), structure_oracle=_structure_oracle,
        support_policy=_support_policy,
        band_table=table, stratum_key=F.STRATUM, declared_band_id=F.BAND_ID,
        declared_band_digest=table.provenance.calibration_content_digest,
        origin_transition_id=F.TXN,
        safety_gate=F.safety_gate(),
        declared_policy=F.declared_policy(),
        feedback_enabled=True,
        cost_meter=_module_meter(),
    )
    kw.update(over)
    return run_mechanism_views(**kw)


def test_the_executor_runs_every_mechanism_view():
    results = _exec()
    assert {r.axes.mechanism_view for r in results} == set(MECHANISM_VIEWS)


def test_each_view_produces_two_arms_under_identical_seeds():
    for result in _exec():
        assert result.arm_a.realized_propagation_seed == result.arm_b.realized_propagation_seed


@pytest.mark.parametrize("view", ["feedback_off", "endpoint_change"])
def test_arms_that_do_not_perturb_the_source_share_it_exactly(view):
    """A2 and V2 share the pre-feedback prefix; regenerating it per arm would reintroduce exactly
    the root-sampling variance the matched design exists to remove.

    Only the non-source views can assert this: source_change and source_shuffle perturb the source
    on purpose, and asserting equality there would be asserting the intervention did nothing."""
    result = next(r for r in _exec() if r.axes.mechanism_view == view)
    assert result.arm_a.cycle.source.state_id == result.arm_b.cycle.source.state_id


@pytest.mark.parametrize("view", ["source_change", "source_shuffle"])
def test_the_source_perturbing_views_really_change_the_source(view):
    """The mirror: an intervention that left the source identical would make its whole contrast
    vacuous, and the verdict would read as "no dependence" for the wrong reason."""
    result = next(r for r in _exec() if r.axes.mechanism_view == view)
    assert result.arm_a.cycle.source.tokens != result.arm_b.cycle.source.tokens


def test_every_view_reports_either_a_verdict_or_a_parity_violation():
    """Never neither, and never both.

    A view that produced neither would be silently unusable; a view that produced both would be
    laundering a malformed contrast into evidence.  With the runner enforcing parity, a policy that
    cannot hold the action vector fixed under the intervention surfaces HERE, as a named violation,
    rather than as a difference nobody can attribute."""
    for result in _exec():
        has_verdict = result.verdict is not None
        has_violation = bool(result.parity_violation)
        assert has_verdict != has_violation, result.axes.kind
        if has_verdict:
            assert result.verdict.subject


def test_a_parity_failure_surfaces_as_a_typed_null_from_the_RUNNER():
    """Enforcement lives in the runner, not in a post-hoc comparison.

    The difference is observable and it matters: with runner enforcement the offending arm never
    runs to completion -- it returns NULL_INVALID_POLICY_RESULT naming the parity failure, so no
    downstream consumer can mistake a mismatched contrast for a completed one.  A post-hoc check
    would let the arm commit and only flag it afterwards."""
    violated = [r for r in _exec() if r.parity_violation]
    assert violated, "no view exercised the parity path; the control would be vacuous"
    for result in violated:
        assert result.arm_b.cycle.outcome is st.TransitionOutcome.NULL_INVALID_POLICY_RESULT
        assert "parity" in result.arm_b.cycle.detail
        assert result.arm_b.cycle.projected is None


def test_the_endpoint_change_view_really_hands_the_second_arm_a_different_endpoint():
    """Otherwise the contrast is a comparison of an endpoint with itself, and its verdict would
    read as "the kernel ignores the endpoint" for entirely the wrong reason."""
    result = next(r for r in _exec() if r.axes.mechanism_view == "endpoint_change")
    a, b = result.arm_a.cycle, result.arm_b.cycle
    if b.outcome is st.TransitionOutcome.COMMITTED:
        assert a.selected_endpoint.endpoint_id != b.selected_endpoint.endpoint_id
    else:
        # Too few admissible endpoints to form the pair is a legitimate typed null, but it must be
        # reported rather than silently degenerating into a self-comparison.
        assert b.selected_endpoint is None or b.projected is None
        assert result.verdict is None or result.parity_violation


def test_the_source_change_view_perturbs_the_source():
    result = next(r for r in _exec() if r.axes.mechanism_view == "source_change")
    assert result.arm_b.cycle.source.tokens != result.arm_a.cycle.source.tokens


def test_the_source_shuffle_view_preserves_the_source_multiset():
    result = next(r for r in _exec() if r.axes.mechanism_view == "source_shuffle")
    a, b = result.arm_a.cycle.source, result.arm_b.cycle.source
    editable = list(a.editable_positions)
    assert sorted(a.tokens[i] for i in editable) == sorted(b.tokens[i] for i in editable)


def test_the_feedback_off_view_leaves_one_arm_unprojected():
    """A2 is the same run with feedback disabled; its arm must have no projection at all."""
    result = next(r for r in _exec() if r.axes.mechanism_view == "feedback_off")
    assert result.arm_b.cycle.projected is None


def test_a_support_parity_violation_is_reported_not_hidden():
    """If the two arms' policies chose different-sized supports, the contrast is malformed and the
    executor must say so rather than return a difference that cannot be attributed."""
    calls = {"n": 0}

    def lopsided(source, endpoint, coordinates):
        calls["n"] += 1
        decision = _support_policy(source, endpoint, coordinates)
        if calls["n"] % 2 == 0 and isinstance(decision, pol.PolicyDecision):
            # give the second arm one extra reopen, taken from its carries
            carry = list(decision.carry_from_source)
            if carry:
                moved = carry.pop()
                import dataclasses
                return dataclasses.replace(
                    decision, reopen=tuple(sorted(decision.reopen + (moved,))),
                    carry_from_source=tuple(carry))
        return decision

    results = _exec(support_policy=lopsided)
    assert any(r.parity_violation for r in results)


def test_a_parity_violated_view_carries_no_dependence_claim():
    """A malformed contrast must not also report a verdict: that would launder a broken experiment
    into evidence."""
    calls = {"n": 0}

    def lopsided(source, endpoint, coordinates):
        calls["n"] += 1
        decision = _support_policy(source, endpoint, coordinates)
        if calls["n"] % 2 == 0 and isinstance(decision, pol.PolicyDecision):
            carry = list(decision.carry_from_source)
            if carry:
                moved = carry.pop()
                import dataclasses
                return dataclasses.replace(
                    decision, reopen=tuple(sorted(decision.reopen + (moved,))),
                    carry_from_source=tuple(carry))
        return decision

    for result in _exec(support_policy=lopsided):
        if result.parity_violation:
            assert result.verdict is None


# --------------------------------------------------------------------------------------------
# not-yet-frozen axes are refused, not half-run
# --------------------------------------------------------------------------------------------


def test_a_reward_ordered_endpoint_relation_is_declared_but_not_executable_yet():
    """PLAN §2.5 freezes the FeedbackSupportPolicySpec only after real one-cycle transmission.
    Until then the interface exists and the gate refuses to run, so nobody mistakes a placeholder
    for evidence."""
    axes = InterventionAxes(mechanism_view="feedback_off", endpoint_relation="reward_ordered",
                            policy_variant="as_configured")
    with pytest.raises(NotYetFrozenError, match="reward_ordered|frozen"):
        _exec(interventions=(axes,))


def test_a_source_off_policy_variant_is_declared_but_not_executable_yet():
    axes = InterventionAxes(mechanism_view="feedback_off", endpoint_relation="as_selected",
                            policy_variant="source_off")
    with pytest.raises(NotYetFrozenError, match="source_off|frozen"):
        _exec(interventions=(axes,))


def test_the_neutral_levels_do_execute():
    axes = InterventionAxes(mechanism_view="feedback_off", endpoint_relation="as_selected",
                            policy_variant="as_configured")
    assert len(_exec(interventions=(axes,))) == 1


# --------------------------------------------------------------------------------------------
# A2 is a VIEW of one realized pre-feedback pool, not a second run of the same recipe
# --------------------------------------------------------------------------------------------


class _CountingSampler:
    """Counts the exact completions the whole paired executor performs."""

    def __init__(self):
        self._inner = PositionDependentDFMSampler(mask_token_id=F.MASK, vocab_size=VOCAB)
        self.source_captures = 0
        self.completions = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def sample(self, **kw):
        if kw.get("continuation_resume") is not None:
            self.completions += 1
        if kw.get("continuation") is not None:
            self.source_captures += 1
        return self._inner.sample(**kw)


def test_every_arm_reads_one_realized_pre_feedback_pool():
    """PLAN §2.3/§4.3: A2 is "the archive state immediately before feedback" of the SAME run.

    Re-executing the recipe per arm is not that.  Identical seeds make the two executions produce
    equal values under a deterministic fake oracle, so the difference is invisible in a unit test --
    but the pool is the thing both arms are supposed to SHARE, and two executions share nothing.
    A real Head on a GPU is not bit-reproducible, and a real structure oracle is a cache with
    state, so under the conditions the experiment actually runs the two arms would be comparing
    against different realized pools while reporting a matched contrast.

    The cost is the observable: one shared pool means the source prefix and its K lookaheads are
    paid ONCE for the whole executor, not once per arm per view.
    """
    sampler = _CountingSampler()
    source_seeds = (5001, 5002, 5003)
    results = _exec(sampler=sampler, source_fork_seeds=source_seeds)
    assert len(results) >= 2, "the contrast needs more than one view to be worth sharing"

    assert sampler.source_captures == 1, (
        f"the source prefix was captured {sampler.source_captures} times; every arm of every view "
        "is defined over ONE source state"
    )
    # Each arm that actually propagated screens its OWN descendants -- that work is per-arm by
    # definition.  Everything before the projection is not.
    propagated_arms = sum(
        1 for r in results for arm in (r.arm_a, r.arm_b) if arm.cycle.propagated is not None)
    per_arm_descendants = _pair_context().n_descendant_lookaheads
    assert sampler.completions == len(source_seeds) + propagated_arms * per_arm_descendants, (
        f"expected one shared pool of {len(source_seeds)} plus {propagated_arms} x "
        f"{per_arm_descendants} descendants; the pre-feedback pool was generated more than once"
    )


def test_the_a2_arm_returns_the_shared_pool_itself_not_a_copy_of_it():
    """Object identity, because equality is exactly what a re-run would also satisfy."""
    results = {r.axes.mechanism_view: r for r in _exec()}
    a2 = results["feedback_off"]
    assert a2.arm_b.cycle.projected is None, "the A2 arm must have no projection at all"
    for view, result in results.items():
        assert result.arm_a.cycle.endpoints is a2.arm_b.cycle.endpoints, (
            f"view {view!r} arm A was screened against a different endpoint pool than A2"
        )
        assert result.arm_a.cycle.source is a2.arm_b.cycle.source


def test_each_executed_arm_has_an_independent_post_feedback_archive():
    """The source/lookahead pool is shared, but an arm's descendants are not shared history."""
    results = _exec()
    checked = 0
    for result in results:
        if result.axes.mechanism_view == "feedback_off":
            continue
        arm_a = result.arm_a.cycle
        arm_b = result.arm_b.cycle
        if not arm_a.descendant_endpoints or not arm_b.descendant_endpoints:
            continue
        checked += 1
        assert arm_a.archive is not arm_b.archive
        a_descendants = {endpoint.endpoint_id for endpoint in arm_a.descendant_endpoints}
        b_descendants = {endpoint.endpoint_id for endpoint in arm_b.descendant_endpoints}
        a_archive = {row.endpoint_id for row in arm_a.archive.raw_rows()}
        b_archive = {row.endpoint_id for row in arm_b.archive.raw_rows()}
        assert a_descendants <= a_archive
        assert b_descendants <= b_archive
        assert a_descendants.isdisjoint(b_archive)
        assert b_descendants.isdisjoint(a_archive)
    assert checked, "the fixture exercised no two-arm post-feedback archive pair"


def test_a_source_perturbation_holds_the_endpoint_pool_fixed():
    """The source probes ask whether q_phi is blind to P, so y must not move with it.

    Regenerating the lookaheads from the perturbed source changes P AND y together, and a
    difference in the projection could then be attributed to either.
    """
    results = {r.axes.mechanism_view: r for r in _exec()}
    for view in ("source_change", "source_shuffle"):
        result = results[view]
        assert result.arm_b.cycle.endpoints is result.arm_a.cycle.endpoints, (
            f"{view} regenerated its endpoints, so the contrast confounds P with y"
        )
        assert result.arm_b.cycle.source is not result.arm_a.cycle.source, (
            f"{view} did not actually perturb the source"
        )


def test_the_head_is_asked_once_per_distinct_pre_feedback_sequence():
    """A shared pool is scored once; a re-run scores it again under a different call."""
    oracle = _FakeHeadOracle()
    results = _exec(head_oracle=oracle)
    pre_feedback = {e.sequence_md5 for r in results for e in r.arm_a.cycle.endpoints}
    descendant_batches = sum(
        1 for r in results for arm in (r.arm_a, r.arm_b) if arm.cycle.propagated is not None)
    assert oracle.calls == 1 + descendant_batches, (
        f"the Head was called {oracle.calls} times for {len(pre_feedback)} shared pre-feedback "
        f"sequences and {descendant_batches} descendant pools"
    )


# --------------------------------------------------------------------------------------------
# PLAN §2.6 intervention 2: hold endpoint and support FIXED, ablate the source
# --------------------------------------------------------------------------------------------


def test_the_source_ablation_arm_actually_produces_a_verdict():
    """PLAN §2.6 intervention 2 is "hold endpoint `y` and support FIXED, change or ablate
    compatible source state `P`", and PLAN §8.3 reads its null result as a CLASSIFICATION:
    "source ablation has no effect: classify the mechanism as source-forgetting v0-style reopen".

    The arm therefore has to be able to return an answer.  Re-deriving the support from the ablated
    source and then checking its CARDINALITY cannot: masking a resolved position necessarily moves
    it between support classes, so the action vector changes by construction and the parity check
    kills the arm every time.  `source_shuffle` preserves resolvedness and still passes, so the
    executor looks green while the one control that separates trajectory-coupled feedback from
    source-forgetting reopen silently yields nothing.
    """
    results = _exec()
    ablation = {r.axes.mechanism_view: r for r in results}["source_change"]
    if ablation.verdict is not None:
        return
    # Where the coordinate admits no fixed-support ablation the arm may legitimately carry no
    # dependence claim -- but it must say WHY in its own terms.  A bare parity message reads as a
    # broken experiment and sends the operator to fix the wrong thing.
    assert "support was NOT held fixed" in ablation.parity_violation, (
        f"the arm returned no verdict and did not name the cause: {ablation.parity_violation}")


def test_the_source_ablation_arm_holds_the_support_partition_fixed():
    """"Fixed" means the SAME four sets, not merely the same four cardinalities.

    Two partitions can share an action vector and still act on different positions, which is
    exactly the confound this intervention exists to exclude.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.paired import _fixed_support_ablation

    results = _exec()
    ablation = {r.axes.mechanism_view: r for r in results}["source_change"]
    pinned, targets = _fixed_support_ablation(ablation.arm_a.cycle)
    if pinned is None:
        assert targets == ()
        return
    a = ablation.arm_a.cycle.projected.support
    b = ablation.arm_b.cycle.projected.support
    for name in ("write_from_endpoint", "inject_from_source_feedback", "reopen",
                 "carry_from_source"):
        assert getattr(a, name) == getattr(b, name), f"{name} differs between the paired arms"


def test_the_ablated_position_really_lost_its_source_byte():
    """The guard must not be satisfiable by ablating nothing."""
    results = _exec()
    ablation = {r.axes.mechanism_view: r for r in results}["source_change"]
    a_tokens = ablation.arm_a.cycle.source.tokens
    b_tokens = ablation.arm_b.cycle.source.tokens
    changed = [i for i, (x, y) in enumerate(zip(a_tokens, b_tokens)) if x != y]
    assert changed, "no source byte was ablated at all"

    mask = ablation.arm_b.cycle.source.mask_token_id
    assert all(b_tokens[i] == mask for i in changed)


# --------------------------------------------------------------------------------------------
# each arm is its own transition
# --------------------------------------------------------------------------------------------


def test_each_arm_runs_under_its_own_transition_id():
    """Arm A and arm B are two transitions, not one attempted twice.

    The ledger keys its event ids on the transition id and counts a repeat under one event as a
    RETRY -- a guard that exists to stop one arm from silently drawing more compute than its match.
    Sharing the id across arms blinds exactly that guard: on the cluster a matched two-view run
    read as 13 retries against a 2-retry cap with nothing retried at all (job 12098305).
    """
    seen = []
    for result in _exec():
        for arm in (result.arm_a, result.arm_b):
            projected = arm.cycle.projected
            if projected is not None:
                seen.append((result.axes.mechanism_view, arm.arm_slot,
                             projected.origin_transition_id))
    assert seen, "no arm produced a projection; the fixture cannot prove anything"
    ids = [txn for _, _, txn in seen]
    assert len(set(ids)) == len(ids), f"transition ids collide across arms: {seen}"


def test_no_arm_reuses_the_callers_transition_id_for_its_own_projection():
    """The caller's id belongs to the shared pre-feedback stage, which is A2 and ran once."""
    for result in _exec():
        for arm in (result.arm_a, result.arm_b):
            projected = arm.cycle.projected
            if projected is not None:
                assert projected.origin_transition_id != F.TXN


def test_an_arm_transition_id_is_a_well_formed_transition_id():
    """It must stay parseable as the identity layer's own format, not a suffixed string."""
    from inverse_folding.reference_flow.fusion_v2 import identity as ident

    for result in _exec():
        for arm in (result.arm_a, result.arm_b):
            projected = arm.cycle.projected
            if projected is not None:
                assert ident.id_namespace(projected.origin_transition_id)
                assert ":txn:d0:" in projected.origin_transition_id
