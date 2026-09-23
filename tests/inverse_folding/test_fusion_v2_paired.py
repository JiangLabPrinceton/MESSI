"""V2F5-2: paired feedback-transmission evidence (PLAN task V2F5, §2.6).

This is the layer that makes the cycle FALSIFIABLE.  A committed cycle proves only that the wiring
runs; a paired cycle asks whether the projected state actually depends on the things the mechanism
claims it depends on.

PLAN V2F5 requires four paired views plus two comparators:

* **feedback-off** (A2): the same source, no projection at all;
* **endpoint-change**: the same source and the same support, a DIFFERENT selected endpoint;
* **source-change / ablation**: the same endpoint and the same support, a perturbed source;
* **source-shuffle**: the same endpoint and support, the source's own bytes permuted;
* a **reward-ordered endpoint pair**; and
* a **policy-level source-off view** matched on support cardinality, horizon, seeds, and resources.

The seed law is the sharp edge.  PLAN §2.6: "All paired descendants use ``FeedbackPairSeedContext``;
treatment content may not enter the shared seed."  If a treatment could reach the seed, the two arms
would draw different randomness and every observed difference would be confounded with sampling
variance -- the exact failure the design exists to rule out.  So the tests below check not that the
seeds "look independent", but that the two arms draw the IDENTICAL seed while differing in treatment.

The acceptance bullet that keeps this honest: "a deliberately source-blind kernel fails the
deterministic source-dependence check".  A positive control that never fires is decoration.
"""

from __future__ import annotations

import dataclasses

import pytest

from inverse_folding.reference_flow.fusion_v2 import seeds as sd
from inverse_folding.reference_flow.fusion_v2 import state as st
from inverse_folding.reference_flow.fusion_v2_runtime.paired import (
    INTERVENTIONS,
    V2PairedError,
    ablate_source,
    build_paired_contrasts,
    shuffle_source,
    source_dependence_verdict,
)
from tests.inverse_folding import _v2_fixtures as F


def _pair_context(**over):
    kw = dict(
        seed_schema=sd.V2_SEED_ENCODING_VERSION,
        campaign_id="v2-canary", split_role="dev", master_seed=20260805, protein_id="5ZHV_B",
        depth=0, r_step=F.R_STEP, c_next_step=F.C_NEXT, pair_ordinal=0,
        n_forks=2, n_descendant_lookaheads=3,
    )
    kw.update(over)
    return sd.FeedbackPairSeedContext(**kw)


def _contrasts(context=None, fork_index=0, **kw):
    return build_paired_contrasts(
        context=context or _pair_context(), fork_index=fork_index,
        c_source_step=F.C_SOURCE, **kw)


def _arms(context=None, fork_index=0):
    """Every arm of every contrast, flattened."""
    return [arm for contrast in _contrasts(context, fork_index) for arm in contrast.arms]


# --------------------------------------------------------------------------------------------
# the four paired views exist and are declared
# --------------------------------------------------------------------------------------------


def test_the_four_required_paired_views_are_declared():
    """PLAN V2F5 names them explicitly; a missing arm is a missing falsifier."""
    assert {"feedback_off", "endpoint_change", "source_change", "source_shuffle"} <= \
        set(INTERVENTIONS)


def test_build_returns_one_contrast_per_declared_mechanism_view():
    """``intervention_kind`` is now the canonical encoding of a point in the three-axis space, so
    the mechanism view is read off the axes rather than off the string."""
    contrasts = _contrasts()
    assert {c.axes.mechanism_view for c in contrasts} == set(INTERVENTIONS)
    assert all(c.intervention_kind == c.axes.kind for c in contrasts)


def test_every_arm_declares_its_own_treatment_identity():
    """PLAN acceptance: "all paired identities and realized seeds are explicit"."""
    arms = _arms(_pair_context(), 0)
    identities = [arm.treatment_identity for arm in arms]
    assert len(set(identities)) == len(identities)
    assert all(identity for identity in identities)


def test_the_two_arms_of_a_contrast_share_one_match_identity():
    """Within a contrast the arms differ ONLY in treatment; across contrasts they are different
    experiments and must not collide."""
    contrasts = _contrasts()
    for contrast in contrasts:
        assert contrast.arm_a.shared_match_identity == contrast.arm_b.shared_match_identity
    assert len({c.arm_a.shared_match_identity for c in contrasts}) == len(contrasts)


def test_each_contrast_is_exactly_two_arms():
    """PLAN §2.6's pairing law: a contrast is one treatment against one matched control."""
    for contrast in _contrasts():
        assert len(contrast.arms) == 2
        assert {arm.arm_slot for arm in contrast.arms} == {"arm_a", "arm_b"}
        assert contrast.arm_a.treatment_identity != contrast.arm_b.treatment_identity


# --------------------------------------------------------------------------------------------
# THE SEED LAW: treatment may not reach the seed
# --------------------------------------------------------------------------------------------


def test_all_paired_arms_draw_the_identical_propagation_seed():
    """PLAN §2.6: "treatment content may not enter the shared seed".  If it did, the arms would
    differ in randomness as well as in treatment, and no observed difference could be attributed."""
    arms = _arms(_pair_context(), 0)
    assert len({arm.realized_propagation_seed for arm in arms}) == 1


def test_all_paired_arms_draw_the_identical_lookahead_seeds():
    arms = _arms(_pair_context(), 0)
    assert len({arm.realized_lookahead_seeds for arm in arms}) == 1


def test_every_contrast_survives_the_v2f1_matched_pair_assertion():
    """The V2F1 seed layer re-derives every seed from context alone and enforces the pairing law:
    exactly two rows, identical seeds, distinct treatment.  A treatment folded into a seed fails
    there on every real run, not only under test."""
    for contrast in _contrasts():
        sd.assert_matched_pair_seeds([arm.seed_record for arm in contrast.arms])


def test_a_different_fork_index_changes_the_seed():
    """Forks must be collision-free even though arms are not."""
    a = _arms(_pair_context(), 0)[0]
    b = _arms(_pair_context(), 1)[0]
    assert a.realized_propagation_seed != b.realized_propagation_seed


def test_a_different_pair_ordinal_changes_the_seed():
    a = _arms(_pair_context(pair_ordinal=0), 0)[0]
    b = _arms(_pair_context(pair_ordinal=1), 0)[0]
    assert a.realized_propagation_seed != b.realized_propagation_seed


def test_a_different_depth_changes_the_seed():
    a = _arms(_pair_context(depth=0), 0)[0]
    b = _arms(_pair_context(depth=1), 0)[0]
    assert a.realized_propagation_seed != b.realized_propagation_seed


def test_the_seed_plan_is_computable_before_any_model_runs():
    """PLAN §2.6's exclusion law is provable by DRY RUN: if the seed table can be enumerated with
    no state, no endpoint, and no digest in hand, then no treatment can possibly have entered it."""
    arms = _arms(_pair_context(), 0)
    assert all(isinstance(arm.realized_propagation_seed, int) for arm in arms)
    assert all(arm.realized_lookahead_seeds for arm in arms)


def test_lookahead_seeds_are_disjoint_within_an_arm():
    arm = _arms(_pair_context(), 0)[0]
    assert len(set(arm.realized_lookahead_seeds)) == len(arm.realized_lookahead_seeds)


def test_the_propagation_seed_is_disjoint_from_the_lookahead_seeds():
    arm = _arms(_pair_context(), 0)[0]
    assert arm.realized_propagation_seed not in arm.realized_lookahead_seeds


# --------------------------------------------------------------------------------------------
# source interventions
# --------------------------------------------------------------------------------------------


def test_ablation_changes_the_source_bytes():
    source = F.source()
    ablated = ablate_source(source, positions=(3,))
    assert ablated.tokens != source.tokens
    assert ablated.tokens[3] == source.mask_token_id


def test_ablation_preserves_the_hard_anchors():
    source = F.source()
    ablated = ablate_source(source, positions=(3,))
    assert ablated.hard_anchors == source.hard_anchors


def test_ablation_refuses_to_touch_an_anchor():
    """An ablation that removed a constraint would be a different experiment, not a control."""
    with pytest.raises(V2PairedError, match="anchor"):
        ablate_source(F.source(), positions=(0,))


def test_shuffling_permutes_the_editable_bytes_without_changing_the_multiset():
    source = F.source()
    shuffled = shuffle_source(source, seed=7)
    editable = list(source.editable_positions)
    assert sorted(shuffled.tokens[i] for i in editable) == \
        sorted(source.tokens[i] for i in editable)


def test_shuffling_preserves_the_hard_anchors():
    source = F.source()
    shuffled = shuffle_source(source, seed=7)
    assert shuffled.hard_anchors == source.hard_anchors
    for position, token in source.hard_anchors:
        assert shuffled.tokens[position] == token


def test_shuffling_is_deterministic_in_its_seed():
    assert shuffle_source(F.source(), seed=7).tokens == shuffle_source(F.source(), seed=7).tokens


def test_the_shuffle_seed_actually_selects_the_permutation():
    """Sweep seeds and require at least two distinct outcomes.

    Comparing one pair of seeds is not enough -- with a handful of movable positions two seeds can
    legitimately land on the same permutation, so a single-pair assertion has to be written with an
    escape clause, and an escape clause makes the test vacuous.  A sweep has no such problem."""
    outcomes = {shuffle_source(F.source(), seed=seed).tokens for seed in range(12)}
    assert len(outcomes) >= 2, "the shuffle ignores its seed: every seed gave one permutation"


def test_every_shuffle_outcome_preserves_the_multiset():
    source = F.source()
    editable = list(source.editable_positions)
    expected = sorted(source.tokens[i] for i in editable)
    for seed in range(12):
        shuffled = shuffle_source(source, seed=seed)
        assert sorted(shuffled.tokens[i] for i in editable) == expected


# --------------------------------------------------------------------------------------------
# THE POSITIVE CONTROL: a source-blind kernel must FAIL
# --------------------------------------------------------------------------------------------


def _perturbed_source_projection():
    """A projection built from a source whose CARRIED byte differs.  Changing only the fork seed
    would not do: the seed is not a compared field, so it would make the passing case vacuous."""
    source = F.source()
    tokens = list(source.tokens)
    tokens[3] = 19
    provenance = list(source.provenance_by_pos)
    provenance[3] = F.provenance(19, kind=st.ActiveOriginKind.DENOISER_SAMPLE,
                                 step=F.EARLY_COMMITS[3])
    return F.projected(source=F.source(tokens=tuple(tokens),
                                       provenance_by_pos=tuple(provenance)))


def test_a_source_dependent_projection_passes_the_check():
    """The control must have a passing case, or it proves nothing about the failing one."""
    verdict = source_dependence_verdict(baseline=F.projected(),
                                        perturbed=_perturbed_source_projection())
    assert verdict.depends_on_source


def test_a_source_blind_projection_fails_the_check():
    """PLAN V2F5 acceptance: "a deliberately source-blind kernel fails the deterministic
    source-dependence check".  Two projections that are byte-identical after the source was
    perturbed prove the kernel never read it."""
    baseline = F.projected()
    verdict = source_dependence_verdict(baseline=baseline, perturbed=baseline)
    assert not verdict.depends_on_source
    assert "identical" in verdict.detail


def test_the_verdict_names_what_actually_differed():
    verdict = source_dependence_verdict(baseline=F.projected(),
                                        perturbed=_perturbed_source_projection())
    assert "tokens" in verdict.differing_fields


def test_an_endpoint_blind_projection_fails_the_endpoint_check():
    """The mirror control: if changing the selected endpoint changes nothing, the kernel ignored
    the endpoint and any 'feedback' it appears to transmit is an artefact."""
    baseline = F.projected()
    verdict = source_dependence_verdict(baseline=baseline, perturbed=baseline,
                                        subject="endpoint")
    assert not verdict.depends_on_source
    assert verdict.subject == "endpoint"


# --------------------------------------------------------------------------------------------
# the policy-level source-off comparator
# --------------------------------------------------------------------------------------------


def test_the_policy_source_off_comparator_lives_on_its_own_axis():
    """PLAN §2.5's source-off comparator is a POLICY-level control, not a mechanism view.  It has
    an interface today and its real gate waits for the FeedbackSupportPolicySpec freeze."""
    from inverse_folding.reference_flow.fusion_v2_runtime.paired import POLICY_VARIANTS

    assert "source_off" in POLICY_VARIANTS
    assert "source_off" not in INTERVENTIONS


def test_the_reward_ordered_endpoint_pair_lives_on_its_own_axis():
    """PLAN V2F5 asks for a reward-ordered endpoint pair, which is a claim about SELECTION rather
    than about the mechanism -- so it is a level of the endpoint-relation axis."""
    from inverse_folding.reference_flow.fusion_v2_runtime.paired import ENDPOINT_RELATIONS

    assert "reward_ordered" in ENDPOINT_RELATIONS
    assert "reward_ordered" not in INTERVENTIONS


def test_arms_declare_the_horizon_they_are_matched_on():
    arms = _arms(_pair_context(), 0)
    for arm in arms:
        assert arm.r_step == F.R_STEP
        assert arm.c_next_step == F.C_NEXT


# --------------------------------------------------------------------------------------------
# fail-closed
# --------------------------------------------------------------------------------------------


def test_two_arms_may_not_share_a_treatment_identity():
    """A control that silently equals its treatment reads as a null effect rather than as a broken
    design, so the V2F1 seed layer refuses the pair outright."""
    contrast = _contrasts()[0]
    same = dataclasses.replace(contrast.arm_b.seed_record,
                               treatment_identity=contrast.arm_a.treatment_identity)
    with pytest.raises(sd.MatchedPairSeedError):
        sd.assert_matched_pair_seeds([contrast.arm_a.seed_record, same])


def test_an_unknown_intervention_is_refused():
    with pytest.raises(V2PairedError, match="intervention"):
        _contrasts(interventions=("vibes_off",))


def test_a_negative_fork_index_is_refused():
    with pytest.raises(V2PairedError):
        _contrasts(fork_index=-1)
