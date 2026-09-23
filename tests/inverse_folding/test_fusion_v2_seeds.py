"""V2F1 steps 7-8: ``fusion_v2.seeds``.

Ordinary V2 namespaces plus the PLAN §2.6 matched-pair law. The load-bearing property is that a
matched descendant seed is derivable *before either arm of a pair exists*, so nothing that differs
between the two arms can reach it: if a treatment digest could change its own matched randomness,
every paired control in V2F5 silently compares two different random streams instead of two
different interventions, and nothing fails.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from inverse_folding.reference_flow.fusion_v2 import seeds as sd
from inverse_folding.reference_flow.fusion_v2.errors import V2Error


def _ctx(**over):
    kw = dict(seed_schema="v2seed-1", campaign_id="v2_mech_dev", split_role="v2_dev",
              master_seed=20260804, protein_id="5ZHV_B")
    kw.update(over)
    return sd.V2SeedContext(**kw)


def _pair(**over):
    kw = dict(seed_schema="v2seed-1", campaign_id="v2_mech_dev", split_role="v2_dev",
              master_seed=20260804, protein_id="5ZHV_B", depth=0, r_step=40, c_next_step=60,
              pair_ordinal=0, n_forks=4, n_descendant_lookaheads=2)
    kw.update(over)
    return sd.FeedbackPairSeedContext(**kw)


def _record(ctx=None, *, fork_index=0, arm_slot="arm_a", treatment="endpoint:A", **over):
    ctx = ctx or _pair()
    kw = dict(
        context=ctx, fork_index=fork_index, arm_slot=arm_slot,
        intervention_kind="endpoint_change", treatment_identity=treatment,
        shared_match_identity="support:abc|horizon:40-60", c_source_step=50,
        realized_propagation_seed=ctx.matched_descendant_seed(fork_index),
        realized_lookahead_seeds=tuple(
            ctx.matched_descendant_lookahead_seed(fork_index, j)
            for j in range(ctx.n_descendant_lookaheads)
        ),
    )
    kw.update(over)
    return sd.MatchedPairSeedRecord(**kw)


# --------------------------------------------------------------------------------------------
# Step 7 - ordinary namespaces
# --------------------------------------------------------------------------------------------

def test_seeds_are_process_independent():
    """A seed derived under a different PYTHONHASHSEED must be the same seed.

    Mirrors the V1 guarantee at ``test_reference_flow_fusion_v1_seeds.py``; without it a rerun on
    another node silently draws a different trajectory while claiming the same identity.
    """
    code = (
        "from inverse_folding.reference_flow.fusion_v2 import seeds as sd;"
        "c = sd.V2SeedContext(seed_schema='v2seed-1', campaign_id='v2_mech_dev',"
        " split_role='v2_dev', master_seed=20260804, protein_id='5ZHV_B');"
        "print(c.depth0_root_seed(checkpoint_step=50, root_index=3),"
        " c.lookahead_seed(depth=0, source_state_id='s', lookahead_index=1))"
    )
    outs = set()
    for hashseed in ("0", "1", "12345"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True, env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"})
        outs.add(out.stdout.strip())
    assert len(outs) == 1, outs


def test_the_four_namespaces_are_disjoint_at_identical_coordinates():
    """``v2_lookahead`` and ``a2_extra_lookahead`` must never collide: an extra A2 future would
    otherwise repeat a shared pre-feedback draw (PLAN §4.3)."""
    ctx = _ctx()
    drawn = {
        "lookahead": ctx.lookahead_seed(depth=0, source_state_id="s0", lookahead_index=1),
        "a2_extra": ctx.a2_extra_lookahead_seed(depth=0, source_state_id="s0", extra_index=1),
        "root": ctx.depth0_root_seed(checkpoint_step=0, root_index=1),
        "matched": _pair().matched_descendant_seed(1),
    }
    assert len(set(drawn.values())) == len(drawn), drawn


def test_lookahead_seeds_are_bound_to_both_depth_and_source_state():
    """``stationary_checkpoint`` repeats a sampler step across depths, and two active-population
    members share a depth, so neither alone identifies a stream (PLAN §2.1, §4.4)."""
    ctx = _ctx()
    base = ctx.lookahead_seed(depth=0, source_state_id="s0", lookahead_index=0)
    assert base != ctx.lookahead_seed(depth=1, source_state_id="s0", lookahead_index=0)
    assert base != ctx.lookahead_seed(depth=0, source_state_id="s1", lookahead_index=0)
    assert base != ctx.lookahead_seed(depth=0, source_state_id="s0", lookahead_index=1)


def test_v2_encoding_version_differs_from_v1():
    from inverse_folding.reference_flow.fusion.v1_seeds import SEED_ENCODING_VERSION

    assert sd.V2_SEED_ENCODING_VERSION != SEED_ENCODING_VERSION


def test_a2_and_v2_cannot_draw_different_pre_feedback_lookaheads():
    """PLAN §2.3/§4.3: A2 and V2 may not regenerate separate initial roots or lookaheads.

    Structural, not conventional: ``V2SeedContext`` has no ``arm`` or ``view`` field to differ on.
    """
    assert not any(f in ("arm", "view", "arm_role") for f in sd.V2SeedContext.__dataclass_fields__)


@pytest.mark.parametrize("bad", [{"master_seed": 1.0}, {"master_seed": True}, {"protein_id": ""},
                                {"seed_schema": ""}, {"campaign_id": ""}, {"split_role": ""}])
def test_seed_context_fails_closed(bad):
    with pytest.raises(V2Error):
        _ctx(**bad)


# --------------------------------------------------------------------------------------------
# Step 8 - the matched-pair law
# --------------------------------------------------------------------------------------------

def test_the_pair_seed_table_is_computable_before_either_arm_exists():
    """The executable proof of PLAN §2.6's "frozen before either arm exists".

    Every input is declared config plus a schedule coordinate plus an ordinal; nothing is knowable
    only after an arm is materialized. So the dry-run table must equal the realized table.
    """
    ctx = _pair()
    dry_run = ctx.enumerate_pair_seeds()
    assert len(dry_run) == ctx.n_forks * (1 + ctx.n_descendant_lookaheads)
    realized = {}
    for fork in range(ctx.n_forks):
        realized[f"fork{fork}:propagation"] = ctx.matched_descendant_seed(fork)
        for j in range(ctx.n_descendant_lookaheads):
            realized[f"fork{fork}:lookahead{j}"] = ctx.matched_descendant_lookahead_seed(fork, j)
    assert dry_run == realized


def test_the_paired_api_offers_no_channel_for_treatment_content():
    """PLAN §2.6 exclusion list, made structural: the only per-call arguments are bounded ints.

    A digest has no channel because there is no ``str``/``bytes``/``Mapping`` parameter anywhere in
    the paired derivation path, and no field on the context can carry an arm or intervention label.
    """
    banned = {"arm", "arm_slot", "intervention_kind", "treatment_identity", "endpoint_id",
              "endpoint_digest", "source_state_id", "source_digest", "projected_state_digest",
              "transition_id", "policy_digest", "policy_config_digest", "config_digest", "phase"}
    assert banned.isdisjoint(sd.FeedbackPairSeedContext.__dataclass_fields__)
    with pytest.raises(TypeError):
        _pair().matched_descendant_seed(0, endpoint_digest="deadbeef")


def test_both_arms_of_a_pair_draw_identical_seeds():
    ctx = _pair()
    a = _record(ctx, arm_slot="arm_a", treatment="endpoint:A")
    b = _record(ctx, arm_slot="arm_b", treatment="endpoint:B")
    assert a.realized_propagation_seed == b.realized_propagation_seed
    assert a.realized_lookahead_seeds == b.realized_lookahead_seeds
    sd.assert_matched_pair_seeds([a, b])


def test_a_persisted_seed_that_is_not_the_derived_seed_cannot_be_written():
    """§7.2: 'a treatment endpoint/source/state digest changes a matched descendant seed'.

    Re-derivation happens at record construction, from the context alone -- so an implementation
    that folded a treatment field into the seed fails on every real run, not only under test.
    """
    ctx = _pair()
    with pytest.raises(sd.MatchedPairSeedError):
        _record(ctx, realized_propagation_seed=ctx.matched_descendant_seed(0) ^ 1)
    with pytest.raises(sd.MatchedPairSeedError):
        _record(ctx, realized_lookahead_seeds=(ctx.matched_descendant_lookahead_seed(0, 0) ^ 1,))


def test_pairing_law_rejects_a_control_that_equals_its_treatment():
    """Two arms with the same treatment identity are not a paired intervention."""
    ctx = _pair()
    same = [_record(ctx, arm_slot="arm_a", treatment="endpoint:A"),
            _record(ctx, arm_slot="arm_b", treatment="endpoint:A")]
    with pytest.raises(sd.MatchedPairSeedError):
        sd.assert_matched_pair_seeds(same)


@pytest.mark.parametrize(
    "over", [{"intervention_kind": "source_change"}, {"shared_match_identity": "other"},
             {"c_source_step": 51}],
)
def test_pairing_law_rejects_an_unmatched_shared_field(over):
    """Everything held fixed must actually be equal across the pair (PLAN §2.6, §8.4)."""
    ctx = _pair()
    pair = [_record(ctx, arm_slot="arm_a", treatment="endpoint:A"),
            _record(ctx, arm_slot="arm_b", treatment="endpoint:B", **over)]
    with pytest.raises(sd.MatchedPairSeedError):
        sd.assert_matched_pair_seeds(pair)


def test_pairing_law_requires_exactly_two_distinct_arm_slots():
    ctx = _pair()
    with pytest.raises(sd.MatchedPairSeedError):
        sd.assert_matched_pair_seeds([_record(ctx, arm_slot="arm_a", treatment="endpoint:A")])
    with pytest.raises(sd.MatchedPairSeedError):
        sd.assert_matched_pair_seeds([
            _record(ctx, arm_slot="arm_a", treatment="endpoint:A"),
            _record(ctx, arm_slot="arm_a", treatment="endpoint:B"),
        ])


def test_pair_id_is_derived_not_accepted():
    """PLAN §2.6: the pair ID is frozen before either arm exists, so it cannot be an input."""
    assert "pair_id" not in sd.FeedbackPairSeedContext.__dataclass_fields__
    ctx = _pair()
    assert ctx.pair_id == sd.derive_pair_id(
        protein_id="5ZHV_B", depth=0, r_step=40, c_next_step=60, pair_ordinal=0
    )
    assert _pair(pair_ordinal=1).pair_id != ctx.pair_id


@pytest.mark.parametrize(
    "over",
    [{"n_forks": 0}, {"n_descendant_lookaheads": 0}, {"r_step": 60}, {"r_step": 61},
     {"depth": -1}, {"pair_ordinal": -1}, {"master_seed": 1.5}],
)
def test_pair_context_fails_closed(over):
    with pytest.raises(V2Error):
        _pair(**over)


@pytest.mark.parametrize("bad_fork", [-1, 4, 99])
def test_fork_index_is_bounded_by_the_frozen_fork_count(bad_fork):
    """The bound is also the anti-smuggling guard: a content digest coerced to an int cannot land
    inside a small declared range."""
    with pytest.raises(V2Error):
        _pair().matched_descendant_seed(bad_fork)


def test_realized_manifest_collapses_pairs_and_fails_closed_on_collision():
    """A naive per-arm map would false-positive on the intended within-pair equality."""
    ctx, pair_ctx = _ctx(), _pair()
    manifest = sd.realized_v2_seed_manifest(
        ctx,
        depth0_root_draws=((50, 0), (50, 1)),
        lookahead_draws=((0, "s0", 0), (0, "s0", 1)),
        a2_extra_draws=((0, "s0", 0),),
        pair_records=(_record(pair_ctx, arm_slot="arm_a", treatment="endpoint:A"),
                      _record(pair_ctx, arm_slot="arm_b", treatment="endpoint:B")),
    )
    assert len(manifest) == 2 + 2 + 1 + (1 + pair_ctx.n_descendant_lookaheads)
    assert len(set(manifest.values())) == len(manifest)
    with pytest.raises(V2Error):
        sd.realized_v2_seed_manifest(ctx, depth0_root_draws=((50, 0), (50, 0)),
                                     lookahead_draws=(), a2_extra_draws=())
