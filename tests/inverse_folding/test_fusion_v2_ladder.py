"""V2F6: the general D>=1 depth ladder, the stationary comparator, and typed stopping.

PLAN V2F6 generalizes the verified one-cycle transition to depth ``D >= 1`` "without changing its
semantics", and adds one compact stationary comparator "through the same segment API".

Two things this suite is careful about, because both are ways the generalization could quietly
become a different mechanism:

**Both laws must call the SAME executor.**  A progressive ladder and a stationary comparator that
ran through different code would not be comparable -- the comparator would measure the second
implementation, not the schedule.  So the ladder is one loop parameterized by a coordinate law.

**Depth must not buy multiplicity.**  PLAN §4.4: "No family receives extra ancestry mass solely
because it emitted more identical siblings."  A ladder that let a family fan out on duplicates
would report depth as improvement when it was really just sampling more.

Breadth and depth are DECLARED, never allocated.  PLAN's revision separates implementing the
recursion from authorizing real ``D>1`` runs, so there is no cost model here and no automatic
allocator: the schedule states how many lookaheads each depth gets, and the engine obeys it.
Production ``D>1`` ships behind an explicit launch flag that defaults to off.
"""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from inverse_folding.reference_flow.fusion_v2 import state as st
from inverse_folding.reference_flow.fusion_v2.schedule import CoordinateLaw
from inverse_folding.reference_flow.fusion_v2_runtime.ladder import (
    DepthPlan,
    LadderOutcome,
    ProductionDepthDisabledError,
    StoppingReason,
    V2LadderError,
    run_depth_ladder,
)
from inverse_folding.reference_flow.fusion_v2_runtime.capture import FullyResolvedRootError
from inverse_folding.reference_flow.sampler import PositionDependentDFMSampler
from tests.inverse_folding import _v2_fixtures as F
from tests.inverse_folding.test_fusion_v2_cycle import (
    ALPHABET,
    VOCAB,
    _FakeHeadOracle,
    _cfg,
    _denoiser,
    _structure_oracle,
    _module_meter,
    _support_policy,
    _wide_band_table,
)

#: A LONGER protein than the shared fixture's length 6.
#:
#: The ladder is the only V2 suite that must keep a state live across two full cycles, and it needs
#: headroom at BOTH ends of every rung: enough already-resolved history before each ``r_d`` for the
#: probe policy to have something to carry, and enough still-unresolved mass at ``c_2`` for the
#: deepest capture to remain a live partial state.  With 5 editable positions the linear schedule
#: leaves about 2 unresolved at step 62, so whether depth 2 exists at all comes down to the draw --
#: a suite sitting there passes or fails on the seed, not on the ladder, and would silently stop
#: exercising depth 2 the day the seed derivation changed.  23 editable positions leave ~9.
LADDER_L = 24

#: Coordinates chosen so every captured state stays pre-terminal.
C0, R1, C1, R2, C2 = 50, 40, 62, 52, 72
#: The stationary comparator re-enters twice and returns to the SAME checkpoint both times.
R_STAT_A, R_STAT_B, C_STAT = 40, 45, 50


def _ladder_band_table():
    """A permissive band at EVERY re-entry step the ladder uses.

    ``lookup_band`` is exact and refuses to interpolate, so a ladder whose depths re-enter at
    different steps needs a calibrated row for each of them.  Band calibration itself is tested in
    ``test_fusion_v2_schedule``; narrowing it here would fail this suite for an unrelated reason."""
    import inverse_folding.reference_flow.fusion_v2.schedule as sch

    bands = tuple(
        sch.make_band(
            step=step, stratum_key=F.STRATUM, levels=sch.QuantileLevels(levels=(0.1, 0.5, 0.9)),
            rho_quantiles=(0.05, 0.5, 0.95), unresolved_quantiles=(LADDER_L, 2, 1),
            rho_accept=sch.BandInterval(lo=0.0, hi=1.0, lo_level=0.1, hi_level=0.9),
            unresolved_accept=sch.BandInterval(lo=1.0, hi=float(LADDER_L), lo_level=0.9,
                                              hi_level=0.1),
            combination_rule="both_axes", n_attempts=32, n_captured=32,
            n_editable_min=1, n_editable_max=LADDER_L,
        )
        for step in sorted({R1, R2, R_STAT_A, R_STAT_B})   # R1 and R_STAT_A coincide
    )
    return sch.make_band_table(provenance=F.band_provenance(), bands=bands)


def _progressive_plan(**over):
    kw = dict(
        law=CoordinateLaw.PROGRESSIVE_CHECKPOINT,
        depth_cap=2,
        cycles=((R1, C0, C1), (R2, C1, C2)),
        lookaheads_per_depth=(3, 2),
        n_steps=F.N_STEPS,
    )
    kw.update(over)
    return DepthPlan(**kw)


def _stationary_plan(**over):
    """The predeclared stationary D=2 schedule.

    ``stationary_checkpoint`` means the checkpoint does not move AT ALL: every depth re-enters at
    its own ``r_d`` and returns to the same ``c``.  It is the compact comparator, not a schedule
    that merely happens to repeat a step at the end."""
    kw = dict(
        law=CoordinateLaw.STATIONARY_CHECKPOINT,
        depth_cap=2,
        cycles=((R_STAT_A, C_STAT, C_STAT), (R_STAT_B, C_STAT, C_STAT)),
        lookaheads_per_depth=(3, 2),
        n_steps=F.N_STEPS,
    )
    kw.update(over)
    return DepthPlan(**kw)


def _run_kwargs(**over):
    table = _ladder_band_table()
    kw = dict(
        plan=_progressive_plan(),
        sampler=PositionDependentDFMSampler(mask_token_id=F.MASK, vocab_size=VOCAB),
        denoiser=_denoiser, config=_cfg(), sequence_length=LADDER_L,
        h_values=np.zeros(LADDER_L, dtype=np.float32), residue_token_ids=F.AA,
        alphabet=ALPHABET, fixed_tokens={0: 10}, lineage=F.lineage(),
        mask_token_id=F.MASK, aa_token_ids=F.AA, conditioning=F.conditioning(
            # The kernel refuses a band table the run's own frozen provenance does not
            # name, so the fixture binds the two instead of leaving them unrelated.
            schedule_band_calibration=table.provenance.calibration_content_digest),
        safety_reference=F.safety_reference(length=LADDER_L),
        head_oracle=_FakeHeadOracle(length=LADDER_L), structure_oracle=_structure_oracle,
        support_policy=_support_policy,
        band_table=table, stratum_key=F.STRATUM, declared_band_id=F.BAND_ID,
        declared_band_digest=table.provenance.calibration_content_digest,
        safety_gate=F.safety_gate(length=LADDER_L),
        declared_policy=F.declared_policy(),
        feedback_enabled=True,
        cost_meter=_module_meter(),
        campaign_id="v2-canary", split_role="dev", master_seed=20260805,
        allow_production_depth_gt_1=True,
    )
    kw.update(over)
    return kw


def _run(**over):
    return run_depth_ladder(**_run_kwargs(**over))


# --------------------------------------------------------------------------------------------
# the plan is declared, not allocated
# --------------------------------------------------------------------------------------------


def test_breadth_is_declared_per_depth():
    """PLAN's revision: no automatic allocator.  The schedule states the breadth and the engine
    obeys it, so nothing here depends on a cost model that does not exist yet."""
    plan = _progressive_plan(lookaheads_per_depth=(4, 1))
    assert plan.lookaheads_at(0) == 4
    assert plan.lookaheads_at(1) == 1


def test_the_breadth_vector_must_cover_every_depth():
    with pytest.raises(V2LadderError, match="lookaheads_per_depth|depth"):
        _progressive_plan(lookaheads_per_depth=(3,))


def test_a_progressive_schedule_must_strictly_advance():
    """PLAN V2F6: "progressive checkpoints strictly advance"."""
    from inverse_folding.reference_flow.fusion_v2.errors import V2Error

    # DepthPlan delegates ordering to schedule.make_cycle, so the precise error comes from there;
    # duplicating the rule here is how the ladder would drift from the rest of V2.
    with pytest.raises(V2Error, match="advance|progressive|c_next"):
        _progressive_plan(cycles=((R1, C0, C1), (R2, C1, C1)))


def test_a_stationary_schedule_holds_the_checkpoint_still():
    plan = _stationary_plan()
    assert all(c_next == c_source for _, c_source, c_next in plan.cycles)


def test_the_cycles_must_chain():
    """Depth d's next checkpoint is depth d+1's source; a gap would silently skip a segment."""
    with pytest.raises(V2LadderError, match="chain|source"):
        _progressive_plan(cycles=((R1, C0, C1), (R2, C1 + 1, C2)))


def test_a_schedule_that_exhausts_the_horizon_is_refused():
    """PLAN V2F6: "reject exhausted horizons"."""
    from inverse_folding.reference_flow.fusion_v2.errors import V2Error

    with pytest.raises(V2Error, match="n_steps"):
        _progressive_plan(cycles=((R1, C0, C1), (R2, C1, F.N_STEPS)))


def test_the_depth_cap_must_match_the_declared_cycles():
    with pytest.raises(V2LadderError, match="depth_cap"):
        _progressive_plan(depth_cap=3)


# --------------------------------------------------------------------------------------------
# production D>1 is launch-disabled
# --------------------------------------------------------------------------------------------


def test_production_depth_greater_than_one_is_disabled_by_default():
    """PLAN V2F6: production ``D>1`` stays launch-disabled until the real one-cycle transmission
    gate and the policy directionality gate pass.  The capability exists; the authorization does
    not, and the default must be the safe one."""
    with pytest.raises(ProductionDepthDisabledError, match="D>1|launch"):
        _run(allow_production_depth_gt_1=False)


def test_depth_one_runs_without_the_flag():
    """The gate is on D>1, not on the mechanism: a single cycle needs no authorization."""
    outcome = _run(plan=_progressive_plan(depth_cap=1, cycles=((R1, C0, C1),),
                                          lookaheads_per_depth=(3,)),
                   allow_production_depth_gt_1=False)
    assert isinstance(outcome, LadderOutcome)


def test_the_flag_is_recorded_on_the_outcome():
    """A run must say whether it was authorized, or a later reader cannot tell a capability test
    from a production result."""
    assert _run().production_depth_authorized is True


def test_exploratory_depth_override_is_narrow_and_separately_recorded():
    """An exploratory launch is not evidence that the production gate passed."""
    outcome = _run(
        allow_production_depth_gt_1=False,
        exploratory_depth_override=True,
        phase="capability_ladder",
        split_role="exploratory_deep_smoke",
    )
    assert outcome.production_depth_authorized is False
    assert outcome.exploratory_depth_override is True


@pytest.mark.parametrize(
    ("phase", "split_role"),
    [
        ("policy_qualification", "exploratory_deep_smoke"),
        ("capability_ladder", "dev"),
    ],
)
def test_exploratory_depth_override_refuses_non_exploratory_identity(phase, split_role):
    with pytest.raises(ProductionDepthDisabledError, match="exploratory|capability_ladder"):
        _run(
            allow_production_depth_gt_1=False,
            exploratory_depth_override=True,
            phase=phase,
            split_role=split_role,
        )


# --------------------------------------------------------------------------------------------
# the ladder runs and both laws share one executor
# --------------------------------------------------------------------------------------------


def test_the_progressive_ladder_reaches_the_declared_depth():
    outcome = _run()
    assert outcome.depth_reached == 2
    assert len(outcome.cycles) == 2


def test_the_next_rung_uses_the_advanced_policy_identity():
    """A committed donor must become the reference encoded by the next policy answer.

    This is intentionally a ladder wiring test rather than a second reward-law test: the exact
    Head gate is covered in ``test_fusion_v2_head_directed_policy``.  Here a tiny policy exposes
    only an immutable generation in its identity and advances after a passing typed verdict.  If
    the ladder kept reusing its construction-time policy, both feedback events would carry the
    same digest and this test would fail.
    """
    calls = []

    class AdvancingProbe:
        def __init__(self, generation=0):
            self.generation = generation

        def __call__(self, source, endpoint, coordinates):
            decision = _support_policy(source, endpoint, coordinates)
            identity = dataclasses.replace(
                decision.policy,
                policy_config_digest=F.digest(f"incumbent-generation-{self.generation}"),
            )
            evidence = types.SimpleNamespace(donor_gate=f"pass-{self.generation}")
            return dataclasses.replace(
                decision, policy=identity, decision_evidence=evidence)

        def advance_lineage_incumbent(self, *, donor, verdict, accepted_at_depth):
            calls.append((self.generation, donor.endpoint_id, verdict, accepted_at_depth))
            return AdvancingProbe(self.generation + 1)

    outcome = _run(support_policy=AdvancingProbe())
    assert [record.cycle.policy_identity.policy_config_digest for record in outcome.cycles] == [
        F.digest("incumbent-generation-0"),
        F.digest("incumbent-generation-1"),
    ]
    assert calls[0][0] == 0
    assert calls[0][2:] == ("pass-0", 1)


def test_depth0_generated_rank_bootstraps_the_ladder_without_a_wt_donor_verdict():
    """Policy v2 intentionally has no global WT comparison at D0.

    The cycle proves the selected endpoint is rank zero and records the bootstrap gate kind.  The
    ladder must pass ``verdict=None`` to the policy's own guarded advancement exactly at depth zero,
    then require ordinary donor verdicts again at later depths.
    """
    from inverse_folding.reference_flow.fusion_v2.reward import DEPTH0_BOOTSTRAP_RULE

    calls = []

    class BootstrapProbe:
        depth0_incumbent_rule = DEPTH0_BOOTSTRAP_RULE

        def __init__(self, generation=0):
            self.generation = generation

        def __call__(self, source, endpoint, coordinates):
            decision = _support_policy(source, endpoint, coordinates)
            verdict = None if self.generation == 0 else f"pass-{self.generation}"
            evidence = types.SimpleNamespace(
                donor_gate=verdict,
                reward_gate_kind=(DEPTH0_BOOTSTRAP_RULE
                                  if self.generation == 0 else "strict_improvement"),
            )
            return dataclasses.replace(decision, decision_evidence=evidence)

        def advance_lineage_incumbent(self, *, donor, verdict, accepted_at_depth):
            calls.append((self.generation, verdict, accepted_at_depth, donor.endpoint_id))
            return BootstrapProbe(self.generation + 1)

    outcome = _run(support_policy=BootstrapProbe())

    assert outcome.depth_reached == 2
    assert calls[0][0:3] == (0, None, 1)
    assert calls[1][0:3] == (1, "pass-1", 2)


def test_the_stationary_ladder_reaches_the_declared_depth():
    outcome = _run(plan=_stationary_plan())
    assert outcome.depth_reached == 2


def test_both_laws_run_through_the_same_segment_executor():
    """PLAN V2F6: "both laws call the same segment executor".  Two implementations would make the
    comparator measure the second implementation rather than the schedule."""
    progressive = _run()
    stationary = _run(plan=_stationary_plan())
    assert progressive.segment_executor_id == stationary.segment_executor_id


def test_depth_one_of_the_ladder_equals_the_standalone_one_cycle_runner():
    """PLAN V2F6: "depth-1 equals the V2F5 runner".  The generalization must not change the
    semantics of the transition it generalizes, so a one-rung ladder and a direct one-cycle call on
    the SAME coordinates, seeds and band table must produce the same states byte for byte."""
    from inverse_folding.reference_flow.fusion_v2_runtime.cycle import run_one_cycle

    plan = _progressive_plan(depth_cap=1, cycles=((R1, C0, C1),), lookaheads_per_depth=(3,))
    ladder = _run(plan=plan)
    record = ladder.cycles[0]

    table = _ladder_band_table()
    single = run_one_cycle(
        sampler=PositionDependentDFMSampler(mask_token_id=F.MASK, vocab_size=VOCAB),
        denoiser=_denoiser, config=_cfg(), sequence_length=LADDER_L,
        h_values=np.zeros(LADDER_L, dtype=np.float32), residue_token_ids=F.AA,
        alphabet=ALPHABET, fixed_tokens={0: 10}, lineage=F.lineage(),
        mask_token_id=F.MASK, aa_token_ids=F.AA, conditioning=F.conditioning(
            # The kernel refuses a band table the run's own frozen provenance does not
            # name, so the fixture binds the two instead of leaving them unrelated.
            schedule_band_calibration=table.provenance.calibration_content_digest),
        safety_reference=F.safety_reference(length=LADDER_L),
        c_source_step=C0, r_step=R1, c_next_step=C1,
        source_fork_seeds=record.source_fork_seeds,
        descendant_fork_seeds=record.descendant_fork_seeds,
        descendant_propagation_seed=record.propagation_seed,
        head_oracle=_FakeHeadOracle(length=LADDER_L), structure_oracle=_structure_oracle,
        support_policy=_support_policy,
        band_table=table, stratum_key=F.STRATUM, declared_band_id=F.BAND_ID,
        declared_band_digest=table.provenance.calibration_content_digest,
        origin_transition_id=record.cycle.projected.origin_transition_id,
        safety_gate=F.safety_gate(length=LADDER_L),
        declared_policy=F.declared_policy(),
        feedback_enabled=True,
        cost_meter=_module_meter(),
    )
    assert single.outcome is record.cycle.outcome
    assert single.source.state_id == record.cycle.source.state_id
    assert single.propagated.content_digest == record.cycle.propagated.content_digest


# --------------------------------------------------------------------------------------------
# lineage: acyclic, and identities never alias
# --------------------------------------------------------------------------------------------


def test_the_depth_two_lineage_is_a_chain():
    outcome = _run()
    first, second = outcome.cycles
    assert second.cycle.source.state_id == first.cycle.propagated.state_id
    assert second.cycle.propagated.lineage.depth == 2


def test_the_lineage_is_acyclic():
    """Every PROPAGATED state is new.

    A rung's source is the previous rung's propagated capture -- that shared node is the chain, not
    a cycle -- so acyclicity is about the states the ladder CREATES, and asserting over sources too
    would flag the chain itself."""
    outcome = _run()
    created = [r.cycle.propagated.state_id for r in outcome.cycles
               if r.cycle.propagated is not None]
    assert len(set(created)) == len(created)
    for earlier, later in zip(outcome.cycles, outcome.cycles[1:]):
        assert later.cycle.source.state_id == earlier.cycle.propagated.state_id


def test_a_stationary_ladder_reuses_a_sampler_step_without_reusing_an_identity():
    """PLAN V2F6: "stationary depth-2 may reuse a sampler step without reusing a state/event
    identity".  HistoryKey(depth, step) is what keeps two events at one step distinct."""
    outcome = _run(plan=_stationary_plan())
    first, second = outcome.cycles
    assert first.cycle.propagated.sampler_step == second.cycle.propagated.sampler_step
    assert first.cycle.propagated.state_id != second.cycle.propagated.state_id


def test_every_transition_has_its_own_identity():
    outcome = _run()
    ids = [r.cycle.projected.origin_transition_id for r in outcome.cycles
           if r.cycle.projected is not None]
    assert len(set(ids)) == len(ids)


# --------------------------------------------------------------------------------------------
# the substrate holds at every depth
# --------------------------------------------------------------------------------------------


def test_every_propagation_segment_has_zero_background_remask():
    """PLAN V2F6 and §7.2: not one background-remask event, at any depth."""
    for record in _run().cycles:
        if record.cycle.segment is not None:
            assert record.cycle.segment.background_remask_events == 0


def test_unresolved_mass_only_rises_at_an_explicit_projection():
    for record in _run().cycles:
        cycle = record.cycle
        if cycle.projected is None or cycle.propagated is None:
            continue
        editable = cycle.projected.editable_positions
        after_projection = sum(1 for i in editable
                               if cycle.projected.tokens[i] == F.MASK)
        after_segment = sum(1 for i in editable
                            if cycle.propagated.tokens[i] == F.MASK)
        assert after_segment <= after_projection


def test_every_depth_uses_the_frozen_no_remask_substrate():
    """PLAN V2F6: "Every causal arm uses the frozen no-remask substrate"."""
    outcome = _run()
    assert outcome.substrate_digest
    assert all(r.substrate_digest == outcome.substrate_digest for r in outcome.cycles)


# --------------------------------------------------------------------------------------------
# seeds across depths
# --------------------------------------------------------------------------------------------


def test_seeds_are_disjoint_across_depths():
    """A depth that reused a shallower depth's seed would regenerate an endpoint the archive
    already holds, so the extra depth would measure nothing."""
    outcome = _run()
    used = []
    for record in outcome.cycles:
        used.extend(record.source_fork_seeds)
        used.extend(record.descendant_fork_seeds)
        used.append(record.propagation_seed)
    assert len(set(used)) == len(used)


def test_the_ladder_is_deterministic():
    a, b = _run(), _run()
    assert [r.cycle.propagated.content_digest for r in a.cycles] == \
        [r.cycle.propagated.content_digest for r in b.cycles]


def test_a_different_master_seed_changes_the_ladder():
    a = _run(master_seed=20260805)
    b = _run(master_seed=20260806)
    assert a.cycles[0].propagation_seed != b.cycles[0].propagation_seed


# --------------------------------------------------------------------------------------------
# archive carryover and no multiplicity purchase
# --------------------------------------------------------------------------------------------


def test_one_archive_carries_across_every_depth():
    """PLAN §2.3: the archive is the run's memory; a per-depth archive would lose the elite."""
    outcome = _run()
    assert outcome.archive is not None
    ids = {row.endpoint_id for row in outcome.archive.raw_rows()}
    for record in outcome.cycles:
        assert {e.endpoint_id for e in record.cycle.endpoints} <= ids


def test_the_elite_is_always_a_definitively_feasible_endpoint():
    outcome = _run()
    if outcome.best_definitive is not None:
        assert outcome.best_definitive.feasibility_level is st.FeasibilityLevel.DEFINITIVE


def test_depth_does_not_buy_ancestry_on_duplicate_siblings():
    """PLAN §4.4: "No family receives extra ancestry mass solely because it emitted more identical
    siblings."  A ladder that fanned out on duplicates would report depth as improvement when it
    was really just sampling more."""
    outcome = _run()
    for record in outcome.cycles:
        if record.cycle.selected_endpoint is None:
            continue
        chosen = record.cycle.selected_endpoint.sequence_md5
        siblings = [e for e in record.cycle.endpoints if e.sequence_md5 == chosen]
        assert record.n_selected_lineages == 1, (
            f"{len(siblings)} identical siblings bought {record.n_selected_lineages} lineages")


# --------------------------------------------------------------------------------------------
# typed stopping
# --------------------------------------------------------------------------------------------


def test_reaching_the_declared_depth_stops_on_the_depth_cap():
    assert _run().stopping_reason is StoppingReason.DEPTH_CAP


def test_every_stopping_reason_is_typed():
    """PLAN §4.5 names them; an untyped stop cannot be aggregated across a cohort."""
    assert {"DEPTH_CAP", "NO_ADMISSIBLE_ENDPOINT", "INVALID_PROJECTION", "NO_NOVEL_DESCENDANT",
            "NO_PRETERMINAL_ROOT", "TERMINAL_BEST_LOOKAHEAD", "FRONTIER_PLATEAU",
            "DIVERSITY_COLLAPSE", "OPERATIONAL_CEILING"} <= \
        {member.name for member in StoppingReason}


def test_a_run_where_nothing_folds_stops_on_no_admissible_endpoint():
    outcome = _run(structure_oracle=lambda request: __import__(
        "inverse_folding.reference_flow.fusion.v1_admission", fromlist=["StructureOutcome"]
    ).StructureOutcome(feasible=False, failure_reason="scTM 0.2"))
    assert outcome.stopping_reason is StoppingReason.NO_ADMISSIBLE_ENDPOINT
    assert outcome.depth_reached == 0


def test_a_stop_preserves_the_best_existing_archive_state():
    """PLAN §4.5: "Any stop returns the best existing definitive archive state; it never destroys
    or replaces it with a worse descendant"."""
    outcome = _run()
    assert outcome.best_definitive is not None
    rows = {row.endpoint_id for row in outcome.archive.raw_rows()}
    assert outcome.best_definitive.endpoint_id in rows


def test_a_null_branch_does_not_empty_the_archive():
    """PLAN §4.5: any stop returns the best existing definitive archive state.

    The refusal is ATTRIBUTED, because a refusal is a policy answer and PLAN §5.3 requires every
    feedback event -- null ones included -- to name the policy that produced it.  Leaving it
    anonymous would make this test depend on a hole in the kernel's identity gate rather than on
    the archive behaviour it is about.
    """
    from inverse_folding.reference_flow.fusion_v2.policy import PolicyRejection

    from tests.inverse_folding.test_fusion_v2_cycle import _policy_identity

    class DecliningPolicy:
        def __call__(self, source, endpoint, coordinates):
            del source, endpoint, coordinates
            return PolicyRejection(reason="declined", policy=_policy_identity())

        def advance_lineage_incumbent(self, **kwargs):
            del kwargs
            raise AssertionError("a null/stalled cycle must not advance the incumbent")

    outcome = _run(support_policy=DecliningPolicy())
    assert outcome.stopping_reason is StoppingReason.INVALID_PROJECTION
    assert outcome.archive.raw_rows()


# --------------------------------------------------------------------------------------------
# cost ledger across the ladder
# --------------------------------------------------------------------------------------------


def test_the_ladder_reports_the_total_logical_dfe():
    """The total is the root prefix the ladder paid plus every rung's REALIZED cost.

    Screening-plus-segment is not that sum: it drops each rung's descendant pool, which the next
    rung's inherited screening used to stand in for -- leaving the deepest rung's pool uncharged.
    """
    outcome = _run()
    total = outcome.root_capture_logical_dfe + sum(
        r.cycle.cost.total_logical_dfe for r in outcome.cycles)
    assert outcome.total_logical_dfe == total


def test_a_fully_resolved_root_is_retried_with_a_new_seed():
    class RetryOnce(PositionDependentDFMSampler):
        def __init__(self):
            super().__init__(mask_token_id=F.MASK, vocab_size=VOCAB)
            self.seeds = []

        def sample(self, **kwargs):
            if kwargs.get("continuation") is not None:
                self.seeds.append(int(kwargs["config"].sampler.seed))
            if kwargs.get("continuation") is not None and len(self.seeds) == 1:
                raise FullyResolvedRootError("rho_edit == 1.0")
            return super().sample(**kwargs)

    sampler = RetryOnce()
    outcome = _run(sampler=sampler, root_capture_max_retries=1)

    assert outcome.root_capture_attempts_used == 2
    assert outcome.root_capture_logical_dfe == 2 * C0
    assert outcome.root_capture_status == "captured"
    assert outcome.root_capture_seed == sampler.seeds[-1]
    assert outcome.root_capture_n_unresolved_editable > 0
    assert outcome.root_capture_rho_edit < 1.0
    assert sampler.seeds[0] != sampler.seeds[1]


@pytest.mark.parametrize("max_retries", [0, 2])
def test_exhausted_root_retries_return_one_small_typed_failure_record(max_retries):
    class AlwaysResolved:
        def sample(self, **kwargs):
            del kwargs
            raise FullyResolvedRootError("rho_edit == 1.0")

    outcome = _run(
        sampler=AlwaysResolved(), root_capture_max_retries=max_retries)

    assert outcome.stopping_reason is StoppingReason.NO_PRETERMINAL_ROOT
    assert outcome.root_capture_attempts_used == max_retries + 1
    assert outcome.root_capture_status == "fully_resolved"
    assert outcome.root_capture_detail == "rho_edit == 1.0"
    assert outcome.root_capture_n_unresolved_editable == 0
    assert outcome.root_capture_rho_edit == 1.0
    assert outcome.root_capture_seed is not None
    assert outcome.cycles == () and outcome.best_definitive is None


def test_the_source_prefix_is_charged_once_per_depth_not_once_per_lookahead():
    """PLAN §3.4: C_screen = c + K(S-c); charging the prefix K times would inflate every cost
    comparison the study makes."""
    outcome = _run()
    record = outcome.cycles[0]
    k = len(record.cycle.endpoints)
    # The ladder captures depth 0 itself, so the prefix sits on the LADDER's own field and the
    # rung carries only the tails it forked.  C_screen is the two of them together, once.
    assert record.cycle.cost.source_prefix_logical_dfe == 0
    assert outcome.root_capture_logical_dfe == C0
    assert record.cycle.cost.screening_logical_dfe == k * (F.N_STEPS - C0)
    assert outcome.root_capture_logical_dfe + record.cycle.cost.screening_logical_dfe == \
        C0 + k * (F.N_STEPS - C0)


# --------------------------------------------------------------------------------------------
# every ladder seed carries the run's declared identity (PLAN §2.6)
# --------------------------------------------------------------------------------------------


def _all_seeds(outcome):
    """Every seed the ladder actually realized, flattened."""
    seeds = []
    for record in outcome.cycles:
        seeds.extend(record.source_fork_seeds)
        seeds.extend(record.descendant_fork_seeds)
        seeds.append(record.propagation_seed)
    assert seeds, "the ladder must realize at least one seed for this test to mean anything"
    return seeds


@pytest.mark.parametrize("field,other", [
    ("campaign_id", "a-different-campaign"),
    ("split_role", "test"),
])
def test_every_ladder_seed_carries_the_run_identity(field, other):
    """PLAN §2.6: a seed is derived from campaign, split, master seed, protein and coordinates.

    A ladder that seeded off ``master_seed`` alone would hand two different campaigns the same
    random stream.  Two campaigns are then not independent replicates -- they are one replicate
    reported twice, and any variance estimate taken across them is an underestimate by construction.
    """
    base = _run()
    changed = _run(**{field: other})
    assert _all_seeds(base) != _all_seeds(changed), (
        f"changing {field} left every realized seed untouched"
    )


def test_ladder_seeds_are_recomputable_from_the_declared_context_alone():
    """The seeds must be a function of the declared identity, not of execution order.

    PLAN §2.6 makes the seed table computable before either arm exists; that property is what lets
    ``--dry-run`` publish the exact streams a run will use.  A seed drawn from a running counter
    cannot be predicted, so nothing can be checked against it after the fact.
    """
    from inverse_folding.reference_flow.fusion_v2.seeds import (
        V2_SEED_ENCODING_VERSION,
        FeedbackPairSeedContext,
    )

    plan = _progressive_plan()
    outcome = _run(plan=plan)
    r_step, _, c_next = plan.coordinates_at(0)
    context = FeedbackPairSeedContext(
        seed_schema=V2_SEED_ENCODING_VERSION, campaign_id="v2-canary", split_role="dev",
        master_seed=20260805, protein_id=F.lineage().protein_id, depth=0,
        r_step=r_step, c_next_step=c_next, pair_ordinal=0,
        n_forks=1, n_descendant_lookaheads=plan.lookaheads_at(min(1, plan.depth_cap - 1)),
    )
    assert outcome.cycles[0].propagation_seed == context.matched_descendant_seed(0)
    assert list(outcome.cycles[0].descendant_fork_seeds) == [
        context.matched_descendant_lookahead_seed(0, index)
        for index in range(len(outcome.cycles[0].descendant_fork_seeds))
    ]


# --------------------------------------------------------------------------------------------
# the safety ratchet advances with the ladder
# --------------------------------------------------------------------------------------------


def test_the_safety_ledger_advances_one_depth_per_committed_rung():
    """PLAN §2.7: the cumulative reference is frozen at depth 0 and the lineage ADVANCES.

    A ladder that never advanced the ledger would leave it at depth 0 forever.  Two things break
    then: the incremental gate can never be satisfied (its reference is the immediate parent, which
    a depth-0 ledger does not have), and the ledger stops being a record of which design the
    lineage actually descended through.
    """
    outcome = _run()
    assert outcome.depth_reached == 2
    assert outcome.final_safety_ledger is not None
    assert outcome.final_safety_ledger.depth == outcome.depth_reached
    parent = outcome.final_safety_ledger.immediate_parent
    assert parent is not None
    assert parent.endpoint_id == outcome.cycles[-1].cycle.selected_endpoint.endpoint_id


def test_the_cumulative_reference_is_the_same_object_at_every_depth():
    """The ratchet is inescapable only because the reference is carried by object identity.

    If a deeper rung could re-base it, three steps that each add a small parent-relative increase
    would each look safe while their SUM crossed the depth-0 threshold -- which is exactly the
    drift the cumulative gate exists to catch.
    """
    outcome = _run()
    gate = F.safety_gate(length=LADDER_L)
    assert outcome.final_safety_ledger.cumulative_reference is not gate.ledger.cumulative_reference
    # ...but within ONE run it never changes:
    assert outcome.final_safety_ledger.cumulative_reference.reference_binding_id == \
        gate.ledger.cumulative_reference.reference_binding_id


def test_a_ladder_that_stops_early_reports_the_ledger_it_actually_reached():
    """A stopped ladder must not claim a depth it never advanced through."""
    plan = _progressive_plan(depth_cap=1, cycles=((R1, C0, C1),), lookaheads_per_depth=(3,))
    outcome = _run(plan=plan)
    assert outcome.final_safety_ledger.depth == outcome.depth_reached


# --------------------------------------------------------------------------------------------
# the ladder's reported cost is its own execution graph
# --------------------------------------------------------------------------------------------


class _CountingDenoiser:
    """Counts every forward pass the ladder actually performs.

    One denoiser call is one lane-step, which is exactly the unit ``total_logical_dfe`` claims to
    report.  Counting them is the only way to tell a cost MODEL from a cost ACCOUNTING: an
    arithmetic projection agrees with itself no matter what the run did.
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, x_t, t, struct):
        self.calls += 1
        return _denoiser(x_t, t, struct)


def test_the_ladder_total_equals_the_forward_passes_it_actually_ran():
    """PLAN §3.4.  Two specific errors this pins down, both of which inflate or deflate the number
    a matched-compute claim is judged on:

    * every rung above depth 0 inherits its source, so charging its ``c_d`` prefix again adds a
      prefix per depth that no forward pass corresponds to;
    * the DEEPEST rung's descendant pool is screened and then never charged, because the old total
      only reached it through the next rung's inherited screening -- and there is no next rung.
    """
    denoiser = _CountingDenoiser()
    outcome = _run(denoiser=denoiser)
    assert outcome.depth_reached == 2, "this fixture must exercise a real two-rung ladder"
    assert outcome.total_logical_dfe == denoiser.calls


def test_the_stationary_comparator_also_reports_what_it_ran():
    denoiser = _CountingDenoiser()
    outcome = _run(plan=_stationary_plan(), denoiser=denoiser)
    assert outcome.total_logical_dfe == denoiser.calls


# --------------------------------------------------------------------------------------------
# a descendant pool that repeats what the archive already holds is not a novel descendant
# --------------------------------------------------------------------------------------------


def _constant_denoiser(x_t, t, struct):
    """Collapses every fork onto ONE sequence: the same argmax at every position, every step.

    A degenerate denoiser is the cheap way to reach the state the ladder must recognise -- a
    non-empty descendant pool that carries nothing the archive has not already recorded.
    """
    import torch

    length = int(x_t.shape[0])
    logits = torch.full((length, VOCAB), float("-inf"), dtype=torch.float32)
    for position in range(length):
        for token in sorted(F.AA):
            logits[position, token] = 0.0
        logits[position, sorted(F.AA)[0]] = 10.0
    return logits


def test_a_descendant_pool_that_repeats_the_archive_stops_the_ladder():
    """PLAN §4.5's ``no novel descendant``.

    The old check was ``the pool is non-empty``, which a degenerate run satisfies forever: every
    rung re-derives sequences the archive already holds, the ladder runs to its depth cap, and the
    artifact reports ``depth_cap`` over a single unique sequence.  Depth then reads as search when
    it was repetition, and every per-depth rate computed from it is measured over duplicates.

    ``no novel descendant`` is the reason PLAN names; the predicate is chosen HERE, because PLAN
    does not define novelty: it is the archive's own ``sequence_equivalence_key``, i.e. a pool is
    novel when it carries an equivalence class no earlier depth produced.
    """
    outcome = _run(denoiser=_constant_denoiser)
    unique = {endpoint.sequence_md5 for endpoint in outcome.archive.endpoints()}
    assert len(unique) == 1, "the fixture must actually collapse onto one sequence"
    assert outcome.stopping_reason is StoppingReason.NO_NOVEL_DESCENDANT
    assert outcome.depth_reached < 2


def test_an_ordinary_ladder_is_not_stopped_by_the_novelty_check():
    """The guard must not be satisfiable by refusing everything."""
    outcome = _run()
    assert outcome.stopping_reason is StoppingReason.DEPTH_CAP
    assert outcome.depth_reached == 2


def test_the_ladder_refuses_a_missing_arm_identity_or_meter_before_paying_the_prefix():
    """The refusal has to come BEFORE the root capture, or it is not worth having.

    ``run_one_cycle`` would reject both arguments anyway -- but only after the ladder had already
    run a full ``c_0``-step prefix, which is real forward passes spent to discover a typo.  The
    counter proves the check fires first rather than merely eventually.
    """
    for missing in ("feedback_enabled", "cost_meter"):
        denoiser = _CountingDenoiser()
        kwargs = {k: v for k, v in _run_kwargs(denoiser=denoiser).items() if k != missing}
        with pytest.raises(V2LadderError, match=missing):
            run_depth_ladder(**kwargs)
        assert denoiser.calls == 0, f"the ladder paid a prefix before refusing {missing!r}"


def test_the_preflight_projection_reconciles_with_a_realized_ladder():
    """The launch gate must bound the run that actually happens.

    ``project_v2_budget`` is pure arithmetic over the declared schedule and can agree with itself
    forever; the only thing that makes it a bound rather than a second cost model is that it equals
    the graph the ladder executes on the SAME schedule.  Both sides are checked against the
    forward-pass counter, so neither can drift toward the other.
    """
    from inverse_folding.reference_flow.fusion_v2.config import load_v2_config
    from scripts.rf_fusion_v2_preflight import project_v2_budget
    from tests.inverse_folding.test_fusion_v2_config import _mapping

    config = load_v2_config(_mapping(**{
        "substrate.n_steps": F.N_STEPS,
        "schedule.depth_cap": 2,
        "schedule.min_lookahead_tail_steps": 10,
        "schedule.points": [
            {"depth": 0, "r_step": R1, "c_source_step": C0, "c_next_step": C1,
             "n_lookaheads": 3, "band_key": F.STRATUM},
            {"depth": 1, "r_step": R2, "c_source_step": C1, "c_next_step": C2,
             "n_lookaheads": 2, "band_key": F.STRATUM},
        ],
        "caps.max_logical_dfe": 10 ** 6,
        "caps.max_head_calls": 10 ** 6,
        "caps.max_definitive_refolds": 10 ** 6,
    }))

    denoiser = _CountingDenoiser()
    outcome = _run(denoiser=denoiser)
    assert outcome.depth_reached == 2, "the reconciliation needs a ladder that ran to its cap"

    projection = project_v2_budget(config, n_proteins=1)
    assert projection.total_logical_dfe == denoiser.calls
    assert projection.total_logical_dfe == outcome.total_logical_dfe
