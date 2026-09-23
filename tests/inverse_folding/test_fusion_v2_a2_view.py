"""V2F4-3: the A2 feedback-off view and matched extra-lookahead allocation (PLAN §2.3, §4.3).

A2 is **a view of the same run**, not a second run.  PLAN §2.3: "The archive state immediately
before feedback is the prospective A2 view.  A2 and V2 may not regenerate separate initial roots,
lookaheads, Head results, or structure results."  PLAN §4.3 adds that A2 "is not a separate
root-generation run and may not be reconstructed post hoc from only selected rows".

That is the whole experimental design.  If A2 regenerated its own roots, the comparison would
confound the feedback mechanism with root-sampling variance, and every reported difference would be
uninterpretable.  So the acceptance bullet is exact: "A2 and V2 pre-feedback records have identical
endpoint IDs and costs."

The second half is the matched-compute reallocation.  V2 spends resource after feedback (a
projection, a propagation segment, descendant lookaheads); A2 spends the SAME resource on
additional exact futures from the UNCHANGED source, under declared disjoint seeds.  Which resource
"the same" means is a required, no-default choice over a closed vocabulary, because structure
dominates cost (one refold is about 81 DFE) and matching on the wrong component silently hands one
arm more compute.
"""

from __future__ import annotations

import dataclasses

import pytest

from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
from inverse_folding.reference_flow.fusion_v2 import state as st
from inverse_folding.reference_flow.fusion_v2.config import A2_RESOURCE_COMPONENTS
from inverse_folding.reference_flow.fusion_v2_runtime.a2_view import (
    ResourceLedger,
    V2A2Error,
    matched_extra_lookahead_budget,
    snapshot_a2_view,
)
from inverse_folding.reference_flow.fusion_v2_runtime.archive import ExactArchive
from tests.inverse_folding import _v2_fixtures as F


def _endpoint(*, seq="ACDEFG", fork_index=0, fork_seed=4242, risk=-9.0, **over):
    from inverse_folding.reference_flow.fusion.state import sequence_md5

    md5 = sequence_md5(seq)
    kw = dict(
        sequence=seq, sequence_md5=md5,
        head_binding=dataclasses.replace(F.endpoint().head_binding, sequence_md5=md5),
        head_score=dataclasses.replace(F.endpoint().head_score, sequence_md5=md5,
                                       global_risk=risk),
        head_global_risk=risk, fork_index=fork_index, fork_seed=fork_seed,
        replay=st.ReplayIdentity(mode="fork", rng_state=None, fork_seed=fork_seed,
                                 replay_state_hash=F.E),
        feasibility_level=st.FeasibilityLevel.DEFINITIVE,
        structure_outcome=StructureOutcome(feasible=True, metrics={"scTM": 0.9}),
    )
    kw.update(over)
    return F.endpoint(**kw)


def _archive(n=3):
    archive = ExactArchive()
    for index in range(n):
        archive.admit(
            _endpoint(seq="ACDEF" + "GHIKL"[index], fork_index=index, fork_seed=index + 1,
                      risk=-5.0 - index),
            depth=0,
        )
    return archive


def _ledger(**over):
    kw = dict(logical_dfe=200, head_calls=3, definitive_refolds=1, gpu_seconds=12.5,
              walltime_s=30.0)
    kw.update(over)
    return ResourceLedger(**kw)


def _snapshot(archive, *, depth, **over):
    source = F.source()
    return snapshot_a2_view(
        archive,
        protein_id=source.lineage.protein_id,
        source_state_id=source.state_id,
        depth=depth,
        **over,
    )


# --------------------------------------------------------------------------------------------
# A2 is a VIEW of the same archive
# --------------------------------------------------------------------------------------------


def test_the_view_carries_every_pre_feedback_endpoint_id():
    archive = _archive()
    view = _snapshot(archive, depth=0)
    assert view.endpoint_ids == tuple(e.endpoint_id for e in archive.endpoints())


def test_the_view_is_taken_from_the_same_rows_not_regenerated():
    """PLAN §2.3: A2 and V2 "may not regenerate separate initial roots, lookaheads, Head results,
    or structure results".  Identity of the endpoint ids is what proves they are the same rows."""
    archive = _archive()
    view = _snapshot(archive, depth=0)
    for endpoint in archive.endpoints():
        assert endpoint.endpoint_id in view.endpoint_ids
        assert view.content_digest_by_endpoint[endpoint.endpoint_id] == endpoint.content_digest


def test_the_view_records_the_depth_it_was_taken_at():
    assert _snapshot(_archive(), depth=2).depth == 2


def test_the_view_is_frozen_against_later_archive_growth():
    """The A2 view is the archive state IMMEDIATELY BEFORE feedback.  If it tracked the live
    archive, post-feedback descendants would leak into the control arm."""
    archive = _archive()
    view = _snapshot(archive, depth=0)
    before = len(view.endpoint_ids)
    archive.admit(_endpoint(seq="ACDEFM", fork_index=9, fork_seed=99), depth=1)
    assert len(view.endpoint_ids) == before


def test_the_view_of_an_empty_archive_is_empty_not_an_error():
    assert _snapshot(ExactArchive(), depth=0).endpoint_ids == ()


def test_the_view_may_not_be_built_from_a_filtered_subset():
    """PLAN §4.3: A2 "may not be reconstructed post hoc from only selected rows"."""
    archive = _archive()
    with pytest.raises(V2A2Error, match="subset|selected|complete"):
        _snapshot(archive, depth=0, endpoint_ids=(archive.endpoints()[0].endpoint_id,))


# --------------------------------------------------------------------------------------------
# identical costs before feedback
# --------------------------------------------------------------------------------------------


def test_the_view_carries_the_pre_feedback_cost_ledger():
    ledger = _ledger()
    view = _snapshot(_archive(), depth=0, pre_feedback_cost=ledger)
    assert view.pre_feedback_cost == ledger


def test_a2_and_v2_pre_feedback_costs_are_one_measurement_not_two():
    """The acceptance bullet: "A2 and V2 pre-feedback records have identical endpoint IDs and
    costs".  They are identical because they are the SAME prefix, recorded once."""
    ledger = _ledger()
    a = _snapshot(_archive(), depth=0, pre_feedback_cost=ledger)
    b = _snapshot(_archive(), depth=0, pre_feedback_cost=ledger)
    assert a.pre_feedback_cost == b.pre_feedback_cost
    assert a.endpoint_ids == b.endpoint_ids


# --------------------------------------------------------------------------------------------
# matched extra-lookahead allocation
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("component", sorted(A2_RESOURCE_COMPONENTS))
def test_every_declared_matching_resource_is_supported(component):
    budget = matched_extra_lookahead_budget(
        v2_post_feedback_cost=_ledger(), matching_resource=component,
        per_lookahead_cost=_ledger(logical_dfe=50, head_calls=1, definitive_refolds=1,
                                   gpu_seconds=4.0, walltime_s=10.0),
    )
    assert budget.matched_component == component


def test_the_budget_matches_on_the_declared_component():
    """V2 spent 200 logical DFE after feedback; one extra lookahead costs 50, so A2 gets 4."""
    budget = matched_extra_lookahead_budget(
        v2_post_feedback_cost=_ledger(logical_dfe=200), matching_resource="logical_dfe",
        per_lookahead_cost=_ledger(logical_dfe=50))
    assert budget.n_extra_lookaheads == 4


def test_matching_on_a_different_component_gives_a_different_budget():
    """Structure dominates cost -- one refold is about 81 DFE -- so matching on refolds and
    matching on DFE are materially different experiments.  This is exactly why the component is a
    required choice rather than a default."""
    post = _ledger(logical_dfe=200, definitive_refolds=4)
    per = _ledger(logical_dfe=50, definitive_refolds=1)
    by_dfe = matched_extra_lookahead_budget(
        v2_post_feedback_cost=post, matching_resource="logical_dfe", per_lookahead_cost=per)
    by_refold = matched_extra_lookahead_budget(
        v2_post_feedback_cost=post, matching_resource="definitive_refolds",
        per_lookahead_cost=dataclasses.replace(per, definitive_refolds=2))
    assert by_dfe.n_extra_lookaheads != by_refold.n_extra_lookaheads


def test_an_unknown_matching_resource_is_refused():
    with pytest.raises(V2A2Error, match="matching_resource|vocabulary"):
        matched_extra_lookahead_budget(
            v2_post_feedback_cost=_ledger(), matching_resource="vibes",
            per_lookahead_cost=_ledger(logical_dfe=50))


def test_the_matching_resource_has_no_default():
    """CLAUDE.md forbids scientific defaults; PLAN leaves which resource A2 matches on open, so it
    must be stated by the caller rather than guessed here."""
    with pytest.raises(TypeError):
        matched_extra_lookahead_budget(
            v2_post_feedback_cost=_ledger(), per_lookahead_cost=_ledger())


def test_a_zero_per_lookahead_cost_is_refused():
    """Dividing by zero would report an unbounded budget, which reads as "A2 gets infinite compute"
    rather than "the cost model is broken"."""
    with pytest.raises(V2A2Error):
        matched_extra_lookahead_budget(
            v2_post_feedback_cost=_ledger(), matching_resource="logical_dfe",
            per_lookahead_cost=_ledger(logical_dfe=0))


def test_the_budget_reports_the_unmatched_remainder_explicitly():
    """Matching on one component leaves the others unmatched.  The complement must be REPORTED,
    not silently ignored: an arm that quietly received more refolds is not a control."""
    budget = matched_extra_lookahead_budget(
        v2_post_feedback_cost=_ledger(logical_dfe=200), matching_resource="logical_dfe",
        per_lookahead_cost=_ledger(logical_dfe=50))
    assert set(budget.unmatched_components) == set(A2_RESOURCE_COMPONENTS) - {"logical_dfe"}


@pytest.mark.parametrize("component", sorted(A2_RESOURCE_COMPONENTS))
def test_the_unmatched_set_is_the_exact_complement(component):
    budget = matched_extra_lookahead_budget(
        v2_post_feedback_cost=_ledger(), matching_resource=component,
        per_lookahead_cost=_ledger())
    assert set(budget.unmatched_components) | {component} == set(A2_RESOURCE_COMPONENTS)
    assert component not in budget.unmatched_components


def test_the_budget_reports_the_residual_it_could_not_spend():
    """220 DFE at 50 per lookahead buys 4 and leaves 20 unspent; hiding the residual would make the
    arms look matched when they are not."""
    budget = matched_extra_lookahead_budget(
        v2_post_feedback_cost=_ledger(logical_dfe=220), matching_resource="logical_dfe",
        per_lookahead_cost=_ledger(logical_dfe=50))
    assert budget.n_extra_lookaheads == 4
    assert budget.residual == pytest.approx(20.0)


def test_a_budget_smaller_than_one_lookahead_buys_nothing_and_says_so():
    budget = matched_extra_lookahead_budget(
        v2_post_feedback_cost=_ledger(logical_dfe=10), matching_resource="logical_dfe",
        per_lookahead_cost=_ledger(logical_dfe=50))
    assert budget.n_extra_lookaheads == 0
    assert budget.residual == pytest.approx(10.0)


# --------------------------------------------------------------------------------------------
# disjoint seeds for the extra futures
# --------------------------------------------------------------------------------------------


def test_extra_lookahead_seeds_are_disjoint_from_the_pre_feedback_pool():
    """PLAN §4.3: the extra futures use "declared disjoint seeds".  Reusing a pre-feedback seed
    would regenerate an endpoint A2 already has, so the extra compute would buy nothing."""
    budget = matched_extra_lookahead_budget(
        v2_post_feedback_cost=_ledger(logical_dfe=200), matching_resource="logical_dfe",
        per_lookahead_cost=_ledger(logical_dfe=50), used_seeds=(1, 2, 3), seed_base=1)
    assert set(budget.extra_fork_seeds).isdisjoint({1, 2, 3})
    assert len(budget.extra_fork_seeds) == budget.n_extra_lookaheads


def test_extra_lookahead_seeds_are_disjoint_from_each_other():
    budget = matched_extra_lookahead_budget(
        v2_post_feedback_cost=_ledger(logical_dfe=500), matching_resource="logical_dfe",
        per_lookahead_cost=_ledger(logical_dfe=50), used_seeds=(7,), seed_base=1)
    assert len(set(budget.extra_fork_seeds)) == len(budget.extra_fork_seeds)


def test_the_seed_allocation_is_deterministic():
    kw = dict(v2_post_feedback_cost=_ledger(logical_dfe=200), matching_resource="logical_dfe",
              per_lookahead_cost=_ledger(logical_dfe=50), used_seeds=(1, 2), seed_base=90)
    assert matched_extra_lookahead_budget(**kw).extra_fork_seeds == \
        matched_extra_lookahead_budget(**kw).extra_fork_seeds


# --------------------------------------------------------------------------------------------
# assembling the resource vector out of the compute journal
# --------------------------------------------------------------------------------------------
#
# Nothing above this line had a non-test caller: the amounts A2 matching is computed from live in
# the journal, per phase, and were never assembled into a ``ResourceLedger``.  These tests pin the
# assembly, because a matching claim computed from an unmeasured vector is the same defect as a
# hardcoded zero wearing a different name.


TXN = "5ZHV_B:fam0:txn:d0:r40-c60:abcdef012345"


def _event(event_id, phase, *, attempt=None, logical_dfe=0, head_calls=0, structure_attempts=0,
           gpu_seconds=0.0, walltime_s=0.0, status="ok"):
    """One raw ledger row, in the ``asdict`` form the shard payload carries."""
    return dict(
        event_id=event_id, attempt_id=attempt or f"att:{event_id}", protein_id="5ZHV_B",
        arm="v2", phase=phase, status=status, logical_dfe=logical_dfe, head_calls=head_calls,
        structure_attempts=structure_attempts, physical_forwards=0, batch_size=1,
        physical_forwards_charged=True, structure_cache_hits=0, head_cache_hits=0,
        gpu_seconds=gpu_seconds, walltime_s=walltime_s, physical_cost_status="observed",
        outcome=None,
    )


@pytest.mark.parametrize("phase,stage", [
    ("root_capture", "prefix"),
    ("screen", "source_pool"),
    ("projection", "feedback"),
    ("segment", "feedback"),
    ("descendant_screen", "descendant_pool"),
])
def test_every_ledger_phase_names_the_stage_of_the_rung_that_paid_it(phase, stage):
    from inverse_folding.reference_flow.fusion_v2_runtime.a2_view import ledger_stage

    assert ledger_stage(phase=phase, event_id=f"{TXN}:{phase}") == stage


@pytest.mark.parametrize("marker,stage", [
    ("source", "source_pool"),
    ("descendant", "descendant_pool"),
])
@pytest.mark.parametrize("phase", ["head", "structure"])
def test_head_and_structure_are_attributed_to_the_pool_that_requested_them(marker, stage, phase):
    """Head and structure are charged on BOTH sides of the feedback boundary.

    Attributing them by phase alone would put the descendant pool's refolds -- the dominant term of
    everything V2 spends after feedback -- on the pre-feedback side, and the matched allocation
    would be computed from a number that excludes the very work it is supposed to reassign.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.a2_view import ledger_stage

    assert ledger_stage(phase=phase, event_id=f"{TXN}:{marker}:{phase}:ep1") == stage


@pytest.mark.parametrize("event_id,phase", [
    (f"{TXN}:head", "head"),
    (f"{TXN}:structure:ep1", "structure"),
    (f"{TXN}:whatever", "projection_v2"),
])
def test_an_event_that_cannot_be_attributed_to_a_stage_is_refused(event_id, phase):
    """A row binned by a fallback would move real spend to whichever side the fallback favours,
    silently, and the two arms' costs would differ by an amount nobody could see."""
    from inverse_folding.reference_flow.fusion_v2_runtime.a2_view import ledger_stage

    with pytest.raises(V2A2Error, match="attribut|stage|phase"):
        ledger_stage(phase=phase, event_id=event_id)


def test_the_assembled_vector_carries_all_five_resource_components():
    """FUSION_V2 §6.1: the envelope "retains at least DFE, Head calls, definitive refolds, GPU time,
    and walltime" and "is not silently collapsed into one invented universal cost"."""
    from inverse_folding.reference_flow.fusion_v2_runtime.a2_view import stage_ledgers

    rows = [
        _event(f"{TXN}:descendant_screen", "descendant_screen", logical_dfe=80,
               gpu_seconds=1.5, walltime_s=4.0),
        _event(f"{TXN}:descendant:head", "head", head_calls=2, gpu_seconds=0.5, walltime_s=1.0),
        _event(f"{TXN}:descendant:structure:ep1", "structure", structure_attempts=1,
               gpu_seconds=2.5, walltime_s=2.5),
    ]
    ledgers = stage_ledgers(rows)
    assert ledgers["descendant_pool"] == ResourceLedger(
        logical_dfe=80, head_calls=2, definitive_refolds=1, gpu_seconds=4.5, walltime_s=7.5)
    # Stages nothing was charged to are present and zero, never absent: a missing key would make a
    # stage that ran nothing indistinguishable from one whose rows were dropped.
    assert ledgers["feedback"] == ResourceLedger()


def test_a_replayed_ledger_row_is_not_charged_twice():
    """Resume re-reads fragments it has already seen.  A vector that double-counted them would
    inflate exactly the number the matched allocation is divided by."""
    from inverse_folding.reference_flow.fusion_v2_runtime.a2_view import stage_ledgers

    row = _event(f"{TXN}:segment", "segment", logical_dfe=20, gpu_seconds=0.4)
    assert stage_ledgers([row, dict(row)])["feedback"].logical_dfe == 20


def test_a_per_lookahead_unit_cost_may_be_fractional():
    """Three refolds over two lookaheads is 1.5 refolds each.  Rounding it to an int would change
    the matched budget, which is the number the whole depth-versus-breadth contrast rests on."""
    unit = ResourceLedger(logical_dfe=100, definitive_refolds=3).per_unit(2)
    assert unit.component("definitive_refolds") == pytest.approx(1.5)
    assert unit.component("logical_dfe") == pytest.approx(50.0)


def test_a_unit_cost_over_no_lookaheads_is_refused():
    with pytest.raises(V2A2Error, match="lookahead|zero|positive"):
        ResourceLedger(logical_dfe=100).per_unit(0)


# --------------------------------------------------------------------------------------------
# what the artifact is allowed to call "matched"
# --------------------------------------------------------------------------------------------


def _allocation(**over):
    from inverse_folding.reference_flow.fusion_v2_runtime.a2_view import a2_matched_allocation

    kw = dict(
        matching_resource="definitive_refolds",
        pre_feedback_cost=ResourceLedger(logical_dfe=300, head_calls=3, definitive_refolds=3),
        v2_post_feedback_cost=ResourceLedger(logical_dfe=120, head_calls=2, definitive_refolds=2),
        per_lookahead_cost=ResourceLedger(logical_dfe=50, head_calls=1, definitive_refolds=1),
        executed_extra_lookaheads=0,
        post_feedback_spend_observed=True,
    )
    kw.update(over)
    return a2_matched_allocation(**kw)


def test_an_allocation_that_was_never_executed_is_not_reported_as_matched():
    """The defect this closes: the artifact reported ``matched_extra_lookaheads: 0`` beside a
    declared ``matching_resource``, and a depth-versus-breadth analysis reads that pair as "the two
    arms were compute-matched on refolds" -- when V2 in fact received a whole post-feedback segment
    and a descendant pool that A2 never got."""
    allocation = _allocation()
    assert allocation.matched is False
    assert allocation.status == "declared_but_not_executed"
    # The column that would otherwise read as a matching claim carries nothing to claim with.
    assert allocation.reported_matching_resource is None
    assert allocation.owed_extra_lookaheads == 2
    assert allocation.report()["detail"]


def test_an_executed_allocation_reports_the_resource_it_matched_on():
    """The guard above must not be satisfiable by refusing to report a matching that really ran."""
    allocation = _allocation(executed_extra_lookaheads=2)
    assert allocation.matched is True
    assert allocation.status == "matched"
    assert allocation.reported_matching_resource == "definitive_refolds"


def test_an_arm_that_never_ran_the_feedback_stage_cannot_derive_what_a2_is_owed():
    """An A2 shard has no post-feedback spend of its own: the amount it is owed is defined by its
    paired V2 run, which this shard cannot see.  Reporting its own zero as "matched" would let the
    control arm certify a matching nobody computed."""
    allocation = _allocation(
        v2_post_feedback_cost=ResourceLedger(), post_feedback_spend_observed=False)
    assert allocation.matched is False
    assert allocation.status == "not_derivable_in_this_shard"
    assert allocation.owed_extra_lookaheads is None
    assert allocation.reported_matching_resource is None


def test_a_view_with_nothing_to_reassign_says_so_rather_than_claiming_a_match():
    """A rung that stopped before projecting spent nothing after its view.  That is genuinely
    matched -- but it must be distinguishable from an allocation that was owed and skipped."""
    allocation = _allocation(v2_post_feedback_cost=ResourceLedger())
    assert allocation.status == "nothing_to_match"
    assert allocation.matched is True


def test_an_unmeasured_unit_cost_cannot_be_turned_into_an_owed_count():
    """A zero unit cost would report an unbounded budget.  The honest answer is that the shard did
    not measure what one extra exact future costs, not that A2 is owed infinitely many."""
    allocation = _allocation(per_lookahead_cost=ResourceLedger())
    assert allocation.status == "unit_cost_not_measured"
    assert allocation.owed_extra_lookaheads is None
    assert allocation.matched is False


def test_the_allocation_report_keeps_the_whole_resource_vector():
    """FUSION_V2 §6.1: the full vector "remains attached to every plotted point"."""
    report = _allocation().report()
    for key in ("v2_post_feedback_cost", "a2_pre_feedback_cost", "per_lookahead_cost"):
        assert set(report[key]) == set(A2_RESOURCE_COMPONENTS), key
    assert set(report["declared_unmatched_components"]) == \
        set(A2_RESOURCE_COMPONENTS) - {"definitive_refolds"}
    assert report["matched_allocation_executed"] is False
    assert report["owed_extra_lookaheads"] == 2


def test_the_owed_futures_are_declared_with_seeds_disjoint_from_the_ones_already_used():
    """PLAN §4.3: the extra futures use "declared disjoint seeds".  A seed the run already forked
    from would regenerate an endpoint A2 already holds."""
    allocation = _allocation(used_seeds=(1, 2, 3))
    assert len(allocation.extra_fork_seeds) == allocation.owed_extra_lookaheads
    assert set(allocation.extra_fork_seeds).isdisjoint({1, 2, 3})
