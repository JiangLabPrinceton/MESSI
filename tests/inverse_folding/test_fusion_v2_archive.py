"""V2F4-2: the monotone archive, feasibility promotion, and family-aware selection.

PLAN §2.3: "Archive update is monotone and must preserve the previous best definitively feasible
endpoint even if every new descendant regresses.  Distinct logical endpoints remain visible even
when sequences converge.  Duplicate sequence or sibling multiplicity may not buy extra ancestry
mass; selection must operate on declared lineage families or sequence-equivalence classes while
preserving all raw endpoint rows for audit."

PLAN §4.2: an endpoint may enter as ``unvalidated`` but must become ``definitive`` before it can
become feedback ancestry, support the reported feasible frontier, or be returned as a final design.

Three failure modes drive the design:

* **a regressing elite** would let one bad descendant erase the best result the run ever found, and
  the reported frontier would then understate the method;
* **duplicate ancestry mass** would let a family that happened to emit more identical siblings win
  selection on multiplicity rather than on quality -- a pure artefact of sampling variance; and
* **a provisional endpoint purchasing ancestry** would let feedback propagate from a design whose
  structure was never actually evaluated.
"""

from __future__ import annotations

import dataclasses

import pytest

from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
from inverse_folding.reference_flow.fusion_v2 import state as st
from inverse_folding.reference_flow.fusion_v2_runtime.archive import (
    ExactArchive,
    V2ArchiveError,
    select_family_representatives,
)
from tests.inverse_folding import _v2_fixtures as F


def _endpoint(*, seq="ACDEFG", fork_index=0, fork_seed=4242, risk=-9.0, **over):
    """A scored endpoint.  ``risk`` is the Head's global risk: LOWER IS BETTER."""
    from inverse_folding.reference_flow.fusion.state import sequence_md5

    md5 = sequence_md5(seq)
    binding = dataclasses.replace(
        F.endpoint().head_binding, sequence_md5=md5,
    )
    score = dataclasses.replace(
        F.endpoint().head_score, sequence_md5=md5, global_risk=risk,
    )
    kw = dict(sequence=seq, sequence_md5=md5, head_binding=binding, head_score=score,
              head_global_risk=risk, fork_index=fork_index, fork_seed=fork_seed,
              # The replay stream is bound to the endpoint's OWN fork seed; leaving the fixture's
              # default here would make every sibling claim the same stream.
              replay=st.ReplayIdentity(mode="fork", rng_state=None, fork_seed=fork_seed,
                                       replay_state_hash=F.E),
              feasibility_level=st.FeasibilityLevel.UNVALIDATED, structure_outcome=None)
    kw.update(over)
    return F.endpoint(**kw)


def _definitive(**over):
    kw = dict(feasibility_level=st.FeasibilityLevel.DEFINITIVE,
              structure_outcome=StructureOutcome(feasible=True, metrics={"scTM": 0.91}))
    kw.update(over)
    return _endpoint(**kw)


# --------------------------------------------------------------------------------------------
# every complete lookahead is retained
# --------------------------------------------------------------------------------------------



def _archive_with_one_endpoint():
    """One admitted, still-unvalidated endpoint in a fresh archive."""
    archive = ExactArchive()
    endpoint = F.endpoint()
    archive.admit(endpoint, depth=0)
    return archive, endpoint


def test_every_admitted_endpoint_is_retained_as_a_raw_row():
    """PLAN §2.3: "Every complete lookahead is retained as an exact endpoint record"."""
    archive = ExactArchive()
    archive.admit(_endpoint(seq="ACDEFG", fork_index=0), depth=0)
    archive.admit(_endpoint(seq="ACDEFH", fork_index=1, fork_seed=4243), depth=0)
    assert len(archive.raw_rows()) == 2


def test_identical_sequences_from_different_forks_stay_visible_as_distinct_rows():
    """PLAN §2.3: "Distinct logical endpoints remain visible even when sequences converge"."""
    archive = ExactArchive()
    archive.admit(_endpoint(seq="ACDEFG", fork_index=0, fork_seed=1), depth=0)
    archive.admit(_endpoint(seq="ACDEFG", fork_index=1, fork_seed=2), depth=0)
    assert len(archive.raw_rows()) == 2
    assert len({row.endpoint_id for row in archive.raw_rows()}) == 2


def test_readmitting_the_same_endpoint_twice_is_refused():
    """The same endpoint id twice is a bookkeeping error, not a duplicate sibling."""
    archive = ExactArchive()
    endpoint = _endpoint()
    archive.admit(endpoint, depth=0)
    with pytest.raises(V2ArchiveError, match="already"):
        archive.admit(endpoint, depth=0)


def test_an_admitted_row_records_the_depth_it_was_first_seen_at():
    archive = ExactArchive()
    archive.admit(_endpoint(), depth=2)
    assert archive.raw_rows()[0].first_depth_seen == 2


# --------------------------------------------------------------------------------------------
# monotone elite
# --------------------------------------------------------------------------------------------


def test_a_better_definitive_endpoint_becomes_the_elite():
    archive = ExactArchive()
    archive.admit(_definitive(seq="ACDEFG", risk=-5.0), depth=0)
    archive.admit(_definitive(seq="ACDEFH", risk=-9.0, fork_index=1, fork_seed=2), depth=0)
    assert archive.elite().sequence == "ACDEFH"


def test_a_worse_descendant_cannot_overwrite_the_elite():
    """PLAN §2.3: the archive "must preserve the previous best definitively feasible endpoint even
    if every new descendant regresses"."""
    archive = ExactArchive()
    archive.admit(_definitive(seq="ACDEFG", risk=-9.0), depth=0)
    archive.admit(_definitive(seq="ACDEFH", risk=-1.0, fork_index=1, fork_seed=2), depth=1)
    assert archive.elite().sequence == "ACDEFG"


def test_equal_risk_elite_is_independent_of_insertion_order():
    first = _definitive(seq="ACDEFG", risk=-9.0, fork_index=0, fork_seed=1)
    second = _definitive(seq="ACDEFH", risk=-9.0, fork_index=1, fork_seed=2)
    expected = min(first.endpoint_id, second.endpoint_id)
    observed = []
    for order in ((first, second), (second, first)):
        archive = ExactArchive()
        for endpoint in order:
            archive.admit(endpoint, depth=0)
        observed.append(archive.elite().endpoint_id)
    assert observed == [expected, expected]


def test_the_elite_survives_a_whole_generation_of_regression():
    archive = ExactArchive()
    archive.admit(_definitive(seq="ACDEFG", risk=-9.0), depth=0)
    for index in range(5):
        archive.admit(
            _definitive(seq="ACDEF" + "HIKLM"[index], risk=-1.0 + index,
                        fork_index=index + 1, fork_seed=100 + index),
            depth=1,
        )
    assert archive.elite().sequence == "ACDEFG"
    assert archive.elite().head_global_risk == -9.0


def test_an_unvalidated_endpoint_never_becomes_the_elite():
    """PLAN §4.2: an endpoint must be definitive before it can "support the reported feasible
    frontier"."""
    archive = ExactArchive()
    archive.admit(_endpoint(seq="ACDEFG", risk=-99.0), depth=0)
    assert archive.elite() is None


def test_a_provisional_endpoint_never_becomes_the_elite():
    archive = ExactArchive()
    archive.admit(_endpoint(seq="ACDEFG", risk=-99.0,
                            feasibility_level=st.FeasibilityLevel.PROVISIONAL), depth=0)
    assert archive.elite() is None


def test_a_provisional_endpoint_with_a_passing_structure_is_still_not_the_elite():
    """The sharp case.  A provisional endpoint with NO structure result is excluded by the
    structure check alone, so testing only that would let a broken feasibility check pass.  Here
    the structure was evaluated and passed, and the endpoint STILL must not lead the frontier --
    PLAN §4.2 requires DEFINITIVE, and provisional means the verdict is not final."""
    archive = ExactArchive()
    endpoint = _endpoint(seq="ACDEFG", risk=-99.0)
    archive.admit(endpoint, depth=0)
    archive.promote(endpoint.endpoint_id, feasibility_level=st.FeasibilityLevel.PROVISIONAL,
                    structure_outcome=StructureOutcome(feasible=True, metrics={"scTM": 0.95}),
                    depth=0)
    assert archive.elite() is None
    assert not archive.may_become_ancestry(endpoint.endpoint_id)


def test_a_provisional_endpoint_with_a_passing_structure_is_not_selectable():
    """Same sharp case, at the selection boundary rather than the elite boundary."""
    archive = ExactArchive()
    endpoint = _endpoint(seq="ACDEFG", risk=-99.0)
    archive.admit(endpoint, depth=0)
    archive.promote(endpoint.endpoint_id, feasibility_level=st.FeasibilityLevel.PROVISIONAL,
                    structure_outcome=StructureOutcome(feasible=True), depth=0)
    assert select_family_representatives(archive.endpoints()) == ()


def test_an_infeasible_definitive_endpoint_is_unrepresentable():
    """Stronger than an archive guard: the V2F1 state layer already refuses to CONSTRUCT a
    definitive endpoint whose structure verdict failed, so an infeasible design cannot even reach
    the archive claiming definitive feasibility."""
    from inverse_folding.reference_flow.fusion_v2.errors import V2Error

    with pytest.raises(V2Error, match="structure"):
        _endpoint(seq="ACDEFG", risk=-99.0, feasibility_level=st.FeasibilityLevel.DEFINITIVE,
                  structure_outcome=StructureOutcome(feasible=False, failure_reason="scTM 0.31"))


# --------------------------------------------------------------------------------------------
# feasibility promotion
# --------------------------------------------------------------------------------------------


def test_an_endpoint_can_be_promoted_from_unvalidated_to_definitive():
    archive = ExactArchive()
    endpoint = _endpoint(seq="ACDEFG", risk=-9.0)
    archive.admit(endpoint, depth=0)
    assert archive.elite() is None
    archive.promote(endpoint.endpoint_id, feasibility_level=st.FeasibilityLevel.DEFINITIVE,
                    structure_outcome=StructureOutcome(feasible=True, metrics={"scTM": 0.9}),
                    depth=0)
    assert archive.elite().sequence == "ACDEFG"


def test_feasibility_may_never_regress():
    archive = ExactArchive()
    endpoint = _endpoint()
    archive.admit(endpoint, depth=0)
    archive.promote(endpoint.endpoint_id, feasibility_level=st.FeasibilityLevel.DEFINITIVE,
                    structure_outcome=StructureOutcome(feasible=True), depth=0)
    with pytest.raises(V2ArchiveError, match="regress|advance"):
        archive.promote(endpoint.endpoint_id,
                        feasibility_level=st.FeasibilityLevel.PROVISIONAL,
                        structure_outcome=None, depth=1)


def test_promoting_to_definitive_requires_an_evaluated_structure():
    """PLAN §4.2 requires the structure result to be explicit; "definitive" without an evaluated
    outcome is a claim with no measurement behind it."""
    archive = ExactArchive()
    endpoint = _endpoint()
    archive.admit(endpoint, depth=0)
    with pytest.raises(V2ArchiveError):
        archive.promote(endpoint.endpoint_id,
                        feasibility_level=st.FeasibilityLevel.DEFINITIVE,
                        structure_outcome=None, depth=0)


def test_promoting_an_unknown_endpoint_is_refused():
    archive = ExactArchive()
    with pytest.raises(V2ArchiveError, match="unknown|not in"):
        archive.promote("endpoint:nobody", feasibility_level=st.FeasibilityLevel.DEFINITIVE,
                        structure_outcome=StructureOutcome(feasible=True), depth=0)


def test_the_structure_outcome_stays_explicit_on_the_row():
    """PLAN §4.2: "Structure cache hit/miss, execution, metrics, failure reason, and cost must
    remain explicit"."""
    archive = ExactArchive()
    endpoint = _endpoint()
    archive.admit(endpoint, depth=0)
    outcome = StructureOutcome(feasible=True, metrics={"scTM": 0.88})
    archive.promote(endpoint.endpoint_id, feasibility_level=st.FeasibilityLevel.DEFINITIVE,
                    structure_outcome=outcome, depth=0)
    assert archive.structure_outcome(endpoint.endpoint_id) == outcome


# --------------------------------------------------------------------------------------------
# ancestry admission
# --------------------------------------------------------------------------------------------


def test_only_a_definitive_endpoint_may_become_feedback_ancestry():
    archive = ExactArchive()
    endpoint = _endpoint()
    archive.admit(endpoint, depth=0)
    assert not archive.may_become_ancestry(endpoint.endpoint_id)
    archive.promote(endpoint.endpoint_id, feasibility_level=st.FeasibilityLevel.DEFINITIVE,
                    structure_outcome=StructureOutcome(feasible=True), depth=0)
    assert archive.may_become_ancestry(endpoint.endpoint_id)


def test_a_failed_structure_verdict_cannot_promote_an_endpoint_to_ancestry():
    """Promotion is where a real run learns the structure verdict, so the archive must refuse a
    failing one rather than record it as definitive."""
    archive = ExactArchive()
    endpoint = _endpoint()
    archive.admit(endpoint, depth=0)
    archive.promote(endpoint.endpoint_id, feasibility_level=st.FeasibilityLevel.PROVISIONAL,
                    structure_outcome=StructureOutcome(feasible=False, failure_reason="clash"),
                    depth=0)
    assert not archive.may_become_ancestry(endpoint.endpoint_id)


# --------------------------------------------------------------------------------------------
# family-aware selection: multiplicity may not buy ancestry
# --------------------------------------------------------------------------------------------


def test_selection_returns_one_representative_per_sequence_equivalence_class():
    """PLAN §2.3: "Duplicate sequence or sibling multiplicity may not buy extra ancestry mass"."""
    rows = [
        _definitive(seq="ACDEFG", fork_index=0, fork_seed=1, risk=-5.0),
        _definitive(seq="ACDEFG", fork_index=1, fork_seed=2, risk=-5.0),
        _definitive(seq="ACDEFG", fork_index=2, fork_seed=3, risk=-5.0),
        _definitive(seq="ACDEFH", fork_index=3, fork_seed=4, risk=-4.0),
    ]
    chosen = select_family_representatives(rows)
    assert len(chosen) == 2
    assert {c.sequence for c in chosen} == {"ACDEFG", "ACDEFH"}


def test_a_family_with_more_identical_siblings_does_not_win_on_multiplicity():
    """Three copies of a worse design must not out-vote one copy of a better one."""
    rows = [
        _definitive(seq="ACDEFG", fork_index=i, fork_seed=i + 1, risk=-1.0)
        for i in range(3)
    ] + [_definitive(seq="ACDEFH", fork_index=9, fork_seed=99, risk=-9.0)]
    chosen = select_family_representatives(rows)
    assert chosen[0].sequence == "ACDEFH"


def test_selection_is_independent_of_row_order():
    rows = [
        _definitive(seq="ACDEFG", fork_index=0, fork_seed=1, risk=-3.0),
        _definitive(seq="ACDEFH", fork_index=1, fork_seed=2, risk=-7.0),
        _definitive(seq="ACDEFI", fork_index=2, fork_seed=3, risk=-5.0),
    ]
    forward = [c.sequence for c in select_family_representatives(rows)]
    backward = [c.sequence for c in select_family_representatives(list(reversed(rows)))]
    assert forward == backward


def test_selection_keeps_every_raw_row_available_for_audit():
    """Selection narrows what may become ancestry; it must not delete evidence."""
    archive = ExactArchive()
    for index in range(3):
        archive.admit(_definitive(seq="ACDEFG", fork_index=index, fork_seed=index + 1), depth=0)
    chosen = select_family_representatives(archive.endpoints())
    assert len(chosen) == 1
    assert len(archive.raw_rows()) == 3


def test_selection_ignores_endpoints_that_may_not_become_ancestry():
    rows = [
        _endpoint(seq="ACDEFG", fork_index=0, fork_seed=1, risk=-99.0),      # unvalidated
        _definitive(seq="ACDEFH", fork_index=1, fork_seed=2, risk=-1.0),
    ]
    chosen = select_family_representatives(rows)
    assert [c.sequence for c in chosen] == ["ACDEFH"]


def test_selection_of_an_empty_pool_is_empty_not_an_error():
    """PLAN §4.5 requires typed null behaviour: no admissible endpoint is a normal outcome."""
    assert select_family_representatives([]) == ()


# --------------------------------------------------------------------------------------------
# forking a view: shared history, separate futures
# --------------------------------------------------------------------------------------------


def test_a_forked_view_starts_from_the_same_rows():
    archive = ExactArchive()
    archive.admit(F.endpoint(), depth=0)
    forked = archive.fork_view()
    assert [row.endpoint_id for row in forked.raw_rows()] == \
        [row.endpoint_id for row in archive.raw_rows()]
    assert forked.elite() is archive.elite()


def test_writing_into_a_forked_view_never_reaches_the_original():
    """Two arms over one pre-feedback pool still generate their OWN descendants.

    PLAN §2.3 defines the A2 view as the archive immediately before feedback, so an arm that could
    see the other arm's post-feedback descendants would not be reading a control at all.
    """
    archive = ExactArchive()
    archive.admit(F.endpoint(), depth=0)
    before = {row.endpoint_id for row in archive.raw_rows()}

    forked = archive.fork_view()
    descendant = F.endpoint(fork_index=7)
    assert descendant.endpoint_id not in before, "the fixture must produce a NEW row"
    forked.admit(descendant, depth=1)

    assert {row.endpoint_id for row in archive.raw_rows()} == before
    assert descendant.endpoint_id in {row.endpoint_id for row in forked.raw_rows()}


def test_a_forked_view_promotes_independently():
    """Promotion mutates row bookkeeping; the fork must not share those row objects."""
    archive = ExactArchive()
    endpoint = F.endpoint(feasibility_level=st.FeasibilityLevel.UNVALIDATED,
                          structure_outcome=None)
    archive.admit(endpoint, depth=0)
    forked = archive.fork_view()
    forked.promote(endpoint.endpoint_id, feasibility_level=st.FeasibilityLevel.DEFINITIVE,
                   structure_outcome=StructureOutcome(feasible=True, metrics={"scTM": 0.9}),
                   depth=0)
    assert forked.may_become_ancestry(endpoint.endpoint_id)
    assert not archive.may_become_ancestry(endpoint.endpoint_id), (
        "promoting inside a fork changed the original's ancestry eligibility"
    )


def test_the_row_digest_still_names_the_endpoint_the_row_holds_after_promotion():
    """The digest is a JOIN KEY, and promotion changes the endpoint's content.

    ``promote`` replaces ``row.endpoint`` so ``elite()`` cannot hand back a record still labelled
    unvalidated -- but the row's ``endpoint_content_digest`` was computed once at admit time and
    never refreshed.  The ``archive`` table and the ``complete_endpoints`` table both carry that
    column, so after promotion they name different content for the same endpoint and the join
    between them is silently wrong.
    """
    archive, endpoint = _archive_with_one_endpoint()
    admitted_digest = next(iter(archive.raw_rows())).endpoint_content_digest
    archive.promote(endpoint.endpoint_id, feasibility_level=st.FeasibilityLevel.DEFINITIVE,
                    structure_outcome=StructureOutcome(feasible=True, metrics={"scTM": 0.9}),
                    depth=0)
    row = next(iter(archive.raw_rows()))
    stored = next(iter(archive.endpoints()))
    assert row.endpoint_content_digest != admitted_digest, (
        "promotion changed the endpoint but not its digest")
    assert row.endpoint_content_digest == stored.content_digest
