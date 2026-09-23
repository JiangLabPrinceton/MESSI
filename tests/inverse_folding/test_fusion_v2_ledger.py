"""V2F7: the compute ledger, hard caps, and the append-only attempt journal (PLAN §3.4, §5.3-5.4).

Three separable claims, and the suite keeps them separable because conflating them is how a cost
report stops being evidence:

**Logical assigned work** is what the method decided to spend.  It is counted ONCE per logical
event, so a retried attempt cannot inflate the number the science is compared on.

**Observed physical work** is what the machine actually burned.  It sums per distinct attempt, so
re-emitting the same attempt row (a resume replaying a fragment) does not double-charge it.

**Unknown physical work** is what a backend burned but could not report.  PLAN §5.4 is explicit
that this must persist as ``unknown_after_start`` "rather than zero or a guessed value" -- a zero
here would read as "this attempt was free", which is the one thing it certainly was not.
"""

from __future__ import annotations

import json

import pytest

from inverse_folding.reference_flow.fusion_v2.errors import V2Error
from inverse_folding.reference_flow.fusion_v2_runtime.ledger import (
    UNKNOWN_AFTER_START,
    AttemptJournal,
    CapsExceeded,
    V2LedgerError,
    V2LedgerEvent,
    aggregate_v2_ledger,
    check_caps,
)


def _event(**over):
    kw = dict(
        event_id="evt:5ZHV_B:d0:screen", attempt_id="att:0001", protein_id="5ZHV_B",
        arm="v2", phase="screen", status="ok",
        logical_dfe=200, head_calls=4, structure_attempts=4,
        physical_forwards=200, batch_size=1, walltime_s=1.5, gpu_seconds=1.2,
    )
    kw.update(over)
    return V2LedgerEvent(**kw)


def _caps(**over):
    from inverse_folding.reference_flow.fusion_v2.config import V2CapsConfig

    kw = dict(max_logical_dfe=10_000, max_head_calls=100, max_definitive_refolds=100,
              max_gpu_seconds=1000, max_walltime_s=1000, max_retries=2,
              retry_scope="per_request")
    kw.update(over)
    return V2CapsConfig(**kw)


# --------------------------------------------------------------------------------------------
# the three kinds of work stay separate
# --------------------------------------------------------------------------------------------


def test_logical_work_is_counted_once_per_logical_event():
    """A retry re-runs the machine; it does not re-decide the method's spend.

    Counting the logical DFE twice would make a run that hit a transient failure look like it
    allocated more compute than its matched control -- the exact comparison V2 rests on.
    """
    totals = aggregate_v2_ledger([
        _event(attempt_id="att:0001", status="retried"),
        _event(attempt_id="att:0002", status="ok"),
    ])
    assert totals["logical_dfe"] == 200
    assert totals["n_logical_events"] == 1
    assert totals["n_physical_attempts"] == 2
    assert totals["n_retries"] == 1


def test_physical_work_sums_per_attempt_and_is_idempotent_under_replay():
    """A resume re-reads fragments it already aggregated once."""
    rows = [_event(attempt_id="att:0001"), _event(attempt_id="att:0002", status="retried")]
    once = aggregate_v2_ledger(rows)
    twice = aggregate_v2_ledger(rows + rows)
    assert once == twice
    assert once["physical_forwards"] == 400


def test_a_conflicting_second_row_for_one_attempt_is_refused():
    """One physical execution has ONE physical cost; keeping the last write would let a bogus
    count silently replace a measured one."""
    with pytest.raises(V2LedgerError, match="conflicting"):
        aggregate_v2_ledger([_event(), _event(physical_forwards=999)])


def test_unknown_physical_work_is_never_folded_into_the_observed_total():
    """PLAN §5.4: persist ``unknown_after_start`` rather than zero or a guess.

    The row still carries a zero in the observed counters -- there is nothing else to put there --
    but the STATUS says that zero is not a measurement, and the aggregate reports the unknown
    attempts as their own quantity.  Folding them in either direction produces a number that reads
    as fact and is not one.
    """
    totals = aggregate_v2_ledger([
        _event(attempt_id="att:0001"),
        _event(attempt_id="att:0002", status="failed", physical_cost_status=UNKNOWN_AFTER_START,
               physical_forwards=0, gpu_seconds=0.0, walltime_s=0.0),
    ])
    assert totals["physical_forwards"] == 200, "an unknown attempt contributed a fabricated count"
    assert totals["n_attempts_unknown_physical"] == 1
    assert totals["physical_cost_complete"] is False


def test_a_row_may_not_claim_both_an_observed_cost_and_an_unknown_one():
    with pytest.raises(V2LedgerError, match="unknown_after_start"):
        _event(physical_cost_status=UNKNOWN_AFTER_START, physical_forwards=7)


def test_a_run_with_every_cost_reported_says_so():
    assert aggregate_v2_ledger([_event()])["physical_cost_complete"] is True


# --------------------------------------------------------------------------------------------
# a batched forward is charged once, not once per lane
# --------------------------------------------------------------------------------------------


def test_a_batched_group_is_charged_once_for_its_shared_forwards():
    """PLAN §3.1 records "physical batched forwards".

    Each lane's logical DFE is its own, but the forwards were shared.  Emitting one physical row
    per lane would multiply the real GPU cost by the batch width, which is precisely the number a
    matched-compute claim must not overstate.
    """
    lanes = [
        _event(event_id=f"evt:lane{i}", attempt_id="att:batch0", logical_dfe=200,
               physical_forwards=200, batch_size=4,
               # only ONE lane carries the shared physical row; the rest declare it already paid
               physical_forwards_charged=(i == 0))
        for i in range(4)
    ]
    totals = aggregate_v2_ledger(lanes)
    assert totals["logical_dfe"] == 800, "each lane's own logical spend must still be counted"
    assert totals["physical_forwards"] == 200, "the shared batch was charged more than once"


def test_a_lane_that_does_not_carry_the_shared_row_may_not_report_a_different_cost():
    with pytest.raises(V2LedgerError, match="charged"):
        aggregate_v2_ledger([
            _event(event_id="evt:a", attempt_id="att:batch0", batch_size=2,
                   physical_forwards=200, physical_forwards_charged=True),
            _event(event_id="evt:b", attempt_id="att:batch0", batch_size=2,
                   physical_forwards=17, physical_forwards_charged=False),
        ])


# --------------------------------------------------------------------------------------------
# hard caps
# --------------------------------------------------------------------------------------------


def test_a_breached_cap_is_reported_and_names_the_field():
    verdict = check_caps(aggregate_v2_ledger([_event(logical_dfe=20_000)]), _caps())
    assert verdict.within is False
    assert "max_logical_dfe" in verdict.breached


def test_caps_are_hard_and_can_be_asserted():
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import assert_within_caps

    totals = aggregate_v2_ledger([_event(head_calls=1000)])
    with pytest.raises(CapsExceeded, match="max_head_calls"):
        assert_within_caps(totals, _caps())
    assert_within_caps(aggregate_v2_ledger([_event()]), _caps()) is None


def test_unknown_physical_cost_makes_the_physical_caps_unverifiable_and_fails_closed():
    """A budget you cannot measure is not a budget you are inside.

    A run that lost a backend mid-attempt burned an unknown amount of GPU time.  Reporting
    "within caps" on the remaining observed rows would claim a measurement that was never made.
    """
    totals = aggregate_v2_ledger([
        _event(),
        _event(attempt_id="att:0002", status="failed", physical_cost_status=UNKNOWN_AFTER_START,
               physical_forwards=0, gpu_seconds=0.0, walltime_s=0.0),
    ])
    verdict = check_caps(totals, _caps())
    assert verdict.within is False
    assert "max_gpu_seconds" in verdict.unverifiable
    assert "max_walltime_s" in verdict.unverifiable
    # ...but a LOGICAL cap is still verifiable: logical work is assigned, not measured.
    assert "max_logical_dfe" not in verdict.unverifiable


def test_the_retry_cap_counts_retries_not_attempts():
    totals = aggregate_v2_ledger([
        _event(attempt_id=f"att:{i:04d}", status="retried" if i < 3 else "ok")
        for i in range(4)
    ])
    assert totals["n_retries"] == 3
    assert check_caps(totals, _caps(max_retries=2)).within is False
    assert check_caps(totals, _caps(max_retries=3)).within is True


# --------------------------------------------------------------------------------------------
# the append-only attempt journal
# --------------------------------------------------------------------------------------------


def test_a_request_is_journaled_before_it_is_executed(tmp_path):
    """PLAN §5.4: "Oracle requests are journaled before execution and outcomes afterward."

    The whole point is the crash case: if the process dies inside the oracle, the only evidence
    that work was started -- and therefore that budget was burned -- is the line written BEFORE it
    ran.  A journal written after the fact records nothing about the run that failed.
    """
    path = tmp_path / "attempts.jsonl"
    journal = AttemptJournal(path)
    attempt_id = journal.open_request(
        event_id="evt:a", protein_id="5ZHV_B", arm="v2", phase="screen",
        request_kind="head", request_digest="d" * 64,
    )
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["record"] == "requested"
    assert lines[0]["attempt_id"] == attempt_id
    assert "outcome" not in lines[0]


def test_the_outcome_is_appended_after_execution_and_never_rewrites_the_request(tmp_path):
    path = tmp_path / "attempts.jsonl"
    journal = AttemptJournal(path)
    attempt_id = journal.open_request(
        event_id="evt:a", protein_id="5ZHV_B", arm="v2", phase="screen",
        request_kind="head", request_digest="d" * 64,
    )
    journal.close_observed(attempt_id, status="ok", physical_forwards=200, gpu_seconds=1.0,
                           walltime_s=1.5, head_calls=4)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["record"] for line in lines] == ["requested", "observed"]
    assert lines[0]["attempt_id"] == lines[1]["attempt_id"] == attempt_id


def test_an_abandoned_attempt_records_unknown_cost_rather_than_zero(tmp_path):
    path = tmp_path / "attempts.jsonl"
    journal = AttemptJournal(path)
    attempt_id = journal.open_request(
        event_id="evt:a", protein_id="5ZHV_B", arm="v2", phase="structure",
        request_kind="structure", request_digest="d" * 64,
    )
    journal.close_unknown(attempt_id, status="failed", reason="backend died mid-refold")
    row = json.loads(path.read_text().splitlines()[-1])
    assert row["record"] == "abandoned"
    assert row["physical_cost_status"] == UNKNOWN_AFTER_START
    assert "reason" in row


def test_closing_an_attempt_that_was_never_opened_is_refused(tmp_path):
    journal = AttemptJournal(tmp_path / "attempts.jsonl")
    with pytest.raises(V2LedgerError, match="never opened|unknown attempt"):
        journal.close_observed("att:ghost", status="ok", physical_forwards=1, gpu_seconds=0.0,
                               walltime_s=0.0, head_calls=0)


def test_an_attempt_cannot_be_closed_twice(tmp_path):
    journal = AttemptJournal(tmp_path / "attempts.jsonl")
    attempt_id = journal.open_request(
        event_id="evt:a", protein_id="5ZHV_B", arm="v2", phase="screen",
        request_kind="head", request_digest="d" * 64,
    )
    journal.close_observed(attempt_id, status="ok", physical_forwards=1, gpu_seconds=0.0,
                           walltime_s=0.0, head_calls=0)
    with pytest.raises(V2LedgerError, match="already closed"):
        journal.close_unknown(attempt_id, status="failed", reason="late")


def test_the_journal_survives_reopening_and_reports_what_was_left_open(tmp_path):
    """A crash between request and outcome is the case the journal exists for."""
    path = tmp_path / "attempts.jsonl"
    journal = AttemptJournal(path)
    open_id = journal.open_request(
        event_id="evt:a", protein_id="5ZHV_B", arm="v2", phase="screen",
        request_kind="head", request_digest="d" * 64,
    )
    closed_id = journal.open_request(
        event_id="evt:b", protein_id="5ZHV_B", arm="v2", phase="screen",
        request_kind="head", request_digest="e" * 64,
    )
    journal.close_observed(closed_id, status="ok", physical_forwards=1, gpu_seconds=0.0,
                           walltime_s=0.0, head_calls=1)

    reopened = AttemptJournal(path)
    assert reopened.open_attempts() == (open_id,)


def test_events_derived_from_a_journal_carry_the_unknown_status_through(tmp_path):
    """The journal is the source of the ledger, so the sentinel must survive the conversion."""
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import events_from_journal

    path = tmp_path / "attempts.jsonl"
    journal = AttemptJournal(path)
    attempt_id = journal.open_request(
        event_id="evt:a", protein_id="5ZHV_B", arm="v2", phase="structure",
        request_kind="structure", request_digest="d" * 64, logical_dfe=0,
        structure_attempts=1,
    )
    journal.close_unknown(attempt_id, status="failed", reason="oom")
    events = events_from_journal(path)
    assert len(events) == 1
    assert events[0].physical_cost_status == UNKNOWN_AFTER_START
    assert aggregate_v2_ledger(events)["physical_cost_complete"] is False


def test_an_attempt_left_open_becomes_an_unknown_cost_event_not_a_free_one(tmp_path):
    """A process that died mid-oracle burned budget.  Reading its absence as zero is the failure
    PLAN §5.4 names; the reconstruction must charge it as unknown."""
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import events_from_journal

    path = tmp_path / "attempts.jsonl"
    journal = AttemptJournal(path)
    journal.open_request(
        event_id="evt:a", protein_id="5ZHV_B", arm="v2", phase="screen",
        request_kind="head", request_digest="d" * 64,
    )
    events = events_from_journal(path)
    assert len(events) == 1
    assert events[0].physical_cost_status == UNKNOWN_AFTER_START
    assert events[0].status == "failed"


def test_the_ledger_error_is_a_v2_error():
    """So a driver catching V2Error records a typed failure rather than dying on a bare exception."""
    assert issubclass(V2LedgerError, V2Error)
    assert issubclass(CapsExceeded, V2LedgerError)


def test_failed_assigned_work_remains_in_the_logical_ledger():
    """A requested logical event remains assigned even when its physical attempt fails."""
    totals = aggregate_v2_ledger([
        _event(event_id="evt:ok", attempt_id="att:0001", status="ok", logical_dfe=200,
               physical_forwards=200),
        _event(event_id="evt:doomed", attempt_id="att:0002", status="failed", logical_dfe=500,
               physical_forwards=500),
    ])
    assert totals["logical_dfe"] == 700
    assert totals["n_logical_events"] == 2
    assert totals["physical_forwards"] == 700, "the failed attempt's real burn was dropped"


def test_failed_only_and_failed_then_ok_costs_are_conserved_and_idempotent():
    failed = _event(status="failed", logical_dfe=7, physical_forwards=7)
    assert aggregate_v2_ledger([failed])["logical_dfe"] == 7

    succeeded = _event(attempt_id="att:0002", status="ok", logical_dfe=7,
                       physical_forwards=7)
    totals = aggregate_v2_ledger([failed, succeeded])
    assert totals["logical_dfe"] == 7
    assert totals["n_retries"] == 1
    assert aggregate_v2_ledger([failed, succeeded, failed]) == totals


def test_a_deferred_event_assigns_nothing():
    totals = aggregate_v2_ledger([
        _event(event_id="evt:later", status="deferred", logical_dfe=900,
               physical_forwards=0, gpu_seconds=0.0, walltime_s=0.0),
    ])
    assert totals["logical_dfe"] == 0
    assert totals["n_logical_events"] == 0


def test_two_commits_of_one_event_may_not_disagree_on_what_was_assigned():
    """Last-write-wins would let a second, wrong row silently redefine the method's spend."""
    with pytest.raises(V2LedgerError, match="conflicting assigned logical"):
        aggregate_v2_ledger([
            _event(attempt_id="att:0001", logical_dfe=200),
            _event(attempt_id="att:0002", logical_dfe=201),
        ])


# --------------------------------------------------------------------------------------------
# the meter: what binds the journal to a running cycle
# --------------------------------------------------------------------------------------------


from inverse_folding.reference_flow.fusion_v2_runtime import ledger as ledger_mod  # noqa: E402


def _meter(tmp_path, **over):
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import CostMeter

    kw = dict(journal=AttemptJournal(tmp_path / "attempts.jsonl"), protein_id="5ZHV_B",
              arm="v2", gpu_clock=lambda: 0.0)
    kw.update(over)
    return CostMeter(**kw)


def test_a_meter_requires_a_gpu_instrument_rather_than_assuming_one(tmp_path):
    """A run that cannot say what measured its GPU time cannot certify a GPU cap.

    Defaulting the clock would write ``gpu_seconds: 0.0`` for a run that used a GPU heavily, and
    ``check_caps`` treats GPU-seconds as a MEASURED quantity -- so the cap would read as satisfied
    on a number nobody took.  A CPU run passing ``lambda: 0.0`` is making a true measurement; the
    difference is that it is stated.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import CostMeter

    with pytest.raises(TypeError):
        CostMeter(journal=AttemptJournal(tmp_path / "a.jsonl"), protein_id="5ZHV_B", arm="v2")


def test_an_attempt_that_reports_its_cost_closes_observed(tmp_path):
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import events_from_journal

    meter = _meter(tmp_path)
    with meter.attempt(event_id="evt:screen:d0", phase="screen", request_kind="lookahead_pool",
                       request_digest="abc", logical_dfe=150) as receipt:
        receipt.observe(physical_forwards=150)
    events = events_from_journal(tmp_path / "attempts.jsonl")
    assert len(events) == 1
    assert events[0].status == "ok"
    assert events[0].physical_cost_status == ledger_mod.OBSERVED
    assert events[0].logical_dfe == 150
    assert events[0].physical_forwards == 150


def test_an_attempt_whose_body_raises_persists_unknown_after_start(tmp_path):
    """PLAN §5.4: a backend that dies mid-call burned budget nobody measured.

    The exception still propagates -- the journal records what was spent, it does not swallow the
    failure.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import events_from_journal

    meter = _meter(tmp_path)
    with pytest.raises(RuntimeError):
        with meter.attempt(event_id="evt:head:d0", phase="head", request_kind="head_batch",
                           request_digest="abc", head_calls=3):
            raise RuntimeError("CUDA out of memory")
    events = events_from_journal(tmp_path / "attempts.jsonl")
    assert len(events) == 1
    assert events[0].physical_cost_status == UNKNOWN_AFTER_START
    assert events[0].status == "failed"


def test_an_attempt_that_forgets_to_report_is_unknown_and_loud(tmp_path):
    """Silently closing it as free would make forgotten instrumentation indistinguishable from a
    genuinely costless call."""
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import (
        V2LedgerError,
        events_from_journal,
    )

    meter = _meter(tmp_path)
    with pytest.raises(V2LedgerError):
        with meter.attempt(event_id="evt:seg:d0", phase="segment", request_kind="segment",
                           request_digest="abc", logical_dfe=20):
            pass
    events = events_from_journal(tmp_path / "attempts.jsonl")
    assert events[0].physical_cost_status == UNKNOWN_AFTER_START


def test_the_meter_carries_the_shards_identity_onto_every_row(tmp_path):
    """``aggregate_v2_ledger`` refuses one event id under two identities, so the identity has to
    come from the shard rather than from each call site remembering to pass it."""
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import events_from_journal

    meter = _meter(tmp_path, protein_id="9L2Q_A", arm="a2")
    with meter.attempt(event_id="evt:x", phase="head", request_kind="head_batch",
                       request_digest="abc", head_calls=1) as receipt:
        receipt.observe(physical_forwards=0)
    event = events_from_journal(tmp_path / "attempts.jsonl")[0]
    assert (event.protein_id, event.arm) == ("9L2Q_A", "a2")


def test_the_meter_measures_walltime_it_does_not_accept_it(tmp_path):
    """Walltime is measurable in process; letting a caller declare it would make the one field
    nothing else can cross-check into a free parameter."""
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import events_from_journal

    meter = _meter(tmp_path)
    with meter.attempt(event_id="evt:x", phase="head", request_kind="head_batch",
                       request_digest="abc", head_calls=1) as receipt:
        receipt.observe(physical_forwards=0)
    assert events_from_journal(tmp_path / "attempts.jsonl")[0].walltime_s >= 0.0


def test_a_gpu_clock_that_is_not_an_instrument_is_refused(tmp_path):
    """The likely mistake is passing a NUMBER -- "the GPU seconds" -- instead of the thing that
    measures them.  That reads as a declaration of cost and would be silently ignored."""
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import CostMeter, V2LedgerError

    with pytest.raises(V2LedgerError, match="gpu_clock"):
        CostMeter(journal=AttemptJournal(tmp_path / "a.jsonl"), protein_id="5ZHV_B", arm="v2",
                  gpu_clock=0.0)


def test_aggregation_is_a_function_of_the_row_SET_not_of_arrival_order():
    """PLAN §5.4 routes the ledger through resume, and fragment arrival order is the scheduler's.

    A batch lane that is not charged for its attempt must be validated against the lane that IS
    charged.  That check only ran when the charged row had already been seen, so the same row set
    raised in one order and silently accepted a contradictory lane in the other -- and the lane's
    number was then dropped without a word.
    """
    def ev(**kw):
        base = dict(event_id="e1", attempt_id="att:1", protein_id="P", arm="v2",
                    phase="screen", status="ok")
        base.update(kw)
        return V2LedgerEvent(**base)

    charged = ev(physical_forwards=150, physical_forwards_charged=True, batch_size=3)
    contradicting = ev(event_id="e2", physical_forwards=999,
                       physical_forwards_charged=False, batch_size=3)
    for order in ([charged, contradicting], [contradicting, charged]):
        with pytest.raises(V2LedgerError):
            aggregate_v2_ledger(order)


def test_a_consistent_batch_aggregates_in_either_order():
    """The guard must not be satisfiable by refusing every batch."""
    def ev(**kw):
        base = dict(event_id="e1", attempt_id="att:1", protein_id="P", arm="v2",
                    phase="screen", status="ok")
        base.update(kw)
        return V2LedgerEvent(**base)

    charged = ev(physical_forwards=150, physical_forwards_charged=True, batch_size=3)
    lane = ev(event_id="e2", physical_forwards=150, physical_forwards_charged=False, batch_size=3)
    assert aggregate_v2_ledger([charged, lane])["physical_forwards"] == \
        aggregate_v2_ledger([lane, charged])["physical_forwards"] == 150
