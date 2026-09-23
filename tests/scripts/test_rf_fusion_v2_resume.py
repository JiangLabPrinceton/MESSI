"""V2F7: content-bound resume and fragment aggregation (PLAN §5.4).

A resume is a claim -- "this was already computed, under exactly these conditions, so I may skip
it".  Every test here attacks a way that claim could be false while still looking true.

The V2F7 acceptance list drives the cases: crash after paid work, retry, shard reorder,
empty/partial cohort, stale input/config, duplicate fragments, and zero-success.
"""

from __future__ import annotations

import json

import pytest

from scripts.rf_fusion_v2_resume import (
    AggregateReport,
    FragmentVerdict,
    RunSignature,
    V2ResumeError,
    aggregate_fragments,
    read_fragment,
    scan_fragments,
    validate_fragment,
    write_fragment,
)

TABLES = ("complete_endpoints", "feedback_events")


def _sig(**over):
    kw = dict(config_digest="c" * 64, campaign_id="v2-canary", split_role="dev", arm_role="v2",
              protein_id="5ZHV_B", input_signature="i" * 64, code_revision="deadbeef")
    kw.update(over)
    return RunSignature(**kw)


def _write(tmp_path, signature, *, status="ok", rows=1, name=None):
    path = tmp_path / f"{name or signature.protein_id}.json"
    write_fragment(path, signature=signature, status=status, payload={
        "complete_endpoints": [{"endpoint_id": f"{signature.protein_id}:{i}"}
                               for i in range(rows)],
        "feedback_events": [],
        "ledger_events": [],
    })
    return path


# --------------------------------------------------------------------------------------------
# exact run-signature equality
# --------------------------------------------------------------------------------------------


def test_a_fragment_from_the_same_run_is_reusable(tmp_path):
    path = _write(tmp_path, _sig())
    assert validate_fragment(path, expected=_sig()).accepted


@pytest.mark.parametrize("field,value,status", [
    ("config_digest", "d" * 64, "stale_config"),
    ("input_signature", "j" * 64, "stale_input"),
    ("code_revision", "cafebabe", "stale_code"),
    ("campaign_id", "other-campaign", "foreign_run"),
    ("split_role", "test", "foreign_run"),
    ("arm_role", "a2", "foreign_run"),
    ("exploratory_depth_override", True, "foreign_run"),
    ("production_depth_authorized", True, "foreign_run"),
])
def test_a_changed_scientific_condition_makes_paid_work_unreusable(tmp_path, field, value, status):
    """Each of these names a DIFFERENT operator action, so each gets its own status.

    A foreign run needs a different command; a stale config needs a decision about the science; a
    stale code revision needs a rebuild.  Reporting all three as "skipped" would make them look
    like the same problem, and the most dangerous one -- silently mixing two experiments' results
    into one table -- is the one that looks most benign.
    """
    path = _write(tmp_path, _sig())
    verdict = validate_fragment(path, expected=_sig(**{field: value}))
    assert verdict.accepted is False
    assert verdict.status == status


def test_a_fragment_whose_digest_disagrees_with_its_own_fields_is_corrupt(tmp_path):
    """Tampering, not staleness: the two would otherwise be reported the same way."""
    path = _write(tmp_path, _sig())
    data = read_fragment(path)
    data["run_sig"] = "0" * 64
    path.write_text(json.dumps(data, sort_keys=True))
    verdict = validate_fragment(path, expected=_sig())
    assert verdict.status == "corrupt"
    assert "digest" in verdict.detail


def test_a_fragment_with_no_signature_at_all_is_refused(tmp_path):
    """Work that cannot name its own conditions may never be reused."""
    path = tmp_path / "5ZHV_B.json"
    path.write_text(json.dumps({"payload": {"complete_endpoints": []}}))
    assert validate_fragment(path, expected=_sig()).status == "corrupt"


def test_a_truncated_fragment_is_refused_rather_than_partially_parsed(tmp_path):
    """A shard killed mid-write must not look like completed work."""
    path = tmp_path / "5ZHV_B.json"
    path.write_text('{"fragment_schema": "v2frag-1", "signat')
    assert validate_fragment(path, expected=_sig()).status == "corrupt"


def test_the_fragment_write_is_atomic(tmp_path):
    """The temp file must never be left behind as something a later scan would read."""
    _write(tmp_path, _sig())
    assert [p.name for p in tmp_path.iterdir()] == ["5ZHV_B.json"]


# --------------------------------------------------------------------------------------------
# the V2F7 acceptance cases
# --------------------------------------------------------------------------------------------


def test_shard_order_does_not_change_the_aggregate(tmp_path):
    """Shards finish in whatever order the scheduler gives them.

    An aggregate that depended on arrival order would make a re-run produce a different artifact
    from the same evidence, and nothing downstream could be compared across runs.
    """
    proteins = ["A_1", "B_2", "C_3"]
    expected = {p: _sig(protein_id=p) for p in proteins}
    for p in proteins:
        _write(tmp_path, expected[p], rows=2)

    first = aggregate_fragments(tmp_path, requested_cohort=proteins,
                                expected_by_protein=expected, table_names=TABLES)
    reversed_cohort = list(reversed(proteins))
    second = aggregate_fragments(tmp_path, requested_cohort=reversed_cohort,
                                 expected_by_protein=expected, table_names=TABLES)
    assert first.tables == second.tables
    assert first.n_ok == second.n_ok == 3


def test_a_partial_cohort_names_the_proteins_that_are_missing(tmp_path):
    """A cohort that quietly aggregated only the shards that finished would publish a partial
    result under the whole cohort's name, and every rate computed from it would be conditioned on
    success."""
    proteins = ["A_1", "B_2", "C_3"]
    expected = {p: _sig(protein_id=p) for p in proteins}
    _write(tmp_path, expected["A_1"])
    report = aggregate_fragments(tmp_path, requested_cohort=proteins,
                                 expected_by_protein=expected, table_names=TABLES)
    assert report.missing_proteins == ("B_2", "C_3")
    assert report.complete is False
    assert report.any_success is True


def test_an_empty_cohort_directory_yields_an_empty_but_well_formed_report(tmp_path):
    report = aggregate_fragments(tmp_path, requested_cohort=["A_1"],
                                 expected_by_protein={"A_1": _sig(protein_id="A_1")},
                                 table_names=TABLES)
    assert isinstance(report, AggregateReport)
    assert report.accepted == () and report.n_ok == 0
    assert report.missing_proteins == ("A_1",)
    assert report.any_success is False
    assert report.tables == {name: [] for name in TABLES}


def test_a_byte_identical_duplicate_fragment_is_idempotent(tmp_path):
    """A shard that re-ran and wrote the same answer is not a problem to report."""
    signature = _sig()
    _write(tmp_path, signature, name="5ZHV_B")
    _write(tmp_path, signature, name="5ZHV_B__retry")
    report = aggregate_fragments(tmp_path, requested_cohort=["5ZHV_B"],
                                 expected_by_protein={"5ZHV_B": signature}, table_names=TABLES)
    assert report.rejected == ()
    assert len(report.tables["complete_endpoints"]) == 1, "the duplicate was aggregated twice"


def test_two_conflicting_fragments_for_one_protein_are_refused(tmp_path):
    """Keeping either would be a coin flip about which run's result the cohort reports."""
    signature = _sig()
    _write(tmp_path, signature, rows=1, name="5ZHV_B")
    _write(tmp_path, signature, rows=5, name="5ZHV_B__second")
    report = aggregate_fragments(tmp_path, requested_cohort=["5ZHV_B"],
                                 expected_by_protein={"5ZHV_B": signature}, table_names=TABLES)
    assert [v.status for v in report.rejected] == ["duplicate_conflict"]
    assert report.complete is False


def test_a_stale_fragment_is_reported_not_silently_skipped(tmp_path):
    """The whole difference from the V1 aggregator.

    A bare ``continue`` on a stale checkpoint means a cohort with every fragment stale is
    indistinguishable from a cohort that was never run -- both aggregate to nothing, and only one
    of them is a configuration error someone must fix.
    """
    _write(tmp_path, _sig())
    report = aggregate_fragments(tmp_path, requested_cohort=["5ZHV_B"],
                                 expected_by_protein={"5ZHV_B": _sig(config_digest="d" * 64)},
                                 table_names=TABLES)
    assert [v.status for v in report.rejected] == ["stale_config"]
    assert report.missing_proteins == ("5ZHV_B",)
    assert report.complete is False


def test_a_fragment_for_a_protein_outside_the_cohort_is_reported(tmp_path):
    """A stray fragment from another run in the same directory must not be absorbed."""
    _write(tmp_path, _sig(protein_id="STRAY_X"))
    report = aggregate_fragments(tmp_path, requested_cohort=["5ZHV_B"],
                                 expected_by_protein={"5ZHV_B": _sig()}, table_names=TABLES)
    assert [v.status for v in report.rejected] == ["foreign_run"]


def test_zero_success_is_distinguishable_from_no_work(tmp_path):
    """PLAN V2F7 acceptance: "all proteins fail but the driver exits zero" is a case to prevent.

    A run where every protein was processed and every one failed has fragments, no missing
    proteins, and no successes.  A run that never started has none of the three.  The report must
    tell them apart, or the driver cannot pick different exit codes for them.
    """
    proteins = ["A_1", "B_2"]
    expected = {p: _sig(protein_id=p) for p in proteins}
    for p in proteins:
        _write(tmp_path, expected[p], status="failed", rows=0)
    report = aggregate_fragments(tmp_path, requested_cohort=proteins,
                                 expected_by_protein=expected, table_names=TABLES)
    assert report.missing_proteins == ()
    assert report.rejected == ()
    assert report.n_ok == 0
    assert report.any_success is False
    assert report.complete is True, "every requested protein DID produce a valid fragment"


def test_paid_work_survives_a_crash_and_is_not_recomputed(tmp_path):
    """A fragment written before the crash is accepted by the resume, so the tail is not re-paid."""
    signature = _sig()
    _write(tmp_path, signature, rows=3)
    # A second process starts with the same signature and finds the work already there.
    report = aggregate_fragments(tmp_path, requested_cohort=["5ZHV_B"],
                                 expected_by_protein={"5ZHV_B": signature}, table_names=TABLES)
    assert report.complete is True
    assert len(report.tables["complete_endpoints"]) == 3


def test_a_duplicate_protein_in_the_requested_cohort_is_a_caller_error(tmp_path):
    with pytest.raises(V2ResumeError, match="duplicate protein"):
        aggregate_fragments(tmp_path, requested_cohort=["A_1", "A_1"],
                            expected_by_protein={"A_1": _sig(protein_id="A_1")})


def test_scan_is_canonically_ordered(tmp_path):
    """So a report is a function of the fragment SET, not of filesystem iteration order."""
    expected = {p: _sig(protein_id=p) for p in ("A_1", "B_2", "C_3")}
    for name in ("C_3", "A_1", "B_2"):
        _write(tmp_path, expected[name])
    verdicts = scan_fragments(tmp_path, expected_by_protein=expected)
    assert [v.fragment_id for v in verdicts] == ["A_1", "B_2", "C_3"]


def test_a_verdict_status_outside_the_closed_set_is_refused():
    with pytest.raises(V2ResumeError, match="status must be one of"):
        FragmentVerdict("p", "f", "sort_of_ok")


# --------------------------------------------------------------------------------------------
# a fragment is bound to its own CONTENT, not only to the run that wrote it
# --------------------------------------------------------------------------------------------


def test_a_result_status_outside_the_closed_vocabulary_cannot_be_written(tmp_path):
    """``running`` is the dangerous one.

    A half-finished shard that named itself ``running`` would be accepted by a resume and skipped
    forever, so the cohort would report a protein as done that no process ever finished.  The
    vocabulary is closed at the WRITE boundary so such a fragment cannot come into existence.
    """
    with pytest.raises(V2ResumeError, match="result status must be one of"):
        write_fragment(tmp_path / "5ZHV_B.json", signature=_sig(), status="running", payload={})
    assert list(tmp_path.iterdir()) == [], "a refused write still left a fragment on disk"


def test_a_fragment_claiming_an_unknown_result_status_is_never_reused(tmp_path):
    """Hand-written or produced by other code: an unreadable outcome is not a reusable outcome."""
    path = _write(tmp_path, _sig())
    data = read_fragment(path)
    data["status"] = "running"
    path.write_text(json.dumps(data, sort_keys=True))
    verdict = validate_fragment(path, expected=_sig())
    assert verdict.status == "corrupt"
    assert "running" in verdict.detail


def test_an_edited_payload_is_refused_even_though_the_signature_still_matches(tmp_path):
    """``run_sig`` digests the run's CONDITIONS; it says nothing about the result.

    Without a result digest, an endpoint table can be edited to say anything at all and the
    fragment still validates -- the resume would then reuse a number no run ever produced.
    """
    path = _write(tmp_path, _sig(), rows=1)
    data = read_fragment(path)
    data["payload"]["complete_endpoints"] = [{"endpoint_id": "fabricated"}]
    path.write_text(json.dumps(data, sort_keys=True))
    verdict = validate_fragment(path, expected=_sig())
    assert verdict.status == "corrupt"
    assert "payload" in verdict.detail or "result" in verdict.detail


def test_a_failure_edited_into_a_success_is_refused(tmp_path):
    """The status is part of the result, so it is digested with it.

    Flipping ``failed`` to ``ok`` is the cheapest possible way to manufacture a cohort success rate.
    """
    path = _write(tmp_path, _sig(), status="failed", rows=0)
    data = read_fragment(path)
    data["status"] = "ok"
    path.write_text(json.dumps(data, sort_keys=True))
    assert validate_fragment(path, expected=_sig()).status == "corrupt"


def test_a_fragment_with_no_result_digest_at_all_is_refused(tmp_path):
    """Written by an older writer that bound no result: it cannot prove what it contains."""
    path = _write(tmp_path, _sig())
    data = read_fragment(path)
    del data["result_sig"]
    path.write_text(json.dumps(data, sort_keys=True))
    assert validate_fragment(path, expected=_sig()).status == "corrupt"


def test_an_unknown_fragment_schema_version_is_never_accepted(tmp_path):
    """``fragment_schema`` was written and never read, which is the same as not having one.

    A fragment from a different writer version may lay out its payload differently; reading it under
    this version's assumptions is how a table silently changes meaning between runs.
    """
    path = _write(tmp_path, _sig())
    data = read_fragment(path)
    data["fragment_schema"] = "v2frag-2"
    path.write_text(json.dumps(data, sort_keys=True))
    verdict = validate_fragment(path, expected=_sig())
    assert verdict.status == "corrupt"
    assert "v2frag-2" in verdict.detail


@pytest.mark.parametrize("payload", [
    {"complete_endpoints": [["endpoint_id"]]},  # rows must be mappings, not positional lists
    {"complete_endpoints": [None]},
    {"complete_endpoints": [{"endpoint_id": "x"}, "not_a_row"]},
])
def test_a_payload_that_is_not_table_shaped_is_refused_at_the_write_boundary(tmp_path, payload):
    """``aggregate_fragments`` EXTENDS its tables from these lists.

    A positional row would be written under whatever column happened to align, so a malformed
    payload does not fail -- it publishes.  Refused where it is produced, so the error names the
    runner that returned it rather than a fragment nobody can explain later.
    """
    with pytest.raises(V2ResumeError, match="table row|mapping"):
        write_fragment(tmp_path / "5ZHV_B.json", signature=_sig(), status="ok", payload=payload)
    assert list(tmp_path.iterdir()) == []


def test_a_table_key_holding_a_string_is_refused_before_it_is_aggregated(tmp_path):
    """A str is iterable, so ``extend()`` would spread it one character per row.

    Whether a key is a TABLE is knowable only from the declared table names, so this is checked
    where those names are known rather than guessed at the write boundary.
    """
    path = _write(tmp_path, _sig())
    data = read_fragment(path)
    data["payload"]["complete_endpoints"] = "abc"
    path.write_text(json.dumps(data, sort_keys=True))
    verdict = validate_fragment(path, expected=_sig(), table_names=TABLES)
    assert verdict.status == "corrupt"
    assert "complete_endpoints" in verdict.detail


def test_a_verdict_carries_the_fragments_own_result_status(tmp_path):
    """Admissible and successful are different questions.

    A ``failed`` fragment is perfectly valid evidence -- of a failure.  A driver that reads only
    ``accepted`` would skip that protein forever instead of retrying it, so the verdict must report
    both facts separately.
    """
    ok = validate_fragment(_write(tmp_path, _sig(), name="ok_one"), expected=_sig())
    assert ok.accepted and ok.result_status == "ok" and ok.records_success is True

    failed = validate_fragment(
        _write(tmp_path, _sig(), status="failed", rows=0, name="failed_one"), expected=_sig())
    assert failed.accepted is True, "a recorded failure is still valid evidence"
    assert failed.result_status == "failed"
    assert failed.records_success is False, "a failed shard must be retried, not skipped"


def test_a_rejected_verdict_records_no_result_status(tmp_path):
    """Nothing may be inferred about the outcome of a fragment that was not admitted."""
    path = _write(tmp_path, _sig())
    verdict = validate_fragment(path, expected=_sig(config_digest="d" * 64))
    assert verdict.status == "stale_config"
    assert verdict.result_status is None and verdict.records_success is False
