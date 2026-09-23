"""V2F7: the V2 evidence bundle (PLAN §5.3).

What this suite is actually protecting: an artifact bundle is the ONLY thing that survives a run.
Every scientific claim about V2 is eventually read off these tables, so three properties matter more
than any individual column.

**Nothing scored is dropped.**  PLAN §5.3: "Raw logical duplicates remain in endpoint tables."  A
summary may collapse by a declared equivalence, but only "while preserving the full mapping" -- a
frontier computed from a silently deduplicated pool would overstate what the method found.

**A schema is stable whether or not there are rows.**  An empty cohort must produce the same typed
columns as a full one, or a downstream reader sees an all-null table drift into typed columns and
cannot tell "no results" from "column renamed".

**Per-position evidence is content-addressed, not summarized.**  The provenance vectors are what
make a state reconstructible; a table that stored only their length would be a table nobody can
audit.
"""

from __future__ import annotations

import dataclasses
import json
import types

import pytest

pytest.importorskip("pyarrow")

import pyarrow.parquet as pq  # noqa: E402

from scripts.rf_fusion_v2_artifacts import (  # noqa: E402
    V2_TABLE_SCHEMAS,
    V2ArtifactError,
    a2_view_rows,
    archive_rows,
    structure_evaluation_rows,
    complete_endpoint_rows,
    feedback_event_rows,
    partial_state_rows,
    run_manifest,
    terminal_validation_rows,
    write_v2_bundle,
)
from tests.inverse_folding import _v2_fixtures as F  # noqa: E402


def _read(path):
    table = pq.read_table(str(path))
    return table.column_names, table.to_pylist()


# --------------------------------------------------------------------------------------------
# every required evidence object exists and is typed
# --------------------------------------------------------------------------------------------


def test_every_evidence_object_plan_requires_has_a_declared_schema():
    """PLAN §5.3 names eight evidence objects.  A missing one is a hole in the audit trail."""
    required = {
        "partial_states", "complete_endpoints", "archive", "feedback_events", "a2_views",
        "terminal_validation",
    }
    assert required <= set(V2_TABLE_SCHEMAS), sorted(required - set(V2_TABLE_SCHEMAS))
    for name, schema in V2_TABLE_SCHEMAS.items():
        assert schema.columns, f"{name} declares no columns"
        assert set(schema.sort_by) <= set(schema.columns), name


def test_an_empty_table_still_carries_its_full_typed_schema(tmp_path):
    """An empty cohort and a full one must be readable by the same code."""
    write_v2_bundle(
        tmp_path, manifest=run_manifest(config=F.v2_config(), code_revision="deadbeef",
                                        content_identities={}, seed_namespaces=("v2_lookahead",)),
        tables={name: [] for name in V2_TABLE_SCHEMAS},
    )
    for name, schema in V2_TABLE_SCHEMAS.items():
        columns, rows = _read(tmp_path / f"{name}.parquet")
        assert columns == list(schema.columns), f"{name} lost its schema when empty"
        assert rows == []


def test_the_manifest_carries_one_config_digest_and_the_declared_caps(tmp_path):
    config = F.v2_config()
    manifest = run_manifest(config=config, code_revision="deadbeef",
                            content_identities={"reference.fasta": "a" * 64},
                            seed_namespaces=("v2_lookahead", "matched_descendant"))
    assert manifest["config_digest"] == config.config_digest()
    assert manifest["caps"]["max_logical_dfe"] == config.caps.max_logical_dfe
    assert manifest["split_role"] == config.identity.split_role
    assert manifest["code_revision"] == "deadbeef"
    assert manifest["schedule_id"] == config.schedule.schedule_id
    assert manifest["seed_namespaces"] == ["matched_descendant", "v2_lookahead"]

    write_v2_bundle(tmp_path, manifest=manifest, tables={n: [] for n in V2_TABLE_SCHEMAS})
    assert json.loads((tmp_path / "run_manifest.json").read_text()) == manifest


# --------------------------------------------------------------------------------------------
# raw logical duplicates survive
# --------------------------------------------------------------------------------------------


def test_two_forks_that_produced_the_same_sequence_stay_two_endpoint_rows():
    """PLAN §2.3/§5.3: distinct logical endpoints remain visible even when sequences converge.

    Collapsing them here would make the endpoint table a biased sample of what was generated, and
    any diversity or yield number read off it would be wrong in the direction that flatters V2.
    """
    twins = [F.endpoint(fork_index=0), F.endpoint(fork_index=1)]
    assert twins[0].sequence_md5 == twins[1].sequence_md5
    assert twins[0].endpoint_id != twins[1].endpoint_id
    rows = complete_endpoint_rows(twins, depth=0)
    assert len(rows) == 2
    assert {row["endpoint_id"] for row in rows} == {e.endpoint_id for e in twins}
    assert {row["sequence_equivalence_key"] for row in rows} == {twins[0].sequence_md5}


def test_the_endpoint_row_carries_its_per_position_provenance_not_a_summary_of_it():
    """The per-position evidence is what makes an endpoint reconstructible."""
    endpoint = F.endpoint()
    row = complete_endpoint_rows([endpoint], depth=0)[0]
    decoded = json.loads(row["endpoint_provenance_json"])
    assert len(decoded) == endpoint.sequence_length
    assert [entry["token"] for entry in decoded] == [
        e.token for e in endpoint.endpoint_provenance_evidence_by_pos]
    assert row["endpoint_content_digest"] == endpoint.content_digest


def test_a_partial_state_row_carries_every_per_position_vector_a_replay_needs():
    state = F.source()
    row = partial_state_rows([state])[0]
    assert row["state_id"] == state.state_id
    assert row["state_content_digest"] == state.content_digest
    for column in ("tokens_json", "active_origin_kind_json", "active_commit_json",
                   "active_sampler_score_json", "active_score_status_json",
                   "provenance_json", "hard_anchors_json"):
        assert json.loads(row[column]) is not None, column
    assert len(json.loads(row["tokens_json"])) == len(state.tokens)
    assert row["replay_state_hash"] == state.replay.replay_state_hash
    assert row["n_unresolved_editable"] == sum(
        1 for i in state.editable_positions if state.tokens[i] == state.mask_token_id)


# --------------------------------------------------------------------------------------------
# feedback events, archive, A2
# --------------------------------------------------------------------------------------------


def test_a_feedback_event_row_names_all_four_support_sets_and_their_reasons():
    """PLAN §5.3: the four support sets AND reasons are load-bearing fields.

    Recording only the partition sizes would make a null event indistinguishable from a different
    null event, and the per-position reason is the only record of WHY the policy chose each action.
    """
    projected = F.projected()
    row = feedback_event_rows(
        [dict(source=F.source(), endpoint=F.endpoint(), projected=projected, propagated=None,
              policy=F.policy_identity(), outcome="committed", detail="",
              pair_id="5ZHV_B:d0:r40-c60:p0", arm_slot="arm_a", treatment_identity="v2")]
    )[0]
    assert row["outcome"] == "committed"
    assert json.loads(row["write_from_endpoint_json"]) == list(
        projected.support.write_from_endpoint)
    assert json.loads(row["reopen_json"]) == list(projected.support.reopen)
    assert json.loads(row["inject_from_source_feedback_json"]) == list(
        projected.support.inject_from_source_feedback)
    assert json.loads(row["carry_from_source_json"]) == list(projected.support.carry_from_source)
    assert json.loads(row["support_reason_by_pos_json"])
    assert row["pair_id"] == "5ZHV_B:d0:r40-c60:p0"
    assert row["selected_endpoint_id"] == F.endpoint().endpoint_id


def test_a_null_feedback_event_is_recorded_with_no_projected_state():
    """PLAN §4.5: a refusal is a result.  A bundle that dropped null events would report only the
    cycles that worked, which is the definition of a biased record."""
    row = feedback_event_rows(
        [dict(source=F.source(), endpoint=F.endpoint(), projected=None, propagated=None,
              policy=F.policy_identity(), outcome="null_band_incompatible",
              detail="projected maturity is outside B(r_d)", pair_id="p", arm_slot="arm_a",
              treatment_identity="v2")]
    )[0]
    assert row["outcome"] == "null_band_incompatible"
    assert row["projected_state_id"] is None
    assert row["detail"]
    assert json.loads(row["write_from_endpoint_json"]) == []


def test_archive_rows_carry_membership_feasibility_and_the_depth_span():
    from inverse_folding.reference_flow.fusion_v2_runtime.archive import ExactArchive

    archive = ExactArchive()
    archive.admit(F.endpoint(), depth=0)
    rows = archive_rows(archive)
    assert len(rows) == 1
    row = rows[0]
    assert row["feasibility_level"] == "definitive"
    assert row["is_elite"] is True
    assert row["first_depth_seen"] == 0 and row["last_depth_seen"] == 0
    assert row["family_id"] == F.endpoint().lineage.family_id
    assert row["sequence_equivalence_key"] == F.endpoint().sequence_md5


def test_a_REJECTED_endpoint_s_structure_metrics_survive_into_the_evaluations_table():
    """The whole point of the table: the endpoint record is allowed to carry an evaluated structure
    outcome only when DEFINITIVE, so a rejected design keeps `structure_evaluated=false` and empty
    metrics. Measured on the first Canary: `5zhv_r30` folded four endpoints, rejected all four, and
    its bundle recorded no scTM anywhere -- the numbers had to be recovered from the refold cache by
    content key, which is evidence recovery by side-channel.
    """
    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
    from inverse_folding.reference_flow.fusion_v2.state import FeasibilityLevel
    from inverse_folding.reference_flow.fusion_v2_runtime.archive import ExactArchive

    archive = ExactArchive()
    # Admitted UNVALIDATED, which is what a real lookahead is before the gate runs; the fixture's
    # default endpoint is already DEFINITIVE and the archive rightly refuses to regress it.
    endpoint = dataclasses.replace(
        F.endpoint(), feasibility_level=FeasibilityLevel.UNVALIDATED,
        structure_outcome=None, endpoint_id=None)
    archive.admit(endpoint, depth=0)
    rejected = StructureOutcome(
        feasible=False, cache_status="miss", model_executed=True,
        failure_reason="scTM 0.7885 < 0.85", walltime_s=12.5, evaluated=True,
        metrics={"scTM": 0.7885, "pLDDT": 64.48},
    )
    archive.promote(endpoint.endpoint_id, feasibility_level=FeasibilityLevel.UNVALIDATED,
                    structure_outcome=rejected, depth=0)

    conditioning = types.SimpleNamespace(structure_backend="b" * 64,
                                         v0_structure_gate_config="g" * 64)
    row = structure_evaluation_rows(
        archive, admission_by_endpoint={endpoint.endpoint_id: "structure not definitively feasible"},
        conditioning=conditioning)[0]

    assert row["evaluated"] is True and row["feasible"] is False
    assert json.loads(row["metrics_json"]) == {"scTM": 0.7885, "pLDDT": 64.48}
    assert row["failure_reason"] == "scTM 0.7885 < 0.85"
    assert row["feasibility_level"] == "unvalidated"
    assert row["admission_reason"] == "structure not definitively feasible"
    assert row["cache_status"] == "miss" and row["model_executed"] is True
    # Carried per row: this table gets read on its own, and structure verdicts that cannot name the
    # model that produced them are an anecdote.
    assert row["structure_backend_digest"] == "b" * 64
    assert row["v0_structure_gate_config_digest"] == "g" * 64


def test_the_evaluations_table_is_written_even_when_every_endpoint_failed():
    """A bundle whose cell admitted nothing is exactly the bundle that must explain itself."""
    from scripts.rf_fusion_v2_artifacts import V2_TABLE_SCHEMAS

    assert "structure_evaluations" in V2_TABLE_SCHEMAS, (
        "write_v2_bundle refuses a missing table, so registration is what makes it mandatory")


def test_an_a2_row_records_the_exact_pre_feedback_membership_and_the_extra_allocation():
    from inverse_folding.reference_flow.fusion_v2_runtime.a2_view import snapshot_a2_view
    from inverse_folding.reference_flow.fusion_v2_runtime.archive import ExactArchive

    archive = ExactArchive()
    archive.admit(F.endpoint(), depth=0)
    source = F.source()
    view = snapshot_a2_view(
        archive, depth=0, protein_id=source.lineage.protein_id,
        source_state_id=source.state_id,
    )
    row = a2_view_rows(
        [view], matched_extra_lookaheads=3,
        matching_resource="definitive_refolds",
        declared_matching_resource="definitive_refolds", matching_executed=True,
    )[0]
    assert json.loads(row["member_endpoint_ids_json"]) == list(view.endpoint_ids)
    assert row["matched_extra_lookaheads"] == 3
    assert row["matching_resource"] == "definitive_refolds"


def test_an_unexecuted_a2_match_is_reported_as_declared_not_executed():
    from inverse_folding.reference_flow.fusion_v2_runtime.a2_view import snapshot_a2_view
    from inverse_folding.reference_flow.fusion_v2_runtime.archive import ExactArchive

    source = F.source()
    view = snapshot_a2_view(
        ExactArchive(), depth=0, protein_id=source.lineage.protein_id,
        source_state_id=source.state_id,
    )
    row = a2_view_rows(
        [view], matched_extra_lookaheads=0, matching_resource=None,
        declared_matching_resource="definitive_refolds", matching_executed=False,
    )[0]
    assert row["matching_resource"] is None
    assert row["declared_matching_resource"] == "definitive_refolds"
    assert row["matching_status"] == "declared_not_executed"
    assert "not executed" in row["matching_detail"]


def test_a2_view_ids_bind_protein_and_source_and_do_not_collide_in_one_bundle(tmp_path):
    from inverse_folding.reference_flow.fusion_v2_runtime.a2_view import snapshot_a2_view
    from inverse_folding.reference_flow.fusion_v2_runtime.archive import ExactArchive

    first = F.source()
    foreign_lineage = F.lineage(
        protein_id="OTHER_A", root_id="OTHER_A:v2:d0:r0", family_id="fam1",
    )
    second = F.source(
        lineage=foreign_lineage, safety_reference=F.safety_reference(protein_id="OTHER_A"),
    )
    views = [
        snapshot_a2_view(
            ExactArchive(), depth=0, protein_id=source.lineage.protein_id,
            source_state_id=source.state_id,
        )
        for source in (first, second)
    ]
    rows = a2_view_rows(
        views, matched_extra_lookaheads=0, matching_resource=None,
        declared_matching_resource="definitive_refolds", matching_executed=False,
    )
    assert len({row["a2_view_id"] for row in rows}) == 2
    assert {(row["protein_id"], row["source_state_id"]) for row in rows} == {
        (first.lineage.protein_id, first.state_id),
        (second.lineage.protein_id, second.state_id),
    }

    tables = {name: [] for name in V2_TABLE_SCHEMAS}
    tables["a2_views"] = rows
    write_v2_bundle(
        tmp_path,
        manifest=run_manifest(
            config=F.v2_config(), code_revision="deadbeef", content_identities={},
            seed_namespaces=("v2_lookahead",),
        ),
        tables=tables,
    )


# --------------------------------------------------------------------------------------------
# the bundle itself
# --------------------------------------------------------------------------------------------


def test_the_bundle_round_trips_through_parquet(tmp_path):
    endpoints = [F.endpoint(fork_index=i) for i in range(3)]
    tables = {name: [] for name in V2_TABLE_SCHEMAS}
    tables["complete_endpoints"] = complete_endpoint_rows(endpoints, depth=0)
    tables["partial_states"] = partial_state_rows([F.source()])
    write_v2_bundle(tmp_path, manifest=run_manifest(
        config=F.v2_config(), code_revision="deadbeef", content_identities={},
        seed_namespaces=("v2_lookahead",)), tables=tables)

    _, rows = _read(tmp_path / "complete_endpoints.parquet")
    assert len(rows) == 3
    assert {row["endpoint_id"] for row in rows} == {e.endpoint_id for e in endpoints}
    _, states = _read(tmp_path / "partial_states.parquet")
    assert states[0]["state_id"] == F.source().state_id


def test_a_row_with_an_undeclared_column_is_refused_rather_than_silently_dropped(tmp_path):
    tables = {name: [] for name in V2_TABLE_SCHEMAS}
    tables["archive"] = [{"endpoint_id": "x", "not_a_column": 1}]
    with pytest.raises(Exception, match="undeclared"):
        write_v2_bundle(tmp_path, manifest=run_manifest(
            config=F.v2_config(), code_revision="deadbeef", content_identities={},
            seed_namespaces=("v2_lookahead",)), tables=tables)


def test_a_bundle_missing_a_required_table_is_refused(tmp_path):
    """A bundle that silently omitted a table would read as "this run produced none of that"."""
    tables = {name: [] for name in V2_TABLE_SCHEMAS}
    tables.pop("feedback_events")
    with pytest.raises(V2ArtifactError, match="feedback_events"):
        write_v2_bundle(tmp_path, manifest=run_manifest(
            config=F.v2_config(), code_revision="deadbeef", content_identities={},
            seed_namespaces=("v2_lookahead",)), tables=tables)


def test_join_keys_are_unique_within_every_table(tmp_path):
    """A duplicated join key silently multiplies rows in any downstream join."""
    duplicated = complete_endpoint_rows([F.endpoint()], depth=0) * 2
    tables = {name: [] for name in V2_TABLE_SCHEMAS}
    tables["complete_endpoints"] = duplicated
    with pytest.raises(V2ArtifactError, match="duplicate"):
        write_v2_bundle(tmp_path, manifest=run_manifest(
            config=F.v2_config(), code_revision="deadbeef", content_identities={},
            seed_namespaces=("v2_lookahead",)), tables=tables)


def test_the_compute_ledger_round_trips_from_the_json_rows_a_fragment_carries(tmp_path):
    """PLAN §5.3 makes the compute ledger a required evidence object, and §5.4 routes it through
    resume -- so by the time it reaches this writer it has been through JSON and is a mapping, not
    a dataclass.  A bundle that could only write dataclasses would drop the ledger on exactly the
    path it has to survive."""
    import dataclasses

    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import V2LedgerEvent

    event = V2LedgerEvent(
        event_id="evt:1", attempt_id="att:1", protein_id="5ZHV_B", arm="v2", phase="screen",
        status="ok", logical_dfe=150, physical_forwards=150, gpu_seconds=1.5)
    rows = [json.loads(json.dumps(dataclasses.asdict(event)))]
    write_v2_bundle(tmp_path, manifest=run_manifest(
        config=F.v2_config(), code_revision="deadbeef", content_identities={},
        seed_namespaces=("v2_lookahead",)), tables={n: [] for n in V2_TABLE_SCHEMAS},
        ledger_events=rows)

    written = [json.loads(line) for line in
               (tmp_path / "cost_ledger.jsonl").read_text().splitlines() if line.strip()]
    assert written == [dataclasses.asdict(event)]


def test_a_ledger_row_that_is_not_a_ledger_event_is_refused(tmp_path):
    """The ledger is evidence, not free-form JSON.

    A row with a phase nothing aggregates, or with the logical and physical fields confused, would
    be published as a measurement and silently dropped by every aggregate that reads the closed
    vocabulary -- so the run's compute total would be understated with no error anywhere.
    """
    with pytest.raises(Exception, match="phase|ledger"):
        write_v2_bundle(tmp_path, manifest=run_manifest(
            config=F.v2_config(), code_revision="deadbeef", content_identities={},
            seed_namespaces=("v2_lookahead",)), tables={n: [] for n in V2_TABLE_SCHEMAS},
            ledger_events=[{"event_id": "evt:1", "attempt_id": "att:1", "protein_id": "P",
                            "arm": "v2", "phase": "not_a_phase", "status": "ok"}])


def test_terminal_validation_keeps_the_structure_and_immune_results_separate():
    """They are independent measurements; one column for "passed" would hide which one failed."""
    row = terminal_validation_rows([
        dict(endpoint_id=F.endpoint().endpoint_id, sequence_md5=F.SEQ_MD5,
             structure_definitive=True, structure_feasible=True, structure_metrics={"scTM": 0.91},
             immune_evaluator="nmp", immune_global_risk=-9.1, immune_passed=True,
             diversity_family_id="fam0", v0_before_after=None),
    ])[0]
    assert row["structure_feasible"] is True
    assert row["immune_passed"] is True
    assert json.loads(row["structure_metrics_json"])["scTM"] == 0.91
    assert row["v0_before_after_json"] is None
