"""Typed Fusion V2 artifact schemas, stable writers, and evidence tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python scripts/...` as well as `import scripts...`
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion_v2.errors import V2Error  # noqa: E402
from inverse_folding.reference_flow.fusion_v2.identity import canonical_digest  # noqa: E402

# REUSE, not reimplementation: the V1 writers already give an explicit Arrow schema, declared
# column order, a stable sort, atomic manifest writes and the append-safe JSONL ledger format.
# V2 adds its own column vocabulary through ``write_stable_parquet(types=...)`` -- an argument, not
# a second parquet implementation (``AGENTS.md`` §5 reuse rule).
from scripts.rf_artifact_io import (  # noqa: E402,F401
    TableSchema,
    write_cost_ledger_jsonl,
    write_manifest,
    write_stable_parquet,
)

__all__ = [
    "V2ArtifactError",
    "V2_TABLE_SCHEMAS",
    "DUAL_TABLE_SCHEMAS",
    "SCORABLE_CONTRAST_VIEWS",
    "POLICY_QUALIFICATION_CONTRAST_VIEWS",
    "V2_COLUMN_TYPES",
    "DUAL_COLUMN_TYPES",
    "dual_terminal_summary_rows",
    "dual_feedback_evidence_rows",
    "dual_endpoint_evidence_rows",
    "run_manifest",
    "partial_state_rows",
    "complete_endpoint_rows",
    "archive_rows",
    "structure_evaluation_rows",
    "feedback_event_rows",
    "a2_view_rows",
    "terminal_validation_rows",
    "MECHANISM_CONTRAST_VIEWS",
    "free_domain",
    "mechanism_contrast_rows",
    "write_v2_bundle",
]


class V2ArtifactError(V2Error):
    """A bundle contract violation: a missing table, a duplicate join key, a malformed row."""


def _schema(columns: Sequence[str], sort_by: Sequence[str]) -> TableSchema:
    return TableSchema(columns=tuple(columns), sort_by=tuple(sort_by))


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


#: The frozen V2 table registry.  One entry per PLAN §5.3 evidence object except the run manifest
#: (JSON, not tabular) and the compute ledger (JSONL, written by the ledger module).
V2_TABLE_SCHEMAS: dict[str, TableSchema] = {
    "partial_states": _schema(
        ["state_id", "state_content_digest", "layer", "protein_id", "root_id", "family_id",
         "depth", "parent_state_id", "parent_transition_id", "origin_endpoint_id",
         "sampler_step", "r_step", "c_next_step", "n_steps", "coordinate_law",
         "length", "n_editable", "n_unresolved_editable", "rho_edit",
         "tokens_json", "mask_token_id", "hard_anchors_json", "editable_positions_json",
         "active_origin_kind_json", "feedback_origin_ref_json", "active_commit_json",
         "origin_transition_id_json", "active_sampler_score_json", "active_score_status_json",
         "provenance_json", "temporary_protection_json", "expired_protection_json",
         "replay_mode", "replay_fork_seed", "replay_state_hash",
         "accumulated_lineage_dfe", "conditioning_digest", "safety_reference_id"],
        ["state_id"],
    ),
    "complete_endpoints": _schema(
        ["endpoint_id", "endpoint_content_digest", "protein_id", "root_id", "family_id", "depth",
         "source_state_id", "source_state_content_digest", "fork_index", "fork_seed",
         "sequence", "sequence_md5", "sequence_equivalence_key", "sequence_length",
         "head_evaluator_digest", "head_window_grid_digest", "head_global_risk",
         "head_score_json", "feasibility_level", "structure_evaluated", "structure_feasible",
         "structure_metrics_json", "endpoint_provenance_json",
         "replay_mode", "replay_fork_seed", "replay_state_hash", "cost_event_ids_json"],
        ["endpoint_id"],
    ),
    "archive": _schema(
        ["endpoint_id", "endpoint_content_digest", "protein_id", "root_id", "family_id",
         "sequence_equivalence_key", "feasibility_level", "is_elite", "elite_rank",
         "is_diversity_frontier", "first_depth_seen", "last_depth_seen",
         "may_become_ancestry", "admission_reason"],
        ["endpoint_id"],
    ),
    "feedback_events": _schema(
        ["transition_id", "outcome", "detail", "protein_id", "family_id", "depth",
         "source_state_id", "source_state_content_digest", "selected_endpoint_id",
         "selected_endpoint_content_digest", "projected_state_id", "projected_state_digest",
         "propagated_state_id", "propagated_state_digest",
         "policy_id", "policy_version", "policy_config_digest", "policy_spec_digest",
         "policy_is_diagnostic",
         "write_from_endpoint_json", "inject_from_source_feedback_json", "reopen_json",
         "carry_from_source_json", "support_reason_by_pos_json",
         "n_write_from_endpoint", "n_inject_from_source_feedback", "n_reopen",
         "n_carry_from_source",
         "assimilation_json", "temporary_protection_json",
         "r_step", "c_source_step", "c_next_step", "declared_band_id", "declared_band_digest",
         "pair_id", "arm_slot", "treatment_identity", "descendant_fork_seed",
         # V2F5A (PLAN §5.3, A.5): "Every directional transition must preserve enough evidence to
         # reconstruct why each support position was chosen."  Null for every policy that emits
         # only the four support sets -- the diagnostic probe has no candidate ranking to record --
         # and populated on a typed STALL as well as on a decision, because a stall that kept no
         # rejected-candidate counts cannot be told from a policy nobody asked.
         "policy_stall_reason", "head_evidence_consulted",
         "incumbent_id", "incumbent_sequence_md5", "incumbent_global_risk",
         "safety_reference_sequence_md5", "donor_gate_passed", "donor_gate_reason",
         "donor_gate_margin", "donor_gate_epsilon",
         "incumbent_window_evidence_digest", "safety_window_evidence_digest",
         "u_target", "band_center_rule", "write_cap", "n_legal_write_candidates",
         "n_positive_contributions", "m_band_min", "m_band_max", "required_reopen",
         "n_legal_reopen_candidates", "policy_head_calls",
         "write_candidate_evidence_json", "reopen_candidate_evidence_json",
         "policy_calibration_json", "reward_gate_kind",
         "attribution_reference_sequence_md5"],
        # ``arm_slot`` and ``treatment_identity`` are part of the KEY, not merely columns.  A
        # matched pair runs both arms off one source through one transition id, so without them two
        # arms of the same contrast collide on the join key and the bundle refuses a table whose
        # own columns were designed to hold exactly that.  For an unpaired ladder run both are
        # constant, so the key is unchanged.
        ["transition_id", "source_state_id", "arm_slot", "treatment_identity"],
    ),
    "a2_views": _schema(
        ["a2_view_id", "protein_id", "source_state_id", "depth", "n_members",
         "member_endpoint_ids_json", "member_content_digests_json",
         "matched_extra_lookaheads", "matching_resource", "declared_matching_resource",
         "matching_status", "matching_detail", "unmatched_components_json"],
        ["a2_view_id"],
    ),
    # Every structure evaluation the run performed, PASSED OR FAILED.
    #
    # The endpoint table cannot carry this: the state layer allows an evaluated structure outcome
    # only on a DEFINITIVE endpoint, so a rejected design keeps ``structure_evaluated=false`` and
    # empty metrics -- deliberately, because "evaluated and failed" is bookkeeping about a design
    # rather than a property of it.  The consequence, measured on the first Canary, is that the one
    # cell that most needs diagnosing is the one whose bundle cannot say why: `5zhv_r30` folded
    # four endpoints, rejected all four, and its own artifacts recorded no scTM at all.  Recovering
    # them meant reaching into the refold cache by content key, which is evidence recovery by
    # side-channel.  This table is the evidence, and it changes no endpoint semantics.
    "structure_evaluations": _schema(
        ["endpoint_id", "protein_id", "depth", "sequence_md5", "evaluated", "feasible",
         "failure_reason", "cache_status", "model_executed", "walltime_s", "metrics_json",
         "feasibility_level", "admission_reason",
         "structure_backend_digest", "v0_structure_gate_config_digest"],
        ["endpoint_id"],
    ),
    # Runbook §7: the mechanism cohort's readout, one row per MATCHED DESCENDANT PAIR.
    #
    # Neither of the two checks that already exist answers the question this table is for.  The
    # Canary showed the feedback machinery EXECUTES; ``source_dependence_verdict`` shows the kernel
    # is not blind to what it was handed -- but both stop at the projected state.  Whether the
    # injected identity survives propagation to a completed descendant is the actual scientific
    # claim of V2, and nothing measured it.
    "mechanism_contrasts": _schema(
        ["protein_id", "view", "source_index", "fork_index",
         "source_seed", "source_state_id", "source_unresolved_editable",
         "analyzable", "reason", "parity_violation",
         "n_free", "n_diff_free", "hamming_free",
         "n_editable_scored", "n_diff_editable", "hamming_editable",
         "contrastable", "n_input_diff", "write_positions_json",
         "arm_a_descendant_id", "arm_b_descendant_id",
         "arm_a_sequence_md5", "arm_b_sequence_md5"],
        ["protein_id", "view", "source_index", "fork_index"],
    ),
    "terminal_validation": _schema(
        ["endpoint_id", "sequence_md5", "structure_definitive", "structure_feasible",
         "structure_metrics_json", "immune_evaluator", "immune_global_risk", "immune_passed",
         "whole_landscape_max_increase", "whole_landscape_positive_mass",
         "whole_landscape_positive_count", "whole_landscape_n_windows",
         "whole_landscape_reference_binding_id", "incremental_gate_status",
         "diversity_family_id", "v0_before_after_json"],
        ["endpoint_id"],
    ),
}

#: Explicit Arrow types for the V2 vocabulary.  Declared here rather than registered in the V1
#: module's global name sets, so the two vocabularies cannot collide on a shared column name.
V2_COLUMN_TYPES: dict[str, str] = {
    **{name: "int" for name in (
        "depth", "sampler_step", "r_step", "c_next_step", "c_source_step", "n_steps", "length",
        "n_editable", "n_unresolved_editable", "mask_token_id", "accumulated_lineage_dfe",
        "fork_index", "sequence_length", "elite_rank", "first_depth_seen", "last_depth_seen",
        "n_write_from_endpoint", "n_inject_from_source_feedback", "n_reopen",
        "n_carry_from_source", "n_members", "matched_extra_lookaheads",
        "whole_landscape_positive_count", "whole_landscape_n_windows",
        "source_index", "n_free", "n_diff_free", "n_editable_scored", "n_diff_editable",
        "n_input_diff", "source_unresolved_editable",
        "u_target", "write_cap", "n_legal_write_candidates", "n_positive_contributions",
        "m_band_min", "m_band_max", "required_reopen", "n_legal_reopen_candidates",
        "policy_head_calls",
    )},
    **{name: "uint" for name in ("fork_seed", "replay_fork_seed", "descendant_fork_seed",
                                 "source_seed")},
    **{name: "float" for name in ("rho_edit", "head_global_risk", "immune_global_risk",
                                  "whole_landscape_max_increase",
                                  "whole_landscape_positive_mass", "walltime_s",
                                  "hamming_free", "hamming_editable",
                                  "incumbent_global_risk", "donor_gate_margin",
                                  "donor_gate_epsilon")},
    **{name: "bool" for name in (
        "structure_evaluated", "structure_feasible", "structure_definitive", "is_elite",
        "is_diversity_frontier", "may_become_ancestry", "policy_is_diagnostic", "immune_passed",
        "evaluated", "feasible", "model_executed", "analyzable", "contrastable",
        "head_evidence_consulted", "donor_gate_passed",
    )},
}


# --------------------------------------------------------------------------------------------
# run manifest
# --------------------------------------------------------------------------------------------


def run_manifest(
    *, config: Any, code_revision: str, content_identities: Mapping[str, str],
    seed_namespaces: Sequence[str], production_depth_authorized: bool = False,
    exploratory_depth_override: bool = False,
) -> dict:
    """The run's whole identity, derivable with no model and no file access (PLAN §5.1, §5.3).

    ``config_digest`` is taken from the config object itself rather than recomputed here, so
    ``--print-config``, ``--dry-run`` and the realized run can never disagree about which config
    they described.
    """
    return {
        "schema_version": config.schema_version,
        "config_digest": config.config_digest(),
        "campaign_id": config.identity.campaign_id,
        "split_role": config.identity.split_role,
        "phase": config.identity.phase,
        "master_seed": int(config.identity.master_seed),
        "seed_schema": config.identity.seed_schema,
        "code_revision": str(code_revision),
        "arm_role": config.arm.arm_role,
        "feedback_enabled": bool(config.arm.feedback_enabled),
        "a2_matching_resource": config.arm.a2_matching_resource,
        "schedule_id": config.schedule.schedule_id,
        "coordinate_law": config.schedule.coordinate_law.value,
        "depth_cap": int(config.schedule.depth_cap),
        "active_population_width": int(config.schedule.active_population_width),
        "production_depth_authorized": bool(production_depth_authorized),
        "exploratory_depth_override": bool(exploratory_depth_override),
        "caps": {
            "max_logical_dfe": int(config.caps.max_logical_dfe),
            "max_head_calls": int(config.caps.max_head_calls),
            "max_definitive_refolds": int(config.caps.max_definitive_refolds),
            "max_gpu_seconds": int(config.caps.max_gpu_seconds),
            "max_walltime_s": int(config.caps.max_walltime_s),
            "max_retries": int(config.caps.max_retries),
            "retry_scope": config.caps.retry_scope,
        },
        "content_identities": {str(k): str(v) for k, v in sorted(dict(content_identities).items())},
        "seed_namespaces": sorted(str(name) for name in seed_namespaces),
    }


# --------------------------------------------------------------------------------------------
# row builders
# --------------------------------------------------------------------------------------------


def _commit_payload(commit) -> list | None:
    return None if commit is None else [commit.depth, commit.step]


def partial_state_rows(states: Iterable[Any]) -> list[dict]:
    """One row per live or projected partial state, carrying every vector a replay needs."""
    rows: list[dict] = []
    for state in states:
        tokens = list(state.tokens)
        editable = [int(i) for i in state.editable_positions]
        mask_token_id = int(state.mask_token_id)
        unresolved = sum(1 for i in editable if tokens[i] == mask_token_id)
        is_projected = hasattr(state, "r_step")
        rows.append({
            "state_id": state.state_id,
            "state_content_digest": state.content_digest,
            "layer": "projected" if is_projected else "live",
            "protein_id": state.lineage.protein_id,
            "root_id": state.lineage.root_id,
            "family_id": state.lineage.family_id,
            "depth": int(state.lineage.depth),
            "parent_state_id": state.lineage.parent_state_id,
            "parent_transition_id": state.lineage.parent_transition_id,
            "origin_endpoint_id": state.lineage.origin_endpoint_id,
            "sampler_step": int(getattr(state, "sampler_step", getattr(state, "r_step", 0))),
            "r_step": int(state.r_step) if is_projected else None,
            "c_next_step": int(state.c_next_step) if is_projected else None,
            "n_steps": int(state.n_steps),
            "coordinate_law": (
                state.coordinate_law.value if is_projected
                and hasattr(state.coordinate_law, "value") else None),
            "length": len(tokens),
            "n_editable": len(editable),
            "n_unresolved_editable": unresolved,
            "rho_edit": (
                float(len(editable) - unresolved) / float(len(editable)) if editable else 0.0),
            "tokens_json": _json([int(t) for t in tokens]),
            "mask_token_id": mask_token_id,
            "hard_anchors_json": _json([[int(p), int(t)] for p, t in state.hard_anchors]),
            "editable_positions_json": _json(editable),
            "active_origin_kind_json": _json(
                [kind.value for kind in state.active_origin_kind_by_pos]),
            "feedback_origin_ref_json": _json(
                [ref.value for ref in state.feedback_origin_ref_by_pos]),
            "active_commit_json": _json(
                [_commit_payload(c) for c in state.active_commit_depth_step_by_pos]),
            "origin_transition_id_json": _json(list(state.origin_transition_id_by_pos)),
            "active_sampler_score_json": _json(
                [None if v is None else float(v) for v in state.active_sampler_score_by_pos]),
            "active_score_status_json": _json(
                [status.value for status in state.active_score_status_by_pos]),
            "provenance_json": _json(
                [prov.canonical_payload() for prov in state.provenance_by_pos]),
            "temporary_protection_json": _json([
                protection.canonical_payload()
                for protection in getattr(state, "active_temporary_protection", ())
            ]),
            "expired_protection_json": _json([
                protection.canonical_payload()
                for protection in getattr(state, "expired_protection", ())
            ]),
            "replay_mode": getattr(getattr(state, "replay", None), "mode", None),
            "replay_fork_seed": getattr(getattr(state, "replay", None), "fork_seed", None),
            "replay_state_hash": getattr(getattr(state, "replay", None), "replay_state_hash",
                                         None),
            "accumulated_lineage_dfe": int(getattr(
                state, "accumulated_lineage_dfe", getattr(state, "inherited_lineage_dfe", 0))),
            # V2ConditioningIdentity exposes its canonical payload, not a digest property;
            # hashing it here keeps ONE conditioning identity per state row without
            # inventing a second definition of what that identity is.
            "conditioning_digest": canonical_digest(state.conditioning.canonical_payload()),
            "safety_reference_id": state.safety_reference.reference_id,
        })
    return rows


def complete_endpoint_rows(endpoints: Iterable[Any], *, depth: int) -> list[dict]:
    """One row per LOGICAL endpoint.

    Two forks that converged on the same sequence produce two rows with one shared
    ``sequence_equivalence_key``.  A summary table may collapse on that key; this one may not.
    """
    rows: list[dict] = []
    for endpoint in endpoints:
        structure = endpoint.structure_outcome
        rows.append({
            "endpoint_id": endpoint.endpoint_id,
            "endpoint_content_digest": endpoint.content_digest,
            "protein_id": endpoint.protein_id,
            "root_id": endpoint.lineage.root_id,
            "family_id": endpoint.lineage.family_id,
            "depth": int(depth),
            "source_state_id": endpoint.source_state_id,
            "source_state_content_digest": endpoint.source_state_content_digest,
            "fork_index": int(endpoint.fork_index),
            "fork_seed": int(endpoint.fork_seed),
            "sequence": endpoint.sequence,
            "sequence_md5": endpoint.sequence_md5,
            "sequence_equivalence_key": endpoint.sequence_md5,
            "sequence_length": int(endpoint.sequence_length),
            "head_evaluator_digest": endpoint.head_binding.evaluator.digest(),
            "head_window_grid_digest": endpoint.head_binding.window_grid_digest,
            "head_global_risk": float(endpoint.head_global_risk),
            "head_score_json": _json(endpoint.canonical_payload()["head_score"]),
            "feasibility_level": endpoint.feasibility_level.value,
            "structure_evaluated": bool(getattr(structure, "evaluated", False)),
            "structure_feasible": bool(getattr(structure, "feasible", False)),
            "structure_metrics_json": _json(dict(getattr(structure, "metrics", None) or {})),
            "endpoint_provenance_json": _json([
                evidence.canonical_payload()
                for evidence in endpoint.endpoint_provenance_evidence_by_pos
            ]),
            "replay_mode": endpoint.replay.mode,
            "replay_fork_seed": endpoint.replay.fork_seed,
            "replay_state_hash": endpoint.replay.replay_state_hash,
            "cost_event_ids_json": _json(list(endpoint.cost_event_ids)),
        })
    return rows


def archive_rows(archive: Any, *, admission_by_endpoint: Mapping[str, str] | None = None
                 ) -> list[dict]:
    """One row per archived endpoint, with membership and the reason it may or may not be ancestry.

    ``admission_reason`` is threaded from the safety gate rather than re-derived: an archive that
    recomputed eligibility could disagree with the decision the run actually made.
    """
    reasons = dict(admission_by_endpoint or {})
    by_id = {endpoint.endpoint_id: endpoint for endpoint in archive.endpoints()}
    rows: list[dict] = []
    for entry in archive.raw_rows():
        endpoint = by_id[entry.endpoint_id]
        rows.append({
            "endpoint_id": entry.endpoint_id,
            "endpoint_content_digest": entry.endpoint_content_digest,
            "protein_id": entry.lineage.protein_id,
            "root_id": entry.lineage.root_id,
            "family_id": entry.lineage.family_id,
            "sequence_equivalence_key": entry.sequence_equivalence_key,
            "feasibility_level": entry.feasibility_level.value,
            "is_elite": bool(entry.membership.is_elite),
            "elite_rank": entry.membership.elite_rank,
            "is_diversity_frontier": bool(entry.membership.is_diversity_frontier),
            "first_depth_seen": int(entry.first_depth_seen),
            "last_depth_seen": int(entry.last_depth_seen),
            "may_become_ancestry": bool(archive.may_become_ancestry(entry.endpoint_id)),
            "admission_reason": reasons.get(entry.endpoint_id),
            **({} if endpoint is None else {}),
        })
    return rows


def structure_evaluation_rows(
    archive: Any, *, admission_by_endpoint: Mapping[str, str] | None = None,
    conditioning: Any = None,
) -> list[dict]:
    """One row per endpoint the structure gate looked at -- including every one it rejected.

    Read off the ARCHIVE rather than off the endpoints, because that is where a failed verdict is
    kept: ``ExactArchive.promote`` stores the outcome for every endpoint regardless of level, while
    the endpoint record is allowed to carry one only when DEFINITIVE.  Reading the archive
    therefore needs no new plumbing through the cycle and cannot disagree with the decision the run
    made.

    The backend and gate-config digests are repeated on every row on purpose.  They are also in the
    manifest, but this table is the one that gets carried off alone to answer "why did nothing pass"
    -- and a table of structure verdicts that cannot say which model produced them is an anecdote.
    """
    reasons = dict(admission_by_endpoint or {})
    backend = getattr(conditioning, "structure_backend", None)
    gate_config = getattr(conditioning, "v0_structure_gate_config", None)
    rows: list[dict] = []
    for entry in archive.raw_rows():
        outcome = archive.structure_outcome(entry.endpoint_id)
        metrics = dict(getattr(outcome, "metrics", None) or {}) if outcome is not None else {}
        rows.append({
            "endpoint_id": entry.endpoint_id,
            "protein_id": entry.lineage.protein_id,
            "depth": int(entry.first_depth_seen),
            "sequence_md5": entry.sequence_equivalence_key,
            # ``evaluated`` false with no outcome at all means the gate never ran for this
            # endpoint; false WITH an outcome means it ran and could not produce a verdict.  The
            # two are different facts and the row keeps them apart.
            "evaluated": bool(getattr(outcome, "evaluated", False)),
            "feasible": bool(getattr(outcome, "feasible", False)),
            "failure_reason": getattr(outcome, "failure_reason", None),
            "cache_status": getattr(outcome, "cache_status", None),
            "model_executed": bool(getattr(outcome, "model_executed", False)),
            "walltime_s": float(getattr(outcome, "walltime_s", 0.0) or 0.0),
            "metrics_json": _json({k: float(v) for k, v in metrics.items()}),
            "feasibility_level": entry.feasibility_level.value,
            "admission_reason": reasons.get(entry.endpoint_id),
            "structure_backend_digest": backend,
            "v0_structure_gate_config_digest": gate_config,
        })
    return rows


def feedback_event_rows(events: Iterable[Mapping[str, Any]]) -> list[dict]:
    """One row per feedback transition, committed or null.

    A null event is recorded in full.  PLAN §4.5 makes a refusal a result, and a bundle that kept
    only the cycles that committed would be a record of the successes alone.
    """
    rows: list[dict] = []
    for event in events:
        source = event["source"]
        endpoint = event.get("endpoint")
        projected = event.get("projected")
        propagated = event.get("propagated")
        policy = event.get("policy")
        support = getattr(projected, "support", None)

        def _set(name):
            return list(getattr(support, name, ()) or ())

        rows.append({
            "transition_id": (
                projected.origin_transition_id if projected is not None
                else event.get("transition_id", f"null:{source.state_id}")),
            "outcome": event["outcome"],
            "detail": event.get("detail", ""),
            "protein_id": source.lineage.protein_id,
            "family_id": source.lineage.family_id,
            "depth": int(source.lineage.depth),
            "source_state_id": source.state_id,
            "source_state_content_digest": source.content_digest,
            "selected_endpoint_id": None if endpoint is None else endpoint.endpoint_id,
            "selected_endpoint_content_digest": (
                None if endpoint is None else endpoint.content_digest),
            "projected_state_id": None if projected is None else projected.state_id,
            "projected_state_digest": None if projected is None else projected.content_digest,
            "propagated_state_id": None if propagated is None else propagated.state_id,
            "propagated_state_digest": None if propagated is None else propagated.content_digest,
            "policy_id": getattr(policy, "policy_id", None),
            "policy_version": getattr(policy, "policy_version", None),
            "policy_config_digest": getattr(policy, "policy_config_digest", None),
            "policy_spec_digest": getattr(policy, "policy_spec_digest", None),
            "policy_is_diagnostic": bool(getattr(policy, "is_diagnostic_only", False)),
            "write_from_endpoint_json": _json(_set("write_from_endpoint")),
            "inject_from_source_feedback_json": _json(_set("inject_from_source_feedback")),
            "reopen_json": _json(_set("reopen")),
            "carry_from_source_json": _json(_set("carry_from_source")),
            "support_reason_by_pos_json": _json(
                {str(k): str(v) for k, v in dict(getattr(support, "reason_by_pos", {}) or {})
                 .items()}),
            "n_write_from_endpoint": len(_set("write_from_endpoint")),
            "n_inject_from_source_feedback": len(_set("inject_from_source_feedback")),
            "n_reopen": len(_set("reopen")),
            "n_carry_from_source": len(_set("carry_from_source")),
            "assimilation_json": _json(list(event.get("assimilation", []) or [])),
            "temporary_protection_json": _json([
                protection.canonical_payload()
                for protection in getattr(projected, "active_temporary_protection", ())
            ]),
            "r_step": None if projected is None else int(projected.r_step),
            "c_source_step": int(source.sampler_step),
            "c_next_step": None if projected is None else int(projected.c_next_step),
            "declared_band_id": getattr(projected, "declared_band_id", None),
            "declared_band_digest": getattr(projected, "declared_band_digest", None),
            "pair_id": event.get("pair_id"),
            "arm_slot": event.get("arm_slot"),
            "treatment_identity": event.get("treatment_identity"),
            "descendant_fork_seed": (
                None if projected is None else int(projected.descendant_fork_seed)),
            **_policy_evidence_columns(event.get("policy_evidence")),
        })
    return rows


#: Every V2F5A evidence column, so a bundle whose policy emits none still has the same SHAPE.  A
#: schema that grew or shrank with the policy would make two arms of one campaign unjoinable.
_POLICY_EVIDENCE_COLUMNS = (
    "policy_stall_reason", "head_evidence_consulted", "incumbent_id", "incumbent_sequence_md5",
    "incumbent_global_risk", "safety_reference_sequence_md5", "donor_gate_passed",
    "donor_gate_reason", "donor_gate_margin", "donor_gate_epsilon",
    "incumbent_window_evidence_digest", "safety_window_evidence_digest", "u_target",
    "band_center_rule", "write_cap", "n_legal_write_candidates", "n_positive_contributions",
    "m_band_min", "m_band_max", "required_reopen", "n_legal_reopen_candidates",
    "policy_head_calls", "write_candidate_evidence_json", "reopen_candidate_evidence_json",
    "policy_calibration_json",
    "reward_gate_kind", "attribution_reference_sequence_md5",
)


def _policy_evidence_columns(evidence: Any) -> dict:
    """Flatten one ``HeadDirectedDecisionEvidence`` into the feedback-event row.

    The scalar decision inputs become columns so a cohort can be filtered and aggregated in SQL --
    "how many sources had at least one legal positive write" must not require parsing JSON in every
    row -- while the per-position candidate tables stay JSON, because their length varies per
    transition and a wide table would be mostly nulls.
    """
    if evidence is None:
        return {name: None for name in _POLICY_EVIDENCE_COLUMNS}
    payload = evidence.canonical_payload()
    gate = payload.get("donor_gate") or {}
    return {
        "policy_stall_reason": payload.get("stall_reason"),
        "head_evidence_consulted": bool(payload.get("head_evidence_consulted")),
        "incumbent_id": payload.get("incumbent_id"),
        "incumbent_sequence_md5": payload.get("incumbent_sequence_md5"),
        "incumbent_global_risk": gate.get("incumbent_global_risk"),
        "safety_reference_sequence_md5": payload.get("safety_reference_sequence_md5"),
        "donor_gate_passed": None if not gate else bool(gate.get("passed")),
        "donor_gate_reason": gate.get("reason"),
        "donor_gate_margin": gate.get("margin"),
        "donor_gate_epsilon": gate.get("epsilon_r"),
        "incumbent_window_evidence_digest": payload.get("incumbent_window_evidence_digest"),
        "safety_window_evidence_digest": payload.get("safety_window_evidence_digest"),
        "u_target": payload.get("u_target"),
        "band_center_rule": payload.get("band_center_rule"),
        "write_cap": payload.get("write_cap"),
        "n_legal_write_candidates": payload.get("n_legal_write_candidates"),
        "n_positive_contributions": payload.get("n_positive_contributions"),
        "m_band_min": payload.get("m_band_min"),
        "m_band_max": payload.get("m_band_max"),
        "required_reopen": payload.get("required_reopen"),
        "n_legal_reopen_candidates": payload.get("n_legal_reopen_candidates"),
        "policy_head_calls": payload.get("head_calls"),
        "write_candidate_evidence_json": _json(payload.get("write_candidates", [])),
        "reopen_candidate_evidence_json": _json(payload.get("reopen_candidates", [])),
        "policy_calibration_json": _json(payload.get("calibration", {})),
        "reward_gate_kind": payload.get("reward_gate_kind"),
        "attribution_reference_sequence_md5": payload.get(
            "attribution_reference_sequence_md5"),
    }


def a2_view_rows(
    views: Iterable[Any], *, matched_extra_lookaheads: int,
    matching_resource: str | None,
    declared_matching_resource: str,
    matching_executed: bool,
    unmatched_components: Mapping[str, Any] | None = None,
) -> list[dict]:
    """One row per A2 view: the exact pre-feedback membership and what A2 was given instead.

    The unmatched components are reported rather than netted away.  PLAN §4.3 requires A2's extra
    allocation to be matched on ONE declared resource; every other resource is then unmatched by
    construction, and hiding that would make the control look tighter than it is.
    """
    if not isinstance(matching_executed, bool):
        raise V2ArtifactError("matching_executed must be an explicit bool")
    if not isinstance(matched_extra_lookaheads, int) or isinstance(
        matched_extra_lookaheads, bool
    ) or matched_extra_lookaheads < 0:
        raise V2ArtifactError("matched_extra_lookaheads must be a non-negative int")
    if not isinstance(declared_matching_resource, str) or not declared_matching_resource:
        raise V2ArtifactError("declared_matching_resource must be a non-empty str")
    if matching_executed:
        if matching_resource != declared_matching_resource:
            raise V2ArtifactError(
                "executed matching_resource must equal declared_matching_resource"
            )
        matching_status = "executed"
        matching_detail = "matched-extra allocation executed"
    else:
        if matching_resource is not None or matched_extra_lookaheads != 0:
            raise V2ArtifactError(
                "an unexecuted match must report no matching_resource and zero extra lookaheads"
            )
        matching_status = "declared_not_executed"
        matching_detail = "matching declared but not executed; no matched-extra allocation was run"

    rows: list[dict] = []
    for view in views:
        member_ids = list(view.endpoint_ids)
        rows.append({
            "a2_view_id": view.a2_view_id,
            "protein_id": view.protein_id,
            "source_state_id": view.source_state_id,
            "depth": int(view.depth),
            "n_members": len(member_ids),
            "member_endpoint_ids_json": _json(member_ids),
            "member_content_digests_json": _json(
                {str(k): str(v) for k, v in dict(view.content_digest_by_endpoint).items()}),
            "matched_extra_lookaheads": int(matched_extra_lookaheads),
            "matching_resource": matching_resource,
            "declared_matching_resource": declared_matching_resource,
            "matching_status": matching_status,
            "matching_detail": matching_detail,
            "unmatched_components_json": _json(dict(unmatched_components or {})),
        })
    return rows


# --------------------------------------------------------------------------------------------
# mechanism contrasts (runbook §7)
# --------------------------------------------------------------------------------------------

#: The views whose DESCENDANTS form a scorable contrast.
#:
#: ``feedback_off`` is absent because it is a view of the shared pre-feedback stage rather than a
#: second execution: that stage runs with feedback disabled, stops after the A2 view, and therefore
#: produces no descendants to pair.  ``source_change`` is absent because its fixed-support ablation
#: masks a carried position, which moves that position INTO the free domain on one arm only -- the
#: two arms would then be scored on different domains.  Both still run and still write their
#: ordinary rows; they are simply not contrasts this statistic can carry.
MECHANISM_CONTRAST_VIEWS: tuple[str, ...] = ("endpoint_change", "source_shuffle")

#: V2F5A's contrast (PLAN §8.4).  Scorable here, but on a NARROWER domain: the two support laws
#: reopen different positions by construction, so their free domains differ and only the
#: INTERSECTION is comparable position by position.  Its primary readout is not in this table at
#: all -- it is the paired descendant HEAD risk, recovered by joining ``arm_{a,b}_descendant_id``
#: to ``complete_endpoints.head_global_risk``, which needs no shared domain.  The Hamming columns
#: are its mechanism secondary.
POLICY_QUALIFICATION_CONTRAST_VIEWS: tuple[str, ...] = ("support_law",)

#: Every view ``mechanism_contrast_rows`` will score.
SCORABLE_CONTRAST_VIEWS: tuple[str, ...] = (
    MECHANISM_CONTRAST_VIEWS + POLICY_QUALIFICATION_CONTRAST_VIEWS)


def free_domain(projected: Any) -> tuple[int, ...]:
    """The positions feedback can still steer: editable, and still unresolved at re-entry.

    This one predicate is the whole domain definition, and it is stated as a property of the
    projected state rather than as a list of exclusions so that it cannot fall out of date with the
    kernel.  It excludes, automatically:

    * **hard anchors** -- they are not in ``editable_positions`` at all;
    * **the endpoint-written position** -- ``write_from_endpoint`` is resolved by the write, and it
      is the one position two different endpoints differ at BY CONSTRUCTION;
    * **``inject_from_source_feedback`` and ``carry_from_source``** -- both resolved, and both
      carrying source bytes verbatim.  Under ``source_shuffle`` those bytes are permuted, so the
      arms differ there by construction with no mechanism involved: on the three executed Canary
      coordinates that is 46/101, 90/277 and 119/277 positions.  A whole-editable Hamming would
      read 0.33-0.46 for a kernel that transmits nothing, and a 0.02 gate on it is not a gate.

    What remains is masked on BOTH arms at ``r_d`` and can only be decided by forward propagation.
    A difference there is transmission or it is nothing.
    """
    mask = projected.mask_token_id
    return tuple(position for position in projected.editable_positions
                 if projected.tokens[position] == mask)


def _hamming(sequence_a: str, sequence_b: str, positions: Sequence[int]) -> int:
    return sum(1 for position in positions if sequence_a[position] != sequence_b[position])


def mechanism_contrast_rows(
    *,
    protein_id: str,
    source_index: int,
    source_seed: int,
    source_state_id: str | None,
    source_unresolved_editable: int | None,
    view: str,
    arm_a: Any,
    arm_b: Any,
    parity_violation: str = "",
) -> list[dict]:
    """Score one mechanism view's matched descendant pairs for one source prefix.

    Returns one row per matched fork.  A prefix that produced no scorable contrast returns a SINGLE
    row saying why rather than no rows at all: the sampling unit is the source prefix, and a
    silently absent prefix is indistinguishable from one that was never attempted.
    """
    if view not in SCORABLE_CONTRAST_VIEWS:
        raise V2ArtifactError(
            f"{view!r} is not a scorable contrast; expected one of "
            f"{list(SCORABLE_CONTRAST_VIEWS)}"
        )

    common = {
        "protein_id": str(protein_id), "view": str(view),
        "source_index": int(source_index), "source_seed": int(source_seed),
        "source_state_id": source_state_id,
        "source_unresolved_editable": (None if source_unresolved_editable is None
                                       else int(source_unresolved_editable)),
        "parity_violation": str(parity_violation),
    }

    def unscorable(reason: str) -> list[dict]:
        return [{
            **common, "fork_index": 0, "analyzable": False, "reason": reason,
            "n_free": None, "n_diff_free": None, "hamming_free": None,
            "n_editable_scored": None, "n_diff_editable": None, "hamming_editable": None,
            "contrastable": None, "n_input_diff": None, "write_positions_json": _json([]),
            "arm_a_descendant_id": None, "arm_b_descendant_id": None,
            "arm_a_sequence_md5": None, "arm_b_sequence_md5": None,
        }]

    if parity_violation:
        return unscorable(f"support parity violated: {parity_violation}")

    projected_a = getattr(arm_a, "projected", None)
    projected_b = getattr(arm_b, "projected", None)
    if projected_a is None or projected_b is None:
        # Carry the CYCLE's own words.  The commonest way arm B has no projection is
        # ``endpoint_rank=1`` on a pool with one admissible endpoint -- a fact about how many
        # lookaheads cleared the gate, not about the mechanism -- and a generic "no projection"
        # would make that indistinguishable from a band refusal or a policy rejection.
        arm = arm_a if projected_a is None else arm_b
        slot = "A" if projected_a is None else "B"
        outcome = getattr(getattr(arm, "outcome", None), "value", None)
        detail = str(getattr(arm, "detail", "") or "").strip()
        said = f" [{outcome}]" if outcome else ""
        said += f" {detail}" if detail else ""
        return unscorable(
            f"arm {slot} produced no projection; the cycle stopped before feedback.{said}")

    free = free_domain(projected_a)
    if free != free_domain(projected_b):
        if view not in POLICY_QUALIFICATION_CONTRAST_VIEWS:
            return unscorable("the arms do not share a free domain, so no position is comparable "
                              "between them")
        # The support-law arms reopen DIFFERENT positions -- that IS the treatment -- so their free
        # domains differ by construction and demanding equality would kill every pair.  Scored on
        # the intersection, which is the largest set on which a position-by-position comparison
        # means the same thing on both arms.  ``n_free`` therefore reports the comparable width,
        # not either arm's own.
        free = tuple(sorted(set(free) & set(free_domain(projected_b))))
    if not free:
        return unscorable("the free domain is empty; this coordinate leaves nothing for "
                          "propagation to decide")

    written = set(projected_a.support.write_from_endpoint) | set(
        projected_b.support.write_from_endpoint)
    # The whole-editable reading is kept alongside the free-domain one, minus the written position,
    # because it is the denominator a threshold might instead be frozen on -- and because the gap
    # between the two columns is the measurement of how much of the domain could never move.
    editable_scored = tuple(sorted(set(projected_a.editable_positions) - written))

    descendants_a = tuple(getattr(arm_a, "descendant_endpoints", ()) or ())
    descendants_b = tuple(getattr(arm_b, "descendant_endpoints", ()) or ())
    if len(descendants_a) != len(descendants_b):
        return unscorable(
            f"the arms produced {len(descendants_a)} and {len(descendants_b)} descendant(s); "
            "zipping them would drop the tail and report a matched contrast that is not one")
    if not descendants_a:
        return unscorable("neither arm produced a descendant to compare")

    # Whether the intervention actually changed what the arms were handed.  Two endpoints that
    # happen to agree at the written position make the arms byte-identical INPUTS, so their
    # descendants are identical for a reason that has nothing to do with transmission; such a pair
    # is a structural zero and the analysis must be able to exclude it by a pre-registered rule
    # rather than by looking at the outcome.
    n_input_diff = _hamming(
        "".join(chr(t) for t in projected_a.tokens),
        "".join(chr(t) for t in projected_b.tokens),
        range(len(projected_a.tokens)),
    )

    rows: list[dict] = []
    for fork_index, (endpoint_a, endpoint_b) in enumerate(zip(descendants_a, descendants_b)):
        sequence_a, sequence_b = endpoint_a.sequence, endpoint_b.sequence
        if len(sequence_a) != len(projected_a.tokens) or len(sequence_b) != len(sequence_a):
            raise V2ArtifactError(
                f"descendant length {len(sequence_a)}/{len(sequence_b)} does not match the "
                f"projected state's {len(projected_a.tokens)}; the position indices would not align"
            )
        n_diff_free = _hamming(sequence_a, sequence_b, free)
        n_diff_editable = _hamming(sequence_a, sequence_b, editable_scored)
        rows.append({
            **common, "fork_index": int(fork_index), "analyzable": True, "reason": "",
            "n_free": len(free), "n_diff_free": n_diff_free,
            "hamming_free": n_diff_free / len(free),
            "n_editable_scored": len(editable_scored), "n_diff_editable": n_diff_editable,
            "hamming_editable": (n_diff_editable / len(editable_scored)
                                 if editable_scored else None),
            "contrastable": bool(n_input_diff), "n_input_diff": int(n_input_diff),
            "write_positions_json": _json(sorted(written)),
            "arm_a_descendant_id": getattr(endpoint_a, "endpoint_id", None),
            "arm_b_descendant_id": getattr(endpoint_b, "endpoint_id", None),
            "arm_a_sequence_md5": getattr(endpoint_a, "sequence_md5", None),
            "arm_b_sequence_md5": getattr(endpoint_b, "sequence_md5", None),
        })
    return rows


def terminal_validation_rows(records: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Structure and immune results stay in SEPARATE columns.

    They are independent measurements of different failure modes; one "passed" column would hide
    which of them a design actually failed, and a design that folds but opens a hotspot is exactly
    the case V2's safety contract exists for.
    """
    rows: list[dict] = []
    for record in records:
        rows.append({
            "endpoint_id": record["endpoint_id"],
            "sequence_md5": record["sequence_md5"],
            "structure_definitive": bool(record.get("structure_definitive", False)),
            "structure_feasible": bool(record.get("structure_feasible", False)),
            "structure_metrics_json": _json(dict(record.get("structure_metrics") or {})),
            "immune_evaluator": record.get("immune_evaluator"),
            "immune_global_risk": (
                None if record.get("immune_global_risk") is None
                else float(record["immune_global_risk"])),
            "immune_passed": bool(record.get("immune_passed", False)),
            "whole_landscape_max_increase": record.get("whole_landscape_max_increase"),
            "whole_landscape_positive_mass": record.get("whole_landscape_positive_mass"),
            "whole_landscape_positive_count": record.get("whole_landscape_positive_count"),
            "whole_landscape_n_windows": record.get("whole_landscape_n_windows"),
            "whole_landscape_reference_binding_id": record.get(
                "whole_landscape_reference_binding_id"),
            "incremental_gate_status": record.get("incremental_gate_status"),
            "diversity_family_id": record.get("diversity_family_id"),
            "v0_before_after_json": (
                None if record.get("v0_before_after") is None
                else _json(dict(record["v0_before_after"]))),
        })
    return rows


# --------------------------------------------------------------------------------------------
# the bundle
# --------------------------------------------------------------------------------------------


#: The Dual overlay's own tables. Kept in a SEPARATE registry, not appended to V2_TABLE_SCHEMAS,
#: because ``write_v2_bundle`` requires every declared table and writes it even when empty -- so
#: registering them there would add three empty parquet files to every legacy bundle, which is a
#: Dual-off change. A run with no overlay passes no ``dual_tables`` and its bundle is unchanged.
DUAL_TABLE_SCHEMAS: dict[str, TableSchema] = {
    "dual_endpoint_evidence": _schema(
        ["endpoint_id", "arm", "protein_id", "sequence_md5", "sequence_length",
         "window_grid_digest", "allele_a", "allele_b", "raw_risk_a", "raw_risk_b",
         "raw_density_a", "raw_density_b", "u_a", "u_b", "joint_risk", "joint_density",
         "active_worst", "objective_mode", "objective_digest", "calibration_digest",
         "head_binding_digest_a", "head_binding_digest_b",
         "hotspot_b_max_increase", "hotspot_b_positive_mass", "hotspot_b_positive_count"],
        ["endpoint_id", "arm"],
    ),
    "dual_feedback_evidence": _schema(
        ["transition_id", "arm", "position", "protein_id", "donor_endpoint_id",
         "incumbent_id", "donor_joint_value", "incumbent_joint_value", "donor_margin",
         "counterfactual_sequence_md5", "raw_a", "raw_b", "contribution_a", "contribution_b",
         "joint_contribution", "write_eligible", "selection_rank", "selected",
         "reopen_new_hotspot_a", "reopen_new_hotspot_b", "reopen_worsening_a",
         "reopen_worsening_b", "reopen_residual_a", "reopen_residual_b",
         "reopen_winning_allele", "objective_digest"],
        ["transition_id", "arm", "position"],
    ),
    "dual_terminal_summary": _schema(
        ["protein_id", "root_id", "arm", "n_endpoints", "n_definitive",
         "best_joint_risk", "best_joint_risk_endpoint_id", "best_raw_risk_a", "best_raw_risk_b",
         "d0_joint_risk", "last_improving_depth", "typed_stop", "objective_digest",
         "calibration_digest"],
        ["protein_id", "root_id", "arm"],
    ),
}


#: Explicit Arrow types for the Dual vocabulary.  A SEPARATE registry for the same reason the
#: schemas are separate: a Dual name must not be able to retype a legacy column.
#:
#: Declared rather than inferred, and asserted complete below.  An untyped numeric column is
#: inferred from the rows present, so an EMPTY table -- which ``write_v2_bundle`` writes for every
#: declared table -- types ``raw_risk_a`` as string, and the first real ``-1.2`` then raises
#: ``ArrowTypeError`` at the point where a shard tries to publish its evidence.
DUAL_COLUMN_TYPES: dict[str, str] = {
    **{name: "int" for name in (
        "sequence_length", "hotspot_b_positive_count", "position", "selection_rank",
        "n_endpoints", "n_definitive", "last_improving_depth",
    )},
    **{name: "float" for name in (
        "raw_risk_a", "raw_risk_b", "raw_density_a", "raw_density_b", "u_a", "u_b",
        "joint_risk", "joint_density", "hotspot_b_max_increase", "hotspot_b_positive_mass",
        "donor_joint_value", "incumbent_joint_value", "donor_margin",
        "raw_a", "raw_b", "contribution_a", "contribution_b", "joint_contribution",
        "reopen_residual_a", "reopen_residual_b",
        "best_joint_risk", "best_raw_risk_a", "best_raw_risk_b", "d0_joint_risk",
    )},
    **{name: "bool" for name in ("write_eligible", "selected")},
    # The reopen conjuncts carry the NORMALIZED VALUE each allele contributed, exactly as the
    # legacy ``ReopenCandidateEvidence.new_hotspot`` / ``worsening`` are floats -- not flags. They
    # were declared bool, and the first real ``dual_feedback_evidence`` write raised
    # ``ArrowInvalid: Could not convert 0.0 with type float: tried to convert to boolean``. The
    # round-trip test missed it because it passed an EMPTY feedback table.
    **{name: "float" for name in (
        "reopen_new_hotspot_a", "reopen_new_hotspot_b",
        "reopen_worsening_a", "reopen_worsening_b",
    )},
    **{name: "str" for name in (
        "endpoint_id", "arm", "protein_id", "sequence_md5", "window_grid_digest",
        "allele_a", "allele_b", "active_worst", "objective_mode", "objective_digest",
        "calibration_digest", "head_binding_digest_a", "head_binding_digest_b",
        "transition_id", "donor_endpoint_id", "incumbent_id", "counterfactual_sequence_md5",
        "reopen_winning_allele", "root_id", "best_joint_risk_endpoint_id", "typed_stop",
    )},
}

#: Every declared Dual column must have a declared type.  Checked at import so a schema edit that
#: adds a column cannot ship with that column silently inferred.
_UNTYPED_DUAL_COLUMNS = sorted(
    {column for schema in DUAL_TABLE_SCHEMAS.values() for column in schema.columns}
    - set(DUAL_COLUMN_TYPES)
)
if _UNTYPED_DUAL_COLUMNS:
    raise V2ArtifactError(
        f"Dual artifact column(s) {_UNTYPED_DUAL_COLUMNS} have no declared Arrow type; an "
        "inferred column types itself from the rows present, so an empty shard would publish a "
        "numeric measurement as a string and the first real row would fail to append"
    )


def dual_endpoint_evidence_rows(evidence_by_endpoint: Mapping[str, Any], *, arm: str,
                                head_binding_digests: Mapping[str, tuple[str, str]] | None = None,
                                ) -> list[dict]:
    """One row per exact endpoint both Heads scored.

    This is the table the two-axis return boundary is read off, so it carries BOTH raw risks, both
    normalized coordinates and the joint value -- not the joint value alone. A front cannot be
    recovered from a scalar.
    """
    bindings = dict(head_binding_digests or {})
    rows = []
    for endpoint_id, evidence in sorted(evidence_by_endpoint.items()):
        risk, density = evidence.risk, evidence.density
        hotspot = evidence.hotspot_b
        digest_a, digest_b = bindings.get(
            endpoint_id, (evidence.a.head_binding_digest, evidence.b.head_binding_digest))
        rows.append({
            "endpoint_id": str(endpoint_id), "arm": str(arm),
            "protein_id": evidence.protein_id, "sequence_md5": evidence.sequence_md5,
            "sequence_length": int(evidence.sequence_length),
            "window_grid_digest": evidence.window_grid_digest,
            "allele_a": evidence.a.allele, "allele_b": evidence.b.allele,
            "raw_risk_a": float(evidence.a.raw_risk), "raw_risk_b": float(evidence.b.raw_risk),
            # None, not 0.0: a run that did not build the density axis did not measure zero.
            "raw_density_a": _optional_float(evidence.a.raw_density),
            "raw_density_b": _optional_float(evidence.b.raw_density),
            "u_a": float(risk.u_a), "u_b": float(risk.u_b),
            "joint_risk": float(risk.value),
            "joint_density": None if density is None else float(density.value),
            "active_worst": risk.active_worst.value,
            "objective_mode": risk.mode.value,
            "objective_digest": risk.objective_digest,
            "calibration_digest": evidence.calibration_digest,
            "head_binding_digest_a": digest_a, "head_binding_digest_b": digest_b,
            "hotspot_b_max_increase": (
                None if hotspot is None else float(hotspot.max_increase)),
            "hotspot_b_positive_mass": (
                None if hotspot is None else float(hotspot.positive_mass)),
            "hotspot_b_positive_count": (
                None if hotspot is None else int(hotspot.positive_count)),
        })
    return rows


def dual_feedback_evidence_rows(events: Sequence[Mapping[str, Any]], *, arm: str) -> list[dict]:
    """One row per position the joint law attributed, from the decisions that actually ran.

    Read off the DECISION's own evidence rather than recomputed: a second computation here would be
    a different measurement of the same cycle, and the two could disagree with nothing to say which
    one the run acted on.
    """
    rows = []
    for event in events:
        evidence = event.get("policy_evidence")
        dual = getattr(evidence, "dual", None)
        if dual is None:
            continue
        transition_id = str(event.get("transition_id") or _transition_of(event))
        protein_id = str(event.get("protein_id") or _protein_of(event))
        gate = getattr(evidence, "donor_gate", None)
        joint = getattr(gate, "joint", None)
        # Read off the DECISION EVIDENCE's own per-position rows, not off a "decision" key: the
        # event mappings the shard builds carry no such key, so `selected` was structurally always
        # False and `selection_rank` always None -- two columns that looked measured and were not.
        write_by_position = {int(row.position): row
                             for row in getattr(evidence, "write_candidates", ())}
        reopen_selection = {int(row.position): row
                            for row in getattr(evidence, "reopen_candidates", ())}
        reopen_by_position = {int(row.position): row for row in dual.reopen}
        contribution_by_position = {int(c.position): c for c in dual.contributions}
        # The UNION of the two position sets, not the write set alone.  Write candidates are the
        # source-MASKED positions and reopen candidates the source-RESOLVED ones, so the two are
        # disjoint by construction: keying rows off the contributions alone emitted a reopen column
        # that was structurally always empty, and the union reducer's winner -- the one field that
        # distinguishes a joint reopen table from a single-Head one -- never appeared.
        for position in sorted(set(contribution_by_position) | set(reopen_by_position)):
            reopen = reopen_by_position.get(position)
            contribution = contribution_by_position.get(position)
            rows.append({
                "transition_id": transition_id, "arm": str(arm), "position": position,
                "protein_id": protein_id,
                "donor_endpoint_id": str(getattr(evidence, "donor_endpoint_id", "")),
                "incumbent_id": str(getattr(evidence, "incumbent_id", "") or ""),
                "donor_joint_value": (None if joint is None else float(joint.donor_value)),
                "incumbent_joint_value": (
                    None if joint is None else float(joint.incumbent_value)),
                "donor_margin": (None if joint is None else float(joint.margin)),
                # None throughout for a REOPEN-only position: it was never a write candidate, so
                # no counterfactual was scored for it.  Absent, not zero.
                "counterfactual_sequence_md5": (
                    "" if contribution is None else contribution.counterfactual_sequence_md5),
                "raw_a": _optional_float(getattr(contribution, "raw_a", None)),
                "raw_b": _optional_float(getattr(contribution, "raw_b", None)),
                "contribution_a": _optional_float(
                    getattr(contribution, "contribution_a", None)),
                "contribution_b": _optional_float(
                    getattr(contribution, "contribution_b", None)),
                "joint_contribution": (
                    None if contribution is None else float(contribution.contribution)),
                # Against the floor the run ACTUALLY applied, carried on the decision evidence.
                # Recomputing on 0.0 made the table contradict the decision it claims to read off.
                "write_eligible": (
                    None if contribution is None
                    else bool(contribution.contribution > float(dual.write_epsilon))),
                "selection_rank": _selection_rank(
                    write_by_position.get(position), reopen_selection.get(position)),
                "selected": _was_selected(
                    write_by_position.get(position), reopen_selection.get(position)),
                "reopen_new_hotspot_a": _conjunct(reopen, "new_hotspot", "value_a"),
                "reopen_new_hotspot_b": _conjunct(reopen, "new_hotspot", "value_b"),
                "reopen_worsening_a": _conjunct(reopen, "worsening", "value_a"),
                "reopen_worsening_b": _conjunct(reopen, "worsening", "value_b"),
                "reopen_residual_a": _conjunct(reopen, "residual_burden", "value_a"),
                "reopen_residual_b": _conjunct(reopen, "residual_burden", "value_b"),
                "reopen_winning_allele": "" if reopen is None else reopen.winning_allele,
                "objective_digest": str(dual.objective_digest),
            })
    return rows


def dual_terminal_summary_rows(evidence_by_endpoint: Mapping[str, Any], *, arm: str,
                               protein_id: str, root_id: str, n_definitive: int,
                               d0_endpoint_ids: Sequence[str] = (),
                               last_improving_depth: int | None = None,
                               typed_stop: str = "") -> list[dict]:
    """One row per (protein, root, arm): where the joint objective ended up."""
    if not evidence_by_endpoint:
        return []
    best_id = min(evidence_by_endpoint,
                  key=lambda k: (float(evidence_by_endpoint[k].risk.value), k))
    best = evidence_by_endpoint[best_id]
    d0 = [evidence_by_endpoint[k] for k in d0_endpoint_ids if k in evidence_by_endpoint]
    return [{
        "protein_id": str(protein_id), "root_id": str(root_id), "arm": str(arm),
        "n_endpoints": len(evidence_by_endpoint), "n_definitive": int(n_definitive),
        "best_joint_risk": float(best.risk.value), "best_joint_risk_endpoint_id": str(best_id),
        "best_raw_risk_a": float(best.a.raw_risk), "best_raw_risk_b": float(best.b.raw_risk),
        "d0_joint_risk": (None if not d0 else min(float(e.risk.value) for e in d0)),
        "last_improving_depth": last_improving_depth,
        "typed_stop": str(typed_stop),
        "objective_digest": best.risk.objective_digest,
        "calibration_digest": best.calibration_digest,
    }]


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _selection_rank(write_row: Any, reopen_row: Any) -> int | None:
    """The rank this position took in whichever set it belonged to; ``None`` if it never ranked."""
    for row in (write_row, reopen_row):
        rank = getattr(row, "selection_rank", None)
        if rank is not None:
            return int(rank)
    return None


def _was_selected(write_row: Any, reopen_row: Any) -> bool:
    return bool(getattr(write_row, "selected", False) or getattr(reopen_row, "selected", False))


def _conjunct(reopen: Any, name: str, side: str) -> float | None:
    if reopen is None:
        return None
    return _optional_float(getattr(getattr(reopen, name), side))


def _transition_of(event: Mapping[str, Any]) -> str:
    projected = event.get("projected")
    return str(getattr(projected, "origin_transition_id", "") or "")


def _protein_of(event: Mapping[str, Any]) -> str:
    source = event.get("source")
    return str(getattr(getattr(source, "lineage", None), "protein_id", "") or "")


def write_v2_bundle(
    out_dir: Any, *, manifest: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping]],
    ledger_events: Sequence[Any] = (),
    dual_tables: Mapping[str, Sequence[Mapping]] | None = None,
) -> None:
    """Write the whole evidence bundle for one shard.

    Every declared table is written even when empty, and a MISSING table is an error rather than an
    omission: a bundle that silently skipped one would read as "this run produced none of that",
    which is a different claim entirely.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    missing = sorted(set(V2_TABLE_SCHEMAS) - set(tables))
    if missing:
        raise V2ArtifactError(
            f"bundle is missing required table(s) {missing}; an absent table reads as 'this run "
            "produced none of that', which is a claim rather than an omission"
        )
    unknown = sorted(set(tables) - set(V2_TABLE_SCHEMAS))
    if unknown:
        raise V2ArtifactError(f"bundle carries undeclared table(s) {unknown}")
    if dual_tables is not None:
        dual_missing = sorted(set(DUAL_TABLE_SCHEMAS) - set(dual_tables))
        if dual_missing:
            raise V2ArtifactError(
                f"a Dual bundle is missing required table(s) {dual_missing}; a run that declared "
                "an overlay and then omitted one of its tables reads as 'this run produced none of "
                "that'"
            )
        dual_unknown = sorted(set(dual_tables) - set(DUAL_TABLE_SCHEMAS))
        if dual_unknown:
            raise V2ArtifactError(f"bundle carries undeclared Dual table(s) {dual_unknown}")

    for name, schema in V2_TABLE_SCHEMAS.items():
        rows = list(tables[name])
        keys = [tuple(row.get(k) for k in schema.sort_by) for row in rows]
        if len(set(keys)) != len(keys):
            duplicated = sorted({key for key in keys if keys.count(key) > 1})
            raise V2ArtifactError(
                f"table {name!r} has duplicate join key(s) {duplicated[:3]}; a duplicated key "
                "silently multiplies rows in every downstream join"
            )
        write_stable_parquet(
            out / f"{name}.parquet", rows,
            columns=schema.columns, sort_by=schema.sort_by,
            types={c: V2_COLUMN_TYPES[c] for c in schema.columns if c in V2_COLUMN_TYPES},
        )

    if dual_tables is not None:
        # Same writer, same stable ordering, same duplicate-key refusal -- only the registry
        # differs, so a Dual table cannot acquire looser guarantees than a legacy one.
        for name, schema in DUAL_TABLE_SCHEMAS.items():
            rows = list(dual_tables[name])
            keys = [tuple(row.get(k) for k in schema.sort_by) for row in rows]
            if len(set(keys)) != len(keys):
                duplicated = sorted({key for key in keys if keys.count(key) > 1})
                raise V2ArtifactError(
                    f"table {name!r} has duplicate join key(s) {duplicated[:3]}"
                )
            write_stable_parquet(
                out / f"{name}.parquet", rows,
                columns=schema.columns, sort_by=schema.sort_by,
                types={c: DUAL_COLUMN_TYPES[c] for c in schema.columns},
            )

    write_manifest(out / "run_manifest.json", manifest)
    # Written even when EMPTY.  An absent file and an empty one are different claims -- "no ledger
    # was kept" versus "a ledger was kept and it is empty" -- and only the second is auditable.
    write_cost_ledger_jsonl(out / "cost_ledger.jsonl", _validated_ledger_events(ledger_events))


def _validated_ledger_events(events: Sequence[Any]) -> Sequence[Any]:
    """Check every ledger row against the typed event contract, and publish it UNCHANGED.

    A shard's ledger reaches this writer as PLAIN MAPPINGS: PLAN §5.4 routes it through resume, and
    a fragment is JSON on disk, so the dataclass does not survive the trip.  Rebuilding the event
    here re-runs the whole ``V2LedgerEvent`` contract -- the closed phase and status vocabularies,
    non-negative counters, and the rule that one row may not report both a measured and an unknown
    physical cost -- at the boundary where the ledger becomes an artifact.

    Without it, a row carrying a phase nothing aggregates would be published as a measurement and
    then silently skipped by every reader of that closed vocabulary, so the run's compute total
    would be understated with no error raised anywhere.

    What is written is the row as GIVEN, never the reconstruction: a writer that persisted the
    rebuilt event would fill absent fields with dataclass defaults, and a defaulted ``0.0`` in a
    physical column is indistinguishable in the artifact from a measurement of zero.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import (  # noqa: PLC0415
        V2LedgerError,
        V2LedgerEvent,
    )

    for index, event in enumerate(events):
        if isinstance(event, V2LedgerEvent):
            continue
        if not isinstance(event, Mapping):
            raise V2ArtifactError(
                f"ledger row {index} is a {type(event).__name__}; the compute ledger is typed "
                "evidence (PLAN §5.3), not free-form JSON"
            )
        try:
            V2LedgerEvent(**dict(event))
        except (V2LedgerError, TypeError) as exc:
            raise V2ArtifactError(f"ledger row {index} is not a V2LedgerEvent: {exc}") from exc
    return events
