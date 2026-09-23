#!/usr/bin/env python3
"""Materialize an exact V2 archive selection as a Phase-C/v0 ``generated.parquet``.

The archive table is a membership view, not sequence evidence.  This exporter therefore joins it
to ``complete_endpoints.parquet`` on the persisted ``endpoint_id`` and validates the redundant
content/protein/feasibility identities before selecting anything.  It never reconstructs lineage
from sequence bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion.state import sequence_md5  # noqa: E402
from inverse_folding.reference_flow.fusion.v1_alloc import (  # noqa: E402
    HeadInputError,
    validate_complete_aa20,
)
from inverse_folding.reference_flow.constraints import load_constraint_manifest  # noqa: E402
from inverse_folding.reference_flow.fusion_v2.errors import V2Error  # noqa: E402
from inverse_folding.reference_flow.fusion_v2.identity import (  # noqa: E402
    HeadEvaluatorIdentity,
    window_grid_digest,
)
from inverse_folding.reference_flow.official_selection import (  # noqa: E402
    two_axis_pareto_order,
)

__all__ = ["V2FacadeError", "materialize_archive_facade", "main"]


class V2FacadeError(V2Error):
    """The named evidence bundles cannot support an exact facade selection."""


_ARCHIVE_REQUIRED = {
    "endpoint_id",
    "endpoint_content_digest",
    "protein_id",
    "sequence_equivalence_key",
    "feasibility_level",
    "is_elite",
    "elite_rank",
}
_ENDPOINT_REQUIRED = {
    "endpoint_id",
    "endpoint_content_digest",
    "protein_id",
    "depth",
    "sequence",
    "sequence_md5",
    "sequence_equivalence_key",
    "sequence_length",
    "head_evaluator_digest",
    "head_window_grid_digest",
    "head_global_risk",
    "feasibility_level",
    "structure_evaluated",
    "structure_feasible",
}
_OUTPUT_COLUMNS = (
    "protein_id",
    "design_idx",
    "sequence",
    "entry_source_id",
    "endpoint_id",
    "endpoint_content_digest",
    "depth",
    "head_global_risk",
    "head_evaluator_digest",
    "head_window_grid_digest",
    "sequence_md5",
)
_MULTIROOT_MODES = (
    "feasible_immune_pareto",
    "structure_rejected_fallback",
)
_MULTIROOT_SPLIT_ROLES = frozenset({
    "exploratory_highrisk_ceiling_v1",
    "exploratory_highrisk_ceiling_v2",
    "exploratory_highrisk_breadth_ceiling_v1",
    "exploratory_testset_design_v1",
    # Compatibility identity of the already-launched 2026-08-22 DRB1*04:01 fast campaign. New
    # test-set campaigns use the generic role above; accepting this exact historical role lets the
    # same four-root evidence validator close that campaign without relabelling its manifests.
    "exploratory_fast_generalization_0401",
})
_MULTIROOT_COMMON_CONTENT = (
    "cohort_table",
    "dplm_checkpoint",
    "head_checkpoint",
    "head_config",
    "projection_policy_spec",
    "reference_sequences",
    "rf_sampler_config",
    "schedule_band_calibration",
    "structure_backend",
    "structure_config",
    "v0_structure_gate_config",
)
_MULTIROOT_PROTEIN_CONTENT = (
    "backbone",
    "complete_reference_sequence",
    "constraint_manifest",
    "fixed_token_policy",
)
_MULTIROOT_EXPECTED_SEED_COUNT = 4
_MULTIROOT_SEED_SCHEMA = "v2seed-1"
_MULTIROOT_SEED_NAMESPACES = frozenset({
    "a2_extra_lookahead",
    "matched_descendant",
    "v2_depth0_root",
    "v2_lookahead",
})
_MULTIROOT_CAP_INTEGER_FIELDS = (
    "max_definitive_refolds",
    "max_gpu_seconds",
    "max_head_calls",
    "max_logical_dfe",
    "max_retries",
    "max_walltime_s",
)
_MULTIROOT_CAP_KEYS = frozenset((*_MULTIROOT_CAP_INTEGER_FIELDS, "retry_scope"))
_MASTER_SEED_GRID_SCHEMA = "rf-fusion-v2-master-seed-grid-v1"
_SELECTION_PROVENANCE_SCHEMA = "rf-fusion-v2-selection-provenance-v1"
_MULTIROOT_ENDPOINT_REQUIRED = {"head_score_json", "root_id"}
_STRUCTURE_REQUIRED = {
    "endpoint_id",
    "protein_id",
    "depth",
    "sequence_md5",
    "evaluated",
    "feasible",
    "metrics_json",
    "feasibility_level",
    "structure_backend_digest",
    "v0_structure_gate_config_digest",
}
_TERMINAL_REQUIRED = {
    "endpoint_id",
    "sequence_md5",
    "structure_definitive",
    "structure_feasible",
    "structure_metrics_json",
    "immune_evaluator",
    "immune_global_risk",
    "immune_passed",
}
_MULTIROOT_OUTPUT_COLUMNS = _OUTPUT_COLUMNS + (
    "head_positive_mass_density",
    "pareto_rank",
    "selection_status",
    "terminal_validated",
    "structure_evaluated",
    "structure_feasible",
    "feasibility_level",
    "n_source_endpoints",
    "n_source_roots",
    "source_endpoint_ids_json",
    "source_root_ids_json",
    "source_provenance_json",
    "master_seed_grid_json",
    "master_seed_grid_digest",
    "selection_provenance_digest",
)
_FALLBACK_OUTPUT_COLUMNS = _MULTIROOT_OUTPUT_COLUMNS + ("scTM", "pLDDT")

# These fields define the one exploratory experiment whose per-protein bundles may be merged into
# a cohort facade.  Protein-bound content (backbone, constraints, band cell, safety reference and
# the resolved config digest) is deliberately absent: requiring those to match would make every
# legitimate multi-protein cohort unmergeable.
_COMMON_MANIFEST_FIELDS = (
    "campaign_id",
    "phase",
    "split_role",
    "code_revision",
    "schedule_id",
    "coordinate_law",
    "depth_cap",
    "exploratory_depth_override",
    "production_depth_authorized",
    "structure_backend",
    "v0_structure_gate_config",
    "structure_config",
)
_REQUIRED_STRUCTURE_CONTENT = ("structure_backend", "v0_structure_gate_config")


def _bundle_root(raw: str | Path) -> Path:
    path = Path(raw).expanduser().resolve()
    if path.is_file() and path.name in {"archive.parquet", "complete_endpoints.parquet"}:
        path = path.parent
    if not path.is_dir():
        raise V2FacadeError(f"V2 bundle is not a directory: {path}")
    return path


def _required_manifest_text(manifest: Mapping[str, Any], field: str, *, path: Path) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value.strip():
        raise V2FacadeError(f"{path} requires non-empty string {field}")
    return value


def _load_manifest_identity(root: Path) -> dict[str, Any]:
    """Read one bundle's run identity before any endpoint tables are merged.

    The exporter is intentionally specific to the unblinded uricase capability ladder.  A bundle
    from qualification, production authorization, or another exploratory split is valid evidence
    for its own question, but it is not a cell in this cohort and must not be made one by a parquet
    concatenation.
    """
    path = root / "run_manifest.json"
    if not path.is_file():
        raise V2FacadeError(f"V2 bundle is missing run_manifest.json: {root}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V2FacadeError(f"cannot read valid JSON run_manifest {path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise V2FacadeError(f"{path} must contain a JSON object")

    phase = _required_manifest_text(manifest, "phase", path=path)
    if phase != "capability_ladder":
        raise V2FacadeError(
            f"{path} phase must be 'capability_ladder', got {phase!r}"
        )
    split_role = _required_manifest_text(manifest, "split_role", path=path)
    if split_role != "exploratory_uricase":
        raise V2FacadeError(
            f"{path} split_role must be 'exploratory_uricase', got {split_role!r}"
        )
    if manifest.get("exploratory_depth_override") is not True:
        raise V2FacadeError(
            f"{path} requires exploratory_depth_override=true; an ordinary or production D>1 "
            "run cannot enter this exploratory facade"
        )
    if manifest.get("production_depth_authorized") is not False:
        raise V2FacadeError(
            f"{path} requires production_depth_authorized=false; exploratory and production "
            "evidence are non-interchangeable"
        )

    depth_cap = manifest.get("depth_cap")
    if isinstance(depth_cap, bool) or not isinstance(depth_cap, int) or depth_cap < 2:
        raise V2FacadeError(f"{path} requires integer depth_cap >= 2, got {depth_cap!r}")
    content = manifest.get("content_identities")
    if not isinstance(content, Mapping):
        raise V2FacadeError(f"{path} requires object content_identities")
    structure: dict[str, str | None] = {}
    for role in _REQUIRED_STRUCTURE_CONTENT:
        value = content.get(role)
        if not isinstance(value, str) or not value.strip():
            raise V2FacadeError(
                f"{path} requires content identity {role!r}; the common structure instrument "
                "cannot be inferred from endpoint rows"
            )
        structure[role] = value
    optional_structure = content.get("structure_config")
    if optional_structure is not None and (
        not isinstance(optional_structure, str) or not optional_structure.strip()
    ):
        raise V2FacadeError(
            f"{path} content identity 'structure_config' must be a non-empty string when present"
        )
    structure["structure_config"] = optional_structure

    return {
        "campaign_id": _required_manifest_text(manifest, "campaign_id", path=path),
        "phase": phase,
        "split_role": split_role,
        "code_revision": _required_manifest_text(manifest, "code_revision", path=path),
        "schedule_id": _required_manifest_text(manifest, "schedule_id", path=path),
        "coordinate_law": _required_manifest_text(manifest, "coordinate_law", path=path),
        "depth_cap": depth_cap,
        "exploratory_depth_override": True,
        "production_depth_authorized": False,
        **structure,
    }


def _validate_common_manifest_identity(roots: Sequence[Path]) -> None:
    identities = [(root, _load_manifest_identity(root)) for root in roots]
    baseline_root, baseline = identities[0]
    for root, identity in identities[1:]:
        for field in _COMMON_MANIFEST_FIELDS:
            if identity[field] != baseline[field]:
                raise V2FacadeError(
                    f"V2 bundle manifest identity mismatch for {field}: "
                    f"{baseline_root} has {baseline[field]!r}, {root} has {identity[field]!r}"
                )


def _load_multiroot_caps_identity(
    manifest: Mapping[str, Any], *, path: Path,
) -> tuple[tuple[str, object], ...]:
    caps = manifest.get("caps")
    if not isinstance(caps, Mapping):
        raise V2FacadeError(f"{path} caps must be an object")
    observed_keys = frozenset(caps)
    if observed_keys != _MULTIROOT_CAP_KEYS:
        raise V2FacadeError(
            f"{path} caps must contain exactly {sorted(_MULTIROOT_CAP_KEYS)}, "
            f"got {sorted(map(str, observed_keys))}"
        )
    for field in _MULTIROOT_CAP_INTEGER_FIELDS:
        value = caps[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise V2FacadeError(
                f"{path} caps.{field} must be an integer >= 0, got {value!r}"
            )
    retry_scope = caps["retry_scope"]
    if not isinstance(retry_scope, str) or not retry_scope.strip():
        raise V2FacadeError(
            f"{path} caps.retry_scope must be a non-empty string"
        )

    realized = manifest.get("realized_caps")
    if not isinstance(realized, Mapping):
        raise V2FacadeError(f"{path} realized_caps must be an object")
    if realized.get("breached") != []:
        raise V2FacadeError(
            f"{path} realized_caps.breached must be empty, "
            f"got {realized.get('breached')!r}"
        )
    if realized.get("unverifiable") != []:
        raise V2FacadeError(
            f"{path} realized_caps.unverifiable must be empty, "
            f"got {realized.get('unverifiable')!r}"
        )
    if realized.get("within") is not True:
        raise V2FacadeError(f"{path} requires realized_caps.within=true")

    return tuple((field, caps[field]) for field in sorted(_MULTIROOT_CAP_KEYS))


def _load_multiroot_manifest_identity(root: Path) -> dict[str, Any]:
    """Read the closed high-risk identity used only by explicit multiroot modes."""
    path = root / "run_manifest.json"
    if not path.is_file():
        raise V2FacadeError(f"V2 bundle is missing run_manifest.json: {root}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V2FacadeError(f"cannot read valid JSON run_manifest {path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise V2FacadeError(f"{path} must contain a JSON object")

    caps_identity = _load_multiroot_caps_identity(manifest, path=path)

    schema_version = _required_manifest_text(manifest, "schema_version", path=path)
    if schema_version != "v2cfg-1":
        raise V2FacadeError(
            f"{path} schema_version must be 'v2cfg-1', got {schema_version!r}"
        )

    phase = _required_manifest_text(manifest, "phase", path=path)
    if phase != "capability_ladder":
        raise V2FacadeError(
            f"{path} phase must be 'capability_ladder', got {phase!r}"
        )
    split_role = _required_manifest_text(manifest, "split_role", path=path)
    if split_role not in _MULTIROOT_SPLIT_ROLES:
        raise V2FacadeError(
            f"{path} split_role must be one of {sorted(_MULTIROOT_SPLIT_ROLES)}, "
            f"got {split_role!r}"
        )
    if manifest.get("exploratory_depth_override") is not True:
        raise V2FacadeError(
            f"{path} requires exploratory_depth_override=true for multiroot selection"
        )
    if manifest.get("production_depth_authorized") is not False:
        raise V2FacadeError(
            f"{path} requires production_depth_authorized=false for multiroot selection"
        )
    depth_cap = manifest.get("depth_cap")
    if isinstance(depth_cap, bool) or not isinstance(depth_cap, int) or depth_cap < 2:
        raise V2FacadeError(f"{path} requires integer depth_cap >= 2, got {depth_cap!r}")
    master_seed = manifest.get("master_seed")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise V2FacadeError(
            f"{path} requires integer master_seed >= 0, got {master_seed!r}"
        )
    arm_role = _required_manifest_text(manifest, "arm_role", path=path)
    if arm_role != "v2":
        raise V2FacadeError(f"{path} arm_role must be 'v2', got {arm_role!r}")
    if manifest.get("feedback_enabled") is not True:
        raise V2FacadeError(f"{path} requires feedback_enabled=true")
    active_population_width = manifest.get("active_population_width")
    if (
        isinstance(active_population_width, bool)
        or not isinstance(active_population_width, int)
        or active_population_width != 1
    ):
        raise V2FacadeError(
            f"{path} active_population_width must be integer 1, "
            f"got {active_population_width!r}"
        )
    a2_matching_resource = _required_manifest_text(
        manifest, "a2_matching_resource", path=path,
    )
    if a2_matching_resource != "definitive_refolds":
        raise V2FacadeError(
            f"{path} a2_matching_resource must be 'definitive_refolds', "
            f"got {a2_matching_resource!r}"
        )
    seed_schema = _required_manifest_text(manifest, "seed_schema", path=path)
    if seed_schema != _MULTIROOT_SEED_SCHEMA:
        raise V2FacadeError(
            f"{path} seed_schema must be {_MULTIROOT_SEED_SCHEMA!r}, got {seed_schema!r}"
        )
    seed_namespaces_raw = manifest.get("seed_namespaces")
    if not isinstance(seed_namespaces_raw, list) or any(
        not isinstance(value, str) or not value.strip() for value in seed_namespaces_raw
    ):
        raise V2FacadeError(f"{path} seed_namespaces must be a list of non-empty strings")
    if len(set(seed_namespaces_raw)) != len(seed_namespaces_raw):
        raise V2FacadeError(f"{path} seed_namespaces contains duplicates")
    seed_namespaces = frozenset(seed_namespaces_raw)
    if seed_namespaces != _MULTIROOT_SEED_NAMESPACES:
        raise V2FacadeError(
            f"{path} seed_namespaces must equal {sorted(_MULTIROOT_SEED_NAMESPACES)}, "
            f"got {sorted(seed_namespaces)}"
        )

    requested_cohort = manifest.get("requested_cohort")
    if (
        not isinstance(requested_cohort, list)
        or len(requested_cohort) != 1
        or not isinstance(requested_cohort[0], str)
        or not requested_cohort[0].strip()
    ):
        raise V2FacadeError(
            f"{path} requested_cohort must contain exactly one non-empty protein id"
        )
    requested_protein = requested_cohort[0]
    if manifest.get("accepted_fragments") != [requested_protein]:
        raise V2FacadeError(
            f"{path} accepted_fragments must exactly equal requested_cohort"
        )
    if manifest.get("missing_proteins") != []:
        raise V2FacadeError(f"{path} missing_proteins must be empty")
    if manifest.get("rejected_fragments") != []:
        raise V2FacadeError(f"{path} rejected_fragments must be empty")
    n_ok = manifest.get("n_ok")
    if isinstance(n_ok, bool) or not isinstance(n_ok, int) or n_ok not in {0, 1}:
        raise V2FacadeError(f"{path} n_ok must be integer 0 or 1, got {n_ok!r}")

    content = manifest.get("content_identities")
    if not isinstance(content, Mapping):
        raise V2FacadeError(f"{path} requires object content_identities")
    required_roles = _MULTIROOT_COMMON_CONTENT + _MULTIROOT_PROTEIN_CONTENT
    content_identity: dict[str, str] = {}
    for role in required_roles:
        value = content.get(role)
        if not isinstance(value, str) or not value.strip():
            raise V2FacadeError(f"{path} requires non-empty content identity {role!r}")
        content_identity[role] = value

    return {
        "schema_version": schema_version,
        "campaign_id": _required_manifest_text(manifest, "campaign_id", path=path),
        "phase": phase,
        "split_role": split_role,
        "code_revision": _required_manifest_text(manifest, "code_revision", path=path),
        "schedule_id": _required_manifest_text(manifest, "schedule_id", path=path),
        "coordinate_law": _required_manifest_text(manifest, "coordinate_law", path=path),
        "depth_cap": depth_cap,
        "master_seed": master_seed,
        "requested_protein": requested_protein,
        "arm_role": arm_role,
        "feedback_enabled": True,
        "active_population_width": active_population_width,
        "a2_matching_resource": a2_matching_resource,
        "seed_schema": seed_schema,
        "seed_namespaces": tuple(sorted(seed_namespaces)),
        "caps_identity": caps_identity,
        "exploratory_depth_override": True,
        "production_depth_authorized": False,
        "content_identities": content_identity,
    }


def _validate_multiroot_manifest_identity(roots: Sequence[Path]) -> dict[Path, dict[str, Any]]:
    identities = {root: _load_multiroot_manifest_identity(root) for root in roots}
    baseline_root = roots[0]
    baseline = identities[baseline_root]
    common_fields = (
        "schema_version",
        "campaign_id",
        "phase",
        "split_role",
        "code_revision",
        "schedule_id",
        "coordinate_law",
        "depth_cap",
        "exploratory_depth_override",
        "production_depth_authorized",
        "arm_role",
        "feedback_enabled",
        "active_population_width",
        "a2_matching_resource",
        "seed_schema",
        "seed_namespaces",
        "caps_identity",
    )
    for root in roots[1:]:
        identity = identities[root]
        for field in common_fields:
            if identity[field] != baseline[field]:
                raise V2FacadeError(
                    f"V2 multiroot manifest identity mismatch for {field}: "
                    f"{baseline_root} has {baseline[field]!r}, "
                    f"{root} has {identity[field]!r}"
                )
        for role in _MULTIROOT_COMMON_CONTENT:
            expected = baseline["content_identities"][role]
            observed = identity["content_identities"][role]
            if observed != expected:
                raise V2FacadeError(
                    f"V2 multiroot content identity mismatch for {role}: "
                    f"{baseline_root} has {expected!r}, {root} has {observed!r}"
                )
    return identities


def _require_columns(frame: pd.DataFrame, required: set[str], *, table: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise V2FacadeError(f"{table} is missing required columns {missing}")


def _require_unique_endpoint_ids(frame: pd.DataFrame, *, table: Path) -> None:
    if frame["endpoint_id"].isna().any():
        raise V2FacadeError(f"{table} contains null endpoint_id")
    if frame["endpoint_id"].astype(str).str.len().eq(0).any():
        raise V2FacadeError(f"{table} contains empty endpoint_id")
    duplicate = frame.loc[frame["endpoint_id"].duplicated(keep=False), "endpoint_id"]
    if not duplicate.empty:
        raise V2FacadeError(
            f"{table} contains duplicate endpoint_id values {sorted(set(map(str, duplicate)))[:3]}"
        )


def _strict_bool(value: object, *, field: str, endpoint_id: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise V2FacadeError(
            f"endpoint {endpoint_id} has non-boolean {field}={value!r}; no truthy coercion allowed"
        )
    return bool(value)


def _load_joined_bundle(
    root: Path, *, endpoint_required: set[str] | None = None,
) -> pd.DataFrame:
    archive_path = root / "archive.parquet"
    endpoint_path = root / "complete_endpoints.parquet"
    for path in (archive_path, endpoint_path):
        if not path.is_file():
            raise V2FacadeError(f"V2 bundle is missing {path.name}: {root}")
    archive = pd.read_parquet(archive_path)
    endpoints = pd.read_parquet(endpoint_path)
    _require_columns(archive, _ARCHIVE_REQUIRED, table=archive_path)
    _require_columns(
        endpoints,
        _ENDPOINT_REQUIRED | set(endpoint_required or ()),
        table=endpoint_path,
    )
    _require_unique_endpoint_ids(archive, table=archive_path)
    _require_unique_endpoint_ids(endpoints, table=endpoint_path)

    joined = archive.merge(
        endpoints,
        on="endpoint_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_archive", "_endpoint"),
        indicator=True,
    )
    unmatched = joined[joined["_merge"] != "both"]
    if not unmatched.empty:
        examples = sorted(map(str, unmatched["endpoint_id"]))[:3]
        side = sorted(set(map(str, unmatched["_merge"])))
        raise V2FacadeError(
            f"bundle {root} has endpoint ids with no complete endpoint/archive peer: "
            f"merge_status={side} examples={examples}"
        )
    joined = joined.drop(columns=["_merge"])

    for field in (
        "endpoint_content_digest",
        "protein_id",
        "sequence_equivalence_key",
        "feasibility_level",
    ):
        left = joined[f"{field}_archive"].astype("string")
        right = joined[f"{field}_endpoint"].astype("string")
        mismatch = left.isna() | right.isna() | left.ne(right)
        if mismatch.any():
            ids = sorted(map(str, joined.loc[mismatch, "endpoint_id"]))[:3]
            raise V2FacadeError(
                f"bundle {root} has {field} identity mismatch for endpoint(s) {ids}"
            )

    joined["source_bundle"] = str(root)
    return joined


def _validate_endpoint_identity_row(row: pd.Series) -> None:
    endpoint_id = str(row["endpoint_id"])
    sequence = row["sequence"]
    try:
        expected_length = int(row["sequence_length"])
        validate_complete_aa20(sequence, expected_length)
    except (HeadInputError, TypeError, ValueError) as exc:
        raise V2FacadeError(
            f"endpoint {endpoint_id} is not complete uppercase AA20: {exc}"
        ) from exc
    observed_md5 = sequence_md5(sequence)
    if str(row["sequence_md5"]) != observed_md5:
        raise V2FacadeError(
            f"endpoint {endpoint_id} sequence_md5 does not digest its AA20 sequence"
        )
    if str(row["sequence_equivalence_key_endpoint"]) != observed_md5:
        raise V2FacadeError(
            f"endpoint {endpoint_id} sequence_equivalence_key does not equal sequence_md5"
        )
    try:
        risk = float(row["head_global_risk"])
    except (TypeError, ValueError) as exc:
        raise V2FacadeError(f"endpoint {endpoint_id} has invalid Head risk") from exc
    if not math.isfinite(risk):
        raise V2FacadeError(f"endpoint {endpoint_id} has non-finite Head risk {risk!r}")
    try:
        depth_value = float(row["depth"])
    except (TypeError, ValueError) as exc:
        raise V2FacadeError(f"endpoint {endpoint_id} has invalid depth") from exc
    if not math.isfinite(depth_value) or not depth_value.is_integer() or depth_value < 0:
        raise V2FacadeError(f"endpoint {endpoint_id} has invalid depth {row['depth']!r}")


def _validate_endpoint_row(row: pd.Series) -> None:
    endpoint_id = str(row["endpoint_id"])
    definitive = (
        str(row["feasibility_level_archive"]) == "definitive"
        and str(row["feasibility_level_endpoint"]) == "definitive"
    )
    evaluated = _strict_bool(
        row["structure_evaluated"], field="structure_evaluated", endpoint_id=endpoint_id,
    )
    feasible = _strict_bool(
        row["structure_feasible"], field="structure_feasible", endpoint_id=endpoint_id,
    )
    if not (definitive and evaluated and feasible):
        raise V2FacadeError(
            f"endpoint {endpoint_id} is not definitive-feasible: "
            f"archive_level={row['feasibility_level_archive']!r} "
            f"endpoint_level={row['feasibility_level_endpoint']!r} "
            f"evaluated={evaluated} feasible={feasible}"
        )
    _validate_endpoint_identity_row(row)


def _validate_terminal_candidate(row: pd.Series) -> None:
    endpoint_id = str(row["endpoint_id"])
    if row.get("terminal_merge_status") != "both":
        raise V2FacadeError(
            f"endpoint {endpoint_id} is missing terminal-validation evidence"
        )
    if str(row["sequence_md5_terminal"]) != str(row["sequence_md5"]):
        raise V2FacadeError(
            f"endpoint {endpoint_id} terminal sequence_md5 does not match endpoint evidence"
        )
    structure_definitive = _strict_bool(
        row["structure_definitive_terminal"],
        field="terminal_validation.structure_definitive",
        endpoint_id=endpoint_id,
    )
    structure_feasible = _strict_bool(
        row["structure_feasible_terminal"],
        field="terminal_validation.structure_feasible",
        endpoint_id=endpoint_id,
    )
    immune_passed = _strict_bool(
        row["immune_passed_terminal"],
        field="terminal_validation.immune_passed",
        endpoint_id=endpoint_id,
    )
    if not structure_definitive:
        raise V2FacadeError(
            f"endpoint {endpoint_id} terminal structure_definitive is false"
        )
    if not structure_feasible:
        raise V2FacadeError(
            f"endpoint {endpoint_id} terminal structure_feasible is false"
        )
    if not immune_passed:
        raise V2FacadeError(f"endpoint {endpoint_id} terminal immune_passed is false")
    if str(row["immune_evaluator_terminal"]) != str(row["head_evaluator_digest"]):
        raise V2FacadeError(
            f"endpoint {endpoint_id} terminal immune_evaluator does not match Head identity"
        )
    immune_risk_raw = row["immune_global_risk_terminal"]
    if isinstance(immune_risk_raw, (bool, np.bool_)):
        raise V2FacadeError(
            f"endpoint {endpoint_id} terminal immune_global_risk is not finite numeric evidence"
        )
    try:
        immune_risk = float(immune_risk_raw)
    except (TypeError, ValueError) as exc:
        raise V2FacadeError(
            f"endpoint {endpoint_id} terminal immune_global_risk is not finite numeric evidence"
        ) from exc
    endpoint_risk = float(row["head_global_risk"])
    if not math.isfinite(immune_risk) or immune_risk != endpoint_risk:
        raise V2FacadeError(
            f"endpoint {endpoint_id} terminal immune_global_risk does not exactly match Head risk"
        )
    structure_metrics = row["structure_metrics_json_terminal"]
    if not isinstance(structure_metrics, str) or not structure_metrics.strip():
        raise V2FacadeError(
            f"endpoint {endpoint_id} terminal structure_metrics_json is empty"
        )
    try:
        metrics_payload = json.loads(structure_metrics)
    except json.JSONDecodeError as exc:
        raise V2FacadeError(
            f"endpoint {endpoint_id} terminal structure_metrics_json is invalid: {exc}"
        ) from exc
    if not isinstance(metrics_payload, Mapping):
        raise V2FacadeError(
            f"endpoint {endpoint_id} terminal structure_metrics_json must contain an object"
        )


def _candidate_rows(
    joined: pd.DataFrame, *, require_terminal_evidence: bool = False,
) -> pd.DataFrame:
    eligible_mask = (
        joined["feasibility_level_archive"].eq("definitive")
        & joined["feasibility_level_endpoint"].eq("definitive")
        & joined["structure_evaluated"].eq(True)  # noqa: E712 - exact evidence value
        & joined["structure_feasible"].eq(True)  # noqa: E712 - exact evidence value
    )
    candidates = joined.loc[eligible_mask].copy()
    terminal_validated: list[bool] = []
    for _, row in candidates.iterrows():
        _validate_endpoint_row(row)
        if require_terminal_evidence:
            _validate_terminal_candidate(row)
        terminal_validated.append(require_terminal_evidence)
    if require_terminal_evidence:
        candidates["terminal_validated_evidence"] = terminal_validated
    return candidates


def _attach_structure_evaluations(
    root: Path,
    joined: pd.DataFrame,
    *,
    identity: Mapping[str, Any],
) -> pd.DataFrame:
    path = root / "structure_evaluations.parquet"
    if not path.is_file():
        raise V2FacadeError(f"V2 bundle is missing structure_evaluations.parquet: {root}")
    structure = pd.read_parquet(path)
    _require_columns(structure, _STRUCTURE_REQUIRED, table=path)
    _require_unique_endpoint_ids(structure, table=path)
    structure = structure.rename(columns={
        column: f"{column}_structure"
        for column in structure.columns if column != "endpoint_id"
    })
    merged = joined.merge(
        structure,
        on="endpoint_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        examples = sorted(map(str, unmatched["endpoint_id"]))[:3]
        raise V2FacadeError(
            f"bundle {root} has endpoint ids with no structure-evaluation peer: {examples}"
        )
    merged = merged.drop(columns=["_merge"])
    for endpoint_field, structure_field in (
        ("protein_id_endpoint", "protein_id_structure"),
        ("sequence_md5", "sequence_md5_structure"),
    ):
        left = merged[endpoint_field].astype("string")
        right = merged[structure_field].astype("string")
        mismatch = left.isna() | right.isna() | left.ne(right)
        if mismatch.any():
            ids = sorted(map(str, merged.loc[mismatch, "endpoint_id"]))[:3]
            raise V2FacadeError(
                f"bundle {root} has structure {structure_field} identity mismatch for {ids}"
            )
    endpoint_depth = pd.to_numeric(merged["depth"], errors="coerce")
    structure_depth = pd.to_numeric(merged["depth_structure"], errors="coerce")
    mismatch = endpoint_depth.isna() | structure_depth.isna() | endpoint_depth.ne(structure_depth)
    if mismatch.any():
        ids = sorted(map(str, merged.loc[mismatch, "endpoint_id"]))[:3]
        raise V2FacadeError(f"bundle {root} has structure depth identity mismatch for {ids}")
    endpoint_level = merged["feasibility_level_endpoint"].astype("string")
    structure_level = merged["feasibility_level_structure"].astype("string")
    mismatch = endpoint_level.isna() | structure_level.isna() | endpoint_level.ne(structure_level)
    if mismatch.any():
        ids = sorted(map(str, merged.loc[mismatch, "endpoint_id"]))[:3]
        raise V2FacadeError(
            f"bundle {root} has structure feasibility_level identity mismatch for {ids}"
        )
    for column, content_role in (
        ("structure_backend_digest_structure", "structure_backend"),
        ("v0_structure_gate_config_digest_structure", "v0_structure_gate_config"),
    ):
        expected = str(identity["content_identities"][content_role])
        observed = merged[column].astype("string")
        mismatch = observed.isna() | observed.ne(expected)
        if mismatch.any():
            ids = sorted(map(str, merged.loc[mismatch, "endpoint_id"]))[:3]
            raise V2FacadeError(
                f"bundle {root} has {column.removesuffix('_structure')} mismatch against "
                f"run_manifest authority for {ids}"
            )
    return merged


def _attach_terminal_validation(root: Path, joined: pd.DataFrame) -> pd.DataFrame:
    path = root / "terminal_validation.parquet"
    if not path.is_file():
        raise V2FacadeError(f"V2 bundle is missing terminal_validation.parquet: {root}")
    terminal = pd.read_parquet(path)
    _require_columns(terminal, _TERMINAL_REQUIRED, table=path)
    _require_unique_endpoint_ids(terminal, table=path)
    terminal = terminal.rename(columns={
        column: f"{column}_terminal"
        for column in terminal.columns if column != "endpoint_id"
    })
    merged = joined.merge(
        terminal,
        on="endpoint_id",
        how="outer",
        validate="one_to_one",
        indicator="terminal_merge_status",
    )
    orphan = merged[merged["terminal_merge_status"] == "right_only"]
    if not orphan.empty:
        examples = sorted(map(str, orphan["endpoint_id"]))[:3]
        raise V2FacadeError(
            f"bundle {root} has terminal-validation endpoint ids with no archive peer: {examples}"
        )
    return merged


def _prepare_multiroot_bundle(
    root: Path, *, mode: str, identity: Mapping[str, Any],
) -> tuple[pd.DataFrame, str]:
    joined = _load_joined_bundle(root, endpoint_required=_MULTIROOT_ENDPOINT_REQUIRED)
    if "root_id_endpoint" in joined.columns:
        root_values = joined["root_id_endpoint"].astype("string")
        if "root_id_archive" in joined.columns:
            archive_values = joined["root_id_archive"].astype("string")
            mismatch = root_values.isna() | archive_values.isna() | root_values.ne(archive_values)
            if mismatch.any():
                ids = sorted(map(str, joined.loc[mismatch, "endpoint_id"]))[:3]
                raise V2FacadeError(f"bundle {root} has root_id identity mismatch for {ids}")
    elif "root_id" in joined.columns:
        root_values = joined["root_id"].astype("string")
    else:  # pragma: no cover - guarded by endpoint_required, retained for schema clarity
        raise V2FacadeError(f"bundle {root} has no endpoint root_id")
    if root_values.isna().any() or root_values.str.len().eq(0).any():
        raise V2FacadeError(f"bundle {root} contains null or empty endpoint root_id")
    if root_values.nunique() != 1:
        raise V2FacadeError(f"bundle {root} contains more than one root_id")
    joined["source_root_id"] = root_values.astype(str)

    proteins = sorted(set(map(str, joined["protein_id_endpoint"].dropna().unique())))
    if len(proteins) != 1:
        raise V2FacadeError(
            f"multiroot bundle {root} must contain exactly one protein, found {proteins}"
        )
    if identity["requested_protein"] != proteins[0]:
        raise V2FacadeError(
            f"multiroot bundle {root} requested_cohort declares "
            f"{identity['requested_protein']!r}, but tables contain {proteins[0]!r}"
        )
    if mode == "structure_rejected_fallback":
        joined = _attach_structure_evaluations(root, joined, identity=identity)
    else:
        joined = _attach_terminal_validation(root, joined)
    return joined, proteins[0]


def _head_positive_mass_density(
    row: pd.Series,
    *,
    manifest_identity: Mapping[str, Any],
    window_grid_cache: dict[str, tuple[tuple[int, int, int], ...]],
) -> tuple[float, str, str]:
    endpoint_id = str(row["endpoint_id"])
    raw = row["head_score_json"]
    if not isinstance(raw, str) or not raw.strip():
        raise V2FacadeError(f"endpoint {endpoint_id} has empty head_score_json")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise V2FacadeError(
            f"endpoint {endpoint_id} has invalid head_score_json: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise V2FacadeError(f"endpoint {endpoint_id} head_score_json must contain an object")
    expected_protein = str(row["protein_id_endpoint"])
    if payload.get("protein_id") != expected_protein:
        raise V2FacadeError(
            f"endpoint {endpoint_id} head_score_json protein_id does not match endpoint row"
        )
    if payload.get("sequence_md5") != str(row["sequence_md5"]):
        raise V2FacadeError(
            f"endpoint {endpoint_id} head_score_json sequence_md5 does not match endpoint row"
        )
    payload_length = payload.get("sequence_length")
    sequence_length = int(row["sequence_length"])
    if (
        isinstance(payload_length, bool)
        or not isinstance(payload_length, int)
        or payload_length != sequence_length
    ):
        raise V2FacadeError(
            f"endpoint {endpoint_id} head_score_json sequence_length does not match endpoint row"
        )
    allele = payload.get("allele")
    if not isinstance(allele, str) or not allele.strip():
        raise V2FacadeError(
            f"endpoint {endpoint_id} head_score_json allele must be a non-empty string"
        )
    score_scale = payload.get("score_scale")
    if not isinstance(score_scale, str) or not score_scale.strip():
        raise V2FacadeError(
            f"endpoint {endpoint_id} head_score_json score_scale must be a non-empty string"
        )
    windows = payload.get("windows")
    if not isinstance(windows, list):
        raise V2FacadeError(
            f"endpoint {endpoint_id} head_score_json windows must be a list"
        )
    window_coordinates: list[tuple[int, int, int]] = []
    for index, window in enumerate(windows):
        if not isinstance(window, Mapping):
            raise V2FacadeError(
                f"endpoint {endpoint_id} head_score_json windows[{index}] must be an object"
            )
        z = window.get("z")
        if isinstance(z, bool):
            raise V2FacadeError(
                f"endpoint {endpoint_id} head_score_json windows[{index}].z is not finite"
            )
        try:
            z_value = float(z)
        except (TypeError, ValueError) as exc:
            raise V2FacadeError(
                f"endpoint {endpoint_id} head_score_json windows[{index}].z is not finite"
            ) from exc
        if not math.isfinite(z_value):
            raise V2FacadeError(
                f"endpoint {endpoint_id} head_score_json windows[{index}].z is not finite"
            )
        coordinate_values = []
        for field, minimum in (("start_0b", 0), ("end_0b", 0), ("k", 1)):
            value = window.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise V2FacadeError(
                    f"endpoint {endpoint_id} head_score_json windows[{index}].{field} "
                    f"must be integer >= {minimum}"
                )
            coordinate_values.append(value)
        window_coordinates.append(tuple(coordinate_values))
    if not window_coordinates:
        raise V2FacadeError(
            f"endpoint {endpoint_id} head_score_json carries an empty windows grid"
        )
    claimed_grid_digest = str(row["head_window_grid_digest"])
    expected_coordinates = window_grid_cache.get(claimed_grid_digest)
    if expected_coordinates is None:
        window_records = (
            SimpleNamespace(start_0b=start, end_0b=end, k=k)
            for start, end, k in window_coordinates
        )
        try:
            observed_grid_digest = window_grid_digest(window_records)
        except (AttributeError, TypeError, ValueError, V2Error) as exc:
            raise V2FacadeError(
                f"endpoint {endpoint_id} head_score_json carries an invalid windows grid: {exc}"
            ) from exc
        if observed_grid_digest != claimed_grid_digest:
            raise V2FacadeError(
                f"endpoint {endpoint_id} recomputed head_window_grid_digest does not match "
                "endpoint row"
            )
        expected_coordinates = tuple(sorted(window_coordinates))
        window_grid_cache[claimed_grid_digest] = expected_coordinates
    else:
        observed_coordinates = tuple(window_coordinates)
        if observed_coordinates != expected_coordinates:
            observed_coordinates = tuple(sorted(window_coordinates))
        if observed_coordinates != expected_coordinates:
            raise V2FacadeError(
                f"endpoint {endpoint_id} recomputed head_window_grid_digest does not match "
                "endpoint row"
            )
    window_ks = [coordinate[2] for coordinate in window_coordinates]
    try:
        observed_evaluator_digest = HeadEvaluatorIdentity(
            allele=allele,
            score_scale=score_scale,
            window_k_min=min(window_ks),
            window_k_max=max(window_ks),
            head_config_hash=manifest_identity["content_identities"]["head_config"],
            head_checkpoint_digest=(
                manifest_identity["content_identities"]["head_checkpoint"]
            ),
        ).digest()
    except (KeyError, TypeError, ValueError, V2Error) as exc:
        raise V2FacadeError(
            f"endpoint {endpoint_id} cannot reconstruct its HeadEvaluatorIdentity: {exc}"
        ) from exc
    if observed_evaluator_digest != str(row["head_evaluator_digest"]):
        raise V2FacadeError(
            f"endpoint {endpoint_id} recomputed head_evaluator_digest does not exactly match "
            "endpoint row"
        )
    hotspots = payload.get("residue_hotspot")
    if not isinstance(hotspots, list):
        raise V2FacadeError(
            f"endpoint {endpoint_id} head_score_json requires residue_hotspot list"
        )
    if len(hotspots) != sequence_length:
        raise V2FacadeError(
            f"endpoint {endpoint_id} residue_hotspot length {len(hotspots)} does not equal "
            f"sequence_length {sequence_length}"
        )
    values: list[float] = []
    for index, value in enumerate(hotspots):
        if isinstance(value, bool):
            raise V2FacadeError(
                f"endpoint {endpoint_id} residue_hotspot[{index}] is boolean"
            )
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise V2FacadeError(
                f"endpoint {endpoint_id} residue_hotspot[{index}] is not numeric"
            ) from exc
        if not math.isfinite(number):
            raise V2FacadeError(
                f"endpoint {endpoint_id} residue_hotspot[{index}] is non-finite"
            )
        values.append(number)
    if isinstance(payload.get("global_risk"), bool):
        raise V2FacadeError(
            f"endpoint {endpoint_id} head_score_json requires finite global_risk"
        )
    try:
        json_risk = float(payload["global_risk"])
    except (KeyError, TypeError, ValueError) as exc:
        raise V2FacadeError(
            f"endpoint {endpoint_id} head_score_json requires finite global_risk"
        ) from exc
    row_risk = float(row["head_global_risk"])
    if not math.isfinite(json_risk) or json_risk != row_risk:
        raise V2FacadeError(
            f"endpoint {endpoint_id} head_score_json global_risk does not match endpoint row"
        )
    density = math.fsum(max(value, 0.0) for value in values) / sequence_length
    return density, allele, score_scale


def _rejected_structure_metrics(row: pd.Series) -> tuple[float, float]:
    endpoint_id = str(row["endpoint_id"])
    _validate_endpoint_identity_row(row)
    evaluated = _strict_bool(
        row["evaluated_structure"], field="structure_evaluations.evaluated",
        endpoint_id=endpoint_id,
    )
    feasible = _strict_bool(
        row["feasible_structure"], field="structure_evaluations.feasible",
        endpoint_id=endpoint_id,
    )
    endpoint_feasible = _strict_bool(
        row["structure_feasible"], field="structure_feasible", endpoint_id=endpoint_id,
    )
    if not evaluated or feasible or endpoint_feasible:
        raise V2FacadeError(
            f"endpoint {endpoint_id} is not a definitive structure rejection: "
            f"evaluated={evaluated} structure_result_feasible={feasible} "
            f"endpoint_feasible={endpoint_feasible}"
        )
    raw = row["metrics_json_structure"]
    if not isinstance(raw, str) or not raw.strip():
        raise V2FacadeError(f"endpoint {endpoint_id} has empty structure metrics_json")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise V2FacadeError(
            f"endpoint {endpoint_id} has invalid structure metrics_json: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise V2FacadeError(f"endpoint {endpoint_id} structure metrics_json must be an object")
    values: list[float] = []
    for field in ("scTM", "pLDDT"):
        value = payload.get(field)
        if isinstance(value, bool):
            raise V2FacadeError(
                f"endpoint {endpoint_id} structure metrics_json requires finite {field}"
            )
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise V2FacadeError(
                f"endpoint {endpoint_id} structure metrics_json requires finite {field}"
            ) from exc
        if not math.isfinite(number):
            raise V2FacadeError(
                f"endpoint {endpoint_id} structure metrics_json requires finite {field}"
            )
        values.append(number)
    return values[0], values[1]


def _validate_one_head_domain(frame: pd.DataFrame) -> None:
    columns = [
        "head_evaluator_digest",
        "head_window_grid_digest",
        "head_payload_allele",
        "head_payload_score_scale",
    ]
    if frame[columns].isna().any().any():
        raise V2FacadeError("multiroot candidate pool carries null Head identity")
    for column in columns:
        if frame[column].astype(str).str.len().eq(0).any():
            raise V2FacadeError("multiroot candidate pool carries empty Head identity")
    if frame["head_evaluator_digest"].astype(str).nunique() != 1:
        raise V2FacadeError(
            "multiroot candidate pool does not share one Head evaluator identity"
        )
    payload_domains = frame[[
        "head_evaluator_digest", "head_payload_allele", "head_payload_score_scale",
    ]].astype(str).drop_duplicates()
    if len(payload_domains) != 1:
        raise V2FacadeError(
            "multiroot candidate pool does not share one Head allele/score-scale domain"
        )
    grid_counts = frame.assign(
        _protein=frame["protein_id_endpoint"].astype(str),
        _grid=frame["head_window_grid_digest"].astype(str),
    ).groupby("_protein")["_grid"].nunique()
    if grid_counts.gt(1).any():
        proteins = sorted(map(str, grid_counts[grid_counts.gt(1)].index))[:3]
        raise V2FacadeError(
            "multiroot candidate pool has multiple Head window-grid identities within "
            f"protein(s) {proteins}"
        )


def _collapse_multiroot_sequences(
    pool: pd.DataFrame, *, metric_columns: Sequence[str],
) -> pd.DataFrame:
    collapsed: list[pd.Series] = []
    grouped = pool.groupby(["protein_id_endpoint", "sequence_md5"], sort=True, dropna=False)
    for (protein_id, sequence_digest), group in grouped:
        ordered = group.sort_values("endpoint_id", kind="mergesort")
        if ordered["sequence"].nunique(dropna=False) != 1:
            raise V2FacadeError(
                f"protein {protein_id!r} sequence_md5 {sequence_digest!r} maps to multiple "
                "sequence byte strings"
            )
        for column in metric_columns:
            values = [float(value) for value in ordered[column]]
            if any(value != values[0] for value in values[1:]):
                raise V2FacadeError(
                    f"protein {protein_id!r} converged sequence {sequence_digest!r} has "
                    f"conflicting {column} values across roots"
                )
        for column in ("head_evaluator_digest", "head_window_grid_digest"):
            if ordered[column].astype(str).nunique() != 1:
                raise V2FacadeError(
                    f"protein {protein_id!r} converged sequence {sequence_digest!r} has "
                    f"conflicting {column} values across roots"
                )
        provenance = sorted(
            ({
                "endpoint_id": str(row["endpoint_id"]),
                "master_seed": int(row["source_master_seed"]),
                "root_id": str(row["source_root_id"]),
                "source_bundle": str(row["source_bundle"]),
                "depth": int(row["depth"]),
            } for _, row in ordered.iterrows()),
            key=lambda item: (
                item["master_seed"], item["root_id"], item["endpoint_id"],
                item["source_bundle"], item["depth"],
            ),
        )
        representative = ordered.iloc[0].copy()
        endpoint_ids = sorted({item["endpoint_id"] for item in provenance})
        root_identities = sorted({
            (item["master_seed"], item["root_id"])
            for item in provenance
        })
        root_ids = [
            {"master_seed": master_seed, "root_id": root_id}
            for master_seed, root_id in root_identities
        ]
        representative["n_source_endpoints"] = len(endpoint_ids)
        representative["n_source_roots"] = len(root_identities)
        representative["source_endpoint_ids_json"] = json.dumps(
            endpoint_ids, separators=(",", ":"),
        )
        representative["source_root_ids_json"] = json.dumps(
            root_ids, sort_keys=True, separators=(",", ":"),
        )
        representative["source_provenance_json"] = json.dumps(
            provenance, sort_keys=True, separators=(",", ":"),
        )
        collapsed.append(representative)
    return pd.DataFrame(collapsed)


def _first_pareto_front(
    frame: pd.DataFrame, *, minimize: Sequence[str], maximize: Sequence[str] = (),
) -> pd.DataFrame:
    objective_columns = list(minimize) + list(maximize)
    values = frame.loc[:, objective_columns].astype(float).to_numpy(copy=True)
    if maximize:
        values[:, len(minimize):] *= -1.0
    dominated = np.zeros(len(frame), dtype=bool)
    for index in range(len(frame)):
        no_worse = np.all(values <= values[index], axis=1)
        strictly_better = np.any(values < values[index], axis=1)
        no_worse[index] = False
        dominated[index] = bool(np.any(no_worse & strictly_better))
    return frame.loc[~dominated].copy()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_multiroot_constraints(
    pool: pd.DataFrame,
    *,
    identities: Mapping[Path, Mapping[str, Any]],
    constraint_manifest: str | Path | None,
) -> None:
    contributing_roots = sorted(set(map(str, pool["source_bundle"])))
    by_path = {str(path): identity for path, identity in identities.items()}
    constraint_digests = {
        by_path[root]["content_identities"]["constraint_manifest"]
        for root in contributing_roots
    }
    fixed_policy_digests = {
        by_path[root]["content_identities"]["fixed_token_policy"]
        for root in contributing_roots
    }
    if constraint_digests != fixed_policy_digests:
        raise V2FacadeError(
            "multiroot bundle constraint_manifest and fixed_token_policy identities disagree"
        )

    if constraint_manifest is None:
        # Import lazily so the legacy facade path remains independent of the materializer.
        from scripts.materialize_v2_canary_config import no_constraint_manifest_digest

        absence = no_constraint_manifest_digest()
        if constraint_digests != {absence}:
            raise V2FacadeError(
                "a constrained multiroot bundle requires an explicit constraint manifest"
            )
        return

    path = Path(constraint_manifest).expanduser().resolve()
    if not path.is_file():
        raise V2FacadeError(f"constraint manifest does not exist or is not a file: {path}")
    observed_digest = _sha256_file(path)
    if constraint_digests != {observed_digest}:
        raise V2FacadeError(
            f"constraint manifest digest {observed_digest} does not match contributing "
            f"bundle identities {sorted(constraint_digests)}"
        )
    try:
        manifest = load_constraint_manifest(path)
        for _, row in pool.iterrows():
            constraint = manifest.constraint_for_protein(str(row["protein_id_endpoint"]))
            constraint.validate_against_sequence(str(row["sequence"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise V2FacadeError(f"constraint manifest anchor validation failed: {exc}") from exc


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _validate_expected_master_seeds(
    expected_master_seeds: Sequence[int] | None,
) -> tuple[int, ...]:
    if expected_master_seeds is None:
        raise V2FacadeError("expected_master_seeds is required in multiroot Pareto modes")
    if isinstance(expected_master_seeds, (str, bytes)):
        raise V2FacadeError("expected_master_seeds must be a sequence of integers")
    seeds = list(expected_master_seeds)
    if len(seeds) != _MULTIROOT_EXPECTED_SEED_COUNT:
        raise V2FacadeError(
            f"multiroot Pareto modes require exactly {_MULTIROOT_EXPECTED_SEED_COUNT} "
            "distinct master seeds"
        )
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise V2FacadeError(
                f"expected_master_seeds must contain integers >= 0, got {seed!r}"
            )
    if len(set(seeds)) != len(seeds):
        raise V2FacadeError("expected_master_seeds contains duplicate values")
    return tuple(sorted(seeds))


def _master_seed_grid_provenance(
    identity: Mapping[str, Any], expected_master_seeds: Sequence[int],
) -> tuple[str, str]:
    payload = {
        "schema_version": _MASTER_SEED_GRID_SCHEMA,
        "campaign_id": identity["campaign_id"],
        "split_role": identity["split_role"],
        "schedule_id": identity["schedule_id"],
        "seed_schema": identity["seed_schema"],
        "seed_namespaces": list(identity["seed_namespaces"]),
        "expected_roots_per_protein": len(expected_master_seeds),
        "master_seeds": list(expected_master_seeds),
    }
    payload_json = _canonical_json(payload)
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    return payload_json, digest


def _selection_provenance_digest(
    row: pd.Series, *, mode: str, master_seed_grid_digest: str,
) -> str:
    payload = {
        "schema_version": _SELECTION_PROVENANCE_SCHEMA,
        "selection_status": mode,
        "protein_id": str(row["protein_id_endpoint"]),
        "sequence_md5": str(row["sequence_md5"]),
        "source_endpoint_ids": json.loads(str(row["source_endpoint_ids_json"])),
        "source_roots": json.loads(str(row["source_root_ids_json"])),
        "master_seed_grid_digest": master_seed_grid_digest,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _multiroot_output(
    front: pd.DataFrame,
    *,
    mode: str,
    selection_status: str,
    master_seed_grid_json: str,
    master_seed_grid_digest: str,
) -> pd.DataFrame:
    if "_official_selection_rank" in front.columns:
        sort_columns = ["protein_id_endpoint", "_official_selection_rank"]
        ascending = [True, True]
    elif mode == "feasible_immune_pareto":
        sort_columns = [
            "protein_id_endpoint", "head_global_risk", "head_positive_mass_density",
            "sequence_md5", "endpoint_id",
        ]
        ascending = [True, True, True, True, True]
    else:
        sort_columns = [
            "protein_id_endpoint", "head_positive_mass_density", "scTM", "pLDDT",
            "sequence_md5", "endpoint_id",
        ]
        ascending = [True, True, False, False, True, True]
    chosen = front.sort_values(sort_columns, ascending=ascending, kind="mergesort").copy()
    chosen["design_idx"] = chosen.groupby("protein_id_endpoint", sort=False).cumcount()
    chosen["selection_provenance_digest"] = [
        _selection_provenance_digest(
            row, mode=selection_status, master_seed_grid_digest=master_seed_grid_digest,
        )
        for _, row in chosen.iterrows()
    ]
    if mode == "feasible_immune_pareto":
        terminal_validated = chosen["terminal_validated_evidence"].astype(bool)
        structure_evaluated = chosen["structure_evaluated"].astype(bool)
        structure_feasible = chosen["structure_feasible"].astype(bool)
    else:
        terminal_validated = pd.Series(False, index=chosen.index, dtype=bool)
        structure_evaluated = chosen["evaluated_structure"].astype(bool)
        structure_feasible = chosen["feasible_structure"].astype(bool)
    out = pd.DataFrame({
        "protein_id": chosen["protein_id_endpoint"].astype(str),
        "design_idx": chosen["design_idx"].astype(int),
        "sequence": chosen["sequence"].astype(str),
        "entry_source_id": chosen["endpoint_id"].astype(str),
        "endpoint_id": chosen["endpoint_id"].astype(str),
        "endpoint_content_digest": chosen["endpoint_content_digest_endpoint"].astype(str),
        "depth": chosen["depth"].astype(int),
        "head_global_risk": chosen["head_global_risk"].astype(float),
        "head_evaluator_digest": chosen["head_evaluator_digest"].astype(str),
        "head_window_grid_digest": chosen["head_window_grid_digest"].astype(str),
        "sequence_md5": chosen["sequence_md5"].astype(str),
        "head_positive_mass_density": chosen["head_positive_mass_density"].astype(float),
        "pareto_rank": chosen.get(
            "_pareto_layer", pd.Series(1, index=chosen.index),
        ).astype(int),
        "selection_status": selection_status,
        "terminal_validated": terminal_validated,
        "structure_evaluated": structure_evaluated,
        "structure_feasible": structure_feasible,
        "feasibility_level": chosen["feasibility_level_endpoint"].astype(str),
        "n_source_endpoints": chosen["n_source_endpoints"].astype(int),
        "n_source_roots": chosen["n_source_roots"].astype(int),
        "source_endpoint_ids_json": chosen["source_endpoint_ids_json"].astype(str),
        "source_root_ids_json": chosen["source_root_ids_json"].astype(str),
        "source_provenance_json": chosen["source_provenance_json"].astype(str),
        "master_seed_grid_json": master_seed_grid_json,
        "master_seed_grid_digest": master_seed_grid_digest,
        "selection_provenance_digest": chosen["selection_provenance_digest"].astype(str),
    })
    columns = _MULTIROOT_OUTPUT_COLUMNS
    if mode == "structure_rejected_fallback":
        out["scTM"] = chosen["scTM"].astype(float)
        out["pLDDT"] = chosen["pLDDT"].astype(float)
        columns = _FALLBACK_OUTPUT_COLUMNS
    out = out.sort_values(["protein_id", "design_idx"], kind="mergesort").reset_index(drop=True)
    if out[["protein_id", "design_idx"]].duplicated().any():
        raise V2FacadeError("multiroot selection produced duplicate facade keys")
    if out["entry_source_id"].duplicated().any():
        raise V2FacadeError("multiroot selection produced duplicate representative endpoint ids")
    return out.loc[:, columns]


def _materialize_multiroot_selection(
    bundles: Sequence[str | Path],
    *,
    mode: str,
    official: bool,
    final_candidates_per_protein: int | None,
    constraint_manifest: str | Path | None,
    expected_master_seeds: Sequence[int] | None,
) -> pd.DataFrame:
    expected_seeds = _validate_expected_master_seeds(expected_master_seeds)
    expected_seed_set = frozenset(expected_seeds)
    if not bundles:
        raise V2FacadeError("at least one V2 bundle is required")
    roots = [_bundle_root(path) for path in bundles]
    if len(set(roots)) != len(roots):
        raise V2FacadeError("the same V2 bundle was supplied more than once")
    identities = _validate_multiroot_manifest_identity(roots)
    joined_by_root: list[tuple[Path, pd.DataFrame, str]] = []
    protein_content: dict[str, tuple[Path, dict[str, str]]] = {}
    protein_bundles: dict[str, list[tuple[Path, int]]] = {}
    for root in roots:
        frame, protein_id = _prepare_multiroot_bundle(
            root, mode=mode, identity=identities[root],
        )
        master_seed = int(identities[root]["master_seed"])
        frame["source_master_seed"] = master_seed
        content = identities[root]["content_identities"]
        observed = {role: content[role] for role in _MULTIROOT_PROTEIN_CONTENT}
        previous = protein_content.get(protein_id)
        if previous is not None and previous[1] != observed:
            raise V2FacadeError(
                f"protein {protein_id!r} has conflicting protein-bound content identities "
                f"across roots {previous[0]} and {root}"
            )
        protein_content[protein_id] = (root, observed)
        protein_bundles.setdefault(protein_id, []).append((root, master_seed))
        joined_by_root.append((root, frame, protein_id))

    for protein_id in sorted(protein_bundles):
        entries = protein_bundles[protein_id]
        seeds = [seed for _, seed in entries]
        duplicate_seeds = sorted({seed for seed in seeds if seeds.count(seed) > 1})
        if duplicate_seeds:
            raise V2FacadeError(
                f"protein {protein_id!r} has duplicate master_seed {duplicate_seeds[0]} "
                "across root bundles"
            )
        seed_set = frozenset(seeds)
        if seed_set != expected_seed_set or len(entries) != len(expected_seeds):
            missing = sorted(expected_seed_set - seed_set)
            unexpected = sorted(seed_set - expected_seed_set)
            raise V2FacadeError(
                f"protein {protein_id!r} does not realize the complete expected master-seed "
                f"grid: missing={missing}, unexpected={unexpected}, bundles={len(entries)}"
            )

    master_seed_grid_json, master_seed_grid_digest = _master_seed_grid_provenance(
        identities[roots[0]], expected_seeds,
    )

    joined = pd.concat([frame for _, frame, _ in joined_by_root], ignore_index=True)
    if joined.empty:
        raise V2FacadeError("the named V2 bundles contain no archived endpoints")
    duplicate_ids = joined.loc[joined["endpoint_id"].duplicated(keep=False), "endpoint_id"]
    if not duplicate_ids.empty:
        raise V2FacadeError(
            "endpoint_id is duplicated across V2 bundles: "
            f"{sorted(set(map(str, duplicate_ids)))[:3]}"
        )

    feasible = _candidate_rows(
        joined, require_terminal_evidence=mode == "feasible_immune_pareto",
    )
    if mode == "feasible_immune_pareto":
        pool = feasible.copy()
    else:
        successful_proteins = set(map(str, feasible["protein_id_endpoint"].unique()))
        pool = joined.loc[
            ~joined["protein_id_endpoint"].astype(str).isin(successful_proteins)
        ].copy()
    if pool.empty:
        raise V2FacadeError(f"{mode} candidate pool is empty")

    densities: list[float] = []
    head_alleles: list[str] = []
    head_score_scales: list[str] = []
    window_grid_cache: dict[str, tuple[tuple[int, int, int], ...]] = {}
    manifest_by_bundle = {str(root): identity for root, identity in identities.items()}
    structure_metrics: list[tuple[float, float]] = []
    for _, row in pool.iterrows():
        if mode == "structure_rejected_fallback":
            structure_metrics.append(_rejected_structure_metrics(row))
        source_bundle = str(row["source_bundle"])
        manifest_identity = manifest_by_bundle.get(source_bundle)
        if manifest_identity is None:  # pragma: no cover - source_bundle is set by the join
            raise V2FacadeError(
                f"endpoint {row['endpoint_id']} has no source manifest identity"
            )
        density, allele, score_scale = _head_positive_mass_density(
            row,
            manifest_identity=manifest_identity,
            window_grid_cache=window_grid_cache,
        )
        densities.append(density)
        head_alleles.append(allele)
        head_score_scales.append(score_scale)
    pool["head_positive_mass_density"] = densities
    pool["head_payload_allele"] = head_alleles
    pool["head_payload_score_scale"] = head_score_scales
    if structure_metrics:
        pool["scTM"] = [item[0] for item in structure_metrics]
        pool["pLDDT"] = [item[1] for item in structure_metrics]
    _validate_one_head_domain(pool)
    _validate_multiroot_constraints(
        pool, identities=identities, constraint_manifest=constraint_manifest,
    )

    metric_columns = ["head_global_risk", "head_positive_mass_density"]
    if mode == "structure_rejected_fallback":
        metric_columns.extend(["scTM", "pLDDT"])
    collapsed = _collapse_multiroot_sequences(pool, metric_columns=metric_columns)
    selected_groups: list[pd.DataFrame] = []
    for protein_id in sorted(set(map(str, collapsed["protein_id_endpoint"]))):
        protein = collapsed[
            collapsed["protein_id_endpoint"].astype(str) == protein_id
        ].copy()
        if final_candidates_per_protein is not None:
            first = protein["head_global_risk"].astype(float).tolist()
            if mode == "feasible_immune_pareto":
                second = protein["head_positive_mass_density"].astype(float).tolist()
            else:
                # The fallback front minimizes immune mass while maximizing scTM.
                first = protein["head_positive_mass_density"].astype(float).tolist()
                second = (-protein["scTM"].astype(float)).tolist()
            order = two_axis_pareto_order(
                first=first,
                second=second,
                tie_keys=protein["sequence_md5"].astype(str).tolist(),
            )
            order = order[:final_candidates_per_protein]
            picked = protein.iloc[[row.index for row in order]].copy()
            picked["_pareto_layer"] = [row.pareto_layer for row in order]
            picked["_official_selection_rank"] = [row.selection_rank for row in order]
        elif mode == "feasible_immune_pareto":
            picked = _first_pareto_front(
                protein,
                minimize=("head_global_risk", "head_positive_mass_density"),
            )
        else:
            picked = _first_pareto_front(
                protein,
                minimize=("head_positive_mass_density",),
                maximize=("scTM",),
            )
        selected_groups.append(picked)
    selected = pd.concat(selected_groups, ignore_index=True)
    if selected.empty:
        raise V2FacadeError(f"{mode} selection produced zero rows")
    if final_candidates_per_protein is None:
        selection_status = mode
    elif official:
        selection_status = f"official_{mode}"
    else:
        selection_status = f"bounded_{mode}"
    return _multiroot_output(
        selected,
        mode=mode,
        selection_status=selection_status,
        master_seed_grid_json=master_seed_grid_json,
        master_seed_grid_digest=master_seed_grid_digest,
    )


def materialize_archive_facade(
    bundles: Sequence[str | Path],
    *,
    mode: str,
    k: int | None = None,
    require_k: bool = False,
    official: bool = False,
    final_candidates_per_protein: int | None = None,
    constraint_manifest: str | Path | None = None,
    expected_master_seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Return a deterministic Phase-C facade selected from explicit V2 bundle evidence.

    ``top-k`` first collapses sequence-equivalent logical endpoints to the stable representative
    under ``(head_global_risk, endpoint_id)``.  Raw multiplicity remains intact in the source V2
    bundle, but cannot buy duplicate round-0 ancestry mass in the optional v0 suffix.

    The two multiroot modes are a separate high-risk selection interface. They alone permit the
    same protein in multiple root bundles; neither can relax the legacy definitive-feasible path.
    """
    if final_candidates_per_protein is not None and (
        isinstance(final_candidates_per_protein, bool)
        or not isinstance(final_candidates_per_protein, int)
        or final_candidates_per_protein < 1
    ):
        raise V2FacadeError("final_candidates_per_protein must be an integer >= 1")
    if official and final_candidates_per_protein is None:
        final_candidates_per_protein = 8
    if mode in _MULTIROOT_MODES:
        if k is not None or require_k:
            raise V2FacadeError("k/require_k are invalid in multiroot Pareto modes")
        return _materialize_multiroot_selection(
            bundles,
            mode=mode,
            official=bool(official),
            final_candidates_per_protein=final_candidates_per_protein,
            constraint_manifest=constraint_manifest,
            expected_master_seeds=expected_master_seeds,
        )
    if official or final_candidates_per_protein is not None:
        raise V2FacadeError(
            "official/final_candidates_per_protein are valid only in multiroot Pareto modes"
        )
    if mode not in {"elite", "top-k"}:
        raise V2FacadeError("mode must be 'elite' or 'top-k'")
    if constraint_manifest is not None:
        raise V2FacadeError("constraint_manifest is valid only in multiroot Pareto modes")
    if expected_master_seeds is not None:
        raise V2FacadeError(
            "expected_master_seeds is valid only in multiroot Pareto modes"
        )
    if mode == "top-k":
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise V2FacadeError("top-k mode requires integer k >= 1")
    elif k is not None or require_k:
        raise V2FacadeError("k/require_k are valid only in top-k mode")
    if not bundles:
        raise V2FacadeError("at least one V2 bundle is required")

    roots = [_bundle_root(path) for path in bundles]
    if len(set(roots)) != len(roots):
        raise V2FacadeError("the same V2 bundle was supplied more than once")
    # Establish that these are cells of ONE frozen experiment before opening their endpoint
    # tables.  Once frames are concatenated, source-bundle provenance is too late to prevent a
    # mixed campaign from looking like one cohort.
    _validate_common_manifest_identity(roots)
    joined_by_root = [(root, _load_joined_bundle(root)) for root in roots]

    protein_owner: dict[str, Path] = {}
    for root, frame in joined_by_root:
        proteins = set(map(str, frame["protein_id_archive"].dropna().unique()))
        for protein_id in proteins:
            previous = protein_owner.get(protein_id)
            if previous is not None:
                raise V2FacadeError(
                    f"protein {protein_id!r} appears in more than one bundle: {previous}, {root}"
                )
            protein_owner[protein_id] = root
    joined = pd.concat([frame for _, frame in joined_by_root], ignore_index=True)
    if joined.empty or not protein_owner:
        raise V2FacadeError("the named V2 bundles contain no archived endpoints")
    duplicated_ids = joined.loc[
        joined["endpoint_id"].duplicated(keep=False), "endpoint_id"
    ]
    if not duplicated_ids.empty:
        raise V2FacadeError(
            "endpoint_id is duplicated across V2 bundles: "
            f"{sorted(set(map(str, duplicated_ids)))[:3]}"
        )

    candidates = _candidate_rows(joined)

    selected: list[pd.DataFrame] = []
    for protein_id in sorted(protein_owner):
        all_rows = joined[joined["protein_id_archive"].astype(str) == protein_id]
        pool = candidates[candidates["protein_id_archive"].astype(str) == protein_id].copy()
        if mode == "elite":
            elite_mask = [
                _strict_bool(
                    row["is_elite"], field="is_elite", endpoint_id=str(row["endpoint_id"]),
                )
                for _, row in all_rows.iterrows()
            ]
            elite_rows = all_rows.loc[elite_mask]
            if len(elite_rows) != 1:
                raise V2FacadeError(
                    f"protein {protein_id!r} requires exactly one current archive elite; "
                    f"found {len(elite_rows)}"
                )
            elite_row = elite_rows.iloc[0]
            try:
                elite_rank = float(elite_row["elite_rank"])
            except (TypeError, ValueError) as exc:
                raise V2FacadeError(
                    f"protein {protein_id!r} current archive elite has invalid elite_rank"
                ) from exc
            if not math.isfinite(elite_rank) or not elite_rank.is_integer() or elite_rank != 0:
                raise V2FacadeError(
                    f"protein {protein_id!r} current archive elite must carry elite_rank=0"
                )
            _validate_endpoint_row(elite_row)
            pick = elite_rows.copy()
        else:
            if pool.empty:
                raise V2FacadeError(
                    f"protein {protein_id!r} has no definitive-feasible archive endpoint"
                )
            pool = pool.sort_values(
                ["head_global_risk", "endpoint_id"], kind="mergesort",
            )
            pool = pool.drop_duplicates("sequence_equivalence_key_endpoint", keep="first")
            if require_k and len(pool) < int(k):
                raise V2FacadeError(
                    f"top-k requires {k} distinct definitive-feasible endpoints for protein "
                    f"{protein_id!r}, but it has {len(pool)}"
                )
            pick = pool.head(int(k))
        pick = pick.sort_values(["head_global_risk", "endpoint_id"], kind="mergesort").copy()
        pick["design_idx"] = range(len(pick))
        selected.append(pick)

    chosen = pd.concat(selected, ignore_index=True)
    if chosen.empty:
        raise V2FacadeError("selection produced zero facade rows")
    head_domains = chosen[["head_evaluator_digest", "head_window_grid_digest"]].drop_duplicates()
    if chosen[["head_evaluator_digest", "head_window_grid_digest"]].isna().any().any():
        raise V2FacadeError("selected V2 endpoints carry null Head identity")
    if any(
        not str(value)
        for value in chosen["head_evaluator_digest"].tolist()
        + chosen["head_window_grid_digest"].tolist()
    ):
        raise V2FacadeError("selected V2 endpoints carry empty Head identity")
    if len(head_domains) != 1:
        raise V2FacadeError(
            "selected V2 endpoints do not share one frozen Head evaluator/window-grid identity"
        )
    out = pd.DataFrame({
        "protein_id": chosen["protein_id_endpoint"].astype(str),
        "design_idx": chosen["design_idx"].astype(int),
        "sequence": chosen["sequence"].astype(str),
        "entry_source_id": chosen["endpoint_id"].astype(str),
        "endpoint_id": chosen["endpoint_id"].astype(str),
        "endpoint_content_digest": chosen["endpoint_content_digest_endpoint"].astype(str),
        "depth": chosen["depth"].astype(int),
        "head_global_risk": chosen["head_global_risk"].astype(float),
        "head_evaluator_digest": chosen["head_evaluator_digest"].astype(str),
        "head_window_grid_digest": chosen["head_window_grid_digest"].astype(str),
        "sequence_md5": chosen["sequence_md5"].astype(str),
    })
    out = out.sort_values(["protein_id", "design_idx"], kind="mergesort").reset_index(drop=True)
    if out[["protein_id", "design_idx"]].duplicated().any():
        raise V2FacadeError("selection produced duplicate (protein_id, design_idx) facade keys")
    if out["entry_source_id"].duplicated().any():
        raise V2FacadeError("selection produced duplicate endpoint lineage keys")
    return out.loc[:, _OUTPUT_COLUMNS]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", action="append", required=True,
        help=(
            "V2 bundle directory; repeat for disjoint proteins in elite/top-k or for multiple "
            "roots of the same high-risk protein in explicit multiroot modes"
        ),
    )
    parser.add_argument(
        "--mode", required=True,
        choices=("elite", "top-k", *_MULTIROOT_MODES),
    )
    parser.add_argument("--k", type=int, help="maximum distinct endpoints per protein in top-k mode")
    parser.add_argument(
        "--require-k", action="store_true",
        help="fail if any protein has fewer than K distinct definitive-feasible endpoints",
    )
    parser.add_argument(
        "--constraint-manifest",
        help="required by multiroot modes when bundle identities declare hard constraints",
    )
    parser.add_argument(
        "--expected-master-seed", action="append", type=int,
        help=(
            "one authoritative campaign master seed; repeat exactly four times in multiroot "
            "modes to bind and verify the complete (protein, master_seed) grid"
        ),
    )
    parser.add_argument(
        "--official",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "emit the paper-facing Pareto-layer panel; defaults to eight candidates per "
            "protein without changing or deleting the source evidence bundles"
        ),
    )
    parser.add_argument(
        "--final-candidates-per-protein",
        type=int,
        help=(
            "explicit up-to-N final panel size in multiroot modes; works with or without "
            "--official and overrides its default N=8"
        ),
    )
    parser.add_argument("--output", required=True, help="output generated.parquet path")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.official is None:
        args.official = args.mode in _MULTIROOT_MODES
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite to replace it: {output}")
    try:
        facade = materialize_archive_facade(
            args.bundle, mode=args.mode, k=args.k, require_k=bool(args.require_k),
            official=bool(args.official),
            final_candidates_per_protein=args.final_candidates_per_protein,
            constraint_manifest=args.constraint_manifest,
            expected_master_seeds=args.expected_master_seed,
        )
    except V2FacadeError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    facade.to_parquet(output, index=False)
    if args.official or args.final_candidates_per_protein is not None:
        requested = args.final_candidates_per_protein
        if requested is None:
            requested = 8
        counts = facade.groupby("protein_id", sort=True).size()
        manifest = {
            "schema_version": "rf-fusion-v2-official-selection-v1",
            "official": bool(args.official),
            "mode": str(args.mode),
            "requested_candidates_per_protein": int(requested),
            "realized_counts": {str(key): int(value) for key, value in counts.items()},
            "output": str(output),
            "output_sha256": _sha256_file(output),
            "source_bundles_preserved": True,
        }
        manifest_path = output.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        f"wrote {len(facade)} rows across {facade['protein_id'].nunique()} proteins to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
