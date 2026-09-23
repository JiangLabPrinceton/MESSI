"""Run registry, protocol signature, and cross-run comparability (Module G).

Implements PLAN.md Tasks G0-G6:
  - G0: Frozen registry row schema + comparability contract
  - G1: Deterministic run_id + protocol_signature
  - G2: Registry writer with strict fail-fast validation
  - G3: Trainer/inference integration helpers (build_registry_row)
  - G4: Comparability guard (is_comparable)
  - G5: Backfill from existing run directories
  - G6: Smoke-ready end-to-end flow
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# ── G0: Frozen Registry Schema ──────────────────────────────────────────────

REQUIRED_REGISTRY_KEYS = frozenset({
    "run_id",
    "timestamp",
    "manifest_version",
    "diff_ids_applied",
    "resolved_config_path",
    "best_checkpoint_path",
    "primary_metrics",
    "config_hash",
    "protocol_signature",
})

REQUIRED_METRICS_KEYS = frozenset({"logit_gap"})

# Keys that define comparability (G4).
COMPARABILITY_KEYS = frozenset({"manifest_version", "protocol_signature"})


def validate_registry_row(row: dict) -> list[str]:
    """Validate a registry row against the frozen schema (G0).

    Returns list of error strings (empty = valid).
    """
    errors = []

    missing = REQUIRED_REGISTRY_KEYS - set(row.keys())
    if missing:
        errors.append(f"Missing required keys: {sorted(missing)}")
        return errors

    # Type checks
    if not isinstance(row["run_id"], str) or not row["run_id"]:
        errors.append("run_id must be a non-empty string")
    if not isinstance(row["timestamp"], (int, float)):
        errors.append("timestamp must be a number")
    if not isinstance(row["manifest_version"], str):
        errors.append("manifest_version must be a string")
    if not isinstance(row["diff_ids_applied"], list):
        errors.append("diff_ids_applied must be a list")
    if not isinstance(row["config_hash"], str):
        errors.append("config_hash must be a string")
    if not isinstance(row["protocol_signature"], str) or not row["protocol_signature"]:
        errors.append("protocol_signature must be a non-empty string")

    # primary_metrics
    pm = row["primary_metrics"]
    if not isinstance(pm, dict):
        errors.append("primary_metrics must be a dict")
    else:
        missing_m = REQUIRED_METRICS_KEYS - set(pm.keys())
        if missing_m:
            errors.append(f"primary_metrics missing keys: {sorted(missing_m)}")

    # Path fields: validate they are strings (existence checked in append)
    for path_key in ("resolved_config_path", "best_checkpoint_path"):
        if not isinstance(row[path_key], str):
            errors.append(f"{path_key} must be a string")

    return errors


# ── G1: Run Identity + Protocol Signature ────────────────────────────────────

def generate_run_id(seed_str: str | None = None) -> str:
    """Generate a sortable, collision-safe run_id.

    Format: `run_<YYYYMMDD>_<HHMMSS>_<hash6>`.
    If seed_str is provided, the hash suffix is deterministic.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    if seed_str is not None:
        h = hashlib.sha256(seed_str.encode()).hexdigest()[:6]
    else:
        h = hashlib.sha256(f"{time.time()}".encode()).hexdigest()[:6]
    return f"run_{ts}_{h}"


# Fields that contribute to protocol_signature (order-stable).
_PROTOCOL_SIG_FIELDS = [
    "min_k",
    "max_k",
    "hotspot_center_method",
    "hotspot_clamp",
    "chunking.context_len",
    "chunking.stride",
    "chunking.margin",
    "chunking.stitch_mode",
    "loss.tau",
    "neg_ratio",
]


def compute_protocol_signature(cfg: dict) -> str:
    """Compute deterministic protocol signature from comparability-critical fields (G1).

    Extracts a fixed set of protocol-critical keys from a resolved config,
    canonically serializes them, and returns a short hash.

    Args:
        cfg: resolved config dict (may be flat or nested).

    Returns:
        12-char hex digest.
    """
    extracted = {}
    for key in _PROTOCOL_SIG_FIELDS:
        parts = key.split(".")
        val = cfg
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p)
            else:
                val = None
                break
        extracted[key] = val

    canonical = json.dumps(extracted, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ── G2: Registry Writer ─────────────────────────────────────────────────────

def validate_artifact_paths(row: dict, check_existence: bool = True) -> list[str]:
    """Validate that referenced artifact paths exist on disk."""
    errors = []
    for key in ("resolved_config_path", "best_checkpoint_path"):
        path = Path(row[key])
        if check_existence and not path.exists():
            errors.append(f"{key} not found: {row[key]}")
    return errors


def validate_checkpoint_consistency(row: dict) -> list[str]:
    """Validate checkpoint metadata matches registry row fields (G2)."""
    errors = []
    ckpt_path = Path(row["best_checkpoint_path"])
    if not ckpt_path.exists():
        return []  # skip if path doesn't exist (already caught by path check)

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        errors.append(f"Failed to load checkpoint: {exc}")
        return errors

    meta = ckpt.get("metadata", {})
    if meta.get("config_hash") != row["config_hash"]:
        errors.append(
            f"config_hash mismatch: checkpoint={meta.get('config_hash')} "
            f"vs row={row['config_hash']}"
        )
    if meta.get("manifest_version") != row["manifest_version"]:
        errors.append(
            f"manifest_version mismatch: checkpoint={meta.get('manifest_version')} "
            f"vs row={row['manifest_version']}"
        )
    return errors


def append_registry_row(
    row: dict,
    registry_path: Path | str,
    check_paths: bool = True,
    check_checkpoint: bool = True,
) -> Path:
    """Validate and append a single registry row to JSONL file (G2).

    Args:
        row: registry row dict.
        registry_path: path to run_registry.jsonl.
        check_paths: whether to check artifact path existence.
        check_checkpoint: whether to validate checkpoint metadata consistency.

    Returns:
        Path to registry file.

    Raises:
        ValueError: on any validation failure.
    """
    registry_path = Path(registry_path)

    # Schema validation
    schema_errors = validate_registry_row(row)
    if schema_errors:
        raise ValueError(f"Registry row schema errors: {schema_errors}")

    # Path existence
    if check_paths:
        path_errors = validate_artifact_paths(row, check_existence=True)
        if path_errors:
            raise ValueError(f"Artifact path errors: {path_errors}")

    # Checkpoint consistency
    if check_checkpoint and check_paths:
        ckpt_errors = validate_checkpoint_consistency(row)
        if ckpt_errors:
            raise ValueError(f"Checkpoint consistency errors: {ckpt_errors}")

    # Dedup: reject if run_id already present
    if registry_path.exists():
        existing_ids = {r.get("run_id") for r in load_registry(registry_path)}
        if row["run_id"] in existing_ids:
            raise ValueError(
                f"Duplicate run_id '{row['run_id']}' already in {registry_path}"
            )

    # Atomic append
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=False, default=str) + "\n"
    with open(registry_path, "a") as f:
        f.write(line)

    return registry_path


def load_registry(registry_path: Path | str) -> list[dict]:
    """Load all registry rows from JSONL file."""
    registry_path = Path(registry_path)
    if not registry_path.exists():
        return []
    rows = []
    with open(registry_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num}: {exc}") from exc
    return rows


# ── G3: Integration Helpers ──────────────────────────────────────────────────

def build_registry_row(
    run_id: str,
    run_dir: Path | str,
    best_checkpoint_path: Path | str,
    config_hash: str,
    manifest_version: str,
    diff_ids_applied: list[str],
    protocol_signature: str,
    primary_metrics: dict,
    prediction_digest: str | None = None,
) -> dict:
    """Build a complete registry row from trainer/inference outputs (G3).

    Args:
        run_id: unique run identifier.
        run_dir: path to run directory (contains resolved_config.yaml).
        best_checkpoint_path: path to best.pt.
        config_hash: deterministic config hash.
        manifest_version: data manifest version string.
        diff_ids_applied: list of codemap diff IDs applied.
        protocol_signature: protocol signature hash.
        primary_metrics: dict with at least logit_gap.
        prediction_digest: optional F-stage prediction summary digest.

    Returns:
        Schema-valid registry row dict.
    """
    run_dir = Path(run_dir)
    row = {
        "run_id": run_id,
        "timestamp": time.time(),
        "manifest_version": manifest_version,
        "diff_ids_applied": list(diff_ids_applied),
        "resolved_config_path": str(run_dir / "resolved_config.yaml"),
        "best_checkpoint_path": str(best_checkpoint_path),
        "primary_metrics": dict(primary_metrics),
        "config_hash": config_hash,
        "protocol_signature": protocol_signature,
    }

    if prediction_digest is not None:
        row["prediction_digest"] = prediction_digest

    return row


def build_registry_row_from_summary(
    run_summary: dict,
    run_dir: Path | str,
    manifest_version: str | None = None,
    diff_ids_applied: list[str] | None = None,
    protocol_signature: str | None = None,
    prediction_digest: str | None = None,
) -> dict:
    """Build registry row from an existing run_summary.json (G3/G5 helper).

    Extracts run_id, config_hash, metrics from the summary.
    Resolution order for each field: explicit arg > summary value > fallback.
    """
    run_dir = Path(run_dir)

    run_id = run_summary.get("run_id")
    if run_id is None:
        run_id = run_dir.name

    config_hash = run_summary.get("config_hash", "")

    # Resolve manifest_version: explicit arg > summary > "v1.1"
    if manifest_version is None:
        manifest_version = run_summary.get("manifest_version", "v1.1")

    # Resolve protocol_signature: explicit arg > summary > "" (empty will fail G0 validation)
    if protocol_signature is None:
        protocol_signature = run_summary.get("protocol_signature", "")
    best_monitor_value = run_summary.get("best_monitor_value")
    monitor_metric = run_summary.get("monitor_metric", "logit_gap")

    primary_metrics = {monitor_metric: best_monitor_value}
    if "logit_gap" not in primary_metrics:
        primary_metrics["logit_gap"] = best_monitor_value

    best_ckpt = run_dir / "best.pt"

    return build_registry_row(
        run_id=run_id,
        run_dir=run_dir,
        best_checkpoint_path=best_ckpt,
        config_hash=config_hash,
        manifest_version=manifest_version,
        diff_ids_applied=list(diff_ids_applied or []),
        protocol_signature=protocol_signature,
        primary_metrics=primary_metrics,
        prediction_digest=prediction_digest,
    )


# ── G4: Comparability Guard ─────────────────────────────────────────────────

def is_comparable(run_a: dict, run_b: dict) -> dict:
    """Check whether two registry rows represent comparable runs (G4).

    Returns:
        dict with:
          - comparable: bool
          - reason: "comparable" | "manifest_mismatch" | "protocol_mismatch"
          - details: explanatory string
    """
    for key in COMPARABILITY_KEYS:
        if key not in run_a:
            raise ValueError(f"run_a missing comparability key: {key}")
        if key not in run_b:
            raise ValueError(f"run_b missing comparability key: {key}")

    # Empty comparability fields make runs non-comparable (G4 strict guard)
    for key in COMPARABILITY_KEYS:
        empty_sides = []
        if not run_a[key]:
            empty_sides.append("run_a")
        if not run_b[key]:
            empty_sides.append("run_b")
        if empty_sides:
            return {
                "comparable": False,
                "reason": "unknown_protocol",
                "details": (
                    f"{key} is empty in {' and '.join(empty_sides)}; "
                    f"cannot determine comparability"
                ),
            }

    if run_a["manifest_version"] != run_b["manifest_version"]:
        return {
            "comparable": False,
            "reason": "manifest_mismatch",
            "details": (
                f"manifest_version: {run_a['manifest_version']} "
                f"vs {run_b['manifest_version']}"
            ),
        }

    if run_a["protocol_signature"] != run_b["protocol_signature"]:
        return {
            "comparable": False,
            "reason": "protocol_mismatch",
            "details": (
                f"protocol_signature: {run_a['protocol_signature']} "
                f"vs {run_b['protocol_signature']}"
            ),
        }

    return {
        "comparable": True,
        "reason": "comparable",
        "details": "manifest_version and protocol_signature match",
    }


def write_comparability_report(
    run_a: dict,
    run_b: dict,
    output_dir: Path | str,
) -> Path:
    """Write pairwise comparability report artifact (G4)."""
    result = is_comparable(run_a, run_b)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    id_a = run_a.get("run_id", "unknown_a")
    id_b = run_b.get("run_id", "unknown_b")
    report = {
        "run_a": id_a,
        "run_b": id_b,
        **result,
        "timestamp": time.time(),
    }

    filename = f"{id_a}__{id_b}.json"
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    return path


# ── G5: Backfill ─────────────────────────────────────────────────────────────

def backfill_registry(
    metrics_dir: Path | str,
    registry_path: Path | str,
    manifest_version: str | None = None,
    protocol_signature: str | None = None,
) -> dict:
    """Scan existing run directories and backfill valid rows into registry (G5).

    Resolution order for each field: explicit arg > run_summary value > fallback.
    Runs whose protocol_signature cannot be recovered (not in summary and not
    provided by caller) will fail G0 schema validation and be skipped with an
    explicit reason — matching the G5 "unverifiable = explicitly marked" contract.

    Args:
        metrics_dir: directory containing run_* subdirectories.
        registry_path: path to run_registry.jsonl.
        manifest_version: override manifest version (None = read from summary).
        protocol_signature: override protocol signature (None = read from summary).

    Returns:
        dict with {appended: list[str], skipped: list[dict{run_dir, reason}]}.
    """
    metrics_dir = Path(metrics_dir)
    registry_path = Path(registry_path)
    appended = []
    skipped = []

    # Load existing registry to avoid duplicates
    existing_ids = set()
    for row in load_registry(registry_path):
        existing_ids.add(row.get("run_id", ""))

    if not metrics_dir.exists():
        return {"appended": appended, "skipped": skipped}

    for run_dir in sorted(metrics_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue

        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            skipped.append({"run_dir": str(run_dir), "reason": "missing run_summary.json"})
            continue

        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception as exc:
            skipped.append({"run_dir": str(run_dir), "reason": f"invalid run_summary.json: {exc}"})
            continue

        row = build_registry_row_from_summary(
            summary, run_dir,
            manifest_version=manifest_version,
            protocol_signature=protocol_signature,
        )

        if row["run_id"] in existing_ids:
            skipped.append({"run_dir": str(run_dir), "reason": "already in registry"})
            continue

        # Validate schema
        schema_errors = validate_registry_row(row)
        if schema_errors:
            skipped.append({"run_dir": str(run_dir), "reason": f"schema errors: {schema_errors}"})
            continue

        # Validate paths (soft: only check what exists)
        path_errors = validate_artifact_paths(row, check_existence=True)
        if path_errors:
            skipped.append({"run_dir": str(run_dir), "reason": f"path errors: {path_errors}"})
            continue

        # Append (skip checkpoint consistency for backfill — may be from older format)
        try:
            append_registry_row(
                row, registry_path,
                check_paths=True, check_checkpoint=False,
            )
            appended.append(row["run_id"])
            existing_ids.add(row["run_id"])
        except ValueError as exc:
            skipped.append({"run_dir": str(run_dir), "reason": str(exc)})

    return {"appended": appended, "skipped": skipped}
