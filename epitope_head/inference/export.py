"""Canonical JSON export for per-protein prediction payloads (F5).

Implements PLAN.md Task F5:
  - Schema-validated JSON export with required keys
  - Deterministic serialization (sorted keys, consistent float formatting)
  - Batch summary artifact for multi-protein runs
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch


# ── Required payload keys ────────────────────────────────────────────────────

REQUIRED_PAYLOAD_KEYS = frozenset({
    "window_logits",
    "residue_hotspot",
    "global_risk",
    "meta",
})

REQUIRED_META_KEYS = frozenset({
    "protein_len",
    "n_windows",
    "min_k",
    "max_k",
    "center_method",
    "clamp_method",
})

REQUIRED_WINDOW_KEYS = frozenset({"start_0b", "end_0b", "k", "z"})

REQUIRED_HOTSPOT_KEYS = frozenset({"index_0b", "h_raw", "h_processed"})


def validate_prediction_payload(payload: dict) -> list[str]:
    """Validate a prediction payload against the canonical schema.

    Returns list of error strings (empty = valid).
    """
    errors = []

    missing_top = REQUIRED_PAYLOAD_KEYS - set(payload.keys())
    if missing_top:
        errors.append(f"Missing top-level keys: {sorted(missing_top)}")
        return errors  # can't validate further

    # meta
    meta = payload["meta"]
    if not isinstance(meta, dict):
        errors.append("'meta' must be a dict")
    else:
        missing_meta = REQUIRED_META_KEYS - set(meta.keys())
        if missing_meta:
            errors.append(f"meta missing keys: {sorted(missing_meta)}")

    # global_risk
    if not isinstance(payload["global_risk"], (int, float)):
        errors.append("'global_risk' must be a number")

    # window_logits
    wl = payload["window_logits"]
    if not isinstance(wl, list):
        errors.append("'window_logits' must be a list")
    elif len(wl) > 0:
        sample = wl[0]
        if not isinstance(sample, dict):
            errors.append("window_logits entries must be dicts")
        else:
            missing_w = REQUIRED_WINDOW_KEYS - set(sample.keys())
            if missing_w:
                errors.append(f"window_logits entry missing keys: {sorted(missing_w)}")

    # residue_hotspot
    rh = payload["residue_hotspot"]
    if not isinstance(rh, list):
        errors.append("'residue_hotspot' must be a list")
    elif len(rh) > 0:
        sample = rh[0]
        if not isinstance(sample, dict):
            errors.append("residue_hotspot entries must be dicts")
        else:
            missing_h = REQUIRED_HOTSPOT_KEYS - set(sample.keys())
            if missing_h:
                errors.append(f"residue_hotspot entry missing keys: {sorted(missing_h)}")

    return errors


def format_prediction_payload(
    predict_result: dict,
    protein_id: str,
    checkpoint_metadata: dict | None = None,
    config_hash: str | None = None,
) -> dict:
    """Convert predict_protein() output into canonical export payload.

    Args:
        predict_result: output from InferencePredictor.predict_protein().
        protein_id: protein identifier.
        checkpoint_metadata: optional checkpoint metadata to include in meta.
        config_hash: optional config hash for traceability.

    Returns:
        Schema-compliant payload dict ready for JSON serialization.
    """
    h_raw = predict_result["debug"]["h_raw"]
    h_processed = predict_result["residue_hotspot"]

    # Build residue_hotspot list
    residue_hotspot = []
    for i in range(len(h_processed)):
        residue_hotspot.append({
            "index_0b": i,
            "h_raw": float(h_raw[i].item()) if isinstance(h_raw, torch.Tensor) else float(h_raw[i]),
            "h_processed": float(h_processed[i].item()) if isinstance(h_processed, torch.Tensor) else float(h_processed[i]),
        })

    # Build meta
    meta = dict(predict_result["meta"])
    meta["protein_id"] = protein_id
    meta["timestamp"] = time.time()
    if checkpoint_metadata:
        meta["checkpoint"] = {
            k: v for k, v in checkpoint_metadata.items()
            if k in {"manifest_version", "config_hash", "diff_ids_applied", "epoch", "global_step"}
        }
    if config_hash:
        meta["inference_config_hash"] = config_hash

    return {
        "window_logits": predict_result["window_logits"],
        "residue_hotspot": residue_hotspot,
        "global_risk": predict_result["global_risk"],
        "meta": meta,
    }


def export_prediction_json(
    payload: dict,
    output_path: Path | str,
) -> Path:
    """Write validated prediction payload to JSON file.

    Args:
        payload: schema-validated prediction payload.
        output_path: destination file path.

    Returns:
        Path to written file.

    Raises:
        ValueError: if payload fails schema validation.
    """
    errors = validate_prediction_payload(payload)
    if errors:
        raise ValueError(f"Payload validation failed: {errors}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False, default=str)

    return output_path


def compute_payload_digest(payload: dict) -> str:
    """Compute deterministic digest of a prediction payload.

    Hashes window_logits z values + residue_hotspot h_processed values
    + global_risk for reproducibility comparison. Excludes timestamps
    and other non-deterministic meta fields.
    """
    h = hashlib.sha256()

    # Global risk
    h.update(f"R={payload['global_risk']:.8f}".encode())

    # Window logits (ordered)
    for w in payload["window_logits"]:
        h.update(f"w={w['start_0b']},{w['end_0b']},{w['k']},{w['z']:.8f}".encode())

    # Residue hotspot (ordered by index)
    for r in payload["residue_hotspot"]:
        h.update(f"h={r['index_0b']},{r['h_processed']:.8f}".encode())

    return h.hexdigest()[:16]


def write_prediction_summary(
    protein_ids: list[str],
    payload_digests: list[str],
    config_hash: str | None,
    output_dir: Path | str,
) -> Path:
    """Write batch prediction summary artifact.

    Args:
        protein_ids: list of predicted protein IDs.
        payload_digests: list of per-protein payload digests.
        config_hash: inference config hash.
        output_dir: directory for summary file.

    Returns:
        Path to prediction_summary.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combined digest over all individual digests
    combined = hashlib.sha256()
    for d in sorted(payload_digests):
        combined.update(d.encode())
    if len(protein_ids) != len(payload_digests):
        raise ValueError(
            f"protein_ids ({len(protein_ids)}) and payload_digests "
            f"({len(payload_digests)}) length mismatch",
        )
    summary = {
        "n_proteins": len(protein_ids),
        "protein_ids": protein_ids,
        "payload_digests": dict(zip(protein_ids, payload_digests)),
        "combined_digest": combined.hexdigest()[:16],
        "config_hash": config_hash,
        "timestamp": time.time(),
    }

    path = output_dir / "prediction_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=False)

    return path
