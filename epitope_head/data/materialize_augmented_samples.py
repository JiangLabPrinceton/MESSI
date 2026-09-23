"""Stage J — Task J6: Materialize train-only augmented ProteinSamples.

Converts mutation registry rows into augmented ProteinSample rows that are
schema-compatible with the existing training pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def materialize_augmented_samples(
    registry_df: pd.DataFrame,
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create augmented ProteinSample rows from mutation registry.

    Each registry row produces one augmented sample with:
    - Unique protein_id: AUG::<source>::<pos><wt>><mut>
    - Mutated protein_seq (1 AA substitution)
    - positives_json = source positives - affected_spans

    Args:
        registry_df: mutation registry with columns from build_registry_rows()
        source_df: original protein_samples parquet

    Returns:
        DataFrame with same schema as source_df, containing augmented rows only.
    """
    if len(registry_df) == 0:
        return pd.DataFrame(columns=source_df.columns)

    # Build source lookup
    source_lookup = {}
    for _, row in source_df.iterrows():
        source_lookup[row["protein_id"]] = row

    aug_rows = []
    for _, reg in registry_df.iterrows():
        src_pid = reg["source_protein_id"]
        if src_pid not in source_lookup:
            logger.warning(f"Source protein {src_pid} not found in parquet, skipping")
            continue

        src = source_lookup[src_pid]
        src_seq = src["protein_seq"]
        pos = reg["mut_pos_0b"]
        mut_aa = reg["mut_aa"]

        # Build mutant sequence
        mut_seq = src_seq[:pos] + mut_aa + src_seq[pos + 1:]

        # Remove affected spans from positives
        src_positives = json.loads(src["positives_json"])
        affected = set()
        for a in json.loads(reg["affected_spans_json"]):
            affected.add((a["start_0b"], a["end_0b"], a["pep_len"]))

        remaining_positives = [
            p for p in src_positives
            if (p["start_0b"], p["end_0b"], p["pep_len"]) not in affected
        ]

        # Build provenance metadata
        src_meta = json.loads(src["metadata_json"]) if pd.notna(src.get("metadata_json")) else {}
        aug_meta = {
            "augmentation": {
                "source_protein_id": src_pid,
                "mut_pos_0b": pos,
                "wt_aa": reg["wt_aa"],
                "mut_aa": mut_aa,
                "n_affected_spans": reg["n_affected_spans"],
                "max_delta_rank": reg["max_delta_rank"],
            },
            **{k: v for k, v in src_meta.items() if k != "augmentation"},
        }

        aug_rows.append({
            "protein_id": reg["mutant_protein_id"],
            "allele": src["allele"],
            "protein_seq": mut_seq,
            "positives_json": json.dumps(remaining_positives),
            "sequence_length": len(mut_seq),
            "input_span_count": src.get("input_span_count", 0),
            "positive_count": len(remaining_positives),
            "duplicate_span_count": 0,
            "metadata_json": json.dumps(aug_meta),
        })

    return pd.DataFrame(aug_rows)


def validate_augmented_samples(
    aug_df: pd.DataFrame,
    source_df: pd.DataFrame,
    val_ids: set[str],
    test_ids: set[str],
) -> list[str]:
    """Validate augmented samples for split safety and schema compliance.

    Returns list of error messages (empty = all good).
    """
    errors = []

    # Schema check
    required_cols = {"protein_id", "allele", "protein_seq", "positives_json", "sequence_length"}
    missing = required_cols - set(aug_df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    # Split safety: no augmented IDs in val/test
    aug_ids = set(aug_df["protein_id"])
    val_leak = aug_ids & val_ids
    if val_leak:
        errors.append(f"Augmented IDs leaked into val: {val_leak}")
    test_leak = aug_ids & test_ids
    if test_leak:
        errors.append(f"Augmented IDs leaked into test: {test_leak}")

    # No reuse of source protein_ids
    source_ids = set(source_df["protein_id"])
    reused = aug_ids & source_ids
    if reused:
        errors.append(f"Augmented rows reuse source protein_ids: {reused}")

    # Unique IDs
    if aug_df["protein_id"].duplicated().any():
        n_dup = aug_df["protein_id"].duplicated().sum()
        errors.append(f"{n_dup} duplicate augmented protein_ids")

    # Provenance check: all should have AUG:: prefix
    non_aug = aug_df[~aug_df["protein_id"].str.startswith("AUG::")]
    if len(non_aug) > 0:
        errors.append(f"{len(non_aug)} rows missing AUG:: prefix")

    return errors
