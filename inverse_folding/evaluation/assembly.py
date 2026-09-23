"""Test set assembly, artifact generation, and validation.

Merges Tier 1/2/3 entries into a single test_proteins.parquet,
validates every row against the frozen schema, and provides
post-hoc validation for the assembled set.
"""

from typing import Dict, List

import pandas as pd

from inverse_folding.evaluation.schema import (
    EvalSchemaError,
    TEST_PROTEIN_COLUMNS,
    validate_test_protein_entry,
)


class AssemblyError(Exception):
    """Raised when test set assembly encounters a fatal error."""


def assemble_test_set(entries: List[dict]) -> pd.DataFrame:
    """Merge tier entries into a single DataFrame.

    Validates each entry against the frozen schema and rejects
    duplicate protein_ids.

    Args:
        entries: list of per-protein dicts conforming to TEST_PROTEIN_COLUMNS.

    Returns:
        Assembled DataFrame.

    Raises:
        AssemblyError: if duplicates found or schema validation fails.
    """
    if not entries:
        raise AssemblyError("No entries to assemble.")

    # Check for duplicates
    pids = [e.get("protein_id", "") for e in entries]
    seen = set()
    duplicates = set()
    for pid in pids:
        if pid in seen:
            duplicates.add(pid)
        seen.add(pid)

    if duplicates:
        raise AssemblyError(
            f"Duplicate protein_id(s): {sorted(duplicates)}"
        )

    # Validate each entry
    for entry in entries:
        try:
            validate_test_protein_entry(entry)
        except EvalSchemaError as e:
            pid = entry.get("protein_id", "<unknown>")
            raise AssemblyError(
                f"Schema validation failed for {pid}: {e}"
            ) from e

    df = pd.DataFrame(entries)
    return df


def validate_test_set(parquet_path: str) -> List[str]:
    """Post-hoc validation of an assembled test set parquet.

    Checks:
      1. No duplicate protein_ids.
      2. No CATH overlap flags remaining (should have been excluded).
      3. All required columns present.

    Returns:
        List of error strings. Empty means valid.
    """
    df = pd.read_parquet(parquet_path)
    errors: List[str] = []

    # Required columns
    missing_cols = TEST_PROTEIN_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {sorted(missing_cols)}")

    # Duplicate protein_ids
    dup_mask = df["protein_id"].duplicated(keep=False)
    if dup_mask.any():
        dups = df.loc[dup_mask, "protein_id"].unique().tolist()
        errors.append(f"Duplicate protein_id(s): {dups}")

    # CATH overlap flags
    if "cath_overlap_flag" in df.columns:
        n_overlap = int(df["cath_overlap_flag"].sum())
        if n_overlap > 0:
            overlap_pids = df.loc[df["cath_overlap_flag"], "protein_id"].tolist()
            errors.append(
                f"{n_overlap} protein(s) with cath_overlap_flag=True "
                f"should have been excluded: {overlap_pids}"
            )

    return errors


def compute_assembly_summary(df: pd.DataFrame) -> dict:
    """Compute summary statistics for an assembled test set."""
    summary = {
        "total_proteins": len(df),
        "tier_counts": df["tier"].value_counts().sort_index().to_dict(),
        "mean_sequence_length": float(df["sequence_length"].mean()),
        "mean_netmhciipan_n_strong": float(df["netmhciipan_n_strong"].mean()),
        "mean_head_global_risk": float(df["head_global_risk"].mean()),
    }
    if "cath_topology" in df.columns:
        n_topologies = df["cath_topology"].dropna().nunique()
        summary["n_unique_topologies"] = n_topologies
    return summary
