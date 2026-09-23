"""Stage A schema validation for SpanRecord DataFrames."""

from __future__ import annotations

import pandas as pd

SPAN_RECORD_REQUIRED = [
    "protein_id", "allele",
    "start_1b", "end_1b", "start_0b", "end_0b", "pep_len",
    "peptide_seq", "source", "dataset_source",
]

SPAN_RECORD_TYPES = {
    "protein_id": "object",
    "allele": "object",
    "start_1b": "int64",
    "end_1b": "int64",
    "start_0b": "int64",
    "end_0b": "int64",
    "pep_len": "int64",
    "peptide_seq": "object",
    "source": "object",
    "dataset_source": "object",
}


def validate_span_records(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce SpanRecord schema. Raises on violations."""
    missing = [c for c in SPAN_RECORD_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"SpanRecord missing columns: {missing}")

    # Select only canonical columns (drop internal bookkeeping)
    df = df[SPAN_RECORD_REQUIRED].copy()

    # Type coercion
    for col, dtype in SPAN_RECORD_TYPES.items():
        df[col] = df[col].astype(dtype)

    # Non-null check
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if len(null_cols) > 0:
        raise ValueError(f"SpanRecord has nulls: {null_cols.to_dict()}")

    # Invariant: pep_len == end_0b - start_0b
    bad = df["pep_len"] != (df["end_0b"] - df["start_0b"])
    if bad.any():
        raise ValueError(f"pep_len invariant violated in {bad.sum()} rows")

    return df
