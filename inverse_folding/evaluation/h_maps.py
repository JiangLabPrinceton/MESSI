"""Phase B h-map artifact loading and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sequence_md5(sequence: str) -> str:
    """Return the lowercase hex md5 digest of a residue sequence.

    Used to bind an h-map row to the exact sequence it was scored on so that
    Phase C1 can fail-fast on any (protein_id, sequence) drift between the
    test parquet and the precomputed h-maps. The sequence is uppercased
    before hashing so that case-only differences are tolerated.
    """
    return hashlib.md5(sequence.upper().encode("utf-8")).hexdigest()


HMAP_COLUMNS = (
    "protein_id",
    "allele",
    "sequence_length",
    "sequence_md5",
    "h_raw",
    "h_processed",
    "global_risk",
    "n_windows",
)

SEQUENCE_MD5_LENGTH = 32

HMAP_META_KEYS = (
    "run_id",
    "allele",
    "head_checkpoint_path",
    "head_checkpoint_metadata",
    "inference_cfg",
    "source_dataset",
    "source_dataset_rowcount",
    "git_commit",
    "timestamp",
    "n_proteins_total",
    "n_proteins_completed",
    "n_proteins_failed",
    "failures",
    "device",
    "wall_clock_seconds",
)

CORPUS_STATS_KEYS = (
    "corpus_h_raw_mean",
    "corpus_h_raw_std",
    "corpus_h_raw_n_residues",
)


class HMapSchemaError(ValueError):
    """Raised when an h-map parquet or metadata sidecar violates the contract."""


def default_meta_path(parquet_path: Path | str) -> Path:
    """Return the sidecar path for ``h_maps_*.parquet``."""
    path = Path(parquet_path)
    return path.with_name(f"{path.stem}.meta.json")


def load_h_maps(
    parquet_path: Path | str,
    meta_path: Path | str | None = None,
    *,
    require_corpus_stats: bool = False,
    allow_partial: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate a Phase B h-map artifact.

    Args:
        parquet_path: h-map parquet path.
        meta_path: optional sidecar JSON path. Defaults to
            ``<parquet_stem>.meta.json``.
        require_corpus_stats: require and validate CATH/B3 corpus statistics.
        allow_partial: allow incomplete checkpoint artifacts where
            ``completed + failed < total``.

    Returns:
        ``(dataframe, metadata)`` with array columns preserved.
    """
    parquet_path = Path(parquet_path)
    if meta_path is None:
        meta_path = default_meta_path(parquet_path)
    meta_path = Path(meta_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"h-map parquet not found: {parquet_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"h-map metadata sidecar not found: {meta_path}")

    df = pd.read_parquet(parquet_path)
    with open(meta_path) as f:
        meta = json.load(f)

    validate_h_map_metadata(meta, allow_partial=allow_partial)
    validate_h_map_dataframe(df, expected_allele=str(meta["allele"]))

    if int(meta["n_proteins_completed"]) != len(df):
        raise HMapSchemaError(
            "metadata n_proteins_completed does not match parquet row count: "
            f"{meta['n_proteins_completed']} != {len(df)}",
        )
    validate_h_map_cross_fields(
        df,
        meta,
        require_corpus_stats=require_corpus_stats,
        allow_partial=allow_partial,
    )

    return df, meta


def validate_h_map_metadata(meta: dict[str, Any], *, allow_partial: bool = False) -> None:
    """Validate required sidecar keys and non-empty checkpoint metadata."""
    missing = [k for k in HMAP_META_KEYS if k not in meta]
    if missing:
        raise HMapSchemaError(f"h-map metadata missing required keys: {missing}")

    checkpoint_meta = meta.get("head_checkpoint_metadata")
    if not isinstance(checkpoint_meta, dict) or not checkpoint_meta:
        raise HMapSchemaError("head_checkpoint_metadata must be a non-empty object")

    failures = meta.get("failures")
    if not isinstance(failures, list):
        raise HMapSchemaError("failures must be a list")
    n_failed = int(meta["n_proteins_failed"])
    if len(failures) != n_failed:
        raise HMapSchemaError(
            "failures length does not match n_proteins_failed: "
            f"{len(failures)} != {n_failed}",
        )

    n_total = int(meta["n_proteins_total"])
    n_completed = int(meta["n_proteins_completed"])
    accounted = n_completed + n_failed
    if allow_partial:
        if accounted > n_total:
            raise HMapSchemaError(
                "partial metadata has completed + failed greater than "
                f"n_proteins_total: {accounted} > {n_total}",
            )
    elif n_total != accounted:
        raise HMapSchemaError(
            "n_proteins_total must equal n_proteins_completed + n_proteins_failed: "
            f"{n_total} != {n_completed} + {n_failed}",
        )

    source_rows = int(meta["source_dataset_rowcount"])
    if source_rows != n_total:
        raise HMapSchemaError(
            "source_dataset_rowcount must match n_proteins_total: "
            f"{source_rows} != {n_total}",
        )


def validate_h_map_dataframe(df: pd.DataFrame, *, expected_allele: str | None = None) -> None:
    """Validate h-map parquet schema and per-row array invariants."""
    missing = [c for c in HMAP_COLUMNS if c not in df.columns]
    if missing:
        raise HMapSchemaError(f"h-map parquet missing required columns: {missing}")
    if df.empty:
        raise HMapSchemaError("h-map parquet is empty")
    if df["protein_id"].duplicated().any():
        dupes = sorted(df.loc[df["protein_id"].duplicated(), "protein_id"].astype(str).unique())
        raise HMapSchemaError(f"h-map parquet contains duplicate protein_id values: {dupes[:5]}")
    if expected_allele is not None:
        row_alleles = set(df["allele"].astype(str).unique())
        if row_alleles != {expected_allele}:
            raise HMapSchemaError(
                f"h-map dataframe allele values {sorted(row_alleles)} do not match "
                f"metadata allele {expected_allele}",
            )

    for idx, row in df.iterrows():
        protein_id = str(row["protein_id"])
        sequence_length = int(row["sequence_length"])
        if sequence_length <= 0:
            raise HMapSchemaError(f"{protein_id}: sequence_length must be positive")

        sequence_md5 = row["sequence_md5"]
        if not isinstance(sequence_md5, str) or len(sequence_md5) != SEQUENCE_MD5_LENGTH:
            raise HMapSchemaError(
                f"{protein_id}: sequence_md5 must be a 32-character hex digest, "
                f"got {sequence_md5!r}"
            )
        if not all(c in "0123456789abcdef" for c in sequence_md5):
            raise HMapSchemaError(
                f"{protein_id}: sequence_md5 must contain only lowercase hex digits, "
                f"got {sequence_md5!r}"
            )

        h_raw = _as_float_array(row["h_raw"], f"{protein_id}: h_raw")
        h_processed = _as_float_array(row["h_processed"], f"{protein_id}: h_processed")
        if len(h_raw) != sequence_length:
            raise HMapSchemaError(
                f"{protein_id}: h_raw length {len(h_raw)} != sequence_length {sequence_length}",
            )
        if len(h_processed) != sequence_length:
            raise HMapSchemaError(
                f"{protein_id}: h_processed length {len(h_processed)} != "
                f"sequence_length {sequence_length}",
            )

        _validate_median_centering(protein_id, h_raw, h_processed)


def validate_h_map_cross_fields(
    df: pd.DataFrame,
    meta: dict[str, Any],
    *,
    require_corpus_stats: bool = False,
    allow_partial: bool = False,
) -> None:
    """Validate metadata fields that require the dataframe contents."""
    if require_corpus_stats:
        missing = [k for k in CORPUS_STATS_KEYS if k not in meta]
        if missing:
            raise HMapSchemaError(f"metadata missing required corpus_h_raw stats: {missing}")

        pooled = _pooled_h_raw(df)
        expected_n = int(sum(int(x) for x in df["sequence_length"]))
        if int(meta["corpus_h_raw_n_residues"]) != expected_n:
            raise HMapSchemaError(
                "corpus_h_raw_n_residues does not match sum(sequence_length): "
                f"{meta['corpus_h_raw_n_residues']} != {expected_n}",
            )
        if pooled.size != expected_n:
            raise HMapSchemaError(
                "pooled h_raw residue count does not match sum(sequence_length): "
                f"{pooled.size} != {expected_n}",
            )
        if pooled.size:
            mean = float(pooled.mean())
            std = float(pooled.std())
            if not np.isclose(float(meta["corpus_h_raw_mean"]), mean, rtol=1e-6, atol=1e-6):
                raise HMapSchemaError("corpus_h_raw_mean does not match parquet h_raw values")
            if not np.isclose(float(meta["corpus_h_raw_std"]), std, rtol=1e-6, atol=1e-6):
                raise HMapSchemaError("corpus_h_raw_std does not match parquet h_raw values")


def _as_float_array(value: Any, label: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value.astype(np.float32, copy=False)
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=np.float32)
    else:
        raise HMapSchemaError(f"{label} must be a list-like array")
    if arr.ndim != 1:
        raise HMapSchemaError(f"{label} must be one-dimensional")
    return arr


def _pooled_h_raw(df: pd.DataFrame) -> np.ndarray:
    arrays = [
        _as_float_array(row["h_raw"], f"{row['protein_id']}: h_raw")
        for _, row in df.iterrows()
    ]
    if not arrays:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(arrays).astype(np.float32, copy=False)


def _validate_median_centering(
    protein_id: str,
    h_raw: np.ndarray,
    h_processed: np.ndarray,
) -> None:
    finite = np.isfinite(h_raw)
    if not finite.any():
        return

    center = _torch_style_median(h_raw[finite])
    expected = h_raw.copy()
    expected[finite] = expected[finite] - center

    processed_finite = np.isfinite(h_processed)
    comparable = finite & processed_finite
    if not comparable.any():
        raise HMapSchemaError(f"{protein_id}: h_processed has no finite centered values")

    if not np.allclose(h_processed[comparable], expected[comparable], rtol=1e-4, atol=1e-5):
        raise HMapSchemaError(
            f"{protein_id}: h_processed is not median-centered h_raw within tolerance",
        )


def _torch_style_median(values: np.ndarray) -> float:
    """Match ``torch.median`` for even-length 1D tensors.

    PyTorch returns the lower middle value for even-length tensors, while
    ``numpy.median`` averages the two middle values.
    """
    if values.size == 0:
        raise HMapSchemaError("cannot compute median of an empty array")
    sorted_values = np.sort(values.astype(np.float32, copy=False))
    return float(sorted_values[(len(sorted_values) - 1) // 2])
