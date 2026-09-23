"""Stage A: Parse mhc_if_v2.tsv into normalized SpanRecord DataFrames.

Covers Tasks A1-A4:
  A1 — raw row reader + structural JSON parsing
  A2 — explode peptide_position_info to one record per protein
  A3 — coordinate normalization (1b inclusive → 0b half-open) + length filter
  A4 — allele filter (strict_sa / balanced_sa profiles)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RejectionLedger:
    """Accumulates rejection counts by reason."""
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, reason: str, n: int = 1):
        self.counts[reason] = self.counts.get(reason, 0) + n

    def summary(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))


def _coerce_integral_coordinate(value) -> int:
    """Parse coordinate as integer without silent truncation."""
    if isinstance(value, bool):
        raise TypeError("bool is not a valid coordinate")

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise TypeError("non-finite coordinate")
        if float(value).is_integer():
            return int(value)
        raise ValueError("non_integral")

    if isinstance(value, str):
        s = value.strip()
        if s == "":
            raise TypeError("empty coordinate string")
        try:
            v = float(s)
        except ValueError as exc:
            raise TypeError("non-numeric coordinate string") from exc
        if not np.isfinite(v):
            raise TypeError("non-finite coordinate string")
        if v.is_integer():
            return int(v)
        raise ValueError("non_integral")

    raise TypeError("unsupported coordinate type")


# ── A1: Row reader + structural parsing ──────────────────────────────────────

def _parse_json_column(series: pd.Series, col_name: str, ledger: RejectionLedger) -> pd.Series:
    """Parse a JSON-encoded column. Returns parsed objects; bad rows become None."""
    def _safe_parse(val):
        if pd.isna(val):
            return None
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return None

    parsed = series.map(_safe_parse)
    n_bad = parsed.isna().sum()
    if n_bad > 0:
        ledger.add(f"json_parse_fail_{col_name}", int(n_bad))
    return parsed


def read_and_parse(tsv_path: str) -> tuple[pd.DataFrame, RejectionLedger]:
    """Read TSV and parse JSON columns. Returns (df_with_parsed_cols, ledger)."""
    ledger = RejectionLedger()
    df = pd.read_csv(tsv_path, sep="\t")
    logger.info("Raw rows loaded: %d", len(df))

    df["_alleles_parsed"] = _parse_json_column(df["alleles"], "alleles", ledger)
    df["_positions_parsed"] = _parse_json_column(
        df["peptide_position_info"], "peptide_position_info", ledger
    )

    # Drop rows where either JSON parse failed
    valid_mask = df["_alleles_parsed"].notna() & df["_positions_parsed"].notna()
    n_drop = (~valid_mask).sum()
    if n_drop > 0:
        ledger.add("structural_parse_drop", int(n_drop))
    df = df[valid_mask].reset_index(drop=True)

    return df, ledger


# ── A2: Explode position info → one record per protein ──────────────────────

def explode_positions(df: pd.DataFrame, ledger: RejectionLedger) -> pd.DataFrame:
    """Explode _positions_parsed so each row has one (protein_id, start, end)."""
    records = []
    for idx, row in df.iterrows():
        positions = row["_positions_parsed"]
        if not isinstance(positions, list) or len(positions) == 0:
            ledger.add("empty_positions", 1)
            continue
        for pos in positions:
            if not isinstance(pos, dict):
                ledger.add("invalid_position_entry", 1)
                continue
            pid = pos.get("protein_id")
            start = pos.get("start")
            end = pos.get("end")
            if pid is None or start is None or end is None:
                ledger.add("missing_position_fields", 1)
                continue
            # start/end may come as float from JSON
            try:
                start_1b = _coerce_integral_coordinate(start)
                end_1b = _coerce_integral_coordinate(end)
            except ValueError:
                ledger.add("non_integral_coordinates", 1)
                continue
            except TypeError:
                ledger.add("non_numeric_coordinates", 1)
                continue
            records.append({
                "protein_id": str(pid),
                "start_1b": start_1b,
                "end_1b": end_1b,
                "peptide_seq": row["peptide_seq"],
                "alleles": row["_alleles_parsed"],
                "resolution": row["resolution"],
                "source": row["source"],
                "dataset_source": row["dataset_source"],
                "_parent_idx": idx,
            })

    out = pd.DataFrame(records)
    logger.info("Exploded to %d per-protein records", len(out))
    return out


# ── A3: Coordinate normalization + length filter ─────────────────────────────

def normalize_coordinates(df: pd.DataFrame, min_k: int, max_k: int,
                          ledger: RejectionLedger) -> pd.DataFrame:
    """Convert 1b-inclusive to 0b-half-open. Filter by peptide length [min_k, max_k]."""
    df = df.copy()

    valid_1b = (df["start_1b"] >= 1) & (df["end_1b"] >= df["start_1b"])
    n_invalid_1b = (~valid_1b).sum()
    if n_invalid_1b > 0:
        ledger.add("invalid_1b_coordinates", int(n_invalid_1b))
        df = df[valid_1b].reset_index(drop=True)

    df["start_0b"] = df["start_1b"] - 1
    df["end_0b"] = df["end_1b"]  # 1b inclusive end == 0b half-open end
    df["pep_len"] = df["end_0b"] - df["start_0b"]

    # Invariant check: pep_len must equal end_1b - start_1b + 1
    invariant = df["pep_len"] == (df["end_1b"] - df["start_1b"] + 1)
    n_violate = (~invariant).sum()
    if n_violate > 0:
        ledger.add("coord_invariant_violation", int(n_violate))
        df = df[invariant].reset_index(drop=True)

    # Length filter
    len_mask = (df["pep_len"] >= min_k) & (df["pep_len"] <= max_k)
    n_short = (df["pep_len"] < min_k).sum()
    n_long = (df["pep_len"] > max_k).sum()
    if n_short > 0:
        ledger.add("pep_too_short", int(n_short))
    if n_long > 0:
        ledger.add("pep_too_long", int(n_long))
    df = df[len_mask].reset_index(drop=True)

    logger.info("After coord normalization + length filter: %d records", len(df))
    return df


# ── A4: Allele filter (SA v1 policy) ────────────────────────────────────────

def _classify_allele_row(alleles: list, resolution: str, target: str) -> str:
    """Classify a row into allele policy category.

    Returns one of:
      'strict'   — exact single-allele match with high_res_single resolution
      'multi'    — multi resolution row containing target allele
      'reject'   — does not qualify for either profile
    """
    if resolution == "high_res_single" and alleles == [target]:
        return "strict"
    if resolution == "multi" and target in alleles:
        return "multi"
    return "reject"


def apply_allele_filter(
    df: pd.DataFrame,
    target_allele: str,
    multi_ratio: float,
    multi_seed: int,
    ledger: RejectionLedger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply SA v1 allele filter. Returns (strict_df, balanced_df).

    strict_df  = rows classified as 'strict' only
    balanced_df = strict rows + deterministic multi_ratio sample of 'multi' rows
    """
    df = df.copy()
    df["_allele_class"] = df.apply(
        lambda r: _classify_allele_row(r["alleles"], r["resolution"], target_allele),
        axis=1,
    )

    strict_rows = df[df["_allele_class"] == "strict"].copy()
    multi_rows = df[df["_allele_class"] == "multi"].copy()
    reject_rows = df[df["_allele_class"] == "reject"]

    ledger.add("allele_strict_kept", len(strict_rows))
    ledger.add("allele_multi_candidates", len(multi_rows))
    ledger.add("allele_rejected", len(reject_rows))

    # Deterministic sampling of multi rows
    n_sample = int(len(multi_rows) * multi_ratio)
    if n_sample > 0 and len(multi_rows) > 0:
        multi_sampled = multi_rows.sample(n=n_sample, random_state=multi_seed)
    else:
        multi_sampled = multi_rows.iloc[:0]
    ledger.add("allele_multi_sampled", len(multi_sampled))

    # Assign the single target allele for all kept rows
    for frame in [strict_rows, multi_sampled]:
        frame["allele"] = target_allele

    # Build outputs
    strict_df = strict_rows.drop(columns=["_allele_class", "alleles"])
    balanced_df = pd.concat(
        [strict_rows, multi_sampled], ignore_index=True
    ).drop(columns=["_allele_class", "alleles"])

    logger.info("Strict profile: %d rows; Balanced profile: %d rows",
                len(strict_df), len(balanced_df))
    return strict_df, balanced_df
