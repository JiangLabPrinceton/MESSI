"""Stage C: ProteinSample aggregation.

Covers Tasks C1-C3:
  C1 — grouping by (protein_id, allele) + positive span deduplication
  C2 — group isolation + sequence consistency
  C3 — metadata construction + invariant checks
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

PROTEIN_SAMPLE_COLUMNS = [
    "protein_id",
    "allele",
    "protein_seq",
    "positives_json",
    "sequence_length",
    "input_span_count",
    "positive_count",
    "duplicate_span_count",
    "metadata_json",
]


@dataclass
class AggStats:
    """Aggregation statistics."""
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, reason: str, n: int = 1):
        self.counts[reason] = self.counts.get(reason, 0) + n

    def summary(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))


def aggregate_protein_samples(
    span_df: pd.DataFrame,
    stats: AggStats,
) -> pd.DataFrame:
    """Aggregate span-level records into protein-centric samples.

    Args:
        span_df: Stage B output with protein_seq attached.
        stats: accumulator for aggregation statistics.

    Returns:
        DataFrame with one row per (protein_id, allele).
    """
    if len(span_df) == 0:
        return pd.DataFrame(columns=PROTEIN_SAMPLE_COLUMNS)

    records = []
    grouped = span_df.groupby(["protein_id", "allele"], sort=True)

    for (pid, allele), group in grouped:
        # C2: sequence consistency — all rows in group must have same protein_seq
        seqs = group["protein_seq"].unique()
        if len(seqs) != 1:
            stats.add("drop_inconsistent_seq", len(group))
            logger.warning("Inconsistent protein_seq for (%s, %s): %d variants",
                           pid, allele, len(seqs))
            continue

        protein_seq = seqs[0]
        input_span_count = len(group)

        # C1: deduplicate positives by (start_0b, end_0b, pep_len), count support
        span_tuples = list(zip(group["start_0b"], group["end_0b"], group["pep_len"]))
        span_counter = Counter(span_tuples)
        unique_spans = sorted(span_counter.keys())  # deterministic ordering
        duplicate_span_count = input_span_count - len(unique_spans)

        # Serialize positives with support_n (observation count per span)
        positives = [
            {"start_0b": int(s), "end_0b": int(e), "pep_len": int(k),
             "support_n": span_counter[(s, e, k)]}
            for s, e, k in unique_spans
        ]
        positives_json = json.dumps(positives, separators=(",", ":"))

        # C3: metadata
        source_counts = Counter(group["source"])
        ds_counts = Counter(group["dataset_source"])
        metadata = {
            "source_counts": dict(sorted(source_counts.items())),
            "dataset_source_counts": dict(sorted(ds_counts.items())),
        }
        metadata_json = json.dumps(metadata, separators=(",", ":"))

        records.append({
            "protein_id": pid,
            "allele": allele,
            "protein_seq": protein_seq,
            "positives_json": positives_json,
            "sequence_length": len(protein_seq),
            "input_span_count": input_span_count,
            "positive_count": len(unique_spans),
            "duplicate_span_count": duplicate_span_count,
            "metadata_json": metadata_json,
        })

        stats.add("proteins_created")
        stats.add("total_input_spans", input_span_count)
        stats.add("total_unique_positives", len(unique_spans))
        stats.add("total_duplicates_removed", duplicate_span_count)

    out = pd.DataFrame(records, columns=PROTEIN_SAMPLE_COLUMNS)
    logger.info("Aggregated %d protein samples from %d span records",
                len(out), len(span_df))
    return out


def validate_protein_samples(df: pd.DataFrame) -> None:
    """Validate ProteinSample invariants. Raises on violation."""
    if len(df) == 0:
        return

    # positive_count == len(positives_json)
    for idx, row in df.iterrows():
        positives = json.loads(row["positives_json"])
        if row["positive_count"] != len(positives):
            raise ValueError(
                f"Row {idx} ({row['protein_id']}): positive_count={row['positive_count']} "
                f"!= len(positives)={len(positives)}"
            )
        support_sum = sum(p.get("support_n", 0) for p in positives)
        if support_sum != row["input_span_count"]:
            raise ValueError(
                f"Row {idx} ({row['protein_id']}): sum(support_n)={support_sum} "
                f"!= input_span_count={row['input_span_count']}"
            )
        if row["duplicate_span_count"] != row["input_span_count"] - row["positive_count"]:
            raise ValueError(
                f"Row {idx} ({row['protein_id']}): duplicate_span_count invariant violated"
            )
        if row["sequence_length"] != len(row["protein_seq"]):
            raise ValueError(
                f"Row {idx} ({row['protein_id']}): sequence_length mismatch"
            )
        # Each positive span boundary check
        seq_len = row["sequence_length"]
        for span in positives:
            s, e = span["start_0b"], span["end_0b"]
            if not (0 <= s < e <= seq_len):
                raise ValueError(
                    f"Row {idx} ({row['protein_id']}): span ({s},{e}) out of range for seq_len={seq_len}"
                )
            if span["pep_len"] != e - s:
                raise ValueError(
                    f"Row {idx} ({row['protein_id']}): span pep_len invariant violated"
                )
            if not isinstance(span.get("support_n"), int) or span["support_n"] < 1:
                raise ValueError(
                    f"Row {idx} ({row['protein_id']}): span support_n must be int >= 1"
                )
