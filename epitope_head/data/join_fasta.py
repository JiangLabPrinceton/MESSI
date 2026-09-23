"""Stage B: FASTA join + sequence validation.

Covers Tasks B1-B4:
  B1 — FASTA index + version-strip fallback lookup
  B2 — Profile-aware join (same logic for strict/balanced)
  B3 — Coordinate range + boundary validation
  B4 — Peptide exact-match validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class JoinStats:
    """Per-profile join statistics."""
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, reason: str, n: int = 1):
        self.counts[reason] = self.counts.get(reason, 0) + n

    def summary(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))


# ── B1: FASTA index + lookup resolver ────────────────────────────────────────

def build_fasta_index(fasta_path: str) -> dict[str, str]:
    """Parse FASTA into {accession: sequence} dict.

    Header format expected: >ACCESSION (no description after space used as key).
    """
    index: dict[str, str] = {}
    current_id = None
    chunks: list[str] = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    index[current_id] = "".join(chunks)
                current_id = line[1:].split()[0]  # first token after >
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        index[current_id] = "".join(chunks)

    logger.info("FASTA index built: %d sequences", len(index))
    return index


def resolve_protein(protein_id: str, fasta_index: dict[str, str]) -> tuple[str | None, str]:
    """Lookup protein_id with version-strip fallback.

    Returns (sequence_or_None, hit_type) where hit_type is one of:
      'exact_hit', 'stripped_hit', 'miss'
    """
    if protein_id in fasta_index:
        return fasta_index[protein_id], "exact_hit"

    # Version-strip fallback: P02751.4 -> P02751
    if "." in protein_id:
        base_id = protein_id.rsplit(".", 1)[0]
        if base_id in fasta_index:
            return fasta_index[base_id], "stripped_hit"

    return None, "miss"


# ── B2-B4: Join + validate ───────────────────────────────────────────────────

def join_and_validate(
    span_df: pd.DataFrame,
    fasta_index: dict[str, str],
    stats: JoinStats,
) -> pd.DataFrame:
    """Attach protein_seq, validate coordinates and peptide match.

    Args:
        span_df: SpanRecord DataFrame (from Stage A)
        fasta_index: {accession: sequence}
        stats: accumulator for join statistics

    Returns:
        DataFrame with added 'protein_seq' column, only valid rows retained.
    """
    records = []

    for _, row in span_df.iterrows():
        pid = row["protein_id"]
        seq, hit_type = resolve_protein(pid, fasta_index)
        stats.add(f"lookup_{hit_type}")

        # B1: unresolved
        if seq is None:
            stats.add("drop_unresolved")
            continue

        seq_len = len(seq)
        start_0b = row["start_0b"]
        end_0b = row["end_0b"]

        # B3: coordinate boundary validation
        if start_0b < 0 or end_0b > seq_len or start_0b >= end_0b:
            stats.add("drop_out_of_range")
            continue

        # B3: pep_len invariant
        if row["pep_len"] != end_0b - start_0b:
            stats.add("drop_len_invariant")
            continue

        # B4: peptide exact-match
        extracted = seq[start_0b:end_0b]
        expected = row["peptide_seq"]
        if extracted != expected:
            stats.add("drop_seq_mismatch")
            continue

        stats.add("retained")
        rec = row.to_dict()
        rec["protein_seq"] = seq
        records.append(rec)

    if records:
        out = pd.DataFrame(records).reset_index(drop=True)
    else:
        # Preserve schema even when all rows are rejected
        out_cols = list(span_df.columns) + ["protein_seq"]
        out = pd.DataFrame(columns=out_cols)
    logger.info("Join complete: %d retained from %d input", len(out), len(span_df))
    return out
