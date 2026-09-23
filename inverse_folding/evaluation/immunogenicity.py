"""Immunogenicity scoring wrappers for IF evaluation.

Two independent scorers:
  1. Epitope head (internal) — reuses InferencePredictor from epitope_head package
  2. NetMHCIIpan 4.3 (external) — reuses StandaloneRunner, adds per-design aggregation

Unit convention: all %Rank_EL values in this module are in PERCENTAGE units
(e.g. 2.0 means 2%), matching NetMHCIIpan's native output format.
PeptideScore.el_rank from the runner is a 0-1 fraction and must be converted
before passing to aggregate_nmp_scores.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ── NetMHCIIpan aggregation ──────────────────────────────────────────────────

_STRONG_BINDER_THRESHOLD = 2.0   # %Rank_EL < 2%
_WEAK_BINDER_THRESHOLD = 10.0    # %Rank_EL < 10%
_TOP_K_FOR_MEAN_BEST = 5

# Peptide lengths to scan (MHC-II binding peptides range 12-25 AA)
NMP_PEP_LENGTHS = list(range(12, 26))


def peptide_scores_to_dataframe(scores: list) -> pd.DataFrame:
    """Convert a list of PeptideScore objects to a DataFrame.

    Converts el_rank from 0-1 fraction to percentage for downstream use.
    """
    if not scores:
        return pd.DataFrame(columns=["peptide", "rank_EL", "pos", "core", "el_score"])

    rows = []
    for s in scores:
        rows.append({
            "peptide": s.peptide,
            "rank_EL": s.el_rank * 100.0,  # fraction → percentage
            "pos": s.pos,
            "core": s.core,
            "el_score": s.el_score,
        })
    return pd.DataFrame(rows)


def score_protein_all_lengths(
    runner,
    protein_id: str,
    sequence: str,
    allele: str,
    pep_lengths: List[int] = None,
) -> pd.DataFrame:
    """Score a protein across all MHC-II peptide lengths using StandaloneRunner.

    Args:
        runner: NetMHCIIpanRunner instance (has score_protein method).
        protein_id: identifier for logging/provenance.
        sequence: amino acid string.
        allele: HLA allele (e.g. "HLA-DRB1*07:01").
        pep_lengths: list of peptide lengths to scan. Defaults to 12-25.

    Returns:
        DataFrame with columns [peptide, rank_EL, pos, core, el_score].
        rank_EL is in percentage units (0-100).
    """
    if pep_lengths is None:
        pep_lengths = NMP_PEP_LENGTHS

    all_scores = []
    for k in pep_lengths:
        if len(sequence) < k:
            continue
        scores = runner.score_protein(protein_id, sequence, allele, k)
        all_scores.extend(scores)

    return peptide_scores_to_dataframe(all_scores)


def aggregate_nmp_batch_scores(
    runner,
    entries: List[tuple[str, str]],
    allele: str,
    pep_lengths: List[int] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Score multiple proteins in batch and aggregate per-protein NMP metrics.

    Uses the runner's batch API so callers can amortize NetMHCIIpan startup
    across multiple proteins and peptide lengths.

    Each result dict includes:
      - Standard aggregation fields (n_strong_binders, etc.)
      - scored_lengths: list of peptide lengths that returned results
      - requested_lengths: list of all requested peptide lengths
      - nmp_status: "complete" if all applicable lengths scored,
                    "partial" if some missing, "timeout" if none scored
    """
    if pep_lengths is None:
        pep_lengths = NMP_PEP_LENGTHS

    filtered_entries = [(pid, seq) for pid, seq in entries if seq]
    if not filtered_entries:
        return {}

    batch_scores = runner.score_batch(filtered_entries, allele, pep_lengths)
    aggregated: Dict[str, Dict[str, Any]] = {}

    # Build per-protein set of applicable lengths (skip lengths > seq length)
    seq_by_id = {pid: seq for pid, seq in filtered_entries}

    for protein_id, by_len in batch_scores.items():
        all_scores = []
        for scores in by_len.values():
            all_scores.extend(scores)

        seq_len = len(seq_by_id.get(protein_id, ""))
        applicable = sorted(pl for pl in pep_lengths if pl <= seq_len)
        scored = sorted(by_len.keys())

        if not applicable:
            status = "complete"
        elif scored == applicable:
            status = "complete"
        elif not scored:
            status = "timeout"
        else:
            status = "partial"

        windows_df = peptide_scores_to_dataframe(all_scores)
        agg = aggregate_nmp_scores(windows_df)
        agg["coverage_fraction"] = compute_coverage_fraction(windows_df, seq_len)
        agg["scored_lengths"] = scored
        agg["requested_lengths"] = applicable
        agg["nmp_status"] = status
        aggregated[protein_id] = agg

    # Mark proteins that got zero results (not even in batch_scores)
    for pid, seq in filtered_entries:
        if pid not in aggregated:
            applicable = sorted(pl for pl in pep_lengths if pl <= len(seq))
            aggregated[pid] = {
                "n_strong_binders": 0,
                "n_weak_binders": 0,
                "mean_best_rank": float("nan"),
                "n_windows_scored": 0,
                "coverage_fraction": 0.0,
                "scored_lengths": [],
                "requested_lengths": applicable,
                "nmp_status": "timeout",
            }

    return aggregated


def compute_coverage_fraction(
    window_df: pd.DataFrame,
    seq_len: int,
    strong_threshold: float = _STRONG_BINDER_THRESHOLD,
) -> float:
    """Fraction of residues covered by ≥1 strong NetMHCIIpan window.

    The length-normalized immunogenicity density used for Tier 2 v2
    stratified selection (PLAN_DATA_SEL §12). A "strong" window has
    ``rank_EL < strong_threshold`` (percentage units). Each strong window
    covers residues ``[pos, pos + len(peptide))`` (0-based, per
    PeptideScore.pos); the covered set is the union across all strong
    windows, clipped to ``[0, seq_len)``.

    Args:
        window_df: per-window DataFrame with columns ``pos``, ``peptide``,
            ``rank_EL`` (percentage units, as from peptide_scores_to_dataframe).
        seq_len: protein length (residue count).
        strong_threshold: %Rank_EL cutoff for a strong binder window.

    Returns:
        Coverage fraction in [0, 1]. 0.0 when seq_len <= 0, the frame is
        empty, or no window is strong.
    """
    if seq_len <= 0 or window_df is None or len(window_df) == 0:
        return 0.0

    strong = window_df[window_df["rank_EL"] < strong_threshold]
    if len(strong) == 0:
        return 0.0

    covered = np.zeros(seq_len, dtype=bool)
    for pos, peptide in zip(strong["pos"].to_numpy(), strong["peptide"].to_numpy()):
        start = max(0, int(pos))
        end = min(seq_len, int(pos) + len(peptide))
        if end > start:
            covered[start:end] = True

    return float(covered.sum()) / float(seq_len)


def aggregate_nmp_scores(scores: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate per-window NetMHCIIpan scores into per-design metrics.

    Args:
        scores: DataFrame with columns [peptide, rank_EL].
            rank_EL must be in PERCENTAGE units (e.g. 2.0 = 2%).
            Each row is one scored window (k-mer).

    Returns:
        Dict with: n_strong_binders, n_weak_binders, mean_best_rank, n_windows_scored.
    """
    n = len(scores)
    if n == 0:
        return {
            "n_strong_binders": 0,
            "n_weak_binders": 0,
            "mean_best_rank": float("nan"),
            "n_windows_scored": 0,
        }

    ranks = scores["rank_EL"]
    n_strong = int((ranks < _STRONG_BINDER_THRESHOLD).sum())
    n_weak = int((ranks < _WEAK_BINDER_THRESHOLD).sum())

    # Mean of top-K lowest (= worst-case) ranks
    top_k = min(_TOP_K_FOR_MEAN_BEST, n)
    sorted_ranks = ranks.sort_values().head(top_k)
    mean_best = float(sorted_ranks.mean())

    return {
        "n_strong_binders": n_strong,
        "n_weak_binders": n_weak,
        "mean_best_rank": mean_best,
        "n_windows_scored": n,
    }
