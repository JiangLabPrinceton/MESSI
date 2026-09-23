#!/usr/bin/env python
"""IEDB test-set benchmark: epitope head and NetMHCIIpan vs ground truth.

Loads IEDB test proteins (protein_samples parquet + test_ids split) and runs
both predictors on each. Ground-truth EL-positive spans come from
`positives_json`. Computes per-protein AUC / AP / Recall@K for both head and
NMP using the exact-span-match label convention from
`epitope_head/training/eval_metrics.py`.

Default NMP mode is "original" (no acceleration) to measure a clean baseline
wall time: batch_size=1, max_lengths_per_call=14, n_workers=1 — i.e. one
subprocess call per protein with all peptide lengths in that call.

Usage:
    python scripts/benchmark_iedb_test.py \
        --protein-samples-parquet outputs/manifests/protein_samples_strict.parquet \
        --test-ids outputs/manifests/splits/strict/test_ids.txt \
        --epitope-ckpt /scratch/.../cnn_himp_a1_res03_drb0701_seed42_cv5_fold0/.../best.pt \
        --netmhciipan-bin /.../netMHCIIpan \
        --allele "HLA-DRB1*07:01" \
        --output-json outputs/benchmark_iedb_drb0701.json \
        [--nmp-mode original|accelerated] \
        [--device cuda]
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

from epitope_head.training.eval_metrics import (
    build_window_labels,
    compute_ap,
    compute_auc,
    compute_recall_at_k,
    enumerate_candidate_windows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IEDB test-set benchmark for head + NMP vs ground truth.",
    )
    p.add_argument("--protein-samples-parquet", required=True,
                   help="protein_samples_strict.parquet (Stage C output).")
    p.add_argument("--test-ids", required=True,
                   help="test_ids.txt (one protein_id per line).")
    p.add_argument("--epitope-ckpt", required=True,
                   help="Epitope head checkpoint (.pt).")
    p.add_argument("--netmhciipan-bin", required=True,
                   help="NetMHCIIpan binary path.")
    p.add_argument("--allele", required=True,
                   help="HLA allele (e.g. HLA-DRB1*07:01).")
    p.add_argument("--output-json", required=True,
                   help="Output JSON path.")
    p.add_argument("--nmp-mode", choices=["original", "accelerated"],
                   default="original",
                   help="NMP mode (default: original — batch_size=1, "
                        "max_lengths_per_call=14, n_workers=1).")
    p.add_argument("--device", default="cuda",
                   help="Device for head inference (default: cuda).")
    p.add_argument("--variant-id", default="LC1",
                   help="Epitope head CNN variant ID (default: LC1).")
    p.add_argument("--config-dir", default=None,
                   help="Epitope head config directory (auto-detect if omitted).")
    p.add_argument("--nmp-timeout", type=int, default=600,
                   help="NMP subprocess timeout in seconds (default: 600).")
    p.add_argument("--min-k", type=int, default=12,
                   help="Minimum peptide length (default: 12).")
    p.add_argument("--max-k", type=int, default=25,
                   help="Maximum peptide length (default: 25).")
    p.add_argument("--recall-ks", type=int, nargs="+", default=[50, 100],
                   help="Recall@K values (default: 50 100).")
    p.add_argument("--precision-ks", type=int, nargs="+",
                   default=[10, 25, 50, 100, 200],
                   help="Precision@K values (default: 10 25 50 100 200).")
    p.add_argument("--precision-recall-targets", type=float, nargs="+",
                   default=[0.3, 0.5, 0.7],
                   help="Target recall values for precision-at-recall "
                        "(default: 0.3 0.5 0.7).")
    # Accelerated-mode overrides (ignored in original mode)
    p.add_argument("--nmp-batch-size", type=int, default=8)
    p.add_argument("--nmp-max-lengths-per-call", type=int, default=4)
    p.add_argument("--nmp-workers", type=int, default=1)
    p.add_argument(
        "--near-miss-analysis",
        action="store_true",
        help="Compute optional exact-vs-overlap near-miss diagnostics.",
    )
    # EL landscape metric suite (see doc/EL_new_evaluation.md).
    p.add_argument(
        "--metric-suite",
        choices=["exact", "residue", "iou_ladder", "emd", "all"],
        default="all",
        help="Which metric families to compute beyond the exact-span "
             "primary/conditional baseline. 'exact' disables the EL-landscape "
             "extensions; 'all' (default) enables residue+iou_ladder+emd.",
    )
    p.add_argument(
        "--no-emd", action="store_true",
        help="Force-skip the M3 EMD family even under --metric-suite all. EMD is "
             "the slowest residue aggregation on very long proteins and is unused "
             "by the CV aggregator; skip it for sweep throughput.",
    )
    p.add_argument(
        "--exact-head", action="store_true",
        help="Wave-4 dual-head: score windows with the exact readout z_exact "
             "(predict_protein(return_exact=True)) instead of the region logit z. "
             "Run a checkpoint both ways and compare exact-AP(z_exact) vs "
             "exact-AP(z_region); no-op for single-head checkpoints (z_exact==z).",
    )
    p.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=[1.0, 0.7, 0.5, 0.0],
        help="IoU tiers for the M6 ladder (1.0=exact, 0.0=any overlap).",
    )
    p.add_argument(
        "--bootstrap-n",
        type=int,
        default=1000,
        help="Bootstrap resamples for 95%% CI over proteins (default: 1000).",
    )
    p.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Bootstrap RNG seed for reproducibility (default: 0).",
    )
    # Windows cache for cheap metric iteration. Inference (steps 2 + 3) is
    # the dominant wall-time; once cached, future metric tweaks reload the
    # raw (start, k) → score dicts and skip head + NMP entirely.
    p.add_argument(
        "--windows-cache",
        default=None,
        help="If set, write head_wins+nmp_wins to this parquet path (with a "
             "sidecar .meta.json) after inference.",
    )
    p.add_argument(
        "--resume-from-cache",
        default=None,
        help="If set, load head_wins+nmp_wins from this parquet path and "
             "skip steps 2 (head) and 3 (NMP). Mutually exclusive with "
             "running fresh inference.",
    )
    p.add_argument(
        "--reuse-nmp-from-cache",
        default=None,
        help="If set, load only nmp_wins from this parquet path and skip "
             "step 3 (NMP); head inference (step 2) still runs fresh from "
             "--epitope-ckpt. For head iteration on a frozen NMP. Cache "
             "metadata must match --allele and cover --min-k/--max-k. "
             "Mutually exclusive with --resume-from-cache.",
    )
    return p.parse_args()


def _log(msg: str) -> None:
    print(msg, flush=True)


def _recall_target_key(target: float) -> str:
    """Stable key suffix for recall targets like 0.3 -> '30'."""
    return str(int(round(target * 100)))


def _compute_precision_at_k(
    scores: torch.Tensor, labels: torch.Tensor, k: int,
) -> float | None:
    """Precision@K on descending-ranked windows."""
    if scores.numel() == 0:
        return None
    sorted_indices = torch.argsort(scores, descending=True)
    top_k = min(k, len(sorted_indices))
    if top_k == 0:
        return None
    top_k_labels = labels[sorted_indices[:top_k]]
    return float(top_k_labels.sum().item() / top_k)


def _compute_precision_at_target_recall(
    scores: torch.Tensor, labels: torch.Tensor, target_recall: float,
) -> float | None:
    """Precision at the shortest prefix achieving recall >= target_recall."""
    n_pos = int(labels.sum().item())
    if n_pos == 0:
        return None
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices].float()
    tp_cumsum = torch.cumsum(sorted_labels, dim=0)
    recalls = tp_cumsum / n_pos
    hits = torch.nonzero(recalls >= target_recall, as_tuple=False)
    if hits.numel() == 0:
        return None
    idx = int(hits[0].item())
    tp = float(tp_cumsum[idx].item())
    return tp / float(idx + 1)


def _load_test_entries(
    parquet_path: str, test_ids_path: str, allele: str,
) -> list[dict]:
    """Load test protein entries for the given allele (no length filtering).

    The CNN encoder handles arbitrary lengths and `InferencePredictor`
    provides chunking for ESM variants, so no length cap is applied.
    """
    with open(test_ids_path) as f:
        test_ids = {line.strip() for line in f if line.strip()}

    df = pd.read_parquet(parquet_path)
    df = df[(df["protein_id"].isin(test_ids)) & (df["allele"] == allele)]

    entries: list[dict] = []
    for _, row in df.iterrows():
        entries.append({
            "protein_id": row["protein_id"],
            "protein_seq": row["protein_seq"],
            "positives": json.loads(row["positives_json"]),
        })
    return entries


def _span_overlap_len(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Return overlap length between two half-open spans."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _span_gap(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Return residue gap between spans, or 0 if they overlap."""
    if a[1] <= b[0]:
        return b[0] - a[1]
    if b[1] <= a[0]:
        return a[0] - b[1]
    return 0


def _span_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Return IoU between two half-open spans."""
    ov = _span_overlap_len(a, b)
    if ov == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - ov
    return ov / union


def _build_overlap_labels(
    candidate_spans: torch.Tensor,
    positives: list[dict],
) -> torch.Tensor:
    """Label windows positive if they overlap any GT span."""
    pos_spans = [
        (int(p["start_0b"]), int(p["end_0b"]))
        for p in positives
    ]
    labels = torch.zeros(candidate_spans.shape[0], dtype=torch.long)
    for i in range(candidate_spans.shape[0]):
        span = (
            int(candidate_spans[i, 0].item()),
            int(candidate_spans[i, 1].item()),
        )
        if any(_span_overlap_len(span, pos) > 0 for pos in pos_spans):
            labels[i] = 1
    return labels


def _build_iou50_labels(
    candidate_spans: torch.Tensor,
    positives: list[dict],
) -> torch.Tensor:
    """Label windows positive if they match any GT span with IoU >= 0.5."""
    pos_spans = [
        (int(p["start_0b"]), int(p["end_0b"]))
        for p in positives
    ]
    labels = torch.zeros(candidate_spans.shape[0], dtype=torch.long)
    for i in range(candidate_spans.shape[0]):
        span = (
            int(candidate_spans[i, 0].item()),
            int(candidate_spans[i, 1].item()),
        )
        if any(_span_iou(span, pos) >= 0.5 for pos in pos_spans):
            labels[i] = 1
    return labels


def _sorted_median(values: list[int]) -> int | None:
    """Median of a pre-sortable integer list, or None if empty."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _empty_fp_bucket_counts() -> dict[str, int]:
    """Exclusive bucket counts for high-score false positives."""
    return {
        "overlap_iou_gte_0_8": 0,
        "overlap_iou_gte_0_5": 0,
        "overlap_iou_lt_0_5": 0,
        "near_gap_le_2": 0,
        "near_gap_le_5": 0,
        "near_gap_le_10": 0,
        "far_gap_gt_10": 0,
    }


def _score_bucket_counts(rows: list[dict], cutoffs: tuple[int, ...]) -> dict[str, dict[str, int]]:
    """Aggregate false-positive buckets for each requested cutoff."""
    out: dict[str, dict[str, int]] = {}
    for cutoff in cutoffs:
        counts = _empty_fp_bucket_counts()
        for row in rows[:cutoff]:
            counts[row["bucket"]] += 1
        out[str(cutoff)] = counts
    return out


def _build_ranked_window_rows(
    spans: torch.Tensor,
    scores_t: torch.Tensor,
) -> tuple[list[dict], dict[tuple[int, int], int]]:
    """Build descending-ranked windows with a span->rank map."""
    rows = []
    for i in range(spans.shape[0]):
        span = (
            int(spans[i, 0].item()),
            int(spans[i, 1].item()),
        )
        rows.append({"span": span, "score": float(scores_t[i].item())})
    rows.sort(key=lambda row: row["score"], reverse=True)
    rank_map = {row["span"]: idx + 1 for idx, row in enumerate(rows)}
    return rows, rank_map


def _analyze_near_miss(
    spans: torch.Tensor,
    positives: list[dict],
    scores_t: torch.Tensor,
    fp_cutoffs: tuple[int, ...] = (10, 25, 50, 100),
) -> dict:
    """Summarize exact-vs-overlap ranking behavior for one protein."""
    pos_spans = [
        (int(p["start_0b"]), int(p["end_0b"]))
        for p in positives
    ]
    pos_set = set(pos_spans)
    if not pos_spans:
        return {
            "exact_ap": None,
            "overlap_ap": None,
            "iou50_ap": None,
            "best_exact_rank": None,
            "median_exact_rank": None,
            "median_best_overlap_nonexact_rank": None,
            "n_gt_total": 0,
            "n_gt_with_overlap_outranking_exact": 0,
            "gt_examples": [],
            "top_fp_breakdown": {
                str(cutoff): _empty_fp_bucket_counts() for cutoff in fp_cutoffs
            },
            "top_shift_pairs": [],
        }
    ranked_rows, rank_map = _build_ranked_window_rows(spans, scores_t)
    overlap_labels = _build_overlap_labels(spans, positives)
    iou50_labels = _build_iou50_labels(spans, positives)

    gt_examples = []
    exact_ranks: list[int] = []
    best_overlap_nonexact_ranks: list[int] = []
    n_outranking = 0
    shift_counts: Counter[tuple[int, int]] = Counter()

    for gt_span in pos_spans:
        exact_rank = rank_map[gt_span]
        exact_ranks.append(exact_rank)
        best_overlap_nonexact = None
        for row in ranked_rows:
            span = row["span"]
            if span == gt_span:
                continue
            if _span_overlap_len(span, gt_span) > 0:
                best_overlap_nonexact = row
                break

        best_rank = None
        best_span = None
        best_iou = None
        if best_overlap_nonexact is not None:
            best_span = best_overlap_nonexact["span"]
            best_rank = rank_map[best_span]
            best_iou = _span_iou(best_span, gt_span)
            best_overlap_nonexact_ranks.append(best_rank)
            shift_counts[(best_span[0] - gt_span[0], best_span[1] - gt_span[1])] += 1
            if best_rank < exact_rank:
                n_outranking += 1

        gt_examples.append({
            "gt_span": [gt_span[0], gt_span[1]],
            "exact_rank": exact_rank,
            "best_overlap_nonexact_rank": best_rank,
            "best_overlap_nonexact_span": (
                None if best_span is None else [best_span[0], best_span[1]]
            ),
            "best_overlap_nonexact_iou": best_iou,
        })

    fp_rows = []
    for row in ranked_rows:
        span = row["span"]
        if span in pos_set:
            continue
        overlap_ious = [
            _span_iou(span, pos_span)
            for pos_span in pos_spans
            if _span_overlap_len(span, pos_span) > 0
        ]
        if overlap_ious:
            best_iou = max(overlap_ious)
            if best_iou >= 0.8:
                bucket = "overlap_iou_gte_0_8"
            elif best_iou >= 0.5:
                bucket = "overlap_iou_gte_0_5"
            else:
                bucket = "overlap_iou_lt_0_5"
        else:
            nearest_gap = min(_span_gap(span, pos_span) for pos_span in pos_spans)
            if nearest_gap <= 2:
                bucket = "near_gap_le_2"
            elif nearest_gap <= 5:
                bucket = "near_gap_le_5"
            elif nearest_gap <= 10:
                bucket = "near_gap_le_10"
            else:
                bucket = "far_gap_gt_10"
        fp_rows.append({"span": span, "bucket": bucket})

    return {
        "exact_ap": compute_ap(scores_t, build_window_labels(spans, positives)),
        "overlap_ap": compute_ap(scores_t, overlap_labels),
        "iou50_ap": compute_ap(scores_t, iou50_labels),
        "best_exact_rank": min(exact_ranks) if exact_ranks else None,
        "median_exact_rank": _sorted_median(exact_ranks),
        "median_best_overlap_nonexact_rank": _sorted_median(best_overlap_nonexact_ranks),
        "n_gt_total": len(pos_spans),
        "n_gt_with_overlap_outranking_exact": n_outranking,
        "gt_examples": gt_examples,
        "top_fp_breakdown": _score_bucket_counts(fp_rows, fp_cutoffs),
        "top_shift_pairs": [
            {
                "delta_start": ds,
                "delta_end": de,
                "count": count,
            }
            for (ds, de), count in shift_counts.most_common(10)
        ],
    }


def _select_near_miss_group(per_protein_row: dict) -> str | None:
    """Group proteins by exact-AP winner for near-miss summaries."""
    head_primary = per_protein_row["metrics"].get("head_primary")
    nmp_primary = per_protein_row["metrics"].get("nmp_primary")
    if head_primary is None or nmp_primary is None:
        return None
    head_ap = head_primary.get("ap")
    nmp_ap = nmp_primary.get("ap")
    if head_ap is None or nmp_ap is None or head_ap == nmp_ap:
        return None
    return "head_better" if head_ap > nmp_ap else "nmp_better"


def _summarize_near_miss_group(rows: list[dict]) -> dict:
    """Aggregate near-miss outputs for a group of proteins."""
    near_rows = [
        row["near_miss"]
        for row in rows
        if row.get("near_miss") is not None
    ]
    if not near_rows:
        return {
            "n_proteins": 0,
            "mean_exact_ap": None,
            "mean_overlap_ap": None,
            "mean_iou50_ap": None,
            "median_exact_rank": None,
            "median_best_overlap_nonexact_rank": None,
            "n_gt_total": 0,
            "n_gt_with_overlap_outranking_exact": 0,
            "gt_with_overlap_outranking_exact_fraction": None,
            "top_fp_breakdown": {},
        }

    exact_aps = [float(row["exact_ap"]) for row in near_rows if row["exact_ap"] is not None]
    overlap_aps = [float(row["overlap_ap"]) for row in near_rows if row["overlap_ap"] is not None]
    iou50_aps = [float(row["iou50_ap"]) for row in near_rows if row["iou50_ap"] is not None]
    exact_ranks = [
        int(gt["exact_rank"])
        for row in near_rows
        for gt in row["gt_examples"]
        if gt.get("exact_rank") is not None
    ]
    overlap_ranks = [
        int(gt["best_overlap_nonexact_rank"])
        for row in near_rows
        for gt in row["gt_examples"]
        if gt.get("best_overlap_nonexact_rank") is not None
    ]
    total_gt = sum(int(row["n_gt_total"]) for row in near_rows)
    outranking_gt = sum(
        int(row["n_gt_with_overlap_outranking_exact"]) for row in near_rows
    )

    cutoffs = sorted({
        cutoff
        for row in near_rows
        for cutoff in row["top_fp_breakdown"]
    }, key=int)
    top_fp_breakdown = {}
    for cutoff in cutoffs:
        counts = _empty_fp_bucket_counts()
        for row in near_rows:
            by_cutoff = row["top_fp_breakdown"].get(cutoff)
            if by_cutoff is None:
                continue
            for key, value in by_cutoff.items():
                counts[key] += int(value)
        top_fp_breakdown[cutoff] = counts

    return {
        "n_proteins": len(near_rows),
        "mean_exact_ap": sum(exact_aps) / len(exact_aps) if exact_aps else None,
        "mean_overlap_ap": sum(overlap_aps) / len(overlap_aps) if overlap_aps else None,
        "mean_iou50_ap": sum(iou50_aps) / len(iou50_aps) if iou50_aps else None,
        "median_exact_rank": _sorted_median(exact_ranks),
        "median_best_overlap_nonexact_rank": _sorted_median(overlap_ranks),
        "n_gt_total": total_gt,
        "n_gt_with_overlap_outranking_exact": outranking_gt,
        "gt_with_overlap_outranking_exact_fraction": (
            outranking_gt / total_gt if total_gt > 0 else None
        ),
        "top_fp_breakdown": top_fp_breakdown,
    }


# ── EL landscape metric suite (doc/EL_new_evaluation.md) ─────────────
#
# All three families (M2a residue, M6 IoU ladder, M3 EMD) operate on the
# same per-protein inputs as the existing primary/conditional metrics.
# Score convention: wins dicts passed to these helpers must already use the
# "higher = more immunogenic" orientation (NMP el_rank pre-inverted by the
# caller). Helpers are pure (no global state) so they are unit-testable.


def _invert_nmp_wins(
    wins: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    """Flip NMP el_rank → higher-is-more-immunogenic convention."""
    return {k: -v for k, v in wins.items()}


# ── M2a · residue-level aggregation ─────────────────────────────────

def _aggregate_residue_scores(
    wins_inv: dict[tuple[int, int], float],
    L: int,
    k_filter: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-residue max score over windows containing r.

    Args:
        wins_inv: (start_0b, k) → score, higher-is-more-immunogenic.
        L: protein length.
        k_filter: if not None, only aggregate windows of this length.

    Returns:
        score_res: float32[L], -inf where no qualifying span covers r.
        mask: bool[L], True where residue was covered.
    """
    score_res = np.full(L, -np.inf, dtype=np.float32)
    for (s, k), v in wins_inv.items():
        if k_filter is not None and k != k_filter:
            continue
        end = min(s + k, L)
        if s >= L or end <= 0 or end <= s:
            continue
        segment = score_res[s:end]
        np.maximum(segment, float(v), out=segment)
    mask = np.isfinite(score_res)
    return score_res, mask


def _build_residue_labels(
    positives: list[dict], L: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_cover, y_density), each int64[L]."""
    y_density = np.zeros(L, dtype=np.int64)
    for p in positives:
        s = int(p["start_0b"])
        e = min(int(p["end_0b"]), L)
        if e <= s:
            continue
        y_density[max(s, 0):e] += 1
    y_cover = (y_density > 0).astype(np.int64)
    return y_cover, y_density


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Pearson r, None if degenerate."""
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    r = float(np.corrcoef(x.astype(np.float64), y.astype(np.float64))[0, 1])
    return None if np.isnan(r) else r


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Spearman ρ via scipy, None if degenerate."""
    if x.size < 2:
        return None
    res = scipy_stats.spearmanr(x, y)
    r = float(res.statistic)
    return None if np.isnan(r) else r


def _compute_residue_metrics(
    score_res: np.ndarray,
    mask: np.ndarray,
    y_cover: np.ndarray,
    y_density: np.ndarray,
) -> Optional[dict]:
    """AUC/AP vs y_cover + Pearson/Spearman vs y_density on covered residues.

    Returns None if not evaluable (no coverage, or all-positive/all-negative
    label vector on the covered subset).
    """
    if mask.sum() == 0:
        return None
    cov = mask
    scores_t = torch.from_numpy(score_res[cov].astype(np.float32))
    labels_t = torch.from_numpy(y_cover[cov].astype(np.int64))
    density_sub = y_density[cov]
    n_pos = int(labels_t.sum().item())
    n_neg = int((labels_t == 0).sum().item())
    if n_pos == 0 or n_neg == 0:
        return None
    return {
        "auc": compute_auc(scores_t, labels_t),
        "ap": compute_ap(scores_t, labels_t),
        "pearson": _pearson_corr(score_res[cov], density_sub),
        "spearman": _spearman_corr(score_res[cov], density_sub),
        "n_residues_scored": int(cov.sum()),
        "n_residues_total": int(score_res.size),
    }


# ── M6 · IoU ladder with COCO-style greedy 1-to-1 assignment ───────

def _iou_greedy_labels(
    k15_spans: np.ndarray,  # [M, 2] (start, end) half-open
    scores: np.ndarray,     # [M]
    positives: list[dict],
    iou_threshold: float,
) -> tuple[np.ndarray, int]:
    """Greedy 1-to-1 matching of predictions → GT spans.

    Predictions are sorted by score desc. For each prediction, match to the
    highest-IoU unassigned GT whose IoU ≥ threshold. Matched → TP, else FP.
    For `iou_threshold == 0.0` (the "any overlap" tier) any overlap > 0
    qualifies.

    Returns (tp_labels, n_gt) where tp_labels is in score-descending order.
    """
    gt_spans = [
        (int(p["start_0b"]), int(p["end_0b"]))
        for p in positives
    ]
    n_gt = len(gt_spans)
    M = int(k15_spans.shape[0])
    if M == 0:
        return np.zeros(0, dtype=np.int64), n_gt

    order = np.argsort(-scores, kind="stable")
    tp = np.zeros(M, dtype=np.int64)
    matched = [False] * n_gt
    for rank, idx in enumerate(order):
        span = (int(k15_spans[idx, 0]), int(k15_spans[idx, 1]))
        best_gt = -1
        best_iou = -1.0
        for g, gt in enumerate(gt_spans):
            if matched[g]:
                continue
            if iou_threshold == 0.0:
                if _span_overlap_len(span, gt) == 0:
                    continue
                iou = _span_iou(span, gt)
            else:
                iou = _span_iou(span, gt)
                if iou < iou_threshold:
                    continue
            if iou > best_iou:
                best_iou = iou
                best_gt = g
        if best_gt >= 0:
            matched[best_gt] = True
            tp[rank] = 1
    return tp, n_gt


def _detection_ap(sorted_tp: np.ndarray, n_gt: int) -> Optional[float]:
    """COCO-style AP: ∑_k (TP_k/k) · I(TP_k) / n_gt."""
    if n_gt == 0 or sorted_tp.size == 0:
        return None
    cum_tp = np.cumsum(sorted_tp)
    ranks = np.arange(1, sorted_tp.size + 1, dtype=np.float64)
    precisions = cum_tp / ranks
    return float((precisions * sorted_tp).sum() / n_gt)


def _detection_recall_at_k(
    sorted_tp: np.ndarray, n_gt: int, k: int,
) -> Optional[float]:
    """Recall@K for detection: fraction of GT matched in top-K predictions."""
    if n_gt == 0:
        return None
    top = sorted_tp[:min(k, sorted_tp.size)]
    return float(top.sum() / n_gt)


def _detection_auc(sorted_tp: np.ndarray) -> Optional[float]:
    """Rank-AUC on the TP/FP binary labels from greedy assignment."""
    if sorted_tp.size == 0:
        return None
    if sorted_tp.sum() == 0 or sorted_tp.sum() == sorted_tp.size:
        return None
    # Implicit scores are the descending rank; reuse compute_auc by
    # assigning strictly decreasing scores.
    scores_t = torch.arange(sorted_tp.size, 0, -1, dtype=torch.float32)
    labels_t = torch.from_numpy(sorted_tp.astype(np.int64))
    return compute_auc(scores_t, labels_t)


def _iou_threshold_key(t: float) -> str:
    if t >= 1.0:
        return "exact"
    if t <= 0.0:
        return "overlap_any"
    return f"iou_{t:.2f}".replace("0.", "0p")


def _compute_iou_ladder(
    wins_inv: dict[tuple[int, int], float],
    L: int,
    positives: list[dict],
    iou_thresholds: list[float],
    recall_ks: tuple[int, ...],
) -> Optional[dict]:
    """M6 ladder over the full multi-k span set (per doc §5 head note).

    IoU is length-agnostic — predictions and GT spans of differing length are
    handled directly by `|w∩g| / |w∪g|`. Returns None if no in-bounds
    predictions exist.
    """
    pred = [
        ((s, s + k), v)
        for (s, k), v in wins_inv.items()
        if 0 <= s and s + k <= L
    ]
    if not pred:
        return None
    spans_np = np.array([[a, b] for ((a, b), _) in pred], dtype=np.int64)
    scores_np = np.array([v for (_, v) in pred], dtype=np.float64)

    out: dict = {
        "n_pred_windows": int(spans_np.shape[0]),
        "n_gt": len(positives),
        "tiers": {},
    }
    for t in iou_thresholds:
        sorted_tp, n_gt = _iou_greedy_labels(spans_np, scores_np, positives, t)
        tier = {
            "ap": _detection_ap(sorted_tp, n_gt),
            "auc": _detection_auc(sorted_tp),
        }
        for rk in recall_ks:
            tier[f"recall_{rk}"] = _detection_recall_at_k(sorted_tp, n_gt, rk)
        out["tiers"][_iou_threshold_key(t)] = tier
    return out


# ── M3 · Normalized 1D EMD ──────────────────────────────────────────

def _compute_emd(
    wins_inv: dict[tuple[int, int], float],
    L: int,
    y_density: np.ndarray,
    k_filter: Optional[int] = None,
) -> Optional[dict]:
    """1D EMD between per-residue pred density and GT coverage density.

    Uses the residue-level `score(r)` from §5.1 (max over the full multi-k
    span set). Returns None if either distribution is degenerate. The
    `k_filter` argument is preserved for ablations only.
    """
    if y_density.sum() == 0:
        return None
    score_res, mask = _aggregate_residue_scores(wins_inv, L, k_filter=k_filter)
    if not mask.all():
        # Residue not covered by any qualifying span → EMD undefined.
        return None
    # softplus with τ=1 (fixed per doc §5.3)
    p_pred_raw = np.log1p(np.exp(np.minimum(score_res.astype(np.float64), 40.0)))
    # handle overflow for large positive scores: softplus(x) ≈ x for x>>0
    big = score_res.astype(np.float64) > 40.0
    p_pred_raw[big] = score_res.astype(np.float64)[big]
    total = p_pred_raw.sum()
    if total <= 0:
        return None
    p_pred = p_pred_raw / total
    p_gt = y_density.astype(np.float64) / y_density.sum()
    cdf_pred = np.cumsum(p_pred)
    cdf_gt = np.cumsum(p_gt)
    emd_norm = float(np.abs(cdf_pred - cdf_gt).sum() / L)
    return {
        "emd_norm": emd_norm,
        "similarity": 1.0 - emd_norm,
    }


# ── Statistics: bootstrap CI + Wilcoxon ──────────────────────────────

def _bootstrap_ci(
    values: list[float],
    n_resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> Optional[dict]:
    """Non-parametric percentile bootstrap CI over proteins."""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return {
        "mean": float(arr.mean()),
        "low": lo,
        "high": hi,
        "n": int(arr.size),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
    }


def _paired_wilcoxon(
    head: list[Optional[float]], nmp: list[Optional[float]],
) -> Optional[dict]:
    """Wilcoxon signed-rank, head - nmp. Matched by position in input lists."""
    assert len(head) == len(nmp)
    pairs = [
        (h, n) for h, n in zip(head, nmp)
        if h is not None and n is not None
    ]
    if len(pairs) < 2:
        return None
    h_arr = np.array([p[0] for p in pairs], dtype=np.float64)
    n_arr = np.array([p[1] for p in pairs], dtype=np.float64)
    diffs = h_arr - n_arr
    if np.all(diffs == 0):
        return {
            "n_pairs": len(pairs),
            "statistic": 0.0,
            "pvalue": 1.0,
            "mean_diff": 0.0,
            "median_diff": 0.0,
            "effect_r": 0.0,
        }
    res = scipy_stats.wilcoxon(h_arr, n_arr, zero_method="wilcox", alternative="two-sided")
    # rank-biserial effect size: r = 1 - 2W/(n(n+1)/2) using signed ranks.
    abs_ranks = scipy_stats.rankdata(np.abs(diffs[diffs != 0]))
    w_plus = abs_ranks[diffs[diffs != 0] > 0].sum()
    n_nonzero = (diffs != 0).sum()
    denom = n_nonzero * (n_nonzero + 1) / 2.0
    effect_r = float(2 * w_plus / denom - 1.0) if denom > 0 else 0.0
    return {
        "n_pairs": len(pairs),
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "mean_diff": float(diffs.mean()),
        "median_diff": float(np.median(diffs)),
        "effect_r": effect_r,
    }


# ── Windows cache (parquet) ──────────────────────────────────────────
#
# Schema (long format makes head/NMP coverage gaps explicit):
#   columns = [protein_id, start, k, head_score, nmp_el_rank]
#   rows    = one per (pid, start, k) where either predictor scored
# Sidecar `<path>.meta.json` records the inference config so a stale cache
# can be flagged before reuse.


def _save_windows_cache(
    path: str,
    head_wins_by_pid: dict[str, dict[tuple[int, int], float]],
    nmp_wins_by_pid: dict[str, dict[tuple[int, int], float]],
    metadata: dict,
) -> None:
    """Write the raw (start, k) → score dicts to a parquet + meta sidecar."""
    pids = sorted(set(head_wins_by_pid) | set(nmp_wins_by_pid))
    rows: list[dict] = []
    for pid in pids:
        h = head_wins_by_pid.get(pid, {})
        n = nmp_wins_by_pid.get(pid, {})
        for key in sorted(set(h) | set(n)):
            s, k = int(key[0]), int(key[1])
            rows.append({
                "protein_id": pid,
                "start": s,
                "k": k,
                "head_score": float(h[key]) if key in h else float("nan"),
                "nmp_el_rank": float(n[key]) if key in n else float("nan"),
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    df.to_parquet(path, index=False)
    with open(path + ".meta.json", "w") as f:
        json.dump(metadata, f, indent=2)


def _validate_nmp_cache_meta(
    cache_meta: dict,
    *,
    allele: str,
    min_k: int,
    max_k: int,
) -> Optional[str]:
    """Return an error message for hard-incompatible NMP cache metadata, or
    None if the cache is usable. Soft mismatches (e.g. nmp_mode) are not
    surfaced here — callers handle them as warnings."""
    if not cache_meta:
        return None
    cache_allele = cache_meta.get("allele")
    if cache_allele and cache_allele != allele:
        return (
            f"cache allele={cache_allele!r} != requested {allele!r}; "
            f"NMP scores are allele-specific. Re-run NMP from scratch."
        )
    cache_min_k = cache_meta.get("min_k")
    cache_max_k = cache_meta.get("max_k")
    if cache_min_k is not None and cache_max_k is not None:
        if cache_min_k > min_k or cache_max_k < max_k:
            return (
                f"cache k-range [{cache_min_k},{cache_max_k}] does not "
                f"cover requested [{min_k},{max_k}]; NMP would be missing "
                f"windows. Re-run NMP from scratch."
            )
    return None


def _load_windows_cache(
    path: str,
) -> tuple[dict[str, dict[tuple[int, int], float]],
           dict[str, dict[tuple[int, int], float]],
           dict]:
    """Reload (head_wins_by_pid, nmp_wins_by_pid, metadata)."""
    df = pd.read_parquet(path)
    head: dict[str, dict[tuple[int, int], float]] = {}
    nmp: dict[str, dict[tuple[int, int], float]] = {}
    for row in df.itertuples(index=False):
        pid = row.protein_id
        key = (int(row.start), int(row.k))
        if pd.notna(row.head_score):
            head.setdefault(pid, {})[key] = float(row.head_score)
        if pd.notna(row.nmp_el_rank):
            nmp.setdefault(pid, {})[key] = float(row.nmp_el_rank)
    meta_path = path + ".meta.json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return head, nmp, meta


def _collect_pp(
    per_protein: list[dict], path: tuple[str, ...],
) -> list[Optional[float]]:
    """Walk a dotted path into each row; return list aligned with per_protein."""
    out: list[Optional[float]] = []
    for row in per_protein:
        cur: object = row
        for key in path:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(key)
            if cur is None:
                break
        out.append(float(cur) if isinstance(cur, (int, float)) else None)
    return out


def main() -> int:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    t0 = time.time()

    # ── Print config ─────────────────────────────────────────────────────
    _log("=" * 60)
    _log("IEDB Test-Set Benchmark: Head + NMP vs Ground Truth")
    _log(f"  Parquet            : {args.protein_samples_parquet}")
    _log(f"  Test IDs           : {args.test_ids}")
    _log(f"  Epitope ckpt       : {args.epitope_ckpt}")
    _log(f"  NMP binary         : {args.netmhciipan_bin}")
    _log(f"  Allele             : {args.allele}")
    _log(f"  NMP mode           : {args.nmp_mode}")
    _log(f"  Device             : {args.device}")
    _log(f"  Variant ID         : {args.variant_id}")
    _log(f"  k range            : [{args.min_k}, {args.max_k}]")
    _log(f"  Recall@K           : {args.recall_ks}")
    _log(f"  Precision@K        : {args.precision_ks}")
    _log(f"  Precision@Recall   : {args.precision_recall_targets}")
    if args.nmp_mode == "accelerated":
        _log(f"  NMP batch size     : {args.nmp_batch_size}")
        _log(f"  NMP lengths/call   : {args.nmp_max_lengths_per_call}")
        _log(f"  NMP workers        : {args.nmp_workers}")
    _log(f"  NMP timeout        : {args.nmp_timeout}s")
    _log(f"  Near-miss analysis : {args.near_miss_analysis}")
    _log(f"  Metric suite       : {args.metric_suite}")
    if args.metric_suite in ("iou_ladder", "all"):
        _log(f"  IoU thresholds     : {args.iou_thresholds}")
    if args.metric_suite != "exact":
        _log(f"  Bootstrap resamples: {args.bootstrap_n} (seed={args.bootstrap_seed})")
    _log(f"  Windows cache      : {args.windows_cache or '(disabled)'}")
    _log(f"  Resume from cache  : {args.resume_from_cache or '(no)'}")
    _log(f"  Reuse NMP from cache: {args.reuse_nmp_from_cache or '(no)'}")
    _log(f"  Output JSON        : {args.output_json}")
    _log("=" * 60)

    if args.resume_from_cache and args.reuse_nmp_from_cache:
        _log("ERROR: --resume-from-cache and --reuse-nmp-from-cache are "
             "mutually exclusive (they're alternative read paths).")
        return 1
    if args.windows_cache and args.resume_from_cache:
        _log("ERROR: --windows-cache (write) and --resume-from-cache (read) "
             "are mutually exclusive. Pick one.")
        return 1
    if args.resume_from_cache and not os.path.exists(args.resume_from_cache):
        _log(f"ERROR: cache file not found: {args.resume_from_cache}")
        return 1
    if args.reuse_nmp_from_cache and not os.path.exists(args.reuse_nmp_from_cache):
        _log(f"ERROR: cache file not found: {args.reuse_nmp_from_cache}")
        return 1

    # Toggles for metric suites (computed once; used in per-protein loop).
    suite = args.metric_suite
    do_residue = suite in ("residue", "all")
    do_iou = suite in ("iou_ladder", "all")
    do_emd = suite in ("emd", "all") and not args.no_emd
    iou_thresholds = list(args.iou_thresholds)

    # ── Step 1: Load test entries ──────────────────────────────────────
    _log("\n[1/4] Loading test entries...")
    entries = _load_test_entries(
        args.protein_samples_parquet, args.test_ids, args.allele,
    )
    if entries:
        lengths = [len(e["protein_seq"]) for e in entries]
        _log(f"  Loaded {len(entries)} test proteins for allele {args.allele} "
             f"(len: min={min(lengths)}, median={sorted(lengths)[len(lengths) // 2]}, "
             f"max={max(lengths)}).")
    else:
        _log("ERROR: No test proteins found.")
        return 1

    # NMP runner config (recorded into output JSON regardless of cache path).
    if args.nmp_mode == "original":
        # One protein, all lengths in one subprocess — the "original" naive call.
        batch_size, max_lengths_per_call, n_workers = 1, 14, 1
    else:
        batch_size = args.nmp_batch_size
        max_lengths_per_call = args.nmp_max_lengths_per_call
        n_workers = args.nmp_workers

    head_wins_by_pid: dict[str, dict[tuple[int, int], float]] = {}
    nmp_wins_by_pid: dict[str, dict[tuple[int, int], float]] = {}
    per_protein_nmp_timing: list[dict] = []
    t_head_elapsed = 0.0
    t_nmp_elapsed = 0.0

    if args.resume_from_cache:
        # ── Steps 2+3 short-circuit ────────────────────────────────────
        _log(f"\n[2-3/4] Resuming from windows cache: {args.resume_from_cache}")
        t_load = time.time()
        head_wins_by_pid, nmp_wins_by_pid, cache_meta = _load_windows_cache(
            args.resume_from_cache,
        )
        _log(f"  Loaded head pids={len(head_wins_by_pid)}, "
             f"nmp pids={len(nmp_wins_by_pid)} in {time.time() - t_load:.2f}s")
        if cache_meta:
            _log(f"  Cache meta: allele={cache_meta.get('allele')!r}  "
                 f"k=[{cache_meta.get('min_k')},{cache_meta.get('max_k')}]  "
                 f"variant_id={cache_meta.get('variant_id')!r}")
            if cache_meta.get("allele") and cache_meta["allele"] != args.allele:
                _log(f"  WARNING: cache allele={cache_meta['allele']!r} != "
                     f"requested {args.allele!r} — proceeding anyway.")
            if (cache_meta.get("min_k") is not None
                    and (cache_meta["min_k"] > args.min_k
                         or cache_meta["max_k"] < args.max_k)):
                _log(f"  WARNING: cache k-range "
                     f"[{cache_meta['min_k']},{cache_meta['max_k']}] is narrower "
                     f"than requested [{args.min_k},{args.max_k}].")
        # Synthetic timing record so the per-protein timing column stays a
        # stable schema; chunk_seconds=0 marks "loaded from cache".
        for e in entries:
            per_protein_nmp_timing.append({
                "protein_id": e["protein_id"],
                "sequence_length": len(e["protein_seq"]),
                "chunk_seconds": 0.0,
                "from_cache": True,
            })
    elif args.reuse_nmp_from_cache:
        # ── Step 3 short-circuit: NMP from cache, head fresh ──────────
        _log(f"\n[2-3/4] Reusing NMP from cache (head will run fresh): "
             f"{args.reuse_nmp_from_cache}")
        t_load = time.time()
        _ignored_head, nmp_wins_by_pid, cache_meta = _load_windows_cache(
            args.reuse_nmp_from_cache,
        )
        _log(f"  Loaded NMP pids={len(nmp_wins_by_pid)} "
             f"(head dict from cache discarded) in {time.time() - t_load:.2f}s")
        # Strict cache validation — NMP scores depend on allele and k coverage,
        # so a mismatched cache would silently corrupt the benchmark.
        if cache_meta:
            _log(f"  Cache meta: allele={cache_meta.get('allele')!r}  "
                 f"k=[{cache_meta.get('min_k')},{cache_meta.get('max_k')}]  "
                 f"nmp_mode={cache_meta.get('nmp_mode')!r}  "
                 f"epitope_ckpt={cache_meta.get('epitope_ckpt')!r}")
            err = _validate_nmp_cache_meta(
                cache_meta,
                allele=args.allele,
                min_k=args.min_k,
                max_k=args.max_k,
            )
            if err is not None:
                _log(f"ERROR: {err}")
                return 1
            if (cache_meta.get("nmp_mode")
                    and cache_meta["nmp_mode"] != args.nmp_mode):
                _log(f"  WARNING: cache nmp_mode={cache_meta['nmp_mode']!r} != "
                     f"requested {args.nmp_mode!r} — proceeding (NMP outputs "
                     f"should be deterministic across modes).")
        # Warn if the cache covers fewer proteins than the current test set —
        # missing proteins fall through to NMP_FILL_SCORE imputation later.
        cached_pids = set(nmp_wins_by_pid)
        test_pids = {e["protein_id"] for e in entries}
        missing_pids = test_pids - cached_pids
        if missing_pids:
            _log(f"  WARNING: {len(missing_pids)} test proteins missing from NMP "
                 f"cache (out of {len(test_pids)}); their NMP windows will be "
                 f"imputed with worst-case el_rank=1.0. Examples: "
                 f"{sorted(missing_pids)[:5]}")
        # Synthetic NMP timing — head timing is real (filled below).
        for e in entries:
            per_protein_nmp_timing.append({
                "protein_id": e["protein_id"],
                "sequence_length": len(e["protein_seq"]),
                "chunk_seconds": 0.0,
                "from_cache": True,
            })

        # ── Step 2: Head inference (fresh) ─────────────────────────────
        _log("\n[2/4] Running epitope head inference (NMP cached)...")
        from scripts.head_runtime import load_epitope_predictor

        config_dir = args.config_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "epitope_head", "configs",
        )
        predictor = load_epitope_predictor(
            config_dir=config_dir,
            checkpoint_path=args.epitope_ckpt,
            variant_id=args.variant_id,
            device=args.device,
        )

        t_head_start = time.time()
        for idx, entry in enumerate(entries, start=1):
            pred = predictor.predict_protein(
                entry["protein_seq"], allele_idx=0, return_exact=args.exact_head,
            )
            wins = {}
            for w in pred["window_logits"]:
                k = w["end_0b"] - w["start_0b"]
                # --exact-head routes ALL window scores to z_exact (falls back to
                # z for single-head checkpoints that carry no z_exact).
                wins[(w["start_0b"], k)] = (
                    w.get("z_exact", w["z"]) if args.exact_head else w["z"]
                )
            head_wins_by_pid[entry["protein_id"]] = wins
            if idx % 20 == 0 or idx == len(entries):
                _log(f"  Head: {idx}/{len(entries)}")
        t_head_elapsed = time.time() - t_head_start
        _log(f"  Head inference complete ({t_head_elapsed:.1f}s)")

        if args.windows_cache:
            _log(f"\n  Writing windows cache (fresh head + reused NMP) → "
                 f"{args.windows_cache}")
            t_save = time.time()
            _save_windows_cache(
                args.windows_cache,
                head_wins_by_pid,
                nmp_wins_by_pid,
                metadata={
                    "allele": args.allele,
                    "min_k": int(args.min_k),
                    "max_k": int(args.max_k),
                    "variant_id": args.variant_id,
                    "epitope_ckpt": args.epitope_ckpt,
                    "n_proteins": len(entries),
                    "nmp_mode": args.nmp_mode,
                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "nmp_source": (
                        f"reused_from:{os.path.abspath(args.reuse_nmp_from_cache)}"
                    ),
                },
            )
            _log(f"  Cache write complete ({time.time() - t_save:.2f}s)")
    else:
        # ── Step 2: Head inference ─────────────────────────────────────
        _log("\n[2/4] Running epitope head inference...")
        from scripts.head_runtime import load_epitope_predictor

        config_dir = args.config_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "epitope_head", "configs",
        )
        predictor = load_epitope_predictor(
            config_dir=config_dir,
            checkpoint_path=args.epitope_ckpt,
            variant_id=args.variant_id,
            device=args.device,
        )

        t_head_start = time.time()
        for idx, entry in enumerate(entries, start=1):
            pred = predictor.predict_protein(
                entry["protein_seq"], allele_idx=0, return_exact=args.exact_head,
            )
            wins = {}
            for w in pred["window_logits"]:
                k = w["end_0b"] - w["start_0b"]
                # --exact-head routes ALL window scores to z_exact (falls back to
                # z for single-head checkpoints that carry no z_exact).
                wins[(w["start_0b"], k)] = (
                    w.get("z_exact", w["z"]) if args.exact_head else w["z"]
                )
            head_wins_by_pid[entry["protein_id"]] = wins
            if idx % 20 == 0 or idx == len(entries):
                _log(f"  Head: {idx}/{len(entries)}")
        t_head_elapsed = time.time() - t_head_start
        _log(f"  Head inference complete ({t_head_elapsed:.1f}s)")

        # ── Step 3: NMP scoring ────────────────────────────────────────
        _log(f"\n[3/4] Running NetMHCIIpan (mode={args.nmp_mode})...")
        from epitope_head.data.netmhciipan_runner import build_runner

        nmp_runner = build_runner(
            backend="standalone",
            binary_path=args.netmhciipan_bin,
            batch_size=batch_size,
            subprocess_timeout=args.nmp_timeout,
            max_lengths_per_call=max_lengths_per_call,
            n_workers=n_workers,
        )

        pep_lengths = list(range(args.min_k, args.max_k + 1))

        t_nmp_start = time.time()
        n_done = 0
        for chunk_start in range(0, len(entries), batch_size):
            chunk = entries[chunk_start:chunk_start + batch_size]
            chunk_pairs = [(e["protein_id"], e["protein_seq"]) for e in chunk]
            t_chunk = time.time()
            batch_out = nmp_runner.score_batch(chunk_pairs, args.allele, pep_lengths)
            chunk_elapsed = time.time() - t_chunk

            for pid, by_len in batch_out.items():
                wins: dict[tuple[int, int], float] = {}
                for pl, scores in by_len.items():
                    for s in scores:
                        wins[(s.pos, pl)] = s.el_rank  # 0-1 fraction
                nmp_wins_by_pid[pid] = wins

            # Per-chunk timing; in original mode each chunk = 1 protein
            for e in chunk:
                per_protein_nmp_timing.append({
                    "protein_id": e["protein_id"],
                    "sequence_length": len(e["protein_seq"]),
                    "chunk_seconds": round(chunk_elapsed, 2),
                })

            n_done += len(chunk)
            elapsed = time.time() - t_nmp_start
            rate = n_done / max(elapsed, 1e-6)
            eta = (len(entries) - n_done) / max(rate, 1e-6)
            pids_label = chunk[0]["protein_id"] if len(chunk) == 1 else (
                f"{chunk[0]['protein_id']}..{chunk[-1]['protein_id']}"
            )
            _log(f"  NMP [{n_done}/{len(entries)}] {pids_label}: "
                 f"chunk {chunk_elapsed:.1f}s | "
                 f"rate {rate * 60:.1f}/min, ETA {eta / 60:.1f}min")

        t_nmp_elapsed = time.time() - t_nmp_start
        _log(f"  NMP scoring complete ({t_nmp_elapsed:.1f}s total, "
             f"{t_nmp_elapsed / max(len(entries), 1):.2f}s/protein mean)")

        if args.windows_cache:
            _log(f"\n  Writing windows cache → {args.windows_cache}")
            t_save = time.time()
            _save_windows_cache(
                args.windows_cache,
                head_wins_by_pid,
                nmp_wins_by_pid,
                metadata={
                    "allele": args.allele,
                    "min_k": int(args.min_k),
                    "max_k": int(args.max_k),
                    "variant_id": args.variant_id,
                    "epitope_ckpt": args.epitope_ckpt,
                    "n_proteins": len(entries),
                    "nmp_mode": args.nmp_mode,
                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
            _log(f"  Cache write complete ({time.time() - t_save:.2f}s)")

    # ── Step 4: Build labels and compute three-tiered metrics ──────────
    # Q1 PRIMARY   : head uses real scores; NMP missing windows imputed with
    #                worst-case el_rank=1.0 (inverted -1.0, i.e. "definitely
    #                not a binder"). Both evaluated on the full ground-truth
    #                label set. This is the head-to-head benchmark answer.
    # Q2 CONDITIONAL: each predictor only on windows it actually scored.
    #                Tells us who's better WHEN THEY SUCCEED.
    # Q3 COVERAGE  : per-predictor coverage stats (scored/total windows,
    #                full-coverage protein count, partial/missing NMP).
    _log("\n[4/4] Computing metrics vs IEDB ground truth...")
    per_protein: list[dict] = []
    n_skipped_no_windows = 0

    # NMP missing windows get imputed to el_rank = 1.0 (100% rank, i.e.
    # NetMHCIIpan's own convention for "not a binder"); inverted to -1.0 to
    # match our higher-is-more-immunogenic convention.
    NMP_FILL_SCORE = -1.0
    RECALL_KS = tuple(args.recall_ks)
    PRECISION_KS = tuple(args.precision_ks)
    PRECISION_RECALL_TARGETS = tuple(args.precision_recall_targets)

    def _extract_scored(
        wins: dict[tuple[int, int], float],
        spans: torch.Tensor,
        invert: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pull scores for windows in `wins`, in canonical span order."""
        score_list: list[float] = []
        mask_list: list[bool] = []
        for i in range(spans.shape[0]):
            s = int(spans[i, 0].item())
            k = int(spans[i, 1].item()) - s
            if (s, k) in wins:
                v = wins[(s, k)]
                score_list.append(-v if invert else v)
                mask_list.append(True)
            else:
                mask_list.append(False)
        return (
            torch.tensor(score_list, dtype=torch.float32),
            torch.tensor(mask_list, dtype=torch.bool),
        )

    def _compute_metric_bundle(
        scores_t: torch.Tensor, labels: torch.Tensor,
    ) -> Optional[dict]:
        """AUC + AP + Recall@K + Precision views, or None if not evaluable."""
        if scores_t.numel() == 0:
            return None
        n_pos = int(labels.sum().item())
        n_neg = int((labels == 0).sum().item())
        if n_pos == 0 or n_neg == 0:
            return None
        out = {
            "auc": compute_auc(scores_t, labels),
            "ap": compute_ap(scores_t, labels),
        }
        for rk in RECALL_KS:
            out[f"recall_{rk}"] = compute_recall_at_k(scores_t, labels, rk)
        for pk in PRECISION_KS:
            out[f"precision_{pk}"] = _compute_precision_at_k(scores_t, labels, pk)
        for target in PRECISION_RECALL_TARGETS:
            target_key = _recall_target_key(target)
            out[f"precision_at_recall_{target_key}"] = (
                _compute_precision_at_target_recall(scores_t, labels, target)
            )
        return out

    def _impute_full(
        scored: torch.Tensor, mask: torch.Tensor, fill: float, n_total: int,
    ) -> torch.Tensor:
        """Embed `scored` into a length-n_total tensor, filling gaps with `fill`."""
        full = torch.full((n_total,), fill, dtype=torch.float32)
        full[mask] = scored
        return full

    def _serialize_score_distribution(
        scores_t: torch.Tensor, labels: torch.Tensor,
    ) -> dict:
        """Store per-label score samples for downstream histogram plotting."""
        pos_scores = scores_t[labels == 1].tolist()
        neg_scores = scores_t[labels == 0].tolist()
        return {
            "n_pos": len(pos_scores),
            "n_neg": len(neg_scores),
            "pos_scores": [float(x) for x in pos_scores],
            "neg_scores": [float(x) for x in neg_scores],
        }

    for entry in entries:
        pid = entry["protein_id"]
        seq = entry["protein_seq"]
        positives = entry["positives"]
        L = len(seq)

        spans = enumerate_candidate_windows(L, args.min_k, args.max_k)
        if spans.shape[0] == 0:
            n_skipped_no_windows += 1
            continue

        labels_full = build_window_labels(spans, positives)
        n_total_win = int(spans.shape[0])
        n_pos_total = int(labels_full.sum().item())

        head_wins = head_wins_by_pid.get(pid, {})
        nmp_wins = nmp_wins_by_pid.get(pid, {})

        head_scored, head_mask = _extract_scored(head_wins, spans, invert=False)
        nmp_scored, nmp_mask = _extract_scored(nmp_wins, spans, invert=True)
        head_n_scored = int(head_mask.sum().item())
        nmp_n_scored = int(nmp_mask.sum().item())

        # Head primary imputation: head is unbounded-logit; fill with
        # min-observed - 1 so missing windows rank strictly below any real
        # head prediction. Head should cover 100% in practice; this is
        # defensive for bug-detection only.
        if head_n_scored > 0:
            head_fill = float(head_scored.min().item()) - 1.0
        else:
            head_fill = -1e9

        head_full = _impute_full(head_scored, head_mask, head_fill, n_total_win)
        nmp_full = _impute_full(
            nmp_scored, nmp_mask, NMP_FILL_SCORE, n_total_win,
        )

        # Primary metrics (Q1): full window set, with imputation.
        # Even 0-coverage predictors stay in the primary benchmark via their
        # all-imputed score vector so complete failures are penalized.
        head_primary = _compute_metric_bundle(head_full, labels_full)
        nmp_primary = _compute_metric_bundle(nmp_full, labels_full)

        # Conditional metrics (Q2): scored subset only.
        head_cond = None
        if head_n_scored >= 10:
            head_cond = _compute_metric_bundle(
                head_scored, labels_full[head_mask],
            )
        nmp_cond = None
        if nmp_n_scored >= 10:
            nmp_cond = _compute_metric_bundle(
                nmp_scored, labels_full[nmp_mask],
            )

        # Coverage status (Q3).
        if nmp_n_scored == 0:
            nmp_status = "missing"
        elif nmp_n_scored == n_total_win:
            nmp_status = "complete"
        else:
            nmp_status = "partial"
        head_status = (
            "complete" if head_n_scored == n_total_win
            else ("missing" if head_n_scored == 0 else "partial")
        )

        row = {
            "protein_id": pid,
            "sequence_length": L,
            "n_positives_total": len(positives),
            "n_windows_total": n_total_win,
            "n_positives_in_windows": n_pos_total,
            "head_n_scored": head_n_scored,
            "nmp_n_scored": nmp_n_scored,
            "head_coverage": head_n_scored / n_total_win,
            "nmp_coverage": nmp_n_scored / n_total_win,
            "head_status": head_status,
            "nmp_status": nmp_status,
            "metrics": {
                "head_primary": head_primary,
                "nmp_primary": nmp_primary,
                "head_conditional": head_cond,
                "nmp_conditional": nmp_cond,
            },
            "score_distributions": {
                "head_primary": _serialize_score_distribution(
                    head_full, labels_full,
                ),
                "nmp_primary": _serialize_score_distribution(
                    nmp_full, labels_full,
                ),
                "head_conditional": _serialize_score_distribution(
                    head_scored, labels_full[head_mask],
                ),
                "nmp_conditional": _serialize_score_distribution(
                    nmp_scored, labels_full[nmp_mask],
                ),
            },
        }
        if args.near_miss_analysis:
            row["near_miss"] = _analyze_near_miss(spans, positives, head_full)

        # ── EL landscape extensions (doc/EL_new_evaluation.md) ────────
        # Re-express both predictors in higher-is-more-immunogenic form
        # once per protein, then pass the inverted-NMP dict to all helpers.
        if do_residue or do_iou or do_emd:
            head_wins_inv = head_wins  # head is already higher-is-better
            nmp_wins_inv = _invert_nmp_wins(nmp_wins)
            y_cover, y_density = _build_residue_labels(positives, L)

        if do_residue:
            # Per doc §5: identical multi-k aggregation for head and NMP.
            head_scores, head_mask = _aggregate_residue_scores(
                head_wins_inv, L, k_filter=None,
            )
            nmp_scores, nmp_mask = _aggregate_residue_scores(
                nmp_wins_inv, L, k_filter=None,
            )
            row["metrics"]["residue"] = {
                "head": _compute_residue_metrics(
                    head_scores, head_mask, y_cover, y_density,
                ),
                "nmp": _compute_residue_metrics(
                    nmp_scores, nmp_mask, y_cover, y_density,
                ),
            }

        if do_iou:
            row["metrics"]["iou_ladder"] = {
                "head": _compute_iou_ladder(
                    head_wins_inv, L, positives,
                    iou_thresholds, RECALL_KS,
                ),
                "nmp": _compute_iou_ladder(
                    nmp_wins_inv, L, positives,
                    iou_thresholds, RECALL_KS,
                ),
            }

        if do_emd:
            row["metrics"]["emd"] = {
                "head": _compute_emd(head_wins_inv, L, y_density),
                "nmp": _compute_emd(nmp_wins_inv, L, y_density),
            }

        per_protein.append(row)

    # ── Macro averages ─────────────────────────────────────────────────
    # Each (predictor, mode) averaged over its own evaluable proteins.

    def _macro_value(predictor: str, mode: str, metric_key: str) -> Optional[float]:
        vals = []
        for r in per_protein:
            bundle = r["metrics"].get(f"{predictor}_{mode}")
            if bundle is not None and bundle.get(metric_key) is not None:
                vals.append(bundle[metric_key])
        return sum(vals) / len(vals) if vals else None

    def _count_evaluable(predictor: str, mode: str) -> int:
        return sum(
            1 for r in per_protein
            if r["metrics"].get(f"{predictor}_{mode}") is not None
        )

    def _build_mode_macro(mode: str) -> dict:
        block: dict = {
            "n_head_evaluable": _count_evaluable("head", mode),
            "n_nmp_evaluable": _count_evaluable("nmp", mode),
        }
        for name in ("head", "nmp"):
            keys = (
                ["auc", "ap"]
                + [f"recall_{rk}" for rk in RECALL_KS]
                + [f"precision_{pk}" for pk in PRECISION_KS]
                + [
                    f"precision_at_recall_{_recall_target_key(target)}"
                    for target in PRECISION_RECALL_TARGETS
                ]
            )
            block[name] = {
                f"pp_{k}": _macro_value(name, mode, k) for k in keys
            }
        return block

    n_pp = max(len(per_protein), 1)
    head_cov_vals = [r["head_coverage"] for r in per_protein]
    nmp_cov_vals = [r["nmp_coverage"] for r in per_protein]
    coverage_block = {
        "head_mean_coverage": sum(head_cov_vals) / n_pp if head_cov_vals else None,
        "nmp_mean_coverage": sum(nmp_cov_vals) / n_pp if nmp_cov_vals else None,
        "n_head_complete": sum(1 for r in per_protein if r["head_status"] == "complete"),
        "n_head_partial": sum(1 for r in per_protein if r["head_status"] == "partial"),
        "n_head_missing": sum(1 for r in per_protein if r["head_status"] == "missing"),
        "n_nmp_complete": sum(1 for r in per_protein if r["nmp_status"] == "complete"),
        "n_nmp_partial": sum(1 for r in per_protein if r["nmp_status"] == "partial"),
        "n_nmp_missing": sum(1 for r in per_protein if r["nmp_status"] == "missing"),
    }

    macro: dict = {
        "n_total": len(entries),
        "n_skipped_no_windows": n_skipped_no_windows,
        "primary": _build_mode_macro("primary"),
        "conditional": _build_mode_macro("conditional"),
        "coverage": coverage_block,
    }

    # ── EL landscape macro blocks ──────────────────────────────────────
    def _macro_mean(values: list[Optional[float]]) -> Optional[float]:
        vs = [float(v) for v in values if v is not None]
        return sum(vs) / len(vs) if vs else None

    if do_residue:
        residue_block: dict = {
            "n_head_evaluable": sum(
                1 for r in per_protein
                if r["metrics"].get("residue", {}).get("head") is not None
            ),
            "n_nmp_evaluable": sum(
                1 for r in per_protein
                if r["metrics"].get("residue", {}).get("nmp") is not None
            ),
        }
        for name in ("head", "nmp"):
            residue_block[name] = {}
            for key in ("auc", "ap", "pearson", "spearman"):
                path = ("metrics", "residue", name, key)
                residue_block[name][f"pp_{key}"] = _macro_mean(
                    _collect_pp(per_protein, path),
                )
        macro["residue"] = residue_block

    if do_iou:
        iou_block: dict = {
            "thresholds": iou_thresholds,
            "n_head_evaluable": sum(
                1 for r in per_protein
                if r["metrics"].get("iou_ladder", {}).get("head") is not None
            ),
            "n_nmp_evaluable": sum(
                1 for r in per_protein
                if r["metrics"].get("iou_ladder", {}).get("nmp") is not None
            ),
        }
        tier_keys = [_iou_threshold_key(t) for t in iou_thresholds]
        tier_metric_keys = ["ap", "auc"] + [f"recall_{rk}" for rk in RECALL_KS]
        for name in ("head", "nmp"):
            iou_block[name] = {}
            for tier in tier_keys:
                tier_out = {}
                for mk in tier_metric_keys:
                    path = ("metrics", "iou_ladder", name, "tiers", tier, mk)
                    tier_out[f"pp_{mk}"] = _macro_mean(
                        _collect_pp(per_protein, path),
                    )
                iou_block[name][tier] = tier_out
        macro["iou_ladder"] = iou_block

    if do_emd:
        emd_block: dict = {
            "n_head_evaluable": sum(
                1 for r in per_protein
                if r["metrics"].get("emd", {}).get("head") is not None
            ),
            "n_nmp_evaluable": sum(
                1 for r in per_protein
                if r["metrics"].get("emd", {}).get("nmp") is not None
            ),
        }
        for name in ("head", "nmp"):
            emd_block[name] = {
                "pp_emd_norm": _macro_mean(
                    _collect_pp(per_protein, ("metrics", "emd", name, "emd_norm")),
                ),
                "pp_similarity": _macro_mean(
                    _collect_pp(per_protein, ("metrics", "emd", name, "similarity")),
                ),
            }
        macro["emd"] = emd_block

    # ── Statistics: bootstrap 95% CI + paired Wilcoxon (head vs NMP) ──
    # Applied to headline metrics where both predictors produce per-protein
    # scalars. Skipped when the suite excludes the relevant family.
    statistics: dict = {"bootstrap_95ci": {}, "wilcoxon_head_vs_nmp": {}}

    headline_specs: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
        # (label, head_path, nmp_path)
        ("primary_ap",
         ("metrics", "head_primary", "ap"),
         ("metrics", "nmp_primary", "ap")),
        ("primary_auc",
         ("metrics", "head_primary", "auc"),
         ("metrics", "nmp_primary", "auc")),
    ]
    if do_residue:
        for metric in ("auc", "ap", "pearson", "spearman"):
            headline_specs.append((
                f"residue_{metric}",
                ("metrics", "residue", "head", metric),
                ("metrics", "residue", "nmp", metric),
            ))
    if do_iou:
        for tier in [_iou_threshold_key(t) for t in iou_thresholds]:
            for metric in ("ap", "auc"):
                headline_specs.append((
                    f"iou_{tier}_{metric}",
                    ("metrics", "iou_ladder", "head", "tiers", tier, metric),
                    ("metrics", "iou_ladder", "nmp", "tiers", tier, metric),
                ))
    if do_emd:
        for metric in ("similarity", "emd_norm"):
            headline_specs.append((
                f"emd_{metric}",
                ("metrics", "emd", "head", metric),
                ("metrics", "emd", "nmp", metric),
            ))

    for label, head_path, nmp_path in headline_specs:
        head_vals = _collect_pp(per_protein, head_path)
        nmp_vals = _collect_pp(per_protein, nmp_path)
        statistics["bootstrap_95ci"][label] = {
            "head": _bootstrap_ci(head_vals, args.bootstrap_n, args.bootstrap_seed),
            "nmp": _bootstrap_ci(nmp_vals, args.bootstrap_n, args.bootstrap_seed + 1),
        }
        statistics["wilcoxon_head_vs_nmp"][label] = _paired_wilcoxon(head_vals, nmp_vals)

    total_elapsed = time.time() - t0

    output = {
        "config": {
            "protein_samples_parquet": args.protein_samples_parquet,
            "test_ids": args.test_ids,
            "epitope_ckpt": args.epitope_ckpt,
            "allele": args.allele,
            "nmp_mode": args.nmp_mode,
            "variant_id": args.variant_id,
            "min_k": args.min_k,
            "max_k": args.max_k,
            "recall_ks": args.recall_ks,
            "precision_ks": args.precision_ks,
            "precision_recall_targets": args.precision_recall_targets,
            "nmp_batch_size": batch_size,
            "nmp_max_lengths_per_call": max_lengths_per_call,
            "nmp_workers": n_workers,
            "metric_suite": args.metric_suite,
            "iou_thresholds": iou_thresholds,
            "bootstrap_n": args.bootstrap_n,
            "bootstrap_seed": args.bootstrap_seed,
            "windows_cache": args.windows_cache,
            "resume_from_cache": args.resume_from_cache,
            "reuse_nmp_from_cache": args.reuse_nmp_from_cache,
        },
        "timing": {
            "total_seconds": round(total_elapsed, 1),
            "head_seconds": round(t_head_elapsed, 1),
            "nmp_seconds": round(t_nmp_elapsed, 1),
            "nmp_mean_per_protein_seconds": round(
                t_nmp_elapsed / max(len(entries), 1), 2,
            ),
        },
        "macro": macro,
        "statistics": statistics,
        "per_protein": per_protein,
        "nmp_timing_per_protein": per_protein_nmp_timing,
    }
    if args.near_miss_analysis:
        nmp_better_rows = [
            row for row in per_protein
            if _select_near_miss_group(row) == "nmp_better"
        ]
        head_better_rows = [
            row for row in per_protein
            if _select_near_miss_group(row) == "head_better"
        ]
        output["near_miss_summary"] = {
            "enabled": True,
            "group_metric": "exact_ap",
            "groups": {
                "nmp_better": _summarize_near_miss_group(nmp_better_rows),
                "head_better": _summarize_near_miss_group(head_better_rows),
                "all": _summarize_near_miss_group(per_protein),
            },
        }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────
    _log(f"\n{'=' * 60}")
    _log("Results (macro-averaged per protein):")
    _log(f"  n_total                  = {macro['n_total']}")
    _log(f"  n_skipped_no_windows     = {macro['n_skipped_no_windows']}")

    _log("\n  [Coverage — Q3: who is more stable/complete?]")
    cov = macro["coverage"]
    if cov["head_mean_coverage"] is not None:
        _log(f"    head mean coverage    = {cov['head_mean_coverage']:.4f}")
    if cov["nmp_mean_coverage"] is not None:
        _log(f"    nmp  mean coverage    = {cov['nmp_mean_coverage']:.4f}")
    _log(f"    head complete/partial/missing = "
         f"{cov['n_head_complete']}/{cov['n_head_partial']}/{cov['n_head_missing']}")
    _log(f"    nmp  complete/partial/missing = "
         f"{cov['n_nmp_complete']}/{cov['n_nmp_partial']}/{cov['n_nmp_missing']}")

    for mode, heading in (
        ("primary",
         "[Primary — Q1: who wins on the same real test set? "
         "NMP missing=worst]"),
        ("conditional",
         "[Conditional — Q2: who wins when they successfully score?]"),
    ):
        _log(f"\n  {heading}")
        block = macro[mode]
        for name in ("head", "nmp"):
            n_eval = block[f"n_{name}_evaluable"]
            _log(f"    [{name.upper()}] evaluable = {n_eval}/{macro['n_total']}")
            metric_keys = (
                ["pp_auc", "pp_ap"]
                + [f"pp_recall_{rk}" for rk in args.recall_ks]
                + [f"pp_precision_{pk}" for pk in args.precision_ks]
                + [
                    f"pp_precision_at_recall_{_recall_target_key(target)}"
                    for target in args.precision_recall_targets
                ]
            )
            for key in metric_keys:
                v = block[name].get(key)
                if v is not None:
                    _log(f"      {key:<14} = {v:.4f}")
                else:
                    _log(f"      {key:<14} = N/A")

    # ── EL-landscape summary ────────────────────────────────────────
    if do_residue:
        _log("\n  [Residue (M2a) — per-residue landscape, k∈[min_k,max_k]]")
        block = macro["residue"]
        for name in ("head", "nmp"):
            n_eval = block[f"n_{name}_evaluable"]
            cells = []
            for key in ("pp_auc", "pp_ap", "pp_pearson", "pp_spearman"):
                v = block[name].get(key)
                cells.append(
                    f"{key[3:]}={v:.4f}" if v is not None else f"{key[3:]}=N/A"
                )
            _log(f"    [{name.upper():4s}] n={n_eval}/{macro['n_total']}  "
                 + "  ".join(cells))

    if do_iou:
        _log("\n  [IoU ladder (M6) — window-level, k=15]")
        block = macro["iou_ladder"]
        _log(f"    n_head_evaluable = {block['n_head_evaluable']}/{macro['n_total']}")
        _log(f"    n_nmp_evaluable  = {block['n_nmp_evaluable']}/{macro['n_total']}")
        for name in ("head", "nmp"):
            _log(f"    [{name.upper()}]")
            for t in iou_thresholds:
                tier = _iou_threshold_key(t)
                cells = []
                for key in (["pp_ap", "pp_auc"]
                            + [f"pp_recall_{rk}" for rk in args.recall_ks]):
                    v = block[name][tier].get(key)
                    cells.append(
                        f"{key[3:]}={v:.4f}" if v is not None else f"{key[3:]}=N/A"
                    )
                _log(f"      {tier:12s}  " + "  ".join(cells))

    if do_emd:
        _log("\n  [EMD (M3) — residue density distance]")
        block = macro["emd"]
        for name in ("head", "nmp"):
            sim = block[name].get("pp_similarity")
            emd = block[name].get("pp_emd_norm")
            sim_s = f"{sim:.4f}" if sim is not None else "N/A"
            emd_s = f"{emd:.4f}" if emd is not None else "N/A"
            n_eval = block[f"n_{name}_evaluable"]
            _log(f"    [{name.upper():4s}] n={n_eval}/{macro['n_total']}  "
                 f"similarity={sim_s}  emd_norm={emd_s}")

    if statistics["wilcoxon_head_vs_nmp"]:
        _log("\n  [Statistics — paired Wilcoxon (head vs NMP, two-sided)]")
        for label, wx in statistics["wilcoxon_head_vs_nmp"].items():
            if wx is None:
                continue
            _log(f"    {label:<36s} n={wx['n_pairs']:3d}  "
                 f"median_Δ={wx['median_diff']:+.4f}  "
                 f"p={wx['pvalue']:.2e}  r={wx['effect_r']:+.3f}")

    _log("")
    _log(f"  Head wall time  = {t_head_elapsed:.1f}s")
    _log(f"  NMP  wall time  = {t_nmp_elapsed:.1f}s "
         f"({t_nmp_elapsed / 60:.1f}min, "
         f"{t_nmp_elapsed / max(len(entries), 1):.2f}s/protein)")
    if args.near_miss_analysis:
        near = output["near_miss_summary"]["groups"]
        _log("  Near-miss groups:")
        _log(f"    nmp_better proteins = {near['nmp_better']['n_proteins']}")
        _log(f"    head_better proteins = {near['head_better']['n_proteins']}")
        _log(f"    all proteins = {near['all']['n_proteins']}")
    _log(f"  Total wall time = {total_elapsed:.1f}s "
         f"({total_elapsed / 60:.1f}min)")
    _log(f"\nSaved to {args.output_json}")
    _log("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
