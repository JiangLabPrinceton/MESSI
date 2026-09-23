"""Per-protein full-window-scan evaluation metrics.

Scores ALL candidate k-mer windows within each validation protein,
compares against ground-truth positive spans, and computes:
  - Per-protein AUC (macro-averaged)
  - Per-protein Average Precision (macro-averaged)
  - Per-protein Recall@K (macro-averaged)
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def enumerate_candidate_windows(
    protein_length: int,
    min_k: int = 12,
    max_k: int = 25,
) -> torch.Tensor:
    """Enumerate all valid (start, end) windows for k in [min_k, max_k].

    Returns:
        spans: [N, 2] int64 tensor, (start_0b, end_0b) half-open.
    """
    spans = []
    for k in range(min_k, max_k + 1):
        for start in range(protein_length - k + 1):
            spans.append((start, start + k))
    if not spans:
        return torch.zeros(0, 2, dtype=torch.long)
    return torch.tensor(spans, dtype=torch.long)


def build_window_labels(
    candidate_spans: torch.Tensor,
    positives: list[dict],
) -> torch.Tensor:
    """Label each candidate window: 1 if exact match with a positive, 0 otherwise."""
    pos_set = {(p["start_0b"], p["end_0b"]) for p in positives}
    labels = torch.zeros(candidate_spans.shape[0], dtype=torch.long)
    for i in range(candidate_spans.shape[0]):
        s = candidate_spans[i, 0].item()
        e = candidate_spans[i, 1].item()
        if (s, e) in pos_set:
            labels[i] = 1
    return labels


# ── Metric functions ────────────────────────────────────────────────────────

def compute_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    """Pairwise AUC: P(score_pos > score_neg) + 0.5 * P(tie)."""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return None
    cmp = (pos_scores.unsqueeze(1) > neg_scores.unsqueeze(0)).float()
    tie = (pos_scores.unsqueeze(1) == neg_scores.unsqueeze(0)).float()
    return float((cmp + 0.5 * tie).mean().item())


def compute_ap(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    """Average Precision (area under Precision-Recall curve)."""
    n_pos = labels.sum().item()
    if n_pos == 0:
        return None
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices].float()
    tp_cumsum = torch.cumsum(sorted_labels, dim=0)
    ranks = torch.arange(1, len(sorted_labels) + 1, device=scores.device).float()
    precisions = tp_cumsum / ranks
    ap = (precisions * sorted_labels).sum().item() / n_pos
    return ap


def compute_recall_at_k(
    scores: torch.Tensor, labels: torch.Tensor, k: int,
) -> float | None:
    """Recall@K: fraction of positives in top-K scored windows."""
    n_pos = labels.sum().item()
    if n_pos == 0:
        return None
    sorted_indices = torch.argsort(scores, descending=True)
    top_k = min(k, len(sorted_indices))
    top_k_labels = labels[sorted_indices[:top_k]]
    return float(top_k_labels.sum().item() / n_pos)


# ── Single-protein evaluation ──────────────────────────────────────────────

SPAN_SCORE_BATCH = 4096  # score windows in batches to bound memory


@torch.no_grad()
def evaluate_single_protein(
    model: nn.Module,
    tokenizer: Callable,
    protein_id: str,
    protein_seq: str,
    positives: list[dict],
    min_k: int,
    max_k: int,
    device: torch.device,
    recall_ks: tuple[int, ...] = (50, 100),
    context_len: int = 1022,
) -> dict | None:
    """Full-window-scan evaluation for a single protein.

    Returns dict of metrics, or None if protein cannot be evaluated
    (too long for single-chunk, no positives, etc.).
    """
    L = len(protein_seq)

    if L > context_len:
        return None

    if not positives:
        return None

    all_spans = enumerate_candidate_windows(L, min_k, max_k)
    if all_spans.shape[0] == 0:
        return None

    labels = build_window_labels(all_spans, positives)

    if labels.sum() == 0 or labels.sum() == labels.numel():
        return None

    # Encode protein once
    tok_out = tokenizer([protein_seq])
    token_ids = tok_out["token_ids"].to(device)
    attention_mask = tok_out["attention_mask"].to(device)

    G, lengths = model.encode_and_project(token_ids, attention_mask)
    G_0 = G[0]
    chunk_len = int(lengths[0].item())

    # Score all windows (batched to bound memory)
    all_spans_dev = all_spans.to(device)
    allele_idx = torch.zeros(all_spans.shape[0], dtype=torch.long, device=device)

    logit_parts: list[torch.Tensor] = []
    for i in range(0, all_spans.shape[0], SPAN_SCORE_BATCH):
        j = min(i + SPAN_SCORE_BATCH, all_spans.shape[0])
        logit_parts.append(
            model.score_spans(G_0, chunk_len, all_spans_dev[i:j], allele_idx[i:j])
        )
    scores = torch.cat(logit_parts)
    labels_dev = labels.to(device)

    result: dict = {
        "protein_id": protein_id,
        "n_windows": int(all_spans.shape[0]),
        "n_pos": int(labels.sum().item()),
        "auc": compute_auc(scores, labels_dev),
        "ap": compute_ap(scores, labels_dev),
    }
    for k in recall_ks:
        result[f"recall_{k}"] = compute_recall_at_k(scores, labels_dev, k)

    return result


# ── Whole-val-set evaluation ───────────────────────────────────────────────

@torch.no_grad()
def full_val_eval(
    model: nn.Module,
    tokenizer: Callable,
    val_entries: list,
    min_k: int,
    max_k: int,
    device: torch.device,
    recall_ks: tuple[int, ...] = (50, 100),
    context_len: int = 1022,
) -> dict:
    """Run per-protein full-window-scan evaluation on the val set.

    Returns a dict with macro-averaged metrics (pp_auc, pp_ap, pp_recall_K, …).
    """
    model.eval()

    results: list[dict] = []
    n_skipped = 0

    for entry in val_entries:
        r = evaluate_single_protein(
            model=model,
            tokenizer=tokenizer,
            protein_id=entry.protein_id,
            protein_seq=entry.protein_seq,
            positives=entry.positives,
            min_k=min_k,
            max_k=max_k,
            device=device,
            recall_ks=recall_ks,
            context_len=context_len,
        )
        if r is not None:
            results.append(r)
        else:
            n_skipped += 1

    if not results:
        logger.warning("full_val_eval: no proteins evaluated")
        metrics: dict = {
            "pp_auc": None,
            "pp_ap": None,
            "pp_n_proteins": 0,
            "pp_n_skipped": n_skipped,
        }
        for k in recall_ks:
            metrics[f"pp_recall_{k}"] = None
        return metrics

    auc_vals = [r["auc"] for r in results if r["auc"] is not None]
    ap_vals = [r["ap"] for r in results if r["ap"] is not None]

    metrics = {
        "pp_auc": sum(auc_vals) / len(auc_vals) if auc_vals else None,
        "pp_ap": sum(ap_vals) / len(ap_vals) if ap_vals else None,
        "pp_n_proteins": len(results),
        "pp_n_skipped": n_skipped,
    }
    for k in recall_ks:
        key = f"recall_{k}"
        vals = [r[key] for r in results if r.get(key) is not None]
        metrics[f"pp_{key}"] = sum(vals) / len(vals) if vals else None

    logger.info(
        "Full-scan eval: %d proteins (skipped %d), pp_auc=%.4f, pp_ap=%.4f",
        len(results), n_skipped,
        metrics["pp_auc"] or 0.0, metrics["pp_ap"] or 0.0,
    )

    return metrics
