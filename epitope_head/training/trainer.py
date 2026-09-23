"""Trainer loop with sanity metrics, checkpointing, and run metadata.

Implements PLAN.md Tasks E5 (trainer loop), E6 (checkpointing/logging), E7 (smoke-ready).
  - Train/val step plumbing with optimizer, scheduler, grad clipping
  - Frozen encoder guard: encoder params excluded from optimizer
  - NaN/Inf guard with fail-fast
  - Sanity metrics: mean_pos_logit, mean_neg_logit, logit_gap, per_protein_auc
  - Epoch checkpoints + best checkpoint by monitor_metric
  - JSONL train/val logs with stable key schema
  - Resolved config snapshot + run summary
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from epitope_head.configs import validate_himp_train_blocks
from epitope_head.training.eval_metrics import full_val_eval
from epitope_head.training.losses import compute_loss, exact_margin_loss, residue_pairwise_margin_loss
from epitope_head.training.span_geom import max_iou_per_window
from epitope_head.training.negatives import sample_negatives
from epitope_head.training.registry import (
    append_registry_row,
    build_registry_row,
    compute_protocol_signature,
    generate_run_id,
)
from epitope_head.training.residue_supervision import (
    aggregate_window_logits_to_residues,
    build_residue_labels,
    enumerate_all_residue_windows,
)

logger = logging.getLogger(__name__)


# ── Sanity Metrics ──────────────────────────────────────────────────────────

@dataclass
class StepMetrics:
    """Metrics from a single train/val step (one chunk)."""
    loss_total: float = 0.0
    loss_intra: float = 0.0
    loss_mp: float = 0.0
    loss_smooth: float = 0.0
    loss_margin: float = 0.0
    # HIMP4: residue ranking loss decomposition
    loss_residue: float = 0.0
    n_residue_pairs: int = 0
    residue_skipped_chunks: int = 0
    n_residue_chunks: int = 0
    # Wave-3: window IoU-ranking loss decomposition
    loss_iou_rank: float = 0.0
    mean_pos_logit: float = 0.0
    mean_neg_logit: float = 0.0
    logit_gap: float = 0.0
    per_protein_auc: float | None = None
    n_pos: int = 0
    n_neg: int = 0

    def to_dict(self) -> dict:
        return {
            "loss_total": self.loss_total,
            "loss_intra": self.loss_intra,
            "loss_mp": self.loss_mp,
            "loss_smooth": self.loss_smooth,
            "loss_margin": self.loss_margin,
            "loss_residue": self.loss_residue,
            "n_residue_pairs": self.n_residue_pairs,
            "residue_skipped_chunks": self.residue_skipped_chunks,
            "n_residue_chunks": self.n_residue_chunks,
            "loss_iou_rank": self.loss_iou_rank,
            "mean_pos_logit": self.mean_pos_logit,
            "mean_neg_logit": self.mean_neg_logit,
            "logit_gap": self.logit_gap,
            "per_protein_auc": self.per_protein_auc,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
        }


def _compute_binary_auc(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
) -> float | None:
    """Compute AUC from positive/negative logits; returns None if undefined."""
    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return None
    # Pairwise AUC: P(score_pos > score_neg) + 0.5 * P(tie)
    cmp = (pos_logits.unsqueeze(1) > neg_logits.unsqueeze(0)).float()
    tie = (pos_logits.unsqueeze(1) == neg_logits.unsqueeze(0)).float()
    return float((cmp + 0.5 * tie).mean().item())


def compute_sanity_metrics(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    loss_dict: dict[str, torch.Tensor],
) -> StepMetrics:
    """Compute sanity metrics from logits and loss terms."""
    mean_pos = pos_logits.mean().item() if pos_logits.numel() > 0 else 0.0
    mean_neg = neg_logits.mean().item() if neg_logits.numel() > 0 else 0.0
    auc = _compute_binary_auc(pos_logits, neg_logits)
    return StepMetrics(
        loss_total=loss_dict["loss_total"].item(),
        loss_intra=loss_dict["loss_intra"].item(),
        loss_mp=loss_dict["loss_mp"].item(),
        loss_smooth=loss_dict["loss_smooth"].item(),
        loss_margin=loss_dict["loss_margin"].item(),
        mean_pos_logit=mean_pos,
        mean_neg_logit=mean_neg,
        logit_gap=mean_pos - mean_neg,
        per_protein_auc=auc,
        n_pos=pos_logits.numel(),
        n_neg=neg_logits.numel(),
    )


def _diagnostic_loss_dict(cat_pos, cat_neg, loss_cfg):
    """Recompute span loss terms over concatenated logits, for logging only.

    In ``iou_rank_only`` the span InfoNCE/margin terms are 0 by definition and
    the iou-rank term needs per-chunk window inputs unavailable on these
    concatenated logits, so report zeros instead of calling ``compute_loss``
    (which would hit its training-time fail-fast). The real backprop loss is
    ``avg_loss``; ``loss_iou_rank`` is attached separately from ``residue_stats``.
    """
    if loss_cfg.get("objective_mode") == "iou_rank_only":
        z = torch.zeros(())
        return {k: z for k in (
            "loss_total", "loss_intra", "loss_mp", "loss_smooth", "loss_margin")}
    # Strip dual-head exact keys — compute_loss does not accept them; the exact
    # term is a trainer-level addition (in _forward_union), not part of the span
    # objective. _forward_union pops them from a local copy, so the original
    # loss_cfg reaching this diagnostic recompute still carries them.
    cfg = {k: v for k, v in loss_cfg.items()
           if k not in ("lambda_exact", "exact_margin_m", "exact_hard_topk")}
    return compute_loss(cat_pos, cat_neg, **cfg)


def aggregate_epoch_metrics(step_metrics_list: list[StepMetrics]) -> dict:
    """Aggregate step metrics into epoch-level summary.

    Excludes empty-positive chunks (n_pos == 0) to avoid diluting metrics.
    """
    # Filter out steps with no positives (empty chunks contribute no gradient)
    step_metrics_list = [m for m in step_metrics_list if m.n_pos > 0]
    if not step_metrics_list:
        return {}
    n = len(step_metrics_list)
    auc_values = [m.per_protein_auc for m in step_metrics_list if m.per_protein_auc is not None]
    # HIMP4: residue stats — average loss_residue over steps that contributed
    # (i.e. had at least one non-skipped residue chunk), so the headline
    # number is not diluted by chunks that legitimately skipped due to far-bg
    # shortage.
    n_residue_chunks_total = sum(m.n_residue_chunks for m in step_metrics_list)
    residue_skipped_total = sum(m.residue_skipped_chunks for m in step_metrics_list)
    residue_pairs_total = sum(m.n_residue_pairs for m in step_metrics_list)
    contributing_steps = [m for m in step_metrics_list if m.loss_residue != 0.0]
    if contributing_steps:
        residue_loss_avg = sum(m.loss_residue for m in contributing_steps) / len(contributing_steps)
    else:
        residue_loss_avg = 0.0
    return {
        "loss_total": sum(m.loss_total for m in step_metrics_list) / n,
        "loss_intra": sum(m.loss_intra for m in step_metrics_list) / n,
        "loss_mp": sum(m.loss_mp for m in step_metrics_list) / n,
        "loss_smooth": sum(m.loss_smooth for m in step_metrics_list) / n,
        "loss_margin": sum(m.loss_margin for m in step_metrics_list) / n,
        "loss_residue": residue_loss_avg,
        "n_residue_pairs": residue_pairs_total,
        "n_residue_chunks": n_residue_chunks_total,
        "residue_skipped_chunks": residue_skipped_total,
        "loss_iou_rank": (
            sum(m.loss_iou_rank for m in step_metrics_list if m.loss_iou_rank != 0.0)
            / max(len([m for m in step_metrics_list if m.loss_iou_rank != 0.0]), 1)
        ),
        "mean_pos_logit": sum(m.mean_pos_logit for m in step_metrics_list) / n,
        "mean_neg_logit": sum(m.mean_neg_logit for m in step_metrics_list) / n,
        "logit_gap": sum(m.logit_gap for m in step_metrics_list) / n,
        "per_protein_auc": (sum(auc_values) / len(auc_values)) if auc_values else None,
        "total_pos": sum(m.n_pos for m in step_metrics_list),
        "total_neg": sum(m.n_neg for m in step_metrics_list),
        "n_steps": n,
    }


# ── NaN Guard ───────────────────────────────────────────────────────────────

class NaNDetected(RuntimeError):
    """Raised when NaN or Inf is detected in loss or logits."""
    pass


def nan_guard(tensor: torch.Tensor, name: str = "tensor"):
    """Check for NaN/Inf and raise immediately."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise NaNDetected(f"NaN/Inf detected in {name}: {tensor}")


# ── Optimizer Builder ───────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build optimizer with only learnable (non-frozen) parameters.

    Splits params into decay (Linear.weight only) and no-decay groups
    (bias, LayerNorm, Embedding, learnable Parameters) to avoid
    regularizing normalization/embedding/bias terms.
    """
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Only apply weight decay to Linear weight matrices
        if p.dim() >= 2 and "embedding" not in name.lower():
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr)
    elif optimizer_name == "adam":
        return torch.optim.Adam(param_groups, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    warmup_steps: int = 500,
    max_epochs: int = 50,
    steps_per_epoch: int = 100,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build LR scheduler with optional warmup."""
    total_steps = max_epochs * steps_per_epoch
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps),
        )
    elif scheduler_name == "constant":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def normalize_loss_cfg(loss_cfg: dict) -> dict:
    """Normalize train config loss keys to compute_loss(...) signature keys."""
    normalized = dict(loss_cfg)
    if "tau_mp" in normalized:
        if "T_mp" in normalized and normalized["T_mp"] != normalized["tau_mp"]:
            raise ValueError("Both 'tau_mp' and 'T_mp' provided with different values")
        normalized["T_mp"] = normalized.pop("tau_mp")

    required = {"tau", "T_mp", "lambda_mp", "lambda_smooth"}
    optional = {"objective_mode", "margin_m", "hard_topk", "lambda_margin",
                "lambda_iou_rank", "iou_rank_margin", "iou_rank_min_gap",
                "lambda_exact", "exact_margin_m", "exact_hard_topk"}
    missing = required - set(normalized.keys())
    if missing:
        raise ValueError(f"Loss config missing required keys after normalization: {sorted(missing)}")

    unknown = set(normalized.keys()) - (required | optional)
    if unknown:
        raise ValueError(f"Loss config has unknown keys: {sorted(unknown)}")

    return normalized


# ── Train / Val Step ────────────────────────────────────────────────────────

def prepare_chunk_spans(
    batch: dict,
    batch_idx: int,
    neg_ratio: int = 7,
    hard_negative_fraction: float = 0.3,
    hard_neg_max_overlap_ratio: float = 0.8,
    hard_neg_offset_range: int = 5,
    neg_length_sampling: str = "match_positive",
    min_k: int = 12,
    max_k: int = 25,
    rng: np.random.RandomState | None = None,
    near_positive_cfg: dict | None = None,
    residue_cfg: dict | None = None,
    chunk_central_margin: int = 0,
) -> tuple[
    list[torch.Tensor], list[torch.Tensor],
    list[torch.Tensor], list[torch.Tensor],
    list[dict],
]:
    """Prepare positive/negative spans for each chunk in batch.

    Converts protein-global positives to chunk-local coordinates,
    filters to spans within chunk bounds, samples negatives.

    Args:
        ... (existing args unchanged) ...
        near_positive_cfg: optional HIMP1 config; when provided,
            ``extras[i]["neg_weights"]`` and ``extras[i]["neg_relations"]``
            are populated parallel to ``neg_spans_list[i]``.
        residue_cfg: optional HIMP2/3 config; when ``residue_cfg["enabled"]``,
            ``extras[i]["residue_meta"]`` (label / ambiguous / far_bg /
            central masks) and ``extras[i]["residue_extra_windows"]`` (only
            non-empty when ``window_mode=all``) are populated.
        chunk_central_margin: residues to exclude on each side when building
            ``central_mask`` for residue supervision.

    Returns:
        Tuple ``(pos_spans, neg_spans, allele_pos, allele_neg, extras)`` where
        ``extras`` is a list with one dict per chunk. Each dict carries:
          - ``neg_weights`` (Tensor[N] | None)
          - ``neg_relations`` (list[SpanRelation] | None)
          - ``residue_meta`` (dict | None) — keys: label, ambiguous_mask,
            far_bg_mask, central_mask
          - ``residue_extra_windows`` (Tensor[W, 2] | None) — extra windows
            for ``window_mode=all``; None for ``sampled``.
    """
    B = len(batch["protein_ids"])
    pos_spans_list = []
    neg_spans_list = []
    allele_pos_list = []
    allele_neg_list = []
    extras_list: list[dict] = []
    _disrupted_stats = {"used": 0, "hard_target": 0, "total_neg": 0}

    residue_enabled = bool(residue_cfg and residue_cfg.get("enabled", False))
    window_mode = (residue_cfg or {}).get("window_mode", "sampled")
    max_windows = int((residue_cfg or {}).get("max_windows_per_chunk", 1024))

    # HIMP1 review fix 2: respect apply_to_span_negatives. When the user
    # disables span-side weighting (e.g. for ablating residue supervision in
    # isolation), the negative sampler must not classify or weight spans.
    # apply_to_residue_labels is honored separately inside build_residue_labels.
    np_cfg_for_negatives = (
        near_positive_cfg
        if (
            near_positive_cfg is not None
            and bool(near_positive_cfg.get("apply_to_span_negatives", True))
        )
        else None
    )

    for i in range(B):
        chunk_start = batch["chunk_starts"][i].item()
        chunk_end = batch["chunk_ends"][i].item()
        chunk_len = chunk_end - chunk_start
        positives = batch["positives"][i]

        # Filter positives to those within chunk bounds (chunk-local coords)
        local_positives = []
        for p in positives:
            s, e = p["start_0b"], p["end_0b"]
            if s >= chunk_start and e <= chunk_end:
                local_positives.append({
                    "start_0b": s - chunk_start,
                    "end_0b": e - chunk_start,
                    "pep_len": p["pep_len"],
                })

        # Convert disrupted spans to chunk-local coords for hard negative priority
        local_disrupted = None
        raw_disrupted = batch.get("disrupted_spans", [None] * B)[i]
        if raw_disrupted:
            local_disrupted = []
            for ds in raw_disrupted:
                s, e = ds["start_0b"], ds["end_0b"]
                if s >= chunk_start and e <= chunk_end:
                    local_disrupted.append({
                        "start_0b": s - chunk_start,
                        "end_0b": e - chunk_start,
                        "pep_len": ds["pep_len"],
                    })
            if not local_disrupted:
                local_disrupted = None

        chunk_extras: dict = {
            "neg_weights": None,
            "neg_relations": None,
            "residue_meta": None,
            "residue_extra_windows": None,
        }
        if local_positives:
            pos_spans = torch.tensor(
                [[p["start_0b"], p["end_0b"]] for p in local_positives],
                dtype=torch.long,
            )
            # Sample negatives in chunk-local space
            negs = sample_negatives(
                protein_length=chunk_len,
                positives=local_positives,
                neg_ratio=neg_ratio,
                hard_negative_fraction=hard_negative_fraction,
                hard_neg_max_overlap_ratio=hard_neg_max_overlap_ratio,
                hard_neg_offset_range=hard_neg_offset_range,
                neg_length_sampling=neg_length_sampling,
                min_k=min_k,
                max_k=max_k,
                rng=rng,
                strict=False,
                disrupted_spans=local_disrupted,
                near_positive_cfg=np_cfg_for_negatives,
            )
            # Accumulate disrupted-fill stats if available
            _disrupted_stats["used"] += getattr(negs, "_n_disrupted_used", 0)
            _disrupted_stats["hard_target"] += getattr(negs, "_n_hard_target", 0)
            _disrupted_stats["total_neg"] += getattr(negs, "_n_total_target", 0)
            neg_spans = torch.tensor(
                [[n["start_0b"], n["end_0b"]] for n in negs],
                dtype=torch.long,
            ) if negs else torch.zeros(0, 2, dtype=torch.long)

            # HIMP1 extras: relations + weights, only when span weighting is
            # actually enabled (np_cfg_for_negatives gates apply_to_span_negatives).
            if np_cfg_for_negatives is not None and hasattr(negs, "_weights"):
                chunk_extras["neg_relations"] = list(getattr(negs, "_relations", []))
                w = getattr(negs, "_weights", [])
                chunk_extras["neg_weights"] = torch.tensor(w, dtype=torch.float32)
        else:
            pos_spans = torch.zeros(0, 2, dtype=torch.long)
            neg_spans = torch.zeros(0, 2, dtype=torch.long)

        # HIMP2 extras: residue masks + (optional) extra windows for window_mode=all.
        if residue_enabled:
            # Review fix 4: per-chunk trusted interior matching
            # ChunkPlan.trusted_interior semantics: only drop the seam
            # margin, not the protein N-/C-termini. ``chunk_central_margin``
            # carries the configured seam width; we apply it on the left
            # iff this chunk is not the first, and on the right iff it is
            # not the last (i.e. its end reaches sequence_length).
            seam = int(chunk_central_margin)
            seq_lens = batch.get("sequence_lengths")
            chunk_seq_len = (
                int(seq_lens[i].item()) if seq_lens is not None else chunk_end
            )
            left_seam = 0 if chunk_start == 0 else seam
            right_seam = 0 if chunk_end >= chunk_seq_len else seam
            central_start_local = min(chunk_len, max(0, left_seam))
            central_end_local = max(central_start_local, chunk_len - right_seam)

            residue_meta = build_residue_labels(
                positives=local_positives,
                chunk_len=chunk_len,
                central_start=central_start_local,
                central_end=central_end_local,
                near_positive_cfg=near_positive_cfg,
            )
            chunk_extras["residue_meta"] = residue_meta
            if window_mode == "all":
                # Enumerate residue-supervision windows over the trusted
                # central region (per-chunk, NOT symmetric).
                extra_windows = enumerate_all_residue_windows(
                    central_start=central_start_local,
                    central_end=central_end_local,
                    min_k=int(min_k),
                    max_k=int(max_k),
                    max_windows=max_windows,
                )
                if extra_windows:
                    chunk_extras["residue_extra_windows"] = torch.tensor(
                        extra_windows, dtype=torch.long,
                    )
                else:
                    chunk_extras["residue_extra_windows"] = torch.zeros(0, 2, dtype=torch.long)

        pos_spans_list.append(pos_spans)
        neg_spans_list.append(neg_spans)
        allele_pos_list.append(torch.zeros(pos_spans.shape[0], dtype=torch.long))
        allele_neg_list.append(torch.zeros(neg_spans.shape[0], dtype=torch.long))
        extras_list.append(chunk_extras)

    # Log disrupted-fill ratio if any disrupted spans were used
    if _disrupted_stats["used"] > 0:
        logger.debug(
            "  disrupted hard-neg fill: %d/%d hard quota (%.1f%%), %d/%d total neg (%.1f%%)",
            _disrupted_stats["used"], _disrupted_stats["hard_target"],
            100.0 * _disrupted_stats["used"] / max(_disrupted_stats["hard_target"], 1),
            _disrupted_stats["used"], _disrupted_stats["total_neg"],
            100.0 * _disrupted_stats["used"] / max(_disrupted_stats["total_neg"], 1),
        )

    return (
        pos_spans_list, neg_spans_list,
        allele_pos_list, allele_neg_list,
        extras_list,
    )


def _split_himp_kwargs(neg_cfg: dict) -> tuple[dict, dict | None, dict | None, int]:
    """Pull HIMP-only kwargs out of ``neg_cfg`` so the legacy ``**neg_cfg``
    expansion still matches ``prepare_chunk_spans``'s positional kwargs."""
    legacy = dict(neg_cfg)
    near_positive_cfg = legacy.pop("near_positive_cfg", None)
    residue_cfg = legacy.pop("residue_cfg", None)
    chunk_central_margin = int(legacy.pop("chunk_central_margin", 0))
    return legacy, near_positive_cfg, residue_cfg, chunk_central_margin


def _forward_union_and_compute_losses(
    model: nn.Module,
    batch: dict,
    pos_spans_list: list[torch.Tensor],
    neg_spans_list: list[torch.Tensor],
    allele_pos_list: list[torch.Tensor],
    allele_neg_list: list[torch.Tensor],
    extras_list: list[dict],
    loss_cfg: dict,
    residue_cfg: dict | None,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], dict, int]:
    """Run the union (pos + neg + residue-extra) forward and compute per-chunk
    span + residue losses. Returns ``(avg_loss, all_pos_logits_detached,
    all_neg_logits_detached, residue_stats, n_chunks_with_pos)``.

    ``residue_stats`` keys: ``loss_residue_sum``, ``n_residue_pairs``,
    ``residue_skipped_chunks``, ``n_residue_chunks``.
    """
    token_ids = batch["token_ids"]
    attention_mask = batch["attention_mask"]
    chunk_lengths = batch["chunk_ends"] - batch["chunk_starts"]
    device = token_ids.device

    residue_enabled = bool(residue_cfg and residue_cfg.get("enabled", False))
    lambda_residue = float((residue_cfg or {}).get("lambda_residue", 0.0))
    residue_aggregation = (residue_cfg or {}).get("aggregation", "log_mean_exp")
    residue_agg_params = (residue_cfg or {}).get("aggregation_params", {})
    residue_loss_mode = (residue_cfg or {}).get("loss_mode", "pairwise_margin")
    residue_margin_m = float((residue_cfg or {}).get("margin_m_residue", 0.5))
    residue_min_far_bg = int((residue_cfg or {}).get("min_far_bg_residues", 4))

    # Build per-chunk forward batches: union of (pos, neg, residue_extra) spans.
    all_spans_list = []
    all_allele_list = []
    pos_counts = []
    neg_counts = []
    residue_extra_counts = []  # number of residue-only extra windows appended
    for i in range(len(pos_spans_list)):
        ps = pos_spans_list[i]
        ns = neg_spans_list[i]
        ap = allele_pos_list[i]
        an = allele_neg_list[i]
        extras_i = extras_list[i] if i < len(extras_list) else {}
        residue_extra = extras_i.get("residue_extra_windows")

        parts = []
        allele_parts = []
        if ps.shape[0] > 0:
            parts.append(ps)
            allele_parts.append(ap)
        if ns.shape[0] > 0:
            parts.append(ns)
            allele_parts.append(an)
        n_extra = 0
        if residue_enabled and residue_extra is not None and residue_extra.shape[0] > 0:
            parts.append(residue_extra)
            allele_parts.append(torch.zeros(residue_extra.shape[0], dtype=torch.long))
            n_extra = int(residue_extra.shape[0])

        if parts:
            combined_spans = torch.cat(parts, dim=0)
            combined_allele = torch.cat(allele_parts, dim=0)
        else:
            combined_spans = torch.zeros(0, 2, dtype=torch.long)
            combined_allele = torch.zeros(0, dtype=torch.long)

        all_spans_list.append(combined_spans.to(device))
        all_allele_list.append(combined_allele.to(device))
        pos_counts.append(int(ps.shape[0]))
        neg_counts.append(int(ns.shape[0]))
        residue_extra_counts.append(n_extra)

    # Wave-4 dual-head: strip exact-head keys (compute_loss does not accept them)
    # from a local copy, and request the z_exact readout only when the exact term
    # is active. want_dual=False => bit-for-bit legacy forward + loss.
    loss_cfg = dict(loss_cfg)
    lambda_exact = float(loss_cfg.pop("lambda_exact", 0.0))
    exact_margin_m = float(loss_cfg.pop("exact_margin_m", 0.3))
    exact_hard_topk = int(loss_cfg.pop("exact_hard_topk", 8))
    want_dual = lambda_exact > 0.0 and getattr(model, "enable_boundary_head", False)

    # Only pass return_dual when actually needed, so legacy/stub models (and any
    # caller without the kwarg) stay bit-for-bit compatible.
    if want_dual:
        logits_list = model(token_ids, attention_mask, all_spans_list, all_allele_list,
                            chunk_lengths, return_dual=True)
    else:
        logits_list = model(token_ids, attention_mask, all_spans_list, all_allele_list,
                            chunk_lengths)

    # Per-chunk: span loss + (optional) residue loss
    all_pos_logits: list[torch.Tensor] = []
    all_neg_logits: list[torch.Tensor] = []
    total_loss = torch.tensor(0.0, device=device)
    residue_stats = {
        "loss_residue_sum": 0.0,
        "n_residue_pairs": 0,
        "residue_skipped_chunks": 0,
        "n_residue_chunks": 0,
        "loss_iou_rank_sum": 0.0,
        "n_iou_rank_chunks": 0,
        "loss_exact_sum": 0.0,
        "n_exact_chunks": 0,
    }
    lambda_iou_rank = float(loss_cfg.get("lambda_iou_rank", 0.0))
    n_chunks_with_pos = 0

    for i, item in enumerate(logits_list):
        # Dual readout unpacks to (z_region, z_exact); legacy is a bare z_region.
        if want_dual:
            logits, z_exact_i = item
        else:
            logits, z_exact_i = item, None
        pc = pos_counts[i]
        nc = neg_counts[i]
        if pc == 0:
            continue
        pos_logits = logits[:pc]
        neg_logits = logits[pc:pc + nc]

        nan_guard(pos_logits, f"pos_logits[chunk={i}]")
        nan_guard(neg_logits, f"neg_logits[chunk={i}]")

        # HIMP1: pull negative weights from extras (None when not configured).
        extras_i = extras_list[i] if i < len(extras_list) else {}
        neg_w = extras_i.get("neg_weights")
        if neg_w is not None:
            neg_w = neg_w.to(device)

        # Wave-3: window IoU-ranking auxiliary. Rank the union candidate windows
        # (pos ∪ neg) by their max IoU to this chunk's GT positives so the span
        # scorer learns the M6 region-AP ordering. No-op when lambda_iou_rank==0.
        iou_rank_kwargs: dict = {}
        if lambda_iou_rank > 0.0:
            win = torch.cat([pos_spans_list[i], neg_spans_list[i]], dim=0).to(device)
            gt = pos_spans_list[i].to(device)
            iou_rank_kwargs = {
                "window_logits": logits[: pc + nc],
                "window_ious": max_iou_per_window(win, gt),
            }

        loss_dict = compute_loss(
            pos_logits, neg_logits, **loss_cfg, neg_weights=neg_w, **iou_rank_kwargs
        )
        nan_guard(loss_dict["loss_total"], f"loss_total[chunk={i}]")
        chunk_loss = loss_dict["loss_total"]
        if iou_rank_kwargs:
            residue_stats["loss_iou_rank_sum"] += float(loss_dict["loss_iou_rank"].detach().item())
            residue_stats["n_iou_rank_chunks"] += 1

        # Wave-4 dual-head: exact term on z_exact. Gradient reaches only the
        # boundary head (z_exact = stopgrad(z_region) + boundary_head(detach(phi))),
        # so the region/residue landscape trajectory is unchanged.
        if want_dual and z_exact_i is not None:
            loss_exact = exact_margin_loss(
                z_exact_i[:pc], z_exact_i[pc:pc + nc],
                margin_m=exact_margin_m, hard_topk=exact_hard_topk,
            )
            chunk_loss = chunk_loss + lambda_exact * loss_exact
            residue_stats["loss_exact_sum"] += float(loss_exact.detach().item())
            residue_stats["n_exact_chunks"] += 1

        # HIMP3: residue ranking loss
        if residue_enabled and extras_i.get("residue_meta") is not None:
            residue_stats["n_residue_chunks"] += 1
            meta = extras_i["residue_meta"]
            chunk_len = int(meta["label"].shape[0])
            # Residue-supervision windows = pos ∪ neg ∪ residue_extra.
            # window_logits parallel to the union forward order: [pos | neg | extra].
            n_extra = residue_extra_counts[i]
            window_logits = logits[: pc + nc + n_extra]
            # Reconstruct windows list parallel to window_logits.
            windows: list[tuple[int, int]] = []
            for k in range(pc):
                s, e = pos_spans_list[i][k].tolist()
                windows.append((int(s), int(e)))
            for k in range(nc):
                s, e = neg_spans_list[i][k].tolist()
                windows.append((int(s), int(e)))
            if n_extra:
                extras_w = extras_i["residue_extra_windows"]
                for k in range(n_extra):
                    s, e = extras_w[k].tolist()
                    windows.append((int(s), int(e)))

            residue_scores = aggregate_window_logits_to_residues(
                window_logits=window_logits,
                windows=windows,
                chunk_len=chunk_len,
                mode=residue_aggregation,
                params=residue_agg_params,
            )
            # Review fix 3: residues not covered by any residue-supervision
            # window receive -inf scores from the aggregator. They MUST NOT
            # leak into the ranking loss as "valid" pos / far_bg / central
            # residues — otherwise n_pairs is inflated, ranking gaps are
            # meaningless (-inf vs finite), and min_far_bg_residues skip can
            # be circumvented. Intersect every mask with finite-score support.
            finite_mask = torch.isfinite(residue_scores)
            label_t = torch.tensor(meta["label"], device=device).bool() & finite_mask
            far_bg_t = torch.tensor(meta["far_bg_mask"], device=device) & finite_mask
            central_t = torch.tensor(meta["central_mask"], device=device) & finite_mask
            if residue_loss_mode != "pairwise_margin":
                raise NotImplementedError(
                    f"residue.loss_mode={residue_loss_mode!r} not implemented in v0"
                )
            r_loss, r_meta = residue_pairwise_margin_loss(
                residue_scores=residue_scores,
                label=label_t,
                far_bg_mask=far_bg_t,
                central_mask=central_t,
                margin_m=residue_margin_m,
                min_far_bg=residue_min_far_bg,
            )
            if r_meta["skipped"]:
                residue_stats["residue_skipped_chunks"] += 1
            else:
                residue_stats["loss_residue_sum"] += float(r_loss.detach().item())
                residue_stats["n_residue_pairs"] += int(r_meta["n_pairs"])
                chunk_loss = chunk_loss + float(lambda_residue) * r_loss

        total_loss = total_loss + chunk_loss
        n_chunks_with_pos += 1
        all_pos_logits.append(pos_logits.detach())
        all_neg_logits.append(neg_logits.detach())

    if n_chunks_with_pos == 0:
        return (
            torch.tensor(0.0, device=device),
            all_pos_logits, all_neg_logits,
            residue_stats, 0,
        )

    avg_loss = total_loss / n_chunks_with_pos
    return avg_loss, all_pos_logits, all_neg_logits, residue_stats, n_chunks_with_pos


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    loss_cfg: dict,
    neg_cfg: dict,
    grad_clip: float = 1.0,
    scheduler=None,
    warmup_steps: int = 0,
    global_step: int = 0,
    rng: np.random.RandomState | None = None,
) -> StepMetrics:
    """Execute one training step on a batch of chunks.

    Returns aggregated StepMetrics across all chunks in the batch.
    """
    model.train()
    loss_cfg = normalize_loss_cfg(loss_cfg)

    legacy_neg_cfg, near_positive_cfg, residue_cfg, chunk_central_margin = \
        _split_himp_kwargs(neg_cfg)

    pos_spans_list, neg_spans_list, allele_pos_list, allele_neg_list, extras_list = \
        prepare_chunk_spans(
            batch, batch_idx=0, rng=rng,
            near_positive_cfg=near_positive_cfg,
            residue_cfg=residue_cfg,
            chunk_central_margin=chunk_central_margin,
            **legacy_neg_cfg,
        )

    # Skip batch if no positives at all
    total_pos = sum(s.shape[0] for s in pos_spans_list)
    if total_pos == 0:
        return StepMetrics()

    avg_loss, all_pos_logits, all_neg_logits, residue_stats, n_chunks_with_pos = \
        _forward_union_and_compute_losses(
            model=model, batch=batch,
            pos_spans_list=pos_spans_list, neg_spans_list=neg_spans_list,
            allele_pos_list=allele_pos_list, allele_neg_list=allele_neg_list,
            extras_list=extras_list,
            loss_cfg=loss_cfg, residue_cfg=residue_cfg,
        )

    if n_chunks_with_pos == 0:
        return StepMetrics()

    # Backward + optimize
    optimizer.zero_grad()
    avg_loss.backward()

    if grad_clip > 0:
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            grad_clip,
        )

    optimizer.step()

    # Warmup: linear ramp from 0 to base_lr, then cosine decay
    if warmup_steps > 0 and global_step < warmup_steps:
        warmup_factor = (global_step + 1) / warmup_steps
        for pg in optimizer.param_groups:
            pg["lr"] = pg.get("initial_lr", pg["lr"]) * warmup_factor
    elif scheduler is not None:
        scheduler.step()

    # Compute metrics. ``compute_loss`` here is recomputed unweighted on the
    # concatenated logits as a *diagnostic* (so loss_intra / loss_margin
    # reflect the unweighted span objective). The reported ``loss_total``
    # however is overridden to ``avg_loss``, which already includes the
    # per-negative weights and ``lambda_residue * loss_residue`` term — i.e.
    # the actual quantity that ``avg_loss.backward()`` minimized.
    cat_pos = torch.cat(all_pos_logits) if all_pos_logits else torch.tensor([])
    cat_neg = torch.cat(all_neg_logits) if all_neg_logits else torch.tensor([])
    loss_for_metrics = _diagnostic_loss_dict(cat_pos, cat_neg, loss_cfg)

    metrics = compute_sanity_metrics(cat_pos, cat_neg, loss_for_metrics)
    metrics.loss_total = float(avg_loss.detach().item())
    _attach_residue_stats(metrics, residue_stats)
    return metrics


def _attach_residue_stats(metrics: StepMetrics, residue_stats: dict) -> None:
    """Mutate ``metrics`` with per-step residue aggregates."""
    n_chunks = int(residue_stats.get("n_residue_chunks", 0))
    n_skipped = int(residue_stats.get("residue_skipped_chunks", 0))
    n_contributing = max(n_chunks - n_skipped, 0)
    if n_contributing > 0:
        metrics.loss_residue = float(residue_stats["loss_residue_sum"]) / n_contributing
    else:
        metrics.loss_residue = 0.0
    metrics.n_residue_pairs = int(residue_stats.get("n_residue_pairs", 0))
    metrics.residue_skipped_chunks = n_skipped
    metrics.n_residue_chunks = n_chunks
    n_iou = int(residue_stats.get("n_iou_rank_chunks", 0))
    metrics.loss_iou_rank = (
        float(residue_stats.get("loss_iou_rank_sum", 0.0)) / n_iou if n_iou > 0 else 0.0
    )


@torch.no_grad()
def val_step(
    model: nn.Module,
    batch: dict,
    loss_cfg: dict,
    neg_cfg: dict,
    rng: np.random.RandomState | None = None,
) -> StepMetrics:
    """Execute one validation step (no grad, no optimizer)."""
    model.eval()
    loss_cfg = normalize_loss_cfg(loss_cfg)

    legacy_neg_cfg, near_positive_cfg, residue_cfg, chunk_central_margin = \
        _split_himp_kwargs(neg_cfg)

    pos_spans_list, neg_spans_list, allele_pos_list, allele_neg_list, extras_list = \
        prepare_chunk_spans(
            batch, batch_idx=0, rng=rng,
            near_positive_cfg=near_positive_cfg,
            residue_cfg=residue_cfg,
            chunk_central_margin=chunk_central_margin,
            **legacy_neg_cfg,
        )

    total_pos = sum(s.shape[0] for s in pos_spans_list)
    if total_pos == 0:
        return StepMetrics()

    avg_loss, all_pos_logits, all_neg_logits, residue_stats, n_chunks_with_pos = \
        _forward_union_and_compute_losses(
            model=model, batch=batch,
            pos_spans_list=pos_spans_list, neg_spans_list=neg_spans_list,
            allele_pos_list=allele_pos_list, allele_neg_list=allele_neg_list,
            extras_list=extras_list,
            loss_cfg=loss_cfg, residue_cfg=residue_cfg,
        )

    if n_chunks_with_pos == 0:
        return StepMetrics()

    cat_pos = torch.cat(all_pos_logits) if all_pos_logits else torch.tensor([])
    cat_neg = torch.cat(all_neg_logits) if all_neg_logits else torch.tensor([])
    loss_dict = _diagnostic_loss_dict(cat_pos, cat_neg, loss_cfg)
    metrics = compute_sanity_metrics(cat_pos, cat_neg, loss_dict)
    # Override loss_total with the actual HIMP objective (weighted span +
    # lambda_residue * residue), matching what train_step backprops on.
    metrics.loss_total = float(avg_loss.detach().item())
    _attach_residue_stats(metrics, residue_stats)
    return metrics


# ── E6: Checkpointing & Logging ────────────────────────────────────────────

CHECKPOINT_METADATA_KEYS = frozenset({
    "epoch", "global_step", "monitor_metric", "monitor_value",
    "config_hash", "manifest_version", "diff_ids_applied",
})

LOG_ENTRY_KEYS = frozenset({
    "epoch", "phase", "loss_total", "loss_intra", "loss_mp", "loss_smooth",
    "loss_margin", "mean_pos_logit", "mean_neg_logit", "logit_gap",
    "per_protein_auc", "total_pos", "total_neg", "n_steps", "timestamp",
    # HIMP4: residue ranking loss decomposition (always present; 0 when disabled).
    "loss_residue", "lambda_residue", "residue_skipped_chunks",
    "n_residue_pairs", "n_residue_chunks",
})


def config_hash(cfg: dict) -> str:
    """Deterministic hash of resolved config."""
    serialized = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    monitor_metric: str,
    monitor_value: float,
    cfg_hash: str,
    path: Path,
    manifest_version: str = "v1.1",
    diff_ids_applied: list[str] | None = None,
):
    """Save model checkpoint with metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": {
            "epoch": epoch,
            "global_step": global_step,
            "monitor_metric": monitor_metric,
            "monitor_value": monitor_value,
            "config_hash": cfg_hash,
            "manifest_version": manifest_version,
            "diff_ids_applied": list(diff_ids_applied or []),
        },
    }, path)


def load_checkpoint(path: Path) -> dict:
    """Load checkpoint and validate metadata keys."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt.get("metadata", {})
    missing = CHECKPOINT_METADATA_KEYS - set(meta.keys())
    if missing:
        raise ValueError(f"Checkpoint missing metadata keys: {missing}")
    return ckpt


def write_log_entry(log_path: Path, entry: dict):
    """Append a JSONL log entry with schema validation."""
    missing = LOG_ENTRY_KEYS - set(entry.keys())
    if missing:
        raise ValueError(f"Log entry missing keys: {missing}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def save_resolved_config(cfg: dict, run_dir: Path):
    """Save resolved config snapshot."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


# ── E5+E6+E7: Trainer ──────────────────────────────────────────────────────

class Trainer:
    """Training loop with checkpointing, logging, and early stopping.

    Wires together E1-E4 components into an executable training pipeline.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        train_cfg: dict,
        run_dir: Path,
        device: torch.device | str = "cpu",
        registry_path: Path | str | None = None,
        wandb_cfg: dict | None = None,
        # Per-protein full-scan evaluation (optional; enabled when all are provided)
        val_entries: list | None = None,
        tokenizer=None,
        min_k: int = 12,
        max_k: int = 25,
        context_len: int = 1022,
        epoch_hook=None,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Per-protein eval resources
        self.val_entries = val_entries
        self.tokenizer = tokenizer
        self.min_k = min_k
        self.max_k = max_k
        self.context_len = context_len
        self._pp_eval_enabled = (val_entries is not None and tokenizer is not None)
        self.cfg = train_cfg
        self.run_dir = Path(run_dir)
        self.registry_path = Path(registry_path) if registry_path is not None else None
        self.epoch_hook = epoch_hook
        self._epoch_aux_metrics: dict = {}
        self._last_epoch_aux_metrics: dict = {}

        # Build optimizer (only learnable params)
        self.optimizer = build_optimizer(
            model,
            optimizer_name=train_cfg["optimizer"],
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
        # Store initial_lr for warmup linear ramp
        for pg in self.optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]

        # Build scheduler
        steps_per_epoch = len(train_loader) if hasattr(train_loader, '__len__') else 100
        self.scheduler = build_scheduler(
            self.optimizer,
            scheduler_name=train_cfg["scheduler"],
            warmup_steps=train_cfg["warmup_steps"],
            max_epochs=train_cfg["max_epochs"],
            steps_per_epoch=steps_per_epoch,
        )

        # Loss config (normalized to compute_loss signature)
        self.loss_cfg = normalize_loss_cfg(train_cfg["loss"])

        # Negative sampling config + HIMP1/2 hooks. Re-validate the HIMP
        # blocks defensively so that any cfg reaching the trainer (whether
        # from load_train_config, an override merge, or an ad-hoc test
        # fixture) is guaranteed to satisfy the HIMP0 schema.
        validate_himp_train_blocks(train_cfg)
        np_cfg = train_cfg.get("near_positive")
        if isinstance(np_cfg, dict) and not np_cfg.get("enabled", False):
            np_cfg = None
        res_cfg = train_cfg.get("residue")
        if isinstance(res_cfg, dict) and not res_cfg.get("enabled", False):
            res_cfg = None

        chunk_margin = 0
        if "chunking" in train_cfg and isinstance(train_cfg["chunking"], dict):
            if bool(train_cfg["chunking"].get("enabled", False)):
                chunk_margin = int(train_cfg["chunking"].get("margin", 0))

        self.neg_cfg = {
            "neg_ratio": train_cfg["neg_ratio"],
            "hard_negative_fraction": train_cfg["hard_negative_fraction"],
            "hard_neg_max_overlap_ratio": train_cfg["hard_neg_max_overlap_ratio"],
            "hard_neg_offset_range": train_cfg["hard_neg_offset_range"],
            "neg_length_sampling": train_cfg["neg_length_sampling"],
            "near_positive_cfg": np_cfg,
            "residue_cfg": res_cfg,
            "chunk_central_margin": chunk_margin,
        }
        self._lambda_residue = float((res_cfg or {}).get("lambda_residue", 0.0))

        # HIMP4 startup banner: print every new hyperparameter so SLURM logs
        # capture the effective resolved config.
        logger.info(
            "HIMP near_positive: enabled=%s schedule=%s near_gap_max=%s "
            "schedule_params=%s metric=%s apply_to_span_negatives=%s "
            "apply_to_residue_labels=%s",
            np_cfg is not None,
            (np_cfg or {}).get("schedule"),
            (np_cfg or {}).get("near_gap_max"),
            (np_cfg or {}).get("schedule_params"),
            (np_cfg or {}).get("metric"),
            (np_cfg or {}).get("apply_to_span_negatives"),
            (np_cfg or {}).get("apply_to_residue_labels"),
        )
        logger.info(
            "HIMP residue: enabled=%s lambda_residue=%s aggregation=%s "
            "aggregation_params=%s loss_mode=%s margin_m_residue=%s "
            "window_mode=%s max_windows_per_chunk=%s min_far_bg_residues=%s "
            "central_margin=%s",
            res_cfg is not None,
            self._lambda_residue,
            (res_cfg or {}).get("aggregation"),
            (res_cfg or {}).get("aggregation_params"),
            (res_cfg or {}).get("loss_mode"),
            (res_cfg or {}).get("margin_m_residue"),
            (res_cfg or {}).get("window_mode"),
            (res_cfg or {}).get("max_windows_per_chunk"),
            (res_cfg or {}).get("min_far_bg_residues"),
            chunk_margin,
        )

        # Checkpointing
        self.cfg_hash = config_hash(train_cfg)
        self.monitor_metric = train_cfg["monitor_metric"]
        # Determine early stopping direction: loss metrics are minimized, others maximized
        self.monitor_mode = "min" if "loss" in self.monitor_metric else "max"
        self.best_monitor_value = float("inf") if self.monitor_mode == "min" else float("-inf")
        self.patience_counter = 0
        self.early_stopping_patience = train_cfg["early_stopping_patience"]
        self.checkpoint_every_n = train_cfg["checkpoint_every_n_epochs"]
        self.diff_ids_applied = list(train_cfg.get("diff_ids_applied", []))

        # Logging
        self.train_log_path = self.run_dir / "train_log.jsonl"
        self.val_log_path = self.run_dir / "val_log.jsonl"
        self.global_step = 0

        # Separate RNGs for train/val negative sampling (avoid cross-contamination)
        self.train_rng = np.random.RandomState(train_cfg["seed"])
        self.val_rng = np.random.RandomState(train_cfg["seed"] + 1)

        # Run identity (G1/G3)
        self.run_id = generate_run_id(seed_str=self.cfg_hash)
        self.manifest_version = str(train_cfg.get("manifest_version", "v1.1"))
        self.protocol_signature = compute_protocol_signature(train_cfg)

        # Save resolved config
        save_resolved_config(train_cfg, self.run_dir)

        # W&B integration
        self._wandb = None
        if wandb_cfg and wandb_cfg.get("enabled", False):
            try:
                import wandb
                self._wandb = wandb
                wandb.init(
                    entity=wandb_cfg.get("entity"),
                    project=wandb_cfg.get("project", "Immune-Design"),
                    name=wandb_cfg.get("name", self.run_id),
                    config=train_cfg,
                    dir=str(self.run_dir),
                    resume="allow",
                )
                wandb.watch(self.model, log="gradients", log_freq=50)
                logger.info("W&B initialized: %s", wandb.run.url or wandb.run.id)
            except ImportError:
                logger.warning("wandb not installed, skipping W&B logging")
            except Exception as e:
                logger.warning("wandb init failed: %s", e)

    def _frozen_encoder_guard(self):
        """Verify encoder params still have requires_grad=False after step.

        Only applies when the encoder is expected to be frozen (e.g. ESM-2).
        Trainable encoders (E1/E2) skip this check.
        """
        encoder = self.model.encoder
        # Skip guard if encoder has any trainable params (it's intentionally trainable)
        has_trainable = any(p.requires_grad for p in encoder.parameters())
        if has_trainable:
            # For trainable encoders, this guard is not applicable
            # Check if the encoder was supposed to be frozen (FrozenESMEncoder pattern)
            if not hasattr(encoder, 'esm'):
                return  # Non-ESM encoder, trainable by design
        for name, p in encoder.named_parameters():
            if p.requires_grad:
                raise RuntimeError(
                    f"Frozen encoder param '{name}' has requires_grad=True after step"
                )

    def train_epoch(self, epoch: int) -> dict:
        """Run one training epoch. Returns epoch metrics dict."""
        # Reset train RNG each epoch with epoch-dependent seed: each epoch's
        # negatives are deterministic given (seed, epoch), while still varying
        # across epochs. Mirrors val_rng's per-epoch reset (line 768).
        self.train_rng = np.random.RandomState(self.cfg["seed"] + epoch)
        step_metrics_list = []
        self._epoch_disrupted_used = 0
        self._epoch_hard_target = 0
        self._epoch_total_neg = 0

        if hasattr(self.train_loader, 'batch_sampler') and \
           hasattr(self.train_loader.batch_sampler, 'set_epoch'):
            self.train_loader.batch_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(self.train_loader):
            # Move tensors to device
            batch = self._to_device(batch)

            metrics = train_step(
                model=self.model,
                batch=batch,
                optimizer=self.optimizer,
                loss_cfg=self.loss_cfg,
                neg_cfg=self.neg_cfg,
                grad_clip=self.cfg["grad_clip"],
                scheduler=self.scheduler,
                warmup_steps=self.cfg["warmup_steps"],
                global_step=self.global_step,
                rng=self.train_rng,
            )
            step_metrics_list.append(metrics)
            self.global_step += 1

        # Frozen encoder guard (after full epoch)
        self._frozen_encoder_guard()

        epoch_metrics = aggregate_epoch_metrics(step_metrics_list)
        epoch_metrics["epoch"] = epoch
        epoch_metrics["phase"] = "train"
        epoch_metrics["timestamp"] = time.time()
        epoch_metrics["lambda_residue"] = self._lambda_residue
        if self._epoch_aux_metrics:
            epoch_metrics.update(self._epoch_aux_metrics)
            self._last_epoch_aux_metrics = dict(self._epoch_aux_metrics)

        write_log_entry(self.train_log_path, epoch_metrics)
        return epoch_metrics

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict:
        """Run one validation epoch. Returns epoch metrics dict."""
        # Reset val RNG each epoch for deterministic, comparable val metrics
        self.val_rng = np.random.RandomState(self.cfg["seed"] + 1)
        step_metrics_list = []

        for batch_idx, batch in enumerate(self.val_loader):
            batch = self._to_device(batch)
            metrics = val_step(
                model=self.model,
                batch=batch,
                loss_cfg=self.loss_cfg,
                neg_cfg=self.neg_cfg,
                rng=self.val_rng,
            )
            step_metrics_list.append(metrics)

        epoch_metrics = aggregate_epoch_metrics(step_metrics_list)
        epoch_metrics["epoch"] = epoch
        epoch_metrics["phase"] = "val"
        epoch_metrics["timestamp"] = time.time()
        epoch_metrics["lambda_residue"] = self._lambda_residue

        # Per-protein full-window-scan evaluation
        if self._pp_eval_enabled:
            pp_metrics = full_val_eval(
                model=self.model,
                tokenizer=self.tokenizer,
                val_entries=self.val_entries,
                min_k=self.min_k,
                max_k=self.max_k,
                device=self.device,
                recall_ks=(50, 100),
                context_len=self.context_len,
            )
            epoch_metrics.update(pp_metrics)

        write_log_entry(self.val_log_path, epoch_metrics)
        return epoch_metrics

    def fit(self, max_epochs: int | None = None) -> dict:
        """Full training loop with checkpointing and early stopping.

        Returns run summary dict.
        """
        max_epochs = max_epochs or self.cfg["max_epochs"]
        logger.info("Starting training for %d epochs, run_dir=%s", max_epochs, self.run_dir)

        for epoch in range(max_epochs):
            # Optional epoch hook (e.g., runtime augmentation rebuilds train_loader)
            if self.epoch_hook is not None:
                self._epoch_aux_metrics = dict(self.epoch_hook(self, epoch) or {})
            else:
                self._epoch_aux_metrics = {}

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.val_epoch(epoch)

            monitor_val = val_metrics.get(self.monitor_metric)
            if monitor_val is None:
                monitor_val = float("-inf") if self.monitor_mode == "max" else float("inf")

            # Checkpoint every N epochs
            if (epoch + 1) % self.checkpoint_every_n == 0:
                ckpt_path = self.run_dir / f"epoch_{epoch}.pt"
                save_checkpoint(
                    self.model, self.optimizer, epoch, self.global_step,
                    self.monitor_metric, monitor_val, self.cfg_hash, ckpt_path,
                    diff_ids_applied=self.diff_ids_applied,
                )

            # Best checkpoint (direction-aware: min for loss, max for gap/auc)
            improved = (monitor_val < self.best_monitor_value) if self.monitor_mode == "min" \
                else (monitor_val > self.best_monitor_value)
            if improved:
                self.best_monitor_value = monitor_val
                self.patience_counter = 0
                best_path = self.run_dir / "best.pt"
                save_checkpoint(
                    self.model, self.optimizer, epoch, self.global_step,
                    self.monitor_metric, monitor_val, self.cfg_hash, best_path,
                    diff_ids_applied=self.diff_ids_applied,
                )
            else:
                self.patience_counter += 1

            # W&B epoch logging
            if self._wandb is not None:
                log_dict = {"epoch": epoch}
                for k, v in train_metrics.items():
                    if isinstance(v, (int, float)):
                        log_dict[f"train/{k}"] = v
                for k, v in val_metrics.items():
                    if isinstance(v, (int, float)):
                        log_dict[f"val/{k}"] = v
                log_dict["lr"] = self.optimizer.param_groups[0]["lr"]
                log_dict["patience"] = self.patience_counter
                if hasattr(self.model, "scorer") and hasattr(self.model.scorer, "log_logit_scale"):
                    log_dict["logit_scale"] = self.model.scorer.log_logit_scale.exp().item()
                self._wandb.log(log_dict, step=epoch)

            pp_auc_str = "%.4f" % val_metrics["pp_auc"] if val_metrics.get("pp_auc") is not None else "N/A"
            pp_ap_str = "%.4f" % val_metrics["pp_ap"] if val_metrics.get("pp_ap") is not None else "N/A"
            logger.info(
                "Epoch %d — train_loss=%.4f, val_loss=%.4f, "
                "pp_auc=%s, pp_ap=%s, patience=%d/%d",
                epoch, train_metrics.get("loss_total", 0),
                val_metrics.get("loss_total", 0),
                pp_auc_str, pp_ap_str,
                self.patience_counter, self.early_stopping_patience,
            )

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        # Run summary (includes run_id for G3 registry integration)
        summary = {
            "run_id": self.run_id,
            "final_epoch": epoch,
            "best_monitor_value": self.best_monitor_value,
            "monitor_metric": self.monitor_metric,
            "global_steps": self.global_step,
            "config_hash": self.cfg_hash,
            "run_dir": str(self.run_dir),
            "manifest_version": self.manifest_version,
            "protocol_signature": self.protocol_signature,
        }
        aug_summary = {
            k.removeprefix("aug_"): v
            for k, v in self._last_epoch_aux_metrics.items()
            if k.startswith("aug_")
        }
        if aug_summary:
            summary["augmentation"] = aug_summary

        with open(self.run_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # G3: auto-append registry row if registry_path configured
        if self.registry_path is not None:
            best_ckpt = self.run_dir / "best.pt"
            if best_ckpt.exists():
                primary_metrics = {
                    self.monitor_metric: self.best_monitor_value,
                }
                if "logit_gap" not in primary_metrics:
                    primary_metrics["logit_gap"] = self.best_monitor_value
                row = build_registry_row(
                    run_id=self.run_id,
                    run_dir=self.run_dir,
                    best_checkpoint_path=best_ckpt,
                    config_hash=self.cfg_hash,
                    manifest_version=self.manifest_version,
                    diff_ids_applied=self.diff_ids_applied,
                    protocol_signature=self.protocol_signature,
                    primary_metrics=primary_metrics,
                )
                append_registry_row(row, self.registry_path)
                logger.info("Registry row appended for run_id=%s", self.run_id)
            else:
                logger.warning(
                    "Skipping registry write: best.pt not found at %s", best_ckpt,
                )

        if self._wandb is not None:
            self._wandb.log({"best_monitor_value": self.best_monitor_value})
            self._wandb.finish()

        return summary

    def _to_device(self, batch: dict) -> dict:
        """Move tensor values in batch to device."""
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved


# ── E7: Smoke Diagnostics ──────────────────────────────────────────────────

def verify_run_artifacts(run_dir: Path, n_epochs: int, checkpoint_every_n: int) -> dict:
    """Verify all expected training artifacts exist and are loadable.

    Returns dict with 'ok' bool and 'missing'/'errors' lists.
    """
    run_dir = Path(run_dir)
    missing = []
    errors = []

    # Required files
    required = ["train_log.jsonl", "val_log.jsonl", "resolved_config.yaml",
                 "run_summary.json", "best.pt"]
    for f in required:
        if not (run_dir / f).exists():
            missing.append(f)

    # Epoch checkpoints
    for e in range(n_epochs):
        if (e + 1) % checkpoint_every_n == 0:
            name = f"epoch_{e}.pt"
            if not (run_dir / name).exists():
                missing.append(name)

    # Loadability checks
    for pt_file in run_dir.glob("*.pt"):
        try:
            ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
            meta = ckpt.get("metadata", {})
            meta_missing = CHECKPOINT_METADATA_KEYS - set(meta.keys())
            if meta_missing:
                errors.append(f"{pt_file.name}: missing metadata keys {meta_missing}")
        except Exception as exc:
            errors.append(f"{pt_file.name}: load failed: {exc}")

    # Log schema checks
    for log_name in ["train_log.jsonl", "val_log.jsonl"]:
        log_path = run_dir / log_name
        if log_path.exists():
            with open(log_path) as f:
                for i, line in enumerate(f):
                    entry = json.loads(line)
                    entry_missing = LOG_ENTRY_KEYS - set(entry.keys())
                    if entry_missing:
                        errors.append(f"{log_name}:{i}: missing keys {entry_missing}")

    # Run summary loadability
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            for key in ["final_epoch", "best_monitor_value", "config_hash"]:
                if key not in summary:
                    errors.append(f"run_summary.json missing '{key}'")
        except Exception as exc:
            errors.append(f"run_summary.json: load failed: {exc}")

    return {"ok": len(missing) == 0 and len(errors) == 0,
            "missing": missing, "errors": errors}


def compute_run_digest(run_dir: Path) -> str:
    """Compute a deterministic digest of training logs for reproducibility comparison.

    Hashes the content of train_log.jsonl + val_log.jsonl + run_summary.json.
    """
    run_dir = Path(run_dir)
    h = hashlib.sha256()
    for fname in ["train_log.jsonl", "val_log.jsonl", "run_summary.json"]:
        fpath = run_dir / fname
        if fpath.exists():
            h.update(fpath.read_bytes())
    return h.hexdigest()[:16]


def boundary_distance_bucket_stats(
    span_starts: list[int],
    chunk_starts: list[int],
    chunk_ends: list[int],
    scores: list[float],
    n_buckets: int = 4,
) -> list[dict]:
    """Compute per-bucket score statistics by distance from chunk boundary.

    For each span, d_boundary = min(span_start - chunk_start, chunk_end - span_start).
    Spans are grouped into equal-frequency buckets.

    Returns list of dicts with {bucket, d_min, d_max, mean_score, std_score, count}.
    """
    if not span_starts:
        return []

    distances = []
    for s, cs, ce in zip(span_starts, chunk_starts, chunk_ends):
        d = min(s - cs, ce - s)
        distances.append(d)

    # Sort by distance and partition into buckets
    indexed = sorted(zip(distances, scores), key=lambda x: x[0])
    bucket_size = max(1, len(indexed) // n_buckets)

    buckets = []
    for b in range(n_buckets):
        start_idx = b * bucket_size
        end_idx = start_idx + bucket_size if b < n_buckets - 1 else len(indexed)
        if start_idx >= len(indexed):
            break
        bucket_items = indexed[start_idx:end_idx]
        ds = [x[0] for x in bucket_items]
        ss = [x[1] for x in bucket_items]
        buckets.append({
            "bucket": b,
            "d_min": min(ds),
            "d_max": max(ds),
            "mean_score": sum(ss) / len(ss),
            "std_score": float(np.std(ss)) if len(ss) > 1 else 0.0,
            "count": len(ss),
        })

    return buckets


def long_vs_short_comparison(
    protein_lengths: list[int],
    logit_gaps: list[float],
    aucs: list[float | None],
    threshold: int = 1022,
) -> dict:
    """Compare metrics between long (chunked) and short (single-chunk) proteins.

    Returns dict with {short: {n, mean_gap, mean_auc}, long: {n, mean_gap, mean_auc}}.
    """
    short_gaps, long_gaps = [], []
    short_aucs, long_aucs = [], []

    for length, gap, auc in zip(protein_lengths, logit_gaps, aucs):
        if length <= threshold:
            short_gaps.append(gap)
            if auc is not None:
                short_aucs.append(auc)
        else:
            long_gaps.append(gap)
            if auc is not None:
                long_aucs.append(auc)

    def _stats(gaps, aucs_list):
        return {
            "n": len(gaps),
            "mean_gap": sum(gaps) / len(gaps) if gaps else None,
            "mean_auc": sum(aucs_list) / len(aucs_list) if aucs_list else None,
        }

    return {
        "short": _stats(short_gaps, short_aucs),
        "long": _stats(long_gaps, long_aucs),
        "threshold": threshold,
    }
