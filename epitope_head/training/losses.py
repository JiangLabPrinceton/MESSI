"""Training losses: InfoNCE primary + optional multi-positive and smoothness.

Implements PLAN.md Task E4:
  - Intra-protein InfoNCE with temperature tau (stable log-sum-exp)
  - Optional multi-positive softmax branch (lambda_mp, disabled in v0)
  - Optional smoothness regularizer (lambda_smooth, disabled in v0)
  - Decomposed loss dict for logging
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _stable_log_sum_exp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable log-sum-exp: log(sum(exp(x))) = max + log(sum(exp(x - max)))."""
    x_max = x.max(dim=dim, keepdim=True).values
    return x_max.squeeze(dim) + torch.log(torch.exp(x - x_max).sum(dim=dim))


def info_nce_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    tau: float = 0.1,
    neg_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Intra-protein InfoNCE loss with optional per-negative weighting.

    Default (``neg_weights=None``) evaluates:

        L = -log(exp(z_pos/tau) / (exp(z_pos/tau) + sum_j exp(z_neg_j/tau)))

    With weights ``w_j ∈ [0, 1]``, the partition is reweighted:

        L = -log(exp(z_pos/tau) / (exp(z_pos/tau) + sum_j w_j * exp(z_neg_j/tau)))

    Equivalently each negative's logit is shifted by ``log(w_j)`` before
    log-sum-exp, with ``w_j == 0`` producing ``-inf`` (negative excluded).

    Args:
        pos_logits: ``[P]`` logits for positive spans.
        neg_logits: ``[N]`` logits for negative spans.
        tau: temperature (default 0.1).
        neg_weights: optional ``[N]`` non-negative weights. ``None`` recovers
            legacy behavior bit-for-bit.

    Returns:
        Scalar loss averaged over positives.
    """

    # check tau is valid:
    if tau <= 0.0:
        raise ValueError("tau must be positive")

    if pos_logits.numel() == 0:
        return torch.tensor(0.0, device=pos_logits.device, requires_grad=True)

    P = pos_logits.shape[0]

    # Scale by temperature
    pos_scaled = pos_logits / tau  # [P]
    neg_scaled = neg_logits / tau  # [N]

    # Apply per-negative weights via log-shift (None → no-op).
    if neg_weights is not None:
        if neg_weights.shape[0] != neg_scaled.shape[0]:
            raise ValueError(
                f"neg_weights shape {tuple(neg_weights.shape)} does not match "
                f"neg_logits shape {tuple(neg_scaled.shape)}"
            )
        # log(w) with w==0 → -inf so that contribution vanishes inside logsumexp.
        eps = torch.finfo(neg_scaled.dtype).tiny
        neg_w = neg_weights.to(neg_scaled.device).to(neg_scaled.dtype)
        log_w = torch.where(
            neg_w > 0,
            torch.log(neg_w.clamp_min(eps)),
            torch.full_like(neg_w, float("-inf")),
        )
        neg_scaled = neg_scaled + log_w

    losses = torch.zeros(P, device=pos_logits.device)
    for i in range(P):
        all_logits = torch.cat([pos_scaled[i:i+1], neg_scaled])  # [1 + N]
        log_denom = _stable_log_sum_exp(all_logits, dim=0)
        losses[i] = -pos_scaled[i] + log_denom

    return losses.mean()


def multi_positive_loss(
    pos_logits: torch.Tensor,
    T_mp: float = 0.1,
) -> torch.Tensor:
    """Multi-positive softmax loss (optional, disabled in v0).

    When multiple positives exist for a chunk, encourages uniform
    confidence across them via entropy-like regularization.

    L_mp = -mean_i(log(softmax(pos_logits / T_mp)[i]))

    Args:
        pos_logits: [P] logits for positive spans (P >= 2 for meaningful loss).
        T_mp: temperature for multi-positive softmax.

    Returns:
        Scalar loss (0 if P < 2).
    """
    if T_mp <= 0.0:
        raise ValueError("T_mp must be positive")
    if pos_logits.numel() < 2:
        return torch.tensor(0.0, device=pos_logits.device, requires_grad=True)

    log_probs = F.log_softmax(pos_logits / T_mp, dim=0)  # [P]
    return -log_probs.mean()


def smoothness_loss(
    per_residue_scores: torch.Tensor,
    chunk_len: int,
) -> torch.Tensor:
    """Smoothness regularizer on per-residue score profile (optional, disabled in v0).

    Penalizes large differences between adjacent residue scores.
    L_smooth = mean((s[i+1] - s[i])^2)

    Args:
        per_residue_scores: [L] score per residue in chunk.
        chunk_len: actual residue count.

    Returns:
        Scalar loss (0 if chunk_len < 2).
    """
    if chunk_len < 2:
        return torch.tensor(0.0, device=per_residue_scores.device, requires_grad=True)

    valid = per_residue_scores[:chunk_len]
    diffs = valid[1:] - valid[:-1]
    return (diffs ** 2).mean()


def margin_hard_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    margin_m: float = 0.5,
    hard_topk: int = 8,
    neg_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Margin-based hard-example loss with optional per-negative weighting.

    Default (``neg_weights=None``) selects top-k hardest negatives by raw
    logit and averages ``relu(m - gap)`` over all ``[P, k]`` pairs.

    With weights, negatives carrying ``w == 0`` are excluded *before* the
    top-k selection (treated as if absent). Among the surviving negatives,
    the resulting ``[P, k]`` pair losses are weighted-averaged by the
    selected negatives' weights, so partial-weight negatives contribute
    proportionally to their schedule output.

    Args:
        pos_logits: ``[P]`` positive logits.
        neg_logits: ``[N]`` negative logits.
        margin_m: hinge margin (default 0.5).
        hard_topk: number of hardest negatives per positive (default 8).
        neg_weights: optional ``[N]`` non-negative weights. ``None`` recovers
            legacy behavior bit-for-bit.

    Returns:
        Scalar margin loss (0 if no positives or no surviving negatives).
    """
    if hard_topk < 1:
        raise ValueError("hard_topk must be >= 1")

    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return torch.tensor(0.0, device=pos_logits.device, requires_grad=True)

    if neg_weights is not None:
        if neg_weights.shape[0] != neg_logits.shape[0]:
            raise ValueError(
                f"neg_weights shape {tuple(neg_weights.shape)} does not match "
                f"neg_logits shape {tuple(neg_logits.shape)}"
            )
        # Filter out fully-ignored negatives (w == 0) before topk selection.
        keep = neg_weights > 0
        if not keep.any():
            return torch.tensor(0.0, device=pos_logits.device, requires_grad=True)
        neg_logits_f = neg_logits[keep]
        neg_w_f = neg_weights[keep].to(neg_logits_f.dtype).to(neg_logits_f.device)

        k = min(hard_topk, neg_logits_f.numel())
        hard_neg, top_idx = torch.topk(neg_logits_f, k)  # [k]
        top_w = neg_w_f[top_idx]                          # [k]

        gaps = pos_logits.unsqueeze(1) - hard_neg.unsqueeze(0)  # [P, k]
        pair_loss = F.relu(margin_m - gaps)                     # [P, k]
        weighted = pair_loss * top_w.unsqueeze(0)               # [P, k]
        denom = top_w.sum().clamp_min(torch.finfo(weighted.dtype).tiny) * pos_logits.numel()
        return weighted.sum() / denom

    # Legacy path
    k = min(hard_topk, neg_logits.numel())
    hard_neg, _ = torch.topk(neg_logits, k)  # [k], highest logits

    # Pairwise margin: [P, k]
    gaps = pos_logits.unsqueeze(1) - hard_neg.unsqueeze(0)  # [P, k]
    losses = F.relu(margin_m - gaps)  # [P, k]

    return losses.mean()


def exact_margin_loss(
    pos_exact_logits: torch.Tensor,
    neg_exact_logits: torch.Tensor,
    margin_m: float = 0.3,
    hard_topk: int = 8,
) -> torch.Tensor:
    """Wave-4 dual-head exact objective on the ``z_exact`` readout.

    Ranks exact-match positive windows above the top-k hardest negatives via
    ``relu(margin_m - (z_pos - z_neg))``, reusing the margin-hard machinery
    (no ``neg_weights`` — the exact head sits downstream of the region negative
    schedule). Returns a differentiable ``0`` when positives or negatives are
    empty. This term is added to the total at weight ``lambda_exact``; because
    ``z_exact = stopgrad(z_region) + boundary_head(detach(phi))`` its gradient
    reaches only the boundary head, never the landscape trunk.
    """
    return margin_hard_loss(
        pos_exact_logits, neg_exact_logits,
        margin_m=margin_m, hard_topk=hard_topk, neg_weights=None,
    )


def residue_pairwise_margin_loss(
    residue_scores: torch.Tensor,
    label: torch.Tensor,
    far_bg_mask: torch.Tensor,
    central_mask: torch.Tensor,
    margin_m: float = 0.5,
    min_far_bg: int = 4,
) -> tuple[torch.Tensor, dict]:
    """Pairwise margin hinge ranking loss over residue scores (HIMP3).

    For each ``(positive_residue, far_bg_residue)`` pair restricted to the
    chunk's central region, accumulates ``relu(margin_m - (s_pos - s_neg))``
    and returns the mean. Residues that are not covered by any window
    (``-inf`` scores) MUST already be excluded from ``far_bg_mask`` by the
    caller — this function does not silently drop them.

    Args:
        residue_scores: ``[L]`` per-residue scores (output of
            ``aggregate_window_logits_to_residues``).
        label: ``[L]`` 0/1 (uint8 / bool / int) binary coverage labels.
        far_bg_mask: ``[L]`` bool, residues eligible as ranking negatives.
        central_mask: ``[L]`` bool, residues inside the chunk's central
            region (boundary residues are excluded from training pairs).
        margin_m: hinge margin ``m`` (default 0.5).
        min_far_bg: chunks with fewer central far-bg residues than this are
            **skipped** (loss=0, ``meta["skipped"]=True``) so per-chunk
            pair counts stay statistically meaningful in dense alleles.

    Returns:
        ``(loss, meta)`` with ``meta`` containing:
          - ``skipped``: bool, True if loss is degenerate (no pairs).
          - ``n_pos``: int, central-region positive residue count.
          - ``n_far_bg``: int, central-region far-bg residue count.
          - ``n_pairs``: int, number of pairs that contributed to the loss.
    """
    device = residue_scores.device

    label_bool = (label > 0) if label.dtype != torch.bool else label
    pos_in_central = label_bool & central_mask
    neg_in_central = far_bg_mask & central_mask

    n_pos = int(pos_in_central.sum().item())
    n_far_bg = int(neg_in_central.sum().item())

    meta = {
        "skipped": False,
        "n_pos": n_pos,
        "n_far_bg": n_far_bg,
        "n_pairs": 0,
    }

    # Skip when either side is empty or far-bg below threshold.
    if n_pos == 0 or n_far_bg < int(min_far_bg):
        meta["skipped"] = True
        return torch.zeros((), device=device, requires_grad=False), meta

    pos_scores = residue_scores[pos_in_central]   # [n_pos]
    neg_scores = residue_scores[neg_in_central]   # [n_far_bg]

    # Pairwise gaps: pos[i] - neg[j] → [n_pos, n_far_bg]
    gaps = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
    losses = torch.relu(float(margin_m) - gaps)
    loss = losses.mean()
    meta["n_pairs"] = n_pos * n_far_bg
    return loss, meta


def window_iou_rank_loss(
    logits: torch.Tensor,
    ious: torch.Tensor,
    margin_m: float = 0.5,
    min_iou_gap: float = 0.1,
    weight_by_gap: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Pairwise margin ranking over candidate windows by IoU-to-GT (Wave-3).

    For every ordered pair ``(i, j)`` with ``ious[i] - ious[j] >= min_iou_gap``,
    enforces ``logits[i] >= logits[j] + margin_m`` via
    ``relu(margin_m - (logits[i] - logits[j]))``. This trains the span scorer to
    rank candidate windows by region overlap — the exact quantity the M6 IoU
    ladder measures — actively pulling high-IoU (overlapping) windows *above*
    low-IoU / far windows, rather than merely down-weighting them. Pairs with an
    IoU separation below ``min_iou_gap`` (ambiguous near-ties) are excluded.

    Args:
        logits: ``[W]`` candidate-window span-scorer logits.
        ious: ``[W]`` per-window max IoU to any GT span, in ``[0, 1]``.
        margin_m: hinge margin separating higher-IoU from lower-IoU windows.
        min_iou_gap: minimum IoU difference for a pair to contribute.
        weight_by_gap: weight each pair by its IoU gap (larger gaps dominate).

    Returns:
        ``(loss, meta)`` with ``meta["n_pairs"]``. Degenerate cases (``W < 2`` or
        no qualifying pair) return a differentiable scalar ``0``.
    """
    device = logits.device
    meta = {"n_pairs": 0}
    if logits.shape[0] < 2:
        return torch.zeros((), device=device, requires_grad=True), meta

    diff_iou = ious.unsqueeze(1) - ious.unsqueeze(0)        # [W, W], i vs j
    mask = diff_iou >= float(min_iou_gap)                   # i should out-rank j
    n_pairs = int(mask.sum().item())
    if n_pairs == 0:
        return torch.zeros((), device=device, requires_grad=True), meta

    gap_logit = logits.unsqueeze(1) - logits.unsqueeze(0)   # [W, W], want > 0
    pair_loss = torch.relu(float(margin_m) - gap_logit)
    pl = pair_loss[mask]
    if weight_by_gap:
        w = diff_iou[mask].to(pl.dtype)
        loss = (pl * w).sum() / w.sum().clamp_min(torch.finfo(pl.dtype).tiny)
    else:
        loss = pl.mean()
    meta["n_pairs"] = n_pairs
    return loss, meta


def compute_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    tau: float = 0.1,
    T_mp: float = 0.1,
    lambda_mp: float = 0.0,
    lambda_smooth: float = 0.0,
    per_residue_scores: torch.Tensor | None = None,
    chunk_len: int = 0,
    objective_mode: str = "infonce",
    margin_m: float = 0.5,
    hard_topk: int = 8,
    lambda_margin: float = 0.5,
    neg_weights: torch.Tensor | None = None,
    window_logits: torch.Tensor | None = None,
    window_ious: torch.Tensor | None = None,
    lambda_iou_rank: float = 0.0,
    iou_rank_margin: float = 0.5,
    iou_rank_min_gap: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute total loss with decomposed terms.

    Args:
        pos_logits: [P] positive span logits.
        neg_logits: [N] negative span logits.
        tau: InfoNCE temperature.
        T_mp: multi-positive temperature.
        lambda_mp: weight for multi-positive term (0 = disabled).
        lambda_smooth: weight for smoothness term (0 = disabled).
        per_residue_scores: [L] optional residue scores (needed if lambda_smooth > 0).
        chunk_len: actual residue count (needed if lambda_smooth > 0).
        objective_mode: "infonce" | "mixed_margin" | "margin_only".
        margin_m: margin threshold for hard-example loss.
        hard_topk: number of hardest negatives per positive.
        lambda_margin: weight for margin term in mixed_margin mode.

    Returns:
        Dict with keys:
          loss_total: scalar, the backprop target.
          loss_intra: scalar, InfoNCE term.
          loss_mp: scalar, multi-positive term (0 if disabled).
          loss_smooth: scalar, smoothness term (0 if disabled).
          loss_margin: scalar, margin hard-example term (0 if infonce mode).
    """
    device = pos_logits.device

    # InfoNCE (always computed unless margin_only)
    if objective_mode in ("infonce", "mixed_margin"):
        loss_intra = info_nce_loss(pos_logits, neg_logits, tau=tau, neg_weights=neg_weights)
    else:
        loss_intra = torch.tensor(0.0, device=device)

    # Margin hard-example loss
    if objective_mode in ("mixed_margin", "margin_only"):
        loss_margin = margin_hard_loss(
            pos_logits, neg_logits,
            margin_m=margin_m, hard_topk=hard_topk,
            neg_weights=neg_weights,
        )
    else:
        loss_margin = torch.tensor(0.0, device=device)

    # Window IoU-ranking auxiliary (Wave-3): rank candidate windows by IoU-to-GT.
    if lambda_iou_rank > 0.0 and window_logits is not None and window_ious is not None:
        loss_iou_rank, _ = window_iou_rank_loss(
            window_logits, window_ious,
            margin_m=iou_rank_margin, min_iou_gap=iou_rank_min_gap,
        )
    else:
        loss_iou_rank = torch.tensor(0.0, device=device)

    # Multi-positive (skip computation when disabled)
    if lambda_mp > 0.0:
        loss_mp = multi_positive_loss(pos_logits, T_mp=T_mp)
    else:
        loss_mp = torch.tensor(0.0, device=device)

    # Smoothness (skip computation when disabled)
    if lambda_smooth > 0.0 and per_residue_scores is not None:
        loss_smooth = smoothness_loss(per_residue_scores, chunk_len)
    elif lambda_smooth > 0.0 and per_residue_scores is None:
        raise ValueError("per_residue_scores is required when lambda_smooth > 0.0")
    else:
        loss_smooth = torch.tensor(0.0, device=device)

    # Total loss by mode
    if objective_mode == "iou_rank_only":
        # Span objective is PURE IoU-graded window ranking: InfoNCE + margin OFF.
        # Fail-fast — a missing ranking signal would silently train the span
        # scorer on nothing (the residue landscape loss is added separately by
        # the trainer and is unaffected by this mode).
        if lambda_iou_rank <= 0.0 or window_logits is None or window_ious is None:
            raise ValueError(
                "objective_mode='iou_rank_only' requires lambda_iou_rank>0 and "
                "window_logits/window_ious (the span objective has nothing to rank)."
            )
        loss_total = lambda_mp * loss_mp + lambda_smooth * loss_smooth
    elif objective_mode == "infonce":
        loss_total = loss_intra + lambda_mp * loss_mp + lambda_smooth * loss_smooth
    elif objective_mode == "mixed_margin":
        loss_total = loss_intra + lambda_margin * loss_margin + lambda_mp * loss_mp + lambda_smooth * loss_smooth
    elif objective_mode == "margin_only":
        loss_total = loss_margin + lambda_mp * loss_mp + lambda_smooth * loss_smooth
    else:
        raise ValueError(f"Unknown objective_mode: {objective_mode}")

    # Additive Wave-3 term (no-op when lambda_iou_rank == 0 -> legacy bit-for-bit;
    # the SOLE span term in iou_rank_only mode).
    loss_total = loss_total + lambda_iou_rank * loss_iou_rank

    return {
        "loss_total": loss_total,
        "loss_intra": loss_intra,
        "loss_mp": loss_mp,
        "loss_smooth": loss_smooth,
        "loss_margin": loss_margin,
        "loss_iou_rank": loss_iou_rank,
    }
