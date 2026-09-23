"""Window/span geometry for the Wave-3 IoU-aligned span objective.

Vectorized, torch-friendly half-open-span IoU, matching the convention used by
the benchmark metric (``scripts/benchmark_iedb_test.py::_span_iou``): a span is a
half-open interval ``[start, end)``; ``IoU = overlap / union`` and ``0`` on no
overlap. These helpers feed the window IoU-ranking loss (``losses.window_iou_rank_loss``)
so training optimizes the same region-overlap quantity the M6 IoU ladder measures.
"""
from __future__ import annotations

import torch


def window_iou_matrix(windows: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    """IoU between every window and every positive span (half-open).

    Args:
        windows: ``[W, 2]`` long/int tensor of ``[start, end)`` per candidate window.
        positives: ``[P, 2]`` long/int tensor of GT ``[start, end)`` spans.

    Returns:
        ``[W, P]`` float tensor of IoU in ``[0, 1]``. Empty (W==0 or P==0) returns
        a correctly-shaped zero tensor.
    """
    W = windows.shape[0]
    P = positives.shape[0]
    if W == 0 or P == 0:
        return torch.zeros((W, P), dtype=torch.float32, device=windows.device)

    w = windows.to(torch.float32)
    p = positives.to(torch.float32)
    ws, we = w[:, 0].unsqueeze(1), w[:, 1].unsqueeze(1)   # [W,1]
    ps, pe = p[:, 0].unsqueeze(0), p[:, 1].unsqueeze(0)   # [1,P]

    inter = (torch.minimum(we, pe) - torch.maximum(ws, ps)).clamp_min(0.0)   # [W,P]
    len_w = (we - ws).clamp_min(0.0)
    len_p = (pe - ps).clamp_min(0.0)
    union = (len_w + len_p - inter).clamp_min(1e-9)
    iou = torch.where(inter > 0, inter / union, torch.zeros_like(inter))
    return iou


def max_iou_per_window(windows: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    """Per-window max IoU over all positive spans (``[W]``, 0 if no positives)."""
    W = windows.shape[0]
    if W == 0:
        return torch.zeros((0,), dtype=torch.float32, device=windows.device)
    if positives.shape[0] == 0:
        return torch.zeros((W,), dtype=torch.float32, device=windows.device)
    return window_iou_matrix(windows, positives).max(dim=1).values
