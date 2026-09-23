"""Utilities for flank ablation inference experiments."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Iterator

import pandas as pd
import torch


CNN_ENCODER_TYPES = frozenset({"dilated_cnn", "multiscale_cnn"})


def binary_auc_from_scores(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> float | None:
    """Pairwise binary AUC: P(pos > neg) + 0.5 * P(pos == neg)."""
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return None
    cmp = (pos_scores.unsqueeze(1) > neg_scores.unsqueeze(0)).float()
    tie = (pos_scores.unsqueeze(1) == neg_scores.unsqueeze(0)).float()
    return float((cmp + 0.5 * tie).mean().item())


@contextlib.contextmanager
def flank_ablation_context(span_feature_builder) -> Iterator[None]:
    """Temporarily replace all flank features with pad_left/pad_right.

    This enforces inference-time ablation where every span is treated as
    if it were at both boundaries.
    """
    original_forward = span_feature_builder.forward
    d_proj = int(span_feature_builder.d_proj)

    def _ablated_forward(G, spans, chunk_len, allele_idx):
        phi = original_forward(G, spans, chunk_len, allele_idx)
        if phi.numel() == 0:
            return phi
        phi = phi.clone()
        n = phi.shape[0]
        left = span_feature_builder.pad_left.unsqueeze(0).expand(n, d_proj)
        right = span_feature_builder.pad_right.unsqueeze(0).expand(n, d_proj)
        phi[:, 3 * d_proj: 4 * d_proj] = left
        phi[:, 4 * d_proj: 5 * d_proj] = right
        return phi

    span_feature_builder.forward = _ablated_forward
    try:
        yield
    finally:
        span_feature_builder.forward = original_forward


def load_split_proteins_for_pp_auc(
    samples_path: Path | str,
    split_ids_path: Path | str,
    max_proteins: int | None = None,
) -> list[dict]:
    """Load protein sequences and positive span sets for pp_AUC evaluation."""
    samples_path = Path(samples_path)
    split_ids_path = Path(split_ids_path)

    split_ids = []
    with open(split_ids_path) as f:
        for line in f:
            pid = line.strip()
            if pid:
                split_ids.append(pid)
    split_id_set = set(split_ids)

    df = pd.read_parquet(samples_path, columns=["protein_id", "protein_seq", "positives_json"])
    df = df[df["protein_id"].isin(split_id_set)].copy()
    # deterministic order: split id order first, fallback by protein_id
    rank = {pid: i for i, pid in enumerate(split_ids)}
    df["_rank"] = df["protein_id"].map(rank).fillna(10**9).astype(int)
    df = df.sort_values(by=["_rank", "protein_id"]).drop(columns=["_rank"])

    records = []
    for _, row in df.iterrows():
        positives = json.loads(row["positives_json"]) if row["positives_json"] else []
        pos_set = {(int(p["start_0b"]), int(p["end_0b"])) for p in positives}
        records.append(
            {
                "protein_id": str(row["protein_id"]),
                "sequence": str(row["protein_seq"]),
                "positive_spans": pos_set,
            },
        )
        if max_proteins is not None and len(records) >= max_proteins:
            break

    return records


def pp_auc_from_prediction(
    window_logits: list[dict],
    positive_spans: set[tuple[int, int]],
) -> float | None:
    """Compute protein-level pairwise AUC from predicted windows."""
    if not window_logits:
        return None
    pos_scores = []
    neg_scores = []
    for w in window_logits:
        span = (int(w["start_0b"]), int(w["end_0b"]))
        z = float(w["z"])
        if span in positive_spans:
            pos_scores.append(z)
        else:
            neg_scores.append(z)

    if not pos_scores or not neg_scores:
        return None
    return binary_auc_from_scores(
        torch.tensor(pos_scores, dtype=torch.float32),
        torch.tensor(neg_scores, dtype=torch.float32),
    )


def resolve_cnn_variant_profile(ablation_cfg: dict, variant_id: str) -> dict:
    """Resolve one ablation profile and enforce CNN-only encoder type."""
    profiles = ablation_cfg.get("profiles", {})
    if variant_id not in profiles:
        raise ValueError(f"Unknown ablation variant_id: {variant_id}")

    profile = profiles[variant_id]
    encoder_type = profile.get("encoder_type")
    if encoder_type not in CNN_ENCODER_TYPES:
        raise ValueError(
            f"Variant '{variant_id}' is not a CNN profile (encoder_type={encoder_type}). "
            f"Allowed CNN encoder types: {sorted(CNN_ENCODER_TYPES)}",
        )
    return profile
