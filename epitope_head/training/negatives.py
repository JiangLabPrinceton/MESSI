"""Negative span sampler with positive exclusion and hard/easy ratio.

Implements PLAN.md Task E2 and codemap §7:
  - Global positive exclusion per protein
  - Hard negatives: offset-based near positives
  - Easy negatives: random spans
  - Length sampling: match_positive distribution or uniform
  - Fail-fast on ratio shortfall (strict mode)

HIMP1 extension: when ``near_positive_cfg`` is provided, each returned negative
is annotated with its ``SpanRelation`` and a scalar penalty ``weight`` (both as
parallel ``_relations`` / ``_weights`` lists on the returned ``_NegList``). The
default ``None`` value preserves legacy behavior bit-for-bit.
"""

from __future__ import annotations

import logging

import numpy as np

from epitope_head.training.near_positive import (
    SpanRelation,
    classify_span_relation,
    compute_negative_weight,
)

logger = logging.getLogger(__name__)


class NegativeSamplingShortfall(RuntimeError):
    """Raised when target negative count cannot be met in strict mode."""
    pass


def sample_negatives(
    protein_length: int,
    positives: list[dict],
    neg_ratio: int = 15,
    hard_negative_fraction: float = 0.3,
    hard_neg_max_overlap_ratio: float = 0.8,
    hard_neg_offset_range: int = 20,
    neg_length_sampling: str = "match_positive",
    min_k: int = 12,
    max_k: int = 25,
    rng: np.random.RandomState | None = None,
    strict: bool = True,
    disrupted_spans: list[dict] | None = None,
    near_positive_cfg: dict | None = None,
) -> list[dict]:
    """Sample negative spans for one protein.

    Args:
        protein_length: full protein length L.
        positives: list of {start_0b, end_0b, pep_len, ...} dicts.
        neg_ratio: negatives per positive.
        hard_negative_fraction: fraction of negatives that are hard.
        hard_neg_max_overlap_ratio: hard negs must have overlap < this * pos_len.
        hard_neg_offset_range: max offset for hard negative generation.
        neg_length_sampling: "match_positive" or "uniform".
        min_k, max_k: allowed peptide length range.
        rng: random state for reproducibility.
        strict: if True, raise NegativeSamplingShortfall when target count
            or hard/easy ratio cannot be met. If False, backfill hard
            shortfall with easy negatives and log a warning.
        disrupted_spans: optional list of disrupted span dicts from runtime
            augmentation. These are guaranteed hard negatives that fill the
            hard quota first before offset-based sampling.
        near_positive_cfg: optional dict enabling HIMP1 relation classification
            and per-negative weights. Recognized keys:
              - ``near_gap_max`` (int): end-point distance threshold.
              - ``schedule`` (str): one of ``ignore``, ``linear_clamp``, ``sigmoid``.
              - ``schedule_params`` (dict): kwargs forwarded to the schedule fn.
              - ``metric`` (str): currently only ``endpoint_gap`` supported.
            When ``None``, the returned ``_NegList`` does not carry relation /
            weight metadata and behavior is identical to legacy callers.

    Returns:
        List of negative span dicts {start_0b, end_0b, pep_len}.

    Raises:
        NegativeSamplingShortfall: in strict mode, when total count is not met
            or hard negative target is not met.
    """
    if rng is None:
        rng = np.random.RandomState()

    if not positives:
        return []

    # Build positive exclusion set
    positive_set = {(p["start_0b"], p["end_0b"]) for p in positives}

    # Build positive length distribution for match_positive sampling
    pos_lengths = [p["pep_len"] for p in positives]

    total_neg = neg_ratio * len(positives)
    n_hard = int(round(total_neg * hard_negative_fraction))
    n_easy = total_neg - n_hard

    hard_negatives = []
    hard_sources: list[str] = []  # "disrupted" | "offset" — for relation tagging
    easy_negatives = []

    # --- Disrupted spans fill hard quota first (Stage J / HIMP1) ---
    # Counterfactual_disrupted spans are mutation-derived pseudo-negatives.
    # PLAN_EPI_IMP §5 HIMP1 #3 mandates that they are preserved with weight
    # 1.0 even when their coordinates overlap a remaining WT positive — the
    # mutation-derived evidence overrides the WT label. (In the current
    # pipeline ``positives`` is already the post-mutation list with
    # disrupted spans removed, so an exact coordinate clash is rare; the
    # filter is removed here to make the contract robust to any caller
    # that supplies WT positives instead.)
    n_disrupted_used = 0
    if disrupted_spans:
        for ds in disrupted_spans:
            if len(hard_negatives) >= n_hard:
                break
            hard_negatives.append({
                "start_0b": ds["start_0b"],
                "end_0b": ds["end_0b"],
                "pep_len": ds["pep_len"],
            })
            hard_sources.append("disrupted")
            n_disrupted_used += 1

    # --- Hard negatives: offset from random positive (fill remaining quota) ---
    hard_attempts = 0
    max_hard_attempts = n_hard * 20
    while len(hard_negatives) < n_hard and hard_attempts < max_hard_attempts:
        hard_attempts += 1
        anchor = positives[rng.randint(len(positives))]
        a_start = anchor["start_0b"]
        a_end = anchor["end_0b"]
        a_len = anchor["pep_len"]

        offset = rng.randint(1, hard_neg_offset_range + 1)
        if rng.random() < 0.5:
            offset = -offset

        new_start = a_start + offset
        new_end = new_start + a_len

        if new_start < 0 or new_end > protein_length:
            continue

        overlap_start = max(a_start, new_start)
        overlap_end = min(a_end, new_end)
        overlap = max(0, overlap_end - overlap_start)
        if overlap >= hard_neg_max_overlap_ratio * a_len:
            continue

        if (new_start, new_end) in positive_set:
            continue

        hard_negatives.append({
            "start_0b": new_start,
            "end_0b": new_end,
            "pep_len": a_len,
        })
        hard_sources.append("offset")

    # Check hard negative shortfall
    hard_shortfall = n_hard - len(hard_negatives)
    if hard_shortfall > 0:
        if strict:
            raise NegativeSamplingShortfall(
                f"Hard negative shortfall: wanted {n_hard}, got {len(hard_negatives)} "
                f"(protein_length={protein_length}, n_positives={len(positives)})"
            )
        else:
            logger.warning(
                "Hard negative shortfall: wanted %d, got %d for protein_length=%d. "
                "Backfilling %d with easy negatives.",
                n_hard, len(hard_negatives), protein_length, hard_shortfall,
            )
            # Backfill: increase easy target to compensate
            n_easy += hard_shortfall

    # --- Easy negatives: random spans ---
    easy_attempts = 0
    max_easy_attempts = n_easy * 20
    while len(easy_negatives) < n_easy and easy_attempts < max_easy_attempts:
        easy_attempts += 1

        if neg_length_sampling == "match_positive":
            k = pos_lengths[rng.randint(len(pos_lengths))]
        else:
            k = rng.randint(min_k, max_k + 1)

        if k > protein_length:
            continue

        start = rng.randint(0, protein_length - k + 1)
        end = start + k

        if (start, end) in positive_set:
            continue

        easy_negatives.append({
            "start_0b": start,
            "end_0b": end,
            "pep_len": k,
        })

    negatives = hard_negatives + easy_negatives

    # Check total shortfall
    total_shortfall = total_neg - len(negatives)
    if total_shortfall > 0:
        if strict:
            raise NegativeSamplingShortfall(
                f"Total negative shortfall: wanted {total_neg}, got {len(negatives)} "
                f"(protein_length={protein_length}, n_positives={len(positives)})"
            )
        else:
            logger.warning(
                "Total negative shortfall: wanted %d, got %d for protein_length=%d.",
                total_neg, len(negatives), protein_length,
            )

    # Attach disrupted-fill stats via subclass (preserves list interface)
    class _NegList(list):
        pass
    result = _NegList(negatives)
    result._n_disrupted_used = n_disrupted_used
    result._n_hard_target = n_hard
    result._n_total_target = total_neg

    # HIMP1: when near-positive config is supplied, attach relation/weight
    # metadata as parallel lists. Source-tagged negatives override classifier
    # output so disrupted spans always carry COUNTERFACTUAL_DISRUPTED + w=1.0.
    if near_positive_cfg is not None:
        np_cfg = near_positive_cfg
        near_gap_max = int(np_cfg["near_gap_max"])
        schedule_name = str(np_cfg["schedule"])
        schedule_params = np_cfg.get("schedule_params") or {}

        positives_xy = [(p["start_0b"], p["end_0b"]) for p in positives]

        relations: list[SpanRelation] = []
        weights: list[float] = []

        # Hard block: index parallel to hard_sources.
        for idx, neg in enumerate(hard_negatives):
            if idx < len(hard_sources) and hard_sources[idx] == "disrupted":
                rel = SpanRelation.COUNTERFACTUAL_DISRUPTED
                gap = 0
            else:
                rel, gap = classify_span_relation(
                    (neg["start_0b"], neg["end_0b"]),
                    positives_xy,
                    near_gap_max=near_gap_max,
                )
            relations.append(rel)
            weights.append(
                compute_negative_weight(
                    rel, gap, near_gap_max, schedule_name, schedule_params,
                )
            )

        # Easy block: never disrupted.
        for neg in easy_negatives:
            rel, gap = classify_span_relation(
                (neg["start_0b"], neg["end_0b"]),
                positives_xy,
                near_gap_max=near_gap_max,
            )
            relations.append(rel)
            weights.append(
                compute_negative_weight(
                    rel, gap, near_gap_max, schedule_name, schedule_params,
                )
            )

        result._relations = relations
        result._weights = weights

    return result
