"""HIMP2 tests: residue label builder + window enumeration + aggregation.

Covers PLAN_EPI_IMP.md §5 HIMP2 TDD Gate:
  1. RED: residue label builder produces values > 1 under binary_coverage when a
     residue is covered by multiple positive spans.
  2. RED: aggregation produces residue scores for residues outside the chunk's
     central region.
  3. RED: aggregation=topk_mean silently falls back instead of raising.
  4. GREEN: binary coverage labels are stable (∈ {0, 1}).
  5. GREEN: aggregation functions match expected outputs on small fixtures.
  6. GREEN: residue scores are defined only on the chunk's central region.

Coordinate convention: spans are 0-based half-open [start, end).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from epitope_head.training.residue_supervision import (
    aggregate_window_logits_to_residues,
    build_residue_labels,
    enumerate_all_residue_windows,
)


def _pos(start: int, length: int) -> dict:
    return {"start_0b": start, "end_0b": start + length, "pep_len": length}


# ── Residue labels: binary coverage + ambiguous + far_bg + central ──────────

class TestResidueLabelBuilder:

    def test_binary_coverage_basic(self):
        positives = [_pos(10, 15)]  # covers [10, 25)
        out = build_residue_labels(
            positives=positives, chunk_len=40, central_start=0,
            near_positive_cfg=None,
        )
        label = out["label"]
        assert label.shape == (40,)
        assert label.dtype == np.int64 or label.dtype == np.int32 or label.dtype == bool or label.dtype == np.uint8
        assert label[0:10].sum() == 0
        assert label[10:25].all()
        assert label[25:].sum() == 0

    # TDD Gate #4: binary coverage stable under duplicates.
    def test_binary_coverage_duplicate_positives_stay_binary(self):
        positives = [_pos(10, 15), _pos(10, 15), _pos(10, 15)]
        out = build_residue_labels(
            positives=positives, chunk_len=40, central_start=0,
            near_positive_cfg=None,
        )
        label = out["label"]
        # All values in {0, 1} regardless of positive duplication.
        assert set(np.unique(label).tolist()) <= {0, 1}
        assert label[10:25].all()

    def test_binary_coverage_overlapping_positives(self):
        positives = [_pos(10, 15), _pos(20, 15)]  # [10,25) ∪ [20,35)
        out = build_residue_labels(
            positives=positives, chunk_len=40, central_start=0,
            near_positive_cfg=None,
        )
        label = out["label"]
        assert label[10:35].all()
        assert label[35:].sum() == 0

    def test_central_mask_respects_margin(self):
        positives = [_pos(10, 15)]
        chunk_len = 40
        margin = 5
        out = build_residue_labels(
            positives=positives, chunk_len=chunk_len, central_start=margin, central_end=chunk_len - margin,
            near_positive_cfg=None,
        )
        central = out["central_mask"]
        assert central.shape == (chunk_len,)
        # central region is [margin, chunk_len - margin)
        assert not central[:margin].any()
        assert central[margin:chunk_len - margin].all()
        assert not central[chunk_len - margin:].any()

    def test_far_bg_excludes_positives_and_ambiguous(self):
        positives = [_pos(20, 10)]  # [20, 30)
        np_cfg = {
            "enabled": True,
            "near_gap_max": 5,
            "apply_to_residue_labels": True,
        }
        out = build_residue_labels(
            positives=positives, chunk_len=50, central_start=0,
            near_positive_cfg=np_cfg,
        )
        label = out["label"]
        ambiguous = out["ambiguous_mask"]
        far_bg = out["far_bg_mask"]

        # Inside positive region: not far_bg, not ambiguous
        assert not far_bg[20:30].any()
        assert not ambiguous[20:30].any()

        # Within near_gap_max of positive boundary but not inside: ambiguous
        # near_gap_max=5 → residues 15..19 (left side) and 30..34 (right side)
        assert ambiguous[15:20].all()
        assert ambiguous[30:35].all()

        # Outside near_gap_max: far_bg
        assert far_bg[:15].all()
        assert far_bg[35:].all()
        # Ambiguous and far_bg are disjoint.
        assert not (ambiguous & far_bg).any()

    def test_apply_to_residue_labels_false_disables_ambiguity(self):
        positives = [_pos(20, 10)]
        np_cfg = {
            "enabled": True,
            "near_gap_max": 5,
            "apply_to_residue_labels": False,
        }
        out = build_residue_labels(
            positives=positives, chunk_len=50, central_start=0,
            near_positive_cfg=np_cfg,
        )
        ambiguous = out["ambiguous_mask"]
        # Ambiguous is empty when apply_to_residue_labels=False.
        assert not ambiguous.any()

    def test_no_positives_all_far_bg(self):
        out = build_residue_labels(
            positives=[], chunk_len=20, central_start=0,
            near_positive_cfg=None,
        )
        assert out["label"].sum() == 0
        assert out["far_bg_mask"].all()
        assert not out["ambiguous_mask"].any()


# ── Window enumeration (window_mode=all) ────────────────────────────────────

class TestEnumerateAllResidueWindows:

    def test_basic_enumeration(self):
        windows = enumerate_all_residue_windows(
            central_start=0, central_end=20,
            min_k=12, max_k=15, max_windows=1024,
        )
        # k=12: starts 0..8 → 9 windows; k=13: 0..7 → 8; k=14: 0..6 → 7; k=15: 0..5 → 6
        assert len(windows) == 9 + 8 + 7 + 6
        for s, e in windows:
            assert 0 <= s
            assert e <= 20
            assert 12 <= e - s <= 15

    def test_central_region_respected(self):
        # Central region [5, 25) → k=12 starts in [5, 13] (e ≤ 25)
        windows = enumerate_all_residue_windows(
            central_start=5, central_end=25,
            min_k=12, max_k=12, max_windows=1024,
        )
        assert len(windows) == 25 - 5 - 12 + 1  # 9
        for s, e in windows:
            assert s >= 5
            assert e <= 25

    def test_max_windows_cap_exceeded_raises(self):
        # k∈[12,25] over [0, 1022) → very large set, far above small cap.
        with pytest.raises(ValueError, match=r"max_windows"):
            enumerate_all_residue_windows(
                central_start=0, central_end=1022,
                min_k=12, max_k=25, max_windows=64,
            )

    def test_empty_when_central_region_too_small(self):
        windows = enumerate_all_residue_windows(
            central_start=0, central_end=5,
            min_k=12, max_k=15, max_windows=1024,
        )
        assert windows == []


# ── Aggregation ─────────────────────────────────────────────────────────────

class TestAggregateWindowLogitsToResidues:

    def test_max_aggregation_basic(self):
        # chunk_len=10, two windows: w0 = [0,5) logit=1.0, w1 = [3,8) logit=2.0
        windows = [(0, 5), (3, 8)]
        logits = torch.tensor([1.0, 2.0])
        scores = aggregate_window_logits_to_residues(
            window_logits=logits, windows=windows, chunk_len=10,
            mode="max", params=None,
        )
        assert scores.shape == (10,)
        # residues 0..2: only w0 → 1.0
        assert torch.all(scores[0:3] == 1.0)
        # residues 3..4: w0 and w1 → max=2.0
        assert torch.all(scores[3:5] == 2.0)
        # residues 5..7: only w1 → 2.0
        assert torch.all(scores[5:8] == 2.0)
        # residues 8..9: not covered → -inf
        assert torch.all(torch.isinf(scores[8:10]) & (scores[8:10] < 0))

    def test_log_mean_exp_uniform_logits_recovers_value(self):
        # All windows on the same residues with same logit value → lme = value.
        windows = [(0, 5), (0, 5), (0, 5)]
        logits = torch.tensor([3.0, 3.0, 3.0])
        scores = aggregate_window_logits_to_residues(
            window_logits=logits, windows=windows, chunk_len=10,
            mode="log_mean_exp", params={"beta": 1.0},
        )
        # All covered residues should be 3.0; uncovered residues -inf.
        for i in range(0, 5):
            assert scores[i].item() == pytest.approx(3.0, abs=1e-6)
        assert torch.all(torch.isinf(scores[5:]) & (scores[5:] < 0))

    def test_log_mean_exp_known_two_value_case(self):
        # logits [0.0, 2.0], beta=1 → lme = log((e^0 + e^2)/2) ≈ log((1+e^2)/2)
        windows = [(0, 4), (0, 4)]
        logits = torch.tensor([0.0, 2.0])
        scores = aggregate_window_logits_to_residues(
            window_logits=logits, windows=windows, chunk_len=10,
            mode="log_mean_exp", params={"beta": 1.0},
        )
        expected = math.log((math.exp(0.0) + math.exp(2.0)) / 2.0)
        for i in range(0, 4):
            assert scores[i].item() == pytest.approx(expected, abs=1e-6)

    # TDD Gate #2: residues outside coverage are explicitly invalidated.
    def test_residue_outside_all_windows_is_neg_inf(self):
        windows = [(2, 5)]
        logits = torch.tensor([1.0])
        scores = aggregate_window_logits_to_residues(
            window_logits=logits, windows=windows, chunk_len=10,
            mode="max", params=None,
        )
        assert torch.all(torch.isinf(scores[0:2]) & (scores[0:2] < 0))
        assert torch.all(scores[2:5] == 1.0)
        assert torch.all(torch.isinf(scores[5:]) & (scores[5:] < 0))

    # TDD Gate #3: topk_mean must fail loudly, not silently fall back.
    def test_topk_mean_aggregation_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="topk_mean"):
            aggregate_window_logits_to_residues(
                window_logits=torch.tensor([1.0]),
                windows=[(0, 5)],
                chunk_len=10,
                mode="topk_mean",
                params=None,
            )

    def test_unknown_aggregation_raises_value_error(self):
        with pytest.raises(ValueError, match="aggregation"):
            aggregate_window_logits_to_residues(
                window_logits=torch.tensor([1.0]),
                windows=[(0, 5)],
                chunk_len=10,
                mode="median",
                params=None,
            )

    def test_gradient_flows_through_lme(self):
        # log_mean_exp must be differentiable through the window logits.
        windows = [(0, 4), (0, 4)]
        logits = torch.tensor([0.0, 2.0], requires_grad=True)
        scores = aggregate_window_logits_to_residues(
            window_logits=logits, windows=windows, chunk_len=5,
            mode="log_mean_exp", params={"beta": 1.0},
        )
        loss = scores[0:4].sum()
        loss.backward()
        assert logits.grad is not None
        # Both window contributions are positive (softmax weights).
        assert (logits.grad > 0).all()

    def test_gradient_flows_through_max(self):
        # max aggregation also needs a path for the winning window.
        windows = [(0, 4), (0, 4)]
        logits = torch.tensor([0.0, 2.0], requires_grad=True)
        scores = aggregate_window_logits_to_residues(
            window_logits=logits, windows=windows, chunk_len=5,
            mode="max", params=None,
        )
        loss = scores[0:4].sum()
        loss.backward()
        assert logits.grad is not None
        # Only w1 (logit=2.0) wins → grad of w1 == 4 (one for each of 4 residues).
        assert logits.grad[1].item() == pytest.approx(4.0, abs=1e-6)
        assert logits.grad[0].item() == pytest.approx(0.0, abs=1e-6)
