"""Module I loss contract tests.

Covers:
  I1: margin_hard_loss function
  I2: compute_loss with objective_mode variants
  I3: trainer schema updates for loss_margin
"""

from __future__ import annotations

import pytest
import torch

from epitope_head.training.losses import compute_loss, margin_hard_loss
from epitope_head.training.trainer import (
    LOG_ENTRY_KEYS,
    StepMetrics,
    normalize_loss_cfg,
)


# ── I1: margin_hard_loss ──────────────────────────────────────────────────────

class TestI1MarginHardLoss:
    """Tests for margin_hard_loss function."""

    def test_all_gaps_above_margin_returns_zero(self):
        """When all pos-neg gaps exceed margin, loss should be 0."""
        pos = torch.tensor([2.0, 3.0])
        neg = torch.tensor([-1.0, -2.0, -3.0])
        loss = margin_hard_loss(pos, neg, margin_m=0.5, hard_topk=2)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_gap_below_margin_positive_loss(self):
        """When gap < margin, loss should be positive and match manual calc."""
        pos = torch.tensor([1.0])
        neg = torch.tensor([0.8])  # gap = 0.2, margin = 0.5 → relu(0.3) = 0.3
        loss = margin_hard_loss(pos, neg, margin_m=0.5, hard_topk=1)
        assert loss.item() == pytest.approx(0.3, abs=1e-6)

    def test_topk_greater_than_n_neg_fallback(self):
        """hard_topk > N_neg should use all negatives without error."""
        pos = torch.tensor([1.0])
        neg = torch.tensor([0.9])  # only 1 neg, topk=8
        loss = margin_hard_loss(pos, neg, margin_m=0.5, hard_topk=8)
        # gap = 0.1, loss = relu(0.5 - 0.1) = 0.4
        assert loss.item() == pytest.approx(0.4, abs=1e-6)

    @pytest.mark.parametrize("hard_topk", [0, -1])
    def test_invalid_hard_topk_raises(self, hard_topk):
        """hard_topk must be >= 1."""
        pos = torch.tensor([1.0])
        neg = torch.tensor([0.9, 0.8])
        with pytest.raises(ValueError, match="hard_topk must be >= 1"):
            margin_hard_loss(pos, neg, margin_m=0.5, hard_topk=hard_topk)

    def test_topk_selects_hardest_negatives(self):
        """Top-k should select the highest-scoring negatives."""
        pos = torch.tensor([1.0])
        neg = torch.tensor([0.0, 0.5, 0.9, 0.1])  # hardest = 0.9, 0.5
        loss_k2 = margin_hard_loss(pos, neg, margin_m=0.5, hard_topk=2)
        # With k=2: hardest = [0.9, 0.5]
        # gaps = [0.1, 0.5] → relu(0.5-0.1, 0.5-0.5) = [0.4, 0.0] → mean = 0.2
        assert loss_k2.item() == pytest.approx(0.2, abs=1e-6)

    def test_empty_pos_returns_zero(self):
        """No positives → loss = 0."""
        pos = torch.tensor([])
        neg = torch.tensor([1.0, 2.0])
        loss = margin_hard_loss(pos, neg, margin_m=0.5)
        assert loss.item() == 0.0

    def test_empty_neg_returns_zero(self):
        """No negatives → loss = 0."""
        pos = torch.tensor([1.0])
        neg = torch.tensor([])
        loss = margin_hard_loss(pos, neg, margin_m=0.5)
        assert loss.item() == 0.0

    def test_gradient_flows(self):
        """Margin loss should support backprop."""
        pos = torch.tensor([1.0], requires_grad=True)
        neg = torch.tensor([0.9], requires_grad=True)
        loss = margin_hard_loss(pos, neg, margin_m=0.5)
        loss.backward()
        assert pos.grad is not None
        assert neg.grad is not None


# ── I2: compute_loss with objective_mode ──────────────────────────────────────

class TestI2ComputeLossObjectiveMode:
    """Tests for compute_loss with different objective modes."""

    def test_infonce_mode_backward_compat(self):
        """infonce mode should produce same result as before (loss_margin=0)."""
        pos = torch.tensor([0.5, 0.6])
        neg = torch.tensor([0.3, 0.4, 0.2])
        result = compute_loss(pos, neg, tau=1.0, T_mp=0.1,
                              lambda_mp=0.0, lambda_smooth=0.0,
                              objective_mode="infonce")
        assert "loss_margin" in result
        assert result["loss_margin"].item() == 0.0
        assert result["loss_intra"].item() > 0.0

    def test_mixed_margin_adds_margin_term(self):
        """mixed_margin mode should include both InfoNCE and margin terms."""
        pos = torch.tensor([1.0])
        neg = torch.tensor([0.9])
        result = compute_loss(pos, neg, tau=0.1, T_mp=0.1,
                              lambda_mp=0.0, lambda_smooth=0.0,
                              objective_mode="mixed_margin",
                              margin_m=0.5, hard_topk=8, lambda_margin=0.5)
        assert result["loss_intra"].item() > 0.0
        assert result["loss_margin"].item() > 0.0
        # Total should include both
        expected_total = result["loss_intra"].item() + 0.5 * result["loss_margin"].item()
        assert result["loss_total"].item() == pytest.approx(expected_total, abs=1e-5)

    def test_margin_only_no_infonce(self):
        """margin_only mode: loss_intra should be 0, loss_margin should be nonzero."""
        pos = torch.tensor([1.0])
        neg = torch.tensor([0.9])
        result = compute_loss(pos, neg, tau=0.1, T_mp=0.1,
                              lambda_mp=0.0, lambda_smooth=0.0,
                              objective_mode="margin_only",
                              margin_m=0.5, hard_topk=8, lambda_margin=0.5)
        assert result["loss_intra"].item() == 0.0
        assert result["loss_margin"].item() > 0.0
        assert result["loss_total"].item() == pytest.approx(result["loss_margin"].item(), abs=1e-5)

    # ── Wave-3 A2: iou_rank_only — ranking REPLACES InfoNCE+margin ──────────────

    def test_iou_rank_only_zeros_infonce_and_margin(self):
        """iou_rank_only: both InfoNCE and margin terms are off; total == λ·iou_rank."""
        pos = torch.tensor([1.0, 0.8])
        neg = torch.tensor([0.9, 0.2, 0.1])
        win_logits = torch.cat([pos, neg])                      # [pos | neg] order
        win_ious = torch.tensor([1.0, 0.9, 0.6, 0.0, 0.0])       # graded IoU spectrum
        result = compute_loss(pos, neg, tau=0.1, T_mp=0.1,
                              lambda_mp=0.0, lambda_smooth=0.0,
                              objective_mode="iou_rank_only",
                              window_logits=win_logits, window_ious=win_ious,
                              lambda_iou_rank=1.0, iou_rank_margin=0.5,
                              iou_rank_min_gap=0.1)
        assert result["loss_intra"].item() == 0.0
        assert result["loss_margin"].item() == 0.0
        assert result["loss_iou_rank"].item() > 0.0
        assert result["loss_total"].item() == pytest.approx(
            result["loss_iou_rank"].item(), abs=1e-6)

    def test_iou_rank_only_requires_window_inputs(self):
        """iou_rank_only must fail-fast (no placeholder 0 loss) when the span
        objective has nothing to rank — a silent 0 would train nothing."""
        pos = torch.tensor([1.0])
        neg = torch.tensor([0.9])
        with pytest.raises(ValueError, match="iou_rank_only"):
            compute_loss(pos, neg, tau=0.1, T_mp=0.1,
                         objective_mode="iou_rank_only",
                         lambda_iou_rank=0.0)

    def test_iou_rank_only_gradient_flows(self):
        """The ranking term must backprop to the window logits in this mode."""
        pos = torch.tensor([1.0, 0.8], requires_grad=True)
        neg = torch.tensor([0.9, 0.2, 0.1], requires_grad=True)
        win_logits = torch.cat([pos, neg])
        win_ious = torch.tensor([1.0, 0.9, 0.6, 0.0, 0.0])
        result = compute_loss(pos, neg, tau=0.1, T_mp=0.1,
                              objective_mode="iou_rank_only",
                              window_logits=win_logits, window_ious=win_ious,
                              lambda_iou_rank=0.7)
        result["loss_total"].backward()
        assert pos.grad is not None and torch.isfinite(pos.grad).all()

    def test_diagnostic_loss_dict_iou_rank_only_no_failfast(self):
        """train/val diagnostic recompute over concatenated logits must NOT hit
        compute_loss's iou_rank_only fail-fast (it has no window inputs) — it
        returns zeroed span terms instead. Regression for the A2 train_step/
        val_step crash."""
        from epitope_head.training.trainer import _diagnostic_loss_dict
        cat_pos = torch.tensor([1.0, 0.5])
        cat_neg = torch.tensor([0.2, 0.1])
        d = _diagnostic_loss_dict(cat_pos, cat_neg,
                                  {"objective_mode": "iou_rank_only", "lambda_iou_rank": 1.0})
        for k in ("loss_total", "loss_intra", "loss_mp", "loss_smooth", "loss_margin"):
            assert d[k].item() == 0.0

    def test_diagnostic_loss_dict_other_modes_delegate(self):
        """Non-iou_rank_only modes still compute the real diagnostic loss."""
        from epitope_head.training.trainer import _diagnostic_loss_dict
        cat_pos = torch.tensor([1.0, 0.5])
        cat_neg = torch.tensor([0.2, 0.1])
        d = _diagnostic_loss_dict(cat_pos, cat_neg, {"objective_mode": "infonce"})
        assert d["loss_intra"].item() > 0.0

    def test_diagnostic_loss_dict_strips_dualhead_exact_keys(self):
        """Dual-head exact keys must be stripped before compute_loss (the diagnostic
        recompute sees the un-popped loss_cfg). Regression for the a1res03_exact
        smoke crash: compute_loss() got an unexpected keyword 'lambda_exact'."""
        from epitope_head.training.trainer import _diagnostic_loss_dict
        cat_pos = torch.tensor([1.0, 0.5])
        cat_neg = torch.tensor([0.2, 0.1])
        cfg = {"objective_mode": "mixed_margin", "margin_m": 0.5, "hard_topk": 8,
               "lambda_margin": 0.1, "lambda_exact": 0.1, "exact_margin_m": 0.3,
               "exact_hard_topk": 8}
        d = _diagnostic_loss_dict(cat_pos, cat_neg, cfg)  # must not raise
        assert "loss_total" in d


# ── I3: Trainer schema updates ────────────────────────────────────────────────

class TestI3TrainerSchema:
    """Tests for trainer schema changes supporting margin loss."""

    def test_normalize_loss_cfg_accepts_optional_keys(self):
        """normalize_loss_cfg should accept new margin-related optional keys."""
        cfg = {
            "tau": 1.0, "T_mp": 0.1, "lambda_mp": 0.0, "lambda_smooth": 0.0,
            "objective_mode": "mixed_margin", "margin_m": 0.5,
            "hard_topk": 8, "lambda_margin": 0.5,
        }
        result = normalize_loss_cfg(cfg)
        assert result["objective_mode"] == "mixed_margin"
        assert result["margin_m"] == 0.5

    def test_normalize_loss_cfg_rejects_unknown_keys(self):
        """Unknown keys should still be rejected."""
        cfg = {
            "tau": 1.0, "T_mp": 0.1, "lambda_mp": 0.0, "lambda_smooth": 0.0,
            "bogus_key": 42,
        }
        with pytest.raises(ValueError, match="unknown keys"):
            normalize_loss_cfg(cfg)

    def test_step_metrics_has_loss_margin(self):
        """StepMetrics should have loss_margin field."""
        m = StepMetrics(loss_margin=0.42)
        assert m.loss_margin == 0.42
        d = m.to_dict()
        assert "loss_margin" in d
        assert d["loss_margin"] == 0.42

    def test_log_entry_keys_contains_loss_margin(self):
        """LOG_ENTRY_KEYS should include loss_margin."""
        assert "loss_margin" in LOG_ENTRY_KEYS
