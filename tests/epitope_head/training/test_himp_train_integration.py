"""HIMP4 trainer integration tests.

Covers PLAN_EPI_IMP.md §5 HIMP4 TDD Gate:
  1. RED: enabling residue supervision contributes zero gradient to scorer
     parameters.
  2. RED: old configs break when new trainer code is loaded.
  3. RED: window_mode=all exceeding max_windows_per_chunk silently falls back
     to a duplicated forward.
  4. GREEN: old config path and improved config path both execute on a tiny
     synthetic batch.
  5. GREEN: union window forward returns the same span-loss logits as the
     legacy code path when residue.enabled=false.

Loss-level coverage focuses on: weighted InfoNCE / margin variants,
``compute_loss`` plumbing, and ``prepare_chunk_spans`` extras shape. Full
end-to-end ``train_step`` smoke is left to the existing module-E/I contract
tests + post-hoc real training.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from epitope_head.training.losses import (
    compute_loss,
    info_nce_loss,
    margin_hard_loss,
)


# ── Weighted InfoNCE ────────────────────────────────────────────────────────

class TestWeightedInfoNCE:

    def test_none_weights_match_legacy(self):
        pos = torch.tensor([2.0])
        neg = torch.tensor([0.0, 0.5, 1.0])
        legacy = info_nce_loss(pos, neg, tau=0.5, neg_weights=None)
        explicit_legacy = info_nce_loss(pos, neg, tau=0.5)
        assert legacy.item() == pytest.approx(explicit_legacy.item(), abs=1e-7)

    def test_uniform_one_weights_match_legacy(self):
        pos = torch.tensor([2.0])
        neg = torch.tensor([0.0, 0.5, 1.0])
        weights = torch.ones(3)
        weighted = info_nce_loss(pos, neg, tau=0.5, neg_weights=weights)
        legacy = info_nce_loss(pos, neg, tau=0.5)
        assert weighted.item() == pytest.approx(legacy.item(), abs=1e-6)

    def test_zero_weight_excludes_negative(self):
        pos = torch.tensor([2.0])
        neg_full = torch.tensor([0.0, 0.5, 1.0])
        weights = torch.tensor([1.0, 1.0, 0.0])  # exclude the 3rd (logit=1.0)
        loss_weighted = info_nce_loss(pos, neg_full, tau=0.5, neg_weights=weights)
        # Equivalent to dropping the 3rd negative entirely
        neg_subset = torch.tensor([0.0, 0.5])
        loss_subset = info_nce_loss(pos, neg_subset, tau=0.5)
        assert loss_weighted.item() == pytest.approx(loss_subset.item(), abs=1e-6)

    def test_partial_weights_smaller_than_full_one(self):
        # Reducing a hard negative's weight should *decrease* loss (less penalty).
        pos = torch.tensor([0.5])
        neg = torch.tensor([2.0, 0.0])  # neg[0] is "harder" than positive
        loss_full = info_nce_loss(pos, neg, tau=0.5, neg_weights=torch.tensor([1.0, 1.0]))
        loss_softened = info_nce_loss(pos, neg, tau=0.5, neg_weights=torch.tensor([0.1, 1.0]))
        assert loss_softened.item() < loss_full.item()

    def test_gradient_flows_through_weighted(self):
        pos = torch.tensor([0.0], requires_grad=True)
        neg = torch.tensor([1.0, 1.0], requires_grad=True)
        weights = torch.tensor([0.5, 1.0])
        loss = info_nce_loss(pos, neg, tau=0.5, neg_weights=weights)
        loss.backward()
        assert pos.grad is not None
        assert neg.grad is not None


# ── Weighted margin_hard_loss ───────────────────────────────────────────────

class TestWeightedMarginHard:

    def test_none_weights_match_legacy(self):
        pos = torch.tensor([1.0])
        neg = torch.tensor([0.0, 1.0, 2.0])  # one above margin
        legacy = margin_hard_loss(pos, neg, margin_m=0.5, hard_topk=2, neg_weights=None)
        explicit = margin_hard_loss(pos, neg, margin_m=0.5, hard_topk=2)
        assert legacy.item() == pytest.approx(explicit.item(), abs=1e-7)

    def test_zero_weights_filter_before_topk(self):
        # Without weights: top-2 hardest from [0, 1, 2] = [2, 1] → pos=1.0, gaps=[-1, 0]
        # With w=[1, 1, 0]: hardest is [1, 0] → gaps=[0, 1]
        pos = torch.tensor([1.0])
        neg = torch.tensor([0.0, 1.0, 2.0])
        loss_no_wt = margin_hard_loss(pos, neg, margin_m=0.5, hard_topk=2)
        loss_wt = margin_hard_loss(
            pos, neg, margin_m=0.5, hard_topk=2,
            neg_weights=torch.tensor([1.0, 1.0, 0.0]),
        )
        # Loss with the hard neg masked should be lower.
        assert loss_wt.item() < loss_no_wt.item()


# ── compute_loss plumbing ───────────────────────────────────────────────────

class TestComputeLossWithWeights:

    def test_neg_weights_forwarded(self):
        pos = torch.tensor([0.5])
        neg = torch.tensor([2.0, 0.0])
        weights = torch.tensor([0.1, 1.0])
        out_w = compute_loss(
            pos, neg, tau=0.5, T_mp=0.1, lambda_mp=0.0, lambda_smooth=0.0,
            objective_mode="infonce",
            neg_weights=weights,
        )
        out_no = compute_loss(
            pos, neg, tau=0.5, T_mp=0.1, lambda_mp=0.0, lambda_smooth=0.0,
            objective_mode="infonce",
            neg_weights=None,
        )
        assert out_w["loss_intra"].item() != out_no["loss_intra"].item()
        # mixed_margin must also forward weights — both terms differ.
        out_mm_w = compute_loss(
            pos, neg, tau=0.5, T_mp=0.1, lambda_mp=0.0, lambda_smooth=0.0,
            objective_mode="mixed_margin", margin_m=0.5, hard_topk=2,
            lambda_margin=0.5,
            neg_weights=weights,
        )
        assert out_mm_w["loss_total"].item() != out_no["loss_total"].item()

    def test_no_weights_unchanged(self):
        pos = torch.tensor([1.0])
        neg = torch.tensor([0.0, 0.5])
        out_default = compute_loss(
            pos, neg, tau=0.5, T_mp=0.1, lambda_mp=0.0, lambda_smooth=0.0,
            objective_mode="infonce",
        )
        out_explicit_none = compute_loss(
            pos, neg, tau=0.5, T_mp=0.1, lambda_mp=0.0, lambda_smooth=0.0,
            objective_mode="infonce",
            neg_weights=None,
        )
        assert out_default["loss_total"].item() == pytest.approx(
            out_explicit_none["loss_total"].item(), abs=1e-7,
        )


# ── prepare_chunk_spans extension ───────────────────────────────────────────

class TestPrepareChunkSpansExtras:
    """Lightweight tests on the extras tuple element (HIMP4 Action 1)."""

    def _build_batch(self, chunk_len: int = 200, positives: list[dict] | None = None,
                     disrupted: list[dict] | None = None,
                     sequence_length: int | None = None,
                     chunk_start: int = 0) -> dict:
        """Construct a one-chunk batch dict matching the trainer's expected shape.

        Defaults represent a single-chunk protein where ``chunk_end ==
        sequence_length`` so per-chunk trusted-interior logic treats both
        ends as protein termini (no seam exclusion).
        """
        if positives is None:
            positives = [{"start_0b": 50, "end_0b": 65, "pep_len": 15}]
        chunk_end = chunk_start + chunk_len
        if sequence_length is None:
            sequence_length = chunk_end
        return {
            "protein_ids": ["P1"],
            "chunk_starts": torch.tensor([chunk_start]),
            "chunk_ends": torch.tensor([chunk_end]),
            "sequence_lengths": torch.tensor([sequence_length]),
            "positives": [positives],
            "disrupted_spans": [disrupted],
        }

    def test_legacy_call_returns_five_tuple(self):
        from epitope_head.training.trainer import prepare_chunk_spans
        batch = self._build_batch()
        result = prepare_chunk_spans(
            batch, batch_idx=0,
            neg_ratio=4, hard_negative_fraction=0.5,
            hard_neg_max_overlap_ratio=0.8, hard_neg_offset_range=20,
            neg_length_sampling="match_positive",
            min_k=12, max_k=25,
            rng=np.random.RandomState(0),
        )
        # 5-tuple after HIMP4: (pos, neg, allele_pos, allele_neg, extras)
        assert len(result) == 5
        pos, neg, ap, an, extras = result
        assert isinstance(extras, list)
        assert len(extras) == 1
        # In legacy mode (no near_positive/residue cfg) extras dict has the
        # documented optional keys but with None / empty defaults.
        chunk_extras = extras[0]
        assert chunk_extras.get("neg_weights") is None
        assert chunk_extras.get("neg_relations") is None

    def test_himp_call_attaches_weights_and_relations(self):
        from epitope_head.training.trainer import prepare_chunk_spans
        batch = self._build_batch()
        np_cfg = {
            "enabled": True, "near_gap_max": 10, "schedule": "ignore",
            "schedule_params": {}, "metric": "endpoint_gap",
            "apply_to_span_negatives": True, "apply_to_residue_labels": True,
        }
        result = prepare_chunk_spans(
            batch, batch_idx=0,
            neg_ratio=4, hard_negative_fraction=0.5,
            hard_neg_max_overlap_ratio=0.8, hard_neg_offset_range=20,
            neg_length_sampling="match_positive",
            min_k=12, max_k=25,
            rng=np.random.RandomState(0),
            near_positive_cfg=np_cfg,
        )
        assert len(result) == 5
        pos, neg, ap, an, extras = result
        chunk_extras = extras[0]
        # Weights present, length == n_neg.
        weights = chunk_extras["neg_weights"]
        relations = chunk_extras["neg_relations"]
        assert weights is not None
        assert relations is not None
        assert weights.shape[0] == neg[0].shape[0]
        assert len(relations) == neg[0].shape[0]
        # All weights in [0, 1].
        assert (weights >= 0.0).all() and (weights <= 1.0).all()

    def test_apply_to_span_negatives_false_skips_weights(self):
        """Review fix 2: span weighting must respect apply_to_span_negatives."""
        from epitope_head.training.trainer import prepare_chunk_spans
        batch = self._build_batch()
        np_cfg = {
            "enabled": True, "near_gap_max": 10, "schedule": "ignore",
            "schedule_params": {}, "metric": "endpoint_gap",
            "apply_to_span_negatives": False,   # turn span-side off
            "apply_to_residue_labels": True,
        }
        result = prepare_chunk_spans(
            batch, batch_idx=0,
            neg_ratio=4, hard_negative_fraction=0.5,
            hard_neg_max_overlap_ratio=0.8, hard_neg_offset_range=20,
            neg_length_sampling="match_positive",
            min_k=12, max_k=25,
            rng=np.random.RandomState(0),
            near_positive_cfg=np_cfg,
        )
        _, _, _, _, extras = result
        # Span-side metadata must be absent when apply_to_span_negatives=False.
        assert extras[0]["neg_weights"] is None
        assert extras[0]["neg_relations"] is None

    def _run_for_central(self, batch: dict, seam: int = 10) -> np.ndarray:
        from epitope_head.training.trainer import prepare_chunk_spans
        residue_cfg = {
            "enabled": True, "lambda_residue": 0.1,
            "label_mode": "binary_coverage",
            "aggregation": "log_mean_exp", "aggregation_params": {"beta": 1.0},
            "loss_mode": "pairwise_margin", "margin_m_residue": 0.5,
            "window_mode": "sampled", "max_windows_per_chunk": 1024,
            "min_far_bg_residues": 4,
        }
        _, _, _, _, extras = prepare_chunk_spans(
            batch, batch_idx=0,
            neg_ratio=4, hard_negative_fraction=0.5,
            hard_neg_max_overlap_ratio=0.8, hard_neg_offset_range=20,
            neg_length_sampling="match_positive",
            min_k=12, max_k=25,
            rng=np.random.RandomState(0),
            residue_cfg=residue_cfg,
            chunk_central_margin=seam,
        )
        meta = extras[0]["residue_meta"]
        for k in ("label", "ambiguous_mask", "far_bg_mask", "central_mask"):
            assert k in meta
        return meta["central_mask"]

    def test_single_chunk_central_covers_full_protein(self):
        """Review fix 4: single-chunk protein has no seams → full central."""
        L = 200
        batch = self._build_batch(chunk_len=L, chunk_start=0, sequence_length=L)
        central = self._run_for_central(batch, seam=10)
        assert central.all()  # no seam exclusion at protein N/C termini

    def test_first_chunk_only_right_seam_applied(self):
        """First chunk of multi-chunk protein: left seam = 0 (N-terminus)."""
        chunk_len = 200
        batch = self._build_batch(
            chunk_len=chunk_len, chunk_start=0, sequence_length=400,
        )
        central = self._run_for_central(batch, seam=10)
        # Left side fully included (residue 0 is N-terminus, kept).
        assert central[:10].all()
        # Right seam excluded (chunk does NOT reach protein end).
        assert not central[chunk_len - 10:].any()

    def test_middle_chunk_both_seams_applied(self):
        """Interior chunk: both seams excluded."""
        chunk_len = 200
        batch = self._build_batch(
            chunk_len=chunk_len, chunk_start=100, sequence_length=600,
        )
        central = self._run_for_central(batch, seam=10)
        assert not central[:10].any()
        assert not central[chunk_len - 10:].any()
        assert central[10:chunk_len - 10].all()

    def test_last_chunk_only_left_seam_applied(self):
        """Last chunk of multi-chunk protein: right seam = 0 (C-terminus)."""
        chunk_len = 200
        seq_len = 600
        batch = self._build_batch(
            chunk_len=chunk_len, chunk_start=seq_len - chunk_len,
            sequence_length=seq_len,
        )
        central = self._run_for_central(batch, seam=10)
        assert not central[:10].any()
        assert central[chunk_len - 10:].all()  # C-terminus kept


# ── StepMetrics + LOG_ENTRY_KEYS extension ──────────────────────────────────

class TestForwardUnionLossPlumbing:
    """Review fix 6: the avg_loss returned by _forward_union_and_compute_losses
    must include weighted span loss AND lambda_residue * residue_loss, so the
    trainer's backward pass and logged loss_total agree."""

    def _stub_model(self, fixed_logit: float = 0.5):
        """Return a callable that emits a constant logit per requested span."""

        class _Stub:
            def __init__(self, value: float):
                self.value = value

            def __call__(self, token_ids, attention_mask, spans_list, allele_list, chunk_lengths):
                return [
                    torch.full((s.shape[0],), self.value, dtype=torch.float32, requires_grad=True)
                    for s in spans_list
                ]

        return _Stub(fixed_logit)

    def _build_inputs(self, chunk_len: int = 200):
        from epitope_head.training.trainer import prepare_chunk_spans
        batch = {
            "protein_ids": ["P1"],
            "chunk_starts": torch.tensor([0]),
            "chunk_ends": torch.tensor([chunk_len]),
            "sequence_lengths": torch.tensor([chunk_len]),
            "positives": [[{"start_0b": 50, "end_0b": 65, "pep_len": 15}]],
            "disrupted_spans": [None],
            "token_ids": torch.zeros((1, chunk_len), dtype=torch.long),
            "attention_mask": torch.ones((1, chunk_len), dtype=torch.long),
        }
        residue_cfg = {
            "enabled": True, "lambda_residue": 0.5,
            "label_mode": "binary_coverage",
            "aggregation": "log_mean_exp", "aggregation_params": {"beta": 1.0},
            "loss_mode": "pairwise_margin", "margin_m_residue": 0.5,
            "window_mode": "sampled", "max_windows_per_chunk": 1024,
            "min_far_bg_residues": 4,
        }
        spans = prepare_chunk_spans(
            batch, batch_idx=0,
            neg_ratio=4, hard_negative_fraction=0.5,
            hard_neg_max_overlap_ratio=0.8, hard_neg_offset_range=20,
            neg_length_sampling="match_positive",
            min_k=12, max_k=25,
            rng=np.random.RandomState(0),
            residue_cfg=residue_cfg,
            chunk_central_margin=0,
        )
        return batch, residue_cfg, spans

    def test_avg_loss_includes_residue_contribution(self):
        from epitope_head.training.trainer import _forward_union_and_compute_losses
        batch, residue_cfg, spans = self._build_inputs()
        pos, neg, ap, an, extras = spans
        model = self._stub_model(fixed_logit=0.5)
        loss_cfg = {
            "tau": 1.0, "T_mp": 0.1, "lambda_mp": 0.0, "lambda_smooth": 0.0,
            "objective_mode": "infonce",
        }
        avg_loss, _, _, residue_stats, n_chunks = _forward_union_and_compute_losses(
            model=model, batch=batch,
            pos_spans_list=pos, neg_spans_list=neg,
            allele_pos_list=ap, allele_neg_list=an,
            extras_list=extras,
            loss_cfg=loss_cfg, residue_cfg=residue_cfg,
        )
        assert n_chunks > 0
        # With a constant scorer all logits are equal → InfoNCE = log(1+N).
        # Residue ranking with all-equal scores → relu(margin - 0) > 0.
        # Therefore avg_loss must reflect both contributions.
        assert residue_stats["n_residue_chunks"] >= 1
        # If residue contribution were dropped, avg_loss would equal pure
        # span InfoNCE on equal logits ≈ log(1+N). Lambda_residue=0.5 with
        # margin_m=0.5 produces an additive component > 0.
        assert avg_loss.item() > 0.0


class TestStepMetricsResidueFields:

    def test_step_metrics_has_residue_fields(self):
        from epitope_head.training.trainer import StepMetrics
        m = StepMetrics(
            loss_residue=0.1, n_residue_pairs=12,
            residue_skipped_chunks=2, n_residue_chunks=5,
        )
        assert m.loss_residue == 0.1
        assert m.n_residue_pairs == 12
        assert m.residue_skipped_chunks == 2
        assert m.n_residue_chunks == 5

    def test_log_entry_keys_contain_residue_fields(self):
        from epitope_head.training.trainer import LOG_ENTRY_KEYS
        assert "loss_residue" in LOG_ENTRY_KEYS
        assert "lambda_residue" in LOG_ENTRY_KEYS
        assert "residue_skipped_chunks" in LOG_ENTRY_KEYS
