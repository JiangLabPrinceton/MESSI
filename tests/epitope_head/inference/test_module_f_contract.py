"""Module F contract tests (F0-F7): config, checkpoint, encoding, scoring, aggregation, export, repro, smoke."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from epitope_head.configs import load_inference_config
from epitope_head.inference.export import (
    compute_payload_digest,
    export_prediction_json,
    format_prediction_payload,
    validate_prediction_payload,
    write_prediction_summary,
)
from epitope_head.inference.predictor import (
    InferencePredictor,
    build_epitope_scorer_from_config,
)
from epitope_head.training.trainer import save_checkpoint


class DeterministicFrozenEncoder(nn.Module):
    """Returns deterministic residue embeddings from token ids."""

    def __init__(self, d_enc: int = 16):
        super().__init__()
        self.d_enc = d_enc

    @torch.no_grad()
    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _ = token_ids.shape
        real_counts = attention_mask.sum(dim=1)
        lengths = real_counts - 2  # strip BOS + EOS
        l_max = int(lengths.max().item())

        out = torch.zeros(bsz, l_max, self.d_enc)
        for i in range(bsz):
            l_i = int(lengths[i].item())
            for j in range(l_i):
                tok = int(token_ids[i, j + 1].item())  # skip BOS
                out[i, j, tok % self.d_enc] = 1.0
        return out, lengths


def simple_tokenize(seqs: list[str]) -> dict[str, torch.Tensor]:
    """BOS/EOS tokenizer for tests, deterministic and model-agnostic."""
    if not seqs:
        raise ValueError("empty sequence batch")

    max_len = max(len(s) for s in seqs) + 2
    token_ids = torch.full((len(seqs), max_len), 1, dtype=torch.long)  # PAD=1
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)

    for i, seq in enumerate(seqs):
        ids = [0] + [4 + (ord(ch) % 20) for ch in seq] + [2]  # BOS=0, EOS=2
        token_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = True

    return {"token_ids": token_ids, "attention_mask": attention_mask}


def _valid_inference_yaml() -> dict:
    return {
        "inference": {
            "checkpoint_path": "outputs/checkpoints/best.pt",
            "min_k": 12,
            "max_k": 25,
            "hotspot_center_method": "median",
            "hotspot_clamp": "none",
            "chunking": {
                "enabled": True,
                "context_len": 1022,
                "stride": 512,
                "margin": 32,
                "stitch_mode": "per_residue_stitch",
                "enable_reliability": True,
            },
            "device": "cpu",
        },
    }


def _valid_model_cfg(d_enc: int = 16, d_proj: int = 8) -> dict:
    return {
        "encoder_name": "mock",
        "freeze_encoder": True,
        "d_enc": d_enc,
        "d_proj": d_proj,
        "length_embedding_dim": 4,
        "allele_embedding_dim": 4,
        "scorer_hidden_dim": 16,
        "scorer_activation": "gelu",
        "pad_left_init": "zeros",
        "pad_right_init": "zeros",
    }


class TestF0InferenceConfig:
    def test_missing_top_level_inference_key_raises(self, tmp_path: Path):
        path = tmp_path / "inference.yaml"
        with open(path, "w") as f:
            yaml.safe_dump({"wrong": {}}, f)

        with pytest.raises(ValueError, match="top-level 'inference'"):
            load_inference_config(path)

    def test_missing_required_key_raises(self, tmp_path: Path):
        cfg = _valid_inference_yaml()
        del cfg["inference"]["max_k"]
        path = tmp_path / "inference.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)

        with pytest.raises(ValueError, match="missing required keys"):
            load_inference_config(path)

    def test_invalid_enum_value_raises(self, tmp_path: Path):
        cfg = _valid_inference_yaml()
        cfg["inference"]["hotspot_clamp"] = "sigmoid"
        path = tmp_path / "inference.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)

        with pytest.raises(ValueError, match="hotspot_clamp"):
            load_inference_config(path)

    def test_valid_config_loads(self, tmp_path: Path):
        cfg = _valid_inference_yaml()
        path = tmp_path / "inference.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)

        out = load_inference_config(path)
        assert out["min_k"] == 12
        assert out["max_k"] == 25
        assert out["chunking"]["context_len"] == 1022

    def test_min_k_must_be_frozen_to_12(self, tmp_path: Path):
        cfg = _valid_inference_yaml()
        cfg["inference"]["min_k"] = 11
        path = tmp_path / "inference.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)

        with pytest.raises(ValueError, match="must be frozen to 12"):
            load_inference_config(path)

    def test_max_k_must_be_frozen_to_25(self, tmp_path: Path):
        cfg = _valid_inference_yaml()
        cfg["inference"]["max_k"] = 26
        path = tmp_path / "inference.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)

        with pytest.raises(ValueError, match="must be frozen to 25"):
            load_inference_config(path)

    def test_chunking_enabled_must_be_true(self, tmp_path: Path):
        cfg = _valid_inference_yaml()
        cfg["inference"]["chunking"]["enabled"] = False
        path = tmp_path / "inference.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)

        with pytest.raises(ValueError, match="chunking.enabled must be true"):
            load_inference_config(path)

    def test_stitch_mode_must_be_per_residue_stitch(self, tmp_path: Path):
        cfg = _valid_inference_yaml()
        cfg["inference"]["chunking"]["stitch_mode"] = "center_weighted"
        path = tmp_path / "inference.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)

        with pytest.raises(ValueError, match="stitch_mode must be frozen to per_residue_stitch"):
            load_inference_config(path)


class TestF1CheckpointAndInit:
    def _save_valid_ckpt(self, tmp_path: Path, model: nn.Module) -> Path:
        ckpt = tmp_path / "best.pt"
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        save_checkpoint(
            model=model,
            optimizer=opt,
            epoch=1,
            global_step=10,
            monitor_metric="logit_gap",
            monitor_value=0.1,
            cfg_hash="abc123",
            path=ckpt,
            manifest_version="v1.1",
            diff_ids_applied=["d001"],
        )
        return ckpt

    def test_missing_checkpoint_metadata_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.pt"
        torch.save({"metadata": {"epoch": 1}, "model_state_dict": {}}, bad)

        model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
        inf_cfg = _valid_inference_yaml()["inference"]

        with pytest.raises(ValueError, match="metadata"):
            InferencePredictor.from_checkpoint(
                checkpoint_path=bad,
                model=model,
                inference_cfg=inf_cfg,
                tokenize_fn=simple_tokenize,
            )

    def test_state_dict_shape_mismatch_raises(self, tmp_path: Path):
        model_a = build_epitope_scorer_from_config(_valid_model_cfg(d_proj=8), DeterministicFrozenEncoder())
        ckpt = self._save_valid_ckpt(tmp_path, model_a)

        # Different d_proj -> incompatible projection/scorer shapes
        model_b = build_epitope_scorer_from_config(_valid_model_cfg(d_proj=7), DeterministicFrozenEncoder())
        inf_cfg = _valid_inference_yaml()["inference"]

        with pytest.raises(ValueError, match="state_dict"):
            InferencePredictor.from_checkpoint(
                checkpoint_path=ckpt,
                model=model_b,
                inference_cfg=inf_cfg,
                tokenize_fn=simple_tokenize,
            )

    def test_from_checkpoint_success(self, tmp_path: Path):
        model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
        ckpt = self._save_valid_ckpt(tmp_path, model)

        inf_cfg = _valid_inference_yaml()["inference"]
        predictor = InferencePredictor.from_checkpoint(
            checkpoint_path=ckpt,
            model=model,
            inference_cfg=inf_cfg,
            tokenize_fn=simple_tokenize,
        )
        assert predictor.model is model
        assert not predictor.model.training
        assert predictor.checkpoint_metadata["manifest_version"] == "v1.1"

    def test_unimplemented_stitch_mode_fails_fast(self, tmp_path: Path):
        model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
        ckpt = self._save_valid_ckpt(tmp_path, model)
        inf_cfg = _valid_inference_yaml()["inference"]
        inf_cfg["chunking"]["stitch_mode"] = "center_weighted"

        with pytest.raises(ValueError, match="Unsupported stitch_mode"):
            InferencePredictor.from_checkpoint(
                checkpoint_path=ckpt,
                model=model,
                inference_cfg=inf_cfg,
                tokenize_fn=simple_tokenize,
            )


class TestF2SequenceEncoding:
    def _build_predictor(self, tmp_path: Path) -> InferencePredictor:
        model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ckpt = tmp_path / "best.pt"
        save_checkpoint(
            model=model,
            optimizer=opt,
            epoch=1,
            global_step=10,
            monitor_metric="logit_gap",
            monitor_value=0.2,
            cfg_hash="abc123",
            path=ckpt,
            manifest_version="v1.1",
            diff_ids_applied=["f001"],
        )
        return InferencePredictor.from_checkpoint(
            checkpoint_path=ckpt,
            model=model,
            inference_cfg=_valid_inference_yaml()["inference"],
            tokenize_fn=simple_tokenize,
        )

    def test_short_sequence_single_pass_shape(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        seq = "A" * 100

        emb, debug = predictor.encode_sequence(seq)
        assert emb.shape[0] == len(seq)
        assert emb.shape[1] == predictor.model.projection.linear.out_features
        assert debug["n_chunks"] == 1

    def test_long_sequence_chunked_stitch_shape(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        seq = "ACDEFGHIKLMNPQRSTVWY" * 80  # 1600 aa > 1022

        emb, debug = predictor.encode_sequence(seq)
        assert emb.shape[0] == len(seq)
        assert debug["n_chunks"] > 1
        assert len(debug["chunk_starts"]) == debug["n_chunks"]

    def test_long_sequence_deterministic(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        seq = "ACDEFGHIKLMNPQRSTVWY" * 75  # 1500 aa

        emb1, dbg1 = predictor.encode_sequence(seq)
        emb2, dbg2 = predictor.encode_sequence(seq)

        assert torch.allclose(emb1, emb2)
        assert dbg1["chunk_starts"] == dbg2["chunk_starts"]
        assert dbg1["residue_owner_chunk"] == dbg2["residue_owner_chunk"]

    def test_reliability_reported_in_debug(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        seq = "A" * 1500

        _, debug = predictor.encode_sequence(seq)
        reliability = debug["residue_reliability"]
        assert len(reliability) == len(seq)
        assert all(0.0 <= r <= 1.0 for r in reliability)


class TestF3WindowEnumeration:
    def _build_predictor(self, tmp_path: Path) -> InferencePredictor:
        model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ckpt = tmp_path / "best.pt"
        from epitope_head.training.trainer import save_checkpoint

        save_checkpoint(
            model=model, optimizer=opt, epoch=1, global_step=10,
            monitor_metric="logit_gap", monitor_value=0.2, cfg_hash="abc123",
            path=ckpt, manifest_version="v1.1", diff_ids_applied=["f003"],
        )
        return InferencePredictor.from_checkpoint(
            checkpoint_path=ckpt, model=model,
            inference_cfg=_valid_inference_yaml()["inference"],
            tokenize_fn=simple_tokenize,
        )

    def test_window_count_formula(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        L = 50
        seq = "A" * L
        G, _ = predictor.encode_sequence(seq)
        entries, z = predictor.enumerate_and_score(G, L, 12, 25)
        expected = sum(max(0, L - k + 1) for k in range(12, 26))
        assert len(entries) == expected
        assert z.shape[0] == expected

    def test_window_ordering(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        L = 40
        seq = "A" * L
        G, _ = predictor.encode_sequence(seq)
        entries, _ = predictor.enumerate_and_score(G, L, 12, 25)
        for i in range(1, len(entries)):
            prev = (entries[i - 1]["start_0b"], entries[i - 1]["k"])
            curr = (entries[i]["start_0b"], entries[i]["k"])
            assert prev <= curr, f"Window ordering violated at index {i}"

    def test_short_protein_no_windows(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        L = 10
        # Encode a short sequence (still need valid embedding)
        seq = "A" * L
        G, _ = predictor.encode_sequence(seq)
        entries, z = predictor.enumerate_and_score(G, L, 12, 25)
        assert len(entries) == 0
        assert z.shape[0] == 0

    def test_logits_finite(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        seq = "ACDEFGHIKLMNPQRSTVWY" * 3  # 60 aa
        G, _ = predictor.encode_sequence(seq)
        entries, z = predictor.enumerate_and_score(G, len(seq), 12, 25)
        assert torch.isfinite(z).all(), "Found NaN or Inf in logits"

    def test_batched_matches_unbatched(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        seq = "A" * 50
        G, _ = predictor.encode_sequence(seq)
        _, z_small = predictor.enumerate_and_score(G, 50, 12, 25, window_batch_size=8)
        _, z_large = predictor.enumerate_and_score(G, 50, 12, 25, window_batch_size=9999)
        assert torch.allclose(z_small, z_large), "Batched vs unbatched logits differ"


class TestF4Aggregation:
    def _build_predictor(self, tmp_path: Path) -> InferencePredictor:
        model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ckpt = tmp_path / "best.pt"
        from epitope_head.training.trainer import save_checkpoint

        save_checkpoint(
            model=model, optimizer=opt, epoch=1, global_step=10,
            monitor_metric="logit_gap", monitor_value=0.2, cfg_hash="abc123",
            path=ckpt, manifest_version="v1.1", diff_ids_applied=["f004"],
        )
        return InferencePredictor.from_checkpoint(
            checkpoint_path=ckpt, model=model,
            inference_cfg=_valid_inference_yaml()["inference"],
            tokenize_fn=simple_tokenize,
        )

    def test_hotspot_length_equals_protein_length(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        seq = "A" * 50
        G, _ = predictor.encode_sequence(seq)
        entries, z = predictor.enumerate_and_score(G, len(seq), 12, 25)
        h_raw, h_proc, R = predictor.aggregate_hotspot_and_risk(
            entries, z, len(seq), "median", "none",
        )
        assert h_raw.shape[0] == len(seq)
        assert h_proc.shape[0] == len(seq)

    def test_global_risk_is_scalar_and_finite(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        seq = "A" * 50
        G, _ = predictor.encode_sequence(seq)
        entries, z = predictor.enumerate_and_score(G, len(seq), 12, 25)
        _, _, R = predictor.aggregate_hotspot_and_risk(
            entries, z, len(seq), "none", "none",
        )
        assert isinstance(R, float)
        assert math.isfinite(R)

    def test_log_mean_exp_formula(self, tmp_path: Path):
        """Verify log-mean-exp on a toy example with known values."""
        predictor = self._build_predictor(tmp_path)
        # Manually construct toy window entries covering residue 0.
        z_vals = torch.tensor([1.0, 2.0, 3.0])
        entries = [
            {"start_0b": 0, "end_0b": 1, "k": 1},
            {"start_0b": 0, "end_0b": 1, "k": 1},
            {"start_0b": 0, "end_0b": 1, "k": 1},
        ]
        h_raw, _, _ = predictor.aggregate_hotspot_and_risk(entries, z_vals, 1, "none", "none")
        expected = torch.logsumexp(z_vals, dim=0) - math.log(3)
        assert torch.allclose(h_raw[0], expected, atol=1e-5)

    def test_center_median_shifts_median_to_zero(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        seq = "A" * 50
        G, _ = predictor.encode_sequence(seq)
        entries, z = predictor.enumerate_and_score(G, len(seq), 12, 25)
        _, h_proc, _ = predictor.aggregate_hotspot_and_risk(
            entries, z, len(seq), "median", "none",
        )
        assert abs(torch.median(h_proc).item()) < 1e-5

    def test_clamp_softplus_nonnegative(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        seq = "A" * 50
        G, _ = predictor.encode_sequence(seq)
        entries, z = predictor.enumerate_and_score(G, len(seq), 12, 25)
        _, h_proc, _ = predictor.aggregate_hotspot_and_risk(
            entries, z, len(seq), "median", "softplus",
        )
        assert (h_proc >= 0).all()


class TestF3F4Integration:
    def _build_predictor(self, tmp_path: Path) -> InferencePredictor:
        model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ckpt = tmp_path / "best.pt"
        from epitope_head.training.trainer import save_checkpoint

        save_checkpoint(
            model=model, optimizer=opt, epoch=1, global_step=10,
            monitor_metric="logit_gap", monitor_value=0.2, cfg_hash="abc123",
            path=ckpt, manifest_version="v1.1", diff_ids_applied=["f034"],
        )
        return InferencePredictor.from_checkpoint(
            checkpoint_path=ckpt, model=model,
            inference_cfg=_valid_inference_yaml()["inference"],
            tokenize_fn=simple_tokenize,
        )

    def test_predict_protein_returns_required_keys(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        seq = "A" * 50
        result = predictor.predict_protein(seq)
        assert "window_logits" in result
        assert "residue_hotspot" in result
        assert "global_risk" in result
        assert "meta" in result
        assert "debug" in result
        assert result["meta"]["protein_len"] == 50
        assert result["residue_hotspot"].shape[0] == 50

    def test_predict_protein_deterministic(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        seq = "ACDEFGHIKLMNPQRSTVWY" * 3  # 60 aa
        r1 = predictor.predict_protein(seq)
        r2 = predictor.predict_protein(seq)
        assert torch.allclose(r1["residue_hotspot"], r2["residue_hotspot"])
        assert r1["global_risk"] == r2["global_risk"]
        assert len(r1["window_logits"]) == len(r2["window_logits"])


# ── F5: Canonical JSON Export ────────────────────────────────────────────────

class TestF5Export:
    def _build_predictor(self, tmp_path: Path) -> InferencePredictor:
        model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ckpt = tmp_path / "best.pt"
        save_checkpoint(
            model=model, optimizer=opt, epoch=1, global_step=10,
            monitor_metric="logit_gap", monitor_value=0.2, cfg_hash="abc123",
            path=ckpt, manifest_version="v1.1", diff_ids_applied=["f005"],
        )
        return InferencePredictor.from_checkpoint(
            checkpoint_path=ckpt, model=model,
            inference_cfg=_valid_inference_yaml()["inference"],
            tokenize_fn=simple_tokenize,
        )

    def test_format_payload_has_required_keys(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        result = predictor.predict_protein("A" * 50)
        payload = format_prediction_payload(result, "P12345")
        errors = validate_prediction_payload(payload)
        assert errors == [], f"Validation errors: {errors}"

    def test_residue_hotspot_has_correct_entries(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        result = predictor.predict_protein("A" * 30)
        payload = format_prediction_payload(result, "P00001")
        rh = payload["residue_hotspot"]
        assert len(rh) == 30
        assert rh[0]["index_0b"] == 0
        assert rh[29]["index_0b"] == 29
        assert "h_raw" in rh[0]
        assert "h_processed" in rh[0]

    def test_window_logits_entries_have_required_keys(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        result = predictor.predict_protein("A" * 50)
        payload = format_prediction_payload(result, "P00002")
        for w in payload["window_logits"][:5]:
            assert set(w.keys()) >= {"start_0b", "end_0b", "k", "z"}

    def test_meta_includes_checkpoint_metadata(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        result = predictor.predict_protein("A" * 50)
        payload = format_prediction_payload(
            result, "P00003",
            checkpoint_metadata=predictor.checkpoint_metadata,
            config_hash="deadbeef",
        )
        meta = payload["meta"]
        assert meta["protein_id"] == "P00003"
        assert "checkpoint" in meta
        assert meta["checkpoint"]["manifest_version"] == "v1.1"
        assert meta["inference_config_hash"] == "deadbeef"

    def test_export_writes_valid_json(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        result = predictor.predict_protein("A" * 50)
        payload = format_prediction_payload(result, "P00004")
        out_path = tmp_path / "predictions" / "P00004.json"
        written = export_prediction_json(payload, out_path)
        assert written.exists()
        with open(written) as f:
            loaded = json.load(f)
        assert loaded["global_risk"] == payload["global_risk"]
        assert len(loaded["window_logits"]) == len(payload["window_logits"])

    def test_export_rejects_invalid_payload(self, tmp_path: Path):
        bad_payload = {"window_logits": [], "meta": {}}  # missing keys
        with pytest.raises(ValueError, match="Payload validation"):
            export_prediction_json(bad_payload, tmp_path / "bad.json")

    def test_write_prediction_summary(self, tmp_path: Path):
        summary_path = write_prediction_summary(
            protein_ids=["P1", "P2"],
            payload_digests=["aabb", "ccdd"],
            config_hash="xyz",
            output_dir=tmp_path / "predictions",
        )
        assert summary_path.exists()
        with open(summary_path) as f:
            s = json.load(f)
        assert s["n_proteins"] == 2
        assert "combined_digest" in s
        assert s["payload_digests"]["P1"] == "aabb"


# ── F6: Reproducibility and Payload Validation ──────────────────────────────

class TestF6Reproducibility:
    def _build_predictor(self, tmp_path: Path) -> InferencePredictor:
        torch.manual_seed(42)
        model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ckpt = tmp_path / "best.pt"
        save_checkpoint(
            model=model, optimizer=opt, epoch=1, global_step=10,
            monitor_metric="logit_gap", monitor_value=0.2, cfg_hash="abc123",
            path=ckpt, manifest_version="v1.1", diff_ids_applied=["f006"],
        )
        return InferencePredictor.from_checkpoint(
            checkpoint_path=ckpt, model=model,
            inference_cfg=_valid_inference_yaml()["inference"],
            tokenize_fn=simple_tokenize,
        )

    def test_payload_digest_deterministic(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        seq = "ACDEFGHIKLMNPQRSTVWY" * 3
        r1 = predictor.predict_protein(seq)
        r2 = predictor.predict_protein(seq)
        p1 = format_prediction_payload(r1, "P1")
        p2 = format_prediction_payload(r2, "P2")
        d1 = compute_payload_digest(p1)
        d2 = compute_payload_digest(p2)
        assert d1 == d2, f"Digests differ: {d1} vs {d2}"

    def test_no_nan_inf_in_payload(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        seq = "ACDEFGHIKLMNPQRSTVWY" * 3
        result = predictor.predict_protein(seq)
        payload = format_prediction_payload(result, "P_check")

        assert math.isfinite(payload["global_risk"])
        for w in payload["window_logits"]:
            assert math.isfinite(w["z"]), f"Non-finite z at {w}"
        for r in payload["residue_hotspot"]:
            assert math.isfinite(r["h_raw"]), f"Non-finite h_raw at {r}"
            assert math.isfinite(r["h_processed"]), f"Non-finite h_processed at {r}"

    def test_exported_json_reloads_identically(self, tmp_path: Path):
        predictor = self._build_predictor(tmp_path)
        result = predictor.predict_protein("A" * 40)
        payload = format_prediction_payload(result, "P_reload")
        path = export_prediction_json(payload, tmp_path / "P_reload.json")

        with open(path) as f:
            reloaded = json.load(f)
        assert compute_payload_digest(reloaded) == compute_payload_digest(payload)


# ── F7: End-to-End Inference Smoke Gate ──────────────────────────────────────

class TestF7SmokeGate:
    def _build_predictor(self, tmp_path: Path) -> InferencePredictor:
        torch.manual_seed(99)
        model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ckpt = tmp_path / "best.pt"
        save_checkpoint(
            model=model, optimizer=opt, epoch=2, global_step=20,
            monitor_metric="logit_gap", monitor_value=0.3, cfg_hash="smoke",
            path=ckpt, manifest_version="v1.1", diff_ids_applied=["f007"],
        )
        return InferencePredictor.from_checkpoint(
            checkpoint_path=ckpt, model=model,
            inference_cfg=_valid_inference_yaml()["inference"],
            tokenize_fn=simple_tokenize,
        )

    def test_smoke_short_protein_e2e(self, tmp_path: Path):
        """Short protein (L=50 < 1022): full pipeline export."""
        predictor = self._build_predictor(tmp_path)
        seq = "ACDEFGHIKLMNPQRSTVWY" * 3  # 60 aa
        result = predictor.predict_protein(seq)
        payload = format_prediction_payload(
            result, "SHORT_01",
            checkpoint_metadata=predictor.checkpoint_metadata,
        )
        out = export_prediction_json(payload, tmp_path / "predictions" / "SHORT_01.json")
        assert out.exists()

        with open(out) as f:
            loaded = json.load(f)
        # Shape sanity
        L = len(seq)
        expected_windows = sum(max(0, L - k + 1) for k in range(12, 26))
        assert len(loaded["window_logits"]) == expected_windows
        assert len(loaded["residue_hotspot"]) == L
        assert math.isfinite(loaded["global_risk"])

    def test_smoke_long_protein_e2e(self, tmp_path: Path):
        """Long protein (L=1500 > 1022): chunked pipeline export."""
        predictor = self._build_predictor(tmp_path)
        seq = "ACDEFGHIKLMNPQRSTVWY" * 75  # 1500 aa
        result = predictor.predict_protein(seq)
        payload = format_prediction_payload(
            result, "LONG_01",
            checkpoint_metadata=predictor.checkpoint_metadata,
        )
        out = export_prediction_json(payload, tmp_path / "predictions" / "LONG_01.json")
        assert out.exists()

        with open(out) as f:
            loaded = json.load(f)
        L = len(seq)
        expected_windows = sum(max(0, L - k + 1) for k in range(12, 26))
        assert len(loaded["window_logits"]) == expected_windows
        assert len(loaded["residue_hotspot"]) == L
        assert math.isfinite(loaded["global_risk"])
        # Confirm chunking was used
        assert result["debug"]["encode"]["n_chunks"] > 1

    def test_smoke_batch_summary(self, tmp_path: Path):
        """Multi-protein batch with summary artifact."""
        predictor = self._build_predictor(tmp_path)
        pred_dir = tmp_path / "predictions"

        ids = []
        digests = []
        for name, seq in [("P_short", "A" * 50), ("P_long", "A" * 1500)]:
            result = predictor.predict_protein(seq)
            payload = format_prediction_payload(result, name)
            export_prediction_json(payload, pred_dir / f"{name}.json")
            ids.append(name)
            digests.append(compute_payload_digest(payload))

        summary_path = write_prediction_summary(ids, digests, "smoke_hash", pred_dir)
        assert summary_path.exists()
        with open(summary_path) as f:
            s = json.load(f)
        assert s["n_proteins"] == 2
        assert set(s["protein_ids"]) == {"P_short", "P_long"}
