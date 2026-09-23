"""T2 contract tests: batch facade for InferencePredictor (Phase D1).

Verifies:
1. ``predict_protein(window_batch_size=...)`` forwards through ``enumerate_and_score``
   and produces output identical to the default batch size for the same sequence.
2. ``predict_proteins(records)`` returns one result per record in input order.
3. The batch facade reuses a single ``InferencePredictor`` instance (the model is
   not re-constructed per record) — proved via an ``enumerate_and_score`` counter.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from epitope_head.inference.predictor import (
    InferencePredictor,
    build_epitope_scorer_from_config,
)
from tests.epitope_head.inference.test_module_f_contract import (
    DeterministicFrozenEncoder,
    _valid_inference_yaml,
    _valid_model_cfg,
    simple_tokenize,
)


def _build_predictor(tmp_path: Path) -> InferencePredictor:
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


def test_predict_protein_window_batch_size_matches_default(tmp_path: Path):
    """Forwarding window_batch_size must not change scores or ordering."""
    predictor = _build_predictor(tmp_path)
    seq = "ACDEFGHIKLMNPQRSTVWY" * 2  # 40 residues

    default_result = predictor.predict_protein(seq)
    batched_result = predictor.predict_protein(seq, window_batch_size=2)

    assert len(default_result["window_logits"]) == len(batched_result["window_logits"])
    # Batched matmul changes float32 accumulation order; tolerate ~1e-5 drift.
    for d, b in zip(default_result["window_logits"], batched_result["window_logits"]):
        assert d["start_0b"] == b["start_0b"]
        assert d["end_0b"] == b["end_0b"]
        assert d["k"] == b["k"]
        assert d["z"] == pytest.approx(b["z"], abs=1e-5)

    torch.testing.assert_close(
        default_result["residue_hotspot"],
        batched_result["residue_hotspot"],
        atol=1e-5,
        rtol=1e-5,
    )
    assert default_result["global_risk"] == pytest.approx(
        batched_result["global_risk"], abs=1e-5
    )


def test_predict_proteins_preserves_input_order(tmp_path: Path):
    predictor = _build_predictor(tmp_path)
    records = [
        ("protein_A", "ACDEFGHIKL" * 3),
        ("protein_B", "MNPQRSTVWY" * 3),
        ("protein_C", "ACDEFG" * 5),
    ]

    results = predictor.predict_proteins(records)

    assert len(results) == 3
    assert [r["protein_id"] for r in results] == ["protein_A", "protein_B", "protein_C"]
    assert results[0]["prediction"]["meta"]["protein_len"] == len(records[0][1])
    assert results[1]["prediction"]["meta"]["protein_len"] == len(records[1][1])
    assert results[2]["prediction"]["meta"]["protein_len"] == len(records[2][1])


def test_predict_proteins_batched_encode_matches_serial_predictions(tmp_path: Path):
    predictor = _build_predictor(tmp_path)
    records = [
        ("protein_A", "ACDEFGHIKLMNPQRSTVWY" * 2),
        ("protein_B", "MNPQRSTVWYACDEFGHIKL" * 2),
    ]

    serial = [
        predictor.predict_protein(seq, window_batch_size=7)
        for _, seq in records
    ]
    batched = predictor.predict_proteins(records, window_batch_size=7)

    assert [r["protein_id"] for r in batched] == ["protein_A", "protein_B"]
    for serial_pred, batched_row in zip(serial, batched):
        batched_pred = batched_row["prediction"]
        assert len(serial_pred["window_logits"]) == len(batched_pred["window_logits"])
        for s, b in zip(serial_pred["window_logits"], batched_pred["window_logits"]):
            assert (s["start_0b"], s["end_0b"], s["k"]) == (
                b["start_0b"],
                b["end_0b"],
                b["k"],
            )
            assert b["z"] == pytest.approx(s["z"], abs=1e-5)
        torch.testing.assert_close(
            batched_pred["residue_hotspot"],
            serial_pred["residue_hotspot"],
            atol=1e-5,
            rtol=1e-5,
        )
        assert batched_pred["global_risk"] == pytest.approx(
            serial_pred["global_risk"], abs=1e-5
        )


def test_same_length_records_score_spans_in_one_batched_scorer_pass(tmp_path: Path):
    """Same-length candidates share one span template, so the scorer MLP must run
    over all K*W spans in ceil(K*W / window_batch_size) passes — NOT one pass per
    candidate. Drives the batched span-scoring path."""
    predictor = _build_predictor(tmp_path)
    # 4 distinct sequences, identical length (30) -> identical span template.
    records = [
        ("p1", "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"),
        ("p2", "MNPQRSTVWYACDEFGHIKLMNPQRSTVWY"),
        ("p3", "WYACDEFGHIKLMNPQRSTVWYACDEFGHI"),
        ("p4", "GHIKLMNPQRSTVWYACDEFGHIKLMNPQR"),
    ]
    assert len({len(seq) for _, seq in records}) == 1  # precondition: same length

    scorer_calls = {"n": 0}

    def _hook(_module, _inp, _out):
        scorer_calls["n"] += 1

    handle = predictor.model.scorer.register_forward_hook(_hook)
    try:
        predictor.predict_proteins(records, window_batch_size=4096)
    finally:
        handle.remove()

    # K*W = 4 * 175 = 700 <= 4096 -> exactly one batched scorer pass for all
    # candidates. The serial per-candidate path would call the scorer 4 times.
    assert scorer_calls["n"] == 1


@pytest.mark.parametrize(
    "center_method,clamp_method",
    [("median", "none"), ("mean", "relu"), ("median", "softplus")],
)
def test_batched_same_length_aggregation_matches_serial(
    tmp_path: Path, center_method: str, clamp_method: str
):
    """The batched span+aggregation path must match the per-candidate serial path
    (window z, h_raw, residue_hotspot, global_risk) across center/clamp variants."""
    predictor = _build_predictor(tmp_path)
    predictor.inference_cfg["hotspot_center_method"] = center_method
    predictor.inference_cfg["hotspot_clamp"] = clamp_method

    records = [
        ("p1", "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"),
        ("p2", "MNPQRSTVWYACDEFGHIKLMNPQRSTVWY"),
        ("p3", "WYACDEFGHIKLMNPQRSTVWYACDEFGHI"),
    ]
    assert len({len(seq) for _, seq in records}) == 1

    serial = [predictor.predict_protein(seq) for _, seq in records]
    batched = predictor.predict_proteins(records)

    for serial_pred, batched_row in zip(serial, batched):
        batched_pred = batched_row["prediction"]
        assert [
            (w["start_0b"], w["end_0b"], w["k"]) for w in serial_pred["window_logits"]
        ] == [(w["start_0b"], w["end_0b"], w["k"]) for w in batched_pred["window_logits"]]
        for s, b in zip(serial_pred["window_logits"], batched_pred["window_logits"]):
            assert b["z"] == pytest.approx(s["z"], abs=1e-5)
        torch.testing.assert_close(
            batched_pred["residue_hotspot"], serial_pred["residue_hotspot"],
            atol=1e-5, rtol=1e-5,
        )
        torch.testing.assert_close(
            batched_pred["debug"]["h_raw"], serial_pred["debug"]["h_raw"],
            atol=1e-5, rtol=1e-5,
        )
        assert batched_pred["global_risk"] == pytest.approx(
            serial_pred["global_risk"], abs=1e-5
        )


@pytest.mark.parametrize("length", [12, 13, 26])
def test_batched_same_length_matches_serial_at_boundary_lengths(tmp_path: Path, length: int):
    """Batched aggregation must match serial at boundary lengths: L=min_k (single
    window), L=min_k+1, and L just above max_k — exercising small-W cover gathers."""
    predictor = _build_predictor(tmp_path)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    records = [
        ("p1", (alphabet * 2)[:length]),
        ("p2", (alphabet * 2)[3 : 3 + length]),
    ]
    assert len({len(seq) for _, seq in records}) == 1
    assert len(records[0][1]) == length

    serial = [predictor.predict_protein(seq) for _, seq in records]
    batched = predictor.predict_proteins(records)

    for serial_pred, batched_row in zip(serial, batched):
        batched_pred = batched_row["prediction"]
        assert len(serial_pred["window_logits"]) == len(batched_pred["window_logits"])
        for s, b in zip(serial_pred["window_logits"], batched_pred["window_logits"]):
            assert (s["start_0b"], s["end_0b"], s["k"]) == (b["start_0b"], b["end_0b"], b["k"])
            assert b["z"] == pytest.approx(s["z"], abs=1e-5)
        torch.testing.assert_close(
            batched_pred["residue_hotspot"], serial_pred["residue_hotspot"],
            atol=1e-5, rtol=1e-5,
        )
        assert batched_pred["global_risk"] == pytest.approx(
            serial_pred["global_risk"], abs=1e-5
        )


def test_predict_proteins_uses_one_batched_encoder_call_for_short_records():
    class EncodeCountingPredictor(InferencePredictor):
        encode_chunk_calls = 0

        def _encode_chunk(self, seq: str):
            EncodeCountingPredictor.encode_chunk_calls += 1
            return super()._encode_chunk(seq)

    model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
    predictor = EncodeCountingPredictor(
        model=model,
        inference_cfg=_valid_inference_yaml()["inference"],
        tokenize_fn=simple_tokenize,
    )
    EncodeCountingPredictor.encode_chunk_calls = 0

    predictor.predict_proteins(
        [
            ("p1", "ACDEFGHIKL" * 2),
            ("p2", "MNPQRSTVWY" * 2),
            ("p3", "ACDEFG" * 4),
        ]
    )

    assert EncodeCountingPredictor.encode_chunk_calls == 0


def test_predict_proteins_reuses_single_model_instance(tmp_path: Path):
    """Facade must not re-instantiate the underlying model per record."""

    class CountingPredictor(InferencePredictor):
        enumerate_call_count = 0
        observed_model_ids: set[int] = set()

        def enumerate_and_score(self, *args, **kwargs):
            CountingPredictor.enumerate_call_count += 1
            CountingPredictor.observed_model_ids.add(id(self.model))
            return super().enumerate_and_score(*args, **kwargs)

    model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
    predictor = CountingPredictor(
        model=model,
        inference_cfg=_valid_inference_yaml()["inference"],
        tokenize_fn=simple_tokenize,
    )

    records = [
        ("p1", "ACDEFGHIKL" * 2),
        ("p2", "MNPQRSTVWY" * 2),
        ("p3", "ACDEFG" * 3),
    ]
    CountingPredictor.enumerate_call_count = 0
    CountingPredictor.observed_model_ids = set()

    _ = predictor.predict_proteins(records)

    assert CountingPredictor.enumerate_call_count == len(records)
    assert CountingPredictor.observed_model_ids == {id(model)}
    assert predictor.model is model


def test_predict_proteins_forwards_window_batch_size(tmp_path: Path):
    """Explicit window_batch_size propagates into enumerate_and_score on the
    mixed-length fallback path (same-length records route to the batched span
    path, which chunks the scorer directly — see the batched test below)."""

    class RecordingPredictor(InferencePredictor):
        recorded_batch_sizes: list[int] = []

        def enumerate_and_score(self, *args, **kwargs):
            RecordingPredictor.recorded_batch_sizes.append(int(kwargs.get("window_batch_size", -1)))
            return super().enumerate_and_score(*args, **kwargs)

    model = build_epitope_scorer_from_config(_valid_model_cfg(), DeterministicFrozenEncoder())
    predictor = RecordingPredictor(
        model=model,
        inference_cfg=_valid_inference_yaml()["inference"],
        tokenize_fn=simple_tokenize,
    )

    RecordingPredictor.recorded_batch_sizes = []
    # Mixed lengths (30, 24) -> per-candidate fallback -> enumerate_and_score runs.
    predictor.predict_proteins(
        [("p1", "ACDEFG" * 5), ("p2", "MNPQRS" * 4)], window_batch_size=3
    )

    assert RecordingPredictor.recorded_batch_sizes == [3, 3]


def test_batched_span_path_chunks_scorer_by_window_batch_size(tmp_path: Path):
    """On the same-length batched path, window_batch_size chunks the K*W spans
    fed to the scorer MLP: ceil(K*W / wbs) passes."""
    predictor = _build_predictor(tmp_path)
    # 2 same-length (30) candidates -> W = 175 windows each -> K*W = 350 spans.
    records = [
        ("p1", "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"),
        ("p2", "MNPQRSTVWYACDEFGHIKLMNPQRSTVWY"),
    ]
    assert len({len(seq) for _, seq in records}) == 1

    scorer_calls = {"n": 0}

    def _hook(_module, _inp, _out):
        scorer_calls["n"] += 1

    handle = predictor.model.scorer.register_forward_hook(_hook)
    try:
        predictor.predict_proteins(records, window_batch_size=200)
    finally:
        handle.remove()

    # K*W = 350, wbs = 200 -> ceil(350 / 200) = 2 scorer passes.
    assert scorer_calls["n"] == 2
