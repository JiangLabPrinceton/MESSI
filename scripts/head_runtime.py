"""Shared Head configuration identity and production scorer construction."""
from __future__ import annotations
import hashlib
from pathlib import Path

def _compute_head_config_hash(config_dir: Path) -> str:
    """SHA-256 over the three head config YAMLs in order (model/ablation/inference)."""
    h = hashlib.sha256()
    for fname in ("model.yaml", "model_ablation.yaml", "inference.yaml"):
        path = Path(config_dir) / fname
        if path.exists():
            h.update(path.read_bytes())
        h.update(b"\x00")
    return h.hexdigest()


def build_head_scorer(
    setup,
    *,
    window_k_min: int,
    window_k_max: int,
    static_cache_path: Path | None = None,
    static_cache_meta_path: Path | None = None,
    static_cache_policy: str = "lazy_write",
):
    """Construct the production InferencePredictor and wrap it in OnlineHeadScorer.

    Called ONCE per process (when the controller is enabled). The head model is
    reused across all proteins and designs.
    """
    # Local import keeps the script importable without the head deps in unit tests.
    from epitope_head.configs import (
        load_ablation_config,
        load_inference_config,
        load_model_config,
    )
    from scripts.score_head import build_predictor

    from inverse_folding.reference_flow.head_scoring import OnlineHeadScorer

    predictor = build_predictor(
        model_cfg=load_model_config(setup.head_config_dir / "model.yaml"),
        ablation_cfg=load_ablation_config(setup.head_config_dir / "model_ablation.yaml"),
        inference_cfg=load_inference_config(setup.head_config_dir / "inference.yaml"),
        variant_id=setup.head_variant_id,
        checkpoint_path=setup.head_checkpoint,
        device=setup.head_device,
    )
    return OnlineHeadScorer(
        predictor=predictor,
        allele=setup.allele,
        allele_idx=setup.head_allele_idx,
        head_checkpoint_digest=hashlib.sha256(
            setup.head_checkpoint.read_bytes()
        ).hexdigest() if setup.head_checkpoint.exists() else "",
        head_config_hash=_compute_head_config_hash(setup.head_config_dir),
        score_scale=setup.config.head.score_scale,
        window_k_min=int(window_k_min),
        window_k_max=int(window_k_max),
        static_cache_path=static_cache_path,
        static_cache_meta_path=static_cache_meta_path,
        static_cache_policy=static_cache_policy,
        window_batch_size=setup.head_window_batch_size,
    )



def load_epitope_predictor(config_dir: str, checkpoint_path: str, variant_id: str, device: str):
    """Use the canonical checkpoint constructor for benchmark and inference paths."""
    from epitope_head.configs import (
        load_ablation_config, load_inference_config, load_model_config,
    )
    from scripts.score_head import build_predictor

    config_dir = Path(config_dir)
    return build_predictor(
        model_cfg=load_model_config(config_dir / "model.yaml"),
        ablation_cfg=load_ablation_config(config_dir / "model_ablation.yaml"),
        inference_cfg=load_inference_config(config_dir / "inference.yaml"),
        variant_id=variant_id, checkpoint_path=Path(checkpoint_path), device=device,
    )
