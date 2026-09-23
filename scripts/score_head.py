"""Score sequences with an allele-matched canonical Head checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from epitope_head.configs import (
    load_ablation_config,
    load_inference_config,
    load_model_config,
)
from epitope_head.inference.export import (
    compute_payload_digest,
    export_prediction_json,
    format_prediction_payload,
    write_prediction_summary,
)
from epitope_head.inference.flank_ablation import resolve_cnn_variant_profile
from epitope_head.inference.predictor import InferencePredictor
from epitope_head.training.encoders import build_encoder
from epitope_head.training.model import EpitopeScorer
from epitope_head.training.trainer import config_hash

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Epitope Head CNN inference launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config-dir",
        type=str,
        default=str(PROJECT_ROOT / "epitope_head" / "configs"),
        help="Directory containing model.yaml, model_ablation.yaml, and inference.yaml",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the allele-matched canonical Head checkpoint",
    )
    p.add_argument(
        "--variant-id",
        type=str,
        default="LC1",
        choices=("LC1",),
        help="Canonical multiscale CNN profile",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. cpu / cuda / cuda:0 / mps",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "predictions"),
        help="Directory to write per-protein prediction JSON files",
    )
    p.add_argument(
        "--input-fasta",
        type=str,
        default=None,
        help="Input FASTA path for batch prediction",
    )
    p.add_argument(
        "--protein-id",
        type=str,
        default=None,
        help="Protein ID for single-sequence mode",
    )
    p.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Amino-acid sequence for single-sequence mode",
    )
    p.add_argument(
        "--max-proteins",
        type=int,
        default=None,
        help="Optional cap on number of FASTA proteins to process",
    )
    p.add_argument(
        "--stdout-json",
        action="store_true",
        help="Print full prediction payload to stdout (useful for single-sequence visualization)",
    )
    return p.parse_args()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Read FASTA as list of (protein_id, sequence)."""
    records: list[tuple[str, str]] = []
    cur_id: str | None = None
    cur_seq: list[str] = []

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    records.append((cur_id, "".join(cur_seq).upper()))
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None:
        records.append((cur_id, "".join(cur_seq).upper()))
    return records


def resolve_inputs(args: argparse.Namespace) -> list[tuple[str, str]]:
    has_single = args.protein_id is not None or args.sequence is not None
    has_fasta = args.input_fasta is not None

    if has_single and has_fasta:
        raise ValueError("Use either single-sequence args (--protein-id/--sequence) or --input-fasta, not both")
    if not has_single and not has_fasta:
        raise ValueError("Provide either (--protein-id and --sequence) or --input-fasta")

    if has_single:
        if args.protein_id is None or args.sequence is None:
            raise ValueError("Single-sequence mode requires both --protein-id and --sequence")
        return [(args.protein_id, args.sequence.upper())]

    records = read_fasta(Path(args.input_fasta))
    if args.max_proteins is not None:
        records = records[: args.max_proteins]
    return records


def build_predictor(
    model_cfg: dict,
    ablation_cfg: dict,
    inference_cfg: dict,
    variant_id: str,
    checkpoint_path: Path,
    device: str,
) -> InferencePredictor:
    profile_cfg = resolve_cnn_variant_profile(ablation_cfg, variant_id)
    encoder_type = profile_cfg["encoder_type"]
    d_enc = int(profile_cfg["d_enc"])
    encoder_cfg = profile_cfg["encoder_cfg"]
    frozen = ablation_cfg["frozen_constants"]

    inference_cfg = dict(inference_cfg)
    inference_cfg["device"] = device

    encoder, tokenizer = build_encoder(encoder_type, d_enc, encoder_cfg)
    model = EpitopeScorer(
        encoder=encoder,
        d_enc=d_enc,
        d_proj=int(frozen["d_proj"]),
        length_emb_dim=int(model_cfg["length_embedding_dim"]),
        allele_emb_dim=int(model_cfg["allele_embedding_dim"]),
        min_k=int(frozen["min_k"]),
        max_k=int(frozen["max_k"]),
        n_alleles=int(model_cfg.get("n_alleles", 1)),
        scorer_hidden_dim=int(frozen["scorer_hidden_dim"]),
        scorer_activation=str(frozen["scorer_activation"]),
        scorer_dropout=float(frozen.get("scorer_dropout", model_cfg.get("scorer_dropout", 0.3))),
        logit_scale_init=float(model_cfg.get("logit_scale_init", 10.0)),
        logit_scale_max=float(model_cfg.get("logit_scale_max", 20.0)),
        projection_layer_norm=bool(model_cfg.get("projection_layer_norm", True)),
        pad_left_init=str(model_cfg.get("pad_left_init", "zeros")),
        pad_right_init=str(model_cfg.get("pad_right_init", "zeros")),
    )
    logger.info("Using CNN encoder variant=%s (%s)", variant_id, encoder_type)
    return InferencePredictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        inference_cfg=inference_cfg,
        tokenize_fn=tokenizer,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    config_dir = Path(args.config_dir)
    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_cfg = load_model_config(config_dir / "model.yaml")
    ablation_cfg = load_ablation_config(config_dir / "model_ablation.yaml")
    inference_cfg = load_inference_config(config_dir / "inference.yaml")
    predictor = build_predictor(
        model_cfg=model_cfg,
        ablation_cfg=ablation_cfg,
        inference_cfg=inference_cfg,
        variant_id=args.variant_id,
        checkpoint_path=checkpoint_path,
        device=args.device,
    )

    records = resolve_inputs(args)
    logger.info("Loaded %d input proteins", len(records))
    out_dir.mkdir(parents=True, exist_ok=True)

    inf_cfg_hash = config_hash({"inference": inference_cfg, "model": model_cfg})
    ids: list[str] = []
    digests: list[str] = []

    for idx, (protein_id, seq) in enumerate(records, 1):
        result = predictor.predict_protein(seq)
        payload = format_prediction_payload(
            result,
            protein_id=protein_id,
            checkpoint_metadata=predictor.checkpoint_metadata,
            config_hash=inf_cfg_hash,
        )
        output_path = out_dir / f"{protein_id}.json"
        export_prediction_json(payload, output_path)
        digest = compute_payload_digest(payload)

        ids.append(protein_id)
        digests.append(digest)
        logger.info("[%d/%d] wrote %s (digest=%s)", idx, len(records), output_path, digest)

        if args.stdout_json:
            print(json.dumps(payload, indent=2, default=str))

    summary_path = write_prediction_summary(ids, digests, inf_cfg_hash, out_dir)
    logger.info("Prediction summary: %s", summary_path)
    print(json.dumps({"n_proteins": len(ids), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
