"""Train the canonical a1res03 multiscale CNN Head (LC1)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from epitope_head.configs import (
    load_ablation_config,
    load_data_config,
    load_model_config,
    load_train_config,
    validate_himp_train_blocks,
)
from epitope_head.data.netmhciipan_mutation import (
    MutationRegistryIndex,
    apply_runtime_augmentation,
)
from epitope_head.training.datamodule import (
    build_chunk_samples,
    build_dataloader,
    load_augmented_proteins,
    load_split_proteins,
)
from epitope_head.training.encoders import AATokenizer, build_encoder
from epitope_head.training.model import EpitopeScorer, FrozenESMEncoder, ESMTokenizer
from epitope_head.training.trainer import Trainer

logger = logging.getLogger(__name__)

VALID_ENCODER_IDS = {"LC1"}
STAGE_I_VARIANT_IDS = {"LC1"}
VALID_VARIANT_IDS = VALID_ENCODER_IDS | STAGE_I_VARIANT_IDS
VALID_SEEDS = {42, 43, 44}


def apply_fixed_epoch_overrides(train_cfg: dict, max_epochs, no_early_stopping: bool) -> dict:
    """Pin the epoch budget and (optionally) switch early stopping off.

    Used by the full-data / production refit: training on train+val+test leaves no
    honest val split, so the epoch is transferred from the CV goal epoch and held
    fixed. `checkpoint_every_n_epochs` is deliberately left alone — the caller
    aligns the budget with the save grid so the target epoch lands on disk.
    """
    if max_epochs is not None:
        if max_epochs < 1:
            raise ValueError(f"--max-epochs must be >= 1, got {max_epochs}")
        train_cfg["max_epochs"] = max_epochs
    if no_early_stopping:
        train_cfg["early_stopping_patience"] = train_cfg["max_epochs"] + 1
    return train_cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Epitope Head v2 encoder ablation (single run)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--variant-id", type=str, default="LC1", choices=sorted(VALID_VARIANT_IDS),
                    help="Canonical multiscale CNN profile")
    p.add_argument("--encoder-id", type=str, default=None, choices=sorted(VALID_ENCODER_IDS),
                    help="(Deprecated: use --variant-id) Encoder ablation profile (E0, E1, E2)")
    p.add_argument("--seed", type=int, required=True,
                    help="Random seed (must be one of 42, 43, 44)")
    p.add_argument("--profile", type=str, default="strict", choices=["strict", "balanced"],
                    help="Data profile (strict or balanced)")
    p.add_argument("--device", type=str, default="cpu",
                    help="Torch device (cpu, cuda, cuda:0, mps)")

    p.add_argument("--config-dir", type=str,
                    default=str(PROJECT_ROOT / "epitope_head" / "configs"),
                    help="Config directory")
    p.add_argument("--override-config", type=str,
                    default=str(PROJECT_ROOT / "epitope_head/configs/cnn_himp_a1_res03.yaml"),
                    help="Path to a YAML file with highest-priority overrides. "
                         "Applied AFTER all other configs (train.yaml + ablation loss_overrides). "
                         "Format mirrors train.yaml, e.g. train.loss.tau, train.lr, etc.")
    p.add_argument("--data-dir", type=str, default=None,
                    help="Manifest directory (default: outputs/manifests)")
    p.add_argument("--splits-subdir", type=str, default=None,
                    help="Wave-3 CV: subdir under splits/<profile>/ holding fold-specific "
                         "train_ids.txt / val_ids.txt (e.g. 'cv5/fold0'). Default uses the "
                         "top-level split.")
    p.add_argument("--output-root", type=str, default=None,
                    help="Root for ablation outputs (default: outputs/ablation/encoder_v2)")
    p.add_argument("--max-epochs", type=int, default=None,
                    help="Override the config's max_epochs. For a full-data refit the epoch "
                         "budget is transferred from CV (there is no held-out val to stop on); "
                         "keep it on the checkpoint_every_n_epochs grid so the target epoch is "
                         "actually saved.")
    p.add_argument("--no-early-stopping", action="store_true",
                    help="Disable early stopping (patience > max_epochs) so a fixed-epoch run "
                         "reaches its budget. Required for full-data refits, whose val split is "
                         "inside the training set and therefore cannot stop training honestly.")

    smoke = p.add_argument_group("smoke testing")
    smoke.add_argument("--smoke", action="store_true",
                       help="Smoke mode: small data + few epochs")
    smoke.add_argument("--smoke-n-train", type=int, default=20)
    smoke.add_argument("--smoke-n-val", type=int, default=5)
    smoke.add_argument("--smoke-epochs", type=int, default=2)

    wb = p.add_argument_group("wandb")
    wb.add_argument("--wandb", action="store_true")
    wb.add_argument("--wandb-entity", type=str, default=None)
    wb.add_argument("--wandb-project", type=str, default="Immune-Design")
    wb.add_argument("--wandb-name", type=str, default=None)

    aug = p.add_argument_group("augmentation (Stage J)")
    aug.add_argument("--aug-train-parquet", type=str, default=None,
                     help="(Deprecated) Path to augmented train-only parquet (static union)")
    aug.add_argument("--registry-path", type=str, default=None,
                     help="Path to mutation registry parquet for runtime p_aug replacement")
    aug.add_argument("--p-aug", type=float, default=0.20,
                     help="Probability of replacing a train protein with a mutant (default: 0.20)")

    return p.parse_args()


def main() -> dict:
    args = parse_args()

    # Resolve variant-id vs deprecated encoder-id
    if args.variant_id is not None:
        variant_id = args.variant_id
    elif args.encoder_id is not None:
        variant_id = args.encoder_id
        logger.warning("--encoder-id is deprecated, use --variant-id instead")
    else:
        raise SystemExit("Error: one of --variant-id or --encoder-id is required")

    is_stage_i = variant_id in STAGE_I_VARIANT_IDS

    # Validate seed
    if args.seed not in VALID_SEEDS and not args.smoke:
        logger.warning("Seed %d not in canonical set %s — results may not be comparable", args.seed, VALID_SEEDS)

    log_level = logging.DEBUG if args.smoke else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # ── Load configs ──────────────────────────────────────────────────
    config_dir = Path(args.config_dir)
    data_cfg = load_data_config(config_dir / "data.yaml")
    model_cfg = load_model_config(config_dir / "model.yaml")
    train_cfg = load_train_config(config_dir / "train.yaml")
    ablation_cfg = load_ablation_config(config_dir / "model_ablation.yaml")

    profile_cfg = ablation_cfg["profiles"][variant_id]
    encoder_type = profile_cfg["encoder_type"]
    d_enc = profile_cfg["d_enc"]
    encoder_cfg = profile_cfg["encoder_cfg"]

    logger.info("Ablation run: variant=%s, seed=%d, profile=%s", variant_id, args.seed, args.profile)

    # ── Apply loss_overrides from profile (Module I) ────────────────
    loss_overrides = profile_cfg.get("loss_overrides", {})
    if loss_overrides:
        train_cfg["loss"].update(loss_overrides)
        logger.info("Loss overrides applied: %s", loss_overrides)

    # ── Stage-I: override monitor metric to pp_ap ─────────────────
    if is_stage_i:
        stage_i_constants = ablation_cfg.get("stage_i_frozen_constants", {})
        if "monitor_metric" in stage_i_constants:
            train_cfg["monitor_metric"] = stage_i_constants["monitor_metric"]
            logger.info("Stage-I monitor metric: %s", train_cfg["monitor_metric"])

    # ── User override config (HIGHEST priority) ──────────────────
    if args.override_config:
        override_path = Path(args.override_config)
        if not override_path.exists():
            logger.error("Override config not found: %s", override_path)
            sys.exit(1)
        with open(override_path) as f:
            override_raw = yaml.safe_load(f)

        def _deep_update(base: dict, patch: dict) -> dict:
            for k, v in patch.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    _deep_update(base[k], v)
                else:
                    base[k] = v
            return base

        # train overrides: prefer the explicit `train:` block; a bare mapping
        # (no train/model wrapper) is treated as the train block for back-compat.
        has_model = isinstance(override_raw, dict) and "model" in override_raw
        train_override = override_raw.get("train", {} if has_model else override_raw)
        _deep_update(train_cfg, train_override)
        # Re-validate HIMP blocks after override merge — load_train_config only
        # validated the base yaml, so unknown schedule names / wrong types in
        # the override would otherwise reach training silently.
        validate_himp_train_blocks(train_cfg)
        # Wave-4: an optional `model:` block configures the head itself
        # (e.g. dual-head enable_boundary_head); merge it into model_cfg.
        if has_model:
            _deep_update(model_cfg, override_raw["model"])
        logger.info("User override config applied from %s: %s", override_path, train_override)

    # ── Override seed ─────────────────────────────────────────────────
    train_cfg["seed"] = args.seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if train_cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(True)

    # ── Resolve data paths ────────────────────────────────────────────
    manifest_dir = Path(args.data_dir) if args.data_dir else (PROJECT_ROOT / "outputs" / "manifests")
    profile = args.profile
    samples_path = manifest_dir / f"protein_samples_{profile}.parquet"
    split_dir = manifest_dir / "splits" / profile
    if getattr(args, "splits_subdir", None):
        # Wave-3 CV: read fold-specific id files, e.g. splits/<profile>/cv5/fold0/.
        split_dir = split_dir / args.splits_subdir
    train_ids_path = split_dir / "train_ids.txt"
    val_ids_path = split_dir / "val_ids.txt"

    for p in [samples_path, train_ids_path, val_ids_path]:
        if not p.exists():
            logger.error("Required artifact missing: %s", p)
            sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────
    train_entries = load_split_proteins(samples_path, train_ids_path)
    val_entries = load_split_proteins(samples_path, val_ids_path)
    logger.info("Loaded %d train, %d val proteins", len(train_entries), len(val_entries))

    # ── Stage J: runtime augmentation via registry (preferred) ─────
    registry_idx = None
    if args.registry_path:
        registry_idx = MutationRegistryIndex.from_parquet(args.registry_path)
        logger.info(
            "Loaded mutation registry: %d eligible proteins, %d eligible mutations, p_aug=%.2f",
            registry_idx.n_eligible_proteins, registry_idx.n_eligible_mutations, args.p_aug,
        )
    elif args.aug_train_parquet:
        # Deprecated static union fallback
        aug_entries = load_augmented_proteins(args.aug_train_parquet)
        logger.info("(deprecated) Static union: %d augmented train entries from %s",
                     len(aug_entries), args.aug_train_parquet)
        train_entries = train_entries + aug_entries

    # ── Fixed-epoch overrides (full-data refit) ───────────────────────
    apply_fixed_epoch_overrides(train_cfg, args.max_epochs, args.no_early_stopping)
    if args.max_epochs is not None or args.no_early_stopping:
        logger.info("Fixed-epoch overrides: max_epochs=%d, early_stopping_patience=%d",
                    train_cfg["max_epochs"], train_cfg["early_stopping_patience"])

    # ── Smoke overrides ───────────────────────────────────────────────
    if args.smoke:
        train_entries = train_entries[:args.smoke_n_train]
        val_entries = val_entries[:args.smoke_n_val]
        train_cfg["max_epochs"] = args.smoke_epochs
        train_cfg["checkpoint_every_n_epochs"] = 1
        train_cfg["early_stopping_patience"] = args.smoke_epochs + 1
        train_cfg["num_workers"] = 0
        logger.info("SMOKE MODE: %d train, %d val, %d epochs",
                     len(train_entries), len(val_entries), train_cfg["max_epochs"])

    if sys.platform == "darwin" and train_cfg["num_workers"] > 0:
        logger.warning("macOS: forcing num_workers=0")
        train_cfg["num_workers"] = 0

    # ── Build chunks ──────────────────────────────────────────────────
    chunking = train_cfg["chunking"]
    ck_params = dict(context_len=chunking["context_len"], stride=chunking["stride"], margin=chunking["margin"])
    train_chunks = build_chunk_samples(train_entries, **ck_params)
    val_chunks = build_chunk_samples(val_entries, **ck_params)
    logger.info("Chunks: %d train, %d val", len(train_chunks), len(val_chunks))

    # CNN encoders can handle arbitrary lengths at eval — don't skip any protein
    if encoder_type in ("dilated_cnn", "multiscale_cnn"):
        eval_context_len = max(e.sequence_length for e in train_entries + val_entries)
    else:
        eval_context_len = ck_params["context_len"]

    # ── Build encoder + tokenizer ─────────────────────────────────────
    device = torch.device(args.device)

    if encoder_type == "esm2_frozen":
        logger.info("Loading ESM-2: %s ...", encoder_cfg["encoder_name"])
        try:
            import esm
        except ImportError:
            logger.error("Package 'fair-esm' not installed. Use --encoder-id E1 or E2 for non-ESM ablation.")
            sys.exit(1)
        esm_model, alphabet = getattr(esm.pretrained, encoder_cfg["encoder_name"])()
        encoder = FrozenESMEncoder(esm_model, d_enc=d_enc)
        tokenizer = ESMTokenizer(alphabet)
    else:
        encoder, tokenizer = build_encoder(encoder_type, d_enc, encoder_cfg)

    trainable_enc_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info("Encoder: %s, trainable params: %d", encoder_type, trainable_enc_params)

    # ── Build dataloaders ─────────────────────────────────────────────
    train_loader = build_dataloader(
        train_chunks, max_tokens=train_cfg["max_tokens"],
        shuffle=True, seed=args.seed, num_workers=train_cfg["num_workers"],
        tokenize_fn=tokenizer,
    )
    val_loader = build_dataloader(
        val_chunks, max_tokens=train_cfg["max_tokens"],
        shuffle=False, seed=args.seed, num_workers=0,
        tokenize_fn=tokenizer,
    )

    # ── Build model ───────────────────────────────────────────────────
    # Use frozen constants from ablation config for head params
    frozen = ablation_cfg["frozen_constants"]
    model = EpitopeScorer(
        encoder=encoder,
        d_enc=d_enc,
        d_proj=frozen["d_proj"],
        length_emb_dim=model_cfg["length_embedding_dim"],
        allele_emb_dim=model_cfg["allele_embedding_dim"],
        min_k=frozen["min_k"],
        max_k=frozen["max_k"],
        n_alleles=1,
        scorer_hidden_dim=frozen["scorer_hidden_dim"],
        scorer_activation=frozen["scorer_activation"],
        scorer_dropout=frozen.get("scorer_dropout", 0.3),
        # Wave-4 dual-head (default off -> legacy). Enabled via a `model:` block
        # in the experiment override config.
        enable_boundary_head=bool(model_cfg.get("enable_boundary_head", False)),
        boundary_head_hidden_dim=int(model_cfg.get("boundary_head_hidden_dim", 64)),
        boundary_head_dropout=float(model_cfg.get("boundary_head_dropout", 0.1)),
        use_core_scorer=bool(model_cfg.get("use_core_scorer", False)),
    )
    if model.enable_boundary_head:
        logger.info("Dual-head ENABLED (boundary_head_hidden_dim=%d)",
                    int(model_cfg.get("boundary_head_hidden_dim", 64)))
    if model.span_features.use_core_scorer:
        logger.info("Core-aware scorer ENABLED (9-mer-core feature, d_phi=%d)",
                    model.span_features.d_phi)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d trainable / %d total params", trainable, total)

    # ── Run directory ─────────────────────────────────────────────────
    if args.output_root:
        output_root = Path(args.output_root)
    elif is_stage_i:
        stage_i_constants = ablation_cfg.get("stage_i_frozen_constants", {})
        output_root = PROJECT_ROOT / stage_i_constants.get("output_root", "outputs/ablation/cnn_enhance_v2")
    else:
        output_root = PROJECT_ROOT / "outputs" / "ablation" / "encoder_v2"
    suffix = "smoke" if args.smoke else f"seed_{args.seed}"
    run_dir = output_root / "runs" / variant_id / suffix
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    # ── Registry ──────────────────────────────────────────────────────
    registry_path = output_root / "metrics" / "run_registry.jsonl"
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Resolved config snapshot ──────────────────────────────────────
    train_cfg_for_trainer = dict(train_cfg)
    train_cfg_for_trainer["manifest_version"] = "v1.1"
    train_cfg_for_trainer["diff_ids_applied"] = ["d001", "d002"]
    train_cfg_for_trainer["ablation_encoder_id"] = variant_id

    resolved = {
        "data": data_cfg,
        "model": model_cfg,
        "train": train_cfg_for_trainer,
        "ablation": {
            "variant_id": variant_id,
            "encoder_type": encoder_type,
            "d_enc": d_enc,
            "encoder_cfg": encoder_cfg,
            "frozen_constants": frozen,
        },
        "runtime": {
            "profile": profile,
            "device": str(device),
            "smoke": args.smoke,
            "seed": args.seed,
            "train_proteins": len(train_entries),
            "val_proteins": len(val_entries),
            "train_chunks": len(train_chunks),
            "val_chunks": len(val_chunks),
            "trainable_params": trainable,
            "total_params": total,
        },
    }
    with open(run_dir / "full_resolved_config.json", "w") as f:
        json.dump(resolved, f, indent=2, default=str)

    # ── W&B ───────────────────────────────────────────────────────────
    wandb_cfg = None
    if args.wandb:
        wandb_cfg = {
            "enabled": True,
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": args.wandb_name or f"{variant_id}_seed{args.seed}",
        }

    # ── Build epoch hook for runtime augmentation ──────────────────
    epoch_hook = None
    if registry_idx is not None:
        # Closure captures: train_entries (base), registry_idx, args, ck_params, train_cfg, tokenizer
        _base_train_entries = list(train_entries)  # snapshot base entries

        def _aug_epoch_hook(trainer_obj, epoch):
            """Rebuild train_loader with runtime-augmented entries each epoch."""
            aug_entries, aug_stats = apply_runtime_augmentation(
                _base_train_entries, registry_idx,
                p_aug=args.p_aug, seed=args.seed, epoch=epoch, return_stats=True,
            )
            aug_chunks = build_chunk_samples(aug_entries, **ck_params)
            trainer_obj.train_loader = build_dataloader(
                aug_chunks, max_tokens=train_cfg["max_tokens"],
                shuffle=True, seed=args.seed + epoch,
                num_workers=train_cfg["num_workers"],
                tokenize_fn=tokenizer,
            )
            log_stats = {}
            for k, v in aug_stats.items():
                if k == "effective_aug_fraction":
                    log_stats["aug_effective_fraction"] = v
                else:
                    log_stats[f"aug_{k}"] = v
            return log_stats

        epoch_hook = _aug_epoch_hook

    # ── Train ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg_for_trainer,
        run_dir=run_dir,
        device=device,
        registry_path=registry_path,
        wandb_cfg=wandb_cfg,
        val_entries=val_entries,
        tokenizer=tokenizer,
        min_k=data_cfg["min_k"],
        max_k=data_cfg["max_k"],
        context_len=eval_context_len,
        epoch_hook=epoch_hook,
    )

    logger.info("=" * 60)
    logger.info("Starting ablation: variant=%s, seed=%d, epochs=%d, device=%s",
                variant_id, args.seed, train_cfg_for_trainer["max_epochs"], device)
    logger.info("=" * 60)

    summary = trainer.fit()

    logger.info("=" * 60)
    logger.info("Training complete. Summary:")
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    main()
