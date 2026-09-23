"""Config loader for Epitope Head."""

from pathlib import Path
import yaml

_CONFIG_DIR = Path(__file__).parent


# ── HIMP v0 frozen option sets (PLAN_EPI_IMP.md §5 HIMP0) ────────────────────

NEAR_POSITIVE_SCHEDULES_V0 = frozenset({"ignore", "linear_clamp", "sigmoid"})
NEAR_POSITIVE_METRICS_V0 = frozenset({"endpoint_gap"})

RESIDUE_AGGREGATIONS_V0 = frozenset({"max", "log_mean_exp"})
RESIDUE_LOSS_MODES_V0 = frozenset({"pairwise_margin"})
RESIDUE_LABEL_MODES_V0 = frozenset({"binary_coverage"})
RESIDUE_WINDOW_MODES_V0 = frozenset({"sampled", "all"})

NEAR_POSITIVE_DEFAULTS = {
    "enabled": False,
    "near_gap_max": 10,
    "schedule": "ignore",
    "schedule_params": {},
    "metric": "endpoint_gap",
    "apply_to_span_negatives": True,
    "apply_to_residue_labels": True,
}

RESIDUE_DEFAULTS = {
    "enabled": False,
    "lambda_residue": 0.0,
    "label_mode": "binary_coverage",
    "aggregation": "log_mean_exp",
    "aggregation_params": {},
    "loss_mode": "pairwise_margin",
    "margin_m_residue": 0.5,
    "window_mode": "sampled",
    "max_windows_per_chunk": 1024,
    "min_far_bg_residues": 4,
}


def _validate_and_default_near_positive(cfg: dict | None) -> dict:
    """Validate and default-fill the train.near_positive sub-block.

    Returns a dict containing all NEAR_POSITIVE_DEFAULTS keys with user
    overrides applied. Raises ValueError on unknown keys, unsupported
    schedule / metric names, or out-of-range values.
    """
    if cfg is None:
        return dict(NEAR_POSITIVE_DEFAULTS)

    if not isinstance(cfg, dict):
        raise ValueError("train.near_positive must be a dict")

    unknown = set(cfg.keys()) - set(NEAR_POSITIVE_DEFAULTS.keys())
    if unknown:
        raise ValueError(
            f"train.near_positive has unknown keys: {sorted(unknown)}"
        )

    out = dict(NEAR_POSITIVE_DEFAULTS)
    out.update(cfg)

    if not isinstance(out["enabled"], bool):
        raise ValueError("train.near_positive.enabled must be a bool")

    near_gap_max = out["near_gap_max"]
    if not isinstance(near_gap_max, int) or isinstance(near_gap_max, bool) or near_gap_max < 0:
        raise ValueError(
            f"train.near_positive.near_gap_max must be a non-negative int, got {near_gap_max!r}"
        )

    if out["schedule"] not in NEAR_POSITIVE_SCHEDULES_V0:
        raise ValueError(
            f"train.near_positive.schedule must be one of "
            f"{sorted(NEAR_POSITIVE_SCHEDULES_V0)}, got {out['schedule']!r}"
        )

    if not isinstance(out["schedule_params"], dict):
        raise ValueError(
            f"train.near_positive.schedule_params must be a dict, "
            f"got {type(out['schedule_params']).__name__}"
        )

    if out["metric"] not in NEAR_POSITIVE_METRICS_V0:
        raise ValueError(
            f"train.near_positive.metric must be one of "
            f"{sorted(NEAR_POSITIVE_METRICS_V0)}, got {out['metric']!r}"
        )

    if not isinstance(out["apply_to_span_negatives"], bool):
        raise ValueError("train.near_positive.apply_to_span_negatives must be a bool")
    if not isinstance(out["apply_to_residue_labels"], bool):
        raise ValueError("train.near_positive.apply_to_residue_labels must be a bool")

    return out


def _validate_and_default_residue(cfg: dict | None) -> dict:
    """Validate and default-fill the train.residue sub-block.

    Returns a dict containing all RESIDUE_DEFAULTS keys with user overrides
    applied. Raises ValueError on unknown keys, unsupported names, or
    out-of-range values.
    """
    if cfg is None:
        return dict(RESIDUE_DEFAULTS)

    if not isinstance(cfg, dict):
        raise ValueError("train.residue must be a dict")

    unknown = set(cfg.keys()) - set(RESIDUE_DEFAULTS.keys())
    if unknown:
        raise ValueError(
            f"train.residue has unknown keys: {sorted(unknown)}"
        )

    out = dict(RESIDUE_DEFAULTS)
    out.update(cfg)

    if not isinstance(out["enabled"], bool):
        raise ValueError("train.residue.enabled must be a bool")

    lam = out["lambda_residue"]
    if not isinstance(lam, (int, float)) or isinstance(lam, bool) or lam < 0:
        raise ValueError(
            f"train.residue.lambda_residue must be a non-negative number, got {lam!r}"
        )

    if out["label_mode"] not in RESIDUE_LABEL_MODES_V0:
        raise ValueError(
            f"train.residue.label_mode must be one of "
            f"{sorted(RESIDUE_LABEL_MODES_V0)}, got {out['label_mode']!r}"
        )

    if out["aggregation"] not in RESIDUE_AGGREGATIONS_V0:
        raise ValueError(
            f"train.residue.aggregation must be one of "
            f"{sorted(RESIDUE_AGGREGATIONS_V0)}, got {out['aggregation']!r}"
        )

    if not isinstance(out["aggregation_params"], dict):
        raise ValueError(
            f"train.residue.aggregation_params must be a dict, "
            f"got {type(out['aggregation_params']).__name__}"
        )

    if out["loss_mode"] not in RESIDUE_LOSS_MODES_V0:
        raise ValueError(
            f"train.residue.loss_mode must be one of "
            f"{sorted(RESIDUE_LOSS_MODES_V0)}, got {out['loss_mode']!r}"
        )

    margin_m = out["margin_m_residue"]
    if not isinstance(margin_m, (int, float)) or isinstance(margin_m, bool) or margin_m < 0:
        raise ValueError(
            f"train.residue.margin_m_residue must be a non-negative number, got {margin_m!r}"
        )

    if out["window_mode"] not in RESIDUE_WINDOW_MODES_V0:
        raise ValueError(
            f"train.residue.window_mode must be one of "
            f"{sorted(RESIDUE_WINDOW_MODES_V0)}, got {out['window_mode']!r}"
        )

    mwpc = out["max_windows_per_chunk"]
    if not isinstance(mwpc, int) or isinstance(mwpc, bool) or mwpc <= 0:
        raise ValueError(
            f"train.residue.max_windows_per_chunk must be a positive int, got {mwpc!r}"
        )

    mfbr = out["min_far_bg_residues"]
    if not isinstance(mfbr, int) or isinstance(mfbr, bool) or mfbr < 0:
        raise ValueError(
            f"train.residue.min_far_bg_residues must be a non-negative int, got {mfbr!r}"
        )

    return out


def validate_himp_train_blocks(train_cfg: dict) -> dict:
    """Re-run HIMP block validation on a (potentially override-merged) cfg.

    ``load_train_config`` validates the *base* yaml. When the launcher applies
    ``--override-config`` via raw deep-merge (e.g. ``train_v2_ablation.py``),
    user overrides bypass that gate, so unknown schedule names, wrong types,
    or unknown keys can slip through. This function re-validates the
    ``near_positive`` and ``residue`` blocks in-place and writes back the
    defaulted dicts. It is idempotent — calling it on an already-validated
    cfg is a no-op (defaults match the input).

    Returns the same ``train_cfg`` for chaining.
    """
    if not isinstance(train_cfg, dict):
        raise ValueError("train_cfg must be a dict")
    train_cfg["near_positive"] = _validate_and_default_near_positive(
        train_cfg.get("near_positive")
    )
    train_cfg["residue"] = _validate_and_default_residue(train_cfg.get("residue"))
    return train_cfg


def load_data_config(path: Path | str | None = None) -> dict:
    """Load and validate data config. Raises on missing required keys."""
    if path is None:
        path = _CONFIG_DIR / "data.yaml"
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    data = cfg.get("data")
    if data is None:
        raise ValueError("Config missing top-level 'data' key")

    required = ["target_allele", "min_k", "max_k", "coord_mode"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    if not isinstance(data["target_allele"], str) or len(data["target_allele"]) == 0:
        raise ValueError("target_allele must be a non-empty string")

    return data


def load_model_config(path: Path | str | None = None) -> dict:
    """Load and validate model config."""
    if path is None:
        path = _CONFIG_DIR / "model.yaml"
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    model = cfg.get("model")
    if model is None:
        raise ValueError("Config missing top-level 'model' key")

    required = ["encoder_name", "freeze_encoder", "d_proj",
                 "scorer_hidden_dim", "scorer_activation"]
    missing = [k for k in required if k not in model]
    if missing:
        raise ValueError(f"Model config missing required keys: {missing}")

    return model


def load_train_config(path: Path | str | None = None) -> dict:
    """Load and validate training config.

    Validates all E0-frozen keys including nested loss.* and chunking.* sub-configs.
    Raises ValueError on any missing or mistyped key.
    """
    if path is None:
        path = _CONFIG_DIR / "train.yaml"
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    train = cfg.get("train")
    if train is None:
        raise ValueError("Config missing top-level 'train' key")

    # ── Top-level required keys (E0 frozen) ──
    required_top = [
        # Negative sampling
        "neg_ratio", "hard_negative_fraction", "hard_neg_max_overlap_ratio",
        "hard_neg_offset_range", "neg_length_sampling",
        # Loss (sub-dict validated below)
        "loss",
        # Optimization
        "lr", "weight_decay", "optimizer", "scheduler",
        "warmup_steps", "max_epochs", "grad_clip",
        # Batching
        "max_tokens", "num_workers",
        # Reproducibility
        "seed", "deterministic",
        # Checkpointing
        "checkpoint_every_n_epochs", "early_stopping_patience", "monitor_metric",
        # Chunking (sub-dict validated below)
        "chunking",
    ]
    missing = [k for k in required_top if k not in train]
    if missing:
        raise ValueError(f"Train config missing required keys: {missing}")

    # ── loss.* sub-keys ──
    loss = train["loss"]
    if not isinstance(loss, dict):
        raise ValueError("train.loss must be a dict")
    required_loss = ["tau", "T_mp", "lambda_mp", "lambda_smooth"]
    missing_loss = [k for k in required_loss if k not in loss]
    if missing_loss:
        raise ValueError(f"train.loss missing required keys: {missing_loss}")

    # ── chunking.* sub-keys ──
    chunking = train["chunking"]
    if not isinstance(chunking, dict):
        raise ValueError("train.chunking must be a dict")
    required_chunking = [
        "enabled", "context_len", "stride", "margin",
        "stitch_mode", "window_mode", "enable_reliability",
    ]
    missing_chunking = [k for k in required_chunking if k not in chunking]
    if missing_chunking:
        raise ValueError(f"train.chunking missing required keys: {missing_chunking}")

    # ── HIMP optional sub-blocks (default: disabled, backward-compatible) ──
    train["near_positive"] = _validate_and_default_near_positive(
        train.get("near_positive")
    )
    train["residue"] = _validate_and_default_residue(train.get("residue"))

    return train


def load_ablation_config(path: Path | str | None = None) -> dict:
    """Load and validate encoder ablation config (Module H contract).

    Validates that each profile has required encoder keys and that
    frozen constants match expected values.
    """
    if path is None:
        path = _CONFIG_DIR / "model_ablation.yaml"
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    ablation = cfg.get("ablation")
    if ablation is None:
        raise ValueError("Config missing top-level 'ablation' key")

    profiles = ablation.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("ablation.profiles must be a non-empty dict")

    VALID_ENCODER_TYPES = {"esm2_frozen", "dilated_cnn", "shallow_transformer", "multiscale_cnn"}
    REQUIRED_PROFILE_KEYS = {"encoder_type", "d_enc", "trainable_encoder", "encoder_cfg"}

    ENCODER_CFG_KEYS = {
        "esm2_frozen": {"encoder_name"},
        "dilated_cnn": {"token_emb_dim", "n_blocks", "kernel_size", "dilations",
                        "hidden_channels", "block_dropout"},
        "shallow_transformer": {"d_model", "n_layers", "n_heads", "ffn_dim",
                                "dropout", "max_seq_len"},
        "multiscale_cnn": {"token_emb_dim", "n_blocks", "dilations",
                           "hidden_channels", "branch_channels", "block_dropout"},
    }

    OPTIONAL_PROFILE_KEYS = {"loss_overrides"}

    for profile_id, profile in profiles.items():
        missing = REQUIRED_PROFILE_KEYS - set(profile.keys())
        if missing:
            raise ValueError(
                f"Ablation profile '{profile_id}' missing required keys: {sorted(missing)}"
            )

        enc_type = profile["encoder_type"]
        if enc_type not in VALID_ENCODER_TYPES:
            raise ValueError(
                f"Profile '{profile_id}': encoder_type '{enc_type}' not in {sorted(VALID_ENCODER_TYPES)}"
            )

        enc_cfg = profile["encoder_cfg"]
        if not isinstance(enc_cfg, dict):
            raise ValueError(f"Profile '{profile_id}': encoder_cfg must be a dict")

        expected_keys = ENCODER_CFG_KEYS[enc_type]
        missing_enc = expected_keys - set(enc_cfg.keys())
        if missing_enc:
            raise ValueError(
                f"Profile '{profile_id}': encoder_cfg missing keys: {sorted(missing_enc)}"
            )

    return ablation


def load_inference_config(path: Path | str | None = None) -> dict:
    """Load and validate inference config (Module F0 contract)."""
    if path is None:
        path = _CONFIG_DIR / "inference.yaml"
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    inference = cfg.get("inference")
    if inference is None:
        raise ValueError("Config missing top-level 'inference' key")

    required_top = [
        "checkpoint_path",
        "min_k",
        "max_k",
        "hotspot_center_method",
        "hotspot_clamp",
        "chunking",
        "device",
    ]
    missing = [k for k in required_top if k not in inference]
    if missing:
        raise ValueError(f"Inference config missing required keys: {missing}")

    if not isinstance(inference["min_k"], int) or not isinstance(inference["max_k"], int):
        raise ValueError("inference.min_k and inference.max_k must be integers")
    if inference["min_k"] > inference["max_k"]:
        raise ValueError("inference.min_k must be <= inference.max_k")
    if inference["min_k"] != 12:
        raise ValueError("inference.min_k must be frozen to 12 for v1.1")
    if inference["max_k"] != 25:
        raise ValueError("inference.max_k must be frozen to 25 for v1.1")

    center_allowed = {"median", "mean", "none"}
    if inference["hotspot_center_method"] not in center_allowed:
        raise ValueError(
            f"inference.hotspot_center_method must be one of {sorted(center_allowed)}",
        )

    clamp_allowed = {"softplus", "relu", "none"}
    if inference["hotspot_clamp"] not in clamp_allowed:
        raise ValueError(
            f"inference.hotspot_clamp must be one of {sorted(clamp_allowed)}",
        )

    chunking = inference["chunking"]
    if not isinstance(chunking, dict):
        raise ValueError("inference.chunking must be a dict")

    required_chunking = [
        "enabled",
        "context_len",
        "stride",
        "margin",
        "stitch_mode",
        "enable_reliability",
    ]
    missing_chunking = [k for k in required_chunking if k not in chunking]
    if missing_chunking:
        raise ValueError(f"inference.chunking missing required keys: {missing_chunking}")

    stitch_allowed = {"per_residue_stitch", "center_weighted"}
    if chunking["stitch_mode"] not in stitch_allowed:
        raise ValueError(
            f"inference.chunking.stitch_mode must be one of {sorted(stitch_allowed)}",
        )
    if not bool(chunking["enabled"]):
        raise ValueError("inference.chunking.enabled must be true in v1.1")
    if int(chunking["context_len"]) != 1022:
        raise ValueError("inference.chunking.context_len must be frozen to 1022 for v1.1")
    if int(chunking["stride"]) != 512:
        raise ValueError("inference.chunking.stride must be frozen to 512 for v1.1")
    if int(chunking["margin"]) != 32:
        raise ValueError("inference.chunking.margin must be frozen to 32 for v1.1")
    if chunking["stitch_mode"] != "per_residue_stitch":
        raise ValueError(
            "inference.chunking.stitch_mode must be frozen to per_residue_stitch in v1.1",
        )

    return inference
