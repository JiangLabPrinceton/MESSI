"""Baseline config schema validation for Inverse Folding v1.

Enforces the frozen contract from PLAN_IF.md Task K0:
  - Required keys must be present and non-null
  - Frozen values (adapter-only training, checkpoint, seed) must match contract
  - Artifact root must be non-empty
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class BaselineConfigError(Exception):
    """Raised when a baseline config violates the frozen contract."""


# ── Frozen contract values (PLAN_IF.md §K0) ─────────────────────────────────

_FROZEN_VALUES = {
    "checkpoint_id": "airkingbd/dplm_650m",
    "trainable_params_pattern": "adapter",
    "gvp_frozen": True,
    "backbone_frozen": True,
    "seed": 42,
}

_REQUIRED_KEYS = [
    "checkpoint_id",
    "trainable_params_pattern",
    "dataset_root",
    "dplm_commit",
    "seed",
    "artifact_root",
]

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "dplm_v1.yaml"


# ── Public API ───────────────────────────────────────────────────────────────

def validate_baseline_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a baseline config dict against the frozen K0 contract.

    Returns the config unchanged if valid; raises BaselineConfigError otherwise.
    """
    if not cfg or "baseline" not in cfg:
        raise BaselineConfigError(
            "Config must contain a 'baseline' section"
        )

    b = cfg["baseline"]

    # Check required keys exist and are non-null
    for key in _REQUIRED_KEYS:
        if key not in b:
            raise BaselineConfigError(
                f"Missing required key: '{key}'"
            )
        if b[key] is None:
            raise BaselineConfigError(
                f"Required key '{key}' must not be null"
            )

    # Check frozen values match contract
    for key, expected in _FROZEN_VALUES.items():
        if key in b and b[key] != expected:
            raise BaselineConfigError(
                f"Frozen value mismatch for '{key}': "
                f"expected {expected!r}, got {b[key]!r}"
            )

    # Check artifact_root is non-empty
    if not b.get("artifact_root"):
        raise BaselineConfigError(
            "Field 'artifact_root' must be a non-empty string"
        )

    return cfg


def load_baseline_config(
    override_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the default baseline config, optionally merged with overrides.

    The default config is ``inverse_folding/configs/dplm_v1.yaml``.
    Override files are shallow-merged into the ``baseline`` section.
    """
    with open(_DEFAULT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    if override_path is not None:
        with open(override_path) as f:
            overrides = yaml.safe_load(f)
        if overrides and "baseline" in overrides:
            cfg["baseline"].update(overrides["baseline"])

    return cfg
