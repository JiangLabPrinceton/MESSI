"""YAML schema and validation for Phase C1 reference-flow runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


class ReferenceFlowConfigError(ValueError):
    """Raised when a reference-flow YAML violates the frozen contract."""


@dataclass(frozen=True)
class RemaskConfig:
    """Optional inference-time refinement on committed positions.

    When ``enabled``, after each unmask step (except the final step) the
    sampler re-masks the lowest-score committed positions following DPLM's
    ``reparam-uncond-deterministic-linear`` rule. The cutoff fraction is
    ``1 - (step + 1) / n_steps``. This is *not* part of the position-dependent
    DFM derivation: it is an inference-only refinement layered on top of the
    position-dependent unmasking schedule (analogous to MaskGIT iterative
    decoding). The position-dependent schedule still drives *first*-unmask
    timing and remains the only place the h-map signal is injected.

    ``fraction_scale`` multiplies the per-step remask cutoff ``1 - (step+1)/T``.
    At ``1.0`` (default) the remask is byte-identical to the legacy rule; at
    ``0.0`` no positions are remasked while ``enabled`` stays ``True`` (so the
    controller ``post_step`` / telemetry path still runs). This isolates the
    remask attractor cleanly: ``enabled=False`` would also skip ``post_step``,
    conflating "no remask" with "no controller post-step lifecycle".
    """

    enabled: bool = False
    fraction_scale: float = 1.0


@dataclass(frozen=True)
class SamplerConfig:
    n_steps: int
    seed: int
    temperature: float = 1.0
    n_designs_per_protein: int = 1
    remask: RemaskConfig = RemaskConfig()


@dataclass(frozen=True)
class ScheduleConfig:
    base_form: str


@dataclass(frozen=True)
class AmplificationConfig:
    form: str
    h_source: str
    c: float | None = None
    mu: float = 0.0
    kappa: float | None = None
    p: float | None = None
    g_max_cap: float = 20.0


@dataclass(frozen=True)
class HShuffleConfig:
    enabled: bool = False
    seed: int | None = None


@dataclass(frozen=True)
class ReferenceFlowConfig:
    sampler: SamplerConfig
    schedule: ScheduleConfig
    amplification: AmplificationConfig
    h_shuffle: HShuffleConfig


def load_reference_flow_config(path: str | Path) -> ReferenceFlowConfig:
    """Load and validate a reference-flow YAML file."""
    config_path = Path(path)
    with open(config_path) as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ReferenceFlowConfigError("top-level YAML payload must be a mapping")
    return materialize_reference_flow_config(payload)


def materialize_reference_flow_config(payload: dict[str, Any]) -> ReferenceFlowConfig:
    sampler = payload.get("sampler")
    schedule = payload.get("schedule")
    amplification = payload.get("amplification")
    h_shuffle = payload.get("h_shuffle", {})

    if not isinstance(sampler, dict):
        raise ReferenceFlowConfigError("sampler section is required")
    if not isinstance(schedule, dict):
        raise ReferenceFlowConfigError("schedule section is required")
    if not isinstance(amplification, dict):
        raise ReferenceFlowConfigError("amplification section is required")
    if not isinstance(h_shuffle, dict):
        raise ReferenceFlowConfigError("h_shuffle section must be a mapping")

    remask_payload = sampler.get("remask", {})
    if not isinstance(remask_payload, dict):
        raise ReferenceFlowConfigError("sampler.remask must be a mapping")
    remask_cfg = RemaskConfig(
        enabled=bool(remask_payload.get("enabled", False)),
        fraction_scale=float(remask_payload.get("fraction_scale", 1.0)),
    )

    sampler_cfg = SamplerConfig(
        n_steps=_require_int(sampler, "n_steps"),
        seed=_require_int(sampler, "seed"),
        temperature=float(sampler.get("temperature", 1.0)),
        n_designs_per_protein=int(sampler.get("n_designs_per_protein", 1)),
        remask=remask_cfg,
    )
    if sampler_cfg.n_steps <= 0:
        raise ReferenceFlowConfigError("sampler.n_steps must be positive")
    if sampler_cfg.temperature <= 0.0:
        raise ReferenceFlowConfigError("sampler.temperature must be positive")
    if sampler_cfg.n_designs_per_protein <= 0:
        raise ReferenceFlowConfigError("sampler.n_designs_per_protein must be positive")
    if not 0.0 <= sampler_cfg.remask.fraction_scale <= 1.0:
        raise ReferenceFlowConfigError(
            "sampler.remask.fraction_scale must be in [0, 1]"
        )

    base_form = str(schedule.get("base_form"))
    if base_form not in {"linear", "cosine", "cubic"}:
        raise ReferenceFlowConfigError("schedule.base_form must be linear|cosine|cubic")
    schedule_cfg = ScheduleConfig(base_form=base_form)

    form = str(amplification.get("form"))
    h_source = str(amplification.get("h_source"))
    if form not in {"constant_one", "linear_clamp", "sigmoid", "power"}:
        raise ReferenceFlowConfigError(
            "amplification.form must be constant_one|linear_clamp|sigmoid|power"
        )
    if h_source not in {"h_raw", "h_processed", "h_normalized_corpus"}:
        raise ReferenceFlowConfigError(
            "amplification.h_source must be h_raw|h_processed|h_normalized_corpus"
        )

    c = amplification.get("c")
    if form != "constant_one" and c is None:
        raise ReferenceFlowConfigError("amplification.c is required for non-constant forms")
    kappa = amplification.get("kappa")
    if form == "sigmoid" and kappa is None:
        raise ReferenceFlowConfigError("amplification.kappa is required for sigmoid form")
    p = amplification.get("p")
    if form == "power" and p is None:
        raise ReferenceFlowConfigError("amplification.p is required for power form")

    amplification_cfg = AmplificationConfig(
        form=form,
        h_source=h_source,
        c=float(c) if c is not None else None,
        mu=float(amplification.get("mu", 0.0)),
        kappa=float(kappa) if kappa is not None else None,
        p=float(p) if p is not None else None,
        g_max_cap=float(amplification.get("g_max_cap", 20.0)),
    )
    if amplification_cfg.g_max_cap < 1.0:
        raise ReferenceFlowConfigError("amplification.g_max_cap must be >= 1")

    enabled = bool(h_shuffle.get("enabled", False))
    shuffle_seed = h_shuffle.get("seed")
    if enabled and shuffle_seed is None:
        raise ReferenceFlowConfigError("h_shuffle.seed is required when h_shuffle.enabled=true")
    h_shuffle_cfg = HShuffleConfig(
        enabled=enabled,
        seed=int(shuffle_seed) if shuffle_seed is not None else None,
    )

    return ReferenceFlowConfig(
        sampler=sampler_cfg,
        schedule=schedule_cfg,
        amplification=amplification_cfg,
        h_shuffle=h_shuffle_cfg,
    )


def with_reference_flow_overrides(
    config: ReferenceFlowConfig,
    *,
    n_steps: int | None = None,
    seed: int | None = None,
    n_designs_per_protein: int | None = None,
) -> ReferenceFlowConfig:
    """Return a config with CLI overrides applied to the sampler block."""
    sampler = replace(
        config.sampler,
        n_steps=config.sampler.n_steps if n_steps is None else int(n_steps),
        seed=config.sampler.seed if seed is None else int(seed),
        n_designs_per_protein=(
            config.sampler.n_designs_per_protein
            if n_designs_per_protein is None
            else int(n_designs_per_protein)
        ),
    )
    return replace(config, sampler=sampler)


def reference_flow_config_to_dict(config: ReferenceFlowConfig) -> dict[str, Any]:
    """Serialize a dataclass config into a plain nested mapping."""
    return asdict(config)


def _require_int(mapping: dict[str, Any], key: str) -> int:
    if key not in mapping:
        raise ReferenceFlowConfigError(f"missing required field: {key}")
    return int(mapping[key])
