"""Typed loader + fail-fast validation + canonical hashing for the Fusion config (§2).

One top-level ``fusion:`` block. Calibration-gate values (§2.1) carry NO silent numeric
default — a missing one is rejected, never invented. Only the frozen v0 modes are accepted.
Every resolved field is serializable and canonically hashable for the run manifest.

Pure: stdlib + yaml only (no torch).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


def _finite_nonneg(value: float) -> bool:
    return math.isfinite(value) and value >= 0.0


def _finite_pos(value: float) -> bool:
    return math.isfinite(value) and value > 0.0

_SELECTION_MODES = frozenset({"greedy", "beam", "fk"})
_ACTIVE_SITE_METRICS = frozenset({"none", "legacy_ca_shell", "sidechain_max_anchor"})
_STRUCTURE_BACKENDS = frozenset({"esmfold", "esmfold2_live"})


class FusionConfigError(ValueError):
    """Raised when a Fusion YAML violates the §2 contract."""


# --------------------------------------------------------------------------- #
# extraction helpers — a missing calibration-gate value is an error, not a default
# --------------------------------------------------------------------------- #
def _require(node: dict, key: str, ctx: str) -> Any:
    if not isinstance(node, dict) or key not in node or node[key] is None:
        raise FusionConfigError(f"missing required field: {ctx}.{key}")
    return node[key]


def _req_float(node: dict, key: str, ctx: str) -> float:
    return float(_require(node, key, ctx))


def _req_int(node: dict, key: str, ctx: str) -> int:
    return int(_require(node, key, ctx))


def _req_str(node: dict, key: str, ctx: str) -> str:
    return str(_require(node, key, ctx))


def _as_bool(node: dict, key: str, default: bool, ctx: str) -> bool:
    """Strict bool: accept a real YAML bool or the exact strings true/false (case-insensitive).
    Plain ``bool("false")`` is truthy, so a quoted ``"false"`` would silently enable a gate —
    fail-fast instead of coercing an ambiguous value."""
    val = node.get(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str) and val.strip().lower() in ("true", "false"):
        return val.strip().lower() == "true"
    raise FusionConfigError(f"{ctx}.{key} must be a boolean (got {val!r})")


def _sub(node: dict, key: str, ctx: str) -> dict:
    val = _require(node, key, ctx)
    if not isinstance(val, dict):
        raise FusionConfigError(f"{ctx}.{key} must be a mapping")
    return val


# --------------------------------------------------------------------------- #
# section dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CoreConfig:
    enabled: bool
    seed: int
    population_size: int
    n_rounds: int


@dataclass(frozen=True)
class HandoffConfig:
    mode: str


@dataclass(frozen=True)
class TargetingConfig:
    mode: str
    registers_per_parent: int
    halo_radius: int


@dataclass(frozen=True)
class ObjectiveConfig:
    head_field: str
    min_head_improvement: float
    max_offtarget_window_increase: float


@dataclass(frozen=True)
class ExplicitMovesConfig:
    enabled: bool
    max_edit_order: int
    max_raw_candidates_per_parent: int
    pair_seed_budget: int


@dataclass(frozen=True)
class RfReopenMovesConfig:
    enabled: bool
    children_per_parent: int


@dataclass(frozen=True)
class RepairMovesConfig:
    enabled: bool
    repair_shortlist_per_parent: int
    children_per_edit: int
    sampler_steps: int
    local_remask: bool


@dataclass(frozen=True)
class MovesConfig:
    explicit: ExplicitMovesConfig
    rf_reopen: RfReopenMovesConfig
    repair: RepairMovesConfig


@dataclass(frozen=True)
class StructureConfig:
    backend: str
    surrogate_mode: str
    scTM_min: float
    active_site_metric: str
    active_site_RMSD_max: float | None
    max_anchor_sidechain_RMSD_max: float | None
    active_site_shell_radius: float
    scRMSD_max: float | None
    cache_dir: str
    max_refolds_per_parent: int


@dataclass(frozen=True)
class SelectionConfig:
    mode: str
    beta: tuple[float, ...]
    resample_method: str


@dataclass(frozen=True)
class TelemetryConfig:
    candidate_detail: bool
    window_detail: bool
    trajectory_detail: bool


@dataclass(frozen=True)
class FusionConfig:
    core: CoreConfig
    handoff: HandoffConfig
    targeting: TargetingConfig
    objective: ObjectiveConfig
    moves: MovesConfig
    structure: StructureConfig
    selection: SelectionConfig
    telemetry: TelemetryConfig

    def to_canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_canonical_dict(), sort_keys=True,
                             separators=(",", ":"), default=list)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# beta schedule
# --------------------------------------------------------------------------- #
def _resolve_beta(selection: dict, n_rounds: int) -> tuple[float, ...]:
    n = n_rounds + 1
    if "beta" in selection and selection["beta"] is not None:
        beta = tuple(float(x) for x in selection["beta"])
        if len(beta) != n:
            raise FusionConfigError(
                f"selection.beta length {len(beta)} != n_rounds+1 ({n})")
        _validate_beta(beta)
        return beta
    beta_start = _req_float(selection, "beta_start", "selection")
    beta_end = _req_float(selection, "beta_end", "selection")
    if n == 1:
        beta = (beta_start,)
    else:
        step = (beta_end - beta_start) / (n - 1)
        beta = tuple(beta_start + step * i for i in range(n))
    _validate_beta(beta)
    return beta


def _validate_beta(beta: tuple[float, ...]) -> None:
    if not all(_finite_nonneg(b) for b in beta):
        raise FusionConfigError("selection.beta values must be finite and >= 0")


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #
def load_fusion_config(source: str | Path | dict) -> FusionConfig:
    """Load + validate a Fusion config from a YAML path or an already-parsed mapping."""
    if isinstance(source, (str, Path)):
        with open(source) as fh:
            raw = yaml.safe_load(fh)
    else:
        raw = source
    if not isinstance(raw, dict) or "fusion" not in raw:
        raise FusionConfigError("config must have a top-level 'fusion:' block")
    f = raw["fusion"]

    population_size = _req_int(f, "population_size", "fusion")
    n_rounds = _req_int(f, "n_rounds", "fusion")
    if population_size <= 0:
        raise FusionConfigError("fusion.population_size must be positive")
    if n_rounds < 0:
        raise FusionConfigError("fusion.n_rounds must be non-negative")
    core = CoreConfig(
        enabled=_as_bool(f, "enabled", True, "fusion"),
        seed=_req_int(f, "seed", "fusion"),
        population_size=population_size,
        n_rounds=n_rounds,
    )

    handoff = HandoffConfig(mode=_req_str(_sub(f, "handoff", "fusion"), "mode", "handoff"))
    if handoff.mode != "terminal_population":
        raise FusionConfigError(
            f"handoff.mode {handoff.mode!r} unsupported in v0 (only terminal_population)")

    tgt = _sub(f, "targeting", "fusion")
    targeting = TargetingConfig(
        mode=_req_str(tgt, "mode", "targeting"),
        registers_per_parent=int(tgt.get("registers_per_parent", 1)),
        halo_radius=_req_int(tgt, "halo_radius", "targeting"),
    )
    if targeting.mode != "head_max_window":
        raise FusionConfigError(
            f"targeting.mode {targeting.mode!r} unsupported in v0 (only head_max_window)")
    if targeting.registers_per_parent != 1:
        raise FusionConfigError("targeting.registers_per_parent must be 1 in v0")
    if targeting.halo_radius < 0:
        raise FusionConfigError("targeting.halo_radius must be >= 0")

    obj = _sub(f, "objective", "fusion")
    objective = ObjectiveConfig(
        head_field=_req_str(obj, "head_field", "objective"),
        min_head_improvement=_req_float(obj, "min_head_improvement", "objective"),
        max_offtarget_window_increase=_req_float(obj, "max_offtarget_window_increase", "objective"),
    )
    if objective.head_field != "global_risk":
        raise FusionConfigError("objective.head_field must be global_risk in v0")
    if not _finite_nonneg(objective.min_head_improvement):
        raise FusionConfigError("objective.min_head_improvement must be finite and >= 0")
    if not _finite_nonneg(objective.max_offtarget_window_increase):
        raise FusionConfigError("objective.max_offtarget_window_increase must be finite and >= 0")

    mv = _sub(f, "moves", "fusion")
    ex = _sub(mv, "explicit", "moves")
    explicit = ExplicitMovesConfig(
        enabled=_as_bool(ex, "enabled", True, "moves.explicit"),
        max_edit_order=int(ex.get("max_edit_order", 2)),
        max_raw_candidates_per_parent=_req_int(ex, "max_raw_candidates_per_parent", "moves.explicit"),
        pair_seed_budget=_req_int(ex, "pair_seed_budget", "moves.explicit"),
    )
    if explicit.max_edit_order not in (1, 2):
        raise FusionConfigError("moves.explicit.max_edit_order must be 1 or 2")
    if explicit.max_raw_candidates_per_parent <= 0 or explicit.pair_seed_budget < 0:
        raise FusionConfigError("explicit budgets must be positive / non-negative")

    ro = _sub(mv, "rf_reopen", "moves")
    rf_reopen = RfReopenMovesConfig(
        enabled=_as_bool(ro, "enabled", False, "moves.rf_reopen"),
        children_per_parent=int(ro.get("children_per_parent", 0)),
    )
    rp = _sub(mv, "repair", "moves")
    repair = RepairMovesConfig(
        enabled=_as_bool(rp, "enabled", False, "moves.repair"),
        repair_shortlist_per_parent=int(rp.get("repair_shortlist_per_parent", 0)),
        children_per_edit=int(rp.get("children_per_edit", 0)),
        sampler_steps=int(rp.get("sampler_steps", 0)),
        local_remask=_as_bool(rp, "local_remask", False, "moves.repair"),
    )
    if not (explicit.enabled or rf_reopen.enabled or repair.enabled):
        raise FusionConfigError("at least one move family must be enabled")
    for name, val in (("repair.repair_shortlist_per_parent", repair.repair_shortlist_per_parent),
                      ("repair.children_per_edit", repair.children_per_edit),
                      ("repair.sampler_steps", repair.sampler_steps),
                      ("rf_reopen.children_per_parent", rf_reopen.children_per_parent)):
        if val < 0:
            raise FusionConfigError(f"moves.{name} must be non-negative")
    # an ENABLED move family must carry a usable budget (else the H3/reopen arm silently degenerates)
    if rf_reopen.enabled and rf_reopen.children_per_parent < 1:
        raise FusionConfigError("moves.rf_reopen.enabled requires children_per_parent >= 1")
    # rf_reopen reuses the same DPLM repair_fn as edit-repair, so the shared sampler kernel
    # (moves.repair.sampler_steps) must be usable even when edit-repair is disabled.
    if rf_reopen.enabled and repair.sampler_steps < 1:
        raise FusionConfigError(
            "moves.rf_reopen.enabled requires moves.repair.sampler_steps >= 1 (shared sampler kernel)")
    if repair.enabled and not (repair.repair_shortlist_per_parent >= 1
                               and repair.children_per_edit >= 1 and repair.sampler_steps >= 1):
        raise FusionConfigError(
            "moves.repair.enabled requires repair_shortlist_per_parent / children_per_edit / "
            "sampler_steps all >= 1")
    moves = MovesConfig(explicit=explicit, rf_reopen=rf_reopen, repair=repair)

    st = _sub(f, "structure", "fusion")
    surrogate_mode = _req_str(st, "surrogate_mode", "structure")
    if surrogate_mode != "none":
        raise FusionConfigError(
            f"structure.surrogate_mode {surrogate_mode!r} unimplemented in v0 (only none)")
    scTM_min = _req_float(st, "scTM_min", "structure")
    if not (math.isfinite(scTM_min) and 0.0 < scTM_min <= 1.0):
        raise FusionConfigError("structure.scTM_min must be finite in (0, 1]")
    as_max = st.get("active_site_RMSD_max")
    if as_max is not None:
        as_max = float(as_max)
        if not _finite_pos(as_max):  # NaN/inf ceiling would silently disable the shell gate
            raise FusionConfigError("structure.active_site_RMSD_max must be finite and positive")
    active_site_metric = str(st.get("active_site_metric", "legacy_ca_shell"))
    if active_site_metric not in _ACTIVE_SITE_METRICS:
        raise FusionConfigError(
            "structure.active_site_metric must be one of "
            f"{sorted(_ACTIVE_SITE_METRICS)}"
        )
    sidechain_max = st.get("max_anchor_sidechain_RMSD_max")
    if sidechain_max is not None:
        sidechain_max = float(sidechain_max)
        if not _finite_pos(sidechain_max):
            raise FusionConfigError(
                "structure.max_anchor_sidechain_RMSD_max must be finite and positive"
            )
    if active_site_metric == "sidechain_max_anchor" and sidechain_max is None:
        raise FusionConfigError(
            "structure.active_site_metric=sidechain_max_anchor requires "
            "structure.max_anchor_sidechain_RMSD_max"
        )
    scrmsd_max = st.get("scRMSD_max")
    if scrmsd_max is not None:
        scrmsd_max = float(scrmsd_max)
        if not _finite_pos(scrmsd_max):
            raise FusionConfigError("structure.scRMSD_max must be finite and positive")
    shell_radius = float(st.get("active_site_shell_radius", 6.0))
    if not _finite_pos(shell_radius):
        raise FusionConfigError("structure.active_site_shell_radius must be finite and positive")
    backend = _req_str(st, "backend", "structure")
    if backend not in _STRUCTURE_BACKENDS:
        raise FusionConfigError(
            f"structure.backend must be one of {sorted(_STRUCTURE_BACKENDS)}"
        )
    structure = StructureConfig(
        backend=backend,
        surrogate_mode=surrogate_mode,
        scTM_min=scTM_min,
        active_site_metric=active_site_metric,
        active_site_RMSD_max=as_max,
        max_anchor_sidechain_RMSD_max=sidechain_max,
        active_site_shell_radius=shell_radius,
        scRMSD_max=scrmsd_max,
        cache_dir=_req_str(st, "cache_dir", "structure"),
        max_refolds_per_parent=_req_int(st, "max_refolds_per_parent", "structure"),
    )
    if structure.max_refolds_per_parent <= 0:
        raise FusionConfigError("structure.max_refolds_per_parent must be positive")

    sel = _sub(f, "selection", "fusion")
    mode = _req_str(sel, "mode", "selection")
    if mode not in _SELECTION_MODES:
        raise FusionConfigError(f"selection.mode must be one of {sorted(_SELECTION_MODES)}")
    resample_method = str(sel.get("resample_method", "multinomial"))
    if resample_method != "multinomial":
        raise FusionConfigError("selection.resample_method must be multinomial in v0")
    selection = SelectionConfig(
        mode=mode,
        beta=_resolve_beta(sel, core.n_rounds),
        resample_method=resample_method,
    )

    tel = f.get("telemetry", {}) or {}
    telemetry = TelemetryConfig(
        candidate_detail=_as_bool(tel, "candidate_detail", True, "telemetry"),
        window_detail=_as_bool(tel, "window_detail", False, "telemetry"),
        trajectory_detail=_as_bool(tel, "trajectory_detail", True, "telemetry"),
    )

    return FusionConfig(
        core=core, handoff=handoff, targeting=targeting, objective=objective,
        moves=moves, structure=structure, selection=selection, telemetry=telemetry,
    )


def fusion_config_to_dict(config: FusionConfig) -> dict[str, Any]:
    return config.to_canonical_dict()


# --- Public aliases for fusion_v2 reuse (additive; zero behavior change) ---------------------
# require_explicit_bool already closes the quoted-"false" truthiness trap; re-deriving these in a
# V2 loader would re-open a fixed bug. The V2 config shares the helpers, never the config types.
require_key = _require
require_int = _req_int
require_float = _req_float
require_str = _req_str
require_explicit_bool = _as_bool
require_submapping = _sub
require_finite_nonneg = _finite_nonneg
require_finite_pos = _finite_pos
