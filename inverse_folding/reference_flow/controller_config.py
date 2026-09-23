"""YAML schema and validation for Phase D controller configs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class ControllerConfigError(ValueError):
    """Raised when a controller YAML violates the frozen D-phase contract."""


_ALLOWED_MODES = frozenset({"monitor_only", "d2_logits", "d3_revisit", "d2_d3_full"})
# §A2 D2 candidate-score source (doc §8.4 Path A): ``local`` block-span ensemble
# (default) vs ``terminal`` full-completion P_cheap probe.
_ALLOWED_CANDIDATE_SCORE_SOURCES = frozenset({"local", "terminal"})
_MODES_REQUIRING_D2 = frozenset({"d2_logits", "d2_d3_full"})
_MODES_REQUIRING_D3 = frozenset({"d3_revisit", "d2_d3_full"})

# Stage B typed-targeting allowed values (PLAN_RF_UNI_CTRL.md Task B2).
_ALLOWED_TARGETING_MODES = frozenset({"static_excess", "typed_actionability"})
# Stage B.1 within-block editable-position ranking source (PLAN_RF_UNI_CTRL.md).
# 'e_fresh' is reserved (deferred until the field-floor fix), so it is NOT allowed.
_ALLOWED_WITHIN_BLOCK_SOURCES = frozenset({"legacy_excess", "v_target"})
_ALLOWED_TAU_REF_SOURCES = frozenset({"static_median", "first_reliable_refresh_median"})
_RESERVED_TAU_REF_SOURCES = frozenset({"first_reliable_refresh_median"})
_ALLOWED_D3_EVIDENCE_SOURCES = frozenset({"legacy_window_excess", "typed_fresh"})

# Stage C.1 global-pressure actuator allowed values (PLAN_RF_UNI_CTRL.md C1.2/G3).
# ``trajectory_median_G`` (legacy mean-based driver) is recognized but rejected
# when global_pressure is enabled — C.1 outcome runs must use the thresholded G.
_ALLOWED_PRESSURE_MAPPINGS = frozenset({"smoothstep"})
_ALLOWED_PRESSURE_SOURCES = frozenset(
    {"trajectory_thresholded_G", "trajectory_median_G", "self_conditioned_probe"}
)
_REQUIRED_C1_PRESSURE_SOURCE = "trajectory_thresholded_G"
# Pressure sources allowed when global_pressure is actuating (PLAN_RF_SC_GR.md
# Task SC1.1): the Stage C.1 thresholded-G driver OR the SC-GR probe. The legacy
# mean-based ``trajectory_median_G`` stays rejected when enabled.
_ENABLED_PRESSURE_SOURCES = frozenset(
    {"trajectory_thresholded_G", "self_conditioned_probe"}
)
_ALLOWED_TAU_PROM_SOURCES = frozenset({"calibration_json"})

# Self-Conditioned GR probe allowed values (PLAN_RF_SC_GR.md Task SC0.1 / SC1.1).
# ``beta_pressure`` (SC1) gates D2 beta by ``g_GR=smoothstep(B_sc)``.
_ALLOWED_SCGR_MODES = frozenset({"monitor_only", "beta_pressure"})
_ALLOWED_SCGR_ARMS = frozenset({"fresh", "self_conditioned"})
_ALLOWED_SCGR_CONFIDENCE_SOURCES = frozenset({"canonical_softmax_max"})
_ALLOWED_SCGR_UPDATE_SOURCES = frozenset({"structural_argmax"})
_ALLOWED_SCGR_REFRESH_POLICIES = frozenset({"all_typed_refreshes"})
# SC1.1 actuation knobs. Aggregator names map to the SC0.2 telemetry stems
# (G_mean_excess / G_topm_lse / G_supra_mass_tau_*).
_ALLOWED_SCGR_ACTUATION_AGGREGATORS = frozenset(
    {"mean_excess", "topm_lse", "supra_mass"}
)
_ALLOWED_SCGR_ACTUATION_REDUCE = frozenset({"median", "ema"})

# Allocation layer (PLAN_PLANNER_SC_GR.md §2). ``selection_field_mode`` chooses
# WHICH per-residue field ranks D2 position selection: ``v_target`` (default,
# current typed behavior = the H1 uniform-allocation control), ``v_target_x_alloc``
# (tilt the typed field by the frozen editability mass Phi_i), or ``flat``
# (content-blind pseudo-random control). The allocation ``arm`` reuses the SC-GR
# probe arms (Phi_i is derived from one arm's per-residue residue_excess).
_ALLOWED_SELECTION_FIELD_MODES = frozenset(
    {
        "v_target",
        "v_target_x_alloc",
        "v_target_triage",
        "v_target_terminal_union",
        "flat",
    }
)
# Selection modes that CONSUME the frozen Phi_i (and so require the SC-GR pressure
# freeze path). ``v_target_x_alloc`` (superseded, §8.3), ``v_target_triage``
# (§A1 / doc §8.4 Path A) and ``v_target_terminal_union`` (§A3 / doc §8.4 Fork A)
# all fall back to raw v_target if Phi_i never freezes.
_PHI_CONSUMING_SELECTION_MODES = frozenset(
    {"v_target_x_alloc", "v_target_triage", "v_target_terminal_union"}
)


@dataclass(frozen=True)
class CompletionConfig:
    method: str = "argmax"


@dataclass(frozen=True)
class HeadConfig:
    score_scale: str = "raw_logit"
    static_cache_policy: str = "lazy_write"
    local_risk_aggregation: str = "LME"


@dataclass(frozen=True)
class D2Config:
    enabled: bool = False
    beta: float = 1.0
    eta: float = 0.7
    struct_temperature: float = 1.0
    delta_struct: float = 1.5
    context_pnll_h0: float = 2.0
    context_jsd_h0: float = 0.5
    epsilon: float = 1.0e-8
    candidate_mode: str = "structure_topk"
    top_k_tokens: int = 4
    max_positions_per_block: int = 2
    max_candidates_per_block: int = 32
    selection_score: str = "residue_excess_then_low_entropy"
    min_delta_R_improvement: float = 0.0
    min_ess_fraction: float = 0.25
    max_abs_logit_shift: float = 5.0
    paired_uncorrected_sample: bool = True
    completion_ensemble_enabled: bool = True
    completion_ensemble_size: int = 3
    completion_ensemble_scope: str = "local_windows"
    completion_ensemble_rescore_top_m: int = 8
    completion_ensemble_use_variance_gate: bool = False
    sticky_ttl_steps: int = 5
    sticky_clear_on_selected: bool = True
    sticky_clear_on_remask: bool = True
    sticky_overwrite_on_refresh: bool = True
    # §A2 (doc §8.4 Path A / RAR 0019 P_cheap): how D2 ranks candidate tuples.
    # ``local`` (default, byte-identical) scores the local block-span completion
    # ensemble (ΔR_B on the merged active-block span). ``terminal`` swaps the
    # rescore for a full-sequence terminal-completion probe (P_cheap): each
    # shortlisted candidate is rolled out to ``candidate_terminal_K_P`` full
    # completions (paired seeds across candidates), Ω(B)-LME risk − r_current in
    # the SAME head-logit units as local ΔR_B (so exp(−β·ΔR) stays consistent).
    candidate_score_source: str = "local"
    candidate_terminal_K_P: int = 4


@dataclass(frozen=True)
class D3Config:
    enabled: bool = False
    window_to_residue_projection: str = "max_covering_window"
    gamma_min: float = 0.40
    gamma_max: float = 0.90
    lambda_commit: float = 1.0
    alpha_struct: float = 0.3
    d2_evidence_nu: float = 1.0
    d2_evidence_ttl_steps: int = 5
    d2_evidence_requires_benefit: bool = True
    zscore_epsilon: float = 1.0e-6
    same_refresh_grace: bool = True
    final_freeze_steps: int = 1
    # Stage B (PLAN_RF_UNI_CTRL.md Task B5): which residue-level evidence drives
    # the D3 EMA ``m_i``. ``legacy_window_excess`` keeps the Stage A projection;
    # ``typed_fresh`` feeds the typed ``e_fresh`` field (d2_d3_full_stageB only).
    evidence_source: str = "legacy_window_excess"


@dataclass(frozen=True)
class AttributionConfig:
    write_paired_counterfactual: bool = True
    write_independent_delta_R_i: bool = True
    productive_delta_logp: float = 0.5


@dataclass(frozen=True)
class ControlsConfig:
    allow_wrong_allele_head: bool = True
    allow_shuffled_head: bool = True


@dataclass(frozen=True)
class TargetingConfig:
    """Stage B typed-actionability targeting (PLAN_RF_UNI_CTRL.md Task B2).

    ``mode='static_excess'`` (default) keeps the legacy ``z_dyn - z_static``
    active-window path bit-for-bit. ``mode='typed_actionability'`` builds the
    typed ``A_i(t)`` field and selects active windows from ``v_target``.
    """

    mode: str = "static_excess"
    tau_ref_source: str = "static_median"
    tau_ref_quantile: float = 0.50
    projection: str = "max_covering_window"
    env_seed_top_current: int = 16
    env_seed_top_uncertain: int = 16
    env_seed_max_windows: int = 32
    env_ensemble_size: int = 3
    env_consistency_floor: float = 1.0 / 3.0
    softor_tau: float = 0.5
    r_ctx_floor: float = 0.25
    mem_half_life_refreshes: float = 3.0
    mem_cold_start: str = "zero"
    cluster_radius: int = 4
    cluster_min_mass: float = 3.0
    cluster_floor: float = 0.25
    use_cluster_for_active_blocks: bool = False
    active_window_source: str = "v_target"
    active_window_min_excess: float = 0.0
    write_actionability_telemetry: bool = True
    # Stage B.1 (PLAN_RF_UNI_CTRL.md): within-block editable-position ranking
    # source. 'legacy_excess' (default) = legacy residue_excess, bit-for-bit;
    # 'v_target' ranks within-block positions by the typed field so the typed
    # signal reaches the D2 position layer. 'e_fresh' is reserved (not v1).
    within_block_source: str = "legacy_excess"
    # Allocation layer (PLAN_PLANNER_SC_GR.md §2). Which per-residue field ranks
    # D2 position selection at the two selection seams. 'v_target' (default) is
    # the current typed behavior (= the H1 uniform-allocation control).
    selection_field_mode: str = "v_target"


@dataclass(frozen=True)
class GlobalPressureConfig:
    """Global-pressure actuator config (PLAN_RF_UNI_CTRL.md Task B2 / C1.2).

    Stage B keeps ``enabled=False`` and only logs ``G(t)`` / the diagnostic
    ``g_GR(t)``; it never scales beta/lambda. Stage C.1 sets ``enabled=True`` and
    scales D2 ``beta`` (when ``scale_beta``) and D3 ``lambda`` (when
    ``scale_lambda``) by the trajectory-level smoothstep pressure ``g_GR`` =
    ``smoothstep(B_GR; B_low, B_high, g_min, g_max)`` where ``B_GR`` is the median
    of reliable per-refresh ``G_step`` (PLAN_RF_UNI_CTRL.md Stage C.1).

    ``B_low`` / ``B_high`` come from a calibration JSON stamped at run setup
    (never hardcoded). The primary C.1 YAML sets ``enabled=true`` but omits the
    band; materialization therefore tolerates a missing band, and
    :func:`validate_global_pressure_runtime` fails fast at run setup if the band
    was never stamped. ``g_min`` defaults to ``0.0`` so low-burden trajectories
    reach zero pressure (PLAN G1); the Stage B diagnostic ``0.25`` floor must be
    pinned explicitly in any config that still wants it.
    """

    enabled: bool = False
    g_min: float = 0.0
    g_max: float = 1.0
    # SC2 two-sided gain (PLAN_RF_SC_GR.md Task SC2.1). ``amplify=false`` (default)
    # keeps ``g_max == 1.0`` for a beta_pressure run (protect-low only).
    # ``amplify=true`` unlocks ``g_min < 1.0 < g_max <= 3.0`` so low burden →
    # suppress (g_min) and high burden → amplify (g_max) on the one smoothstep.
    amplify: bool = False
    # Stage C.1 actuator. ``G_step`` is the prominence-thresholded mass
    # ``(1/L) Σ ReLU(u_pressure − τ_prom)`` (G3 / RAR 0006); ``pressure_source`` is
    # its trajectory aggregation. ``trajectory_thresholded_G`` = median over
    # refreshes of that thresholded mass; the legacy ``trajectory_median_G``
    # (median of the un-thresholded mean) is recognized but rejected when enabled.
    pressure_source: str = "trajectory_thresholded_G"
    # τ_prom: the second prominence cut, stamped from the calibration JSON. None
    # in Stage B ⇒ G_step reduces to mean(u_pressure) (byte-identical diagnostic).
    tau_prom_source: str | None = None
    tau_prom: float | None = None
    mapping: str = "smoothstep"
    B_low: float | None = None
    B_high: float | None = None
    min_reliable_refreshes: int = 1
    unready_g: float = 0.0
    scale_beta: bool = True
    scale_lambda: bool = True
    # Stage B diagnostic-sigmoid provenance. The diagnostic ``g_GR`` uses
    # hardcoded ``G0=0`` / ``s_G=1`` anchors, so these strings are inert metadata
    # retained for back-compat; they do not drive the Stage C.1 actuator.
    G0_source: str = "stageB_pilot_median"
    s_G_source: str = "stageB_pilot_iqr_half"


@dataclass(frozen=True)
class SelfConditionedGRConfig:
    """Self-Conditioned GR monitor probe config (PLAN_RF_SC_GR.md Task SC0.1).

    SC-GR is a monitor-only sensing layer: at each refresh it builds fresh and
    self-conditioned pseudo-terminal completions pre-D2, scores them with the
    frozen head, and writes burden-estimator telemetry. It never changes logits,
    D2/D3 state, ``global_pressure``, or the Phase C schedule. ``enabled=False``
    (default) makes the controller skip the probe entirely.

    Arms: ``fresh`` samples every masked position from the structural softmax;
    ``self_conditioned`` reuses the previous refresh's structural-argmax token at
    positions whose previous canonical-softmax confidence cleared
    ``confidence_threshold`` and samples the rest. The window→trajectory
    aggregators (``mean_excess`` / ``top_m`` LSE / supra-threshold mass over
    ``supra_tau_values``) are all logged; SC0 picks none — the analysis CLI
    selects the actuation aggregator later (Task SC0.5).
    """

    enabled: bool = False
    mode: str = "monitor_only"
    arms: tuple[str, ...] = ("fresh", "self_conditioned")
    ensemble_size: int = 3
    struct_temperature: float = 1.0
    confidence_source: str = "canonical_softmax_max"
    confidence_threshold: float = 0.70
    update_state_from: str = "structural_argmax"
    probe_refresh_policy: str = "all_typed_refreshes"
    top_m: int = 16
    lse_temperature: float = 1.0
    supra_tau_values: tuple[float, ...] = (11.75,)
    write_probe_telemetry: bool = True
    # Per-residue ``r_i`` map persistence (SC-GR signal-direction follow-up):
    # when True the probe appends the per-completion ``residue_excess`` array to
    # sc_gr_probe_samples.parquet for sub-protein r_i targeting validation.
    # Default False keeps existing monitor telemetry byte-identical.
    write_residue_telemetry: bool = False
    # SC1.1 actuation knobs (only consumed when ``mode='beta_pressure'``). The
    # aggregator/arm are the SC0.5 monitor decision; ``B_sc`` is the early-frozen
    # per-design burden over the first ``freeze_after_reliable_refreshes`` reliable
    # refreshes, reduced by ``actuation_reduce``. In ``monitor_only`` mode these
    # are inert defaults.
    actuation_aggregator: str = "topm_lse"
    actuation_arm: str = "fresh"
    freeze_after_reliable_refreshes: int = 1
    actuation_reduce: str = "median"


@dataclass(frozen=True)
class AllocationConfig:
    """Allocation layer (PLAN_PLANNER_SC_GR.md §2). Reweights the selection
    field ``v_target`` by a frozen per-residue editability mass ``Phi_i``.

    WHERE-only: it never touches ``v_target`` construction, pressure, memory, or
    the denoising schedule. ``Phi_i`` is distinct from the typed-actionability
    field ``A_i(t)`` — it only *reweights* the field at the selection seams.
    ``enabled=False`` (default) makes the controller skip the freeze entirely;
    the ``selection_field_mode`` on :class:`TargetingConfig` is authoritative for
    whether Phi_i is actually consumed.
    """

    enabled: bool = False
    arm: str = "fresh"
    smooth_half_width: int = 4
    reweight_c: float = 1.0
    reweight_eps: float = 0.05
    # §A1 triage (doc §8.4 Path A): consumed only by selection_field_mode=
    # 'v_target_triage'. ``eligible_quantile`` = the v_target eligibility floor
    # (top-(1−q) by v_target are eligible); ``triage_lambda`` = the bounded Phi_i
    # tie-break weight (v_target keeps coefficient 1 and dominates).
    eligible_quantile: float = 0.5
    triage_lambda: float = 0.3
    # §A3 terminal-aware active-register targeting (doc §8.4 Fork A): consumed only
    # by selection_field_mode='v_target_terminal_union'. ``terminal_eligible_quantile``
    # = the terminal gate (top-(1−q) by the register-smoothed Phi_i are terminal-
    # eligible, unioned with the locally-actionable set); ``terminal_lambda`` = the
    # bounded Phi_i rank weight in the union (v_target keeps coefficient 1).
    terminal_eligible_quantile: float = 0.9
    terminal_lambda: float = 0.3


@dataclass(frozen=True)
class ActiveWindowsConfig:
    excess_threshold: float = 0.0
    max_windows: int = 16
    selection: str = "threshold_then_top_n"
    merge_overlapping_scoring_windows: bool = True


@dataclass(frozen=True)
class ReliabilityConfig:
    time_k: float = 20.0
    entropy_h0: float = 1.5
    min_completion_fraction: float = 0.5
    min_rho_to_emit_event: float = 0.0


@dataclass(frozen=True)
class TelemetryConfig:
    write_refresh_log: bool = True
    write_controller_events: bool = True
    write_per_protein_summary: bool = True


@dataclass(frozen=True)
class ControllerConfig:
    """Controller config for the Phase D adaptive layer.

    When ``enabled=False`` the controller is a no-op and downstream wiring must
    skip predictor construction and telemetry writers. The other fields hold
    safe defaults so a disabled YAML may omit them entirely.
    """

    enabled: bool
    mode: str = "monitor_only"
    t_start: float = 0.5
    refresh_interval: int = 5
    completion: CompletionConfig = CompletionConfig()
    head: HeadConfig = HeadConfig()
    active_windows: ActiveWindowsConfig = ActiveWindowsConfig()
    reliability: ReliabilityConfig = ReliabilityConfig()
    d2: D2Config = D2Config()
    d3: D3Config = D3Config()
    attribution: AttributionConfig = AttributionConfig()
    controls: ControlsConfig = ControlsConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    targeting: TargetingConfig = TargetingConfig()
    global_pressure: GlobalPressureConfig = GlobalPressureConfig()
    self_conditioned_gr: SelfConditionedGRConfig = SelfConditionedGRConfig()
    allocation: AllocationConfig = AllocationConfig()


def load_controller_config(path: str | Path) -> ControllerConfig:
    config_path = Path(path)
    with open(config_path) as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ControllerConfigError("top-level YAML payload must be a mapping")
    return materialize_controller_config(payload)


def materialize_controller_config(payload: dict[str, Any]) -> ControllerConfig:
    controller_payload = payload.get("controller")
    if not isinstance(controller_payload, dict):
        raise ControllerConfigError("controller section is required")

    enabled = bool(controller_payload.get("enabled", False))
    if not enabled:
        return ControllerConfig(enabled=False)

    mode = str(controller_payload.get("mode", ""))
    if mode not in _ALLOWED_MODES:
        raise ControllerConfigError(
            "controller.mode must be one of "
            f"{sorted(_ALLOWED_MODES)} (got {mode!r})"
        )

    t_start = float(controller_payload.get("t_start", -1.0))
    if not (0.0 <= t_start <= 1.0):
        raise ControllerConfigError(
            f"controller.t_start must lie in [0, 1] (got {t_start})"
        )

    refresh_interval = int(controller_payload.get("refresh_interval", 0))
    if refresh_interval <= 0:
        raise ControllerConfigError(
            f"controller.refresh_interval must be positive (got {refresh_interval})"
        )

    completion_payload = _require_mapping(controller_payload, "completion")
    completion_method = str(completion_payload.get("method", ""))
    if completion_method != "argmax":
        raise ControllerConfigError(
            f"controller.completion.method must be 'argmax' for D1 (got {completion_method!r})"
        )
    completion = CompletionConfig(method=completion_method)

    head_payload = _require_mapping(controller_payload, "head")
    score_scale = str(head_payload.get("score_scale", ""))
    if score_scale != "raw_logit":
        raise ControllerConfigError(
            f"controller.head.score_scale must be 'raw_logit' for D1 (got {score_scale!r})"
        )
    static_cache_policy = str(head_payload.get("static_cache_policy", "lazy_write"))
    if static_cache_policy not in {"lazy_write", "read_only"}:
        raise ControllerConfigError(
            "controller.head.static_cache_policy must be 'lazy_write' or 'read_only' "
            f"(got {static_cache_policy!r})"
        )
    local_risk_aggregation = str(head_payload.get("local_risk_aggregation", "LME"))
    if local_risk_aggregation != "LME":
        raise ControllerConfigError(
            "controller.head.local_risk_aggregation must be 'LME' for the first D2/D3 "
            f"implementation (got {local_risk_aggregation!r})"
        )
    head = HeadConfig(
        score_scale=score_scale,
        static_cache_policy=static_cache_policy,
        local_risk_aggregation=local_risk_aggregation,
    )

    aw_payload = _require_mapping(controller_payload, "active_windows")
    selection = str(aw_payload.get("selection", ""))
    if selection != "threshold_then_top_n":
        raise ControllerConfigError(
            "controller.active_windows.selection must be 'threshold_then_top_n' for D1 "
            f"(got {selection!r})"
        )
    max_windows = int(aw_payload.get("max_windows", 0))
    if max_windows <= 0:
        raise ControllerConfigError(
            f"controller.active_windows.max_windows must be positive (got {max_windows})"
        )
    active_windows = ActiveWindowsConfig(
        excess_threshold=float(aw_payload.get("excess_threshold", 0.0)),
        max_windows=max_windows,
        selection=selection,
        merge_overlapping_scoring_windows=bool(
            aw_payload.get("merge_overlapping_scoring_windows", True)
        ),
    )

    rel_payload = _require_mapping(controller_payload, "reliability")
    time_k = float(rel_payload.get("time_k", 20.0))
    if time_k <= 0.0:
        raise ControllerConfigError(
            f"controller.reliability.time_k must be positive (got {time_k})"
        )
    entropy_h0 = float(rel_payload.get("entropy_h0", 1.5))
    if entropy_h0 <= 0.0:
        raise ControllerConfigError(
            f"controller.reliability.entropy_h0 must be positive (got {entropy_h0})"
        )
    min_completion_fraction = float(rel_payload.get("min_completion_fraction", 0.5))
    if not (0.0 <= min_completion_fraction <= 1.0):
        raise ControllerConfigError(
            "controller.reliability.min_completion_fraction must lie in [0, 1] "
            f"(got {min_completion_fraction})"
        )
    min_rho_to_emit_event = float(rel_payload.get("min_rho_to_emit_event", 0.0))
    if not (0.0 <= min_rho_to_emit_event <= 1.0):
        raise ControllerConfigError(
            "controller.reliability.min_rho_to_emit_event must lie in [0, 1] "
            f"(got {min_rho_to_emit_event})"
        )
    reliability = ReliabilityConfig(
        time_k=time_k,
        entropy_h0=entropy_h0,
        min_completion_fraction=min_completion_fraction,
        min_rho_to_emit_event=min_rho_to_emit_event,
    )

    d2 = _materialize_d2(controller_payload.get("d2"))
    d3 = _materialize_d3(controller_payload.get("d3"))
    _cross_validate_mode(mode=mode, d2=d2, d3=d3)
    attribution = _materialize_attribution(controller_payload.get("attribution"))
    controls = _materialize_controls(controller_payload.get("controls"))
    targeting = _materialize_targeting(controller_payload.get("targeting"))
    global_pressure = _materialize_global_pressure(
        controller_payload.get("global_pressure")
    )
    self_conditioned_gr = _materialize_self_conditioned_gr(
        controller_payload.get("self_conditioned_gr")
    )
    allocation = _materialize_allocation(controller_payload.get("allocation"))
    _cross_validate_stage_bc(
        targeting=targeting, global_pressure=global_pressure, d3=d3
    )
    _cross_validate_scgr(
        self_conditioned_gr=self_conditioned_gr, global_pressure=global_pressure
    )
    _cross_validate_allocation(
        targeting=targeting,
        allocation=allocation,
        self_conditioned_gr=self_conditioned_gr,
        global_pressure=global_pressure,
    )

    tel_payload = controller_payload.get("telemetry", {})
    if not isinstance(tel_payload, dict):
        raise ControllerConfigError("controller.telemetry must be a mapping")
    telemetry = TelemetryConfig(
        write_refresh_log=bool(tel_payload.get("write_refresh_log", True)),
        write_controller_events=bool(tel_payload.get("write_controller_events", True)),
        write_per_protein_summary=bool(tel_payload.get("write_per_protein_summary", True)),
    )

    return ControllerConfig(
        enabled=True,
        mode=mode,
        t_start=t_start,
        refresh_interval=refresh_interval,
        completion=completion,
        head=head,
        active_windows=active_windows,
        reliability=reliability,
        d2=d2,
        d3=d3,
        attribution=attribution,
        controls=controls,
        telemetry=telemetry,
        targeting=targeting,
        global_pressure=global_pressure,
        self_conditioned_gr=self_conditioned_gr,
        allocation=allocation,
    )


def _materialize_d2(payload: Any) -> D2Config:
    if payload is None:
        return D2Config()
    if not isinstance(payload, dict):
        raise ControllerConfigError("controller.d2 must be a mapping when present")

    enabled = bool(payload.get("enabled", False))
    if not enabled:
        # Carry-through of disabled section: ignore other fields, return default
        # disabled config so cross-mode validation still sees enabled=False.
        return D2Config(enabled=False)

    top_k_tokens = int(payload.get("top_k_tokens", 0))
    if top_k_tokens <= 0:
        raise ControllerConfigError(
            f"controller.d2.top_k_tokens must be positive (got {top_k_tokens})"
        )
    max_positions = int(payload.get("max_positions_per_block", 0))
    if max_positions <= 0:
        raise ControllerConfigError(
            f"controller.d2.max_positions_per_block must be positive (got {max_positions})"
        )
    max_candidates = int(payload.get("max_candidates_per_block", 0))
    if max_candidates <= 0:
        raise ControllerConfigError(
            f"controller.d2.max_candidates_per_block must be positive (got {max_candidates})"
        )
    beta = float(payload.get("beta", 1.0))
    if beta < 0.0:
        raise ControllerConfigError(
            f"controller.d2.beta must be >= 0 (got {beta})"
        )
    eta = float(payload.get("eta", 0.7))
    if eta < 0.0:
        raise ControllerConfigError(f"controller.d2.eta must be >= 0 (got {eta})")
    struct_temperature = float(payload.get("struct_temperature", 1.0))
    if struct_temperature <= 0.0:
        raise ControllerConfigError(
            "controller.d2.struct_temperature must be positive "
            f"(got {struct_temperature})"
        )
    delta_struct = float(payload.get("delta_struct", 1.5))
    if delta_struct <= 0.0:
        raise ControllerConfigError(
            f"controller.d2.delta_struct must be positive (got {delta_struct})"
        )
    context_pnll_h0 = float(payload.get("context_pnll_h0", 2.0))
    if context_pnll_h0 <= 0.0:
        raise ControllerConfigError(
            f"controller.d2.context_pnll_h0 must be positive (got {context_pnll_h0})"
        )
    context_jsd_h0 = float(payload.get("context_jsd_h0", 0.5))
    if context_jsd_h0 <= 0.0:
        raise ControllerConfigError(
            f"controller.d2.context_jsd_h0 must be positive (got {context_jsd_h0})"
        )
    epsilon = float(payload.get("epsilon", 1.0e-8))
    if epsilon <= 0.0:
        raise ControllerConfigError(
            f"controller.d2.epsilon must be positive (got {epsilon})"
        )
    candidate_mode = str(payload.get("candidate_mode", "structure_topk"))
    if candidate_mode != "structure_topk":
        raise ControllerConfigError(
            "controller.d2.candidate_mode must be 'structure_topk' for the first "
            f"D2 implementation (got {candidate_mode!r})"
        )
    selection_score = str(payload.get("selection_score", "residue_excess_then_low_entropy"))
    if selection_score != "residue_excess_then_low_entropy":
        raise ControllerConfigError(
            "controller.d2.selection_score must be 'residue_excess_then_low_entropy' "
            f"(got {selection_score!r})"
        )
    min_delta_R_improvement = float(payload.get("min_delta_R_improvement", 0.0))
    if min_delta_R_improvement < 0.0:
        raise ControllerConfigError(
            "controller.d2.min_delta_R_improvement must be >= 0 "
            f"(got {min_delta_R_improvement})"
        )
    min_ess_fraction = float(payload.get("min_ess_fraction", 0.25))
    if not (0.0 <= min_ess_fraction <= 1.0):
        raise ControllerConfigError(
            "controller.d2.min_ess_fraction must lie in [0, 1] "
            f"(got {min_ess_fraction})"
        )
    max_abs_logit_shift = float(payload.get("max_abs_logit_shift", 5.0))
    if max_abs_logit_shift <= 0.0:
        raise ControllerConfigError(
            "controller.d2.max_abs_logit_shift must be positive "
            f"(got {max_abs_logit_shift})"
        )
    completion_ensemble_size = int(payload.get("completion_ensemble_size", 3))
    if completion_ensemble_size < 1:
        raise ControllerConfigError(
            "controller.d2.completion_ensemble_size must be >= 1 "
            f"(got {completion_ensemble_size})"
        )
    completion_ensemble_scope = str(
        payload.get("completion_ensemble_scope", "local_windows")
    )
    if completion_ensemble_scope != "local_windows":
        raise ControllerConfigError(
            "controller.d2.completion_ensemble_scope must be 'local_windows' "
            f"for Stage A (got {completion_ensemble_scope!r})"
        )
    completion_ensemble_rescore_top_m = int(
        payload.get("completion_ensemble_rescore_top_m", 8)
    )
    if completion_ensemble_rescore_top_m < 1:
        raise ControllerConfigError(
            "controller.d2.completion_ensemble_rescore_top_m must be >= 1 "
            f"(got {completion_ensemble_rescore_top_m})"
        )
    completion_ensemble_use_variance_gate = bool(
        payload.get("completion_ensemble_use_variance_gate", False)
    )
    if completion_ensemble_use_variance_gate:
        raise ControllerConfigError(
            "controller.d2.completion_ensemble_use_variance_gate must be false "
            "for Stage A"
        )
    sticky_ttl_steps = int(payload.get("sticky_ttl_steps", 5))
    if sticky_ttl_steps < 1:
        raise ControllerConfigError(
            f"controller.d2.sticky_ttl_steps must be >= 1 (got {sticky_ttl_steps})"
        )
    candidate_score_source = str(
        payload.get("candidate_score_source", "local")
    )
    if candidate_score_source not in _ALLOWED_CANDIDATE_SCORE_SOURCES:
        raise ControllerConfigError(
            "controller.d2.candidate_score_source must be one of "
            f"{sorted(_ALLOWED_CANDIDATE_SCORE_SOURCES)} "
            f"(got {candidate_score_source!r})"
        )
    candidate_terminal_K_P = int(payload.get("candidate_terminal_K_P", 4))
    if candidate_terminal_K_P < 1:
        raise ControllerConfigError(
            "controller.d2.candidate_terminal_K_P must be >= 1 "
            f"(got {candidate_terminal_K_P})"
        )
    # §A2: the terminal probe rescoring rides the completion-ensemble shortlist
    # path (_process_block). With completion_ensemble_enabled=false that branch is
    # skipped and D2 falls back to the local argmax ΔR_B — a silent degrade that
    # would still be labelled 'terminal'. Require the ensemble path to be on.
    if (
        candidate_score_source == "terminal"
        and not bool(payload.get("completion_ensemble_enabled", True))
    ):
        raise ControllerConfigError(
            "controller.d2.candidate_score_source='terminal' requires "
            "controller.d2.completion_ensemble_enabled=true (the terminal probe "
            "rescores the ensemble shortlist; with it off the rescore is skipped "
            "and D2 silently falls back to the local argmax ΔR_B)"
        )
    return D2Config(
        enabled=True,
        beta=beta,
        eta=eta,
        struct_temperature=struct_temperature,
        delta_struct=delta_struct,
        context_pnll_h0=context_pnll_h0,
        context_jsd_h0=context_jsd_h0,
        epsilon=epsilon,
        candidate_mode=candidate_mode,
        top_k_tokens=top_k_tokens,
        max_positions_per_block=max_positions,
        max_candidates_per_block=max_candidates,
        selection_score=selection_score,
        min_delta_R_improvement=min_delta_R_improvement,
        min_ess_fraction=min_ess_fraction,
        max_abs_logit_shift=max_abs_logit_shift,
        paired_uncorrected_sample=bool(payload.get("paired_uncorrected_sample", True)),
        completion_ensemble_enabled=bool(payload.get("completion_ensemble_enabled", True)),
        completion_ensemble_size=completion_ensemble_size,
        completion_ensemble_scope=completion_ensemble_scope,
        completion_ensemble_rescore_top_m=completion_ensemble_rescore_top_m,
        completion_ensemble_use_variance_gate=completion_ensemble_use_variance_gate,
        sticky_ttl_steps=sticky_ttl_steps,
        sticky_clear_on_selected=bool(payload.get("sticky_clear_on_selected", True)),
        sticky_clear_on_remask=bool(payload.get("sticky_clear_on_remask", True)),
        sticky_overwrite_on_refresh=bool(payload.get("sticky_overwrite_on_refresh", True)),
        candidate_score_source=candidate_score_source,
        candidate_terminal_K_P=candidate_terminal_K_P,
    )


def _materialize_d3(payload: Any) -> D3Config:
    if payload is None:
        return D3Config()
    if not isinstance(payload, dict):
        raise ControllerConfigError("controller.d3 must be a mapping when present")

    enabled = bool(payload.get("enabled", False))
    projection = str(payload.get("window_to_residue_projection", "max_covering_window"))
    if projection != "max_covering_window":
        raise ControllerConfigError(
            "controller.d3.window_to_residue_projection must be 'max_covering_window' "
            f"for the first D3 implementation (got {projection!r})"
        )
    gamma_min = float(payload.get("gamma_min", 0.4))
    gamma_max = float(payload.get("gamma_max", 0.9))
    if not (0.0 <= gamma_min <= 1.0 and 0.0 <= gamma_max <= 1.0):
        raise ControllerConfigError(
            "controller.d3.gamma_min and gamma_max must lie in [0, 1] "
            f"(got gamma_min={gamma_min}, gamma_max={gamma_max})"
        )
    if gamma_min > gamma_max:
        raise ControllerConfigError(
            "controller.d3.gamma_min must be <= gamma_max "
            f"(got gamma_min={gamma_min}, gamma_max={gamma_max})"
        )
    lambda_commit = float(payload.get("lambda_commit", 1.0))
    if lambda_commit < 0.0:
        raise ControllerConfigError(
            f"controller.d3.lambda_commit must be >= 0 (got {lambda_commit})"
        )
    alpha_struct = float(payload.get("alpha_struct", 0.3))
    if not (0.0 <= alpha_struct <= 1.0):
        raise ControllerConfigError(
            f"controller.d3.alpha_struct must lie in [0, 1] (got {alpha_struct})"
        )
    d2_evidence_nu = float(payload.get("d2_evidence_nu", 1.0))
    if d2_evidence_nu < 0.0:
        raise ControllerConfigError(
            f"controller.d3.d2_evidence_nu must be >= 0 (got {d2_evidence_nu})"
        )
    d2_evidence_ttl_steps = int(payload.get("d2_evidence_ttl_steps", 5))
    if d2_evidence_ttl_steps < 1:
        raise ControllerConfigError(
            "controller.d3.d2_evidence_ttl_steps must be >= 1 "
            f"(got {d2_evidence_ttl_steps})"
        )
    zscore_epsilon = float(payload.get("zscore_epsilon", 1.0e-6))
    if zscore_epsilon <= 0.0:
        raise ControllerConfigError(
            f"controller.d3.zscore_epsilon must be positive (got {zscore_epsilon})"
        )
    final_freeze_steps = int(payload.get("final_freeze_steps", 1))
    if final_freeze_steps < 1:
        raise ControllerConfigError(
            "controller.d3.final_freeze_steps must be >= 1 "
            f"(got {final_freeze_steps})"
        )
    evidence_source = str(payload.get("evidence_source", "legacy_window_excess"))
    if evidence_source not in _ALLOWED_D3_EVIDENCE_SOURCES:
        raise ControllerConfigError(
            "controller.d3.evidence_source must be one of "
            f"{sorted(_ALLOWED_D3_EVIDENCE_SOURCES)} (got {evidence_source!r})"
        )
    return D3Config(
        enabled=enabled,
        window_to_residue_projection=projection,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        lambda_commit=lambda_commit,
        alpha_struct=alpha_struct,
        d2_evidence_nu=d2_evidence_nu,
        d2_evidence_ttl_steps=d2_evidence_ttl_steps,
        d2_evidence_requires_benefit=bool(
            payload.get("d2_evidence_requires_benefit", True)
        ),
        zscore_epsilon=zscore_epsilon,
        same_refresh_grace=bool(payload.get("same_refresh_grace", True)),
        final_freeze_steps=final_freeze_steps,
        evidence_source=evidence_source,
    )


def _materialize_attribution(payload: Any) -> AttributionConfig:
    if payload is None:
        return AttributionConfig()
    if not isinstance(payload, dict):
        raise ControllerConfigError("controller.attribution must be a mapping when present")
    productive_delta_logp = float(payload.get("productive_delta_logp", 0.5))
    if productive_delta_logp < 0.0:
        raise ControllerConfigError(
            "controller.attribution.productive_delta_logp must be >= 0 "
            f"(got {productive_delta_logp})"
        )
    return AttributionConfig(
        write_paired_counterfactual=bool(payload.get("write_paired_counterfactual", True)),
        write_independent_delta_R_i=bool(payload.get("write_independent_delta_R_i", True)),
        productive_delta_logp=productive_delta_logp,
    )


def _materialize_controls(payload: Any) -> ControlsConfig:
    if payload is None:
        return ControlsConfig()
    if not isinstance(payload, dict):
        raise ControllerConfigError("controller.controls must be a mapping when present")
    return ControlsConfig(
        allow_wrong_allele_head=bool(payload.get("allow_wrong_allele_head", True)),
        allow_shuffled_head=bool(payload.get("allow_shuffled_head", True)),
    )


def _materialize_targeting(payload: Any) -> TargetingConfig:
    if payload is None:
        return TargetingConfig()
    if not isinstance(payload, dict):
        raise ControllerConfigError(
            "controller.targeting must be a mapping when present"
        )
    defaults = TargetingConfig()

    mode = str(payload.get("mode", defaults.mode))
    if mode not in _ALLOWED_TARGETING_MODES:
        raise ControllerConfigError(
            "controller.targeting.mode must be one of "
            f"{sorted(_ALLOWED_TARGETING_MODES)} (got {mode!r})"
        )

    tau_ref_source = str(payload.get("tau_ref_source", defaults.tau_ref_source))
    if tau_ref_source not in _ALLOWED_TAU_REF_SOURCES:
        raise ControllerConfigError(
            "controller.targeting.tau_ref_source must be one of "
            f"{sorted(_ALLOWED_TAU_REF_SOURCES)} (got {tau_ref_source!r})"
        )
    if tau_ref_source in _RESERVED_TAU_REF_SOURCES:
        raise NotImplementedError(
            "controller.targeting.tau_ref_source='first_reliable_refresh_median' "
            "is reserved but not implemented in Stage B v1; only 'static_median' "
            "is available. Provide a static window cache instead."
        )

    tau_ref_quantile = float(payload.get("tau_ref_quantile", defaults.tau_ref_quantile))
    if not (0.0 <= tau_ref_quantile <= 1.0):
        raise ControllerConfigError(
            "controller.targeting.tau_ref_quantile must lie in [0, 1] "
            f"(got {tau_ref_quantile})"
        )

    projection = str(payload.get("projection", defaults.projection))
    if projection != "max_covering_window":
        raise ControllerConfigError(
            "controller.targeting.projection must be 'max_covering_window' for "
            f"Stage B v1 (got {projection!r})"
        )

    env_seed_top_current = int(
        payload.get("env_seed_top_current", defaults.env_seed_top_current)
    )
    env_seed_top_uncertain = int(
        payload.get("env_seed_top_uncertain", defaults.env_seed_top_uncertain)
    )
    env_seed_max_windows = int(
        payload.get("env_seed_max_windows", defaults.env_seed_max_windows)
    )
    for name, val in (
        ("env_seed_top_current", env_seed_top_current),
        ("env_seed_top_uncertain", env_seed_top_uncertain),
        ("env_seed_max_windows", env_seed_max_windows),
    ):
        if val < 0:
            raise ControllerConfigError(
                f"controller.targeting.{name} must be >= 0 (got {val})"
            )

    env_ensemble_size = int(payload.get("env_ensemble_size", defaults.env_ensemble_size))
    if env_ensemble_size < 1:
        raise ControllerConfigError(
            "controller.targeting.env_ensemble_size must be >= 1 "
            f"(got {env_ensemble_size})"
        )

    env_consistency_floor = float(
        payload.get("env_consistency_floor", defaults.env_consistency_floor)
    )
    if not (0.0 <= env_consistency_floor <= 1.0):
        raise ControllerConfigError(
            "controller.targeting.env_consistency_floor must lie in [0, 1] "
            f"(got {env_consistency_floor})"
        )

    softor_tau = float(payload.get("softor_tau", defaults.softor_tau))
    if softor_tau <= 0.0:
        raise ControllerConfigError(
            f"controller.targeting.softor_tau must be positive (got {softor_tau})"
        )

    r_ctx_floor = float(payload.get("r_ctx_floor", defaults.r_ctx_floor))
    if not (0.0 <= r_ctx_floor <= 1.0):
        raise ControllerConfigError(
            f"controller.targeting.r_ctx_floor must lie in [0, 1] (got {r_ctx_floor})"
        )

    mem_half_life_refreshes = float(
        payload.get("mem_half_life_refreshes", defaults.mem_half_life_refreshes)
    )
    if mem_half_life_refreshes <= 0.0:
        raise ControllerConfigError(
            "controller.targeting.mem_half_life_refreshes must be positive "
            f"(got {mem_half_life_refreshes})"
        )

    mem_cold_start = str(payload.get("mem_cold_start", defaults.mem_cold_start))
    if mem_cold_start != "zero":
        raise ControllerConfigError(
            "controller.targeting.mem_cold_start must be 'zero' for Stage B v1 "
            f"(got {mem_cold_start!r})"
        )

    cluster_radius = int(payload.get("cluster_radius", defaults.cluster_radius))
    if cluster_radius < 0:
        raise ControllerConfigError(
            f"controller.targeting.cluster_radius must be >= 0 (got {cluster_radius})"
        )
    cluster_min_mass = float(payload.get("cluster_min_mass", defaults.cluster_min_mass))
    if cluster_min_mass <= 0.0:
        raise ControllerConfigError(
            "controller.targeting.cluster_min_mass must be positive "
            f"(got {cluster_min_mass})"
        )
    cluster_floor = float(payload.get("cluster_floor", defaults.cluster_floor))
    if not (0.0 <= cluster_floor <= 1.0):
        raise ControllerConfigError(
            f"controller.targeting.cluster_floor must lie in [0, 1] (got {cluster_floor})"
        )

    active_window_source = str(
        payload.get("active_window_source", defaults.active_window_source)
    )
    if active_window_source != "v_target":
        raise ControllerConfigError(
            "controller.targeting.active_window_source must be 'v_target' for "
            f"Stage B v1 (got {active_window_source!r})"
        )
    active_window_min_excess = float(
        payload.get("active_window_min_excess", defaults.active_window_min_excess)
    )
    if active_window_min_excess < 0.0:
        raise ControllerConfigError(
            "controller.targeting.active_window_min_excess must be >= 0 "
            f"(got {active_window_min_excess})"
        )
    within_block_source = str(
        payload.get("within_block_source", defaults.within_block_source)
    )
    if within_block_source not in _ALLOWED_WITHIN_BLOCK_SOURCES:
        raise ControllerConfigError(
            "controller.targeting.within_block_source must be one of "
            f"{sorted(_ALLOWED_WITHIN_BLOCK_SOURCES)} (got {within_block_source!r}); "
            "'e_fresh' is reserved (deferred per RAR 0006)"
        )

    selection_field_mode = str(
        payload.get("selection_field_mode", defaults.selection_field_mode)
    )
    if selection_field_mode not in _ALLOWED_SELECTION_FIELD_MODES:
        raise ControllerConfigError(
            "controller.targeting.selection_field_mode must be one of "
            f"{sorted(_ALLOWED_SELECTION_FIELD_MODES)} (got {selection_field_mode!r})"
        )

    return TargetingConfig(
        mode=mode,
        tau_ref_source=tau_ref_source,
        tau_ref_quantile=tau_ref_quantile,
        projection=projection,
        env_seed_top_current=env_seed_top_current,
        env_seed_top_uncertain=env_seed_top_uncertain,
        env_seed_max_windows=env_seed_max_windows,
        env_ensemble_size=env_ensemble_size,
        env_consistency_floor=env_consistency_floor,
        softor_tau=softor_tau,
        r_ctx_floor=r_ctx_floor,
        mem_half_life_refreshes=mem_half_life_refreshes,
        mem_cold_start=mem_cold_start,
        cluster_radius=cluster_radius,
        cluster_min_mass=cluster_min_mass,
        cluster_floor=cluster_floor,
        use_cluster_for_active_blocks=bool(
            payload.get(
                "use_cluster_for_active_blocks",
                defaults.use_cluster_for_active_blocks,
            )
        ),
        active_window_source=active_window_source,
        active_window_min_excess=active_window_min_excess,
        write_actionability_telemetry=bool(
            payload.get(
                "write_actionability_telemetry",
                defaults.write_actionability_telemetry,
            )
        ),
        within_block_source=within_block_source,
        selection_field_mode=selection_field_mode,
    )


def _materialize_allocation(payload: Any) -> AllocationConfig:
    if payload is None:
        return AllocationConfig()
    if not isinstance(payload, dict):
        raise ControllerConfigError(
            "controller.allocation must be a mapping when present"
        )
    defaults = AllocationConfig()
    arm = str(payload.get("arm", defaults.arm))
    if arm not in _ALLOWED_SCGR_ARMS:
        raise ControllerConfigError(
            "controller.allocation.arm must be one of "
            f"{sorted(_ALLOWED_SCGR_ARMS)} (got {arm!r})"
        )
    smooth_half_width = int(
        payload.get("smooth_half_width", defaults.smooth_half_width)
    )
    if smooth_half_width < 0:
        raise ControllerConfigError(
            "controller.allocation.smooth_half_width must be >= 0 "
            f"(got {smooth_half_width})"
        )
    reweight_c = float(payload.get("reweight_c", defaults.reweight_c))
    if reweight_c < 0.0:
        raise ControllerConfigError(
            f"controller.allocation.reweight_c must be >= 0 (got {reweight_c})"
        )
    reweight_eps = float(payload.get("reweight_eps", defaults.reweight_eps))
    if reweight_eps < 0.0:
        raise ControllerConfigError(
            f"controller.allocation.reweight_eps must be >= 0 (got {reweight_eps})"
        )
    eligible_quantile = float(
        payload.get("eligible_quantile", defaults.eligible_quantile)
    )
    if not (0.0 <= eligible_quantile <= 1.0):
        raise ControllerConfigError(
            "controller.allocation.eligible_quantile must lie in [0, 1] "
            f"(got {eligible_quantile})"
        )
    triage_lambda = float(payload.get("triage_lambda", defaults.triage_lambda))
    if triage_lambda < 0.0:
        raise ControllerConfigError(
            f"controller.allocation.triage_lambda must be >= 0 (got {triage_lambda})"
        )
    terminal_eligible_quantile = float(
        payload.get("terminal_eligible_quantile", defaults.terminal_eligible_quantile)
    )
    if not (0.0 <= terminal_eligible_quantile <= 1.0):
        raise ControllerConfigError(
            "controller.allocation.terminal_eligible_quantile must lie in [0, 1] "
            f"(got {terminal_eligible_quantile})"
        )
    terminal_lambda = float(
        payload.get("terminal_lambda", defaults.terminal_lambda)
    )
    if terminal_lambda < 0.0:
        raise ControllerConfigError(
            "controller.allocation.terminal_lambda must be >= 0 "
            f"(got {terminal_lambda})"
        )
    return AllocationConfig(
        enabled=bool(payload.get("enabled", defaults.enabled)),
        arm=arm,
        smooth_half_width=smooth_half_width,
        reweight_c=reweight_c,
        reweight_eps=reweight_eps,
        eligible_quantile=eligible_quantile,
        triage_lambda=triage_lambda,
        terminal_eligible_quantile=terminal_eligible_quantile,
        terminal_lambda=terminal_lambda,
    )


def _materialize_global_pressure(payload: Any) -> GlobalPressureConfig:
    if payload is None:
        return GlobalPressureConfig()
    if not isinstance(payload, dict):
        raise ControllerConfigError(
            "controller.global_pressure must be a mapping when present"
        )
    defaults = GlobalPressureConfig()
    enabled = bool(payload.get("enabled", defaults.enabled))
    g_min = float(payload.get("g_min", defaults.g_min))
    g_max = float(payload.get("g_max", defaults.g_max))
    if g_min < 0.0:
        raise ControllerConfigError(
            f"controller.global_pressure.g_min must be >= 0 (got {g_min})"
        )
    if g_max < g_min:
        raise ControllerConfigError(
            "controller.global_pressure.g_max must be >= g_min "
            f"(got g_min={g_min}, g_max={g_max})"
        )
    mapping = str(payload.get("mapping", defaults.mapping))
    if mapping not in _ALLOWED_PRESSURE_MAPPINGS:
        raise ControllerConfigError(
            "controller.global_pressure.mapping must be one of "
            f"{sorted(_ALLOWED_PRESSURE_MAPPINGS)} (got {mapping!r})"
        )
    pressure_source = str(payload.get("pressure_source", defaults.pressure_source))
    if pressure_source not in _ALLOWED_PRESSURE_SOURCES:
        raise ControllerConfigError(
            "controller.global_pressure.pressure_source must be one of "
            f"{sorted(_ALLOWED_PRESSURE_SOURCES)} (got {pressure_source!r})"
        )
    # G3 / RAR 0006 + SC1.1: an actuating run must use the thresholded-G driver
    # OR the SC-GR probe; the legacy mean-based trajectory_median_G is rejected.
    if enabled and pressure_source not in _ENABLED_PRESSURE_SOURCES:
        raise ControllerConfigError(
            "controller.global_pressure.enabled=true requires pressure_source in "
            f"{sorted(_ENABLED_PRESSURE_SOURCES)} (got {pressure_source!r}); the "
            "legacy trajectory_median_G is not valid for an actuating run"
        )
    raw_tau_prom_source = payload.get("tau_prom_source", defaults.tau_prom_source)
    tau_prom_source = (
        None if raw_tau_prom_source is None else str(raw_tau_prom_source)
    )
    if tau_prom_source is not None and tau_prom_source not in _ALLOWED_TAU_PROM_SOURCES:
        raise ControllerConfigError(
            "controller.global_pressure.tau_prom_source must be one of "
            f"{sorted(_ALLOWED_TAU_PROM_SOURCES)} when present "
            f"(got {tau_prom_source!r})"
        )
    raw_tau_prom = payload.get("tau_prom", defaults.tau_prom)
    tau_prom = None if raw_tau_prom is None else float(raw_tau_prom)
    raw_B_low = payload.get("B_low", defaults.B_low)
    raw_B_high = payload.get("B_high", defaults.B_high)
    B_low = None if raw_B_low is None else float(raw_B_low)
    B_high = None if raw_B_high is None else float(raw_B_high)
    # Fail fast on a hardcoded inverted band. Presence (band must exist when
    # enabled) is deferred to validate_global_pressure_runtime, because the
    # primary C.1 YAML omits the band and stamps it from the calibration JSON
    # after load (PLAN_RF_UNI_CTRL.md C1.3).
    if B_low is not None and B_high is not None and not (B_high > B_low):
        raise ControllerConfigError(
            "controller.global_pressure requires B_high > B_low "
            f"(got B_low={B_low}, B_high={B_high})"
        )
    min_reliable_refreshes = int(
        payload.get("min_reliable_refreshes", defaults.min_reliable_refreshes)
    )
    if min_reliable_refreshes < 1:
        raise ControllerConfigError(
            "controller.global_pressure.min_reliable_refreshes must be >= 1 "
            f"(got {min_reliable_refreshes})"
        )
    unready_g = float(payload.get("unready_g", defaults.unready_g))
    if not (0.0 <= unready_g <= g_max):
        raise ControllerConfigError(
            "controller.global_pressure.unready_g must be within [0, g_max] "
            f"(got unready_g={unready_g}, g_max={g_max})"
        )
    amplify = bool(payload.get("amplify", defaults.amplify))
    # SC2.1: a two-sided gain needs g_min below 1 (suppress low) and g_max above
    # 1 (amplify high), bounded to keep beta scaling sane.
    if amplify and not (g_min < 1.0 < g_max <= 3.0):
        raise ControllerConfigError(
            "controller.global_pressure.amplify=true requires "
            f"g_min < 1.0 < g_max <= 3.0 (got g_min={g_min}, g_max={g_max})"
        )
    return GlobalPressureConfig(
        enabled=enabled,
        g_min=g_min,
        g_max=g_max,
        amplify=amplify,
        pressure_source=pressure_source,
        tau_prom_source=tau_prom_source,
        tau_prom=tau_prom,
        mapping=mapping,
        B_low=B_low,
        B_high=B_high,
        min_reliable_refreshes=min_reliable_refreshes,
        unready_g=unready_g,
        scale_beta=bool(payload.get("scale_beta", defaults.scale_beta)),
        scale_lambda=bool(payload.get("scale_lambda", defaults.scale_lambda)),
        G0_source=str(payload.get("G0_source", defaults.G0_source)),
        s_G_source=str(payload.get("s_G_source", defaults.s_G_source)),
    )


def validate_global_pressure_runtime(global_pressure: GlobalPressureConfig) -> None:
    """Fail fast if Stage C.1 pressure is enabled but its band is not stamped.

    Called at run setup AFTER the calibration JSON is stamped into the config
    (PLAN_RF_UNI_CTRL.md C1.3). ``_materialize_global_pressure`` deliberately
    tolerates ``enabled=true`` with a missing ``B_low``/``B_high`` so the primary
    C.1 YAML can be loaded before the band is known; this guard ensures no run
    ever actuates with an un-calibrated (or inverted) band — no placeholder
    anchors allowed (CLAUDE.md fail-fast rule).
    """
    if not global_pressure.enabled:
        return
    # τ_prom is stamped from the calibration JSON and required ONLY for the
    # thresholded G_step driver (G3); the SC-GR probe source (SC1.1) bins B_sc
    # directly and needs no τ_prom. The band is required for both sources.
    if (
        global_pressure.pressure_source == _REQUIRED_C1_PRESSURE_SOURCE
        and global_pressure.tau_prom is None
    ):
        raise ControllerConfigError(
            "controller.global_pressure.enabled=true with pressure_source="
            f"{_REQUIRED_C1_PRESSURE_SOURCE!r} requires tau_prom to be stamped from "
            "the Stage C calibration JSON (--global-pressure-calibration-json); it "
            "may not be hardcoded"
        )
    if global_pressure.B_low is None or global_pressure.B_high is None:
        raise ControllerConfigError(
            "controller.global_pressure.enabled=true requires B_low and B_high to "
            "be stamped from the Stage C calibration JSON "
            "(--global-pressure-calibration-json); neither may be hardcoded"
        )
    if not (global_pressure.B_high > global_pressure.B_low):
        raise ControllerConfigError(
            "controller.global_pressure requires B_high > B_low "
            f"(got B_low={global_pressure.B_low}, B_high={global_pressure.B_high})"
        )


def _cross_validate_stage_bc(
    *,
    targeting: TargetingConfig,
    global_pressure: GlobalPressureConfig,
    d3: D3Config,
) -> None:
    """Enforce the Stage B/C cross-config contract (PLAN_RF_UNI_CTRL.md C1.2).

    Stage C.1 builds ``g_GR`` from the typed pressure field, so global-pressure
    actuation is only valid under typed targeting. The typed ``e_fresh`` D3 input
    likewise only exists when typed targeting builds the field and D3 is active.
    Catching these at materialization turns a mid-run crash (missing ``e_fresh``
    / no-op pressure) into a config-time fail-fast. The presence of a calibrated
    ``B_low``/``B_high`` band is checked separately at run setup by
    :func:`validate_global_pressure_runtime` (the band is stamped post-load).
    """
    if global_pressure.enabled and targeting.mode != "typed_actionability":
        raise ControllerConfigError(
            "controller.global_pressure.enabled=true (Stage C.1 actuation) requires "
            "controller.targeting.mode='typed_actionability'; g_GR is built from the "
            "typed pressure field"
        )
    # G10 / RAR 0006 M9: the primary C.1 base is the typed_target / legacy-D3 arm;
    # typed_fresh inflates the D3 EMA (harmful) and is deferred. Reject it when
    # pressure actuation is enabled.
    if global_pressure.enabled and d3.evidence_source != "legacy_window_excess":
        raise ControllerConfigError(
            "controller.global_pressure.enabled=true (Stage C.1) requires "
            "controller.d3.evidence_source='legacy_window_excess' (G10); "
            f"got {d3.evidence_source!r} (typed_fresh is harmful per RAR 0006 M9 "
            "and deferred)"
        )
    if d3.evidence_source == "typed_fresh":
        if targeting.mode != "typed_actionability":
            raise ControllerConfigError(
                "controller.d3.evidence_source='typed_fresh' requires "
                "controller.targeting.mode='typed_actionability' (the typed "
                "e_fresh field only exists under typed targeting)"
            )
        if not d3.enabled:
            raise ControllerConfigError(
                "controller.d3.evidence_source='typed_fresh' requires "
                "controller.d3.enabled=true (D3 must be active to consume it)"
            )


def _materialize_self_conditioned_gr(payload: Any) -> SelfConditionedGRConfig:
    if payload is None:
        return SelfConditionedGRConfig()
    if not isinstance(payload, dict):
        raise ControllerConfigError(
            "controller.self_conditioned_gr must be a mapping when present"
        )
    defaults = SelfConditionedGRConfig()

    enabled = bool(payload.get("enabled", defaults.enabled))

    mode = str(payload.get("mode", defaults.mode))
    if mode not in _ALLOWED_SCGR_MODES:
        raise ControllerConfigError(
            "controller.self_conditioned_gr.mode must be one of "
            f"{sorted(_ALLOWED_SCGR_MODES)} (got {mode!r}); 'beta_pressure' is "
            "unlocked later (PLAN_RF_SC_GR.md Task SC1.1)"
        )

    raw_arms = payload.get("arms", list(defaults.arms))
    if not isinstance(raw_arms, (list, tuple)):
        raise ControllerConfigError(
            "controller.self_conditioned_gr.arms must be a list "
            f"(got {raw_arms!r})"
        )
    arms = tuple(str(a) for a in raw_arms)
    if not arms:
        raise ControllerConfigError(
            "controller.self_conditioned_gr.arms must be non-empty"
        )
    bad_arms = [a for a in arms if a not in _ALLOWED_SCGR_ARMS]
    if bad_arms:
        raise ControllerConfigError(
            "controller.self_conditioned_gr.arms must be a subset of "
            f"{sorted(_ALLOWED_SCGR_ARMS)} (got {list(arms)!r})"
        )

    ensemble_size = int(payload.get("ensemble_size", defaults.ensemble_size))
    if ensemble_size < 1:
        raise ControllerConfigError(
            "controller.self_conditioned_gr.ensemble_size must be >= 1 "
            f"(got {ensemble_size})"
        )

    struct_temperature = float(
        payload.get("struct_temperature", defaults.struct_temperature)
    )
    if struct_temperature <= 0.0:
        raise ControllerConfigError(
            "controller.self_conditioned_gr.struct_temperature must be positive "
            f"(got {struct_temperature})"
        )

    confidence_source = str(
        payload.get("confidence_source", defaults.confidence_source)
    )
    if confidence_source not in _ALLOWED_SCGR_CONFIDENCE_SOURCES:
        raise ControllerConfigError(
            "controller.self_conditioned_gr.confidence_source must be one of "
            f"{sorted(_ALLOWED_SCGR_CONFIDENCE_SOURCES)} (got {confidence_source!r})"
        )

    confidence_threshold = float(
        payload.get("confidence_threshold", defaults.confidence_threshold)
    )
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ControllerConfigError(
            "controller.self_conditioned_gr.confidence_threshold must lie in "
            f"[0, 1] (got {confidence_threshold})"
        )

    update_state_from = str(
        payload.get("update_state_from", defaults.update_state_from)
    )
    if update_state_from not in _ALLOWED_SCGR_UPDATE_SOURCES:
        raise ControllerConfigError(
            "controller.self_conditioned_gr.update_state_from must be one of "
            f"{sorted(_ALLOWED_SCGR_UPDATE_SOURCES)} (got {update_state_from!r})"
        )

    probe_refresh_policy = str(
        payload.get("probe_refresh_policy", defaults.probe_refresh_policy)
    )
    if probe_refresh_policy not in _ALLOWED_SCGR_REFRESH_POLICIES:
        raise ControllerConfigError(
            "controller.self_conditioned_gr.probe_refresh_policy must be one of "
            f"{sorted(_ALLOWED_SCGR_REFRESH_POLICIES)} (got {probe_refresh_policy!r})"
        )

    top_m = int(payload.get("top_m", defaults.top_m))
    if top_m < 1:
        raise ControllerConfigError(
            f"controller.self_conditioned_gr.top_m must be >= 1 (got {top_m})"
        )

    lse_temperature = float(payload.get("lse_temperature", defaults.lse_temperature))
    if lse_temperature <= 0.0:
        raise ControllerConfigError(
            "controller.self_conditioned_gr.lse_temperature must be positive "
            f"(got {lse_temperature})"
        )

    raw_supra = payload.get("supra_tau_values", list(defaults.supra_tau_values))
    if not isinstance(raw_supra, (list, tuple)):
        raise ControllerConfigError(
            "controller.self_conditioned_gr.supra_tau_values must be a list "
            f"(got {raw_supra!r})"
        )
    supra_tau_values = tuple(float(v) for v in raw_supra)
    if not supra_tau_values:
        raise ControllerConfigError(
            "controller.self_conditioned_gr.supra_tau_values must be non-empty"
        )
    for v in supra_tau_values:
        if not math.isfinite(v) or v < 0.0:
            raise ControllerConfigError(
                "controller.self_conditioned_gr.supra_tau_values must be finite "
                f"and >= 0 (got {list(supra_tau_values)!r})"
            )

    actuation_aggregator = str(
        payload.get("actuation_aggregator", defaults.actuation_aggregator)
    )
    if actuation_aggregator not in _ALLOWED_SCGR_ACTUATION_AGGREGATORS:
        raise ControllerConfigError(
            "controller.self_conditioned_gr.actuation_aggregator must be one of "
            f"{sorted(_ALLOWED_SCGR_ACTUATION_AGGREGATORS)} "
            f"(got {actuation_aggregator!r})"
        )
    # supra_mass actuation needs an unambiguous tau (PLAN_RF_SC_GR.md SC1.1).
    if actuation_aggregator == "supra_mass" and len(supra_tau_values) != 1:
        raise ControllerConfigError(
            "controller.self_conditioned_gr.actuation_aggregator='supra_mass' "
            "requires exactly one supra_tau_values element so the tau is "
            f"unambiguous (got {list(supra_tau_values)!r})"
        )

    actuation_arm = str(payload.get("actuation_arm", defaults.actuation_arm))
    if actuation_arm not in _ALLOWED_SCGR_ARMS:
        raise ControllerConfigError(
            "controller.self_conditioned_gr.actuation_arm must be one of "
            f"{sorted(_ALLOWED_SCGR_ARMS)} (got {actuation_arm!r})"
        )
    # You cannot actuate on an arm that is not probed.
    if actuation_arm not in arms:
        raise ControllerConfigError(
            "controller.self_conditioned_gr.actuation_arm must be one of the probed "
            f"arms {list(arms)!r} (got {actuation_arm!r})"
        )

    freeze_after_reliable_refreshes = int(
        payload.get(
            "freeze_after_reliable_refreshes",
            defaults.freeze_after_reliable_refreshes,
        )
    )
    if freeze_after_reliable_refreshes < 1:
        raise ControllerConfigError(
            "controller.self_conditioned_gr.freeze_after_reliable_refreshes must "
            f"be >= 1 (got {freeze_after_reliable_refreshes})"
        )

    actuation_reduce = str(payload.get("actuation_reduce", defaults.actuation_reduce))
    if actuation_reduce not in _ALLOWED_SCGR_ACTUATION_REDUCE:
        raise ControllerConfigError(
            "controller.self_conditioned_gr.actuation_reduce must be one of "
            f"{sorted(_ALLOWED_SCGR_ACTUATION_REDUCE)} (got {actuation_reduce!r})"
        )

    return SelfConditionedGRConfig(
        enabled=enabled,
        mode=mode,
        arms=arms,
        ensemble_size=ensemble_size,
        struct_temperature=struct_temperature,
        confidence_source=confidence_source,
        confidence_threshold=confidence_threshold,
        update_state_from=update_state_from,
        probe_refresh_policy=probe_refresh_policy,
        top_m=top_m,
        lse_temperature=lse_temperature,
        supra_tau_values=supra_tau_values,
        write_probe_telemetry=bool(
            payload.get("write_probe_telemetry", defaults.write_probe_telemetry)
        ),
        write_residue_telemetry=bool(
            payload.get("write_residue_telemetry", defaults.write_residue_telemetry)
        ),
        actuation_aggregator=actuation_aggregator,
        actuation_arm=actuation_arm,
        freeze_after_reliable_refreshes=freeze_after_reliable_refreshes,
        actuation_reduce=actuation_reduce,
    )


def _cross_validate_scgr(
    *,
    self_conditioned_gr: SelfConditionedGRConfig,
    global_pressure: GlobalPressureConfig,
) -> None:
    """Enforce the SC-GR mode/actuation contract (PLAN_RF_SC_GR.md SC0.1 / SC1.1).

    ``monitor_only`` is a pure sensing layer: it must not run while the
    ``global_pressure`` actuator is on, so the probe telemetry is never confounded
    by an active gain. ``beta_pressure`` (SC1) instead *requires* the actuator,
    fed by the SC-GR probe, scaling beta only (``scale_lambda=false``, RAR 0008)
    and never amplifying (``g_max==1.0``; amplify is SC2).
    """
    scgr = self_conditioned_gr
    # Reverse constraint (both directions must agree): whenever the actuator
    # consumes the SC-GR probe, the probe MUST be the active producer of B_sc.
    # Otherwise _scgr_actuation_B_sc stays None at runtime and g_GR silently
    # degrades to unready_g (the "SC pressure" experiment becomes a no-op or
    # mis-scales). Checked independent of scgr.enabled so a probe-disabled config
    # with pressure_source=self_conditioned_probe fails fast.
    if (
        global_pressure.enabled
        and global_pressure.pressure_source == "self_conditioned_probe"
        and not (scgr.enabled and scgr.mode == "beta_pressure")
    ):
        raise ControllerConfigError(
            "controller.global_pressure.pressure_source='self_conditioned_probe' "
            "requires controller.self_conditioned_gr.enabled=true and "
            "mode='beta_pressure' (the probe must produce B_sc; otherwise g_GR "
            "silently degrades to unready_g)"
        )
    if not scgr.enabled:
        return
    if scgr.mode == "monitor_only":
        if global_pressure.enabled:
            raise ControllerConfigError(
                "controller.self_conditioned_gr.mode='monitor_only' requires "
                "controller.global_pressure.enabled=false; the monitor probe must "
                "not run alongside an active pressure gate (PLAN_RF_SC_GR.md SC0.1)"
            )
    elif scgr.mode == "beta_pressure":
        if not global_pressure.enabled:
            raise ControllerConfigError(
                "controller.self_conditioned_gr.mode='beta_pressure' requires "
                "controller.global_pressure.enabled=true (SC1 gates beta by g_GR)"
            )
        if global_pressure.pressure_source != "self_conditioned_probe":
            raise ControllerConfigError(
                "controller.self_conditioned_gr.mode='beta_pressure' requires "
                "controller.global_pressure.pressure_source='self_conditioned_probe' "
                f"(got {global_pressure.pressure_source!r})"
            )
        if global_pressure.scale_lambda:
            raise ControllerConfigError(
                "controller.self_conditioned_gr.mode='beta_pressure' requires "
                "controller.global_pressure.scale_lambda=false (RAR 0008: scaling "
                "lambda reverts suppressed designs and loses D3's commit)"
            )
        # SC2.1: g_max>1 (amplify-high) is allowed ONLY when amplify=true, so a
        # coder cannot silently amplify (SC1 stays g_max==1.0). The amplify
        # bounds (g_min<1<g_max<=3) are validated in _materialize_global_pressure.
        if not global_pressure.amplify and global_pressure.g_max != 1.0:
            raise ControllerConfigError(
                "controller.self_conditioned_gr.mode='beta_pressure' requires "
                "controller.global_pressure.g_max=1.0 unless "
                "global_pressure.amplify=true (amplify g>1 is SC2); "
                f"got g_max={global_pressure.g_max}"
            )


def _cross_validate_allocation(
    *,
    targeting: TargetingConfig,
    allocation: AllocationConfig,
    self_conditioned_gr: SelfConditionedGRConfig,
    global_pressure: GlobalPressureConfig,
) -> None:
    """Enforce the allocation-layer contract (PLAN_PLANNER_SC_GR.md §2 / Task 4-5).

    ``Φ_i`` is derived from the SC-GR probe's per-residue ``residue_excess`` and is
    frozen INSIDE the SC-GR pressure path (``_scgr_frozen_pressure``). So a config
    that asks for ``v_target_x_alloc`` but cannot actually form/freeze ``Φ_i`` would
    silently degrade the treatment arm to the raw ``v_target`` control — a circular
    no-op that looks like a valid experiment. These checks turn every such
    degeneration into a config-time fail-fast (no silent control collapse).
    """
    scgr = self_conditioned_gr
    gp = global_pressure
    if allocation.enabled:
        # Φ_i comes from the probe; an arm that is never probed yields an empty
        # r_i and Φ_i never freezes.
        if not scgr.enabled:
            raise ControllerConfigError(
                "controller.allocation.enabled=true requires "
                "controller.self_conditioned_gr.enabled=true (Φ_i is derived from "
                "the SC-GR probe's per-residue residue_excess)"
            )
        if allocation.arm not in scgr.arms:
            raise ControllerConfigError(
                f"controller.allocation.arm={allocation.arm!r} must be one of the "
                f"probed controller.self_conditioned_gr.arms {list(scgr.arms)!r} "
                "(Φ_i reduces over that arm's residue_excess; an unprobed arm "
                "yields an empty r_i)"
            )
    mode = targeting.selection_field_mode
    if mode in _PHI_CONSUMING_SELECTION_MODES:
        if not allocation.enabled:
            raise ControllerConfigError(
                f"controller.targeting.selection_field_mode={mode!r} "
                "requires controller.allocation.enabled=true; otherwise Φ_i never "
                "forms and the selection field silently degrades to the raw "
                "v_target control (the treatment arm becomes its own control)"
            )
        if not (
            gp.enabled and gp.pressure_source == "self_conditioned_probe"
        ):
            raise ControllerConfigError(
                f"controller.targeting.selection_field_mode={mode!r} "
                "requires controller.global_pressure.enabled=true with "
                "pressure_source='self_conditioned_probe' (Φ_i is frozen inside the "
                "SC-GR pressure path; without it Φ_i never freezes and the arm "
                "degrades to v_target)"
            )


def _cross_validate_mode(*, mode: str, d2: D2Config, d3: D3Config) -> None:
    """Enforce mode-vs-enabled cross constraints (PLAN_RF.md §D2-D3 rules 3-5)."""

    if mode in _MODES_REQUIRING_D2 and not d2.enabled:
        raise ControllerConfigError(
            f"controller.mode={mode!r} requires controller.d2.enabled=true"
        )
    if mode in _MODES_REQUIRING_D3 and not d3.enabled:
        raise ControllerConfigError(
            f"controller.mode={mode!r} requires controller.d3.enabled=true"
        )
    if mode == "d2_logits" and d3.enabled:
        raise ControllerConfigError(
            "controller.mode='d2_logits' requires controller.d3.enabled=false"
        )
    if mode == "d3_revisit" and d2.enabled:
        raise ControllerConfigError(
            "controller.mode='d3_revisit' requires controller.d2.enabled=false"
        )


def controller_config_to_dict(config: ControllerConfig) -> dict[str, Any]:
    return asdict(config)


def controller_config_hash(config: ControllerConfig) -> str:
    """Deterministic SHA-256 over the serialized config.

    Used to stamp manifest provenance so resumed runs can detect controller
    config drift independently of head/checkpoint hashes.
    """
    payload = json.dumps(controller_config_to_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ControllerConfigError(f"controller.{key} section is required")
    return value
