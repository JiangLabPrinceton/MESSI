"""Stage B typed-actionability pure operators (PLAN_RF_UNI_CTRL.md §"Mathematical
Definitions", Tasks B1/B3).

This module holds the residue-level ``A_i(t)`` field math as pure functions:
max-covering window projection, the anchored SoftOR union, the
proposal-envelope ``peak * consistency`` aggregation, the fresh-evidence /
memory firewall, the target-vs-pressure split, and the global-pressure mass.

Design constraints carried from the PLAN:

* Projection is **max-covering window** (not the head's internal LME), to keep
  focal hotspots visible to D2 targeting.
* ``SoftOR`` is anchored so ``SoftOR(0,...) = 0`` and ``SoftOR(x, 0,...) = x``;
  any single positive channel can make a residue actionable, but the union
  grows sublinearly versus the arithmetic sum.
* The memory firewall is enforced by signature: ``update_memory`` accepts only
  ``previous_b_mem``, ``e_fresh``, and ``half_life_refreshes``. It never accepts
  ``v_target``, ``u_pressure``, or D3 ``m_i``.

All residue arrays are shape ``(L,)`` with ``L`` the protein length.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Projection + SoftOR (Task B1)
# ---------------------------------------------------------------------------


def max_covering_window_projection(
    *,
    length: int,
    windows: Sequence[Mapping[str, float | int]],
) -> np.ndarray:
    """Per residue, the max ``score`` over windows whose ``[start, end)`` covers it.

    Residues covered by no window are 0.0. This is the PLAN's
    ``Proj_i(z_w) = max_{w: i in span(w)} z_w`` used for ``b_cur`` / ``b_env``;
    it deliberately differs from the head's window→residue LME projection so
    that a single focal high-risk window is not averaged away.
    """
    values = np.full(int(length), -np.inf, dtype=float)
    for window in windows:
        start = int(window["start"])
        end = int(window["end"])
        score = float(window["score"])
        values[start:end] = np.maximum(values[start:end], score)
    values[~np.isfinite(values)] = 0.0
    return values


def soft_or(values: Iterable[float], *, tau: float) -> float:
    """Anchored soft union ``tau * log(1 + sum_k (exp(x_k/tau) - 1))``.

    Properties (asserted by tests):
    ``SoftOR(0,...) = 0``; ``SoftOR(x, 0,...) = x``; ``SoftOR(x, y) >= max(x, y)``;
    sublinear versus ``x + y``. Expects nonnegative evidence.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    if np.any(arr < -1e-12):
        raise ValueError("soft_or expects nonnegative evidence values")
    if float(np.max(arr)) == 0.0:
        return 0.0
    return float(tau * np.log1p(np.sum(np.expm1(arr / tau))))


def _soft_or_stack(channels: Sequence[np.ndarray], *, tau: float) -> np.ndarray:
    """Per-residue anchored SoftOR across a list of ``(L,)`` channel arrays.

    Because ``expm1(0) = 0`` and ``log1p(0) = 0``, the anchoring properties of
    :func:`soft_or` hold elementwise without a special-case branch:
    ``SoftOR(0,...) = 0`` and ``SoftOR(x, 0,...) = x``. SoftOR is associative,
    so stacking ``[SoftOR(a, b), c]`` equals ``SoftOR(a, b, c)``.
    """
    stack = np.stack([np.asarray(c, dtype=float) for c in channels], axis=0)
    if np.any(stack < -1e-12):
        raise ValueError("soft_or expects nonnegative evidence values")
    return (tau * np.log1p(np.sum(np.expm1(stack / float(tau)), axis=0))).astype(
        np.float64, copy=False
    )


# ---------------------------------------------------------------------------
# Field functions (Task B3)
# ---------------------------------------------------------------------------


def positive_evidence_weight(r_ctx: np.ndarray, *, floor: float) -> np.ndarray:
    """``w_pos(r) = floor + (1 - floor) * r`` — a reliability weight with a floor.

    The floor keeps unreliable context from erasing early positive current
    evidence (PLAN_RF_UNI_CTRL.md "Context Reliability").
    """
    r = np.asarray(r_ctx, dtype=float)
    return float(floor) + (1.0 - float(floor)) * r


def excess_over_tau(values: np.ndarray, *, tau_ref: float) -> np.ndarray:
    """``max(0, values - tau_ref)`` — positive excess above the scalar background."""
    return np.maximum(0.0, np.asarray(values, dtype=float) - float(tau_ref))


def envelope_burden_from_excess_samples(
    samples: np.ndarray, *, consistency_floor: float
) -> np.ndarray:
    """``peak * (c_floor + (1 - c_floor) * consistency)`` over envelope samples.

    ``samples`` has shape ``(K_env, L)`` holding ``e_env_{i,s}`` (already excess
    over ``tau_ref_B``). ``peak_i = max_s`` and ``consistency_i`` is the fraction
    of envelope samples with positive excess. With ``K_env = 3`` this is the
    PLAN's max*consistency surrogate for CVaR (PLAN "Proposal-Envelope Burden").
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"samples must be 2D (K_env, L), got shape {arr.shape}")
    peak = arr.max(axis=0)
    consistency = (arr > 0.0).mean(axis=0)
    floor = float(consistency_floor)
    return peak * (floor + (1.0 - floor) * consistency)


def compute_fresh_evidence(
    *,
    b_cur: np.ndarray,
    b_env: np.ndarray,
    r_ctx: np.ndarray,
    r_ctx_floor: float,
    tau: float,
) -> np.ndarray:
    """``e_fresh = SoftOR(w_pos(r_ctx) * b_cur, b_env)`` — memory excluded.

    Fresh evidence is the only thing allowed to update ``b_mem``; it must never
    read memory (PLAN "Fresh Evidence And Memory Firewall").
    """
    weight = positive_evidence_weight(r_ctx, floor=r_ctx_floor)
    weighted_cur = weight * np.asarray(b_cur, dtype=float)
    return _soft_or_stack([weighted_cur, np.asarray(b_env, dtype=float)], tau=tau)


def update_memory(
    *,
    previous_b_mem: np.ndarray | None,
    e_fresh: np.ndarray,
    half_life_refreshes: float,
) -> np.ndarray:
    """EMA memory ``b_mem = gamma * prev + (1 - gamma) * e_fresh``.

    ``gamma = 2 ** (-1 / H)`` so memory halves after ``H`` zero-evidence
    refreshes. Cold start (``previous_b_mem is None``) begins from zero
    (``mem_cold_start = zero``). By signature this function can only consume
    fresh evidence — never ``v_target``, ``u_pressure``, or D3 ``m_i`` — which
    enforces the memory firewall at the call site.
    """
    e = np.asarray(e_fresh, dtype=float)
    gamma = 2.0 ** (-1.0 / float(half_life_refreshes))
    prev = np.zeros_like(e) if previous_b_mem is None else np.asarray(
        previous_b_mem, dtype=float
    )
    return gamma * prev + (1.0 - gamma) * e


def compute_target_evidence(
    *,
    fresh_evidence: np.ndarray,
    b_mem: np.ndarray,
    tau: float,
) -> np.ndarray:
    """``v_target = SoftOR(e_fresh, b_mem)`` — the unclustered targeting field.

    By SoftOR associativity this equals
    ``SoftOR(w_pos(r_ctx) * b_cur, b_env, b_mem)``. ``v_target`` drives
    active-block discovery; cluster support is applied only for global pressure.
    """
    return _soft_or_stack(
        [np.asarray(fresh_evidence, dtype=float), np.asarray(b_mem, dtype=float)],
        tau=tau,
    )


def cluster_support_multiplier(
    v_target: np.ndarray,
    *,
    tau: float,
    radius: int,
    min_mass: float,
) -> np.ndarray:
    """Soft cluster support ``support_i = min(1, (sum_{|j-i|<=radius} a_j)/min_mass)``.

    ``a_i = v_i / (v_i + tau)`` is a bounded per-residue activation. Returned
    before the floor is applied, so it is logged directly as the
    ``cluster_support`` telemetry column.
    """
    v = np.asarray(v_target, dtype=float)
    a = v / (v + float(tau))
    L = v.shape[0]
    r = int(radius)
    mm = float(min_mass)
    idx = np.arange(L)
    lo = np.maximum(0, idx - r)
    hi = np.minimum(L, idx + r + 1)
    prefix = np.concatenate(([0.0], np.cumsum(a, dtype=float)))
    mass = prefix[hi] - prefix[lo]
    return np.minimum(1.0, mass / mm).astype(float, copy=False)


def cluster_supported_pressure(
    *,
    v_target: np.ndarray,
    tau: float,
    radius: int,
    min_mass: float,
    floor: float,
) -> np.ndarray:
    """``u_pressure = v_target * (floor + (1 - floor) * support)``.

    Cluster support down-weights isolated spikes for the global pressure
    aggregate but is never applied to ``v_target`` itself, so a focal hotspot
    stays fully visible to active-block discovery (PLAN "Targeting Versus
    Pressure").
    """
    v = np.asarray(v_target, dtype=float)
    support = cluster_support_multiplier(
        v, tau=tau, radius=radius, min_mass=min_mass
    )
    return v * (float(floor) + (1.0 - float(floor)) * support)


def global_pressure_mass(u_pressure: np.ndarray) -> float:
    """``G(t) = (1/L) sum_i u_pressure_i`` — length-normalized thresholded mass.

    Division by length is required so global pressure does not scale with
    protein length (PLAN "Global Pressure").
    """
    arr = np.asarray(u_pressure, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def prominence_thresholded_mass(u_pressure: np.ndarray, *, tau_prom: float) -> float:
    """``G_step = (1/L) Σ_i ReLU(u_pressure_i − τ_prom)`` — prominence-thresholded mass.

    The Stage C.1 pressure driver (PLAN_RF_UNI_CTRL.md G3 / RAR 0006 M11). The
    dense-field mean (:func:`global_pressure_mass`) is floor-dominated and does
    not discriminate burden (the typed field is positive at ~100% of residues),
    so a second prominence cut ``τ_prom`` keeps only the prominent excess.

    ``τ_prom`` is **NOT** a re-subtraction of ``tau_ref_B`` (which was applied
    once at field formation, when ``b_cur``/``b_env`` were taken as excess over
    ``tau_ref_B``). ``u_pressure`` is the downstream aggregated field (SoftOR →
    memory EMA → cluster scaling), already nonnegative; ``τ_prom`` is a separate
    prominence cut that selects which portions of that field count as actuator
    burden. ``τ_prom`` is calibrated from the typed pilot. With ``τ_prom = 0``
    this reduces exactly to :func:`global_pressure_mass` for the nonnegative
    ``u_pressure``, so a Stage B run (no calibration ⇒ ``τ_prom=0``) keeps its
    diagnostic ``G`` unchanged.
    """
    arr = np.asarray(u_pressure, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.maximum(0.0, arr - float(tau_prom)).mean())


def global_pressure_scalar(
    G: float,
    *,
    g_min: float,
    g_max: float,
    G0: float,
    s_G: float,
) -> float:
    """Bounded pressure scalar ``g_min + (g_max - g_min) * sigmoid((G - G0)/s_G)``.

    This is the Stage B **diagnostic-only** scalar for a single per-refresh
    ``G`` (logged, never used to scale beta/lambda). Stage C.1 uses the separate
    trajectory-level :func:`smoothstep_pressure` actuator over ``B_GR`` instead.
    Because no Stage B pilot calibration is available before C.1, callers pass
    ``G0=0.0`` and ``s_G=1.0`` for the raw-sigmoid diagnostic.
    """
    z = (float(G) - float(G0)) / float(s_G)
    sigmoid = 1.0 / (1.0 + math.exp(-z))
    return float(g_min) + (float(g_max) - float(g_min)) * sigmoid


def smoothstep_pressure(
    B_GR: float,
    *,
    B_low: float,
    B_high: float,
    g_min: float,
    g_max: float,
) -> float:
    """Clipped cubic-smoothstep pressure mapping (PLAN_RF_UNI_CTRL.md Task C1.1).

    ::

        q = clip((B_GR - B_low) / (B_high - B_low), 0, 1)
        s = 3 q^2 - 2 q^3
        g = g_min + (g_max - g_min) * s

    This is the Stage C.1 **actuator** mapping; it is distinct from
    :func:`global_pressure_scalar` (the Stage B sigmoid *diagnostic* of a single
    per-refresh ``G``). The smoothstep hard-saturates: ``B_GR <= B_low`` returns
    exactly ``g_min`` and ``B_GR >= B_high`` returns exactly ``g_max``. That hard
    floor is what lets low-burden trajectories reach ``g_min = 0`` in the primary
    C.1 config (PLAN_RF_UNI_CTRL.md §"Stage C.1 Runtime Definitions"), which the
    sigmoid's asymptote cannot.

    ``B_high`` must be strictly greater than ``B_low``; config loading already
    fails fast on this, but the pure function guards it too.
    """
    denom = float(B_high) - float(B_low)
    if denom <= 0.0:
        raise ValueError(
            "smoothstep_pressure requires B_high > B_low "
            f"(got B_low={B_low}, B_high={B_high})"
        )
    q = (float(B_GR) - float(B_low)) / denom
    q = min(1.0, max(0.0, q))
    s = q * q * (3.0 - 2.0 * q)
    return float(g_min) + (float(g_max) - float(g_min)) * s


def protein_pressure_burden(G_values: list[float]) -> float:
    """Trajectory pressure burden ``B_GR`` = median of finite per-refresh ``G``.

    Uses the **median** (not the mean or the latest value) so a single
    high-``G`` refresh spike cannot dominate the per-design actuation
    (PLAN_RF_UNI_CTRL.md G2: "Actuation reads a stable per-design trajectory
    pressure state ``B_GR``"). Non-finite entries are dropped; an empty or
    all-non-finite input returns ``0.0`` (no observed burden -> ``g_min`` after
    the smoothstep). The controller only feeds reliable (finite) ``G`` values
    here, so the finite filter is a defensive guard.
    """
    arr = np.asarray(list(G_values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.median(arr))
