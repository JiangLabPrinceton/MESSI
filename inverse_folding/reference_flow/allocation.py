"""Allocation layer — per-residue editability mass ``Φ_i`` (PLAN_PLANNER_SC_GR.md §2).

DISTINCT from the typed-actionability field ``A_i(t)`` in ``actionability.py``:
``Φ_i`` is a mean-normalized editability mass derived from the SC-GR prospective
risk ``r_i``; it only *reweights* the actionability field's ``v_target`` at the
selection seams. It never replaces ``A_i(t)``, never touches pressure/memory, and
never enters the denoising schedule. All arrays are shape ``(L,)``.
"""
from __future__ import annotations

import hashlib

import numpy as np


def smooth_window_max(values: np.ndarray, *, half_width: int) -> np.ndarray:
    """Per residue, max over ``[i - half_width, i + half_width]`` (register grain,
    RAR 0020 9-residue window ⇒ ``half_width = 4``)."""
    v = np.asarray(values, dtype=float)
    L = int(v.shape[0])
    r = int(half_width)
    out = np.zeros(L, dtype=float)
    for i in range(L):
        lo, hi = max(0, i - r), min(L, i + r + 1)
        out[i] = float(v[lo:hi].max()) if hi > lo else 0.0
    return out


def allocation_mass(r_i: np.ndarray, *, half_width: int) -> np.ndarray:
    """Mean-normalized editability mass ``Φ_i`` from per-residue risk ``r_i``.

    Smooth to register grain, normalize over the protein to **mean 1** (high-risk
    regions ⇒ ``Φ_i > 1``). All-zero ``r_i`` ⇒ uniform ``Φ_i ≡ 1`` (degrade to the
    ``v_target`` baseline).
    """
    smoothed = smooth_window_max(np.asarray(r_i, dtype=float), half_width=half_width)
    L = int(smoothed.shape[0])
    total = float(smoothed.sum())
    if L == 0:
        return smoothed
    if total <= 0.0:
        return np.ones(L, dtype=float)
    return smoothed * (L / total)


def reweight_by_allocation(
    v_target: np.ndarray, allocation: np.ndarray, *, c: float, eps: float
) -> np.ndarray:
    """Selection field ``s_alloc_i = v_target_i * (eps + c * Φ_i)``.

    Uniform ``Φ_i ≡ 1`` ⇒ constant rescale of ``v_target`` (preserves baseline
    selection order, the H1 uniform control); tilted ``Φ_i`` reorders toward
    high prospective-immune regions. Reweights the SELECTION field only — never
    ``v_target`` itself (pressure stays on raw ``v_target``) and never memory.
    """
    v = np.asarray(v_target, dtype=float)
    phi = np.asarray(allocation, dtype=float)
    return v * (float(eps) + float(c) * phi)


def stable_seed(*parts: object) -> int:
    """Process-stable 64-bit seed from arbitrary parts via sha256.

    Reproducible across Della jobs — unlike the built-in ``hash()`` (salted by
    ``PYTHONHASHSEED``). Mirrors the ``_rng_for`` idiom (self_conditioned_gr.py).
    """
    key = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


# ---------------------------------------------------------------------------
# §A1 — revised r_i consumption: triage / tie-break within the eligible set
# (PLAN_PLANNER_SC_GR.md §A1; doc/Self-Cond_GR.md §8.4 Path A)
# ---------------------------------------------------------------------------


def _norm_rank(x: np.ndarray) -> np.ndarray:
    """Tie-aware fractional ranks in (0, 1): equal values share the MEAN of their
    sorted positions, so ties are NOT broken by original index. Empty ⇒ empty.

    This is load-bearing for :func:`triage_field` / :func:`terminal_union_field`:
    at tied (e.g. zero / low) ``v_target`` — A3's core promotion regime — an
    index-broken rank would span the full (0, 1) and swamp the bounded ``Φ_i``
    nudge, so the *position* would decide selection instead of the terminal signal
    (review P1). With a shared rank on ties, the additive ``Φ_i`` term becomes the
    real tie-break. Distinct values are evenly spaced — identical to a plain
    argsort rank, so all-distinct callers are unaffected.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n == 0:
        return x.astype(float)
    order = np.argsort(x, kind="stable")
    sx = x[order]
    _uniq, inv, counts = np.unique(sx, return_inverse=True, return_counts=True)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    avg_pos = (starts + (counts - 1) / 2.0)[inv]  # mean 0-indexed position per tie group
    result = np.empty(n, dtype=float)
    result[order] = (avg_pos + 0.5) / n
    return result


def triage_field(
    v_target: np.ndarray,
    allocation: np.ndarray,
    *,
    eligible_quantile: float,
    triage_lambda: float,
    floor: float = 0.0,
) -> np.ndarray:
    """Path-A triage (doc §8.4): within the v_target-eligible set, rank by
    ``norm_rank(v_target) + triage_lambda * norm_rank(Φ_i)``; ineligible ⇒ 0.

    ``v_target`` keeps coefficient 1 and dominates; ``Φ_i`` is a bounded
    (``triage_lambda``) tie-break/nudge.

    **Strict eligibility (doc §8.4 — load-bearing).** Eligibility is the
    top-(1−eligible_quantile) AMONG positions with **strictly-positive
    actionability** ``v_target > floor`` (``floor`` = the pipeline's
    ``active_window_min_excess``, default 0). The quantile is taken over the
    POSITIVE values only, so a sparse ``v_target`` whose median is 0 (or any
    non-positive site) can NEVER become eligible — this would otherwise re-admit
    exactly the low-actionability sites whose edits did not convert (RAR 0019 M4)
    and re-create the ``v_target·Φ`` failure (the multiplicative reweight's harm).
    All-``v_target``-non-positive ⇒ nothing eligible ⇒ all-zero field. ``Φ_i ≡ 1``
    (or ``triage_lambda`` 0) ⇒ pure ``v_target`` order within eligible.
    """
    v = np.asarray(v_target, dtype=float)
    phi = np.asarray(allocation, dtype=float)
    L = v.shape[0]
    if L == 0:
        return v
    out = np.zeros(L, dtype=float)
    positive = v > float(floor)
    if not positive.any():
        return out  # no genuinely-actionable site ⇒ select nothing (no τ_v widening)
    thr = float(np.quantile(v[positive], eligible_quantile))
    eligible = positive & (v >= thr)
    idx = np.where(eligible)[0]
    if idx.size:
        out[idx] = _norm_rank(v[idx]) + float(triage_lambda) * _norm_rank(phi[idx])
    return out


def terminal_union_field(
    v_target: np.ndarray,
    allocation: np.ndarray,
    *,
    terminal_eligible_quantile: float,
    terminal_lambda: float,
    floor: float = 0.0,
) -> np.ndarray:
    """A3 terminal-aware active-register targeting (doc §8.4 Fork A): rank the
    UNION of the locally-actionable and the terminally-hot registers by
    ``norm_rank(v_target) + terminal_lambda * norm_rank(Φ_i)``; ineligible ⇒ 0.

    Eligibility is ``local | terminal``: ``local`` = strictly-positive
    actionability (``v_target > floor``, ``floor = active_window_min_excess``);
    ``terminal`` = the register-smoothed ``Φ_i`` in the top-(1−``terminal_eligible_quantile``)
    by ``Φ``. Over the union, ``v_target`` keeps coefficient 1 and dominates;
    ``Φ_i`` is a bounded (``terminal_lambda``) additive rank nudge.

    **The principled inverse of :func:`triage_field` (A1 — load-bearing).**
    Triage is a WITHIN-eligible tie-break: a low-``v_target`` site can never
    activate. A3 PROMOTES high-``Φ`` low-``v_target`` registers INTO the active
    set through a SEPARATE terminal gate — NOT by lowering the ``v_target`` bar
    (widening ``τ_v``). C0a M4 found those registers are skipped for low *local*
    head evidence (``b_env``), not structure, so the terminal channel supplies the
    actionability evidence the local term misses. Because ``Φ`` is the frozen,
    register-smoothed allocation mass, both the terminal quantile and the
    window-max selection unit are register-grain.

    **Terminal-gate guards (load-bearing).** The gate quantile is taken over the
    **positive (discriminative) ``Φ`` subset only** — mirroring :func:`triage_field`'s
    ``np.quantile(v[positive], …)`` — so it can never widen to dead (``v_target ≤
    floor``) sites and re-create the falsified ``v_target·Φ`` harm (RAR 0019 M4 /
    §8.2.1). Two degenerate regimes both collapse a full-array quantile and must be
    excluded: (i) the realistic **sparse** ``Φ`` (``allocation_mass`` window-max-
    smooths a clipped, mostly-zero ``r_i``, so ``Φ`` is 0 across most of the
    protein) — a full-array ``quantile`` lands in the zero baseline whenever the hot
    region is smaller than the ``(1−q)`` tail and ``Φ ≥ 0`` admits everyone;
    (ii) the **uniform** ``Φ`` (an all-zero ``r_i`` makes ``allocation_mass`` return
    ones) with no discriminative signal. Restricting to positives handles (i); the
    ``max > min`` flat guard handles (ii). In both, the terminal set is empty and A3
    admits nothing beyond the local set.
    """
    v = np.asarray(v_target, dtype=float)
    phi = np.asarray(allocation, dtype=float)
    L = v.shape[0]
    if L == 0:
        return v
    out = np.zeros(L, dtype=float)
    local = v > float(floor)
    # Terminal eligibility = top-(1−q) AMONG strictly-positive (discriminative) Φ
    # only (mirror triage_field's positive-subset quantile). A full-array quantile
    # over the realistic SPARSE Φ would collapse to the zero baseline and admit
    # every dead site — the τ_v-widening §8.2.1 / RAR 0019 M4 forbid. The flat
    # guard additionally excludes a uniform Φ (all-zero r_i ⇒ allocation_mass ones).
    phi_positive = phi > 0.0
    if phi_positive.any() and float(np.nanmax(phi)) > float(np.nanmin(phi)):
        thr = float(np.quantile(phi[phi_positive], terminal_eligible_quantile))
        terminal = phi_positive & (phi >= thr)
    else:
        terminal = np.zeros(L, dtype=bool)
    eligible = local | terminal
    idx = np.where(eligible)[0]
    if idx.size:
        out[idx] = _norm_rank(v[idx]) + float(terminal_lambda) * _norm_rank(phi[idx])
    return out
