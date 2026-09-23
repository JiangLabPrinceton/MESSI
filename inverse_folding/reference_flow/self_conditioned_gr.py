"""Self-Conditioned GR monitor probe — pure helpers (PLAN_RF_SC_GR.md Task SC0.2).

SC-GR is a monitor-only sensing layer (``doc/Self-Cond_GR.md``). At each refresh
it builds ``K`` one-shot pseudo-terminal completions pre-D2 from the structural
logits, scores them with the frozen head, and reduces the head's burden map to
trajectory-level estimators. These helpers are deterministic and free of any
controller / D2 / D3 state:

* ``build_probe_samples`` fills masked positions of the two arms. The ``fresh``
  arm samples every masked position from the canonical-AA structural softmax; the
  ``self_conditioned`` arm reuses the previous refresh's structural-argmax token
  at positions whose previous canonical-softmax confidence cleared the threshold,
  and samples the rest. Committed positions are always fixed.
* ``update_state_from_structural_argmax`` produces the carried state from the
  deterministic structural argmax + canonical-softmax confidence — never from
  head scores or either arm's stochastic samples (firewall by signature).
* ``compute_risk_aggregates`` projects one completion's head windows to a
  per-residue excess map (the full-sequence ``b_cur`` scope) and reduces it with
  the three monitored amplify-aligned aggregators.
* ``summarize_probe_refresh`` reduces the ``K`` per-arm completions to the
  per-refresh telemetry row (robust ``median`` for ``B_sc``; ``max`` / ``std``
  for the over-K instability diagnostic).

Reuses the projection primitives from :mod:`actionability` so the SC-GR residue
map is identical to ``b_cur`` up to the ensemble + aggregator differences.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from inverse_folding.reference_flow.actionability import (
    excess_over_tau,
    max_covering_window_projection,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SCGRState:
    """Detached self-conditioned state carried across refreshes (per design).

    ``prev_x1_hat`` is the previous refresh's deterministic structural-argmax
    terminal estimate (tokens, shape ``(L,)``); ``prev_confidence`` is the
    previous refresh's per-position canonical-softmax max-probability. Neither is
    head-scored nor drawn from any arm's samples (``doc/Self-Cond_GR.md`` §3).
    """

    prev_x1_hat: np.ndarray
    prev_confidence: np.ndarray


@dataclass(frozen=True)
class SCGRProbeSample:
    """One pseudo-terminal completion (pre-score) with its sampling metadata."""

    arm: str
    sample_idx: int
    tokens: np.ndarray
    num_masked: int
    num_reused_from_prev: int
    reuse_fraction: float
    mean_prev_confidence_reused: float
    mean_sample_entropy: float
    state_bootstrap_flag: bool


@dataclass(frozen=True)
class SCGRRiskAggregates:
    """Trajectory-level burden aggregates for one scored completion.

    ``residue_excess`` is the per-residue map ``max(0, Proj_i(z) - tau_ref_B)``
    the scalar aggregators reduce (the full-sequence ``b_cur`` scope). It is
    surfaced so the per-residue ``r_i`` signal can be persisted for sub-protein
    targeting validation; the scalar fields are unchanged reductions of it.
    """

    G_mean_excess: float
    G_topm_lse: float
    supra_masses: dict[float, float]
    head_risk_LME: float
    head_risk_max: float
    residue_excess: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )


# ---------------------------------------------------------------------------
# Internal numeric helpers
# ---------------------------------------------------------------------------


def _rng_for(rng_key: Sequence[object], arm: str, sample_idx: int) -> np.random.Generator:
    """Deterministic per-sample RNG seeded from protein/design/seed/refresh/arm/idx.

    Uses a dedicated hash-derived seed so the probe never touches the sampler's
    RNG stream (the firewall must not perturb generation).
    """
    parts = list(rng_key) + [arm, sample_idx]
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _canonical_softmax(logits_row: np.ndarray, canonical_ids: np.ndarray, *, temperature: float) -> np.ndarray:
    z = np.asarray(logits_row, dtype=float)[canonical_ids] / float(temperature)
    z = z - z.max()
    e = np.exp(z)
    s = e.sum()
    if s <= 0.0:
        return np.full(canonical_ids.shape[0], 1.0 / canonical_ids.shape[0])
    return e / s


def _entropy(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    nz = p > 0.0
    return float(-np.sum(p[nz] * np.log(p[nz])))


def _logmeanexp(values: np.ndarray, *, temperature: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    T = float(temperature)
    scaled = arr / T
    m = float(np.max(scaled))
    return float(T * (m + np.log(np.mean(np.exp(scaled - m)))))


def supra_tau_label(tau: float) -> str:
    """Format a supra threshold for column names: ``11.75 -> '11p75'``."""
    return f"{float(tau):g}".replace(".", "p")


# ---------------------------------------------------------------------------
# Probe construction
# ---------------------------------------------------------------------------


def build_probe_samples(
    *,
    x_t: np.ndarray,
    structural_logits: np.ndarray,
    mask_token_id: int,
    canonical_token_ids: Sequence[int],
    arm: str,
    ensemble_size: int,
    struct_temperature: float,
    confidence_threshold: float,
    prev_state: SCGRState | None,
    rng_key: Sequence[object],
) -> list[SCGRProbeSample]:
    """Build ``ensemble_size`` pseudo-terminal completions for one probe arm.

    Committed positions (``x_t != mask_token_id``) are fixed. Masked positions:
    ``fresh`` samples every position from the canonical-AA structural softmax;
    ``self_conditioned`` reuses ``prev_state.prev_x1_hat[i]`` when
    ``prev_state.prev_confidence[i] >= confidence_threshold`` and samples the
    rest. With no ``prev_state``, ``self_conditioned`` degrades to ``fresh`` and
    flags ``state_bootstrap_flag``.
    """
    tokens0 = np.asarray(x_t, dtype=np.int64)
    L = tokens0.shape[0]
    canonical = np.asarray(list(canonical_token_ids), dtype=np.int64)
    if canonical.size == 0:
        raise ValueError("build_probe_samples requires non-empty canonical_token_ids")
    masked = tokens0 == int(mask_token_id)
    masked_idx = np.flatnonzero(masked)
    num_masked = int(masked_idx.size)

    bootstrap = arm == "self_conditioned" and prev_state is None
    use_recycle = arm == "self_conditioned" and prev_state is not None
    if use_recycle:
        prev_tokens = np.asarray(prev_state.prev_x1_hat, dtype=np.int64)
        prev_conf = np.asarray(prev_state.prev_confidence, dtype=float)
        if prev_tokens.shape[0] != L or prev_conf.shape[0] != L:
            raise ValueError(
                "prev_state arrays must match x_t length "
                f"(got {prev_tokens.shape[0]}/{prev_conf.shape[0]} vs L={L})"
            )
        reuse_mask = masked & (prev_conf >= float(confidence_threshold))
    else:
        reuse_mask = np.zeros(L, dtype=bool)

    reuse_idx = np.flatnonzero(reuse_mask)
    sample_idx_positions = np.array(
        [i for i in masked_idx if not reuse_mask[i]], dtype=np.int64
    )

    # Precompute per-position canonical softmax + entropy at sampled positions.
    probs_by_pos: dict[int, np.ndarray] = {}
    entropies: list[float] = []
    for i in sample_idx_positions:
        p = _canonical_softmax(
            structural_logits[i], canonical, temperature=struct_temperature
        )
        probs_by_pos[int(i)] = p
        entropies.append(_entropy(p))
    mean_sample_entropy = float(np.mean(entropies)) if entropies else 0.0

    num_reused = int(reuse_idx.size)
    reuse_fraction = float(num_reused / num_masked) if num_masked else 0.0
    if num_reused:
        mean_prev_confidence_reused = float(prev_conf[reuse_idx].mean())
    else:
        mean_prev_confidence_reused = 0.0

    samples: list[SCGRProbeSample] = []
    for k in range(int(ensemble_size)):
        rng = _rng_for(rng_key, arm, k)
        tokens = tokens0.copy()
        if use_recycle and num_reused:
            tokens[reuse_idx] = prev_tokens[reuse_idx]
        for i in sample_idx_positions:
            p = probs_by_pos[int(i)]
            j = int(rng.choice(canonical.size, p=p))
            tokens[i] = int(canonical[j])
        samples.append(
            SCGRProbeSample(
                arm=arm,
                sample_idx=k,
                tokens=tokens,
                num_masked=num_masked,
                num_reused_from_prev=num_reused,
                reuse_fraction=reuse_fraction,
                mean_prev_confidence_reused=mean_prev_confidence_reused,
                mean_sample_entropy=mean_sample_entropy,
                state_bootstrap_flag=bool(bootstrap),
            )
        )
    return samples


def update_state_from_structural_argmax(
    *,
    structural_argmax_tokens: np.ndarray,
    structural_logits: np.ndarray,
    canonical_token_ids: Sequence[int],
) -> SCGRState:
    """Build :class:`SCGRState` from the structural argmax + canonical confidence.

    Confidence is the per-position canonical-softmax max-probability (renormalized
    over canonical AA tokens, temperature 1.0) — DELIBERATELY not full-vocab
    entropy, so it matches the canonical-AA sampling distribution and avoids
    special/non-canonical-token bias. No head score is consulted (firewall).
    """
    tokens = np.asarray(structural_argmax_tokens, dtype=np.int64).copy()
    canonical = np.asarray(list(canonical_token_ids), dtype=np.int64)
    logits = np.asarray(structural_logits, dtype=float)
    L = logits.shape[0]
    confidence = np.empty(L, dtype=float)
    for i in range(L):
        p = _canonical_softmax(logits[i], canonical, temperature=1.0)
        confidence[i] = float(p.max())
    return SCGRState(prev_x1_hat=tokens, prev_confidence=confidence)


# ---------------------------------------------------------------------------
# Risk aggregation
# ---------------------------------------------------------------------------


def compute_risk_aggregates(
    *,
    windows: Sequence[Mapping[str, float | int]],
    length: int,
    tau_ref_B: float,
    top_m: int,
    lse_temperature: float,
    supra_tau_values: Sequence[float],
) -> SCGRRiskAggregates:
    """Reduce one completion's head windows to the monitored burden aggregators.

    Projects over the **full sequence** (every window the head returns) — the
    ``b_cur`` scope (``controller.py`` ``_compute_actionability_state``), NOT the
    seed-union ``b_env`` scope which zeroes non-seed residues. The three trajectory
    aggregators (``mean_excess`` / top-m LSE / supra-threshold mass) are all
    length-normalized; an empty window set yields all-zero aggregates.
    """
    win_list = [
        {"start": int(w["start"]), "end": int(w["end"]), "score": float(w["score"])}
        for w in windows
    ]
    h = max_covering_window_projection(length=int(length), windows=win_list)
    residue_excess = excess_over_tau(h, tau_ref=float(tau_ref_B))

    if residue_excess.size:
        G_mean_excess = float(residue_excess.mean())
        k = min(int(top_m), residue_excess.size)
        top = np.sort(residue_excess)[::-1][:k]
        G_topm_lse = _logmeanexp(top, temperature=float(lse_temperature))
        supra_masses = {
            float(tau): float(np.maximum(0.0, residue_excess - float(tau)).mean())
            for tau in supra_tau_values
        }
    else:
        G_mean_excess = 0.0
        G_topm_lse = 0.0
        supra_masses = {float(tau): 0.0 for tau in supra_tau_values}

    z = np.array([float(w["score"]) for w in win_list], dtype=float)
    head_risk_max = float(z.max()) if z.size else 0.0
    head_risk_LME = _logmeanexp(z, temperature=1.0) if z.size else 0.0

    return SCGRRiskAggregates(
        G_mean_excess=G_mean_excess,
        G_topm_lse=G_topm_lse,
        supra_masses=supra_masses,
        head_risk_LME=head_risk_LME,
        head_risk_max=head_risk_max,
        residue_excess=residue_excess,
    )


# ---------------------------------------------------------------------------
# Per-refresh reduction
# ---------------------------------------------------------------------------


def summarize_probe_refresh(
    *,
    per_sample: Sequence[tuple[SCGRProbeSample, SCGRRiskAggregates]],
    old_argmax_aggregates: SCGRRiskAggregates,
    supra_tau_values: Sequence[float],
) -> list[dict]:
    """Reduce the per-sample completions to one telemetry row per arm.

    ``B_sc_<metric>_median`` is the robust central estimate that would set the
    gain (``median`` over the ``K`` completions — NOT a high quantile, which is
    not estimable at ``K ≈ 3–4``). ``G_<metric>_max`` / ``_std`` are the over-K
    instability diagnostic, kept separate from ``B_sc`` (``doc/Self-Cond_GR.md``
    §3 step 5). ``old_argmax_*`` carries the single-argmax baseline burden so the
    monitor can compare new-ensemble vs old-``B_GR`` head-to-head.
    """
    tau_labels = [(float(t), supra_tau_label(t)) for t in supra_tau_values]

    # Group by arm, preserving first-seen order.
    arms: list[str] = []
    grouped: dict[str, list[tuple[SCGRProbeSample, SCGRRiskAggregates]]] = {}
    for sample, agg in per_sample:
        if sample.arm not in grouped:
            grouped[sample.arm] = []
            arms.append(sample.arm)
        grouped[sample.arm].append((sample, agg))

    def _old(metric_key: str) -> float:
        return float(getattr(old_argmax_aggregates, metric_key))

    rows: list[dict] = []
    for arm in arms:
        items = grouped[arm]
        samples = [s for s, _ in items]
        aggs = [a for _, a in items]

        mean_excess = np.array([a.G_mean_excess for a in aggs], dtype=float)
        topm = np.array([a.G_topm_lse for a in aggs], dtype=float)

        row: dict = {
            "arm": arm,
            "ensemble_size_effective": len(items),
            "B_sc_mean_excess_median": float(np.median(mean_excess)),
            "B_sc_topm_lse_median": float(np.median(topm)),
            "G_mean_excess_max": float(np.max(mean_excess)),
            "G_mean_excess_std": float(np.std(mean_excess)),
            "G_topm_lse_max": float(np.max(topm)),
            "G_topm_lse_std": float(np.std(topm)),
            "num_masked": int(samples[0].num_masked),
            "reuse_fraction_mean": float(
                np.mean([s.reuse_fraction for s in samples])
            ),
            "state_bootstrap_flag": bool(samples[0].state_bootstrap_flag),
            "old_argmax_G_mean_excess": _old("G_mean_excess"),
            "old_argmax_G_topm_lse": _old("G_topm_lse"),
        }
        for tau, label in tau_labels:
            supra = np.array([a.supra_masses[tau] for a in aggs], dtype=float)
            row[f"B_sc_supra_mass_tau_{label}_median"] = float(np.median(supra))
            row[f"G_supra_mass_tau_{label}_max"] = float(np.max(supra))
            row[f"G_supra_mass_tau_{label}_std"] = float(np.std(supra))
            row[f"old_argmax_G_supra_mass_tau_{label}"] = float(
                old_argmax_aggregates.supra_masses[tau]
            )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Per-residue K reduction (PLAN_PLANNER_SC_GR.md Task 1)
# ---------------------------------------------------------------------------


def reduce_residue_excess_over_k(
    per_sample: "Sequence[tuple[SCGRProbeSample, SCGRRiskAggregates]]",
    *,
    arm: str,
) -> np.ndarray:
    """Median over the K completions of one arm's per-residue ``residue_excess``.

    Returns the per-residue prospective-risk map ``r_i`` (RAR 0020 form:
    median-over-K, ``fresh`` arm). Empty completions skipped; no match ⇒ empty.
    """
    arrays = [
        a.residue_excess
        for s, a in per_sample
        if s.arm == arm and a.residue_excess.size
    ]
    if not arrays:
        return np.empty(0, dtype=float)
    return np.median(np.stack(arrays, axis=0), axis=0)
