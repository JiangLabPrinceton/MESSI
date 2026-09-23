"""Phase D3 EMA commit/revisit primitives (PLAN_RF.md §"D3 step behavior").

Pure functions only. The D3Handler in ``controller.py`` owns ``m_i`` state
across refreshes; this module supplies the projection, EMA update,
z-scoring, and commit-score construction. ``scores[]`` is never mutated here.

Conventions
-----------
* All arrays are residue-indexed with shape ``(L,)``; ``L`` is the protein
  length.
* ``window_to_residue_projection`` is fixed to ``max_covering_window`` for
  the first D3 implementation (PLAN §D2-D3 validation 11).
* Z-scoring is done across all currently committed residues in the protein,
  not only active-block residues (PLAN §D3-8). Below ``zscore_epsilon`` or
  with fewer than two eligible residues, that z-score term contributes 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .controller_config import D3Config


# ---------------------------------------------------------------------------
# Window → residue projection (PLAN §D3-3)
# ---------------------------------------------------------------------------


def project_window_excess_to_residue(
    *,
    windows: Sequence,
    window_excess: Sequence[float],
    sequence_length: int,
) -> np.ndarray:
    """For each residue, the max ``window_excess`` over windows covering it.

    ``windows`` is a sequence of records exposing ``start_0b`` and ``end_0b``
    on half-open ``[start, end)`` residue coordinates. Windows with
    non-positive excess never lower a residue that another window already
    raised: the max acts as a logical OR over positive excesses.
    """
    L = int(sequence_length)
    out = np.zeros(L, dtype=np.float64)
    for w, exc in zip(windows, window_excess):
        if float(exc) <= 0.0:
            continue
        s = max(0, int(getattr(w, "start_0b")))
        e = min(L, int(getattr(w, "end_0b")))
        for i in range(s, e):
            if float(exc) > out[i]:
                out[i] = float(exc)
    return out


# ---------------------------------------------------------------------------
# Stage B D3 evidence-source firewall (PLAN_RF_UNI_CTRL.md Task B5)
# ---------------------------------------------------------------------------


def compute_d3_evidence_input(
    *,
    d3_config: D3Config,
    legacy_e_i: np.ndarray,
    typed_fresh_i: np.ndarray | None,
) -> np.ndarray:
    """Select the residue evidence that drives the D3 EMA ``m_i``.

    ``legacy_window_excess`` (default and every existing D3-only config) keeps
    the Stage A window-excess projection. ``typed_fresh`` (d2_d3_full_stageB)
    feeds the typed ``e_fresh`` field, which by construction excludes ``b_mem``
    — so this never lets memory-inflated targeting flow back into the D3 memory
    update (PLAN "Fresh Evidence And Memory Firewall").
    """
    if d3_config.evidence_source == "typed_fresh":
        if typed_fresh_i is None:
            raise ValueError(
                "d3.evidence_source='typed_fresh' requires a typed_fresh_i array "
                "(the Stage B e_fresh field); got None"
            )
        return np.asarray(typed_fresh_i, dtype=np.float64)
    return np.asarray(legacy_e_i, dtype=np.float64)


# ---------------------------------------------------------------------------
# Residue-level reliability (PLAN §D3-4)
# ---------------------------------------------------------------------------


def compute_residue_reliability(
    *,
    active_blocks: Sequence,
    sequence_length: int,
) -> np.ndarray:
    """For each residue, max ``rho_B`` among active blocks that cover it."""
    L = int(sequence_length)
    out = np.zeros(L, dtype=np.float64)
    for blk in active_blocks:
        s = max(0, int(getattr(blk, "residue_start_0b")))
        e = min(L, int(getattr(blk, "residue_end_0b")))
        rho = float(getattr(blk, "rho_B"))
        for i in range(s, e):
            if rho > out[i]:
                out[i] = rho
    return out


# ---------------------------------------------------------------------------
# EMA gamma and update (PLAN §D3-5/D3-6)
# ---------------------------------------------------------------------------


def compute_ema_gamma(
    *,
    rho_i: np.ndarray,
    gamma_min: float,
    gamma_max: float,
) -> np.ndarray:
    """Reliability-dependent EMA decay factor in ``[gamma_min, gamma_max]``.

    rho=1 (reliable) → gamma_min (fast update toward new ``e_i``); rho=0
    (unreliable) → gamma_max (slow update, hold prior).
    """
    rho = np.clip(np.asarray(rho_i, dtype=np.float64), 0.0, 1.0)
    return float(gamma_min) + (float(gamma_max) - float(gamma_min)) * (1.0 - rho)


def update_ema(
    *,
    m_prev: np.ndarray | None,
    e_i: np.ndarray,
    rho_i: np.ndarray,
    gamma_min: float,
    gamma_max: float,
) -> np.ndarray:
    """One-refresh EMA update with cold-start convention.

    Cold start (``m_prev is None``): ``m = e_i`` directly so the first
    observation is not pre-smoothed by an arbitrary prior.
    """
    e = np.asarray(e_i, dtype=np.float64)
    if m_prev is None:
        return e.copy()
    gamma = compute_ema_gamma(rho_i=rho_i, gamma_min=gamma_min, gamma_max=gamma_max)
    return gamma * np.asarray(m_prev, dtype=np.float64) + (1.0 - gamma) * e


# ---------------------------------------------------------------------------
# Structural chosen-token log-prob for committed residues (PLAN §D3-7)
# ---------------------------------------------------------------------------


def compute_chosen_token_logprob(
    *,
    struct_logits: torch.Tensor,
    x_t: torch.Tensor,
    mask_token_id: int,
) -> np.ndarray:
    """Per-residue chosen-token log-softmax under the pre-D2 structural logits.

    Returns ``NaN`` for masked positions; callers must combine with the
    eligibility mask before z-scoring so masked positions never enter the
    standardization statistics.
    """
    L = int(struct_logits.shape[0])
    log_probs = torch.log_softmax(struct_logits, dim=-1).detach().cpu()
    x = x_t.detach().cpu()
    out = np.full(L, np.nan, dtype=np.float64)
    for i in range(L):
        tok = int(x[i].item())
        if tok == int(mask_token_id):
            continue
        out[i] = float(log_probs[i, tok].item())
    return out


# ---------------------------------------------------------------------------
# Z-score across eligible residues (PLAN §D3-8)
# ---------------------------------------------------------------------------


def z_score(
    *,
    values: np.ndarray,
    eligibility_mask: np.ndarray,
    zscore_epsilon: float,
) -> np.ndarray:
    """Standardize ``values`` over eligible residues; return zeros if degenerate.

    Eligible residues are committed positions (``x_t != mask_token_id``).
    Below two eligible residues or std < ``zscore_epsilon`` the term
    contributes zero (PLAN §D3-7 / D3-8 fallback).
    """
    vals = np.asarray(values, dtype=np.float64)
    mask = np.asarray(eligibility_mask, dtype=bool)
    out = np.zeros_like(vals, dtype=np.float64)
    eligible = vals[mask]
    eligible = eligible[~np.isnan(eligible)]
    if eligible.size < 2:
        return out
    mu = float(eligible.mean())
    sigma = float(eligible.std(ddof=0))
    if sigma < float(zscore_epsilon):
        return out
    standardized = np.where(mask, (vals - mu) / sigma, 0.0)
    standardized = np.where(np.isnan(standardized), 0.0, standardized)
    return standardized.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Commit score + same-refresh grace (PLAN §D3-9/D3-10)
# ---------------------------------------------------------------------------


def compute_commit_score(
    *,
    z_ell: np.ndarray,
    z_m: np.ndarray,
    lambda_commit: float,
    grace_positions: Sequence[int] | None = None,
) -> np.ndarray:
    """``commit_score_i = z(ell_i) - lambda * z(m_i)`` with same-refresh grace.

    For ``grace_positions`` (D2 corrected in current refresh) only the
    structural-confidence term remains; the immune penalty is dropped for
    this refresh per PLAN §D3-10. They are NOT protected from
    structure-low-confidence remask.
    """
    z_ell_arr = np.asarray(z_ell, dtype=np.float64)
    z_m_arr = np.asarray(z_m, dtype=np.float64)
    score = z_ell_arr - float(lambda_commit) * z_m_arr
    if grace_positions:
        for i in grace_positions:
            score[int(i)] = float(z_ell_arr[int(i)])
    return score


def compute_stage_a_rank_score(
    *,
    structural_logits: torch.Tensor,
    x_t: torch.Tensor,
    scores: np.ndarray,
    mask_token_id: int,
    m_i: np.ndarray | None,
    d2_evidence: np.ndarray | None,
    alpha_struct: float,
    lambda_commit: float,
    d2_evidence_nu: float,
    zscore_epsilon: float,
) -> np.ndarray:
    """Stage A multi-objective rank face for D2 actuation.

    ``scores`` is the sampler's sampled-token log-prob under the actually used
    logits. It is copied into a local array and never mutated here.
    """
    L = int(x_t.shape[0])
    eligibility = (x_t != int(mask_token_id)).detach().cpu().numpy()
    ell_struct = compute_chosen_token_logprob(
        struct_logits=structural_logits,
        x_t=x_t,
        mask_token_id=int(mask_token_id),
    )
    ell_sample = np.asarray(scores, dtype=np.float64).copy()
    if ell_sample.shape != (L,):
        raise ValueError(f"scores must have shape ({L},), got {ell_sample.shape}")
    if m_i is None:
        m_arr = np.zeros(L, dtype=np.float64)
    else:
        m_arr = np.asarray(m_i, dtype=np.float64)
        if m_arr.shape != (L,):
            raise ValueError(f"m_i must have shape ({L},), got {m_arr.shape}")
    if d2_evidence is None:
        d2_arr = np.zeros(L, dtype=np.float64)
    else:
        d2_arr = np.asarray(d2_evidence, dtype=np.float64)
        if d2_arr.shape != (L,):
            raise ValueError(
                f"d2_evidence must have shape ({L},), got {d2_arr.shape}"
            )

    z_ell_struct = z_score(
        values=ell_struct,
        eligibility_mask=eligibility,
        zscore_epsilon=float(zscore_epsilon),
    )
    z_ell_sample = z_score(
        values=ell_sample,
        eligibility_mask=eligibility,
        zscore_epsilon=float(zscore_epsilon),
    )
    z_m = z_score(
        values=m_arr,
        eligibility_mask=eligibility,
        zscore_epsilon=float(zscore_epsilon),
    )
    z_d2 = z_score(
        values=d2_arr,
        eligibility_mask=eligibility,
        zscore_epsilon=float(zscore_epsilon),
    )
    alpha = float(alpha_struct)
    rank = (
        alpha * z_ell_struct
        + (1.0 - alpha) * z_ell_sample
        - float(lambda_commit) * z_m
        + float(d2_evidence_nu) * z_d2
    )
    return np.where(eligibility, rank, 0.0).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Final freeze (PLAN §D3-12, validation rule 13)
# ---------------------------------------------------------------------------


def select_freeze_protected_positions(
    *,
    x_t: torch.Tensor,
    mask_token_id: int,
    step: int,
    n_steps: int,
    final_freeze_steps: int,
) -> tuple[int, ...]:
    """Return all currently committed positions during the final freeze window.

    Outside the freeze window, returns the empty tuple so the controller
    imposes no remask protection.
    """
    if int(step) < int(n_steps) - int(final_freeze_steps):
        return ()
    committed = (
        (x_t != int(mask_token_id))
        .nonzero(as_tuple=False)
        .flatten()
        .tolist()
    )
    return tuple(int(p) for p in committed)


# ---------------------------------------------------------------------------
# D3Handler — stateless orchestrator composed into the controller (PLAN §D3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class D3RefreshOutcome:
    """D3 post-sampling refresh output consumed by ``controller.post_step``."""

    commit_score: np.ndarray
    e_i: np.ndarray
    m_i: np.ndarray
    rho_i: np.ndarray
    grace_positions: tuple[int, ...]
    # Stage C.1 global-pressure attribution (PLAN_RF_UNI_CTRL.md C1.6). ``lambda_eff``
    # is the value used in the commit-score immune penalty (``lambda_commit * g_GR``
    # when scaling, else the configured ``lambda_commit``). All None in legacy /
    # pressure-off runs so existing telemetry is unchanged.
    lambda_base: float | None = None
    lambda_eff: float | None = None
    g_GR_effective: float | None = None


class D3Handler:
    """One-shot composition of D3 pure primitives for the post-sampling hook.

    State (``m_i`` across refreshes) lives in ``RefreshState`` on the
    controller; the handler is stateless and only owns ``D3Config``.
    """

    def __init__(self, config: D3Config) -> None:
        self.config = config

    def run_refresh(
        self,
        *,
        windows: Sequence,
        window_excess: Sequence[float],
        active_blocks: Sequence,
        sequence_length: int,
        structural_logits: torch.Tensor,
        x_t: torch.Tensor,
        mask_token_id: int,
        m_prev: np.ndarray | None,
        corrected_positions: Sequence[int],
        typed_fresh: np.ndarray | None = None,
        lambda_override: float | None = None,
        g_GR_effective: float | None = None,
    ) -> D3RefreshOutcome:
        # Stage C.1 (PLAN_RF_UNI_CTRL.md C1.6): ``lambda_override`` is the EFFECTIVE
        # immune-penalty weight (``lambda_commit * g_GR``) from the controller's
        # pressure actuator; ``None`` keeps the configured ``lambda_commit``.
        lambda_base = float(self.config.lambda_commit)
        lambda_eff = lambda_base if lambda_override is None else float(lambda_override)
        legacy_e_i = project_window_excess_to_residue(
            windows=windows,
            window_excess=window_excess,
            sequence_length=int(sequence_length),
        )
        # Stage B firewall (PLAN_RF_UNI_CTRL.md Task B5): d2_d3_full_stageB feeds
        # the typed e_fresh; every legacy config keeps the window-excess input.
        e_i = compute_d3_evidence_input(
            d3_config=self.config,
            legacy_e_i=legacy_e_i,
            typed_fresh_i=typed_fresh,
        )
        rho_i = compute_residue_reliability(
            active_blocks=active_blocks, sequence_length=int(sequence_length)
        )
        m_i = update_ema(
            m_prev=m_prev,
            e_i=e_i,
            rho_i=rho_i,
            gamma_min=float(self.config.gamma_min),
            gamma_max=float(self.config.gamma_max),
        )
        ell_i_cur = compute_chosen_token_logprob(
            struct_logits=structural_logits,
            x_t=x_t,
            mask_token_id=int(mask_token_id),
        )
        eligibility = (x_t != int(mask_token_id)).detach().cpu().numpy()
        z_ell = z_score(
            values=ell_i_cur,
            eligibility_mask=eligibility,
            zscore_epsilon=float(self.config.zscore_epsilon),
        )
        z_m = z_score(
            values=m_i,
            eligibility_mask=eligibility,
            zscore_epsilon=float(self.config.zscore_epsilon),
        )
        grace_positions = (
            tuple(int(p) for p in corrected_positions)
            if self.config.same_refresh_grace
            else ()
        )
        commit_score = compute_commit_score(
            z_ell=z_ell,
            z_m=z_m,
            lambda_commit=lambda_eff,
            grace_positions=grace_positions if grace_positions else None,
        )
        return D3RefreshOutcome(
            commit_score=commit_score,
            e_i=e_i,
            m_i=m_i,
            rho_i=rho_i,
            grace_positions=grace_positions,
            lambda_base=lambda_base,
            lambda_eff=lambda_eff,
            g_GR_effective=(None if g_GR_effective is None else float(g_GR_effective)),
        )
