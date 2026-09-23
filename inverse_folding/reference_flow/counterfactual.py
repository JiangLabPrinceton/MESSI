"""Phase D2 hard-counterfactual primitives (PLAN_RF.md §"D2 step behavior").

Pure functions only. The D2Handler in ``controller.py`` orchestrates these
primitives, owns the head-scorer batch call, and writes telemetry. Keeping
this module head-scorer-free makes every component independently testable.

Conventions
-----------
* Positions are 0-based residue indices into ``x_t`` / ``struct_logits``.
* ``struct_logits`` has shape ``(L, V)`` and is per-protein at one sampler step.
* ``canonical_token_ids`` enumerates the AA tokens that may appear in a
  designed sequence; special tokens (mask/pad/cls/eos) must be excluded by
  the caller so the candidate support is always sampleable.
* Local risk aggregation is LME (log-mean-exp) for the first D2/D3
  implementation (PLAN_RF.md §D2-D3 validation 7).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .controller_config import D2Config
from .self_conditioned_gr import build_probe_samples


# ---------------------------------------------------------------------------
# Position selection (PLAN §D2 step behavior 5-6)
# ---------------------------------------------------------------------------


def select_editable_positions(
    *,
    start_0b: int,
    end_0b: int,
    x_t: torch.Tensor,
    mask_token_id: int,
    residue_excess: np.ndarray,
    per_pos_entropy: torch.Tensor,
    max_positions: int,
    score_source: str = "legacy_excess",
    typed_field: np.ndarray | None = None,
) -> tuple[int, ...]:
    """Pick at most ``max_positions`` masked residues inside ``[start, end)``.

    Ranking primary key (descending), then per-position structural entropy
    ascending (lower entropy first as tiebreak). The primary key is selected by
    ``score_source`` (PLAN_RF_UNI_CTRL.md Stage B.1):

    * ``"legacy_excess"`` (default) — ``residue_excess`` (bit-for-bit the legacy
      behaviour; every existing/static config uses this).
    * ``"v_target"`` — the typed actionability field ``typed_field`` (the typed
      signal reaching the within-block position layer). Falls back to
      ``residue_excess`` when ``typed_field is None`` (e.g. static mode), so a
      misconfigured caller degrades to legacy rather than erroring.
    """
    masked: list[int] = []
    for i in range(int(start_0b), int(end_0b)):
        if int(x_t[i].item()) == int(mask_token_id):
            masked.append(i)
    if not masked:
        return ()
    if score_source == "v_target" and typed_field is not None:
        primary = np.asarray(typed_field, dtype=float)
    else:
        primary = np.asarray(residue_excess, dtype=float)
    ranked = sorted(
        masked,
        key=lambda i: (-float(primary[i]), float(per_pos_entropy[i].item())),
    )
    return tuple(ranked[: int(max_positions)])


# ---------------------------------------------------------------------------
# Candidate support (PLAN §D2 step behavior 7)
# ---------------------------------------------------------------------------


def build_candidate_support(
    *,
    struct_logits: torch.Tensor,
    positions: Sequence[int],
    top_k: int,
    canonical_token_ids: Sequence[int],
) -> dict[int, tuple[int, ...]]:
    """Return per-position top-``top_k`` canonical-AA tokens by structural logit.

    Special tokens never enter the support because we restrict to
    ``canonical_token_ids`` before the top-k.
    """
    canonical_arr = np.asarray(list(canonical_token_ids), dtype=np.int64)
    if canonical_arr.size == 0:
        raise ValueError("canonical_token_ids must be non-empty")
    k = min(int(top_k), int(canonical_arr.shape[0]))
    out: dict[int, tuple[int, ...]] = {}
    for i in positions:
        logits_i = struct_logits[int(i), :].detach().cpu()
        logits_canonical = logits_i[torch.from_numpy(canonical_arr)]
        topk_idx = torch.topk(logits_canonical, k).indices.numpy()
        out[int(i)] = tuple(int(canonical_arr[j]) for j in topk_idx)
    return out


def build_safe_candidate_support(
    *,
    struct_logits: torch.Tensor,
    positions: Sequence[int],
    top_k: int,
    canonical_token_ids: Sequence[int],
    delta_struct: float,
) -> dict[int, tuple[int, ...]]:
    """Trust-region structural support used by Stage A.

    Tokens are filtered by log-probability drop from the structural top-1
    canonical token, then capped to ``top_k`` by structural log-prob. The
    top-1 token is always retained.
    """
    canonical_arr = np.asarray(list(canonical_token_ids), dtype=np.int64)
    if canonical_arr.size == 0:
        raise ValueError("canonical_token_ids must be non-empty")
    k = min(int(top_k), int(canonical_arr.shape[0]))
    out: dict[int, tuple[int, ...]] = {}
    for i in positions:
        log_probs_i = torch.log_softmax(struct_logits[int(i), :], dim=-1).detach().cpu()
        vals = log_probs_i[torch.from_numpy(canonical_arr)].numpy().astype(np.float64)
        order = np.argsort(-vals)
        top_val = float(vals[order[0]])
        safe_order = [
            int(j)
            for j in order.tolist()
            if float(vals[int(j)]) >= top_val - float(delta_struct)
        ]
        if not safe_order:
            safe_order = [int(order[0])]
        if int(order[0]) not in safe_order:
            safe_order.insert(0, int(order[0]))
        safe_order = safe_order[:k]
        out[int(i)] = tuple(int(canonical_arr[j]) for j in safe_order)
    return out


# ---------------------------------------------------------------------------
# Candidate enumeration (PLAN §D2 step behavior 8)
# ---------------------------------------------------------------------------


def enumerate_candidates(
    *,
    positions: Sequence[int],
    K_i_per_pos: dict[int, tuple[int, ...]],
    max_candidates: int,
    struct_logits: torch.Tensor,
    seed_tuple: tuple,
    struct_temperature: float = 1.0,
) -> tuple[tuple[tuple[int, ...], ...], str]:
    """Build the per-block candidate set.

    * Cartesian product when ``prod(|K_i|) <= max_candidates``.
    * Else i.i.d. with-replacement sampling from per-position structural
      softmax restricted to ``K_i``. Duplicates are retained because they are
      part of the Monte Carlo estimate that feeds ESS.

    Sampling RNG is derived deterministically from ``seed_tuple`` so a refresh
    can be reproduced from telemetry alone.
    """
    pos_list = list(positions)
    K_lists = [K_i_per_pos[int(p)] for p in pos_list]
    n_cart = 1
    for K in K_lists:
        n_cart *= max(1, len(K))
    if n_cart <= int(max_candidates):
        return tuple(tuple(combo) for combo in product(*K_lists)), "cartesian"

    rng = np.random.default_rng(_seed_from_tuple(seed_tuple))
    per_pos_choice_arrays: list[np.ndarray] = []
    per_pos_probs: list[np.ndarray] = []
    for p, K in zip(pos_list, K_lists):
        K_arr = np.asarray(K, dtype=np.int64)
        per_pos_choice_arrays.append(K_arr)
        logits_K = (
            struct_logits[int(p), torch.from_numpy(K_arr)]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            / float(struct_temperature)
        )
        logits_K = logits_K - float(logits_K.max())
        probs = np.exp(logits_K)
        probs = probs / probs.sum()
        per_pos_probs.append(probs)
    samples: list[tuple[int, ...]] = []
    for _ in range(int(max_candidates)):
        cand: list[int] = []
        for K_arr, probs in zip(per_pos_choice_arrays, per_pos_probs):
            choice = int(rng.choice(K_arr.shape[0], p=probs))
            cand.append(int(K_arr[choice]))
        samples.append(tuple(cand))
    return tuple(samples), "sampled"


def _seed_from_tuple(tup: tuple) -> int:
    """Deterministic 64-bit unsigned seed from a generic tuple key."""
    h = hashlib.sha256(repr(tup).encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


# ---------------------------------------------------------------------------
# Local risk + per-candidate joint structural mass (PLAN §D2 step 10-12)
# ---------------------------------------------------------------------------


def compute_local_risk(
    *,
    window_risks: Sequence[float],
    omega_indices: Sequence[int],
    aggregation: str = "LME",
) -> float:
    """Restrict to ``Omega(B)`` window indices and aggregate via LME."""
    indices = [int(i) for i in omega_indices]
    if not indices:
        return float("-inf")
    vals = np.asarray([float(window_risks[i]) for i in indices], dtype=np.float64)
    if aggregation == "LME":
        m = float(vals.max())
        if not math.isfinite(m):
            return m
        return m + math.log(float(np.exp(vals - m).mean()))
    raise ValueError(f"unsupported aggregation: {aggregation!r}")


def compute_local_risk_batch(
    *,
    window_risks: np.ndarray,
    omega_indices: Sequence[int],
    aggregation: str = "LME",
) -> np.ndarray:
    """Vectorized ``compute_local_risk`` for a ``[K, W]`` window-risk matrix."""
    if aggregation != "LME":
        raise ValueError(f"unsupported aggregation: {aggregation!r}")
    arr = np.asarray(window_risks, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"window_risks must be [K, W], got shape {arr.shape}")
    indices = np.asarray([int(i) for i in omega_indices], dtype=np.int64)
    if indices.size == 0:
        return np.full((arr.shape[0],), float("-inf"), dtype=np.float64)
    vals = arr[:, indices]
    m = vals.max(axis=1)
    out = np.empty((arr.shape[0],), dtype=np.float64)
    finite = np.isfinite(m)
    out[~finite] = m[~finite]
    if finite.any():
        centered = vals[finite] - m[finite, None]
        out[finite] = m[finite] + np.log(np.exp(centered).mean(axis=1))
    return out


def _score_window_risk_matrix(
    *,
    scorer: Any,
    protein_id: str,
    records: list[tuple[str, str]],
) -> np.ndarray:
    """Score records as a compact ``[K, W]`` matrix when the scorer supports it."""
    if not records:
        return np.zeros((0, 0), dtype=np.float64)
    if hasattr(scorer, "score_window_risk_batch_same_protein"):
        compact = scorer.score_window_risk_batch_same_protein(
            protein_id=str(protein_id), records=records
        )
        return np.asarray(compact.window_risks, dtype=np.float64)

    batch = scorer.score_batch_same_protein(
        protein_id=str(protein_id), records=records
    )
    return np.asarray(
        [[float(w.z) for w in score.windows] for score in batch.scores],
        dtype=np.float64,
    )


def compute_Q_B_per_candidate(
    *,
    candidates: Sequence[tuple[int, ...]],
    positions: Sequence[int],
    K_i_per_pos: dict[int, tuple[int, ...]],
    struct_logits: torch.Tensor,
    struct_temperature: float = 1.0,
) -> np.ndarray:
    """Joint structural probability of each candidate, factorized over positions."""
    pos_list = list(positions)
    log_prob_lookup: list[dict[int, float]] = []
    for p in pos_list:
        K = K_i_per_pos[int(p)]
        K_arr = np.asarray(K, dtype=np.int64)
        logits_K = (
            struct_logits[int(p), torch.from_numpy(K_arr)]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            / float(struct_temperature)
        )
        logits_K = logits_K - float(logits_K.max())
        probs = np.exp(logits_K)
        probs = probs / probs.sum()
        log_prob_lookup.append(
            {int(K_arr[j]): float(np.log(probs[j])) for j in range(K_arr.shape[0])}
        )
    Q = np.zeros(len(candidates), dtype=np.float64)
    for c_idx, cand in enumerate(candidates):
        lp = 0.0
        for j, tok in enumerate(cand):
            lp += log_prob_lookup[j][int(tok)]
        Q[c_idx] = float(np.exp(lp))
    return Q


def compute_context_pnll(
    *,
    struct_logits: torch.Tensor,
    x_t: torch.Tensor,
    mask_token_id: int,
    start_0b: int,
    end_0b: int,
) -> float | None:
    """Mean pseudo-NLL over committed context residues in ``[start, end)``."""
    log_probs = torch.log_softmax(struct_logits, dim=-1).detach().cpu()
    return compute_context_pnll_from_log_probs(
        log_probs=log_probs,
        x_t=x_t,
        mask_token_id=mask_token_id,
        start_0b=start_0b,
        end_0b=end_0b,
    )


def compute_context_pnll_from_log_probs(
    *,
    log_probs: torch.Tensor,
    x_t: torch.Tensor,
    mask_token_id: int,
    start_0b: int,
    end_0b: int,
) -> float | None:
    """Mean pseudo-NLL using precomputed ``log_softmax(struct_logits)``."""
    s = max(0, int(start_0b))
    e = min(int(x_t.shape[0]), int(end_0b))
    if e <= s:
        return None
    lp = log_probs.detach()
    if lp.device.type != "cpu":
        lp = lp.cpu()
    x = x_t.detach().cpu()
    span_tokens = x[s:e].to(torch.long)
    committed = span_tokens != int(mask_token_id)
    if not bool(committed.any()):
        return None
    positions = torch.arange(s, e, dtype=torch.long)[committed]
    tokens = span_tokens[committed]
    vals = -lp[positions, tokens]
    return float(vals.to(torch.float64).mean().item())


@dataclass(frozen=True)
class EnsembleDiagnostics:
    mean_delta_R: dict[int, float]
    std_delta_R: dict[int, float]
    sign_consistency: dict[int, float]
    rank_flip_rate: float


def compute_ensemble_diagnostics(
    *,
    argmax_delta_R: np.ndarray,
    ensemble_delta_R_by_candidate: dict[int, np.ndarray],
) -> EnsembleDiagnostics:
    """Summarize local-completion ensemble deltas without applying a gate."""
    argmax_arr = np.asarray(argmax_delta_R, dtype=np.float64)
    mean_delta: dict[int, float] = {}
    std_delta: dict[int, float] = {}
    sign_consistency: dict[int, float] = {}
    for idx, vals in ensemble_delta_R_by_candidate.items():
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            continue
        mean_delta[int(idx)] = float(arr.mean())
        std_delta[int(idx)] = float(arr.std(ddof=0))
        arg_sign = np.sign(float(argmax_arr[int(idx)])) if int(idx) < argmax_arr.size else 0.0
        if arg_sign == 0.0:
            sign_consistency[int(idx)] = 0.0
        else:
            sign_consistency[int(idx)] = float((np.sign(arr) == arg_sign).mean())
    rank_flip_rate = 0.0
    if mean_delta:
        indices = sorted(mean_delta)
        arg_order = sorted(indices, key=lambda i: float(argmax_arr[i]))
        ens_order = sorted(indices, key=lambda i: float(mean_delta[i]))
        flips = sum(1 for a, b in zip(arg_order, ens_order) if a != b)
        rank_flip_rate = float(flips) / float(len(indices))
    return EnsembleDiagnostics(
        mean_delta_R=mean_delta,
        std_delta_R=std_delta,
        sign_consistency=sign_consistency,
        rank_flip_rate=rank_flip_rate,
    )


# ---------------------------------------------------------------------------
# Weights + ESS + marginals (PLAN §D2 step 13-16)
# ---------------------------------------------------------------------------


def compute_weights(
    *,
    mode: str,
    Q_B: np.ndarray,
    delta_R_B: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Branch-specific unnormalized immune importance weight.

    * Cartesian enumeration: ``w(a) = Q_B(a) * exp(-beta * ΔR_B(a))``.
    * Sampled from ``Q_B``: ``w(a) = exp(-beta * ΔR_B(a))`` (``Q_B`` is the proposal).
    """
    expo = np.exp(-float(beta) * np.asarray(delta_R_B, dtype=np.float64))
    if mode == "cartesian":
        return np.asarray(Q_B, dtype=np.float64) * expo
    if mode == "sampled":
        return expo
    raise ValueError(f"unknown candidate mode: {mode!r}")


def compute_ess(weights: np.ndarray) -> float:
    """Effective sample size from unnormalized weights (zero-safe)."""
    arr = np.asarray(weights, dtype=np.float64)
    s = float(arr.sum())
    s2 = float((arr * arr).sum())
    if s2 <= 0.0:
        return 0.0
    return (s * s) / s2


def project_marginals(
    *,
    candidates: Sequence[tuple[int, ...]],
    weights_normalized: np.ndarray,
    positions: Sequence[int],
    K_i_per_pos: dict[int, tuple[int, ...]],
) -> dict[int, dict[int, float]]:
    """Marginal probability per (position, token in K_i) under given weights."""
    pos_list = list(positions)
    marginals: dict[int, dict[int, float]] = {
        int(p): {int(t): 0.0 for t in K_i_per_pos[int(p)]} for p in pos_list
    }
    for c_idx, cand in enumerate(candidates):
        w = float(weights_normalized[c_idx])
        for j, p in enumerate(pos_list):
            tok = int(cand[j])
            if tok in marginals[int(p)]:
                marginals[int(p)][tok] += w
    return marginals


# ---------------------------------------------------------------------------
# Logit correction (PLAN §D2 step 17-20)
# ---------------------------------------------------------------------------


def compute_delta_logit(
    *,
    pi_marginals: dict[int, dict[int, float]],
    Q_marginals: dict[int, dict[int, float]],
    eta: float,
    rho_B: float,
    epsilon: float,
    max_abs_shift: float,
) -> dict[tuple[int, int], float]:
    """Per ``(position, token)`` log-ratio correction restricted to candidate support."""
    factor = float(eta) * float(rho_B)
    out: dict[tuple[int, int], float] = {}
    for p, pi_p in pi_marginals.items():
        Q_p = Q_marginals[int(p)]
        for tok in pi_p:
            pi_val = float(pi_p[int(tok)])
            Q_val = float(Q_p[int(tok)])
            shift = factor * (math.log(pi_val + float(epsilon)) - math.log(Q_val + float(epsilon)))
            if shift > float(max_abs_shift):
                shift = float(max_abs_shift)
            elif shift < -float(max_abs_shift):
                shift = -float(max_abs_shift)
            out[(int(p), int(tok))] = float(shift)
    return out


def apply_logit_correction(
    *,
    struct_logits: torch.Tensor,
    delta_logit: dict[tuple[int, int], float],
) -> torch.Tensor:
    """Return a ``.clone()``'d logits tensor with ``delta_logit`` applied.

    The original ``struct_logits`` MUST remain untouched because D3 commit
    scoring and paired-attribution sampling both rely on the uncorrected
    structural distribution.
    """
    corrected = struct_logits.clone()
    for (i, tok), shift in delta_logit.items():
        corrected[int(i), int(tok)] = corrected[int(i), int(tok)] + float(shift)
    return corrected


# ---------------------------------------------------------------------------
# Feasibility (PLAN §D2 step 15)
# ---------------------------------------------------------------------------


def compute_feasibility(
    *,
    delta_R_B: np.ndarray,
    min_delta_R_improvement: float,
) -> tuple[bool, float]:
    """A block is feasible when at least one candidate has ``ΔR_B < -threshold``."""
    arr = np.asarray(delta_R_B, dtype=np.float64)
    if arr.size == 0:
        return False, float("inf")
    best = float(arr.min())
    return (best < -float(min_delta_R_improvement)), best


# ---------------------------------------------------------------------------
# D2Handler — stateless orchestrator composed into the controller (PLAN §D2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class D2BlockOutcome:
    """Per-block D2 diagnostics and (when applied) logit correction.

    ``skipped_reason`` is ``None`` when the block contributes a correction.
    ``"no_editable"`` → block had no masked positions to write into.
    ``"low_ess"`` → ESS fraction below ``min_ess_fraction``; ``g_ESS=0``.
    ``"not_feasible"`` → no candidate met the ``min_delta_R_improvement`` gate.

    ``r_current`` is the LME risk over ``omega_indices`` of the hard-completion
    sequence at refresh time. Used by the post-sampling realized ΔR pass to
    compute ``delta_R_corrected`` / ``delta_R_uncorrected`` (PLAN_RF.md D0
    schema) without re-scoring the baseline.
    """

    block_id: int
    omega_indices: tuple[int, ...]
    editable_positions: tuple[int, ...]
    K_i_per_pos: dict[int, tuple[int, ...]]
    candidate_mode: str
    candidate_count: int
    delta_R_B: tuple[float, ...]
    Q_B: tuple[float, ...]
    weights_unnormalized: tuple[float, ...]
    ess: float
    ess_fraction: float
    g_ESS_candidates: float
    feasible: bool
    best_delta_R_B: float
    mean_delta_R_B: float
    rho_B_effective: float
    corrected_positions: tuple[int, ...]
    delta_logit: dict[tuple[int, int], float]
    skipped_reason: str | None
    r_current: float = float("nan")
    safe_support_sizes: dict[int, int] = field(default_factory=dict)
    candidate_count_argmax: int = 0
    candidate_count_ensemble: int = 0
    argmax_best_delta_R_B: float = float("nan")
    ensemble_best_delta_R_B: float = float("nan")
    argmax_to_ensemble_rank_flip_rate: float = 0.0
    argmax_to_ensemble_rank_flip_flag: bool = False
    ensemble_delta_R_std: float | None = None
    ensemble_sign_consistency: float | None = None
    context_pnll: float | None = None
    g_pnll: float | None = None
    context_jsd: float | None = None
    g_stability: float = 1.0
    # Stage C.1 global-pressure actuation provenance (PLAN_RF_UNI_CTRL.md C1.5).
    # ``beta_base`` is the configured beta; ``beta_eff`` is the value actually used
    # for candidate weights (``beta_base * g_GR`` when scaling, else ``beta_base``);
    # ``g_GR_effective`` is the trajectory pressure scalar. All None in Stage B /
    # static runs (no override passed), so legacy telemetry is unchanged.
    beta_base: float | None = None
    beta_eff: float | None = None
    g_GR_effective: float | None = None
    # §A2 (PLAN_PLANNER_SC_GR.md): which candidate scorer ranked this block —
    # 'local' (block-span ensemble ΔR_B) or 'terminal' (full-completion P_cheap).
    candidate_score_source: str = "local"


@dataclass(frozen=True)
class D2RefreshOutcome:
    """Aggregated D2 output for one refresh step."""

    corrected_logits: torch.Tensor
    block_outcomes: tuple[D2BlockOutcome, ...]
    corrected_positions: frozenset


class D2Handler:
    """Composes D2 pure primitives into a single per-refresh correction step.

    The handler owns ``D2Config`` only; everything else (scorer, decode
    callable, structural logits, current completion) is supplied per call so
    the same handler instance can be shared across designs.
    """

    def __init__(self, config: D2Config) -> None:
        self.config = config

    def correct_logits(
        self,
        *,
        structural_logits: torch.Tensor,
        active_blocks: Sequence,
        x_t: torch.Tensor,
        mask_token_id: int,
        completed_tokens: torch.Tensor,
        decode_tokens: Callable[[torch.Tensor], str],
        per_pos_entropy: torch.Tensor,
        struct_log_probs: torch.Tensor,
        residue_excess: np.ndarray,
        current_window_risks: Sequence[float],
        scorer: Any,
        canonical_token_ids: Sequence[int],
        protein_id: str,
        seed: int,
        design_idx: int,
        refresh_step: int,
        beta_override: float | None = None,
        g_GR_effective: float | None = None,
        within_block_source: str = "legacy_excess",
        typed_field: np.ndarray | None = None,
    ) -> D2RefreshOutcome:
        # Stage C.1 (PLAN_RF_UNI_CTRL.md C1.5): ``beta_override`` is the EFFECTIVE
        # beta (``beta * g_GR``) supplied by the controller's global-pressure
        # actuator; ``None`` keeps the configured beta (legacy / pressure-off).
        # The delta logits are computed ONCE here with ``beta_eff``; the sticky
        # re-delivery path replays the stored deltas and never recomputes beta.
        beta_base = float(self.config.beta)
        beta_eff = beta_base if beta_override is None else float(beta_override)
        g_eff = None if g_GR_effective is None else float(g_GR_effective)

        block_outcomes: list[D2BlockOutcome] = []
        delta_logit_acc: dict[tuple[int, int], float] = {}
        corrected_positions: set[int] = set()

        for blk in active_blocks:
            outcome = self._process_block(
                block=blk,
                structural_logits=structural_logits,
                x_t=x_t,
                mask_token_id=mask_token_id,
                completed_tokens=completed_tokens,
                decode_tokens=decode_tokens,
                per_pos_entropy=per_pos_entropy,
                struct_log_probs=struct_log_probs,
                residue_excess=residue_excess,
                current_window_risks=current_window_risks,
                scorer=scorer,
                canonical_token_ids=canonical_token_ids,
                protein_id=protein_id,
                seed=seed,
                design_idx=design_idx,
                refresh_step=refresh_step,
                beta_eff=beta_eff,
                within_block_source=within_block_source,
                typed_field=typed_field,
            )
            # Stamp pressure provenance + the §A2 candidate-score source on every
            # block outcome (including the skipped early-return paths) so telemetry
            # attribution is complete.
            outcome = replace(
                outcome,
                beta_base=beta_base,
                beta_eff=beta_eff,
                g_GR_effective=g_eff,
                candidate_score_source=str(self.config.candidate_score_source),
            )
            block_outcomes.append(outcome)
            if outcome.skipped_reason is None:
                for key, shift in outcome.delta_logit.items():
                    delta_logit_acc[key] = delta_logit_acc.get(key, 0.0) + float(shift)
                for pos in outcome.corrected_positions:
                    corrected_positions.add(int(pos))

        if delta_logit_acc:
            corrected = apply_logit_correction(
                struct_logits=structural_logits, delta_logit=delta_logit_acc
            )
        else:
            corrected = structural_logits.clone()
        return D2RefreshOutcome(
            corrected_logits=corrected,
            block_outcomes=tuple(block_outcomes),
            corrected_positions=frozenset(corrected_positions),
        )

    def _score_completion_ensemble(
        self,
        *,
        candidates: Sequence[tuple[int, ...]],
        candidate_indices: Sequence[int],
        editable: Sequence[int],
        block,
        structural_logits: torch.Tensor,
        x_t: torch.Tensor,
        mask_token_id: int,
        completed_tokens: torch.Tensor,
        decode_tokens: Callable[[torch.Tensor], str],
        scorer: Any,
        canonical_token_ids: Sequence[int],
        protein_id: str,
        seed: int,
        design_idx: int,
        refresh_step: int,
        r_current: float,
        omega: tuple[int, ...],
    ) -> dict[int, np.ndarray]:
        records: list[tuple[str, str]] = []
        mapping: list[int] = []
        editable_set = {int(p) for p in editable}
        canonical_arr = np.asarray(list(canonical_token_ids), dtype=np.int64)
        for cand_idx in candidate_indices:
            cand = candidates[int(cand_idx)]
            for ens_idx in range(int(self.config.completion_ensemble_size)):
                tokens = completed_tokens.detach().clone()
                for pos, tok in zip(editable, cand):
                    tokens[int(pos)] = int(tok)
                rng = np.random.default_rng(
                    _seed_from_tuple(
                        (
                            int(seed),
                            str(protein_id),
                            int(design_idx),
                            int(refresh_step),
                            int(block.block_id),
                            int(cand_idx),
                            int(ens_idx),
                        )
                    )
                )
                # Stage A v1 approximates the local Omega(B) completion field
                # by the merged active-block residue span. The exact union of
                # all covering scoring-window receptive fields is a Stage B
                # refinement; keeping the span here avoids changing D1 active
                # block geometry during the actuation test.
                for pos in range(
                    int(block.residue_start_0b), int(block.residue_end_0b)
                ):
                    if pos in editable_set:
                        continue
                    if int(x_t[pos].item()) != int(mask_token_id):
                        continue
                    logits_K = (
                        structural_logits[pos, torch.from_numpy(canonical_arr)]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                        / float(self.config.struct_temperature)
                    )
                    logits_K = logits_K - float(logits_K.max())
                    probs = np.exp(logits_K)
                    probs = probs / probs.sum()
                    choice = int(rng.choice(canonical_arr.shape[0], p=probs))
                    tokens[pos] = int(canonical_arr[choice])
                records.append(
                    (
                        f"d2_b{int(block.block_id)}_c{int(cand_idx)}_e{int(ens_idx)}",
                        decode_tokens(tokens),
                    )
                )
                mapping.append(int(cand_idx))
        if not records:
            return {}
        window_risk_matrix = _score_window_risk_matrix(
            scorer=scorer, protein_id=str(protein_id), records=records
        )
        local_risks = compute_local_risk_batch(
            window_risks=window_risk_matrix,
            omega_indices=omega,
            aggregation="LME",
        )
        out: dict[int, list[float]] = {int(i): [] for i in candidate_indices}
        for risk, cand_idx in zip(local_risks, mapping):
            out[int(cand_idx)].append(float(risk - float(r_current)))
        return {
            int(idx): np.asarray(vals, dtype=np.float64)
            for idx, vals in out.items()
        }

    def _score_completion_terminal(
        self,
        *,
        candidates: Sequence[tuple[int, ...]],
        candidate_indices: Sequence[int],
        editable: Sequence[int],
        block,
        structural_logits: torch.Tensor,
        x_t: torch.Tensor,
        mask_token_id: int,
        completed_tokens: torch.Tensor,
        decode_tokens: Callable[[torch.Tensor], str],
        scorer: Any,
        canonical_token_ids: Sequence[int],
        protein_id: str,
        seed: int,
        design_idx: int,
        refresh_step: int,
        r_current: float,
        omega: tuple[int, ...],
        K_P: int,
    ) -> dict[int, np.ndarray]:
        """§A2 terminal candidate probe (P_cheap; doc §8.4 Path A / RAR 0019).

        Parallels :meth:`_score_completion_ensemble` but rolls each candidate out
        to a **full-sequence terminal completion** (all remaining masked positions
        filled, not just the block span) ``K_P`` times. The fill RNG key EXCLUDES
        the candidate index, so the ``K_P`` fills are **paired across candidates**
        (same context draws; only the committed editable tokens differ). Risk is
        ``Ω(B)-LME(window_risks) − r_current`` in the SAME head-logit units as the
        local ``ΔR_B`` (so the resampling weight ``exp(−β·ΔR)`` stays consistent;
        deliberately NOT the SC-GR ``G_topm_lse`` aggregate).
        """
        base_x = x_t.detach().cpu().numpy().astype(np.int64)
        struct_np = structural_logits.detach().cpu().numpy().astype(float)
        # Paired across candidates: the fill seed depends on the block, NOT the
        # candidate, so candidate A and B share the same K_P context completions.
        rng_key = (
            int(seed),
            str(protein_id),
            int(design_idx),
            int(refresh_step),
            int(block.block_id),
        )
        records: list[tuple[str, str]] = []
        mapping: list[int] = []
        for cand_idx in candidate_indices:
            cand = candidates[int(cand_idx)]
            x_cand = base_x.copy()
            for pos, tok in zip(editable, cand):
                x_cand[int(pos)] = int(tok)
            samples = build_probe_samples(
                x_t=x_cand,
                structural_logits=struct_np,
                mask_token_id=int(mask_token_id),
                canonical_token_ids=canonical_token_ids,
                arm="fresh",
                ensemble_size=int(K_P),
                struct_temperature=float(self.config.struct_temperature),
                confidence_threshold=1.0,  # unused for the fresh arm
                prev_state=None,
                rng_key=rng_key,
            )
            for ens_idx, s in enumerate(samples):
                records.append(
                    (
                        f"d2t_b{int(block.block_id)}_c{int(cand_idx)}_e{int(ens_idx)}",
                        decode_tokens(torch.as_tensor(s.tokens, dtype=torch.long)),
                    )
                )
                mapping.append(int(cand_idx))
        if not records:
            return {}
        window_risk_matrix = _score_window_risk_matrix(
            scorer=scorer, protein_id=str(protein_id), records=records
        )
        local_risks = compute_local_risk_batch(
            window_risks=window_risk_matrix,
            omega_indices=omega,
            aggregation="LME",
        )
        out: dict[int, list[float]] = {int(i): [] for i in candidate_indices}
        for risk, cand_idx in zip(local_risks, mapping):
            out[int(cand_idx)].append(float(risk - float(r_current)))
        return {
            int(idx): np.asarray(vals, dtype=np.float64)
            for idx, vals in out.items()
        }

    def _process_block(
        self,
        *,
        block,
        structural_logits: torch.Tensor,
        x_t: torch.Tensor,
        mask_token_id: int,
        completed_tokens: torch.Tensor,
        decode_tokens: Callable[[torch.Tensor], str],
        per_pos_entropy: torch.Tensor,
        struct_log_probs: torch.Tensor,
        residue_excess: np.ndarray,
        current_window_risks: Sequence[float],
        scorer: Any,
        canonical_token_ids: Sequence[int],
        protein_id: str,
        seed: int,
        design_idx: int,
        refresh_step: int,
        beta_eff: float | None = None,
        within_block_source: str = "legacy_excess",
        typed_field: np.ndarray | None = None,
    ) -> D2BlockOutcome:
        # ``beta_eff`` is the Stage C.1 effective beta (PLAN_RF_UNI_CTRL.md C1.5);
        # None falls back to the configured beta so legacy callers are unchanged.
        beta_used = float(self.config.beta) if beta_eff is None else float(beta_eff)
        omega = tuple(int(i) for i in block.window_indices)
        # Stage B.1: rank within-block positions by ``typed_field`` (v_target) when
        # within_block_source='v_target'; else legacy residue_excess.
        editable = select_editable_positions(
            start_0b=int(block.residue_start_0b),
            end_0b=int(block.residue_end_0b),
            x_t=x_t,
            mask_token_id=int(mask_token_id),
            residue_excess=residue_excess,
            per_pos_entropy=per_pos_entropy,
            max_positions=int(self.config.max_positions_per_block),
            score_source=within_block_source,
            typed_field=typed_field,
        )
        if not editable:
            # No editable masked positions inside this block → no candidate
            # enumeration / head re-score happens, so ``r_current`` is not
            # available here. We pass NaN explicitly to document that no
            # realized ΔR can be computed for no_editable blocks (and the
            # realized-ΔR pass at post_step also skips blocks with
            # ``skipped_reason != None``).
            current_window_risks_tuple = tuple(float(z) for z in current_window_risks)
            r_current_unused = compute_local_risk(
                window_risks=current_window_risks_tuple,
                omega_indices=omega,
                aggregation="LME",
            )
            return _empty_block_outcome(
                block_id=int(block.block_id),
                omega=omega,
                reason="no_editable",
                r_current=float(r_current_unused),
            )

        r_current = compute_local_risk(
            window_risks=current_window_risks,
            omega_indices=omega,
            aggregation="LME",
        )
        context_pnll = compute_context_pnll_from_log_probs(
            log_probs=struct_log_probs,
            x_t=x_t,
            mask_token_id=int(mask_token_id),
            # Same Stage A v1 locality approximation as the completion
            # ensemble: use the merged active-block span rather than expanding
            # to every scoring-window receptive-field edge.
            start_0b=int(block.residue_start_0b),
            end_0b=int(block.residue_end_0b),
        )
        if context_pnll is None:
            return _empty_block_outcome(
                block_id=int(block.block_id),
                omega=omega,
                reason="no_committed_context",
                r_current=float(r_current),
                context_pnll=None,
                g_pnll=0.0,
            )
        g_pnll = float(math.exp(-float(context_pnll) / float(self.config.context_pnll_h0)))
        g_stability = 1.0
        context_jsd = None

        K = build_safe_candidate_support(
            struct_logits=structural_logits,
            positions=editable,
            top_k=int(self.config.top_k_tokens),
            canonical_token_ids=canonical_token_ids,
            delta_struct=float(self.config.delta_struct),
        )
        candidates, mode = enumerate_candidates(
            positions=editable,
            K_i_per_pos=K,
            max_candidates=int(self.config.max_candidates_per_block),
            struct_logits=structural_logits,
            seed_tuple=(
                int(seed),
                str(protein_id),
                int(design_idx),
                int(refresh_step),
                int(block.block_id),
            ),
            struct_temperature=float(self.config.struct_temperature),
        )

        # Build candidate sequences and score in one batch.
        records: list[tuple[str, str]] = []
        for c_idx, cand in enumerate(candidates):
            tokens = completed_tokens.detach().clone()
            for pos, tok in zip(editable, cand):
                tokens[int(pos)] = int(tok)
            records.append(
                (f"d2_b{int(block.block_id)}_c{c_idx}", decode_tokens(tokens))
            )
        window_risk_matrix = _score_window_risk_matrix(
            scorer=scorer, protein_id=str(protein_id), records=records
        )
        delta_R_argmax = (
            compute_local_risk_batch(
                window_risks=window_risk_matrix,
                omega_indices=omega,
                aggregation="LME",
            )
            - float(r_current)
        )
        argmax_best_delta_R_B = (
            float(delta_R_argmax.min()) if delta_R_argmax.size else float("nan")
        )

        candidate_indices = tuple(range(len(candidates)))
        ensemble_diag = compute_ensemble_diagnostics(
            argmax_delta_R=delta_R_argmax,
            ensemble_delta_R_by_candidate={},
        )
        if self.config.completion_ensemble_enabled and len(candidates) > 0:
            shortlist_size = min(
                int(self.config.completion_ensemble_rescore_top_m),
                int(len(candidates)),
            )
            candidate_indices = tuple(
                int(i) for i in np.argsort(delta_R_argmax)[:shortlist_size].tolist()
            )
            # §A2: the shortlist (top-m by local argmax ΔR) is rescored either by
            # the local block-span ensemble (default, byte-identical) or by the
            # terminal full-completion probe (P_cheap). Same return contract
            # (dict[cand_idx] -> per-sample ΔR), same head-logit units.
            if self.config.candidate_score_source == "terminal":
                ensemble_delta_by_idx = self._score_completion_terminal(
                    candidates=candidates,
                    candidate_indices=candidate_indices,
                    editable=editable,
                    block=block,
                    structural_logits=structural_logits,
                    x_t=x_t,
                    mask_token_id=int(mask_token_id),
                    completed_tokens=completed_tokens,
                    decode_tokens=decode_tokens,
                    scorer=scorer,
                    canonical_token_ids=canonical_token_ids,
                    protein_id=protein_id,
                    seed=seed,
                    design_idx=design_idx,
                    refresh_step=refresh_step,
                    r_current=float(r_current),
                    omega=omega,
                    K_P=int(self.config.candidate_terminal_K_P),
                )
            else:
                ensemble_delta_by_idx = self._score_completion_ensemble(
                    candidates=candidates,
                    candidate_indices=candidate_indices,
                    editable=editable,
                    block=block,
                    structural_logits=structural_logits,
                    x_t=x_t,
                    mask_token_id=int(mask_token_id),
                    completed_tokens=completed_tokens,
                    decode_tokens=decode_tokens,
                    scorer=scorer,
                    canonical_token_ids=canonical_token_ids,
                    protein_id=protein_id,
                    seed=seed,
                    design_idx=design_idx,
                    refresh_step=refresh_step,
                    r_current=float(r_current),
                    omega=omega,
                )
            ensemble_diag = compute_ensemble_diagnostics(
                argmax_delta_R=delta_R_argmax,
                ensemble_delta_R_by_candidate=ensemble_delta_by_idx,
            )
            delta_R_B = np.array(
                [ensemble_diag.mean_delta_R[int(i)] for i in candidate_indices],
                dtype=np.float64,
            )
            candidates_effective = tuple(candidates[int(i)] for i in candidate_indices)
        else:
            delta_R_B = delta_R_argmax
            candidates_effective = tuple(candidates)

        # GUARD (Stage A counterfactual review #7): sampled candidate
        # enumeration combined with an ensemble shortlist that strictly reduces
        # the candidate set makes the uniform-over-shortlist Q marginal (built
        # in the ``mode == "sampled"`` branch of the projection below) a biased
        # estimate of Q_B^safe — the shortlist is a delta_R-selected subset, not
        # an i.i.d. Q draw, so the log(pi/Q) correction is distorted. The
        # canonical Stage A preset never reaches sampled mode
        # (n_cart = prod(|K_i|) <= max_candidates_per_block → cartesian), so this
        # path is untested. Fail fast instead of silently emitting a biased
        # correction. Before widening the candidate budget so n_cart exceeds
        # max_candidates_per_block, project pi and Q over the FULL candidate set
        # (argmax delta_R for non-shortlisted candidates, ensemble mean for the
        # shortlist) so uniform-over-sample stays an unbiased Q estimate.
        if mode == "sampled" and len(candidates_effective) != len(candidates):
            raise NotImplementedError(
                "D2 sampled candidate mode combined with ensemble shortlisting "
                "is not supported in Stage A: the uniform-over-shortlist Q "
                "marginal is a biased estimate of Q_B^safe. Keep the candidate "
                "budget so enumeration stays cartesian "
                "(prod(|K_i|) <= max_candidates_per_block), or implement "
                "full-candidate-set projection before enabling this path."
            )

        ensemble_best_delta_R_B = (
            float(delta_R_B.min()) if delta_R_B.size else float("nan")
        )

        Q_B = compute_Q_B_per_candidate(
            candidates=candidates_effective,
            positions=editable,
            K_i_per_pos=K,
            struct_logits=structural_logits,
            struct_temperature=float(self.config.struct_temperature),
        )
        weights = compute_weights(
            mode=mode, Q_B=Q_B, delta_R_B=delta_R_B, beta=beta_used
        )
        ess = compute_ess(weights)
        ess_fraction = ess / float(len(candidates_effective)) if candidates_effective else 0.0
        feasible, best_dR = compute_feasibility(
            delta_R_B=delta_R_B,
            min_delta_R_improvement=float(self.config.min_delta_R_improvement),
        )
        mean_dR = float(delta_R_B.mean()) if delta_R_B.size else float("nan")

        # Low-ESS gate ⇒ g_ESS_candidates = 0 ⇒ rho_B_effective = 0 ⇒ no correction.
        if ess_fraction < float(self.config.min_ess_fraction):
            return D2BlockOutcome(
                block_id=int(block.block_id),
                omega_indices=omega,
                editable_positions=tuple(editable),
                K_i_per_pos=K,
                candidate_mode=mode,
                candidate_count=len(candidates_effective),
                delta_R_B=tuple(float(x) for x in delta_R_B.tolist()),
                Q_B=tuple(float(x) for x in Q_B.tolist()),
                weights_unnormalized=tuple(float(x) for x in weights.tolist()),
                ess=float(ess),
                ess_fraction=float(ess_fraction),
                g_ESS_candidates=0.0,
                feasible=feasible,
                best_delta_R_B=float(best_dR),
                mean_delta_R_B=float(mean_dR),
                rho_B_effective=0.0,
                corrected_positions=(),
                delta_logit={},
                skipped_reason="low_ess",
                r_current=float(r_current),
                safe_support_sizes={int(p): len(K[int(p)]) for p in editable},
                candidate_count_argmax=len(candidates),
                candidate_count_ensemble=len(candidates_effective),
                argmax_best_delta_R_B=float(argmax_best_delta_R_B),
                ensemble_best_delta_R_B=float(ensemble_best_delta_R_B),
                argmax_to_ensemble_rank_flip_rate=float(ensemble_diag.rank_flip_rate),
                argmax_to_ensemble_rank_flip_flag=bool(ensemble_diag.rank_flip_rate > 0.0),
                ensemble_delta_R_std=_mean_optional(ensemble_diag.std_delta_R),
                ensemble_sign_consistency=_mean_optional(ensemble_diag.sign_consistency),
                context_pnll=float(context_pnll),
                g_pnll=float(g_pnll),
                context_jsd=context_jsd,
                g_stability=float(g_stability),
            )
        g_ESS_candidates = 1.0

        if not feasible:
            return D2BlockOutcome(
                block_id=int(block.block_id),
                omega_indices=omega,
                editable_positions=tuple(editable),
                K_i_per_pos=K,
                candidate_mode=mode,
                candidate_count=len(candidates_effective),
                delta_R_B=tuple(float(x) for x in delta_R_B.tolist()),
                Q_B=tuple(float(x) for x in Q_B.tolist()),
                weights_unnormalized=tuple(float(x) for x in weights.tolist()),
                ess=float(ess),
                ess_fraction=float(ess_fraction),
                g_ESS_candidates=float(g_ESS_candidates),
                feasible=False,
                best_delta_R_B=float(best_dR),
                mean_delta_R_B=float(mean_dR),
                rho_B_effective=0.0,
                corrected_positions=(),
                delta_logit={},
                skipped_reason="not_feasible",
                r_current=float(r_current),
                safe_support_sizes={int(p): len(K[int(p)]) for p in editable},
                candidate_count_argmax=len(candidates),
                candidate_count_ensemble=len(candidates_effective),
                argmax_best_delta_R_B=float(argmax_best_delta_R_B),
                ensemble_best_delta_R_B=float(ensemble_best_delta_R_B),
                argmax_to_ensemble_rank_flip_rate=float(ensemble_diag.rank_flip_rate),
                argmax_to_ensemble_rank_flip_flag=bool(ensemble_diag.rank_flip_rate > 0.0),
                ensemble_delta_R_std=_mean_optional(ensemble_diag.std_delta_R),
                ensemble_sign_consistency=_mean_optional(ensemble_diag.sign_consistency),
                context_pnll=float(context_pnll),
                g_pnll=float(g_pnll),
                context_jsd=context_jsd,
                g_stability=float(g_stability),
            )

        if beta_used == 0.0:
            # Reached either by a configured beta=0 or by a Stage C.1 g_GR that
            # drives beta_eff to 0 (low-burden suppression): both yield the
            # uncorrected / beta=0 posterior. The optional "global_pressure_zero"
            # skip reason is intentionally deferred (PLAN_RF_UNI_CTRL.md C1.5).
            return D2BlockOutcome(
                block_id=int(block.block_id),
                omega_indices=omega,
                editable_positions=tuple(editable),
                K_i_per_pos=K,
                candidate_mode=mode,
                candidate_count=len(candidates_effective),
                delta_R_B=tuple(float(x) for x in delta_R_B.tolist()),
                Q_B=tuple(float(x) for x in Q_B.tolist()),
                weights_unnormalized=tuple(float(x) for x in weights.tolist()),
                ess=float(ess),
                ess_fraction=float(ess_fraction),
                g_ESS_candidates=float(g_ESS_candidates),
                feasible=True,
                best_delta_R_B=float(best_dR),
                mean_delta_R_B=float(mean_dR),
                rho_B_effective=0.0,
                corrected_positions=(),
                delta_logit={},
                skipped_reason="beta_zero",
                r_current=float(r_current),
                safe_support_sizes={int(p): len(K[int(p)]) for p in editable},
                candidate_count_argmax=len(candidates),
                candidate_count_ensemble=len(candidates_effective),
                argmax_best_delta_R_B=float(argmax_best_delta_R_B),
                ensemble_best_delta_R_B=float(ensemble_best_delta_R_B),
                argmax_to_ensemble_rank_flip_rate=float(ensemble_diag.rank_flip_rate),
                argmax_to_ensemble_rank_flip_flag=bool(ensemble_diag.rank_flip_rate > 0.0),
                ensemble_delta_R_std=_mean_optional(ensemble_diag.std_delta_R),
                ensemble_sign_consistency=_mean_optional(ensemble_diag.sign_consistency),
                context_pnll=float(context_pnll),
                g_pnll=float(g_pnll),
                context_jsd=context_jsd,
                g_stability=float(g_stability),
            )

        w_sum = float(weights.sum())
        w_norm = weights / w_sum if w_sum > 0.0 else weights
        pi = project_marginals(
            candidates=candidates_effective,
            weights_normalized=w_norm,
            positions=editable,
            K_i_per_pos=K,
        )
        # Q marginal source depends on enumeration mode:
        # * Cartesian: each candidate appears exactly once with weight Q_B,
        #   so the projection is exactly the analytic per-position softmax
        #   over K_i. Use the analytic form — it is exact and avoids small
        #   numerical drift from re-projection.
        # * Sampled: candidates are i.i.d. draws from Q_B; the corresponding
        #   Q marginal is the empirical projection of the SAME sample with
        #   uniform weights. Using the analytic per-position softmax would
        #   make ``pi`` empirical and ``Q`` analytical, so even at beta=0
        #   the log-ratio would not vanish at finite sample sizes. Project
        #   both ``pi`` and ``Q`` from the same draws so beta=0 yields a
        #   strict null (PLAN_RF.md §"D2 step behavior" 19).
        if mode == "sampled":
            # ``candidates_effective`` here is the FULL i.i.d. sample: the
            # review-#7 guard above rejects sampled mode whenever the ensemble
            # shortlist reduced the candidate set, so uniform-over-sample is an
            # unbiased empirical estimate of the Q_B^safe marginal (projecting
            # both pi and Q from the same draws also keeps the beta=0 null
            # strict, PLAN_RF.md §"D2 step behavior" 19).
            n_cand = len(candidates_effective)
            uniform = (
                np.full(n_cand, 1.0 / float(n_cand), dtype=np.float64)
                if n_cand > 0
                else np.zeros(0, dtype=np.float64)
            )
            Q_marg = project_marginals(
                candidates=candidates_effective,
                weights_normalized=uniform,
                positions=editable,
                K_i_per_pos=K,
            )
        elif len(candidates_effective) != len(candidates):
            q_sum = float(Q_B.sum())
            q_norm = Q_B / q_sum if q_sum > 0.0 else Q_B
            Q_marg = project_marginals(
                candidates=candidates_effective,
                weights_normalized=q_norm,
                positions=editable,
                K_i_per_pos=K,
            )
        else:
            Q_marg = _per_position_Q_marginal(
                positions=editable,
                K_i_per_pos=K,
                struct_logits=structural_logits,
                struct_temperature=float(self.config.struct_temperature),
            )

        rho_B_effective = (
            float(block.g_time)
            * float(g_pnll)
            * float(g_stability)
            * float(g_ESS_candidates)
        )
        rho_B_effective = max(0.0, min(1.0, rho_B_effective))

        delta = compute_delta_logit(
            pi_marginals=pi,
            Q_marginals=Q_marg,
            eta=float(self.config.eta),
            rho_B=float(rho_B_effective),
            epsilon=float(self.config.epsilon),
            max_abs_shift=float(self.config.max_abs_logit_shift),
        )

        return D2BlockOutcome(
            block_id=int(block.block_id),
            omega_indices=omega,
            editable_positions=tuple(editable),
            K_i_per_pos=K,
            candidate_mode=mode,
            candidate_count=len(candidates_effective),
            delta_R_B=tuple(float(x) for x in delta_R_B.tolist()),
            Q_B=tuple(float(x) for x in Q_B.tolist()),
            weights_unnormalized=tuple(float(x) for x in weights.tolist()),
            ess=float(ess),
            ess_fraction=float(ess_fraction),
            g_ESS_candidates=float(g_ESS_candidates),
            feasible=True,
            best_delta_R_B=float(best_dR),
            mean_delta_R_B=float(mean_dR),
            rho_B_effective=float(rho_B_effective),
            corrected_positions=tuple(editable),
            delta_logit=delta,
            skipped_reason=None,
            r_current=float(r_current),
            safe_support_sizes={int(p): len(K[int(p)]) for p in editable},
            candidate_count_argmax=len(candidates),
            candidate_count_ensemble=len(candidates_effective),
            argmax_best_delta_R_B=float(argmax_best_delta_R_B),
            ensemble_best_delta_R_B=float(ensemble_best_delta_R_B),
            argmax_to_ensemble_rank_flip_rate=float(ensemble_diag.rank_flip_rate),
            argmax_to_ensemble_rank_flip_flag=bool(ensemble_diag.rank_flip_rate > 0.0),
            ensemble_delta_R_std=_mean_optional(ensemble_diag.std_delta_R),
            ensemble_sign_consistency=_mean_optional(ensemble_diag.sign_consistency),
            context_pnll=float(context_pnll),
            g_pnll=float(g_pnll),
            context_jsd=context_jsd,
            g_stability=float(g_stability),
        )


def _empty_block_outcome(
    *,
    block_id: int,
    omega: tuple[int, ...],
    reason: str,
    r_current: float = float("nan"),
    context_pnll: float | None = None,
    g_pnll: float | None = None,
) -> D2BlockOutcome:
    return D2BlockOutcome(
        block_id=block_id,
        omega_indices=omega,
        editable_positions=(),
        K_i_per_pos={},
        candidate_mode="",
        candidate_count=0,
        delta_R_B=(),
        Q_B=(),
        weights_unnormalized=(),
        ess=0.0,
        ess_fraction=0.0,
        g_ESS_candidates=0.0,
        feasible=False,
        best_delta_R_B=float("nan"),
        mean_delta_R_B=float("nan"),
        rho_B_effective=0.0,
        corrected_positions=(),
        delta_logit={},
        skipped_reason=reason,
        r_current=float(r_current),
        context_pnll=context_pnll,
        g_pnll=g_pnll,
    )


def _mean_optional(values: dict[int, float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(list(values.values()), dtype=np.float64)
    if arr.size == 0:
        return None
    return float(arr.mean())


def _per_position_Q_marginal(
    *,
    positions: Sequence[int],
    K_i_per_pos: dict[int, tuple[int, ...]],
    struct_logits: torch.Tensor,
    struct_temperature: float = 1.0,
) -> dict[int, dict[int, float]]:
    out: dict[int, dict[int, float]] = {}
    for p in positions:
        K = K_i_per_pos[int(p)]
        K_arr = np.asarray(K, dtype=np.int64)
        logits_K = (
            struct_logits[int(p), torch.from_numpy(K_arr)]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            / float(struct_temperature)
        )
        logits_K = logits_K - float(logits_K.max())
        probs = np.exp(logits_K)
        probs = probs / probs.sum()
        out[int(p)] = {int(K_arr[j]): float(probs[j]) for j in range(K_arr.shape[0])}
    return out
