"""V2F3a: the sampler-neutral arbitrary-token log-probability primitive (PLAN §3.1).

PLAN §3.1 requires, before the segment API is built, one shared primitive

    l_j = log softmax(z_j / T)_{a_j}

that "accepts raw base logits, canonical AA20 token IDs, and a finite positive sampler temperature;
it draws no RNG and adds no DFE".

V2 needs it because projection injects identities the denoiser never sampled -- a selected endpoint
byte, or a carried source identity -- which arrive at ``pending_assimilation`` with no active score.
Scoring them any other way would let an injected identity compete for survival on a scale the
natural tokens were never measured on (PLAN §7.2: "assimilation score differs from categorical
chosen-token log-probability at temperature not equal to one").

The quantity is defined by the V1 sampler and is reproduced here rather than refactored into it:
``sampler._sample_categorical`` computes ``torch.log_softmax(logits, dim=-1)`` and gathers the drawn
row, with the temperature division applied by its callers immediately before the call.  Changing
that function to delegate here would modify published V1 behaviour for no gain, so instead
``test_fusion_v2_token_logprob`` asserts byte-equality between the two on random inputs at several
temperatures.  Agreement is proven, not assumed.

PLAN §3.1 also forbids importing the D3 chosen-token helper: that helper is controller-family
evidence and does not expose the sampler-temperature contract.  Nothing here imports it.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["token_logprob"]


def token_logprob(
    logits: torch.Tensor,
    token_ids,
    *,
    temperature: float,
) -> np.ndarray:
    """Log-probability of an ARBITRARY token under the sampler's own normalization.

    Pure: takes no generator, mutates nothing, adds no DFE.  Inserting a call into a trajectory
    therefore cannot perturb replay.

    Args:
        logits: ``(rows, vocab)`` raw denoiser logits, BEFORE any temperature division.
        token_ids: one token id per row.
        temperature: the sampler temperature actually in force.  Required -- there is no default,
            because a wrongly assumed temperature silently rescales every assimilated score.

    Returns:
        ``(rows,)`` float64 log-probabilities, the dtype the sampler stores.
    """
    scale = float(temperature)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")

    if logits.ndim != 2:
        raise ValueError(f"logits must be (rows, vocab), got shape {tuple(logits.shape)}")
    ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    rows, vocab = int(logits.shape[0]), int(logits.shape[-1])
    if ids.shape[0] != rows:
        raise ValueError(
            f"token_ids has {ids.shape[0]} entries but logits has {rows} rows; one token is "
            "scored per row"
        )
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= vocab):
        raise ValueError(
            f"token_ids must lie in [0, {vocab}), got range "
            f"[{int(ids.min())}, {int(ids.max())}]"
        )

    log_probs = torch.log_softmax(logits / scale, dim=-1).cpu().numpy()
    return log_probs[np.arange(rows), ids].astype(np.float64, copy=False)
