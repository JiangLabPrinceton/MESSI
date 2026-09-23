"""Base schedules for position-dependent discrete flow matching."""

from __future__ import annotations

import numpy as np


_VALID_FORMS = {"linear", "cosine", "cubic"}


def _as_array(t: float | np.ndarray) -> np.ndarray:
    return np.asarray(t, dtype=np.float64)


def base_schedule_value(t: float | np.ndarray, form: str) -> np.ndarray:
    """Return ``kappa_base(t)`` for the configured schedule form."""
    tt = _as_array(t)
    _validate_form(form)
    if form == "linear":
        return tt
    if form == "cosine":
        return 1.0 - np.cos(0.5 * np.pi * tt)
    return 3.0 * np.square(tt) - 2.0 * np.power(tt, 3.0)


def base_schedule_derivative(t: float | np.ndarray, form: str) -> np.ndarray:
    """Return the analytical derivative of ``kappa_base(t)``."""
    tt = _as_array(t)
    _validate_form(form)
    if form == "linear":
        return np.ones_like(tt, dtype=np.float64)
    if form == "cosine":
        return 0.5 * np.pi * np.sin(0.5 * np.pi * tt)
    return 6.0 * tt - 6.0 * np.square(tt)


def positionwise_unmask_probabilities(
    *,
    t: float,
    dt: float,
    g_values: np.ndarray,
    base_form: str,
) -> np.ndarray:
    """Return per-position Bernoulli unmask probabilities for one DFM step."""
    g = np.asarray(g_values, dtype=np.float64)
    if g.ndim != 1:
        raise ValueError("g_values must be one-dimensional")
    if np.any(g < 1.0):
        raise ValueError("g_values must be >= 1")

    base = float(base_schedule_value(t, base_form))
    base_prime = float(base_schedule_derivative(t, base_form))

    if base <= 0.0:
        derivative = np.where(np.isclose(g, 1.0), base_prime, 0.0)
        kappa = np.zeros_like(g, dtype=np.float64)
    else:
        log_base = max(np.log(base), -50.0)
        kappa = np.exp(g * log_base)
        derivative = g * np.exp((g - 1.0) * log_base) * base_prime

    denom = np.maximum(1.0 - kappa, 1e-12)
    rate = derivative / denom
    probs = np.clip(rate * dt, 0.0, 1.0)
    return probs.astype(np.float64, copy=False)


def _validate_form(form: str) -> None:
    if form not in _VALID_FORMS:
        raise ValueError(
            f"unsupported base_form={form!r}; expected one of {sorted(_VALID_FORMS)}"
        )
