"""Tier 2 pre-screening: dual-scorer agreement filter.

Checks both NetMHCIIpan and epitope head for DRB1*07:01
presentation risk. Only candidates passing both thresholds
proceed to diversity sampling.

Frozen thresholds (PLAN_DATA_SEL §L3):
  - NMP: n_strong_windows >= 5 (windows with %Rank_EL < 2%)
  - Head: global_risk >= batch median (risk_median_threshold)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PrescreenResult:
    """Result of applying Tier 2 pre-screening filters."""

    passed: bool
    reason: str


def apply_tier2_filters(
    nmp_result: dict,
    head_result: dict,
    min_strong_windows: int = 5,
    risk_median_threshold: float = 0.5,
) -> PrescreenResult:
    """Apply dual-scorer threshold filter for Tier 2 candidate selection.

    Args:
        nmp_result: dict with 'n_strong_windows' and 'mean_best_rank'.
        head_result: dict with 'global_risk' and 'n_hotspot_positions'.
        min_strong_windows: minimum NetMHCIIpan strong binder windows.
        risk_median_threshold: minimum epitope head global risk.

    Returns:
        PrescreenResult with pass/fail decision and reason.
    """
    n_strong = nmp_result["n_strong_windows"]
    global_risk = head_result["global_risk"]

    if n_strong < min_strong_windows:
        return PrescreenResult(
            passed=False,
            reason=f"NMP n_strong_windows={n_strong} < {min_strong_windows}",
        )

    if global_risk < risk_median_threshold:
        return PrescreenResult(
            passed=False,
            reason=f"Head global_risk={global_risk:.3f} < {risk_median_threshold:.3f}",
        )

    return PrescreenResult(
        passed=True,
        reason="passed both NMP and head thresholds",
    )
