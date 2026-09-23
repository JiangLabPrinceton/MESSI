"""Complete-state Head objective, target responsibility, and new-hotspot gate (§1.2/§1.3).

Whole-sequence ``global_risk`` drives selection; local LME delta over the target register is
telemetry only; the parent-relative off-halo new-hotspot burden is the hard admissibility
signal. Parent and child window coordinate grids must align exactly (same length + k-range in
inverse folding), otherwise evaluation fails rather than silently comparing mismatched windows.

Consumes duck-typed Head objects: a score exposes ``.windows`` (each with
``start_0b``/``end_0b``/``k``/``z``) and ``.global_risk``. Pure Python — no torch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class ObjectiveError(ValueError):
    """Raised on a missing global_risk or a parent/child window-coordinate mismatch."""


@dataclass(frozen=True)
class TargetRegister:
    start_0b: int
    end_0b: int
    editable_positions: tuple[int, ...]
    z: float


@dataclass(frozen=True)
class NewHotspot:
    max_increase: float
    positive_mass: float
    positive_count: int


# --------------------------------------------------------------------------- #
# whole-sequence objective
# --------------------------------------------------------------------------- #
def global_risk_of(head_score) -> float:
    """R_H(x) = head_score.global_risk. A missing (None) or non-finite value is a hard error (§1.2)."""
    gr = getattr(head_score, "global_risk", None)
    if gr is None:
        raise ObjectiveError(
            "global_risk is None — a dynamically scored HeadScore must populate it "
            "(static-cache windows-only scores are not valid selection inputs)")
    if not math.isfinite(float(gr)):
        raise ObjectiveError(f"global_risk must be finite, got {gr!r}")
    return float(gr)


# --------------------------------------------------------------------------- #
# target register responsibility
# --------------------------------------------------------------------------- #
def _sorted_windows(head_score):
    # highest raw risk first; deterministic ties by start_0b, end_0b, k ascending.
    # Validate z finiteness here too (consistent fail-closed contract) — a corrupted z must
    # not silently steer target selection.
    for w in head_score.windows:
        _finite_z(w)
    return sorted(head_score.windows, key=lambda w: (-w.z, w.start_0b, w.end_0b, w.k))


def select_target_register(head_score, *, anchors: set[int]) -> TargetRegister | None:
    """Highest-z window that has at least one editable (non-anchor) position (§1.2).

    Anchors are excluded from the editable set but not from Head responsibility; if the
    top window is fully anchored the next window is tried. Returns None when no window has
    an editable position (caller emits ``stalled_no_editable_target``).
    """
    for w in _sorted_windows(head_score):
        editable = tuple(p for p in range(w.start_0b, w.end_0b) if p not in anchors)
        if editable:
            return TargetRegister(start_0b=w.start_0b, end_0b=w.end_0b,
                                  editable_positions=editable, z=float(w.z))
    return None


# --------------------------------------------------------------------------- #
# window alignment
# --------------------------------------------------------------------------- #
def _coord(w):
    return (w.start_0b, w.end_0b, w.k)


def _finite_z(w) -> float:
    z = float(w.z)
    if not math.isfinite(z):  # a corrupted z must not silently become max(0, nan)=0
        raise ObjectiveError(f"non-finite window z at {_coord(w)}: {w.z!r}")
    return z


def _aligned_z(child, parent):
    """Return {coord: (z_child, z_parent)} requiring identical coordinate sets + finite z."""
    cmap = {_coord(w): _finite_z(w) for w in child.windows}
    pmap = {_coord(w): _finite_z(w) for w in parent.windows}
    if set(cmap) != set(pmap):
        raise ObjectiveError(
            "parent/child window coordinate sets differ; cannot align windows exactly")
    return {c: (cmap[c], pmap[c]) for c in cmap}


def _lme(values) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    m = max(vals)
    return m + math.log(math.fsum(math.exp(v - m) for v in vals) / len(vals))


def _overlaps(coord, start, end) -> bool:
    s, e, _k = coord
    return s < end and e > start


# --------------------------------------------------------------------------- #
# local target delta (telemetry) and off-halo new-hotspot (gate signal)
# --------------------------------------------------------------------------- #
def local_target_delta(child, parent, *, target_start: int, target_end: int) -> float:
    """ΔR_local = LME over windows overlapping the target of z(child) minus z(parent) (§1.2)."""
    aligned = _aligned_z(child, parent)
    child_z = [zc for c, (zc, _zp) in aligned.items() if _overlaps(c, target_start, target_end)]
    parent_z = [zp for c, (_zc, zp) in aligned.items() if _overlaps(c, target_start, target_end)]
    return _lme(child_z) - _lme(parent_z)


def new_hotspot(child, parent, *, halo_start: int, halo_end: int) -> NewHotspot:
    """Parent-relative off-halo hotspot: N_H = max_{w∩H=∅} max(0, z_w(y) - z_w(x)) (§1.3)."""
    aligned = _aligned_z(child, parent)
    deltas = [max(0.0, zc - zp)
              for c, (zc, zp) in aligned.items()
              if not _overlaps(c, halo_start, halo_end)]
    positives = [d for d in deltas if d > 0.0]
    return NewHotspot(
        max_increase=max(deltas) if deltas else 0.0,
        positive_mass=math.fsum(positives),
        positive_count=len(positives),
    )


# --- Public aliases for fusion_v2 reuse (additive; zero behavior change) ---------------------
# doc/FUSION_V2.md section 4.3 names the window-alignment primitive as the one reusable part of the
# v0 hotspot machinery. new_hotspot() itself is deliberately NOT reused: it is off-halo and
# rolling-parent, and its max_increase falls back to 0.0 on an empty delta list (fail-open).
aligned_window_z = _aligned_z
finite_window_z = _finite_z
window_coord = _coord
