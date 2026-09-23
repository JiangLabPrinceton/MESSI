"""Injected oracle bundle, provenance-keyed structure cache, and the absolute structure gate.

Head-only (§0.3(1)): the bundle has NO ``nmp_fn`` field — NetMHCIIpan has no runtime role.
Real Head/structure wiring lives in the driver (F8); this module is torch-free and defines
only the contracts + the deterministic feasibility logic that the evaluator (F4) consumes.

- ``head_fn(protein_id, sequences) -> list[HeadScore]`` (parallel to input; caller dedups)
- ``struct_fn(protein_id, sequence) -> StructureMetrics`` (ESMFold refold + selected geometry)
"""
from __future__ import annotations

import math
from collections import namedtuple
from typing import TYPE_CHECKING

from .state import sequence_md5


def _finite(value) -> bool:
    return value is not None and math.isfinite(float(value))

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import FusionConfig

# Head-only oracle bundle — deliberately WITHOUT nmp_fn (the refiner's Oracles namedtuple
# carries a mandatory nmp_fn; copying it verbatim would force a StandaloneRunner import).
FusionOracles = namedtuple("FusionOracles", ["head_fn", "struct_fn"])


class StructureCache:
    """``(protein_id, seq_md5, backbone_digest, backend, config_hash)``-keyed refold cache.

    A cache hit is valid only if ALL provenance fields match (§1.4); a differing target
    backbone, refold backend, or structure-config hash forces recomputation rather than a
    silent stale reuse.
    """

    def __init__(self, *, backbone_digest: str, backend: str, config_hash: str) -> None:
        self.backbone_digest = str(backbone_digest)
        self.backend = str(backend)
        self.config_hash = str(config_hash)
        self._store: dict = {}
        self.hits = 0
        self.misses = 0

    def _key(self, protein_id: str, sequence: str):
        return (str(protein_id), sequence_md5(sequence),
                self.backbone_digest, self.backend, self.config_hash)

    def evaluate(self, protein_id: str, sequence: str, struct_fn):
        """Return ``(metrics, was_cache_hit)``; compute via ``struct_fn`` on a miss."""
        key = self._key(protein_id, sequence)
        if key in self._store:
            self.hits += 1
            return self._store[key], True
        self.misses += 1
        metrics = struct_fn(protein_id, sequence)
        self._store[key] = metrics
        return metrics, False


def structure_feasible(metrics, config: "FusionConfig", *, has_active_site: bool) -> tuple[bool, str]:
    """Absolute target-backbone gate (§1.4). NOT the refiner's seed-relative gate.

    scTM floor is always enforced. Unless the active-site metric is explicitly ``none``, an
    anchored protein also uses the configured legacy C-alpha shell or all-heavy-side-chain gate.
    Missing/incomplete enabled metrics fail closed. A no-anchor protein skips that gate (N/A).
    """
    st = config.structure
    scTM = getattr(metrics, "scTM", None)
    if not _finite(scTM):  # None / NaN / inf must fail closed, never slip past `<`
        return False, f"scTM not finite: {scTM}"
    if float(scTM) < st.scTM_min:
        return False, f"scTM {scTM} < scTM_min {st.scTM_min}"
    if has_active_site and st.active_site_metric != "none":
        if st.active_site_metric == "legacy_ca_shell":
            if st.active_site_RMSD_max is None:
                return False, (
                    "anchored protein but active_site_RMSD_max not configured (fail closed)"
                )
            asr = getattr(metrics, "active_site_RMSD", None)
            if not _finite(asr):
                return False, (
                    "active_site_RMSD unavailable/not finite on anchored protein: "
                    f"{asr}"
                )
            if float(asr) > st.active_site_RMSD_max:
                return False, f"active_site_RMSD {asr} > {st.active_site_RMSD_max}"
        elif st.active_site_metric == "sidechain_max_anchor":
            if getattr(metrics, "active_site_complete", None) is not True:
                return False, "side-chain active-site metrics incomplete"
            sidechain_rmsd = getattr(metrics, "max_anchor_sidechain_RMSD", None)
            if not _finite(sidechain_rmsd):
                return False, (
                    "max_anchor_sidechain_RMSD unavailable/not finite on anchored protein: "
                    f"{sidechain_rmsd}"
                )
            if float(sidechain_rmsd) > st.max_anchor_sidechain_RMSD_max:
                return False, (
                    f"max_anchor_sidechain_RMSD {sidechain_rmsd} > "
                    f"{st.max_anchor_sidechain_RMSD_max}"
                )
        else:  # config loader prevents this; retain a fail-closed runtime boundary.
            return False, f"unknown active_site_metric: {st.active_site_metric}"
    if st.scRMSD_max is not None:
        scr = getattr(metrics, "scRMSD", None)
        if not _finite(scr):
            return False, f"scRMSD unavailable/not finite but scRMSD_max configured: {scr}"
        if float(scr) > st.scRMSD_max:
            return False, f"scRMSD {scr} > {st.scRMSD_max}"
    return True, "ok"
