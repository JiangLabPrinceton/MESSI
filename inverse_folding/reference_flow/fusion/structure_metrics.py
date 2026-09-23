"""Legacy active-site C-alpha shell RMSD producer — compatibility mode only.

``active_site_spatial_shell6A_ca_rmsd`` = ``sqrt(mean(sc_ca_distance^2))`` over the 6 Å spatial
shell (residues whose reference Cα is within ``radius`` of any anchor Cα), where the per-residue
``sc_ca_distance`` comes from a single global full-trace Kabsch superposition
(``evaluation.sc_rmsd.compute_ca_self_consistency`` — NOT a local re-superposition). Shell
membership depends only on the reference structure + anchors, so it is computed once per protein.

This module is pure (numpy only, no torch/PDB I/O): the driver parses the reference Cα trace
and runs the refold+Kabsch pass, then calls these helpers. That keeps the geometry unit-testable.
"""
from __future__ import annotations

import numpy as np


def shell_indices_from_ca(ca_coords, anchor_indices, radius: float) -> frozenset[int]:
    """Residues whose Cα is within ``radius`` of ANY anchor Cα (anchors included, distance 0).

    ``ca_coords`` is an ``(L, 3)`` array of reference Cα coordinates; ``anchor_indices`` are
    0-based residue indices. Returns the 0-based shell index set (empty if no anchors).
    """
    anchors = sorted(int(a) for a in anchor_indices)
    if not anchors:
        return frozenset()
    ca = np.asarray(ca_coords, dtype=float)
    if ca.ndim != 2 or ca.shape[1] != 3:
        raise ValueError(f"ca_coords must be (L,3), got {ca.shape}")
    if anchors[0] < 0:  # negative index would silently wrap under numpy fancy-indexing
        raise ValueError(f"anchor index {anchors[0]} is negative")
    if max(anchors) >= len(ca):
        raise ValueError(f"anchor index {max(anchors)} out of range for {len(ca)} residues")
    anchor_ca = ca[anchors]                                   # (A, 3)
    dist = np.linalg.norm(ca[:, None, :] - anchor_ca[None, :, :], axis=2)  # (L, A)
    within = (dist <= float(radius)).any(axis=1)
    return frozenset(int(i) for i in np.nonzero(within)[0])


def shell_rmsd_from_rows(rows, shell_indices) -> float | None:
    """RMS of per-residue ``sc_ca_distance`` over the shell subset (§1.4).

    ``rows`` is the per-residue table from ``compute_ca_self_consistency`` (each row has
    ``residue_idx`` 0-based and ``sc_ca_distance`` in Å). Returns ``None`` when the shell is
    empty (no anchors) so the caller can treat the gate as N/A.
    """
    shell = {int(i) for i in shell_indices}
    if not shell:
        return None
    dists = [float(r["sc_ca_distance"]) for r in rows if int(r["residue_idx"]) in shell]
    if not dists:
        return None
    return float(np.sqrt(np.mean(np.square(dists))))
