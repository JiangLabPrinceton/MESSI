"""Normalize Protenix v2 inference output into the shared refold cache.

Protenix writes, per fold ``name``:
  ``<out>/<name>/seed_<seed>/predictions/<name>_sample_<rank>.cif`` (mmCIF, per-atom
  pLDDT on the 0-100 scale in the B-factor column) and a matching
  ``<name>_summary_confidence_sample_<rank>.json`` (``plddt`` = mean pLDDT 0-100,
  ``ranking_score`` = model-selection scalar). ``_sample_0`` is the best sample WITHIN
  a seed (ranking-sorted); across seeds the global best is the seed whose ``_sample_0``
  has the max ``ranking_score`` (there is no cross-seed ranking file).

Protenix is invoked across a SLURM boundary (its own env / the AF3 interpreter); this
module only READS the produced files and normalizes them, so it imports nothing from
Protenix. It reuses ``refold_normalize`` for the mmCIF->single-chain-PDB + 0-100 pLDDT
sidecar conversion, and the Protenix fold ``name`` is set to the ``cache_key`` so the
on-disk name already equals the cache key.
"""

from __future__ import annotations

import glob
import json
import os

from inverse_folding.evaluation.refold_normalize import (
    mean_ca_plddt_from_pdb,
    normalize_to_cache,
)


def _summary_for_cif(cif_path: str) -> str:
    """The ``_summary_confidence_sample_0.json`` next to a ``_sample_0.cif``."""
    return cif_path.replace("_sample_0.cif", "_summary_confidence_sample_0.json")


def read_ranking_score(summary_path: str) -> float:
    with open(summary_path) as handle:
        return float(json.load(handle)["ranking_score"])


def read_mean_plddt(summary_path: str) -> float:
    """Mean pLDDT (0-100) from a Protenix summary-confidence JSON."""
    with open(summary_path) as handle:
        return float(json.load(handle)["plddt"])


def find_best_sample(protenix_out_dir: str, name: str) -> tuple[str, str]:
    """Return (best_cif_path, best_summary_json_path) = global max ranking_score.

    Fail-fast (FileNotFoundError) if Protenix produced no ranked sample for ``name``.
    """
    pattern = os.path.join(protenix_out_dir, name, "seed_*", "predictions", f"{name}_sample_0.cif")
    cifs = sorted(glob.glob(pattern))
    if not cifs:
        raise FileNotFoundError(
            f"protenix: no prediction found for '{name}' under {protenix_out_dir} "
            f"(expected {pattern})"
        )
    best_cif = None
    best_summary = None
    best_score = float("-inf")
    for cif in cifs:
        summary = _summary_for_cif(cif)
        if not os.path.isfile(summary):
            raise FileNotFoundError(f"protenix: missing summary json for {cif}: {summary}")
        score = read_ranking_score(summary)
        if score > best_score:
            best_score, best_cif, best_summary = score, cif, summary
    return best_cif, best_summary


def normalize_protenix_to_cache(
    protenix_out_dir: str,
    name: str,
    *,
    cache_dir: str,
    key: str,
    protein_only: bool = False,
) -> dict[str, str]:
    """Pick the global-best Protenix sample for ``name`` and write it into the cache.

    Writes ``<cache_dir>/<key>.pdb`` (single-chain) + ``<key>.plddt`` (0-100).
    """
    best_cif, best_summary = find_best_sample(protenix_out_dir, name)
    mean_plddt = read_mean_plddt(best_summary)  # 0-100 already
    result = normalize_to_cache(
        cache_dir,
        key,
        cif_path=best_cif,
        mean_plddt=mean_plddt,
        native_scale="0-100",
        protein_only=protein_only,
    )
    # Protenix's summary is an all-atom/complex aggregate, while canonical v2
    # validates and reports the per-residue CA mean from the protein cache PDB.
    ca_mean = mean_ca_plddt_from_pdb(result["pdb_path"])
    if ca_mean is not None:
        with open(result["plddt_path"], "w") as handle:
            handle.write(f"{ca_mean:.4f}")
    return result
