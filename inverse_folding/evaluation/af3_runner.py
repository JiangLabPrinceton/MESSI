"""Normalize AlphaFold3 (official DeepMind AF3 v3.0.3) inference output into the
shared refold cache.

AF3 writes, per fold ``name`` (``sanitised_name`` keeps alnum/``_``/``-``/``.`` and does
NOT lowercase, so it equals the ``cache_key``):
  ``<out>/<name>/<name>_model.cif``          top-ranked best model (mmCIF, single chain
                                             for a monomer, per-atom pLDDT 0-100 in B-factor)
  ``<out>/<name>/<name>_confidences.json``   ``atom_plddts`` (0-100), pae, ...
  ``<out>/<name>/<name>_summary_confidences.json``  ptm / iptm / ranking_score

AF3 is invoked across a SLURM boundary (its own interpreter); this module only READS the
produced files and reuses ``refold_normalize`` for the mmCIF->single-chain-PDB + 0-100
pLDDT-sidecar conversion.
"""

from __future__ import annotations

import json
import os

from inverse_folding.evaluation.refold_normalize import (
    mean_ca_plddt_from_pdb,
    normalize_to_cache,
)


def read_mean_plddt_af3(confidences_path: str) -> float:
    """Mean pLDDT (0-100) = mean of ``atom_plddts`` in an AF3 confidences JSON."""
    with open(confidences_path) as handle:
        data = json.load(handle)
    arr = data["atom_plddts"]
    if not arr:
        raise ValueError(f"af3: empty atom_plddts in {confidences_path}")
    return float(sum(arr) / len(arr))


def normalize_af3_to_cache(
    af3_out_dir: str, name: str, *, cache_dir: str, key: str
) -> dict[str, str]:
    """Write AF3's best model for ``name`` into the cache as ``<key>.pdb`` + ``.plddt`` (0-100).

    Fail-fast (FileNotFoundError) if AF3 produced no model / confidences for ``name``.
    """
    fold_dir = os.path.join(af3_out_dir, name)
    cif = os.path.join(fold_dir, f"{name}_model.cif")
    conf = os.path.join(fold_dir, f"{name}_confidences.json")
    if not os.path.isfile(cif):
        raise FileNotFoundError(
            f"af3: no model for '{name}' under {af3_out_dir} (expected {cif})"
        )
    if not os.path.isfile(conf):
        raise FileNotFoundError(f"af3: missing confidences json for {name}: {conf}")
    mean_plddt = read_mean_plddt_af3(conf)  # all-atom mean (0-100); fallback only
    result = normalize_to_cache(
        cache_dir, key, cif_path=cif, mean_plddt=mean_plddt, native_scale="0-100"
    )
    # Overwrite the sidecar with the per-residue CA mean of the emitted cache PDB so it matches
    # evaluate_phase_c's v2 predicted_global_plddt (the all-atom atom_plddts mean disagrees and
    # trips _validate_v2_prediction_plddt's fail-closed consistency check).
    ca_mean = mean_ca_plddt_from_pdb(result["pdb_path"])
    if ca_mean is not None:
        with open(result["plddt_path"], "w") as handle:
            handle.write(f"{ca_mean:.4f}")
    return result
