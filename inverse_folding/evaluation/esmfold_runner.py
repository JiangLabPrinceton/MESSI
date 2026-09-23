"""ESMFold structure prediction wrapper for IF evaluation.

Predicts 3D structures from amino acid sequences using ESMFold,
with per-(protein_id, seq_hash) caching to avoid redundant predictions.
"""

import hashlib
import os
from typing import Dict, Optional


def seq_hash(sequence: str) -> str:
    """Deterministic short hash for a protein sequence."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:12]


def cache_key(protein_id: str, sequence: str) -> str:
    """Cache key safe to use as a single filesystem basename."""
    return f"{_cache_token(protein_id)}_{seq_hash(sequence)}"


def _cache_token(value: str) -> str:
    token = str(value)
    separators = {os.sep, os.altsep, "/", "\\"}
    for separator in separators:
        if separator:
            token = token.replace(separator, "_")
    token = token.replace("\x00", "_")
    return token or "protein"


def predict_structure(
    sequence: str,
    protein_id: str,
    design_id: str,
    cache_dir: Optional[str] = None,
    model: object = None,
) -> Dict:
    """Predict structure with ESMFold, returning PDB string and pLDDT.

    Args:
        sequence: amino acid string.
        protein_id: identifier for the source protein.
        design_id: identifier for this specific design.
        cache_dir: if set, cache PDB outputs by (protein_id, seq_hash).
        model: pre-loaded ESMFold model instance. If None, raises on first call.

    Returns:
        Dict with keys: pdb_string, pLDDT, pdb_path (if cached).
    """
    key = cache_key(protein_id, sequence)

    # Check cache
    if cache_dir is not None:
        pdb_path = os.path.join(cache_dir, f"{key}.pdb")
        plddt_path = os.path.join(cache_dir, f"{key}.plddt")
        if os.path.isfile(pdb_path) and os.path.isfile(plddt_path):
            with open(pdb_path) as f:
                pdb_string = f.read()
            with open(plddt_path) as f:
                plddt = float(f.read().strip())
            return {
                "pdb_string": pdb_string,
                "pLDDT": plddt,
                "pdb_path": pdb_path,
                "cache_hit": True,
            }

    # Predict
    if model is None:
        raise RuntimeError(
            "ESMFold model not loaded. Call load_esmfold_model() first."
        )

    with _no_grad():
        output = model.infer_pdb(sequence)

    # Extract mean pLDDT from PDB B-factor column
    plddt = _extract_mean_plddt(output)

    # Write cache
    pdb_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        pdb_path = os.path.join(cache_dir, f"{key}.pdb")
        with open(pdb_path, "w") as f:
            f.write(output)
        with open(os.path.join(cache_dir, f"{key}.plddt"), "w") as f:
            f.write(f"{plddt:.4f}")

    return {
        "pdb_string": output,
        "pLDDT": plddt,
        "pdb_path": pdb_path,
        "cache_hit": False,
    }


def load_esmfold_model(device: str = "cuda"):
    """Load ESMFold model. Requires `esm` package."""
    import esm as esm_module
    model = esm_module.pretrained.esmfold_v1()
    model = model.eval()
    model = model.to(device)
    return model


def _extract_mean_plddt(pdb_string: str) -> float:
    """Extract mean pLDDT from PDB B-factor column (ESMFold convention)."""
    b_factors = []
    for line in pdb_string.split("\n"):
        if line.startswith("ATOM") and len(line) >= 66:
            # B-factor is columns 61-66
            try:
                bf = float(line[60:66].strip())
                b_factors.append(bf)
            except ValueError:
                continue
    if not b_factors:
        return 0.0
    return sum(b_factors) / len(b_factors)


def _no_grad():
    """Context manager for torch.no_grad(), imported lazily."""
    import torch
    return torch.no_grad()
