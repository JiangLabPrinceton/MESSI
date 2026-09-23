"""Refold backend dispatcher for Phase C evaluation.

Three dispatch classes:
  * in-process live (``esmfold`` v1): a torch model is loaded once via
    ``load_refold_model`` and folded per design inside the eval loop.
  * isolated live (``esmfold2_live``): Fusion starts one persistent worker with
    the current Python/torch runtime and an isolated Biohub package overlay.
  * cache-read (``esmfold2``, ``protenix``): the structures are produced by a
    SEPARATE SLURM pre-compute step in a foreign conda/tool env (which cannot be
    imported into ``immune-design``) and normalized into the SAME on-disk cache
    the esmfold runner uses (``<cache_dir>/<cache_key>.pdb`` + ``<cache_key>.plddt``,
    pLDDT on the 0-100 scale). For these backends ``load_refold_model`` returns
    ``None`` and ``refold`` is a pure cache read keyed on ``(protein_id, sequence)``.

Every backend returns the same normalized dict: a real single-chain ``.pdb``
``pdb_path`` (never ``None``) and a ``pLDDT`` float on the 0-100 scale, so the
structural eval (TMalign scTM with Chain_1 = predicted, canonical global C-alpha
and side-chain metrics, foldability, structural.parquet) is format- and
confidence-scale-consistent regardless of backend.
"""

from __future__ import annotations

import os
from typing import Any

# Backends whose structures are produced by an out-of-env pre-compute step and
# read from the shared refold cache at eval time (never folded in-process).
# af3 = official DeepMind AlphaFold3 (shared install, its own interpreter).
CACHE_READ_BACKENDS = frozenset({"esmfold2", "protenix", "af3"})


def load_refold_model(backend: str, *, device: str = "cuda", **backend_options: Any) -> Any:
    """Load the model object required by a refold backend.

    Returns ``None`` for cache-read backends (esmfold2, protenix): they run in a
    foreign env and are never loaded into the eval process.
    """
    if backend == "esmfold":
        from inverse_folding.evaluation.esmfold_runner import load_esmfold_model

        return load_esmfold_model(device=device)
    if backend == "esmfold2_live":
        from inverse_folding.evaluation.esmfold2_live import ESMFold2LiveClient

        return ESMFold2LiveClient(device=device, **backend_options)
    if backend in CACHE_READ_BACKENDS:
        return None
    raise ValueError(f"unsupported refold backend: {backend}")


def refold(
    sequence: str,
    protein_id: str,
    design_id: str,
    *,
    backend: str,
    cache_dir: str | None = None,
    model: Any = None,
) -> dict[str, Any]:
    """Dispatch refolding to the configured backend."""
    if backend == "esmfold":
        from inverse_folding.evaluation.esmfold_runner import predict_structure

        return predict_structure(
            sequence=sequence,
            protein_id=protein_id,
            design_id=design_id,
            cache_dir=cache_dir,
            model=model,
        )
    if backend == "esmfold2_live":
        if model is None or not hasattr(model, "predict"):
            raise RuntimeError(
                "refold backend 'esmfold2_live' requires a loaded model from "
                "load_refold_model()"
            )
        return model.predict(
            sequence=sequence,
            protein_id=protein_id,
            design_id=design_id,
            cache_dir=cache_dir,
        )
    if backend in CACHE_READ_BACKENDS:
        return _read_cached_structure(backend, protein_id, sequence, cache_dir)
    raise ValueError(f"unsupported refold backend: {backend}")


def _read_cached_structure(
    backend: str, protein_id: str, sequence: str, cache_dir: str | None
) -> dict[str, Any]:
    """Read a pre-computed, normalized structure from the shared refold cache.

    Fail-fast (not a placeholder / not ``None``, and NOT an 'out of memory'
    message that would wrongly trip the eval's cuda->cpu OOM retry) when the
    precompute did not populate the required key.
    """
    from inverse_folding.evaluation.esmfold_runner import cache_key

    if cache_dir is None:
        raise RuntimeError(
            f"refold backend '{backend}' is cache-read but cache_dir is None; "
            "a precompute step must populate --refold-cache-dir before struct eval"
        )
    key = cache_key(protein_id, sequence)
    pdb_path = os.path.join(cache_dir, f"{key}.pdb")
    plddt_path = os.path.join(cache_dir, f"{key}.plddt")
    if not (os.path.isfile(pdb_path) and os.path.isfile(plddt_path)):
        raise RuntimeError(
            f"refold backend '{backend}': precompute missing for key '{key}' "
            f"(protein_id={protein_id}); expected {pdb_path} + .plddt. "
            f"Run the {backend} precompute step before struct eval."
        )
    with open(plddt_path) as handle:
        plddt = float(handle.read().strip())
    return {
        "pdb_string": None,
        "pLDDT": plddt,
        "pdb_path": pdb_path,
        "cache_hit": True,
    }
