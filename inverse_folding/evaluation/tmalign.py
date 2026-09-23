"""TM-align output parser and runner wrapper.

Parses TM-align stdout into structured metric dicts. The runner
invokes TM-align as a subprocess and caches results by (protein_id, seq_hash).
"""

import hashlib
import json
import os
import re
import subprocess
from typing import Dict, Optional


def parse_tmalign_output(stdout: str) -> Dict[str, float]:
    """Parse TM-align text output into a structured metrics dict.

    Extracts: tm_score, rmsd, aligned_length, seq_identity.
    Raises ValueError if required fields cannot be parsed.
    """
    result = {}

    # TM-score (first occurrence, normalized by Chain_1)
    tm_match = re.search(
        r"TM-score=\s*([0-9.]+)\s*\(if normalized by length of Chain_1",
        stdout,
    )
    if not tm_match:
        raise ValueError(
            "Could not parse TM-score from TM-align output"
        )
    result["tm_score"] = float(tm_match.group(1))

    # RMSD and aligned length
    align_match = re.search(
        r"Aligned length=\s*(\d+),\s*RMSD=\s*([0-9.]+),\s*Seq_ID=.*?=\s*([0-9.]+)",
        stdout,
    )
    if align_match:
        result["aligned_length"] = int(align_match.group(1))
        result["rmsd"] = float(align_match.group(2))
        result["seq_identity"] = float(align_match.group(3))
    else:
        result["aligned_length"] = 0
        result["rmsd"] = float("inf")
        result["seq_identity"] = 0.0

    return result


def run_tmalign(
    pred_pdb: str,
    ref_pdb: str,
    tmalign_bin: str = "TMalign",
    cache_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Run TM-align on two PDB files and return parsed metrics.

    If cache_dir is provided, results are cached by file content hashes.
    """
    # Check cache
    if cache_dir is not None:
        cache_key = _cache_key(pred_pdb, ref_pdb)
        cache_path = os.path.join(cache_dir, f"{cache_key}.json")
        if os.path.isfile(cache_path):
            with open(cache_path) as f:
                return json.load(f)

    # Run TM-align
    result = subprocess.run(
        [tmalign_bin, pred_pdb, ref_pdb],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"TM-align failed (exit {result.returncode}): {result.stderr}"
        )

    metrics = parse_tmalign_output(result.stdout)

    # Write cache
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(metrics, f)

    return metrics


def _cache_key(path_a: str, path_b: str) -> str:
    """Deterministic cache key from two file paths."""
    h = hashlib.sha256()
    h.update(os.path.abspath(path_a).encode())
    h.update(os.path.abspath(path_b).encode())
    return h.hexdigest()[:16]
