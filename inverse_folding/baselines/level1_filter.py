"""N1: Level 1 post-hoc filter baseline.

Implements the "generate then filter" comparison arm:
given K candidates from unconditional DPLM generation (M3 eta=0),
select the one with minimum epitope-head risk.

Reads directly from M3 sweep outputs (all_candidates/ directory).
"""

import hashlib
import os
import re
from typing import Any, Dict, List

import numpy as np


# Regex for M3 candidate FASTA header format:
#   >{protein_id}_cand{i} | risk={value} | hash={hex}
_HEADER_RE = re.compile(
    r">(\S+)\s*\|\s*risk=([0-9.eE+-]+)\s*\|\s*hash=(\w+)"
)


def parse_candidates_fasta(fasta_path: str) -> Dict[str, Any]:
    """Parse an M3 all_candidates FASTA file.

    Returns dict with keys: sequences, risks, hashes.
    """
    sequences: List[str] = []
    risks: List[float] = []
    hashes: List[str] = []
    current_seq_parts: List[str] = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Flush previous sequence
                if current_seq_parts:
                    sequences.append("".join(current_seq_parts))
                    current_seq_parts = []

                m = _HEADER_RE.match(line)
                if m:
                    risks.append(float(m.group(2)))
                    hashes.append(m.group(3))
                else:
                    risks.append(float("nan"))
                    hashes.append("")
            else:
                current_seq_parts.append(line)

        # Flush last sequence
        if current_seq_parts:
            sequences.append("".join(current_seq_parts))

    if not sequences:
        raise ValueError(f"No candidates found in {fasta_path}")

    return {
        "sequences": sequences,
        "risks": np.array(risks, dtype=np.float64),
        "hashes": hashes,
    }


def select_argmin_candidate(
    risks: np.ndarray,
    sequences: List[str],
) -> Dict[str, Any]:
    """Select the candidate with lowest risk (argmin).

    Ties are broken by selecting the first occurrence (np.argmin behavior).
    """
    risks = np.asarray(risks, dtype=np.float64)
    idx = int(np.argmin(risks))
    selected_seq = sequences[idx]

    return {
        "selected_index": idx,
        "selected_risk": float(risks[idx]),
        "selected_sequence": selected_seq,
        "selected_seq_hash": hashlib.sha256(
            selected_seq.encode()
        ).hexdigest()[:12],
        "selection_rule": "argmin_risk",
        "candidate_risks": risks.tolist(),
    }


def filter_protein_set(
    candidates_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """Apply argmin filter to all proteins in an M3 all_candidates/ directory.

    Returns {protein_id: selection_result}.
    """
    results: Dict[str, Dict[str, Any]] = {}

    if not os.path.isdir(candidates_dir):
        return results

    for fname in sorted(os.listdir(candidates_dir)):
        if not (fname.endswith(".fasta") or fname.endswith(".fa")):
            continue

        protein_id = os.path.splitext(fname)[0]
        fasta_path = os.path.join(candidates_dir, fname)
        parsed = parse_candidates_fasta(fasta_path)

        result = select_argmin_candidate(parsed["risks"], parsed["sequences"])
        results[protein_id] = result

    return results
