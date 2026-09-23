"""Test set curation logic for IF evaluation.

Selects PDB structures for the fixed evaluation cohort with:
  - Length, resolution, and chain quality filters
  - Overlap exclusion against CATH training data
  - WT hotspot map pre-computation
  - Explicit provenance and rejection tracking
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import pandas as pd


@dataclass
class CurationConfig:
    """Parameters for test set curation."""
    min_length: int = 100
    max_length: int = 500
    max_resolution: float = 2.5  # angstroms
    require_single_chain: bool = True
    min_hotspot_positions: int = 3  # positions with h_i > median + 1*std
    target_size: int = 80  # aim for ~80 test proteins


@dataclass
class CurationLedger:
    """Tracks acceptance and rejection decisions for auditability."""
    accepted: List[Dict] = field(default_factory=list)
    rejected: List[Dict] = field(default_factory=list)
    overlap_excluded: List[Dict] = field(default_factory=list)

    def accept(self, protein_id: str, reason: str = "passed_all_filters"):
        self.accepted.append({"protein_id": protein_id, "reason": reason})

    def reject(self, protein_id: str, reason: str):
        self.rejected.append({"protein_id": protein_id, "reason": reason})

    def exclude_overlap(self, protein_id: str, matched_cath_id: str):
        self.overlap_excluded.append({
            "protein_id": protein_id,
            "matched_cath_id": matched_cath_id,
        })

    def summary(self) -> Dict:
        return {
            "n_accepted": len(self.accepted),
            "n_rejected": len(self.rejected),
            "n_overlap_excluded": len(self.overlap_excluded),
            "rejection_reasons": _count_reasons(self.rejected),
        }

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "accepted": self.accepted,
                "rejected": self.rejected,
                "overlap_excluded": self.overlap_excluded,
                "summary": self.summary(),
            }, f, indent=2)


def load_cath_chain_names(
    splits_json_path: str,
    split_keys: Optional[List[str]] = None,
) -> Set[str]:
    """Load CATH chain names for overlap checking.

    Args:
        splits_json_path: path to chain_set_splits.json.
        split_keys: which splits to include. Defaults to ["train"] only,
            per PLAN_IF L1 (exclude overlap with the CATH training substrate
            used in Module K, not validation/test).
    """
    if split_keys is None:
        split_keys = ["train"]

    with open(splits_json_path) as f:
        splits = json.load(f)
    names = set()
    for key in split_keys:
        if key in splits:
            names.update(splits[key])
    return names


def check_overlap(protein_id: str, cath_names: Set[str]) -> Optional[str]:
    """Check if a PDB protein_id overlaps with CATH training data.

    Checks both exact match and PDB-code-level match (first 4 chars).
    Returns the matched CATH ID if overlap found, else None.
    """
    # Exact match
    if protein_id in cath_names:
        return protein_id

    # PDB code match (e.g., "1ABC" matches "1ABCA00")
    pdb_code = protein_id[:4].upper()
    for cath_name in cath_names:
        if cath_name[:4].upper() == pdb_code:
            return cath_name

    return None


def filter_candidate(
    protein_id: str,
    sequence_length: int,
    resolution: Optional[float],
    n_chains: int,
    cfg: CurationConfig,
    ledger: CurationLedger,
) -> bool:
    """Apply curation filters to a single candidate protein.

    Returns True if the protein passes all filters.
    """
    if sequence_length < cfg.min_length:
        ledger.reject(protein_id, f"too_short ({sequence_length} < {cfg.min_length})")
        return False

    if sequence_length > cfg.max_length:
        ledger.reject(protein_id, f"too_long ({sequence_length} > {cfg.max_length})")
        return False

    if resolution is not None and resolution > cfg.max_resolution:
        ledger.reject(protein_id, f"low_resolution ({resolution} > {cfg.max_resolution})")
        return False

    if cfg.require_single_chain and n_chains > 1:
        ledger.reject(protein_id, f"multi_chain ({n_chains} chains)")
        return False

    return True


def _count_reasons(entries: List[Dict]) -> Dict[str, int]:
    """Count rejection reasons."""
    counts: Dict[str, int] = {}
    for entry in entries:
        reason = entry.get("reason", "unknown")
        # Extract the base reason (before parenthetical details)
        base = reason.split("(")[0].strip()
        counts[base] = counts.get(base, 0) + 1
    return counts
