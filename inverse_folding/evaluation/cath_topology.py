"""CATH topology assignment and diversity sampling for Tier 2.

Assigns CATH topology codes to PDB proteins and samples a structurally
diverse subset maximizing topology coverage with guaranteed representation
across CATH architecture classes.
"""

import json
import os
from typing import Optional

import pandas as pd


# CATH architecture classes (first digit of topology code)
_CATH_CLASSES = {
    "1": "mainly_alpha",
    "2": "mainly_beta",
    "3": "alpha_beta",
    "4": "few_secondary_structures",
}


def assign_topology(
    protein_id: str,
    cath_domain_list_path: str,
) -> Optional[str]:
    """Look up CATH topology code for a protein's PDB code.

    Accepts either:
      1. A CathDomainList file where each non-comment line has
         `domain_name class arch topology homology ...`
      2. A `chain_set.jsonl` file where each entry carries `name` and `CATH`.

    Matches on PDB code (first 4 chars of domain name, case-insensitive)
    against the first 4 chars of protein_id.

    Returns topology string (e.g., "1.10.490") or None if not found.
    """
    pdb_code = protein_id[:4].upper()

    if os.path.basename(cath_domain_list_path) == "chain_set.jsonl":
        with open(cath_domain_list_path) as f:
            for line in f:
                entry = json.loads(line)
                domain_name = entry.get("name", "")
                cath_codes = entry.get("CATH", [])
                if domain_name[:4].upper() == pdb_code and cath_codes:
                    return cath_codes[0]
        return None

    with open(cath_domain_list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            domain_pdb = parts[0][:4].upper()
            if domain_pdb == pdb_code:
                cls, arch, topo = parts[1], parts[2], parts[3]
                return f"{cls}.{arch}.{topo}"
    return None


def _extract_class(topology: str) -> str:
    """Extract CATH architecture class (first digit) from topology string."""
    if topology and "." in topology:
        return topology.split(".")[0]
    return "__unknown__"


def sample_diverse(
    candidates_df: pd.DataFrame,
    max_per_topology: int = 2,
    target_total: int = 50,
) -> pd.DataFrame:
    """Sample a structurally diverse subset from pre-screened candidates.

    Two-phase sampling:
      Phase 1 (class coverage): pick the top-scoring candidate from each
        CATH architecture class (1=alpha, 2=beta, 3=alpha/beta, 4=few SS)
        to guarantee cross-class representation.
      Phase 2 (topology diversity): from remaining candidates, sample up
        to max_per_topology per topology group, prioritizing higher NMP
        signal. Truncate to target_total.

    Args:
        candidates_df: DataFrame with columns [protein_id, cath_topology,
            netmhciipan_n_strong].
        max_per_topology: max candidates per topology group.
        target_total: maximum total candidates to return.

    Returns:
        Sampled DataFrame subset.
    """
    df = candidates_df.copy()
    if df.empty:
        return df

    # Sort globally by NMP signal descending
    df = df.sort_values("netmhciipan_n_strong", ascending=False)

    # Assign class column
    topo_filled = df["cath_topology"].fillna("__null__")
    df = df.assign(_cath_class=topo_filled.apply(_extract_class))

    # Phase 1: guarantee one candidate per architecture class
    phase1_ids = set()
    for cls_code in sorted(_CATH_CLASSES.keys()):
        cls_candidates = df[df["_cath_class"] == cls_code]
        if not cls_candidates.empty:
            phase1_ids.add(cls_candidates.iloc[0]["protein_id"])

    # Phase 2: topology-capped sampling from all candidates
    sampled_parts = []
    topo_col = topo_filled

    for _, group in df.groupby(topo_col, sort=False):
        sampled_parts.append(group.head(max_per_topology))

    if not sampled_parts:
        return df.iloc[:0].drop(columns=["_cath_class"], errors="ignore")

    result = pd.concat(sampled_parts, ignore_index=True)

    # Ensure phase 1 picks are included even if they got dropped
    phase1_missing = phase1_ids - set(result["protein_id"])
    if phase1_missing:
        extras = df[df["protein_id"].isin(phase1_missing)]
        result = pd.concat([result, extras], ignore_index=True)
        result = result.drop_duplicates(subset="protein_id", keep="first")

    # Truncate to target while preserving phase 1 class-coverage picks
    if len(result) > target_total:
        # Separate phase 1 picks (must keep) from the rest
        phase1_mask = result["protein_id"].isin(phase1_ids)
        phase1_rows = result[phase1_mask]
        other_rows = result[~phase1_mask].sort_values(
            "netmhciipan_n_strong", ascending=False,
        )
        remaining_slots = target_total - len(phase1_rows)
        if remaining_slots > 0:
            result = pd.concat(
                [phase1_rows, other_rows.head(remaining_slots)],
                ignore_index=True,
            )
        else:
            result = phase1_rows.head(target_total).reset_index(drop=True)
    else:
        result = result.sort_values(
            "netmhciipan_n_strong", ascending=False,
        ).reset_index(drop=True)

    # Remove internal column
    result = result.drop(columns=["_cath_class"], errors="ignore")

    return result
