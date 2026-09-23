"""Frozen metric schema for the IF evaluation pipeline.

Defines required column sets for each output table and a validator
that rejects incomplete rows. All downstream modules (M, N) must
produce outputs conforming to these schemas.

Also defines the per-protein test set schema (PLAN_DATA_SEL §L0)
with tier-specific nullable rules.
"""

from typing import Dict, FrozenSet, Set

import pandas as pd


class EvalSchemaError(Exception):
    """Raised when evaluation output violates the frozen metric schema."""


# ── Frozen column sets (PLAN_IF.md §L0) ─────────────────────────────────────

LEGACY_STRUCTURAL_COLUMNS: FrozenSet[str] = frozenset({
    "protein_id",
    "design_id",
    "sequence",
    "scTM",
    "pLDDT",
    "bb_RMSD",
    "recovery",
    "foldability",
})

LEGACY_STRUCTURAL_RESIDUE_COLUMNS: FrozenSet[str] = frozenset({
    "protein_id",
    "design_id",
    "design_idx",
    "residue_idx",
    "residue_idx_1based",
    "ref_chain_id",
    "ref_resseq",
    "ref_icode",
    "ref_aa",
    "design_aa",
    "sc_ca_distance",
    "ref_ca_x",
    "ref_ca_y",
    "ref_ca_z",
    "pred_ca_x",
    "pred_ca_y",
    "pred_ca_z",
    "aligned_pred_ca_x",
    "aligned_pred_ca_y",
    "aligned_pred_ca_z",
    "refold_backend",
})

STRUCTURAL_V2_COLUMNS: FrozenSet[str] = frozenset({
    "protein_id",
    "design_id",
    "design_idx",
    "sequence",
    "scTM",
    "global_ca_RMSD",
    "pLDDT",
    "reference_pLDDT",
    "predicted_active_site_mean_pLDDT",
    "predicted_active_site_min_pLDDT",
    "reference_active_site_mean_pLDDT",
    "reference_active_site_min_pLDDT",
    "active_site_sidechain_RMSD",
    "max_anchor_sidechain_RMSD",
    "max_anchor_atom_distance",
    "active_site_sidechain_atom_count",
    "anchor_count",
    "matched_anchor_count",
    "active_site_complete",
    "recovery",
    "foldability",
    "refold_backend",
})

STRUCTURAL_V2_RESIDUE_COLUMNS: FrozenSet[str] = frozenset({
    "protein_id",
    "design_id",
    "design_idx",
    "residue_idx",
    "residue_idx_1based",
    "ref_chain_id",
    "ref_resseq",
    "ref_icode",
    "ref_aa",
    "design_aa",
    "is_anchor",
    "match_status",
    "symmetry_swap_applied",
    "reference_sidechain_atom_count",
    "predicted_sidechain_atom_count",
    "sidechain_atom_count",
    "sidechain_sq_error_sum",
    "sidechain_RMSD",
    "sidechain_max_atom_distance",
    "predicted_pLDDT",
    "reference_pLDDT",
    "refold_backend",
})

# v2 is the canonical structural benchmark contract. Legacy names remain
# available only for explicit compatibility callers during migration.
STRUCTURAL_COLUMNS: FrozenSet[str] = STRUCTURAL_V2_COLUMNS
STRUCTURAL_RESIDUE_COLUMNS: FrozenSet[str] = STRUCTURAL_V2_RESIDUE_COLUMNS

IMMUNOGENICITY_HEAD_COLUMNS: FrozenSet[str] = frozenset({
    "protein_id",
    "design_id",
    "global_risk",
    "mean_hotspot",
    "max_hotspot",
    "n_hotspot_positions",
})

IMMUNOGENICITY_NMP_COLUMNS: FrozenSet[str] = frozenset({
    "protein_id",
    "design_id",
    "n_strong_binders",
    "n_weak_binders",
    "mean_best_rank",
    "n_windows_scored",
})

COMPARISON_COLUMNS: FrozenSet[str] = frozenset({
    "protein_id",
    "design_id",
    "delta_global_risk_head",
    "delta_mean_best_rank_nmp",
    "delta_n_strong_binders",
    "hotspot_reduction",
    "mutation_count",
    "risk_per_mutation",
})


# ── Per-protein test set schema (PLAN_DATA_SEL §L0) ──────────────────────────

TEST_PROTEIN_COLUMNS: Set[str] = {
    "protein_id",
    "tier",
    "sequence",
    "sequence_length",
    "pdb_path",
    "resolution",
    "cath_overlap_flag",
    "cath_overlap_id",
    "head_train_overlap_flag",
    "netmhciipan_n_strong",
    "netmhciipan_mean_best_rank",
    "head_global_risk",
    "head_n_hotspot",
    "cath_topology",
    "experimental_epitopes_json",
    "literature_evidence",
    "selection_reason",
}

TIER_REQUIRED_FIELDS: Dict[int, Set[str]] = {
    1: {"experimental_epitopes_json"},
    2: {"cath_topology"},
    3: {"literature_evidence"},
}

_VALID_TIERS = {1, 2, 3}


def _allows_null_cath_topology(entry: dict) -> bool:
    """Tier 2 rows rebuilt without CATH diversity may omit topology."""
    return (
        entry.get("tier") == 2
        and entry.get("selection_reason") == "pre-screened-no-diversity"
    )


def validate_test_protein_entry(entry: dict) -> dict:
    """Validate a single test protein entry against the frozen schema.

    Checks:
      1. All fields in TEST_PROTEIN_COLUMNS are present.
      2. ``tier`` is 1, 2, or 3.
      3. Tier-specific fields are non-null for their tier.

    Returns the entry unchanged if valid; raises EvalSchemaError otherwise.
    """
    # Universal required fields (must be present as keys)
    missing = TEST_PROTEIN_COLUMNS - set(entry.keys())
    if missing:
        raise EvalSchemaError(
            f"Missing required fields: {sorted(missing)}"
        )

    # Tier validity
    tier = entry.get("tier")
    if tier not in _VALID_TIERS:
        raise EvalSchemaError(
            f"Invalid tier={tier!r}; must be one of {sorted(_VALID_TIERS)}"
        )

    # Tier-specific non-null checks
    for field in TIER_REQUIRED_FIELDS.get(tier, set()):
        if field == "cath_topology" and _allows_null_cath_topology(entry):
            continue
        if entry.get(field) is None:
            raise EvalSchemaError(
                f"Field '{field}' must not be null for tier {tier}"
            )

    return entry


# ── Public API ───────────────────────────────────────────────────────────────

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: FrozenSet[str],
) -> pd.DataFrame:
    """Validate that a DataFrame contains all required columns and is non-empty.

    Returns the DataFrame unchanged if valid; raises EvalSchemaError otherwise.
    """
    if df.empty:
        raise EvalSchemaError(
            f"DataFrame is empty; expected columns: {sorted(required_columns)}"
        )

    missing = required_columns - set(df.columns)
    if missing:
        raise EvalSchemaError(
            f"Missing required columns: {sorted(missing)}"
        )

    return df
