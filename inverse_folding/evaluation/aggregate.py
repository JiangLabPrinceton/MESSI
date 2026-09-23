"""Alignment validator for IF evaluation tables.

Comparison and bundle-writer helpers were removed when their only caller
(`scripts/evaluate_if.py`) was retired in B4.
"""

import pandas as pd

from inverse_folding.evaluation.schema import EvalSchemaError


def validate_bundle_alignment(
    structural: pd.DataFrame,
    immuno_head: pd.DataFrame,
    immuno_nmp: pd.DataFrame,
) -> None:
    """Verify that all three metric tables are row-aligned.

    Checks: same row count, same (protein_id, design_id) pairs in same order.
    """
    counts = {
        "structural": len(structural),
        "immuno_head": len(immuno_head),
        "immuno_nmp": len(immuno_nmp),
    }
    unique_counts = set(counts.values())
    if len(unique_counts) > 1:
        raise EvalSchemaError(
            f"Mismatched row count across tables: {counts}"
        )

    # Check (protein_id, design_id) pair alignment
    for id_col in ("protein_id", "design_id"):
        if id_col in structural.columns and id_col in immuno_head.columns:
            if not (structural[id_col].values == immuno_head[id_col].values).all():
                raise EvalSchemaError(
                    f"Mismatched {id_col} order between structural and immuno_head tables"
                )
        if id_col in structural.columns and id_col in immuno_nmp.columns:
            if not (structural[id_col].values == immuno_nmp[id_col].values).all():
                raise EvalSchemaError(
                    f"Mismatched {id_col} order between structural and immuno_nmp tables"
                )
