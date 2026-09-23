"""Canonical structural-v2 contract for refinement shard merging."""

from __future__ import annotations

import json
import sys

import pandas as pd

from scripts import merge_refine_shards as merge


def test_merge_uses_canonical_v2_structure_without_suffix_table(tmp_path, monkeypatch):
    refined = tmp_path / "shard00of01" / "refined"
    refined.mkdir(parents=True)
    pd.DataFrame([{
        "protein_id": "P1", "design_idx": 0,
        "sequence_original": "AAAA", "sequence_refined": "AAAC",
        "core_count_before": 1, "core_count_after": 0,
        "n_mutations": 1, "muts": "3:C", "scTM_after": 0.9,
    }]).to_parquet(refined / "refined_designs.parquet")
    pd.DataFrame([{
        "protein_id": "P1", "design_id": "design_0000", "design_idx": 0,
        "sequence": "AAAC", "scTM": 0.9, "global_ca_RMSD": 0.7,
        "pLDDT": 91.0, "active_site_sidechain_RMSD": 0.8,
        "max_anchor_sidechain_RMSD": 0.9, "max_anchor_atom_distance": 1.2,
        "active_site_complete": True, "recovery": 0.75, "foldability": True,
        "refold_backend": "esmfold2",
    }]).to_parquet(refined / "structural.parquet")
    (refined / "final_metrics_status.json").write_text(json.dumps({"ok": True}))

    out = tmp_path / "merged"
    monkeypatch.setattr(
        sys,
        "argv",
        ["merge_refine_shards.py", "--run-dir", str(tmp_path), "--out-dir", str(out)],
    )
    merge.main()

    master = pd.read_parquet(out / "master.parquet")
    assert master.loc[0, "global_ca_RMSD"] == 0.7
    assert master.loc[0, "active_site_sidechain_RMSD"] == 0.8
    assert "bb_RMSD" not in master
    assert "structural_v2" not in merge.PER_DESIGN_TABLES


def test_merge_head_mode_emits_complete_two_axis_pareto_front(tmp_path, monkeypatch):
    refined = tmp_path / "shard00of01" / "refined"
    refined.mkdir(parents=True)
    rows = []
    for design_idx, (sequence, risk, density) in enumerate([
        ("AAAC", 1.0, 1.0),
        ("AAAD", 0.8, 1.0),
        ("AAAE", 1.0, 0.8),
        ("AAAF", 1.1, 1.1),
    ]):
        rows.append({
            "protein_id": "P1",
            "design_idx": design_idx,
            "target_source": "head",
            "sequence_original": "AAAA",
            "sequence_refined": sequence,
            "core_count_before": None,
            "core_count_after": None,
            "head_global_risk_before": 1.2,
            "head_global_risk_after": risk,
            "head_positive_mass_density_before": 1.2,
            "head_positive_mass_density_after": density,
            "n_mutations": 1,
            "muts": f"A3{sequence[-1]}",
            "scTM_after": 0.9,
        })
    pd.DataFrame(rows).to_parquet(refined / "refined_designs.parquet")
    (refined / "final_metrics_status.json").write_text(json.dumps({"ok": True}))

    out = tmp_path / "merged"
    monkeypatch.setattr(
        sys,
        "argv",
        ["merge_refine_shards.py", "--run-dir", str(tmp_path), "--out-dir", str(out)],
    )
    merge.main()

    front = pd.read_parquet(out / "head_pareto_per_seed.parquet")
    assert set(front["sequence_refined"]) == {"AAAD", "AAAE"}
    assert set(front["head_pareto_rank"]) == {1}
    assert not (out / "best_count0_per_seed.parquet").exists()
    assert not (out / "top1_per_seed.parquet").exists()
    summary = json.loads((out / "merge_summary.json").read_text())
    assert summary["target_source"] == "head"
    assert summary["head_pareto"]["rows"] == 2


def test_official_head_merge_is_lineage_balanced_and_candidate_count_is_independent(
    tmp_path, monkeypatch,
):
    refined = tmp_path / "shard00of01" / "refined"
    refined.mkdir(parents=True)
    rows = []
    specifications = {
        "SEED1": [("AAAC", 0.0, 2.0), ("AAAD", 1.0, 0.0)],
        "SEED2": [("AAAE", 0.5, 1.5), ("AAAF", 1.5, 0.5)],
    }
    for sequence_original, candidates in specifications.items():
        for sequence, risk, density in candidates:
            rows.append({
                "protein_id": "P1",
                "design_idx": len(rows),
                "target_source": "head",
                "sequence_original": sequence_original,
                "sequence_refined": sequence,
                "core_count_before": None,
                "core_count_after": None,
                "head_global_risk_before": 2.0,
                "head_global_risk_after": risk,
                "head_positive_mass_density_before": 2.0,
                "head_positive_mass_density_after": density,
                "n_mutations": 1,
                "muts": "A3C",
                "scTM_after": 0.9,
                "seed_group": "product",
            })
    pd.DataFrame(rows).to_parquet(refined / "refined_designs.parquet")
    (refined / "final_metrics_status.json").write_text(json.dumps({"ok": True}))

    out = tmp_path / "merged"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_refine_shards.py",
            "--run-dir", str(tmp_path),
            "--out-dir", str(out),
            # The count knob creates the final panel even without --official.
            "--final-candidates-per-protein", "2",
        ],
    )
    merge.main()

    selected = pd.read_parquet(out / "final_candidates.parquet")
    assert set(selected["sequence_original"]) == {"SEED1", "SEED2"}
    assert set(selected["sequence_refined"]) == {"AAAC", "AAAE"}
    assert list(selected["final_selection_rank"]) == [1, 2]
    manifest = json.loads((out / "final_candidates_manifest.json").read_text())
    assert manifest["requested_candidates_per_protein"] == 2
    assert manifest["official"] is False
    assert manifest["realized_counts"] == {"P1::product": 2}


def test_official_head_merge_defaults_to_eight_but_keeps_at_most_one_per_seed(
    tmp_path, monkeypatch,
):
    refined = tmp_path / "shard00of01" / "refined"
    refined.mkdir(parents=True)
    pd.DataFrame([
        {
            "protein_id": "P1", "design_idx": index, "target_source": "head",
            "sequence_original": "SEED1", "sequence_refined": sequence,
            "core_count_before": None, "core_count_after": None,
            "head_global_risk_before": 3.0, "head_global_risk_after": risk,
            "head_positive_mass_density_before": 3.0,
            "head_positive_mass_density_after": density,
            "n_mutations": 1, "muts": "A3C", "scTM_after": 0.9,
            "seed_group": "product",
        }
        for index, (sequence, risk, density) in enumerate([
            ("AAAC", 0.0, 2.0), ("AAAD", 1.0, 0.0), ("AAAE", 2.0, 1.0),
        ])
    ]).to_parquet(refined / "refined_designs.parquet")
    (refined / "final_metrics_status.json").write_text(json.dumps({"ok": True}))
    out = tmp_path / "merged"
    monkeypatch.setattr(
        sys,
        "argv",
        ["merge_refine_shards.py", "--run-dir", str(tmp_path), "--out-dir", str(out),
         "--official"],
    )

    merge.main()

    selected = pd.read_parquet(out / "final_candidates.parquet")
    # The full two-axis front remains available, but the delivered panel keeps only the
    # minimum-global-risk representative from this one seed lineage.
    assert set(selected["sequence_refined"]) == {"AAAC"}
    manifest = json.loads((out / "final_candidates_manifest.json").read_text())
    assert manifest["requested_candidates_per_protein"] == 8
    assert manifest["official"] is True
