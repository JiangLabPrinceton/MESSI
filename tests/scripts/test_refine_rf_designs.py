"""Driver smoke for scripts/refine_rf_designs.py (PLAN_RF_REFINE.md R4).

Fake oracles bypass build_oracles (no torch/NMP/ESMFold). Builds a sharded 2-protein
generated.parquet + imm_head.parquet, runs refine mode, and asserts both output
schemas plus a killable core reaching count 0.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scripts.refine_rf_designs as refine_driver
from types import SimpleNamespace

from inverse_folding.reference_flow.refine import StructureMetrics
from scripts.refine_rf_designs import (
    Oracles,
    _atomic_write,
    _canonical_allele,
    _load_catalytic_indices_by_protein,
    _load_sharded,
    _run_final_metrics,
    _retain_official_head_rows,
    build_arg_parser,
    build_oracles,
    run_refinement,
)


def _fake_oracles() -> Oracles:
    def nmp_fn(pid, seqs):
        out = []
        for s in seqs:
            if pid == "A" and s[10] == "A":   # protein A has one core killable by editing pos 10
                out.append([{"pos": 10, "pep_length": 9, "peptide": s[10:19],
                             "core": s[10:19], "rank_EL": 0.001}])
            else:
                out.append([])
        return out

    def head_fn(pid, seqs):
        return np.array([[9.0 if s[10] == "A" else 0.0] for s in seqs])

    def struct_fn(pid, seq):
        return StructureMetrics(scTM=0.90, pLDDT=90.0)

    return Oracles(head_fn=head_fn, nmp_fn=nmp_fn, struct_fn=struct_fn,
                   window_coords_fn=lambda seq: (np.array([0]), np.array([len(seq)])))


def _fake_head_target_oracles(*, two_peaks=False):
    """Residue-local Head peaks with forbidden NMP/window-only oracles."""
    def forbidden_oracle(*_args, **_kwargs):
        raise AssertionError("Head target mode must not call NMP or window-only Head")

    def head_score_fn(_pid, seqs):
        scores = []
        for seq in seqs:
            active = (12, 20) if two_peaks else (12,)
            hotspots = np.zeros(len(seq), dtype=float)
            for rank, position in enumerate(active):
                hotspots[position] = (
                    0.30 - 0.05 * rank if seq[position] == "A" else 0.10
                )
            scores.append(SimpleNamespace(
                global_risk=float(sum(seq[position] == "A" for position in active)),
                residue_hotspot=tuple(hotspots),
                sequence_length=len(seq),
                windows=(SimpleNamespace(start_0b=0, end_0b=25, k=25, z=1.0),),
            ))
        return scores

    return SimpleNamespace(
        head_fn=forbidden_oracle,
        head_score_fn=head_score_fn,
        nmp_fn=forbidden_oracle,
        struct_fn=lambda _pid, _seq: StructureMetrics(scTM=0.90, pLDDT=90.0),
        window_coords_fn=lambda seq: (np.array([0]), np.array([len(seq)])),
    )


def _write_inputs(tmp_path):
    gen_dir = tmp_path / "run" / "generation" / "u_shard00of01_seed42"
    gen_dir.mkdir(parents=True)
    pd.DataFrame([
        {"protein_id": "A", "design_idx": 0, "sequence": "A" * 30, "seed": 42, "wall_seconds": 1.0},
        {"protein_id": "B", "design_idx": 0, "sequence": "M" * 30, "seed": 42, "wall_seconds": 1.0},
    ]).to_parquet(gen_dir / "generated.parquet")

    eval_dir = tmp_path / "eval_immune"
    ev_shard = eval_dir / "u_shard00of01_seed42_imm_full"
    ev_shard.mkdir(parents=True)
    pd.DataFrame([
        {"protein_id": "A", "design_idx": 0, "global_risk": 1.0},
        {"protein_id": "B", "design_idx": 0, "global_risk": 2.0},
    ]).to_parquet(ev_shard / "imm_head.parquet")

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "schema_version: uricase_active_site_v0\n"
        "description: smoke\n"
        "entries:\n"
        "  - protein_id: A\n"
        "    hard_anchors:\n"
        "      - {index_0b: 0, expected_aa: A, label: a0}\n"
    )
    return tmp_path / "run", eval_dir, manifest


def test_refine_smoke_writes_both_schemas_and_kills_a_core(tmp_path):
    run_dir, eval_dir, manifest = _write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    args = build_arg_parser().parse_args(['--target-source', 'nmp', '--no-official',
        "--run-dir", str(run_dir), "--eval-immune-dir", str(eval_dir),
        "--constraint-manifest", str(manifest), "--allele", "HLA-DRB1_07_01",
        "--proteins", "all", "--seeds-per-protein", "1", "--mode", "refine",
        "--beam-width", "4", "--max-rounds", "5", "--scTM-eps", "0.05",
        "--out-dir", str(out_dir),
    ])
    rc = run_refinement(args, _fake_oracles())
    assert rc == 0

    rich = pd.read_parquet(out_dir / "refined" / "refined_designs.parquet")
    assert {
        "protein_id", "design_idx", "sequence_original", "sequence_refined",
        "n_mutations", "core_count_before", "core_count_after", "scTM_after",
        "pLDDT_after", "diverged",
    } <= set(rich.columns)

    ev = pd.read_parquet(out_dir / "refined" / "evaluator_ready.parquet")
    assert list(ev.columns) == ["protein_id", "design_idx", "sequence", "seed", "wall_seconds"]

    # (iii) protein A's killable core is eliminated.
    a = rich[rich["protein_id"] == "A"].iloc[0]
    assert a["core_count_before"] == 1
    assert a["core_count_after"] == 0
    assert a["sequence_refined"][10] != "A"   # the eliminating edit is at pos 10

    # evaluator_ready sequence is the refined sequence, joined on (protein_id, design_idx).
    merged = ev.merge(rich, on=["protein_id", "design_idx"])
    assert (merged["sequence"] == merged["sequence_refined"]).all()


def test_ceiling_smoke_writes_candidate_table(tmp_path):
    run_dir, eval_dir, manifest = _write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    args = build_arg_parser().parse_args(['--target-source', 'nmp', '--no-official',
        "--run-dir", str(run_dir), "--eval-immune-dir", str(eval_dir),
        "--constraint-manifest", str(manifest), "--allele", "HLA-DRB1_07_01",
        "--proteins", "A", "--seeds-per-protein", "1", "--mode", "ceiling",
        "--out-dir", str(out_dir),
    ])
    assert run_refinement(args, _fake_oracles()) == 0
    cand = pd.read_parquet(out_dir / "ceiling" / "candidates.parquet")
    assert {"protein_id", "design_idx", "muts", "head_proxy", "core_count",
            "rank_margin_mass", "scTM", "pLDDT", "passed", "eliminates"} <= set(cand.columns)
    assert bool(cand["eliminates"].any())   # editing pos 10 drops the core -> eliminates


def test_refine_from_seed_table(tmp_path):
    # --seed-table bypasses --run-dir best-of-N selection: refine the listed seeds directly.
    seed_table = tmp_path / "seeds.parquet"
    pd.DataFrame([
        {"protein_id": "A", "design_id": "design_0000", "sequence": "A" * 30, "seed": 42},
    ]).to_parquet(seed_table)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "schema_version: uricase_active_site_v0\ndescription: smoke\nentries:\n"
        "  - protein_id: A\n    hard_anchors:\n      - {index_0b: 0, expected_aa: A, label: a0}\n"
    )
    out_dir = tmp_path / "out"
    args = build_arg_parser().parse_args(['--target-source', 'nmp', '--no-official',
        "--seed-table", str(seed_table), "--constraint-manifest", str(manifest),
        "--allele", "HLA-DRB1_07_01", "--mode", "refine", "--beam-width", "4",
        "--max-rounds", "5", "--out-dir", str(out_dir),
    ])
    assert run_refinement(args, _fake_oracles()) == 0
    rich = pd.read_parquet(out_dir / "refined" / "refined_designs.parquet")
    a = rich[rich["protein_id"] == "A"].iloc[0]
    assert a["core_count_after"] == 0   # killable core eliminated, seed sourced from the table


def test_head_target_source_is_head_only_and_edits_thresholded_residue_peak(tmp_path):
    seed_table = tmp_path / "seeds.parquet"
    pd.DataFrame([{
        "protein_id": "A",
        "design_id": "design_0000",
        "sequence": "A" * 30,
        "seed": 42,
    }]).to_parquet(seed_table)

    base = [
        "--seed-table", str(seed_table), "--allele", "HLA-DRB1_07_01",
        "--mode", "refine", "--beam-width", "4", "--max-rounds", "3",
        "--max-pairs", "0", "--head-aa-per-position", "1", "--no-eval-metrics",
    ]
    head_out = tmp_path / "head"
    head_args = build_arg_parser().parse_args(
        base + ["--target-source", "head", "--out-dir", str(head_out)]
    )

    assert run_refinement(head_args, _fake_head_target_oracles()) == 0
    head_row = pd.read_parquet(
        head_out / "refined" / "refined_designs.parquet"
    ).iloc[0]
    assert head_row["sequence_refined"][12] != "A"
    assert set(np.flatnonzero(np.asarray(list(head_row["sequence_refined"])) != "A")) == {12}
    assert pd.isna(head_row["core_count_before"])
    assert pd.isna(head_row["core_count_after"])
    assert head_row["head_global_risk"] == 0.0
    assert head_row["head_global_risk_before"] == 1.0
    assert head_row["head_global_risk_after"] == 0.0
    assert head_row["head_positive_mass_density"] == pytest.approx(0.10 / 30.0)
    assert head_row["head_positive_mass_density_before"] == pytest.approx(0.30 / 30.0)
    assert head_row["head_positive_mass_density_after"] == pytest.approx(0.10 / 30.0)
    assert head_row["head_positive_hotspot_positions_before"] == 1
    assert head_row["head_positive_hotspot_positions_after"] == 1
    assert head_row["head_residue_threshold"] == pytest.approx(0.15)
    assert list(head_row["head_target_positions_0b_before"]) == [12]
    assert list(head_row["head_target_hotspot_values_before"]) == [0.30]
    assert head_row["n_head_residues_above_threshold_before"] == 1
    assert head_row["n_head_local_maxima_pre_cap_before"] == 1
    assert head_row["n_head_target_positions_before"] == 1
    assert list(head_row["head_target_positions_0b_after"]) == []
    assert head_row["n_head_target_positions_after"] == 0


def test_head_ceiling_scores_all_singles_then_capped_round_robin_doubles(tmp_path):
    seed_table = tmp_path / "seeds.parquet"
    pd.DataFrame([{
        "protein_id": "A", "design_id": "design_0000", "sequence": "A" * 30, "seed": 42,
    }]).to_parquet(seed_table)
    out_dir = tmp_path / "head_ceiling"
    args = build_arg_parser().parse_args(['--no-official',
        "--seed-table", str(seed_table), "--allele", "HLA-DRB1_07_01",
        "--target-source", "head", "--mode", "ceiling", "--out-dir", str(out_dir),
        "--head-aa-per-position", "1", "--max-pairs", "1",
    ])

    assert run_refinement(args, _fake_head_target_oracles(two_peaks=True)) == 0
    candidates = pd.read_parquet(out_dir / "ceiling" / "candidates.parquet")
    assert len(candidates) == 2 * 19 + 1
    assert candidates["head_candidate_stage"].value_counts().to_dict() == {
        "single": 38,
        "double": 1,
    }
    assert set(position for row in candidates["positions"] for position in row) == {12, 20}
    assert list(candidates.iloc[-1]["positions"]) == [12, 20]


def test_official_head_retention_keeps_exact_per_seed_front_and_default_is_opt_in():
    rows = [
        {"sequence_refined": "AAAC", "head_global_risk_after": 0.0,
         "head_positive_mass_density_after": 2.0},
        {"sequence_refined": "AAAD", "head_global_risk_after": 1.0,
         "head_positive_mass_density_after": 0.0},
        {"sequence_refined": "AAAE", "head_global_risk_after": 2.0,
         "head_positive_mass_density_after": 2.0},
    ]

    retained = _retain_official_head_rows(rows)

    assert [row["sequence_refined"] for row in retained] == ["AAAC", "AAAD"]
    args = build_arg_parser().parse_args(
        ['--target-source', 'nmp', '--no-official', "--seed-table", "t", "--allele", "A", "--out-dir", "o"]
    )
    assert args.official is False
    assert args.final_candidates_per_protein is None


def test_official_head_run_persists_seed_group_and_requested_final_count(tmp_path):
    seed_table = tmp_path / "seeds.parquet"
    pd.DataFrame([{
        "protein_id": "A",
        "design_id": "design_0000",
        "sequence": "A" * 30,
        "seed": 42,
        "seed_group": "product",
        "selection_rule_id": "upstream_rule_v1",
    }]).to_parquet(seed_table)
    out = tmp_path / "official"
    args = build_arg_parser().parse_args([
        "--seed-table", str(seed_table),
        "--allele", "HLA-DRB1_07_01",
        "--target-source", "head",
        "--mode", "refine",
        "--max-rounds", "1",
        "--max-pairs", "0",
        "--head-aa-per-position", "1",
        "--official",
        "--final-candidates-per-protein", "5",
        "--no-eval-metrics",
        "--out-dir", str(out),
    ])

    assert run_refinement(args, _fake_head_target_oracles()) == 0
    refined = pd.read_parquet(out / "refined" / "refined_designs.parquet")
    assert set(refined["seed_group"]) == {"product"}
    assert set(refined["selection_rule_id"]) == {"upstream_rule_v1"}
    config = json.loads((out / "refined" / "refine_config.json").read_text())
    assert bool(config["official"])
    assert int(config["final_candidates_per_protein"]) == 5


def test_refine_help_lists_key_flags():
    # R4 review fixes + PLAN update: the new seed-table / max-path-mutations knobs must exist.
    parser = build_arg_parser()
    dests = {a.dest for a in parser._actions}
    assert {"test_set_parquet", "refold_cache_dir", "netmhciipan_bin", "constraint_manifest",
            "mode", "nmp_batch_size", "topB", "seed_table", "max_path_mutations",
            "refinement_structure_metrics", "max_anchor_sidechain_RMSD_max",
            "structural_metrics_v2", "refold_model", "esmfold2_site_packages",
            "structure_gate_profile", "gate_scTM_min", "gate_cat_max_scRMSD_max",
            "gate_predicted_active_site_min_pLDDT_min", "target_source",
            "head_residue_threshold", "head_max_target_positions",
            "head_aa_per_position", "official",
            "final_candidates_per_protein"} <= dests


def test_refiner_defaults_to_esmfold2_and_has_no_numeric_protocol_gate_defaults():
    args = build_arg_parser().parse_args(
        ['--target-source', 'nmp', '--no-official', "--seed-table", "t", "--allele", "A", "--out-dir", "o"]
    )
    assert args.refold_model == "esmfold2_live"
    assert args.structure_gate_profile == "protocol"
    assert args.scTM_eps is None
    assert args.gate_scTM_min is None
    assert args.gate_cat_max_scRMSD_max is None
    assert args.gate_predicted_active_site_min_pLDDT_min is None
    assert args.head_residue_threshold == pytest.approx(0.15)
    assert args.head_max_target_positions == 12
    assert args.head_aa_per_position == 2


@pytest.mark.parametrize("override", [
    ["--head-residue-threshold", "-0.1"],
    ["--head-residue-threshold", "nan"],
    ["--head-max-target-positions", "0"],
    ["--head-aa-per-position", "0"],
    ["--head-aa-per-position", "20"],
])
def test_head_target_knobs_fail_closed_before_input_loading(override):
    args = build_arg_parser().parse_args(['--no-official',
        "--seed-table", "missing.parquet", "--allele", "A", "--out-dir", "out",
        "--target-source", "head", *override,
    ])
    with pytest.raises(ValueError, match="Head refinement requires"):
        run_refinement(args, _fake_head_target_oracles())




def test_build_oracles_fail_fast_on_missing_heavy_inputs():
    # The guard fires BEFORE any torch/NMP/ESMFold import (review CRITICAL fixes:
    # netmhciipan_bin/head-checkpoint/head-config-dir/pdb-root/test-set/esmfold-cache).
    args = build_arg_parser().parse_args(['--target-source', 'nmp', '--no-official',
        "--run-dir", "x", "--eval-immune-dir", "y", "--allele", "A", "--out-dir", "z",
    ])
    with pytest.raises(ValueError, match=r"build_oracles requires"):
        build_oracles(args)


def test_head_build_oracles_does_not_require_netmhciipan():
    args = build_arg_parser().parse_args(['--no-official',
        "--seed-table", "x", "--allele", "A", "--out-dir", "z",
        "--target-source", "head",
    ])
    with pytest.raises(ValueError) as exc:
        build_oracles(args)
    assert "--netmhciipan-bin" not in str(exc.value)


def test_build_oracles_validates_sidechain_gate_before_heavy_imports():
    base = [
        "--run-dir", "x", "--allele", "A", "--out-dir", "z",
        "--esmfold-cache-dir", "cache", "--test-set-parquet", "test.parquet",
        "--pdb-root", "pdb", "--head-checkpoint", "head.pt",
        "--head-config-dir", "configs", "--netmhciipan-bin", "nmp",
        "--refold-model", "esmfold",
    ]
    args = build_arg_parser().parse_args(
        base + ["--structure-gate-profile", "legacy",
                "--max-anchor-sidechain-RMSD-max", "1.5"]
    )
    with pytest.raises(ValueError, match="requires --refinement-structure-metrics sidechain"):
        build_oracles(args)

    args = build_arg_parser().parse_args(
        base + [
            "--structure-gate-profile", "legacy",
            "--refinement-structure-metrics", "sidechain",
            "--max-anchor-sidechain-RMSD-max", "1.5",
        ]
    )
    with pytest.raises(ValueError, match="requires --constraint-manifest"):
        build_oracles(args)


def test_build_oracles_requires_complete_protocol_gate_before_heavy_imports():
    base = [
        "--run-dir", "x", "--allele", "A", "--out-dir", "z",
        "--esmfold-cache-dir", "cache", "--test-set-parquet", "test.parquet",
        "--pdb-root", "pdb", "--head-checkpoint", "head.pt",
        "--head-config-dir", "configs", "--netmhciipan-bin", "nmp",
        "--esmfold2-site-packages", "site-packages",
    ]
    args = build_arg_parser().parse_args(base + ["--gate-scTM-min", "0.94"])
    with pytest.raises(ValueError, match="requires all three"):
        build_oracles(args)


def test_load_catalytic_indices_uses_protocol_annotation_and_v0_fallback(tmp_path):
    protocol_manifest = tmp_path / "protocol.yaml"
    protocol_manifest.write_text(
        "schema_version: uricase_active_site_v1\n"
        "annotation_provenance:\n"
        "  direct_functional_union_uniprot_1b: [2]\n"
        "entries:\n"
        "  - protein_id: P\n"
        "    hard_anchors:\n"
        "      - {index_0b: 0, expected_aa: A}\n"
        "      - {index_0b: 1, expected_aa: C}\n"
    )
    parsed = refine_driver.load_constraint_manifest(protocol_manifest)
    assert _load_catalytic_indices_by_protein(protocol_manifest, parsed) == {"P": {1}}

    v0_manifest = tmp_path / "v0.yaml"
    v0_manifest.write_text(
        "schema_version: uricase_active_site_v0\n"
        "entries:\n"
        "  - protein_id: P\n"
        "    hard_anchors:\n"
        "      - {index_0b: 3, expected_aa: A}\n"
    )
    parsed = refine_driver.load_constraint_manifest(v0_manifest)
    assert _load_catalytic_indices_by_protein(v0_manifest, parsed) == {"P": {3}}


def test_canonical_allele_normalizes_underscore_tag():
    # NMP score_batch normalizes '*'->'_' and drops ':'. The run-dir filesystem tag
    # 'HLA-DRB1_07_01' would mis-normalize to an INVALID 'DRB1_07_01'; the driver must
    # canonicalize it to 'HLA-DRB1*07:01' (-> valid 'DRB1_0701'). Already-canonical passes through.
    assert _canonical_allele("HLA-DRB1_07_01") == "HLA-DRB1*07:01"
    assert _canonical_allele("HLA-DRB1_04_01") == "HLA-DRB1*04:01"
    assert _canonical_allele("HLA-DRB1*07:01") == "HLA-DRB1*07:01"


def test_load_sharded_accepts_direct_file_and_bare_shards(tmp_path):
    df = pd.DataFrame([{"protein_id": "A", "design_idx": 0, "sequence": "AA"}])
    # (a) --run-dir pointing directly at a single clean parquet FILE
    f = tmp_path / "clean.parquet"
    df.to_parquet(f)
    assert len(_load_sharded(f, "generation", "generated.parquet")) == 1
    # (b) bare '*shard*/' dirs directly under root (NO 'generation/' wrapper)
    d = tmp_path / "run"
    (d / "x_shard00of02_ailab").mkdir(parents=True)
    (d / "x_shard01of02_ailab").mkdir()
    df.to_parquet(d / "x_shard00of02_ailab" / "generated.parquet")
    df.to_parquet(d / "x_shard01of02_ailab" / "generated.parquet")
    assert len(_load_sharded(d, "generation", "generated.parquet")) == 2
    # (c) flat merged parquet directly inside the dir
    d2 = tmp_path / "merged"
    d2.mkdir()
    df.to_parquet(d2 / "generated.parquet")
    assert len(_load_sharded(d2, "generation", "generated.parquet")) == 1


def test_refine_without_eval_immune_refines_provided_designs(tmp_path):
    # User hands a clean generated.parquet (already-chosen seqs to refine); no imm_head /
    # global_risk. --eval-immune-dir is optional -> refine every provided design.
    gen = tmp_path / "clean.parquet"
    pd.DataFrame([
        {"protein_id": "A", "design_idx": 0, "sequence": "A" * 30, "seed": 42, "wall_seconds": 1.0},
    ]).to_parquet(gen)
    out_dir = tmp_path / "out"
    args = build_arg_parser().parse_args(['--target-source', 'nmp', '--no-official',
        "--run-dir", str(gen), "--allele", "HLA-DRB1*07:01",
        "--proteins", "all", "--seeds-per-protein", "1", "--mode", "refine",
        "--beam-width", "4", "--max-rounds", "5", "--out-dir", str(out_dir),
    ])
    assert run_refinement(args, _fake_oracles()) == 0
    rich = pd.read_parquet(out_dir / "refined" / "refined_designs.parquet")
    a = rich[rich["protein_id"] == "A"].iloc[0]
    assert a["core_count_before"] == 1
    assert a["core_count_after"] == 0


def test_shard_seeds_balanced_and_stride_partition_disjointly():
    from scripts.refine_rf_designs import _shard_seeds

    seeds = {
        "A": [{"sequence": "x", "design_idx": 0, "seed": -1, "_cost": 60.0},
              {"sequence": "y", "design_idx": 1, "seed": -1, "_cost": 5.0}],
        "B": [{"sequence": "z", "design_idx": 0, "seed": -1, "_cost": 30.0},
              {"sequence": "w", "design_idx": 1, "seed": -1, "_cost": 30.0}],
    }
    n = 2
    for mode in ("balanced", "stride"):
        parts = [_shard_seeds(seeds, n, i, mode) for i in range(n)]
        allseqs = sorted(s["sequence"] for p in parts for lst in p.values() for s in lst)
        assert allseqs == ["w", "x", "y", "z"]                 # every seed once, no drop/dup
        assert _shard_seeds(seeds, n, 0, mode) == parts[0]     # deterministic partition


def test_incremental_nmp_default_on_and_toggle():
    base = ["--seed-table", "t", "--allele", "A", "--out-dir", "o"]
    assert build_arg_parser().parse_args(base).incremental_nmp is True
    assert build_arg_parser().parse_args(base + ["--no-incremental-nmp"]).incremental_nmp is False
    a = build_arg_parser().parse_args(base + ["--n-shards", "24", "--shard-idx", "7"])
    assert a.n_shards == 24 and a.shard_idx == 7 and a.shard_by == "balanced"


def test_eval_metrics_flags_defaults_and_toggle():
    base = ["--seed-table", "t", "--allele", "A", "--out-dir", "o"]
    a = build_arg_parser().parse_args(base)
    assert a.emit_eval_metrics is True          # full-metrics emission on by default
    assert a.strong_binder_threshold == 2.0     # evaluate_phase_c default (percent rank_EL)
    assert a.hotspot_threshold == 0.5
    assert a.imm_full is False
    assert a.structural_metrics_v2 is True
    assert a.refinement_structure_metrics == ""
    b = build_arg_parser().parse_args(
        base + ["--no-eval-metrics", "--imm-full",
                "--strong-binder-threshold", "1.0", "--hotspot-threshold", "0.6",
                "--structural-metrics-v2", "--refinement-structure-metrics",
                "global_ca_rmsd,sidechain", "--max-anchor-sidechain-RMSD-max", "1.5"])
    assert b.emit_eval_metrics is False and b.imm_full is True
    assert b.strong_binder_threshold == 1.0 and b.hotspot_threshold == 0.6
    assert b.structural_metrics_v2 is True
    assert b.refinement_structure_metrics == "global_ca_rmsd,sidechain"
    assert b.max_anchor_sidechain_RMSD_max == 1.5


def test_run_final_metrics_writes_v2_to_canonical_paths(tmp_path, monkeypatch):
    from scripts import evaluate_phase_c as evaluator

    refined_dir = tmp_path / "refined"
    refined_dir.mkdir()
    pd.DataFrame([
        {"protein_id": "P1", "design_idx": 0, "sequence": "AAAA", "seed": 0,
         "wall_seconds": 0.0},
    ]).to_parquet(refined_dir / "evaluator_ready.parquet")
    generated = pd.DataFrame([
        {"protein_id": "P1", "design_id": "design_0000", "design_idx": 0,
         "sequence": "AAAA"},
    ])
    monkeypatch.setattr(evaluator, "load_generated_designs", lambda path: generated)
    monkeypatch.setattr(
        evaluator,
        "load_test_lookup",
        lambda path: (pd.DataFrame(), {"P1": {"sequence": "AAAA", "sequence_length": 4}}),
    )
    monkeypatch.setattr(evaluator, "build_head_predictor", lambda **kwargs: object())
    monkeypatch.setattr(evaluator, "build_nmp_runner", lambda **kwargs: object())
    monkeypatch.setattr(
        evaluator,
        "evaluate_immunogenicity_rows",
        lambda *args, **kwargs: (
            pd.DataFrame([{"protein_id": "P1", "design_id": "design_0000"}]),
            pd.DataFrame([{"protein_id": "P1", "design_id": "design_0000"}]),
            [],
            pd.DataFrame(),
            pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(evaluator, "_load_anchor_indices_by_protein", lambda *args: {"P1": {1}})
    call = {}

    def fake_structural(*args, **kwargs):
        call.update(kwargs)
        return (
            pd.DataFrame([{
                "protein_id": "P1", "design_id": "design_0000", "design_idx": 0,
                "sequence": "AAAA", "scTM": 0.9, "global_ca_RMSD": 0.7,
                "pLDDT": 91.0, "active_site_sidechain_RMSD": 0.8,
            }]),
            [],
            pd.DataFrame([{
                "protein_id": "P1", "design_id": "design_0000", "design_idx": 0,
                "residue_idx": 1, "sidechain_RMSD": 0.8,
            }]),
        )

    monkeypatch.setattr(evaluator, "evaluate_structural_rows", fake_structural)
    monkeypatch.setattr(refine_driver, "_free_gpu", lambda: None)
    args = build_arg_parser().parse_args(['--target-source', 'nmp', '--no-official',
        "--seed-table", "unused", "--allele", "A", "--out-dir", str(tmp_path),
        "--test-set-parquet", "test.parquet", "--pdb-root", "pdb",
        "--head-checkpoint", "head.pt", "--head-config-dir", "configs",
        "--netmhciipan-bin", "nmp", "--esmfold-cache-dir", "cache",
    ])

    _run_final_metrics(args)

    assert call["return_v2_metrics"] is True
    assert call["legacy_metrics"] is False
    assert call["refold_backend"] == "esmfold2"
    structural = pd.read_parquet(refined_dir / "structural.parquet")
    assert "global_ca_RMSD" in structural and "bb_RMSD" not in structural
    assert (refined_dir / "structural_residues.parquet").exists()
    assert not (refined_dir / "structural_v2.parquet").exists()
    status = pd.read_json(refined_dir / "final_metrics_status.json", typ="series")
    assert status["structural_metrics_version"] == "v2"


def test_run_final_metrics_head_mode_never_builds_or_writes_nmp(tmp_path, monkeypatch):
    from scripts import evaluate_phase_c as evaluator

    refined_dir = tmp_path / "refined"
    refined_dir.mkdir()
    pd.DataFrame([{
        "protein_id": "P1", "design_idx": 0, "sequence": "AAAA", "seed": 0,
        "wall_seconds": 0.0,
    }]).to_parquet(refined_dir / "evaluator_ready.parquet")
    generated = pd.DataFrame([{
        "protein_id": "P1", "design_id": "design_0000", "design_idx": 0,
        "sequence": "AAAA",
    }])
    monkeypatch.setattr(evaluator, "load_generated_designs", lambda _path: generated)
    monkeypatch.setattr(
        evaluator,
        "load_test_lookup",
        lambda _path: (pd.DataFrame(), {"P1": {"sequence": "AAAA"}}),
    )
    monkeypatch.setattr(evaluator, "build_head_predictor", lambda **_kwargs: object())

    def forbidden_nmp(**_kwargs):
        raise AssertionError("Head mode must not construct NetMHCIIpan")

    monkeypatch.setattr(evaluator, "build_nmp_runner", forbidden_nmp)
    immune_call = {}

    def fake_immune(*_args, **kwargs):
        immune_call.update(kwargs)
        return (
            pd.DataFrame([{"protein_id": "P1", "design_id": "design_0000"}]),
            pd.DataFrame(),
            [],
            pd.DataFrame(),
            pd.DataFrame(),
        )

    monkeypatch.setattr(evaluator, "evaluate_immunogenicity_rows", fake_immune)
    monkeypatch.setattr(evaluator, "_load_anchor_indices_by_protein", lambda *_args: {})
    monkeypatch.setattr(
        evaluator,
        "evaluate_structural_rows",
        lambda *_args, **_kwargs: (pd.DataFrame(), [], pd.DataFrame()),
    )
    monkeypatch.setattr(refine_driver, "_free_gpu", lambda: None)
    args = build_arg_parser().parse_args(['--no-official',
        "--seed-table", "unused", "--allele", "A", "--out-dir", str(tmp_path),
        "--target-source", "head", "--test-set-parquet", "test.parquet",
        "--pdb-root", "pdb", "--head-checkpoint", "head.pt",
        "--head-config-dir", "configs", "--esmfold-cache-dir", "cache",
    ])

    _run_final_metrics(args)

    assert immune_call["run_nmp"] is False
    assert immune_call["nmp_runner"] is None
    assert (refined_dir / "imm_head.parquet").exists()
    assert not (refined_dir / "imm_nmp.parquet").exists()
    assert not (refined_dir / "imm_nmp_peptides.parquet").exists()


def test_run_final_metrics_skips_when_no_evaluator_ready(tmp_path, capsys):
    # No refined/evaluator_ready.parquet -> graceful skip (no crash, no output files, no heavy import).
    args = build_arg_parser().parse_args(
        ['--target-source', 'nmp', '--no-official', "--seed-table", "t", "--allele", "A", "--out-dir", str(tmp_path)])
    _run_final_metrics(args)
    assert "skipping" in capsys.readouterr().out
    assert not list(tmp_path.rglob("imm_head.parquet"))
    assert not list(tmp_path.rglob("structural.parquet"))


def test_atomic_write_roundtrips(tmp_path):
    df = pd.DataFrame({"protein_id": ["A", "B"], "design_idx": [0, 1], "scTM": [0.9, 0.8]})
    _atomic_write(tmp_path, df, "structural.parquet")
    back = pd.read_parquet(tmp_path / "structural.parquet")
    pd.testing.assert_frame_equal(back, df)
    assert not (tmp_path / "structural.parquet.tmp").exists()   # temp cleaned by atomic rename
