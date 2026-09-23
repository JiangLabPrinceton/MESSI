"""Tests for post-prediction protein-complex interface metrics (D2 tetramer + hetero-dimer)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inverse_folding.evaluation.complex_interfaces import (
    aggregate_homomer_alanine_scan,
    canonical_protein_atoms,
    classify_d2_interface_pairs,
    classify_interface_pairs,
    interface_pair_metrics,
    interface_residue_candidates,
    pair_residue_contacts,
    pair_salt_bridges,
    parse_rosetta_ddg_scan_log,
    parse_rosetta_scorefile,
    _reconcile_rosetta_alanine_rows,
    summarize_interface_pairs,
)
from scripts.eval_complex_gate import add_interface_wt_normalization, tier1_from_summary
from scripts.eval_tetramer_reference import _paths_overlap, main as reference_main


def _pair_table() -> pd.DataFrame:
    values = {
        ("A", "B"): 6000.0,
        ("A", "C"): 700.0,
        ("A", "D"): 5100.0,
        ("B", "C"): 5000.0,
        ("B", "D"): 650.0,
        ("C", "D"): 5900.0,
    }
    return pd.DataFrame([
        {
            "chain_1": a,
            "chain_2": b,
            "bsa_total_a2": bsa,
            "bsa_per_partner_a2": bsa / 2,
            "n_residue_contacts_8a": int(bsa / 30),
            "n_salt_bridges_4a": int(bsa / 2000),
        }
        for (a, b), bsa in values.items()
    ])


def _atom_array(rows):
    import biotite.structure as struc

    arr = struc.AtomArray(len(rows))
    arr.coord = np.asarray([row[0] for row in rows], dtype=float)
    arr.chain_id = np.asarray([row[1] for row in rows])
    arr.res_id = np.asarray([row[2] for row in rows])
    arr.res_name = np.asarray([row[3] for row in rows])
    arr.atom_name = np.asarray([row[4] for row in rows])
    arr.element = np.asarray([row[5] for row in rows])
    arr.hetero = np.asarray([
        row[6] if len(row) > 6 else False
        for row in rows
    ], dtype=bool)
    return arr


def _residue(chain, res_id, res_name, origin, *, hetero=False, sidechain="CB"):
    x, y, z = origin
    rows = [
        ((x, y, z), chain, res_id, res_name, "N", "N", hetero),
        ((x + 1.0, y, z), chain, res_id, res_name, "CA", "C", hetero),
        ((x + 2.0, y, z), chain, res_id, res_name, "C", "C", hetero),
        ((x + 2.5, y, z), chain, res_id, res_name, "O", "O", hetero),
    ]
    if sidechain:
        rows.append(((x + 1.0, y + 1.0, z), chain, res_id, res_name,
                     sidechain, "O" if sidechain == "OG" else "C", hetero))
    return rows


def test_d2_classification_keeps_disjoint_symmetry_copies_together():
    out = classify_d2_interface_pairs(_pair_table())
    classes = {
        row.chain_1 + row.chain_2: row.interface_class
        for row in out.itertuples()
    }
    assert classes == {
        "AB": "interface_1",
        "AC": "diagonal",
        "AD": "interface_2",
        "BC": "interface_2",
        "BD": "diagonal",
        "CD": "interface_1",
    }
    assert out.groupby("interface_class").size().to_dict() == {
        "diagonal": 2,
        "interface_1": 2,
        "interface_2": 2,
    }


def test_interface_summary_uses_worst_copy_in_the_correct_direction():
    pairs = classify_d2_interface_pairs(_pair_table())
    pairs["rosetta_dg_separated_reu"] = [-20, np.nan, -12, -10, np.nan, -25]
    pairs["rosetta_delta_unsat_hbonds"] = [1, np.nan, 3, 5, np.nan, 2]
    summary = summarize_interface_pairs(pairs)

    assert summary["interface_1_bsa_total_min_a2"] == 5900.0
    assert summary["interface_2_bsa_total_min_a2"] == 5000.0
    assert summary["diagonal_bsa_total_max_a2"] == 700.0
    assert summary["interface_1_rosetta_dg_separated_max_reu"] == -20.0
    assert summary["interface_2_rosetta_dg_separated_max_reu"] == -10.0
    assert summary["interface_2_rosetta_delta_unsat_hbonds_max"] == 5.0
    assert summary["interface_topology_bsa_gap_a2"] == 4300.0


def test_hetero_dimer_classification_marks_the_single_pair_biological():
    """A binder:target complex has one interface and no symmetry copy or diagonal."""
    pairs = pd.DataFrame([{
        "chain_1": "A", "chain_2": "B", "bsa_total_a2": 1800.0,
        "bsa_per_partner_a2": 900.0, "n_residue_contacts_8a": 60, "n_salt_bridges_4a": 3,
    }])
    out = classify_interface_pairs(pairs)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["interface_class"] == "interface_1"
    assert row["interface_class_rank"] == 1
    assert row["interface_copy"] == 1
    assert bool(row["is_biological_interface"]) is True
    # the D2 columns exist for schema stability but carry no meaning without a D2 graph
    assert row["d2_matching"] == ""
    assert np.isnan(row["d2_matching_mean_bsa_total_a2"])


def test_hetero_dimer_summary_degenerates_to_the_single_copy_and_omits_topology():
    """With no symmetry copy the 'worst copy' statistic equals the mean; no diagonal to compare."""
    pairs = classify_interface_pairs(pd.DataFrame([{
        "chain_1": "A", "chain_2": "B", "bsa_total_a2": 1800.0,
        "bsa_per_partner_a2": 900.0, "n_residue_contacts_8a": 60, "n_salt_bridges_4a": 3,
    }]))
    pairs["rosetta_dg_separated_reu"] = [-31.5]
    summary = summarize_interface_pairs(pairs)

    assert summary["interface_1_bsa_total_mean_a2"] == 1800.0
    assert summary["interface_1_bsa_total_min_a2"] == 1800.0
    assert summary["interface_1_rosetta_dg_separated_max_reu"] == -31.5
    assert not any(k.startswith("interface_2_") for k in summary)
    assert not any(k.startswith("diagonal_") for k in summary)
    assert "interface_topology_bsa_gap_a2" not in summary


def test_d2_summary_is_unchanged_when_the_hetero_path_exists():
    """Regression guard: the four-chain uricase summary must be byte-identical to before."""
    pairs = classify_interface_pairs(_pair_table())
    assert pairs.equals(classify_d2_interface_pairs(_pair_table()))
    pairs["rosetta_dg_separated_reu"] = [-20, np.nan, -12, -10, np.nan, -25]
    summary = summarize_interface_pairs(pairs)
    assert summary["interface_1_bsa_total_min_a2"] == 5900.0
    assert summary["diagonal_bsa_total_max_a2"] == 700.0
    assert summary["interface_topology_bsa_gap_a2"] == 4300.0


def test_interface_pair_metrics_supports_two_chains_and_rejects_unsupported_counts():
    rows = []
    rows += _residue("A", 1, "LEU", (0.0, 0.0, 0.0))
    rows += _residue("B", 1, "ASP", (0.0, 4.0, 0.0))
    rows += _residue("C", 1, "LYS", (0.0, 8.0, 0.0))
    arr2 = _atom_array(rows[:10])
    out = interface_pair_metrics(arr2, ["A", "B"])
    assert list(out["chain_pair"]) == ["A:B"]
    assert bool(out.iloc[0]["is_biological_interface"]) is True

    with pytest.raises(ValueError, match="two or four protein chains"):
        interface_pair_metrics(_atom_array(rows), ["A", "B", "C"])


def test_tier1_confidence_clamps_to_the_available_chain_count():
    """A two-chain complex must not index a 4x4 slice out of a 2x2 confidence matrix."""
    j = {
        "iptm": 0.95, "ptm": 0.95, "plddt": 0.9, "ranking_score": 0.9, "has_clash": False,
        "chain_pair_iptm": [[0.98, 0.95], [0.95, 0.97]],
        "chain_pair_gpde": [[0.5, 1.2], [1.2, 0.6]],
        "chain_plddt": [0.93, 0.88],
    }
    t1 = tier1_from_summary(j, n_protein=4)
    assert t1["min_pp_chain_pair_iptm"] == pytest.approx(0.95)
    assert t1["mean_pp_chain_pair_gpde"] == pytest.approx(1.2)
    assert t1["min_prot_chain_plddt"] == pytest.approx(0.88)


def test_contacts_and_salt_bridges_count_residue_pairs_not_atom_pairs():
    chain_a = _atom_array([
        ((0.0, 0.0, 0.0), "A", 1, "ASP", "CB", "C"),
        ((0.0, 0.0, 0.0), "A", 1, "ASP", "OD1", "O"),
        ((0.0, 0.5, 0.0), "A", 1, "ASP", "OD2", "O"),
        ((20.0, 0.0, 0.0), "A", 2, "GLY", "CA", "C"),
    ])
    chain_b = _atom_array([
        ((3.5, 0.0, 0.0), "B", 1, "ARG", "CB", "C"),
        ((3.5, 0.0, 0.0), "B", 1, "ARG", "NH1", "N"),
        ((3.5, 0.5, 0.0), "B", 1, "ARG", "NH2", "N"),
        ((40.0, 0.0, 0.0), "B", 2, "GLY", "CA", "C"),
    ])
    assert pair_residue_contacts(chain_a, chain_b, cutoff=8.0) == 1
    assert pair_salt_bridges(chain_a, chain_b, cutoff=4.0) == 1


def test_canonical_protein_atoms_keeps_complete_polymer_and_normalizes_sac():
    rows = []
    rows += _residue("A", 1, "SAC", (0.0, 0.0, 0.0), hetero=True, sidechain="OG")
    rows.append(((-1.0, 0.0, 0.0), "A", 1, "SAC", "C1A", "C", True))
    rows += _residue("A", 2, "ALA", (4.0, 0.0, 0.0))
    rows.append(((8.0, 0.0, 0.0), "A", 296, "SER", "N", "N", False))
    rows.extend([
        ((9.0, 0.0, 0.0), "A", 700, "CYS", "CB", "C", True),
        ((10.0, 0.0, 0.0), "A", 700, "CYS", "SG", "S", True),
    ])

    out = canonical_protein_atoms(_atom_array(rows))

    assert set(out.res_id) == {1, 2}
    assert set(out.res_name[out.res_id == 1]) == {"SER"}
    assert "C1A" not in set(out.atom_name)
    assert not out.hetero.any()


def test_interface_residue_candidates_returns_both_sides_and_min_distance():
    rows = []
    rows += _residue("A", 10, "LEU", (0.0, 0.0, 0.0))
    rows += _residue("A", 20, "LYS", (30.0, 0.0, 0.0))
    rows += _residue("B", 11, "ASP", (0.0, 4.0, 0.0))
    rows += _residue("B", 21, "GLU", (60.0, 0.0, 0.0))
    arr = _atom_array(rows)

    out = interface_residue_candidates(arr, "A", "B", cutoff=5.0)

    assert set(zip(out["source_chain"], out["res_id"])) == {("A", 10), ("B", 11)}
    assert out["min_cross_chain_heavy_atom_distance_a"].max() == pytest.approx(3.0)


def test_parse_and_aggregate_rosetta_ddg_scan_output():
    wt, parsed = parse_rosetta_ddg_scan_log(
        "job wild-type binding ddG = -155.368\n"
        " Residue A49ALA->ALA : 0.0000\n"
        " Residue B236GLU->ALA : -6.2583\n"
    )
    assert wt == pytest.approx(-155.368)
    assert parsed.to_dict("records") == [
        {
            "rosetta_chain": "A", "res_id": 49, "wt_residue_name3": "ALA",
            "mut_residue_name3": "ALA", "ddg_bind_reu": 0.0,
        },
        {
            "rosetta_chain": "B", "res_id": 236, "wt_residue_name3": "GLU",
            "mut_residue_name3": "ALA", "ddg_bind_reu": -6.2583,
        },
    ]

    long = pd.DataFrame([
        {"interface_pair": "A:B", "source_chain": "A", "res_id": 10,
         "ins_code": "", "wt_residue_name3": "LEU", "ddg_bind_reu": 2.0,
         "is_native_alanine": False, "is_glycine_to_alanine": False,
         "is_proline_to_alanine": False,
         "is_interpretable_sidechain_alanine": True},
        {"interface_pair": "A:B", "source_chain": "B", "res_id": 10,
         "ins_code": "", "wt_residue_name3": "LEU", "ddg_bind_reu": 2.2,
         "is_native_alanine": False, "is_glycine_to_alanine": False,
         "is_proline_to_alanine": False,
         "is_interpretable_sidechain_alanine": True},
    ])
    by_position = aggregate_homomer_alanine_scan(long)
    assert by_position.loc[0, "n_chain_sides"] == 2
    assert by_position.loc[0, "ddg_bind_mean_reu"] == pytest.approx(2.1)
    assert by_position.loc[0, "ddg_bind_chain_range_reu"] == pytest.approx(0.2)


def test_disulfide_cysteines_omitted_by_rosetta_are_retained_as_noninterpretable():
    rows = []
    rows += _residue("A", 10, "CYS", (0.0, 0.0, 0.0), sidechain=None)
    rows += _residue("A", 20, "CYS", (0.0, 3.0, 0.0), sidechain=None)
    rows += _residue("B", 11, "LEU", (0.0, 0.0, 3.0))
    rows.extend(
        [
            ((1.0, 1.0, 0.0), "A", 10, "CYS", "SG", "S", False),
            ((1.0, 3.05, 0.0), "A", 20, "CYS", "SG", "S", False),
        ]
    )
    arr = _atom_array(rows)
    candidates = pd.DataFrame(
        [
            {
                "source_chain": "A", "rosetta_chain": "A", "res_id": res_id,
                "ins_code": "", "wt_residue_name3": "CYS",
                "is_interpretable_sidechain_alanine": True,
            }
            for res_id in (10, 20)
        ]
        + [
            {
                "source_chain": "B", "rosetta_chain": "B", "res_id": 11,
                "ins_code": "", "wt_residue_name3": "LEU",
                "is_interpretable_sidechain_alanine": True,
            }
        ]
    )
    parsed = pd.DataFrame(
        [
            {
                "rosetta_chain": "B", "res_id": 11,
                "wt_residue_name3": "LEU", "mut_residue_name3": "ALA",
                "ddg_bind_reu": 1.25,
            }
        ]
    )

    merged = _reconcile_rosetta_alanine_rows(
        arr,
        candidates,
        parsed,
        chain_a="A",
        chain_b="B",
    )

    disulfides = merged[merged["is_disulfide_cysteine"]]
    assert set(disulfides.res_id) == {10, 20}
    assert disulfides.ddg_bind_reu.isna().all()
    assert not disulfides.is_interpretable_sidechain_alanine.any()
    assert merged.loc[merged.res_id.eq(11), "ddg_bind_reu"].iloc[0] == pytest.approx(1.25)


def test_one_sided_disulfide_makes_homomer_position_noninterpretable():
    long = pd.DataFrame(
        [
            {
                "interface_pair": "A:C",
                "source_chain": "A",
                "res_id": 145,
                "ins_code": "",
                "wt_residue_name3": "CYS",
                "ddg_bind_reu": float("nan"),
                "is_native_alanine": False,
                "is_glycine_to_alanine": False,
                "is_proline_to_alanine": False,
                "is_disulfide_cysteine": True,
                "is_interpretable_sidechain_alanine": False,
            },
            {
                "interface_pair": "A:C",
                "source_chain": "C",
                "res_id": 145,
                "ins_code": "",
                "wt_residue_name3": "CYS",
                "ddg_bind_reu": -117.0,
                "is_native_alanine": False,
                "is_glycine_to_alanine": False,
                "is_proline_to_alanine": False,
                "is_disulfide_cysteine": False,
                "is_interpretable_sidechain_alanine": True,
            },
        ]
    )
    by_position = aggregate_homomer_alanine_scan(long).iloc[0]
    assert bool(by_position.is_disulfide_cysteine)
    assert not bool(by_position.is_interpretable_sidechain_alanine)
    assert by_position.ddg_bind_mean_reu == pytest.approx(-117.0)
    assert pd.isna(by_position.sidechain_ddg_bind_rank_desc)


def test_parse_rosetta_scorefile_preserves_canonical_interface_fields(tmp_path):
    score = tmp_path / "scores.sc"
    score.write_text(
        "SEQUENCE:\n"
        "SCORE: total_score dSASA_int dG_separated dG_separated/dSASAx100 "
        "packstat hbonds_int hbond_E_fraction delta_unsatHbonds description\n"
        "SCORE: -100.0 2100.0 -20.0 -0.952 0.71 8 0.23 2 ifc_000000_0001\n"
    )
    out = parse_rosetta_scorefile(score)
    assert out.loc[0, "dG_separated/dSASAx100"] == pytest.approx(-0.952)
    assert out.loc[0, "packstat"] == pytest.approx(0.71)
    assert out.loc[0, "description"] == "ifc_000000_0001"


def test_wt_normalization_uses_real_parent_wt_and_can_fail_fast():
    df = pd.DataFrame([
        {
            "parent": "P1", "kind": "WT", "predicted": True,
            "interface_metrics_ok": True, "interface_1_bsa_total_min_a2": 5000.0,
        },
        {
            "parent": "P1", "kind": "design", "predicted": True,
            "interface_metrics_ok": True, "interface_1_bsa_total_min_a2": 4000.0,
        },
    ])
    out = add_interface_wt_normalization(df, require_wt=True)
    assert out.loc[1, "wt_interface_1_bsa_total_min_a2"] == 5000.0
    assert out.loc[1, "interface_1_bsa_total_min_a2_ratio_to_wt"] == pytest.approx(0.8)
    assert bool(out.loc[1, "interface_wt_baseline_available"])

    without_wt = df[df["kind"] != "WT"].copy()
    with pytest.raises(ValueError, match="P1"):
        add_interface_wt_normalization(without_wt, require_wt=True)


def test_reference_rosetta_requires_separate_run_and_log_roots(tmp_path, capsys):
    structure = tmp_path / "reference.pdb"
    structure.write_text("")

    with pytest.raises(SystemExit, match="2"):
        reference_main([
            "--structure", str(structure),
            "--name", "ref",
            "--out-dir", str(tmp_path / "work"),
            "--rosetta-interface-analyzer", "InterfaceAnalyzer",
        ])
    assert "requires --run-dir" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        reference_main([
            "--structure", str(structure),
            "--name", "ref",
            "--out-dir", str(tmp_path / "work"),
            "--run-dir", str(tmp_path / "work" / "runtime"),
            "--log-dir", str(tmp_path / "logs"),
            "--rosetta-interface-analyzer", "InterfaceAnalyzer",
        ])
    assert "must be separate directory trees" in capsys.readouterr().err

    assert _paths_overlap(tmp_path / "run", tmp_path / "run" / "nested")
    assert not _paths_overlap(tmp_path / "run", tmp_path / "logs")


# --------------------------------------------------------------------------------------
# Per-parent catalytic numbering.
#
# The Q00511 catalytic panel (Lys11/Thr58/His257 + cross-protomer Asn255) is hardcoded at
# index_0b 10/57/256/254. Measured against the Active-15 constraint manifests, only 2 of the
# 15 parents actually sit at those indices (Q00511 and Q0CXR4); A0A9P8P4R1 is offset by 26
# (Lys11 -> 36). Applying the Q00511 numbers to another parent silently measures the ligand
# to the WRONG residues, so an unresolvable parent must yield NO catalytic panel rather than
# a plausible-looking wrong one.
# --------------------------------------------------------------------------------------

def test_catalytic_residues_default_to_the_legacy_panel_only_for_q00511():
    from scripts.eval_complex_gate import CATALYTIC_CORE, XPROT_W1, resolve_catalytic_residues

    core, partner = resolve_catalytic_residues("Q00511", None)
    assert core == CATALYTIC_CORE
    assert partner == XPROT_W1


def test_catalytic_residues_are_absent_for_an_unmapped_non_q00511_parent():
    from scripts.eval_complex_gate import resolve_catalytic_residues

    core, partner = resolve_catalytic_residues("A0A9P8P4R1", None)
    assert core == {} and partner == {}


def test_catalytic_residues_come_from_the_map_when_supplied():
    from scripts.eval_complex_gate import resolve_catalytic_residues

    cat_map = {
        "A0A9P8P4R1": {"Lys11": 36, "Thr58": 84, "His257": 287, "Asn255": 285},
        "Q00511": {"Lys11": 10, "Thr58": 57, "His257": 256, "Asn255": 254},
    }
    core, partner = resolve_catalytic_residues("A0A9P8P4R1", cat_map)
    assert core == {"Lys11": 36, "Thr58": 84, "His257": 287}
    assert partner == {"Asn255": 285}
    # the map must win over the legacy constants even for Q00511 itself
    core_q, partner_q = resolve_catalytic_residues("Q00511", cat_map)
    assert core_q == {"Lys11": 10, "Thr58": 57, "His257": 256}
    assert partner_q == {"Asn255": 254}


def test_catalytic_map_loads_per_parent_indices_from_a_manifest_table(tmp_path):
    from scripts.eval_complex_gate import load_catalytic_map

    p = tmp_path / "catmap.parquet"
    pd.DataFrame([
        {"parent": "Q00511", "label": "Lys11", "index_0b": 10},
        {"parent": "Q00511", "label": "Asn255", "index_0b": 254},
        {"parent": "P25689", "label": "Lys11", "index_0b": 22},
        {"parent": "P25689", "label": "Asn255", "index_0b": 261},
    ]).to_parquet(p, index=False)
    m = load_catalytic_map(str(p))
    assert m["Q00511"]["Lys11"] == 10
    assert m["P25689"]["Asn255"] == 261


def test_catalytic_map_rejects_a_parent_whose_panel_is_incomplete(tmp_path):
    """A half-populated parent must fail loudly, not silently fall back to Q00511 numbering."""
    from scripts.eval_complex_gate import load_catalytic_map

    p = tmp_path / "partial.parquet"
    pd.DataFrame([
        {"parent": "P25689", "label": "Lys11", "index_0b": 22},
        {"parent": "P25689", "label": "Lys11", "index_0b": 23},
    ]).to_parquet(p, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_catalytic_map(str(p))
