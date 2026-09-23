"""Phase B4 evaluator contract tests."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_phase_c import (
    aggregate_nmp_scores_with_threshold,
    build_manifest,
    compute_recovery,
    evaluate_immunogenicity_rows,
    evaluate_structural_rows,
    load_generated_designs,
    mode_outputs_exist,
    output_paths,
    probe_nmp_version,
    resolve_nmp_runtime_params,
)


_AA3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


def _write_ca_pdb(path: Path, sequence: str, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
    lines = []
    dx, dy, dz = offset
    for idx, aa in enumerate(sequence, start=1):
        x = (idx - 1) * 1.6 + dx
        y = ((idx - 1) % 2) * 0.7 + dy
        z = ((idx - 1) % 3) * 0.4 + dz
        lines.append(
            f"ATOM  {idx:5d}  CA  {_AA3[aa]:>3} A{idx:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00           C\n"
        )
    lines.append("END\n")
    path.write_text("".join(lines))


_TEST_SIDECHAINS = {
    "A": ("CB",),
    "C": ("CB", "SG"),
    "G": (),
    "T": ("CB", "OG1", "CG2"),
}


def _write_all_atom_pdb(
    path: Path,
    sequence: str,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    plddt: float = 88.0,
) -> None:
    lines = []
    serial = 1
    dx, dy, dz = offset
    for idx, aa in enumerate(sequence, start=1):
        ca = (
            (idx - 1) * 1.6 + dx,
            ((idx - 1) % 2) * 0.7 + dy,
            ((idx - 1) % 3) * 0.4 + dz,
        )
        atoms = {
            "N": (ca[0] - 0.5, ca[1], ca[2]),
            "CA": ca,
            "C": (ca[0] + 0.5, ca[1], ca[2]),
            "O": (ca[0] + 0.8, ca[1] + 0.2, ca[2]),
        }
        for atom_idx, atom_name in enumerate(_TEST_SIDECHAINS[aa], start=1):
            atoms[atom_name] = (ca[0], ca[1] + atom_idx * 0.5, ca[2] + atom_idx * 0.2)
        for atom_name, (x, y, z) in atoms.items():
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {_AA3[aa]:>3} A{idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{plddt:6.2f}          {atom_name[0]:>2}\n"
            )
            serial += 1
    lines.append("END\n")
    path.write_text("".join(lines))


class _FakePredictor:
    def predict_protein(self, sequence: str):
        hotspot = [float(idx) for idx, _ in enumerate(sequence, start=1)]
        return {
            "global_risk": float(len(sequence)) / 10.0,
            "residue_hotspot": pd.Series(hotspot, dtype=float).to_numpy(),
        }


class _FakeRunner:
    def score_batch(self, entries, allele, pep_lengths):
        del allele, pep_lengths
        out = {}
        for pid, seq in entries:
            scores = []
            for idx in range(max(1, len(seq) - 2)):
                rank = 0.01 if idx == 0 else 0.03
                scores.append(
                    SimpleNamespace(
                        peptide=seq[idx: idx + 3],
                        el_rank=rank,
                        pos=idx,
                        core="AAA",
                        el_score=1.0,
                    )
                )
            out[pid] = {12: scores}
        return out


def _generated_fixture() -> pd.DataFrame:
    rows = []
    for protein_id, seqs in {
        "p1": ["AAAA", "AAAT"],
        "p2": ["CCCCC", "CCCCA"],
        "p3": ["GGGGGG", "GGGGGA"],
    }.items():
        for design_idx, sequence in enumerate(seqs):
            rows.append(
                {
                    "protein_id": protein_id,
                    "design_idx": design_idx,
                    "design_id": f"design_{design_idx:04d}",
                    "sequence": sequence,
                    "seed": 42,
                    "wall_seconds": 0.1,
                }
            )
    return pd.DataFrame(rows)


def _test_lookup_fixture(tmp_path: Path) -> dict[str, dict]:
    lookup = {}
    for protein_id, sequence in {
        "p1": "AAAA",
        "p2": "CCCCC",
        "p3": "GGGGGG",
    }.items():
        pdb_path = tmp_path / f"{protein_id}.pdb"
        _write_ca_pdb(pdb_path, sequence)
        lookup[protein_id] = {
            "protein_id": protein_id,
            "sequence": sequence,
            "sequence_length": len(sequence),
            "pdb_path": pdb_path.name,
        }
    return lookup


def test_evaluate_immunogenicity_rows_matches_predictor_and_nmp_counts():
    generated = _generated_fixture()
    head_df, nmp_df, failures = evaluate_immunogenicity_rows(
        generated,
        predictor=_FakePredictor(),
        nmp_runner=_FakeRunner(),
        allele="HLA-DRB1*07:01",
        strong_binder_threshold=2.0,
        nmp_batch_size=2,
        progress_every=0,
    )
    assert failures == []
    assert len(head_df) == 6
    assert len(nmp_df) == 6
    assert head_df.iloc[0]["global_risk"] == pytest.approx(0.4)
    assert nmp_df.iloc[0]["n_strong_binders"] == 1


def test_selection_authority_round_trips_through_head_and_nmp_outputs(tmp_path):
    generated = _generated_fixture()
    generated["selection_status"] = "feasible_immune_pareto"
    generated["terminal_validated"] = True
    generated["structure_feasible"] = True
    generated["selection_provenance_digest"] = [
        f"{index:064x}" for index in range(1, len(generated) + 1)
    ]
    path = tmp_path / "generated.parquet"
    generated.to_parquet(path, index=False)
    loaded = load_generated_designs(path)

    head_df, nmp_df, failures = evaluate_immunogenicity_rows(
        loaded,
        predictor=_FakePredictor(),
        nmp_runner=_FakeRunner(),
        allele="HLA-DRB1*07:01",
        nmp_batch_size=2,
        progress_every=0,
    )

    assert failures == []
    authority = [
        "selection_status",
        "terminal_validated",
        "structure_feasible",
        "selection_provenance_digest",
    ]
    pd.testing.assert_frame_equal(
        head_df[["protein_id", "design_idx", *authority]].reset_index(drop=True),
        loaded[["protein_id", "design_idx", *authority]].reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        nmp_df[["protein_id", "design_idx", *authority]].reset_index(drop=True),
        loaded[["protein_id", "design_idx", *authority]].reset_index(drop=True),
        check_dtype=False,
    )


def test_selection_authority_partial_input_fails_closed(tmp_path):
    generated = _generated_fixture()
    generated["selection_status"] = "structure_rejected_fallback"
    path = tmp_path / "generated.parquet"
    generated.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="selection authority columns"):
        load_generated_designs(path)


def test_probe_nmp_version_prefers_standard_data_version_file(tmp_path, monkeypatch):
    binary = tmp_path / "netMHCIIpan"
    binary.write_text("binary-placeholder")
    version_path = tmp_path / "data" / "version"
    version_path.parent.mkdir()
    version_path.write_text("  NetMHCIIpan version 4.3i\n")

    def _must_not_probe(*args, **kwargs):
        del args, kwargs
        raise AssertionError("binary probe must not run when data/version exists")

    monkeypatch.setattr("scripts.evaluate_phase_c.subprocess.run", _must_not_probe)

    assert probe_nmp_version(binary) == "NetMHCIIpan version 4.3i"


def test_probe_nmp_version_fallback_accepts_only_explicit_version_line(
    tmp_path, monkeypatch,
):
    binary = tmp_path / "netMHCIIpan"
    binary.write_text("binary-placeholder")
    monkeypatch.setattr(
        "scripts.evaluate_phase_c.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="NetMHCIIpan version 4.3i\n", stderr="", returncode=0,
        ),
    )

    assert probe_nmp_version(binary) == "NetMHCIIpan version 4.3i"


def test_probe_nmp_version_never_records_usage_or_error_as_version(
    tmp_path, monkeypatch,
):
    binary = tmp_path / "netMHCIIpan"
    binary.write_text("binary-placeholder")
    monkeypatch.setattr(
        "scripts.evaluate_phase_c.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="Usage: netMHCIIpan [options]\n",
            stderr="ERROR: input file is missing\n",
            returncode=1,
        ),
    )

    assert probe_nmp_version(binary) is None


def test_resolve_nmp_runtime_params_original_matches_iedb_baseline():
    args = SimpleNamespace(
        nmp_mode="original",
        nmp_batch_size=8,
        nmp_max_lengths_per_call=4,
        nmp_workers=8,
    )

    assert resolve_nmp_runtime_params(args) == (1, 14, 1)


def test_resolve_nmp_runtime_params_accelerated_uses_explicit_knobs():
    args = SimpleNamespace(
        nmp_mode="accelerated",
        nmp_batch_size=6,
        nmp_max_lengths_per_call=3,
        nmp_workers=5,
    )

    assert resolve_nmp_runtime_params(args) == (6, 3, 5)


def test_evaluate_structural_rows_emits_expected_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    generated = _generated_fixture()
    test_lookup = _test_lookup_fixture(tmp_path)

    def _fake_refold(sequence, protein_id, design_id, backend, cache_dir=None, model=None):
        pdb_path = tmp_path / f"{protein_id}_{design_id}.pdb"
        _write_ca_pdb(pdb_path, sequence, offset=(10.0, -2.0, 5.0))
        return {
            "pdb_path": str(pdb_path),
            "pLDDT": 88.0,
        }

    monkeypatch.setattr(
        "scripts.evaluate_phase_c.load_refold_model",
        lambda backend, device="cuda": object(),
    )
    monkeypatch.setattr(
        "scripts.evaluate_phase_c.refold",
        _fake_refold,
    )
    monkeypatch.setattr(
        "inverse_folding.evaluation.tmalign.run_tmalign",
        lambda pred_pdb, ref_pdb, tmalign_bin="TMalign", cache_dir=None: {
            "tm_score": 0.75,
            "rmsd": 1.25,
        },
    )
    monkeypatch.setattr(
        "inverse_folding.reference_flow.runtime.resolve_structure_path",
        lambda entry, pdb_root: Path(pdb_root) / str(entry["pdb_path"]),
    )

    df, failures, residues_df = evaluate_structural_rows(
        generated,
        test_lookup,
        pdb_root=tmp_path,
        refold_backend="esmfold",
        device="cuda",
        tmalign_bin="TMalign",
        esmfold_cache_dir=str(tmp_path / ".cache"),
        progress_every=0,
        return_residue_metrics=True,
    )
    assert failures == []
    assert len(df) == 6
    assert set(["protein_id", "design_id", "design_idx", "sequence", "scTM", "pLDDT", "bb_RMSD", "scRMSD", "recovery", "foldability", "refold_backend"]).issubset(df.columns)
    assert (df["refold_backend"] == "esmfold").all()
    assert df.loc[(df["protein_id"] == "p1") & (df["design_idx"] == 1), "recovery"].iloc[0] == pytest.approx(0.75)
    assert df["scRMSD"].max() < 1e-5
    assert len(residues_df) == sum(len(seq) for seq in generated["sequence"])
    assert set(
        [
            "protein_id",
            "design_id",
            "design_idx",
            "residue_idx",
            "residue_idx_1based",
            "ref_aa",
            "design_aa",
            "sc_ca_distance",
            "aligned_pred_ca_x",
        ]
    ).issubset(residues_df.columns)
    p1_mut = residues_df[
        (residues_df["protein_id"] == "p1")
        & (residues_df["design_idx"] == 1)
        & (residues_df["residue_idx"] == 3)
    ].iloc[0]
    assert p1_mut["residue_idx_1based"] == 4
    assert p1_mut["ref_aa"] == "A"
    assert p1_mut["design_aa"] == "T"
    assert residues_df["sc_ca_distance"].max() < 1e-5


def test_evaluate_structural_rows_v2_is_complete_and_indexable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from inverse_folding.evaluation.schema import (
        STRUCTURAL_V2_COLUMNS,
        STRUCTURAL_V2_RESIDUE_COLUMNS,
    )

    generated = _generated_fixture()
    test_lookup = _test_lookup_fixture(tmp_path)
    for protein_id, test_row in test_lookup.items():
        _write_all_atom_pdb(tmp_path / test_row["pdb_path"], test_row["sequence"])

    def _fake_refold(sequence, protein_id, design_id, backend, cache_dir=None, model=None):
        pdb_path = tmp_path / f"{protein_id}_{design_id}.pdb"
        _write_all_atom_pdb(pdb_path, sequence, offset=(10.0, -2.0, 5.0))
        return {"pdb_path": str(pdb_path), "pLDDT": 88.0}

    monkeypatch.setattr("scripts.evaluate_phase_c.load_refold_model", lambda *args, **kwargs: object())
    monkeypatch.setattr("scripts.evaluate_phase_c.refold", _fake_refold)
    monkeypatch.setattr(
        "inverse_folding.evaluation.tmalign.run_tmalign",
        lambda **kwargs: {"tm_score": 0.91, "rmsd": 1.2},
    )
    monkeypatch.setattr(
        "inverse_folding.reference_flow.runtime.resolve_structure_path",
        lambda entry, pdb_root: Path(pdb_root) / str(entry["pdb_path"]),
    )

    monkeypatch.setattr(
        "inverse_folding.evaluation.sc_rmsd.compute_ca_self_consistency",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy C-alpha RMSD must not run in canonical v2 mode")
        ),
    )

    v2, failures, v2_residues = evaluate_structural_rows(
        generated,
        test_lookup,
        pdb_root=tmp_path,
        refold_backend="esmfold2",
        device="cpu",
        tmalign_bin="TMalign",
        esmfold_cache_dir=str(tmp_path / ".cache"),
        progress_every=0,
        return_residue_metrics=True,
        return_v2_metrics=True,
        legacy_metrics=False,
        anchor_indices_by_protein={"p1": {0}, "p2": {0}},
    )

    assert failures == []
    assert len(v2) == 6
    assert STRUCTURAL_V2_COLUMNS <= set(v2.columns)
    assert STRUCTURAL_V2_RESIDUE_COLUMNS <= set(v2_residues.columns)
    assert v2["global_ca_RMSD"].max() < 1e-5
    assert v2["pLDDT"].tolist() == pytest.approx([88.0] * len(v2))
    assert v2.loc[v2["protein_id"] == "p1", "active_site_complete"].all()
    assert len(v2_residues) == sum(len(seq) for seq in generated["sequence"])

    changed = v2_residues[
        (v2_residues["protein_id"] == "p1")
        & (v2_residues["design_idx"] == 1)
        & (v2_residues["residue_idx"] == 3)
    ].iloc[0]
    assert changed["match_status"] == "residue_identity_mismatch"
    assert pd.isna(changed["sidechain_RMSD"])


def test_af3_refold_backend_is_cache_read():
    # af3 is now a cache-read backend (official DeepMind AlphaFold3); with no cache_dir
    # it fails fast with a clear, non-OOM error (not NotImplementedError).
    from inverse_folding.evaluation.refold import refold

    with pytest.raises(RuntimeError) as exc:
        refold("AAAA", "p1", "design_0000", backend="af3", cache_dir=None)
    assert "cache_dir" in str(exc.value)
    assert "out of memory" not in str(exc.value).lower()


def test_build_manifest_populates_digests(tmp_path: Path):
    generated_path = tmp_path / "generated.parquet"
    generated_df = pd.DataFrame(
        [{"protein_id": "p1", "design_idx": 0, "sequence": "AAAA", "seed": 42, "wall_seconds": 0.1}]
    )
    generated_df.to_parquet(generated_path, index=False)
    test_set_path = tmp_path / "test.parquet"
    pd.DataFrame(
        [{"protein_id": "p1", "sequence": "AAAA", "sequence_length": 4, "pdb_path": "p1.pdb"}]
    ).to_parquet(test_set_path, index=False)
    ckpt_path = tmp_path / "best.pt"
    ckpt_path.write_bytes(b"checkpoint")
    nmp_bin = tmp_path / "netMHCIIpan"
    nmp_bin.write_text("#!/bin/sh\necho netMHCIIpan 4.3\n")
    nmp_bin.chmod(0o755)

    args = SimpleNamespace(
        generated_parquet=str(generated_path),
        test_set_parquet=str(test_set_path),
        epitope_ckpt=str(ckpt_path),
        netmhciipan_bin=str(nmp_bin),
        allele="HLA-DRB1*07:01",
        refold_model="esmfold2",
        structural_metrics_v2=True,
        constraint_manifest=None,
    )
    manifest = build_manifest(
        args=args,
        run_id="eval_all_esmfold_DRB1_07_01_20260424T080000Z",
        modes_run=["imm", "struct"],
        rows_per_mode={"imm": 1, "struct": 1},
        wall_seconds_per_mode={"imm": 1.0, "struct": 2.0},
        generated_df=generated_df,
    )
    assert manifest["git_sha"]
    assert manifest["head_ckpt_digest"]
    assert manifest["generated_parquet_sha256"]
    assert manifest["structural_metrics_v2"] is True
    assert manifest["structural_metrics_version"] == "v2"
    assert manifest["constraint_manifest_digest"] is None


def test_aggregate_nmp_scores_with_custom_threshold():
    scores = pd.DataFrame(
        [
            {"rank_EL": 1.5},
            {"rank_EL": 2.5},
            {"rank_EL": 9.0},
        ]
    )
    agg = aggregate_nmp_scores_with_threshold(scores, strong_binder_threshold=2.0)
    assert agg["n_strong_binders"] == 1
    assert agg["n_weak_binders"] == 3
    assert math.isclose(agg["mean_best_rank"], (1.5 + 2.5 + 9.0) / 3.0)


def test_compute_recovery_matches_expected_identity():
    assert compute_recovery("AAAT", "AAAA") == pytest.approx(0.75)


def test_structural_v2_completion_requires_both_standalone_files(tmp_path: Path):
    from inverse_folding.evaluation.schema import (
        STRUCTURAL_COLUMNS,
        STRUCTURAL_RESIDUE_COLUMNS,
    )

    paths = output_paths(tmp_path)
    pd.DataFrame(columns=sorted(STRUCTURAL_COLUMNS)).to_parquet(paths["structural"])
    pd.DataFrame(columns=sorted(STRUCTURAL_RESIDUE_COLUMNS)).to_parquet(
        paths["structural_residues"]
    )
    complete, _ = mode_outputs_exist("struct", paths)
    assert complete is True
    assert "structural_v2" not in paths
    assert "structural_v2_residues" not in paths
