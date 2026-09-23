from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


AA3 = {"K": "LYS", "D": "ASP", "F": "PHE", "G": "GLY"}
SIDECHAIN_ATOMS = {
    "K": ("CB", "CG", "CD", "CE", "NZ"),
    "D": ("CB", "CG", "OD1", "OD2"),
    "F": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
}


def _reference_atoms() -> list[dict[str, tuple[float, float, float]]]:
    ca = [(0.0, 0.0, 0.0), (3.0, 0.5, 0.0), (1.0, 3.0, 1.0)]
    out = []
    for idx, aa in enumerate("KDF"):
        x, y, z = ca[idx]
        atoms = {
            "N": (x - 0.8, y, z),
            "CA": (x, y, z),
            "C": (x + 0.8, y, z),
            "O": (x + 1.2, y + 0.4, z),
        }
        for atom_idx, name in enumerate(SIDECHAIN_ATOMS[aa], start=1):
            atoms[name] = (x, y + atom_idx * 0.7, z + atom_idx * 0.2)
        out.append(atoms)
    return out


def _transform(coord: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = coord
    return (-y + 10.0, x - 4.0, z + 2.0)


def _predicted_atoms() -> list[dict[str, tuple[float, float, float]]]:
    reference = _reference_atoms()
    predicted = [{name: _transform(coord) for name, coord in atoms.items()} for atoms in reference]

    # A real side-chain displacement after global C-alpha alignment.
    nz = reference[0]["NZ"]
    predicted[0]["NZ"] = _transform((nz[0] + 2.0, nz[1], nz[2]))

    # ASP OD1/OD2 are naming-ambiguous and must be matched by the lower-error swap.
    predicted[1]["OD1"] = _transform(reference[1]["OD2"])
    predicted[1]["OD2"] = _transform(reference[1]["OD1"])
    return predicted


def _write_pdb(
    path: Path,
    atoms_by_residue: list[dict[str, tuple[float, float, float]]],
    *,
    plddt: tuple[float, ...],
    sequence: str = "KDF",
) -> None:
    lines = []
    serial = 1
    for residue_idx, (aa, atoms) in enumerate(zip(sequence, atoms_by_residue, strict=True), start=1):
        for atom_name, (x, y, z) in atoms.items():
            element = atom_name[0]
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {AA3[aa]:>3s} A{residue_idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{plddt[residue_idx - 1]:6.2f}          "
                f"{element:>2s}\n"
            )
            serial += 1
    lines.append("END\n")
    path.write_text("".join(lines))


def _write_cif(
    path: Path,
    atoms_by_residue: list[dict[str, tuple[float, float, float]]],
    *,
    plddt: tuple[float, ...],
) -> None:
    lines = [
        "data_metrics\n",
        "#\n",
        "loop_\n",
        "_atom_site.group_PDB\n",
        "_atom_site.id\n",
        "_atom_site.type_symbol\n",
        "_atom_site.label_atom_id\n",
        "_atom_site.label_alt_id\n",
        "_atom_site.label_comp_id\n",
        "_atom_site.label_asym_id\n",
        "_atom_site.label_entity_id\n",
        "_atom_site.label_seq_id\n",
        "_atom_site.pdbx_PDB_ins_code\n",
        "_atom_site.Cartn_x\n",
        "_atom_site.Cartn_y\n",
        "_atom_site.Cartn_z\n",
        "_atom_site.occupancy\n",
        "_atom_site.B_iso_or_equiv\n",
        "_atom_site.auth_seq_id\n",
        "_atom_site.auth_asym_id\n",
        "_atom_site.pdbx_PDB_model_num\n",
    ]
    serial = 1
    for residue_idx, (aa, atoms) in enumerate(zip("KDF", atoms_by_residue, strict=True), start=1):
        for atom_name, (x, y, z) in atoms.items():
            lines.append(
                f"ATOM {serial} {atom_name[0]} {atom_name} . {AA3[aa]} A 1 {residue_idx} ? "
                f"{x:.3f} {y:.3f} {z:.3f} 1.00 {plddt[residue_idx - 1]:.2f} "
                f"{residue_idx} A 1\n"
            )
            serial += 1
    lines.append("#\n")
    path.write_text("".join(lines))


def test_import_is_lightweight_and_does_not_pull_scientific_stacks():
    project_root = Path(__file__).resolve().parents[2]
    code = (
        "import json,sys; "
        "import inverse_folding.evaluation.structural_metrics_v2; "
        "print(json.dumps({k: k in sys.modules for k in ['numpy','pandas','torch','Bio']}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(proc.stdout) == {
        "numpy": False,
        "pandas": False,
        "torch": False,
        "Bio": False,
    }


def test_complete_metrics_use_global_ca_alignment_and_symmetry_aware_sidechains(tmp_path: Path):
    from inverse_folding.evaluation.structural_metrics_v2 import (
        ALL_METRICS,
        evaluate_prediction,
        prepare_reference_context,
    )

    ref = tmp_path / "ref.pdb"
    pred = tmp_path / "pred.pdb"
    _write_pdb(ref, _reference_atoms(), plddt=(95.0, 90.0, 85.0))
    _write_pdb(pred, _predicted_atoms(), plddt=(85.0, 75.0, 65.0))

    context = prepare_reference_context(ref, ref_sequence="KDF", anchor_indices={0, 1, 2})
    result = evaluate_prediction(
        context,
        pred,
        design_sequence="KDF",
        metrics=ALL_METRICS,
    )

    assert result.global_ca_rmsd == pytest.approx(0.0, abs=1e-6)
    assert result.predicted_global_plddt == pytest.approx(75.0)
    assert result.predicted_active_site_mean_plddt == pytest.approx(75.0)
    assert result.predicted_active_site_min_plddt == pytest.approx(65.0)
    assert result.reference_global_plddt == pytest.approx(90.0)
    assert result.active_site_sidechain_rmsd == pytest.approx(0.5, abs=1e-6)
    assert result.max_anchor_sidechain_rmsd == pytest.approx(math.sqrt(4.0 / 5.0), abs=1e-6)
    assert result.max_anchor_atom_distance == pytest.approx(2.0, abs=1e-6)
    assert result.matched_anchor_count == 3
    assert result.active_site_complete is True

    rows = {row.residue_idx: row for row in result.residue_metrics}
    assert rows[0].sidechain_atom_count == 5
    assert rows[0].sidechain_sq_error_sum == pytest.approx(4.0, abs=1e-6)
    assert rows[1].sidechain_rmsd == pytest.approx(0.0, abs=1e-6)
    assert rows[1].symmetry_swap_applied is True
    assert rows[2].match_status == "ok"


def test_glycine_anchor_is_a_valid_zero_atom_sidechain(tmp_path: Path):
    from inverse_folding.evaluation.structural_metrics_v2 import (
        ALL_METRICS,
        aggregate_residue_sidechain_metrics,
        evaluate_prediction,
        prepare_reference_context,
    )

    reference = _reference_atoms()
    reference[2] = {
        name: coord
        for name, coord in reference[2].items()
        if name in {"N", "CA", "C", "O"}
    }
    predicted = [
        {name: _transform(coord) for name, coord in atoms.items()}
        for atoms in reference
    ]
    ref = tmp_path / "ref_gly.pdb"
    pred = tmp_path / "pred_gly.pdb"
    _write_pdb(ref, reference, plddt=(95.0, 90.0, 85.0), sequence="KDG")
    _write_pdb(pred, predicted, plddt=(85.0, 75.0, 65.0), sequence="KDG")

    context = prepare_reference_context(ref, ref_sequence="KDG", anchor_indices={0, 1, 2})
    result = evaluate_prediction(
        context,
        pred,
        design_sequence="KDG",
        metrics=ALL_METRICS,
    )

    assert result.active_site_complete is True
    assert result.matched_anchor_count == 2
    assert result.active_site_sidechain_atom_count == 9
    assert result.active_site_sidechain_rmsd == pytest.approx(0.0, abs=1e-6)
    assert result.max_anchor_sidechain_rmsd == pytest.approx(0.0, abs=1e-6)
    assert result.residue_metrics[2].match_status == "no_sidechain"

    aggregate = aggregate_residue_sidechain_metrics(result.residue_metrics, {0, 1, 2})
    assert aggregate.residue_count == 3
    assert aggregate.sidechain_atom_count == 9
    assert aggregate.pooled_sidechain_rmsd == pytest.approx(0.0, abs=1e-6)


def test_selective_metrics_skip_alignment_and_sidechain_rows(tmp_path: Path):
    from inverse_folding.evaluation.structural_metrics_v2 import (
        METRIC_PLDDT,
        evaluate_prediction,
        prepare_reference_context,
    )

    ref = tmp_path / "ref.pdb"
    pred = tmp_path / "pred.pdb"
    _write_pdb(ref, _reference_atoms(), plddt=(95.0, 90.0, 85.0))
    _write_pdb(pred, _predicted_atoms(), plddt=(85.0, 75.0, 65.0))
    context = prepare_reference_context(ref, ref_sequence="KDF", anchor_indices=())

    result = evaluate_prediction(
        context,
        pred,
        design_sequence="KDF",
        metrics={METRIC_PLDDT},
    )

    assert result.predicted_global_plddt == pytest.approx(75.0)
    assert result.global_ca_rmsd is None
    assert result.active_site_sidechain_rmsd is None
    assert result.residue_metrics == ()


def test_anchor_identity_mismatch_fails_closed(tmp_path: Path):
    from inverse_folding.evaluation.structural_metrics_v2 import (
        ALL_METRICS,
        StructuralMetricError,
        evaluate_prediction,
        prepare_reference_context,
    )

    ref = tmp_path / "ref.pdb"
    pred = tmp_path / "pred.pdb"
    _write_pdb(ref, _reference_atoms(), plddt=(95.0, 90.0, 85.0))
    _write_pdb(pred, _predicted_atoms(), plddt=(85.0, 75.0, 65.0))
    context = prepare_reference_context(ref, ref_sequence="KDF", anchor_indices={0})

    with pytest.raises(StructuralMetricError, match="anchor identity mismatch"):
        evaluate_prediction(context, pred, design_sequence="FDF", metrics=ALL_METRICS)


def test_summary_row_is_standalone_for_benchmark_consumers(tmp_path: Path):
    from inverse_folding.evaluation.structural_metrics_v2 import (
        ALL_METRICS,
        build_benchmark_summary_row,
        evaluate_prediction,
        prepare_reference_context,
    )

    ref = tmp_path / "ref.pdb"
    pred = tmp_path / "pred.pdb"
    _write_pdb(ref, _reference_atoms(), plddt=(95.0, 90.0, 85.0))
    _write_pdb(pred, _predicted_atoms(), plddt=(85.0, 75.0, 65.0))
    result = evaluate_prediction(
        prepare_reference_context(ref, ref_sequence="KDF", anchor_indices={0, 1, 2}),
        pred,
        design_sequence="KDF",
        metrics=ALL_METRICS,
    )

    row = build_benchmark_summary_row(
        result,
        protein_id="p1",
        design_id="design_0000",
        design_idx=0,
        sequence="KDF",
        sc_tm=0.91,
        recovery=2.0 / 3.0,
        refold_backend="af3",
    )

    required = {
        "protein_id",
        "design_id",
        "design_idx",
        "sequence",
        "scTM",
        "global_ca_RMSD",
        "pLDDT",
        "reference_pLDDT",
        "active_site_sidechain_RMSD",
        "max_anchor_sidechain_RMSD",
        "max_anchor_atom_distance",
        "predicted_active_site_mean_pLDDT",
        "predicted_active_site_min_pLDDT",
        "reference_active_site_mean_pLDDT",
        "reference_active_site_min_pLDDT",
        "anchor_count",
        "matched_anchor_count",
        "active_site_complete",
        "recovery",
        "foldability",
        "refold_backend",
    }
    assert required <= set(row)
    assert row["global_ca_RMSD"] == pytest.approx(0.0, abs=1e-6)
    assert row["pLDDT"] == pytest.approx(75.0)
    assert row["scTM"] == pytest.approx(0.91)


def test_mmcif_reference_and_arbitrary_index_aggregation(tmp_path: Path):
    from inverse_folding.evaluation.structural_metrics_v2 import (
        ALL_METRICS,
        aggregate_residue_sidechain_metrics,
        build_benchmark_residue_rows,
        evaluate_prediction,
        prepare_reference_context,
    )

    ref = tmp_path / "ref.cif"
    pred = tmp_path / "pred.pdb"
    _write_cif(ref, _reference_atoms(), plddt=(95.0, 90.0, 85.0))
    _write_pdb(pred, _predicted_atoms(), plddt=(85.0, 75.0, 65.0))
    result = evaluate_prediction(
        prepare_reference_context(ref, ref_sequence="KDF", anchor_indices={0, 1, 2}),
        pred,
        design_sequence="KDF",
        metrics=ALL_METRICS,
    )

    subset = aggregate_residue_sidechain_metrics(result.residue_metrics, {0, 1})
    assert subset.residue_count == 2
    assert subset.sidechain_atom_count == 9
    assert subset.pooled_sidechain_rmsd == pytest.approx(math.sqrt(4.0 / 9.0), abs=1e-6)
    assert subset.max_residue_sidechain_rmsd == pytest.approx(math.sqrt(4.0 / 5.0), abs=1e-6)
    assert subset.max_atom_distance == pytest.approx(2.0, abs=1e-6)

    serialized = build_benchmark_residue_rows(
        result,
        protein_id="p1",
        design_id="design_0000",
        design_idx=0,
        refold_backend="esmfold",
    )
    serialized_subset = aggregate_residue_sidechain_metrics(serialized, {0, 1})
    assert serialized_subset == subset
