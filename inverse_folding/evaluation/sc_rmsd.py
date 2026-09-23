"""C-alpha self-consistency RMSD helpers for structural evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import shlex
from typing import Any, Iterable

import numpy as np


_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass(frozen=True)
class CaResidue:
    """One residue-level C-alpha coordinate in trace order."""

    residue_idx: int
    residue_name: str
    aa: str
    chain_id: str
    resseq: str
    icode: str
    coord: np.ndarray


def compute_ca_self_consistency(
    *,
    pred_pdb: str | Path,
    ref_pdb: str | Path,
    protein_id: str,
    design_id: str,
    design_idx: int,
    design_sequence: str,
    ref_sequence: str,
    refold_backend: str,
) -> tuple[float, list[dict[str, Any]]]:
    """Return DPLM-style aggregate scRMSD and per-residue C-alpha distances.

    The predicted C-alpha trace is rigidly superimposed onto the reference trace
    once over all matched residues. Per-residue distances are then measured in
    that shared aligned frame, so an arbitrary residue-index subset can be
    summarized later as ``sqrt(mean(sc_ca_distance ** 2))`` without refolding.
    """

    pred_trace = parse_ca_trace(pred_pdb)
    ref_trace = parse_ca_trace(ref_pdb)
    n = len(ref_trace)
    if n == 0:
        raise ValueError(f"{ref_pdb}: no C-alpha residues found")
    if len(pred_trace) != n:
        raise ValueError(
            f"C-alpha trace length mismatch for {protein_id} {design_id}: "
            f"pred={len(pred_trace)} ref={n}"
        )
    if len(design_sequence) != n or len(ref_sequence) != n:
        raise ValueError(
            f"sequence/C-alpha length mismatch for {protein_id} {design_id}: "
            f"design_seq={len(design_sequence)} ref_seq={len(ref_sequence)} ca={n}"
        )

    ref_coords = np.vstack([res.coord for res in ref_trace]).astype(float)
    pred_coords = np.vstack([res.coord for res in pred_trace]).astype(float)
    aligned_pred = kabsch_align(mobile=pred_coords, target=ref_coords)
    distances = np.linalg.norm(aligned_pred - ref_coords, axis=1)
    sc_rmsd = float(math.sqrt(float(np.mean(np.square(distances)))))

    rows: list[dict[str, Any]] = []
    for idx, (ref_res, pred_coord, aligned_coord, distance) in enumerate(
        zip(ref_trace, pred_coords, aligned_pred, distances, strict=True)
    ):
        rows.append(
            {
                "protein_id": str(protein_id),
                "design_id": str(design_id),
                "design_idx": int(design_idx),
                "residue_idx": int(idx),
                "residue_idx_1based": int(idx + 1),
                "ref_chain_id": ref_res.chain_id,
                "ref_resseq": ref_res.resseq,
                "ref_icode": ref_res.icode,
                "ref_aa": ref_sequence[idx],
                "design_aa": design_sequence[idx],
                "sc_ca_distance": float(distance),
                "ref_ca_x": float(ref_coords[idx, 0]),
                "ref_ca_y": float(ref_coords[idx, 1]),
                "ref_ca_z": float(ref_coords[idx, 2]),
                "pred_ca_x": float(pred_coord[0]),
                "pred_ca_y": float(pred_coord[1]),
                "pred_ca_z": float(pred_coord[2]),
                "aligned_pred_ca_x": float(aligned_coord[0]),
                "aligned_pred_ca_y": float(aligned_coord[1]),
                "aligned_pred_ca_z": float(aligned_coord[2]),
                "refold_backend": str(refold_backend),
            }
        )
    return sc_rmsd, rows


def kabsch_align(*, mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rigidly align ``mobile`` coordinates onto ``target`` coordinates."""

    mobile_arr = np.asarray(mobile, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    if mobile_arr.shape != target_arr.shape:
        raise ValueError(f"coordinate shape mismatch: {mobile_arr.shape} != {target_arr.shape}")
    if mobile_arr.ndim != 2 or mobile_arr.shape[1] != 3:
        raise ValueError(f"expected coordinates with shape (N, 3), got {mobile_arr.shape}")
    if mobile_arr.shape[0] == 0:
        raise ValueError("cannot align empty coordinate arrays")

    mobile_center = mobile_arr.mean(axis=0)
    target_center = target_arr.mean(axis=0)
    mobile_zero = mobile_arr - mobile_center
    target_zero = target_arr - target_center
    cov = mobile_zero.T @ target_zero
    u_mat, _singular_values, vt_mat = np.linalg.svd(cov)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u_mat @ vt_mat)) or 1.0
    rotation = u_mat @ correction @ vt_mat
    return mobile_zero @ rotation + target_center


def parse_ca_trace(path: str | Path) -> list[CaResidue]:
    """Parse a PDB or mmCIF file into trace-ordered C-alpha residues."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return _parse_mmcif_ca_trace(path)
    return _parse_pdb_ca_trace(path)


def _parse_pdb_ca_trace(path: Path) -> list[CaResidue]:
    residues: list[CaResidue] = []
    seen: set[tuple[str, str, str]] = set()
    with open(path) as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if len(line) < 54:
                continue
            atom_name = line[12:16].strip()
            altloc = line[16].strip()
            if atom_name != "CA" or altloc not in {"", "A"}:
                continue
            resname = line[17:20].strip().upper()
            chain_id = line[21].strip() or "?"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain_id, resseq, icode)
            if key in seen:
                continue
            try:
                coord = np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=float,
                )
            except ValueError:
                continue
            seen.add(key)
            residues.append(
                CaResidue(
                    residue_idx=len(residues),
                    residue_name=resname,
                    aa=_THREE_TO_ONE.get(resname, "X"),
                    chain_id=chain_id,
                    resseq=resseq,
                    icode=icode,
                    coord=coord,
                )
            )
    return residues


def _parse_mmcif_ca_trace(path: Path) -> list[CaResidue]:
    tokens = list(_iter_cif_tokens(path))
    idx = 0
    while idx < len(tokens):
        if tokens[idx] != "loop_":
            idx += 1
            continue
        idx += 1
        headers: list[str] = []
        while idx < len(tokens) and tokens[idx].startswith("_"):
            headers.append(tokens[idx])
            idx += 1
        if not headers:
            continue
        n_headers = len(headers)
        if not any(header.startswith("_atom_site.") for header in headers):
            while idx < len(tokens) and tokens[idx] != "loop_" and not tokens[idx].startswith("_"):
                idx += n_headers
            continue

        residues: list[CaResidue] = []
        seen: set[tuple[str, str, str]] = set()
        while idx + n_headers <= len(tokens):
            if tokens[idx] == "loop_" or tokens[idx].startswith("_"):
                break
            record = dict(zip(headers, tokens[idx: idx + n_headers], strict=True))
            idx += n_headers
            atom_name = _clean_cif_value(
                record.get("_atom_site.label_atom_id")
                or record.get("_atom_site.auth_atom_id")
                or ""
            )
            altloc = _clean_cif_value(record.get("_atom_site.label_alt_id") or "")
            if atom_name != "CA" or altloc not in {"", ".", "?", "A"}:
                continue
            resname = _clean_cif_value(
                record.get("_atom_site.label_comp_id")
                or record.get("_atom_site.auth_comp_id")
                or ""
            ).upper()
            chain_id = _clean_cif_value(
                record.get("_atom_site.auth_asym_id")
                or record.get("_atom_site.label_asym_id")
                or "?"
            )
            resseq = _clean_cif_value(
                record.get("_atom_site.auth_seq_id")
                or record.get("_atom_site.label_seq_id")
                or ""
            )
            icode = _clean_cif_value(record.get("_atom_site.pdbx_PDB_ins_code") or "")
            if icode in {".", "?"}:
                icode = ""
            key = (chain_id, resseq, icode)
            if key in seen:
                continue
            try:
                coord = np.array(
                    [
                        float(_clean_cif_value(record["_atom_site.Cartn_x"])),
                        float(_clean_cif_value(record["_atom_site.Cartn_y"])),
                        float(_clean_cif_value(record["_atom_site.Cartn_z"])),
                    ],
                    dtype=float,
                )
            except (KeyError, ValueError):
                continue
            seen.add(key)
            residues.append(
                CaResidue(
                    residue_idx=len(residues),
                    residue_name=resname,
                    aa=_THREE_TO_ONE.get(resname, "X"),
                    chain_id=chain_id or "?",
                    resseq=resseq,
                    icode=icode,
                    coord=coord,
                )
            )
        return residues
    return []


def _iter_cif_tokens(path: Path) -> Iterable[str]:
    in_text_block = False
    with open(path) as handle:
        for raw_line in handle:
            if raw_line.startswith(";"):
                in_text_block = not in_text_block
                continue
            if in_text_block:
                continue
            line = raw_line.strip()
            if not line or line == "#":
                continue
            yield from shlex.split(line, comments=True, posix=True)


def _clean_cif_value(value: str) -> str:
    value = str(value).strip()
    if value in {".", "?"}:
        return value
    return value.strip("'\"")
