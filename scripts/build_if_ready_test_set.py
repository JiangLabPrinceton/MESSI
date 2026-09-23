#!/usr/bin/env python
"""Build an inverse-folding-ready test set from resolved backbone residues.

The assembled IF test-set parquet stores biological/full sequences. Structure-
conditioned inverse folding needs one residue per available backbone coordinate,
so this script creates a derived parquet whose ``sequence`` column is the
canonicalized sequence of resolved residues in the downloaded structure.

The original parquet and raw structures are never modified. Cleaned single-chain
PDB files are written alongside the derived parquet and should be used as the
Phase C ``PDB_ROOT``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from Bio.Data import PDBData
from Bio.PDB import MMCIFIO, MMCIFParser, PDBIO, PDBParser, Select


CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
AA3_TO1_EXT = {k.upper(): v for k, v in PDBData.protein_letters_3to1_extended.items()}
AA1_TO3 = {k: v.upper() for k, v in PDBData.protein_letters_1to3.items()}
REQUIRED_BACKBONE_ATOMS = ("N", "CA", "C", "O")
PDB_CHAIN_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}_[A-Za-z0-9]{1,4}$")


@dataclass(frozen=True)
class ResolvedResidue:
    residue_id: tuple[str, int, str]
    author_resnum: int
    insertion_code: str
    original_resname: str
    canonical_resname: str
    aa: str


class ChainResidueSelect(Select):
    """PDBIO selector for one model, one chain, and selected residues."""

    def __init__(
        self,
        *,
        model_idx: int,
        chain_id: str,
        residue_ids: set[tuple[str, int, str]],
        atom_full_ids: set[tuple],
    ) -> None:
        self.model_idx = model_idx
        self.chain_id = chain_id
        self.residue_ids = residue_ids
        self.atom_full_ids = atom_full_ids

    def accept_model(self, model: Any) -> bool:
        return int(model.id) == self.model_idx

    def accept_chain(self, chain: Any) -> bool:
        return str(chain.id) == self.chain_id

    def accept_residue(self, residue: Any) -> bool:
        return residue.id in self.residue_ids

    def accept_atom(self, atom: Any) -> bool:
        return atom.get_full_id() in self.atom_full_ids


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an IF-ready test-set parquet from resolved PDB residues.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--pdb-root", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--output-structure-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--failure-csv", required=True)
    parser.add_argument(
        "--min-if-length",
        type=int,
        default=50,
        help="Minimum resolved sequence length retained in the IF-ready parquet.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.8,
        help="Minimum resolved/full sequence length ratio retained. Default 0.8 drops "
             "heavily-truncated structure fragments (deviate from whole-protein redesign, "
             "collapse structurally). Pass 0.0 to keep all resolved rows (e.g. full-length "
             "predicted backbones).",
    )
    parser.add_argument(
        "--model-idx",
        type=int,
        default=0,
        help="Zero-based model index to extract from multi-model PDB files.",
    )
    parser.add_argument(
        "--allow-cif",
        action="store_true",
        help="Allow mmCIF input structures and write cleaned mmCIF outputs.",
    )
    parser.add_argument(
        "--default-chain",
        default="A",
        help="Chain ID to use when protein_id is not in PDB_CHAIN format.",
    )
    return parser.parse_args(argv)


def safe_file_id(protein_id: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in protein_id)


def infer_chain_id(protein_id: str, default_chain: str | None = None) -> str:
    if not PDB_CHAIN_ID_RE.match(protein_id):
        if default_chain:
            return default_chain
        raise ValueError("protein_id is not a PDB-chain id")
    return protein_id.split("_", 1)[1]


def resolve_structure_path(row: dict[str, Any], pdb_root: Path) -> Path:
    protein_id = str(row["protein_id"])
    safe_id = safe_file_id(protein_id)
    pdb_path = str(row.get("pdb_path") or "")
    candidates: list[Path] = []

    if pdb_path:
        raw_path = Path(pdb_path)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend([
                pdb_root / raw_path,
                pdb_root / raw_path.name,
                pdb_root / f"{raw_path.stem}.pdb",
                pdb_root / f"{raw_path.stem}.cif",
            ])

    candidates.extend([
        pdb_root / f"{protein_id}.pdb",
        pdb_root / f"{protein_id}.cif",
        pdb_root / f"{safe_id}.pdb",
        pdb_root / f"{safe_id}.cif",
    ])

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not resolve structure under {pdb_root}")


def canonicalize_resname(resname: str) -> tuple[str, str]:
    code = resname.strip().upper()
    aa = AA3_TO1_EXT.get(code)
    if aa is None:
        raise ValueError(f"unknown_residue:{code}")
    if aa not in CANONICAL_AA:
        raise ValueError(f"non_canonical_residue:{code}->{aa}")
    return aa, AA1_TO3[aa]


def has_required_backbone(residue: Any) -> bool:
    atom_names = {atom.get_name().strip() for atom in residue.get_atoms()}
    return all(atom_name in atom_names for atom_name in REQUIRED_BACKBONE_ATOMS)


def extract_resolved_residues(
    pdb_path: Path,
    *,
    chain_id: str,
    model_idx: int,
) -> tuple[Any, list[ResolvedResidue], Counter[str]]:
    suffix = pdb_path.suffix.lower()
    if suffix == ".cif":
        parser = MMCIFParser(QUIET=True)
    elif suffix == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"unsupported_format:{suffix}")
    structure = parser.get_structure("if_ready", str(pdb_path))
    models = list(structure.get_models())
    if model_idx >= len(models):
        raise ValueError(f"model_idx={model_idx} out of range; n_models={len(models)}")
    model = models[model_idx]
    chain = next((ch for ch in model.get_chains() if ch.id == chain_id), None)
    if chain is None:
        raise ValueError(f"chain_not_found:{chain_id}")

    stats: Counter[str] = Counter()
    candidates: list[ResolvedResidue] = []
    for residue in chain.get_residues():
        original_resname = residue.get_resname().strip().upper()
        try:
            aa, canonical_resname = canonicalize_resname(original_resname)
        except ValueError:
            if has_required_backbone(residue):
                raise
            stats["skipped_non_amino_acid"] += 1
            continue

        if not has_required_backbone(residue):
            stats["skipped_missing_backbone"] += 1
            continue

        hetflag, author_resnum, icode = residue.id
        candidates.append(
            ResolvedResidue(
                residue_id=residue.id,
                author_resnum=int(author_resnum),
                insertion_code=str(icode).strip(),
                original_resname=original_resname,
                canonical_resname=canonical_resname,
                aa=aa,
            )
        )
        if original_resname != canonical_resname:
            stats[f"modified:{original_resname}->{canonical_resname}"] += 1
        if hetflag != " ":
            stats["het_amino_acid_residue"] += 1

    # Bio.PDB can represent a standard residue and a HET modified residue with
    # the same author residue number as two Residue objects. biotite/DPLM groups
    # by residue id and will collapse these, so deduplicate here with a stable
    # preference for the standard ATOM residue.
    resolved_by_author: dict[tuple[int, str], ResolvedResidue] = {}
    for candidate in candidates:
        key = (candidate.author_resnum, candidate.insertion_code)
        existing = resolved_by_author.get(key)
        if existing is None:
            resolved_by_author[key] = candidate
            continue
        stats["duplicate_author_residue_id"] += 1
        existing_is_standard = existing.residue_id[0] == " "
        candidate_is_standard = candidate.residue_id[0] == " "
        if candidate_is_standard and not existing_is_standard:
            resolved_by_author[key] = candidate

    resolved = list(resolved_by_author.values())
    if not resolved:
        raise ValueError("no_resolved_backbone_residues")
    return structure, resolved, stats


def write_clean_structure(
    *,
    structure: Any,
    clean_path: Path,
    model_idx: int,
    chain_id: str,
    resolved: list[ResolvedResidue],
) -> None:
    residue_ids = {r.residue_id for r in resolved}
    canonical_by_id = {r.residue_id: r.canonical_resname for r in resolved}
    atom_full_ids: set[tuple] = set()
    for model in structure.get_models():
        if int(model.id) != model_idx:
            continue
        for chain in model.get_chains():
            if str(chain.id) != chain_id:
                continue
            for residue in chain.get_residues():
                if residue.id in canonical_by_id:
                    residue.resname = canonical_by_id[residue.id]
                    for atom in select_single_altloc_atoms(residue):
                        atom.set_altloc(" ")
                        atom_full_ids.add(atom.get_full_id())

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    if clean_path.suffix.lower() == ".cif":
        io = MMCIFIO()
    else:
        io = PDBIO()
    io.set_structure(structure)
    io.save(
        str(clean_path),
        select=ChainResidueSelect(
            model_idx=model_idx,
            chain_id=chain_id,
            residue_ids=residue_ids,
            atom_full_ids=atom_full_ids,
        ),
    )


def select_single_altloc_atoms(residue: Any) -> list[Any]:
    """Select at most one conformer per atom name, preferring highest occupancy."""
    selected: dict[str, Any] = {}
    for atom in residue.get_atoms():
        atom_name = atom.get_name().strip()
        current = selected.get(atom_name)
        if current is None:
            selected[atom_name] = atom
            continue
        atom_occ = atom.get_occupancy() or 0.0
        current_occ = current.get_occupancy() or 0.0
        atom_altloc = atom.get_altloc() or " "
        current_altloc = current.get_altloc() or " "
        if (atom_occ, atom_altloc == " ") > (current_occ, current_altloc == " "):
            selected[atom_name] = atom
    return list(selected.values())


def build_sequence_mapping(original: str, resolved: str) -> list[int | None]:
    """Needleman-Wunsch alignment mapping resolved positions to original indices."""
    n = len(original)
    m = len(resolved)
    match_score = 2
    mismatch_score = -1
    gap_score = -2
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i * gap_score
        trace[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = j * gap_score
        trace[0][j] = "left"

    for i in range(1, n + 1):
        oi = original[i - 1]
        for j in range(1, m + 1):
            rj = resolved[j - 1]
            diag = dp[i - 1][j - 1] + (match_score if oi == rj else mismatch_score)
            up = dp[i - 1][j] + gap_score
            left = dp[i][j - 1] + gap_score
            best = max(diag, up, left)
            dp[i][j] = best
            if best == diag:
                trace[i][j] = "diag"
            elif best == up:
                trace[i][j] = "up"
            else:
                trace[i][j] = "left"

    mapping_reversed: list[int | None] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = trace[i][j]
        if step == "diag":
            mapping_reversed.append(i - 1)
            i -= 1
            j -= 1
        elif step == "left":
            mapping_reversed.append(None)
            j -= 1
        else:
            i -= 1
    return list(reversed(mapping_reversed))


def sequence_sha1(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("ascii")).hexdigest()


def process_row(
    row: dict[str, Any],
    *,
    pdb_root: Path,
    output_structure_dir: Path,
    model_idx: int,
    min_if_length: int,
    min_coverage: float,
    allow_cif: bool,
    default_chain: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    protein_id = str(row["protein_id"])
    original_sequence = str(row["sequence"]).upper()
    original_length = int(row["sequence_length"])
    try:
        chain_id = str(row.get("if_chain_id") or row.get("chain") or "").strip()
        if not chain_id:
            chain_id = infer_chain_id(protein_id, default_chain=default_chain)
        structure_path = resolve_structure_path(row, pdb_root)
        if structure_path.suffix.lower() == ".cif" and not allow_cif:
            raise ValueError("unsupported_format:cif")
        if structure_path.suffix.lower() not in {".pdb", ".cif"}:
            raise ValueError(f"unsupported_format:{structure_path.suffix.lower()}")

        structure, resolved, residue_stats = extract_resolved_residues(
            structure_path,
            chain_id=chain_id,
            model_idx=model_idx,
        )
        if_sequence = "".join(r.aa for r in resolved)
        if_length = len(if_sequence)
        coverage = if_length / float(original_length) if original_length > 0 else 0.0
        if if_length < min_if_length:
            raise ValueError(f"if_sequence_too_short:{if_length}<min_if_length={min_if_length}")
        if coverage < min_coverage:
            raise ValueError(f"coverage_too_low:{coverage:.4f}<min_coverage={min_coverage:.4f}")
        if set(if_sequence) - CANONICAL_AA:
            raise ValueError("if_sequence_non_canonical")

        safe_id = safe_file_id(protein_id)
        clean_suffix = ".cif" if structure_path.suffix.lower() == ".cif" else ".pdb"
        clean_path = output_structure_dir / f"{safe_id}{clean_suffix}"
        write_clean_structure(
            structure=structure,
            clean_path=clean_path,
            model_idx=model_idx,
            chain_id=chain_id,
            resolved=resolved,
        )

        out = dict(row)
        out["original_sequence"] = original_sequence
        out["original_sequence_length"] = original_length
        out["original_pdb_path"] = row.get("pdb_path")
        out["original_sequence_sha1"] = sequence_sha1(original_sequence)
        out["sequence"] = if_sequence
        out["sequence_length"] = if_length
        out["pdb_path"] = clean_path.name
        out["if_ready"] = True
        out["if_chain_id"] = chain_id
        out["if_source_structure"] = str(structure_path)
        out["if_sequence_sha1"] = sequence_sha1(if_sequence)
        out["if_sequence_coverage"] = coverage
        out["if_length_delta"] = if_length - original_length
        out["if_sequence_in_original"] = if_sequence in original_sequence
        out["if_sequence_offset"] = (
            original_sequence.index(if_sequence)
            if if_sequence in original_sequence
            else None
        )
        out["if_to_original_seq_idx_json"] = json.dumps(
            build_sequence_mapping(original_sequence, if_sequence),
            separators=(",", ":"),
        )
        out["if_author_resnums_json"] = json.dumps(
            [r.author_resnum for r in resolved],
            separators=(",", ":"),
        )
        out["if_insertion_codes_json"] = json.dumps(
            [r.insertion_code for r in resolved],
            separators=(",", ":"),
        )
        out["if_original_resnames_json"] = json.dumps(
            [r.original_resname for r in resolved],
            separators=(",", ":"),
        )
        out["if_modified_residue_counts_json"] = json.dumps(
            dict(sorted(residue_stats.items())),
            sort_keys=True,
            separators=(",", ":"),
        )
        return out, None
    except Exception as exc:  # noqa: BLE001 - persisted for data audit.
        return None, {
            "protein_id": protein_id,
            "tier": row.get("tier"),
            "reason": str(exc).split(":", 1)[0],
            "detail": f"{type(exc).__name__}: {exc}",
            "sequence_length": original_length,
            "pdb_path": row.get("pdb_path"),
        }


def summarize(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    by_tier: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_tier[str(row.get("tier", "NA"))]["ready"] += 1
    for failure in failures:
        tier = str(failure.get("tier", "NA"))
        by_tier[tier]["failed"] += 1

    reason_counts = Counter(str(f["reason"]) for f in failures)
    deltas = [int(r["if_length_delta"]) for r in rows]
    coverages = [float(r["if_sequence_coverage"]) for r in rows]
    return {
        "n_ready": len(rows),
        "n_failed": len(failures),
        "by_tier": {tier: dict(counts) for tier, counts in sorted(by_tier.items())},
        "failure_reasons": dict(reason_counts.most_common()),
        "if_length_delta": {
            "min": min(deltas) if deltas else None,
            "max": max(deltas) if deltas else None,
            "median": float(pd.Series(deltas).median()) if deltas else None,
        },
        "if_sequence_coverage": {
            "min": min(coverages) if coverages else None,
            "median": float(pd.Series(coverages).median()) if coverages else None,
            "max": max(coverages) if coverages else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_if_length <= 0:
        print("ERROR: --min-if-length must be positive.")
        return 2
    if not 0.0 <= args.min_coverage <= 1.0:
        print("ERROR: --min-coverage must be in [0, 1].")
        return 2

    input_parquet = Path(args.input_parquet)
    pdb_root = Path(args.pdb_root)
    output_parquet = Path(args.output_parquet)
    output_structure_dir = Path(args.output_structure_dir)
    manifest_json = Path(args.manifest_json)
    failure_csv = Path(args.failure_csv)

    df = pd.read_parquet(input_parquet)
    required = {"protein_id", "sequence", "sequence_length", "pdb_path"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: input parquet missing required columns: {sorted(missing)}")
        return 1
    if df["protein_id"].duplicated().any():
        print("ERROR: input parquet contains duplicate protein_id rows")
        return 1

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        ready_row, failure = process_row(
            row.to_dict(),
            pdb_root=pdb_root,
            output_structure_dir=output_structure_dir,
            model_idx=args.model_idx,
            min_if_length=args.min_if_length,
            min_coverage=args.min_coverage,
            allow_cif=args.allow_cif,
            default_chain=args.default_chain,
        )
        if ready_row is not None:
            rows.append(ready_row)
        if failure is not None:
            failures.append(failure)
        if idx % 250 == 0 or idx == len(df):
            print(
                f"[{idx}/{len(df)}] ready={len(rows)} failed={len(failures)} "
                f"last={row['protein_id']}"
            )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    failure_csv.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(output_parquet, index=False)
    pd.DataFrame(failures).to_csv(failure_csv, index=False)

    manifest = {
        "input_parquet": str(input_parquet.resolve()),
        "pdb_root": str(pdb_root.resolve()),
        "output_parquet": str(output_parquet.resolve()),
        "output_structure_dir": str(output_structure_dir.resolve()),
        "failure_csv": str(failure_csv.resolve()),
        "n_input": int(len(df)),
        "min_if_length": int(args.min_if_length),
        "min_coverage": float(args.min_coverage),
        "model_idx": int(args.model_idx),
        **summarize(rows, failures),
    }
    with open(manifest_json, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print("=" * 60)
    print("IF-ready test set build complete")
    print(f"  input rows          : {len(df)}")
    print(f"  ready rows          : {len(rows)}")
    print(f"  failed rows         : {len(failures)}")
    print(f"  output parquet      : {output_parquet}")
    print(f"  clean structure dir : {output_structure_dir}")
    print(f"  manifest            : {manifest_json}")
    print(f"  failures            : {failure_csv}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
