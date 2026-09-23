#!/usr/bin/env python
"""ProteinMPNN inverse-folding baseline with optional NetMHCIIpan post-hoc filter.

Pipeline:
  1. Optionally stage IF-ready structures into continuous-residue-numbered
     PDBs aligned to a test-set parquet.
  2. Parse a folder of input PDBs into a ProteinMPNN jsonl.
  3. Run the vendored ProteinMPNN (`DRAKES/drakes_protein/ProteinMPNN/`) to
     generate ``--num-seq-per-target`` designs per structure.
  4. Optionally score each design with the project's NetMHCIIpan
     ``StandaloneRunner`` and select an argmin-risk design per protein.
  5. Write ``generated.parquet`` / ``generated.fasta`` with the columns used by
     ``scripts/run_if_phase_c0.py`` so downstream evaluation (Phase B4) is
     plug-compatible. When the NMP filter is enabled, also emit
     ``selection.parquet`` (one selected design per protein) and
     ``selection.json`` (per-design risk metrics).

NMP filter convention:
  * Multi-chain designs (joined by ``/`` in MPNN output) are scored chain by
    chain; the aggregate risk is the sum across chains.
  * Risk metrics (controlled by ``--nmp-rule``):
      ``min_n_sb``      — count of windows with ``%Rank_EL <= --nmp-rank-threshold``
                          (lower = better; tie-break by ``min_mean_rank``)
      ``min_mean_rank`` — mean ``%Rank_EL`` across all scored windows (lower = better)

When ``--test-set-parquet`` is provided, ``--input-pdb-folder`` is treated as
the canonical IF-ready PDB root. Structures are resolved from the parquet,
rewritten into a private staging folder with continuous residue numbering, and
validated against the parquet sequence before ProteinMPNN sees them. This keeps
ProteinMPNN's official parser/model unchanged while avoiding author-numbering
gaps being converted into ``X`` tokens.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from Bio.Data import PDBData
from Bio.PDB import MMCIFParser, PDBParser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTEINMPNN_ROOT = PROJECT_ROOT / "DRAKES" / "drakes_protein" / "ProteinMPNN"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
AA3_TO1_EXT = {k.upper(): v for k, v in PDBData.protein_letters_3to1_extended.items()}
AA1_TO3 = {k: v.upper() for k, v in PDBData.protein_letters_1to3.items()}
REQUIRED_BACKBONE_ATOMS = ("N", "CA", "C", "O")
PDB_CHAIN_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}_[A-Za-z0-9]{1,4}$")


@dataclass(frozen=True)
class StagedInput:
    protein_id: str
    stage_id: str
    source_path: str
    staged_path: str
    original_chain_id: str
    staged_chain_id: str
    sequence_length: int


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ProteinMPNN inverse-folding baseline with optional NMP filter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--input-pdb-folder", required=True, type=Path,
                   help="Directory containing input PDB/CIF files, or the IF-ready PDB root "
                        "when --test-set-parquet is set.")
    p.add_argument("--test-set-parquet", type=Path, default=None,
                   help="Optional IF-ready test-set parquet. When set, only these proteins "
                        "are staged for ProteinMPNN, with continuous residue numbering.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory to write generated.parquet / fasta / configs.")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow writing into a non-empty output directory.")

    # ProteinMPNN
    p.add_argument("--mpnn-model-name", default="v_48_020",
                   choices=["v_48_002", "v_48_010", "v_48_020", "v_48_030"])
    p.add_argument("--num-seq-per-target", type=int, default=8)
    p.add_argument("--sampling-temp", default="0.1",
                   help="Space-separated list of temperatures (passed verbatim to MPNN).")
    p.add_argument("--mpnn-batch-size", type=int, default=1)
    p.add_argument("--mpnn-seed", type=int, default=42)
    p.add_argument("--mpnn-device", default="0",
                   help="CUDA device index for ProteinMPNN (e.g. '0'). Use 'cpu' for CPU.")
    p.add_argument("--ca-only", action="store_true",
                   help="Use CA-only ProteinMPNN model weights.")
    p.add_argument("--design-chains", default="",
                   help="Space-separated chain IDs to design "
                        "(default: design all chains in each PDB).")
    p.add_argument("--keep-mpnn-raw", action="store_true",
                   help="Keep ProteinMPNN raw seqs/ folder (default: kept under <output>/_mpnn_raw).")

    # NMP filter (optional)
    p.add_argument("--apply-nmp-filter", action="store_true",
                   help="Score each design with NetMHCIIpan and pick argmin-risk per protein.")
    p.add_argument("--netmhciipan-bin", type=Path, default=None,
                   help="Path to NetMHCIIpan binary (required when --apply-nmp-filter).")
    p.add_argument("--allele", default=None,
                   help="HLA-DRB1*XX:YY allele (required when --apply-nmp-filter).")
    p.add_argument("--nmp-pep-lengths", default="15",
                   help="Space-separated peptide lengths for NMP scanning.")
    p.add_argument("--nmp-rank-threshold", type=float, default=2.0,
                   help="%%Rank_EL strong-binder threshold in percent (default: 2.0%%).")
    p.add_argument("--nmp-rule", default="min_n_sb",
                   choices=["min_n_sb", "min_mean_rank"],
                   help="Selection rule among MPNN candidates per protein.")
    p.add_argument("--nmp-batch-size", type=int, default=30)
    p.add_argument("--nmp-max-lengths-per-call", type=int, default=4)
    p.add_argument("--nmp-workers", type=int, default=1)
    p.add_argument("--nmp-timeout", type=int, default=600)

    args = p.parse_args(argv)

    # Validation
    if args.num_seq_per_target <= 0:
        p.error("--num-seq-per-target must be positive")
    if args.mpnn_batch_size <= 0:
        p.error("--mpnn-batch-size must be positive")
    if args.test_set_parquet is not None and not args.test_set_parquet.exists():
        p.error(f"--test-set-parquet not found: {args.test_set_parquet}")
    if args.test_set_parquet is not None and args.design_chains.strip() not in {"", "A"}:
        p.error("--test-set-parquet stages single-chain inputs as chain A; use --design-chains A or omit it")
    if args.apply_nmp_filter:
        if args.netmhciipan_bin is None:
            p.error("--netmhciipan-bin is required when --apply-nmp-filter is set")
        if args.allele is None:
            p.error("--allele is required when --apply-nmp-filter is set")
        if not args.netmhciipan_bin.exists():
            p.error(f"NetMHCIIpan binary not found: {args.netmhciipan_bin}")
        if args.nmp_rank_threshold <= 0:
            p.error("--nmp-rank-threshold must be positive")
    return args


# ─────────────────────────────────────────────────────────────────────────────
# IF-ready staging for ProteinMPNN
# ─────────────────────────────────────────────────────────────────────────────

def safe_file_id(protein_id: str) -> str:
    """Return a filesystem-safe id matching the IF-ready data convention."""
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in protein_id)


def infer_chain_id(protein_id: str, default_chain: str = "A") -> str:
    if not PDB_CHAIN_ID_RE.match(protein_id):
        return default_chain
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
            candidates.append(pdb_root / raw_path)
            candidates.append(pdb_root / raw_path.name)
            candidates.append(pdb_root / f"{raw_path.stem}.pdb")
            candidates.append(pdb_root / f"{raw_path.stem}.cif")

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
    raise FileNotFoundError(
        f"{protein_id}: could not resolve structure under {pdb_root}"
    )


def _load_structure(path: Path) -> Any:
    if path.suffix.lower() == ".cif":
        parser = MMCIFParser(QUIET=True)
    elif path.suffix.lower() == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"unsupported structure suffix for ProteinMPNN staging: {path}")
    return parser.get_structure(path.stem, str(path))


def _select_chain(structure: Any, chain_id: str) -> Any:
    model = next(structure.get_models())
    chains = list(model.get_chains())
    for chain in chains:
        if str(chain.id) == chain_id:
            return chain
    if len(chains) == 1:
        return chains[0]
    raise ValueError(
        f"chain {chain_id!r} not found; available chains={[str(c.id) for c in chains]}"
    )


def _residue_to_aa(residue: Any) -> str:
    resname = residue.get_resname().strip().upper()
    aa = AA3_TO1_EXT.get(resname)
    if aa is None:
        raise ValueError(f"unknown residue {resname}")
    if aa not in CANONICAL_AA:
        raise ValueError(f"non-canonical residue {resname}->{aa}")
    return aa


def _selected_atoms_by_name(residue: Any) -> dict[str, Any]:
    """Pick at most one atom per atom name, preferring highest occupancy."""
    selected: dict[str, Any] = {}
    for atom in residue.get_atoms():
        name = atom.get_name().strip()
        current = selected.get(name)
        if current is None:
            selected[name] = atom
            continue
        atom_occ = atom.get_occupancy() or 0.0
        current_occ = current.get_occupancy() or 0.0
        if atom_occ > current_occ:
            selected[name] = atom
    return selected


def _atom_fullname(atom: Any) -> str:
    fullname = atom.get_fullname()
    if isinstance(fullname, str) and len(fullname) == 4:
        return fullname
    return f"{atom.get_name().strip():>4}"[:4]


def _write_staged_pdb(
    *,
    row: dict[str, Any],
    pdb_root: Path,
    output_path: Path,
    staged_chain_id: str = "A",
) -> StagedInput:
    protein_id = str(row["protein_id"])
    expected_sequence = str(row["sequence"]).upper()
    expected_length = int(row["sequence_length"])
    if len(expected_sequence) != expected_length:
        raise ValueError(
            f"{protein_id}: len(sequence)={len(expected_sequence)} "
            f"!= sequence_length={expected_length}"
        )
    if set(expected_sequence) - CANONICAL_AA:
        raise ValueError(f"{protein_id}: test-set sequence contains non-canonical AA")
    if expected_length > 9999:
        raise ValueError(f"{protein_id}: sequence too long for staged PDB numbering")

    source_path = resolve_structure_path(row, pdb_root)
    chain_id = str(row.get("if_chain_id") or row.get("chain") or "").strip()
    if not chain_id:
        chain_id = infer_chain_id(protein_id)
    structure = _load_structure(source_path)
    chain = _select_chain(structure, chain_id)

    residues: list[tuple[Any, str, dict[str, Any]]] = []
    for residue in chain.get_residues():
        aa = _residue_to_aa(residue)
        atoms_by_name = _selected_atoms_by_name(residue)
        missing = [atom for atom in REQUIRED_BACKBONE_ATOMS if atom not in atoms_by_name]
        if missing:
            raise ValueError(
                f"{protein_id}: residue {residue.id!r} missing backbone atoms {missing}"
            )
        residues.append((residue, aa, atoms_by_name))

    observed_sequence = "".join(aa for _residue, aa, _atoms in residues)
    if observed_sequence != expected_sequence:
        raise ValueError(
            f"{protein_id}: staged structure sequence mismatch "
            f"(structure_len={len(observed_sequence)}, parquet_len={expected_length})"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serial = 1
    with open(output_path, "w") as handle:
        for res_idx, (_residue, aa, atoms_by_name) in enumerate(residues, start=1):
            resname = AA1_TO3[aa]
            atoms = sorted(
                atoms_by_name.values(),
                key=lambda atom: int(atom.get_serial_number() or 0),
            )
            for atom in atoms:
                coord = atom.get_coord()
                occupancy = atom.get_occupancy()
                bfactor = atom.get_bfactor()
                element = (atom.element or atom.get_name().strip()[:1]).strip().upper()
                handle.write(
                    "ATOM  "
                    f"{serial:5d} "
                    f"{_atom_fullname(atom)}"
                    " "
                    f"{resname:>3} "
                    f"{staged_chain_id[:1]:1}"
                    f"{res_idx:4d}"
                    " "
                    "   "
                    f"{float(coord[0]):8.3f}"
                    f"{float(coord[1]):8.3f}"
                    f"{float(coord[2]):8.3f}"
                    f"{float(occupancy if occupancy is not None else 1.0):6.2f}"
                    f"{float(bfactor if bfactor is not None else 0.0):6.2f}"
                    "          "
                    f"{element[:2]:>2}"
                    "  \n"
                )
                serial += 1
        handle.write("TER\nEND\n")

    return StagedInput(
        protein_id=protein_id,
        stage_id=output_path.stem,
        source_path=str(source_path),
        staged_path=str(output_path),
        original_chain_id=str(chain.id),
        staged_chain_id=staged_chain_id[:1],
        sequence_length=expected_length,
    )


def prepare_mpnn_input(
    args: argparse.Namespace,
    mpnn_out: Path,
) -> tuple[Path, dict[str, str], dict[str, int]]:
    """Return the PDB folder ProteinMPNN should parse plus stage-id metadata."""
    if args.test_set_parquet is None:
        return args.input_pdb_folder, {}, {}

    df = pd.read_parquet(args.test_set_parquet).copy()
    required = {"protein_id", "sequence", "sequence_length", "pdb_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"test-set parquet missing required columns: {sorted(missing)}")
    df["protein_id"] = df["protein_id"].astype(str)
    if df["protein_id"].duplicated().any():
        raise ValueError("test-set parquet contains duplicate protein_id values")

    staged_dir = mpnn_out / "staged_pdbs"
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True)

    stage_to_protein: dict[str, str] = {}
    expected_lengths: dict[str, int] = {}
    manifest: list[dict[str, Any]] = []
    used_stage_ids: set[str] = set()
    for row in df.to_dict("records"):
        protein_id = str(row["protein_id"])
        stage_id = safe_file_id(protein_id)
        if stage_id in used_stage_ids:
            raise ValueError(f"safe stage id collision: {stage_id}")
        used_stage_ids.add(stage_id)
        staged_path = staged_dir / f"{stage_id}.pdb"
        staged = _write_staged_pdb(
            row=row,
            pdb_root=args.input_pdb_folder,
            output_path=staged_path,
        )
        stage_to_protein[stage_id] = protein_id
        expected_lengths[protein_id] = int(row["sequence_length"])
        manifest.append(staged.__dict__)

    with open(mpnn_out / "staged_inputs_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    with open(mpnn_out / "stage_id_to_protein_id.json", "w") as handle:
        json.dump(stage_to_protein, handle, indent=2, sort_keys=True)
    print(
        f"[stage] wrote {len(manifest)} continuous-numbered PDBs under {staged_dir}",
        flush=True,
    )
    return staged_dir, stage_to_protein, expected_lengths


# ─────────────────────────────────────────────────────────────────────────────
# ProteinMPNN invocation
# ─────────────────────────────────────────────────────────────────────────────

def _check_proteinmpnn_install() -> None:
    runner = PROTEINMPNN_ROOT / "protein_mpnn_run.py"
    if not runner.exists():
        raise FileNotFoundError(
            f"Vendored ProteinMPNN not found at {runner}. "
            "Expected DRAKES/drakes_protein/ProteinMPNN/protein_mpnn_run.py."
        )


def _run_subprocess(cmd: list[str], cwd: Path) -> None:
    print(f"[exec] (cwd={cwd}) " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd))


def run_proteinmpnn(
    args: argparse.Namespace,
    mpnn_out: Path,
) -> tuple[Path, dict[str, str], dict[str, int]]:
    """Run ProteinMPNN and return ``(seqs_dir, stage_id_map, expected_lengths)``."""
    _check_proteinmpnn_install()
    mpnn_out.mkdir(parents=True, exist_ok=True)
    mpnn_input_folder, stage_id_map, expected_lengths = prepare_mpnn_input(args, mpnn_out)

    helper = PROTEINMPNN_ROOT / "helper_scripts"
    parsed_jsonl = mpnn_out / "parsed_chains.jsonl"
    _run_subprocess(
        [
            sys.executable, str(helper / "parse_multiple_chains.py"),
            "--input_path", str(mpnn_input_folder.resolve()),
            "--output_path", str(parsed_jsonl),
        ],
        cwd=PROTEINMPNN_ROOT,
    )

    chain_id_jsonl = ""
    if args.design_chains.strip():
        chain_id_jsonl = str(mpnn_out / "chain_ids.jsonl")
        _run_subprocess(
            [
                sys.executable, str(helper / "assign_fixed_chains.py"),
                "--input_path", str(parsed_jsonl),
                "--output_path", chain_id_jsonl,
                "--chain_list", args.design_chains,
            ],
            cwd=PROTEINMPNN_ROOT,
        )

    runner = PROTEINMPNN_ROOT / "protein_mpnn_run.py"
    cmd = [
        sys.executable, str(runner),
        "--jsonl_path", str(parsed_jsonl),
        "--out_folder", str(mpnn_out),
        "--num_seq_per_target", str(args.num_seq_per_target),
        "--sampling_temp", args.sampling_temp,
        "--batch_size", str(args.mpnn_batch_size),
        "--seed", str(args.mpnn_seed),
        "--model_name", args.mpnn_model_name,
        "--device", args.mpnn_device,
    ]
    if args.ca_only:
        cmd.append("--ca_only")
    if chain_id_jsonl:
        cmd += ["--chain_id_jsonl", chain_id_jsonl]
    _run_subprocess(cmd, cwd=PROTEINMPNN_ROOT)

    seqs_dir = mpnn_out / "seqs"
    if not seqs_dir.is_dir():
        raise RuntimeError(f"ProteinMPNN did not produce seqs/ under {mpnn_out}")
    return seqs_dir, stage_id_map, expected_lengths


# ─────────────────────────────────────────────────────────────────────────────
# FASTA parsing — extract designs from MPNN seqs/<pid>.fa
# ─────────────────────────────────────────────────────────────────────────────

def _parse_kv_header(header: str) -> dict[str, str]:
    """Parse 'key=val, key=val, ...' MPNN header into a dict."""
    out: dict[str, str] = {}
    for kv in header.split(","):
        kv = kv.strip()
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_mpnn_seqs(
    seqs_dir: Path,
    protein_id_by_stage_id: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return {protein_id: [design_record, ...]} parsed from MPNN's *.fa files."""
    rows: dict[str, list[dict[str, Any]]] = {}
    protein_id_by_stage_id = protein_id_by_stage_id or {}
    for fa_path in sorted(seqs_dir.glob("*.fa")):
        stage_id = fa_path.stem
        pid = protein_id_by_stage_id.get(stage_id, stage_id)
        designs: list[dict[str, Any]] = []
        cur_header: str | None = None
        is_design = False
        design_idx = 0
        for line in fa_path.read_text().splitlines():
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                cur_header = line[1:]
                kv = _parse_kv_header(cur_header)
                # Design lines have a 'sample' key; the WT/input header does not.
                is_design = "sample" in kv
                continue
            if cur_header is None:
                continue
            if not is_design:
                continue
            kv = _parse_kv_header(cur_header)
            designs.append({
                "protein_id": pid,
                "design_idx": design_idx,
                "sequence": line.strip(),
                "mpnn_score": float(kv.get("score", "nan")),
                "mpnn_global_score": float(kv.get("global_score", "nan")),
                "seq_recovery": float(kv.get("seq_recovery", "nan")),
                "mpnn_temperature": float(kv.get("T", "nan")),
                "mpnn_sample": int(kv.get("sample", "0")),
            })
            design_idx += 1
        rows[pid] = designs
    return rows


def validate_generated_designs(
    designs: dict[str, list[dict[str, Any]]],
    *,
    expected_lengths: dict[str, int] | None = None,
) -> None:
    """Fail fast on sequences that downstream head/NMP cannot consume."""
    expected_lengths = expected_lengths or {}
    failures: list[str] = []
    for pid, design_list in designs.items():
        expected_len = expected_lengths.get(pid)
        for d in design_list:
            sequence = str(d["sequence"]).upper()
            chars = set(sequence)
            invalid = chars - CANONICAL_AA - set("/")
            if invalid:
                failures.append(
                    f"{pid}/design_{d['design_idx']}: non-canonical "
                    f"{''.join(sorted(invalid))}"
                )
                continue
            if expected_len is not None:
                if "/" in sequence:
                    failures.append(
                        f"{pid}/design_{d['design_idx']}: unexpected multi-chain output"
                    )
                    continue
                if len(sequence) != expected_len:
                    failures.append(
                        f"{pid}/design_{d['design_idx']}: len={len(sequence)} "
                        f"expected={expected_len}"
                    )
    if failures:
        preview = "; ".join(failures[:10])
        raise ValueError(
            f"ProteinMPNN generated invalid downstream sequences "
            f"({len(failures)} failures; first: {preview})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# NMP filter
# ─────────────────────────────────────────────────────────────────────────────

def _designed_chain_seqs(sequence: str) -> list[str]:
    """Split a multi-chain MPNN sequence on '/' and drop empties."""
    return [s for s in sequence.split("/") if s]


def score_designs_with_nmp(
    designs: dict[str, list[dict[str, Any]]],
    *,
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    """Annotate each design with `n_sb` and `mean_rank` (sum/mean across chains)."""
    from epitope_head.data.netmhciipan_runner import build_runner

    runner = build_runner(
        backend="standalone",
        binary_path=args.netmhciipan_bin,
        batch_size=args.nmp_batch_size,
        subprocess_timeout=args.nmp_timeout,
        max_lengths_per_call=args.nmp_max_lengths_per_call,
        n_workers=args.nmp_workers,
    )
    pep_lengths = [int(x) for x in args.nmp_pep_lengths.split() if x]
    if not pep_lengths:
        raise ValueError("--nmp-pep-lengths must list at least one length")
    threshold_frac = float(args.nmp_rank_threshold) / 100.0

    # Build a flat list of (entry_id → (pid, design_idx, chain_idx, seq))
    entries: list[tuple[str, str]] = []
    entry_back: list[tuple[str, int, int]] = []
    for pid, design_list in designs.items():
        for d in design_list:
            chains = _designed_chain_seqs(d["sequence"])
            for ci, chain_seq in enumerate(chains):
                entry_id = f"{pid}__d{d['design_idx']}__c{ci}"
                entries.append((entry_id, chain_seq))
                entry_back.append((pid, d["design_idx"], ci))

    if not entries:
        return designs

    print(
        f"[nmp] scoring {len(entries)} chains over lengths={pep_lengths}, "
        f"allele={args.allele}",
        flush=True,
    )
    t0 = time.time()
    batch_out = runner.score_batch(entries, args.allele, pep_lengths)
    t_nmp = time.time() - t0
    print(f"[nmp] score_batch wall = {t_nmp:.1f}s", flush=True)

    # Aggregate per-design across chains.
    agg: dict[tuple[str, int], dict[str, float | int]] = {}
    for (entry_id, _seq), (pid, di, _ci) in zip(entries, entry_back):
        scored = batch_out.get(entry_id, {})
        bucket = agg.setdefault((pid, di), {"n_windows": 0, "sum_rank": 0.0, "n_sb": 0})
        for pl, peptide_scores in scored.items():
            for ps in peptide_scores:
                bucket["n_windows"] += 1
                bucket["sum_rank"] += float(ps.el_rank)
                if float(ps.el_rank) <= threshold_frac:
                    bucket["n_sb"] += 1

    annotated: dict[str, list[dict[str, Any]]] = {}
    for pid, design_list in designs.items():
        out = []
        for d in design_list:
            bucket = agg.get((pid, d["design_idx"]), {})
            n_w = int(bucket.get("n_windows", 0))
            mean_rank = (
                float(bucket["sum_rank"]) / n_w if n_w > 0 else float("nan")
            )
            d2 = dict(d)
            d2["nmp_n_windows"] = n_w
            d2["nmp_n_sb"] = int(bucket.get("n_sb", 0))
            d2["nmp_mean_rank"] = mean_rank
            out.append(d2)
        annotated[pid] = out
    return annotated


def select_per_protein(
    annotated: dict[str, list[dict[str, Any]]],
    rule: str,
) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    for pid, design_list in annotated.items():
        if not design_list:
            continue
        if rule == "min_n_sb":
            # Tie-break with mean_rank to deterministically resolve ties.
            ranked = sorted(
                design_list,
                key=lambda d: (
                    d.get("nmp_n_sb", 10**9),
                    d.get("nmp_mean_rank", 1.0),
                    d["design_idx"],
                ),
            )
        elif rule == "min_mean_rank":
            ranked = sorted(
                design_list,
                key=lambda d: (
                    d.get("nmp_mean_rank", 1.0),
                    d.get("nmp_n_sb", 10**9),
                    d["design_idx"],
                ),
            )
        else:
            raise ValueError(f"unknown selection rule: {rule}")
        winner = ranked[0]
        selections.append({
            "protein_id": pid,
            "selected_design_idx": int(winner["design_idx"]),
            "sequence": winner["sequence"],
            "nmp_n_sb": int(winner.get("nmp_n_sb", -1)),
            "nmp_mean_rank": float(winner.get("nmp_mean_rank", float("nan"))),
            "nmp_n_windows": int(winner.get("nmp_n_windows", 0)),
            "n_candidates": len(design_list),
            "rule": rule,
        })
    return selections


# ─────────────────────────────────────────────────────────────────────────────
# Output writing
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(
    output_dir: Path,
    *,
    designs: dict[str, list[dict[str, Any]]],
    selections: list[dict[str, Any]] | None,
    run_config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # generated.parquet
    rows: list[dict[str, Any]] = []
    for pid in sorted(designs.keys()):
        for d in designs[pid]:
            rows.append(d)
    df = pd.DataFrame(rows)
    df.to_parquet(output_dir / "generated.parquet", index=False)

    # generated.fasta
    with open(output_dir / "generated.fasta", "w") as f:
        for pid in sorted(designs.keys()):
            for d in designs[pid]:
                hdr = (
                    f">{pid}|design_idx={d['design_idx']}"
                    f"|mpnn_score={d.get('mpnn_score', float('nan')):.4f}"
                    f"|seq_recovery={d.get('seq_recovery', float('nan')):.4f}"
                    f"|T={d.get('mpnn_temperature', float('nan')):.3f}"
                )
                if "nmp_n_sb" in d:
                    hdr += f"|n_sb={d['nmp_n_sb']}|mean_rank={d['nmp_mean_rank']:.4f}"
                f.write(hdr + "\n")
                f.write(d["sequence"] + "\n")

    # run_config.yaml
    with open(output_dir / "run_config.yaml", "w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)

    # selection (only when NMP filter ran)
    if selections is not None:
        sel_df = pd.DataFrame(selections)
        sel_df.to_parquet(output_dir / "selection.parquet", index=False)
        with open(output_dir / "selection.json", "w") as f:
            json.dump(selections, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Echo all hyperparameters/config args for experiment confirmation.
    print("─" * 72)
    print("ProteinMPNN baseline + NMP filter")
    print("─" * 72)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:32s} = {v}")
    print("─" * 72, flush=True)

    if not args.input_pdb_folder.is_dir():
        raise SystemExit(f"--input-pdb-folder is not a directory: {args.input_pdb_folder}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"--output-dir is not empty: {args.output_dir} (pass --overwrite)"
        )

    mpnn_out = args.output_dir / "_mpnn_raw"
    if mpnn_out.exists():
        shutil.rmtree(mpnn_out)
    seqs_dir, stage_id_map, expected_lengths = run_proteinmpnn(args, mpnn_out)
    designs = parse_mpnn_seqs(seqs_dir, protein_id_by_stage_id=stage_id_map)
    if not designs:
        raise SystemExit(f"No designs parsed from {seqs_dir}")
    validate_generated_designs(designs, expected_lengths=expected_lengths)
    if expected_lengths:
        missing = sorted(set(expected_lengths) - set(designs))
        unexpected = sorted(set(designs) - set(expected_lengths))
        if missing or unexpected:
            raise SystemExit(
                "ProteinMPNN output/test-set mismatch: "
                f"missing={missing[:10]} total_missing={len(missing)}; "
                f"unexpected={unexpected[:10]} total_unexpected={len(unexpected)}"
            )
    n_designs = sum(len(v) for v in designs.values())
    print(
        f"[mpnn] parsed {n_designs} designs across {len(designs)} proteins",
        flush=True,
    )

    selections: list[dict[str, Any]] | None = None
    if args.apply_nmp_filter:
        designs = score_designs_with_nmp(designs, args=args)
        selections = select_per_protein(designs, rule=args.nmp_rule)
        print(
            f"[nmp] selected {len(selections)} proteins (rule={args.nmp_rule})",
            flush=True,
        )

    run_config = {
        "method": "proteinmpnn_baseline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_proteins": len(designs),
        "n_designs": n_designs,
        "n_selected": len(selections) if selections is not None else 0,
        "n_staged_inputs": len(expected_lengths),
        "proteinmpnn_root": str(PROTEINMPNN_ROOT),
    }
    write_outputs(
        args.output_dir,
        designs=designs,
        selections=selections,
        run_config=run_config,
    )

    print(f"[done] artifacts written under {args.output_dir}")
    print(f"  generated.parquet  ({n_designs} rows)")
    print(f"  generated.fasta")
    print(f"  run_config.yaml")
    if selections is not None:
        print(f"  selection.parquet  ({len(selections)} rows)")
        print(f"  selection.json")
    if not args.keep_mpnn_raw:
        # Trim bulky helper artifacts but keep parsed inputs, seqs, and staging
        # manifest for audit.
        for child in mpnn_out.iterdir():
            if child.name not in {
                "seqs",
                "parsed_chains.jsonl",
                "staged_inputs_manifest.json",
                "stage_id_to_protein_id.json",
            }:
                if child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child)
    return 0


if __name__ == "__main__":
    sys.exit(main())
