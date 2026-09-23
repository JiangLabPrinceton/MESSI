#!/usr/bin/env python
"""Score static tetramer references and build homolog-specific contact consensus masks."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.evaluation.complex_interfaces import (
    aggregate_homomer_alanine_scan,
    canonical_protein_atoms,
    interface_pair_metrics,
    run_rosetta_alanine_scan,
    run_rosetta_interface_analyzer,
    summarize_interface_pairs,
)
from inverse_folding.evaluation.tetramer_contact_consensus import (
    aggregate_contact_consensus,
    assign_explicit_mechanistic_classes,
    canonical_residue_pair_contacts,
    compute_relay_pair_evidence,
    contact_candidates_with_mapping,
    extract_exact_homomer_mapping,
    load_position_annotations,
    normalize_aa_sequence,
    read_single_fasta,
    resolve_relay_mechanism,
)


def _load_structure(path: Path):
    lower_name = path.name.lower()
    if lower_name.endswith((".cif", ".mmcif")):
        from biotite.structure.io.pdbx import CIFFile, get_structure

        return get_structure(CIFFile.read(str(path)), model=1)
    if lower_name.endswith((".pdb", ".ent", ".pdb1")):
        from biotite.structure.io.pdb import PDBFile

        return PDBFile.read(str(path)).get_structure(model=1)
    raise ValueError(f"unsupported structure format: {path}")


def _parse_chain_pair(value: str) -> tuple[str, str]:
    fields = [field.strip() for field in re.split(r"[:,]", value) if field.strip()]
    if len(fields) != 2 or fields[0] == fields[1]:
        raise argparse.ArgumentTypeError(
            f"chain pair must contain two distinct IDs separated by ':' (got {value!r})"
        )
    return fields[0], fields[1]


def _parse_matching(value: str) -> tuple[tuple[str, str], tuple[str, str]]:
    fields = [field.strip() for field in value.split(",") if field.strip()]
    if len(fields) != 2:
        raise argparse.ArgumentTypeError(
            "catalytic matching must contain two chain pairs, e.g. A:B,C:D"
        )
    return _parse_chain_pair(fields[0]), _parse_chain_pair(fields[1])


def _parse_chains(value: str) -> list[str]:
    chains = [field.strip() for field in value.split(",") if field.strip()]
    if len(chains) != 4 or len(set(chains)) != 4:
        raise argparse.ArgumentTypeError("--chains requires four distinct comma-separated IDs")
    return chains


def _write_protein_pdb(arr, path: Path) -> None:
    from biotite.structure.io.pdb import PDBFile

    path.parent.mkdir(parents=True, exist_ok=True)
    pdb = PDBFile()
    pdb.set_structure(canonical_protein_atoms(arr))
    pdb.write(str(path))


def _write_table(df: pd.DataFrame, path_stem: Path) -> list[Path]:
    parquet_path = path_stem.with_suffix(".parquet")
    csv_path = path_stem.with_suffix(".csv")
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return [parquet_path, csv_path]


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structure", type=Path, action="append", required=True,
        help="tetramer PDB/CIF; repeat for a homolog-specific WT sample ensemble",
    )
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--protein-id",
        help="canonical parent identifier; enables exact-mapped contact-consensus mode",
    )
    canonical = parser.add_mutually_exclusive_group()
    canonical.add_argument("--canonical-fasta", type=Path)
    canonical.add_argument("--canonical-sequence")
    parser.add_argument(
        "--canonical-start-0b", type=int,
        help="explicit canonical start for a structured substring; default: unique exact match",
    )
    parser.add_argument(
        "--sample-id", action="append", default=[],
        help="sample identifier paired with each --structure; default: sanitized file stem",
    )
    parser.add_argument(
        "--position-annotations", type=Path,
        help="optional CSV/TSV/parquet position annotations joined after mask calculation",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="persistent reference tables and standardized structure (project work/ layer)",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        help="Rosetta inputs, protocols, commands, and scorefiles (project run/ layer)",
    )
    parser.add_argument(
        "--log-dir", type=Path,
        help="captured Rosetta stdout/stderr (project logs/ layer)",
    )
    parser.add_argument(
        "--chains", type=_parse_chains,
        help="four comma-separated protein chain IDs; default: auto-detect",
    )
    mechanism = parser.add_mutually_exclusive_group()
    mechanism.add_argument(
        "--catalytic-pair", type=_parse_chain_pair,
        help="one chain pair that identifies the catalytic D2 matching, e.g. A:B",
    )
    mechanism.add_argument(
        "--catalytic-matching", type=_parse_matching,
        help="the two catalytic symmetry-copy pairs, e.g. A:B,C:D",
    )
    mechanism.add_argument(
        "--relay-annotation", type=Path,
        help="independent donor/acceptor index_0b JSON used to resolve catalytic geometry",
    )
    parser.add_argument("--relay-contact-cutoff", type=float)
    parser.add_argument("--relay-min-contacts-per-direction", type=int)
    parser.add_argument(
        "--residue-pair-contact-cutoff", type=float, default=5.0,
        help="minimum-heavy-atom cutoff for exact within/inter-chain canonical pair maps",
    )
    parser.add_argument("--sasa-probe-radius", type=float, default=1.4)
    parser.add_argument("--sasa-point-number", type=int, default=1000)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--salt-bridge-cutoff", type=float, default=4.0)
    parser.add_argument(
        "--rosetta-interface-analyzer",
        help="optional InterfaceAnalyzer executable",
    )
    parser.add_argument(
        "--rosetta-scripts",
        help="optional rosetta_scripts executable for per-residue binding ddG scans",
    )
    parser.add_argument(
        "--alanine-scan-pair", action="append", type=_parse_chain_pair, default=[],
        help="repeatable original-chain pair, e.g. --alanine-scan-pair A:B",
    )
    parser.add_argument("--interface-residue-cutoff", type=float, default=5.0)
    parser.add_argument("--rosetta-score-function", default="ref2015")
    parser.add_argument("--rosetta-timeout-seconds", type=int, default=3600)
    return parser

def _resolve_sample_ids(structures: list[Path], requested: list[str]) -> list[str]:
    if requested and len(requested) != len(structures):
        raise ValueError(
            f"--sample-id count ({len(requested)}) must equal --structure count "
            f"({len(structures)})"
        )
    sample_ids = requested or [
        re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")
        for path in structures
    ]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("sample identifiers must not be empty")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", sample_id) for sample_id in sample_ids):
        raise ValueError("sample identifiers may contain only letters, numbers, '.', '_' and '-'")
    duplicates = sorted({sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1})
    if duplicates:
        raise ValueError(f"sample identifiers must be unique; duplicates={duplicates}")
    return sample_ids


def _relay_site_indices(
    payload: dict,
    *,
    index_key: str,
    record_key: str,
    canonical_sequence: str,
) -> list[int]:
    values = payload.get(index_key, payload.get(record_key))
    if values is None or not isinstance(values, list) or not values:
        raise ValueError(
            f"relay annotation requires a non-empty {index_key!r} or {record_key!r} list"
        )
    indices = []
    for value in values:
        if isinstance(value, dict):
            if "index_0b" not in value:
                raise ValueError(f"relay site record lacks index_0b: {value}")
            index = int(value["index_0b"])
            if value.get("expected_aa") is not None:
                if not 0 <= index < len(canonical_sequence):
                    raise ValueError(f"relay index_0b={index} lies outside canonical sequence")
                expected = str(value["expected_aa"]).upper()
                actual = canonical_sequence[index]
                if expected != actual:
                    raise ValueError(
                        f"relay annotation expected_aa={expected} differs from canonical {actual} "
                        f"at index_0b={index}"
                    )
        else:
            index = int(value)
        if not 0 <= index < len(canonical_sequence):
            raise ValueError(f"relay index_0b={index} lies outside canonical sequence")
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise ValueError(f"relay annotation contains duplicate indices in {index_key}")
    return sorted(indices)


def _load_relay_annotation(
    path: Path,
    *,
    protein_id: str,
    canonical_sequence: str,
    cutoff_override: float | None,
    min_contacts_override: int | None,
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("relay annotation must be a JSON object")
    annotated_id = payload.get("protein_id")
    if annotated_id is not None and str(annotated_id) != protein_id:
        raise ValueError(
            f"relay annotation protein_id={annotated_id!r} does not match {protein_id!r}"
        )
    donors = _relay_site_indices(
        payload,
        index_key="donor_indices_0b",
        record_key="donors",
        canonical_sequence=canonical_sequence,
    )
    acceptors = _relay_site_indices(
        payload,
        index_key="acceptor_indices_0b",
        record_key="acceptors",
        canonical_sequence=canonical_sequence,
    )
    cutoff = float(
        cutoff_override
        if cutoff_override is not None
        else payload.get("contact_cutoff_a", 5.0)
    )
    min_contacts = int(
        min_contacts_override
        if min_contacts_override is not None
        else payload.get("min_contacts_per_direction", 1)
    )
    if cutoff <= 0:
        raise ValueError("relay contact cutoff must be positive")
    if min_contacts <= 0:
        raise ValueError("relay min_contacts_per_direction must be positive")
    return {
        "donor_indices_0b": donors,
        "acceptor_indices_0b": acceptors,
        "contact_cutoff_a": cutoff,
        "min_contacts_per_direction": min_contacts,
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _annotate_exact_pair_classes(
    contacts: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    out = contacts.copy()
    if out.empty:
        for column in (
            "chain_pair", "bsa_rank_class", "mechanistic_class", "consensus_class",
            "interface_copy", "d2_matching", "mechanism_resolution",
        ):
            out[column] = pd.Series(dtype="object")
        return out
    pair_lookup = pair_df.set_index("chain_pair")
    out["chain_pair"] = [
        f"{row.chain_1}:{row.chain_2}"
        if bool(row.is_inter_chain)
        else f"{row.chain_1}:{row.chain_1}"
        for row in out.itertuples(index=False)
    ]
    for column in (
        "bsa_rank_class", "mechanistic_class", "consensus_class", "interface_copy",
        "d2_matching", "mechanism_resolution",
    ):
        values = []
        for row in out.itertuples(index=False):
            if not bool(row.is_inter_chain):
                values.append(0 if column == "interface_copy" else "within_chain")
            else:
                values.append(pair_lookup.loc[row.chain_pair, column])
        out[column] = values
    return out


def _contact_mask_payload(
    consensus: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    protein_id: str,
    name: str,
    cutoff: float,
) -> dict[str, object]:
    n_samples = int(consensus["n_samples"].iloc[0])
    mechanisms = set(pairs["mechanistic_class"].astype(str))
    mechanism_status = "unresolved" if mechanisms == {"unresolved"} else "resolved"
    classes = {}
    for class_name, group in consensus.groupby("consensus_class", sort=False):
        mechanisms_for_class = sorted(set(group["mechanistic_class"].astype(str)))
        classes[str(class_name)] = {
            "mechanistic_class": (
                mechanisms_for_class[0] if len(mechanisms_for_class) == 1 else "unresolved"
            ),
            "bsa_rank_classes_seen": sorted(
                set(
                    value
                    for values in group["bsa_rank_classes_seen"].astype(str)
                    for value in values.split(",")
                    if value
                )
            ),
            "positions_index_0b": {
                "all_N": group.loc[group["mask_all_N"], "index_0b"].astype(int).tolist(),
                "5ofN": group.loc[group["mask_5ofN"], "index_0b"].astype(int).tolist(),
                "4ofN": group.loc[group["mask_4ofN"], "index_0b"].astype(int).tolist(),
                "1ofN": group.loc[group["mask_1ofN"], "index_0b"].astype(int).tolist(),
            },
        }
    return {
        "schema_version": "tetramer_contact_masks_v1",
        "protein_id": protein_id,
        "name": name,
        "n_samples": n_samples,
        "contact_cutoff_a": float(cutoff),
        "mechanism_status": mechanism_status,
        "mask_semantics": {
            "sample_event": "canonical position contacts across a matching chain pair",
            "symmetry_copy_requirement": "both copies",
            "kofN": "position satisfies the both-copy event in at least k samples",
            "literal_thresholds": True,
            "5ofN_available": n_samples >= 5,
            "4ofN_available": n_samples >= 4,
            "annotations_excluded_from_masks": True,
        },
        "classes": classes,
    }


def _run_contact_consensus_mode(
    args,
    *,
    structures: list[Path],
    canonical_sequence: str,
    fasta_header: str | None,
    command: str,
) -> int:
    canonical = normalize_aa_sequence(canonical_sequence, source="canonical sequence")
    sample_ids = _resolve_sample_ids(structures, list(args.sample_id))
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    relay = None
    if args.relay_annotation is not None:
        relay = _load_relay_annotation(
            args.relay_annotation,
            protein_id=args.protein_id,
            canonical_sequence=canonical,
            cutoff_override=args.relay_contact_cutoff,
            min_contacts_override=args.relay_min_contacts_per_direction,
        )

    arrays = {}
    mappings = {}
    source_meta = {}
    pair_frames = []
    mapping_frames = []
    output_paths: list[Path] = []
    for sample_id, source in zip(sample_ids, structures):
        arr = _load_structure(source)
        protein = canonical_protein_atoms(arr)
        detected_chains = sorted(set(protein.chain_id.astype(str)))
        chains = args.chains or detected_chains
        if len(chains) != 4 or any(chain not in detected_chains for chain in chains):
            raise ValueError(
                f"sample {sample_id}: expected exactly four selected protein chains; "
                f"selected={chains}, detected={detected_chains}"
            )
        mapping = extract_exact_homomer_mapping(
            arr,
            canonical_sequence=canonical,
            chains=chains,
            canonical_start_0b=args.canonical_start_0b,
        )
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        mapping.insert(0, "source_sha256", source_sha256)
        mapping.insert(0, "source_structure", str(source))
        mapping.insert(0, "sample_id", sample_id)
        mapping.insert(0, "name", args.name)
        mapping.insert(0, "protein_id", args.protein_id)
        mapping_frames.append(mapping)

        pairs = interface_pair_metrics(
            arr,
            chains,
            sasa_probe_radius=args.sasa_probe_radius,
            sasa_point_number=args.sasa_point_number,
            contact_cutoff=args.contact_cutoff,
            salt_bridge_cutoff=args.salt_bridge_cutoff,
        )
        pairs["bsa_rank_class"] = pairs["interface_class"].astype(str)
        pairs.insert(0, "source_sha256", source_sha256)
        pairs.insert(0, "source_structure", str(source))
        pairs.insert(0, "sample_id", sample_id)
        pairs.insert(0, "name", args.name)
        pairs.insert(0, "protein_id", args.protein_id)
        if relay is not None:
            pairs = compute_relay_pair_evidence(
                arr,
                mapping,
                pairs,
                donor_indices_0b=relay["donor_indices_0b"],
                acceptor_indices_0b=relay["acceptor_indices_0b"],
                cutoff=float(relay["contact_cutoff_a"]),
                min_contacts_per_direction=int(relay["min_contacts_per_direction"]),
            )
        pair_frames.append(pairs)
        arrays[sample_id] = arr
        mappings[sample_id] = mapping
        source_meta[sample_id] = {
            "path": str(source),
            "sha256": source_sha256,
            "chains": list(chains),
        }
        standardized = out_dir / f"{args.name}_{sample_id}_protein_standardized.pdb"
        _write_protein_pdb(arr, standardized)
        output_paths.append(standardized)

    pairs = pd.concat(pair_frames, ignore_index=True)
    if "is_biological_interface" not in pairs:
        raise ValueError("D2 pair table lacks the BSA-rank biological-interface flag")
    pairs = pairs.rename(columns={
        "is_biological_interface": "bsa_is_biological_interface",
    })
    if relay is not None:
        pairs = resolve_relay_mechanism(pairs)
        mechanism_source = "relay_annotation"
    else:
        pairs = assign_explicit_mechanistic_classes(
            pairs,
            catalytic_pair=args.catalytic_pair,
            catalytic_matching=args.catalytic_matching,
        )
        mechanism_source = (
            "explicit_pair" if args.catalytic_pair is not None
            else "explicit_matching" if args.catalytic_matching is not None
            else "none"
        )

    mechanism_resolved = pairs["mechanistic_class"].astype(str) != "unresolved"
    pairs["mechanistic_is_biological_interface"] = pd.Series(
        pd.NA, index=pairs.index, dtype="boolean"
    )
    pairs.loc[
        mechanism_resolved, "mechanistic_is_biological_interface"
    ] = pairs.loc[mechanism_resolved, "mechanistic_class"].isin({
        "catalytic", "assembly"
    })

    candidate_frames = []
    residue_pair_frames = []
    for sample_id in sample_ids:
        sample_pairs = pairs[pairs["sample_id"].astype(str) == sample_id].copy()
        candidates = contact_candidates_with_mapping(
            arrays[sample_id],
            mappings[sample_id],
            sample_pairs,
            cutoff=args.interface_residue_cutoff,
        )
        candidates.insert(0, "source_sha256", source_meta[sample_id]["sha256"])
        candidates.insert(0, "source_structure", source_meta[sample_id]["path"])
        candidates.insert(0, "name", args.name)
        candidates.insert(0, "protein_id", args.protein_id)
        candidate_frames.append(candidates)

        exact_pairs = canonical_residue_pair_contacts(
            arrays[sample_id],
            mappings[sample_id],
            cutoff=args.residue_pair_contact_cutoff,
        )
        exact_pairs = _annotate_exact_pair_classes(exact_pairs, sample_pairs)
        exact_pairs.insert(0, "source_sha256", source_meta[sample_id]["sha256"])
        exact_pairs.insert(0, "source_structure", source_meta[sample_id]["path"])
        exact_pairs.insert(0, "sample_id", sample_id)
        exact_pairs.insert(0, "name", args.name)
        exact_pairs.insert(0, "protein_id", args.protein_id)
        residue_pair_frames.append(exact_pairs)

    mappings_all = pd.concat(mapping_frames, ignore_index=True)
    candidates_all = pd.concat(candidate_frames, ignore_index=True)
    residue_pairs_all = pd.concat(residue_pair_frames, ignore_index=True)
    consensus_core = aggregate_contact_consensus(
        candidates_all,
        pairs,
        canonical_sequence=canonical,
    )
    mask_payload = _contact_mask_payload(
        consensus_core,
        pairs,
        protein_id=args.protein_id,
        name=args.name,
        cutoff=args.interface_residue_cutoff,
    )

    consensus = consensus_core.copy()
    if args.position_annotations is not None:
        annotations = load_position_annotations(
            args.position_annotations, canonical, protein_id=args.protein_id
        )
        if "protein_id" in annotations:
            annotation_ids = sorted(set(annotations["protein_id"].dropna().astype(str)))
            if annotation_ids and annotation_ids != [args.protein_id]:
                raise ValueError(
                    f"position annotation protein_id values {annotation_ids} do not match "
                    f"{args.protein_id!r}"
                )
            annotations = annotations.drop(columns="protein_id")
        consensus = consensus.merge(
            annotations, on="index_0b", how="left", validate="many_to_one"
        )
    consensus.insert(0, "name", args.name)
    consensus.insert(0, "protein_id", args.protein_id)

    table_specs = [
        (mappings_all, out_dir / f"{args.name}_residue_mapping"),
        (pairs, out_dir / f"{args.name}_contact_chain_pairs"),
        (candidates_all, out_dir / f"{args.name}_contact_candidates"),
        (residue_pairs_all, out_dir / f"{args.name}_residue_pair_contacts"),
        (consensus, out_dir / f"{args.name}_contact_consensus"),
    ]
    for table, stem in table_specs:
        output_paths.extend(_write_table(table, stem))
    mask_path = out_dir / f"{args.name}_contact_masks.json"
    mask_path.write_text(json.dumps(mask_payload, indent=2, sort_keys=True) + "\n")
    output_paths.append(mask_path)

    entrypoint_path = Path(__file__).resolve()
    contact_library_path = (
        PROJECT_ROOT / "inverse_folding" / "evaluation" / "tetramer_contact_consensus.py"
    ).resolve()
    canonical_source = (
        {
            "kind": "fasta",
            "path": str(args.canonical_fasta.resolve()),
            "sha256": hashlib.sha256(args.canonical_fasta.read_bytes()).hexdigest(),
            "selection": "exact_full_header",
        }
        if args.canonical_fasta is not None
        else {"kind": "inline_sequence"}
    )
    metadata_path = out_dir / f"{args.name}_contact_metadata.json"
    metadata = {
        "schema_version": "tetramer_contact_consensus_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "persistent_output_dir": str(out_dir),
        "implementation": {
            "entrypoint": str(entrypoint_path),
            "entrypoint_sha256": hashlib.sha256(entrypoint_path.read_bytes()).hexdigest(),
            "contact_library": str(contact_library_path),
            "contact_library_sha256": hashlib.sha256(
                contact_library_path.read_bytes()
            ).hexdigest(),
        },
        "protein_id": args.protein_id,
        "name": args.name,
        "canonical_fasta_header": fasta_header,
        "canonical_source": canonical_source,
        "canonical_length": len(canonical),
        "canonical_sequence_md5": hashlib.md5(canonical.encode()).hexdigest(),
        "canonical_sequence_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "sources": source_meta,
        "parameters": {
            "canonical_start_0b": args.canonical_start_0b,
            "sasa_probe_radius_a": args.sasa_probe_radius,
            "sasa_point_number": args.sasa_point_number,
            "bsa_contact_cutoff_a": args.contact_cutoff,
            "salt_bridge_cutoff_a": args.salt_bridge_cutoff,
            "interface_residue_cutoff_a": args.interface_residue_cutoff,
            "residue_pair_contact_cutoff_a": args.residue_pair_contact_cutoff,
            "residue_pair_contact_definition": "minimum heavy-atom distance below cutoff",
            "mask_symmetry_copy_requirement": "both copies",
            "mask_threshold_counts": [5, 4, 1],
        },
        "mechanism": {
            "source": mechanism_source,
            "status": mask_payload["mechanism_status"],
            "relay": relay,
            "resolution_by_sample": {
                str(sample_id): sorted(set(group["mechanism_resolution"].astype(str)))[0]
                for sample_id, group in pairs.groupby("sample_id", sort=False)
            },
            "bsa_rank_is_not_mechanism": True,
        },
        "position_annotations": (
            {
                "path": str(args.position_annotations.resolve()),
                "sha256": hashlib.sha256(args.position_annotations.read_bytes()).hexdigest(),
                "excluded_from_contact_masks": True,
            }
            if args.position_annotations is not None else None
        ),
        "row_counts": {
            "residue_mapping": int(len(mappings_all)),
            "contact_chain_pairs": int(len(pairs)),
            "contact_candidates": int(len(candidates_all)),
            "residue_pair_contacts": int(len(residue_pairs_all)),
            "contact_consensus": int(len(consensus)),
        },
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "biotite": _package_version("biotite"),
            "scipy": _package_version("scipy"),
        },
        "outputs": [str(path) for path in [*output_paths, metadata_path]],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote homolog-specific contact consensus for {args.protein_id}: "
        f"{len(sample_ids)} samples, {len(candidates_all)} candidate rows, "
        f"{len(residue_pairs_all)} exact residue-pair contacts -> {out_dir}"
    )
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(cli_argv)
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *cli_argv])
    structures = [path.resolve() for path in args.structure]
    for structure in structures:
        if not structure.is_file():
            parser.error(f"structure does not exist: {structure}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        parser.error("--name may contain only letters, numbers, '.', '_' and '-'")
    if args.sasa_point_number <= 0:
        parser.error("--sasa-point-number must be positive")
    if args.interface_residue_cutoff <= 0:
        parser.error("--interface-residue-cutoff must be positive")
    if args.residue_pair_contact_cutoff <= 0:
        parser.error("--residue-pair-contact-cutoff must be positive")
    if args.canonical_start_0b is not None and args.canonical_start_0b < 0:
        parser.error("--canonical-start-0b must be non-negative")
    if args.alanine_scan_pair and not args.rosetta_scripts:
        parser.error("--alanine-scan-pair requires --rosetta-scripts")
    rosetta_requested = bool(args.rosetta_interface_analyzer or args.alanine_scan_pair)

    contact_mode = bool(
        len(structures) > 1
        or args.protein_id
        or args.canonical_fasta
        or args.canonical_sequence
        or args.sample_id
        or args.position_annotations
        or args.catalytic_pair
        or args.catalytic_matching
        or args.relay_annotation
    )
    if contact_mode:
        if not args.protein_id:
            parser.error("contact-consensus mode requires --protein-id")
        if args.canonical_fasta is None and args.canonical_sequence is None:
            parser.error(
                "contact-consensus mode requires --canonical-fasta or --canonical-sequence"
            )
        if rosetta_requested:
            parser.error(
                "contact-consensus mode keeps energy separate: run Rosetta independently and "
                "join its position table with --position-annotations"
            )
        if args.relay_contact_cutoff is not None and args.relay_annotation is None:
            parser.error("--relay-contact-cutoff requires --relay-annotation")
        if (
            args.relay_min_contacts_per_direction is not None
            and args.relay_annotation is None
        ):
            parser.error("--relay-min-contacts-per-direction requires --relay-annotation")
        if args.canonical_fasta is not None:
            fasta_header, canonical_sequence = read_single_fasta(
                args.canonical_fasta, protein_id=args.protein_id
            )
        else:
            fasta_header = None
            canonical_sequence = normalize_aa_sequence(
                args.canonical_sequence, source="--canonical-sequence"
            )
        return _run_contact_consensus_mode(
            args,
            structures=structures,
            canonical_sequence=canonical_sequence,
            fasta_header=fasta_header,
            command=command,
        )

    if len(structures) != 1:
        parser.error("legacy static-reference mode requires exactly one --structure")
    if rosetta_requested and args.run_dir is None:
        parser.error("Rosetta scoring requires --run-dir")
    if rosetta_requested and args.log_dir is None:
        parser.error("Rosetta scoring requires --log-dir")

    source = structures[0]
    out_dir = args.out_dir.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir is not None else None
    log_dir = args.log_dir.resolve() if args.log_dir is not None else None
    roots = [("out", out_dir), ("run", run_dir), ("log", log_dir)]
    for index, (left_name, left_path) in enumerate(roots):
        if left_path is None:
            continue
        for right_name, right_path in roots[index + 1:]:
            if right_path is not None and _paths_overlap(left_path, right_path):
                parser.error(
                    f"--{left_name}-dir and --{right_name}-dir must be separate directory trees"
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    arr = _load_structure(source)
    protein = canonical_protein_atoms(arr)
    detected_chains = sorted(set(protein.chain_id.astype(str)))
    chains = args.chains or detected_chains
    if len(chains) != 4 or any(chain not in detected_chains for chain in chains):
        parser.error(
            f"expected exactly four selected protein chains; selected={chains}, "
            f"detected={detected_chains}"
        )

    output_paths = []
    cleaned_path = out_dir / f"{args.name}_protein_standardized.pdb"
    _write_protein_pdb(arr, cleaned_path)
    output_paths.append(cleaned_path)

    chain_rows = []
    protein_ins_codes = (
        protein.ins_code.astype(str)
        if "ins_code" in protein.get_annotation_categories()
        else np.full(len(protein), "", dtype="U1")
    )
    for chain in chains:
        chain_mask = protein.chain_id == chain
        residue_keys = sorted(set(zip(
            protein.res_id[chain_mask].astype(int), protein_ins_codes[chain_mask]
        )))
        residue_ids = sorted(set(res_id for res_id, _ in residue_keys))
        missing_internal = sorted(set(range(residue_ids[0], residue_ids[-1] + 1)) - set(residue_ids))
        chain_rows.append({
            "name": args.name,
            "chain": chain,
            "n_standardized_residues": len(residue_keys),
            "first_res_id": residue_ids[0],
            "last_res_id": residue_ids[-1],
            "missing_internal_res_ids": ",".join(map(str, missing_internal)),
        })
    chain_summary = pd.DataFrame(chain_rows)
    output_paths.extend(_write_table(
        chain_summary, out_dir / f"{args.name}_chain_summary"
    ))

    pairs = interface_pair_metrics(
        arr,
        chains,
        sasa_probe_radius=args.sasa_probe_radius,
        sasa_point_number=args.sasa_point_number,
        contact_cutoff=args.contact_cutoff,
        salt_bridge_cutoff=args.salt_bridge_cutoff,
    )
    pairs.insert(0, "name", args.name)
    if args.rosetta_interface_analyzer:
        assert run_dir is not None and log_dir is not None
        pairs = run_rosetta_interface_analyzer(
            pairs,
            lambda _: arr,
            executable=args.rosetta_interface_analyzer,
            run_dir=run_dir / "interface_analyzer",
            log_dir=log_dir / "interface_analyzer",
            timeout_seconds=args.rosetta_timeout_seconds,
        )

    summary = {
        "name": args.name,
        "source_structure": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "protein_chains": ",".join(chains),
        "n_protein_chains": len(chains),
        "standardized_residue_count_min": int(chain_summary["n_standardized_residues"].min()),
        "standardized_residue_count_max": int(chain_summary["n_standardized_residues"].max()),
        "sasa_probe_radius_a": float(args.sasa_probe_radius),
        "sasa_point_number": int(args.sasa_point_number),
        "contact_cutoff_a": float(args.contact_cutoff),
        "salt_bridge_cutoff_a": float(args.salt_bridge_cutoff),
        **summarize_interface_pairs(pairs),
    }
    output_paths.extend(_write_table(pairs, out_dir / f"{args.name}_interface_pairs"))

    scan = None
    by_position = None
    if args.alanine_scan_pair:
        assert run_dir is not None and log_dir is not None
        scan = run_rosetta_alanine_scan(
            arr,
            args.alanine_scan_pair,
            executable=args.rosetta_scripts,
            run_dir=run_dir / "alanine_scan",
            log_dir=log_dir / "alanine_scan",
            timeout_seconds=args.rosetta_timeout_seconds,
            interface_cutoff=args.interface_residue_cutoff,
            score_function=args.rosetta_score_function,
        )
        class_by_pair = pairs.set_index("chain_pair")["interface_class"].to_dict()
        scan["interface_class"] = scan["interface_pair"].map(class_by_pair)
        if scan["interface_class"].isna().any():
            unknown = sorted(scan.loc[scan["interface_class"].isna(), "interface_pair"].unique())
            raise ValueError(f"alanine-scan pairs are absent from tetramer pair table: {unknown}")
        by_position = aggregate_homomer_alanine_scan(scan)
        output_paths.extend(_write_table(
            scan, out_dir / f"{args.name}_alanine_scan_chain_residue"
        ))
        output_paths.extend(_write_table(
            by_position, out_dir / f"{args.name}_alanine_scan_by_position"
        ))
        for interface_pair, group in by_position.groupby("interface_pair", sort=True):
            prefix = "alanine_scan_" + interface_pair.replace(":", "")
            interpretable = group[group["is_interpretable_sidechain_alanine"]]
            summary[f"{prefix}_n_positions"] = int(len(group))
            summary[f"{prefix}_n_interpretable_sidechain_positions"] = int(len(interpretable))
            summary[f"{prefix}_all_mutations_ddg_bind_mean_reu"] = float(
                group["ddg_bind_mean_reu"].mean()
            )
            summary[f"{prefix}_all_mutations_ddg_bind_max_reu"] = float(
                group["ddg_bind_mean_reu"].max()
            )
            summary[f"{prefix}_sidechain_ddg_bind_mean_reu"] = float(
                interpretable["ddg_bind_mean_reu"].mean()
            )
            summary[f"{prefix}_sidechain_ddg_bind_max_reu"] = float(
                interpretable["ddg_bind_mean_reu"].max()
            )
            summary[f"{prefix}_sidechain_ddg_bind_min_reu"] = float(
                interpretable["ddg_bind_mean_reu"].min()
            )
            summary[f"{prefix}_max_chain_range_reu"] = float(
                group["ddg_bind_chain_range_reu"].max()
            )

    summary_df = pd.DataFrame([summary])
    output_paths.extend(_write_table(summary_df, out_dir / f"{args.name}_interface_summary"))
    summary_json_path = out_dir / f"{args.name}_interface_summary.json"
    summary_json_path.write_text(summary_df.to_json(orient="records", indent=2) + "\n")
    output_paths.append(summary_json_path)

    metadata = {
        "schema_version": "tetramer_reference_metrics_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "source_structure": str(source),
        "source_sha256": summary["source_sha256"],
        "persistent_output_dir": str(out_dir),
        "runtime_dir": str(run_dir) if run_dir is not None else None,
        "log_dir": str(log_dir) if log_dir is not None else None,
        "command": command,
        "parameters": {
            "chains": chains,
            "sasa_probe_radius_a": args.sasa_probe_radius,
            "sasa_point_number": args.sasa_point_number,
            "contact_cutoff_a": args.contact_cutoff,
            "salt_bridge_cutoff_a": args.salt_bridge_cutoff,
            "interface_residue_cutoff_a": args.interface_residue_cutoff,
            "alanine_scan_pairs": [list(pair) for pair in args.alanine_scan_pair],
            "rosetta_score_function": args.rosetta_score_function,
            "alanine_scan_repack_bound": False,
            "alanine_scan_repack_unbound": False,
            "ddg_sign_convention": "mutant_minus_wildtype",
            "interpretable_sidechain_scan_excludes": [
                "ALA", "GLY", "PRO", "DISULFIDE_CYS"
            ],
        },
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "biotite": _package_version("biotite"),
            "rosetta_interface_analyzer": args.rosetta_interface_analyzer,
            "rosetta_scripts": args.rosetta_scripts,
        },
        "outputs": [str(path) for path in output_paths],
        "runtime_artifacts": (
            [str(path) for path in sorted(run_dir.rglob("*")) if path.is_file()]
            if run_dir is not None else []
        ),
        "log_artifacts": (
            [str(path) for path in sorted(log_dir.rglob("*")) if path.is_file()]
            if log_dir is not None else []
        ),
        "row_counts": {
            "interface_pairs": int(len(pairs)),
            "chain_summary": int(len(chain_summary)),
            "alanine_scan_chain_residue": int(len(scan)) if scan is not None else 0,
            "alanine_scan_by_position": int(len(by_position)) if by_position is not None else 0,
        },
    }
    metadata_path = out_dir / "metadata.json"
    metadata["outputs"].append(str(metadata_path))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Wrote 1 summary row and {len(pairs)} interface-pair rows to {out_dir}")
    if scan is not None:
        print(
            f"Wrote {len(scan)} chain-residue and {len(by_position)} position-level "
            "alanine-scan rows"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
