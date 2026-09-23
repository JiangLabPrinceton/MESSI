"""Homolog-specific D2 tetramer contact consensus with exact canonical numbering.

The BSA rank labels emitted by :mod:`complex_interfaces` are intentionally kept separate from
mechanistic labels.  A catalytic matching is assigned only from an explicit chain pair/matching or
from an independently supplied donor--acceptor relay annotation.  With neither source, the
mechanism remains unresolved and consensus is reported by BSA rank class.

This module is predictor-independent and torch-free.  Contact masks contain coordinate-derived
contacts only; optional functional or energy annotations are joined by the CLI after mask
calculation and never participate in contact inclusion.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from inverse_folding.evaluation.complex_interfaces import (
    canonical_protein_atoms,
    interface_residue_candidates,
)


AA3_TO_1 = {
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
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AA20 = frozenset(AA3_TO_1.values())


def normalize_aa_sequence(sequence: str, *, source: str = "sequence") -> str:
    """Normalize an AA20 sequence and reject gaps, ambiguous residues, and empty input."""

    normalized = "".join(str(sequence).split()).upper()
    if not normalized:
        raise ValueError(f"{source}: empty amino-acid sequence")
    invalid = sorted(set(normalized) - AA20)
    if invalid:
        raise ValueError(f"{source}: sequence contains non-AA20 symbols {invalid}")
    return normalized


def read_single_fasta(path: Path, *, protein_id: str | None = None) -> tuple[str, str]:
    """Read one FASTA record, optionally selected by an exact header/protein id."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[tuple[str, list[str]]] = []
    header: str | None = None
    chunks: list[str] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, chunks))
            header = line[1:].strip()
            chunks = []
            if not header:
                raise ValueError(f"{path}:{line_number}: empty FASTA header")
        elif header is None:
            raise ValueError(f"{path}:{line_number}: sequence appears before FASTA header")
        else:
            chunks.append(line)
    if header is not None:
        records.append((header, chunks))
    if protein_id is None:
        if len(records) != 1:
            raise ValueError(f"{path}: expected exactly one FASTA record, found {len(records)}")
        fasta_header, fasta_chunks = records[0]
    else:
        matching = [record for record in records if record[0] == str(protein_id)]
        if not matching:
            raise ValueError(
                f"{path}: no FASTA record has exact header {protein_id!r}"
            )
        if len(matching) > 1:
            raise ValueError(
                f"{path}: duplicate FASTA records have exact header {protein_id!r}"
            )
        fasta_header, fasta_chunks = matching[0]
    return fasta_header, normalize_aa_sequence(
        "".join(fasta_chunks), source=f"{path} FASTA record {fasta_header!r}"
    )


def _ins_codes(arr) -> np.ndarray:
    if "ins_code" in arr.get_annotation_categories():
        return arr.ins_code.astype(str)
    return np.full(len(arr), "", dtype="U1")


def _ordered_chain_residues(protein, chain: str) -> list[dict[str, object]]:
    chain_arr = protein[protein.chain_id.astype(str) == str(chain)]
    if len(chain_arr) == 0:
        raise ValueError(f"chain {chain}: no canonical protein residues")
    ins_codes = _ins_codes(chain_arr)
    keys = list(dict.fromkeys(zip(chain_arr.res_id.astype(int), ins_codes)))
    rows: list[dict[str, object]] = []
    for chain_position_0b, (res_id, ins_code) in enumerate(keys):
        mask = (chain_arr.res_id == res_id) & (ins_codes == ins_code)
        names = set(chain_arr.res_name[mask].astype(str))
        if len(names) != 1:
            raise ValueError(
                f"chain {chain}: ambiguous residue identity at {res_id}{ins_code}"
            )
        name3 = next(iter(names))
        aa = AA3_TO_1.get(name3)
        if aa is None:
            raise ValueError(
                f"chain {chain}: residue {res_id}{ins_code} is not canonical AA20 ({name3})"
            )
        rows.append({
            "chain": str(chain),
            "chain_position_0b": int(chain_position_0b),
            "res_id": int(res_id),
            "ins_code": str(ins_code),
            "observed_residue_name3": name3,
            "observed_aa": aa,
        })
    return rows


def _exact_substring_start(
    observed: str,
    canonical: str,
    *,
    chain: str,
    canonical_start_0b: int | None,
) -> int:
    if canonical_start_0b is not None:
        start = int(canonical_start_0b)
        if start < 0 or start + len(observed) > len(canonical):
            raise ValueError(
                f"chain {chain}: canonical_start_0b={start} cannot map {len(observed)} residues "
                f"into a canonical sequence of length {len(canonical)}"
            )
        expected = canonical[start:start + len(observed)]
        if observed != expected:
            raise ValueError(
                f"chain {chain}: structure sequence is not the exact canonical substring at "
                f"canonical_start_0b={start}; observed={observed!r}, expected={expected!r}"
            )
        return start

    starts = [
        start
        for start in range(len(canonical) - len(observed) + 1)
        if canonical.startswith(observed, start)
    ]
    if not starts:
        raise ValueError(
            f"chain {chain}: structure sequence is not an exact canonical substring; "
            f"observed length={len(observed)}, canonical length={len(canonical)}"
        )
    if len(starts) > 1:
        raise ValueError(
            f"chain {chain}: structure sequence has multiple exact occurrences in the canonical "
            f"sequence at starts {starts}; provide canonical_start_0b explicitly"
        )
    return starts[0]


def extract_exact_homomer_mapping(
    arr,
    *,
    canonical_sequence: str,
    chains: Sequence[str],
    canonical_start_0b: int | None = None,
) -> pd.DataFrame:
    """Map every residue of four exact homomer chains to canonical ``index_0b``.

    Author residue numbers are retained as structure keys but are never treated as canonical
    positions.  Each complete chain sequence must be one exact, contiguous canonical substring;
    an ambiguous repeated substring requires an explicit ``canonical_start_0b``.  All four chains
    must resolve to the same canonical interval.
    """

    canonical = normalize_aa_sequence(canonical_sequence, source="canonical sequence")
    selected = [str(chain) for chain in chains]
    if len(selected) != 4 or len(set(selected)) != 4:
        raise ValueError(f"exact homotetramer mapping requires four distinct chains, got {selected}")
    protein = canonical_protein_atoms(arr)
    detected = set(protein.chain_id.astype(str))
    missing = sorted(set(selected) - detected)
    if missing:
        raise ValueError(f"selected protein chains are absent from structure: {missing}")

    mapped_rows: list[dict[str, object]] = []
    intervals: dict[str, tuple[int, int]] = {}
    for chain in selected:
        rows = _ordered_chain_residues(protein, chain)
        observed = "".join(str(row["observed_aa"]) for row in rows)
        start = _exact_substring_start(
            observed,
            canonical,
            chain=chain,
            canonical_start_0b=canonical_start_0b,
        )
        intervals[chain] = (start, start + len(rows))
        for row in rows:
            index_0b = start + int(row["chain_position_0b"])
            mapped_rows.append({
                **row,
                "index_0b": index_0b,
                "position_1b": index_0b + 1,
                "expected_aa": canonical[index_0b],
            })

    unique_intervals = set(intervals.values())
    if len(unique_intervals) != 1:
        raise ValueError(
            "homomer chains map to different canonical intervals: "
            + ", ".join(f"{chain}={interval}" for chain, interval in intervals.items())
        )
    mapping = pd.DataFrame(mapped_rows)
    if mapping.duplicated(["chain", "res_id", "ins_code"]).any():
        raise ValueError("duplicate structure residue key in exact homomer mapping")
    if mapping.duplicated(["chain", "index_0b"]).any():
        raise ValueError("duplicate canonical index within a homomer chain")
    if not (mapping["observed_aa"] == mapping["expected_aa"]).all():  # pragma: no cover
        raise RuntimeError("internal exact-mapping invariant failed")
    return mapping.sort_values(["chain", "chain_position_0b"]).reset_index(drop=True)


def _pair_key(chain_a: str, chain_b: str) -> tuple[str, str]:
    return tuple(sorted((str(chain_a), str(chain_b))))


def _matching_edges(label: str) -> frozenset[tuple[str, str]]:
    fields = [field.strip() for field in str(label).split("|") if field.strip()]
    edges = []
    for field in fields:
        chains = [chain.strip() for chain in field.split(":") if chain.strip()]
        if len(chains) != 2 or chains[0] == chains[1]:
            raise ValueError(f"invalid D2 matching edge {field!r} in {label!r}")
        edges.append(_pair_key(*chains))
    if len(edges) != 2 or len(set(edges)) != 2:
        raise ValueError(f"D2 matching must contain two distinct edges, got {label!r}")
    return frozenset(edges)


def _ensure_bsa_rank_class(pair_df: pd.DataFrame) -> pd.DataFrame:
    out = pair_df.copy()
    if "bsa_rank_class" not in out:
        if "interface_class" not in out:
            raise ValueError("pair table requires bsa_rank_class or interface_class")
        out["bsa_rank_class"] = out["interface_class"].astype(str)
    required = {
        "sample_id", "chain_1", "chain_2", "chain_pair", "d2_matching",
        "d2_matching_mean_bsa_total_a2", "interface_copy", "bsa_rank_class",
    }
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"pair table missing required columns {missing}")
    return out


def _assign_classes_for_catalytic_matching(
    sample: pd.DataFrame,
    catalytic_matching: str,
) -> pd.DataFrame:
    out = sample.copy()
    matching_bsa = (
        out.groupby("d2_matching", sort=False)["d2_matching_mean_bsa_total_a2"]
        .first()
        .astype(float)
    )
    if catalytic_matching not in matching_bsa.index:
        raise ValueError(
            f"sample {out['sample_id'].iloc[0]}: catalytic matching {catalytic_matching!r} "
            f"is absent; available={list(matching_bsa.index)}"
        )
    remaining = matching_bsa.drop(index=catalytic_matching).sort_values(
        ascending=False, kind="mergesort"
    )
    if len(remaining) != 2:
        raise ValueError("D2 mechanism assignment requires exactly three perfect matchings")
    assembly_matching, diagonal_matching = remaining.index.tolist()
    class_by_matching = {
        catalytic_matching: "catalytic",
        assembly_matching: "assembly",
        diagonal_matching: "diagonal",
    }
    out["mechanistic_class"] = out["d2_matching"].map(class_by_matching)
    out["consensus_class"] = out["mechanistic_class"]
    return out


def assign_explicit_mechanistic_classes(
    pair_df: pd.DataFrame,
    *,
    catalytic_pair: tuple[str, str] | None = None,
    catalytic_matching: Iterable[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Assign catalytic/assembly/diagonal from an explicit matching declaration.

    With no declaration, every mechanistic label is ``unresolved`` while ``consensus_class``
    remains the BSA rank class.  This deliberately prevents ``interface_1`` from silently becoming
    synonymous with the catalytic interface.
    """

    if catalytic_pair is not None and catalytic_matching is not None:
        raise ValueError("provide catalytic_pair or catalytic_matching, not both")
    pairs = _ensure_bsa_rank_class(pair_df)
    if catalytic_pair is None and catalytic_matching is None:
        pairs["mechanistic_class"] = "unresolved"
        pairs["mechanism_resolution"] = "unresolved_no_annotation"
        pairs["consensus_class"] = pairs["bsa_rank_class"]
        return pairs

    requested_edges = (
        frozenset(_pair_key(*edge) for edge in catalytic_matching)
        if catalytic_matching is not None
        else None
    )
    if requested_edges is not None and len(requested_edges) != 2:
        raise ValueError("explicit catalytic matching must contain exactly two distinct pairs")
    frames = []
    for sample_id, sample in pairs.groupby("sample_id", sort=False):
        if catalytic_pair is not None:
            requested_pair = _pair_key(*catalytic_pair)
            hit = sample[
                [
                    _pair_key(row.chain_1, row.chain_2) == requested_pair
                    for row in sample.itertuples()
                ]
            ]
            if len(hit) != 1:
                raise ValueError(
                    f"sample {sample_id}: explicit catalytic pair {requested_pair} matched "
                    f"{len(hit)} pair rows"
                )
            selected_matching = str(hit.iloc[0]["d2_matching"])
        else:
            matching_lookup = {
                _matching_edges(label): str(label)
                for label in sample["d2_matching"].astype(str).unique()
            }
            if requested_edges not in matching_lookup:
                raise ValueError(
                    f"sample {sample_id}: explicit catalytic matching {sorted(requested_edges)} "
                    f"is absent; available={sorted(map(sorted, matching_lookup))}"
                )
            selected_matching = matching_lookup[requested_edges]
        resolved = _assign_classes_for_catalytic_matching(sample, selected_matching)
        resolved["mechanism_resolution"] = "explicit"
        frames.append(resolved)
    return pd.concat(frames, ignore_index=True)


def _residue_atom_coords(
    protein,
    mapping_lookup: dict[tuple[str, int], tuple[int, str]],
    chain: str,
    index_0b: int,
) -> np.ndarray:
    try:
        res_id, ins_code = mapping_lookup[(str(chain), int(index_0b))]
    except KeyError as error:
        raise ValueError(
            f"relay index_0b={index_0b} is not mapped on chain {chain}"
        ) from error
    protein_ins = _ins_codes(protein)
    mask = (
        (protein.chain_id.astype(str) == str(chain))
        & (protein.res_id.astype(int) == int(res_id))
        & (protein_ins == str(ins_code))
    )
    coords = protein.coord[mask]
    if len(coords) == 0:  # pragma: no cover - mapping/protein invariant
        raise RuntimeError(f"mapped residue atoms disappeared for {chain}:{res_id}{ins_code}")
    return np.asarray(coords, dtype=float)


def _directional_relay_stats(
    protein,
    mapping_lookup,
    donor_chain: str,
    acceptor_chain: str,
    donor_indices: Sequence[int],
    acceptor_indices: Sequence[int],
    cutoff: float,
) -> dict[str, object]:
    contacts: list[list[int]] = []
    minima: list[float] = []
    for donor in donor_indices:
        donor_coords = _residue_atom_coords(
            protein, mapping_lookup, donor_chain, int(donor)
        )
        for acceptor in acceptor_indices:
            acceptor_coords = _residue_atom_coords(
                protein, mapping_lookup, acceptor_chain, int(acceptor)
            )
            distance = float(np.linalg.norm(
                donor_coords[:, None, :] - acceptor_coords[None, :, :], axis=-1
            ).min())
            minima.append(distance)
            if distance < cutoff:
                contacts.append([int(donor), int(acceptor)])
    return {
        "min_distance": min(minima) if minima else np.nan,
        "contact_count": len(contacts),
        "contacts": contacts,
    }


def compute_relay_pair_evidence(
    arr,
    mapping: pd.DataFrame,
    pair_df: pd.DataFrame,
    *,
    donor_indices_0b: Sequence[int],
    acceptor_indices_0b: Sequence[int],
    cutoff: float = 5.0,
    min_contacts_per_direction: int = 1,
) -> pd.DataFrame:
    """Add bidirectional donor--acceptor relay geometry to every D2 chain pair."""

    if cutoff <= 0:
        raise ValueError("relay contact cutoff must be positive")
    if min_contacts_per_direction <= 0:
        raise ValueError("min_contacts_per_direction must be positive")
    donors = tuple(sorted(set(map(int, donor_indices_0b))))
    acceptors = tuple(sorted(set(map(int, acceptor_indices_0b))))
    if not donors or not acceptors:
        raise ValueError("relay annotation requires non-empty donor and acceptor index sets")
    mapped_indices = set(mapping["index_0b"].astype(int))
    missing = sorted((set(donors) | set(acceptors)) - mapped_indices)
    if missing:
        raise ValueError(f"relay annotation indices are absent from the exact mapping: {missing}")

    protein = canonical_protein_atoms(arr)
    mapping_lookup = {
        (str(row.chain), int(row.index_0b)): (int(row.res_id), str(row.ins_code))
        for row in mapping.itertuples(index=False)
    }
    out = _ensure_bsa_rank_class(pair_df)
    records = []
    for row in out.itertuples(index=False):
        forward = _directional_relay_stats(
            protein, mapping_lookup, str(row.chain_1), str(row.chain_2),
            donors, acceptors, cutoff,
        )
        reverse = _directional_relay_stats(
            protein, mapping_lookup, str(row.chain_2), str(row.chain_1),
            donors, acceptors, cutoff,
        )
        records.append({
            "relay_forward_min_heavy_atom_distance_a": forward["min_distance"],
            "relay_reverse_min_heavy_atom_distance_a": reverse["min_distance"],
            "relay_min_heavy_atom_distance_a": float(np.nanmin([
                forward["min_distance"], reverse["min_distance"]
            ])),
            "relay_forward_contact_pairs": int(forward["contact_count"]),
            "relay_reverse_contact_pairs": int(reverse["contact_count"]),
            "relay_forward_contacts_index_0b_json": json.dumps(forward["contacts"]),
            "relay_reverse_contacts_index_0b_json": json.dumps(reverse["contacts"]),
            "relay_supported": (
                int(forward["contact_count"]) >= min_contacts_per_direction
                and int(reverse["contact_count"]) >= min_contacts_per_direction
            ),
        })
    evidence = pd.concat([out.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    return evidence


def resolve_relay_mechanism(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Resolve a unique catalytic D2 matching supported in both symmetry copies.

    If any sample is ambiguous or lacks one supported copy, the complete ensemble remains
    mechanistically unresolved and consensus falls back to BSA rank classes.  Per-sample proposed
    classes and resolution reasons remain in ``relay_sample_class`` and
    ``mechanism_resolution`` for audit.
    """

    pairs = _ensure_bsa_rank_class(pair_df)
    if "relay_supported" not in pairs:
        raise ValueError("relay pair table requires relay_supported")
    frames = []
    all_samples_resolved = True
    for sample_id, sample in pairs.groupby("sample_id", sort=False):
        supported_matchings = []
        for matching, group in sample.groupby("d2_matching", sort=False):
            if len(group) != 2:
                raise ValueError(
                    f"sample {sample_id}: D2 matching {matching!r} must have two symmetry copies"
                )
            if group["relay_supported"].astype(bool).all():
                supported_matchings.append(str(matching))
        if len(supported_matchings) == 1:
            resolved = _assign_classes_for_catalytic_matching(
                sample, supported_matchings[0]
            )
            resolved["relay_sample_class"] = resolved["mechanistic_class"]
            resolved["mechanism_resolution"] = "relay_unique_both_copies"
        else:
            all_samples_resolved = False
            resolved = sample.copy()
            resolved["relay_sample_class"] = "unresolved"
            resolved["mechanistic_class"] = "unresolved"
            resolved["consensus_class"] = resolved["bsa_rank_class"]
            resolved["mechanism_resolution"] = (
                "relay_no_matching_with_both_copies"
                if not supported_matchings
                else "relay_multiple_matchings_with_both_copies"
            )
        frames.append(resolved)
    out = pd.concat(frames, ignore_index=True)
    if not all_samples_resolved:
        out["mechanistic_class"] = "unresolved"
        out["consensus_class"] = out["bsa_rank_class"]
    return out


def contact_candidates_with_mapping(
    arr,
    mapping: pd.DataFrame,
    pair_df: pd.DataFrame,
    *,
    cutoff: float = 5.0,
) -> pd.DataFrame:
    """Return every chain-pair 5 A candidate with an exact canonical index."""

    pairs = _ensure_bsa_rank_class(pair_df)
    if pairs["sample_id"].nunique() != 1:
        raise ValueError("contact_candidates_with_mapping expects exactly one sample")
    lookup = mapping.set_index(["chain", "res_id", "ins_code"])
    frames = []
    pair_columns = [
        "sample_id", "chain_pair", "bsa_rank_class", "interface_copy", "d2_matching",
        "d2_matching_mean_bsa_total_a2", "mechanistic_class", "consensus_class",
    ]
    for pair in pairs.itertuples(index=False):
        candidates = interface_residue_candidates(
            arr, str(pair.chain_1), str(pair.chain_2), cutoff=cutoff
        )
        if candidates.empty:
            continue
        mapped_rows = []
        for candidate in candidates.itertuples(index=False):
            key = (
                str(candidate.source_chain), int(candidate.res_id), str(candidate.ins_code)
            )
            if key not in lookup.index:
                raise ValueError(f"contact candidate has no exact canonical mapping: {key}")
            mapped = lookup.loc[key]
            mapped_rows.append({
                **candidate._asdict(),
                "index_0b": int(mapped["index_0b"]),
                "position_1b": int(mapped["position_1b"]),
                "expected_aa": str(mapped["expected_aa"]),
                "observed_aa": str(mapped["observed_aa"]),
            })
        frame = pd.DataFrame(mapped_rows)
        values = pair._asdict()
        for column in pair_columns:
            frame[column] = values[column]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[
            *pair_columns, "source_chain", "partner_chain", "res_id", "ins_code",
            "index_0b", "position_1b", "expected_aa", "observed_aa",
            "min_cross_chain_heavy_atom_distance_a",
        ])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(
        ["sample_id", "bsa_rank_class", "interface_copy", "source_chain", "index_0b"]
    ).reset_index(drop=True)


def canonical_residue_pair_contacts(
    arr,
    mapping: pd.DataFrame,
    *,
    cutoff: float = 5.0,
) -> pd.DataFrame:
    """Return exact within- and inter-chain residue-pair contacts in canonical numbering.

    A contact uses the existing Q00511 EV-validation geometry: minimum heavy-atom distance below
    ``cutoff`` (default 5 A).  Raw chain-copy rows are retained.  ``is_tetramer_only`` means that
    the canonical position pair is observed inter-chain but not within any monomer chain in the
    same sample.  Equal canonical indices across two protomers are retained and flagged, although
    they are not a valid Potts ``i,j`` pair.
    """

    if cutoff <= 0:
        raise ValueError("residue-pair contact cutoff must be positive")
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - project dependency boundary
        raise RuntimeError("scipy is required for exact residue-pair contact enumeration") from error

    selected_chains = set(mapping["chain"].astype(str))
    protein = canonical_protein_atoms(arr)
    protein = protein[np.isin(protein.chain_id.astype(str), sorted(selected_chains))]
    protein_ins = _ins_codes(protein)
    mapping_by_key = {
        (str(row.chain), int(row.res_id), str(row.ins_code)): row
        for row in mapping.itertuples(index=False)
    }
    residue_keys = list(dict.fromkeys(zip(
        protein.chain_id.astype(str), protein.res_id.astype(int), protein_ins
    )))
    residue_index_by_key = {key: index for index, key in enumerate(residue_keys)}
    missing = sorted(set(residue_keys) - set(mapping_by_key))
    if missing:
        raise ValueError(f"structure residues are absent from exact canonical mapping: {missing[:5]}")
    atom_residue_indices = np.asarray([
        residue_index_by_key[(str(chain), int(res_id), str(ins_code))]
        for chain, res_id, ins_code in zip(
            protein.chain_id, protein.res_id, protein_ins
        )
    ], dtype=int)
    tree = cKDTree(np.asarray(protein.coord, dtype=float))
    atom_pairs = tree.query_pairs(r=float(cutoff), output_type="ndarray")
    minima: dict[tuple[int, int], float] = {}
    if len(atom_pairs):
        distances = np.linalg.norm(
            protein.coord[atom_pairs[:, 0]] - protein.coord[atom_pairs[:, 1]], axis=1
        )
        for atom_pair, distance in zip(atom_pairs, distances):
            if float(distance) >= cutoff:
                continue
            residue_a = int(atom_residue_indices[int(atom_pair[0])])
            residue_b = int(atom_residue_indices[int(atom_pair[1])])
            if residue_a == residue_b:
                continue
            key = tuple(sorted((residue_a, residue_b)))
            minima[key] = min(minima.get(key, np.inf), float(distance))

    rows = []
    for (residue_a, residue_b), distance in minima.items():
        key_a, key_b = residue_keys[residue_a], residue_keys[residue_b]
        mapped_a, mapped_b = mapping_by_key[key_a], mapping_by_key[key_b]
        # Stable structural orientation: chain first, then canonical index and author key.
        left = (key_a, mapped_a)
        right = (key_b, mapped_b)
        if (
            str(mapped_b.chain), int(mapped_b.index_0b), int(mapped_b.res_id), str(mapped_b.ins_code)
        ) < (
            str(mapped_a.chain), int(mapped_a.index_0b), int(mapped_a.res_id), str(mapped_a.ins_code)
        ):
            left, right = right, left
        key_1, row_1 = left
        key_2, row_2 = right
        index_1, index_2 = int(row_1.index_0b), int(row_2.index_0b)
        is_inter = str(row_1.chain) != str(row_2.chain)
        rows.append({
            "chain_1": str(row_1.chain),
            "res_id_1": int(row_1.res_id),
            "ins_code_1": str(row_1.ins_code),
            "index_1_0b": index_1,
            "aa_1": str(row_1.expected_aa),
            "chain_2": str(row_2.chain),
            "res_id_2": int(row_2.res_id),
            "ins_code_2": str(row_2.ins_code),
            "index_2_0b": index_2,
            "aa_2": str(row_2.expected_aa),
            "canonical_pair_min_0b": min(index_1, index_2),
            "canonical_pair_max_0b": max(index_1, index_2),
            "is_same_canonical_index": index_1 == index_2,
            "is_inter_chain": is_inter,
            "min_heavy_atom_distance_a": distance,
        })
    columns = [
        "chain_1", "res_id_1", "ins_code_1", "index_1_0b", "aa_1",
        "chain_2", "res_id_2", "ins_code_2", "index_2_0b", "aa_2",
        "canonical_pair_min_0b", "canonical_pair_max_0b", "is_same_canonical_index",
        "is_inter_chain", "min_heavy_atom_distance_a", "observed_within_chain_any_copy",
        "is_tetramer_only", "contact_kind",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)
    within_pairs = {
        (int(row.canonical_pair_min_0b), int(row.canonical_pair_max_0b))
        for row in out.loc[~out["is_inter_chain"]].itertuples(index=False)
    }
    canonical_pairs = list(zip(out["canonical_pair_min_0b"], out["canonical_pair_max_0b"]))
    out["observed_within_chain_any_copy"] = [pair in within_pairs for pair in canonical_pairs]
    out["is_tetramer_only"] = (
        out["is_inter_chain"] & ~out["observed_within_chain_any_copy"]
    )
    out["contact_kind"] = np.select(
        [
            ~out["is_inter_chain"],
            out["is_tetramer_only"],
        ],
        ["within_chain", "tetramer_only"],
        default="inter_chain_shared_with_monomer",
    )
    return out[columns].sort_values([
        "is_inter_chain", "chain_1", "chain_2", "index_1_0b", "index_2_0b"
    ]).reset_index(drop=True)


def aggregate_contact_consensus(
    candidates: pd.DataFrame,
    pair_df: pd.DataFrame,
    *,
    canonical_sequence: str,
) -> pd.DataFrame:
    """Aggregate position contacts over samples and both D2 symmetry copies.

    The ``kofN`` masks are conservative: a position must contact in **both** matching copies in at
    least ``k`` samples.  Copy occupancy and any-copy sample occupancy are also emitted so no
    information is hidden by this mask definition.  For fewer than four or five samples, the
    corresponding literal threshold mask is empty rather than silently changing its meaning.
    """

    canonical = normalize_aa_sequence(canonical_sequence, source="canonical sequence")
    pairs = _ensure_bsa_rank_class(pair_df)
    required = {"consensus_class", "mechanistic_class"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"mechanism-annotated pair table missing columns {missing}")
    sample_ids = list(dict.fromkeys(pairs["sample_id"].astype(str)))
    if not sample_ids:
        raise ValueError("cannot aggregate an empty pair table")
    classes = list(dict.fromkeys(pairs["consensus_class"].astype(str)))
    for (sample_id, class_name), group in pairs.groupby(
        ["sample_id", "consensus_class"], sort=False
    ):
        if len(group) != 2 or group["chain_pair"].nunique() != 2:
            raise ValueError(
                f"sample {sample_id} consensus class {class_name!r} must contain exactly two "
                "symmetry-copy chain pairs"
            )

    if candidates.empty:
        contact_lookup: dict[tuple[str, str], set[int]] = {}
    else:
        candidate_required = {"sample_id", "chain_pair", "index_0b"}
        candidate_missing = sorted(candidate_required - set(candidates.columns))
        if candidate_missing:
            raise ValueError(f"candidate table missing columns {candidate_missing}")
        bad_indices = sorted(
            set(candidates["index_0b"].astype(int)) - set(range(len(canonical)))
        )
        if bad_indices:
            raise ValueError(f"candidate indices lie outside canonical sequence: {bad_indices}")
        valid_pairs = set(zip(pairs["sample_id"].astype(str), pairs["chain_pair"].astype(str)))
        observed_pairs = set(zip(
            candidates["sample_id"].astype(str), candidates["chain_pair"].astype(str)
        ))
        unknown_pairs = sorted(observed_pairs - valid_pairs)
        if unknown_pairs:
            raise ValueError(f"candidate rows reference unknown sample/chain pairs: {unknown_pairs}")
        contact_lookup = {
            (str(sample_id), str(chain_pair)): set(group["index_0b"].astype(int))
            for (sample_id, chain_pair), group in candidates.groupby(
                ["sample_id", "chain_pair"], sort=False
            )
        }

    n_samples = len(sample_ids)
    rows = []
    for class_name in classes:
        class_pairs = pairs[pairs["consensus_class"].astype(str) == class_name]
        bsa_classes = ",".join(sorted(set(class_pairs["bsa_rank_class"].astype(str))))
        mechanism_values = set(class_pairs["mechanistic_class"].astype(str))
        mechanism = next(iter(mechanism_values)) if len(mechanism_values) == 1 else "unresolved"
        pairs_by_sample = {
            str(sample_id): group.sort_values("interface_copy")["chain_pair"].astype(str).tolist()
            for sample_id, group in class_pairs.groupby("sample_id", sort=False)
        }
        if set(pairs_by_sample) != set(sample_ids):
            raise ValueError(
                f"consensus class {class_name!r} is not represented in every sample"
            )
        for index_0b, aa in enumerate(canonical):
            n_copy_contacts = 0
            n_samples_any = 0
            n_samples_both = 0
            for sample_id in sample_ids:
                copy_hits = [
                    index_0b in contact_lookup.get((sample_id, chain_pair), set())
                    for chain_pair in pairs_by_sample[sample_id]
                ]
                n_copy_contacts += sum(copy_hits)
                n_samples_any += int(any(copy_hits))
                n_samples_both += int(all(copy_hits))
            rows.append({
                "consensus_class": class_name,
                "mechanistic_class": mechanism,
                "bsa_rank_classes_seen": bsa_classes,
                "index_0b": index_0b,
                "position_1b": index_0b + 1,
                "aa": aa,
                "n_samples": n_samples,
                "n_symmetry_copy_opportunities": 2 * n_samples,
                "n_copy_contacts": n_copy_contacts,
                "copy_contact_frequency": n_copy_contacts / (2 * n_samples),
                "n_samples_any_copy": n_samples_any,
                "sample_any_copy_frequency": n_samples_any / n_samples,
                "n_samples_both_copies": n_samples_both,
                "sample_both_copies_frequency": n_samples_both / n_samples,
                "mask_all_N": n_samples_both == n_samples,
                "mask_5ofN": n_samples_both >= 5,
                "mask_4ofN": n_samples_both >= 4,
                "mask_1ofN": n_samples_both >= 1,
            })
    return pd.DataFrame(rows).sort_values(
        ["consensus_class", "index_0b"]
    ).reset_index(drop=True)


def load_position_annotations(
    path: Path,
    canonical_sequence: str,
    *,
    protein_id: str | None = None,
) -> pd.DataFrame:
    """Load optional target-position annotations without granting mask semantics.

    Accepted long-form index columns are 'index_0b' and 'target_index_0b'.  The latter is
    useful for homolog projections because 'target_aa' is validated against the target canonical
    sequence while a separate reference identity column is retained but never enforced.  The
    frozen uricase 'active_site_anchor_map.csv' semicolon-list schema is expanded per parent.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        annotations = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        annotations = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    else:
        raise ValueError("position annotations must be parquet, CSV, or TSV")
    annotations = annotations.copy()
    if protein_id is not None and "protein_id" in annotations:
        annotations = annotations[
            annotations["protein_id"].astype(str) == str(protein_id)
        ].copy()
        if annotations.empty:
            raise ValueError(f"position annotations contain no row for protein_id={protein_id!r}")

    frozen_columns = {
        "hard_anchor_indices_0b",
        "hard_anchor_labels",
        "hard_anchor_expected_aa",
        "monitored_shell_indices_0b",
        "monitored_shell_labels",
    }
    if "index_0b" not in annotations and "target_index_0b" not in annotations:
        if not frozen_columns.issubset(annotations.columns):
            raise ValueError(
                "position annotations require index_0b, target_index_0b, or the frozen "
                "active_site_anchor_map semicolon columns"
            )
        if len(annotations) != 1:
            raise ValueError(
                "frozen active-site annotation maps require one parent row; pass protein_id"
            )

        def split_cell(value) -> list[str]:
            if pd.isna(value) or not str(value).strip():
                return []
            return [field.strip() for field in str(value).split(";") if field.strip()]

        source_row = annotations.iloc[0]
        expanded = []
        for tier, index_column, label_column, aa_column in (
            (
                "hard_anchor",
                "hard_anchor_indices_0b",
                "hard_anchor_labels",
                "hard_anchor_expected_aa",
            ),
            (
                "monitored_shell",
                "monitored_shell_indices_0b",
                "monitored_shell_labels",
                None,
            ),
        ):
            indices = split_cell(source_row[index_column])
            labels = split_cell(source_row[label_column])
            aas = split_cell(source_row[aa_column]) if aa_column is not None else []
            if len(labels) != len(indices):
                raise ValueError(
                    f"frozen annotation {tier} label/index list lengths differ"
                )
            if aa_column is not None and len(aas) != len(indices):
                raise ValueError(
                    f"frozen annotation {tier} AA/index list lengths differ"
                )
            for item_i, (index, label) in enumerate(zip(indices, labels)):
                expanded.append({
                    "protein_id": (
                        str(source_row["protein_id"])
                        if "protein_id" in source_row.index else protein_id
                    ),
                    "index_0b": int(float(index)),
                    "target_aa": aas[item_i] if aas else None,
                    "functional_label": label,
                    "constraint_tier": tier,
                    "source_config": (
                        source_row.get("source_config")
                        if "source_config" in source_row.index else None
                    ),
                })
        annotations = pd.DataFrame(expanded)
    elif "index_0b" not in annotations:
        annotations = annotations.rename(columns={"target_index_0b": "index_0b"})

    if annotations.empty:
        raise ValueError("position annotations are empty after parent selection")
    annotations["index_0b"] = pd.to_numeric(
        annotations["index_0b"], errors="raise"
    ).astype(int)
    if annotations["index_0b"].duplicated().any():
        raise ValueError("position annotations contain duplicate index_0b values")
    canonical = normalize_aa_sequence(canonical_sequence, source="canonical sequence")
    if not annotations["index_0b"].between(0, len(canonical) - 1).all():
        raise ValueError("position annotation index_0b lies outside canonical sequence")

    actual_aa_column = next(
        (
            column
            for column in ("target_aa", "actual_aa", "aa")
            if column in annotations
        ),
        None,
    )
    if actual_aa_column is not None:
        actual = annotations[actual_aa_column]
        missing_actual = actual.isna() | (actual.astype(str).str.strip() == "")
        if missing_actual.any():
            annotations.loc[missing_actual, actual_aa_column] = annotations.loc[
                missing_actual, "index_0b"
            ].map(lambda index: canonical[int(index)])
        expected = annotations["index_0b"].map(lambda index: canonical[int(index)])
        mismatch = annotations[actual_aa_column].astype(str).str.upper() != expected
        if mismatch.any():
            bad = annotations.loc[
                mismatch, ["index_0b", actual_aa_column]
            ].to_dict("records")
            raise ValueError(
                f"position annotation target AA differs from canonical sequence: {bad}"
            )

    key_columns = {"index_0b", "protein_id"}
    rename = {
        column: f"annotation_{column}"
        for column in annotations.columns
        if column not in key_columns
    }
    return annotations.rename(columns=rename).sort_values("index_0b").reset_index(drop=True)
