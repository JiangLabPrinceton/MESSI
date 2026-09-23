"""Reusable post-prediction interface metrics for predicted protein complexes.

Two complex topologies are supported, dispatched on chain count by ``classify_interface_pairs``:

  * **four chains** — a D2-symmetric homotetramer (uricase). The six pair edges form three
    perfect matchings, ranked by matching-level BSA into two repeated biological interface
    classes plus the diagonal non-interface.
  * **two chains** — a hetero-dimer (a de novo binder and its target). There is one interface,
    no symmetry copy, and no diagonal, so the "conservative copy" statistics degenerate to the
    single measurement and the topology-separation block is omitted rather than fabricated.

Everything below the classifier — BSA, residue contacts, salt bridges, and the Rosetta
InterfaceAnalyzer dimer layer — is topology-agnostic and shared by both.

This module is predictor-independent and torch-free. Coordinate metrics require Biotite only when
called; Rosetta InterfaceAnalyzer is an optional subprocess layer for final-selection scoring.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import pandas as pd

D2_INTERFACE_CLASSES = ("interface_1", "interface_2", "diagonal")

CANONICAL_AMINO_ACIDS = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
})

# Parent mappings are intentionally explicit: silently guessing a modified residue's chemistry can
# corrupt both SASA and Rosetta energies. 1R51 starts with N-acetyl-serine (SAC).
MODIFIED_AMINO_ACID_PARENTS = {
    "SAC": ("SER", frozenset({"N", "CA", "C", "O", "OXT", "CB", "OG"})),
}

ROSETTA_SCORE_FIELDS = {
    "dSASA_int": "rosetta_dsasa_int_a2",
    "dG_separated": "rosetta_dg_separated_reu",
    "dG_separated/dSASAx100": "rosetta_dg_per_sasa_x100",
    "packstat": "rosetta_packstat",
    "hbonds_int": "rosetta_hbonds_int",
    "hbond_E_fraction": "rosetta_hbond_energy_fraction",
    "delta_unsatHbonds": "rosetta_delta_unsat_hbonds",
    "sc_value": "rosetta_shape_complementarity",
    "nres_int": "rosetta_nres_int",
}

ROSETTA_REQUIRED_FIELDS = {
    "dSASA_int",
    "dG_separated",
    "dG_separated/dSASAx100",
    "packstat",
    "hbonds_int",
    "hbond_E_fraction",
    "delta_unsatHbonds",
}


def _ins_codes(arr) -> np.ndarray:
    if "ins_code" in arr.get_annotation_categories():
        return arr.ins_code.astype(str)
    return np.full(len(arr), "", dtype="U1")


def canonical_protein_atoms(arr):
    """Return complete, canonicalized polymer amino-acid heavy atoms.

    Canonical HETATM amino acids are excluded because they may be free crystallization components.
    A residue must contain N/CA/C to prevent incomplete terminal records from entering SASA. Known
    modified polymer residues are mapped only through ``MODIFIED_AMINO_ACID_PARENTS``.
    """
    if len(arr) == 0:
        return arr.copy()

    ins_code = _ins_codes(arr)
    keep = np.zeros(len(arr), dtype=bool)
    residue_keys = dict.fromkeys(zip(
        arr.chain_id.astype(str), arr.res_id.astype(int), ins_code
    ))
    for chain, res_id, insertion in residue_keys:
        indices = np.flatnonzero(
            (arr.chain_id == chain) & (arr.res_id == res_id) & (ins_code == insertion)
        )
        residue_names = set(arr.res_name[indices].astype(str))
        if len(residue_names) != 1:
            continue
        residue_name = next(iter(residue_names))
        atom_names = set(arr.atom_name[indices].astype(str))
        if not {"N", "CA", "C"}.issubset(atom_names):
            continue

        if residue_name in CANONICAL_AMINO_ACIDS:
            hetero = (arr.hetero[indices] if "hetero" in arr.get_annotation_categories()
                      else np.zeros(len(indices), dtype=bool))
            if np.all(hetero):
                continue
            keep[indices] = True
        elif residue_name in MODIFIED_AMINO_ACID_PARENTS:
            _, allowed_atoms = MODIFIED_AMINO_ACID_PARENTS[residue_name]
            keep[indices[np.isin(arr.atom_name[indices], list(allowed_atoms))]] = True

    if "element" in arr.get_annotation_categories():
        keep &= np.char.upper(arr.element.astype(str)) != "H"
    out = arr[keep].copy()
    for modified_name, (parent_name, _) in MODIFIED_AMINO_ACID_PARENTS.items():
        out.res_name[out.res_name == modified_name] = parent_name
    if "hetero" in out.get_annotation_categories():
        out.hetero[:] = False
    return out


def protein_heavy_atoms(arr):
    """Backward-compatible alias for complete canonical polymer heavy atoms."""
    return canonical_protein_atoms(arr)


def interface_residue_candidates(
    arr,
    chain_a: str,
    chain_b: str,
    *,
    cutoff: float = 5.0,
) -> pd.DataFrame:
    """Return residues on either chain with any cross-chain heavy-atom contact below cutoff."""
    if cutoff <= 0:
        raise ValueError(f"interface residue cutoff must be positive, got {cutoff}")
    protein = canonical_protein_atoms(arr)
    chain_arrays = {
        chain_a: protein[protein.chain_id == chain_a],
        chain_b: protein[protein.chain_id == chain_b],
    }
    if any(len(chain_arr) == 0 for chain_arr in chain_arrays.values()):
        raise ValueError(f"cannot extract protein chains for interface {chain_a}:{chain_b}")

    atoms_a, atoms_b = chain_arrays[chain_a], chain_arrays[chain_b]
    distances = np.linalg.norm(
        atoms_a.coord[:, None, :] - atoms_b.coord[None, :, :], axis=-1
    )
    atom_minima = {
        chain_a: distances.min(axis=1),
        chain_b: distances.min(axis=0),
    }
    rows = []
    for source_chain, partner_chain, rosetta_chain in (
        (chain_a, chain_b, "A"),
        (chain_b, chain_a, "B"),
    ):
        chain_arr = chain_arrays[source_chain]
        ins_code = _ins_codes(chain_arr)
        residue_keys = dict.fromkeys(zip(chain_arr.res_id.astype(int), ins_code))
        for res_id, insertion in residue_keys:
            mask = (chain_arr.res_id == res_id) & (ins_code == insertion)
            min_distance = float(atom_minima[source_chain][mask].min())
            if min_distance >= cutoff:
                continue
            residue_names = set(chain_arr.res_name[mask].astype(str))
            if len(residue_names) != 1:
                raise ValueError(
                    f"ambiguous residue identity at {source_chain}:{res_id}{insertion}"
                )
            residue_name = next(iter(residue_names))
            rows.append({
                "interface_pair": f"{chain_a}:{chain_b}",
                "source_chain": source_chain,
                "partner_chain": partner_chain,
                "rosetta_chain": rosetta_chain,
                "res_id": int(res_id),
                "ins_code": str(insertion),
                "wt_residue_name3": residue_name,
                "min_cross_chain_heavy_atom_distance_a": min_distance,
                "is_native_alanine": residue_name == "ALA",
                "is_glycine_to_alanine": residue_name == "GLY",
                "is_proline_to_alanine": residue_name == "PRO",
                "is_interpretable_sidechain_alanine": residue_name not in {"ALA", "GLY", "PRO"},
            })
    columns = [
        "interface_pair", "source_chain", "partner_chain", "rosetta_chain", "res_id",
        "ins_code", "wt_residue_name3", "min_cross_chain_heavy_atom_distance_a",
        "is_native_alanine", "is_glycine_to_alanine", "is_proline_to_alanine",
        "is_interpretable_sidechain_alanine",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["rosetta_chain", "res_id", "ins_code"]
    ).reset_index(drop=True)


def _residue_representatives(chain_arr) -> np.ndarray:
    """One CB coordinate per residue (CA for Gly), with alternate atoms de-duplicated."""
    ins_code = (chain_arr.ins_code if "ins_code" in chain_arr.get_annotation_categories()
                else np.full(len(chain_arr), ""))
    wanted = ((chain_arr.atom_name == "CB") |
              ((chain_arr.res_name == "GLY") & (chain_arr.atom_name == "CA")))
    coords = []
    seen = set()
    for i in np.flatnonzero(wanted):
        key = (int(chain_arr.res_id[i]), str(ins_code[i]))
        if key not in seen:
            seen.add(key)
            coords.append(chain_arr.coord[i])
    return np.asarray(coords, dtype=float).reshape((-1, 3))


def pair_residue_contacts(chain_a, chain_b, cutoff: float = 8.0) -> int:
    """Count cross-chain residue pairs whose CB (CA for Gly) distance is below cutoff."""
    ca = _residue_representatives(chain_a)
    cb = _residue_representatives(chain_b)
    if len(ca) == 0 or len(cb) == 0:
        return 0
    d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
    return int((d < cutoff).sum())


def _charged_atom_sites(chain_arr, residue_atoms: dict[str, set[str]]):
    ins_code = (chain_arr.ins_code if "ins_code" in chain_arr.get_annotation_categories()
                else np.full(len(chain_arr), ""))
    sites = []
    for i in range(len(chain_arr)):
        res_name = str(chain_arr.res_name[i])
        if chain_arr.atom_name[i] not in residue_atoms.get(res_name, set()):
            continue
        residue_key = (int(chain_arr.res_id[i]), str(ins_code[i]), res_name)
        sites.append((residue_key, chain_arr.coord[i]))
    return sites


def _opposite_charge_residue_pairs(acid_sites, base_sites, cutoff: float):
    if not acid_sites or not base_sites:
        return set()
    acid_coord = np.asarray([coord for _, coord in acid_sites])
    base_coord = np.asarray([coord for _, coord in base_sites])
    d = np.linalg.norm(acid_coord[:, None, :] - base_coord[None, :, :], axis=-1)
    return {
        (acid_sites[i][0], base_sites[j][0])
        for i, j in np.argwhere(d < cutoff)
    }


def pair_salt_bridges(chain_a, chain_b, cutoff: float = 4.0) -> int:
    """Count unique Asp/Glu--Lys/Arg residue pairs with charged atoms below cutoff.

    Histidine is excluded because its charge state cannot be inferred reliably from predictor
    coordinates. The optional Rosetta layer supplies protonation-aware H-bond/polar metrics.
    """
    acid_atoms = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
    base_atoms = {"LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"}}
    acid_a = _charged_atom_sites(chain_a, acid_atoms)
    acid_b = _charged_atom_sites(chain_b, acid_atoms)
    base_a = _charged_atom_sites(chain_a, base_atoms)
    base_b = _charged_atom_sites(chain_b, base_atoms)
    ab = _opposite_charge_residue_pairs(acid_a, base_b, cutoff)
    ba = _opposite_charge_residue_pairs(acid_b, base_a, cutoff)
    return len(ab) + len(ba)


def _pair_key(chain_a: str, chain_b: str) -> tuple[str, str]:
    return tuple(sorted((str(chain_a), str(chain_b))))


def classify_d2_interface_pairs(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Classify K4 edges into two D2 interface matchings plus the diagonal matching.

    For four chains, the six pair edges form three perfect matchings. Ranking matching-level mean
    BSA is invariant to predictor chain-label permutations and keeps both symmetry copies together.
    """
    chains = sorted(set(pair_df["chain_1"]) | set(pair_df["chain_2"]))
    if len(chains) != 4 or len(pair_df) != 6:
        raise ValueError(
            f"D2 interface classification requires 4 chains/6 pairs, got {chains}/{len(pair_df)}"
        )
    a, b, c, d = chains
    matchings = [
        (_pair_key(a, b), _pair_key(c, d)),
        (_pair_key(a, c), _pair_key(b, d)),
        (_pair_key(a, d), _pair_key(b, c)),
    ]
    bsa_by_pair = {
        _pair_key(r.chain_1, r.chain_2): float(r.bsa_total_a2)
        for r in pair_df.itertuples()
    }
    ranked = []
    for edges in matchings:
        label = "|".join(f"{x}:{y}" for x, y in edges)
        mean_bsa = float(np.mean([bsa_by_pair[edge] for edge in edges]))
        ranked.append((mean_bsa, label, edges))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    out = pair_df.copy()
    out["interface_class"] = ""
    out["interface_class_rank"] = 0
    out["interface_copy"] = 0
    out["d2_matching"] = ""
    out["d2_matching_mean_bsa_total_a2"] = np.nan
    for rank, ((mean_bsa, matching_label, edges), class_name) in enumerate(
        zip(ranked, D2_INTERFACE_CLASSES), start=1
    ):
        for copy, edge in enumerate(sorted(edges), start=1):
            mask = [
                _pair_key(x, y) == edge
                for x, y in zip(out["chain_1"], out["chain_2"])
            ]
            out.loc[mask, "interface_class"] = class_name
            out.loc[mask, "interface_class_rank"] = rank
            out.loc[mask, "interface_copy"] = copy
            out.loc[mask, "d2_matching"] = matching_label
            out.loc[mask, "d2_matching_mean_bsa_total_a2"] = mean_bsa
    out["is_biological_interface"] = out["interface_class"] != "diagonal"
    return out


def classify_hetero_dimer_pairs(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Classify the single edge of a two-chain hetero-complex (e.g. a binder and its target).

    There is no symmetry copy and no diagonal: the one pair IS the biological interface. The D2
    columns are still emitted so the long table keeps one schema across complex topologies, but
    they carry no meaning here and are left empty/NaN rather than filled with a fabricated value.
    """
    if len(pair_df) != 1:
        raise ValueError(
            f"hetero-dimer classification requires exactly one chain pair, got {len(pair_df)}"
        )
    out = pair_df.copy()
    out["interface_class"] = D2_INTERFACE_CLASSES[0]
    out["interface_class_rank"] = 1
    out["interface_copy"] = 1
    out["d2_matching"] = ""
    out["d2_matching_mean_bsa_total_a2"] = np.nan
    out["is_biological_interface"] = True
    return out


def classify_interface_pairs(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Dispatch chain-pair classification on complex topology.

    Four chains -> the uricase D2 homotetramer path (unchanged). Two chains -> a single
    hetero-interface. Anything else has no agreed interface semantics and fails fast rather than
    being silently scored under the wrong topology.
    """
    n_chains = len(set(pair_df["chain_1"]) | set(pair_df["chain_2"]))
    if n_chains == 4:
        return classify_d2_interface_pairs(pair_df)
    if n_chains == 2:
        return classify_hetero_dimer_pairs(pair_df)
    raise ValueError(
        f"interface classification supports two or four protein chains, got {n_chains}"
    )


def interface_pair_metrics(
    arr,
    pchains,
    *,
    sasa_probe_radius: float = 1.4,
    sasa_point_number: int = 1000,
    contact_cutoff: float = 8.0,
    salt_bridge_cutoff: float = 4.0,
) -> pd.DataFrame:
    """Coordinate-only interface metrics for every chain pair of the complex.

    Four chains gives the six uricase tetramer pairs; two chains gives the single
    binder:target pair.
    """
    import biotite.structure as struc

    if len(pchains) not in (2, 4):
        raise ValueError(
            f"expected two or four protein chains, got {pchains}"
        )
    protein = protein_heavy_atoms(arr)
    chain_atoms = {ch: protein[protein.chain_id == ch] for ch in sorted(pchains)}
    chain_sasa = {
        ch: float(np.nansum(struc.sasa(sub, probe_radius=sasa_probe_radius,
                                      point_number=sasa_point_number)))
        for ch, sub in chain_atoms.items()
    }
    rows = []
    chains = sorted(chain_atoms)
    for i, chain_a in enumerate(chains):
        for chain_b in chains[i + 1:]:
            arr_a, arr_b = chain_atoms[chain_a], chain_atoms[chain_b]
            pair = arr_a + arr_b
            pair_sasa = float(np.nansum(struc.sasa(
                pair, probe_radius=sasa_probe_radius, point_number=sasa_point_number
            )))
            bsa_total = max(0.0, chain_sasa[chain_a] + chain_sasa[chain_b] - pair_sasa)
            rows.append({
                "chain_1": chain_a,
                "chain_2": chain_b,
                "chain_pair": f"{chain_a}:{chain_b}",
                "bsa_total_a2": bsa_total,
                "bsa_per_partner_a2": bsa_total / 2.0,
                "n_residue_contacts_8a": pair_residue_contacts(
                    arr_a, arr_b, cutoff=contact_cutoff
                ),
                "n_salt_bridges_4a": pair_salt_bridges(
                    arr_a, arr_b, cutoff=salt_bridge_cutoff
                ),
            })
    return classify_interface_pairs(pd.DataFrame(rows))


def n_interchain_contacts(arr, pchains, cutoff=8.0) -> int:
    """Total pairwise residue contacts across all protein chains."""
    protein = protein_heavy_atoms(arr)
    n = 0
    for i, chain_a in enumerate(pchains):
        for chain_b in pchains[i + 1:]:
            n += pair_residue_contacts(
                protein[protein.chain_id == chain_a],
                protein[protein.chain_id == chain_b],
                cutoff=cutoff,
            )
    return n


SUMMARY_METRICS = (
    ("bsa_total_a2", "bsa_total", "min", "a2"),
    ("bsa_per_partner_a2", "bsa_per_partner", "min", "a2"),
    ("n_residue_contacts_8a", "n_residue_contacts", "min", "8a"),
    ("n_salt_bridges_4a", "n_salt_bridges", "min", "4a"),
    ("rosetta_dsasa_int_a2", "rosetta_dsasa_int", "min", "a2"),
    ("rosetta_dg_separated_reu", "rosetta_dg_separated", "max", "reu"),
    ("rosetta_dg_per_sasa_x100", "rosetta_dg_per_sasa_x100", "max", ""),
    ("rosetta_dg_per_sasa_reu_per_a2", "rosetta_dg_per_sasa", "max", "reu_per_a2"),
    ("rosetta_packstat", "rosetta_packstat", "min", ""),
    ("rosetta_hbonds_int", "rosetta_hbonds_int", "min", ""),
    ("rosetta_hbond_energy_fraction", "rosetta_hbond_energy_fraction", "min", ""),
    ("rosetta_delta_unsat_hbonds", "rosetta_delta_unsat_hbonds", "max", ""),
    ("rosetta_shape_complementarity", "rosetta_shape_complementarity", "min", ""),
    ("rosetta_nres_int", "rosetta_nres_int", "min", ""),
)


def _summary_column(prefix: str, stem: str, stat: str, unit: str) -> str:
    suffix = f"_{unit}" if unit else ""
    return f"{prefix}_{stem}_{stat}{suffix}"


def summarize_interface_pairs(pair_df: pd.DataFrame) -> dict:
    """Return mean and conservative summaries while retaining copy rows in the long table.

    A D2 tetramer (a diagonal class is present) keeps the strict two-symmetry-copy contract and
    the topology-separation block. A hetero-dimer has one interface and no diagonal, so the
    'conservative copy' statistic degenerates to the single measurement and the topology block is
    omitted rather than computed against an absent baseline.
    """
    rec = {}
    is_d2 = bool((pair_df["interface_class"] == "diagonal").any())
    for class_name in D2_INTERFACE_CLASSES[:2]:
        sub = pair_df[pair_df["interface_class"] == class_name]
        if is_d2:
            if len(sub) != 2:
                raise ValueError(f"{class_name} must contain two symmetry-related chain pairs")
        elif sub.empty:
            continue
        for source, stem, conservative_stat, unit in SUMMARY_METRICS:
            if source not in sub.columns:
                continue
            values = pd.to_numeric(sub[source], errors="coerce")
            rec[_summary_column(class_name, stem, "mean", unit)] = float(values.mean())
            rec[_summary_column(class_name, stem, conservative_stat, unit)] = float(
                getattr(values, conservative_stat)()
            )
    if not is_d2:
        return rec

    diagonal = pair_df[pair_df["interface_class"] == "diagonal"]
    for source, stem, _, unit in SUMMARY_METRICS[:4]:
        values = pd.to_numeric(diagonal[source], errors="coerce")
        rec[_summary_column("diagonal", stem, "mean", unit)] = float(values.mean())
        rec[_summary_column("diagonal", stem, "max", unit)] = float(values.max())
    interface_2_min = rec["interface_2_bsa_total_min_a2"]
    diagonal_max = rec["diagonal_bsa_total_max_a2"]
    rec["interface_topology_bsa_gap_a2"] = interface_2_min - diagonal_max
    rec["interface_topology_bsa_ratio"] = (
        interface_2_min / diagonal_max if diagonal_max > 0 else np.inf
    )
    return rec


def write_rosetta_dimer_pdb(arr, chain_a: str, chain_b: str, out_path: Path) -> None:
    """Write one protein-only interface dimer with Rosetta-safe chain IDs A and B."""
    from biotite.structure.io.pdb import PDBFile

    protein = protein_heavy_atoms(arr)
    dimer = protein[(protein.chain_id == chain_a) | (protein.chain_id == chain_b)].copy()
    if not np.any(dimer.chain_id == chain_a) or not np.any(dimer.chain_id == chain_b):
        raise ValueError(f"cannot extract dimer {chain_a}:{chain_b}")
    original_chain_ids = dimer.chain_id.copy()
    dimer.chain_id[original_chain_ids == chain_a] = "A"
    dimer.chain_id[original_chain_ids == chain_b] = "B"
    if "hetero" in dimer.get_annotation_categories():
        dimer.hetero[:] = False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_file = PDBFile()
    pdb_file.set_structure(dimer)
    pdb_file.write(str(out_path))


def parse_rosetta_scorefile(score_path: Path) -> pd.DataFrame:
    """Parse a Rosetta whitespace scorefile without altering slash-containing field names."""
    header = None
    records = []
    for line in score_path.read_text().splitlines():
        if not line.startswith("SCORE:"):
            continue
        fields = line.split()[1:]
        if "description" in fields:
            header = fields
            continue
        if header is not None and len(fields) == len(header):
            records.append(dict(zip(header, fields)))
    if header is None or not records:
        raise ValueError(f"no score records found in Rosetta scorefile: {score_path}")
    df = pd.DataFrame(records)
    for col in df.columns:
        if col != "description":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


_DDG_SCAN_WT_RE = re.compile(
    r"wild-type binding ddG\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
_DDG_SCAN_ROW_RE = re.compile(
    r"Residue\s+(?P<chain>\S)(?P<res_id>-?\d+)"
    r"(?P<wt>[A-Z0-9]{3})->(?P<mut>[A-Z0-9]{3})\s*:\s*"
    r"(?P<ddg>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def parse_rosetta_ddg_scan_log(text: str) -> tuple[float, pd.DataFrame]:
    """Parse a Rosetta DdGScan report and return WT binding dG plus mutation deltas."""
    wt_values = [float(value) for value in _DDG_SCAN_WT_RE.findall(text)]
    if not wt_values:
        raise ValueError("Rosetta DdGScan log is missing the wild-type binding ddG")
    if not np.allclose(wt_values, wt_values[0], atol=1e-6, rtol=0):
        raise ValueError(f"Rosetta DdGScan log has inconsistent WT binding ddGs: {wt_values}")

    rows = [
        {
            "rosetta_chain": match.group("chain"),
            "res_id": int(match.group("res_id")),
            "wt_residue_name3": match.group("wt"),
            "mut_residue_name3": match.group("mut"),
            "ddg_bind_reu": float(match.group("ddg")),
        }
        for match in _DDG_SCAN_ROW_RE.finditer(text)
    ]
    if not rows:
        raise ValueError("Rosetta DdGScan log contains no per-residue mutation records")
    out = pd.DataFrame(rows)
    if out.duplicated(["rosetta_chain", "res_id"]).any():
        duplicate = out[out.duplicated(["rosetta_chain", "res_id"], keep=False)]
        raise ValueError(
            "Rosetta DdGScan emitted duplicate residue rows: "
            f"{duplicate[['rosetta_chain', 'res_id']].to_dict('records')}"
        )
    return wt_values[0], out


def _reconcile_rosetta_alanine_rows(
    arr,
    candidates: pd.DataFrame,
    parsed: pd.DataFrame,
    *,
    chain_a: str,
    chain_b: str,
    disulfide_cutoff: float = 2.5,
) -> pd.DataFrame:
    """Retain Rosetta-omitted covalent disulfide Cys as noninterpretable rows."""
    key = ["rosetta_chain", "res_id"]
    expected = set(map(tuple, candidates[key].to_numpy()))
    observed = set(map(tuple, parsed[key].to_numpy()))
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if extra:
        raise ValueError(
            f"Rosetta alanine-scan residue mismatch for {chain_a}:{chain_b}; "
            f"missing={missing}, extra={extra}"
        )

    disulfide_keys: set[tuple[str, int]] = set()
    if missing:
        protein = canonical_protein_atoms(arr)
        dimer = protein[
            np.isin(protein.chain_id.astype(str), [str(chain_a), str(chain_b)])
        ]
        insertions = _ins_codes(dimer)
        cysteine_sg = (dimer.res_name.astype(str) == "CYS") & (
            dimer.atom_name.astype(str) == "SG"
        )
        for rosetta_chain, res_id in missing:
            source_chain = chain_a if str(rosetta_chain) == "A" else chain_b
            candidate = candidates[
                candidates["rosetta_chain"].astype(str).eq(str(rosetta_chain))
                & candidates["res_id"].astype(int).eq(int(res_id))
            ]
            if len(candidate) != 1 or str(candidate.iloc[0]["wt_residue_name3"]) != "CYS":
                raise ValueError(
                    f"Rosetta alanine-scan residue mismatch for {chain_a}:{chain_b}; "
                    f"missing={missing}, extra={extra}"
                )
            insertion = str(candidate.iloc[0].get("ins_code", ""))
            target = (
                cysteine_sg
                & (dimer.chain_id.astype(str) == str(source_chain))
                & (dimer.res_id.astype(int) == int(res_id))
                & (insertions.astype(str) == insertion)
            )
            other = cysteine_sg & ~target
            if not target.any() or not other.any():
                raise ValueError(
                    f"Rosetta alanine-scan residue mismatch for {chain_a}:{chain_b}; "
                    f"missing={missing}, extra={extra}"
                )
            min_distance = float(
                np.linalg.norm(
                    dimer.coord[target][:, None, :] - dimer.coord[other][None, :, :],
                    axis=-1,
                ).min()
            )
            if min_distance > disulfide_cutoff:
                raise ValueError(
                    f"Rosetta alanine-scan residue mismatch for {chain_a}:{chain_b}; "
                    f"missing={missing}, extra={extra}"
                )
            disulfide_keys.add((str(rosetta_chain), int(res_id)))

    merged = candidates.merge(
        parsed,
        on=key,
        how="left",
        validate="one_to_one",
        suffixes=("", "_rosetta"),
    )
    merged["is_disulfide_cysteine"] = [
        (str(row.rosetta_chain), int(row.res_id)) in disulfide_keys
        for row in merged.itertuples(index=False)
    ]
    disulfide = merged["is_disulfide_cysteine"]
    merged.loc[disulfide, "wt_residue_name3_rosetta"] = "CYS"
    merged.loc[disulfide, "mut_residue_name3"] = "ALA"
    merged.loc[disulfide, "ddg_bind_reu"] = np.nan
    merged.loc[disulfide, "is_interpretable_sidechain_alanine"] = False
    mismatch = merged[
        merged["wt_residue_name3"] != merged["wt_residue_name3_rosetta"]
    ]
    if not mismatch.empty:
        columns = [
            "source_chain", "res_id", "wt_residue_name3", "wt_residue_name3_rosetta"
        ]
        raise ValueError(
            f"Rosetta residue identity mismatch for {chain_a}:{chain_b}: "
            f"{mismatch[columns].to_dict('records')}"
        )
    return merged.drop(columns=["wt_residue_name3_rosetta"])


def aggregate_homomer_alanine_scan(scan_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate side-specific homomer alanine scans to one row per interface/residue index."""
    scan_df = scan_df.copy()
    if "is_disulfide_cysteine" not in scan_df:
        scan_df["is_disulfide_cysteine"] = False
    required = {
        "interface_pair", "source_chain", "res_id", "ins_code", "wt_residue_name3",
        "ddg_bind_reu", "is_native_alanine", "is_glycine_to_alanine",
        "is_proline_to_alanine", "is_disulfide_cysteine",
        "is_interpretable_sidechain_alanine",
    }
    missing = sorted(required - set(scan_df.columns))
    if missing:
        raise ValueError(f"alanine-scan table is missing required columns: {missing}")
    group_columns = ["interface_pair", "res_id", "ins_code", "wt_residue_name3"]
    grouped = scan_df.groupby(group_columns, dropna=False, sort=True)
    out = grouped.agg(
        source_chains=("source_chain", lambda values: ",".join(sorted(set(map(str, values))))),
        n_chain_sides=("source_chain", "nunique"),
        ddg_bind_mean_reu=("ddg_bind_reu", "mean"),
        ddg_bind_min_reu=("ddg_bind_reu", "min"),
        ddg_bind_max_reu=("ddg_bind_reu", "max"),
        is_native_alanine=("is_native_alanine", "all"),
        is_glycine_to_alanine=("is_glycine_to_alanine", "all"),
        is_proline_to_alanine=("is_proline_to_alanine", "all"),
        is_disulfide_cysteine=("is_disulfide_cysteine", "any"),
        is_interpretable_sidechain_alanine=("is_interpretable_sidechain_alanine", "all"),
    ).reset_index()
    out["ddg_bind_chain_range_reu"] = (
        out["ddg_bind_max_reu"] - out["ddg_bind_min_reu"]
    )
    if "min_cross_chain_heavy_atom_distance_a" in scan_df.columns:
        min_distance = grouped["min_cross_chain_heavy_atom_distance_a"].min().reset_index(
            name="min_cross_chain_heavy_atom_distance_a"
        )
        out = out.merge(min_distance, on=group_columns, how="left", validate="one_to_one")
    for optional in ("interface_class", "rosetta_score_function"):
        if optional in scan_df.columns:
            values = grouped[optional].first().reset_index(name=optional)
            out = out.merge(values, on=group_columns, how="left", validate="one_to_one")
    out["ddg_bind_rank_desc"] = out.groupby("interface_pair")[
        "ddg_bind_mean_reu"
    ].rank(method="min", ascending=False).astype("Int64")
    out["sidechain_ddg_bind_rank_desc"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    interpretable = out["is_interpretable_sidechain_alanine"]
    out.loc[interpretable, "sidechain_ddg_bind_rank_desc"] = (
        out.loc[interpretable]
        .groupby("interface_pair")["ddg_bind_mean_reu"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    return out.sort_values(
        ["interface_pair", "res_id", "ins_code"]
    ).reset_index(drop=True)


def _resolve_executable(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    path = Path(executable)
    if path.is_file() and os.access(path, os.X_OK):
        return str(path.resolve())
    raise FileNotFoundError(f"executable not found or not executable: {executable}")


def _write_ddg_scan_protocol(
    protocol_path: Path,
    *,
    resfile_path: Path,
    score_function: str,
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", score_function):
        raise ValueError(f"invalid Rosetta score-function name: {score_function!r}")
    escaped_resfile = xml_escape(str(resfile_path.resolve()), {'"': "&quot;"})
    escaped_score = xml_escape(score_function, {'"': "&quot;"})
    protocol_path.write_text(
        "<ROSETTASCRIPTS>\n"
        "  <SCOREFXNS>\n"
        f"    <ScoreFunction name=\"scan_score\" weights=\"{escaped_score}\"/>\n"
        "  </SCOREFXNS>\n"
        "  <TASKOPERATIONS>\n"
        f"    <ReadResfile name=\"scan_positions\" filename=\"{escaped_resfile}\"/>\n"
        "  </TASKOPERATIONS>\n"
        "  <MOVERS>\n"
        "    <ddG name=\"binding_dg\" scorefxn=\"scan_score\" chain_num=\"2\"\n"
        "         repack_unbound=\"false\" repack_bound=\"false\"\n"
        "         relax_unbound=\"false\" relax_bound=\"false\"/>\n"
        "  </MOVERS>\n"
        "  <FILTERS>\n"
        "    <DdGScan name=\"alanine_scan\" task_operations=\"scan_positions\"\n"
        "             repeats=\"1\" scorefxn=\"scan_score\" report_diffs=\"1\"\n"
        "             write2pdb=\"0\" ddG_mover=\"binding_dg\"/>\n"
        "  </FILTERS>\n"
        "  <PROTOCOLS>\n"
        "    <Add filter_name=\"alanine_scan\"/>\n"
        "  </PROTOCOLS>\n"
        "  <OUTPUT scorefxn=\"scan_score\"/>\n"
        "</ROSETTASCRIPTS>\n"
    )


def run_rosetta_alanine_scan(
    arr,
    chain_pairs,
    *,
    executable: str,
    run_dir: Path,
    log_dir: Path,
    timeout_seconds: int,
    interface_cutoff: float = 5.0,
    score_function: str = "ref2015",
    seed: int = 1729,
    native_alanine_tolerance: float = 1e-4,
) -> pd.DataFrame:
    """Run no-repack Rosetta DdGScan for selected dimer interfaces.

    ``ddg_bind_reu`` is ``binding_dG(mutant) - binding_dG(WT)``. Positive values therefore mean
    the alanine mutation weakens binding. Only residues with a cross-chain heavy-atom distance
    below ``interface_cutoff`` are scanned. Native Ala->Ala rows are retained as zero controls.
    Rosetta inputs, protocols, commands, and scorefiles are written below ``run_dir``; captured
    stdout/stderr are written below the separate ``log_dir``.
    """
    if timeout_seconds <= 0:
        raise ValueError("Rosetta alanine-scan timeout must be positive")
    resolved_executable = _resolve_executable(executable)
    run_dir = Path(run_dir)
    log_dir = Path(log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for pair_index, pair in enumerate(chain_pairs):
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError(f"invalid interface chain pair: {pair!r}")
        chain_a, chain_b = map(str, pair)
        pair_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{chain_a}{chain_b}")
        pair_dir = run_dir / f"{pair_index:02d}_{pair_slug}"
        pair_log_dir = log_dir / f"{pair_index:02d}_{pair_slug}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair_log_dir.mkdir(parents=True, exist_ok=True)

        candidates = interface_residue_candidates(
            arr, chain_a, chain_b, cutoff=interface_cutoff
        )
        if candidates.empty:
            raise ValueError(
                f"no interface residues found for {chain_a}:{chain_b} within "
                f"{interface_cutoff:.3f} A"
            )
        nonempty_insertions = candidates[candidates["ins_code"].astype(str) != ""]
        if not nonempty_insertions.empty:
            raise ValueError(
                "Rosetta alanine scan does not yet support PDB insertion codes: "
                f"{nonempty_insertions[['source_chain', 'res_id', 'ins_code']].to_dict('records')}"
            )

        input_path = pair_dir / "input_dimer.pdb"
        resfile_path = pair_dir / "scan.resfile"
        protocol_path = pair_dir / "ddg_scan.xml"
        score_path = pair_dir / "ddg_scan.sc"
        stdout_path = pair_log_dir / "ddg_scan.stdout.log"
        stderr_path = pair_log_dir / "ddg_scan.stderr.log"
        command_path = pair_dir / "ddg_scan.command.txt"
        write_rosetta_dimer_pdb(arr, chain_a, chain_b, input_path)

        resfile_lines = ["NATRO", "start"]
        resfile_lines.extend(
            f"{row.res_id} {row.rosetta_chain} PIKAA A"
            for row in candidates.itertuples()
        )
        resfile_path.write_text("\n".join(resfile_lines) + "\n")
        _write_ddg_scan_protocol(
            protocol_path, resfile_path=resfile_path, score_function=score_function
        )
        score_path.unlink(missing_ok=True)

        cmd = [
            resolved_executable,
            "-s", str(input_path.resolve()),
            "-parser:protocol", str(protocol_path.resolve()),
            "-nstruct", "1",
            "-jd2:no_output", "true",
            "-constant_seed",
            "-jran", str(int(seed)),
            "-ignore_unrecognized_res", "true",
            "-overwrite",
            "-out:file:score_only", str(score_path.resolve()),
        ]
        command_path.write_text(shlex.join(cmd) + "\n")
        try:
            result = subprocess.run(
                cmd, cwd=pair_dir, capture_output=True, text=True, timeout=timeout_seconds
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Rosetta alanine scan timed out after {timeout_seconds}s; run dir: {pair_dir}"
            ) from exc
        stdout_path.write_text(result.stdout)
        stderr_path.write_text(result.stderr)
        if result.returncode != 0:
            tail = "\n".join((result.stderr or result.stdout).splitlines()[-30:])
            raise RuntimeError(
                f"Rosetta alanine scan failed with exit code {result.returncode}; "
                f"run dir: {pair_dir}; log dir: {pair_log_dir}\n{tail}"
            )

        wt_binding_dg, parsed = parse_rosetta_ddg_scan_log(result.stdout)
        merged = _reconcile_rosetta_alanine_rows(
            arr,
            candidates,
            parsed,
            chain_a=chain_a,
            chain_b=chain_b,
        )
        merged["wt_binding_dg_reu"] = wt_binding_dg
        merged["mut_binding_dg_reu"] = wt_binding_dg + merged["ddg_bind_reu"]
        merged["ddg_sign_convention"] = "mutant_minus_wildtype"
        merged["rosetta_score_function"] = score_function
        merged["repack_bound"] = False
        merged["repack_unbound"] = False
        merged["interface_residue_cutoff_a"] = float(interface_cutoff)

        native_ala = merged[merged["is_native_alanine"]]
        if not native_ala.empty:
            bad_controls = native_ala[
                native_ala["ddg_bind_reu"].abs() > native_alanine_tolerance
            ]
            if not bad_controls.empty:
                raise RuntimeError(
                    "Rosetta DdGScan failed the native Ala->Ala zero control: "
                    f"{bad_controls[['source_chain', 'res_id', 'ddg_bind_reu']].to_dict('records')}"
                )
        all_results.append(merged)

    if not all_results:
        raise ValueError("at least one interface chain pair is required for alanine scanning")
    return pd.concat(all_results, ignore_index=True)


def _description_to_job_id(description: str, valid_job_ids: set[str]) -> str | None:
    stem = Path(str(description)).stem
    if stem in valid_job_ids:
        return stem
    matches = [job_id for job_id in valid_job_ids if stem.startswith(f"{job_id}_")]
    return matches[0] if len(matches) == 1 else None


def run_rosetta_interface_analyzer(
    pair_df: pd.DataFrame,
    arr_loader,
    *,
    executable: str,
    run_dir: Path,
    log_dir: Path,
    timeout_seconds: int,
) -> pd.DataFrame:
    """Run one InterfaceAnalyzer batch with runtime artifacts and logs kept separate."""
    interface_rows = pair_df[pair_df["is_biological_interface"]].copy()
    if interface_rows.empty:
        return pair_df

    run_dir = Path(run_dir)
    log_dir = Path(log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    input_dir = run_dir / "dimers"
    input_dir.mkdir(parents=True, exist_ok=True)
    jobs = {}
    input_paths = []
    for serial, (row_index, row) in enumerate(interface_rows.iterrows()):
        job_id = f"ifc_{serial:06d}"
        pdb_path = input_dir / f"{job_id}.pdb"
        write_rosetta_dimer_pdb(
            arr_loader(row["name"]), row["chain_1"], row["chain_2"], pdb_path
        )
        jobs[job_id] = row_index
        input_paths.append(pdb_path.resolve())

    list_path = run_dir / "interface_dimers.list"
    score_path = run_dir / "interface_analyzer.sc"
    stdout_path = log_dir / "interface_analyzer.stdout.log"
    stderr_path = log_dir / "interface_analyzer.stderr.log"
    list_path.write_text("\n".join(map(str, input_paths)) + "\n")
    score_path.unlink(missing_ok=True)

    cmd = [
        _resolve_executable(executable),
        "-l", str(list_path.resolve()),
        "-interface", "A_B",
        "-compute_packstat", "true",
        "-compute_interface_sc", "true",
        "-pack_input", "false",
        "-pack_separated", "false",
        "-out:file:score_only", str(score_path.resolve()),
        "-jd2:no_output", "true",
        "-constant_seed",
        "-jran", "1729",
        "-ignore_unrecognized_res", "true",
        "-overwrite",
    ]
    (run_dir / "interface_analyzer.command.txt").write_text(shlex.join(cmd) + "\n")
    try:
        result = subprocess.run(
            cmd, cwd=run_dir, capture_output=True, text=True, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Rosetta InterfaceAnalyzer timed out after {timeout_seconds}s; run dir: {run_dir}"
        ) from exc
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-30:])
        raise RuntimeError(
            f"Rosetta InterfaceAnalyzer failed with exit code {result.returncode}; "
            f"run dir: {run_dir}; log dir: {log_dir}\n{tail}"
        )

    scores = parse_rosetta_scorefile(score_path)
    missing_fields = sorted(ROSETTA_REQUIRED_FIELDS - set(scores.columns))
    if missing_fields:
        raise ValueError(f"Rosetta scorefile is missing required fields: {missing_fields}")
    invalid_fields = sorted(
        field for field in ROSETTA_REQUIRED_FIELDS if scores[field].isna().any()
    )
    if invalid_fields:
        raise ValueError(f"Rosetta scorefile has non-numeric required fields: {invalid_fields}")
    valid_job_ids = set(jobs)
    scores["rosetta_job_id"] = scores["description"].map(
        lambda value: _description_to_job_id(value, valid_job_ids)
    )
    if scores["rosetta_job_id"].isna().any():
        unknown = scores.loc[scores["rosetta_job_id"].isna(), "description"].tolist()
        raise ValueError(f"cannot map Rosetta score descriptions to dimers: {unknown}")
    if scores["rosetta_job_id"].duplicated().any():
        raise ValueError("Rosetta emitted duplicate score rows for an interface dimer")
    if set(scores["rosetta_job_id"]) != valid_job_ids:
        missing_jobs = sorted(valid_job_ids - set(scores["rosetta_job_id"]))
        raise ValueError(f"Rosetta did not emit scores for dimers: {missing_jobs}")

    available_fields = [field for field in ROSETTA_SCORE_FIELDS if field in scores.columns]
    metrics = scores[["rosetta_job_id", *available_fields]].rename(columns=ROSETTA_SCORE_FIELDS)
    metrics["rosetta_dg_per_sasa_reu_per_a2"] = (
        metrics["rosetta_dg_per_sasa_x100"] / 100.0
    )
    out = pair_df.copy()
    out["rosetta_job_id"] = None
    for job_id, row_index in jobs.items():
        out.loc[row_index, "rosetta_job_id"] = job_id
    return out.merge(metrics, on="rosetta_job_id", how="left", validate="many_to_one")
