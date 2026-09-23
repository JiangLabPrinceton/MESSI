#!/usr/bin/env python
"""Post-prediction complex-gate metrics for predicted protein assemblies.

Second-layer benchmark supplement. Two complex topologies share this driver:

  * **D2 homotetramer** (uricase): does a design fold BACK into the native tetramer with its
    inter-protomer catalytic pocket intact?
  * **hetero-dimer** (a de novo binder and its target): does a de-immunized design still form
    the parent's interface? A binder's function is only readable on a COMPLEX -- an apo monomer
    refold places its interface side chains with no partner to order them, so it cannot testify
    about binding (PROTOCOL/binder_interface_deimm_selection.md).

Two tiers + advisory, one row per variant:

  Tier 1 (self-confidence, from the Protenix summary_confidence JSON; cheap pre-filter):
    iptm, ptm, plddt, min protein-protein chain_pair_iptm, min protein chain_plddt,
    mean protein-protein chain_pair_gpde (inter-chain PDE), has_clash, ranking_score.
  Tier 2 (geometry vs a reference tetramer; the actual "refold back"):
    complex_TM  — US-align -mm 1 (symmetry-aware) vs the parent WT-predicted tetramer, and,
                  for the crystal parent (Q00511), vs the 1R51 crystal tetramer.
    xprot_as_dist / _dev — cross-protomer active-site distances (same-chain catalytic core
                  Lys11/Thr58/His257 to the NEAREST neighbour-chain Asn255 W1), and the
                  deviation vs the WT-predicted reference. Only for parents whose numbering
                  matches the Q00511 manifest (index_0b); NaN otherwise (per-parent projection
                  is a v1 extension).
  Assembly (all parents): a tetramer's six chain-pair rows are split into the two repeated
                  D2-interface classes and the diagonal non-interface; a hetero-dimer's single
                  pair IS the interface. Coordinate-only metrics include buried surface area
                  (BSA), residue contacts, and salt bridges.
  Rosetta (optional): InterfaceAnalyzer scores every biological-interface dimer (four for a
                  tetramer, one for a hetero-dimer),
                  adding dSASA, binding dG and dG/dSASA, packstat, interface H-bonds,
                  H-bond energy fraction, shape complementarity, and buried unsatisfied
                  H-bonds. Rosetta runs after prediction and never inside the GPU predictor.

Reference model: each design is scored vs its OWN parent's WT-predicted tetramer (relative), so
the gate cancels systematic model bias. Q00511 additionally gets the absolute crystal check.

The gate is NOT hard-coded here: thresholds are calibrated on same-backend WT predictions and
experimental outcomes. The main output is one row per variant; a companion long table preserves
every chain pair so downstream selection never depends only on lossy aggregate values.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.evaluation.ligand_pocket import (
    CrystalPocketReference,
    ligand_site_metrics,
)
from inverse_folding.evaluation.complex_interfaces import (
    interface_pair_metrics,
    run_rosetta_interface_analyzer,
    summarize_interface_pairs,
)

# Q00511 catalytic core (same-chain) + cross-protomer W1, as manifest index_0b (== predicted res_id).
CATALYTIC_CORE = {"Lys11": 10, "Thr58": 57, "His257": 256}
XPROT_W1 = {"Asn255": 254}
# functional atoms for the distance panel (fallback to CA if absent)
FUNC_ATOM = {"Lys11": "NZ", "Thr58": "OG1", "His257": "NE2", "Asn255": "ND2"}
# parents whose residue numbering matches the Q00511 manifest exactly
Q00511_NUMBERING_PARENTS = {"Q00511"}


def load_catalytic_map(path: str) -> dict[str, dict[str, int]]:
    """Load per-parent catalytic residue indices: parent -> {label: index_0b}.

    Accepts a long table (parquet/csv) with columns ``parent``, ``label``, ``index_0b``.
    Labels use the Q00511 nomenclature (``Lys11``/``Thr58``/``His257``/``Asn255``) so a parent's
    own index can be substituted positionally; the Active-15 constraint manifests emit exactly
    this under ``biological_role: functional_analog_lock``.

    A duplicated (parent, label) is a hard error: silently keeping one of two candidate indices
    is how a pocket measurement ends up on the wrong residue.
    """
    p = Path(path)
    table = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    missing = {"parent", "label", "index_0b"} - set(table.columns)
    if missing:
        raise ValueError(f"catalytic map is missing columns: {sorted(missing)}")
    dup = table.duplicated(subset=["parent", "label"], keep=False)
    if dup.any():
        offenders = table[dup][["parent", "label"]].drop_duplicates().to_dict("records")
        raise ValueError(f"catalytic map has duplicate (parent,label) rows: {offenders[:5]}")
    out: dict[str, dict[str, int]] = {}
    for r in table.itertuples():
        out.setdefault(str(r.parent), {})[str(r.label)] = int(r.index_0b)
    return out


def resolve_catalytic_residues(
    parent: str, catalytic_map: Mapping[str, Mapping[str, int]] | None
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (core_residues, partner_residues) in this PARENT's own numbering.

    Without a map only Q00511 resolves, and it resolves to the legacy hardcoded panel so the
    historical behaviour is byte-identical. Every other parent returns empty, which makes the
    caller SKIP the catalytic/ligand-distance panel instead of measuring it against Q00511's
    indices. That distinction matters: measured on the Active-15 manifests, 13 of 15 parents
    put Lys11/Thr58/His257/Asn255 at different indices (A0A9P8P4R1 is off by 26), and the old
    unconditional call emitted plausible-looking distances to the wrong residues.
    """
    if catalytic_map and parent in catalytic_map:
        m = catalytic_map[parent]
        core = {k: int(m[k]) for k in CATALYTIC_CORE if k in m}
        partner = {k: int(m[k]) for k in XPROT_W1 if k in m}
        return core, partner
    if parent in Q00511_NUMBERING_PARENTS:
        return dict(CATALYTIC_CORE), dict(XPROT_W1)
    return {}, {}
# NOTE: manifest index_0b == crystal resSeq == predicted res_id in the NATIVE mature frame, so the
# residue panels below are shifted by --crystal-offset (0 = native/mature, 1 = legacy leading-Met).


# ---------- Protenix output discovery ----------

def find_pred_samples(pred_root: Path, name: str):
    """Return list of (sample_idx, cif_path, summary_json_path) for a variant, or []."""
    hits = sorted((pred_root / name).glob(f"*/seed_*/predictions"))
    out = []
    for pdir in hits:
        for cif in sorted(pdir.glob(f"{name}_sample_*.cif")):
            m = re.search(r"_sample_(\d+)\.cif$", cif.name)
            if not m:
                continue
            k = int(m.group(1))
            js = pdir / f"{name}_summary_confidence_sample_{k}.json"
            if js.exists():
                out.append((k, cif, js))
    return out


def tier1_from_summary(j: dict, n_protein: int = 4) -> dict:
    """Self-confidence metrics; protein chains are the first n_protein entries (A..D before ligands).

    n_protein is clamped to the matrix size so a complex with fewer chains than the tetramer
    default (e.g. a two-chain binder:target) reads its real protein block instead of raising.
    """
    cpi = np.asarray(j.get("chain_pair_iptm", []), dtype=float)
    cpg = np.asarray(j.get("chain_pair_gpde", []), dtype=float)
    cpl = np.asarray(j.get("chain_plddt", []), dtype=float)
    n_chain = next((int(m.shape[0]) for m in (cpi, cpg) if m.ndim == 2 and m.shape[0]), None)
    if n_chain is None and cpl.size:
        n_chain = int(cpl.size)
    pp = list(range(min(n_protein, n_chain) if n_chain else n_protein))
    prot_pairs = [(i, k) for i in pp for k in pp if i < k]
    min_cpi = float(min(cpi[i, k] for i, k in prot_pairs)) if cpi.size else np.nan
    mean_cpg = float(np.mean([cpg[i, k] for i, k in prot_pairs])) if cpg.size else np.nan
    min_cpl = float(np.min(cpl[pp])) if cpl.size else np.nan
    return dict(
        iptm=float(j.get("iptm", np.nan)), ptm=float(j.get("ptm", np.nan)),
        plddt=float(j.get("plddt", np.nan)), ranking_score=float(j.get("ranking_score", np.nan)),
        has_clash=bool(j.get("has_clash", False)),
        min_pp_chain_pair_iptm=min_cpi, mean_pp_chain_pair_gpde=mean_cpg,
        min_prot_chain_plddt=min_cpl,
    )


def pae_metrics_from_summary(j: dict, n_protein: int = 4) -> dict:
    """Chain-pair PAE/PDE/ipTM blocks from the summary-confidence JSON.

    Protenix writes chain_pair_gpde / chain_pair_iptm / chain_pair_iptm_global / chain_pair_plddt
    as (n_chain, n_chain) matrices covering protein AND ligand entities, so the protein-ligand
    off-diagonal blocks are available without the huge per-token full-data dump (that dump, from
    `pred --need_atom_confidence true`, is only needed for per-token PAE maps). Protein chains are
    the first n_protein entries; any remaining entries are ligand copies.
    """
    out: dict = {}
    n_chain = None
    mats = {}
    for key, mat_name in (("chain_pair_gpde", "gpde"), ("chain_pair_iptm", "iptm"),
                          ("chain_pair_iptm_global", "iptm_global"),
                          ("chain_pair_plddt", "plddt")):
        v = j.get(key)
        if v is None:
            continue
        m = np.asarray(v, dtype=float)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            continue
        mats[mat_name] = m
        n_chain = m.shape[0]
    if n_chain is None:
        return out
    prot = list(range(min(n_protein, n_chain)))
    lig = list(range(len(prot), n_chain))
    pp = [(i, k) for i in prot for k in prot if i < k]
    pl = [(i, k) for i in prot for k in lig]
    for name, m in mats.items():
        if pp:
            vals = np.array([m[i, k] for i, k in pp], dtype=float)
            out[f"pp_chain_pair_{name}_mean"] = float(np.nanmean(vals))
            out[f"pp_chain_pair_{name}_min"] = float(np.nanmin(vals))
            out[f"pp_chain_pair_{name}_max"] = float(np.nanmax(vals))
        if pl:
            vals = np.array([m[i, k] for i, k in pl], dtype=float)
            out[f"pl_chain_pair_{name}_mean"] = float(np.nanmean(vals))
            out[f"pl_chain_pair_{name}_min"] = float(np.nanmin(vals))
            out[f"pl_chain_pair_{name}_max"] = float(np.nanmax(vals))
    cpl = np.asarray(j.get("chain_plddt", []), dtype=float)
    if cpl.size >= n_chain and lig:
        out["ligand_chain_plddt_mean"] = float(np.nanmean(cpl[lig]))
        out["ligand_chain_plddt_min"] = float(np.nanmin(cpl[lig]))
    return out


def best_sample(samples) -> tuple:
    """Pick the sample with the highest ranking_score; return (idx, cif, json, tier1)."""
    best = None
    for k, cif, js in samples:
        j = json.loads(js.read_text())
        t1 = tier1_from_summary(j)
        score = t1["ranking_score"]
        if best is None or (score == score and score > best[4]):  # score==score guards NaN
            best = (k, cif, js, t1, (score if score == score else -1.0))
    return best[:4] if best else None


# ---- ESMFold2 backend (predict_tetramer_esmfold2.py layout: <name>/<name>.cif + _confidence.json) ----

def find_esmfold2_pred(pred_root: Path, name: str):
    """Return (0, cif, conf_json, tier1) for an ESMFold2 prediction, or None."""
    d = pred_root / name
    cif, conf = d / f"{name}.cif", d / f"{name}_confidence.json"
    if not (cif.exists() and conf.exists()):
        return None
    return (0, cif, conf, tier1_from_esmfold2(json.loads(conf.read_text())))


def tier1_from_esmfold2(j: dict) -> dict:
    """Map ESMFold2 confidence to the shared Tier-1 schema. ESMFold2 has NO PDE head, so
    mean_pp_chain_pair_gpde is NaN (the Protenix separator); it exposes pae for a future
    inter-chain-PAE metric. Protein chains are the first n_copies entries (P0..P3 before ligands)."""
    n = int(j.get("n_copies") or 4)
    pp = list(range(n))
    pc = np.asarray(j.get("pair_chains_iptm") or [], dtype=float)
    prot_pairs = [(i, k) for i in pp for k in pp if i < k]
    min_cpi = (float(min(pc[i, k] for i, k in prot_pairs))
               if pc.ndim == 2 and pc.shape[0] >= n else np.nan)
    return dict(
        iptm=float(j.get("iptm") if j.get("iptm") is not None else np.nan),
        ptm=float(j.get("ptm") if j.get("ptm") is not None else np.nan),
        plddt=float(j.get("plddt_mean") if j.get("plddt_mean") is not None else np.nan),
        ranking_score=np.nan, has_clash=False,
        min_pp_chain_pair_iptm=min_cpi, mean_pp_chain_pair_gpde=np.nan,
        min_prot_chain_plddt=np.nan,
    )


# ---------- geometry ----------

def _load_arr(path: Path):
    p = str(path)
    if p.endswith(".cif"):
        from biotite.structure.io.pdbx import CIFFile, get_structure
        return get_structure(CIFFile.read(p), model=1)
    from biotite.structure.io.pdb import PDBFile
    return PDBFile.read(p).get_structure(model=1)


def protein_chains(arr):
    """Chain ids that are protein (have CA and >10 residues)."""
    import biotite.structure as struc  # noqa: F401
    out = []
    for ch in pd.unique(arr.chain_id):
        sub = arr[arr.chain_id == ch]
        if "CA" in set(sub.atom_name) and len(pd.unique(sub.res_id)) > 10:
            out.append(ch)
    return list(out)


def usalign_complex_tm(usalign: str, mobile: Path, ref: Path) -> float:
    """Symmetry-aware complex TM-score normalized by the REFERENCE (Structure_2 = ref)."""
    try:
        r = subprocess.run([usalign, str(mobile), str(ref), "-mm", "1", "-ter", "1"],
                           capture_output=True, text=True, timeout=600)
    except Exception:
        return np.nan
    for ln in r.stdout.splitlines():
        if "normalized by length of Structure_2" in ln:
            m = re.search(r"TM-score=\s*([0-9.]+)", ln)
            if m:
                return float(m.group(1))
    return np.nan


def _atom_coord(arr, chain, res_id, atom, fallback="CA"):
    sub = arr[(arr.chain_id == chain) & (arr.res_id == res_id)]
    if len(sub) == 0:
        return None
    for a in (atom, fallback):
        hit = sub[sub.atom_name == a]
        if len(hit):
            return hit.coord[0]
    return None


def xprot_as_distances(arr, pchains, resid_offset: int = 0, *,
                       core_residues: Mapping[str, int] | None = None,
                       partner_residues: Mapping[str, int] | None = None) -> dict:
    """For each protein chain's catalytic core, distance to the NEAREST neighbour-chain Asn255-ND2.
    Returns per-core median over chains (the interfacial catalytic geometry).

    ``resid_offset`` maps manifest ``index_0b`` to the predicted ``res_id``; it is 0 when the
    prediction uses the native mature frame (index_0b == res_id == crystal resSeq, the standard),
    and 1 for a legacy leading-Met prediction that shifts every residue by one.

    ``core_residues``/``partner_residues`` carry THIS parent's own indices under the Q00511 label
    nomenclature; they default to the Q00511 panel so existing callers are unchanged. Passing
    another parent's protein with Q00511 indices measures the wrong residues silently, so callers
    must resolve the parent's own numbering (``resolve_catalytic_residues``) rather than rely on
    the default.
    """
    core_residues = dict(core_residues if core_residues is not None else CATALYTIC_CORE)
    partner_residues = dict(partner_residues if partner_residues is not None else XPROT_W1)
    if "Asn255" not in partner_residues:
        return {}
    asn_id = partner_residues["Asn255"] + resid_offset
    per_core = {}
    for core_lab, core_id0 in core_residues.items():
        core_id = core_id0 + resid_offset
        dists = []
        for ch in pchains:
            c = _atom_coord(arr, ch, core_id, FUNC_ATOM[core_lab])
            if c is None:
                continue
            best = np.inf
            for nb in pchains:
                if nb == ch:
                    continue
                n = _atom_coord(arr, nb, asn_id, FUNC_ATOM["Asn255"])
                if n is None:
                    continue
                best = min(best, float(np.linalg.norm(c - n)))
            if np.isfinite(best):
                dists.append(best)
        per_core[core_lab] = float(np.median(dists)) if dists else np.nan
    return per_core


WT_NORMALIZATION_FIELDS = {
    "interface_1_bsa_total_min_a2": "ratio",
    "interface_2_bsa_total_min_a2": "ratio",
    "interface_1_n_residue_contacts_min_8a": "ratio",
    "interface_2_n_residue_contacts_min_8a": "ratio",
    "interface_1_n_salt_bridges_min_4a": "delta",
    "interface_2_n_salt_bridges_min_4a": "delta",
    "interface_1_rosetta_dsasa_int_min_a2": "ratio",
    "interface_2_rosetta_dsasa_int_min_a2": "ratio",
    "interface_1_rosetta_dg_separated_max_reu": "delta",
    "interface_2_rosetta_dg_separated_max_reu": "delta",
    "interface_1_rosetta_dg_per_sasa_x100_max": "delta",
    "interface_2_rosetta_dg_per_sasa_x100_max": "delta",
    "interface_1_rosetta_dg_per_sasa_max_reu_per_a2": "delta",
    "interface_2_rosetta_dg_per_sasa_max_reu_per_a2": "delta",
    "interface_1_rosetta_packstat_min": "delta",
    "interface_2_rosetta_packstat_min": "delta",
    "interface_1_rosetta_hbonds_int_min": "delta",
    "interface_2_rosetta_hbonds_int_min": "delta",
    "interface_1_rosetta_hbond_energy_fraction_min": "delta",
    "interface_2_rosetta_hbond_energy_fraction_min": "delta",
    "interface_1_rosetta_delta_unsat_hbonds_max": "delta",
    "interface_2_rosetta_delta_unsat_hbonds_max": "delta",
    "interface_1_rosetta_shape_complementarity_min": "delta",
    "interface_2_rosetta_shape_complementarity_min": "delta",
}


def add_interface_wt_normalization(df: pd.DataFrame, *, require_wt: bool) -> pd.DataFrame:
    """Attach real same-backend WT baselines; never substitute a placeholder baseline."""
    out = df.copy()
    out["interface_wt_baseline_available"] = False
    missing_parents = []
    for parent, group in out.groupby("parent", dropna=False):
        group_index = group.index
        metrics_ok = (group["interface_metrics_ok"] if "interface_metrics_ok" in group
                      else pd.Series(False, index=group.index))
        wt = group[
            group["kind"].astype(str).str.upper().eq("WT")
            & metrics_ok.fillna(False)
        ]
        if wt.empty:
            predicted = (group["predicted"] if "predicted" in group
                         else pd.Series(False, index=group.index))
            if predicted.fillna(False).any():
                missing_parents.append(str(parent))
            continue
        out.loc[group_index, "interface_wt_baseline_available"] = True
        for metric, operation in WT_NORMALIZATION_FIELDS.items():
            if metric not in out.columns:
                continue
            baseline = pd.to_numeric(wt[metric], errors="coerce").median()
            baseline_col = f"wt_{metric}"
            out.loc[group_index, baseline_col] = baseline
            if not np.isfinite(baseline):
                continue
            if operation == "ratio":
                result_col = f"{metric}_ratio_to_wt"
                if baseline != 0:
                    out.loc[group_index, result_col] = out.loc[group_index, metric] / baseline
            else:
                result_col = f"{metric}_delta_vs_wt"
                out.loc[group_index, result_col] = out.loc[group_index, metric] - baseline
    if require_wt and missing_parents:
        raise ValueError(
            "real predicted WT interface baselines are missing for parents: "
            + ", ".join(sorted(set(missing_parents)))
        )
    return out


# ---------- driver ----------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--backend", choices=["protenix", "esmfold2"], default="protenix",
                    help="which predictor's output layout to read (gate is backend-selectable)")
    ap.add_argument(
        "--manifest", required=True,
        help="manifest parquet with id,name,parent,kind,allele (name = pred output-dir stem)",
    )
    ap.add_argument("--usalign", required=True)
    ap.add_argument(
        "--crystal-ref", default=None,
        help="1R51_tetramer_ABCD.pdb (for the crystal parent)",
    )
    ap.add_argument("--crystal-parent", default="Q00511")
    ap.add_argument("--extra-table", default=None,
                    help="optional parquet to left-join on id (e.g. activity/expression)")
    ap.add_argument(
        "--interface-pairs-out", default=None,
        help="chain-pair long-table parquet (default: <out stem>_interface_pairs.parquet)",
    )
    ap.add_argument(
        "--catalytic-map", default=None,
        help="long table (parquet/csv: parent,label,index_0b) giving each parent's OWN catalytic "
             "residue indices under the Q00511 label nomenclature. Without it only Q00511 gets "
             "the cross-protomer distance panel and ligand-to-catalytic distances; every other "
             "parent's panel is omitted rather than measured against Q00511's numbering.",
    )
    ap.add_argument(
        "--crystal-offset", type=int, default=0,
        help="added to crystal residue ids to reach prediction residue ids. 0 when the prediction "
             "uses the same frame as the crystal (the standard: uricase predicted at its native "
             "Met-excluded length); 1 only for legacy leading-Met predictions against a "
             "Met-excluded crystal.",
    )
    ap.add_argument(
        "--crystal-ligand-resname", default=None,
        help="restrict the crystal pocket shell to this ligand residue name (e.g. AZA for 1R51); "
             "default uses every non-standard residue in the reference",
    )
    ap.add_argument("--pocket-shell-radius", type=float, default=8.0,
                    help="crystal pocket shell radius around the reference ligand (A)")
    ap.add_argument("--ligand-contact-cutoff", type=float, default=4.5,
                    help="heavy-atom cutoff for ligand pocket contacts (A)")
    ap.add_argument(
        "--ligand-sites-out", default=None,
        help="per-ligand-copy long-table parquet (default: <out stem>_ligand_sites.parquet)",
    )
    ap.add_argument(
        "--no-ligand-metrics", action="store_true",
        help="skip the ligand/pocket layer entirely (apo predictions)",
    )
    ap.add_argument("--sasa-probe-radius", type=float, default=1.4)
    ap.add_argument("--sasa-point-number", type=int, default=1000)
    ap.add_argument("--contact-cutoff", type=float, default=8.0)
    ap.add_argument("--salt-bridge-cutoff", type=float, default=4.0)
    ap.add_argument(
        "--rosetta-interface-analyzer",
        default=os.environ.get("ROSETTA_INTERFACE_ANALYZER"),
        help="optional InterfaceAnalyzer executable; enables CPU Rosetta interface metrics",
    )
    ap.add_argument(
        "--rosetta-run-dir", "--rosetta-work-dir", dest="rosetta_run_dir", default=None,
        help=(
            "Rosetta dimer, command, and score directory in the experiment run layer "
            "(--rosetta-work-dir is a deprecated alias)"
        ),
    )
    ap.add_argument(
        "--rosetta-log-dir", default=None,
        help="captured Rosetta stdout/stderr directory in the project logs layer",
    )
    ap.add_argument("--rosetta-timeout-seconds", type=int, default=21600)
    ap.add_argument(
        "--require-interface-wt",
        action="store_true",
        help="fail if a predicted parent lacks a real predicted WT row in the manifest",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.sasa_probe_radius <= 0:
        ap.error("--sasa-probe-radius must be positive")
    if args.sasa_point_number <= 0:
        ap.error("--sasa-point-number must be positive")
    if args.contact_cutoff <= 0 or args.salt_bridge_cutoff <= 0:
        ap.error("interface distance cutoffs must be positive")
    if args.rosetta_timeout_seconds <= 0:
        ap.error("--rosetta-timeout-seconds must be positive")
    if args.rosetta_interface_analyzer and not args.rosetta_run_dir:
        ap.error("--rosetta-interface-analyzer requires --rosetta-run-dir")
    if args.rosetta_interface_analyzer and not args.rosetta_log_dir:
        ap.error("--rosetta-interface-analyzer requires --rosetta-log-dir")
    if args.rosetta_run_dir and args.rosetta_log_dir:
        rosetta_run_dir = Path(args.rosetta_run_dir).resolve()
        rosetta_log_dir = Path(args.rosetta_log_dir).resolve()
        if (
            rosetta_run_dir == rosetta_log_dir
            or rosetta_run_dir in rosetta_log_dir.parents
            or rosetta_log_dir in rosetta_run_dir.parents
        ):
            ap.error("--rosetta-run-dir and --rosetta-log-dir must be separate directory trees")

    man = pd.read_parquet(args.manifest)
    required_manifest = {"id", "name", "parent", "kind"}
    missing_manifest = sorted(required_manifest - set(man.columns))
    if missing_manifest:
        raise ValueError(f"manifest is missing required columns: {missing_manifest}")
    pred_root = Path(args.pred_root)
    out_path = Path(args.out)
    pairs_out_path = (Path(args.interface_pairs_out) if args.interface_pairs_out else
                      out_path.with_name(f"{out_path.stem}_interface_pairs.parquet"))
    if pairs_out_path.resolve() == out_path.resolve():
        ap.error("--interface-pairs-out must differ from --out")

    # pass 1: locate each variant's best sample + parse coords once (cache arrays for WT refs)
    best = {}   # name -> (idx, cif, json, tier1)
    arr_cache = {}
    for _, row in man.iterrows():
        name = row["name"]
        if args.backend == "esmfold2":
            bs = find_esmfold2_pred(pred_root, name)
        else:
            samples = find_pred_samples(pred_root, name)
            bs = best_sample(samples) if samples else None
        if bs:
            best[name] = bs

    def get_arr(name):
        if name not in arr_cache:
            arr_cache[name] = _load_arr(best[name][1])
        return arr_cache[name]

    # parent -> WT variant name (the relative reference)
    wt_name_by_parent = {}
    for _, row in man.iterrows():
        if str(row.get("kind", "")).upper() == "WT" and row["name"] in best:
            wt_name_by_parent[row["parent"]] = row["name"]

    catalytic_map = load_catalytic_map(args.catalytic_map) if args.catalytic_map else None
    if catalytic_map is not None:
        covered = sorted(set(man["parent"]) & set(catalytic_map))
        uncovered = sorted(set(man["parent"]) - set(catalytic_map))
        print(f"[catalytic map] {len(catalytic_map)} parents loaded; "
              f"covering {len(covered)}/{man['parent'].nunique()} manifest parents"
              + (f"; NO panel for {uncovered}" if uncovered else ""))

    # crystal pocket reference (built once; needs a holo reference with its ligand)
    crystal_ref = None
    if args.crystal_ref and not args.no_ligand_metrics:
        try:
            crystal_ref = CrystalPocketReference(
                _load_arr(Path(args.crystal_ref)),
                ligand_resname=args.crystal_ligand_resname,
                shell_radius=args.pocket_shell_radius,
            )
            print(f"[eval] crystal pocket shell ({args.pocket_shell_radius} A): "
                  f"{ {k: len(v) for k, v in crystal_ref.sites.items()} }, "
                  f"offset={args.crystal_offset}")
        except Exception as exc:
            print(f"[eval] WARNING: crystal pocket reference unavailable: {exc}")

    rows = []
    pair_frames = []
    ligand_frames = []
    for manifest_row_index, (_, row) in enumerate(man.iterrows()):
        name, parent, kind = row["name"], row["parent"], row.get("kind")
        rec = dict(
            manifest_row_index=manifest_row_index,
            id=row["id"],
            name=name,
            parent=parent,
            kind=kind,
            allele=row.get("allele"),
            predicted=name in best,
            interface_metrics_ok=False,
            interface_metrics_error=(None if name in best else "prediction_missing"),
            rosetta_interface_requested=bool(args.rosetta_interface_analyzer),
            rosetta_interface_metrics_ok=False,
        )
        if name in best:
            k, cif, js, t1 = best[name]
            rec.update(sample=k, **t1)
            arr = get_arr(name)
            pchains = protein_chains(arr)
            rec["n_protein_chains"] = len(pchains)
            if len(pchains) in (2, 4):
                pairs = interface_pair_metrics(
                    arr,
                    pchains,
                    sasa_probe_radius=args.sasa_probe_radius,
                    sasa_point_number=args.sasa_point_number,
                    contact_cutoff=args.contact_cutoff,
                    salt_bridge_cutoff=args.salt_bridge_cutoff,
                )
                pairs.insert(0, "prediction_path", str(cif))
                pairs.insert(0, "sample", k)
                pairs.insert(0, "backend", args.backend)
                pairs.insert(0, "allele", row.get("allele"))
                pairs.insert(0, "kind", kind)
                pairs.insert(0, "parent", parent)
                pairs.insert(0, "name", name)
                pairs.insert(0, "id", row["id"])
                pairs.insert(0, "manifest_row_index", manifest_row_index)
                pairs["bsa_method"] = "Shrake-Rupley/ProtOr"
                pairs["sasa_probe_radius_a"] = args.sasa_probe_radius
                pairs["sasa_point_number"] = args.sasa_point_number
                pairs["contact_cutoff_a"] = args.contact_cutoff
                pairs["salt_bridge_cutoff_a"] = args.salt_bridge_cutoff
                pair_frames.append(pairs)
                rec["interface_metrics_ok"] = True
                rec["interface_metrics_error"] = None
                rec["n_interchain_contacts"] = int(pairs["n_residue_contacts_8a"].sum())
            else:
                rec["interface_metrics_error"] = (
                    f"expected_2_or_4_protein_chains_got_{len(pchains)}"
                )
            # complex TM vs parent WT (relative)
            wt = wt_name_by_parent.get(parent)
            if wt and wt in best:
                rec["complex_TM_vs_wt"] = usalign_complex_tm(args.usalign, cif, best[wt][1])
            # crystal check for the crystal parent
            if args.crystal_ref and parent == args.crystal_parent:
                rec["complex_TM_vs_crystal"] = usalign_complex_tm(
                    args.usalign, cif, Path(args.crystal_ref)
                )
            # cross-protomer active-site distances, in THIS parent's own numbering
            cat_core, cat_partner = resolve_catalytic_residues(parent, catalytic_map)
            rec["catalytic_numbering_resolved"] = bool(cat_core)
            if cat_core:
                xd = xprot_as_distances(
                    arr, pchains, resid_offset=args.crystal_offset,
                    core_residues=cat_core, partner_residues=cat_partner,
                )
                for lab, v in xd.items():
                    rec[f"xprot_{lab}_dist"] = v
            # chain-pair PAE / gPDE / ipTM blocks, incl. the protein-ligand off-diagonal
            if args.backend == "protenix":
                try:
                    rec.update(pae_metrics_from_summary(json.loads(js.read_text()),
                                                        n_protein=len(pchains) or 4))
                except Exception as exc:  # confidence JSON shape drift must not kill the run
                    rec["pae_metrics_error"] = str(exc)[:200]
            # ligand pocket geometry (prediction-intrinsic) + crystal-referenced pocket RMSD
            if not args.no_ligand_metrics:
                try:
                    # Pocket-contact counts need no residue panel, but every *_to_core /
                    # lig_min_dist_<label> column is defined by the catalytic indices. Feeding
                    # Q00511's indices to another parent produces distances to unrelated
                    # residues with no error, so an unresolved parent gets an empty panel and
                    # the distance columns are simply absent.
                    sites = ligand_site_metrics(
                        arr, pchains,
                        core_residues={k: v + args.crystal_offset
                                       for k, v in cat_core.items()},
                        partner_residues={k: v + args.crystal_offset
                                          for k, v in cat_partner.items()},
                        contact_cutoff=args.ligand_contact_cutoff,
                    )
                except Exception as exc:
                    sites = pd.DataFrame()
                    rec["ligand_metrics_error"] = str(exc)[:200]
                if len(sites):
                    if crystal_ref is not None and parent == args.crystal_parent:
                        try:
                            rec.update(crystal_ref.evaluate(arr, pchains,
                                                            offset=args.crystal_offset))
                        except Exception as exc:
                            rec["crystal_pocket_error"] = str(exc)[:200]
                    rec["n_ligand_sites"] = int(len(sites))
                    for col in [c for c in sites.columns
                                if c.startswith(("lig_min_dist_", "n_pocket_contacts",
                                                 "lig_centroid_to_core"))]:
                        v = pd.to_numeric(sites[col], errors="coerce")
                        rec[f"{col}_med"] = float(v.median())
                        rec[f"{col}_worst"] = float(v.max())
                    if "site_is_interprotomer" in sites:
                        rec["frac_sites_interprotomer"] = float(sites.site_is_interprotomer.mean())
                    sites = sites.copy()
                    for c, v in (("manifest_row_index", manifest_row_index), ("id", row["id"]),
                                 ("name", name), ("parent", parent), ("kind", kind),
                                 ("allele", row.get("allele")), ("sample", k)):
                        sites.insert(0, c, v)
                    ligand_frames.append(sites)
        rows.append(rec)

    df = pd.DataFrame(rows)
    pair_df = (pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame(
        columns=[
            "manifest_row_index", "id", "name", "parent", "kind", "allele", "backend",
            "sample", "prediction_path", "chain_1", "chain_2", "chain_pair",
            "bsa_total_a2", "bsa_per_partner_a2", "n_residue_contacts_8a",
            "n_salt_bridges_4a", "interface_class", "interface_class_rank",
            "interface_copy", "d2_matching", "d2_matching_mean_bsa_total_a2",
            "is_biological_interface",
        ]
    ))

    if args.rosetta_interface_analyzer:
        if pair_df.empty:
            raise ValueError(
                "Rosetta interface scoring requested, but no valid tetramer pairs exist"
            )
        rosetta_run_dir = Path(args.rosetta_run_dir)
        rosetta_log_dir = Path(args.rosetta_log_dir)
        rosetta_run_dir.mkdir(parents=True, exist_ok=True)
        rosetta_log_dir.mkdir(parents=True, exist_ok=True)
        pair_df = run_rosetta_interface_analyzer(
            pair_df,
            get_arr,
            executable=args.rosetta_interface_analyzer,
            run_dir=rosetta_run_dir,
            log_dir=rosetta_log_dir,
            timeout_seconds=args.rosetta_timeout_seconds,
        )
        scored_manifest_rows = set(
            pair_df.loc[
                pair_df["is_biological_interface"]
                & pair_df["rosetta_dsasa_int_a2"].notna(),
                "manifest_row_index",
            ]
        )
        df["rosetta_interface_metrics_ok"] = df["manifest_row_index"].isin(
            scored_manifest_rows
        )

    if not pair_df.empty:
        summary_rows = []
        for manifest_row_index, group in pair_df.groupby("manifest_row_index", sort=False):
            summary_rows.append({
                "manifest_row_index": manifest_row_index,
                **summarize_interface_pairs(group),
            })
        df = df.merge(pd.DataFrame(summary_rows), on="manifest_row_index", how="left",
                      validate="one_to_one")
    df = add_interface_wt_normalization(df, require_wt=args.require_interface_wt)

    # deviation of cross-protomer distances vs each parent's WT
    for lab in CATALYTIC_CORE:
        col = f"xprot_{lab}_dist"
        if col in df.columns:
            wtval = {p: df[(df.parent == p) & (df.kind.astype(str).str.upper() == "WT")][col].median()
                     for p in df.parent.unique()}
            df[f"xprot_{lab}_dev"] = df.apply(
                lambda r: (r[col] - wtval.get(r.parent, np.nan)) if pd.notna(r.get(col)) else np.nan, axis=1)

    if args.extra_table:
        extra = pd.read_parquet(args.extra_table)
        keep = [c for c in extra.columns if c not in df.columns or c == "id"]
        df = df.merge(extra[keep], on="id", how="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_out_path.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["manifest_row_index"]).to_parquet(out_path, index=False)
    pair_df.to_parquet(pairs_out_path, index=False)
    if ligand_frames:
        lig_out = (Path(args.ligand_sites_out) if args.ligand_sites_out else
                   out_path.with_name(f"{out_path.stem}_ligand_sites.parquet"))
        lig_out.parent.mkdir(parents=True, exist_ok=True)
        lig_df = pd.concat(ligand_frames, ignore_index=True)
        lig_df.to_parquet(lig_out, index=False)
        print(f"[eval] {len(lig_df)} ligand-site rows -> {lig_out}")
    n_pred = int(df["predicted"].sum())
    print(f"[eval] {len(df)} variants, {n_pred} predicted -> {out_path}")
    print(f"[eval] {len(pair_df)} chain-pair rows -> {pairs_out_path}")
    cols = [c for c in ["id", "parent", "kind", "iptm", "min_pp_chain_pair_iptm",
                        "complex_TM_vs_wt", "complex_TM_vs_crystal", "n_interchain_contacts",
                        "interface_1_bsa_total_min_a2", "interface_2_bsa_total_min_a2",
                        "interface_1_rosetta_packstat_min",
                        "interface_2_rosetta_packstat_min"] if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_rows", 60):
        print(df[df.predicted][cols].to_string(index=False))


if __name__ == "__main__":
    main()
