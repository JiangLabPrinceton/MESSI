"""Ligand-pocket and crystal-referenced pocket geometry for holo homo-oligomer predictions.

Torch-free, Biotite-only, so it can be imported independently of the tetramer-gate CLI. Two layers:

1. **Prediction-intrinsic ligand geometry** (no reference needed): for every ligand copy, the min
   heavy-atom distance to the catalytic core and to the cross-protomer partner residue, the pocket
   contact count, which protein chains contact it, and whether the site is inter-protomer. This is
   the "is the interfacial pocket actually formed" readout.
2. **Crystal-referenced pocket RMSD** (needs a holo reference such as 1R51 with its ligand): the
   reference pocket shell is defined as all protein residues within ``shell_radius`` (default 8 A)
   of the reference ligand; the prediction is superposed onto the reference by the best D2 chain
   permutation (Kabsch on matched CA), and pocket CA / backbone / all-heavy-sidechain RMSDs are
   reported both under that global superposition and under a local pocket-only fit. Ligand centroid
   shift vs the reference ligand is reported per site.

Residue-numbering contract: ``crystal_offset`` is added to reference residue ids to reach prediction
residue ids. For a prediction made in the SAME frame as the crystal (the standard for uricase: the
native Met-excluded sequence) this is 0. It is non-zero only for legacy predictions made in a
shifted frame (e.g. a leading-Met 302 frame against the 301-frame crystal -> offset 1).
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

BACKBONE = frozenset({"N", "CA", "C", "O"})
STANDARD_AA = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)


# --------------------------------------------------------------------------- helpers

def _kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    u, _, vh = np.linalg.svd((mobile - mc).T @ (target - tc))
    if np.linalg.det(u @ vh) < 0:
        u[:, -1] *= -1
    rot = u @ vh
    return rot, tc - mc @ rot


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _atom_index(arr, chains: set[str]) -> dict:
    """(chain, res_id, atom_name) -> {coord, resname, element}, protein atoms only."""
    out = {}
    for i in range(arr.array_length()):
        chain = str(arr.chain_id[i])
        if chain not in chains:
            continue
        out[(chain, int(arr.res_id[i]), str(arr.atom_name[i]))] = {
            "coord": np.asarray(arr.coord[i], dtype=float),
            "resname": str(arr.res_name[i]),
            "element": str(arr.element[i]),
        }
    return out


def _residue_name(idx: Mapping, chain: str, res_id: int) -> str | None:
    for atom in ("CA", "N", "C"):
        hit = idx.get((chain, res_id, atom))
        if hit is not None:
            return hit["resname"]
    return None


def ligand_chains(arr, protein_chain_ids: Sequence[str]) -> list[str]:
    """Chains that carry non-standard (ligand/hetero) residues."""
    pset = {str(c) for c in protein_chain_ids}
    out = []
    for chain in pd.unique(arr.chain_id):
        if str(chain) in pset:
            continue
        sub = arr[arr.chain_id == chain]
        if len(sub) and not np.all(np.isin(sub.res_name, list(STANDARD_AA))):
            out.append(str(chain))
    return out


def _heavy(arr, chain: str) -> np.ndarray:
    sub = arr[(arr.chain_id == chain) & (arr.element != "H")]
    return np.asarray(sub.coord, dtype=float)


# ------------------------------------------------- layer 1: intrinsic ligand geometry

def ligand_site_metrics(
    arr,
    protein_chain_ids: Sequence[str],
    *,
    core_residues: Mapping[str, int],
    partner_residues: Mapping[str, int] | None = None,
    contact_cutoff: float = 4.5,
) -> pd.DataFrame:
    """One row per ligand copy: pocket contacts + distances to catalytic / cross-protomer residues.

    ``core_residues`` / ``partner_residues`` map a label to a PREDICTION residue id (1-based res_id
    as written in the CIF). Distances are min heavy-atom over the nearest protein chain, so the
    readout is independent of predictor chain labelling.
    """
    partner_residues = dict(partner_residues or {})
    prot = arr[np.isin(arr.res_name, list(STANDARD_AA)) & (arr.element != "H")]
    lchains = ligand_chains(arr, protein_chain_ids)
    rows = []
    for lch in lchains:
        lxyz = _heavy(arr, lch)
        if not len(lxyz):
            continue
        rec: dict = {"ligand_chain": lch, "n_ligand_heavy_atoms": int(len(lxyz))}
        d = np.linalg.norm(prot.coord[:, None, :] - lxyz[None, :, :], axis=-1)
        close = d.min(axis=1) < contact_cutoff
        rec["n_pocket_contacts"] = int(close.sum())
        cc = sorted({str(c) for c in prot.chain_id[close]})
        rec["contact_chains"] = ",".join(cc)
        rec["n_contact_chains"] = len(cc)
        for label, rid in {**core_residues, **partner_residues}.items():
            best, best_chain = np.inf, None
            for ch in protein_chain_ids:
                sel = prot[(prot.chain_id == ch) & (prot.res_id == rid)]
                if not len(sel):
                    continue
                dd = float(np.linalg.norm(sel.coord[:, None, :] - lxyz[None, :, :], axis=-1).min())
                if dd < best:
                    best, best_chain = dd, str(ch)
            rec[f"lig_min_dist_{label}"] = best if np.isfinite(best) else np.nan
            rec[f"lig_nearest_chain_{label}"] = best_chain
        # inter-protomer: nearest partner residue sits on a different chain than the nearest core
        if partner_residues and core_residues:
            pl, cl = next(iter(partner_residues)), next(iter(core_residues))
            pc, cch = rec.get(f"lig_nearest_chain_{pl}"), rec.get(f"lig_nearest_chain_{cl}")
            rec["site_is_interprotomer"] = bool(pc and cch and pc != cch)
        core_pts = [
            prot[(prot.chain_id == rec[f"lig_nearest_chain_{lab}"]) & (prot.res_id == rid)].coord.mean(axis=0)
            for lab, rid in core_residues.items()
            if rec.get(f"lig_nearest_chain_{lab}")
        ]
        rec["lig_centroid_to_core_centroid"] = (
            float(np.linalg.norm(lxyz.mean(axis=0) - np.mean(core_pts, axis=0))) if core_pts else np.nan
        )
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------- layer 2: crystal-referenced pocket geometry

class CrystalPocketReference:
    """Holo reference (e.g. 1R51 + AZA): pocket shell residues + reference ligand coordinates."""

    def __init__(self, ref_arr, *, ligand_resname: str | None = None, shell_radius: float = 8.0):
        self.arr = ref_arr
        self.shell_radius = float(shell_radius)
        self.protein_chains = [
            str(c) for c in pd.unique(ref_arr.chain_id)
            if "CA" in set(ref_arr[ref_arr.chain_id == c].atom_name)
            and len(pd.unique(ref_arr[ref_arr.chain_id == c].res_id)) > 10
        ]
        self.index = _atom_index(ref_arr, set(self.protein_chains))
        prot = ref_arr[np.isin(ref_arr.res_name, list(STANDARD_AA)) & (ref_arr.element != "H")]
        lig_mask = ~np.isin(ref_arr.res_name, list(STANDARD_AA)) & (ref_arr.element != "H")
        if ligand_resname:
            lig_mask &= ref_arr.res_name == ligand_resname
        lig = ref_arr[lig_mask]
        if not len(lig):
            raise ValueError("crystal reference contains no ligand atoms")
        self.sites: dict[str, set[tuple[str, int]]] = {}
        self.ligand_coords: dict[str, np.ndarray] = {}
        for site_key in pd.unique(lig.chain_id):
            copy = lig[lig.chain_id == site_key]
            self.ligand_coords[str(site_key)] = np.asarray(copy.coord, dtype=float)
            d = np.linalg.norm(prot.coord[:, None, :] - copy.coord[None, :, :], axis=-1)
            hit = prot[d.min(axis=1) <= self.shell_radius]
            self.sites[str(site_key)] = {
                (str(c), int(r)) for c, r in zip(hit.chain_id, hit.res_id, strict=True)
            }
        self.all_shell = set().union(*self.sites.values()) if self.sites else set()

    # -- superposition ------------------------------------------------------

    def _matched_ca(self, pred_index, mapping, offset):
        ref_xyz, pred_xyz = [], []
        for ref_ch, pred_ch in mapping.items():
            for (c, r, a) in self.index:
                if c != ref_ch or a != "CA":
                    continue
                pk = (pred_ch, r + offset, "CA")
                if pk in pred_index:
                    ref_xyz.append(self.index[(c, r, a)]["coord"])
                    pred_xyz.append(pred_index[pk]["coord"])
        return np.asarray(ref_xyz), np.asarray(pred_xyz)

    def best_chain_mapping(self, pred_index, pred_chains: Sequence[str], offset: int):
        """Best reference->prediction chain permutation by global CA RMSD."""
        best = None
        for perm in itertools.permutations(pred_chains, len(self.protein_chains)):
            mapping = dict(zip(self.protein_chains, perm, strict=True))
            ref_xyz, pred_xyz = self._matched_ca(pred_index, mapping, offset)
            if len(ref_xyz) < 4:
                continue
            rot, trans = _kabsch(pred_xyz, ref_xyz)
            score = _rmsd(pred_xyz @ rot + trans, ref_xyz)
            if best is None or score < best[0]:
                best = (score, mapping, rot, trans)
        if best is None:
            raise ValueError("no valid chain mapping between prediction and crystal reference")
        return best[1], best[2], best[3], best[0]

    # -- pocket metrics -----------------------------------------------------

    def _pocket_rmsd(self, pred_index, mapping, residues, offset, kind, rot, trans):
        ref_xyz, pred_xyz = [], []
        for ref_ch, rid in sorted(residues):
            pred_ch = mapping[ref_ch]
            prid = rid + offset
            if kind == "CA":
                names = ["CA"]
            elif kind == "backbone":
                names = sorted(BACKBONE)
            else:  # sidechain: identity must match, heavy atoms only
                if _residue_name(self.index, ref_ch, rid) != _residue_name(pred_index, pred_ch, prid):
                    continue
                names = sorted(
                    a for (c, r, a), v in self.index.items()
                    if c == ref_ch and r == rid and a not in BACKBONE and v["element"] != "H"
                )
            for atom in names:
                rk, pk = (ref_ch, rid, atom), (pred_ch, prid, atom)
                if rk in self.index and pk in pred_index:
                    ref_xyz.append(self.index[rk]["coord"])
                    pred_xyz.append(pred_index[pk]["coord"])
        if len(ref_xyz) < 3:
            return np.nan, np.nan, len(ref_xyz)
        ref_xyz, pred_xyz = np.asarray(ref_xyz), np.asarray(pred_xyz)
        global_rmsd = _rmsd(pred_xyz @ rot + trans, ref_xyz)
        lrot, ltrans = _kabsch(pred_xyz, ref_xyz)
        return global_rmsd, _rmsd(pred_xyz @ lrot + ltrans, ref_xyz), len(ref_xyz)

    def identity_fraction(self, pred_index, mapping, residues, offset) -> float:
        same = total = 0
        for ref_ch, rid in residues:
            rn = _residue_name(self.index, ref_ch, rid)
            pn = _residue_name(pred_index, mapping[ref_ch], rid + offset)
            if rn is not None and pn is not None:
                total += 1
                same += rn == pn
        return float(same / total) if total else np.nan

    def ligand_centroid_shift(self, arr, pred_chains, rot, trans) -> tuple[float, float]:
        lch = ligand_chains(arr, pred_chains)
        if len(lch) != len(self.ligand_coords) or not lch:
            return np.nan, np.nan
        ref_cent = {k: v.mean(axis=0) for k, v in self.ligand_coords.items()}
        pred_cent = {c: _heavy(arr, c).mean(axis=0) @ rot + trans for c in lch}
        keys = list(ref_cent)
        best = None
        for perm in itertools.permutations(lch):
            dists = [float(np.linalg.norm(ref_cent[k] - pred_cent[p])) for k, p in zip(keys, perm, strict=True)]
            if best is None or sum(dists) < best[0]:
                best = (sum(dists), dists)
        return float(np.mean(best[1])), float(np.max(best[1]))

    def evaluate(self, arr, pred_chains: Sequence[str], *, offset: int = 0) -> dict:
        """All crystal-referenced pocket metrics for one predicted assembly."""
        pred_index = _atom_index(arr, {str(c) for c in pred_chains})
        mapping, rot, trans, global_ca = self.best_chain_mapping(pred_index, pred_chains, offset)
        out = {
            "crystal_offset": int(offset),
            "global_ca_rmsd_to_crystal": global_ca,
            "crystal_chain_mapping": ";".join(f"{r}:{p}" for r, p in mapping.items()),
            "pocket_shell_radius_a": self.shell_radius,
        }
        for kind in ("CA", "backbone", "sidechain"):
            g, l, n = self._pocket_rmsd(pred_index, mapping, self.all_shell, offset, kind, rot, trans)
            out[f"pocket{int(self.shell_radius)}a_{kind.lower()}_rmsd_global"] = g
            out[f"pocket{int(self.shell_radius)}a_{kind.lower()}_rmsd_localfit"] = l
            out[f"pocket{int(self.shell_radius)}a_{kind.lower()}_n_atoms"] = n
        per_ca, per_sc = [], []
        for residues in self.sites.values():
            per_ca.append(self._pocket_rmsd(pred_index, mapping, residues, offset, "CA", rot, trans)[1])
            per_sc.append(self._pocket_rmsd(pred_index, mapping, residues, offset, "sidechain", rot, trans)[1])
        pfx = f"pocket{int(self.shell_radius)}a_per_site"
        out[f"{pfx}_ca_rmsd_localfit_mean"] = float(np.nanmean(per_ca)) if per_ca else np.nan
        out[f"{pfx}_ca_rmsd_localfit_max"] = float(np.nanmax(per_ca)) if per_ca else np.nan
        out[f"{pfx}_sidechain_rmsd_localfit_mean"] = float(np.nanmean(per_sc)) if per_sc else np.nan
        out[f"{pfx}_sidechain_rmsd_localfit_max"] = float(np.nanmax(per_sc)) if per_sc else np.nan
        out[f"pocket{int(self.shell_radius)}a_same_residue_fraction"] = self.identity_fraction(
            pred_index, mapping, self.all_shell, offset
        )
        cm, cx = self.ligand_centroid_shift(arr, pred_chains, rot, trans)
        out["ligand_centroid_shift_vs_crystal_mean"] = cm
        out["ligand_centroid_shift_vs_crystal_max"] = cx
        return out
