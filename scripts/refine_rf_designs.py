#!/usr/bin/env python
"""Head-guided residue refinement, exact per-seed fronts, and structure validation."""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.constraints import load_constraint_manifest
from inverse_folding.reference_flow.refine import (
    editable_positions,
    enumerate_pairs,
    enumerate_round_robin_doubles,
    enumerate_singles,
    extract_target_cores,
    head_objective_dominates,
    head_first_pareto_front,
    head_refinement_metrics,
    hotspot_blocks,
    rank_margin_mass,
    refine_sequence,
    refine_sequence_head,
    select_head_residue_targets,
    select_head_single_aa_choices,
    structure_gate,
)

# Injected oracle bundle; the smoke test builds fakes with this exact shape.
# window_coords_fn(seq) -> (window_starts_0b, window_ends_0b) for THIS sequence's length
# (the head window template is length-dependent — a fixed template mis-maps long proteins).
Oracles = namedtuple(
    "Oracles",
    ["head_fn", "nmp_fn", "struct_fn", "window_coords_fn", "head_score_fn"],
    defaults=(None,),
)

_REFINEMENT_STRUCTURE_METRICS = frozenset(
    {"global_ca_rmsd", "plddt", "sidechain"}
)
_PROTOCOL_GATE_FIELDS = (
    "gate_scTM_min",
    "gate_cat_max_scRMSD_max",
    "gate_predicted_active_site_min_pLDDT_min",
)


def _parse_refinement_structure_metrics(value: str | None) -> frozenset[str]:
    requested = frozenset(
        item.strip().lower() for item in str(value or "").split(",") if item.strip()
    )
    unknown = requested - _REFINEMENT_STRUCTURE_METRICS
    if unknown:
        raise ValueError(
            "unknown --refinement-structure-metrics values: "
            f"{sorted(unknown)}; expected {sorted(_REFINEMENT_STRUCTURE_METRICS)}"
        )
    return requested


def _validate_structure_gate_config(args) -> None:
    protocol_values = {name: getattr(args, name) for name in _PROTOCOL_GATE_FIELDS}
    if args.structure_gate_profile == "protocol":
        missing = [name for name, value in protocol_values.items() if value is None]
        if missing:
            raise ValueError(
                "--structure-gate-profile protocol requires all three thresholds: "
                "--gate-scTM-min, --gate-cat-max-scRMSD-max, and "
                "--gate-predicted-active-site-min-pLDDT-min"
            )
        if any(
            value is not None
            for value in (
                args.scTM_eps,
                args.scRMSD_max,
                args.active_site_RMSD_max,
                args.max_anchor_sidechain_RMSD_max,
            )
        ):
            raise ValueError(
                "protocol structure gate cannot be mixed with legacy structure thresholds"
            )
        sc_tm = float(args.gate_scTM_min)
        cat_max = float(args.gate_cat_max_scRMSD_max)
        min_plddt = float(args.gate_predicted_active_site_min_pLDDT_min)
        if not np.isfinite(sc_tm) or not 0.0 <= sc_tm <= 1.0:
            raise ValueError("--gate-scTM-min must be finite and in [0, 1]")
        if not np.isfinite(cat_max) or cat_max < 0.0:
            raise ValueError("--gate-cat-max-scRMSD-max must be finite and non-negative")
        if not np.isfinite(min_plddt) or not 0.0 <= min_plddt <= 100.0:
            raise ValueError(
                "--gate-predicted-active-site-min-pLDDT-min must be finite and in [0, 100]"
            )
        if not args.constraint_manifest:
            raise ValueError("protocol structure gate requires --constraint-manifest")
        return

    if any(value is not None for value in protocol_values.values()):
        raise ValueError("legacy structure gate cannot use protocol gate thresholds")
    if all(
        value is None
        for value in (
            args.scTM_eps,
            args.scRMSD_max,
            args.active_site_RMSD_max,
            args.max_anchor_sidechain_RMSD_max,
        )
    ):
        raise ValueError("legacy structure gate requires at least one legacy threshold")


def _make_target_window_idx_fn(win_starts, win_ends):
    """Build the cores->covering-window-indices map for a specific sequence length.

    Rebuilt per seed from that seed's actual window template (all point-mutation
    candidates share the seed's length, so the template is stable within a seed but
    NOT across proteins of different lengths)."""
    ws = np.asarray(win_starts)
    we = np.asarray(win_ends)

    def target_window_idx_fn(cores):
        return [
            w for w in range(len(ws))
            if any(c.core_start < we[w] and c.core_start + 9 > ws[w] for c in cores)
        ]

    return target_window_idx_fn

# generated.parquet schema (write_phase_c_outputs) — evaluator_ready must match exactly.
GENERATED_COLUMNS = ["protein_id", "design_idx", "sequence", "seed", "wall_seconds"]
_SEED_PROVENANCE_COLUMNS = (
    "seed_group",
    "selection_rule_id",
    "source_campaign",
    "source_depth",
    "source_root_index",
)


# --------------------------------------------------------------------------- #
# Sharded input loading (glob + concat; falls back to a flat file)
# --------------------------------------------------------------------------- #
def _load_sharded(root: Path, subdir: str, filename: str) -> pd.DataFrame:
    """Load a clean refinement input flexibly (the caller hands whatever is tidiest).

    Accepts, in priority order:
      1. ``root`` is a parquet FILE            -> read it directly (a merged/clean parquet);
      2. ``root/<subdir>/*shard*/filename``    -> return-package layout;
      3. ``root/*shard*/filename``             -> shard dirs directly under root (no wrapper);
      4. flat ``root/<subdir>/filename`` or ``root/filename``.
    Shards are concatenated. The ``generation``/``eval_immune`` wrapper is therefore
    optional, so a pre-organized input works without staging a return-package tree.
    """
    root = Path(root)
    if root.is_file():
        return pd.read_parquet(root)
    paths: list[str] = []
    for pattern in (root / subdir / "*shard*" / filename, root / "*shard*" / filename):
        paths = sorted(glob.glob(str(pattern)))
        if paths:
            break
    if not paths:
        for flat in (root / subdir / filename, root / filename):
            if flat.exists():
                paths = [str(flat)]
                break
    if not paths:
        raise FileNotFoundError(
            f"no {filename} found under {root} (tried '{subdir}/*shard*/', '*shard*/', "
            "a flat file, or a direct parquet path)"
        )
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _canonical_allele(allele: str) -> str:
    """Normalize an allele to the canonical NetMHCIIpan form ``HLA-DRB1*07:01``.

    ``StandaloneRunner.score_batch`` normalizes ``*``->``_`` and drops ``:``; a run-dir
    filesystem tag like ``HLA-DRB1_07_01`` would mis-normalize to an INVALID
    ``DRB1_07_01``. The ``*``/``:`` form maps to a valid ``DRB1_0701``. Already-canonical
    input (contains ``*`` or ``:``) passes through unchanged.
    """
    a = allele.strip()
    if "*" in a or ":" in a:
        return a
    parts = a.split("_")  # 'HLA-DRB1_07_01' -> ['HLA-DRB1', '07', '01'] (gene, field1, field2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return f"{parts[0]}*{parts[1]}:{parts[2]}"
    return a


# --------------------------------------------------------------------------- #
# Proposer: block-structured enumeration (singles exhaustive, head-ranked pairs)
# --------------------------------------------------------------------------- #
def make_propose_fn(oracles: Oracles, protein_id: str, anchors: set, *, max_pairs: int,
                    target_window_idx_fn):
    """Block-structured proposer (PLAN §1). Per hotspot block: exhaustive singles over
    the block's editable pocket positions (minus anchors), then head-ranked pairs among
    the singles (best head-proxy first), capped by ``max_pairs``. This is the legacy
    NMP-pocket proposer; Head-only residue targeting uses its own two-stage pipeline.
    ``target_window_idx_fn`` is the per-seed (length-correct) core->window map."""

    def propose(seq: str, cores):
        if not cores:
            return []
        candidates = []
        for block in hotspot_blocks(cores):
            editable = sorted(
                {p for c in block for p in editable_positions(c, anchors=anchors)}
            )
            if not editable:
                continue
            singles = enumerate_singles(seq, editable)
            candidates.extend(singles)
            if max_pairs > 0 and len(singles) > 1:
                risks = oracles.head_fn(protein_id, [s.seq for s in singles])
                tw = target_window_idx_fn(block) if target_window_idx_fn is not None else None
                proxy = np.array(
                    [
                        (np.asarray(row)[tw].max() if tw else np.asarray(row).max())
                        for row in risks
                    ]
                )
                ranked = [
                    (singles[i].positions[0], singles[i].seq[singles[i].positions[0]])
                    for i in np.argsort(proxy)
                ]
                candidates.extend(enumerate_pairs(seq, ranked, max_pairs=max_pairs))
        return candidates

    return propose


def _head_metrics_batch(oracles, protein_id: str, sequences):
    """Score complete sequences through the rich Head interface and bind row order."""
    if not callable(getattr(oracles, "head_score_fn", None)):
        raise ValueError("--target-source head requires an oracle head_score_fn")
    sequences = list(sequences)
    scores = list(oracles.head_score_fn(protein_id, sequences))
    if len(scores) != len(sequences):
        raise RuntimeError(
            f"Head returned {len(scores)} rows for {len(sequences)} sequences"
        )
    return [
        head_refinement_metrics(
            score,
            protein_id=protein_id,
            sequence=sequence,
        )
        for sequence, score in zip(sequences, scores)
    ]


def _head_candidate_round(oracles, protein_id, sequence, parent_head, anchors, args):
    """One exact two-stage Head proposal round for ceiling mode.

    Refine mode implements the same stages across the whole beam so each stage is
    scored in one batch. Ceiling has one parent, so this helper keeps its candidate
    and metric rows aligned without rescoring singles.
    """
    target = select_head_residue_targets(
        parent_head,
        anchors=anchors,
        threshold=args.head_residue_threshold,
        max_positions=args.head_max_target_positions,
    )
    singles = enumerate_singles(sequence, target.positions)
    if not singles:
        return target, [], []
    single_metrics = _head_metrics_batch(
        oracles, protein_id, [candidate.seq for candidate in singles]
    )
    metrics_by_sequence = {
        candidate.seq: metrics for candidate, metrics in zip(singles, single_metrics)
    }
    choices = select_head_single_aa_choices(
        singles,
        metrics_by_sequence,
        position_order=target.positions,
        max_aa_per_position=args.head_aa_per_position,
    )
    doubles = enumerate_round_robin_doubles(
        sequence,
        choices,
        position_order=target.positions,
        max_candidates=args.max_pairs,
    )
    double_metrics = _head_metrics_batch(
        oracles, protein_id, [candidate.seq for candidate in doubles]
    ) if doubles else []
    return target, singles + doubles, single_metrics + double_metrics


def _head_target_payload(target, *, suffix: str) -> dict:
    """Stable output columns for one residue-target selection."""
    return {
        f"head_target_positions_0b{suffix}": list(target.positions),
        f"head_target_hotspot_values{suffix}": list(target.hotspot_values),
        f"n_head_residues_above_threshold{suffix}": int(target.n_above_threshold),
        f"n_head_local_maxima_pre_cap{suffix}": int(
            target.n_local_maxima_pre_cap
        ),
        f"n_head_target_positions{suffix}": len(target.positions),
    }


# --------------------------------------------------------------------------- #
# Seed selection + anchor resolution
# --------------------------------------------------------------------------- #
def _select_seeds(generated: pd.DataFrame, imm_head: pd.DataFrame, protein_id: str,
                  n_seeds: int) -> list[dict]:
    """Top-``n_seeds`` designs for a protein by ascending global_risk (best de-immunized).

    When ``imm_head is None`` (a clean seq-only input with no global_risk), fall back to
    the provided designs in stable ``design_idx`` order — refine what the caller handed."""
    g = generated[generated["protein_id"].astype(str) == protein_id]
    if g.empty:
        return []
    if imm_head is None:
        return g.sort_values("design_idx").head(n_seeds).to_dict("records")
    risk = imm_head[imm_head["protein_id"].astype(str) == protein_id][
        ["design_idx", "global_risk"]
    ]
    merged = g.merge(risk, on="design_idx", how="left").sort_values(
        "global_risk", na_position="last"
    )
    return merged.head(n_seeds).to_dict("records")


def _anchors_for(manifest, protein_id: str, sequence: str) -> set:
    """Hard-anchor positions for a protein (fail-fast identity validation), or empty."""
    if manifest is None or not manifest.has_protein(protein_id):
        return set()
    constraint = manifest.constraint_for_protein(protein_id)
    constraint.validate_against_sequence(sequence)  # fail-fast AA-identity assert
    return set(constraint.hard_anchor_indices)


def _load_catalytic_indices_by_protein(manifest_path, manifest) -> dict[str, set[int]]:
    """Resolve the protocol's direct-functional catalytic subset as 0-based indices.

    Safety-max manifests carry a top-level UniProt 1-based direct-functional union and are
    intentionally single-protein. Historical v0 manifests define hard anchors as the direct
    functional set, so those anchors are the explicit compatibility mapping.
    """
    import yaml

    with open(manifest_path) as handle:
        raw = yaml.safe_load(handle) or {}
    provenance = raw.get("annotation_provenance") or {}
    direct_1b = provenance.get("direct_functional_union_uniprot_1b")
    if direct_1b:
        if len(manifest.entries) != 1:
            raise ValueError(
                "top-level direct_functional_union_uniprot_1b is only unambiguous for a "
                "single-protein constraint manifest"
            )
        protein_id = next(iter(manifest.entries))
        indices = {int(value) - 1 for value in direct_1b}
        if not indices or min(indices) < 0:
            raise ValueError("direct_functional_union_uniprot_1b must contain positive indices")
        anchors = set(manifest.constraint_for_protein(protein_id).hard_anchor_indices)
        missing = sorted(indices - anchors)
        if missing:
            raise ValueError(
                "direct-functional catalytic indices are not protected hard anchors: "
                f"{missing}"
            )
        return {protein_id: indices}

    if str(manifest.schema_version).endswith("_v0"):
        return {
            protein_id: set(constraint.hard_anchor_indices)
            for protein_id, constraint in manifest.entries.items()
            if constraint.hard_anchor_indices
        }

    raise ValueError(
        "constraint manifest lacks annotation_provenance."
        "direct_functional_union_uniprot_1b; cannot compute cat_max_scRMSD"
    )


def _catalytic_max_sidechain_rmsd(v2_result, catalytic_indices: set[int]) -> float | None:
    rows = [
        row for row in v2_result.residue_metrics
        if int(row.residue_idx) in catalytic_indices
    ]
    if {int(row.residue_idx) for row in rows} != set(catalytic_indices):
        return None
    if any(row.match_status not in {"ok", "no_sidechain"} for row in rows):
        return None
    values = [
        float(row.sidechain_rmsd)
        for row in rows
        if row.match_status == "ok" and row.sidechain_rmsd is not None
    ]
    if not values or not all(np.isfinite(value) for value in values):
        return None
    return max(values)


# --------------------------------------------------------------------------- #
# Ceiling mode: score every proposed candidate for one round (E0 dataset)
# --------------------------------------------------------------------------- #
def _run_ceiling_seed(oracles, protein_id, seed_seq, design_idx, anchors, args) -> list[dict]:
    tw_fn = _make_target_window_idx_fn(*oracles.window_coords_fn(seed_seq))
    seed_rows = oracles.nmp_fn(protein_id, [seed_seq])[0]
    seed_cores = extract_target_cores(seed_rows, anchors=anchors, strong_rank=args.strong_rank)
    seed_count = len(seed_cores)
    seed_margin = rank_margin_mass([x.best_rank for x in seed_cores],
                                   strong_rank=args.strong_rank, margin_band=args.margin_band)
    seed_metrics = oracles.struct_fn(protein_id, seed_seq)
    propose_fn = make_propose_fn(oracles, protein_id, anchors, max_pairs=args.max_pairs,
                                 target_window_idx_fn=tw_fn)
    pool = propose_fn(seed_seq, seed_cores)
    if not pool:
        return []
    risks = oracles.head_fn(protein_id, [c.seq for c in pool])
    tw = tw_fn(seed_cores)
    head_proxy = [
        float(np.asarray(row)[tw].max() if tw else np.asarray(row).max()) for row in risks
    ]
    cand_rows = oracles.nmp_fn(protein_id, [c.seq for c in pool])
    out = []
    for c, hp, rows in zip(pool, head_proxy, cand_rows):
        cores = extract_target_cores(rows, anchors=anchors, strong_rank=args.strong_rank)
        count = len(cores)
        margin = rank_margin_mass([x.best_rank for x in cores], strong_rank=args.strong_rank,
                                  margin_band=args.margin_band)
        # Soft improvement (PLAN §1): count drop, OR count tie with lower margin mass.
        improving = count < seed_count or (count == seed_count and margin < seed_margin)
        sc_tm = plddt = passed = None
        global_ca_rmsd = active_site_sidechain_rmsd = None
        max_anchor_sidechain_rmsd = max_anchor_atom_distance = None
        active_site_complete = active_site_min_plddt = None
        cat_max_scrmsd = predicted_active_site_min_plddt = None
        if improving:
            m = oracles.struct_fn(protein_id, c.seq)
            sc_tm, plddt = m.scTM, m.pLDDT
            global_ca_rmsd = getattr(m, "global_ca_RMSD", None)
            active_site_sidechain_rmsd = getattr(m, "active_site_sidechain_RMSD", None)
            max_anchor_sidechain_rmsd = getattr(m, "max_anchor_sidechain_RMSD", None)
            max_anchor_atom_distance = getattr(m, "max_anchor_atom_distance", None)
            active_site_complete = getattr(m, "active_site_complete", None)
            active_site_min_plddt = getattr(m, "active_site_min_pLDDT", None)
            passed, _ = structure_gate(seed_metrics, m, scTM_eps=args.scTM_eps,
                                       scTM_min=args.gate_scTM_min,
                                       cat_max_scRMSD_max=args.gate_cat_max_scRMSD_max,
                                       predicted_active_site_min_pLDDT_min=(
                                           args.gate_predicted_active_site_min_pLDDT_min
                                       ),
                                       scRMSD_max=args.scRMSD_max,
                                       active_site_RMSD_max=args.active_site_RMSD_max,
                                       max_anchor_sidechain_RMSD_max=(
                                           args.max_anchor_sidechain_RMSD_max
                                       ))
            cat_max_scrmsd = getattr(m, "cat_max_scRMSD", None)
            predicted_active_site_min_plddt = getattr(
                m, "predicted_active_site_min_pLDDT", None
            )
        out.append({
            "protein_id": protein_id, "design_idx": int(design_idx), "muts": c.desc,
            "positions": list(c.positions), "head_proxy": hp, "core_count": count,
            "rank_margin_mass": margin, "scTM": sc_tm, "pLDDT": plddt, "passed": passed,
            "global_ca_RMSD": global_ca_rmsd,
            "active_site_sidechain_RMSD": active_site_sidechain_rmsd,
            "max_anchor_sidechain_RMSD": max_anchor_sidechain_rmsd,
            "max_anchor_atom_distance": max_anchor_atom_distance,
            "active_site_complete": active_site_complete,
            "active_site_min_pLDDT": active_site_min_plddt,
            "cat_max_scRMSD": cat_max_scrmsd,
            "predicted_active_site_min_pLDDT": predicted_active_site_min_plddt,
            "eliminates": bool(count < seed_count),
        })
    return out


def _run_head_ceiling_seed(
    oracles, protein_id, seed_seq, design_idx, anchors, args
) -> list[dict]:
    """Head-only ceiling over thresholded residue-local maxima."""
    seed_head = _head_metrics_batch(oracles, protein_id, [seed_seq])[0]
    seed_metrics = oracles.struct_fn(protein_id, seed_seq)
    seed_target, pool, head_rows = _head_candidate_round(
        oracles, protein_id, seed_seq, seed_head, anchors, args
    )
    if not pool:
        return []
    out = []
    for candidate, head in zip(pool, head_rows):
        improving = head_objective_dominates(head, seed_head)
        target = select_head_residue_targets(
            head,
            anchors=anchors,
            threshold=args.head_residue_threshold,
            max_positions=args.head_max_target_positions,
        )
        metrics = None
        passed = None
        if improving:
            metrics = oracles.struct_fn(protein_id, candidate.seq)
            passed, _reason = structure_gate(
                seed_metrics,
                metrics,
                scTM_eps=args.scTM_eps,
                scTM_min=args.gate_scTM_min,
                cat_max_scRMSD_max=args.gate_cat_max_scRMSD_max,
                predicted_active_site_min_pLDDT_min=(
                    args.gate_predicted_active_site_min_pLDDT_min
                ),
                scRMSD_max=args.scRMSD_max,
                active_site_RMSD_max=args.active_site_RMSD_max,
                max_anchor_sidechain_RMSD_max=args.max_anchor_sidechain_RMSD_max,
            )
        out.append({
            "protein_id": protein_id,
            "design_idx": int(design_idx),
            "target_source": "head",
            "head_candidate_stage": (
                "single" if len(candidate.positions) == 1 else "double"
            ),
            "muts": candidate.desc,
            "positions": list(candidate.positions),
            "head_proxy": head.global_risk,
            "head_global_risk_before": seed_head.global_risk,
            "head_global_risk": head.global_risk,
            "head_positive_mass_density_before": seed_head.positive_mass_density,
            "head_positive_mass_density": head.positive_mass_density,
            "head_positive_mass_before": seed_head.positive_mass,
            "head_positive_mass": head.positive_mass,
            "head_positive_hotspot_positions_before": (
                seed_head.n_positive_hotspot_positions
            ),
            "head_positive_hotspot_positions": head.n_positive_hotspot_positions,
            "head_residue_threshold": float(args.head_residue_threshold),
            "head_max_target_positions": int(args.head_max_target_positions),
            "head_aa_per_position": int(args.head_aa_per_position),
            "head_max_double_candidates": int(args.max_pairs),
            **_head_target_payload(seed_target, suffix="_before"),
            **_head_target_payload(target, suffix=""),
            "core_count": None,
            "rank_margin_mass": None,
            "scTM": None if metrics is None else metrics.scTM,
            "pLDDT": None if metrics is None else metrics.pLDDT,
            "passed": passed,
            "global_ca_RMSD": (
                None if metrics is None else getattr(metrics, "global_ca_RMSD", None)
            ),
            "active_site_sidechain_RMSD": (
                None
                if metrics is None
                else getattr(metrics, "active_site_sidechain_RMSD", None)
            ),
            "max_anchor_sidechain_RMSD": (
                None
                if metrics is None
                else getattr(metrics, "max_anchor_sidechain_RMSD", None)
            ),
            "max_anchor_atom_distance": (
                None
                if metrics is None
                else getattr(metrics, "max_anchor_atom_distance", None)
            ),
            "active_site_complete": (
                None if metrics is None else getattr(metrics, "active_site_complete", None)
            ),
            "active_site_min_pLDDT": (
                None if metrics is None else getattr(metrics, "active_site_min_pLDDT", None)
            ),
            "cat_max_scRMSD": (
                None if metrics is None else getattr(metrics, "cat_max_scRMSD", None)
            ),
            "predicted_active_site_min_pLDDT": (
                None
                if metrics is None
                else getattr(metrics, "predicted_active_site_min_pLDDT", None)
            ),
            "improves": bool(improving),
            "eliminates": None,
        })
    return out


# --------------------------------------------------------------------------- #
# Refine mode: full beam search per seed -> shortlist
# --------------------------------------------------------------------------- #
def _run_refine_seed(oracles, protein_id, seed_seq, orig_design_idx, seed_val, anchors, args):
    tw_fn = _make_target_window_idx_fn(*oracles.window_coords_fn(seed_seq))
    propose_fn = make_propose_fn(oracles, protein_id, anchors, max_pairs=args.max_pairs,
                                 target_window_idx_fn=tw_fn)
    res = refine_sequence(
        protein_id, seed_seq,
        propose_fn=propose_fn, head_fn=oracles.head_fn, nmp_fn=oracles.nmp_fn,
        struct_fn=oracles.struct_fn, anchors=anchors,
        target_window_idx_fn=tw_fn,
        strong_rank=args.strong_rank, margin_band=args.margin_band, scTM_eps=args.scTM_eps,
        scTM_min=args.gate_scTM_min,
        cat_max_scRMSD_max=args.gate_cat_max_scRMSD_max,
        predicted_active_site_min_pLDDT_min=(
            args.gate_predicted_active_site_min_pLDDT_min
        ),
        scRMSD_max=args.scRMSD_max, active_site_RMSD_max=args.active_site_RMSD_max,
        max_anchor_sidechain_RMSD_max=args.max_anchor_sidechain_RMSD_max,
        topB=args.topB, beam_width=args.beam_width, max_rounds=args.max_rounds,
        patience=args.patience, max_path_mutations=args.max_path_mutations,
        refold_cap=args.refold_cap, allow_structure_unknown=args.allow_structure_unknown,
        incremental_nmp=args.incremental_nmp, nmp_context_margin=args.nmp_context_margin,
        log_fn=(lambda m: print(m, flush=True)),
    )
    rich = []
    if res.shortlist:
        for entry in res.shortlist:
            rich.append({
                "protein_id": protein_id, "orig_design_idx": int(orig_design_idx),
                "sequence_original": seed_seq, "sequence_refined": entry["seq"],
                "n_mutations": len(entry["muts"].split(",")) if entry["muts"] else 0,
                "muts": entry["muts"], "core_count_before": res.seed_core_count,
                "core_count_after": entry["core_count"], "scTM_after": entry["scTM"],
                "pLDDT_after": entry["pLDDT"], "scRMSD_after": entry["scRMSD"],
                "active_site_RMSD_after": entry["active_site_RMSD"], "diverged": False,
                "global_ca_RMSD_after": entry["global_ca_RMSD"],
                "active_site_sidechain_RMSD_after": entry["active_site_sidechain_RMSD"],
                "max_anchor_sidechain_RMSD_after": entry["max_anchor_sidechain_RMSD"],
                "max_anchor_atom_distance_after": entry["max_anchor_atom_distance"],
                "active_site_complete_after": entry["active_site_complete"],
                "active_site_min_pLDDT_after": entry["active_site_min_pLDDT"],
                "cat_max_scRMSD_after": entry["cat_max_scRMSD"],
                "predicted_active_site_min_pLDDT_after": entry[
                    "predicted_active_site_min_pLDDT"
                ],
            })
    else:
        # No accepted refinement: emit a no-op row so every seed appears for re-eval.
        m = res.best_structure
        rich.append({
            "protein_id": protein_id, "orig_design_idx": int(orig_design_idx),
            "sequence_original": seed_seq, "sequence_refined": res.best_seq,
            "n_mutations": 0, "muts": "", "core_count_before": res.seed_core_count,
            "core_count_after": res.best_core_count, "scTM_after": getattr(m, "scTM", None),
            "pLDDT_after": getattr(m, "pLDDT", None), "scRMSD_after": getattr(m, "scRMSD", None),
            "active_site_RMSD_after": getattr(m, "active_site_RMSD", None),
            "global_ca_RMSD_after": getattr(m, "global_ca_RMSD", None),
            "active_site_sidechain_RMSD_after": getattr(
                m, "active_site_sidechain_RMSD", None
            ),
            "max_anchor_sidechain_RMSD_after": getattr(
                m, "max_anchor_sidechain_RMSD", None
            ),
            "max_anchor_atom_distance_after": getattr(m, "max_anchor_atom_distance", None),
            "active_site_complete_after": getattr(m, "active_site_complete", None),
            "active_site_min_pLDDT_after": getattr(m, "active_site_min_pLDDT", None),
            "cat_max_scRMSD_after": getattr(m, "cat_max_scRMSD", None),
            "predicted_active_site_min_pLDDT_after": getattr(
                m, "predicted_active_site_min_pLDDT", None
            ),
            "diverged": bool(res.diverged),
        })
    trace = [{"protein_id": protein_id, "orig_design_idx": int(orig_design_idx), **t}
             for t in res.trace]
    return rich, trace, seed_val


def _mutation_summary(reference: str, candidate: str) -> tuple[int, str]:
    edits = [
        (idx, before, after)
        for idx, (before, after) in enumerate(zip(reference, candidate))
        if before != after
    ]
    return len(edits), ",".join(f"{before}{idx}{after}" for idx, before, after in edits)


def _run_head_refine_seed(
    oracles, protein_id, seed_seq, orig_design_idx, seed_val, anchors, args
):
    result = refine_sequence_head(
        protein_id,
        seed_seq,
        head_score_fn=oracles.head_score_fn,
        struct_fn=oracles.struct_fn,
        anchors=anchors,
        residue_threshold=args.head_residue_threshold,
        max_target_positions=args.head_max_target_positions,
        aa_per_position=args.head_aa_per_position,
        max_double_candidates=args.max_pairs,
        scTM_eps=args.scTM_eps,
        scTM_min=args.gate_scTM_min,
        cat_max_scRMSD_max=args.gate_cat_max_scRMSD_max,
        predicted_active_site_min_pLDDT_min=(
            args.gate_predicted_active_site_min_pLDDT_min
        ),
        scRMSD_max=args.scRMSD_max,
        active_site_RMSD_max=args.active_site_RMSD_max,
        max_anchor_sidechain_RMSD_max=args.max_anchor_sidechain_RMSD_max,
        topB=args.topB,
        beam_width=args.beam_width,
        max_rounds=args.max_rounds,
        patience=args.patience,
        max_path_mutations=args.max_path_mutations,
        refold_cap=args.refold_cap,
        allow_structure_unknown=args.allow_structure_unknown,
        log_fn=(lambda message: print(message, flush=True)),
    )

    entries = result.shortlist or [{
        "seq": seed_seq,
        "head": result.seed_head,
        "structure": result.seed_structure,
    }]
    seed_target = select_head_residue_targets(
        result.seed_head,
        anchors=anchors,
        threshold=args.head_residue_threshold,
        max_positions=args.head_max_target_positions,
    )
    rich = []
    for entry in entries:
        sequence = entry["seq"]
        head = entry["head"]
        metrics = entry["structure"]
        target = select_head_residue_targets(
            head,
            anchors=anchors,
            threshold=args.head_residue_threshold,
            max_positions=args.head_max_target_positions,
        )
        n_mutations, mutations = _mutation_summary(seed_seq, sequence)
        rich.append({
            "protein_id": protein_id,
            "orig_design_idx": int(orig_design_idx),
            "target_source": "head",
            "sequence_original": seed_seq,
            "sequence_refined": sequence,
            "n_mutations": n_mutations,
            "muts": mutations,
            "core_count_before": None,
            "core_count_after": None,
            "head_global_risk": head.global_risk,
            "head_global_risk_before": result.seed_head.global_risk,
            "head_global_risk_after": head.global_risk,
            "head_positive_mass_density": head.positive_mass_density,
            "head_positive_mass_before": result.seed_head.positive_mass,
            "head_positive_mass_after": head.positive_mass,
            "head_positive_mass_density_before": (
                result.seed_head.positive_mass_density
            ),
            "head_positive_mass_density_after": head.positive_mass_density,
            "head_positive_hotspot_positions_before": (
                result.seed_head.n_positive_hotspot_positions
            ),
            "head_positive_hotspot_positions_after": (
                head.n_positive_hotspot_positions
            ),
            "head_residue_threshold": float(args.head_residue_threshold),
            "head_max_target_positions": int(args.head_max_target_positions),
            "head_aa_per_position": int(args.head_aa_per_position),
            "head_max_double_candidates": int(args.max_pairs),
            **_head_target_payload(seed_target, suffix="_before"),
            **_head_target_payload(target, suffix="_after"),
            "scTM_after": getattr(metrics, "scTM", None),
            "pLDDT_after": getattr(metrics, "pLDDT", None),
            "scRMSD_after": getattr(metrics, "scRMSD", None),
            "active_site_RMSD_after": getattr(metrics, "active_site_RMSD", None),
            "global_ca_RMSD_after": getattr(metrics, "global_ca_RMSD", None),
            "active_site_sidechain_RMSD_after": getattr(
                metrics, "active_site_sidechain_RMSD", None
            ),
            "max_anchor_sidechain_RMSD_after": getattr(
                metrics, "max_anchor_sidechain_RMSD", None
            ),
            "max_anchor_atom_distance_after": getattr(
                metrics, "max_anchor_atom_distance", None
            ),
            "active_site_complete_after": getattr(metrics, "active_site_complete", None),
            "active_site_min_pLDDT_after": getattr(
                metrics, "active_site_min_pLDDT", None
            ),
            "cat_max_scRMSD_after": getattr(metrics, "cat_max_scRMSD", None),
            "predicted_active_site_min_pLDDT_after": getattr(
                metrics, "predicted_active_site_min_pLDDT", None
            ),
            "diverged": bool(result.diverged),
        })
    if args.official:
        rich = _retain_official_head_rows(rich)
    trace = [
        {
            "protein_id": protein_id,
            "orig_design_idx": int(orig_design_idx),
            "target_source": "head",
            **row,
        }
        for row in result.trace
    ]
    return rich, trace, seed_val


def _retain_official_head_rows(rows):
    """Keep the exact per-seed two-axis Head front in official retention mode.

    This is an output-only reduction.  It runs after the complete search has finished, so it
    cannot alter the beam, proposal stream, structure admissions, or random-number consumption.
    """

    rows = list(rows)
    if not rows:
        return []
    metrics = [
        SimpleNamespace(
            global_risk=float(row["head_global_risk_after"]),
            positive_mass_density=float(row["head_positive_mass_density_after"]),
        )
        for row in rows
    ]
    indices = head_first_pareto_front(metrics)
    return [rows[index] for index in indices]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _resolve_proteins(args, generated: pd.DataFrame) -> list[str]:
    if args.proteins and args.proteins != "all":
        return [p.strip() for p in args.proteins.split(",") if p.strip()]
    return sorted(generated["protein_id"].astype(str).unique())


def _seed_design_idx(row, running: int) -> int:
    """Provenance design_idx for a seed-table row: explicit design_idx, else canonical
    design_NNNN, else a running per-protein index."""
    if "design_idx" in row and pd.notna(row["design_idx"]):
        return int(row["design_idx"])
    match = re.match(r"^design_(\d+)$", str(row.get("design_id", "")))
    return int(match.group(1)) if match else running


def _seeds_from_table(seed_table: str, proteins_arg: str,
                      cost_col: str | None = None) -> dict[str, list[dict]]:
    """Seeds from a self-contained protein_id/design_id/sequence parquet (bypasses
    --run-dir best-of-N selection; refines every listed row). When ``cost_col`` is a
    present column, each seed carries ``_cost`` for difficulty-balanced array sharding."""
    df = pd.read_parquet(seed_table)
    for col in ("protein_id", "sequence"):
        if col not in df.columns:
            raise ValueError(f"--seed-table missing required column {col!r}")
    if proteins_arg and proteins_arg != "all":
        want = {p.strip() for p in proteins_arg.split(",")}
        df = df[df["protein_id"].astype(str).isin(want)]
    has_cost = bool(cost_col) and cost_col in df.columns
    seeds_by_protein: dict[str, list[dict]] = {}
    has_seed = "seed" in df.columns
    for _, row in df.iterrows():
        pid = str(row["protein_id"])
        lst = seeds_by_protein.setdefault(pid, [])
        seed = {
            "sequence": str(row["sequence"]),
            "design_idx": _seed_design_idx(row, len(lst)),
            "seed": int(row["seed"]) if has_seed and pd.notna(row.get("seed")) else -1,
            "_cost": float(row[cost_col]) if has_cost and pd.notna(row.get(cost_col)) else None,
        }
        for column in _SEED_PROVENANCE_COLUMNS:
            if column in df.columns and pd.notna(row.get(column)):
                seed[column] = row[column]
        lst.append(seed)
    return seeds_by_protein


def _shard_seeds(seeds_by_protein: dict[str, list[dict]], n_shards: int, shard_idx: int,
                 shard_by: str) -> dict[str, list[dict]]:
    """Partition the flattened seed list across array tasks (deterministic — every task
    recomputes the identical partition). ``balanced`` = LPT bin-pack by each seed's
    ``_cost`` (falls back to stride if any cost is missing); ``stride`` = round-robin."""
    flat = [(pid, s) for pid in sorted(seeds_by_protein) for s in seeds_by_protein[pid]]
    if shard_by == "balanced" and flat and all(s.get("_cost") is not None for _, s in flat):
        order = sorted(range(len(flat)), key=lambda i: -float(flat[i][1]["_cost"]))
        load = [0.0] * n_shards
        assign: dict[int, int] = {}
        for i in order:
            j = min(range(n_shards), key=lambda s: load[s])
            load[j] += max(float(flat[i][1]["_cost"]), 1.0)
            assign[i] = j
        keep = [flat[i] for i in range(len(flat)) if assign[i] == shard_idx]
    else:
        keep = [flat[i] for i in range(len(flat)) if i % n_shards == shard_idx]
    out: dict[str, list[dict]] = {}
    for pid, s in keep:
        out.setdefault(pid, []).append(s)
    return out


def _all_seeds(generated: pd.DataFrame, protein_id: str) -> list[dict]:
    """Every design for a protein — used when --eval-immune-dir is omitted (no best-of-N)."""
    g = generated[generated["protein_id"].astype(str) == protein_id]
    has_seed = "seed" in g.columns
    return [
        {"sequence": str(r["sequence"]), "design_idx": int(r["design_idx"]),
         "seed": int(r["seed"]) if has_seed and pd.notna(r.get("seed")) else -1}
        for _, r in g.iterrows()
    ]


def _gather_seeds(args) -> tuple[dict[str, list[dict]], list[str]]:
    """Resolve per-protein seed lists from --seed-table, or --run-dir (with optional
    --eval-immune-dir best-of-N; omit it to refine every design)."""
    if args.seed_table:
        seeds_by_protein = _seeds_from_table(args.seed_table, args.proteins, args.shard_cost_col)
    else:
        if not args.run_dir:
            raise ValueError("provide either --seed-table or --run-dir")
        generated = _load_sharded(Path(args.run_dir), "generation", "generated.parquet")
        imm_head = (_load_sharded(Path(args.eval_immune_dir), ".", "imm_head.parquet")
                    if args.eval_immune_dir else None)
        seeds_by_protein = {
            pid: (_select_seeds(generated, imm_head, pid, args.seeds_per_protein)
                  if imm_head is not None else _all_seeds(generated, pid))
            for pid in _resolve_proteins(args, generated)
        }
    if args.n_shards > 1:
        seeds_by_protein = _shard_seeds(seeds_by_protein, args.n_shards, args.shard_idx,
                                        args.shard_by)
    return seeds_by_protein, sorted(seeds_by_protein)


def run_refinement(args, oracles: Oracles) -> int:
    if args.official and (args.mode != "refine" or args.target_source != "head"):
        raise ValueError("--official is supported only by Head refine mode")
    if args.final_candidates_per_protein is not None and (
        isinstance(args.final_candidates_per_protein, bool)
        or int(args.final_candidates_per_protein) < 1
    ):
        raise ValueError("--final-candidates-per-protein must be an integer >= 1")
    if args.target_source == "head" and not callable(
        getattr(oracles, "head_score_fn", None)
    ):
        raise ValueError("--target-source head requires an oracle head_score_fn")
    if args.target_source == "head":
        invalid_positive = [
            name for name in (
                "beam_width", "patience", "head_max_target_positions",
                "head_aa_per_position",
            )
            if int(getattr(args, name)) < 1
        ]
        invalid_nonnegative = [
            name for name in (
                "max_rounds", "max_pairs", "max_path_mutations", "refold_cap"
            )
            if int(getattr(args, name)) < 0
        ]
        if args.topB is not None and int(args.topB) < 0:
            invalid_nonnegative.append("topB")
        if (
            not np.isfinite(float(args.head_residue_threshold))
            or float(args.head_residue_threshold) < 0.0
        ):
            invalid_nonnegative.append("head_residue_threshold")
        if int(args.head_aa_per_position) > 19:
            invalid_positive.append("head_aa_per_position")
        if invalid_positive or invalid_nonnegative:
            raise ValueError(
                "Head refinement requires beam_width/patience/head_max_target_positions "
                ">= 1, head_aa_per_position in [1, 19], and "
                "head_residue_threshold/max_rounds/max_pairs/max_path_mutations/"
                "refold_cap/topB >= 0"
            )
    manifest = load_constraint_manifest(args.constraint_manifest) if args.constraint_manifest else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds_by_protein, proteins = _gather_seeds(args)
    n_seeds = sum(len(v) for v in seeds_by_protein.values())
    print(
        f"[refine] mode={args.mode} proteins={len(proteins)} seeds={n_seeds} "
        f"source={'seed-table' if args.seed_table else 'run-dir'} "
        f"target_source={args.target_source} topB={args.topB} "
        f"beam={args.beam_width} refold_cap={args.refold_cap} strong_rank={args.strong_rank} margin_band={args.margin_band} "
        f"structure_gate={args.structure_gate_profile} "
        f"scTM_min={args.gate_scTM_min} cat_max_scRMSD_max={args.gate_cat_max_scRMSD_max} "
        f"active_site_min_pLDDT_min={args.gate_predicted_active_site_min_pLDDT_min} "
        f"scTM_eps={args.scTM_eps} max_pairs={args.max_pairs} "
        f"head_residue_threshold={args.head_residue_threshold} "
        f"head_max_target_positions={args.head_max_target_positions} "
        f"head_aa_per_position={args.head_aa_per_position} "
        f"max_path_mutations={args.max_path_mutations} "
        f"official={args.official} "
        f"final_candidates_per_protein={args.final_candidates_per_protein}",
        flush=True,
    )

    if args.mode == "ceiling":
        rows = []
        for pid in proteins:
            for seed in seeds_by_protein[pid]:
                anchors = _anchors_for(manifest, pid, str(seed["sequence"]))
                runner = (
                    _run_head_ceiling_seed
                    if args.target_source == "head"
                    else _run_ceiling_seed
                )
                rows.extend(runner(
                    oracles, pid, str(seed["sequence"]), seed["design_idx"], anchors, args))
        (out_dir / "ceiling").mkdir(exist_ok=True)
        pd.DataFrame(rows).to_parquet(out_dir / "ceiling" / "candidates.parquet", index=False)
        print(f"[refine] ceiling: {len(rows)} candidate rows -> ceiling/candidates.parquet", flush=True)
        _write_config(out_dir / "ceiling" / "refine_config.json", args)
        return 0

    # refine mode
    refined_dir = out_dir / "refined"
    refined_dir.mkdir(exist_ok=True)
    _write_config(refined_dir / "refine_config.json", args)
    rich_rows, trace_rows, eval_rows = [], [], []
    per_protein_idx: dict[str, int] = {}
    n_done = 0
    for pid in proteins:
        for seed in seeds_by_protein[pid]:
            anchors = _anchors_for(manifest, pid, str(seed["sequence"]))
            runner = (
                _run_head_refine_seed
                if args.target_source == "head"
                else _run_refine_seed
            )
            rich, trace, seed_val = runner(
                oracles, pid, str(seed["sequence"]), seed["design_idx"],
                int(seed.get("seed", -1)), anchors, args)
            for r in rich:
                if args.official or args.final_candidates_per_protein is not None:
                    for column in _SEED_PROVENANCE_COLUMNS:
                        if column in seed:
                            r[column] = seed[column]
                didx = per_protein_idx.get(pid, 0)
                per_protein_idx[pid] = didx + 1
                r["design_idx"] = didx
                rich_rows.append(r)
                eval_rows.append({
                    "protein_id": pid, "design_idx": didx, "sequence": r["sequence_refined"],
                    "seed": int(seed_val), "wall_seconds": 0.0,
                })
            trace_rows.extend(trace)
            n_done += 1
            # Incremental, ATOMIC flush after every seed: a walltime/GPU-idle kill keeps all
            # completed seeds' designs (the driver otherwise only wrote at the very end).
            _flush_refine_outputs(refined_dir, rich_rows, eval_rows, trace_rows)
            print(f"[refine] seed {n_done} done ({pid}) -> {len(rich_rows)} designs so far",
                  flush=True)

    rich_df = _flush_refine_outputs(refined_dir, rich_rows, eval_rows, trace_rows)
    if args.target_source == "head":
        n_improved = int(
            (
                (rich_df["head_global_risk_after"] <= rich_df["head_global_risk_before"])
                & (
                    rich_df["head_positive_mass_density_after"]
                    <= rich_df["head_positive_mass_density_before"]
                )
                & (
                    (rich_df["head_global_risk_after"] < rich_df["head_global_risk_before"])
                    | (
                        rich_df["head_positive_mass_density_after"]
                        < rich_df["head_positive_mass_density_before"]
                    )
                )
            ).sum()
        ) if len(rich_df) else 0
        detail = f"{n_improved} Pareto-improving Head rows"
    else:
        n0 = int((rich_df["core_count_after"] == 0).sum()) if len(rich_df) else 0
        detail = f"{n0} reached count 0"
    print(
        f"[refine] refine: {len(rich_rows)} refined rows ({detail}) "
        f"-> refined/refined_designs.parquet + evaluator_ready.parquet",
        flush=True,
    )
    return 0


def _write_config(path: Path, args) -> None:
    with open(path, "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)


def _atomic_write(refined_dir: Path, df: pd.DataFrame, name: str) -> None:
    """Write ``df`` to ``refined_dir/name`` via a temp file + atomic rename, so a walltime /
    GPU-idle kill mid-write never leaves a half-written parquet."""
    tmp = refined_dir / (name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(refined_dir / name)


def _flush_refine_outputs(refined_dir: Path, rich_rows, eval_rows, trace_rows) -> pd.DataFrame:
    """Atomically (re)write the cumulative refine outputs so a mid-run kill/timeout keeps
    every completed seed's designs (write to a temp then rename). Returns the rich DataFrame."""
    rich_df = pd.DataFrame(rich_rows).drop(columns=["orig_design_idx"], errors="ignore")
    _atomic_write(refined_dir, rich_df, "refined_designs.parquet")
    _atomic_write(refined_dir, pd.DataFrame(eval_rows, columns=GENERATED_COLUMNS), "evaluator_ready.parquet")
    _atomic_write(refined_dir, pd.DataFrame(trace_rows), "refine_trace.parquet")
    return rich_df


def _free_gpu() -> None:
    """Drop pending refs and free the parent process's CUDA caching allocator."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _close_oracles(oracles: Oracles) -> None:
    closer = getattr(oracles.struct_fn, "close", None)
    if callable(closer):
        closer()


def _write_final_metrics_status(refined_dir: Path, *, ok: bool, error: str | None = None,
                                n_designs: int | None = None, n_failures: int | None = None,
                                structural_metrics_version: str | None = None) -> None:
    """Completeness marker written as the LAST step of the metrics pass (ok=True) or by main()'s
    guard on failure (ok=False). Consumers gate a full metric set on ok=true."""
    status = {"ok": ok}
    if error is not None:
        status["error"] = error
    if n_designs is not None:
        status["n_designs"] = n_designs
    if n_failures is not None:
        status["n_failures"] = n_failures
    if structural_metrics_version is not None:
        status["structural_metrics_version"] = str(structural_metrics_version)
    with open(refined_dir / "final_metrics_status.json", "w") as f:
        json.dump(status, f, indent=2, default=str)


def _run_final_metrics(args) -> None:
    """Emit the full evaluate_phase_c metric tables for EVERY final refined design by
    REUSING evaluate_phase_c's own row builders (zero schema drift), reading the just-written
    ``refined/evaluator_ready.parquet``. Runs once after the refine loop.

    The search keeps only scalar metrics (core_count, scTM) and discards the raw per-window
    NMP rows, per-residue head hotspots, and side-chain geometry; this pass
    re-derives and persists them. Every final sequence was refolded during search,
    so ``refold`` is a cache hit here (keyed on (protein_id, sequence)) and structure is cheap.

    Writes Head and canonical structure tables atomically. Legacy NMP mode additionally writes
    ``imm_nmp.parquet`` (and peptide rows under ``--imm-full``); Head mode never builds NMP and
    intentionally omits those files.
    NOTE: imm_nmp.n_strong_binders (rank_EL% < strong_binder_threshold, per-window) is a
    DIFFERENT quantity from the search objective core_count_after (distinct 9-mer cores).
    """
    refined_dir = Path(args.out_dir) / "refined"
    gen_parquet = refined_dir / "evaluator_ready.parquet"
    if not gen_parquet.exists():
        print(f"[refine] final-metrics: {gen_parquet} missing -- skipping", flush=True)
        return

    from scripts.evaluate_phase_c import (
        load_generated_designs, load_test_lookup,
        build_head_predictor, build_nmp_runner,
        evaluate_immunogenicity_rows, evaluate_structural_rows,
        _load_anchor_indices_by_protein,
    )

    gdf = load_generated_designs(gen_parquet)
    if len(gdf) == 0:
        print("[refine] final-metrics: 0 designs -- skipping", flush=True)
        return
    _test_df, test_lookup = load_test_lookup(args.test_set_parquet)
    run_nmp = args.target_source == "nmp"
    immune_outputs = "imm_head/imm_nmp" if run_nmp else "imm_head (NMP disabled)"
    print(
        f"[refine] final-metrics: {len(gdf)} designs -> {immune_outputs}/structural/"
        f"structural_residues (imm_full={args.imm_full})",
        flush=True,
    )

    # --- immunogenicity: head (global_risk/hotspots) + NMP (n_strong_binders, ...) ---
    predictor = build_head_predictor(
        checkpoint_path=args.head_checkpoint, config_dir=args.head_config_dir,
        variant_id=args.head_variant_id, device=args.head_device)
    nmp_runner = None
    if run_nmp:
        nmp_runner = build_nmp_runner(
            binary_path=args.netmhciipan_bin, batch_size=args.nmp_batch_size,
            n_workers=args.nmp_workers, timeout=args.nmp_timeout,
            max_lengths_per_call=args.nmp_max_lengths_per_call)
    head_df, nmp_df, imm_fail, imm_res_df, imm_pep_df = evaluate_immunogenicity_rows(
        gdf, predictor=predictor, nmp_runner=nmp_runner, allele=args.allele,
        strong_binder_threshold=args.strong_binder_threshold,
        nmp_batch_size=args.nmp_batch_size, hotspot_threshold=args.hotspot_threshold,
        full=args.imm_full, run_nmp=run_nmp, return_full_tables=True)
    del predictor, nmp_runner
    _free_gpu()  # release the head predictor before structural evaluation

    # Persist the immunogenicity tables NOW — the NMP re-score is the CPU-costly part of this
    # pass, and a later structural failure must not discard already-completed work.
    _atomic_write(refined_dir, head_df, "imm_head.parquet")
    if run_nmp:
        _atomic_write(refined_dir, nmp_df, "imm_nmp.parquet")
    if args.imm_full:
        _atomic_write(refined_dir, imm_res_df, "imm_head_residues.parquet")
        if run_nmp:
            _atomic_write(refined_dir, imm_pep_df, "imm_nmp_peptides.parquet")

    # --- structure: canonical v2 summary + index-addressable side-chain rows ---
    anchor_indices_by_protein = _load_anchor_indices_by_protein(
        args.constraint_manifest,
        test_lookup,
    )
    eval_refold_backend = (
        "esmfold2" if args.refold_model == "esmfold2_live" else args.refold_model
    )
    evaluated = evaluate_structural_rows(
        gdf, test_lookup, pdb_root=args.pdb_root, refold_backend=eval_refold_backend,
        device=args.head_device, tmalign_bin="TMalign",
        esmfold_cache_dir=args.refold_cache_dir, return_residue_metrics=True,
        return_v2_metrics=True,
        legacy_metrics=False,
        anchor_indices_by_protein=anchor_indices_by_protein,
    )
    structural_df, struct_fail, struct_res_df = evaluated
    _atomic_write(refined_dir, structural_df, "structural.parquet")
    _atomic_write(refined_dir, struct_res_df, "structural_residues.parquet")

    # surface per-design failures (do not silently drop them)
    failures = list(imm_fail) + list(struct_fail)
    if failures:
        with open(refined_dir / "final_metrics_failures.json", "w") as f:
            json.dump(failures, f, indent=2, default=str)
    # completeness marker (LAST write): consumers gate a full set on ok=true
    _write_final_metrics_status(
        refined_dir,
        ok=True,
        n_designs=len(gdf),
        n_failures=len(failures),
        structural_metrics_version="v2",
    )
    print(
        f"[refine] final-metrics done: imm_head={len(head_df)} "
        f"imm_nmp={len(nmp_df) if run_nmp else 'disabled'} "
        f"structural={len(structural_df)} structural_residues={len(struct_res_df)} "
        f"failures={len(failures)} -> {refined_dir}",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Real oracle construction (heavy deps; bypassed by the smoke test)
# --------------------------------------------------------------------------- #
def build_oracles(args) -> Oracles:
    """Build real Head/structure callables and NMP only for the legacy target mode.

    Only place that touches torch / NetMHCIIpan / refold models. See PLAN_RF_REFINE R4(b).
    """
    required = {
        "--refold-cache-dir": args.refold_cache_dir,     # refold pdb_path is None without it
        "--test-set-parquet": args.test_set_parquet,      # resolve_structure_path needs the row
        "--pdb-root": args.pdb_root,                       # reference backbone for TMalign
        "--head-checkpoint": args.head_checkpoint,         # head predictor weights
        "--head-config-dir": args.head_config_dir,         # head model/ablation/inference yamls
    }
    if args.target_source == "nmp":
        required["--netmhciipan-bin"] = args.netmhciipan_bin
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"build_oracles requires: {', '.join(missing)}")
    _validate_structure_gate_config(args)
    backend_options = {}
    if args.refold_model == "esmfold2_live":
        if not args.esmfold2_site_packages:
            raise ValueError(
                "--refold-model esmfold2_live requires --esmfold2-site-packages"
            )
        if min(
            args.esmfold2_num_loops,
            args.esmfold2_num_sampling_steps,
            args.esmfold2_num_diffusion_samples,
        ) < 1:
            raise ValueError("ESMFold2 loop/step/sample counts must all be positive")
        backend_options = {
            "site_packages": args.esmfold2_site_packages,
            "model_name": args.esmfold2_model,
            "num_loops": args.esmfold2_num_loops,
            "num_sampling_steps": args.esmfold2_num_sampling_steps,
            "num_diffusion_samples": args.esmfold2_num_diffusion_samples,
            "seed": args.esmfold2_seed,
        }
    requested_structure_metrics = _parse_refinement_structure_metrics(
        args.refinement_structure_metrics
    )
    evaluation_structure_metrics = set(requested_structure_metrics)
    if args.structure_gate_profile == "protocol":
        evaluation_structure_metrics.update({"plddt", "sidechain", "residue_metrics"})
    if args.active_site_RMSD_max is not None:
        raise ValueError(
            "--active-site-RMSD-max is the retired C-alpha active-site gate and has no "
            "real standalone oracle; use --max-anchor-sidechain-RMSD-max"
        )
    if (
        args.max_anchor_sidechain_RMSD_max is not None
        and (
            not np.isfinite(args.max_anchor_sidechain_RMSD_max)
            or args.max_anchor_sidechain_RMSD_max <= 0.0
        )
    ):
        raise ValueError("--max-anchor-sidechain-RMSD-max must be finite and positive")
    if (
        args.max_anchor_sidechain_RMSD_max is not None
        and "sidechain" not in requested_structure_metrics
    ):
        raise ValueError(
            "--max-anchor-sidechain-RMSD-max requires "
            "--refinement-structure-metrics sidechain"
        )
    if args.max_anchor_sidechain_RMSD_max is not None and not args.constraint_manifest:
        raise ValueError(
            "--max-anchor-sidechain-RMSD-max requires --constraint-manifest"
        )

    # --- Head (reuse the existing build_head_scorer helper) ---
    from scripts.head_runtime import build_head_scorer  # ControllerSetup-style seam
    from types import SimpleNamespace
    import yaml

    head_config_dir = Path(args.head_config_dir)
    inf = yaml.safe_load(open(head_config_dir / "inference.yaml")) or {}
    inf_sec = inf.get("inference", {}) if isinstance(inf, dict) else {}
    # build_head_scorer reads setup.{head_config_dir, head_checkpoint(Path), head_variant_id,
    # head_device, allele, head_allele_idx, config.head.score_scale, head_window_batch_size}.
    setup = SimpleNamespace(
        head_config_dir=head_config_dir, head_checkpoint=Path(args.head_checkpoint),
        head_variant_id=args.head_variant_id, head_device=args.head_device,
        head_allele_idx=args.head_allele_idx,
        head_window_batch_size=args.head_window_batch_size,
        allele=args.allele,   # OnlineHeadScorer allele
        config=SimpleNamespace(head=SimpleNamespace(score_scale="raw_logit")),  # only value supported
    )
    scorer = build_head_scorer(
        setup, window_k_min=int(inf_sec.get("min_k", 12)),
        window_k_max=int(inf_sec.get("max_k", 25)),
    )

    head_chunk = max(1, int(args.head_chunk))

    def head_fn(pid, seqs):
        # Chunk the encoder forward: a beam-multiplied pool (beam_width x per-state candidates,
        # ~8k on hard seeds) in one forward_batched allocates >60 GiB and OOMs the H200. The
        # window template is length-invariant across a protein's candidates, so per-chunk
        # window_risks [chunk, W] concatenate along axis 0 exactly like a single call.
        seqs = list(seqs)
        rows = []
        for i in range(0, len(seqs), head_chunk):
            chunk = seqs[i:i + head_chunk]
            compact = scorer.score_window_risk_batch_same_protein(
                protein_id=pid, records=[(str(j), s) for j, s in enumerate(chunk)])
            rows.append(np.asarray(compact.window_risks))
        return rows[0] if len(rows) == 1 else np.concatenate(rows, axis=0)

    def head_score_fn(pid, seqs):
        """Rich Head rows for NMP-free mode: global risk plus residue_hotspot."""
        seqs = list(seqs)
        scores = []
        for i in range(0, len(seqs), head_chunk):
            chunk = seqs[i:i + head_chunk]
            batch = scorer.score_batch_same_protein(
                protein_id=pid,
                records=[(str(j), sequence) for j, sequence in enumerate(chunk)],
            )
            scores.extend(batch.scores)
        return scores

    def window_coords_fn(seq):
        # Per-sequence window template (length-dependent) — a fixed template would
        # mis-map cores past its length in ~300aa uricases.
        compact = scorer.score_window_risk_batch_same_protein(
            protein_id="__windows__", records=[("0", seq)])
        return np.asarray(compact.window_starts_0b), np.asarray(compact.window_ends_0b)

    # --- NMP (legacy target mode only; Head mode never imports or constructs it) ---
    nmp_fn = None
    if args.target_source == "nmp":
        from epitope_head.data.netmhciipan_runner import StandaloneRunner
        from inverse_folding.evaluation.immunogenicity import NMP_PEP_LENGTHS

        runner = StandaloneRunner(
            binary_path=args.netmhciipan_bin,
            batch_size=args.nmp_batch_size,
            subprocess_timeout=args.nmp_timeout,
            max_lengths_per_call=args.nmp_max_lengths_per_call,
            n_workers=args.nmp_workers,
        )

        def nmp_fn(pid, seqs):
            entries = [(f"{pid}__c{i}", sequence) for i, sequence in enumerate(seqs)]
            scored = runner.score_batch(entries, args.allele, NMP_PEP_LENGTHS)
            out = []
            for i, sequence in enumerate(seqs):
                key = f"{pid}__c{i}"
                if key not in scored:
                    raise RuntimeError(
                        f"NetMHCIIpan returned no result for {key}; refusing to treat a failed "
                        "NMP call as 'no epitope' (NMP is the final gate)."
                    )
                by_len = scored[key]
                missing = [
                    k for k in NMP_PEP_LENGTHS if k <= len(sequence) and k not in by_len
                ]
                if missing:
                    raise RuntimeError(
                        f"NetMHCIIpan result for {key} is missing peptide lengths {missing}; "
                        "incomplete scoring would under-count cores -- aborting."
                    )
                rows = []
                for _pep_len, peptide_scores in by_len.items():
                    for peptide_score in peptide_scores:
                        rows.append({
                            "pos": peptide_score.pos,
                            "pep_length": peptide_score.pep_length,
                            "peptide": peptide_score.peptide,
                            "core": peptide_score.core,
                            "rank_EL": peptide_score.el_rank,
                        })
                out.append(rows)
            return out

    # --- Structure (configured live refold + TMalign vs the WT/target backbone) ---
    from inverse_folding.evaluation.refold import load_refold_model, refold
    from inverse_folding.evaluation.tmalign import run_tmalign
    from inverse_folding.reference_flow.refine import StructureMetrics
    from inverse_folding.reference_flow.runtime import resolve_structure_path

    evaluate_prediction_v2 = None
    prepare_reference_context_v2 = None
    if evaluation_structure_metrics:
        from inverse_folding.evaluation.structural_metrics_v2 import (
            evaluate_prediction as evaluate_prediction_v2,
            prepare_reference_context as prepare_reference_context_v2,
        )

    model = load_refold_model(
        args.refold_model,
        device=args.head_device,
        **backend_options,
    )
    args._refold_backend = args.refold_model
    args._refold_runtime = getattr(model, "metadata", None)
    test_df = pd.read_parquet(args.test_set_parquet)
    test_lookup = {str(r["protein_id"]): r for _, r in test_df.iterrows()}
    manifest = load_constraint_manifest(args.constraint_manifest) if args.constraint_manifest else None
    catalytic_indices_by_protein = (
        _load_catalytic_indices_by_protein(args.constraint_manifest, manifest)
        if args.structure_gate_profile == "protocol"
        else {}
    )
    ref_path_cache: dict[str, Path] = {}
    reference_context_cache: dict[str, object] = {}

    def _reference_path(pid: str) -> Path:
        if pid not in test_lookup:
            raise KeyError(f"protein {pid!r} is absent from --test-set-parquet")
        if pid not in ref_path_cache:
            ref_path_cache[pid] = Path(resolve_structure_path(test_lookup[pid], args.pdb_root))
        return ref_path_cache[pid]

    def _reference_context(pid: str):
        if pid not in reference_context_cache:
            ref_sequence = str(test_lookup[pid]["sequence"])
            anchors = _anchors_for(manifest, pid, ref_sequence)
            reference_context_cache[pid] = prepare_reference_context_v2(
                _reference_path(pid),
                ref_sequence=ref_sequence,
                anchor_indices=anchors,
            )
        return reference_context_cache[pid]

    def struct_fn(pid, seq):
        pred = refold(seq, pid, "refine", backend=args.refold_model,
                      cache_dir=args.refold_cache_dir, model=model)
        ref = _reference_path(pid)
        tm = run_tmalign(pred_pdb=str(pred["pdb_path"]), ref_pdb=str(ref), cache_dir=None)
        v2 = None
        if evaluation_structure_metrics:
            v2 = evaluate_prediction_v2(
                _reference_context(pid),
                str(pred["pdb_path"]),
                design_sequence=seq,
                metrics=evaluation_structure_metrics,
            )
        cat_max_scrmsd = None
        if args.structure_gate_profile == "protocol":
            if pid not in catalytic_indices_by_protein:
                raise KeyError(f"no catalytic index definition for protein {pid!r}")
            cat_max_scrmsd = _catalytic_max_sidechain_rmsd(
                v2,
                catalytic_indices_by_protein[pid],
            )
        return StructureMetrics(scTM=float(tm["tm_score"]), pLDDT=float(pred["pLDDT"]),
                                scRMSD=float(tm["rmsd"]),
                                global_ca_RMSD=(None if v2 is None else v2.global_ca_rmsd),
                                active_site_sidechain_RMSD=(
                                    None if v2 is None else v2.active_site_sidechain_rmsd
                                ),
                                max_anchor_sidechain_RMSD=(
                                    None if v2 is None else v2.max_anchor_sidechain_rmsd
                                ),
                                max_anchor_atom_distance=(
                                    None if v2 is None else v2.max_anchor_atom_distance
                                ),
                                active_site_complete=(
                                    None if v2 is None else v2.active_site_complete
                                ),
                                active_site_min_pLDDT=(
                                    None if v2 is None
                                    else v2.predicted_active_site_min_plddt
                                ),
                                cat_max_scRMSD=cat_max_scrmsd,
                                predicted_active_site_min_pLDDT=(
                                    None if v2 is None
                                    else v2.predicted_active_site_min_plddt
                                ))

    if callable(getattr(model, "close", None)):
        struct_fn.close = model.close

    return Oracles(
        head_fn=head_fn,
        nmp_fn=nmp_fn,
        struct_fn=struct_fn,
        window_coords_fn=window_coords_fn,
        head_score_fn=head_score_fn,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RF refinement (targeted epitope elimination)")
    p.add_argument("--run-dir", default=None,
                   help="a clean generated.parquet (file), a dir holding it, or "
                        "generation/*shard*/generated.parquet (alternative to --seed-table)")
    p.add_argument("--eval-immune-dir", default=None,
                   help="optional dir holding */imm_head.parquet (global_risk best-of-N seed "
                        "selection); omit to refine EVERY design in --run-dir")
    p.add_argument("--seed-table", default=None,
                   help="self-contained protein_id/design_id/sequence parquet; bypasses "
                        "--run-dir selection and refines every listed seed")
    p.add_argument("--test-set-parquet", default=None, help="test-set parquet (rows for resolve_structure_path)")
    p.add_argument(
        "--constraint-manifest",
        default=None,
        help="active-site hard-anchor manifest (required by the protocol structure gate)",
    )
    p.add_argument("--allele", required=True)
    p.add_argument("--proteins", default="all", help="comma list or 'all'")
    p.add_argument("--seeds-per-protein", type=int, default=3)
    p.add_argument("--mode", choices=("ceiling", "refine"), default="refine")
    p.add_argument("--target-source", choices=("nmp", "head"), default="head",
                   help="nmp = legacy NMP-core objective; head = NMP-free two-axis Head Pareto "
                        "objective using thresholded residue-local maxima")
    p.add_argument(
        "--official",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Head refine only: persist the exact per-seed two-axis Pareto front instead of "
            "all admitted history; --no-official retains admitted history"
        ),
    )
    p.add_argument(
        "--final-candidates-per-protein",
        type=int,
        default=None,
        help=(
            "requested final panel size recorded for merge_refine_shards; explicit values "
            "override the --official default of eight and do not alter search"
        ),
    )
    # head (mirror run_if_phase_c1.py)
    p.add_argument("--head-checkpoint", default=None)
    p.add_argument("--head-config-dir", default=None)
    p.add_argument("--head-variant-id", default=None)
    p.add_argument("--head-device", default="cuda")
    p.add_argument("--head-allele-idx", type=int, default=0)
    p.add_argument("--head-window-batch-size", type=int, default=None)
    p.add_argument("--head-chunk", type=int, default=512,
                   help="max candidate seqs per head encoder forward (bounds GPU mem; the "
                        "beam-multiplied pool OOMs the untiled forward on hard seeds)")
    p.add_argument(
        "--head-residue-threshold",
        type=float,
        default=0.15,
        help="inclusive residue_hotspot cutoff for Head proposal local maxima",
    )
    p.add_argument(
        "--head-max-target-positions",
        type=int,
        default=12,
        help="maximum residue-local maxima opened per parent and round",
    )
    p.add_argument(
        "--head-aa-per-position",
        type=int,
        default=2,
        help="single-mutant Head Pareto representatives retained per position for doubles",
    )
    # nmp (mirror evaluate_phase_c.py accelerated knobs)
    p.add_argument("--netmhciipan-bin", default=None, help="path to the NetMHCIIpan binary")
    p.add_argument("--nmp-batch-size", type=int, default=8)
    p.add_argument("--nmp-max-lengths-per-call", type=int, default=4)
    p.add_argument("--nmp-workers", type=int, default=8)
    p.add_argument("--nmp-timeout", type=int, default=1800)
    # structure
    p.add_argument("--pdb-root", default=None)
    p.add_argument(
        "--refold-cache-dir",
        "--esmfold-cache-dir",
        dest="refold_cache_dir",
        metavar="PATH",
        default=None,
        help="normalized refold cache; --esmfold-cache-dir is a deprecated alias",
    )
    p.add_argument(
        "--refold-model",
        choices=("esmfold2_live", "esmfold"),
        default="esmfold2_live",
        help="live refold backend (default: persistent isolated ESMFold2 worker)",
    )
    p.add_argument("--esmfold2-site-packages", default=None)
    p.add_argument("--esmfold2-model", default="biohub/ESMFold2")
    p.add_argument("--esmfold2-num-loops", type=int, default=3)
    p.add_argument("--esmfold2-num-sampling-steps", type=int, default=50)
    p.add_argument("--esmfold2-num-diffusion-samples", type=int, default=1)
    p.add_argument("--esmfold2-seed", type=int, default=0)
    # incremental NMP splice (EXACT for per-window NMP; the primary NMP-volume cut)
    p.add_argument("--incremental-nmp", dest="incremental_nmp", action="store_true", default=True,
                   help="score only each candidate's mutated sub-sequence + splice vs the seed (exact, default on)")
    p.add_argument("--no-incremental-nmp", dest="incremental_nmp", action="store_false",
                   help="disable the splice; score every candidate over its full length")
    p.add_argument("--nmp-context-margin", type=int, default=30,
                   help="sub-sequence half-window for the splice (>= NMP context radius; 30 covers 25-mer+context)")
    # seed job-array sharding (SLURM --array over seeds; each task refines its shard)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--shard-idx", type=int, default=0, help="this task's shard (SLURM_ARRAY_TASK_ID)")
    p.add_argument("--shard-by", choices=("balanced", "stride"), default="balanced",
                   help="balanced = LPT bin-pack by --shard-cost-col; stride = round-robin")
    p.add_argument("--shard-cost-col", default="nmp_n_strong_binders",
                   help="seed-table column used as the difficulty weight for balanced sharding")
    # search knobs (§7)
    p.add_argument("--strong-rank", type=float, default=0.02)
    p.add_argument("--margin-band", type=float, default=0.10)
    p.add_argument(
        "--structure-gate-profile",
        choices=("protocol", "legacy"),
        default="protocol",
        help="protocol = absolute scTM/catalytic-scRMSD/active-site-confidence gate; "
             "legacy retains the old optional thresholds",
    )
    p.add_argument("--gate-scTM-min", dest="gate_scTM_min", type=float, default=None)
    p.add_argument(
        "--gate-cat-max-scRMSD-max",
        dest="gate_cat_max_scRMSD_max",
        type=float,
        default=None,
    )
    p.add_argument(
        "--gate-predicted-active-site-min-pLDDT-min",
        dest="gate_predicted_active_site_min_pLDDT_min",
        type=float,
        default=None,
    )
    p.add_argument(
        "--scTM-eps",
        dest="scTM_eps",
        type=float,
        default=None,
        help="legacy seed-relative scTM allowance; requires --structure-gate-profile legacy",
    )
    p.add_argument("--scRMSD-max", dest="scRMSD_max", type=float, default=None)
    p.add_argument("--active-site-RMSD-max", dest="active_site_RMSD_max", type=float, default=None)
    p.add_argument(
        "--refinement-structure-metrics",
        default="",
        help=(
            "comma-separated lightweight v2 metrics evaluated inside the refold loop: "
            "global_ca_rmsd,plddt,sidechain (protocol gate always enables plddt+sidechain; "
            "reference contexts are cached)"
        ),
    )
    p.add_argument(
        "--max-anchor-sidechain-RMSD-max",
        dest="max_anchor_sidechain_RMSD_max",
        type=float,
        default=None,
        help=(
            "fail-closed ceiling on the worst per-anchor all-heavy-side-chain RMSD; "
            "requires sidechain in --refinement-structure-metrics and a constraint manifest"
        ),
    )
    p.add_argument("--topB", type=int, default=None)
    p.add_argument("--beam-width", type=int, default=8,
                   help="working states between rounds; topB caps the downstream candidate pool")
    p.add_argument("--max-rounds", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument(
        "--max-pairs",
        type=int,
        default=200,
        help="NMP pair cap; in Head mode, full double-mutant scoring cap per parent/round",
    )
    p.add_argument("--max-path-mutations", type=int, default=8,
                   help="cheap refold-free cap on total edits per search path (branch-waste bound)")
    p.add_argument("--refold-cap", dest="refold_cap", type=int, default=16,
                   help="max count-dropping candidates refolded per round; bounds refold cost "
                        "and yields the lean ranked shortlist (pass a large value to disable)")
    p.add_argument("--allow-structure-unknown", action="store_true")
    # full-metrics emission: after refinement, reuse evaluate_phase_c's row builders on the
    # final designs to persist the metrics the search discards (per-window NMP aggregates,
    # head hotspots, per-residue Kabsch CA RMSD). Refolds are cache hits.
    p.add_argument("--emit-eval-metrics", dest="emit_eval_metrics", action="store_true", default=True,
                   help="emit evaluate_phase_c-schema metric tables (imm_head/imm_nmp/structural/"
                        "structural_residues) for every final design (default on)")
    p.add_argument("--no-eval-metrics", dest="emit_eval_metrics", action="store_false",
                   help="skip the post-refine full-metrics emission")
    p.add_argument("--strong-binder-threshold", dest="strong_binder_threshold", type=float, default=2.0,
                   help="imm_nmp strong-binder rank_EL%% cutoff (evaluate_phase_c default 2.0; distinct "
                        "from --strong-rank, the search's FRACTION distinct-core threshold)")
    p.add_argument("--hotspot-threshold", dest="hotspot_threshold", type=float, default=0.5,
                   help="imm_head n_hotspot_positions cutoff (evaluate_phase_c default 0.5)")
    p.add_argument("--imm-full", dest="imm_full", action="store_true", default=False,
                   help="also emit imm_head_residues + imm_nmp_peptides (evaluate_phase_c --imm-full)")
    p.add_argument(
        "--structural-metrics-v2",
        action="store_true",
        default=True,
        help=(
            "deprecated no-op: v2 is the canonical structural output"
        ),
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--print-config", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.allele = _canonical_allele(args.allele)  # tolerate the 'HLA-DRB1_07_01' run-dir tag
    if args.print_config:
        print("[refine] config: " + json.dumps(vars(args), sort_keys=True, default=str), flush=True)
    oracles = build_oracles(args)
    try:
        rc = run_refinement(args, oracles)
    finally:
        _close_oracles(oracles)
    if rc == 0 and args.mode == "refine" and args.emit_eval_metrics:
        del oracles
        _free_gpu()
        # Best-effort: the refinement outputs are already flushed and are the expensive,
        # protected artifact. An eval-pass failure (poison-pill length mismatch, CUDA OOM on
        # model reload, NMP hiccup) must NOT fail the job or lose refinement — the metrics are
        # cheap and independently re-runnable via evaluate_phase_c on refined/evaluator_ready.parquet.
        try:
            _run_final_metrics(args)
        except Exception:
            import traceback
            print("[refine] final-metrics FAILED (refinement outputs intact; re-run eval on "
                  "refined/evaluator_ready.parquet):", flush=True)
            traceback.print_exc()
            _write_final_metrics_status(Path(args.out_dir) / "refined", ok=False,
                                        error=traceback.format_exc(),
                                        structural_metrics_version="v2")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
