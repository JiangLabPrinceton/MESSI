#!/usr/bin/env python
"""Merge independent refinement worker outputs and select official final panels."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.refine import head_first_pareto_front
from inverse_folding.reference_flow.official_selection import two_axis_pareto_order

# Per-design tables that carry (protein_id, design_idx) 1:1 with refined_designs.
PER_DESIGN_TABLES = [
    "refined_designs", "imm_head", "imm_nmp", "structural", "structural_residues",
    "imm_head_residues", "imm_nmp_peptides",
]
_SEED_PROVENANCE_COLUMNS = (
    "seed_group",
    "selection_rule_id",
    "source_campaign",
    "source_depth",
    "source_root_index",
)


def _shard_idx_of(shard_dir: Path) -> int:
    m = re.search(r"shard(\d+)of\d+", shard_dir.name)
    if not m:
        raise ValueError(f"cannot parse shard index from {shard_dir}")
    return int(m.group(1))


def _uid(df: pd.DataFrame, shard_idx: int) -> pd.Series:
    return (
        "s" + f"{shard_idx:02d}" + "_"
        + df["protein_id"].astype(str) + "_d"
        + df["design_idx"].astype(int).map(lambda i: f"{i:04d}")
    )


def _load_concat(shard_dirs, table, *, per_design):
    frames = []
    present = 0
    for sd in shard_dirs:
        p = sd / "refined" / f"{table}.parquet"
        if not p.exists():
            continue
        present += 1
        df = pd.read_parquet(p)
        if df.empty:
            continue
        si = _shard_idx_of(sd)
        df = df.copy()
        df.insert(0, "shard_idx", si)
        if per_design and {"protein_id", "design_idx"}.issubset(df.columns):
            df.insert(1, "design_uid", _uid(df, si))
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return merged, present


def _target_source(frame: pd.DataFrame) -> str:
    if "target_source" not in frame.columns:
        return "nmp"
    values = set(frame["target_source"].dropna().astype(str))
    if len(values) != 1 or not values <= {"nmp", "head"}:
        raise ValueError(f"refined_designs mixes invalid target sources: {sorted(values)}")
    return next(iter(values))


def _head_pareto_per_seed(master: pd.DataFrame) -> pd.DataFrame:
    required = {"head_global_risk_after", "head_positive_mass_density_after"}
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"Head refinement merge is missing objective columns: {missing}")
    for column in sorted(required):
        values = master[column].astype(float)
        if not values.map(math.isfinite).all():
            raise ValueError(f"Head refinement merge has non-finite {column}")

    fronts = []
    for _key, group in master.groupby(
        ["protein_id", "sequence_original"], sort=True, dropna=False
    ):
        metrics = [
            SimpleNamespace(
                global_risk=float(row.head_global_risk_after),
                positive_mass_density=float(row.head_positive_mass_density_after),
            )
            for row in group.itertuples(index=False)
        ]
        local_indices = head_first_pareto_front(metrics)
        front = group.iloc[list(local_indices)].copy()
        front["head_pareto_rank"] = 1
        front["selection_status"] = "head_refinement_pareto"
        fronts.append(front)
    return pd.concat(fronts, ignore_index=True).sort_values(
        [
            "protein_id",
            "sequence_original",
            "head_global_risk_after",
            "head_positive_mass_density_after",
            "design_uid",
        ],
        kind="mergesort",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refined_md5(sequence: object) -> str:
    return hashlib.md5(str(sequence).encode("utf-8"), usedforsecurity=False).hexdigest()


def _head_final_candidates(front: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Select an up-to-N lineage-balanced final Head panel per protein/seed group.

    Each seed contributes at most its minimum-global-risk Pareto member. The frozen two-axis
    Pareto-layer/rank-sum/content-key law then chooses up to N seed representatives. Thus the
    count knob controls delivered lineages rather than letting one seed fill a protein's panel.
    """

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("final candidates per protein must be an integer >= 1")
    required = {
        "protein_id",
        "sequence_original",
        "sequence_refined",
        "head_global_risk_after",
        "head_positive_mass_density_after",
        "design_uid",
    }
    missing = sorted(required - set(front.columns))
    if missing:
        raise ValueError(f"Head final candidate pool is missing columns: {missing}")
    pool = front.copy()
    if "seed_group" not in pool.columns:
        pool["seed_group"] = "unspecified"
    pool["seed_group"] = pool["seed_group"].fillna("unspecified").astype(str)
    pool["refined_sequence_md5"] = pool["sequence_refined"].map(_refined_md5)

    selected_groups = []
    for (_protein_id, _seed_group), group in pool.groupby(
        ["protein_id", "seed_group"], sort=True, dropna=False,
    ):
        representatives = []
        for _seed, seed_rows in group.groupby(
            "sequence_original", sort=True, dropna=False,
        ):
            representatives.append(seed_rows.sort_values(
                [
                    "head_global_risk_after",
                    "refined_sequence_md5",
                    "design_uid",
                ],
                kind="mergesort",
            ).iloc[0])
        candidate_rows = pd.DataFrame(representatives).drop_duplicates(
            "refined_sequence_md5", keep="first",
        ).reset_index(drop=True)
        order = two_axis_pareto_order(
            first=candidate_rows["head_global_risk_after"].astype(float).tolist(),
            second=candidate_rows[
                "head_positive_mass_density_after"
            ].astype(float).tolist(),
            tie_keys=(
                candidate_rows["refined_sequence_md5"].astype(str)
                + ":"
                + candidate_rows["design_uid"].astype(str)
            ).tolist(),
        )[:limit]
        chosen = []
        for final_rank, ordered in enumerate(order, start=1):
            row = candidate_rows.iloc[ordered.index].copy()
            row["final_selection_rank"] = final_rank
            row["final_seed_round"] = 0
            row["final_pareto_layer"] = ordered.pareto_layer
            row["final_head_global_rank"] = ordered.first_rank
            row["final_head_positive_mass_rank"] = ordered.second_rank
            row["final_head_rank_sum"] = ordered.rank_sum
            row["selection_status"] = "official_final_candidate"
            if "selection_rule_id" in row.index:
                row["source_selection_rule_id"] = row["selection_rule_id"]
            row["selection_rule_id"] = (
                "one_per_seed_then_head_pareto_layers_rank_sum_v1"
            )
            chosen.append(row)
        if chosen:
            selected_groups.append(pd.DataFrame(chosen))
    if not selected_groups:
        return pool.iloc[0:0].copy()
    return pd.concat(selected_groups, ignore_index=True).sort_values(
        ["protein_id", "seed_group", "final_selection_rank"], kind="mergesort",
    ).reset_index(drop=True)


def _write_head_final_candidates(
    *, front: pd.DataFrame, out_dir: Path, limit: int, official: bool,
) -> tuple[pd.DataFrame, dict]:
    selected = _head_final_candidates(front, limit)
    parquet = out_dir / "final_candidates.parquet"
    selected.to_parquet(parquet, index=False)
    fasta = out_dir / "final_candidates.fasta"
    with fasta.open("w", encoding="utf-8") as handle:
        for row in selected.itertuples(index=False):
            handle.write(
                f">{row.protein_id}_{row.seed_group}_rank{int(row.final_selection_rank)}"
                f"_risk{float(row.head_global_risk_after):.6f}"
                f"_density{float(row.head_positive_mass_density_after):.6f}\n"
                f"{row.sequence_refined}\n"
            )
    counts = selected.groupby(["protein_id", "seed_group"], sort=True).size()
    manifest = {
        "schema_version": "rf-refinement-final-candidates-v1",
        "official": bool(official),
        "requested_candidates_per_protein": int(limit),
        "realized_counts": {
            f"{protein_id}::{seed_group}": int(count)
            for (protein_id, seed_group), count in counts.items()
        },
        "selection_rule_id": "one_per_seed_then_head_pareto_layers_rank_sum_v1",
        "source": str(out_dir / "head_pareto_per_seed.parquet"),
        "output": str(parquet),
        "output_sha256": _sha256(parquet),
        "source_artifacts_preserved": True,
    }
    (out_dir / "final_candidates_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selected, manifest


def _refine_output_contract(shard_dirs) -> tuple[bool, int | None, str | None]:
    contracts = []
    for shard_dir in shard_dirs:
        path = shard_dir / "refined" / "refine_config.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        contracts.append((
            bool(payload.get("official", False)),
            payload.get("final_candidates_per_protein"),
            payload.get("seed_table"),
        ))
    if not contracts:
        return False, None, None
    if len(contracts) != len(shard_dirs):
        raise ValueError(
            "only some refinement shards carry refine_config.json; output contract is incomplete"
        )
    if len(set(contracts)) != 1:
        raise ValueError("refinement shards disagree on official/final candidate contract")
    official, count, seed_table = contracts[0]
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("refine_config final_candidates_per_protein must be integer >= 1")
    if seed_table is not None and (
        not isinstance(seed_table, str) or not seed_table.strip()
    ):
        raise ValueError("refine_config seed_table must be a non-empty string when present")
    return official, count, seed_table


def _attach_seed_provenance(front: pd.DataFrame, seed_table: str | None) -> pd.DataFrame:
    if "seed_group" in front.columns or seed_table is None:
        return front
    path = Path(seed_table).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"seed table recorded for official merge does not exist: {path}")
    seeds = pd.read_parquet(path)
    required = {"protein_id", "sequence"}
    missing = sorted(required - set(seeds.columns))
    if missing:
        raise ValueError(f"official seed table is missing columns: {missing}")
    columns = ["protein_id", "sequence"] + [
        column for column in _SEED_PROVENANCE_COLUMNS if column in seeds.columns
    ]
    seeds = seeds[columns].copy().rename(columns={"sequence": "sequence_original"})
    if seeds.duplicated(["protein_id", "sequence_original"]).any():
        raise ValueError("official seed table has duplicate protein/sequence keys")
    joined = front.merge(
        seeds,
        on=["protein_id", "sequence_original"],
        how="left",
        validate="many_to_one",
    )
    if "seed_group" in seeds.columns and joined["seed_group"].isna().any():
        raise ValueError("official merge could not map every refined row to its seed group")
    return joined


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="the array run dir holding shardKKofNN/ subdirs")
    ap.add_argument("--out-dir", default=None, help="default: <run-dir>/merged")
    ap.add_argument("--top-k", type=int, default=1,
                    help="select the top-K refined variants per input seed by lowest "
                         "core_count_after (ties: fewest true_muts, then highest scTM) -> "
                         "top{K}_per_seed.{parquet,fasta}; use K>1 when 0 is not guaranteed reachable")
    ap.add_argument("--dplm-native-structural", default=None,
                    help="DPLM-native structural.parquet; adds dplm_native_scTM (per-protein median scTM) "
                         "and delta_scTM = scTM - dplm_native_scTM to master (fold quality RELATIVE to the "
                         "DPLM base, for proteins whose base model folds poorly in absolute terms)")
    ap.add_argument(
        "--official",
        action="store_true",
        help="write the paper-facing final candidate panel (default eight per protein)",
    )
    ap.add_argument(
        "--final-candidates-per-protein",
        type=int,
        default=None,
        help=(
            "explicit up-to-N final Head candidates per protein/seed-group; works with or "
            "without --official and overrides the recorded/default value"
        ),
    )
    ap.add_argument(
        "--seed-table",
        default=None,
        help=(
            "optional refinement seed table used to restore seed_group/provenance for legacy "
            "runs; new official shards record it in refine_config.json"
        ),
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    shard_dirs = sorted(
        (Path(p) for p in glob.glob(str(run_dir / "shard*of*"))),
        key=_shard_idx_of,
    )
    if not shard_dirs:
        raise SystemExit(f"no shard*of* dirs under {run_dir}")
    config_official, config_count, config_seed_table = _refine_output_contract(shard_dirs)
    official = bool(args.official or config_official)
    final_candidates_per_protein = (
        args.final_candidates_per_protein
        if args.final_candidates_per_protein is not None
        else config_count
    )
    if official and final_candidates_per_protein is None:
        final_candidates_per_protein = 8
    seed_table = args.seed_table if args.seed_table is not None else config_seed_table
    if final_candidates_per_protein is not None and (
        isinstance(final_candidates_per_protein, bool)
        or final_candidates_per_protein < 1
    ):
        raise SystemExit("--final-candidates-per-protein must be an integer >= 1")
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"run_dir": str(run_dir), "n_shard_dirs": len(shard_dirs), "tables": {}}

    # per-design tables
    tables = {}
    for t in PER_DESIGN_TABLES:
        merged, present = _load_concat(shard_dirs, t, per_design=True)
        if present == 0:
            continue
        tables[t] = merged
        merged.to_parquet(out_dir / f"{t}.parquet", index=False)
        summary["tables"][t] = {"rows": int(len(merged)), "shards_present": present}

    # refine_trace (keyed on orig_design_idx, not design_idx -> shard_idx only)
    trace, present = _load_concat(shard_dirs, "refine_trace", per_design=False)
    if present:
        trace.to_parquet(out_dir / "refine_trace.parquet", index=False)
        summary["tables"]["refine_trace"] = {"rows": int(len(trace)), "shards_present": present}

    # final_metrics_status roll-up
    ok = bad = missing = 0
    for sd in shard_dirs:
        st = sd / "refined" / "final_metrics_status.json"
        if not st.exists():
            missing += 1
            continue
        j = json.loads(st.read_text())
        ok += int(bool(j.get("ok")))
        bad += int(not j.get("ok"))
    summary["final_metrics_status"] = {"ok": ok, "failed": bad, "missing": missing}

    rd = tables.get("refined_designs")
    if rd is None or rd.empty:
        (out_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[merge] no refined_designs; wrote {out_dir}/merge_summary.json")
        return

    # master join: objective + structural + imm on design_uid (left from refined_designs)
    master = rd.copy()
    join_cols = {
        "structural": [
            "scTM", "global_ca_RMSD", "pLDDT", "reference_pLDDT",
            "predicted_active_site_mean_pLDDT", "predicted_active_site_min_pLDDT",
            "reference_active_site_mean_pLDDT", "reference_active_site_min_pLDDT",
            "active_site_sidechain_RMSD", "max_anchor_sidechain_RMSD",
            "max_anchor_atom_distance", "active_site_sidechain_atom_count",
            "anchor_count", "matched_anchor_count", "active_site_complete", "recovery",
            "foldability", "refold_backend",
        ],
        "imm_nmp": ["n_strong_binders", "n_weak_binders", "mean_best_rank", "n_windows_scored"],
        "imm_head": ["global_risk", "mean_hotspot", "max_hotspot", "n_hotspot_positions"],
    }
    for t, cols in join_cols.items():
        if t in tables and not tables[t].empty:
            have = [c for c in cols if c in tables[t].columns]
            master = master.merge(
                tables[t][["design_uid", *have]], on="design_uid", how="left", validate="1:1")

    # true edit distance from seed (n_mutations records only the LAST beam step; recompute cumulative)
    def _hamming(a, b):
        return sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    master["true_muts"] = [
        _hamming(str(o), str(r))
        for o, r in zip(master["sequence_original"], master["sequence_refined"])
    ]
    # optional: fold quality RELATIVE to the DPLM base (some proteins fold poorly in absolute
    # terms, so the meaningful signal is scTM - dplm_native_scTM, not absolute scTM)
    if args.dplm_native_structural:
        dn = pd.read_parquet(args.dplm_native_structural)
        base = dn.groupby(dn["protein_id"].astype(str))["scTM"].median()
        master["dplm_native_scTM"] = master["protein_id"].astype(str).map(base)
        if "scTM" in master.columns:
            master["delta_scTM"] = master["scTM"] - master["dplm_native_scTM"]
        summary["dplm_native_structural"] = str(args.dplm_native_structural)
    master.to_parquet(out_dir / "master.parquet", index=False)
    summary["tables"]["master"] = {"rows": int(len(master))}
    target_source = _target_source(master)
    summary["target_source"] = target_source
    if target_source != "head" and (
        official or final_candidates_per_protein is not None
    ):
        raise ValueError(
            "official/final-candidates-per-protein currently require Head refinement"
        )

    if target_source == "head":
        if "core_count_after" in master and master["core_count_after"].notna().any():
            raise ValueError("Head refinement rows must not carry an NMP core-count objective")
        front = _head_pareto_per_seed(master)
        if final_candidates_per_protein is not None:
            front = _attach_seed_provenance(front, seed_table)
        front.to_parquet(out_dir / "head_pareto_per_seed.parquet", index=False)
        with open(out_dir / "head_pareto_per_seed.fasta", "w") as handle:
            for row in front.itertuples(index=False):
                handle.write(
                    f">{row.protein_id}_{row.design_uid}"
                    f"_risk{float(row.head_global_risk_after):.6f}"
                    f"_density{float(row.head_positive_mass_density_after):.6f}"
                    f"_muts{int(row.true_muts)}\n{row.sequence_refined}\n"
                )
        n_seeds = int(
            master[["protein_id", "sequence_original"]].drop_duplicates().shape[0]
        )
        summary["head_pareto"] = {
            "rows": int(len(front)),
            "seeds": n_seeds,
            "proteins": int(front["protein_id"].nunique()),
            "min_rows_per_seed": int(
                front.groupby(["protein_id", "sequence_original"]).size().min()
            ),
            "max_rows_per_seed": int(
                front.groupby(["protein_id", "sequence_original"]).size().max()
            ),
        }
        if final_candidates_per_protein is not None:
            selected, final_manifest = _write_head_final_candidates(
                front=front,
                out_dir=out_dir,
                limit=int(final_candidates_per_protein),
                official=official,
            )
            summary["final_candidates"] = {
                "rows": int(len(selected)),
                "proteins": int(selected["protein_id"].nunique()),
                "requested_per_protein": int(final_candidates_per_protein),
                "official": bool(official),
                "manifest": str(out_dir / "final_candidates_manifest.json"),
                "selection_rule_id": final_manifest["selection_rule_id"],
            }
        (out_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[merge] {len(shard_dirs)} shards -> {out_dir}")
        print(
            f"[merge] Head mode: designs={len(master)} seeds={n_seeds} "
            f"Pareto rows={len(front)}"
        )
        print(
            "[merge] wrote master.parquet + head_pareto_per_seed.{parquet,fasta} "
            "+ merge_summary.json"
        )
        return

    # best count-0 variant per input seed (min true edit distance, then max scTM)
    n_seeds = master["sequence_original"].nunique()
    zero = master[master["core_count_after"] == 0].copy()
    n0_seeds = zero["sequence_original"].nunique()
    best = pd.DataFrame()
    if not zero.empty:
        sort_cols = ["true_muts"] + (["scTM"] if "scTM" in zero.columns else [])
        asc = [True] + ([False] if "scTM" in zero.columns else [])
        best = (zero.sort_values(sort_cols, ascending=asc)
                    .groupby("sequence_original", as_index=False).first())
        best.to_parquet(out_dir / "best_count0_per_seed.parquet", index=False)
        with open(out_dir / "best_count0_per_seed.fasta", "w") as f:
            for r in best.itertuples(index=False):
                f.write(f">{r.protein_id}_{r.design_uid}_muts{int(r.true_muts)}"
                        f"_scTM{getattr(r, 'scTM', float('nan')):.3f}\n{r.sequence_refined}\n")

    # top-K refined variants per input seed by lowest core count (does NOT require reaching 0;
    # the deliverable when 0 is not guaranteed — the K best de-immunized designs per protein)
    if args.top_k and args.top_k >= 1:
        sort_cols = ["core_count_after", "true_muts"] + (["scTM"] if "scTM" in master.columns else [])
        asc = [True, True] + ([False] if "scTM" in master.columns else [])
        topk = (master.sort_values(sort_cols, ascending=asc)
                      .groupby("sequence_original", as_index=False).head(args.top_k))
        topk.to_parquet(out_dir / f"top{args.top_k}_per_seed.parquet", index=False)
        with open(out_dir / f"top{args.top_k}_per_seed.fasta", "w") as f:
            for r in topk.itertuples(index=False):
                f.write(f">{r.protein_id}_{r.design_uid}_core{int(r.core_count_after)}"
                        f"_muts{int(r.true_muts)}_scTM{getattr(r, 'scTM', float('nan')):.3f}"
                        f"\n{r.sequence_refined}\n")
        summary["topk"] = {"k": int(args.top_k), "rows": int(len(topk)),
                           "proteins": int(topk["protein_id"].nunique()),
                           "core_after_median": float(topk["core_count_after"].median())}

    summary["reach0"] = {
        "seeds": int(n_seeds),
        "seeds_with_count0_variant": int(n0_seeds),
        "designs_total": int(len(master)),
        "designs_count0": int((master["core_count_after"] == 0).sum()),
    }
    if not best.empty:
        summary["reach0"]["best_true_muts_median"] = float(best["true_muts"].median())
        if "scTM" in best.columns:
            summary["reach0"]["best_scTM_median"] = float(best["scTM"].median())
    (out_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[merge] {len(shard_dirs)} shards -> {out_dir}")
    print(f"[merge] designs={len(master)} seeds={n_seeds} seeds_with_count0={n0_seeds}/{n_seeds}")
    print(f"[merge] status ok={ok} failed={bad} missing={missing}")
    print(f"[merge] wrote master.parquet + best_count0_per_seed.{{parquet,fasta}} + merge_summary.json")


if __name__ == "__main__":
    main()
