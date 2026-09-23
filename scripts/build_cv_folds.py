#!/usr/bin/env python
"""Construct cluster-aware CV folds for canonical Head training."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from epitope_head.data.cv_folds import (
    partition_clusters_into_folds,
    build_fold_splits,
    assert_no_cluster_leakage,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", required=True,
                    help="e.g. /scratch/.../work/immune-design/manifests/drb0701")
    ap.add_argument("--profile", default="strict")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-subdir", default=None,
                    help="default: cv<n_folds> (e.g. cv5)")
    args = ap.parse_args()

    split_root = Path(args.manifest_dir) / "splits" / args.profile
    ca_path = split_root / "cluster_assignment.parquet"
    if not ca_path.exists():
        raise FileNotFoundError(f"missing {ca_path}")
    ca = pd.read_parquet(ca_path)
    protein_to_cluster = dict(zip(ca["protein_id"].astype(str), ca["cluster_rep"].astype(str)))
    print(f"[cv] {len(protein_to_cluster)} proteins / {ca['cluster_rep'].nunique()} clusters")

    fop = partition_clusters_into_folds(protein_to_cluster, n_folds=args.n_folds, seed=args.seed)
    fold_splits = build_fold_splits(fop, n_folds=args.n_folds)
    assert_no_cluster_leakage(fold_splits, protein_to_cluster)  # hard gate

    out_sub = args.output_subdir or f"cv{args.n_folds}"
    out_root = split_root / out_sub
    print(f"[cv] writing -> {out_root}")
    print(f"{'fold':4} {'train':>6} {'val':>5} {'test':>5}")
    for fs in fold_splits:
        fold_dir = out_root / f"fold{fs['fold']}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            (fold_dir / f"{split}_ids.txt").write_text("\n".join(fs[split]) + "\n")
        print(f"{fs['fold']:<4} {len(fs['train']):>6} {len(fs['val']):>5} {len(fs['test']):>5}")
    # provenance sidecar
    (out_root / "cv_config.json").write_text(
        pd.Series({"n_folds": args.n_folds, "seed": args.seed, "profile": args.profile,
                   "n_proteins": len(protein_to_cluster),
                   "n_clusters": int(ca["cluster_rep"].nunique())}).to_json()
    )
    print(f"[cv] done: {out_root}/fold0..{args.n_folds-1}/ + cv_config.json")


if __name__ == "__main__":
    main()
