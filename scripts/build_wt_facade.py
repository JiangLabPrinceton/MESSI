#!/usr/bin/env python
"""Create evaluation-compatible facades from real reference sequences."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--if-ready", required=True,
                    help="canonical IF-ready test parquet (needs protein_id + sequence)")
    ap.add_argument("--output", required=True,
                    help="full facade parquet path; shards/meta derive from this base")
    ap.add_argument("--run-id", default=None, help="run_id recorded in the meta sidecar")
    ap.add_argument("--n-shards", type=int, default=0,
                    help="if >0, also write N round-robin shard facades for a job array")
    args = ap.parse_args()

    src = pd.read_parquet(args.if_ready)
    for col in ("protein_id", "sequence"):
        if col not in src.columns:
            sys.exit(f"FATAL: '{col}' not in {args.if_ready}")
    facade = (
        src[["protein_id", "sequence"]]
        .assign(protein_id=lambda d: d["protein_id"].astype(str), design_idx=0)
        .loc[:, ["protein_id", "design_idx", "sequence"]]
        .sort_values("protein_id")
        .reset_index(drop=True)
    )
    if facade["protein_id"].duplicated().any():
        sys.exit("FATAL: duplicate protein_id in IF-ready source")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    facade.to_parquet(out, index=False)

    meta = {
        "run_id": args.run_id or out.stem,
        "source_test_set": str(Path(args.if_ready).resolve()),
        "output_parquet": str(out.resolve()),
        "n_rows": int(len(facade)),
        "n_proteins": int(facade["protein_id"].nunique()),
        "design_idx": 0,
        "n_shards": int(args.n_shards),
        "purpose": "WT generated.parquet facade for evaluate_phase_c immune baseline on canonical v2 IF-ready test set",
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[facade] {out}  rows={len(facade)}  proteins={meta['n_proteins']}")
    print(f"[meta]   {meta_path}")

    if args.n_shards and args.n_shards > 0:
        n = args.n_shards
        ids = facade["protein_id"].tolist()  # already sorted
        base = str(out)[: -len(".parquet")] if str(out).endswith(".parquet") else str(out)
        counts = []
        for k in range(n):
            shard_ids = ids[k::n]  # round-robin on sorted ids
            shard = facade[facade["protein_id"].isin(shard_ids)].reset_index(drop=True)
            sp = f"{base}.shard{k:02d}of{n:02d}.parquet"
            shard.to_parquet(sp, index=False)
            counts.append(len(shard))
        print(f"[shards] {n} round-robin shards written: sizes min={min(counts)} "
              f"max={max(counts)} sum={sum(counts)} (== {len(facade)})")
        assert sum(counts) == len(facade), "shard coverage mismatch"


if __name__ == "__main__":
    main()
