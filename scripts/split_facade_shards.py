#!/usr/bin/env python
"""Split a generated facade into deterministic worker input tables."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="generated.parquet (facade) to split.")
    ap.add_argument(
        "--n-shards",
        type=int,
        default=192,
        help="Number of shards (default 192: ~120 designs/shard for a full ~23k pool, "
        "~30 min each at 4 cores on qos=short).",
    )
    ap.add_argument(
        "--out-base",
        required=True,
        help="Output base path; writes <out-base>.shard{k:02d}of{N:02d}.parquet.",
    )
    ap.add_argument("--mode", choices=("round_robin", "contiguous"), default="round_robin")
    args = ap.parse_args()
    if args.n_shards <= 0:
        ap.error("--n-shards must be positive")

    df = pd.read_parquet(args.input)
    n = len(df)
    if n == 0:
        ap.error(f"empty input: {args.input}")
    if args.n_shards > n:
        ap.error(f"--n-shards {args.n_shards} exceeds row count {n}")

    out_base = Path(args.out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    for k in range(args.n_shards):
        if args.mode == "round_robin":
            shard = df.iloc[k :: args.n_shards]
        else:
            idx = np.array_split(np.arange(n), args.n_shards)[k]
            shard = df.iloc[idx]
        path = f"{out_base}.shard{k:02d}of{args.n_shards:02d}.parquet"
        shard.to_parquet(path, index=False)
        total += len(shard)
        print(f"shard {k:02d}/{args.n_shards}: {len(shard)} rows -> {path}")
    assert total == n, f"row-count mismatch after split: {total} != {n}"
    print(f"[done] split {n} rows into {args.n_shards} shards (mode={args.mode})")
    print("Evaluate each shard with scripts/evaluate_phase_c.py and explicit input/output paths.")


if __name__ == "__main__":
    main()
