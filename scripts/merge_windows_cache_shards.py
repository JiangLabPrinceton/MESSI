#!/usr/bin/env python
"""Merge sharded ``benchmark_iedb_test.py --windows-cache`` parquets into one cache.

A full-pool NetMHCIIpan cache is built by splitting the protein pool across
independent Slurm jobs (NMP parallelism inside one process is capped at
``ceil(n_lengths / max_lengths_per_call)``, so extra throughput has to come from
extra processes). Each shard writes its own ``cache_*.parquet`` + ``.meta.json``;
this script concatenates them into the single cache that
``--reuse-nmp-from-cache`` expects.

Coverage is checked hard on purpose: ``benchmark_iedb_test.py`` only WARNS when a
test protein is missing from the cache and then imputes worst-case
``el_rank=1.0``, which silently corrupts the NMP reference line. Pass
``--expect-ids`` to make an incomplete merge a fail-fast error instead.

    python scripts/merge_windows_cache_shards.py \
        --shards /path/cache_1501_shard{0,1,2,3}.parquet \
        --output /path/cache_1501_all1536.parquet \
        --expect-ids /path/splits/strict/cv5/all_ids.txt
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import pandas as pd

# Meta fields that decide whether two shards describe the same NMP measurement.
IDENTITY_FIELDS = ("allele", "min_k", "max_k", "nmp_mode")


def _load_meta(shard: str) -> dict:
    meta_path = shard + ".meta.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"shard has no sidecar metadata: {meta_path}")
    with open(meta_path) as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True,
                    help="shard cache parquets (each needs its .meta.json sidecar)")
    ap.add_argument("--output", required=True, help="merged cache parquet path")
    ap.add_argument("--expect-ids", default=None,
                    help="id file the merged cache must cover EXACTLY (fail-fast)")
    args = ap.parse_args()

    frames, metas, provenance = [], [], []
    for shard in args.shards:
        meta = _load_meta(shard)
        df = pd.read_parquet(shard)
        missing = {"protein_id", "start", "k", "nmp_el_rank"} - set(df.columns)
        if missing:
            raise ValueError(f"{shard}: cache is missing columns {sorted(missing)}")
        metas.append((shard, meta))
        frames.append(df)
        provenance.append({
            "shard": os.path.abspath(shard),
            "n_proteins": int(df["protein_id"].nunique()),
            "n_rows": int(len(df)),
            "saved_at": meta.get("saved_at"),
        })
        print(f"[merge] {shard}: {len(df)} rows / {df['protein_id'].nunique()} proteins")

    # 1. Identity: every shard must describe the same NMP measurement.
    ref_shard, ref_meta = metas[0]
    for shard, meta in metas[1:]:
        for field in IDENTITY_FIELDS:
            if meta.get(field) != ref_meta.get(field):
                raise ValueError(
                    f"shard identity mismatch on {field!r}: "
                    f"{ref_shard}={ref_meta.get(field)!r} vs {shard}={meta.get(field)!r}"
                )

    merged = pd.concat(frames, ignore_index=True)

    # 2. Disjointness: a window scored twice means the shard split was wrong.
    dup_mask = merged.duplicated(subset=["protein_id", "start", "k"], keep=False)
    if dup_mask.any():
        dup_pids = sorted(merged.loc[dup_mask, "protein_id"].astype(str).unique())
        raise ValueError(
            f"{len(dup_pids)} protein(s) appear in more than one shard, "
            f"e.g. {dup_pids[:5]}"
        )

    # 3. Coverage: exact match against the pool the cache claims to serve.
    pids = set(merged["protein_id"].astype(str))
    if args.expect_ids:
        expected = {line.strip() for line in open(args.expect_ids) if line.strip()}
        if pids != expected:
            raise ValueError(
                f"coverage mismatch vs {args.expect_ids}: "
                f"{len(expected - pids)} missing (e.g. {sorted(expected - pids)[:5]}), "
                f"{len(pids - expected)} unexpected (e.g. {sorted(pids - expected)[:5]})"
            )
        print(f"[merge] coverage OK: {len(pids)} proteins == {args.expect_ids}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    merged.to_parquet(args.output, index=False)

    out_meta = dict(ref_meta)
    out_meta.update({
        "n_proteins": len(pids),
        "saved_at": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "merged_from_shards": provenance,
        # The head half of a merged cache mixes per-shard head runs; only the NMP
        # half is meant to be reused (--reuse-nmp-from-cache reruns the head).
        "head_scores_authoritative": False,
    })
    with open(args.output + ".meta.json", "w") as fh:
        json.dump(out_meta, fh, indent=2)

    print(f"[merge] wrote {args.output}: {len(merged)} rows / {len(pids)} proteins")
    print(f"[merge] wrote {args.output}.meta.json")


if __name__ == "__main__":
    main()
