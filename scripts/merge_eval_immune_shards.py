#!/usr/bin/env python
"""Merge independently evaluated immune tables with their provenance."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

TABLES = ["imm_head", "imm_nmp", "imm_head_residues", "imm_nmp_peptides"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--allele-tag", required=True, help="e.g. HLA-DRB1_04_01")
    ap.add_argument("--run-id", required=True, help="BASE run_id (no shard suffix)")
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--expected-proteins", type=int, default=0,
                    help="if >0, assert merged unique protein count matches")
    args = ap.parse_args()

    allele_dir = Path(args.output_root) / args.allele_tag
    shard_dirs = [
        allele_dir / f"{args.run_id}.shard{k:02d}of{args.n_shards:02d}"
        for k in range(args.n_shards)
    ]
    missing = [str(d) for d in shard_dirs if not d.is_dir()]
    if missing:
        sys.exit(f"FATAL: {len(missing)} shard dir(s) missing:\n  " + "\n  ".join(missing))

    merged_dir = allele_dir / args.run_id
    merged_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for table in TABLES:
        frames = []
        for d in shard_dirs:
            p = d / f"{table}.parquet"
            if not p.exists():
                sys.exit(f"FATAL: missing {table}.parquet in {d}")
            frames.append(pd.read_parquet(p))
        df = pd.concat(frames, ignore_index=True)
        # aggregated tables are one row per (protein_id, design_idx): no dup allowed
        if table in ("imm_head", "imm_nmp") and "protein_id" in df.columns:
            if df.duplicated(["protein_id", "design_idx"]).any():
                sys.exit(f"FATAL: duplicate (protein_id, design_idx) in merged {table}")
            df = df.sort_values(["protein_id", "design_idx"]).reset_index(drop=True)
        df.to_parquet(merged_dir / f"{table}.parquet", index=False)
        npro = int(df["protein_id"].nunique()) if "protein_id" in df.columns else None
        summary[table] = {"rows": int(len(df)), "proteins": npro}
        print(f"[merge] {table}: rows={len(df)} proteins={npro}")

    # failures.json — concat the per-shard failure lists
    all_failures = []
    for d in shard_dirs:
        fp = d / "failures.json"
        if fp.exists():
            obj = json.loads(fp.read_text())
            all_failures.extend(obj.get("failures", []) if isinstance(obj, dict) else obj)
    (merged_dir / "failures.json").write_text(
        json.dumps({"failures": all_failures}, indent=2) + "\n"
    )

    n_pro = summary["imm_nmp"]["proteins"]
    if args.expected_proteins and n_pro != args.expected_proteins:
        sys.exit(f"FATAL: merged proteins {n_pro} != expected {args.expected_proteins}")

    manifest = {
        "run_id": args.run_id,
        "allele_tag": args.allele_tag,
        "n_shards": args.n_shards,
        "merged_from": [str(d) for d in shard_dirs],
        "tables": summary,
        "n_failures": len(all_failures),
        "merged_dir": str(merged_dir.resolve()),
    }
    (merged_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[merge] DONE -> {merged_dir}  proteins={n_pro}  failures={len(all_failures)}")


if __name__ == "__main__":
    main()
