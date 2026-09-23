"""Dataset build orchestrator: Stage A-D.

Usage:
    python -m epitope_head.data.build_dataset [--config path/to/data.yaml] [--stage a|b|c|d|ab|bc|cd|abc|abcd]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from epitope_head.configs import load_data_config
from epitope_head.data.parse_mhc_if_v2 import (
    RejectionLedger,
    apply_allele_filter,
    explode_positions,
    normalize_coordinates,
    read_and_parse,
)
from epitope_head.data.join_fasta import JoinStats, build_fasta_index, join_and_validate
from epitope_head.data.make_protein_samples import (
    AggStats,
    aggregate_protein_samples,
    validate_protein_samples,
)
from epitope_head.data.create_splits import (
    build_cluster_assignment,
    build_seq_hash_groups,
    run_mmseqs_cluster,
    validate_no_cross_split,
    validate_split_disjointness,
    weighted_cluster_split,
    write_id_file,
)
from epitope_head.data.validators import validate_span_records

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_manifest_dir(cfg: dict) -> Path:
    """Resolve manifest output directory from config.

    If cfg contains 'output_subdir', outputs go to
    outputs/manifests/{output_subdir}/. Otherwise, outputs/manifests/.
    """
    base = PROJECT_ROOT / "outputs" / "manifests"
    subdir = cfg.get("output_subdir")
    return base / subdir if subdir else base


def build_stage_a(cfg: dict | None = None) -> dict:
    """Run full Stage A pipeline. Returns summary dict."""
    if cfg is None:
        cfg = load_data_config()

    tsv_path = PROJECT_ROOT / cfg["tsv_path"]
    out_dir = _resolve_manifest_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = RejectionLedger()

    # A1: read + parse
    df, parse_ledger = read_and_parse(str(tsv_path))
    ledger.counts.update(parse_ledger.counts)

    # A2: explode positions
    df = explode_positions(df, ledger)

    # A3: coordinate normalization + length filter
    df = normalize_coordinates(df, cfg["min_k"], cfg["max_k"], ledger)

    # A4: allele filter
    sa_cfg = cfg["stage_a"]
    strict_df, balanced_df = apply_allele_filter(
        df,
        target_allele=cfg["target_allele"],
        multi_ratio=sa_cfg["profile_balanced"]["multi_ratio"],
        multi_seed=sa_cfg["profile_balanced"].get("multi_seed", 42),
        ledger=ledger,
    )

    # A5: validate + write
    strict_df = validate_span_records(strict_df)
    balanced_df = validate_span_records(balanced_df)

    strict_path = out_dir / "span_records_strict.parquet"
    balanced_path = out_dir / "span_records_balanced.parquet"
    strict_df.to_parquet(strict_path, index=False)
    balanced_df.to_parquet(balanced_path, index=False)

    summary = {
        "strict_rows": len(strict_df),
        "balanced_rows": len(balanced_df),
        "rejection_ledger": ledger.summary(),
        "artifacts": [str(strict_path), str(balanced_path)],
    }

    logger.info("Stage A complete. Strict: %d, Balanced: %d",
                summary["strict_rows"], summary["balanced_rows"])
    logger.info("Rejection ledger: %s", summary["rejection_ledger"])
    return summary


def build_stage_b(cfg: dict | None = None) -> dict:
    """Run Stage B: FASTA join + validation on both profiles. Returns summary dict."""
    if cfg is None:
        cfg = load_data_config()

    manifest_dir = _resolve_manifest_dir(cfg)
    fasta_path = PROJECT_ROOT / cfg["fasta_path"]

    # Build FASTA index once
    fasta_index = build_fasta_index(str(fasta_path))

    results = {}
    for profile in ["strict", "balanced"]:
        input_path = manifest_dir / f"span_records_{profile}.parquet"
        span_df = pd.read_parquet(input_path)

        stats = JoinStats()
        joined_df = join_and_validate(span_df, fasta_index, stats)

        output_path = manifest_dir / f"span_records_with_seq_{profile}.parquet"
        joined_df.to_parquet(output_path, index=False)

        results[profile] = {
            "input_rows": len(span_df),
            "output_rows": len(joined_df),
            "unique_proteins": joined_df["protein_id"].nunique() if len(joined_df) > 0 else 0,
            "join_stats": stats.summary(),
            "artifact": str(output_path),
        }
        logger.info("Stage B [%s]: %d -> %d rows", profile,
                     len(span_df), len(joined_df))
        logger.info("Stage B [%s] stats: %s", profile, stats.summary())

    return results


def build_stage_c(cfg: dict | None = None) -> dict:
    """Run Stage C: ProteinSample aggregation on both profiles. Returns summary dict."""
    if cfg is None:
        cfg = load_data_config()

    manifest_dir = _resolve_manifest_dir(cfg)

    results = {}
    for profile in ["strict", "balanced"]:
        input_path = manifest_dir / f"span_records_with_seq_{profile}.parquet"
        span_df = pd.read_parquet(input_path)

        stats = AggStats()
        protein_df = aggregate_protein_samples(span_df, stats)
        validate_protein_samples(protein_df)

        output_path = manifest_dir / f"protein_samples_{profile}.parquet"
        protein_df.to_parquet(output_path, index=False)

        results[profile] = {
            "input_spans": len(span_df),
            "output_proteins": len(protein_df),
            "agg_stats": stats.summary(),
            "artifact": str(output_path),
        }
        logger.info("Stage C [%s]: %d spans -> %d proteins", profile,
                     len(span_df), len(protein_df))
        logger.info("Stage C [%s] stats: %s", profile, stats.summary())

    # Strict alias
    alias_path = manifest_dir / "protein_samples.parquet"
    strict_path = manifest_dir / "protein_samples_strict.parquet"
    import shutil
    shutil.copy2(strict_path, alias_path)
    logger.info("Strict alias written: %s", alias_path)

    return results


def build_stage_d(cfg: dict | None = None) -> dict:
    """Run Stage D v1.1: seq_hash + mmseqs cluster split. Returns summary dict."""
    if cfg is None:
        cfg = load_data_config()

    import json
    import shutil

    manifest_dir = _resolve_manifest_dir(cfg)
    splits_dir = manifest_dir / "splits"
    ratios = cfg["split_ratios"]
    seed = cfg["split_seed"]
    mmseqs_bin = cfg.get("mmseqs_bin", "mmseqs")
    cluster_identity = cfg.get("cluster_identity", 0.90)
    cluster_coverage = cfg.get("cluster_coverage", 0.80)
    cluster_cov_mode = cfg.get("cluster_cov_mode", 2)

    results = {}

    for profile in ["strict", "balanced"]:
        input_path = manifest_dir / f"protein_samples_{profile}.parquet"
        protein_df = pd.read_parquet(input_path)
        protein_ids = sorted(protein_df["protein_id"].unique().tolist())

        profile_dir = splits_dir / profile
        profile_dir.mkdir(parents=True, exist_ok=True)

        # D1: seq_hash grouping
        seq_hash_groups = build_seq_hash_groups(protein_df)
        seq_hash_groups.to_parquet(profile_dir / "seq_hash_groups.parquet", index=False)
        n_unique_seq = seq_hash_groups["seq_hash"].nunique()
        logger.info("Stage D [%s]: %d proteins -> %d unique sequences (seq_hash)",
                     profile, len(protein_ids), n_unique_seq)

        # D2: mmseqs clustering
        cluster_df = run_mmseqs_cluster(
            seq_hash_groups, protein_df,
            mmseqs_bin=mmseqs_bin,
            identity=cluster_identity,
            coverage=cluster_coverage,
            cov_mode=cluster_cov_mode,
        )
        cluster_df.to_csv(profile_dir / "mmseq90_clusters.tsv", sep="\t", index=False)
        n_clusters = cluster_df["cluster_rep"].nunique()
        logger.info("Stage D [%s]: %d seq_hashes -> %d clusters",
                     profile, n_unique_seq, n_clusters)

        # Expand to protein_id level
        cluster_assignment = build_cluster_assignment(seq_hash_groups, cluster_df)
        cluster_assignment.to_parquet(profile_dir / "cluster_assignment.parquet", index=False)

        # D3: weighted cluster-level split
        train_ids, val_ids, test_ids = weighted_cluster_split(
            cluster_assignment, protein_df, ratios, seed,
        )

        # D4: validate integrity
        validate_split_disjointness(train_ids, val_ids, test_ids, set(protein_ids))
        validate_no_cross_split(train_ids, val_ids, test_ids, "seq_hash", cluster_assignment)
        validate_no_cross_split(train_ids, val_ids, test_ids, "cluster_rep", cluster_assignment)

        # Write split ID files
        write_id_file(train_ids, profile_dir / "train_ids.txt")
        write_id_file(val_ids, profile_dir / "val_ids.txt")
        write_id_file(test_ids, profile_dir / "test_ids.txt")

        # Diagnostics summary
        pos_counts = dict(zip(protein_df["protein_id"], protein_df["positive_count"]))
        train_weight = sum(pos_counts.get(p, 0) for p in train_ids)
        val_weight = sum(pos_counts.get(p, 0) for p in val_ids)
        test_weight = sum(pos_counts.get(p, 0) for p in test_ids)
        total_weight = train_weight + val_weight + test_weight

        diagnostics = {
            "split_method": "seq_hash + mmseqs_cluster",
            "cluster_params": {
                "identity": cluster_identity,
                "coverage": cluster_coverage,
                "cov_mode": cluster_cov_mode,
            },
            "n_proteins": len(protein_ids),
            "n_unique_sequences": n_unique_seq,
            "n_clusters": n_clusters,
            "split_sizes": {
                "train": len(train_ids), "val": len(val_ids), "test": len(test_ids),
            },
            "split_weights": {
                "train": train_weight, "val": val_weight, "test": test_weight,
                "total": total_weight,
            },
            "split_weight_fractions": {
                "train": round(train_weight / max(total_weight, 1), 4),
                "val": round(val_weight / max(total_weight, 1), 4),
                "test": round(test_weight / max(total_weight, 1), 4),
            },
        }

        diag_path = profile_dir / "split_diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)

        results[profile] = {
            "total_proteins": len(protein_ids),
            "n_unique_sequences": n_unique_seq,
            "n_clusters": n_clusters,
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "weight_fractions": diagnostics["split_weight_fractions"],
        }
        logger.info("Stage D [%s]: %d proteins, %d clusters -> train=%d val=%d test=%d",
                     profile, len(protein_ids), n_clusters,
                     len(train_ids), len(val_ids), len(test_ids))
        logger.info("Stage D [%s] weight fractions: %s",
                     profile, diagnostics["split_weight_fractions"])

    # D5: strict aliases at top-level splits/
    strict_dir = splits_dir / "strict"
    for fname in ["train_ids.txt", "val_ids.txt", "test_ids.txt"]:
        shutil.copy2(strict_dir / fname, splits_dir / fname)
    logger.info("Strict aliases written to %s", splits_dir)

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build dataset artifacts")
    parser.add_argument("--config", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--stage", type=str, default="abcd",
                        choices=["a", "b", "c", "d", "ab", "bc", "cd", "abc", "bcd", "abcd"],
                        help="Which stage(s) to run")
    args = parser.parse_args()

    cfg = load_data_config(args.config) if args.config else None

    if "a" in args.stage:
        summary_a = build_stage_a(cfg)
        print("=== Stage A ===")
        print(summary_a)

    if "b" in args.stage:
        summary_b = build_stage_b(cfg)
        print("=== Stage B ===")
        for profile, info in summary_b.items():
            print(f"  [{profile}] {info}")

    if "c" in args.stage:
        summary_c = build_stage_c(cfg)
        print("=== Stage C ===")
        for profile, info in summary_c.items():
            print(f"  [{profile}] {info}")

    if "d" in args.stage:
        summary_d = build_stage_d(cfg)
        print("=== Stage D ===")
        for profile, info in summary_d.items():
            print(f"  [{profile}] {info}")


if __name__ == "__main__":
    main()
