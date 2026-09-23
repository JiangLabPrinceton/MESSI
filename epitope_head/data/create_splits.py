"""Stage D v1.1: Leakage-aware train/val/test split.

Two-layer isolation:
  D1 — seq_hash grouping (exact-sequence dedup)
  D2 — mmseqs cluster grouping (homology isolation)
  D3 — weighted cluster-level split assignment
  D4 — integrity / diagnostic validation
  D5 — compatibility aliases
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── D1: seq_hash grouping ───────────────────────────────────────────────────

def compute_seq_hash(seq: str) -> str:
    """Deterministic SHA-256 hash of a protein sequence."""
    return hashlib.sha256(seq.encode("ascii")).hexdigest()[:16]


def build_seq_hash_groups(protein_df: pd.DataFrame) -> pd.DataFrame:
    """Build seq_hash -> protein_id mapping.

    Returns DataFrame with columns: protein_id, seq_hash
    One row per protein_id.
    """
    records = []
    for _, row in protein_df.iterrows():
        records.append({
            "protein_id": row["protein_id"],
            "seq_hash": compute_seq_hash(row["protein_seq"]),
        })
    df = pd.DataFrame(records).sort_values("protein_id").reset_index(drop=True)
    return df


# ── D2: mmseqs clustering ──────────────────────────────────────────────────

def run_mmseqs_cluster(
    seq_hash_groups: pd.DataFrame,
    protein_df: pd.DataFrame,
    mmseqs_bin: str = "mmseqs",
    identity: float = 0.90,
    coverage: float = 0.80,
    cov_mode: int = 2,
) -> pd.DataFrame:
    """Run mmseqs cluster on representative sequences (one per seq_hash).

    Returns DataFrame: seq_hash, cluster_rep (representative seq_hash).
    """
    # Pick one representative protein per seq_hash
    seq_map = dict(zip(protein_df["protein_id"], protein_df["protein_seq"]))
    hash_to_rep_pid = (
        seq_hash_groups.groupby("seq_hash")["protein_id"]
        .first()
        .to_dict()
    )
    hash_to_seq = {h: seq_map[pid] for h, pid in hash_to_rep_pid.items()}

    with tempfile.TemporaryDirectory(prefix="mmseqs_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Write representative FASTA (keyed by seq_hash)
        fasta_path = tmpdir / "reps.fasta"
        with open(fasta_path, "w") as f:
            for sh in sorted(hash_to_seq.keys()):
                f.write(f">{sh}\n{hash_to_seq[sh]}\n")

        db_path = tmpdir / "seqDB"
        cluster_db = tmpdir / "clusterDB"
        tsv_path = tmpdir / "clusters.tsv"

        # createdb
        subprocess.run(
            [mmseqs_bin, "createdb", str(fasta_path), str(db_path)],
            check=True, capture_output=True,
        )

        # cluster
        subprocess.run(
            [
                mmseqs_bin, "cluster",
                str(db_path), str(cluster_db), str(tmpdir / "tmp"),
                "--min-seq-id", str(identity),
                "-c", str(coverage),
                "--cov-mode", str(cov_mode),
                "--threads", "1",
            ],
            check=True, capture_output=True,
        )

        # createtsv
        subprocess.run(
            [mmseqs_bin, "createtsv", str(db_path), str(db_path),
             str(cluster_db), str(tsv_path)],
            check=True, capture_output=True,
        )

        # Parse TSV: columns are (representative, member)
        cluster_df = pd.read_csv(
            tsv_path, sep="\t", header=None,
            names=["cluster_rep", "seq_hash"],
        )

    # Ensure all seq_hashes are present
    all_hashes = set(hash_to_seq.keys())
    clustered_hashes = set(cluster_df["seq_hash"])
    missing = all_hashes - clustered_hashes
    if missing:
        raise ValueError(f"mmseqs missed {len(missing)} seq_hashes")

    return cluster_df.sort_values("seq_hash").reset_index(drop=True)


def build_cluster_assignment(
    seq_hash_groups: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> pd.DataFrame:
    """Expand cluster assignment to protein_id level.

    Returns DataFrame: protein_id, seq_hash, cluster_rep
    """
    merged = seq_hash_groups.merge(cluster_df, on="seq_hash", how="left")
    if merged["cluster_rep"].isna().any():
        raise ValueError("Some protein_ids have no cluster assignment")
    return merged.sort_values("protein_id").reset_index(drop=True)


# ── D3: Weighted cluster-level split ────────────────────────────────────────

def weighted_cluster_split(
    cluster_assignment: pd.DataFrame,
    protein_df: pd.DataFrame,
    ratios: list[float],
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Split by cluster, weighted by #unique_positive_spans.

    Greedy bin-packing: sort clusters by weight descending (tie-break by
    cluster_rep lexicographic), assign each to the split most below target.

    Returns (train_ids, val_ids, test_ids) as sorted lists.
    """
    assert len(ratios) == 3
    assert abs(sum(ratios) - 1.0) < 1e-9

    # Compute per-protein positive count
    pos_counts = dict(zip(protein_df["protein_id"], protein_df["positive_count"]))

    # Per-cluster weight = sum of positive_count across member proteins
    cluster_weights = {}
    cluster_proteins = {}
    for cluster_rep, group in cluster_assignment.groupby("cluster_rep"):
        pids = sorted(group["protein_id"].tolist())
        weight = sum(pos_counts.get(pid, 0) for pid in pids)
        cluster_weights[cluster_rep] = weight
        cluster_proteins[cluster_rep] = pids

    # Sort clusters: weight descending, then cluster_rep ascending for tie-break
    sorted_clusters = sorted(
        cluster_weights.keys(),
        key=lambda c: (-cluster_weights[c], c),
    )

    # Shuffle with seed for randomization within same-weight groups
    # (but greedy assignment is deterministic given the sorted order)
    total_weight = sum(cluster_weights.values())
    target_weights = [r * total_weight for r in ratios]
    current_weights = [0.0, 0.0, 0.0]
    split_names = ["train", "val", "test"]
    assignments = {c: None for c in sorted_clusters}

    for cluster_rep in sorted_clusters:
        # Find the split most below its target (relative deficit)
        deficits = [
            (target_weights[i] - current_weights[i]) / max(target_weights[i], 1e-9)
            for i in range(3)
        ]
        best_split = max(range(3), key=lambda i: deficits[i])
        assignments[cluster_rep] = best_split
        current_weights[best_split] += cluster_weights[cluster_rep]

    # Collect protein_ids per split
    split_ids: list[list[str]] = [[], [], []]
    for cluster_rep, split_idx in assignments.items():
        split_ids[split_idx].extend(cluster_proteins[cluster_rep])

    return sorted(split_ids[0]), sorted(split_ids[1]), sorted(split_ids[2])


# ── D4: Validation ─────────────────────────────────────────────────────────

def validate_split_disjointness(
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    universe: set[str],
) -> None:
    """Validate split disjointness and full coverage. Raises on violation."""
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    tv = train_set & val_set
    if tv:
        raise ValueError(f"Train-val overlap: {len(tv)} IDs")
    tt = train_set & test_set
    if tt:
        raise ValueError(f"Train-test overlap: {len(tt)} IDs")
    vt = val_set & test_set
    if vt:
        raise ValueError(f"Val-test overlap: {len(vt)} IDs")

    union = train_set | val_set | test_set
    missing = universe - union
    if missing:
        raise ValueError(f"Missing {len(missing)} IDs from splits")
    extra = union - universe
    if extra:
        raise ValueError(f"Extra {len(extra)} IDs not in universe")


def validate_no_cross_split(
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    group_col: str,
    assignment_df: pd.DataFrame,
) -> None:
    """Validate that no group (seq_hash or cluster_rep) crosses splits."""
    pid_to_split = {}
    for pid in train_ids:
        pid_to_split[pid] = "train"
    for pid in val_ids:
        pid_to_split[pid] = "val"
    for pid in test_ids:
        pid_to_split[pid] = "test"

    assignment_df = assignment_df.copy()
    assignment_df["_split"] = assignment_df["protein_id"].map(pid_to_split)

    for group_val, grp in assignment_df.groupby(group_col):
        splits_in_group = grp["_split"].dropna().unique()
        if len(splits_in_group) > 1:
            raise ValueError(
                f"{group_col}={group_val} spans multiple splits: {sorted(splits_in_group)}"
            )


# ── Utility ─────────────────────────────────────────────────────────────────

def write_id_file(ids: list[str], path: Path) -> None:
    """Write sorted IDs to a text file, one per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for pid in sorted(ids):
            f.write(pid + "\n")


def read_id_file(path: Path) -> list[str]:
    """Read protein IDs from a text file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]
