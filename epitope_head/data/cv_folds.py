"""Leakage-safe cluster-level K-fold cross-validation partitioning (Wave-3).

Folds are assigned at the **mmseqs90 cluster** level, never the protein level, so
homologous sequences (same ``cluster_rep``) never span a train/val/test boundary.
For each fold ``k``: ``test = fold k``, ``val = fold (k+1) % K``, ``train = the rest`` —
so every protein is the held-out *test* exactly once and *val* exactly once, and the
three sets within a fold are cluster-disjoint.

The reported CV test metric = mean ± std over the K held-out test folds; a separate
all-data model (no held-out) is the deployment checkpoint.
"""
from __future__ import annotations

from collections import defaultdict


def partition_clusters_into_folds(
    protein_to_cluster: dict[str, str],
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, int]:
    """Greedy balanced assignment of whole clusters to folds.

    Clusters are ordered by size (desc), tie-broken deterministically, then each is
    placed in the currently-smallest fold (by protein count). Deterministic given
    ``seed`` (used only to break exact size+name ties via a stable rotation).

    Returns ``{protein_id: fold_idx}``.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    clusters: dict[str, list[str]] = defaultdict(list)
    for pid, crep in protein_to_cluster.items():
        clusters[crep].append(pid)

    # Deterministic order: larger clusters first, then by a seed-rotated name key.
    def _key(item):
        crep, members = item
        return (-len(members), _rot(crep, seed))

    ordered = sorted(clusters.items(), key=_key)

    fold_counts = [0] * n_folds
    assignment: dict[str, int] = {}
    for crep, members in ordered:
        f = min(range(n_folds), key=lambda i: (fold_counts[i], i))
        for pid in members:
            assignment[pid] = f
        fold_counts[f] += len(members)
    return assignment


def _rot(s: str, seed: int) -> str:
    """Stable, seed-dependent reordering key for tie-breaking (no RNG)."""
    if not s:
        return s
    k = seed % len(s)
    return s[k:] + s[:k]


def build_fold_splits(
    fold_of_protein: dict[str, int],
    n_folds: int = 5,
) -> list[dict]:
    """Build per-fold train/val/test id lists from a protein→fold map.

    For fold ``k``: test = fold ``k``; val = fold ``(k+1) % n_folds``; train = the rest.
    Returns a list (one entry per fold) of ``{"fold": k, "train": [...], "val": [...],
    "test": [...]}`` with ids sorted for determinism.
    """
    by_fold: dict[int, list[str]] = defaultdict(list)
    for pid, f in fold_of_protein.items():
        by_fold[f].append(pid)

    out = []
    for k in range(n_folds):
        val_fold = (k + 1) % n_folds
        test_ids = sorted(by_fold.get(k, []))
        val_ids = sorted(by_fold.get(val_fold, []))
        train_ids = sorted(
            pid for f in range(n_folds) if f not in (k, val_fold)
            for pid in by_fold.get(f, [])
        )
        out.append({"fold": k, "train": train_ids, "val": val_ids, "test": test_ids})
    return out


def assert_no_cluster_leakage(
    fold_splits: list[dict],
    protein_to_cluster: dict[str, str],
) -> None:
    """Raise if any cluster_rep appears in >1 of {train,val,test} within any fold."""
    for fs in fold_splits:
        seen: dict[str, str] = {}
        for split_name in ("train", "val", "test"):
            for pid in fs[split_name]:
                crep = protein_to_cluster[pid]
                if crep in seen and seen[crep] != split_name:
                    raise AssertionError(
                        f"fold {fs['fold']}: cluster {crep} leaks across "
                        f"{seen[crep]} and {split_name} (protein {pid})"
                    )
                seen[crep] = split_name
