"""TDD for the Protenix refold-cache precompute driver (pure parts).

The driver builds Protenix-dialect JSON jobs (name = cache_key) from a generated
parquet for a SLURM shard, and (after the shared protenix-run fold step) normalizes
each fold into the shared refold cache via protenix_runner. Only the pure builders
are unit-tested here; the fold step is a shell call to the group Protenix install.
"""

from __future__ import annotations

import pandas as pd

from inverse_folding.evaluation.esmfold_runner import cache_key
from scripts.precompute_protenix_refold import build_protenix_json, records_for_shard


def test_build_protenix_json_is_a_list_of_monomer_jobs():
    jobs = build_protenix_json([("k1", "AAAA"), ("k2", "CCCC")])
    assert jobs == [
        {"name": "k1", "sequences": [{"proteinChain": {"sequence": "AAAA", "count": 1}}]},
        {"name": "k2", "sequences": [{"proteinChain": {"sequence": "CCCC", "count": 1}}]},
    ]


DTZ_ANION_SMILES = "O=c1c(Cc2ccccc2)nc2c(-c3ccccc3)[n-]c(-c3ccccc3)cn1-2"


def test_build_protenix_json_with_ligand_appends_entity():
    jobs = build_protenix_json([("k1", "AAAA")], ligand_smiles=DTZ_ANION_SMILES)
    assert jobs == [
        {
            "name": "k1",
            "sequences": [
                {"proteinChain": {"sequence": "AAAA", "count": 1}},
                {"ligand": {"ligand": DTZ_ANION_SMILES, "count": 1}},
            ],
        },
    ]


def test_build_protenix_json_empty_ligand_smiles_is_apo():
    # slurm passes --ligand-smiles "" for apo runs; empty string must stay protein-only
    jobs = build_protenix_json([("k1", "AAAA")], ligand_smiles="")
    assert jobs == [{"name": "k1", "sequences": [{"proteinChain": {"sequence": "AAAA", "count": 1}}]}]


def _parquet(tmp_path):
    df = pd.DataFrame(
        [
            {"protein_id": "p1", "design_idx": 0, "sequence": "AAAA"},
            {"protein_id": "p1", "design_idx": 1, "sequence": "AAAA"},  # exact dup
            {"protein_id": "p2", "design_idx": 0, "sequence": "CCCC"},
            {"protein_id": "p3", "design_idx": 0, "sequence": "GGGG"},
        ]
    )
    pq = tmp_path / "generated.parquet"
    df.to_parquet(pq)
    return str(pq)


def test_records_for_shard_dedups_and_uses_cache_key(tmp_path):
    pq = _parquet(tmp_path)
    allrecs = records_for_shard(pq, n_shards=1, shard_idx=0)
    assert len(allrecs) == 3  # deduped unique (protein_id, sequence)
    keys = {k for k, _ in allrecs}
    assert cache_key("p1", "AAAA") in keys
    assert (cache_key("p2", "CCCC"), "CCCC") in allrecs


def test_records_for_shard_round_robin_partitions_all(tmp_path):
    pq = _parquet(tmp_path)
    s0 = records_for_shard(pq, n_shards=2, shard_idx=0)
    s1 = records_for_shard(pq, n_shards=2, shard_idx=1)
    assert len(s0) + len(s1) == 3
    assert set(s0).isdisjoint(set(s1))
    assert set(s0) | set(s1) == set(records_for_shard(pq, 1, 0))
