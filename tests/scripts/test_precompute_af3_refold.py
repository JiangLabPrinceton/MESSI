"""TDD for the AlphaFold3 refold-cache precompute driver (pure parts)."""

from __future__ import annotations

import pandas as pd

from inverse_folding.evaluation.esmfold_runner import cache_key
from scripts.precompute_af3_refold import build_af3_json, records_for_shard


def test_build_af3_json_no_msa_single_seq_fast_path():
    j = build_af3_json("k1", "AAAA", no_msa=True)
    assert j["name"] == "k1"
    assert j["dialect"] == "alphafold3" and j["version"] == 1
    assert j["modelSeeds"] == [1]
    pc = j["sequences"][0]["protein"]
    assert pc["id"] == "A" and pc["sequence"] == "AAAA"
    # empty MSA + templates make --mode inference skip the genetic search
    assert pc["unpairedMsa"] == "" and pc["pairedMsa"] == "" and pc["templates"] == []


def test_build_af3_json_full_msa_omits_msa_fields():
    j = build_af3_json("k1", "AAAA", no_msa=False)
    pc = j["sequences"][0]["protein"]
    # omitting the MSA fields makes --mode data run the local MSA pipeline
    assert "unpairedMsa" not in pc and "pairedMsa" not in pc and "templates" not in pc
    assert pc["sequence"] == "AAAA"


DTZ_ANION_SMILES = "O=c1c(Cc2ccccc2)nc2c(-c3ccccc3)[n-]c(-c3ccccc3)cn1-2"


def test_build_af3_json_apo_has_no_ligand_entity():
    # default (no ligand_smiles) is protein-only — holo must be opt-in
    j = build_af3_json("k1", "AAAA", no_msa=False)
    assert len(j["sequences"]) == 1
    assert all("ligand" not in s for s in j["sequences"])


def test_build_af3_json_with_ligand_appends_smiles_entity():
    j = build_af3_json("k1", "AAAA", no_msa=False, ligand_smiles=DTZ_ANION_SMILES)
    # protein entity unchanged, ligand appended as a distinct chain id (not "A")
    assert j["sequences"][0]["protein"]["sequence"] == "AAAA"
    ligands = [s["ligand"] for s in j["sequences"] if "ligand" in s]
    assert ligands == [{"id": "L", "smiles": DTZ_ANION_SMILES}]


def test_build_af3_json_empty_ligand_smiles_is_apo():
    # slurm passes --ligand-smiles "" for apo runs; empty string must stay protein-only
    j = build_af3_json("k1", "AAAA", no_msa=True, ligand_smiles="")
    assert all("ligand" not in s for s in j["sequences"])


def test_records_for_shard_dedups_and_uses_cache_key(tmp_path):
    df = pd.DataFrame(
        [
            {"protein_id": "p1", "sequence": "AAAA"},
            {"protein_id": "p1", "sequence": "AAAA"},  # exact dup
            {"protein_id": "p2", "sequence": "CCCC"},
        ]
    )
    pq = tmp_path / "g.parquet"
    df.to_parquet(pq)
    recs = records_for_shard(str(pq), n_shards=1, shard_idx=0)
    assert len(recs) == 2
    assert (cache_key("p1", "AAAA"), "AAAA") in recs
