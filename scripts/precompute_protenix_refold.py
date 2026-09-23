#!/usr/bin/env python
"""Prepare, predict, and normalize Protenix structures with reusable workers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_on_path() -> None:
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


_repo_on_path()
from inverse_folding.evaluation.refold_normalize import fold_records_from_parquet  # noqa: E402


def _split_colabfold_a3m(a3m_text: str, seq: str, out_dir: Path) -> tuple[str, str]:
    """Local ColabFold a3m -> Protenix (pairing.a3m = query-only, homomer self-pair;
    non_pairing.a3m = full depth). Drops any leading ``#len\\tcard`` comment line. Returns
    the two absolute paths for pairedMsaPath / unpairedMsaPath."""
    lines = a3m_text.splitlines()
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    out_dir.mkdir(parents=True, exist_ok=True)
    npair = out_dir / "non_pairing.a3m"
    npair.write_text("\n".join(lines) + "\n")
    pair = out_dir / "pairing.a3m"
    pair.write_text(f">query\n{seq}\n")
    return str(pair), str(npair)


def build_protenix_json(
    records: list[tuple[str, str]], ligand_smiles: str | None = None,
    msa_a3m_dir: Path | None = None, msa_px_dir: Path | None = None,
) -> list[dict]:
    """Protenix-dialect job list (one single-count protein monomer per record).

    ``ligand_smiles`` (optional) cofolds a single-count SMILES ligand for a HOLO
    prediction; empty/None keeps the apo protein-only monomer.

    ``msa_a3m_dir`` (optional): inject our local ColabFold ``<cache_key>.a3m`` — split into
    ``pairedMsaPath``/``unpairedMsaPath`` under ``msa_px_dir/<cache_key>/`` — so
    ``protenix-run pred --use_msa true`` reads the LOCAL MSA and no server call is made
    (replaces the deprecated login-node server path).
    """
    jobs: list[dict] = []
    for key, seq in records:
        chain: dict = {"sequence": seq, "count": 1}
        if msa_a3m_dir is not None:
            a3m = msa_a3m_dir / f"{key}.a3m"
            if not a3m.is_file():
                raise SystemExit(f"--msa-a3m-dir set but {a3m} missing (run local ColabFold MSA first)")
            pair, npair = _split_colabfold_a3m(a3m.read_text(), seq, (msa_px_dir or msa_a3m_dir) / key)
            chain["pairedMsaPath"] = pair
            chain["unpairedMsaPath"] = npair
        sequences: list[dict] = [{"proteinChain": chain}]
        if ligand_smiles:
            sequences.append({"ligand": {"ligand": ligand_smiles, "count": 1}})
        jobs.append({"name": key, "sequences": sequences})
    return jobs


def records_for_shard(parquet_path: str, n_shards: int, shard_idx: int) -> list[tuple[str, str]]:
    """This shard's ``(cache_key, sequence)`` records (round-robin over unique folds)."""
    recs = fold_records_from_parquet(parquet_path)
    return recs[shard_idx::n_shards] if n_shards > 1 else recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("build-json", "normalize", "emit-fasta"), required=True)
    ap.add_argument("--from-parquet", required=True, help="generated parquet (protein_id, sequence)")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--out-json", default=None, help="build-json: Protenix JSON output path")
    ap.add_argument("--fasta-out", default=None,
                    help="emit-fasta: write this shard's <cache_key>\\n<seq> FASTA (feed local ColabFold MSA)")
    ap.add_argument("--msa-a3m-dir", default=None,
                    help="build-json: inject <dir>/<cache_key>.a3m (local ColabFold) as paired/unpaired MSA")
    ap.add_argument("--msa-px-dir", default=None,
                    help="build-json: where to write split pairing/non_pairing.a3m (default: alongside --msa-a3m-dir)")
    ap.add_argument("--ligand-smiles", default=None,
                    help="build-json: cofold this SMILES ligand (count 1) for a HOLO prediction; "
                         "empty/omitted = apo protein-only monomer")
    ap.add_argument("--protenix-out", default=None, help="normalize: Protenix `pred` output dir")
    ap.add_argument("--cache-dir", default=None, help="normalize: shared refold cache dir")
    args = ap.parse_args()

    recs = records_for_shard(args.from_parquet, args.n_shards, args.shard_idx)

    if args.mode == "emit-fasta":
        if not args.fasta_out:
            raise SystemExit("--fasta-out is required for --mode emit-fasta")
        out = Path(args.fasta_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(f">{key}\n{seq}\n" for key, seq in recs))
        print(f"[emit-fasta] shard {args.shard_idx}/{args.n_shards}: {len(recs)} seqs -> {out}")
        return

    if args.mode == "build-json":
        if not args.out_json:
            raise SystemExit("--out-json is required for --mode build-json")
        jobs = build_protenix_json(
            recs, ligand_smiles=args.ligand_smiles,
            msa_a3m_dir=Path(args.msa_a3m_dir) if args.msa_a3m_dir else None,
            msa_px_dir=Path(args.msa_px_dir) if args.msa_px_dir else None,
        )
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(jobs, indent=2))
        src = "colabfold-a3m" if args.msa_a3m_dir else "server/none"
        print(f"[build-json] shard {args.shard_idx}/{args.n_shards}: {len(jobs)} monomer jobs "
              f"(msa={src}) -> {out}")
        return

    # normalize
    if not (args.protenix_out and args.cache_dir):
        raise SystemExit("--protenix-out and --cache-dir are required for --mode normalize")
    from inverse_folding.evaluation.protenix_runner import normalize_protenix_to_cache

    n_ok, missing = 0, []
    for key, _seq in recs:
        try:
            normalize_protenix_to_cache(
                args.protenix_out,
                key,
                cache_dir=args.cache_dir,
                key=key,
                protein_only=bool(args.ligand_smiles),
            )
            n_ok += 1
        except FileNotFoundError:
            missing.append(key)
    print(f"[normalize] shard {args.shard_idx}/{args.n_shards}: {n_ok} cached, "
          f"{len(missing)} missing -> {args.cache_dir}")
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)} Protenix fold(s) missing (first: {missing[:3]}); "
            "precompute incomplete — do not run struct eval on a partial cache"
        )
    print("[done]")


if __name__ == "__main__":
    main()
