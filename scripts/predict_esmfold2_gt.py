#!/usr/bin/env python
"""Predict GT structures for the uricase missing-structure case set with ESMFold2.

Runs in the dedicated `esmfold2` conda env (NOT immune-design: the biohub `esm`
package shadows fair-esm, so it must be isolated). Reads a FASTA, folds each
sequence with ESMFold2, writes per-protein mmCIF (and a best-effort PDB), and
records mean pLDDT / pTM with a pass/fail flag against the GT-eligibility gate.

GT-eligibility gate (both tunable; raw values are always recorded so the
threshold can be revisited without re-folding):
    mean pLDDT >= --plddt-min   (default 90, AF "very high confidence" band)
    pTM        >= --ptm-min      (default 0.80, "strong" global topology)

The manifest is append-only JSONL with resume support: re-running skips ids
already present, so a crashed/timed-out job can be relaunched safely.

Smoke test: pass --limit 1 to fold only the first sequence; the first protein
prints verbose diagnostics (result/plddt types, shapes, pTM, mmCIF/PDB write
status) so the real ESMFold2 API can be confirmed before the full run.
"""
import argparse
import io
import json
import re
import time
from pathlib import Path


def read_fasta(path):
    recs, cur = [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            cur = line[1:].strip()
            recs.append([cur, []])
        elif line.strip() and recs:
            recs[-1][1].append(line.strip())
    return [(pid, "".join(parts)) for pid, parts in recs]


def cif_to_pdb(cif_text):
    """Best-effort mmCIF -> PDB via biotite (installed as an esm dependency)."""
    import numpy as np
    import biotite.structure.io.pdb as pdb
    import biotite.structure.io.pdbx as pdbx

    cif = pdbx.CIFFile.read(io.StringIO(cif_text))
    arr = pdbx.get_structure(cif, model=1, extra_fields=["b_factor"])
    if "b_factor" in arr.get_annotation_categories() and not np.isfinite(arr.b_factor).all():
        arr.del_annotation("b_factor")
    pf = pdb.PDBFile()
    pdb.set_structure(pf, arr)
    sio = io.StringIO()
    pf.write(sio)
    return sio.getvalue()


def _ensure_repo_on_path():
    import sys

    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def build_fold_records_from_parquet(parquet_path):
    """Fold list for the shared refold cache from a generated parquet.

    Returns ``[(header_id, sequence), ...]`` over UNIQUE ``(protein_id, sequence)``
    pairs, with ``header_id = cache_key(protein_id, sequence)`` so the ESMFold2
    on-disk id already equals the shared refold cache key (no key drift).
    """
    import pandas as pd

    _ensure_repo_on_path()
    from inverse_folding.evaluation.esmfold_runner import cache_key

    df = pd.read_parquet(parquet_path)
    for col in ("protein_id", "sequence"):
        if col not in df.columns:
            raise SystemExit(f"FATAL: '{col}' not in {parquet_path}")
    seen, recs = set(), []
    for pid, seq in zip(df["protein_id"].astype(str), df["sequence"].astype(str)):
        pair = (pid, seq)
        if pair in seen:
            continue
        seen.add(pair)
        recs.append((cache_key(pid, seq), seq))
    return recs


def cache_layout_from_manifest(manifest_path, mmcif_dir, cache_dir, native_scale="0-1"):
    """Normalize every folded mmCIF in a manifest into the shared refold cache.

    Writes ``<cache_dir>/<id>.pdb`` (single chain) + ``<id>.plddt`` (0-100) for each
    manifest row, where ``id`` is the cache key and ``file`` locates the mmCIF.
    Resume-safe: processes ALL manifest rows (not just this run's freshly folded
    ones). Returns the number of cache entries written.
    """
    _ensure_repo_on_path()
    from inverse_folding.evaluation.refold_normalize import normalize_to_cache

    n = 0
    for line in Path(manifest_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cif_path = str(Path(mmcif_dir) / f"{row['file']}.cif")
        normalize_to_cache(
            cache_dir, str(row["id"]), cif_path=cif_path,
            mean_plddt=float(row["mean_plddt"]), native_scale=native_scale,
        )
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", default=None, help="Input FASTA (to_fold.fasta). One of --fasta/--from-parquet.")
    ap.add_argument("--from-parquet", default=None,
                    help="Build the fold list from a generated parquet (unique protein_id,sequence; "
                         "header id = cache_key) instead of --fasta.")
    ap.add_argument("--cache-layout", default=None,
                    help="After folding, also normalize each mmCIF into <DIR>/<cache_key>.pdb + .plddt "
                         "(0-100) for the shared refold cache (evaluate_phase_c --refold-model esmfold2).")
    ap.add_argument("--emit-fold-fasta", default=None,
                    help="Build the fold FASTA (header id = cache_key) from --from-parquet and EXIT "
                         "without folding. Run this in immune-design (needs pyarrow); the esmfold2 "
                         "fold step then reads only the emitted FASTA (no parquet).")
    ap.add_argument("--out-dir", required=True, help="Output root (mmcif/, pdb/, gt_manifest.jsonl).")
    ap.add_argument("--plddt-min", type=float, default=0.90,
                    help="Min mean pLDDT. ESMFold2 pLDDT is 0-1 scale; 0.90 == AF 'very high'.")
    ap.add_argument("--ptm-min", type=float, default=0.80, help="Min pTM (0-1 scale).")
    ap.add_argument("--num-loops", type=int, default=3)
    ap.add_argument("--num-sampling-steps", type=int, default=50)
    ap.add_argument("--num-diffusion-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0=all; >0 fold only first N (smoke).")
    ap.add_argument("--num-shards", type=int, default=1, help="Split FASTA into N shards (SLURM array).")
    ap.add_argument("--shard-idx", type=int, default=0, help="Which shard to run, in [0, num_shards).")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "mmcif").mkdir(parents=True, exist_ok=True)
    (out / "pdb").mkdir(parents=True, exist_ok=True)
    # Per-shard manifest so concurrent array tasks never append to the same file
    # (mmcif/pdb filenames are per-protein-id, so they never collide across shards).
    if args.num_shards > 1:
        manifest_path = out / f"gt_manifest.shard{args.shard_idx}of{args.num_shards}.jsonl"
    else:
        manifest_path = out / "gt_manifest.jsonl"

    # Build the fold list BEFORE importing esm/torch. --from-parquet needs pyarrow,
    # which lives in immune-design, NOT the esmfold2 env; --emit-fold-fasta writes the
    # FASTA and returns here, so the parquet read runs in immune-design and the actual
    # fold reads only that FASTA in the esmfold2 env.
    if args.from_parquet:
        recs = build_fold_records_from_parquet(args.from_parquet)
    elif args.fasta:
        recs = read_fasta(args.fasta)
    else:
        raise SystemExit("FATAL: provide --fasta or --from-parquet")

    if args.emit_fold_fasta:
        fasta_out = Path(args.emit_fold_fasta)
        fasta_out.parent.mkdir(parents=True, exist_ok=True)
        with open(fasta_out, "w") as fh:
            for pid, seq in recs:
                fh.write(f">{pid}\n{seq}\n")
        print(f"[emit-fasta] {len(recs)} records -> {fasta_out}", flush=True)
        return

    import torch
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    print(f"[load] torch {torch.__version__} cuda_avail={torch.cuda.is_available()}", flush=True)
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").to(args.device).eval()
    builder = ESMFold2InputBuilder()

    if args.num_shards > 1:
        recs = recs[args.shard_idx :: args.num_shards]  # round-robin; balances length
    if args.limit:
        recs = recs[: args.limit]

    done = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [(pid, seq) for pid, seq in recs if pid not in done]
    print(f"[run] {len(todo)} to fold ({len(done)} already done) -> {out}", flush=True)

    with open(manifest_path, "a") as mf:
        for k, (pid, seq) in enumerate(todo):
            # FASTA ids may carry '/', '|', '(', ':' etc. (composite UniProt-style
            # headers); sanitize for the on-disk filename, keep the original id in
            # the manifest so downstream can join back to the FASTA / CSV.
            safe = re.sub(r"[^A-Za-z0-9._+-]", "_", pid)
            spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)])
            t0 = time.time()
            with torch.no_grad():
                result = builder.fold(
                    model,
                    spi,
                    num_loops=args.num_loops,
                    num_sampling_steps=args.num_sampling_steps,
                    num_diffusion_samples=args.num_diffusion_samples,
                    seed=args.seed,
                )
            dt = time.time() - t0
            mean_plddt = float(result.plddt.mean())
            ptm = float(result.ptm)

            cif = result.complex.to_mmcif()
            (out / "mmcif" / f"{safe}.cif").write_text(cif)
            pdb_ok = True
            try:
                (out / "pdb" / f"{safe}.pdb").write_text(cif_to_pdb(cif))
            except Exception as e:
                pdb_ok = False
                if k == 0:
                    print(f"[diag] pdb-convert failed: {e!r}", flush=True)

            if k == 0:
                print(
                    f"[diag] result={type(result).__name__} "
                    f"plddt={type(result.plddt).__name__}"
                    f"{getattr(result.plddt, 'shape', '')} "
                    f"mean_plddt={mean_plddt:.3f} ptm={ptm:.4f} pdb_ok={pdb_ok}",
                    flush=True,
                )

            passed = (mean_plddt >= args.plddt_min) and (ptm >= args.ptm_min)
            row = {
                "id": pid,
                "file": safe,
                "len": len(seq),
                "mean_plddt": round(mean_plddt, 3),
                "ptm": round(ptm, 4),
                "sec": round(dt, 1),
                "pass": bool(passed),
            }
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            print(
                f"[{k + 1}/{len(todo)}] {pid} len={len(seq)} "
                f"pLDDT={mean_plddt:.3f} pTM={ptm:.3f} {dt:.1f}s "
                f"{'PASS' if passed else 'fail'}",
                flush=True,
            )

    print("[done]", flush=True)

    if args.cache_layout:
        n_cache = cache_layout_from_manifest(
            str(manifest_path), str(out / "mmcif"), args.cache_layout
        )
        print(f"[cache-layout] wrote {n_cache} entries -> {args.cache_layout}", flush=True)


if __name__ == "__main__":
    main()
