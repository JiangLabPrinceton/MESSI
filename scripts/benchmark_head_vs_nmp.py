#!/usr/bin/env python
"""Benchmark: Epitope head vs NetMHCIIpan window-level comparison.

For each protein in the input FASTA:
  1. Head: predict_protein() -> per-window logit z, global_risk
  2. NMP: score_batch() -> per-window %Rank_EL (with length-group acceleration)
  3. Align windows by (position, peptide_length)
  4. Label = NMP strong binder (%Rank_EL < threshold), score = head z
  5. Compute per-protein AUC, AP, Recall@K
  6. Macro-average across evaluable proteins

Outputs JSON with config, macro metrics, and per-protein breakdown.

Usage:
    python scripts/benchmark_head_vs_nmp.py \
        --input-fasta path/to/uricases.fasta \
        --epitope-ckpt path/to/best.pt \
        --netmhciipan-bin path/to/netMHCIIpan \
        --allele "HLA-DRB1*07:01" \
        --output-json outputs/benchmark_results.json \
        [--device cuda] \
        [--nmp-batch-size 8] \
        [--nmp-workers 8] \
        [--nmp-max-lengths-per-call 4]
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Optional

import torch

from epitope_head.training.eval_metrics import (
    compute_ap,
    compute_auc,
    compute_recall_at_k,
)

try:
    from scipy.stats import spearmanr as _spearmanr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark epitope head vs NetMHCIIpan on shared proteins.",
    )
    p.add_argument("--input-fasta", required=True,
                   help="Multi-entry FASTA with protein sequences.")
    p.add_argument("--epitope-ckpt", required=True,
                   help="Epitope head checkpoint (.pt).")
    p.add_argument("--netmhciipan-bin", required=True,
                   help="Path to NetMHCIIpan binary.")
    p.add_argument("--allele", default="HLA-DRB1*07:01",
                   help="HLA allele (default: HLA-DRB1*07:01).")
    p.add_argument("--output-json", required=True,
                   help="Output JSON path for benchmark results.")
    p.add_argument("--device", default="cuda",
                   help="Device for head inference (default: cuda).")
    p.add_argument("--variant-id", default="LC1",
                   help="Epitope head CNN variant ID (default: LC1).")
    p.add_argument("--config-dir", default=None,
                   help="Epitope head config directory (auto-detect if omitted).")
    p.add_argument("--nmp-batch-size", type=int, default=8,
                   help="Proteins per NMP batch call (default: 8).")
    p.add_argument("--nmp-timeout", type=int, default=600,
                   help="NMP subprocess timeout in seconds (default: 600).")
    p.add_argument("--nmp-max-lengths-per-call", type=int, default=4,
                   help="Max peptide lengths per NMP call (default: 4).")
    p.add_argument("--nmp-workers", type=int, default=1,
                   help="Parallel NMP subprocess workers (default: 1).")
    p.add_argument("--strong-binder-threshold", type=float, default=2.0,
                   help="NMP %%Rank_EL threshold for strong binder in %% (default: 2.0).")
    p.add_argument("--recall-ks", type=int, nargs="+", default=[50, 100],
                   help="Recall@K values to compute (default: 50 100).")
    return p.parse_args()


# ── I/O helpers ─────────────────────────────────────────────────────────────

def _read_fasta(path: str) -> dict[str, str]:
    seqs: dict[str, str] = {}
    cid: Optional[str] = None
    cseq: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cid and cseq:
                    seqs[cid] = "".join(cseq)
                cid = line[1:].split()[0]
                cseq = []
            else:
                cseq.append(line)
    if cid and cseq:
        seqs[cid] = "".join(cseq)
    return seqs


def _log(msg: str) -> None:
    print(msg, flush=True)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    t0 = time.time()

    # ── Print config ─────────────────────────────────────────────────────
    _log("=" * 60)
    _log("Benchmark: Epitope Head vs NetMHCIIpan")
    _log(f"  Input FASTA        : {args.input_fasta}")
    _log(f"  Epitope ckpt       : {args.epitope_ckpt}")
    _log(f"  NMP binary         : {args.netmhciipan_bin}")
    _log(f"  Allele             : {args.allele}")
    _log(f"  Device             : {args.device}")
    _log(f"  Variant ID         : {args.variant_id}")
    _log(f"  NMP batch size     : {args.nmp_batch_size}")
    _log(f"  NMP timeout        : {args.nmp_timeout}s")
    _log(f"  NMP lengths/call   : {args.nmp_max_lengths_per_call}")
    _log(f"  NMP workers        : {args.nmp_workers}")
    _log(f"  Strong threshold   : {args.strong_binder_threshold}%")
    _log(f"  Recall@K           : {args.recall_ks}")
    _log(f"  Output JSON        : {args.output_json}")
    _log(f"  scipy available    : {_HAS_SCIPY}")
    _log("=" * 60)

    # ── Step 1: Load sequences ──────────────────────────────────────────
    _log("\n[1/4] Loading sequences...")
    sequences = _read_fasta(args.input_fasta)
    _log(f"  Loaded {len(sequences)} proteins.")
    if not sequences:
        _log("ERROR: No sequences found in FASTA.")
        return 1

    # ── Step 2: Head inference ──────────────────────────────────────────
    _log("\n[2/4] Running epitope head inference...")
    from scripts.head_runtime import load_epitope_predictor

    config_dir = args.config_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "epitope_head", "configs",
    )
    predictor = load_epitope_predictor(
        config_dir=config_dir,
        checkpoint_path=args.epitope_ckpt,
        variant_id=args.variant_id,
        device=args.device,
    )

    # {pid: {"global_risk": float, "windows": {(start_0b, k): z}}}
    head_data: dict[str, dict] = {}
    total = len(sequences)
    t_head_start = time.time()
    for idx, (pid, seq) in enumerate(sequences.items(), start=1):
        pred = predictor.predict_protein(seq, allele_idx=0)
        wins: dict[tuple[int, int], float] = {}
        for w in pred["window_logits"]:
            k = w["end_0b"] - w["start_0b"]
            wins[(w["start_0b"], k)] = w["z"]
        head_data[pid] = {
            "global_risk": float(pred["global_risk"]),
            "windows": wins,
        }
        if idx % 20 == 0 or idx == total:
            elapsed = time.time() - t_head_start
            rate = idx / max(elapsed, 1e-6)
            _log(f"  Head: {idx}/{total} ({rate:.1f} proteins/s)")
    head_elapsed = time.time() - t_head_start
    _log(f"  Head inference complete ({head_elapsed:.1f}s)")

    # ── Step 3: NMP scoring ─────────────────────────────────────────────
    _log("\n[3/4] Running NetMHCIIpan scoring...")
    _log(f"  Config: batch_size={args.nmp_batch_size}, timeout={args.nmp_timeout}s, "
         f"max_lengths_per_call={args.nmp_max_lengths_per_call}, "
         f"workers={args.nmp_workers}")
    from epitope_head.data.netmhciipan_runner import build_runner

    nmp_runner = build_runner(
        backend="standalone",
        binary_path=args.netmhciipan_bin,
        batch_size=args.nmp_batch_size,
        subprocess_timeout=args.nmp_timeout,
        max_lengths_per_call=args.nmp_max_lengths_per_call,
        n_workers=args.nmp_workers,
    )

    pep_lengths = list(range(12, 26))
    threshold_frac = args.strong_binder_threshold / 100.0

    # {pid: {"windows": {(pos, k): el_rank_frac}, "n_strong": int, ...}}
    nmp_data: dict[str, dict] = {}
    entries = list(sequences.items())

    t_nmp_start = time.time()
    n_done = 0
    n_nmp_complete = 0
    n_nmp_partial = 0
    for chunk_start in range(0, len(entries), args.nmp_batch_size):
        chunk = entries[chunk_start:chunk_start + args.nmp_batch_size]
        t_chunk = time.time()
        batch_out = nmp_runner.score_batch(chunk, args.allele, pep_lengths)
        chunk_elapsed = time.time() - t_chunk

        for pid, by_len in batch_out.items():
            wins: dict[tuple[int, int], float] = {}
            for pep_len, scores in by_len.items():
                for s in scores:
                    wins[(s.pos, pep_len)] = s.el_rank  # 0-1 fraction

            # Track completeness
            seq_len = len(sequences[pid])
            applicable = sorted(pl for pl in pep_lengths if pl <= seq_len)
            scored = sorted(by_len.keys())
            status = "complete" if scored == applicable else "partial"
            if status == "complete":
                n_nmp_complete += 1
            else:
                n_nmp_partial += 1

            # Per-protein NMP aggregates
            all_ranks = sorted(wins.values())
            n_strong = sum(1 for r in all_ranks if r < threshold_frac)
            n_weak = sum(1 for r in all_ranks if r < 0.10)
            top_k = min(5, len(all_ranks))
            mean_best_rank_pct = (
                sum(all_ranks[:top_k]) / top_k * 100.0
                if top_k > 0 else float("nan")
            )

            nmp_data[pid] = {
                "windows": wins,
                "n_strong": n_strong,
                "n_weak": n_weak,
                "mean_best_rank_pct": mean_best_rank_pct,
                "n_windows_scored": len(wins),
                "status": status,
            }

        n_done += len(chunk)
        nmp_elapsed = time.time() - t_nmp_start
        rate = n_done / max(nmp_elapsed, 1e-6)
        eta = (len(entries) - n_done) / max(rate, 1e-6)
        _log(f"  NMP: {n_done}/{len(entries)} "
             f"({n_nmp_complete}ok {n_nmp_partial}partial) | "
             f"chunk {chunk_elapsed:.1f}s | "
             f"rate {rate:.2f}/s, ETA {eta / 60:.1f}min")

    nmp_elapsed = time.time() - t_nmp_start
    nmp_missing = set(sequences) - set(nmp_data)
    _log(f"  NMP scoring complete ({nmp_elapsed:.1f}s): "
         f"{n_nmp_complete} complete, {n_nmp_partial} partial, "
         f"{len(nmp_missing)} missing")

    # ── Step 4: Align windows and compute metrics ───────────────────────
    _log("\n[4/4] Computing comparison metrics...")
    per_protein: list[dict] = []
    n_skipped = 0

    for pid, seq in sequences.items():
        head = head_data.get(pid)
        nmp = nmp_data.get(pid)
        if head is None or nmp is None:
            n_skipped += 1
            continue

        head_wins = head["windows"]
        nmp_wins = nmp["windows"]
        common = sorted(set(head_wins) & set(nmp_wins))

        if len(common) < 10:
            n_skipped += 1
            continue

        head_z = torch.tensor(
            [head_wins[k] for k in common], dtype=torch.float32,
        )
        nmp_rank = torch.tensor(
            [nmp_wins[k] for k in common], dtype=torch.float32,
        )
        labels = (nmp_rank < threshold_frac).long()

        n_pos = int(labels.sum().item())
        n_neg = int((labels == 0).sum().item())

        entry: dict = {
            "protein_id": pid,
            "sequence_length": len(seq),
            "head_global_risk": head["global_risk"],
            "nmp_n_strong": nmp["n_strong"],
            "nmp_n_weak": nmp["n_weak"],
            "nmp_mean_best_rank_pct": nmp["mean_best_rank_pct"],
            "n_aligned_windows": len(common),
        }

        if n_pos == 0 or n_neg == 0:
            entry.update({"auc": None, "ap": None, "evaluable": False})
            for rk in args.recall_ks:
                entry[f"recall_{rk}"] = None
            entry["spearman_rho"] = None
        else:
            entry["auc"] = compute_auc(head_z, labels)
            entry["ap"] = compute_ap(head_z, labels)
            entry["evaluable"] = True
            for rk in args.recall_ks:
                entry[f"recall_{rk}"] = compute_recall_at_k(head_z, labels, rk)

            if _HAS_SCIPY:
                rho, _ = _spearmanr(head_z.numpy(), -nmp_rank.numpy())
                entry["spearman_rho"] = (
                    float(rho) if not math.isnan(rho) else None
                )
            else:
                entry["spearman_rho"] = None

        per_protein.append(entry)

    # ── Macro averages ──────────────────────────────────────────────────
    evaluable = [r for r in per_protein if r.get("evaluable")]
    auc_vals = [r["auc"] for r in evaluable if r["auc"] is not None]
    ap_vals = [r["ap"] for r in evaluable if r["ap"] is not None]
    rho_vals = [
        r["spearman_rho"] for r in evaluable
        if r.get("spearman_rho") is not None
    ]

    macro: dict = {
        "pp_auc": sum(auc_vals) / len(auc_vals) if auc_vals else None,
        "pp_ap": sum(ap_vals) / len(ap_vals) if ap_vals else None,
        "pp_spearman_rho": (
            sum(rho_vals) / len(rho_vals) if rho_vals else None
        ),
        "n_evaluated": len(evaluable),
        "n_skipped": n_skipped,
        "n_total": len(sequences),
    }
    for rk in args.recall_ks:
        key = f"recall_{rk}"
        vals = [r[key] for r in evaluable if r.get(key) is not None]
        macro[f"pp_{key}"] = sum(vals) / len(vals) if vals else None

    total_elapsed = time.time() - t0

    output = {
        "config": {
            "input_fasta": args.input_fasta,
            "epitope_ckpt": args.epitope_ckpt,
            "allele": args.allele,
            "strong_binder_threshold_pct": args.strong_binder_threshold,
            "variant_id": args.variant_id,
            "nmp_batch_size": args.nmp_batch_size,
            "nmp_workers": args.nmp_workers,
            "nmp_max_lengths_per_call": args.nmp_max_lengths_per_call,
            "recall_ks": args.recall_ks,
        },
        "macro": macro,
        "per_protein": per_protein,
        "timing": {
            "total_seconds": round(total_elapsed, 1),
            "head_seconds": round(head_elapsed, 1),
            "nmp_seconds": round(nmp_elapsed, 1),
        },
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────
    _log(f"\n{'=' * 60}")
    _log("Results:")
    if macro["pp_auc"] is not None:
        _log(f"  pp_auc           = {macro['pp_auc']:.4f}")
    else:
        _log("  pp_auc           = N/A")
    if macro["pp_ap"] is not None:
        _log(f"  pp_ap            = {macro['pp_ap']:.4f}")
    else:
        _log("  pp_ap            = N/A")
    if macro.get("pp_spearman_rho") is not None:
        _log(f"  pp_spearman_rho  = {macro['pp_spearman_rho']:.4f}")
    for rk in args.recall_ks:
        val = macro.get(f"pp_recall_{rk}")
        if val is not None:
            _log(f"  pp_recall_{rk:<4}   = {val:.4f}")
        else:
            _log(f"  pp_recall_{rk:<4}   = N/A")
    _log(f"  n_evaluated      = {macro['n_evaluated']}/{macro['n_total']}")
    _log(f"  wall time        = {total_elapsed:.1f}s "
         f"(head {head_elapsed:.1f}s + NMP {nmp_elapsed:.1f}s)")
    _log(f"\nSaved to {args.output_json}")
    _log("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
