#!/usr/bin/env python
"""Phase C0 driver: DPLM native-sampler baseline on the IF test set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase C0: run the frozen DPLM native sampler on the IF test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Module K checkpoint (.ckpt).")
    parser.add_argument("--test-set-parquet", required=True)
    parser.add_argument("--pdb-root", required=True, help="Directory containing B1 PDB/CIF files.")
    parser.add_argument("--allele", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--n-designs-per-protein", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--sampling-strategy",
        choices=("argmax", "gumbel_argmax"),
        default="argmax",
        help=(
            "Token selection strategy passed to DPLM native generate(). "
            "argmax preserves the historical deterministic C0 baseline; "
            "gumbel_argmax enables stochastic multi-design sampling."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of proteins processed per DPLM generate() call. "
            "Entries are length-bucketed (sort by sequence_length desc, "
            "chunk by batch_size) before batching. >1 trades exact "
            "per-design seed determinism for GPU utilization."
        ),
    )
    parser.add_argument("--run-id", default=None, help="Optional explicit run_id.")
    parser.add_argument("--flag-name", default=None, help="Optional flag name.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Emit a per-protein progress line every N proteins (0 to disable).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty run_dir (default: refuse).",
    )

    from inverse_folding.observability import add_wandb_cli_args

    add_wandb_cli_args(parser, default_project="mhc-if-phase-c-sampling")

    args = parser.parse_args(argv)

    if args.n_designs_per_protein <= 0:
        parser.error("--n-designs-per-protein must be positive")
    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def generate_rows_for_entries(
    entries: pd.DataFrame,
    generator: Callable[[pd.Series, int, int], dict[str, Any]],
    *,
    n_designs_per_protein: int,
    seed: int,
    progress_every: int = 25,
    wandb_run: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate output rows while preserving input protein ordering."""
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    total_entries = int(len(entries))
    total_designs = total_entries * int(n_designs_per_protein)
    run_start = time.time()

    for entry_idx, (_, entry) in enumerate(entries.iterrows(), start=1):
        protein_id = str(entry["protein_id"])
        expected_length = int(entry["sequence_length"])
        for design_idx in range(n_designs_per_protein):
            design_seed = int(seed) + int(design_idx)
            try:
                result = generator(entry, design_idx, design_seed)
                sequence = str(result["sequence"])
                if len(sequence) != expected_length:
                    raise ValueError(
                        f"{protein_id} design_idx={design_idx}: generated length "
                        f"{len(sequence)} != expected {expected_length}"
                    )
                rows.append(
                    {
                        "protein_id": protein_id,
                        "design_idx": int(design_idx),
                        "sequence": sequence,
                        "seed": design_seed,
                        "wall_seconds": float(result["wall_seconds"]),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - persisted for audit.
                failures.append(
                    {
                        "protein_id": protein_id,
                        "design_idx": int(design_idx),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )

        if progress_every > 0 and (
            entry_idx % progress_every == 0 or entry_idx == total_entries
        ):
            _emit_progress(
                entry_idx=entry_idx,
                total_entries=total_entries,
                protein_id=protein_id,
                n_rows=len(rows),
                n_failures=len(failures),
                total_designs=total_designs,
                run_start=run_start,
            )
            if wandb_run is not None:
                from inverse_folding.observability import log_metrics

                elapsed = time.time() - run_start
                done = len(rows) + len(failures)
                avg = elapsed / float(done) if done > 0 else 0.0
                log_metrics(
                    wandb_run,
                    {
                        "progress/proteins_done": entry_idx,
                        "progress/rows_generated": len(rows),
                        "progress/failures": len(failures),
                        "progress/elapsed_seconds": elapsed,
                        "progress/avg_seconds_per_design": avg,
                    },
                    step=entry_idx,
                )
    return rows, failures


def _length_buckets(
    entries: pd.DataFrame, batch_size: int
) -> list[list[int]]:
    """Sort entry indices by sequence_length desc and chunk into groups
    of ``batch_size``. Returns a list of index lists into ``entries``."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    order = (
        entries["sequence_length"]
        .astype(int)
        .sort_values(ascending=False, kind="stable")
        .index.tolist()
    )
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def generate_rows_for_entries_batched(
    entries: pd.DataFrame,
    batched_generator: Callable[
        [list[pd.Series], list[int], int, int], dict[str, Any]
    ],
    *,
    n_designs_per_protein: int,
    seed: int,
    batch_size: int,
    progress_every: int = 25,
    wandb_run: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Length-bucketed batched generation. ``batched_generator`` receives
    ``(entries_list, design_indices, design_idx_global, design_seed)`` and
    must return ``{"sequences": [str, ...], "wall_seconds": float, ...}``.

    The dict may carry an arbitrary ``"meta"`` for the caller's hooks
    (e.g. ablation diagnostics emission).
    """
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    total_entries = int(len(entries))
    total_designs = total_entries * int(n_designs_per_protein)
    run_start = time.time()
    proteins_done = 0
    buckets = _length_buckets(entries, batch_size)

    for design_idx in range(n_designs_per_protein):
        design_seed = int(seed) + int(design_idx)
        for bucket in buckets:
            sub = [entries.loc[i] for i in bucket]
            protein_ids = [str(s["protein_id"]) for s in sub]
            expected_lengths = [int(s["sequence_length"]) for s in sub]
            try:
                result = batched_generator(sub, expected_lengths, design_idx, design_seed)
                # Partial-safe contract:
                #   sequences[i] is the decoded string for sub[i], or None
                #     if that row failed (in prepare or in generate)
                #   failures: list of {"row_idx": int, "reason": str} for
                #     each None entry in sequences. row_idx indexes into
                #     ``sub`` (i.e. the bucket), not the original entries
                #     DataFrame.
                sequences = list(result.get("sequences", []))
                row_failures = list(result.get("failures", []))
                wall_seconds = float(result["wall_seconds"])
                if len(sequences) != len(sub):
                    raise RuntimeError(
                        f"batched_generator returned {len(sequences)} "
                        f"sequences for {len(sub)} rows"
                    )
                n_success = sum(1 for s in sequences if s is not None)
                per_row_wall = (
                    wall_seconds / float(n_success) if n_success > 0 else 0.0
                )
                # Map row_idx -> failure reason for O(1) lookup
                failure_by_idx = {
                    int(f["row_idx"]): str(f.get("reason", "unknown"))
                    for f in row_failures
                    if "row_idx" in f
                }
                for row_i, (entry, sequence, expected_length) in enumerate(
                    zip(sub, sequences, expected_lengths)
                ):
                    if sequence is None:
                        failures.append(
                            {
                                "protein_id": str(entry["protein_id"]),
                                "design_idx": int(design_idx),
                                "reason": failure_by_idx.get(
                                    row_i, "batched_generator returned None"
                                ),
                            }
                        )
                        continue
                    sequence = str(sequence)
                    if len(sequence) != int(expected_length):
                        failures.append(
                            {
                                "protein_id": str(entry["protein_id"]),
                                "design_idx": int(design_idx),
                                "reason": (
                                    f"length mismatch: generated="
                                    f"{len(sequence)} expected={expected_length}"
                                ),
                            }
                        )
                        continue
                    rows.append(
                        {
                            "protein_id": str(entry["protein_id"]),
                            "design_idx": int(design_idx),
                            "sequence": sequence,
                            "seed": design_seed,
                            "wall_seconds": per_row_wall,
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - last-resort: whole-bucket failure
                for entry in sub:
                    failures.append(
                        {
                            "protein_id": str(entry["protein_id"]),
                            "design_idx": int(design_idx),
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )

            proteins_done = proteins_done + len(sub) if design_idx == 0 else proteins_done

            if progress_every > 0 and (
                len(rows) % progress_every == 0
                or (len(rows) + len(failures)) == total_designs
            ):
                _emit_progress(
                    entry_idx=min(proteins_done, total_entries),
                    total_entries=total_entries,
                    protein_id=protein_ids[-1],
                    n_rows=len(rows),
                    n_failures=len(failures),
                    total_designs=total_designs,
                    run_start=run_start,
                )
                if wandb_run is not None:
                    from inverse_folding.observability import log_metrics

                    elapsed = time.time() - run_start
                    done = len(rows) + len(failures)
                    avg = elapsed / float(done) if done > 0 else 0.0
                    log_metrics(
                        wandb_run,
                        {
                            "progress/proteins_done": min(proteins_done, total_entries),
                            "progress/rows_generated": len(rows),
                            "progress/failures": len(failures),
                            "progress/elapsed_seconds": elapsed,
                            "progress/avg_seconds_per_design": avg,
                        },
                        step=min(proteins_done, total_entries),
                    )
    return rows, failures


def _emit_progress(
    *,
    entry_idx: int,
    total_entries: int,
    protein_id: str,
    n_rows: int,
    n_failures: int,
    total_designs: int,
    run_start: float,
) -> None:
    elapsed = time.time() - run_start
    done = n_rows + n_failures
    if done > 0 and total_designs > done:
        eta = elapsed * (total_designs - done) / float(done)
    else:
        eta = 0.0
    avg = elapsed / float(done) if done > 0 else 0.0
    print(
        f"[progress] {entry_idx}/{total_entries} proteins "
        f"protein_id={protein_id} rows={n_rows} failures={n_failures} "
        f"elapsed={_fmt_hms(elapsed)} avg_per_design={avg:.2f}s eta={_fmt_hms(eta)}",
        flush=True,
    )


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def write_phase_c_outputs(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    run_config: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Materialize the Phase C artifact bundle."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_parquet(output_path / "generated.parquet", index=False)
    _write_generated_fasta(output_path / "generated.fasta", rows)
    with open(output_path / "run_config.yaml", "w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)
    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _write_generated_fasta(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            header = f"{row['protein_id']}__design_{int(row['design_idx']):04d}"
            f.write(f">{header}\n{row['sequence']}\n")


def _build_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return str(args.run_id)
    from inverse_folding.reference_flow.runtime import safe_allele_tag

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"c0_{safe_allele_tag(args.allele)}_{args.flag_name}_{stamp}"


def _build_generator(
    *,
    checkpoint: str,
    pdb_root: str,
    device: str,
    max_iter: int,
    temperature: float,
    sampling_strategy: str,
) -> Callable[[pd.Series, int, int], dict[str, Any]]:
    from inverse_folding.reference_flow.runtime import (
        generate_native_sequence,
        load_if_task,
        prepare_backbone,
    )

    task = load_if_task(checkpoint, device=device)

    def _generator(entry: pd.Series, design_idx: int, design_seed: int) -> dict[str, Any]:
        del design_idx
        started = time.time()
        prepared = prepare_backbone(task=task, entry=entry, pdb_root=pdb_root, device=device)
        sequence = generate_native_sequence(
            task=task,
            prepared=prepared,
            max_iter=max_iter,
            temperature=temperature,
            sampling_strategy=sampling_strategy,
            seed=design_seed,
        )
        return {
            "sequence": sequence,
            "wall_seconds": time.time() - started,
        }

    return _generator


def _build_batched_generator(
    *,
    checkpoint: str,
    pdb_root: str,
    device: str,
    max_iter: int,
    temperature: float,
    sampling_strategy: str,
) -> Callable[
    [list[pd.Series], list[int], int, int], dict[str, Any]
]:
    """Batched generator: one DPLM ``generate()`` call per length bucket.

    Partial-safe: per-entry prepare failures are skipped (not raised);
    a batch-wide generate failure triggers per-protein B=1 retry on the
    surviving entries before being marked as a failure. Output contract::

        {
            "sequences": list[str | None] of length len(entries),
            "failures": list[{"row_idx": int, "reason": str}],
            "wall_seconds": float,
        }
    """
    from inverse_folding.reference_flow.runtime import (
        generate_native_sequence,
        generate_native_sequences_batched,
        load_if_task,
        prepare_backbone,
        prepare_backbone_batch,
    )

    task = load_if_task(checkpoint, device=device)

    def _per_protein_retry(
        entries: list[pd.Series],
        sub_indices: list[int],
        sequences: list[str | None],
        failures: list[dict[str, Any]],
        design_seed: int,
    ) -> None:
        """Fall back to B=1 generate for each entry at the given indices.
        Mutates ``sequences`` and ``failures`` in place."""
        for idx in sub_indices:
            entry = entries[idx]
            try:
                prepared = prepare_backbone(
                    task=task, entry=entry, pdb_root=pdb_root, device=device
                )
                seq = generate_native_sequence(
                    task=task,
                    prepared=prepared,
                    max_iter=max_iter,
                    temperature=temperature,
                    sampling_strategy=sampling_strategy,
                    seed=design_seed,
                )
                sequences[idx] = seq
            except Exception as exc:  # noqa: BLE001 - per-row isolation
                failures.append(
                    {
                        "row_idx": int(idx),
                        "reason": f"per_protein_retry: {type(exc).__name__}: {exc}",
                    }
                )

    def _generator(
        entries: list[pd.Series],
        expected_lengths: list[int],
        design_idx: int,
        design_seed: int,
    ) -> dict[str, Any]:
        del design_idx
        started = time.time()
        sequences: list[str | None] = [None] * len(entries)
        failures: list[dict[str, Any]] = []

        batch, seq_lengths, _paths, prep_failures, kept_indices = prepare_backbone_batch(
            task=task,
            entries=entries,
            pdb_root=pdb_root,
            device=device,
            skip_invalid=True,
        )
        # Map prep_failures back to entry positions in the input list.
        kept_set = set(kept_indices)
        prep_failure_iter = iter(prep_failures)
        for idx in range(len(entries)):
            if idx in kept_set:
                continue
            try:
                pf = next(prep_failure_iter)
                reason = pf.get("reason", "prep_failure")
            except StopIteration:
                reason = "prep_failure"
            failures.append({"row_idx": int(idx), "reason": str(reason)})

        if batch is not None and kept_indices:
            try:
                gen_sequences = generate_native_sequences_batched(
                    task=task,
                    batch=batch,
                    sequence_lengths=seq_lengths,
                    max_iter=max_iter,
                    temperature=temperature,
                    sampling_strategy=sampling_strategy,
                    seed=design_seed,
                )
                for row_i, original_idx in enumerate(kept_indices):
                    sequences[original_idx] = gen_sequences[row_i]
            except Exception as exc:  # noqa: BLE001 - bucket-wide retry
                print(
                    f"[batched-generate] failure on bucket of "
                    f"{len(kept_indices)} proteins ({type(exc).__name__}: {exc}); "
                    f"falling back to per-protein B=1 retry.",
                    flush=True,
                )
                _per_protein_retry(
                    entries=entries,
                    sub_indices=kept_indices,
                    sequences=sequences,
                    failures=failures,
                    design_seed=design_seed,
                )

        return {
            "sequences": sequences,
            "failures": failures,
            "wall_seconds": time.time() - started,
        }

    return _generator


def print_resolved_hyperparams(args: argparse.Namespace, *, run_dir: Path, n_entries: int) -> None:
    resolved = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_set_parquet": str(Path(args.test_set_parquet).resolve()),
        "pdb_root": str(Path(args.pdb_root).resolve()),
        "allele": args.allele,
        "output_root": str(Path(args.output_root).resolve()),
        "run_dir": str(run_dir),
        "n_designs_per_protein": args.n_designs_per_protein,
        "seed": args.seed,
        "max_iter": args.max_iter,
        "temperature": args.temperature,
        "sampling_strategy": args.sampling_strategy,
        "device": args.device,
        "batch_size": args.batch_size,
        "progress_every": args.progress_every,
        "overwrite": args.overwrite,
        "n_input_proteins": n_entries,
    }
    print("============================================================")
    print("Phase C0 resolved parameters")
    for key, value in resolved.items():
        print(f"  {key}: {value}")
    print("============================================================")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from inverse_folding.reference_flow.runtime import (
        checkpoint_digest,
        git_sha,
        load_test_entries,
        safe_allele_tag,
        utc_timestamp,
        write_json,
    )

    run_id = _build_run_id(args)
    run_dir = Path(args.output_root) / safe_allele_tag(args.allele) / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        print(
            f"ERROR: run_dir {run_dir} already exists and is non-empty; "
            "pass --overwrite to allow clobbering or choose a distinct --run-id.",
            file=sys.stderr,
        )
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    entries = load_test_entries(args.test_set_parquet)
    print_resolved_hyperparams(args, run_dir=run_dir, n_entries=len(entries))

    from inverse_folding.observability import (
        finish_wandb,
        init_wandb_from_args,
        set_summary,
    )

    wandb_run = init_wandb_from_args(
        args,
        run_name=run_id,
        config={
            "stage": "phase_c_sampling",
            "arm": "c0_native_sampler",
            "allele": args.allele,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "n_input_proteins": int(len(entries)),
            "n_designs_per_protein": args.n_designs_per_protein,
            "seed": args.seed,
            "max_iter": args.max_iter,
            "temperature": args.temperature,
            "sampling_strategy": args.sampling_strategy,
            "device": args.device,
            "run_dir": str(run_dir),
        },
        extra_tags=["c0", "native_sampler", args.allele],
    )

    run_start = time.time()
    if int(args.batch_size) > 1:
        batched_generator = _build_batched_generator(
            checkpoint=args.checkpoint,
            pdb_root=args.pdb_root,
            device=args.device,
            max_iter=args.max_iter,
            temperature=args.temperature,
            sampling_strategy=args.sampling_strategy,
        )
        rows, failures = generate_rows_for_entries_batched(
            entries,
            batched_generator,
            n_designs_per_protein=args.n_designs_per_protein,
            seed=args.seed,
            batch_size=int(args.batch_size),
            progress_every=args.progress_every,
            wandb_run=wandb_run,
        )
    else:
        generator = _build_generator(
            checkpoint=args.checkpoint,
            pdb_root=args.pdb_root,
            device=args.device,
            max_iter=args.max_iter,
            temperature=args.temperature,
            sampling_strategy=args.sampling_strategy,
        )
        rows, failures = generate_rows_for_entries(
            entries,
            generator,
            n_designs_per_protein=args.n_designs_per_protein,
            seed=args.seed,
            progress_every=args.progress_every,
            wandb_run=wandb_run,
        )
    total_wall_seconds = time.time() - run_start

    run_config = {
        "mode": "c0_native_sampler",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_set_parquet": str(Path(args.test_set_parquet).resolve()),
        "pdb_root": str(Path(args.pdb_root).resolve()),
        "allele": args.allele,
        "device": args.device,
        "sampler": {
            "max_iter": args.max_iter,
            "temperature": args.temperature,
            "sampling_strategy": args.sampling_strategy,
            "n_designs_per_protein": args.n_designs_per_protein,
            "seed": args.seed,
            "batch_size": int(args.batch_size),
        },
    }
    manifest = {
        "run_id": run_id,
        "mode": "c0_native_sampler",
        "git_sha": git_sha(PROJECT_ROOT),
        "checkpoint_digest": checkpoint_digest(args.checkpoint),
        "timestamp": utc_timestamp(),
        "allele": args.allele,
        "n_input_proteins": int(len(entries)),
        "n_rows_generated": int(len(rows)),
        "n_failures": int(len(failures)),
        "failures_path": "failures.json" if failures else None,
    }

    manifest["wall_clock_seconds"] = float(total_wall_seconds)
    n_rows = int(len(rows))
    n_failures = int(len(failures))
    n_total_designs = int(len(entries) * args.n_designs_per_protein)
    avg_per_design = (
        total_wall_seconds / float(n_rows + n_failures) if (n_rows + n_failures) > 0 else 0.0
    )
    failure_rate = (
        (n_failures / float(n_total_designs)) if n_total_designs > 0 else 0.0
    )

    write_phase_c_outputs(run_dir, rows, run_config, manifest)
    if failures:
        write_json(run_dir / "failures.json", {"failures": failures})

    print("============================================================")
    print(
        f"[done] run_id={run_id} "
        f"generated_rows={n_rows}/{n_total_designs} "
        f"failures={n_failures} failure_rate={failure_rate:.2%} "
        f"wall={_fmt_hms(total_wall_seconds)} avg_per_design={avg_per_design:.2f}s "
        f"output_dir={run_dir}"
    )
    print("============================================================")

    set_summary(
        wandb_run,
        {
            "summary/n_input_proteins": int(len(entries)),
            "summary/n_total_designs": n_total_designs,
            "summary/n_rows_generated": n_rows,
            "summary/n_failures": n_failures,
            "summary/failure_rate": failure_rate,
            "summary/wall_seconds": float(total_wall_seconds),
            "summary/avg_seconds_per_design": float(avg_per_design),
        },
    )
    finish_wandb(wandb_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
