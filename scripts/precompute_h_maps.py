#!/usr/bin/env python
"""Pre-compute Phase B per-residue epitope-head h_i maps."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.evaluation.h_maps import (
    default_meta_path,
    load_h_maps,
    sequence_md5,
)


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class InputRecord:
    protein_id: str
    sequence: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute wide per-protein h_i map parquet artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--head-checkpoint", required=True, help="Epitope-head checkpoint path.")
    parser.add_argument(
        "--inference-config",
        required=True,
        help="Inference YAML path. model.yaml and model_ablation.yaml are read from --config-dir.",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Directory containing model.yaml and model_ablation.yaml. Defaults to inference config parent.",
    )
    parser.add_argument("--variant-id", default="LC1", help="Epitope-head ablation/profile id.")
    parser.add_argument("--source", choices=("parquet", "jsonl"), required=True)
    parser.add_argument("--input", required=True, help="Source parquet or JSONL path.")
    parser.add_argument("--id-column", "--id-field", dest="id_column", default=None)
    parser.add_argument("--sequence-column", "--sequence-field", dest="sequence_column", default=None)
    parser.add_argument("--splits-json", default=None, help="Optional CATH split JSON for JSONL input.")
    parser.add_argument("--split", default=None, help="Split key to select from --splits-json.")
    parser.add_argument("--allele", required=True, help='Canonical allele, e.g. "HLA-DRB1*07:01".')
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--output-meta", required=True)
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--window-batch-size", type=int, default=4096)
    parser.add_argument("--fail-pct-threshold", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume-from", default=None, help="Prior partial parquet to resume from.")

    args = parser.parse_args(argv)
    if args.id_column is None:
        parser.error("--id-column/--id-field is required")
    if args.sequence_column is None:
        parser.error("--sequence-column/--sequence-field is required")
    if args.fail_pct_threshold < 0.0 or args.fail_pct_threshold > 1.0:
        parser.error("--fail-pct-threshold must be in [0, 1]")
    if args.window_batch_size <= 0:
        parser.error("--window-batch-size must be positive")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be non-negative")
    if args.source == "parquet" and (args.splits_json or args.split):
        parser.error("--splits-json/--split only apply to --source jsonl")
    if bool(args.splits_json) ^ bool(args.split):
        parser.error("--splits-json and --split must be supplied together")
    return args


def build_predictor(args: argparse.Namespace):
    """Build the real checkpoint-backed epitope-head predictor."""
    from epitope_head.configs import (
        load_ablation_config,
        load_inference_config,
        load_model_config,
    )
    from scripts.score_head import build_predictor as build_inference_predictor

    inference_config = Path(args.inference_config)
    config_dir = Path(args.config_dir) if args.config_dir else inference_config.parent

    model_cfg = load_model_config(config_dir / "model.yaml")
    ablation_cfg = load_ablation_config(config_dir / "model_ablation.yaml")
    inference_cfg = load_inference_config(inference_config)

    return build_inference_predictor(
        model_cfg=model_cfg,
        ablation_cfg=ablation_cfg,
        inference_cfg=inference_cfg,
        variant_id=args.variant_id,
        checkpoint_path=Path(args.head_checkpoint),
        device=args.device,
    )


def load_input_records(args: argparse.Namespace) -> list[InputRecord]:
    input_path = Path(args.input)
    if args.source == "parquet":
        df = pd.read_parquet(input_path)
        missing = [c for c in (args.id_column, args.sequence_column) if c not in df.columns]
        if missing:
            raise ValueError(f"input parquet missing required columns: {missing}")
        return [
            InputRecord(str(row[args.id_column]), str(row[args.sequence_column]).upper())
            for _, row in df.iterrows()
        ]

    split_ids: set[str] | None = None
    if args.splits_json and args.split:
        with open(args.splits_json) as f:
            splits = json.load(f)
        if args.split not in splits:
            raise ValueError(f"split '{args.split}' not found in {args.splits_json}")
        split_ids = {str(x) for x in splits[args.split]}

    records: list[InputRecord] = []
    with open(input_path) as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if args.id_column not in obj or args.sequence_column not in obj:
                raise ValueError(
                    f"JSONL line {line_no} missing {args.id_column}/{args.sequence_column}",
                )
            protein_id = str(obj[args.id_column])
            if split_ids is not None and protein_id not in split_ids:
                continue
            records.append(InputRecord(protein_id, str(obj[args.sequence_column]).upper()))
    return records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    entries = load_input_records(args)
    predictor = build_predictor(args)

    print_resolved_hyperparams(args, entries)

    output_parquet = Path(args.output_parquet)
    output_meta = Path(args.output_meta)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)

    rows_by_id = (
        load_resume_rows(
            resume_from=args.resume_from,
            args=args,
            predictor=predictor,
            entries=entries,
        )
        if args.resume_from
        else {}
    )
    failures: list[dict[str, str]] = []
    partial_path = make_partial_path(output_parquet)

    for attempted, entry in enumerate(entries, start=1):
        if entry.protein_id in rows_by_id:
            continue

        reason = validate_sequence(entry.sequence, source=args.source)
        if reason is not None:
            failures.append({"protein_id": entry.protein_id, "reason": reason})
            if should_abort(failures, entries, args.fail_pct_threshold):
                write_partial(partial_path, rows_by_id, entries, args, predictor, failures, started)
                return 2
            continue

        try:
            rows_by_id[entry.protein_id] = score_entry(
                predictor=predictor,
                entry=entry,
                allele=args.allele,
                window_batch_size=args.window_batch_size,
            )
        except RuntimeError as exc:
            if is_oom(exc) and str(args.device).startswith("cuda"):
                try:
                    rows_by_id[entry.protein_id] = retry_on_cpu(
                        predictor=predictor,
                        entry=entry,
                        allele=args.allele,
                        window_batch_size=args.window_batch_size,
                        original_device=args.device,
                    )
                except Exception as retry_exc:  # noqa: BLE001 - logged into artifact.
                    failures.append({
                        "protein_id": entry.protein_id,
                        "reason": f"oom_cpu_retry_failed:{type(retry_exc).__name__}",
                    })
            else:
                failures.append({
                    "protein_id": entry.protein_id,
                    "reason": f"score_failed:{type(exc).__name__}",
                })
        except Exception as exc:  # noqa: BLE001 - logged into artifact.
            failures.append({
                "protein_id": entry.protein_id,
                "reason": f"score_failed:{type(exc).__name__}",
            })

        if should_abort(failures, entries, args.fail_pct_threshold):
            write_partial(partial_path, rows_by_id, entries, args, predictor, failures, started)
            return 2

        if args.checkpoint_every and attempted % args.checkpoint_every == 0:
            write_partial(partial_path, rows_by_id, entries, args, predictor, failures, started)
            print(f"[checkpoint] wrote {partial_path} at input row {attempted}/{len(entries)}")

    final_rows = ordered_rows(rows_by_id, entries)
    write_dataframe_atomic(pd.DataFrame(final_rows), output_parquet)

    meta = build_metadata(
        args=args,
        predictor=predictor,
        entries=entries,
        rows=final_rows,
        failures=failures,
        wall_clock_seconds=time.time() - started,
    )
    write_json_atomic(meta, output_meta)

    print(
        f"[done] completed={len(final_rows)} failed={len(failures)} "
        f"output={output_parquet} meta={output_meta}",
    )
    return 0


def print_resolved_hyperparams(args: argparse.Namespace, entries: list[InputRecord]) -> None:
    printable = {
        "head_checkpoint": str(Path(args.head_checkpoint).resolve()),
        "inference_config": str(Path(args.inference_config).resolve()),
        "config_dir": str(Path(args.config_dir).resolve()) if args.config_dir else None,
        "variant_id": args.variant_id,
        "source": args.source,
        "input": str(Path(args.input).resolve()),
        "id_column": args.id_column,
        "sequence_column": args.sequence_column,
        "splits_json": str(Path(args.splits_json).resolve()) if args.splits_json else None,
        "split": args.split,
        "allele": args.allele,
        "output_parquet": str(Path(args.output_parquet).resolve()),
        "output_meta": str(Path(args.output_meta).resolve()),
        "device": args.device,
        "window_batch_size": args.window_batch_size,
        "fail_pct_threshold": args.fail_pct_threshold,
        "checkpoint_every": args.checkpoint_every,
        "resume_from": args.resume_from,
        "n_input_records": len(entries),
    }
    print("============================================================")
    print("Phase B h-map precompute resolved parameters")
    for key, value in printable.items():
        print(f"  {key}: {value}")
    print("============================================================")


def validate_sequence(sequence: str, source: str) -> str | None:
    if not sequence:
        return "empty_sequence"
    chars = set(sequence)
    if source == "jsonl":
        if "-" in chars:
            return "gap_character"
        non_allowed = chars - CANONICAL_AA - {"X"}
        if non_allowed:
            return "non_canonical_aa"
        x_count = sequence.count("X")
        if x_count > 0 and (x_count / len(sequence)) > 0.05:
            return "too_many_x"
        return None

    if chars - CANONICAL_AA:
        return "non_canonical_aa"
    return None


def score_entry(
    predictor: Any,
    entry: InputRecord,
    allele: str,
    window_batch_size: int,
) -> dict[str, Any]:
    prediction = predict_with_batch_size(
        predictor=predictor,
        sequence=entry.sequence,
        window_batch_size=window_batch_size,
    )
    h_raw = tensor_like_to_float_list(prediction["debug"]["h_raw"])
    h_processed = tensor_like_to_float_list(prediction["residue_hotspot"])
    return {
        "protein_id": entry.protein_id,
        "allele": allele,
        "sequence_length": len(entry.sequence),
        "sequence_md5": sequence_md5(entry.sequence),
        "h_raw": h_raw,
        "h_processed": h_processed,
        "global_risk": float(prediction["global_risk"]),
        "n_windows": int(prediction.get("meta", {}).get("n_windows", len(prediction["window_logits"]))),
    }


def predict_with_batch_size(predictor: Any, sequence: str, window_batch_size: int) -> dict[str, Any]:
    """Run predictor while honoring ``--window-batch-size`` when possible."""
    if all(hasattr(predictor, attr) for attr in ("encode_sequence", "enumerate_and_score", "aggregate_hotspot_and_risk")):
        min_k = int(getattr(predictor, "min_k"))
        max_k = int(getattr(predictor, "max_k"))
        center_method = predictor.inference_cfg.get("hotspot_center_method", "median")
        clamp_method = predictor.inference_cfg.get("hotspot_clamp", "none")
        G, encode_debug = predictor.encode_sequence(sequence)
        window_entries, z_tensor = predictor.enumerate_and_score(
            G,
            len(sequence),
            min_k,
            max_k,
            0,
            window_batch_size,
        )
        if window_entries:
            h_raw, h_processed, global_risk = predictor.aggregate_hotspot_and_risk(
                window_entries,
                z_tensor,
                len(sequence),
                center_method,
                clamp_method,
            )
        else:
            import torch

            h_raw = torch.zeros(len(sequence), dtype=torch.float32)
            h_processed = torch.zeros(len(sequence), dtype=torch.float32)
            global_risk = float("-inf")
        return {
            "window_logits": window_entries,
            "residue_hotspot": h_processed,
            "global_risk": global_risk,
            "meta": {
                "protein_len": len(sequence),
                "n_windows": len(window_entries),
                "min_k": min_k,
                "max_k": max_k,
                "center_method": center_method,
                "clamp_method": clamp_method,
            },
            "debug": {
                "encode": encode_debug,
                "h_raw": h_raw,
                "z_tensor": z_tensor,
            },
        }
    return predictor.predict_protein(sequence, allele_idx=0)


def retry_on_cpu(
    predictor: Any,
    entry: InputRecord,
    allele: str,
    window_batch_size: int,
    original_device: str,
) -> dict[str, Any]:
    move_predictor(predictor, "cpu")
    try:
        return score_entry(predictor, entry, allele, window_batch_size)
    finally:
        move_predictor(predictor, original_device)


def move_predictor(predictor: Any, device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(predictor, "model"):
        predictor.model.to(device)
    predictor.device = torch.device(device)
    if hasattr(predictor, "inference_cfg"):
        predictor.inference_cfg["device"] = device


def tensor_like_to_float_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("h-map arrays must be one-dimensional")
    return [float(x) for x in arr.tolist()]


def is_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


def should_abort(
    failures: list[dict[str, str]],
    entries: list[InputRecord],
    threshold: float,
) -> bool:
    return bool(entries) and (len(failures) / len(entries)) > threshold


def load_resume_rows(
    resume_from: str,
    args: argparse.Namespace,
    predictor: Any,
    entries: list[InputRecord],
) -> dict[str, dict[str, Any]]:
    path = Path(resume_from)
    meta_path = default_meta_path(path)
    if not meta_path.exists():
        raise ValueError(
            f"resume metadata sidecar not found for {path}: expected {meta_path}",
        )

    df, meta = load_h_maps(
        path,
        meta_path,
        require_corpus_stats=args.source == "jsonl",
        allow_partial=True,
    )
    validate_resume_metadata(meta, args, predictor, entries)
    validate_resume_rows(df, args, entries)

    rows: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        item = row.to_dict()
        rows[str(item["protein_id"])] = normalize_row(item)
    return rows


def validate_resume_metadata(
    meta: dict[str, Any],
    args: argparse.Namespace,
    predictor: Any,
    entries: list[InputRecord],
) -> None:
    expected_allele = args.allele
    if str(meta["allele"]) != expected_allele:
        raise ValueError(
            f"resume allele mismatch: {meta['allele']} != {expected_allele}",
        )

    expected_checkpoint = str(Path(args.head_checkpoint).resolve())
    if str(meta["head_checkpoint_path"]) != expected_checkpoint:
        raise ValueError(
            "resume head_checkpoint_path mismatch: "
            f"{meta['head_checkpoint_path']} != {expected_checkpoint}",
        )

    expected_source = str(Path(args.input).resolve())
    if str(meta["source_dataset"]) != expected_source:
        raise ValueError(
            f"resume source_dataset mismatch: {meta['source_dataset']} != {expected_source}",
        )

    if int(meta["source_dataset_rowcount"]) != len(entries):
        raise ValueError(
            "resume source_dataset_rowcount mismatch: "
            f"{meta['source_dataset_rowcount']} != {len(entries)}",
        )

    expected_checkpoint_meta = dict(getattr(predictor, "checkpoint_metadata", {}) or {})
    if canonical_json(meta["head_checkpoint_metadata"]) != canonical_json(expected_checkpoint_meta):
        raise ValueError("resume head_checkpoint_metadata mismatch")

    expected_inference_cfg = dict(getattr(predictor, "inference_cfg", {}) or {})
    if canonical_json(meta["inference_cfg"]) != canonical_json(expected_inference_cfg):
        raise ValueError("resume inference_cfg mismatch")


def validate_resume_rows(
    df: pd.DataFrame,
    args: argparse.Namespace,
    entries: list[InputRecord],
) -> None:
    current_by_id = {entry.protein_id: entry for entry in entries}
    if len(current_by_id) != len(entries):
        raise ValueError("current input contains duplicate protein_id values")

    for _, row in df.iterrows():
        protein_id = str(row["protein_id"])
        if protein_id not in current_by_id:
            raise ValueError(f"resume row protein_id not present in current input: {protein_id}")
        if str(row["allele"]) != args.allele:
            raise ValueError(
                f"resume row allele mismatch for {protein_id}: {row['allele']} != {args.allele}",
            )
        current_seq = current_by_id[protein_id].sequence
        current_len = len(current_seq)
        row_len = int(row["sequence_length"])
        if row_len != current_len:
            raise ValueError(
                f"resume sequence_length mismatch for {protein_id}: {row_len} != {current_len}",
            )
        expected_md5 = sequence_md5(current_seq)
        if str(row["sequence_md5"]) != expected_md5:
            raise ValueError(
                f"resume sequence_md5 mismatch for {protein_id}: "
                f"{row['sequence_md5']} != {expected_md5}",
            )


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["protein_id"] = str(out["protein_id"])
    out["allele"] = str(out["allele"])
    out["sequence_length"] = int(out["sequence_length"])
    out["sequence_md5"] = str(out["sequence_md5"])
    out["h_raw"] = tensor_like_to_float_list(out["h_raw"])
    out["h_processed"] = tensor_like_to_float_list(out["h_processed"])
    out["global_risk"] = float(out["global_risk"])
    out["n_windows"] = int(out["n_windows"])
    return out


def ordered_rows(
    rows_by_id: dict[str, dict[str, Any]],
    entries: Iterable[InputRecord],
) -> list[dict[str, Any]]:
    return [rows_by_id[e.protein_id] for e in entries if e.protein_id in rows_by_id]


def make_partial_path(output_parquet: Path) -> Path:
    return output_parquet.parent / ".partial" / f"{output_parquet.stem}.partial.parquet"


def write_partial(
    partial_path: Path,
    rows_by_id: dict[str, dict[str, Any]],
    entries: list[InputRecord],
    args: argparse.Namespace,
    predictor: Any,
    failures: list[dict[str, str]],
    started: float,
) -> None:
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    rows = ordered_rows(rows_by_id, entries)
    write_dataframe_atomic(pd.DataFrame(rows), partial_path)
    meta = build_metadata(
        args=args,
        predictor=predictor,
        entries=entries,
        rows=rows,
        failures=failures,
        wall_clock_seconds=time.time() - started,
        is_partial=True,
    )
    write_json_atomic(meta, default_meta_path(partial_path))
    print(f"[partial] wrote {partial_path}")


def write_dataframe_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def build_metadata(
    args: argparse.Namespace,
    predictor: Any,
    entries: list[InputRecord],
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    wall_clock_seconds: float,
    is_partial: bool = False,
) -> dict[str, Any]:
    timestamp = datetime.now().astimezone()
    meta: dict[str, Any] = {
        "run_id": f"h_maps_{allele_tag(args.allele)}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
        "allele": args.allele,
        "head_checkpoint_path": str(Path(args.head_checkpoint).resolve()),
        "head_checkpoint_metadata": dict(getattr(predictor, "checkpoint_metadata", {}) or {}),
        "inference_cfg": dict(getattr(predictor, "inference_cfg", {}) or {}),
        "source_dataset": str(Path(args.input).resolve()),
        "source_dataset_rowcount": len(entries),
        "git_commit": git_commit(),
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "n_proteins_total": len(entries),
        "n_proteins_completed": len(rows),
        "n_proteins_failed": len(failures),
        "failures": failures,
        "device": str(getattr(predictor, "device", args.device)),
        "wall_clock_seconds": float(wall_clock_seconds),
        "is_partial": bool(is_partial),
    }
    if args.source == "jsonl":
        stats = corpus_h_raw_stats(rows)
        meta.update(stats)
    return meta


def corpus_h_raw_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "corpus_h_raw_mean": math.nan,
            "corpus_h_raw_std": math.nan,
            "corpus_h_raw_n_residues": 0,
        }
    arrays = [np.asarray(row["h_raw"], dtype=np.float32) for row in rows]
    pooled = np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float32)
    return {
        "corpus_h_raw_mean": float(pooled.mean()) if pooled.size else math.nan,
        "corpus_h_raw_std": float(pooled.std()) if pooled.size else math.nan,
        "corpus_h_raw_n_residues": int(pooled.size),
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - metadata should still be emitted.
        return "unknown"


def allele_tag(allele: str) -> str:
    """File-safe allele tag, unified to the ``HLA-DRB1_07_01`` form.

    Delegates to reference_flow.runtime.safe_allele_tag (preserves ``-``/``.``),
    prepending a missing ``HLA-`` prefix so both ``HLA-DRB1*07:01`` and
    ``DRB1*07:01`` normalise to ``HLA-DRB1_07_01``.
    """
    from inverse_folding.reference_flow.runtime import safe_allele_tag

    a = allele.strip()
    if not a.upper().startswith("HLA-"):
        a = "HLA-" + a
    return safe_allele_tag(a)


if __name__ == "__main__":
    raise SystemExit(main())
