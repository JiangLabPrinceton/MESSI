#!/usr/bin/env python
"""Evaluate Phase C generated.parquet artifacts without WT-relative mixing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.evaluation.immunogenicity import NMP_PEP_LENGTHS
from inverse_folding.evaluation.schema import (
    IMMUNOGENICITY_HEAD_COLUMNS,
    IMMUNOGENICITY_NMP_COLUMNS,
    LEGACY_STRUCTURAL_RESIDUE_COLUMNS,
    STRUCTURAL_COLUMNS,
    STRUCTURAL_RESIDUE_COLUMNS,
    STRUCTURAL_V2_COLUMNS,
    STRUCTURAL_V2_RESIDUE_COLUMNS,
    validate_dataframe,
)
from inverse_folding.evaluation.refold import CACHE_READ_BACKENDS, load_refold_model, refold
from scripts.run_if_phase_c0 import _fmt_hms


CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
_NMP_WEAK_BINDER_THRESHOLD = 10.0
_NMP_TOP_K_FOR_MEAN_BEST = 5
_SELECTION_AUTHORITY_COLUMNS = (
    "selection_status",
    "terminal_validated",
    "structure_feasible",
    "selection_provenance_digest",
)
_SELECTION_STATUSES = frozenset({
    "feasible_immune_pareto",
    "official_feasible_immune_pareto",
    "official_structure_rejected_fallback",
    "structure_rejected_fallback",
})


class DataConsistencyError(RuntimeError):
    """Raised when generated parquet rows are inconsistent with the test set contract."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase C generated.parquet outputs into schema-aligned per-design tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--generated-parquet", required=True)
    parser.add_argument("--test-set-parquet", required=True)
    parser.add_argument("--allele", required=True)
    parser.add_argument("--mode", choices=("imm", "struct", "all"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--fail-pct-threshold", type=float, default=0.05)

    parser.add_argument("--epitope-ckpt", default=None)
    parser.add_argument("--epitope-config-dir", default=None)
    parser.add_argument("--epitope-variant-id", default="LC1")
    parser.add_argument("--netmhciipan-bin", default=None)
    parser.add_argument(
        "--nmp-mode",
        choices=["original", "accelerated"],
        default="accelerated",
        help=(
            "NetMHCIIpan invocation mode. 'accelerated' uses the explicit "
            "--nmp-batch-size / --nmp-max-lengths-per-call / --nmp-workers "
            "values (default for Phase C). 'original' overrides them to the "
            "naive baseline (batch_size=1, max_lengths_per_call=14, "
            "workers=1) to match the IEDB benchmark contract."
        ),
    )
    parser.add_argument("--nmp-batch-size", type=int, default=8)
    parser.add_argument("--nmp-workers", type=int, default=1)
    parser.add_argument("--nmp-timeout", type=int, default=600)
    parser.add_argument("--nmp-max-lengths-per-call", type=int, default=4)
    parser.add_argument("--strong-binder-threshold", type=float, default=2.0)
    parser.add_argument(
        "--hotspot-threshold",
        type=float,
        default=0.5,
        help="Per-residue hotspot threshold for n_hotspot_positions count.",
    )

    parser.add_argument(
        "--imm-full",
        action="store_true",
        help=(
            "Additionally write per-residue hotspot and per-peptide NMP "
            "long-tables (imm_head_residues.parquet, imm_nmp_peptides.parquet) "
            "alongside the existing aggregated imm_head/imm_nmp parquets. "
            "Default off preserves prior Phase C behavior."
        ),
    )
    parser.add_argument(
        "--no-nmp",
        action="store_true",
        help=(
            "imm mode: score immunogenicity with the epitope head only and "
            "skip NetMHCIIpan entirely. No NMP binary is launched and "
            "imm_nmp.parquet is not produced; only imm_head.parquet is "
            "required/written. Useful for fast head-only iteration."
        ),
    )
    parser.add_argument("--pdb-root", default=None)
    parser.add_argument(
        "--structural-metrics-v2",
        action="store_true",
        default=True,
        help=(
            "Deprecated compatibility flag: structural v2 is now always the canonical "
            "struct/all output."
        ),
    )
    parser.add_argument(
        "--constraint-manifest",
        default=None,
        help=(
            "Optional hard-anchor manifest used by structural v2 active-site metrics. "
            "Without it, v2 still emits complete global and per-residue metrics with "
            "active-site aggregate fields marked not applicable."
        ),
    )
    parser.add_argument(
        "--refold-model",
        choices=("esmfold", "esmfold2", "protenix", "af3"),
        default="esmfold2",
        help="structure-prediction backend for canonical v2 metrics. 'esmfold' folds in-process; "
        "'esmfold2'/'protenix' are cache-read (a separate SLURM precompute populates "
        "--refold-cache-dir); 'af3' is also cache-read.",
    )
    parser.add_argument("--tmalign-bin", default="TMalign")
    # Generic refold cache dir; --esmfold-cache-dir kept as a deprecated alias
    # (both write the same destination so all backends share one cache namespace).
    parser.add_argument("--refold-cache-dir", dest="esmfold_cache_dir", default=None)
    parser.add_argument("--esmfold-cache-dir", dest="esmfold_cache_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default=None)

    from inverse_folding.observability import add_wandb_cli_args

    add_wandb_cli_args(parser, default_project="mhc-if-phase-c-eval")

    args = parser.parse_args(argv)
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if not (0.0 <= args.fail_pct_threshold <= 1.0):
        parser.error("--fail-pct-threshold must be in [0, 1]")
    if args.nmp_batch_size <= 0:
        parser.error("--nmp-batch-size must be positive")
    if args.nmp_workers <= 0:
        parser.error("--nmp-workers must be positive")
    args.nmp_batch_size, args.nmp_max_lengths_per_call, args.nmp_workers = (
        resolve_nmp_runtime_params(args)
    )
    args.device = _resolve_device(args.device)
    return args


def _resolve_device(requested: str) -> str:
    """Fall back to CPU when a CUDA device is requested but unavailable.

    Lets the same command line work on both GPU and CPU partitions without
    forcing callers to override --device manually on CPU-only nodes.
    """
    req = str(requested).strip()
    if not req.startswith("cuda"):
        return req
    try:
        import torch  # local import: keep argparse cold path light
    except ImportError:
        print(
            f"[device] torch not importable; falling back from {req!r} to 'cpu'",
            file=sys.stderr,
            flush=True,
        )
        return "cpu"
    if torch.cuda.is_available():
        return req
    print(
        f"[device] CUDA unavailable on this node; falling back from {req!r} to 'cpu'",
        file=sys.stderr,
        flush=True,
    )
    return "cpu"


def resolve_nmp_runtime_params(args: argparse.Namespace) -> tuple[int, int, int]:
    """Return effective NetMHCIIpan batch/length/worker parameters."""

    if args.nmp_mode == "original":
        # Naive baseline: one protein x all lengths per subprocess, no parallelism.
        # Matches benchmark_iedb_test.py's original semantics.
        return 1, 14, 1
    return (
        int(args.nmp_batch_size),
        int(args.nmp_max_lengths_per_call),
        int(args.nmp_workers),
    )


def build_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return str(args.run_id)
    from inverse_folding.reference_flow.runtime import safe_allele_tag

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    refold_tag = args.refold_model if args.mode in {"struct", "all"} else "na"
    return f"eval_{args.mode}_{refold_tag}_{safe_allele_tag(args.allele)}_{args.tag}_{stamp}"


def load_generated_designs(generated_parquet: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(generated_parquet).copy()
    # Only the three columns below are actually consumed downstream
    # (protein_id / design_idx for identity, sequence for scoring).
    # seed / wall_seconds were vestigial Phase C contract fields tracked
    # by in-house DPLM-based generators; external baselines like
    # ProteinMPNN don't carry them, so we no longer require them.
    required = {"protein_id", "design_idx", "sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"generated parquet missing required columns: {sorted(missing)}")
    df["protein_id"] = df["protein_id"].astype(str)
    df["design_idx"] = df["design_idx"].astype(int)
    df["sequence"] = df["sequence"].astype(str).str.upper()
    if df[["protein_id", "design_idx"]].duplicated().any():
        raise ValueError("generated parquet contains duplicate (protein_id, design_idx) rows")
    _validate_selection_authority(df)
    df["design_id"] = df["design_idx"].map(lambda idx: f"design_{idx:04d}")
    return df


def _validate_selection_authority(df: pd.DataFrame) -> None:
    present = set(_SELECTION_AUTHORITY_COLUMNS) & set(df.columns)
    if not present:
        return
    missing = set(_SELECTION_AUTHORITY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            "generated parquet selection authority columns are partial; missing "
            f"{sorted(missing)}"
        )
    for row in df.itertuples(index=False):
        status = getattr(row, "selection_status")
        if not isinstance(status, str) or status not in _SELECTION_STATUSES:
            raise ValueError(f"invalid selection_status {status!r}")
        terminal = getattr(row, "terminal_validated")
        feasible = getattr(row, "structure_feasible")
        if not isinstance(terminal, (bool, np.bool_)):
            raise ValueError("terminal_validated must be boolean selection authority")
        if not isinstance(feasible, (bool, np.bool_)):
            raise ValueError("structure_feasible must be boolean selection authority")
        if status.endswith("feasible_immune_pareto") and not (terminal and feasible):
            raise ValueError(
                "feasible_immune_pareto requires terminal_validated=true and "
                "structure_feasible=true"
            )
        if status.endswith("structure_rejected_fallback") and (terminal or feasible):
            raise ValueError(
                "structure_rejected_fallback requires terminal_validated=false and "
                "structure_feasible=false"
            )
        digest = getattr(row, "selection_provenance_digest")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(
                "selection_provenance_digest must be a lowercase SHA-256 digest"
            )


def _selection_authority_fields(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        if "selection_status" not in row:
            return {}
        return {column: row[column] for column in _SELECTION_AUTHORITY_COLUMNS}
    if not hasattr(row, "selection_status"):
        return {}
    return {column: getattr(row, column) for column in _SELECTION_AUTHORITY_COLUMNS}


def load_test_lookup(test_set_parquet: str | Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    df = pd.read_parquet(test_set_parquet).copy()
    required = {"protein_id", "sequence", "sequence_length", "pdb_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"test-set parquet missing required columns: {sorted(missing)}")
    df["protein_id"] = df["protein_id"].astype(str)
    if df["protein_id"].duplicated().any():
        raise ValueError("test-set parquet contains duplicate protein_id rows")
    return df, {str(row["protein_id"]): row.to_dict() for _, row in df.iterrows()}


def validate_sequence(sequence: str) -> str | None:
    if not sequence:
        return "empty_sequence"
    chars = set(sequence)
    invalid = chars - CANONICAL_AA
    if invalid:
        return f"non_canonical_aa:{''.join(sorted(invalid))}"
    return None


def build_head_predictor(
    *,
    checkpoint_path: str | Path,
    config_dir: str | Path | None,
    variant_id: str,
    device: str,
):
    from epitope_head.configs import (
        load_ablation_config,
        load_inference_config,
        load_model_config,
    )
    from scripts.score_head import build_predictor

    if config_dir is None:
        config_dir = PROJECT_ROOT / "epitope_head" / "configs"
    config_dir = Path(config_dir)
    return build_predictor(
        model_cfg=load_model_config(config_dir / "model.yaml"),
        ablation_cfg=load_ablation_config(config_dir / "model_ablation.yaml"),
        inference_cfg=load_inference_config(config_dir / "inference.yaml"),
        variant_id=variant_id,
        checkpoint_path=Path(checkpoint_path),
        device=device,
    )


def build_nmp_runner(
    *,
    binary_path: str | Path,
    batch_size: int,
    n_workers: int,
    timeout: int,
    max_lengths_per_call: int,
):
    from epitope_head.data.netmhciipan_runner import build_runner

    return build_runner(
        backend="standalone",
        binary_path=str(binary_path),
        batch_size=batch_size,
        subprocess_timeout=timeout,
        max_lengths_per_call=max_lengths_per_call,
        n_workers=n_workers,
    )


def evaluate_immunogenicity_rows(
    generated_df: pd.DataFrame,
    *,
    predictor: Any,
    nmp_runner: Any,
    allele: str,
    strong_binder_threshold: float = 2.0,
    nmp_batch_size: int = 8,
    progress_every: int = 25,
    hotspot_threshold: float = 0.5,
    full: bool = False,
    run_nmp: bool = True,
    return_full_tables: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
] | tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
]:
    head_rows: list[dict[str, Any]] = []
    nmp_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    residue_rows: list[dict[str, Any]] = []
    peptide_rows: list[dict[str, Any]] = []

    total = len(generated_df)
    started = time.time()

    # Head scoring is row-wise.
    for row_idx, row in enumerate(generated_df.itertuples(index=False), start=1):
        invalid = validate_sequence(str(row.sequence))
        if invalid is not None:
            failures.append(_failure_row(row, stage="imm_head", reason=invalid))
            continue
        try:
            pred = predictor.predict_protein(str(row.sequence))
            hotspot = pred["residue_hotspot"]
            selection_authority = _selection_authority_fields(row)
            head_rows.append(
                {
                    "protein_id": str(row.protein_id),
                    "design_id": str(row.design_id),
                    "design_idx": int(row.design_idx),
                    "global_risk": float(pred["global_risk"]),
                    "mean_hotspot": float(hotspot.mean()),
                    "max_hotspot": float(hotspot.max()),
                    "n_hotspot_positions": int((hotspot > hotspot_threshold).sum()),
                    **selection_authority,
                }
            )
            if full:
                hotspot_arr = hotspot.detach().cpu().numpy() if hasattr(hotspot, "detach") else hotspot
                sequence_str = str(row.sequence)
                for residue_idx, value in enumerate(hotspot_arr):
                    residue_rows.append(
                        {
                            "protein_id": str(row.protein_id),
                            "design_id": str(row.design_id),
                            "design_idx": int(row.design_idx),
                            "residue_idx": int(residue_idx),
                            "residue_aa": sequence_str[residue_idx]
                            if residue_idx < len(sequence_str)
                            else "",
                            "hotspot": float(value),
                            **selection_authority,
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            failures.append(_failure_row(row, stage="imm_head", reason=f"{type(exc).__name__}:{exc}"))
        if progress_every > 0 and (row_idx % progress_every == 0 or row_idx == total):
            _emit_progress(
                stage="imm_head",
                current=row_idx,
                total=total,
                n_rows=len(head_rows),
                n_failures=_count_stage_failures(failures, "imm_head"),
                started=started,
            )

    # NMP scoring is batched, but temp IDs must be unique across designs.
    # When run_nmp is False (head-only mode) the whole NetMHCIIpan pass is
    # skipped: no binary is launched and nmp_df is returned empty.
    nmp_started = time.time()
    row_records = list(generated_df.to_dict("records"))
    nmp_chunk_iter = range(0, len(row_records), nmp_batch_size) if run_nmp else range(0)
    for chunk_start in nmp_chunk_iter:
        chunk = row_records[chunk_start:chunk_start + nmp_batch_size]
        entries: list[tuple[str, str]] = []
        row_by_temp_id: dict[str, dict[str, Any]] = {}
        for row in chunk:
            invalid = validate_sequence(str(row["sequence"]))
            if invalid is not None:
                failures.append(_failure_row_dict(row, stage="imm_nmp", reason=invalid))
                continue
            temp_id = _temp_design_key(str(row["protein_id"]), int(row["design_idx"]))
            entries.append((temp_id, str(row["sequence"])))
            row_by_temp_id[temp_id] = row

        batch_scores: dict[str, dict[int, list[Any]]]
        if entries:
            try:
                batch_scores = nmp_runner.score_batch(entries, allele, NMP_PEP_LENGTHS)
            except Exception as exc:  # noqa: BLE001
                for temp_id, _seq in entries:
                    failures.append(
                        _failure_row_dict(
                            row_by_temp_id[temp_id],
                            stage="imm_nmp",
                            reason=f"{type(exc).__name__}:{exc}",
                        )
                    )
                batch_scores = {}
        else:
            batch_scores = {}

        for temp_id, row in row_by_temp_id.items():
            by_len = batch_scores.get(temp_id, {})
            if not by_len:
                failures.append(_failure_row_dict(row, stage="imm_nmp", reason="timeout_or_no_scores"))
                continue
            scores_df = _scores_by_len_to_dataframe(by_len)
            agg = aggregate_nmp_scores_with_threshold(scores_df, strong_binder_threshold)
            nmp_rows.append(
                {
                    "protein_id": str(row["protein_id"]),
                    "design_id": f"design_{int(row['design_idx']):04d}",
                    "design_idx": int(row["design_idx"]),
                    **agg,
                    **_selection_authority_fields(row),
                }
            )
            if full and not scores_df.empty:
                pid = str(row["protein_id"])
                did = f"design_{int(row['design_idx']):04d}"
                dix = int(row["design_idx"])
                for record in scores_df.to_dict("records"):
                    peptide_rows.append(
                        {
                            "protein_id": pid,
                            "design_id": did,
                            "design_idx": dix,
                            "pep_length": int(record["pep_length"]),
                            "pos": int(record["pos"]),
                            "peptide": str(record["peptide"]),
                            "core": str(record["core"]),
                            "rank_EL": float(record["rank_EL"]),
                            "el_score": float(record["el_score"]),
                            **_selection_authority_fields(row),
                        }
                    )

        done = min(chunk_start + nmp_batch_size, len(row_records))
        if progress_every > 0 and (done % progress_every == 0 or done == len(row_records)):
            _emit_progress(
                stage="imm_nmp",
                current=done,
                total=len(row_records),
                n_rows=len(nmp_rows),
                n_failures=_count_stage_failures(failures, "imm_nmp"),
                started=nmp_started,
            )

    head_df = pd.DataFrame(head_rows)
    nmp_df = pd.DataFrame(nmp_rows)
    residues_df = pd.DataFrame(residue_rows)
    peptides_df = pd.DataFrame(peptide_rows)
    if return_full_tables:
        return head_df, nmp_df, failures, residues_df, peptides_df
    return head_df, nmp_df, failures


def evaluate_structural_rows(
    generated_df: pd.DataFrame,
    test_lookup: dict[str, dict[str, Any]],
    *,
    pdb_root: str | Path,
    refold_backend: str,
    device: str,
    tmalign_bin: str,
    esmfold_cache_dir: str | None = None,
    progress_every: int = 25,
    return_residue_metrics: bool = False,
    return_v2_metrics: bool = False,
    legacy_metrics: bool = True,
    anchor_indices_by_protein: dict[str, set[int] | tuple[int, ...]] | None = None,
) -> Any:
    from inverse_folding.evaluation.tmalign import run_tmalign
    from inverse_folding.reference_flow.runtime import resolve_structure_path

    if not legacy_metrics and not return_v2_metrics:
        raise ValueError("legacy_metrics=False requires return_v2_metrics=True")
    if legacy_metrics:
        from inverse_folding.evaluation.sc_rmsd import compute_ca_self_consistency

    if return_v2_metrics:
        from inverse_folding.evaluation.structural_metrics_v2 import (
            ALL_METRICS,
            build_benchmark_residue_rows,
            build_benchmark_summary_row,
            evaluate_prediction,
            prepare_reference_context,
        )

    rows: list[dict[str, Any]] = []
    residue_rows: list[dict[str, Any]] = []
    v2_rows: list[dict[str, Any]] = []
    v2_residue_rows: list[dict[str, Any]] = []
    v2_reference_contexts: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []

    model = None
    cpu_model = None
    started = time.time()
    total = len(generated_df)

    if refold_backend == "esmfold":
        model = load_refold_model("esmfold", device=device)

    for row_idx, row in enumerate(generated_df.itertuples(index=False), start=1):
        test_row = test_lookup.get(str(row.protein_id))
        if test_row is None:
            failures.append(_failure_row(row, stage="struct", reason="missing_test_set_row"))
            continue

        expected_length = int(test_row["sequence_length"])
        sequence = str(row.sequence)
        if len(sequence) != expected_length:
            raise DataConsistencyError(
                f"{row.protein_id} design_idx={row.design_idx}: generated length "
                f"{len(sequence)} != test-set sequence_length {expected_length}"
            )

        invalid = validate_sequence(sequence)
        if invalid is not None:
            failures.append(_failure_row(row, stage="struct", reason=invalid))
            continue

        try:
            ref_path = resolve_structure_path(test_row, pdb_root)
        except FileNotFoundError:
            failures.append(_failure_row(row, stage="struct", reason="missing_pdb"))
            continue

        protein_id = str(row.protein_id)
        v2_context = None
        if return_v2_metrics:
            try:
                v2_context = v2_reference_contexts.get(protein_id)
                if v2_context is None:
                    anchors = (anchor_indices_by_protein or {}).get(protein_id, ())
                    v2_context = prepare_reference_context(
                        ref_path,
                        ref_sequence=str(test_row["sequence"]),
                        anchor_indices=anchors,
                    )
                    v2_reference_contexts[protein_id] = v2_context
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    _failure_row(
                        row,
                        stage="struct_v2",
                        reason=f"reference_context_failed:{type(exc).__name__}:{exc}",
                    )
                )
                continue

        design_id = str(row.design_id)
        pred = None
        try:
            pred = refold(
                sequence=sequence,
                protein_id=protein_id,
                design_id=design_id,
                backend=refold_backend,
                cache_dir=esmfold_cache_dir,
                model=model,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and str(device).startswith("cuda"):
                if cpu_model is None and refold_backend == "esmfold":
                    cpu_model = load_refold_model("esmfold", device="cpu")
                try:
                    pred = refold(
                        sequence=sequence,
                        protein_id=str(row.protein_id),
                        design_id=design_id,
                        backend=refold_backend,
                        cache_dir=esmfold_cache_dir,
                        model=cpu_model,
                    )
                except Exception as retry_exc:  # noqa: BLE001
                    failures.append(
                        _failure_row(
                            row,
                            stage="struct",
                            reason=f"oom_cpu_retry_failed:{type(retry_exc).__name__}:{retry_exc}",
                        )
                    )
                    continue
            else:
                raise

        recovery = compute_recovery(sequence, str(test_row["sequence"]))
        try:
            tm_metrics = run_tmalign(
                pred_pdb=str(pred["pdb_path"]),
                ref_pdb=str(ref_path),
                tmalign_bin=tmalign_bin,
                cache_dir=None,
            )
            sc_tm = float(tm_metrics["tm_score"])
            bb_rmsd = float(tm_metrics["rmsd"])
            foldability = bool(sc_tm > 0.5)
        except Exception as exc:  # noqa: BLE001
            failures.append(_failure_row(row, stage="struct", reason=f"tmalign_failed:{type(exc).__name__}:{exc}"))
            sc_tm = float("nan")
            bb_rmsd = float("nan")
            foldability = False

        sc_rmsd = float("nan")
        if legacy_metrics:
            try:
                sc_rmsd, per_residue_rows = compute_ca_self_consistency(
                    pred_pdb=str(pred["pdb_path"]),
                    ref_pdb=str(ref_path),
                    protein_id=str(row.protein_id),
                    design_id=design_id,
                    design_idx=int(row.design_idx),
                    design_sequence=sequence,
                    ref_sequence=str(test_row["sequence"]),
                    refold_backend=refold_backend,
                )
                residue_rows.extend(per_residue_rows)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    _failure_row(
                        row,
                        stage="struct_residue",
                        reason=f"{type(exc).__name__}:{exc}",
                    )
                )

        if return_v2_metrics:
            try:
                v2_result = evaluate_prediction(
                    v2_context,
                    str(pred["pdb_path"]),
                    design_sequence=sequence,
                    metrics=ALL_METRICS,
                )
                _validate_v2_prediction_plddt(
                    v2_result.predicted_global_plddt,
                    pred.get("pLDDT"),
                    backend=refold_backend,
                )
                v2_rows.append(
                    build_benchmark_summary_row(
                        v2_result,
                        protein_id=protein_id,
                        design_id=design_id,
                        design_idx=int(row.design_idx),
                        sequence=sequence,
                        sc_tm=sc_tm,
                        recovery=recovery,
                        refold_backend=refold_backend,
                    )
                )
                v2_residue_rows.extend(
                    build_benchmark_residue_rows(
                        v2_result,
                        protein_id=protein_id,
                        design_id=design_id,
                        design_idx=int(row.design_idx),
                        refold_backend=refold_backend,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    _failure_row(
                        row,
                        stage="struct_v2",
                        reason=f"{type(exc).__name__}:{exc}",
                    )
                )
                continue

        if legacy_metrics:
            rows.append(
                {
                    "protein_id": protein_id,
                    "design_id": design_id,
                    "design_idx": int(row.design_idx),
                    "sequence": sequence,
                    "scTM": sc_tm,
                    "pLDDT": float(pred["pLDDT"]),
                    "bb_RMSD": bb_rmsd,
                    "scRMSD": sc_rmsd,
                    "recovery": recovery,
                    "foldability": foldability,
                    "refold_backend": refold_backend,
                }
            )

        if progress_every > 0 and (row_idx % progress_every == 0 or row_idx == total):
            _emit_progress(
                stage="struct",
                current=row_idx,
                total=total,
                n_rows=len(rows) if legacy_metrics else len(v2_rows),
                n_failures=_count_stage_failures(failures, "struct"),
                started=started,
            )

    structural_df = pd.DataFrame(rows)
    structural_residue_df = pd.DataFrame(
        residue_rows,
        columns=sorted(LEGACY_STRUCTURAL_RESIDUE_COLUMNS),
    )
    structural_v2_df = pd.DataFrame(v2_rows, columns=sorted(STRUCTURAL_V2_COLUMNS))
    structural_v2_residue_df = pd.DataFrame(
        v2_residue_rows,
        columns=sorted(STRUCTURAL_V2_RESIDUE_COLUMNS),
    )
    if return_v2_metrics and not legacy_metrics:
        if return_residue_metrics:
            return structural_v2_df, failures, structural_v2_residue_df
        return structural_v2_df, failures
    if return_v2_metrics and return_residue_metrics:
        return (
            structural_df,
            failures,
            structural_residue_df,
            structural_v2_df,
            structural_v2_residue_df,
        )
    if return_v2_metrics:
        return structural_df, failures, structural_v2_df, structural_v2_residue_df
    if return_residue_metrics:
        return structural_df, failures, structural_residue_df
    return structural_df, failures


def compute_recovery(sequence: str, wt_sequence: str) -> float:
    if not wt_sequence:
        return float("nan")
    matches = sum(a == b for a, b in zip(sequence, wt_sequence))
    return matches / float(len(wt_sequence))


def _validate_v2_prediction_plddt(
    structure_mean: float | None,
    reported_mean: float | None,
    *,
    backend: str,
    tolerance: float = 1.0,
) -> None:
    """Reject cache PDBs that lost per-residue confidence during conversion."""

    if reported_mean is None or not math.isfinite(float(reported_mean)):
        raise ValueError(f"{backend} reported pLDDT is missing/not finite: {reported_mean}")
    if structure_mean is None or not math.isfinite(float(structure_mean)):
        raise ValueError(
            f"{backend} cache PDB lacks finite per-residue pLDDT B-factors"
        )
    if abs(float(structure_mean) - float(reported_mean)) > float(tolerance):
        raise ValueError(
            f"{backend} cache PDB pLDDT mean {structure_mean:.3f} disagrees with "
            f"sidecar {float(reported_mean):.3f}; regenerate the cache with the current "
            "refold normalizer so B-factors are preserved"
        )


def aggregate_nmp_scores_with_threshold(
    scores: pd.DataFrame,
    strong_binder_threshold: float,
) -> dict[str, Any]:
    n = len(scores)
    if n == 0:
        return {
            "n_strong_binders": 0,
            "n_weak_binders": 0,
            "mean_best_rank": float("nan"),
            "n_windows_scored": 0,
        }

    ranks = scores["rank_EL"]
    top_k = min(_NMP_TOP_K_FOR_MEAN_BEST, n)
    mean_best_rank = float(ranks.sort_values().head(top_k).mean()) if top_k > 0 else float("nan")
    return {
        "n_strong_binders": int((ranks < strong_binder_threshold).sum()),
        "n_weak_binders": int((ranks < _NMP_WEAK_BINDER_THRESHOLD).sum()),
        "mean_best_rank": mean_best_rank,
        "n_windows_scored": int(n),
    }


def build_resolved_run_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return a vars(args) copy with path-typed CLI values resolved to absolute paths.

    This keeps run_config.yaml's paths in sync with manifest.json, which uses
    Path(...).resolve() for the same fields.
    """
    resolved = vars(args).copy()
    path_fields = (
        "generated_parquet",
        "test_set_parquet",
        "output_root",
        "epitope_ckpt",
        "epitope_config_dir",
        "netmhciipan_bin",
        "pdb_root",
        "esmfold_cache_dir",
    )
    for key in path_fields:
        value = resolved.get(key)
        if value is None:
            continue
        resolved[key] = str(Path(value).resolve())
    # tmalign_bin may be a bare command name like "TMalign"; only resolve if it's
    # an actual filesystem path that exists.
    tmalign_bin = resolved.get("tmalign_bin")
    if tmalign_bin is not None:
        candidate = Path(tmalign_bin)
        if candidate.exists():
            resolved["tmalign_bin"] = str(candidate.resolve())
    return resolved


def build_manifest(
    *,
    args: argparse.Namespace,
    run_id: str,
    modes_run: list[str],
    rows_per_mode: dict[str, int],
    wall_seconds_per_mode: dict[str, float],
    generated_df: pd.DataFrame,
) -> dict[str, Any]:
    from inverse_folding.reference_flow.runtime import git_sha

    struct_in_run = any(str(mode).startswith("struct") for mode in modes_run)
    manifest = {
        "run_id": run_id,
        "modes_run": modes_run,
        "refold_backend": args.refold_model if struct_in_run else None,
        "allele": args.allele,
        "generated_parquet_path": str(Path(args.generated_parquet).resolve()),
        "generated_parquet_sha256": sha256_file(args.generated_parquet),
        "test_set_parquet_path": str(Path(args.test_set_parquet).resolve()),
        "head_ckpt_path": str(Path(args.epitope_ckpt).resolve()) if args.epitope_ckpt else None,
        "head_ckpt_digest": sha256_file(args.epitope_ckpt) if args.epitope_ckpt else None,
        "nmp_binary_path": str(Path(args.netmhciipan_bin).resolve()) if args.netmhciipan_bin else None,
        "nmp_version_string": probe_nmp_version(args.netmhciipan_bin) if args.netmhciipan_bin else None,
        "nmp_mode": getattr(args, "nmp_mode", None),
        "nmp_batch_size": int(getattr(args, "nmp_batch_size", 8)),
        "nmp_max_lengths_per_call": int(getattr(args, "nmp_max_lengths_per_call", 4)),
        "nmp_workers": int(getattr(args, "nmp_workers", 1)),
        "structural_metrics_v2": struct_in_run,
        "structural_metrics_version": "v2" if struct_in_run else None,
        "constraint_manifest_path": (
            str(Path(args.constraint_manifest).resolve())
            if (
                struct_in_run
                and getattr(args, "constraint_manifest", None)
            )
            else None
        ),
        "constraint_manifest_digest": (
            sha256_file(args.constraint_manifest)
            if (
                struct_in_run
                and getattr(args, "constraint_manifest", None)
            )
            else None
        ),
        "git_sha": git_sha(PROJECT_ROOT),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_input_designs": int(len(generated_df)),
        "n_rows_per_mode": rows_per_mode,
        "wall_seconds_per_mode": wall_seconds_per_mode,
    }
    return manifest


def probe_nmp_version(binary_path: str | Path | None) -> str | None:
    if binary_path is None:
        return None
    binary = Path(binary_path).expanduser()
    version_path = binary.parent / "data" / "version"
    try:
        version_text = version_path.read_text(encoding="utf-8").strip()
        if version_text:
            return version_text[:200]
    except (OSError, UnicodeError):
        pass
    try:
        result = subprocess.run(
            [str(binary)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        text = "\n".join([result.stdout.strip(), result.stderr.strip()]).strip()
        for line in text.splitlines():
            candidate = line.strip()
            lowered = candidate.lower()
            if candidate and "netmhcii" in lowered and "version" in lowered:
                return candidate[:200]
    except Exception:  # noqa: BLE001
        return None
    return None


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def output_paths(run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    return {
        "imm_head": run_dir / "imm_head.parquet",
        "imm_nmp": run_dir / "imm_nmp.parquet",
        "imm_head_residues": run_dir / "imm_head_residues.parquet",
        "imm_nmp_peptides": run_dir / "imm_nmp_peptides.parquet",
        "structural": run_dir / "structural.parquet",
        "structural_residues": run_dir / "structural_residues.parquet",
        "manifest": run_dir / "manifest.json",
        "failures": run_dir / "failures.json",
        "run_config": run_dir / "run_config.yaml",
        "partial_dir": run_dir / ".partial",
    }


def mode_outputs_exist(
    mode: str,
    paths: dict[str, Path],
    *,
    no_nmp: bool = False,
) -> tuple[bool, list[Path]]:
    if mode == "imm":
        required = [paths["imm_head"]] if no_nmp else [paths["imm_head"], paths["imm_nmp"]]
    elif mode == "struct":
        required = [paths["structural"], paths["structural_residues"]]
    else:
        raise ValueError(mode)
    existing = [path for path in required if path.exists()]
    complete = len(existing) == len(required)
    if complete and mode == "struct":
        complete = _parquet_has_columns(paths["structural"], STRUCTURAL_COLUMNS) and (
            _parquet_has_columns(paths["structural_residues"], STRUCTURAL_RESIDUE_COLUMNS)
        )
    return complete, existing


def _parquet_has_columns(path: Path, required: set[str] | frozenset[str]) -> bool:
    try:
        import pyarrow.parquet as pq

        return set(required) <= set(pq.read_schema(path).names)
    except Exception:  # noqa: BLE001
        return False


def ensure_run_dir_state(
    *,
    run_dir: Path,
    requested_modes: list[str],
    paths: dict[str, Path],
    overwrite: bool,
    no_nmp: bool = False,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return
    if not any(run_dir.iterdir()):
        return
    missing_for_requested = False
    for mode in requested_modes:
        mode_complete, existing = mode_outputs_exist(
            mode,
            paths,
            no_nmp=no_nmp,
        )
        if existing and not mode_complete:
            raise FileExistsError(
                f"run_dir contains partial outputs for mode={mode}; rerun with --overwrite"
            )
        if not mode_complete:
            missing_for_requested = True
    if missing_for_requested:
        # Allow existing run_dir only when all requested modes can be skipped cleanly.
        raise FileExistsError(
            f"run_dir already exists and is non-empty: {run_dir}. "
            "Use --overwrite or a new --run-id."
        )


def write_run_metadata(
    *,
    paths: dict[str, Path],
    manifest: dict[str, Any],
    run_config: dict[str, Any],
    failures: list[dict[str, Any]],
) -> None:
    with open(paths["manifest"], "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    with open(paths["run_config"], "w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)
    with open(paths["failures"], "w") as f:
        json.dump({"failures": failures}, f, indent=2, sort_keys=True)


def write_partial_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _temp_design_key(protein_id: str, design_idx: int) -> str:
    return f"{protein_id}__design_{design_idx:04d}"


def _scores_by_len_to_dataframe(by_len: dict[int, list[Any]]) -> pd.DataFrame:
    rows = []
    for pep_length, scores in by_len.items():
        for score in scores:
            rows.append(
                {
                    "peptide": score.peptide,
                    "rank_EL": score.el_rank * 100.0,
                    "pos": score.pos,
                    "core": score.core,
                    "el_score": score.el_score,
                    "pep_length": int(pep_length),
                }
            )
    return pd.DataFrame(rows)


def _failure_row(row: Any, *, stage: str, reason: str) -> dict[str, Any]:
    return {
        "protein_id": str(row.protein_id),
        "design_id": str(row.design_id),
        "design_idx": int(row.design_idx),
        "stage": stage,
        "reason": reason,
    }


def _failure_row_dict(row: dict[str, Any], *, stage: str, reason: str) -> dict[str, Any]:
    return {
        "protein_id": str(row["protein_id"]),
        "design_id": f"design_{int(row['design_idx']):04d}",
        "design_idx": int(row["design_idx"]),
        "stage": stage,
        "reason": reason,
    }


def _count_stage_failures(failures: list[dict[str, Any]], stage: str) -> int:
    return sum(1 for row in failures if row.get("stage") == stage)


def _emit_progress(
    *,
    stage: str,
    current: int,
    total: int,
    n_rows: int,
    n_failures: int,
    started: float,
) -> None:
    elapsed = time.time() - started
    done = n_rows + n_failures
    eta = elapsed * max(total - done, 0) / float(done) if done > 0 else 0.0
    avg = elapsed / float(done) if done > 0 else 0.0
    print(
        f"[progress:{stage}] {current}/{total} rows={n_rows} failures={n_failures} "
        f"elapsed={_fmt_hms(elapsed)} avg_per_row={avg:.2f}s eta={_fmt_hms(eta)}",
        flush=True,
    )


def run_mode_imm(
    *,
    args: argparse.Namespace,
    generated_df: pd.DataFrame,
    paths: dict[str, Path],
) -> tuple[int, float, list[dict[str, Any]]]:
    no_nmp = bool(getattr(args, "no_nmp", False))
    predictor = build_head_predictor(
        checkpoint_path=args.epitope_ckpt,
        config_dir=args.epitope_config_dir,
        variant_id=args.epitope_variant_id,
        device=args.device,
    )
    nmp_runner = (
        None
        if no_nmp
        else build_nmp_runner(
            binary_path=args.netmhciipan_bin,
            batch_size=args.nmp_batch_size,
            n_workers=args.nmp_workers,
            timeout=args.nmp_timeout,
            max_lengths_per_call=args.nmp_max_lengths_per_call,
        )
    )
    if no_nmp:
        print("[imm] --no-nmp: head-only scoring, NetMHCIIpan skipped", flush=True)
    started = time.time()
    head_df, nmp_df, failures, residues_df, peptides_df = evaluate_immunogenicity_rows(
        generated_df,
        predictor=predictor,
        nmp_runner=nmp_runner,
        allele=args.allele,
        strong_binder_threshold=args.strong_binder_threshold,
        nmp_batch_size=args.nmp_batch_size,
        progress_every=args.progress_every,
        hotspot_threshold=args.hotspot_threshold,
        full=bool(getattr(args, "imm_full", False)),
        run_nmp=not no_nmp,
        return_full_tables=True,
    )
    failed_keys = {
        (row["protein_id"], int(row["design_idx"]))
        for row in failures
        if row["stage"] in {"imm_head", "imm_nmp"}
    }
    if len(failed_keys) / float(max(len(generated_df), 1)) > args.fail_pct_threshold:
        partial_dir = paths["partial_dir"]
        write_partial_dataframe(partial_dir / "imm_head.parquet", head_df)
        write_partial_dataframe(partial_dir / "imm_nmp.parquet", nmp_df)
        raise RuntimeError("imm mode exceeded fail-pct-threshold")

    validate_dataframe(head_df, IMMUNOGENICITY_HEAD_COLUMNS)
    head_df.to_parquet(paths["imm_head"], index=False)
    if not no_nmp:
        validate_dataframe(nmp_df, IMMUNOGENICITY_NMP_COLUMNS)
        nmp_df.to_parquet(paths["imm_nmp"], index=False)
    if bool(getattr(args, "imm_full", False)):
        # Long-tables are best-effort: emit even if empty so callers can
        # detect the run was --imm-full vs aggregate-only.
        residues_df.to_parquet(paths["imm_head_residues"], index=False)
        if not no_nmp:
            peptides_df.to_parquet(paths["imm_nmp_peptides"], index=False)
        print(
            f"[imm-full] wrote {len(residues_df)} residue rows -> "
            f"{paths['imm_head_residues'].name}, {len(peptides_df)} peptide rows -> "
            f"{paths['imm_nmp_peptides'].name}",
            flush=True,
        )
    return len(head_df), time.time() - started, failures


def run_mode_struct(
    *,
    args: argparse.Namespace,
    generated_df: pd.DataFrame,
    test_lookup: dict[str, dict[str, Any]],
    paths: dict[str, Path],
) -> tuple[int, float, list[dict[str, Any]]]:
    started = time.time()
    anchor_indices_by_protein = _load_anchor_indices_by_protein(
        getattr(args, "constraint_manifest", None),
        test_lookup,
    )
    df, failures, residue_df = evaluate_structural_rows(
        generated_df,
        test_lookup,
        pdb_root=args.pdb_root,
        refold_backend=args.refold_model,
        device=args.device,
        tmalign_bin=args.tmalign_bin,
        esmfold_cache_dir=args.esmfold_cache_dir,
        progress_every=args.progress_every,
        return_residue_metrics=True,
        return_v2_metrics=True,
        legacy_metrics=False,
        anchor_indices_by_protein=anchor_indices_by_protein,
    )
    failed_keys = {
        (row["protein_id"], int(row["design_idx"]))
        for row in failures
        if row["stage"] in {"struct", "struct_v2"}
    }
    if len(failed_keys) / float(max(len(generated_df), 1)) > args.fail_pct_threshold:
        partial_dir = paths["partial_dir"]
        write_partial_dataframe(partial_dir / "structural.parquet", df)
        write_partial_dataframe(partial_dir / "structural_residues.parquet", residue_df)
        raise RuntimeError("struct mode exceeded fail-pct-threshold")

    validate_dataframe(df, STRUCTURAL_COLUMNS)
    if residue_df.empty:
        raise RuntimeError("struct mode produced empty structural_residues.parquet")
    validate_dataframe(residue_df, STRUCTURAL_RESIDUE_COLUMNS)

    df.to_parquet(paths["structural"], index=False)
    residue_df.to_parquet(paths["structural_residues"], index=False)
    return len(df), time.time() - started, failures


def _load_anchor_indices_by_protein(
    constraint_manifest: str | Path | None,
    test_lookup: dict[str, dict[str, Any]],
) -> dict[str, set[int]]:
    """Validate an optional manifest against benchmark reference sequences once."""

    if constraint_manifest is None:
        return {}
    from inverse_folding.reference_flow.constraints import load_constraint_manifest

    manifest = load_constraint_manifest(constraint_manifest)
    anchors: dict[str, set[int]] = {}
    for protein_id, test_row in test_lookup.items():
        if not manifest.has_protein(protein_id):
            continue
        constraint = manifest.constraint_for_protein(protein_id)
        constraint.validate_against_sequence(str(test_row["sequence"]))
        anchors[protein_id] = set(constraint.hard_anchor_indices)
    return anchors


def _log_evaluation_distributions(wandb_run: Any, paths: dict[str, Path]) -> None:
    """Emit per-design metric histograms to wandb at the end of a run."""
    if wandb_run is None:
        return
    try:
        import wandb as _wandb
    except Exception:  # noqa: BLE001
        return

    hist_targets: dict[str, tuple[Path, list[str]]] = {
        "imm_head": (paths["imm_head"], ["global_risk", "mean_hotspot", "max_hotspot", "n_hotspot_positions"]),
        "imm_nmp": (paths["imm_nmp"], ["n_strong_binders", "n_weak_binders", "mean_best_rank", "n_windows_scored"]),
        "structural": (
            paths["structural"],
            [
                "scTM",
                "global_ca_RMSD",
                "pLDDT",
                "active_site_sidechain_RMSD",
                "max_anchor_sidechain_RMSD",
                "max_anchor_atom_distance",
                "recovery",
            ],
        ),
    }

    payload: dict[str, Any] = {}
    for mode, (parquet_path, columns) in hist_targets.items():
        if not parquet_path.exists():
            continue
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:  # noqa: BLE001
            continue
        for col in columns:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            try:
                payload[f"distribution/{mode}/{col}"] = _wandb.Histogram(
                    series.to_numpy()
                )
            except Exception:  # noqa: BLE001
                continue
            payload[f"summary/{mode}/{col}_mean"] = float(series.mean())
            payload[f"summary/{mode}/{col}_median"] = float(series.median())

        if mode == "structural" and "foldability" in df.columns:
            payload[f"summary/{mode}/foldability_rate"] = float(
                pd.Series(df["foldability"], dtype="boolean").fillna(False).mean()
            )

    if payload:
        wandb_run.log(payload)


def print_resolved_hyperparams(args: argparse.Namespace, *, run_dir: Path, n_designs: int) -> None:
    resolved = vars(args).copy()
    resolved["generated_parquet"] = str(Path(args.generated_parquet).resolve())
    resolved["test_set_parquet"] = str(Path(args.test_set_parquet).resolve())
    resolved["output_root"] = str(Path(args.output_root).resolve())
    resolved["run_dir"] = str(run_dir)
    resolved["n_input_designs"] = n_designs
    print("============================================================")
    print("Phase C evaluation resolved parameters")
    for key, value in resolved.items():
        print(f"  {key}: {value}")
    print("============================================================")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generated_df = load_generated_designs(args.generated_parquet)
    _test_df, test_lookup = load_test_lookup(args.test_set_parquet)
    run_id = build_run_id(args)

    from inverse_folding.reference_flow.runtime import safe_allele_tag

    run_dir = Path(args.output_root) / safe_allele_tag(args.allele) / run_id
    paths = output_paths(run_dir)
    requested_modes = ["imm", "struct"] if args.mode == "all" else [args.mode]
    ensure_run_dir_state(
        run_dir=run_dir,
        requested_modes=requested_modes,
        paths=paths,
        overwrite=args.overwrite,
        no_nmp=args.no_nmp,
    )
    print_resolved_hyperparams(args, run_dir=run_dir, n_designs=len(generated_df))

    if "imm" in requested_modes:
        if args.epitope_ckpt is None:
            raise ValueError("--epitope-ckpt is required for imm/all mode")
        if args.netmhciipan_bin is None and not args.no_nmp:
            raise ValueError(
                "--netmhciipan-bin is required for imm/all mode (or pass --no-nmp for head-only)"
            )
    if "struct" in requested_modes:
        if args.pdb_root is None:
            raise ValueError("--pdb-root is required for struct/all mode")
        if args.refold_model in CACHE_READ_BACKENDS and args.esmfold_cache_dir is None:
            raise ValueError(
                f"--refold-cache-dir is required for cache-read backend {args.refold_model!r}"
            )
        if args.esmfold_cache_dir is None:
            args.esmfold_cache_dir = str(run_dir / ".cache" / "esmfold")

    from inverse_folding.observability import (
        finish_wandb,
        init_wandb_from_args,
        log_metrics,
        set_summary,
    )

    wandb_run = init_wandb_from_args(
        args,
        run_name=run_id,
        config={
            "stage": "phase_c_evaluation",
            "modes_requested": requested_modes,
            "allele": args.allele,
            "refold_model": args.refold_model,
            "generated_parquet": str(Path(args.generated_parquet).resolve()),
            "test_set_parquet": str(Path(args.test_set_parquet).resolve()),
            "epitope_ckpt": str(Path(args.epitope_ckpt).resolve()) if args.epitope_ckpt else None,
            "epitope_variant_id": args.epitope_variant_id,
            "n_input_designs": int(len(generated_df)),
            "fail_pct_threshold": args.fail_pct_threshold,
            "hotspot_threshold": args.hotspot_threshold,
            "strong_binder_threshold_pct": args.strong_binder_threshold,
            "device": args.device,
            "run_dir": str(run_dir),
        },
        extra_tags=[
            f"mode={args.mode}",
            f"refold={args.refold_model}",
            args.allele,
        ],
    )

    all_failures: list[dict[str, Any]] = []
    modes_run: list[str] = []
    rows_per_mode: dict[str, int] = {}
    wall_seconds_per_mode: dict[str, float] = {}

    try:
        for mode in requested_modes:
            complete, existing = mode_outputs_exist(
                mode,
                paths,
                no_nmp=args.no_nmp,
            )
            if complete and not args.overwrite:
                print(f"[skip] mode={mode} outputs already exist in {run_dir}", flush=True)
                modes_run.append(f"{mode}:skipped")
                continue
            if existing and not complete and not args.overwrite:
                raise FileExistsError(
                    f"mode={mode} has partial outputs in {run_dir}; rerun with --overwrite"
                )

            if mode == "imm":
                n_rows, wall_seconds, failures = run_mode_imm(
                    args=args,
                    generated_df=generated_df,
                    paths=paths,
                )
            else:
                n_rows, wall_seconds, failures = run_mode_struct(
                    args=args,
                    generated_df=generated_df,
                    test_lookup=test_lookup,
                    paths=paths,
                )
            modes_run.append(mode)
            rows_per_mode[mode] = n_rows
            wall_seconds_per_mode[mode] = wall_seconds
            all_failures.extend(failures)
            log_metrics(
                wandb_run,
                {
                    f"mode/{mode}/n_rows": n_rows,
                    f"mode/{mode}/wall_seconds": float(wall_seconds),
                    f"mode/{mode}/n_failures": int(
                        sum(1 for f in failures if f.get("stage", "").startswith(mode))
                    ),
                },
            )
    except DataConsistencyError as exc:
        write_run_metadata(
            paths=paths,
            manifest=build_manifest(
                args=args,
                run_id=run_id,
                modes_run=modes_run,
                rows_per_mode=rows_per_mode,
                wall_seconds_per_mode=wall_seconds_per_mode,
                generated_df=generated_df,
            ),
            run_config=build_resolved_run_config(args),
            failures=all_failures,
        )
        print(f"ERROR: data consistency: {exc}", file=sys.stderr)
        set_summary(wandb_run, {"summary/aborted_reason": "data_consistency"})
        finish_wandb(wandb_run)
        return 2
    except RuntimeError as exc:
        if "fail-pct-threshold" in str(exc):
            write_run_metadata(
                paths=paths,
                manifest=build_manifest(
                    args=args,
                    run_id=run_id,
                    modes_run=modes_run,
                    rows_per_mode=rows_per_mode,
                    wall_seconds_per_mode=wall_seconds_per_mode,
                    generated_df=generated_df,
                ),
                run_config=build_resolved_run_config(args),
                failures=all_failures,
            )
            print(f"ERROR: {exc}", file=sys.stderr)
            set_summary(wandb_run, {"summary/aborted_reason": "fail_pct_threshold"})
            finish_wandb(wandb_run)
            return 2
        finish_wandb(wandb_run)
        raise

    write_run_metadata(
        paths=paths,
        manifest=build_manifest(
            args=args,
            run_id=run_id,
            modes_run=modes_run,
            rows_per_mode=rows_per_mode,
            wall_seconds_per_mode=wall_seconds_per_mode,
            generated_df=generated_df,
        ),
        run_config=build_resolved_run_config(args),
        failures=all_failures,
    )

    # Banner summary mirroring scripts/run_if_phase_c0.py / run_if_phase_c1.py.
    n_input = max(len(generated_df), 1)
    total_failures = len(all_failures)
    failure_rate = total_failures / float(n_input)
    total_wall = sum(wall_seconds_per_mode.values())
    skipped_modes = [m.split(":", 1)[0] for m in modes_run if m.endswith(":skipped")]
    skipped_str = ",".join(skipped_modes) if skipped_modes else "<none>"

    print("============================================================", flush=True)
    print(f"[done] run_id={run_id}", flush=True)
    print(f"       run_dir={run_dir}", flush=True)
    for mode in ("imm", "struct"):
        if mode in rows_per_mode:
            n_rows = rows_per_mode[mode]
            wall = wall_seconds_per_mode.get(mode, 0.0)
            avg = wall / float(n_rows) if n_rows > 0 else 0.0
            print(
                f"   {mode:<6} : rows={n_rows} wall={_fmt_hms(wall)} avg={avg:.2f}s",
                flush=True,
            )
    print(
        f"   skipped: {skipped_str} | total_failures={total_failures} "
        f"({failure_rate:.2%}) total_wall={_fmt_hms(total_wall)}",
        flush=True,
    )
    print("============================================================", flush=True)

    summary: dict[str, Any] = {
        "summary/n_input_designs": int(n_input),
        "summary/total_failures": int(total_failures),
        "summary/failure_rate": float(failure_rate),
        "summary/total_wall_seconds": float(total_wall),
        "summary/modes_run": ",".join(modes_run),
    }
    for mode, n in rows_per_mode.items():
        summary[f"summary/mode_{mode}_rows"] = int(n)
        summary[f"summary/mode_{mode}_wall_seconds"] = float(
            wall_seconds_per_mode.get(mode, 0.0)
        )
    set_summary(wandb_run, summary)
    _log_evaluation_distributions(wandb_run, paths)
    finish_wandb(wandb_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
