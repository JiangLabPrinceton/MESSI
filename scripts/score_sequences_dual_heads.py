#!/usr/bin/env python
"""Score an exact sequence table with BOTH frozen Dual Heads and one signed overlay (DP1 / DP4).

The paper-facing comparison puts Fusion, WT and ProteinMPNN on one axis pair, so all three have to
be measured by the same two instruments under the same coordinates.  The campaign's own endpoints
get that for free -- the runtime scores them through the overlay.  WT and the comparator panel do
not exist inside any run, so they are scored here, through the SAME ``ProductionHeadOracle``
contract and the SAME signed overlay, rather than through a second scoring path that would have to
be argued into agreement.

Three refusals, because each one is a way the comparison could be wrong while looking right:

* **the overlay must bind these exact Heads.**  ``materialize_v2_canary_config.py`` does not
  cross-check ``--dual-overlay`` against the cell's Head paths, so nothing else in the pipeline
  proves that the affine coordinates were measured on the checkpoints doing the scoring.  A
  mismatch here silently rescales every risk.
* **one sequence, two Heads.**  ``DualObjective.evaluate_scores`` is given both scores and proves
  they describe the same molecule on one window grid before reducing them.  Reporting an A risk
  from one sequence beside a B risk from another is exactly what runbook §7.6 forbids, and it is
  invisible in the output table.
* **no score is invented.**  A non-finite risk fails the run rather than becoming a NaN row that a
  downstream ``dropna`` would quietly turn into a smaller, more favourable cohort.

The reduced value ``J`` and the normalized coordinates come from the overlay's own objective, never
from a local re-implementation of the smooth-max: a second implementation is a second law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_COLUMNS = ("protein_id", "design_idx", "sequence")


class DualScoringError(RuntimeError):
    """The requested scoring cannot bind one sequence table to one signed objective."""


def read_sequences(specs: Sequence[str]) -> pd.DataFrame:
    """``LABEL=PATH`` tables, concatenated with the label kept as the method column."""
    frames = []
    for spec in specs:
        label, _, path = spec.partition("=")
        if not path:
            raise DualScoringError(f"--sequences expects LABEL=PATH, got {spec!r}")
        frame = pd.read_parquet(path)
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise DualScoringError(f"{path} is missing {missing}")
        frames.append(frame[list(REQUIRED_COLUMNS)].assign(
            method=label, source=str(Path(path).resolve())))
    table = pd.concat(frames, ignore_index=True)
    duplicated = table.duplicated(subset=["method", "protein_id", "design_idx"])
    if duplicated.any():
        raise DualScoringError(
            f"{int(duplicated.sum())} duplicated (method, protein_id, design_idx) rows; the join "
            "key of the paired read would not be unique")
    return table


def assert_overlay_binds(overlay: Any, *, evaluator_a: Any, evaluator_b: Any) -> None:
    """The affine coordinates must have been measured on the Heads that are about to score."""
    from inverse_folding.reference_flow.fusion_v2.joint_objective import AlleleRole

    coordinates = overlay.calibration.coordinates("global_risk")
    for role, live in ((AlleleRole.A, evaluator_a), (AlleleRole.B, evaluator_b)):
        bound = coordinates.coordinate(role).evaluator
        if bound.digest() != live.digest():
            raise DualScoringError(
                f"role {role.value}: the overlay was calibrated on Head "
                f"{bound.head_checkpoint_digest[:12]} ({bound.allele}, k={bound.window_k_min}.."
                f"{bound.window_k_max}, {bound.score_scale}) but the live Head is "
                f"{live.head_checkpoint_digest[:12]} ({live.allele}, k={live.window_k_min}.."
                f"{live.window_k_max}, {live.score_scale}); b and s would rescale a risk they were "
                "never measured on")


class _ScoreView:
    """A field view of one Head result, in the shape ``DualObjective.evaluate_scores`` reads.

    ``ProductionHeadOracle`` returns a ``V2HeadResult`` that carries its window-grid digest on
    ``.binding`` (the runtime matches results to completions by that binding rather than by
    position).  ``evaluate_scores`` reads ``window_grid_digest`` off the score itself and refuses a
    pair without it -- correctly, since two grids are two landscapes.  This surfaces the field the
    oracle already computed; it recomputes nothing, because a second scoring path is how a
    calibration and the gate it feeds start measuring different quantities.
    """

    __slots__ = ("protein_id", "sequence_md5", "sequence_length", "window_grid_digest",
                 "allele", "score_scale", "global_risk")

    def __init__(self, result: Any) -> None:
        binding = getattr(result, "binding", None)
        digest = getattr(result, "window_grid_digest", None)
        if digest is None:
            digest = getattr(binding, "window_grid_digest", None)
        if digest is None:
            raise DualScoringError(
                f"the Head result for {getattr(result, 'sequence_md5', '?')} carries no window "
                "grid digest, on itself or on its binding; without it a pair cannot be proved to "
                "describe one landscape")
        self.protein_id = str(result.protein_id)
        self.sequence_md5 = str(result.sequence_md5)
        self.sequence_length = int(result.sequence_length)
        self.window_grid_digest = str(digest)
        self.allele = str(result.allele)
        self.score_scale = str(result.score_scale)
        self.global_risk = float(result.global_risk)


def _require_finite(record: dict[str, Any], digest: str, **values: float) -> None:
    import math

    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise DualScoringError(
                f"{record['protein_id']}:{digest} produced a non-finite {name}; a comparison "
                "cannot silently drop the sequences its instrument failed on")


def score_table(
    table: pd.DataFrame, *, head_a: Any, head_b: Any, objective: Any, chunk: int = 500,
) -> list[dict[str, Any]]:
    import time

    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest

    records = table.to_dict("records")
    for record in records:
        record["sequence"] = str(record["sequence"])
        record["sequence_md5"] = hashlib.md5(
            record["sequence"].encode("utf-8")).hexdigest()

    def pass_over(oracle: Any, label: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        started = time.monotonic()
        for offset in range(0, len(records), chunk):
            batch = records[offset:offset + chunk]
            for result in oracle.score([
                OracleRequest(protein_id=str(r["protein_id"]), sequence=r["sequence"],
                              sequence_md5=r["sequence_md5"],
                              sequence_length=len(r["sequence"]))
                    for r in batch]):
                score = getattr(result, "score", result)
                out[str(score.sequence_md5)] = score
            done = min(offset + chunk, len(records))
            rate = done / max(1e-9, time.monotonic() - started)
            print(f"[dualscore] {label}: {done}/{len(records)} ({rate:.1f} seq/s)", flush=True)
        return out

    scores_a = pass_over(head_a, "head A")
    scores_b = pass_over(head_b, "head B")

    rows: list[dict[str, Any]] = []
    for record in records:
        digest = record["sequence_md5"]
        result_a, result_b = scores_a.get(digest), scores_b.get(digest)
        if result_a is None or result_b is None:
            raise DualScoringError(
                f"{record['protein_id']}:{digest} was not returned by both Heads")
        score_a, score_b = _ScoreView(result_a), _ScoreView(result_b)
        raw_a, raw_b = score_a.global_risk, score_b.global_risk
        # Checked BEFORE the reduction: a non-finite raw risk should be reported as the instrument
        # failure it is, not as whatever the objective happens to raise downstream of it.
        _require_finite(record, digest, R_A=raw_a, R_B=raw_b)
        joint = objective.evaluate_scores(score_a=score_a, score_b=score_b)
        _require_finite(record, digest, u_a=joint.u_a, u_b=joint.u_b, J=joint.value)
        rows.append({
            "method": record["method"], "protein_id": str(record["protein_id"]),
            "design_idx": int(record["design_idx"]), "sequence": record["sequence"],
            "sequence_md5": digest, "sequence_length": len(record["sequence"]),
            "R_A": raw_a, "R_B": raw_b,
            "u_a": float(joint.u_a), "u_b": float(joint.u_b), "J": float(joint.value),
            "active_worst": joint.active_worst.value,
            "objective_digest": joint.objective_digest,
            "source": record["source"],
        })
    rows.sort(key=lambda row: (row["method"], row["protein_id"], row["design_idx"]))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequences", action="append", required=True,
                        help="LABEL=PATH parquet with protein_id/design_idx/sequence (repeatable)")
    parser.add_argument("--overlay", required=True, type=Path,
                        help="the signed dual overlay whose coordinates define u and J")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--out-manifest", type=Path, default=None)
    parser.add_argument("--quantity", default="global_risk")
    parser.add_argument("--score-scale", default="raw_logit")
    parser.add_argument("--window-k-min", type=int, required=True)
    parser.add_argument("--window-k-max", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    for role in ("a", "b"):
        group = parser.add_argument_group(f"head {role.upper()}")
        group.add_argument(f"--head-{role}-config-dir", type=Path, required=True)
        group.add_argument(f"--head-{role}-checkpoint", type=Path, required=True)
        group.add_argument(f"--head-{role}-variant-id", default=None)
        group.add_argument(f"--head-{role}-allele", required=True)
        group.add_argument(f"--head-{role}-allele-idx", type=int, default=0)
        group.add_argument(f"--head-{role}-window-batch-size", type=int, default=64)
    return parser


def _load_role(args: argparse.Namespace, role: str) -> Any:
    from types import SimpleNamespace

    from scripts.analysis.replay_v2_head_directed_policy import _load_head

    return _load_head(SimpleNamespace(
        head_config=getattr(args, f"head_{role}_config_dir"),
        head_checkpoint=getattr(args, f"head_{role}_checkpoint"),
        head_variant_id=getattr(args, f"head_{role}_variant_id"),
        head_allele_idx=getattr(args, f"head_{role}_allele_idx"),
        head_window_batch_size=getattr(args, f"head_{role}_window_batch_size"),
        allele=getattr(args, f"head_{role}_allele"),
        score_scale=args.score_scale, window_k_min=args.window_k_min,
        window_k_max=args.window_k_max, device=args.device))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from inverse_folding.reference_flow.fusion_v2.joint_objective import DualObjective
    from scripts.rf_fusion_v2_preflight import load_dual_overlay_file

    table = read_sequences(args.sequences)
    overlay = load_dual_overlay_file(args.overlay)
    head_a, head_b = _load_role(args, "a"), _load_role(args, "b")
    assert_overlay_binds(overlay, evaluator_a=head_a.evaluator_identity(),
                         evaluator_b=head_b.evaluator_identity())
    objective = DualObjective(overlay.calibration, quantity=args.quantity)
    rows = score_table(table, head_a=head_a, head_b=head_b, objective=objective)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(args.out, index=False)
    manifest = {
        "schema_version": "dual-comparator-scores/1",
        "overlay": str(Path(args.overlay).resolve()),
        "overlay_content_digest": overlay.content_digest,
        "objective_digest": rows[0]["objective_digest"] if rows else None,
        "quantity": args.quantity,
        "head_a": head_a.evaluator_identity().canonical_payload(),
        "head_b": head_b.evaluator_identity().canonical_payload(),
        "sources": sorted(frame["source"].unique().tolist()),
        "rows_by_method": {str(k): int(v) for k, v in frame["method"].value_counts().items()},
        "proteins_by_method": {
            str(method): int(group["protein_id"].nunique())
            for method, group in frame.groupby("method")},
        "out_sha256": hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
    }
    if args.out_manifest is not None:
        args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.out_manifest.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                                     encoding="utf-8")
    print(json.dumps(manifest, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
