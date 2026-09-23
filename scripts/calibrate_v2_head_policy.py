"""Calibrate the two V2F5A Head-scale floors from stored-versus-live repeat scoring.

This is a Head-only cluster calibration.  It reads complete sequences and the Head risks recorded
by an existing V2 cohort, scores the same bytes again with the frozen production Head, and binds
the maximum absolute repeat drift into the typed ``PolicyCalibrationArtifact`` required by
``HeadDirectedCappedPolicy``.  The policy compares DIFFERENCES of two Head scores, so both the donor
margin and the leave-one-out local-contribution floor use the conservative difference bound
``2 * max_abs_repeat_drift``.

No denoiser or structure backend is loaded.  The input bundles, Head paths, policy spec and outputs
are CLI arguments; the selected rows and their canonical digest are persisted so the scalar can be
audited rather than reconstructed from a log line.

``--sequence-bundle`` measures the same bound for a Head that NO bundle on this cluster was ever
scored by -- a newly promoted production checkpoint.  The stored-vs-live path refuses such a
bundle, and rightly: a difference between two checkpoints' scores measures the checkpoint change,
not the instrument's own repeatability.  So this mode reads only the bytes and scores them TWICE
with the live Head, in shuffled order and at a different window batch size, exactly as
``calibrate_v2_dual_objective.score_panel_twice`` does.  Order alone is not enough -- it can leave
every batch the Head forms unchanged, and then the measured drift is a tautology.  The artifact
records which method produced it, and the method is inside the calibration digest, so one row
table can never wear two provenances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion.state import sequence_md5  # noqa: E402
from inverse_folding.reference_flow.fusion_v2.config import (  # noqa: E402
    PolicyCalibratedScalar,
    PolicyCalibrationArtifact,
    V2_POLICY_CALIBRATION_SCHEMA_VERSION,
    policy_calibration_source_ref,
)
from inverse_folding.reference_flow.fusion_v2.identity import canonical_digest  # noqa: E402
from inverse_folding.reference_flow.fusion_v2.policy import (  # noqa: E402
    HEAD_DIRECTED_CAPPED_POLICY_ID,
    REOPEN_COUNT_LAW,
    REOPEN_PRIORITY_LAW,
    SOURCE_GEOMETRY_CONTROL_POLICY_ID,
    WRITE_WINDOW_RULE,
)
from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest  # noqa: E402

CALIBRATION_BUNDLE_SCHEMA = "v2-head-policy-calibration-bundle/1"
STORED_VS_LIVE = "stored_vs_live_global_risk_max_abs"
LIVE_VS_LIVE = "live_vs_live_shuffled_global_risk_max_abs"


class HeadPolicyCalibrationError(RuntimeError):
    """The requested calibration cannot bind one frozen Head to one measured population."""


def _required_columns(frame: pd.DataFrame, *, bundle: Path) -> None:
    required = {
        "protein_id", "sequence", "sequence_md5", "head_global_risk", "head_score_json",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HeadPolicyCalibrationError(
            f"{bundle}/complete_endpoints.parquet is missing {missing}; a Head floor cannot be "
            "calibrated without the exact sequence and the score the run used"
        )


def endpoint_population(
    bundles: Iterable[Path], *, max_sequences_per_protein: int = 0,
) -> list[dict[str, Any]]:
    """Return one deterministic observation per ``(bundle, protein, sequence_md5)``.

    Duplicated endpoints inside one matched run are one Head observation, not extra calibration
    power.  The same sequence in two runs remains two stored-versus-live repeat observations:
    production reduction drift can make their stored scores differ slightly, and refusing exactly
    that case would make the repeatability calibration unable to measure what it was built for.
    """
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in bundles:
        bundle = Path(raw)
        table = bundle / "complete_endpoints.parquet"
        manifest_path = bundle / "run_manifest.json"
        if not table.exists() or not manifest_path.exists():
            raise HeadPolicyCalibrationError(
                f"{bundle} must contain complete_endpoints.parquet and run_manifest.json"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        content = dict(manifest.get("content_identities") or {})
        frame = pd.read_parquet(table)
        _required_columns(frame, bundle=bundle)
        for row in frame.to_dict("records"):
            protein_id = str(row["protein_id"])
            sequence = str(row["sequence"])
            digest = str(row["sequence_md5"])
            if sequence_md5(sequence) != digest:
                raise HeadPolicyCalibrationError(
                    f"{bundle}: sequence_md5 {digest!r} does not key the stored sequence"
                )
            stored = float(row["head_global_risk"])
            key = (str(bundle.resolve()), protein_id, digest)
            record = {
                "protein_id": protein_id,
                "sequence": sequence,
                "sequence_md5": digest,
                "stored_head_global_risk": stored,
                "head_score_json": row["head_score_json"],
                "bundle": str(bundle),
                "manifest_head_config_hash": content.get("head_config"),
                "manifest_head_checkpoint_digest": content.get("head_checkpoint"),
            }
            previous = grouped.get(key)
            if previous is not None:
                if (previous["sequence"] != sequence
                        or previous["stored_head_global_risk"] != stored):
                    raise HeadPolicyCalibrationError(
                        f"{protein_id}:{digest} is duplicated with inconsistent sequence/risk"
                    )
                continue
            grouped[key] = record

    if isinstance(max_sequences_per_protein, bool) or max_sequences_per_protein < 0:
        raise HeadPolicyCalibrationError("max_sequences_per_protein must be an integer >= 0")
    selected: list[dict[str, Any]] = []
    by_protein: dict[str, list[dict[str, Any]]] = {}
    for record in grouped.values():
        by_protein.setdefault(record["protein_id"], []).append(record)
    for protein_id in sorted(by_protein):
        ordered = sorted(by_protein[protein_id], key=lambda row: row["sequence_md5"])
        selected.extend(ordered if max_sequences_per_protein == 0
                        else ordered[:max_sequences_per_protein])
    return selected


def _score_all(oracle: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    """One pass over the whole population, keyed by ``sequence_md5``."""
    results = oracle.score([
        OracleRequest(protein_id=row["protein_id"], sequence=row["sequence"],
                      sequence_md5=row["sequence_md5"], sequence_length=len(row["sequence"]))
        for row in records])
    scored = {str(result.sequence_md5): float(result.global_risk) for result in results}
    if len(scored) != len(records):
        raise HeadPolicyCalibrationError(
            f"the Head returned {len(scored)} distinct results for {len(records)} requests")
    return scored


def score_live_vs_live(
    records: list[dict[str, Any]], *, head_oracle: Any, repeat_oracle: Any, shuffle_seed: int,
) -> list[dict[str, Any]]:
    """Score the same bytes twice with the SAME checkpoint and pair the two passes.

    The second pass is shuffled AND handed a differently batched build of the same checkpoint.
    Either alone can leave every batch the Head forms identical, and a repeat that forms the same
    batches returns the same numbers whether or not the instrument drifts.
    """
    first = _score_all(head_oracle, list(records))
    shuffled = list(records)
    random.Random(shuffle_seed).shuffle(shuffled)
    second = _score_all(repeat_oracle if repeat_oracle is not None else head_oracle, shuffled)
    missing = sorted(set(first) - set(second))
    if missing:
        raise HeadPolicyCalibrationError(
            f"the second pass did not return {len(missing)} sequence(s)")
    rows = [{
        "protein_id": record["protein_id"],
        "sequence_md5": record["sequence_md5"],
        "first_pass_global_risk": first[record["sequence_md5"]],
        "second_pass_global_risk": second[record["sequence_md5"]],
        "abs_repeat_drift": abs(second[record["sequence_md5"]] - first[record["sequence_md5"]]),
        "source_bundle": record["bundle"],
    } for record in records]
    rows.sort(key=lambda row: (row["protein_id"], row["sequence_md5"]))
    return rows


def score_repeatability(
    records: list[dict[str, Any]], *, head_oracle: Any, min_observations_per_protein: int,
    repeat_oracle: Any = None, shuffle_seed: int | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    """Return auditable paired rows and the evaluator they were measured on.

    ``shuffle_seed`` selects the live-vs-live mode: the stored scores are the bytes' provenance,
    not the comparator, so the stored-instrument gate below does not apply and is not run.
    """
    if min_observations_per_protein < 1:
        raise HeadPolicyCalibrationError("min_observations_per_protein must be >= 1")
    evaluator = head_oracle.evaluator_identity()
    by_protein: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_protein.setdefault(record["protein_id"], []).append(record)
    short = {protein: len(rows) for protein, rows in by_protein.items()
             if len(rows) < min_observations_per_protein}
    if short:
        raise HeadPolicyCalibrationError(
            f"too few unique stored sequences for {short}; need at least "
            f"{min_observations_per_protein} per protein"
        )
    if shuffle_seed is not None:
        return score_live_vs_live(
            records, head_oracle=head_oracle, repeat_oracle=repeat_oracle,
            shuffle_seed=int(shuffle_seed)), evaluator

    rows: list[dict[str, Any]] = []
    for protein_id in sorted(by_protein):
        stored_rows = by_protein[protein_id]
        for stored in stored_rows:
            for field, observed in (
                ("manifest_head_config_hash", evaluator.head_config_hash),
                ("manifest_head_checkpoint_digest", evaluator.head_checkpoint_digest),
            ):
                declared = stored.get(field)
                if declared != observed:
                    raise HeadPolicyCalibrationError(
                        f"{stored['bundle']} binds {field}={declared!r}, but the loaded Head reports "
                        f"{observed!r}; repeatability must be measured on the same instrument"
                    )
            score_payload = json.loads(stored["head_score_json"])
            for field, expected in (("allele", evaluator.allele),
                                    ("score_scale", evaluator.score_scale)):
                if score_payload.get(field) != expected:
                    raise HeadPolicyCalibrationError(
                        f"{protein_id}:{stored['sequence_md5']} stored {field}="
                        f"{score_payload.get(field)!r}, live Head declares {expected!r}"
                    )

        live_results = head_oracle.score([
            OracleRequest(
                protein_id=protein_id, sequence=row["sequence"],
                sequence_md5=row["sequence_md5"], sequence_length=len(row["sequence"]),
            ) for row in stored_rows
        ])
        live_by_digest = {result.sequence_md5: result for result in live_results}
        if len(live_by_digest) != len(stored_rows):
            raise HeadPolicyCalibrationError(
                f"Head returned {len(live_by_digest)} unique results for {len(stored_rows)} "
                f"stored sequences of {protein_id}"
            )
        for stored in stored_rows:
            live = live_by_digest.get(stored["sequence_md5"])
            if live is None:
                raise HeadPolicyCalibrationError(
                    f"Head returned no result for {protein_id}:{stored['sequence_md5']}"
                )
            live_risk = float(live.global_risk)
            drift = abs(live_risk - stored["stored_head_global_risk"])
            rows.append({
                "protein_id": protein_id,
                "sequence_md5": stored["sequence_md5"],
                "stored_head_global_risk": stored["stored_head_global_risk"],
                "live_head_global_risk": live_risk,
                "abs_repeat_drift": drift,
                "source_bundle": stored["bundle"],
            })
    rows.sort(key=lambda row: (row["protein_id"], row["sequence_md5"]))
    return rows, evaluator


def _scalar(
    value: float, *, unit: str, source_kind: str, source_id: str,
    artifact: PolicyCalibrationArtifact,
) -> dict[str, Any]:
    source_ref = policy_calibration_source_ref(
        value=value, unit=unit, source_kind=source_kind, source_id=source_id,
        artifact=artifact,
    )
    return PolicyCalibratedScalar(
        value=value, unit=unit, source_kind=source_kind, source_id=source_id,
        source_ref=source_ref, artifact=artifact,
    ).canonical_payload()


DEPTH0_RULE_BY_VERSION = {"v1": "cumulative_safety_reference", "v2": "best_admissible_depth0"}


def _depth0_rule(spec: dict[str, Any], spec_path: Path) -> str:
    """The D0 incumbent rule this spec declares, cross-checked against its policy version.

    ``HeadDirectedCappedPolicy`` binds the two together in both directions, so the pairing is not
    a preference: a v2 spec whose artifact carries the v1 rule raises at runtime.
    """
    version = str(spec.get("policy_version"))
    expected = DEPTH0_RULE_BY_VERSION[version]
    declared = (spec.get("artifact_contract") or {}).get("d0_gate_kind")
    if declared is not None and str(declared) != expected:
        raise HeadPolicyCalibrationError(
            f"{spec_path} declares policy_version={version!r} and d0_gate_kind={declared!r}, but "
            f"the runtime reserves {version!r} for {expected!r}; the artifact cannot bind both")
    return expected


def build_calibration_bundle(
    rows: list[dict[str, Any]], *, evaluator: Any, policy_spec: Path,
    max_counterfactual_head_calls_per_cycle: int, method: str = STORED_VS_LIVE,
    sequence_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the measured difference floor and the frozen 5% cap into one config block."""
    if not rows:
        raise HeadPolicyCalibrationError("no repeat-scoring rows were measured")
    if (isinstance(max_counterfactual_head_calls_per_cycle, bool)
            or max_counterfactual_head_calls_per_cycle < 1):
        raise HeadPolicyCalibrationError(
            "max_counterfactual_head_calls_per_cycle must be a positive integer"
        )
    spec_path = Path(policy_spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    # v1 AND v2: the materializer requires `policy_spec_sha256` to equal the digest of the spec the
    # CELL supplies, and every executed V2 campaign supplies the v2 spec.  A producer that admits
    # only v1 cannot produce the artifact production consumes -- it refuses the real one and accepts
    # only a spec no campaign passes.  The enum stays closed; it is simply the declared set.
    if (spec.get("policy_id") != HEAD_DIRECTED_CAPPED_POLICY_ID
            or spec.get("policy_version") not in ("v1", "v2")):
        raise HeadPolicyCalibrationError(
            f"{spec_path} is not a frozen {HEAD_DIRECTED_CAPPED_POLICY_ID!r} v1/v2 spec"
        )
    if method not in (STORED_VS_LIVE, LIVE_VS_LIVE):
        raise HeadPolicyCalibrationError(f"unknown repeatability method {method!r}")
    calibration_data_digest = canonical_digest({"method": method, "rows": rows})
    measurement = PolicyCalibrationArtifact(
        schema_version=V2_POLICY_CALIBRATION_SCHEMA_VERSION,
        measurement_kind="frozen_head_repeatability",
        allele=evaluator.allele, score_scale=evaluator.score_scale,
        n_observations=len(rows), calibration_data_digest=calibration_data_digest,
    )
    max_drift = max(float(row["abs_repeat_drift"]) for row in rows)
    difference_bound = 2.0 * max_drift

    policy_spec_sha256 = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    policy_spec_digest = canonical_digest({
        "file_sha256": policy_spec_sha256,
        "policy_id": spec["policy_id"], "policy_version": spec["policy_version"],
    })
    cap_artifact = PolicyCalibrationArtifact(
        schema_version=V2_POLICY_CALIBRATION_SCHEMA_VERSION,
        measurement_kind="frozen_declared_fraction",
        allele=evaluator.allele, score_scale=evaluator.score_scale,
        n_observations=1, calibration_data_digest=policy_spec_digest,
    )
    head_directed = {
        "write_cap_editable_fraction": _scalar(
            0.05, unit="editable_fraction", source_kind="runbook_frozen",
            source_id="v2f5a-write-cap-0.05-v1", artifact=cap_artifact),
        "donor_improvement_epsilon": _scalar(
            difference_bound, unit=evaluator.score_scale,
            source_kind="measured_calibration",
            source_id="v2f5a-frozen-head-repeatability-difference-bound-v1",
            artifact=measurement),
        "local_contribution_tolerance": _scalar(
            difference_bound, unit=evaluator.score_scale,
            source_kind="measured_calibration",
            source_id="v2f5a-local-contribution-repeatability-difference-bound-v1",
            artifact=measurement),
        "band_center_rule": "midpoint_tie_low",
        # READ from the spec being bound, not restated as a constant.  `HeadDirectedCappedPolicy`
        # pairs the two: `best_admissible_depth0` requires policy_version v2, and v2 is RESERVED
        # for it, so a v2 spec carrying the v1 default `cumulative_safety_reference` fails at
        # runtime -- once per cell, after the GPU is allocated.  The v2 spec names the value
        # itself in `artifact_contract.d0_gate_kind`; hardcoding it here is how the producer and
        # the spec drift apart.
        "lineage_incumbent_depth0_rule": _depth0_rule(spec, spec_path),
        "lineage_incumbent_update_law": "strict_improvement_by_epsilon",
        "write_candidate_window_rule": WRITE_WINDOW_RULE,
        "reopen_count_law": REOPEN_COUNT_LAW,
        "reopen_priority_law": REOPEN_PRIORITY_LAW,
        "control_policy_id": SOURCE_GEOMETRY_CONTROL_POLICY_ID,
        "control_policy_version": "v1",
        "max_counterfactual_head_calls_per_cycle":
            int(max_counterfactual_head_calls_per_cycle),
    }
    payload = {
        "schema_version": CALIBRATION_BUNDLE_SCHEMA,
        "method": method,
        "head": evaluator.canonical_payload(),
        "n_observations": len(rows),
        "calibration_data_digest": calibration_data_digest,
        "max_abs_repeat_drift": max_drift,
        "difference_noise_bound": difference_bound,
        "policy_spec": str(spec_path),
        "policy_spec_sha256": policy_spec_sha256,
        "head_directed": head_directed,
    }
    if sequence_source is not None:
        # The bytes came from runs of a DIFFERENT instrument.  That is the point of this mode, and
        # recording whose runs they were is what keeps it auditable rather than merely permitted.
        payload["sequence_source"] = sequence_source
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", action="append", default=None, type=Path,
                        help="existing V2 bundle directory scored by THIS Head (repeatable); "
                             "stored-vs-live mode")
    parser.add_argument("--sequence-bundle", action="append", default=None, type=Path,
                        help="V2 bundle directory used only as a byte population (repeatable); "
                             "live-vs-live mode, for a Head no bundle was ever scored by")
    parser.add_argument("--repeat-shuffle-seed", type=int, default=20260826,
                        help="live-vs-live only: the declared order of the second pass")
    parser.add_argument("--repeat-window-batch-size", type=int, default=None,
                        help="live-vs-live only: the second pass's window batch size; defaults to "
                             "half the first pass's, because a repeat that forms the same batches "
                             "measures nothing")
    parser.add_argument("--policy-spec", required=True, type=Path)
    parser.add_argument("--out-rows", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--min-observations-per-protein", type=int, default=64)
    parser.add_argument("--max-sequences-per-protein", type=int, default=0,
                        help="0 means all unique sequences; selection is by sequence_md5")
    parser.add_argument("--max-counterfactual-head-calls-per-cycle", type=int, required=True)
    head = parser.add_argument_group("frozen Head")
    head.add_argument("--head-checkpoint", type=Path, required=True)
    head.add_argument("--head-config", type=Path, required=True)
    head.add_argument("--head-variant-id", default=None)
    head.add_argument("--head-allele-idx", type=int, default=0)
    head.add_argument("--head-window-batch-size", type=int, default=64)
    head.add_argument("--allele", required=True)
    head.add_argument("--score-scale", default="raw_logit")
    head.add_argument("--window-k-min", type=int, required=True)
    head.add_argument("--window-k-max", type=int, required=True)
    head.add_argument("--device", default="cuda")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import copy

    from scripts.analysis.replay_v2_head_directed_policy import _load_head

    if bool(args.bundle) == bool(args.sequence_bundle):
        raise HeadPolicyCalibrationError(
            "pass exactly one of --bundle (stored-vs-live, this Head produced the bundle) or "
            "--sequence-bundle (live-vs-live, the bundle is only a byte population)")
    live = bool(args.sequence_bundle)
    bundles = args.sequence_bundle if live else args.bundle
    records = endpoint_population(
        bundles, max_sequences_per_protein=int(args.max_sequences_per_protein))

    head_oracle = _load_head(args)
    repeat_oracle = None
    sequence_source = None
    if live:
        repeat_args = copy.copy(args)
        repeat_args.head_window_batch_size = int(
            args.repeat_window_batch_size or max(1, int(args.head_window_batch_size) // 2))
        if repeat_args.head_window_batch_size == int(args.head_window_batch_size):
            raise HeadPolicyCalibrationError(
                "the repeat pass must use a different window batch size; an identically batched "
                "repeat returns identical numbers whether or not the instrument drifts")
        repeat_oracle = _load_head(repeat_args)
        sequence_source = {
            "bundles": sorted(str(Path(path).resolve()) for path in bundles),
            "stored_head_config_hash": sorted(
                {str(record["manifest_head_config_hash"]) for record in records}),
            "stored_head_checkpoint_digest": sorted(
                {str(record["manifest_head_checkpoint_digest"]) for record in records}),
            "stored_scores_used": False,
            "repeat_shuffle_seed": int(args.repeat_shuffle_seed),
            "window_batch_sizes": [int(args.head_window_batch_size),
                                   repeat_args.head_window_batch_size],
        }
    rows, evaluator = score_repeatability(
        records, head_oracle=head_oracle, repeat_oracle=repeat_oracle,
        shuffle_seed=int(args.repeat_shuffle_seed) if live else None,
        min_observations_per_protein=int(args.min_observations_per_protein))
    payload = build_calibration_bundle(
        rows, evaluator=evaluator, policy_spec=args.policy_spec,
        max_counterfactual_head_calls_per_cycle=int(
            args.max_counterfactual_head_calls_per_cycle),
        method=LIVE_VS_LIVE if live else STORED_VS_LIVE,
        sequence_source=sequence_source,
    )
    args.out_rows.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out_rows, index=False)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "out_rows": str(args.out_rows), "out_json": str(args.out_json),
        "method": payload["method"],
        "n_observations": payload["n_observations"],
        "max_abs_repeat_drift": payload["max_abs_repeat_drift"],
        "difference_noise_bound": payload["difference_noise_bound"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
