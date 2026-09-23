#!/usr/bin/env python
"""Measure empirical schedule and maturity bands for a declared sampler and cohort."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import re
import statistics
from typing import Any, Mapping, Sequence

from inverse_folding.reference_flow.fusion.v1_seeds import assert_no_seed_collisions, derive_seed

# FROZEN diagnostic inputs (the scientific basis; do NOT tune these to move the grid).
PROTEINS = ("2O4T_A", "5YAA_B", "3O1Q_C", "7V2T_A")  # length-stratified: 90/150/210/280 aa
RHOS = (0.20, 0.30, 0.40, 0.50, 0.70, 0.85)
SEED_BASE = 700_000
B = 8   # root-prefix attempts per (protein, rho)
K = 4   # forked continuations per sampled root for Head-free diversity

#: Seed schema for this scan. Stamped into the calibration artifact so a band can never be consumed
#: by a run whose seeds were drawn under a different law.
SCAN_SEED_SCHEMA = "rho-maturity-scan-2"
_CODE_REVISION_RE = re.compile(r"[0-9a-f]{7,64}")


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_digest_payload(value: Any, *, name: str) -> dict[str, Any]:
    import numpy as np

    if value is None:
        raise ValueError(f"prepared backbone has no {name!r} tensor")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def prepared_backbone_digest(prepared: Any) -> str:
    """Digest the actual featurized coordinate tensor consumed by the denoiser."""
    batch = getattr(prepared, "batch", None)
    if not isinstance(batch, Mapping):
        raise ValueError("prepared backbone must expose a mapping-valued batch")
    return _canonical_digest({
        "schema": "prepared-backbone-coordinates/1",
        "sequence_length": int(prepared.sequence_length),
        "coords": _array_digest_payload(batch.get("coords"), name="coords"),
    })


@dataclasses.dataclass(frozen=True)
class ScanConstraintBindings:
    constraint_manifest_digest: str
    constraint_stratum_id: str
    fixed_tokens_by_protein: dict[str, dict[int, int]]


def classify_constraint_stratum(anchored: Sequence[bool]) -> str:
    """The constraint class of a cohort, from REALIZED fixed-token cardinalities.

    Lives here, in the module that measures the bands, and is imported by the runtime rather than
    restated: a second copy would let the class a band was calibrated under and the class a run
    checks itself against drift apart, which is exactly the disagreement the check exists to catch.
    """
    flags = list(anchored)
    if flags and all(flags):
        return "anchored"
    if any(flags):
        return "mixed"
    return "unconstrained"


def build_constraint_bindings(
    *, protein_ids: Sequence[str], rows_by_protein: Mapping[str, Mapping[str, Any]],
    alphabet: Any, constraint_manifest: str | None,
) -> ScanConstraintBindings:
    """Load, validate, and tokenize the exact hard anchors used by the scan.

    Constraint class is derived from realized fixed-token cardinalities. A caller-supplied stratum
    label can no longer describe an unconstrained scan as anchored.
    """
    ordered_ids = tuple(str(pid) for pid in protein_ids)
    fixed: dict[str, dict[int, int]] = {}
    if constraint_manifest is None:
        for protein_id in ordered_ids:
            fixed[protein_id] = {}
        return ScanConstraintBindings(
            constraint_manifest_digest=_canonical_digest({
                "schema": "scan-constraint-binding/1", "constraint_manifest": None,
            }),
            constraint_stratum_id="unconstrained",
            fixed_tokens_by_protein=fixed,
        )

    from inverse_folding.reference_flow.constraints import load_constraint_manifest

    manifest = load_constraint_manifest(constraint_manifest)
    for protein_id in ordered_ids:
        if not manifest.has_protein(protein_id):
            raise ValueError(
                f"constraint manifest has no explicit entry for scanned protein {protein_id!r}"
            )
        row = rows_by_protein[protein_id]
        sequence = str(row["sequence"]).upper()
        constraint = manifest.constraint_for_protein(protein_id)
        constraint.validate_against_sequence(sequence)
        fixed[protein_id] = {
            int(anchor.index_0b): int(alphabet.get_idx(str(anchor.expected_aa)))
            for anchor in constraint.hard_anchors
        }

    stratum = classify_constraint_stratum(
        [bool(fixed[protein_id]) for protein_id in ordered_ids])
    return ScanConstraintBindings(
        constraint_manifest_digest=str(manifest.manifest_hash),
        constraint_stratum_id=stratum,
        fixed_tokens_by_protein=fixed,
    )


def scan_seed(*, protein_id: str, cell_id: str, attempt_index: int, stream: str = "root") -> int:
    """Process-independent attempt seed.

    Replaces ``SEED_BASE + hash((pid, rid, i)) % 100_000``. Python's ``hash()`` of a str is salted
    per process unless PYTHONHASHSEED is pinned, so the previous construction silently drew a
    DIFFERENT scan on every run while reporting the same seed basis -- and the modulo could alias
    two cells onto one stream. ``derive_seed`` is the audited, type-tagged, length-delimited
    project primitive.
    """
    return derive_seed(SCAN_SEED_SCHEMA, str(protein_id), str(cell_id), str(stream),
                       int(attempt_index))


def quantiles(values: Sequence[float], levels: Sequence[float]) -> tuple[float, ...]:
    """Linear-interpolated quantiles over a sorted copy. Pure; no numpy needed at this size."""
    if not values:
        raise ValueError("cannot take quantiles of an empty sample")
    ordered = sorted(float(v) for v in values)
    out = []
    for level in levels:
        if not 0.0 < float(level) < 1.0:
            raise ValueError(f"quantile level must be in (0, 1), got {level}")
        position = float(level) * (len(ordered) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            out.append(ordered[low])
        else:
            out.append(ordered[low] + (ordered[high] - ordered[low]) * (position - low))
    return tuple(out)


def summarize_step_cell(
    attempts: Sequence[Mapping[str, Any]], *, levels: Sequence[float],
) -> dict[str, Any]:
    """Reduce the raw per-attempt rows of one (step, stratum) cell to band inputs.

    Only SUCCESSFUL captures contribute to the quantiles; failures are counted, never dropped, so a
    cell whose captures mostly failed cannot masquerade as a well-sampled one.
    """
    captured = [row for row in attempts if row.get("captured")]
    n_attempts = len(attempts)
    if not captured:
        return {"n_attempts": n_attempts, "n_captured": 0, "rho_quantiles": (),
                "unresolved_quantiles": (), "n_editable_min": None, "n_editable_max": None}
    rho = quantiles([row["rho_actual"] for row in captured], levels)
    # Mass falls as maturity rises, so the unresolved vector is taken at the MIRRORED levels to keep
    # both vectors indexed by the same "maturity increases with level" convention.
    unresolved = quantiles(
        [row["n_unresolved_editable"] for row in captured], [1.0 - level for level in levels]
    )
    editables = [row["n_editable"] for row in captured]
    return {
        "n_attempts": n_attempts,
        "n_captured": len(captured),
        "rho_quantiles": rho,
        "unresolved_quantiles": tuple(int(round(v)) for v in unresolved),
        "n_editable_min": int(min(editables)),
        "n_editable_max": int(max(editables)),
    }


def _args():
    p = argparse.ArgumentParser(description="Frozen outcome-independent maturity scan (runbook §6).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rf-config", required=True)
    p.add_argument("--test-set", required=True)
    p.add_argument("--pdb-root", required=True)
    p.add_argument("--out-parquet", required=True)
    p.add_argument("--proteins", nargs="+", default=list(PROTEINS))
    p.add_argument("--mode", choices=("rho", "step"), default="rho",
                   help="rho: V1-A crossing scan. step: V2 step-indexed B(r) calibration.")
    p.add_argument("--rhos", nargs="+", type=float, default=list(RHOS),
                   help="rho mode only: the maturity targets to cross.")
    p.add_argument("--steps", nargs="+", type=int,
                   help="step mode only: explicit sampler steps to capture at.")
    p.add_argument("--stratum-key", help="step mode only: cohort stratum the absolute unresolved "
                                         "mass is valid within. Required; no default.")
    p.add_argument("--quantile-levels", nargs="+", type=float,
                   help="step mode only: the quantile grid to measure. Required; no default.")
    p.add_argument("--min-captures-per-cell", type=int,
                   help="step mode only: a cell below this exits non-zero and writes NO "
                        "calibration artifact. Required; no default.")
    p.add_argument("--attempts-per-cell", type=int, default=B)
    p.add_argument("--calibration-id", help="step mode only: identity stamped into the artifact.")
    p.add_argument("--calibration-json", help="step mode only: where to write the B(r) artifact.")
    p.add_argument("--constraint-manifest", help="step mode only: optional hard-anchor manifest; "
                                                  "anchors are applied as sampler fixed tokens.")
    p.add_argument(
        "--code-revision", help="step mode only: explicit git revision of executed code."
    )
    args = p.parse_args()
    if args.mode == "step":
        missing = [name for name in ("steps", "stratum_key", "quantile_levels",
                                     "min_captures_per_cell", "calibration_id", "calibration_json",
                                     "code_revision")
                   if getattr(args, name) in (None, [])]
        if missing:
            p.error("step mode requires " + ", ".join("--" + n.replace("_", "-") for n in missing)
                    + "; these are calibration choices with no default")
        revision = str(args.code_revision).strip().lower()
        if (
            revision.startswith(("unset", "unknown", "placeholder"))
            or not _CODE_REVISION_RE.fullmatch(revision)
        ):
            p.error("step mode code revision must be an explicit non-placeholder 7-64 character "
                    "lowercase hexadecimal revision")
        args.code_revision = revision
    return args


def _hamming_editable(seqs, editable):
    if len(seqs) < 2:
        return 0.0
    return statistics.mean(
        sum(1 for i in editable if a[i] != b[i]) / max(1, len(editable))
        for a, b in itertools.combinations(seqs, 2)
    )


def main():
    args = _args()
    import numpy as np
    import pandas as pd

    from inverse_folding.reference_flow.config import load_reference_flow_config
    from inverse_folding.reference_flow.fusion.v1_records import ConditioningDigest, make_root_id
    from inverse_folding.reference_flow.runtime import (
        build_dplm_denoiser_context, load_if_task, make_dplm_denoiser, prepare_backbone,
    )
    from inverse_folding.reference_flow.sampler import (
        ContinuationRequest, InvalidPreterminalRootError, MaturityNotReachedError,
        PositionDependentDFMSampler,
    )
    from scripts.rf_fusion_model_factory import (
        _canonical_id_to_aa, assert_null_amplification, decode_tokens_to_aa, null_h_values,
        payload_from_checkpoint, resume_from_payload,
    )

    task = load_if_task(args.checkpoint, device="cuda")
    sampler = PositionDependentDFMSampler(mask_token_id=task.alphabet.mask_idx, vocab_size=len(task.alphabet))
    rf = load_reference_flow_config(args.rf_config)
    assert_null_amplification(rf)  # this diagnostic is meaningful only under the null kernel
    id2aa = _canonical_id_to_aa(task.alphabet)
    aa_ids = frozenset(id2aa)
    mask_id = int(task.alphabet.mask_idx)
    cond = ConditioningDigest("dc", "tk", "bb", "cm", "ec", "unconstrained", False, False)
    rows = pd.read_parquet(args.test_set).set_index("protein_id")

    def cfg(seed):
        return dataclasses.replace(rf, sampler=dataclasses.replace(rf.sampler, seed=int(seed)))

    if args.mode == "step":
        return _run_step_mode(
            args=args, sampler=sampler, rf=rf, task=task, aa_ids=aa_ids, mask_id=mask_id,
            rows=rows, cfg=cfg, np=np, pd=pd, ConditioningDigest=ConditioningDigest,
            prepare_backbone=prepare_backbone, make_dplm_denoiser=make_dplm_denoiser,
            build_dplm_denoiser_context=build_dplm_denoiser_context,
            ContinuationRequest=ContinuationRequest,
            MaturityNotReachedError=MaturityNotReachedError,
            InvalidPreterminalRootError=InvalidPreterminalRootError,
            payload_from_checkpoint=payload_from_checkpoint, make_root_id=make_root_id,
            null_h_values=null_h_values,
        )

    records = []
    fork_failures: list[dict] = []
    hdr = f"{'protein':9}{'L':>5}{'rho':>6}{'cross/B':>9}{'|U|':>5}{'rho_act':>8}{'step':>7}{'unres':>7}{'unres%':>8}{'fork_div':>9}"
    print(hdr)
    for pid in args.proteins:
        row = dict(rows.loc[pid]); row["protein_id"] = pid
        prepared = prepare_backbone(task=task, entry=row, pdb_root=args.pdb_root, device="cuda")
        denoiser = make_dplm_denoiser(build_dplm_denoiser_context(task=task, prepared=prepared, use_draft_seq_override=False))
        L = prepared.sequence_length
        for rho in args.rhos:
            rid = f"rho{rho:.3f}"; roots = []; crossed = 0
            for i in range(B):
                seed = scan_seed(protein_id=pid, cell_id=rid, attempt_index=i)
                try:
                    out = sampler.sample(sequence_length=L, h_values=null_h_values(L), denoiser=denoiser,
                        config=cfg(seed), controller=None, struct=None,
                        continuation=ContinuationRequest(at_rho_edit=rho, early_stop=True), residue_token_ids=aa_ids)
                except MaturityNotReachedError:
                    continue
                crossed += 1
                roots.append((seed, payload_from_checkpoint(out.continuation_checkpoint,
                    root_id=make_root_id(pid, "preterminal", rid, i), protein_id=pid, arm_id="preterminal",
                    rho_id=rid, mask_token_id=mask_id, conditioning=cond)))
            uniq = {}
            for seed, p in roots:
                uniq.setdefault(p.root_equivalence_hash, (seed, p))
            divs = []
            for seed, p in list(uniq.values())[:2]:  # Head-free descendant branching
                seqs = []
                for k in range(K):
                    try:
                        o = sampler.sample(sequence_length=L, h_values=null_h_values(L), denoiser=denoiser,
                            config=cfg(seed + 1000 + k), controller=None, struct=None,
                            continuation_resume=resume_from_payload(p, fork_seed=seed + 1000 + k), residue_token_ids=aa_ids)
                        seqs.append(decode_tokens_to_aa(o.tokens, id2aa))
                    except (MaturityNotReachedError, InvalidPreterminalRootError) as exc:
                        # A lost fork is EVIDENCE, not noise: a bare ``except Exception`` here hid
                        # real capture failures behind a diversity mean computed on whatever
                        # survived (PLAN section 3.5).
                        fork_failures.append({"protein_id": pid, "cell_id": rid, "fork_index": k,
                                              "failure_kind": type(exc).__name__})
                if len(seqs) >= 2:
                    divs.append(_hamming_editable(seqs, list(p.editable_positions)))
            if not roots:
                records.append(dict(protein_id=pid, length=L, rho=rho, crossed=0, B=B, n_unique=0,
                                    rho_actual=None, step=None, unresolved=None, unresolved_frac=None, fork_diversity=None))
                print(f"{pid:9}{L:>5}{rho:>6.2f}{'0/'+str(B):>9}{'--':>5}{'no-cross':>8}")
                continue
            nedit = len(roots[0][1].editable_positions)
            ra = float(np.mean([(len(p.editable_positions) - p.n_unresolved_editable) / max(1, len(p.editable_positions)) for _, p in roots]))
            st = float(np.mean([p.step for _, p in roots]))
            un = float(np.mean([p.n_unresolved_editable for _, p in roots]))
            fd = statistics.mean(divs) if divs else float("nan")
            records.append(dict(protein_id=pid, length=L, rho=rho, crossed=crossed, B=B, n_unique=len(uniq),
                                rho_actual=ra, step=st, unresolved=un, unresolved_frac=un / max(1, nedit), fork_diversity=fd))
            print(f"{pid:9}{L:>5}{rho:>6.2f}{str(crossed)+'/'+str(B):>9}{len(uniq):>5}{ra:>8.3f}{st:>7.1f}{un:>7.1f}{100*un/max(1,nedit):>7.1f}%{fd:>9.3f}")

    import os
    os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
    pd.DataFrame(records).to_parquet(args.out_parquet, index=False)
    if fork_failures:
        pd.DataFrame(fork_failures).to_parquet(
            args.out_parquet.replace(".parquet", "_fork_failures.parquet"), index=False)
    print(f"[rho_maturity_scan] wrote {len(records)} rows, {len(fork_failures)} fork failures "
          f"-> {args.out_parquet}")


def _run_step_mode(*, args, sampler, rf, task, aa_ids, mask_id, rows, cfg, np, pd,
                   ConditioningDigest,
                   prepare_backbone, make_dplm_denoiser, build_dplm_denoiser_context,
                   ContinuationRequest, MaturityNotReachedError, InvalidPreterminalRootError,
                   payload_from_checkpoint, make_root_id, null_h_values):
    """Capture at EXPLICIT sampler steps and emit the step-indexed band B(r).

    Every attempt writes exactly one raw row, successful or not, so a cell that mostly failed cannot
    be read as a well-sampled one. A cell below ``--min-captures-per-cell`` exits non-zero and
    writes NO calibration artifact: a band nobody may consume is better than a band nobody can
    trust.
    """
    import os
    import sys

    from inverse_folding.reference_flow.fusion_v2.schedule import (
        SCHEDULE_BAND_SCHEMA_VERSION, SCHEDULE_BAND_SCOPE, BandInterval, BandProvenance,
        QuantileLevels, band_table_payload, bind_band_table, make_band,
    )
    from scripts.rf_fusion_model_factory import coordinate_mask_digest, tokenizer_digest

    if float(getattr(rf.sampler.remask, "fraction_scale", 1.0)) != 0.0:
        raise SystemExit(
            "[rho_maturity_scan] step mode requires remask.fraction_scale=0.0: the V1 crossings "
            "were compressed into the last ~16 steps by repeated global remasking, and a band "
            "calibrated on that substrate may not be used to select V2 coordinates (PLAN 2.1)."
        )

    levels = tuple(float(v) for v in args.quantile_levels)
    attempts_per_cell = int(args.attempts_per_cell)
    raw: list[dict] = []
    seeds_seen: dict[str, int] = {}
    rows_by_protein = {
        str(pid): (dict(rows.loc[pid]) | {"protein_id": str(pid)}) for pid in args.proteins
    }
    constraints = build_constraint_bindings(
        protein_ids=args.proteins,
        rows_by_protein=rows_by_protein,
        alphabet=task.alphabet,
        constraint_manifest=args.constraint_manifest,
    )
    tokenizer_content_digest = tokenizer_digest(task.alphabet)
    checkpoint_content_digest = _digest_file(args.checkpoint)
    sampler_config_content_digest = _digest_file(args.rf_config)
    prepared_backbones: dict[str, str] = {}
    prepared_coordinate_masks: dict[str, str] = {}

    for pid in args.proteins:
        row = rows_by_protein[str(pid)]
        prepared = prepare_backbone(task=task, entry=row, pdb_root=args.pdb_root, device="cuda")
        backbone_content_digest = prepared_backbone_digest(prepared)
        coordinate_content_digest = coordinate_mask_digest(prepared)
        prepared_backbones[str(pid)] = backbone_content_digest
        prepared_coordinate_masks[str(pid)] = coordinate_content_digest
        fixed_tokens = constraints.fixed_tokens_by_protein[str(pid)]
        conditioning = ConditioningDigest(
            dplm_checkpoint=checkpoint_content_digest,
            tokenizer=tokenizer_content_digest,
            backbone_row=backbone_content_digest,
            coordinate_mask=coordinate_content_digest,
            entry_config=sampler_config_content_digest,
            fixed_token_policy=(
                f"constraint-{constraints.constraint_manifest_digest}"
                if args.constraint_manifest else
                f"unconstrained-{constraints.constraint_manifest_digest}"
            ),
            controller_enabled=False,
            h_maps_present=False,
        )
        denoiser = make_dplm_denoiser(build_dplm_denoiser_context(
            task=task, prepared=prepared, use_draft_seq_override=False))
        L = prepared.sequence_length
        for step in sorted(set(int(v) for v in args.steps)):
            cell = f"step{step:04d}"
            for i in range(attempts_per_cell):
                seed = scan_seed(protein_id=pid, cell_id=cell, attempt_index=i)
                seeds_seen[f"{pid}:{cell}:{i}"] = seed
                record = {"protein_id": pid, "length": L, "step": step,
                          "stratum_key": args.stratum_key, "attempt_index": i, "seed": seed,
                          "captured": False, "failure_kind": None, "rho_actual": None,
                          "n_editable": None, "n_unresolved_editable": None,
                          "n_fixed": len(fixed_tokens),
                          "constraint_stratum_id": constraints.constraint_stratum_id,
                          "backbone_digest": backbone_content_digest,
                          "coordinate_mask_digest": coordinate_content_digest}
                try:
                    out = sampler.sample(
                        sequence_length=L, h_values=null_h_values(L), denoiser=denoiser,
                        config=cfg(seed), controller=None, struct=None,
                        fixed_tokens=fixed_tokens or None,
                        continuation=ContinuationRequest(at_step=step, early_stop=True),
                        residue_token_ids=aa_ids)
                except (MaturityNotReachedError, InvalidPreterminalRootError) as exc:
                    record["failure_kind"] = type(exc).__name__
                    raw.append(record)
                    continue
                payload = payload_from_checkpoint(
                    out.continuation_checkpoint,
                    root_id=make_root_id(pid, "v2scan", cell, i), protein_id=pid, arm_id="v2scan",
                    rho_id=cell, mask_token_id=mask_id, conditioning=conditioning)
                n_editable = len(payload.editable_positions)
                record.update(captured=True, rho_actual=float(payload.actual_rho_edit),
                              n_editable=n_editable,
                              n_unresolved_editable=int(payload.n_unresolved_editable))
                raw.append(record)

    assert_no_seed_collisions(seeds_seen)
    os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
    pd.DataFrame(raw).to_parquet(args.out_parquet, index=False)
    print(f"[rho_maturity_scan] wrote {len(raw)} raw attempt rows -> {args.out_parquet}")

    thin = []
    bands = []
    for step in sorted(set(int(v) for v in args.steps)):
        cell_rows = [r for r in raw if r["step"] == step]
        summary = summarize_step_cell(cell_rows, levels=levels)
        print(f"  step {step:>4}  captured {summary['n_captured']}/{summary['n_attempts']}"
              f"  rho_q {['%.3f' % v for v in summary['rho_quantiles']]}"
              f"  unresolved_q {list(summary['unresolved_quantiles'])}")
        if summary["n_captured"] < int(args.min_captures_per_cell):
            thin.append((step, summary["n_captured"]))
            continue
        rho_q, un_q = summary["rho_quantiles"], summary["unresolved_quantiles"]
        bands.append(make_band(
            step=step, stratum_key=args.stratum_key, levels=QuantileLevels(levels),
            rho_quantiles=rho_q, unresolved_quantiles=un_q,
            # The accept interval is the measured outer quantile pair. Which pair defines it is a
            # later scientific calibration (PLAN 3.5); this scan declares no tolerance of its own,
            # it only reports what it measured.
            rho_accept=BandInterval(lo=rho_q[0], hi=rho_q[-1], lo_level=levels[0],
                                    hi_level=levels[-1]),
            unresolved_accept=BandInterval(lo=float(min(un_q)), hi=float(max(un_q)),
                                           lo_level=levels[0], hi_level=levels[-1]),
            combination_rule="both_axes", n_attempts=summary["n_attempts"],
            n_captured=summary["n_captured"], n_editable_min=summary["n_editable_min"],
            n_editable_max=summary["n_editable_max"]))

    if thin:
        print(f"[rho_maturity_scan] FAILED: cells below --min-captures-per-cell "
              f"{args.min_captures_per_cell}: {thin}", file=sys.stderr)
        print("[rho_maturity_scan] no calibration artifact written", file=sys.stderr)
        raise SystemExit(2)

    backbone_digest = _canonical_digest({
        "schema": "scan-backbone-cohort/1", "by_protein": prepared_backbones,
    })
    coordinate_digest = _canonical_digest({
        "schema": "scan-coordinate-mask-cohort/1", "by_protein": prepared_coordinate_masks,
    })
    cohort_digest = _canonical_digest({
        "schema": "scan-cohort/1",
        "test_set_digest": _digest_file(args.test_set),
        "protein_ids": [str(pid) for pid in args.proteins],
    })
    provenance = BandProvenance(
        schema_version=SCHEDULE_BAND_SCHEMA_VERSION,
        calibration_scope=SCHEDULE_BAND_SCOPE,
        calibration_id=args.calibration_id,
        calibration_content_digest=_digest_file(args.out_parquet),
        sampler_config_digest=sampler_config_content_digest,
        tokenizer_digest=tokenizer_content_digest,
        backbone_digest=backbone_digest,
        coordinate_mask_policy_digest=coordinate_digest,
        constraint_manifest_digest=constraints.constraint_manifest_digest,
        constraint_stratum_id=constraints.constraint_stratum_id,
        cohort_digest=cohort_digest,
        raw_attempts_digest=_digest_file(args.out_parquet),
        attempted_seed_digest=_canonical_digest({
            "schema": SCAN_SEED_SCHEMA, "attempted_seeds": seeds_seen,
        }),
        seed_schema=SCAN_SEED_SCHEMA,
        code_revision=str(args.code_revision),
        produced_by="scripts/rho_maturity_scan.py --mode step",
        n_steps=int(rf.sampler.n_steps), base_form=str(rf.schedule.base_form),
        amplification_form=str(rf.amplification.form),
        remask_fraction_scale=float(rf.sampler.remask.fraction_scale),
        head_free=True,
        n_attempted_seeds=len(seeds_seen),
        n_failed_captures=sum(1 for r in raw if not r["captured"]),
    )
    table = bind_band_table(provenance=provenance, bands=bands)
    payload = band_table_payload(table)
    os.makedirs(os.path.dirname(args.calibration_json) or ".", exist_ok=True)
    with open(args.calibration_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    print(f"[rho_maturity_scan] wrote {len(bands)} calibrated bands -> {args.calibration_json}")


def _digest_file(path: str) -> str:
    from inverse_folding.reference_flow.fusion.v1_records import content_digest

    return content_digest(path)


if __name__ == "__main__":
    main()
