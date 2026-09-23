"""Run the oracle-assembly contract chain for a resolved Canary cell, on a login node, in seconds.

**Why this exists as a second gate.** `run_rf_fusion_v2.py --dry-run` deliberately loads no model,
so it returns before `build_v2_oracles` is ever entered — and that is where the launch-blocking
faults actually live. Four consecutive cluster attempts died there, each after the GPU allocation
and the checkpoint load had been paid for:

* `content[complete_reference_sequence]` bound to the reference MANIFEST rather than to the cell's
  own `.seq`, so `bind_cumulative_reference` raised `ReferenceRebindAttempt`;
* `alphabet=model.alphabet`, which `PreparedModel` calls `id_to_aa`;
* an anchored cell's runtime-bound `fixed_token_policy` role left out of the shard inputs, so
  `_conditioning` refused with "no content identity".

Every one is decidable without a GPU, from the REAL artifacts, and this checks them:

1. the band loads under the config's declared content digest, and its constraint stratum matches
   the protein's realized fixed-token class;
2. the stratum manifest names this protein, and `lookup_band` — which never interpolates — admits a
   non-empty reopen envelope at this cell's `r_step`;
3. the reference resolves, and the depth-0 safety reference BINDS, which is the step that compares
   the digest recomputed from the reference bytes against the config's own field;
4. the admission policy and the safety gate build;
5. the support policy the config declares is the one the registry builds, and its
   `policy_spec_digest` equals the run's `projection_policy_spec` content role — the value the
   kernel matches every transition against;
6. every one of the eighteen content roles resolves to a digest through the SAME
   `--shard-input` vector the launcher will pass.

**What it does not cover, stated plainly.** The two torch builds are stubbed, so this cannot catch a
fault inside the DPLM stack, the Head or the structure backend. The stub is checked against
`PreparedModel`'s real attribute surface by AST — the attributes `build_v2_oracles` actually reads —
so it cannot pass by defining a name the real class lacks, which is exactly how the previous fakes
agreed with the bug. The reference Head score is a shaped placeholder: it reaches
`bind_cumulative_reference`'s digest checks, which run before any scoring, and nothing here reports
a Head number.

Exit `0` when every cell passes, `1` otherwise. Registered in `doc/SCRIPTS.md`.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__all__ = ["build_parser", "main", "assemble_check", "StubModel", "model_attributes_read"]

_AA = "ACDEFGHIKLMNPQRSTVWY"


class PreflightAssemblyError(RuntimeError):
    """A resolved cell that would fail during oracle assembly."""


class _StubHeadOracle:
    """Holdable, not askable.

    It satisfies the ``hasattr(head_oracle, 'score')`` and ``callable(...)`` checks the policy and
    the counterfactual scorer perform at construction, and refuses if anything actually calls it --
    so an assembly check can prove the policy is constructible without a placeholder ever
    fabricating a Head number that could be mistaken for a measurement.
    """

    def score(self, requests):  # pragma: no cover - refusal path
        raise PreflightAssemblyError(
            "the assembly preflight's stub Head was asked to score; this check proves the policy "
            "can be BUILT model-free and must never produce a Head value"
        )


class StubModel:
    """Only what ``build_v2_oracles`` reads, under ``PreparedModel``'s OWN names.

    Checked against the real class by :func:`model_attributes_read`, so a renamed attribute is a
    failure here rather than a silent agreement between the stub and the bug.
    """

    id_to_aa = {index: residue for index, residue in enumerate(_AA)}
    aa_token_ids = frozenset(range(len(_AA)))
    mask_token_id = len(_AA)
    tokenizer_digest = "0" * 64
    fixed_token_policy = "unconstrained"
    sampler = object()
    rf_config = object()

    def __init__(self, length: int, fixed: dict | None) -> None:
        self._length, self._fixed = int(length), fixed

    def backbone_and_denoiser(self, protein_id):  # noqa: ARG002
        return object(), object()

    def sequence_length(self, protein_id):  # noqa: ARG002
        return self._length

    def fixed_tokens(self, protein_id):  # noqa: ARG002
        return self._fixed or None

    def coordinate_mask_digest(self, protein_id):  # noqa: ARG002
        return "1" * 64

    def null_h_values(self, length):
        return [1.0] * int(length)


def model_attributes_read() -> list[str]:
    """Every ``model.<attr>`` ``build_v2_oracles`` reads, from the SOURCE.

    A hand-maintained list would be a third place to forget the rename.
    """
    from scripts import rf_fusion_v2_oracles as oracles

    tree = ast.parse(inspect.getsource(oracles.build_v2_oracles))
    return sorted({node.attr for node in ast.walk(tree)
                   if isinstance(node, ast.Attribute)
                   and isinstance(node.value, ast.Name) and node.value.id == "model"})


def _assert_stub_surface() -> list[str]:
    from scripts.rf_fusion_model_factory import PreparedModel

    read = model_attributes_read()
    real = set(dir(PreparedModel)) | set(getattr(PreparedModel, "__annotations__", {}))
    absent = [name for name in read if name not in real]
    if absent:
        raise PreflightAssemblyError(
            f"build_v2_oracles reads model.{absent}, which PreparedModel does not carry")
    unstubbed = [name for name in read if name not in set(dir(StubModel))]
    if unstubbed:
        raise PreflightAssemblyError(
            f"this preflight's stub is missing {unstubbed}, so it would not exercise them")
    return read


class _ReferenceHeadScore:
    """A shaped placeholder that reaches the digest checks and reports no Head number."""

    def __init__(self, protein_id, sequence, *, allele, score_scale, k_min, k_max):
        from inverse_folding.reference_flow.fusion.state import sequence_md5

        self.protein_id = protein_id
        self.sequence_md5 = sequence_md5(sequence)
        self.sequence_length = len(sequence)
        self.allele, self.score_scale = allele, score_scale
        self.windows = tuple(
            type("_Window", (), {"start_0b": start, "end_0b": start + k - 1, "k": k, "z": 0.0})()
            for k in {int(k_min), int(k_max)} for start in (0, 1))
        self.residue_hotspot, self.global_risk = None, 0.0


def _shard_inputs_from(args_file: Path):
    """Parse the launcher's own ``SHARD_INPUTS`` vector, so this checks what will actually run."""
    from scripts.rf_fusion_v2_cohort import ShardInputs

    paths, inside = {}, False
    for raw in args_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("SHARD_INPUTS=("):
            inside = True
            continue
        if inside:
            if line == ")":
                break
            name, sep, value = line.partition("=")
            if sep:
                paths[name.strip()] = value.strip()
    if not paths:
        raise PreflightAssemblyError(f"{args_file} declares no SHARD_INPUTS")
    return ShardInputs(**paths)


def _dual_authority(overlay: Any, *, arm: str, incumbent: Any, reference_head_score: Any):
    """The Dual half of the policy, assembled model-free from the signed overlay.

    The check that used to be missing entirely: an operator materialized a three-arm Dual cell,
    passed every preflight, and only found out on the GPU whether the joint policy could be
    constructed at all -- which is exactly the assembly contract this script exists to prove.

    Role B's SCORE here is the same shaped placeholder role A uses, so the authority is proved
    constructible without either Head being loaded or a number being fabricated that could be
    mistaken for a measurement. J(I_0) is derived from it rather than declared, so a calibration
    whose coordinates cannot resolve the incumbent fails here.
    """
    import copy

    from inverse_folding.reference_flow.fusion_v2.dual_policy import DualSupportAuthority
    from inverse_folding.reference_flow.fusion_v2.joint_objective import build_arm_objective

    declared_b = overlay.head_b_runtime.evaluator
    if overlay.calibration.window is None:
        raise PreflightAssemblyError(
            "the Dual calibration declares no window coordinates; the union reopen law compares "
            "per-window residuals and the aggregate risk scale does not transfer to them")
    score_b = copy.copy(reference_head_score)
    score_b.allele = declared_b.allele
    objective = build_arm_objective(overlay.calibration, arm=arm)
    return DualSupportAuthority(
        objective=objective, window_coordinates=overlay.calibration.window,
        evaluator_b=declared_b, counterfactual_scorer_b=_StubCounterfactualScorer(),
        incumbent_score_b=score_b,
        incumbent_joint_value=float(objective.evaluate(
            raw_a=float(incumbent.head_global_risk),
            raw_b=float(score_b.global_risk)).value),
        safety_reference_score_b=score_b, donor_score_b_by_endpoint={},
        max_counterfactual_sequences_per_cycle=int(
            overlay.max_counterfactual_sequences_per_cycle))


class _StubCounterfactualScorer:
    """Callable, and refuses if called -- the counterfactual half of :class:`_StubHeadOracle`."""

    def __call__(self, protein_id, sequences):  # pragma: no cover - refusal path
        raise PreflightAssemblyError(
            "the assembly preflight's stub role B scorer was asked to score; this check proves the "
            "joint policy can be BUILT model-free")


def assemble_check(config_path: Any, *, protein_id: str, args_file: Any = None,
                   dual_overlay: Any = None, dual_arm: str = "") -> dict:
    """Everything ``build_v2_oracles`` does except the two torch builds."""
    from inverse_folding.reference_flow.fusion_v2 import identity as ident
    from inverse_folding.reference_flow.fusion_v2.safety import (
        bind_admission_policy, bind_cumulative_reference, open_lineage_ledger)
    from inverse_folding.reference_flow.fusion_v2.schedule import load_band_table, lookup_band
    from inverse_folding.reference_flow.fusion_v2_runtime.admission import SafetyGate
    from scripts import rf_fusion_v2_oracles as oracles
    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    config_path = Path(config_path)
    args_file = Path(args_file) if args_file else config_path.with_suffix(".args.sh")
    inputs = _shard_inputs_from(args_file)
    config = load_v2_config_file(config_path)
    report: dict[str, Any] = {"cell": config_path.stem, "protein_id": protein_id,
                              "config_digest": config.config_digest()}

    band = load_band_table(inputs.require("schedule_band_calibration"),
                           expected_content_digest=oracles._content_digest(
                               config, "schedule_band_calibration"))
    stratum = oracles.resolve_stratum(inputs.require("protein_stratum_manifest"), protein_id)
    report["band_id"] = band.provenance.calibration_id
    report["band_constraint_stratum"] = band.provenance.constraint_stratum_id
    report["stratum_key"] = stratum

    # The realized constraint class, taken from the manifest the run will actually load rather than
    # from a flag: a label alone cannot stop an anchored protein reading an unconstrained band.
    manifest_path = inputs.paths.get("constraint_manifest")
    fixed = None
    if manifest_path:
        from inverse_folding.reference_flow.constraints import load_constraint_manifest

        manifest = load_constraint_manifest(manifest_path)
        if manifest.has_protein(protein_id):
            fixed = {int(a.index_0b): 0 for a in
                     manifest.constraint_for_protein(protein_id).hard_anchors} or None
    report["n_hard_anchors"] = 0 if not fixed else len(fixed)
    oracles.assert_constraint_class_matches_band(
        band_table=band, fixed_tokens=fixed, protein_id=protein_id)

    sequence, reference_digest = oracles.resolve_reference(
        inputs.require("complete_reference_manifest"), protein_id)
    report["reference_length"] = len(sequence)

    head_identity = ident.HeadEvaluatorIdentity(
        allele=config.head.allele, score_scale=config.head.score_scale,
        window_k_min=config.head.window_k_min, window_k_max=config.head.window_k_max,
        head_config_hash=oracles._content_digest(config, "head_config"),
        head_checkpoint_digest=oracles._content_digest(config, "head_checkpoint"))
    policy = bind_admission_policy(config, head_identity)
    # Hoisted out of the call below so the SAME object serves both the cumulative reference and the
    # support policy, mirroring how the real oracle stack binds it once and passes it twice.
    reference_head_score = _ReferenceHeadScore(
        protein_id, sequence, allele=config.head.allele,
        score_scale=config.head.score_scale,
        k_min=config.head.window_k_min, k_max=config.head.window_k_max)
    cumulative = bind_cumulative_reference(
        lineage_id=f"{protein_id}:fam0", protein_id=protein_id,
        reference_label=config.safety.cumulative_reference_label,
        reference_sequence=sequence, reference_content_digest=reference_digest, policy=policy,
        head_score=reference_head_score)
    SafetyGate(policy=policy, ledger=open_lineage_ledger(cumulative))
    report["reference_id"] = cumulative.binding.reference_id

    # The head-directed policy needs five bindings the earlier version of this check did not supply,
    # so it could never be constructed here and the operator fell back to a real one-prefix GPU run
    # to discover an assembly failure. Four of the five already existed as locals; the fifth is a
    # stub Head that can be held but not asked. Nothing here loads a model.
    context = {}
    if config.projection.head_directed is not None:
        context = {
            "head_oracle": _StubHeadOracle(),
            "incumbent": oracles.bind_lineage_incumbent(
                config=config, cumulative_reference=cumulative, reference_sequence=sequence,
                evaluator=head_identity, lineage_id=f"{protein_id}:fam0"),
            "safety_reference_score": reference_head_score,
            "evaluator": head_identity,
            "window_grid_digest": cumulative.binding.head_binding.window_grid_digest,
        }
        # The policy identity below is derived from a SHAPED PLACEHOLDER Head score, not from the
        # real Head: the incumbent's payload carries head_global_risk and head_score_digest, and
        # both come from _ReferenceHeadScore. The digest proves the policy ASSEMBLES; it is not the
        # digest the run will record.
        report["policy_config_digest_is_placeholder_bound"] = True
        if dual_overlay is not None:
            if dual_arm not in dual_overlay.arms:
                raise PreflightAssemblyError(
                    f"--dual-arm {dual_arm!r} is not in the overlay's declared bundle "
                    f"{list(dual_overlay.arms)}")
            context["dual"] = _dual_authority(
                dual_overlay, arm=dual_arm, incumbent=context["incumbent"],
                reference_head_score=reference_head_score)
            report["dual_arm"] = dual_arm
            report["dual_allele_b"] = dual_overlay.head_b_runtime.evaluator.allele
    report["dual_mode"] = "none" if dual_overlay is None else "dual"

    support = oracles.resolve_support_policy(config, band_table=band, stratum_key=stratum,
                                             **context)
    if dual_overlay is not None and getattr(support, "dual", None) is None:
        raise PreflightAssemblyError(
            "an overlay was supplied but the resolved policy holds no Dual authority; this cell "
            "would run A-only under a Dual signature")
    identity = support.identity()
    declared = config.declared_policy()
    report["policy_config_digest"] = identity.policy_config_digest
    report["policy_matches_declaration"] = bool(
        identity.policy_id == declared.policy_id
        and identity.policy_version == declared.policy_version
        and identity.is_diagnostic_only == declared.is_diagnostic)
    report["policy_spec_digest_matches_conditioning"] = bool(
        identity.policy_spec_digest
        == oracles._content_digest(config, "projection_policy_spec"))

    conditioning = oracles._conditioning(
        config, model=StubModel(len(sequence), fixed), protein_id=protein_id,
        inputs=inputs, band_table=band)
    report["conditioning_digest"] = conditioning.digest()
    report["fixed_token_policy"] = oracles.fixed_token_policy_label(inputs)

    r_step = int(config.schedule.points[0].r_step)
    cell = lookup_band(band, step=r_step, stratum_key=stratum)
    report["r_step"] = r_step
    report["rho_accept"] = [cell.rho_accept.lo, cell.rho_accept.hi]
    report["unresolved_accept"] = [cell.unresolved_accept.lo, cell.unresolved_accept.hi]
    report["reopen_envelope_non_empty"] = bool(
        cell.rho_accept.lo <= cell.rho_accept.hi
        and cell.unresolved_accept.lo <= cell.unresolved_accept.hi)

    report["ok"] = bool(report["policy_matches_declaration"]
                        and report["policy_spec_digest_matches_conditioning"]
                        and report["reopen_envelope_non_empty"])
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preflight_v2_canary_assembly",
        description="run a resolved Canary cell's oracle-assembly contracts without a GPU")
    parser.add_argument("--cell", nargs="+", required=True, metavar="CONFIG=PROTEIN_ID",
                        help="one resolved config and its protein, e.g. "
                             "<WORK>/v2_canary/resolved_configs/q00511_r30.yaml=Q00511")
    parser.add_argument("--json-out", default=None)
    # Reuse over a new script (doc/SCRIPTS.md Reuse-First Gate): the Dual assembly contract is the
    # same contract, with one more object bound.
    parser.add_argument("--dual-overlay", default=None,
                        help="signed Dual overlay JSON; proves the JOINT policy assembles")
    parser.add_argument("--dual-arm", default="",
                        help="joint | a_only | b_only; required with --dual-overlay")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.dual_overlay) != bool(args.dual_arm):
        raise SystemExit(
            "--dual-overlay and --dual-arm must be supplied together; an overlay without an arm "
            "cannot say which comparison this cell runs")
    overlay = None
    if args.dual_overlay:
        from scripts.rf_fusion_v2_preflight import load_dual_overlay_file

        overlay = load_dual_overlay_file(args.dual_overlay)
    surface = _assert_stub_surface()
    rows, failed = [], False
    for item in args.cell:
        path, sep, protein_id = str(item).rpartition("=")
        if not sep or not path or not protein_id:
            raise SystemExit(f"--cell expects CONFIG=PROTEIN_ID, got {item!r}")
        try:
            rows.append(assemble_check(path, protein_id=protein_id,
                                       dual_overlay=overlay, dual_arm=args.dual_arm))
        except Exception as exc:  # noqa: BLE001 - the refusal IS the result
            failed = True
            rows.append({"cell": Path(path).stem, "protein_id": protein_id, "ok": False,
                         "error": f"{type(exc).__name__}: {exc}"})
    payload = {"model_attributes_checked": surface, "cells": rows,
               "all_pass": not failed and all(row.get("ok") for row in rows)}
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    print(text)
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
