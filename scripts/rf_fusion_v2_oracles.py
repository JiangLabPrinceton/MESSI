"""Bind the declared Head, sampler, structure, and policy artifacts to V2."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion_v2.errors import V2Error  # noqa: E402

__all__ = [
    "V2OracleError",
    "OracleSeams",
    "SUPPORT_POLICY_REGISTRY",
    "V2HeadResult",
    "ProductionHeadOracle",
    "assert_constraint_class_matches_band",
    "build_production_oracles",
    "fixed_token_policy_label",
    "reserved_gpu_seconds_clock",
    "resolve_reference",
    "resolve_stratum",
    "resolve_support_policy",
    "build_dual_stack",
    "bind_lineage_incumbent",
    "build_source_geometry_control",
    "build_v2_oracles",
]


class V2OracleError(V2Error):
    """The production oracle stack could not be assembled from what the run declared."""


# --------------------------------------------------------------------------------------------
# the support-policy registry
# --------------------------------------------------------------------------------------------
#
# A registry rather than a constructor call, for the same reason the diagnostic vocabulary is a
# registry: adding a policy that may drive real transitions is an authority edit in code, under
# review, never a config string that self-authorizes.  ``ExplicitProbePolicy`` is deliberately
# ABSENT: PLAN §2.5 admits it for "deterministic tests and the first state-transition diagnostic",
# but its predeclared sets cannot satisfy the kernel against a stochastically realized state (the
# reopen set must name only source-RESOLVED positions), so a canary configured with it measures
# only typed nulls.  ``state_derived_probe`` is what actually runs the first diagnostic.


def _build_state_derived_probe(*, band_table, stratum_key, config, **_context):
    from inverse_folding.reference_flow.fusion_v2.policy import StateDerivedProbePolicy

    # The SPEC digest is the run's DECLARED ``projection_policy_spec`` content role -- the sha256 of
    # the frozen rule file -- not something this factory derives.  The kernel compares the answering
    # policy's spec digest against ``conditioning.projection_policy_spec``, so deriving it here
    # would make both sides agree by construction and check nothing.  The band and stratum stay out
    # of it and enter ``policy_config_digest`` instead, which is what lets a multi-cell campaign
    # (the four-cell Canary) sign one spec while pinning a different reopen cardinality per cell.
    return StateDerivedProbePolicy(
        band_table=band_table, stratum_key=stratum_key,
        policy_spec_digest=_content_digest(config, "projection_policy_spec"))


def bind_lineage_incumbent(*, config, cumulative_reference, reference_sequence, evaluator,
                           lineage_id):
    """``I_0`` under the run's DECLARED depth-0 rule (PLAN §3.1, V2F5A).

    The rule comes from ``config.projection.head_directed.lineage_incumbent_depth0_rule``.  Under
    ``cumulative_safety_reference`` the bound object is the actual reward incumbent.  Under
    ``best_admissible_depth0`` it is only the already-content-bound D0 safety/local-attribution
    reference: the cycle proves the generated rank-zero endpoint and the ladder adopts that exact
    endpoint as ``I_1`` without a global WT donor gate.

    ``predeclared_external_design`` is a legal declaration in the vocabulary and is deliberately NOT
    executed here: it would need its own content-bound input role for the design's bytes and its own
    Head scoring pass, and inventing either would put an unsigned sequence in the position that
    decides which donors may open feedback.
    """
    from inverse_folding.reference_flow.fusion_v2.reward import (
        DEPTH0_BOOTSTRAP_RULE,
        bind_incumbent_from_safety_reference,
    )

    block = config.projection.head_directed
    if block is None:
        raise V2OracleError(
            "config.projection.head_directed is absent, so the run declared no depth-0 incumbent "
            "rule; the Head-directed policy cannot be built without one"
        )
    rule = block.lineage_incumbent_depth0_rule
    if rule not in {"cumulative_safety_reference", DEPTH0_BOOTSTRAP_RULE}:
        raise V2OracleError(
            f"depth-0 incumbent rule {rule!r} is declared in the vocabulary but has no authorized "
            "runtime binding: it needs its own content-bound input role for the predeclared "
            "design's bytes and its own frozen-Head scoring pass.  Declare "
            "'cumulative_safety_reference' or add the role -- an unsigned sequence may not decide "
            "which donors open feedback"
        )
    return bind_incumbent_from_safety_reference(
        reference=cumulative_reference, reference_sequence=reference_sequence,
        lineage_id=lineage_id, evaluator=evaluator, rule="cumulative_safety_reference",
    )


def _build_head_directed_capped(*, band_table, stratum_key, config, head_oracle=None,
                                incumbent=None, safety_reference_score=None, evaluator=None,
                                window_grid_digest=None, dual=None, **_context):
    """V2F5A's production policy.  Every scientific input arrives bound; none is derived here."""
    from inverse_folding.reference_flow.fusion_v2.policy import HeadDirectedCappedPolicy
    from inverse_folding.reference_flow.fusion_v2_runtime.contribution import (
        CounterfactualHeadScorer,
    )

    missing = [name for name, value in (
        ("head_oracle", head_oracle), ("incumbent", incumbent),
        ("safety_reference_score", safety_reference_score), ("evaluator", evaluator),
        ("window_grid_digest", window_grid_digest),
    ) if value is None]
    if missing:
        raise V2OracleError(
            f"the Head-directed policy needs {missing} from the oracle stack; without the frozen "
            "Head and the bound lineage incumbent it has no applied-dose evidence and no donor "
            "gate, which is the Head-blind law V2F5A exists to replace"
        )
    calibration = config.head_directed_calibration()
    if calibration is None:                                # pragma: no cover - loader guarantees it
        raise V2OracleError("config.projection.head_directed is required for this policy")
    return HeadDirectedCappedPolicy(
        band_table=band_table, stratum_key=stratum_key, incumbent=incumbent,
        safety_reference_score=safety_reference_score, evaluator=evaluator,
        window_grid_digest=window_grid_digest, calibration=calibration,
        incumbent_update_law=config.projection.head_directed.lineage_incumbent_update_law,
        counterfactual_scorer=CounterfactualHeadScorer(head_oracle=head_oracle),
        # ``None`` unless a Dual overlay was resolved.  This is the ONLY difference between an
        # a_only arm and a joint arm: the same policy object, with or without an injected second
        # authority, which is what makes "Head B present" and "Head B deciding" separable.
        dual=dual,
        policy_spec_digest=_content_digest(config, "projection_policy_spec"),
        policy_version=config.projection.support_policy_version,
        depth0_incumbent_rule=config.projection.head_directed.lineage_incumbent_depth0_rule,
    )


def build_source_geometry_control(*, treatment_policy, required_writes, required_reopens):
    """The matched Head-blind control, pinned to the treatment arm's REALIZED cardinalities.

    Built from the treatment policy object rather than from the config alone, so the two arms
    provably share one band table, one stratum, one incumbent and one calibration: a control
    assembled independently could agree today and drift with the next config edit.
    """
    from inverse_folding.reference_flow.fusion_v2.policy import SourceGeometryControlPolicy

    return SourceGeometryControlPolicy(
        band_table=treatment_policy.band_table, stratum_key=treatment_policy.stratum_key,
        incumbent=treatment_policy.incumbent, evaluator=treatment_policy.evaluator,
        calibration=treatment_policy.calibration,
        required_writes=int(required_writes), required_reopens=int(required_reopens),
        policy_spec_digest=treatment_policy.policy_spec_digest,
        policy_version=treatment_policy.policy_version,
    )


def _registry() -> dict[str, Callable[..., Any]]:
    from inverse_folding.reference_flow.fusion_v2.policy import (
        HEAD_DIRECTED_CAPPED_POLICY_ID,
        STATE_DERIVED_PROBE_POLICY_ID,
    )

    return {
        STATE_DERIVED_PROBE_POLICY_ID: _build_state_derived_probe,
        HEAD_DIRECTED_CAPPED_POLICY_ID: _build_head_directed_capped,
    }


#: Read through :func:`resolve_support_policy` so the refusal message can name what IS available.
SUPPORT_POLICY_REGISTRY = _registry


def resolve_support_policy(config: Any, *, band_table: Any, stratum_key: str, **context: Any):
    """Build the policy the run DECLARED, or refuse naming what is implemented.

    The kernel matches the answering identity against ``config.declared_policy()``, so a factory
    that quietly substituted a different policy would produce transitions the run's own declaration
    does not describe -- and the artifact would name the declared one.

    ``context`` carries the bound objects only some policies need (the frozen Head, the lineage
    incumbent, the safety reference's own score).  Passed as keywords rather than positionally so a
    policy that does not need them is unaffected, and so adding one later cannot silently reorder
    what an existing policy receives.
    """
    registry = SUPPORT_POLICY_REGISTRY()
    declared = config.projection.support_policy_id
    build = registry.get(declared)
    if build is None:
        raise V2OracleError(
            f"config.projection.support_policy_id={declared!r} has no authorized implementation; "
            f"available: {sorted(registry)}.  Adding one is an authority edit in code, under "
            "review, never a config string -- a policy that drives real feedback transitions may "
            "not authorize itself"
        )
    policy = build(band_table=band_table, stratum_key=stratum_key, config=config, **context)
    identity = policy.identity()
    if identity.policy_version != config.projection.support_policy_version:
        raise V2OracleError(
            f"the {declared!r} implementation is version {identity.policy_version!r} but the run "
            f"declares {config.projection.support_policy_version!r}; the kernel would refuse every "
            "transition, so this is caught here rather than after the prefix is paid"
        )
    return policy


# --------------------------------------------------------------------------------------------
# injectable seams
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleSeams:
    """Every runtime entry point, injectable and lazily resolved.

    Same pattern as ``rf_fusion_model_factory.ModelSeams`` and for the same reason: importing this
    module, printing the config and dry-running must load no model, and the assembly must be
    testable without a GPU.
    """

    build_model_factory: Any = None
    load_band_table: Any = None
    head_scorer: Any = None
    structure_evaluator: Any = None
    gpu_clock: Any = None

    def resolved(self) -> dict:
        from inverse_folding.reference_flow.fusion_v2.schedule import load_band_table

        from scripts.rf_fusion_model_factory import build_model_factory

        return {
            "build_model_factory": self.build_model_factory or build_model_factory,
            "load_band_table": self.load_band_table or load_band_table,
            "head_scorer": self.head_scorer,
            "structure_evaluator": self.structure_evaluator,
            # A real instrument, not the refusal sentinel.  ``_default_gpu_clock`` answers "nobody
            # named an instrument", and on a GPU node that is a refusal -- correctly, but it left
            # the PRODUCTION stack with no way to run at all: the driver injects no seams, so every
            # cluster launch died in ``CostMeter`` the moment the first attempt opened.  The factory
            # owns device placement, so the factory is what must name the instrument.
            "gpu_clock": self.gpu_clock or reserved_gpu_seconds_clock(),
        }


def resolve_reference(manifest_path: Any, protein_id: str) -> tuple[str, str]:
    """The depth-0 reference sequence for ONE protein, from a ``{protein_id: path}`` manifest.

    PLAN §2.7 binds "a predeclared complete reference sequence" per lineage, and the whole-landscape
    hotspot comparator measures every design against it.  A single cohort-wide file binds every
    protein to the same native: at different lengths the Head binding fails outright, and at equal
    lengths -- the dangerous case -- it SUCCEEDS while comparing each design to another molecule.

    The manifest is refused if it is a bare sequence file, because that is exactly the shape that
    works for a one-protein run and then silently shares one reference across a cohort.  A protein
    absent from it fails closed rather than falling back (PLAN §5.2).

    Returns ``(sequence, declared_digest)``.  The digest is PER PROTEIN and lives in the manifest,
    because ``config.content[complete_reference_sequence].expected_sha256`` is a single value and a
    two-protein cohort has two references -- one config field cannot sign both.  That role's digest
    therefore signs the MANIFEST, and the manifest signs each sequence, so every byte is still bound
    and nothing is signed twice under one number.

    Paths inside the manifest are resolved relative to the manifest itself, so the file can be moved
    with its sequences.  Read as ASCII with NO normalization: ``bind_cumulative_reference``
    recomputes the content digest over these exact bytes.
    """
    import json

    manifest = Path(manifest_path)
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V2OracleError(
            f"the complete-reference manifest {manifest} is not readable JSON: {exc}.  It must be "
            "a {protein_id: path} mapping -- a bare sequence file would bind the whole cohort's "
            "hotspot comparator to one protein's native"
        ) from exc
    if not isinstance(entries, dict) or not entries:
        raise V2OracleError(
            f"{manifest} is not a non-empty {{protein_id: path}} reference manifest"
        )
    entry = entries.get(protein_id)
    if isinstance(entry, str):
        raise V2OracleError(
            f"the reference manifest gives {protein_id!r} a bare path; each entry must be "
            '{"path": ..., "sha256": ...} so the sequence carries its own content identity -- the '
            "config's single complete_reference_sequence digest cannot sign a two-protein cohort"
        )
    if not entry:
        raise V2OracleError(
            f"the reference manifest names no complete reference for {protein_id!r} "
            f"(it has {sorted(entries)}); PLAN §5.2 fails closed on missing content identity, and "
            "falling back to another protein's native would anchor this lineage's safety ratchet "
            "to the wrong molecule"
        )
    path, declared = entry.get("path"), entry.get("sha256")
    if not path or not declared:
        raise V2OracleError(
            f"the reference manifest entry for {protein_id!r} needs both 'path' and 'sha256'"
        )
    sequence = (manifest.parent / path).read_text(encoding="ascii")
    observed = _observed_digest(manifest.parent / path)
    if observed != declared:
        raise V2OracleError(
            f"the reference file for {protein_id!r} hashes to {observed} but the manifest declares "
            f"{declared}; the whole-landscape comparator would be anchored to bytes the run never "
            "signed"
        )
    return sequence, declared


def resolve_stratum(manifest_path: Any, protein_id: str) -> str:
    """The COHORT STRATUM this protein's band was measured on, from a ``{protein_id: key}`` map.

    ``source_writeback``'s own contract: ``stratum_key`` "is deliberately NOT derived from
    ``DepthSchedulePoint.band_key`` -- the Interface Map states they are different keys".
    ``band_key`` names a schedule CELL; ``stratum_key`` names the cohort the band's quantiles were
    measured over.  Conflating them gives every protein in a cohort one stratum, so an anchored
    protein reads its reopen cardinality off a band measured on unconstrained ones.

    A protein absent from the manifest fails closed: defaulting to another protein's stratum is
    precisely the silent mis-binding this exists to prevent.
    """
    import json

    manifest = Path(manifest_path)
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V2OracleError(
            f"the protein-stratum manifest {manifest} is not readable JSON: {exc}"
        ) from exc
    if not isinstance(entries, dict) or not entries:
        raise V2OracleError(f"{manifest} is not a non-empty {{protein_id: stratum_key}} mapping")
    stratum = entries.get(protein_id)
    if not isinstance(stratum, str) or not stratum.strip():
        raise V2OracleError(
            f"the stratum manifest names no cohort stratum for {protein_id!r} "
            f"(it has {sorted(entries)}); the band's quantiles were measured over a stratum, and "
            "guessing which one would pin the reopen cardinality from the wrong distribution"
        )
    return stratum


def fixed_token_policy_label(inputs: Any) -> str:
    """The conditioning's ``fixed_token_policy``, in the vocabulary V1 already established.

    ``build_model_factory`` DEFAULTS this to the literal ``"unconstrained"``, and V2 was never
    passing it -- so an anchored protein's conditioning identity recorded ``unconstrained`` while
    the sampler was enforcing 24 hard anchors read from the manifest.  The anchors themselves were
    always applied (``fixed_tokens`` comes from the loaded manifest, and
    :func:`assert_constraint_class_matches_band` checks the REALIZED class), but the provenance row
    every V2 artifact carries named the wrong policy, and two cells that differ precisely in their
    constraint class would have declared the same one.

    Same spelling as ``rf_fusion_v1_oracles`` -- ``manifest:<sha256 of the manifest FILE>`` -- so
    the V1 entry stack and the V2 ladder describe one constraint policy in one vocabulary rather
    than two labels that happen to mean the same thing.
    """
    manifest = inputs.paths.get("constraint_manifest")
    return f"manifest:{_observed_digest(manifest)}" if manifest else "unconstrained"


def assert_constraint_class_matches_band(*, band_table: Any, fixed_tokens: Any,
                                         protein_id: str) -> None:
    """The declared stratum is a LABEL; the constraint class is a FACT about the protein.

    Hard anchors remove positions from the editable domain, so at equal length an anchored protein
    carries different unresolved mass at the same step than an unconstrained one.  A band measured
    on an unconstrained cohort does not describe it, and the reopen cardinality read off that band
    would be pinned from the wrong distribution -- recorded in the artifact as a legitimate
    projection.  ``mixed`` is compatible with either, because that is what it means.
    """
    from scripts.rho_maturity_scan import classify_constraint_stratum

    band_class = band_table.provenance.constraint_stratum_id
    protein_class = classify_constraint_stratum([bool(fixed_tokens)])
    if band_class != "mixed" and band_class != protein_class:
        raise V2OracleError(
            f"{protein_id!r} is {protein_class} but its band was calibrated on a {band_class} "
            "cohort; anchors change the editable domain, so the band's unresolved-mass quantiles "
            "do not describe this protein and the reopen cardinality would come from the wrong "
            "distribution"
        )
    return None


def reserved_gpu_seconds_clock():
    """The production GPU instrument: cumulative RESERVED GPU-seconds for this allocation.

    ``CostMeter`` reads this at the start and end of every attempt and charges the DIFFERENCE, so
    it must be a cumulative counter rather than a duration; the origin is therefore arbitrary and
    only monotonicity and scale matter.

    **What it measures, stated exactly.** Wall time this process has been running, multiplied by
    the number of CUDA devices visible to it.  On a SLURM ``--gres=gpu:N`` allocation those devices
    are reserved for this job and nobody else may use them, so reserved seconds -- not busy seconds
    -- are the quantity ``max_gpu_seconds`` budgets and the quantity the cluster bills.  It
    over-counts relative to device UTILIZATION, and deliberately: a cap sized on utilization would
    authorize a run that holds a GPU idle for a day.  A process with no CUDA device returns ``0.0``,
    which is a measurement and not a default -- it really burned none.

    Device count is read on every call rather than latched, so a process that has not yet placed a
    model is not credited with GPU seconds it could not have spent.
    """
    import time

    origin = time.monotonic()

    def clock() -> float:
        try:
            import torch  # noqa: PLC0415 - deferred so importing this module loads no torch
        except ImportError:
            return 0.0
        if not torch.cuda.is_available():
            return 0.0
        return (time.monotonic() - origin) * float(torch.cuda.device_count())

    return clock


def _default_gpu_clock():
    """The REFUSAL sentinel: what a stack that named no instrument gets.

    ``check_caps`` treats GPU-seconds as a MEASURED quantity, so a fabricated ``0.0`` would let the
    GPU cap read as satisfied on a number nobody took.  A process that has not initialized CUDA
    truthfully burns none; one that HAS must supply a real instrument -- which the production
    factory now does, via :func:`reserved_gpu_seconds_clock`.  This remains the answer only for a
    caller that reaches past the factory and declines to name one.
    """
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return 0.0
    raise V2OracleError(
        "this process has initialized CUDA but no GPU clock was supplied, so the run cannot say "
        "what measured its GPU seconds; pass OracleSeams(gpu_clock=...) rather than let a "
        "fabricated zero certify max_gpu_seconds"
    )


# --------------------------------------------------------------------------------------------
# the factory
# --------------------------------------------------------------------------------------------


def _content_digest(config: Any, role: str) -> str:
    for row in config.content:
        if row.role == role:
            if row.expected_sha256 is None:
                raise V2OracleError(
                    f"config.content role {role!r} is runtime-bound and carries no declared "
                    "digest, but this stack needs it to content-verify the artifact it names"
                )
            return row.expected_sha256
    raise V2OracleError(f"config.content declares no {role!r} row")


@dataclass(frozen=True)
class V2HeadResult:
    """One production ``HeadScore`` plus the ``HeadScoreBinding`` the V2 runtime binds results by.

    The production ``head_scoring.HeadScore`` already carries every field the whole-landscape
    comparator reads (``windows``/``allele``/``score_scale``/``global_risk``/...), but
    ``fusion_v2_runtime.lookahead.bind_head_scores`` additionally requires each result to carry its
    OWN ``binding``: it matches results to completions by identity rather than by position, so a
    reordered batch has to be detectable.  This wrapper adds that one field and copies the rest
    verbatim -- it does not recompute a single score, because a second scoring path is exactly how
    a calibration and the gate it feeds start measuring different quantities.
    """

    protein_id: str
    sequence_md5: str
    sequence_length: int
    allele: str
    score_scale: str
    windows: tuple
    residue_hotspot: tuple | None
    global_risk: float | None
    binding: Any


class ProductionHeadOracle:
    """The frozen Head behind the V2 runtime's ``score(list[OracleRequest])`` contract.

    Wraps the V1 production ``OnlineHeadScorer`` -- the same object the v0/V1 entry paths score
    through -- so the Canary, the hotspot calibration and the v0 boundary all read one Head.  It
    adds no scoring behaviour; it adapts the batch shape and asserts identity.

    Two refusals are deliberate:

    * the scorer's realized ``(allele, score_scale, window_k_min, window_k_max)`` must equal the
      one the run DECLARED.  ``fusion_v2.safety.bind_admission_policy`` compares the same 4-tuple
      and raises ``HeadIdentityMismatch``; catching it here names the offending field instead of
      failing later with two opaque tuples; and
    * a returned score whose ``sequence_md5`` does not key the sequence it was asked about is a
      hard failure.  The digest is the join key the archive, the refold cache and the admission
      ledger all use, so one naming a different design would silently attach a Head result to a
      sequence it was never computed for.
    """

    def __init__(self, scorer: Any, *, allele: str, score_scale: str,
                 window_k_min: int, window_k_max: int) -> None:
        declared = (str(allele), str(score_scale), int(window_k_min), int(window_k_max))
        observed = (str(getattr(scorer, "allele", "")),
                    str(getattr(scorer, "score_scale", "")),
                    int(getattr(scorer, "window_k_min", -1)),
                    int(getattr(scorer, "window_k_max", -1)))
        if declared != observed:
            names = ("allele", "score_scale", "window_k_min", "window_k_max")
            diff = ", ".join(f"{n}: declared {d!r} != realized {o!r}"
                             for n, d, o in zip(names, declared, observed) if d != o)
            raise V2OracleError(
                f"the run's declared Head domain does not match the scorer that was built ({diff}). "
                "bind_admission_policy compares this exact 4-tuple, so the run would fail with "
                "HeadIdentityMismatch after the checkpoints were already resident; and the "
                "whole-landscape N_H is a maximum over the window grid these fields define, so a "
                "threshold measured under one domain does not bound designs scored under another"
            )
        self._scorer = scorer
        self.allele, self.score_scale = declared[0], declared[1]
        self.window_k_min, self.window_k_max = declared[2], declared[3]

    def evaluator_identity(self):
        """The Head evaluator identity, read off the scorer rather than retyped by a caller."""
        from inverse_folding.reference_flow.fusion_v2.identity import HeadEvaluatorIdentity

        return HeadEvaluatorIdentity(
            allele=self.allele, score_scale=self.score_scale,
            window_k_min=self.window_k_min, window_k_max=self.window_k_max,
            head_config_hash=str(self._scorer.head_config_hash),
            head_checkpoint_digest=str(self._scorer.head_checkpoint_digest),
        )

    def score(self, requests) -> list:
        """Score a batch of ``OracleRequest`` and return one result per DISTINCT sequence.

        Deduplicated by ``sequence_md5`` because the Head is a function of the sequence and
        ``bind_head_scores`` refuses a duplicate result for one digest -- two forks that produced
        identical bytes are one measurement, not two.  Batched per protein because
        ``score_batch_same_protein`` is, and a batch spanning proteins would silently score every
        sequence against the first one's context.
        """
        from inverse_folding.reference_flow.fusion_v2.identity import (
            HeadScoreBinding,
            window_grid_digest,
        )

        evaluator = self.evaluator_identity()
        by_protein: dict[str, dict[str, Any]] = {}
        for request in requests:
            by_protein.setdefault(str(request.protein_id), {}) \
                .setdefault(str(request.sequence_md5), request)

        results: list[V2HeadResult] = []
        for protein_id, unique in by_protein.items():
            ordered = list(unique.values())
            batch = self._scorer.score_batch_same_protein(
                protein_id=protein_id,
                records=[(request.sequence_md5, request.sequence) for request in ordered],
            )
            scores = list(batch.scores)
            if len(scores) != len(ordered):
                raise V2OracleError(
                    f"Head returned {len(scores)} score(s) for {len(ordered)} distinct sequence(s) "
                    f"of {protein_id}; a batch that does not correspond one-to-one cannot be "
                    "matched back to the designs it was asked about"
                )
            for request, score in zip(ordered, scores):
                if str(score.sequence_md5) != str(request.sequence_md5):
                    raise V2OracleError(
                        f"Head result for {protein_id} carries sequence_md5 "
                        f"{score.sequence_md5!r} but was asked about {request.sequence_md5!r}; the "
                        "digest is the archive/cache/ledger join key, so a mismatched one attaches "
                        "a score to a sequence it was never computed for"
                    )
                windows = tuple(score.windows)
                results.append(V2HeadResult(
                    protein_id=str(score.protein_id), sequence_md5=str(score.sequence_md5),
                    sequence_length=int(score.sequence_length), allele=str(score.allele),
                    score_scale=str(score.score_scale), windows=windows,
                    residue_hotspot=(tuple(score.residue_hotspot)
                                     if score.residue_hotspot is not None else None),
                    global_risk=(None if score.global_risk is None else float(score.global_risk)),
                    binding=HeadScoreBinding(
                        protein_id=str(score.protein_id), sequence_md5=str(score.sequence_md5),
                        sequence_length=int(score.sequence_length),
                        window_grid_digest=window_grid_digest(windows), evaluator=evaluator,
                    ),
                ))
        return results


#: Every field ``run_rf_refine_fusion.build_oracles`` reads off its ``args``.  Listed explicitly so
#: an unknown key is a typo caught here rather than an attribute the builder silently never finds.
_V0_ORACLE_FIELDS = (
    "head_checkpoint", "head_config_dir", "head_variant_id", "head_device", "head_allele_idx",
    "head_window_batch_size", "allele", "window_k_min", "window_k_max",
    "test_set_parquet", "pdb_root", "refold_cache_dir", "constraint_manifest",
    "esmfold2_site_packages", "esmfold2_model", "esmfold2_num_loops",
    "esmfold2_num_sampling_steps", "esmfold2_num_diffusion_samples", "esmfold2_seed",
)


def _v0_oracle_args(**over: Any):
    """The ``args``-shaped record ``run_rf_refine_fusion.build_oracles`` reads.

    v0's builder was written against an ``argparse.Namespace``; reaching it from V2 through a
    namespace is what lets both stacks run the SAME Head batch path and the SAME definitive
    structure path.  Reimplementing either here would satisfy the type checker and quietly create
    the second implementation the runbook forbids.

    **Unset knobs take v0's OWN argparse defaults, read from v0's parser rather than copied.**  The
    refold protocol knobs (`esmfold2_model`, `num_loops`, `num_sampling_steps`,
    `num_diffusion_samples`, `seed`) are not optional decoration: v0 defaults them to
    ``biohub/ESMFold2`` / 3 / 50 / 1 / 0, and handing ``None`` down instead would fold the V2
    calibration and the Canary under a protocol v0 never used -- while the artifact still claimed
    the v0 definitive contract.  Reading them off the parser means a future change to v0's protocol
    propagates here instead of leaving two literals to drift apart.
    """
    from types import SimpleNamespace

    defaults = {name: None for name in _V0_ORACLE_FIELDS}
    defaults.update(v0_oracle_arg_defaults())
    unknown = sorted(set(over) - set(defaults))
    if unknown:
        raise V2OracleError(f"unknown v0 oracle argument(s): {unknown}")
    return SimpleNamespace(**{**defaults, **over})


def v0_oracle_arg_defaults() -> dict:
    """v0's OWN declared defaults for the fields its oracle builder reads.

    Read off v0's parser, never transcribed: the refold protocol knobs decide how every endpoint in
    the calibration and the Canary is folded, so a copied literal that fell behind v0 would silently
    fold under a protocol nothing declared.
    """
    from scripts.rf_shared_oracles import build_arg_parser  # noqa: PLC0415

    wanted = set(_V0_ORACLE_FIELDS)
    return {action.dest: action.default
            for action in build_arg_parser()._actions if action.dest in wanted}


def build_production_oracles(
    *, structure_config: Any, head_config_dir: Any, head_checkpoint: Any,
    test_set_parquet: Any, pdb_root: Any, refold_cache_dir: Any,
    allele: str, score_scale: str, window_k_min: int, window_k_max: int,
    head_variant_id: str, head_allele_idx: int = 0, head_window_batch_size: int = 64,
    constraint_manifest: Any = None, device: str = "cuda",
    esmfold2: dict | None = None, build_oracles: Any = None,
) -> tuple[ProductionHeadOracle, Callable[[Any], Any]]:
    """Build the REAL ``(head_oracle, structure_oracle)`` pair from declared cluster paths.

    Both come out of v0's ``build_oracles``: the Head is its ``head_fn`` behind
    :class:`ProductionHeadOracle`, and the structure gate is its ``struct_fn`` composed with
    ``fusion.oracles.structure_feasible`` -- the frozen v0 definitive contract, including the
    active-site branch an anchored protein needs.  ``structure_config`` therefore names the v0
    Fusion config that defines backend, ``scTM_min``, the active-site metric and its thresholds; it
    is neither a PDB root nor a refold checkpoint.

    Returns ``structure_oracle(request) -> StructureOutcome``.  A fold or geometry failure is an
    ``evaluated=True, feasible=False`` verdict, never a deferral: the calibration's 48/64 floor and
    the Canary's admission both read ``evaluated AND feasible``, and a deferred outcome would let an
    endpoint nothing folded count toward either.
    """
    from inverse_folding.reference_flow.fusion.config import load_fusion_config
    from inverse_folding.reference_flow.fusion.oracles import structure_feasible
    from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome

    if build_oracles is None:
        from scripts.rf_shared_oracles import build_oracles  # noqa: PLC0415

    fusion_config = load_fusion_config(str(structure_config))
    args = _v0_oracle_args(
        head_checkpoint=str(head_checkpoint), head_config_dir=str(head_config_dir),
        head_variant_id=str(head_variant_id), head_device=str(device),
        head_allele_idx=int(head_allele_idx),
        head_window_batch_size=int(head_window_batch_size), allele=str(allele),
        window_k_min=int(window_k_min), window_k_max=int(window_k_max),
        test_set_parquet=str(test_set_parquet), pdb_root=str(pdb_root),
        refold_cache_dir=str(refold_cache_dir),
        constraint_manifest=(None if constraint_manifest is None else str(constraint_manifest)),
        **(esmfold2 or {}),
    )
    oracles, manifest = build_oracles(args, fusion_config)

    head_oracle = ProductionHeadOracle(
        _scorer_behind(oracles.head_fn), allele=allele, score_scale=score_scale,
        window_k_min=window_k_min, window_k_max=window_k_max)

    def structure_oracle(request: Any):
        import time

        started = time.perf_counter()
        protein_id = str(request.protein_id)
        # Anchored vs unconstrained is read off the MANIFEST v0 itself resolved, not off a flag the
        # caller passed: the active-site branch must fire for exactly the proteins v0 constrains.
        has_active_site = bool(manifest is not None and manifest.has_protein(protein_id))
        try:
            metrics = oracles.struct_fn(protein_id, request.sequence)
            feasible, reason = structure_feasible(
                metrics, fusion_config, has_active_site=has_active_site)
        except Exception as exc:  # noqa: BLE001 - unverifiable structure fails closed, not deferred
            return StructureOutcome(
                feasible=False, cache_status="miss", model_executed=True,
                failure_reason=f"{type(exc).__name__}: {exc}",
                walltime_s=time.perf_counter() - started, evaluated=True)
        return StructureOutcome(
            feasible=bool(feasible),
            cache_status=("hit" if bool(getattr(metrics, "cache_hit", False)) else "miss"),
            model_executed=bool(getattr(
                metrics, "model_executed", not bool(getattr(metrics, "cache_hit", False)))),
            failure_reason=(None if feasible else str(reason)),
            metrics={name: float(value) for name, value in (
                ("scTM", getattr(metrics, "scTM", None)),
                ("pLDDT", getattr(metrics, "pLDDT", None)),
                ("scRMSD", getattr(metrics, "scRMSD", None)),
                ("active_site_RMSD", getattr(metrics, "active_site_RMSD", None)),
                ("max_anchor_sidechain_RMSD",
                 getattr(metrics, "max_anchor_sidechain_RMSD", None)),
            ) if value is not None},
            walltime_s=time.perf_counter() - started, evaluated=True)

    global _STRUCTURE_STACKS_BUILT
    _STRUCTURE_STACKS_BUILT += 1
    return head_oracle, structure_oracle


#: How many definitive structure/refold stacks this process has constructed. A Dual run must build
#: exactly the same number as a single-Head run -- PLAN DUALF2's acceptance criterion -- and this
#: makes that a checkable runtime fact rather than a code-reading argument.
_STRUCTURE_STACKS_BUILT = 0


def build_role_b_head_oracle(
    *, head_config_dir: Any, head_checkpoint: Any, head_variant_id: str,
    allele: str, score_scale: str, window_k_min: int, window_k_max: int,
    head_allele_idx: int, head_window_batch_size: int,
    evaluator_a: Any, device: str = "cuda", build_head_scorer: Any = None,
):
    """Build role B's frozen Head and its counterfactual scorer, and NO structure stack.

    It enters the same Head-construction chain role A uses, at the lowest point that is still that
    chain and the highest point that touches no structure code -- so the two Heads are built by one
    implementation rather than by two that could drift.

    ``head_allele_idx`` and ``head_window_batch_size`` are REQUIRED here, deliberately unlike the
    role-A path which defaults them to 0 and 64. A silently defaulted allele index for a SECOND
    allele would score B's checkpoint against A's allele-embedding row, which is a wrong number
    rather than a missing one.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.contribution import (
        CounterfactualHeadScorer,
    )

    before = _STRUCTURE_STACKS_BUILT
    args = _v0_oracle_args(
        head_config_dir=str(head_config_dir), head_checkpoint=str(head_checkpoint),
        head_variant_id=str(head_variant_id), head_device=str(device),
        head_allele_idx=int(head_allele_idx),
        head_window_batch_size=int(head_window_batch_size),
        allele=str(allele), window_k_min=int(window_k_min), window_k_max=int(window_k_max),
    )
    from scripts.head_runtime import build_head_scorer as _default_build
    from scripts.rf_shared_oracles import _build_head_scorer_from_args

    scorer_b = _build_head_scorer_from_args(args, build_head_scorer or _default_build)
    head_oracle_b = ProductionHeadOracle(
        scorer_b, allele=str(allele), score_scale=str(score_scale),
        window_k_min=int(window_k_min), window_k_max=int(window_k_max))

    identity_b = head_oracle_b.evaluator_identity()
    if evaluator_a is not None:
        if identity_b == evaluator_a or \
                identity_b.head_checkpoint_digest == evaluator_a.head_checkpoint_digest:
            raise V2OracleError(
                "role B resolved to the same Head as role A; the checkpoint digest is the only "
                "bit that distinguishes the two production Heads, so this is a Head mixup rather "
                "than a bookkeeping difference, and it would make the objective reduce one allele "
                "against itself"
            )
    if _STRUCTURE_STACKS_BUILT != before:
        raise V2OracleError(
            "building role B constructed a structure stack; Dual evaluates structure exactly once "
            "per exact sequence and a second stack would double the refold cost the preflight "
            "budgeted for"
        )
    return head_oracle_b, CounterfactualHeadScorer(head_oracle=head_oracle_b)


def _scorer_behind(head_fn: Any):
    """Recover the ``OnlineHeadScorer`` v0's ``head_fn`` closes over.

    v0 hands back a closure rather than the scorer, and V2 needs the object itself for its identity
    (``head_config_hash`` / ``head_checkpoint_digest``) and its batch call.  Read from the closure
    so there is still exactly ONE scorer: building a second one from the same paths would load the
    Head twice and, worse, make "the identity on the artifact" and "the model that scored" two
    objects that merely agree today.
    """
    for cell in getattr(head_fn, "__closure__", None) or ():
        candidate = cell.cell_contents
        if hasattr(candidate, "score_batch_same_protein") and hasattr(candidate, "head_config_hash"):
            return candidate
    raise V2OracleError(
        "could not recover the OnlineHeadScorer from v0's head_fn closure; V2 needs the scorer "
        "itself for its evaluator identity, and constructing a second one would load the Head "
        "twice and let the recorded identity drift from the model that actually scored"
    )


#: Runtime knobs the v0 Head/structure stack needs that are NOT scientific config: they differ
#: between machines and runs, so they arrive as ``--shard-input NAME=VALUE`` like every other path.
_ESMFOLD2_INPUTS = (
    "esmfold2_site_packages", "esmfold2_model", "esmfold2_num_loops",
    "esmfold2_num_sampling_steps", "esmfold2_num_diffusion_samples", "esmfold2_seed",
)


def _production_oracles_for(config: Any, inputs: Any):
    """Build the real Head + definitive structure oracles for one run from its declared inputs.

    The Head DOMAIN is taken from ``config.head`` and never from a shard input: it is scientific
    (it defines the window grid the whole-landscape maximum is taken over) and it is what
    ``bind_admission_policy`` checks the realized scorer against.  Everything else here is a path.
    """
    esmfold2 = {name: inputs.paths.get(name) for name in _ESMFOLD2_INPUTS
                if inputs.paths.get(name) is not None}
    return build_production_oracles(
        structure_config=inputs.require("v0_structure_gate_config"),
        head_config_dir=inputs.require("head_config"),
        head_checkpoint=inputs.require("head_checkpoint"),
        test_set_parquet=inputs.require("test_set_parquet"),
        pdb_root=inputs.require("pdb_root"),
        refold_cache_dir=inputs.require("refold_cache_dir"),
        allele=config.head.allele, score_scale=config.head.score_scale,
        window_k_min=config.head.window_k_min, window_k_max=config.head.window_k_max,
        head_variant_id=inputs.require("head_variant_id"),
        head_allele_idx=int(inputs.paths.get("head_allele_idx", 0)),
        head_window_batch_size=int(inputs.paths.get("head_window_batch_size", 64)),
        constraint_manifest=inputs.paths.get("constraint_manifest"),
        device=inputs.paths.get("device", "cuda"),
        esmfold2=(esmfold2 or None),
    )


def build_dual_stack(*, overlay: Any, arm: str, inputs: Any, evaluator_a: Any,
                     reference_sequence: str, reference_head_score: Any, incumbent: Any,
                     window_grid_digest: str, device: str = "cuda"):
    """Build role B's Head, prove it is the Head the overlay signed, and bind both Dual objects.

    Returns ``(authority, runtime)`` -- the injected support authority the policy decides with, and
    the per-shard runtime the cycle scores and ranks with.

    This is where ``assert_observed_head_b`` finally means something. It used to be called in the
    driver with the overlay's OWN declared digests on both sides, which compares a value with itself
    and passes for any checkpoint on disk. Here the left-hand side is the identity of a Head that
    was actually opened.

    Role B's paths arrive as shard inputs, never from the overlay: the overlay is signed scientific
    identity and is shared across machines, while a checkpoint path is a property of one filesystem.
    """
    from inverse_folding.reference_flow.fusion_v2.dual_policy import DualSupportAuthority
    from inverse_folding.reference_flow.fusion_v2.joint_objective import build_arm_objective
    from inverse_folding.reference_flow.fusion_v2_runtime.dual_runtime import build_dual_runtime
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest

    binding = overlay.head_b_runtime
    declared = binding.evaluator
    head_b, scorer_b = build_role_b_head_oracle(
        head_config_dir=inputs.require("head_b_config"),
        head_checkpoint=inputs.require("head_b_checkpoint"),
        head_variant_id=binding.variant_id, allele=declared.allele,
        score_scale=declared.score_scale, window_k_min=declared.window_k_min,
        window_k_max=declared.window_k_max, head_allele_idx=binding.allele_idx,
        head_window_batch_size=binding.window_batch_size, evaluator_a=evaluator_a, device=device,
    )
    observed = head_b.evaluator_identity()
    overlay.assert_observed_head_b(
        checkpoint_digest=observed.head_checkpoint_digest,
        config_hash=observed.head_config_hash)

    # Role B's verdict on the SAME reference sequence role A's incumbent was bound from.  Both Heads
    # must have scored it, or J(I_0) is a number one instrument never took.
    from inverse_folding.reference_flow.fusion.state import sequence_md5 as _md5

    reference_b = head_b.score([OracleRequest(
        protein_id=str(getattr(incumbent, "protein_id")), sequence=reference_sequence,
        sequence_md5=_md5(reference_sequence), sequence_length=len(reference_sequence),
    )])[0]
    reference_score_b = getattr(reference_b, "score", reference_b)

    # THE arm decision. ``joint`` reduces both coordinates; ``a_only`` / ``b_only`` order on one.
    # Building DualObjective unconditionally here is what made the three arms of a matched
    # comparison execute the identical law under three different signatures.
    objective = build_arm_objective(overlay.calibration, arm=arm)
    if overlay.calibration.window is None:
        raise V2OracleError(
            "the Dual calibration declares no window coordinates; the union reopen law compares "
            "per-window residuals, and the aggregate risk scale does not transfer to them"
        )
    authority = DualSupportAuthority(
        objective=objective, window_coordinates=overlay.calibration.window,
        evaluator_b=declared, counterfactual_scorer_b=scorer_b,
        incumbent_score_b=reference_score_b,
        incumbent_joint_value=float(objective.evaluate(
            raw_a=float(getattr(incumbent, "head_global_risk")),
            raw_b=float(getattr(reference_score_b, "global_risk"))).value),
        safety_reference_score_b=reference_score_b,
        # Empty at construction: every cycle rebinds it with its own pool, so a donor role B has
        # not scored raises rather than resolving to a stale entry.
        donor_score_b_by_endpoint={},
        max_counterfactual_sequences_per_cycle=int(
            overlay.max_counterfactual_sequences_per_cycle),
    )
    runtime = build_dual_runtime(
        overlay=overlay, arm=arm, head_b=head_b,
        # Role B's verdict on the SAME frozen reference role A's cumulative ratchet uses, so its
        # drift is measured by one instrument against one reference.
        safety_reference_score_b=reference_score_b,
        safety_reference_binding_id=str(getattr(incumbent, "incumbent_id", "")))
    return authority, runtime


def build_v2_oracles(*, protein_id: str, config: Any, inputs: Any, seams: OracleSeams | None = None,
                     dual: Any = None):
    """Assemble the ``{gpu_clock, cycle_kwargs}`` contract.

    Everything expensive is built through the shared model factory; everything scientific is bound
    to ``config``.  Returns the mapping ``scripts.rf_fusion_v2_cohort.run_v2_shard`` consumes.
    """
    resolved = (seams or OracleSeams()).resolved()

    from inverse_folding.reference_flow.fusion_v2 import identity as ident
    from inverse_folding.reference_flow.fusion_v2.safety import (
        bind_admission_policy,
        bind_cumulative_reference,
        open_lineage_ledger,
    )
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest
    from inverse_folding.reference_flow.fusion_v2_runtime.admission import SafetyGate

    # ---- the band, content-verified against the run's own provenance --------------------------
    band_digest = _content_digest(config, "schedule_band_calibration")
    band_table = resolved["load_band_table"](
        inputs.require("schedule_band_calibration"), expected_content_digest=band_digest)
    stratum_key = resolve_stratum(inputs.require("protein_stratum_manifest"), protein_id)

    # ---- the model stack, SHARED with the V1 entry path ---------------------------------------
    model = resolved["build_model_factory"](
        base_if_checkpoint=inputs.require("base_if_checkpoint"),
        rf_sampler_config=inputs.require("rf_sampler_config"),
        test_set_parquet=inputs.require("test_set_parquet"),
        pdb_root=inputs.require("pdb_root"),
        device=inputs.paths.get("device", "cuda"),
        constraint_manifest=inputs.paths.get("constraint_manifest"),
        fixed_token_policy=fixed_token_policy_label(inputs),
    )
    _prepared, denoiser = model.backbone_and_denoiser(protein_id)
    length = model.sequence_length(protein_id)
    # The declared stratum is checked against the protein's REALIZED constraint class before
    # anything else reads the band: a label alone cannot stop an anchored protein from running on
    # an unconstrained calibration.
    assert_constraint_class_matches_band(
        band_table=band_table, fixed_tokens=model.fixed_tokens(protein_id),
        protein_id=protein_id)

    # ---- the depth-0 safety reference, bound to the reference BYTES ---------------------------
    # Read as BYTES with no normalization: ``bind_cumulative_reference`` recomputes the content
    # digest as SHA-256 over the sequence's ASCII bytes and compares it to the config's declared
    # one, so the file this role names must be the canonical sequence itself -- no header, no
    # wrapper, no trailing newline.  Stripping here would make the two digests disagree for a file
    # that is otherwise correct, and silently accepting a stripped variant would put bytes in the
    # gate that the config never signed.
    reference_sequence, reference_digest = resolve_reference(
        inputs.require("complete_reference_manifest"), protein_id)
    # The Head and the definitive structure gate are REQUIRED, so a run that declared neither seam
    # must build the real ones rather than proceed with ``None``.  Before this the factory returned
    # a cycle whose ``head_oracle`` was ``None``; the failure surfaced as an ``AttributeError`` on
    # the first score -- after the DPLM checkpoint was already resident on the GPU.
    if resolved["head_scorer"] is None or resolved["structure_evaluator"] is None:
        produced_head, produced_structure = _production_oracles_for(config, inputs)
        resolved["head_scorer"] = resolved["head_scorer"] or produced_head
        resolved["structure_evaluator"] = resolved["structure_evaluator"] or produced_structure
    head_identity = ident.HeadEvaluatorIdentity(
        allele=config.head.allele, score_scale=config.head.score_scale,
        window_k_min=config.head.window_k_min, window_k_max=config.head.window_k_max,
        head_config_hash=_content_digest(config, "head_config"),
        head_checkpoint_digest=_content_digest(config, "head_checkpoint"),
    )
    policy = bind_admission_policy(config, head_identity)
    # Scored through the SAME batch interface every endpoint goes through: a reference scored by a
    # different path could differ from the designs it is compared against for a reason that is not
    # the design.
    from inverse_folding.reference_flow.fusion.state import sequence_md5

    reference_head_score = resolved["head_scorer"].score([OracleRequest(
        protein_id=protein_id, sequence=reference_sequence,
        sequence_md5=sequence_md5(reference_sequence),
        sequence_length=len(reference_sequence),
    )])[0]
    cumulative = bind_cumulative_reference(
        lineage_id=f"{protein_id}:fam0", protein_id=protein_id,
        reference_label=config.safety.cumulative_reference_label,
        reference_sequence=reference_sequence,
        reference_content_digest=reference_digest,
        policy=policy, head_score=reference_head_score,
    )
    safety_gate = SafetyGate(policy=policy, ledger=open_lineage_ledger(cumulative))

    # ``I_0``, bound under the run's declared depth-0 rule and BEFORE any endpoint is scored -- the
    # reference it comes from is already frozen, which is what PLAN §3.1 requires of the reward
    # baseline.  ``None`` for every policy that does not consume one; ``resolve_support_policy``
    # refuses only if the DECLARED policy needs it.
    incumbent = None
    if config.projection.head_directed is not None:
        incumbent = bind_lineage_incumbent(
            config=config, cumulative_reference=cumulative,
            reference_sequence=reference_sequence, evaluator=head_identity,
            lineage_id=f"{protein_id}:fam0",
        )

    dual_authority = dual_runtime = None
    if dual is not None:
        overlay, arm = dual
        if incumbent is None:
            raise V2OracleError(
                "a Dual arm was resolved but the declared policy binds no lineage incumbent; the "
                "joint donor gate has no reference to compare against"
            )
        dual_authority, dual_runtime = build_dual_stack(
            overlay=overlay, arm=arm, inputs=inputs, evaluator_a=head_identity,
            reference_sequence=reference_sequence, reference_head_score=reference_head_score,
            incumbent=incumbent,
            window_grid_digest=cumulative.binding.head_binding.window_grid_digest,
            device=inputs.paths.get("device", "cuda"))

    return {
        "gpu_clock": resolved["gpu_clock"],
        "cycle_kwargs": dict(
            sampler=model.sampler, denoiser=denoiser, config=model.rf_config,
            sequence_length=length, h_values=model.null_h_values(length),
            # ``id_to_aa``, NOT ``alphabet``: the cycle's ``alphabet`` parameter is a
            # ``Mapping[int, str]`` residue map, and that is what ``PreparedModel`` calls it.  The
            # same rename cost the calibrator a cluster allocation; the AST contract test in
            # ``tests/scripts/test_rf_fusion_v2_oracles.py`` now checks every ``model.<attr>`` this
            # factory reads against the real class, so a third occurrence fails locally.
            residue_token_ids=model.aa_token_ids, alphabet=model.id_to_aa,
            fixed_tokens=model.fixed_tokens(protein_id),
            lineage=ident.LineageRef(
                protein_id=protein_id, root_id=f"{protein_id}:v2:d0:r0", family_id="fam0",
                depth=0, parent_state_id=None, parent_transition_id=None,
                origin_endpoint_id=None),
            mask_token_id=model.mask_token_id, aa_token_ids=model.aa_token_ids,
            conditioning=_conditioning(config, model=model, protein_id=protein_id,
                                       inputs=inputs, band_table=band_table),
            safety_reference=cumulative.binding,
            head_oracle=resolved["head_scorer"], structure_oracle=resolved["structure_evaluator"],
            support_policy=resolve_support_policy(
                config, band_table=band_table, stratum_key=stratum_key,
                head_oracle=resolved["head_scorer"], incumbent=incumbent,
                safety_reference_score=reference_head_score, evaluator=head_identity,
                window_grid_digest=cumulative.binding.head_binding.window_grid_digest,
                dual=dual_authority),
            dual=dual_runtime,
            band_table=band_table, stratum_key=stratum_key,
            declared_band_id=band_table.provenance.calibration_id,
            declared_band_digest=band_table.provenance.calibration_content_digest,
            safety_gate=safety_gate,
        ),
    }


def _observed_digest(path: Any) -> str:
    """SHA-256 over the file's bytes -- the same function the driver signs declared inputs with.

    Two different digest functions over the same role would make the config's declared number and
    the run's observed one incomparable, which is the failure PLAN §5.2 ("binds file CONTENTS, not
    paths alone") exists to prevent.
    """
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _conditioning(config: Any, *, model: Any, protein_id: str, inputs: Any,
                  band_table: Any):
    """The run's content provenance, one digest per PLAN §5.2 role.

    A FROZEN role carries its digest in the config and is used as declared.  A RUNTIME-bound role
    carries none by construction -- it is observed, not declared -- so its digest is computed from
    the file the run supplied for it via ``--shard-input <role>=PATH``.  A role with neither is a
    typed refusal: PLAN §5.2 fails closed on missing content identity, and a conditioning identity
    with a hole in it would sign a run whose inputs nobody can name.
    """
    from inverse_folding.reference_flow.fusion_v2 import identity as ident

    declared = {row.role: row.expected_sha256 for row in config.content}
    supplied = dict(getattr(inputs, "paths", {}) or {})

    resolved: dict[str, str] = {}
    unbound: list[str] = []
    for role, field in ident.CONTENT_ROLE_TO_FIELD.items():
        digest = declared.get(role)
        if not digest:
            path = supplied.get(role)
            digest = _observed_digest(path) if path else None
        if not digest:
            unbound.append(role)
            continue
        resolved[field] = digest
    if unbound:
        raise V2OracleError(
            f"no content identity for role(s) {sorted(unbound)}: they are neither frozen in "
            "config.content nor supplied as --shard-input <role>=PATH.  PLAN §5.2 fails closed on "
            "missing content identity rather than signing a run whose inputs nobody can name"
        )

    base = {field.split(".", 1)[1]: digest
            for field, digest in resolved.items() if field.startswith("base.")}
    own = {field: digest for field, digest in resolved.items() if not field.startswith("base.")}
    # The per-protein coordinate mask is computable only after the backbone is prepared, so it
    # overrides whatever a file-level digest would have said about the mask policy.
    base["coordinate_mask"] = model.coordinate_mask_digest(protein_id)
    base["tokenizer"] = model.tokenizer_digest
    base["fixed_token_policy"] = model.fixed_token_policy
    # ``entry_config`` is the base bundle's name for the sampler config the run declares; the role
    # vocabulary calls it ``rf_sampler_config`` and it appears in BOTH halves, because the V1
    # conditioning bundle is reused verbatim (PLAN §5.2's reuse) while V2 also records it as its
    # own provenance row.
    base["entry_config"] = own["rf_sampler_config"]
    # ``schedule_band_calibration`` is the one role whose config digest and whose SCIENTIFIC
    # identity are different numbers.  ``make_band_table`` rebinds ``calibration_content_digest``
    # to a canonical digest of the TABLE's content; that is what ``declared_band_digest`` carries
    # and what ``q_phi`` compares against this field.  The role's ``expected_sha256`` signs the
    # FILE (PLAN §5.2), which the file digest above already records.  Left conflated, the kernel's
    # own provenance check would refuse every real run.  Same shape as ``coordinate_mask``: an
    # intrinsic identity a file digest cannot express overrides the role digest here.
    own["schedule_band_calibration"] = band_table.provenance.calibration_content_digest
    return ident.V2ConditioningIdentity(base=ident.make_v2_conditioning(**base), **own)
