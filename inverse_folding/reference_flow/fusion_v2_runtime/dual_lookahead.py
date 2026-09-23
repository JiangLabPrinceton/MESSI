"""DUALF2 runtime: fan one deduplicated endpoint set to two frozen Heads, then bind the evidence.

Implementation contract: ``PLAN_RF_FUSION_V2_DUAL_ALLELE.md`` DUALF2.

Role A is the existing production Head and its results go through the unchanged
:func:`fusion_v2_runtime.lookahead.bind_head_scores`, so the ``CompleteEndpoint`` objects a Dual run
produces are the ones a single-Head run would have produced -- same fields, same values, same
identities. Role B never touches them. Everything joint lands in the sidecar.

Two cost rules travel with this module because they are contracts, not bookkeeping. A Dual cycle
issues **two** Head batches, and the ledger's logical identity is
``(event_id, protein_id, arm, phase)`` with no allele dimension: both batches landing on one
``event_id`` would merge first-wins, charge one allele's calls, and register the second batch as a
retry -- which breaches ``max_retries`` and fails the cell. :func:`dual_head_event_id` namespaces
the event by role so the two batches are two events. And both Heads must be asked for the **same
deduplicated sequence set**, which :func:`paired_request_digest` pins, so a divergence is a refusal
rather than a silently half-joint archive.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..fusion_v2.dual_evidence import (
    DualEndpointEvidence,
    HeadResult,
    V2DualEvidenceError,
    bind_dual_evidence_batch,
    positive_mass_density,
)
from ..fusion_v2.identity import HeadEvaluatorIdentity, canonical_digest
from ..fusion_v2.joint_objective import AlleleRole, DualObjective
from .lookahead import bind_head_scores

__all__ = [
    "dual_stage_event_id",
    "dual_head_event_id",
    "paired_request_digest",
    "as_head_result",
    "bind_paired_head_scores",
]


def dual_stage_event_id(event_prefix: str, stage: str, role: AlleleRole) -> str:
    """``<prefix>:<stage>:a`` / ``<prefix>:<stage>:b`` -- one ledger event per allele, per batch.

    The single-Head spelling is ``<prefix>:<stage>``. Namespacing by role rather than reusing it is
    what keeps two batches of the same stage from colliding into one logical event; the legacy
    spelling is left untouched so a Dual-off run's ledger is byte-identical.

    A Dual cycle issues two batches at TWO stages -- the lookahead ``head`` batch and the policy's
    ``counterfactual`` batch -- and both are real Head work on the same scale, so the rule lives
    here once rather than being spelled out at each call site.
    """
    if not isinstance(role, AlleleRole):
        raise V2DualEvidenceError("role must be an AlleleRole")
    if not isinstance(event_prefix, str) or not event_prefix.strip():
        raise V2DualEvidenceError("event_prefix must be a non-empty str")
    if not isinstance(stage, str) or not stage.strip() or ":" in stage:
        raise V2DualEvidenceError(
            f"stage must be a non-empty str without ':', got {stage!r}; the colon is the "
            "namespace separator and a stage carrying one would forge a different event")
    return f"{event_prefix}:{stage}:{role.value.lower()}"


def dual_head_event_id(event_prefix: str, role: AlleleRole) -> str:
    """The lookahead stage's per-allele event id: ``<prefix>:head:a`` / ``<prefix>:head:b``."""
    return dual_stage_event_id(event_prefix, "head", role)


def paired_request_digest(sequence_md5s: Sequence[str]) -> str:
    """Digest of the deduplicated sequence set both Heads must be asked for.

    Order-independent by construction, because a batch is a set of measurements and two orderings of
    one set are one request.
    """
    unique = sorted({str(value) for value in sequence_md5s})
    if not unique:
        raise V2DualEvidenceError("a paired Head request must cover at least one sequence")
    return canonical_digest({"sequences": unique})


def as_head_result(result: Any, *, with_density: bool = False) -> HeadResult:
    """Adapt one oracle result -- which is both the score and the carrier of its binding.

    ``with_density`` is opt-in: the return-boundary axis is computed only for a run that builds that
    front, so an ordinary steering run never pays for it and never carries a half-populated field.
    """
    binding = getattr(result, "binding", None)
    if binding is None:
        raise V2DualEvidenceError(
            "every Head result must carry its own binding; without it the batch can only be "
            "matched by position, and a reordered batch would be undetectable"
        )
    density = None
    if with_density:
        hotspot = getattr(result, "residue_hotspot", None)
        if hotspot is None:
            raise V2DualEvidenceError(
                "the return-boundary axis was requested but this Head result carries no "
                "residue_hotspot"
            )
        density = positive_mass_density(hotspot, int(getattr(binding, "sequence_length")))
    return HeadResult(score=result, binding=binding, density=density)


def bind_paired_head_scores(
    *,
    completions: Sequence[Any],
    results_a: Sequence[Any],
    results_b: Sequence[Any],
    evaluator_a: HeadEvaluatorIdentity,
    evaluator_b: HeadEvaluatorIdentity,
    window_grid_digest: str,
    objective: DualObjective,
    density_objective: DualObjective | None = None,
    cost_event_ids: Sequence[str] = (),
) -> tuple[tuple[Any, ...], tuple[DualEndpointEvidence, ...]]:
    """Return ``(endpoints, dual_evidence)`` for one screen.

    The endpoints are produced by the unchanged single-Head binder from role A alone, so a Dual run
    and a single-Head run agree on them exactly. Only after they exist is role B paired in.
    """
    if evaluator_a == evaluator_b:
        raise V2DualEvidenceError(
            "both roles declare the same Head evaluator identity; one Head has been bound twice"
        )
    with_density = density_objective is not None

    endpoints = bind_head_scores(
        completions=completions, results=results_a, evaluator=evaluator_a,
        window_grid_digest=window_grid_digest, cost_event_ids=cost_event_ids)

    typed_a = tuple(as_head_result(result, with_density=with_density) for result in results_a)
    typed_b = tuple(as_head_result(result, with_density=with_density) for result in results_b)

    digest_a = paired_request_digest([result.sequence_md5 for result in typed_a])
    digest_b = paired_request_digest([result.sequence_md5 for result in typed_b])
    if digest_a != digest_b:
        raise V2DualEvidenceError(
            "the two Heads were asked for different sequence sets; a joint archive assembled from "
            "two different request sets is joint for some endpoints and single-allele for the rest"
        )

    for result in typed_b:
        if result.binding.evaluator != evaluator_b:
            raise V2DualEvidenceError(
                "a role B result was produced by a different evaluator identity than the role B "
                "binding declares; two evaluators are two measuring instruments"
            )
        if result.binding.window_grid_digest != window_grid_digest:
            raise V2DualEvidenceError(
                f"role B window grid mismatch: result declares "
                f"{result.binding.window_grid_digest}, the pool declares {window_grid_digest}"
            )

    evidence = bind_dual_evidence_batch(
        endpoints=endpoints, results_a=typed_a, results_b=typed_b,
        objective=objective, density_objective=density_objective)
    return endpoints, evidence
