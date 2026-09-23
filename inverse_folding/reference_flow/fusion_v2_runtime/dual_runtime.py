"""DUALF1 wiring: the one value that turns a legacy V2 cycle into a Dual cycle.

Implementation contract: ``PLAN_RF_FUSION_V2_DUAL_ALLELE.md`` DUALF2/DUALF3/DUALF5.

The Dual layer was built as a library of laws -- objective, evidence, union policy, selection key --
and every one of them was unit-tested in isolation. None of them was ever CALLED by the production
path: ``run_v2_shard`` resolved an overlay only far enough to compute a run signature, the cycle
still asked one Head, and the archive still ranked by ``head_global_risk``. Three arms would have
executed identical A-only V2 under three different signatures, which is the one failure mode a
matched comparison cannot survive -- every arm agrees, and the agreement means nothing.

This module is the seam. It exists so that the wiring is ONE optional argument threaded through
shard -> ladder -> cycle rather than a Dual branch inside each of them: a run with no overlay passes
``None`` and every line below is unreachable, which is what keeps Dual-off equivalence structural.

It owns exactly one piece of mutable state -- the endpoint-evidence registry -- because the archive
elite is maintained ACROSS cycles under the joint ordering, so the key must be able to see evidence
bound by an earlier rung. Everything else is a value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..fusion_v2.dual_evidence import DualEndpointEvidence, V2DualEvidenceError
from ..fusion_v2.errors import V2Error
from ..fusion_v2.identity import HeadEvaluatorIdentity
from ..fusion_v2.joint_objective import AlleleRole, DualObjective, build_arm_objective
from .dual_lookahead import bind_paired_head_scores, dual_stage_event_id

__all__ = ["V2DualRuntimeError", "DualRuntime", "build_dual_runtime"]


class V2DualRuntimeError(V2Error):
    """A Dual runtime wiring contract was violated."""


@dataclass
class DualRuntime:
    """Role B's half of a running shard: the Head, the objective, and the evidence it accumulates.

    ``arm`` is carried because it is what the artifact rows are keyed by, and because the three arms
    of one matched comparison differ ONLY in which authority is injected -- a run that could not name
    its own arm could not be told apart from its control after the fact.
    """

    overlay: Any
    arm: str
    head_b: Any
    objective: DualObjective
    density_objective: DualObjective | None = None
    #: endpoint_id -> the joint evidence bound for it. Grows as the ladder descends; never pruned,
    #: because the archive elite is compared against endpoints from every earlier depth.
    evidence_by_endpoint: dict[str, DualEndpointEvidence] = field(default_factory=dict)
    #: Role B's score of the frozen cumulative safety reference, and that reference's binding id.
    #: Present => role B's whole-landscape drift is measured for every endpoint. Absent => the run
    #: declared it would not measure it. Never a gate (audit Appendix E): a second catastrophic
    #: gate would bind against nothing on this substrate and would move every endpoint identity.
    safety_reference_score_b: Any = None
    safety_reference_binding_id: str = ""
    #: endpoint_id -> role B's FULL score object. The sidecar above carries only the reduced
    #: values, and the union reopen law compares per-window residuals -- so the windows have to
    #: survive too, or the joint policy could steer but not attribute.
    score_b_by_endpoint: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.objective, DualObjective) \
                or self.objective.quantity != "global_risk":
            raise V2DualRuntimeError(
                "the runtime steers on the global-risk objective; a density view orders a return "
                "boundary, not a lineage"
            )
        if not isinstance(self.arm, str) or not self.arm.strip():
            raise V2DualRuntimeError("arm must be a non-empty str")
        if self.arm not in getattr(self.overlay, "arms", ()):  # pragma: no branch - cheap guard
            raise V2DualRuntimeError(
                f"arm {self.arm!r} is not in the overlay's declared bundle "
                f"{list(getattr(self.overlay, 'arms', ()))}"
            )
        if getattr(self.objective, "arm", "joint") != self.arm:
            raise V2DualRuntimeError(
                f"this runtime declares arm {self.arm!r} but its objective is the "
                f"{getattr(self.objective, 'arm', 'joint')!r} ordering law; the arm label and the "
                "law it names must be one decision, or the artifact records an experiment the "
                "engine did not run"
            )
        if not hasattr(self.head_b, "score"):
            raise V2DualRuntimeError("head_b must expose the oracle .score(requests) contract")

    # -- identity -------------------------------------------------------------------------------

    @property
    def evaluator_b(self) -> HeadEvaluatorIdentity:
        return self.objective.coordinates.coordinate(AlleleRole.B).evaluator

    @property
    def evaluator_a(self) -> HeadEvaluatorIdentity:
        return self.objective.coordinates.coordinate(AlleleRole.A).evaluator

    @property
    def calibration_digest(self) -> str:
        return self.objective.calibration.content_digest

    # -- the second Head batch ------------------------------------------------------------------

    def score_pool(self, *, requests: Sequence[Any], cost_meter: Any, event_prefix: str,
                   stage: str) -> Sequence[Any]:
        """Ask role B for the SAME deduplicated request set role A was asked for.

        Journaled under its own per-allele event id: the ledger's logical identity carries no allele
        dimension, so one shared id would merge the two batches first-wins -- charging one allele
        and recording the other as a retry.
        """
        with cost_meter.attempt(
            event_id=dual_stage_event_id(event_prefix, stage, AlleleRole.B),
            phase="head", request_kind="head_batch",
            request_digest=_request_digest(requests),
            head_calls=len(requests),
        ) as receipt:
            results = self.head_b.score(list(requests))
            receipt.observe(physical_forwards=0, head_calls=len(requests))
        return results

    def bind_pool(self, *, completions: Sequence[Any], results_a: Sequence[Any],
                  results_b: Sequence[Any], window_grid_digest: str,
                  cost_event_ids: Sequence[str] = ()) -> tuple[Any, ...]:
        """Bind both Heads onto one pool and REGISTER the evidence for later ordering.

        Returns the endpoints, which are produced by the unchanged single-Head binder from role A
        alone -- so a Dual run and a single-Head run agree on the endpoint objects exactly.
        """
        endpoints, evidence = bind_paired_head_scores(
            completions=completions, results_a=results_a, results_b=results_b,
            evaluator_a=self.evaluator_a, evaluator_b=self.evaluator_b,
            window_grid_digest=window_grid_digest, objective=self.objective,
            density_objective=self.density_objective, cost_event_ids=cost_event_ids)
        by_md5 = {str(getattr(getattr(result, "binding", result), "sequence_md5", "")):
                  getattr(result, "score", result) for result in results_b}
        for endpoint, row in zip(endpoints, evidence):
            self.register(self._with_hotspot_b(row, by_md5.get(str(endpoint.sequence_md5))))
            score_b = by_md5.get(str(endpoint.sequence_md5))
            if score_b is None:
                raise V2DualRuntimeError(
                    f"endpoint {endpoint.endpoint_id} has no role B score for its own sequence; "
                    "the batch was matched by position rather than by content"
                )
            self.score_b_by_endpoint[str(endpoint.endpoint_id)] = score_b
        return endpoints

    def _with_hotspot_b(self, evidence: DualEndpointEvidence, score_b: Any) -> DualEndpointEvidence:
        """Attach role B's whole-landscape drift, when the run declared a role B reference.

        Without this the three ``hotspot_b_*`` columns were structurally always null and the one
        safety statement the program is allowed to make -- that role B's landscape was WATCHED even
        though it was not gated -- had no producer at all.
        """
        if self.safety_reference_score_b is None or score_b is None:
            return evidence
        from dataclasses import replace as _replace

        from ..fusion_v2.dual_evidence import role_b_hotspot_telemetry

        return _replace(evidence, hotspot_b=role_b_hotspot_telemetry(
            endpoint_id=evidence.endpoint_id, design_score_b=score_b,
            reference_score_b=self.safety_reference_score_b, evaluator_b=self.evaluator_b,
            reference_binding_id=self.safety_reference_binding_id))

    def register(self, evidence: DualEndpointEvidence) -> None:
        endpoint_id = str(evidence.endpoint_id)
        existing = self.evidence_by_endpoint.get(endpoint_id)
        if existing is not None and existing != evidence:
            raise V2DualRuntimeError(
                f"endpoint {endpoint_id} was bound twice with different Dual evidence; one exact "
                "endpoint has one pair of Head verdicts, and two would make its joint value depend "
                "on which binding a reader happened to consult"
            )
        self.evidence_by_endpoint[endpoint_id] = evidence

    # -- what the legacy surfaces consume -------------------------------------------------------

    def rank_key(self, *, quantity: str = "risk") -> Callable[[Any], tuple[float, str]]:
        """The ``(J, endpoint_id)`` ordering, over the evidence bound SO FAR.

        Rebuilt per call rather than captured once: the archive admits endpoints as the ladder
        descends, and a key closed over an early snapshot would refuse every later endpoint.
        """
        from .dual_selection import dual_rank_key

        registry = self.evidence_by_endpoint

        def key(endpoint: Any) -> tuple[float, str]:
            return dual_rank_key(registry, quantity=quantity)(endpoint)

        return key

    def donor_scores(self, endpoints: Sequence[Any]) -> dict[str, Any]:
        """Role B's raw score for each endpoint in a pool, keyed by endpoint id.

        This is what the support authority is rebound with each cycle. It is built from the
        REGISTERED evidence rather than from a second scoring pass, so the number the donor gate
        uses is the same number the artifact publishes.
        """
        scores: dict[str, Any] = {}
        for endpoint in endpoints:
            endpoint_id = str(getattr(endpoint, "endpoint_id"))
            score_b = self.score_b_by_endpoint.get(endpoint_id)
            if score_b is None:
                raise V2DualEvidenceError(
                    f"endpoint {endpoint_id} carries no role B verdict; substituting role A's "
                    "would compare an allele against itself"
                )
            scores[endpoint_id] = score_b
        return scores


def _request_digest(requests: Sequence[Any]) -> str:
    from ..fusion_v2.identity import canonical_digest

    return canonical_digest([str(getattr(r, "sequence_md5", r)) for r in requests])


def build_dual_runtime(*, overlay: Any, arm: str, head_b: Any,
                       safety_reference_score_b: Any = None,
                       safety_reference_binding_id: str = "",
                       with_density: bool = False) -> DualRuntime:
    """Assemble the runtime from a loaded overlay and an already-built role B Head.

    The Head is passed in rather than built here so this module performs no I/O and loads no model:
    the same reason ``fusion_v2`` never touches the filesystem.
    """
    calibration = getattr(overlay, "calibration", None)
    if calibration is None:
        raise V2DualRuntimeError("the overlay carries no calibration")
    density = None
    if with_density:
        if calibration.density is None:
            raise V2DualRuntimeError(
                "the return-boundary axis was requested but the calibration declares no density "
                "coordinates; the second axis cannot be normalized on the risk scale"
            )
        density = build_arm_objective(
            calibration, arm=arm, quantity="positive_mass_density")
    return DualRuntime(
        overlay=overlay, arm=str(arm), head_b=head_b,
        safety_reference_score_b=safety_reference_score_b,
        safety_reference_binding_id=str(safety_reference_binding_id),
        objective=build_arm_objective(calibration, arm=arm), density_objective=density)
