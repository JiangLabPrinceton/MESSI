"""V2F7: strict V2 preflight -- config resolution, one shared digest, and the launch gate.

PLAN V2F7 acceptance: "``--print-config`` and ``--dry-run`` load no model and derive one shared
config digest."  Both halves are load-bearing.

**No model.**  A preflight that imported torch would make the cheapest safety check the most
expensive one, and on a cluster it would burn a GPU allocation to discover a typo.  Nothing in this
module imports torch, pandas, or the sampler.

**One digest.**  ``--print-config``, ``--dry-run`` and the realized run must all name the same
config.  Three code paths computing "the digest" separately is how a run ends up reporting a
configuration it did not use, so the digest comes from :meth:`V2Config.config_digest` and nowhere
else.

The budget projection here is deliberately arithmetic over DECLARED quantities.  It cannot know the
real per-refold cost, so it never pretends to: it projects the logical cost model of PLAN §3.4 and
checks it against the declared caps, and says so.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion_v2.config import (  # noqa: E402
    V2ConfigError,
    load_v2_config,
)
__all__ = [
    "V2PreflightError",
    "BudgetProjection",
    "resolve_v2_config",
    "load_v2_config_file",
    "load_dual_overlay_file",
    "input_signature",
    "project_v2_budget",
    "assert_launch_feasible",
    "print_config_payload",
]


class V2PreflightError(V2ConfigError):
    """A preflight refusal: the run is not launchable as configured."""


def load_v2_config_file(path: Any):
    """Read a YAML config and resolve it.  Loads no model."""
    import yaml

    with open(Path(path)) as handle:
        return resolve_v2_config(yaml.safe_load(handle))



def load_dual_overlay_file(path: Any):
    """Read a signed dual-allele overlay JSON into a fully typed :class:`DualOverlay`.

    The same "read a file, load no model" seam ``load_v2_config_file`` occupies, and for the same
    reason: ``fusion_v2.dual_config`` performs no I/O, so without this step ``--dual-overlay PATH``
    has nothing to call. The Dual import is deferred into the body so a legacy run that never passes
    the flag never loads the Dual layer.
    """
    import json

    from inverse_folding.reference_flow.fusion_v2.dual_config import dual_overlay_from_mapping

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return dual_overlay_from_mapping(payload)

def resolve_v2_config(payload: Mapping[str, Any]):
    """Delegate to the typed loader.  This module adds no defaults of its own (PLAN §5.1)."""
    return load_v2_config(payload)


def input_signature(inputs: Sequence[Any]) -> str:
    """Role-aware content signature of every declared input file.

    Bound to ROLE and CONTENT, never to a path or an mtime.  A role-blind signature lets a cohort
    table and backbone exchange places without changing run identity; a path-bound signature makes
    an alias of the same bytes look like different science.
    """
    rows: list[dict[str, str]] = []
    seen_roles: set[str] = set()
    for item in inputs:
        role = getattr(item, "role", None)
        content_sha256 = getattr(item, "sha256", None)
        if not isinstance(role, str) or not role:
            raise V2PreflightError(
                "every input signature entry must carry a non-empty declared role"
            )
        if role in seen_roles:
            raise V2PreflightError(
                f"input signature role {role!r} was declared twice"
            )
        if (
            not isinstance(content_sha256, str)
            or len(content_sha256) != 64
            or any(char not in "0123456789abcdef" for char in content_sha256)
        ):
            raise V2PreflightError(
                f"input signature role {role!r} must carry a lowercase SHA-256 content digest"
            )
        seen_roles.add(role)
        rows.append({"role": role, "sha256": content_sha256})
    payload = json.dumps(sorted(rows, key=lambda row: row["role"]), sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BudgetProjection:
    """The DECLARED logical cost of a run, per PLAN §3.4, checked against the declared caps.

    Every field is derived arithmetically from the config.  None of it is a measurement, and the
    projection says so rather than implying a timing it cannot know.
    """

    n_proteins: int
    n_steps: int
    #: Conservative number of full one-cycle executions charged per protein.  Ordinary ladders
    #: use one.  A policy-qualification prefix has two descendant-generating arms, so the driver
    #: supplies ``2 * n_prefixes``; shared prefix work is intentionally over-counted rather than
    #: omitted from the launch gate.
    execution_replicates: int
    #: The depth-0 prefix, paid ONCE per protein by the ladder itself.
    root_capture_logical_dfe: int
    #: Conditional reserve for fully-resolved roots; zero work is spent when attempt one succeeds.
    root_capture_retry_reserve_logical_dfe: int
    #: Only depth 0 forks a source pool; deeper rungs inherit the previous rung's descendants.
    per_protein_screen_dfe: int
    per_protein_segment_dfe: int
    #: Every depth forks a descendant pool -- including the deepest, whose pool the old projection
    #: reached only through a next rung that does not exist.
    per_protein_descendant_screen_dfe: int
    per_protein_logical_dfe: int
    total_logical_dfe: int
    max_total_logical_dfe: int
    #: Head calls the SUPPORT POLICY makes while deciding, per protein.  Zero for every policy that
    #: does not consult the Head; the run's declared per-cycle ceiling times the declared depths for
    #: V2F5A's, which the policy enforces so the number is a bound and not a guess.
    per_protein_counterfactual_head_calls: int
    total_head_calls: int
    total_definitive_refolds: int
    breached_caps: tuple[str, ...]
    detail: str

    @property
    def feasible(self) -> bool:
        return not self.breached_caps


def project_v2_budget(
    config: Any, *, n_proteins: int, execution_replicates: int = 1, n_heads: int = 1,
    counterfactual_sequences_per_cycle: int | None = None,
) -> BudgetProjection:
    r"""Project the run's logical cost from the declared schedule.

    Per PLAN §3.4, one source checkpoint ``c`` with ``K`` lookaheads under an ``S``-step sampler
    costs ``C_screen = c + K(S - c)``, and a projected segment from ``r_d`` to ``c_{d+1}`` costs
    exactly ``c_{d+1} - r_d`` on top.  The source prefix is charged ONCE for the whole ladder --
    charging it per lookahead, or per depth, is the specific overcount PLAN §3.4 forbids.

    The projection must reconcile with what ``run_depth_ladder`` executes; the reconciliation is
    asserted against a realized forward-pass count in ``test_fusion_v2_ladder.py``.
    """
    if isinstance(n_proteins, bool) or not isinstance(n_proteins, int) or n_proteins < 0:
        raise V2PreflightError(f"n_proteins must be a non-negative int, got {n_proteins!r}")
    if (isinstance(execution_replicates, bool)
            or not isinstance(execution_replicates, int) or execution_replicates < 1):
        raise V2PreflightError(
            f"execution_replicates must be a positive int, got {execution_replicates!r}"
        )
    n_steps = int(config.substrate.n_steps)
    points = sorted(config.schedule.points, key=lambda point: int(point.depth))
    last = len(points) - 1

    # The graph is transcribed from what ``run_depth_ladder`` actually executes, not from a
    # per-depth formula that happens to look symmetric.  The two differ in exactly the ways that
    # matter: the prefix belongs to the ladder rather than to a rung, only depth 0 forks a source
    # pool, and EVERY depth forks a descendant pool -- the deepest one included, which the previous
    # projection never charged because it only saw a pool through the next rung's inherited
    # screening.  A budget that under-counts is not conservative: it authorizes a run larger than
    # the one the caps were written for.
    root_capture = int(points[0].c_source_step)
    screen = int(points[0].n_lookaheads) * (n_steps - int(points[0].c_source_step))
    segment = 0
    descendant_screen = 0
    #: One Head request and one structure attempt per GENERATED endpoint.
    scored_endpoints = int(points[0].n_lookaheads)
    #: Head calls the SUPPORT POLICY makes while deciding -- zero for every policy but V2F5A's.
    counterfactual_head_calls = 0
    for index, point in enumerate(points):
        segment += int(point.c_next_step) - int(point.r_step)
        # The ladder forks the NEXT depth's declared breadth, clamped at the last rung -- the same
        # ``lookaheads_at(min(d + 1, depth_cap - 1))`` the engine uses.
        breadth = int(points[min(index + 1, last)].n_lookaheads)
        descendant_screen += breadth * (n_steps - int(point.c_next_step))
        scored_endpoints += breadth

    root_capture *= execution_replicates
    root_retry_reserve = root_capture * int(config.caps.max_retries)
    screen *= execution_replicates
    segment *= execution_replicates
    descendant_screen *= execution_replicates
    scored_endpoints *= execution_replicates
    per_protein = root_capture + screen + segment + descendant_screen
    total = per_protein * n_proteins
    max_total = (per_protein + root_retry_reserve) * n_proteins
    # V2F5A: the Head-directed policy scores one leave-one-out counterfactual per legal write
    # candidate WHILE deciding, so a projection counting only the scored endpoints understates the
    # Head budget and ``--dry-run`` would pass a run that then breaches ``max_head_calls`` after
    # paying for a model.  The per-cycle ceiling is declared by the run and ENFORCED by the policy
    # (a source with more legal candidates stalls), so charging it here is a real bound rather than
    # an estimate.  One cycle per declared depth point.
    block = config.projection.head_directed
    # Under Dual the per-cycle CANDIDATE domain is the overlay's own C, not the legacy field --
    # they are different numbers by design (Appendix I S7), and projecting from the legacy one
    # would price a domain the run does not use. ``n_heads`` then multiplies it, so the projection
    # is 2C per cycle exactly as the freeze states.
    per_cycle = (int(block.max_counterfactual_head_calls_per_cycle) if block is not None else 0)
    if counterfactual_sequences_per_cycle is not None:
        per_cycle = int(counterfactual_sequences_per_cycle)
    counterfactual_head_calls = (
        0 if block is None else per_cycle * len(points) * execution_replicates)
    # Every Head batch is issued once PER ALLELE. A Dual run scores the same pools and the same
    # counterfactual sets on two frozen Heads, so a projection built for one Head certifies a
    # launch that will spend twice what the gate authorized against ``max_head_calls``.
    head_calls = (scored_endpoints + counterfactual_head_calls) * int(n_heads)
    refolds = scored_endpoints
    breached = []
    if max_total > int(config.caps.max_logical_dfe):
        breached.append("max_logical_dfe")
    if head_calls * n_proteins > int(config.caps.max_head_calls):
        breached.append("max_head_calls")
    if refolds * n_proteins > int(config.caps.max_definitive_refolds):
        breached.append("max_definitive_refolds")

    return BudgetProjection(
        n_proteins=int(n_proteins), n_steps=n_steps,
        execution_replicates=int(execution_replicates),
        root_capture_logical_dfe=root_capture,
        root_capture_retry_reserve_logical_dfe=root_retry_reserve,
        per_protein_screen_dfe=screen, per_protein_segment_dfe=segment,
        per_protein_descendant_screen_dfe=descendant_screen,
        per_protein_logical_dfe=per_protein, total_logical_dfe=total,
        max_total_logical_dfe=max_total,
        per_protein_counterfactual_head_calls=counterfactual_head_calls,
        total_head_calls=head_calls * n_proteins,
        total_definitive_refolds=refolds * n_proteins,
        breached_caps=tuple(breached),
        detail=(
            f"projected from the DECLARED schedule over {execution_replicates} conservative "
            f"execution replicate(s) per protein, with up to "
            f"{int(config.caps.max_retries) + 1} "
            "root-capture attempts; GPU-seconds and walltime are not "
            "projected because no measured per-refold cost is bound to this config"
        ),
    )


def assert_launch_feasible(projection: BudgetProjection) -> None:
    """Fail CLOSED before any work is paid for.

    A cap discovered mid-run has already burned the budget it was supposed to bound.
    """
    if not projection.feasible:
        raise V2PreflightError(
            f"declared schedule breaches {list(projection.breached_caps)}: projected "
            f"{projection.max_total_logical_dfe} maximum logical DFE, "
            f"{projection.total_head_calls} Head "
            f"calls, {projection.total_definitive_refolds} definitive refolds over "
            f"{projection.n_proteins} protein(s)"
        )


def _head_directed_payload(config: Any) -> dict | None:
    """The V2F5A block as ``--print-config`` reports it, or ``None`` for any other policy.

    The counterfactual count is a numeric HARD CEILING: the policy stalls before scoring when a
    source exposes more legal candidates. Offline replay measures the realized distribution, while
    preflight charges the ceiling so a launch can never pass on a fabricated smaller estimate.
    """
    block = config.projection.head_directed
    if block is None:
        return None
    return {
        "support_policy_id": config.projection.support_policy_id,
        "control_policy_id": block.control_policy_id,
        "control_policy_version": block.control_policy_version,
        "write_cap_editable_fraction": float(block.write_cap_editable_fraction.value),
        "write_cap_source_ref": block.write_cap_editable_fraction.source_ref,
        "donor_improvement_epsilon": float(block.donor_improvement_epsilon.value),
        "donor_improvement_source_ref": block.donor_improvement_epsilon.source_ref,
        "donor_improvement_measurement_kind":
            block.donor_improvement_epsilon.artifact.measurement_kind,
        "local_contribution_tolerance": float(block.local_contribution_tolerance.value),
        "local_contribution_source_ref": block.local_contribution_tolerance.source_ref,
        "local_contribution_measurement_kind":
            block.local_contribution_tolerance.artifact.measurement_kind,
        "band_center_rule": block.band_center_rule,
        "lineage_incumbent_depth0_rule": block.lineage_incumbent_depth0_rule,
        "lineage_incumbent_update_law": block.lineage_incumbent_update_law,
        "write_candidate_window_rule": block.write_candidate_window_rule,
        "reopen_count_law": block.reopen_count_law,
        "reopen_priority_law": block.reopen_priority_law,
        # ENFORCED, not advisory: the policy stalls a source whose legal-candidate count exceeds
        # this, so the projection charging it is a real bound.
        "max_counterfactual_head_calls_per_cycle":
            int(block.max_counterfactual_head_calls_per_cycle),
    }


def print_config_payload(
    config: Any, *, n_proteins: int, declared_inputs: Sequence[Any] = (),
    code_revision: str = "unknown", execution_replicates: int = 1,
    n_heads: int = 1, counterfactual_sequences_per_cycle: int | None = None,
) -> dict:
    """Everything ``--print-config`` and ``--dry-run`` report, with no model loaded.

    Includes the realized seed table's namespaces and the budget projection, so the two commands
    answer the question an operator actually has -- "what will this run do, and can it" -- rather
    than echoing the file back.
    """
    # The SAME projection the launch gate asserts on. Reporting a single-Head budget beside a
    # two-Head gate gives the operator launch evidence that understates Head spend by one allele,
    # and it is that printed number the cost-parity GO check is read against.
    projection = project_v2_budget(
        config, n_proteins=n_proteins, execution_replicates=execution_replicates,
        n_heads=n_heads,
        counterfactual_sequences_per_cycle=counterfactual_sequences_per_cycle)
    return {
        "config_digest": config.config_digest(),
        "schema_version": config.schema_version,
        "campaign_id": config.identity.campaign_id,
        "split_role": config.identity.split_role,
        "phase": config.identity.phase,
        "arm_role": config.arm.arm_role,
        "feedback_enabled": bool(config.arm.feedback_enabled),
        "coordinate_law": config.schedule.coordinate_law.value,
        "depth_cap": int(config.schedule.depth_cap),
        "active_population_width": int(config.schedule.active_population_width),
        "code_revision": str(code_revision),
        "input_signature": input_signature(declared_inputs) if declared_inputs else None,
        "substrate": {
            "n_steps": int(config.substrate.n_steps),
            "temperature": float(config.substrate.temperature),
            "amplification_form": config.substrate.amplification_form,
            "controller_enabled": bool(config.substrate.controller_enabled),
            "remask_enabled": bool(config.substrate.remask_enabled),
            "remask_fraction_scale": float(config.substrate.remask_fraction_scale),
        },
        # V2F5A.  Echoed so ``--print-config`` shows the operator the frozen cap, the calibrated
        # margins and the tie law WITHOUT loading anything; every identity behind them was already
        # validated by ``load_v2_config`` (the typed artifact, the unit against the Head's own
        # score scale, and the source_ref that binds the value to its measurement).
        "head_directed": _head_directed_payload(config),
        "budget_projection": {
            "n_proteins": projection.n_proteins,
            "execution_replicates": projection.execution_replicates,
            "root_capture_logical_dfe": projection.root_capture_logical_dfe,
            "root_capture_retry_reserve_logical_dfe":
                projection.root_capture_retry_reserve_logical_dfe,
            "per_protein_screen_dfe": projection.per_protein_screen_dfe,
            "per_protein_segment_dfe": projection.per_protein_segment_dfe,
            "per_protein_descendant_screen_dfe": projection.per_protein_descendant_screen_dfe,
            "per_protein_logical_dfe": projection.per_protein_logical_dfe,
            "per_protein_counterfactual_head_calls":
                projection.per_protein_counterfactual_head_calls,
            "total_logical_dfe": projection.total_logical_dfe,
            "max_total_logical_dfe": projection.max_total_logical_dfe,
            "total_head_calls": projection.total_head_calls,
            "total_definitive_refolds": projection.total_definitive_refolds,
            "feasible": projection.feasible,
            "breached_caps": list(projection.breached_caps),
            "detail": projection.detail,
        },
        "caps": {
            "max_logical_dfe": int(config.caps.max_logical_dfe),
            "max_head_calls": int(config.caps.max_head_calls),
            "max_definitive_refolds": int(config.caps.max_definitive_refolds),
            "max_gpu_seconds": int(config.caps.max_gpu_seconds),
            "max_walltime_s": int(config.caps.max_walltime_s),
            "max_retries": int(config.caps.max_retries),
        },
    }
