"""V2F7: the per-protein execution stage the driver calls (PLAN V2F7).

**Verification boundary, stated first.**  This module is the one place V2 touches real models, so it
is the one place this repo cannot exercise end to end: it needs torch, a DPLM checkpoint, a Head
checkpoint and PDB inputs.  Its contract with the driver -- ``(status, payload)`` and every exit
condition that follows from it -- IS verified, because the driver's suite drives the whole matrix
through an injected runner.  What is NOT verified locally is that the real oracles behave; that
remains a cluster check, and no local green result should be read as evidence for it.

Everything expensive is built through :mod:`scripts.rf_fusion_model_factory`, the same factory the
V1 oracle path now delegates to, so the two entry paths cannot drift into running different kernels
while both claiming the frozen substrate.
"""

from __future__ import annotations

import dataclasses
import functools
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion_v2.errors import V2Error  # noqa: E402

__all__ = ["V2CohortError", "ShardInputs", "assert_runtime_substrate_matches",
           "build_depth_plan", "run_v2_shard"]

#: Cycle arguments the RUN owns, never the oracle factory.  ``feedback_enabled`` is the arm
#: identity and ``cost_meter`` is the compute journal: a factory able to set either could make a
#: run's engine and its own artifact disagree about which experiment was performed.
#: ``dual`` is deliberately NOT one of these.  The factory is where role B's Head is opened, so the
#: Dual runtime is something it legitimately produces -- what the run owns is the DECISION to run a
#: Dual arm at all, which is checked separately: the driver's resolved overlay and the factory's
#: returned runtime must agree, or a run whose signature claims a Dual arm could execute A-only.
RUN_OWNED_CYCLE_KWARGS = ("feedback_enabled", "cost_meter")

#: Every ``config.substrate`` field that has a counterpart on the runtime ``ReferenceFlowConfig``
#: the oracles factory returns, and where that counterpart lives.
#:
#: These two objects are separate: the V2 config is the DECLARED science (it is what the manifest
#: publishes, what the config digest covers and what the preflight projects a budget from), while
#: the ``ReferenceFlowConfig`` is what the sampler actually obeys.  Nothing compared them, so a
#: 140-step runtime under a 100-step declaration burned 654 logical DFE against a launch gate that
#: had authorized 374 -- and a temperature the digest said was 1.0 silently rescaled every
#: assimilated token log-probability.  PLAN §3.2 requires a mismatch in "sampler steps ... or
#: policy digest" to "fail before a denoiser call".
#:
#: ``rf_config_label`` is deliberately absent: it names the config, it is not a kernel parameter,
#: and a runtime object carries nothing to compare it against.
SUBSTRATE_RUNTIME_FIELDS = (
    ("n_steps", ("sampler", "n_steps")),
    ("temperature", ("sampler", "temperature")),
    ("amplification_form", ("amplification", "form")),
    ("remask_enabled", ("sampler", "remask", "enabled")),
    ("remask_fraction_scale", ("sampler", "remask", "fraction_scale")),
)

#: How many extra exact futures this driver hands an A2 view (PLAN §4.3).  ZERO -- the ladder
#: allocates the breadth its schedule declares and nothing more, and no matched reallocation is
#: implemented.  Named rather than inlined so the artifact's claim has one source: what the run
#: EXECUTED, which the allocation record then compares against what matching would owe.
EXECUTED_A2_EXTRA_LOOKAHEADS = 0

#: The ladder's own root-prefix event.  It is charged before the first rung exists, so it is not
#: reachable through any rung's transition id and is attributed to depth 0 explicitly.
LADDER_ROOT_EVENT_SUFFIX = ":ladder_root"


class V2CohortError(V2Error):
    """A shard could not be assembled from what the run declared."""


class ShardInputs:
    """The runtime paths a real shard needs, kept out of the scientific config.

    PLAN forbids hardcoding cluster paths in modules, and they are not scientific parameters: two
    runs of the same science on two machines have different paths and the same config digest.  They
    therefore arrive from the CLI and are carried here rather than folded into ``V2Config``.
    """

    def __init__(self, **paths: Any) -> None:
        self.paths = dict(paths)

    def require(self, name: str) -> Any:
        value = self.paths.get(name)
        if not value:
            raise V2CohortError(
                f"--{name.replace('_', '-')} is required to execute a shard; "
                "--print-config and --dry-run do not need it"
            )
        return value


def build_depth_plan(config: Any):
    """Turn the config's declared schedule into the ladder's :class:`DepthPlan`.

    Breadth and coordinates are DECLARED, never allocated (PLAN V2F6): this is a transcription, and
    anything it had to infer would be a scientific decision smuggled into a driver.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.ladder import DepthPlan

    points = sorted(config.schedule.points, key=lambda point: int(point.depth))
    return DepthPlan(
        law=config.schedule.coordinate_law,
        depth_cap=int(config.schedule.depth_cap),
        cycles=tuple(
            (int(p.r_step), int(p.c_source_step), int(p.c_next_step)) for p in points),
        lookaheads_per_depth=tuple(int(p.n_lookaheads) for p in points),
        n_steps=int(config.substrate.n_steps),
    )


def _journal_path(
    *, inputs: ShardInputs, out_dir: Any, protein_id: str, run_signature: str,
) -> Path:
    """Where this shard's attempt journal lives.

    A journal is a RUNTIME path, so it arrives from the CLI -- either declared directly as
    ``--journal-dir`` (carried on :class:`ShardInputs`) or derived from the run's own ``out_dir``.
    Nothing is hardcoded here, and a shard with neither is refused rather than silently journaling
    into the working directory, where the next run would append to the same file.
    """
    declared = inputs.paths.get("journal_dir")
    if not declared:
        if not out_dir:
            raise V2CohortError(
                "a shard needs somewhere to journal its oracle requests: pass --out-dir, or a "
                "journal_dir on ShardInputs.  PLAN §5.4 journals a request BEFORE execution, so "
                "the file must exist before the first oracle call rather than be chosen by "
                "whatever the process's working directory happens to be"
            )
        declared = Path(out_dir) / "journals"
    return Path(declared) / run_signature / f"{protein_id}.attempts.jsonl"


def _validate_run_signature(
    *, signature: Any, config: Any, protein_id: str,
    production_depth_authorized: bool = False,
    exploratory_depth_override: bool = False,
):
    """Return a formal signature only when it describes this exact shard."""
    from scripts.rf_fusion_v2_resume import RunSignature

    if not isinstance(signature, RunSignature):
        raise V2CohortError(
            "signature must be a formal RunSignature; an unbound journal can mix attempts from "
            "different runs"
        )
    expected = {
        "config_digest": config.config_digest(),
        "campaign_id": config.identity.campaign_id,
        "split_role": config.identity.split_role,
        "arm_role": config.arm.arm_role,
        "protein_id": str(protein_id),
        "code_revision": config.identity.code_revision,
        "production_depth_authorized": bool(production_depth_authorized),
        "exploratory_depth_override": bool(exploratory_depth_override),
    }
    drift = [
        name for name, value in expected.items()
        if getattr(signature, name) != value
    ]
    if drift:
        raise V2CohortError(
            f"run signature contradicts this shard on {drift}; refusing a foreign journal "
            "namespace"
        )
    return signature


def _gpu_clock(oracles: Mapping[str, Any]):
    """The instrument this run's GPU-seconds are measured with.

    ``CostMeter`` requires one rather than defaulting, because ``check_caps`` treats GPU-seconds as
    a MEASURED quantity: a fabricated ``0.0`` would let a GPU cap read as satisfied on a number
    nobody took.  The factory owns device placement, so it is the one thing that can name the
    instrument; when it does not, a process that has never initialized CUDA truly burned no GPU
    seconds and may say so -- and a process that HAS is refused rather than credited with zero.
    """
    declared = oracles.get("gpu_clock")
    if declared is not None:
        return declared
    try:
        import torch
    except ImportError:                                  # a torch-free fake stack burns no GPU
        return lambda: 0.0
    if torch.cuda.is_initialized():
        raise V2CohortError(
            "this process has initialized CUDA but the oracles factory declared no gpu_clock, so "
            "the run cannot say what measured its GPU seconds; reporting 0.0 would certify the "
            "max_gpu_seconds cap against a number nobody took"
        )
    return lambda: 0.0


def _policy_identity(record, *, config: Any):
    """The identity carried by the DECISION this rung acted on, per rung.

    Read off ``CycleOutcome.policy_identity`` rather than off the policy object, because those are
    two separate sources that can disagree: a callable can name itself one thing and stamp its
    answers with another, and the artifact must record which policy produced THIS transition.  It
    is also per rung rather than per run -- a ladder records one event per depth, and one identity
    reused across all of them could not show a policy swapped mid-ladder.

    PLAN §5.3 makes the policy a load-bearing field of every feedback event, so a feedback arm that
    committed a projection under an unattributable answer is refused: null policy columns would
    leave the one table that records what the policy did unable to say which policy it was.
    """
    identity = record.cycle.policy_identity
    if identity is None and record.cycle.projected is not None and bool(
        config.arm.feedback_enabled
    ):
        raise V2CohortError(
            "a projection was committed under a policy answer that carries no identity: PLAN §5.3 "
            "makes the policy a load-bearing field of every feedback event, and an anonymous "
            "answer makes two runs under two different policies indistinguishable in the one "
            "table that records what the policy did"
        )
    return identity


def _endpoints_by_depth(outcome) -> list[tuple[Any, int]]:
    """Every endpoint the ladder produced, each with the depth it was generated at.

    A rung's descendant pool IS the next rung's source pool (the ladder inherits it rather than
    re-screening it), so the two views must be merged: taking only ``cycle.endpoints`` loses the
    DEEPEST rung's descendants entirely, because no later rung ever inherits them.  Merging without
    de-duplicating would instead write an inherited pool twice and duplicate its join key, which
    silently multiplies rows in every downstream join.
    """
    by_id: dict[str, tuple[Any, int]] = {}
    for record in outcome.cycles:
        depth = int(record.depth)
        for endpoint in record.cycle.endpoints:
            by_id.setdefault(endpoint.endpoint_id, (endpoint, depth))
        for endpoint in record.cycle.descendant_endpoints:
            by_id.setdefault(endpoint.endpoint_id, (endpoint, depth + 1))
    return list(by_id.values())


def _partial_states(outcome) -> list[Any]:
    """Source, projected and propagated states, once each.

    The projected state is what ``q_phi`` produced at ``r_d``; a table with only the source and the
    propagated capture cannot show what the projection did between them.  De-duplicated because a
    rung's source IS the previous rung's propagated capture -- one state, one row.
    """
    by_id: dict[str, Any] = {}
    for record in outcome.cycles:
        for state in (record.cycle.source, record.cycle.projected, record.cycle.propagated):
            if state is not None:
                by_id.setdefault(state.state_id, state)
    return list(by_id.values())


def _admissions(outcome) -> dict[str, Any]:
    """Every admission decision the run made, by endpoint.

    Both pools: a rung that inherited its source pool records no admissions of its own, because the
    decisions were made when that pool was the PREVIOUS rung's descendants.
    """
    decisions: dict[str, Any] = {}
    for record in outcome.cycles:
        for admission in (record.cycle.admissions or ()) + (
                record.cycle.descendant_admissions or ()):
            decisions[admission.endpoint_id] = admission
    return decisions


def _terminal_records(outcome, admissions: Mapping[str, Any]) -> list[dict]:
    """One record per design that reached definitive feasibility -- what the run returns.

    Structure and immune results stay in SEPARATE columns because they are independent measurements
    of different failure modes.  The immune result recorded here is the IN-LOOP whole-landscape gate
    (``AdmissibilityVerdict.cumulative``), and ``immune_evaluator`` names the evaluator that
    produced it: a reader joining it against ``complete_endpoints.head_evaluator_digest`` can see
    that it is the same evaluator, i.e. that no independent re-evaluation has been recorded yet.
    Writing nothing at all would instead read as "this run validated none of its designs".
    """
    archive = outcome.archive
    records: list[dict] = []
    for endpoint in archive.endpoints():
        endpoint_id = endpoint.endpoint_id
        if not archive.may_become_ancestry(endpoint_id):
            continue
        structure = archive.structure_outcome(endpoint_id)
        verdict = getattr(admissions.get(endpoint_id), "verdict", None)
        from inverse_folding.reference_flow.fusion_v2.safety import (  # noqa: PLC0415
            IncrementalInapplicable,
        )

        incremental = None if verdict is None else verdict.incremental
        immune_passed = None if verdict is None else (
            verdict.cumulative.passed
            and (incremental is None or isinstance(incremental, IncrementalInapplicable)
                 or incremental.passed))
        # The MEASUREMENT ancestry eligibility actually turned on (PLAN §2.7), not just its
        # verdict.  A bool plus a reason string cannot be re-checked: a reviewer could see THAT a
        # design was admitted and never what N_H^whole was, against which reference, over how many
        # windows.  Null where no verdict exists, never zero -- an unmeasured landscape and a flat
        # one are different claims.
        cumulative_evidence = None if verdict is None else verdict.cumulative.evidence
        records.append({
            "endpoint_id": endpoint_id,
            "sequence_md5": endpoint.sequence_md5,
            "structure_definitive": (bool(getattr(structure, "evaluated", False))
                                     and bool(getattr(structure, "feasible", False))),
            "structure_feasible": bool(getattr(structure, "feasible", False)),
            "structure_metrics": dict(getattr(structure, "metrics", None) or {}),
            "immune_evaluator": endpoint.head_binding.evaluator.digest(),
            "immune_global_risk": float(endpoint.head_global_risk),
            "immune_passed": bool(immune_passed),
            "diversity_family_id": endpoint.lineage.family_id,
            "whole_landscape_max_increase": (
                None if cumulative_evidence is None
                else float(cumulative_evidence.max_increase)),
            "whole_landscape_positive_mass": (
                None if cumulative_evidence is None
                else float(cumulative_evidence.positive_mass)),
            "whole_landscape_positive_count": (
                None if cumulative_evidence is None
                else int(cumulative_evidence.positive_count)),
            "whole_landscape_n_windows": (
                None if cumulative_evidence is None else int(cumulative_evidence.n_windows)),
            "whole_landscape_reference_binding_id": (
                None if cumulative_evidence is None
                else cumulative_evidence.reference_binding_id),
            "incremental_gate_status": (
                None if verdict is None
                else ("inapplicable_depth0" if isinstance(incremental, IncrementalInapplicable)
                      else ("not_enabled" if incremental is None
                            else ("passed" if incremental.passed else "failed")))),
            "v0_before_after": None,
        })
    return records


def _ledger_rows(journal_path: Path) -> list[dict]:
    """This shard's realized compute ledger, as plain JSON rows.

    Dataclasses would not survive the fragment, which is JSON on disk, so the payload carries the
    ``asdict`` form; :func:`write_v2_bundle` re-checks each row against the typed event contract
    before publishing it.  Reconstructed from the JOURNAL rather than accumulated in memory, so an
    attempt that was opened and never closed is reported as ``unknown_after_start`` (PLAN §5.4)
    instead of vanishing from the run's own account of what it burned.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import events_from_journal

    if not journal_path.exists():
        return []
    return [dataclasses.asdict(event) for event in events_from_journal(journal_path)]


def assert_runtime_substrate_matches(config: Any, runtime: Any) -> None:
    """Bind the DECLARED V2 substrate to the runtime sampler config (PLAN §3.2, §5.1).

    PLAN §3.2: "Tampering or a mismatch in sampler steps, tokenizer, backbone, coordinate mask,
    constraints, reference sequence, runtime null sentinels, or policy digest must fail BEFORE a
    denoiser call."  The declared substrate was reaching only the ``DepthPlan`` and the preflight
    budget projection; the sampler ran against the separate ``ReferenceFlowConfig`` the oracles
    factory returns, and nothing compared them.  Measured consequence: a 140-step runtime against a
    100-step declaration burned 654 logical DFE under a launch gate that had authorized 374 -- a
    75% under-projection of every DFE, Head and refold cap -- and a runtime temperature of 3.0
    against a declared 1.0 was accepted silently, so first-forward assimilation computed token
    log-probabilities at a temperature the config digest and the manifest both reported as 1.0.

    Called from ``run_v2_shard`` before the ladder is entered, because the ladder pays a root
    prefix of real forward passes before its first cycle: a check inside the engine would refuse
    the run only after buying the compute the gate exists to authorize.
    """
    if runtime is None:
        raise V2CohortError(
            "the oracles factory supplied no runtime config, so the declared substrate has "
            "nothing to be checked against; PLAN §5.2 fails closed on missing content identity "
            "rather than proceeding unchecked"
        )
    declared = config.substrate
    observed = (
        ("n_steps", int(declared.n_steps), int(runtime.sampler.n_steps)),
        ("temperature", float(declared.temperature), float(runtime.sampler.temperature)),
        ("amplification_form", str(declared.amplification_form),
         str(runtime.amplification.form)),
        ("remask_enabled", bool(declared.remask_enabled), bool(runtime.sampler.remask.enabled)),
        ("remask_fraction_scale", float(declared.remask_fraction_scale),
         float(runtime.sampler.remask.fraction_scale)),
    )
    drift = [(name, d, r) for name, d, r in observed if d != r]
    if drift:
        raise V2CohortError(
            "the runtime sampler config contradicts the declared V2 substrate on "
            + ", ".join(f"{name} (declared {d!r}, runtime {r!r})" for name, d, r in drift)
            + "; the config digest and the run manifest would describe a substrate the run did "
              "not use, and the launch gate would have been evaluated against it"
        )
    # The one substrate field whose runtime counterpart is an ABSENCE: ``ReferenceFlowConfig`` has
    # no controller surface, so a factory that attached one is declaring a different kernel.
    controller = getattr(runtime, "controller", None)
    if bool(getattr(controller, "enabled", False)) != bool(declared.controller_enabled):
        raise V2CohortError(
            f"the runtime config declares controller_enabled="
            f"{bool(getattr(controller, 'enabled', False))} but the V2 config declares "
            f"{bool(declared.controller_enabled)}; every V2/A2 arm is controller-free (PLAN §5.1) "
            "and a controller is a different kernel, not a different setting"
        )


def run_v2_shard(
    *, protein_id: str, config: Any, signature: Any, out_dir: Any,
    inputs: ShardInputs | None = None,
    oracles_factory=None, production_depth_authorized: bool = False,
    exploratory_depth_override: bool = False,
    dual: Any = None,
) -> tuple[str, Mapping[str, Any]]:
    """Run one protein's V2 ladder and return ``(status, payload)`` for the driver's fragment.

    Returns rather than raises on a scientific null: PLAN §4.5 makes a stopped ladder a RESULT, and
    a shard that raised would be indistinguishable from a crash in the driver's accounting.  A
    genuine wiring fault still raises -- the driver catches :class:`V2Error` and records a failed
    fragment, which keeps the two distinguishable in the artifact.

    **The config is the runtime authority.**  ``feedback_enabled`` is injected here from
    ``config.arm``, not defaulted by the engine and not supplied by the oracles factory.  A factory
    that carried either it or the ``cost_meter`` in its ``cycle_kwargs`` is REFUSED rather than
    overridden: letting the config silently win would still leave a run whose declared oracle stack
    disagrees with the arm it ran, and the refusal names the conflict at the one point where both
    values are visible.  Silent precedence, in either direction, is invisible in the artifact.
    """
    from scripts.rf_fusion_v2_artifacts import (
        a2_view_rows,
        archive_rows,
        complete_endpoint_rows,
        feedback_event_rows,
        partial_state_rows,
        structure_evaluation_rows,
        terminal_validation_rows,
    )

    inputs = inputs or ShardInputs()
    if oracles_factory is None:
        raise V2CohortError(
            "no oracles_factory was supplied.  The real V2 oracle stack (DPLM denoiser, frozen "
            "Head, structure gate) is assembled behind an explicit factory so that importing this "
            "module, printing the config and dry-running never load a model; a shard that built "
            "them implicitly would make --dry-run cost a GPU allocation."
        )

    signature = _validate_run_signature(
        signature=signature, config=config, protein_id=protein_id,
        production_depth_authorized=production_depth_authorized,
        exploratory_depth_override=exploratory_depth_override,
    )

    plan = build_depth_plan(config)
    # Passed only when a Dual overlay was actually resolved, so the long-standing three-argument
    # factory surface is untouched for every legacy run and every injected fake.
    factory_kwargs = dict(protein_id=protein_id, config=config, inputs=inputs)
    if dual is not None:
        factory_kwargs["dual"] = dual
    oracles = oracles_factory(**factory_kwargs)
    cycle_kwargs = dict(oracles["cycle_kwargs"])
    dual_runtime = cycle_kwargs.get("dual")
    if (dual is None) != (dual_runtime is None):
        raise V2CohortError(
            f"the driver resolved dual={dual is not None} but the oracle factory returned "
            f"dual={dual_runtime is not None}; a run whose signature claims a Dual arm and whose "
            "engine executes A-only would produce three identical arms under three signatures"
        )
    conflicting = [name for name in RUN_OWNED_CYCLE_KWARGS if name in cycle_kwargs]
    if conflicting:
        raise V2CohortError(
            f"the oracles factory supplied {conflicting}, which the RUN owns: "
            "feedback_enabled is this run's arm identity and comes from config.arm, and cost_meter "
            "is the journal the cohort aggregates.  A factory that could set either would let the "
            "engine and the artifact describe two different experiments"
        )

    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import (
        AttemptJournal,
        CostMeter,
        V2LedgerEvent,
        aggregate_v2_ledger,
        check_caps,
    )

    journal_path = _journal_path(
        inputs=inputs, out_dir=out_dir, protein_id=protein_id,
        run_signature=signature.value,
    )
    cost_meter = CostMeter(
        journal=AttemptJournal(journal_path), protein_id=str(protein_id),
        # The ARM this shard's every attempt is charged to, taken from the config that declared it.
        arm=str(config.arm.arm_role), gpu_clock=_gpu_clock(oracles),
    )

    # BEFORE the ladder: it pays a root prefix of real forward passes before its first cycle.
    assert_runtime_substrate_matches(config, oracles["cycle_kwargs"].get("config"))

    from inverse_folding.reference_flow.fusion_v2_runtime.ladder import (
        StoppingReason,
        run_depth_ladder,
    )

    outcome = run_depth_ladder(
        plan=plan,
        campaign_id=config.identity.campaign_id,
        split_role=config.identity.split_role,
        phase=config.identity.phase,
        master_seed=int(config.identity.master_seed),
        allow_production_depth_gt_1=bool(production_depth_authorized),
        exploratory_depth_override=bool(exploratory_depth_override),
        root_capture_max_retries=int(config.caps.max_retries),
        feedback_enabled=bool(config.arm.feedback_enabled),
        # Same rule as the arm identity: the run's FROZEN declaration is what the kernel matches
        # the answering policy against, so it comes from the config and never from the factory.
        declared_policy=config.declared_policy(),
        cost_meter=cost_meter,
        **cycle_kwargs,
    )

    endpoint_rows: list[dict] = []
    by_depth: dict[int, list] = {}
    for endpoint, depth in _endpoints_by_depth(outcome):
        by_depth.setdefault(depth, []).append(endpoint)
    for depth in sorted(by_depth):
        endpoint_rows.extend(complete_endpoint_rows(by_depth[depth], depth=depth))

    events = [
        dict(source=record.cycle.source, endpoint=record.cycle.selected_endpoint,
             projected=record.cycle.projected, propagated=record.cycle.propagated,
             policy=_policy_identity(record, config=config),
             policy_evidence=getattr(record.cycle, "policy_evidence", None),
             outcome=record.cycle.outcome.value, detail=record.cycle.detail,
             pair_id=None, arm_slot=None, treatment_identity=config.arm.arm_role)
        for record in outcome.cycles
    ]
    admissions = _admissions(outcome)
    ledger_events = _ledger_rows(journal_path)
    # Checked against the run's OWN realized ledger, not against a projection: a shard that spent
    # past a declared hard cap produced a result outside the protocol the cohort is reporting.
    # ``unverifiable`` is recorded but does not by itself fail the protein -- it means an earlier
    # attempt burned an unmeasured amount, and failing on it would freeze a protein as failed for
    # every future re-run of a cohort that a preemption had already interrupted once.
    verdict = check_caps(
        aggregate_v2_ledger(tuple(V2LedgerEvent(**row) for row in ledger_events)), config.caps)

    payload = {
        "complete_endpoints": endpoint_rows,
        "partial_states": partial_state_rows(_partial_states(outcome)),
        "feedback_events": feedback_event_rows(events),
        "archive": archive_rows(
            outcome.archive,
            admission_by_endpoint={
                endpoint_id: admission.reason for endpoint_id, admission in admissions.items()},
        ),
        # Every structure verdict, PASSED OR FAILED.  A rejected endpoint keeps
        # ``structure_evaluated=false`` and empty metrics by design, so without this table a cell
        # that admitted nothing cannot say WHY from its own bundle -- the first Canary's
        # `5zhv_r30` folded four endpoints, rejected all four, and recorded no scTM anywhere.
        "structure_evaluations": structure_evaluation_rows(
            outcome.archive,
            admission_by_endpoint={
                endpoint_id: admission.reason for endpoint_id, admission in admissions.items()},
            conditioning=oracles["cycle_kwargs"].get("conditioning"),
        ),
        "a2_views": a2_view_rows(
            [record.cycle.a2_view for record in outcome.cycles],
            matched_extra_lookaheads=EXECUTED_A2_EXTRA_LOOKAHEADS,
            matching_resource=None,
            declared_matching_resource=config.arm.a2_matching_resource,
            matching_executed=False,
            unmatched_components={
                "declared_unmatched": list(config.arm.a2_unmatched_reported)},
        ),
        "terminal_validation": terminal_validation_rows(_terminal_records(outcome, admissions)),
        # Empty, and true: a ladder run has no matched arms, so it scores no mechanism contrast.
        # Declared rather than omitted, so the bundle carries the same tables either way.
        "mechanism_contrasts": [],
        "ledger_events": ledger_events,
        "cap_verdict": {
            "within": bool(verdict.within),
            "breached": list(verdict.breached),
            "unverifiable": list(verdict.unverifiable),
            "detail": verdict.detail,
        },
        "stopping_reason": outcome.stopping_reason.value,
        "depth_reached": int(outcome.depth_reached),
        "total_logical_dfe": int(outcome.total_logical_dfe),
        "substrate_digest": outcome.substrate_digest,
        "production_depth_authorized": bool(outcome.production_depth_authorized),
        "exploratory_depth_override": bool(outcome.exploratory_depth_override),
        "root_capture": {
            "attempts_used": int(getattr(outcome, "root_capture_attempts_used", 0)),
            "final_seed": getattr(outcome, "root_capture_seed", None),
            "status": str(getattr(outcome, "root_capture_status", "unknown")),
            "n_unresolved_editable": getattr(
                outcome, "root_capture_n_unresolved_editable", None),
            "rho_edit": getattr(outcome, "root_capture_rho_edit", None),
            "detail": str(getattr(outcome, "root_capture_detail", "")),
        },
    }
    if dual_runtime is not None:
        # Emitted only by a Dual run, and into a SEPARATE registry, so a legacy bundle gains no
        # file. Every number here was read off the objects the cycle and the policy actually
        # decided with -- recomputing them from the legacy tables would be a second measurement of
        # the same run, and the two could disagree with nothing to say which one it acted on.
        from scripts.rf_fusion_v2_artifacts import (
            dual_endpoint_evidence_rows,
            dual_feedback_evidence_rows,
            dual_terminal_summary_rows,
        )

        arm = dual_runtime.arm
        # TOP-LEVEL keys, one per table, exactly like the legacy tables.  Nested under a
        # ``dual_tables`` dict they were invisible to ``aggregate_fragments``, which collects
        # payload keys BY TABLE NAME -- so every Dual shard built its three tables and the driver
        # dropped all of them, and no production run ever wrote a Dual parquet.
        dual_rows = {
            "dual_endpoint_evidence": dual_endpoint_evidence_rows(
                dual_runtime.evidence_by_endpoint, arm=arm),
            "dual_feedback_evidence": dual_feedback_evidence_rows(events, arm=arm),
            "dual_terminal_summary": dual_terminal_summary_rows(
                dual_runtime.evidence_by_endpoint, arm=arm, protein_id=str(protein_id),
                root_id=f"{protein_id}:v2:d0:r0",
                n_definitive=sum(1 for a in admissions.values()
                                 if getattr(a, "admitted", False)),
                d0_endpoint_ids=[str(e.endpoint_id)
                                 for e in by_depth.get(0, ())],
                last_improving_depth=_last_improving_depth(
                    by_depth, dual_runtime.evidence_by_endpoint),
                typed_stop=_terminal_stop_label(outcome)),
        }
        payload.update(dual_rows)
        payload["dual_arm"] = arm
    # "ok" means the ladder advanced at least one depth, left a definitive design behind, and
    # stayed inside every declared hard cap.  A typed stop with nothing definitive is a real result
    # and a real FAILURE for this protein: the cohort must be able to count it as such rather than
    # as a success with an empty table.
    completed = outcome.depth_reached > 0 or \
        outcome.stopping_reason is StoppingReason.TERMINAL_BEST_LOOKAHEAD
    status = "ok" if (completed and outcome.best_definitive is not None
                      and not verdict.breached) else "failed"
    return status, payload


# --------------------------------------------------------------------------------------------
# the mechanism cohort (runbook §7)
# --------------------------------------------------------------------------------------------

#: Namespace for the per-prefix source seeds.  Separate from every namespace the ladder uses, so a
#: mechanism prefix and a ladder root with the same ordinal cannot collide on a seed.
MECHANISM_SEED_NAMESPACE = "v2_mechanism_source"


class _ArmRecord:
    """The two fields ``_endpoints_by_depth`` / ``_partial_states`` / ``_admissions`` read."""

    __slots__ = ("depth", "cycle")

    def __init__(self, depth: int, cycle: Any) -> None:
        self.depth, self.cycle = int(depth), cycle


class _ArmOutcome:
    """One executed arm, shaped like the ladder outcome the row helpers already consume.

    Written as an adapter rather than as a second set of row builders: the mechanism cohort's
    endpoints, states, admissions and terminal records mean exactly what a ladder's do, and a
    parallel implementation is how two tables that claim the same thing start to disagree.
    """

    __slots__ = ("cycles", "archive")

    def __init__(self, cycle: Any) -> None:
        self.cycles = (_ArmRecord(0, cycle),)
        self.archive = cycle.archive


def _first_by(rows: Any, key: str) -> list[dict]:
    """De-duplicate on a join key, first write wins.

    Every arm forks ONE shared archive, so the shared depth-0 pool appears in all of them.  That is
    the design -- it is what makes the arms comparable -- but a bundle that wrote it once per arm
    would duplicate the join key and multiply every downstream join.
    """
    seen: dict[Any, dict] = {}
    for row in rows:
        seen.setdefault(row[key], row)
    return list(seen.values())


def _depth0_point(config: Any):
    for point in config.schedule.points:
        if int(point.depth) == 0:
            return point
    raise V2CohortError("config.schedule declares no depth-0 point; the mechanism cohort runs "
                        "exactly one cycle and has nowhere to start")


def _terminal_stop_label(outcome) -> str:
    """Preserve the cycle's precise reason when the ladder groups projection refusals."""
    if outcome.stopping_reason.value == "invalid_projection" and outcome.cycles:
        cycle = outcome.cycles[-1].cycle
        stall_reason = getattr(cycle.policy_evidence, "stall_reason", None)
        if stall_reason:
            return str(stall_reason)
        return cycle.outcome.value
    return outcome.stopping_reason.value


def _last_improving_depth(by_depth, evidence_by_endpoint) -> int | None:
    """The deepest rung whose best joint value STRICTLY improved on every shallower rung.

    ``None`` when no rung improved on depth zero -- an unmeasured quantity, not a zero. This column
    was filled with the ladder's ``depth_reached``, which is how far the loop got and says nothing
    about whether J ever moved: a run that reached depth 4 and improved only at depth 1 published
    "4", and every per-depth improvement rate read off this table would have been computed on the
    depth cap.
    """
    best_by_depth: dict[int, float] = {}
    for depth, endpoints in by_depth.items():
        values = [float(evidence_by_endpoint[key].risk.value) for key in
                  (str(getattr(e, "endpoint_id")) for e in endpoints)
                  if key in evidence_by_endpoint]
        if values:
            best_by_depth[int(depth)] = min(values)
    running: float | None = None
    last: int | None = None
    for depth in sorted(best_by_depth):
        if running is None or best_by_depth[depth] < running:
            running = best_by_depth[depth]
            last = depth
    # Depth zero is the baseline, not an improvement over anything.
    return None if last is None or last == min(best_by_depth, default=0) else last


def run_v2_mechanism_shard(
    *, protein_id: str, config: Any, signature: Any, out_dir: Any,
    inputs: ShardInputs | None = None,
    oracles_factory=None,
    n_prefixes: int = 16,
    prefix_start: int = 0,
    views: Any = None,
    qualification: bool = False,
    dual: Any = None,
) -> tuple[str, Mapping[str, Any]]:
    """Runbook §7: ``n_prefixes`` INDEPENDENT source prefixes, each through the matched arms.

    The Canary showed the feedback machinery executes.  This shard asks the question the Canary
    could not: does the identity that was projected back actually reach the completed descendant?
    Each prefix is carried once through :func:`run_mechanism_views`, which pays for the source
    prefix and its lookahead pool ONCE and forks every arm off that single realized pool -- so the
    arms differ in the intervention and in nothing else, including in what they were compared
    against.

    **The source prefix is the sampling unit.**  Endpoints of one prefix are correlated (measured
    ICC 0.37-0.51 on the resumed null), so the prefix index is carried on every row and the
    analysis averages within a prefix before treating prefixes as independent.  A prefix that
    produced no scorable contrast still emits a row saying why: a silently absent prefix is
    indistinguishable from one that was never attempted.
    """
    from scripts.rf_fusion_v2_artifacts import (
        MECHANISM_CONTRAST_VIEWS,
        POLICY_QUALIFICATION_CONTRAST_VIEWS,
        a2_view_rows,
        archive_rows,
        complete_endpoint_rows,
        feedback_event_rows,
        mechanism_contrast_rows,
        partial_state_rows,
        structure_evaluation_rows,
    )
    from inverse_folding.reference_flow.fusion_v2.identity import (
        canonical_digest,
        make_transition_id,
    )
    from inverse_folding.reference_flow.fusion_v2.seeds import (
        V2_SEED_ENCODING_VERSION,
        FeedbackPairSeedContext,
        derive_seed,
    )
    from inverse_folding.reference_flow.fusion_v2_runtime.paired import (
        run_mechanism_views,
        run_policy_qualification_view,
    )
    from scripts.rf_fusion_v2_oracles import build_source_geometry_control

    inputs = inputs or ShardInputs()
    if oracles_factory is None:
        raise V2CohortError("no oracles_factory was supplied; see run_v2_shard")
    n_prefixes = int(n_prefixes)
    if n_prefixes < 1:
        raise V2CohortError(f"n_prefixes must be >= 1, got {n_prefixes}")
    # A second BATCH continues the prefix sequence instead of repeating it.  Seeds are derived from
    # the prefix index, so re-running from 0 would reproduce batch 1 exactly -- identical results
    # for the price of the compute, and colliding `source_index` values if the two bundles are then
    # read together.  Offsetting keeps the sampling unit's identity unique across batches, which is
    # what lets an internal pilot analyse both batches as one sample.
    prefix_start = int(prefix_start)
    if prefix_start < 0:
        raise V2CohortError(f"prefix_start must be >= 0, got {prefix_start}")
    qualification = bool(qualification)
    default_views = (POLICY_QUALIFICATION_CONTRAST_VIEWS if qualification
                     else MECHANISM_CONTRAST_VIEWS)
    requested_views = tuple(views) if views is not None else default_views
    unknown = [v for v in requested_views if v not in default_views]
    if unknown:
        raise V2CohortError(
            f"{unknown} name no scorable contrast for this run; expected a subset of "
            f"{list(default_views)}"
        )
    if qualification and config.projection.head_directed is None:
        # The contrast IS the two support laws, so a config that declares no Head-directed policy
        # has only one law to run and the comparison does not exist.  Refused here rather than at
        # the first prefix, so the operator learns it before the shard pays for a model.
        raise V2CohortError(
            "--qualification needs a config declaring projection.head_directed: the contrast is "
            "the Head-directed policy against its matched source-geometry control, and this "
            f"config's support_policy_id is {config.projection.support_policy_id!r}"
        )

    signature = _validate_run_signature(signature=signature, config=config, protein_id=protein_id)

    if dual is not None:
        # This engine is the V2F5A mechanism/qualification contrast: two arms that differ by
        # INTERVENTION, inside ONE run, sharing one realized pool. Allele arms are not compared
        # that way. The Dual arm enters no seed, so two runs differing only in --dual-arm produce
        # an identical depth-zero pool (asserted in test_fusion_v2_dual_arms.py) -- the comparison
        # is already paired, across runs, for free. Threading a Dual runtime through here would
        # give BOTH mechanism arms the same allele law: it answers no allele question while
        # looking like it does.
        raise V2CohortError(
            "the mechanism/qualification shard does not take a Dual overlay: its two arms differ "
            "by intervention inside one run, while allele arms are compared ACROSS runs. Run the "
            "cohort once per --dual-arm and compare the artifacts"
        )
    oracles = oracles_factory(protein_id=protein_id, config=config, inputs=inputs)
    cycle_kwargs = dict(oracles["cycle_kwargs"])
    conflicting = [name for name in RUN_OWNED_CYCLE_KWARGS if name in cycle_kwargs]
    if conflicting:
        raise V2CohortError(
            f"the oracles factory supplied {conflicting}, which the RUN owns; see run_v2_shard")
    assert_runtime_substrate_matches(config, cycle_kwargs.get("config"))

    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import (
        AttemptJournal,
        CostMeter,
        V2LedgerEvent,
        aggregate_v2_ledger,
        check_caps,
    )

    journal_path = _journal_path(
        inputs=inputs, out_dir=out_dir, protein_id=protein_id, run_signature=signature.value)
    cost_meter = CostMeter(
        journal=AttemptJournal(journal_path), protein_id=str(protein_id),
        arm=str(config.arm.arm_role), gpu_clock=_gpu_clock(oracles))

    point = _depth0_point(config)
    master_seed = int(config.identity.master_seed)
    base_config = cycle_kwargs["config"]
    # The three the LADDER supplies on top of the factory's ``cycle_kwargs`` and that
    # ``run_mechanism_views`` does not add for us.  ``lineage`` is NOT among them -- the factory
    # provides it and the ladder merely pops and re-passes it.  Kept honest by
    # ``test_the_mechanism_shard_supplies_every_kwarg_the_ladder_does``, which reads the ladder's
    # own call site rather than a hand-kept list.
    coordinate_law = build_depth_plan(config).law
    lineage = cycle_kwargs["lineage"]

    contrast_rows: list[dict] = []
    endpoint_rows_by_id: dict[str, dict] = {}
    state_rows_by_id: dict[str, dict] = {}
    archive_rows_by_id: dict[str, dict] = {}
    structure_rows_by_id: dict[str, dict] = {}
    events: list[dict] = []
    a2_views: list[Any] = []
    n_scorable = 0

    for prefix_index in range(prefix_start, prefix_start + n_prefixes):
        # The SOURCE seed is what makes prefixes independent; everything else about the sampler is
        # the run's own declared configuration.  Same construction as the resumed-null calibration,
        # so the two populations are the same generative process measured twice.
        source_seed = int(derive_seed(
            MECHANISM_SEED_NAMESPACE, str(protein_id), str(master_seed), "source",
            str(prefix_index)))
        prefix_config = dataclasses.replace(
            base_config, sampler=dataclasses.replace(base_config.sampler, seed=source_seed))
        source_fork_seeds = tuple(int(derive_seed(
            MECHANISM_SEED_NAMESPACE, str(protein_id), str(master_seed), "fork",
            str(prefix_index), str(k))) for k in range(int(point.n_lookaheads)))
        context = FeedbackPairSeedContext(
            seed_schema=V2_SEED_ENCODING_VERSION,
            campaign_id=str(config.identity.campaign_id),
            split_role=str(config.identity.split_role),
            master_seed=master_seed, protein_id=str(protein_id), depth=0,
            r_step=int(point.r_step), c_next_step=int(point.c_next_step),
            pair_ordinal=prefix_index, n_forks=int(point.n_lookaheads),
            n_descendant_lookaheads=int(point.n_lookaheads),
        )

        common = dict(
            protein_id=str(protein_id), source_index=prefix_index, source_seed=source_seed,
            source_state_id=None, source_unresolved_editable=None,
        )
        # One transition per PREFIX: every view and every arm of this prefix acts on the same
        # coordinate, and giving them separate ids would claim they were separate transitions.
        # This is why `feedback_events` keys on arm_slot/treatment_identity as well.
        origin_transition_id = make_transition_id(
            str(lineage.protein_id), str(lineage.family_id), depth=0,
            r_step=int(point.r_step), c_next_step=int(point.c_next_step),
            content_digest=canonical_digest({
                "mechanism_prefix": prefix_index, "source_seed": source_seed,
                "r": int(point.r_step), "c": int(point.c_next_step)}))
        shared = dict(cycle_kwargs,
                      config=prefix_config,
                      context=context, fork_index=0,
                      c_source_step=int(point.c_source_step), r_step=int(point.r_step),
                      c_next_step=int(point.c_next_step),
                      source_fork_seeds=source_fork_seeds,
                      origin_transition_id=origin_transition_id,
                      coordinate_law=coordinate_law,
                      declared_policy=config.declared_policy(),
                      feedback_enabled=True, cost_meter=cost_meter)
        try:
            if qualification:
                # PLAN §8.4.  The control is built FROM the treatment policy object, so both arms
                # provably share one band table, one stratum, one incumbent and one calibration --
                # and it is pinned to the treatment's REALIZED cardinalities, which only exist
                # after arm A has run.  Hence a factory rather than a second policy instance.
                treatment_policy = shared["support_policy"]
                results = (run_policy_qualification_view(
                    **shared,
                    control_policy_factory=functools.partial(
                        build_source_geometry_control, treatment_policy=treatment_policy),
                    control_declared_policy=config.declared_control_policy(),
                ),)
            else:
                results = run_mechanism_views(**dict(shared, interventions=requested_views))
        except V2Error as exc:
            # A prefix that could not be executed is a RESULT of this cohort, recorded once per
            # requested view.  Raising here would lose every prefix already paid for.
            for view in requested_views:
                contrast_rows.extend(mechanism_contrast_rows(
                    **common, view=view, arm_a=None, arm_b=None,
                    parity_violation=f"{type(exc).__name__}: {str(exc)[:180]}"))
            continue

        for result in results:
            view = result.axes.mechanism_view
            # ``PolicyQualificationResult`` names its arms treatment/control -- they ARE arm A and
            # arm B of one pair, and the rest of this loop is identical for both runners.
            arm_a = getattr(result, "arm_a", None) or result.treatment
            arm_b = getattr(result, "arm_b", None) or result.control
            arm_a_cycle, arm_b_cycle = arm_a.cycle, arm_b.cycle
            source = getattr(arm_a_cycle, "source", None)
            per_view = dict(
                common,
                source_state_id=None if source is None else source.state_id,
                source_unresolved_editable=(
                    None if source is None
                    else int(source.realized_maturity.n_unresolved_editable)),
            )
            rows = mechanism_contrast_rows(
                **per_view, view=view, arm_a=arm_a_cycle, arm_b=arm_b_cycle,
                parity_violation=result.parity_violation)
            contrast_rows.extend(rows)
            n_scorable += sum(1 for row in rows if row["analyzable"])

            for arm in (arm_a, arm_b):
                outcome = _ArmOutcome(arm.cycle)
                # Converted to ROWS here, not retained as objects.  A 16-prefix run forks two
                # archives per view per prefix and each holds its endpoints; keeping them all until
                # the end held ~39 GiB against a 40 GiB allocation on ONE prefix.  Rows are small,
                # already de-duplicated by join key, and are all the bundle ever needed.
                local = _admissions(outcome)
                reasons = {eid: adm.reason for eid, adm in local.items()}
                by_depth: dict[int, list] = {}
                for endpoint, depth in _endpoints_by_depth(outcome):
                    if endpoint.endpoint_id not in endpoint_rows_by_id:
                        by_depth.setdefault(depth, []).append(endpoint)
                for depth in sorted(by_depth):
                    for row in complete_endpoint_rows(by_depth[depth], depth=depth):
                        endpoint_rows_by_id.setdefault(row["endpoint_id"], row)
                for state_row in partial_state_rows(_partial_states(outcome)):
                    state_rows_by_id.setdefault(state_row["state_id"], state_row)
                for row in archive_rows(outcome.archive, admission_by_endpoint=reasons):
                    archive_rows_by_id.setdefault(row["endpoint_id"], row)
                for row in structure_evaluation_rows(
                        outcome.archive, admission_by_endpoint=reasons,
                        conditioning=cycle_kwargs.get("conditioning")):
                    structure_rows_by_id.setdefault(row["endpoint_id"], row)
                if arm.cycle.a2_view is not None:
                    a2_views.append(arm.cycle.a2_view)
                events.append(dict(
                    source=arm.cycle.source, endpoint=arm.cycle.selected_endpoint,
                    projected=arm.cycle.projected, propagated=arm.cycle.propagated,
                    policy=_policy_identity(_ArmRecord(0, arm.cycle), config=config),
                    policy_evidence=getattr(arm.cycle, "policy_evidence", None),
                    outcome=arm.cycle.outcome.value, detail=arm.cycle.detail,
                    pair_id=context.pair_id, arm_slot=arm.arm_slot,
                    treatment_identity=arm.treatment_identity))

    ledger_events = _ledger_rows(journal_path)
    verdict = check_caps(
        aggregate_v2_ledger(tuple(V2LedgerEvent(**row) for row in ledger_events)), config.caps)

    payload = {
        "mechanism_contrasts": contrast_rows,
        # Every arm forks the shared archive, so the same endpoint appears in many of them.  First
        # write wins: the shared pool is one pool, and duplicating it would multiply every join.
        "complete_endpoints": list(endpoint_rows_by_id.values()),
        "partial_states": list(state_rows_by_id.values()),
        "feedback_events": feedback_event_rows(events),
        "archive": list(archive_rows_by_id.values()),
        "structure_evaluations": list(structure_rows_by_id.values()),
        "a2_views": _first_by(
            a2_view_rows(
                a2_views, matched_extra_lookaheads=EXECUTED_A2_EXTRA_LOOKAHEADS,
                matching_resource=None,
                declared_matching_resource=config.arm.a2_matching_resource,
                matching_executed=False,
                unmatched_components={
                    "declared_unmatched": list(config.arm.a2_unmatched_reported)}),
            "a2_view_id"),
        # Empty, and that is the TRUE claim: terminal validation is what a ladder returns about the
        # designs it proposes, and this cohort proposes none.  It measures whether a projected
        # identity reaches the descendant, and every design it produces is an instrument reading.
        "terminal_validation": [],
        "ledger_events": ledger_events,
        "cap_verdict": {
            "within": bool(verdict.within), "breached": list(verdict.breached),
            "unverifiable": list(verdict.unverifiable), "detail": verdict.detail,
        },
        "n_prefixes_requested": n_prefixes,
        "prefix_start": prefix_start,
        "n_scorable_contrasts": n_scorable,
        "stopping_reason": ("policy_qualification_complete" if qualification
                            else "mechanism_cohort_complete"),
        "depth_reached": 1 if n_scorable else 0,
        "total_logical_dfe": sum(int(row.get("logical_dfe") or 0) for row in ledger_events),
        "substrate_digest": None,
    }
    # A mechanism shard SUCCEEDS when it produced scorable contrasts and stayed inside its caps.
    # Zero scorable contrasts is a real failure for this protein: every prefix stopped before a
    # projection, and reporting it as success with an empty table would let the cohort be consumed
    # downstream as evidence of no transmission.
    status = "ok" if (n_scorable > 0 and not verdict.breached) else "failed"
    return status, payload
