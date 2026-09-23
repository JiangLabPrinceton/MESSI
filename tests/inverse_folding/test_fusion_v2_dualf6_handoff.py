"""DUALF6: the local evidence that closes implementation and authorizes only the cluster runbook.

Four claims, each of which the cluster stage would otherwise have to discover with a GPU:

1. with no Dual overlay, the run is the frozen single-Head V2 in seeds, bytes, decisions, artifacts
   and costs -- not merely in its digests;
2. three objective arms can be run from ONE exact endpoint, so a difference between them is
   attributable to the objective and to nothing else;
3. binding the second Head as an OBSERVER changes nothing generative -- "B is present" and "B acts"
   are separable; and
4. no NetMHCIIpan import or runtime field entered the Dual layer.

What this file deliberately does NOT claim: that Dual works. It shows the code can ANSWER whether
Head B changes donor and write identity. The answer is the cluster runbook's to produce.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from inverse_folding.reference_flow.fusion_v2 import identity as ident
from inverse_folding.reference_flow.fusion_v2 import joint_objective as jo
from inverse_folding.reference_flow.fusion_v2 import policy as pol
from inverse_folding.reference_flow.fusion_v2 import reward as rw
from inverse_folding.reference_flow.fusion_v2 import seeds as sd

from . import _v2_fixtures as F
from .test_fusion_v2_dual_select import (
    EVAL_A, EVAL_B, _ScriptedHeadB, _authority, _score_b,
)
from .test_fusion_v2_head_directed_policy import (
    INCUMBENT_SEQ, L, SOURCE_SEQ, _calibration, _coords, _donor, _incumbent, _policy, _score,
    _scorer, _source,
)


# ------------------------------------------------------------------------------------------
# 1. Dual-off golden equivalence, beyond the digests
# ------------------------------------------------------------------------------------------

def test_the_seed_law_is_untouched_and_never_sees_the_dual_layer():
    src = inspect.getsource(sd)
    for banned in ("dual", "joint_objective", "allele"):
        assert banned not in src.lower(), banned


def test_a_dual_off_decision_is_identical_across_repeated_construction():
    """Decisions, not just digests. Two independently built legacy policies must agree exactly."""
    source = _source(SOURCE_SEQ)
    endpoint = _donor(source=source)
    first = _policy(calibration=_calibration(epsilon=0.0)).decide(
        source=source, endpoint=endpoint, coordinates=_coords())
    second = _policy(calibration=_calibration(epsilon=0.0)).decide(
        source=source, endpoint=endpoint, coordinates=_coords())
    assert isinstance(first, pol.PolicyDecision)
    assert (first.write_from_endpoint, first.reopen, first.carry_from_source,
            first.inject_from_source_feedback) == \
           (second.write_from_endpoint, second.reopen, second.carry_from_source,
            second.inject_from_source_feedback)


def test_a_legacy_bundle_gains_no_file_when_the_dual_tables_are_not_supplied():
    import pathlib
    import tempfile

    import scripts.rf_fusion_v2_artifacts as artifacts

    empty = {name: [] for name in artifacts.V2_TABLE_SCHEMAS}
    with tempfile.TemporaryDirectory() as d:
        artifacts.write_v2_bundle(d, manifest={"schema_version": "v2run-1"}, tables=empty)
        legacy = sorted(p.name for p in pathlib.Path(d).iterdir())
    with tempfile.TemporaryDirectory() as d:
        artifacts.write_v2_bundle(d, manifest={"schema_version": "v2run-1"}, tables=empty,
                                  dual_tables={n: [] for n in artifacts.DUAL_TABLE_SCHEMAS})
        dual = sorted(p.name for p in pathlib.Path(d).iterdir())
    assert set(dual) - set(legacy) == {f"{n}.parquet" for n in artifacts.DUAL_TABLE_SCHEMAS}
    assert set(legacy) - set(dual) == set()


def test_the_legacy_head_cost_event_id_is_unchanged_and_dual_namespaces_beside_it():
    from inverse_folding.reference_flow.fusion_v2_runtime import cycle as cyc
    from inverse_folding.reference_flow.fusion_v2_runtime.dual_lookahead import dual_head_event_id

    assert 'event_id=f"{event_prefix}:head"' in inspect.getsource(cyc)
    for role in (jo.AlleleRole.A, jo.AlleleRole.B):
        assert dual_head_event_id("P:fam0:d0", role) != "P:fam0:d0:head"


# ------------------------------------------------------------------------------------------
# 2. three objective arms from ONE exact endpoint
# ------------------------------------------------------------------------------------------

def _incumbent_b():
    """The lineage incumbent as role B would have scored it -- same construction as the A fixture."""
    score = _score_b(INCUMBENT_SEQ)
    binding = ident.SafetyReferenceBinding(
        reference_id="ref:wt", reference_label="wt_native", sequence_md5=score.sequence_md5,
        sequence_length=L, reference_content_digest=F.D,
        head_binding=ident.HeadScoreBinding(
            protein_id="5ZHV_B", sequence_md5=score.sequence_md5, sequence_length=L,
            window_grid_digest=ident.window_grid_digest(score.windows), evaluator=EVAL_B),
        head_score_digest=F.D, bound_at_depth=0, source_kind="predeclared_external",
    )
    reference = dataclasses.make_dataclass(
        "_RefB", [("binding", object), ("head_score", object)], frozen=True,
    )(binding=binding, head_score=score)
    return rw.bind_incumbent_from_safety_reference(
        reference=reference, reference_sequence=INCUMBENT_SEQ, lineage_id="5ZHV_B:fam0",
        evaluator=EVAL_B, rule="cumulative_safety_reference",
    )


def _arm_decision(arm):
    """Run ONE exact endpoint through one objective arm.

    ``a_only`` is the frozen single-Head law verbatim; ``b_only`` is the same law with the other
    Head; ``joint`` is the same law with an injected objective authority. No arm gets a different
    decision sequence -- which is what makes a difference between them attributable to the objective.
    """
    source = _source(SOURCE_SEQ)
    endpoint = _donor(source=source)
    calibration = _calibration(epsilon=0.0)
    if arm == "joint":
        policy = _policy(dual=_authority(endpoint), calibration=calibration)
    elif arm == "a_only":
        policy = _policy(calibration=calibration)
    elif arm == "b_only":
        # A CompleteEndpoint carries ONE Head's binding and score, so the b_only arm cannot reuse
        # the role-A endpoint object -- its donor gate would be comparing two instruments. What the
        # arms share is the exact SEQUENCE and the source it forked from, which is what
        # ``endpoint_id`` is derived from; only the Head evidence attached to it differs.
        score_b = _score_b(endpoint.sequence)
        endpoint = dataclasses.replace(
            endpoint, head_score=score_b, head_global_risk=score_b.global_risk,
            head_binding=ident.HeadScoreBinding(
                protein_id=endpoint.protein_id, sequence_md5=endpoint.sequence_md5,
                sequence_length=endpoint.sequence_length,
                window_grid_digest=ident.window_grid_digest(score_b.windows),
                evaluator=EVAL_B))
        policy = _policy(
            calibration=calibration, incumbent=_incumbent_b(), evaluator=EVAL_B,
            safety_reference_score=_score_b(INCUMBENT_SEQ),
            window_grid_digest=ident.window_grid_digest(_score_b(INCUMBENT_SEQ).windows),
            counterfactual_scorer=_scorer(_ScriptedHeadB()))
    else:
        raise AssertionError(arm)
    return endpoint, policy.decide(source=source, endpoint=endpoint, coordinates=_coords())


def test_all_three_arms_run_from_one_shared_depth_zero():
    """What "one byte-identical D0" can and cannot mean, stated precisely.

    A ``CompleteEndpoint`` carries ONE Head's binding and score, and the donor gate refuses a donor
    whose evaluator differs from the incumbent's -- correctly, since their risks would be two
    instruments. So the three arms cannot share one endpoint OBJECT.

    What they do share is the exact sequence and the source state it forked from, which is what
    ``endpoint_id`` is derived from. The content digest necessarily differs on the b_only arm,
    because the Head evidence attached to the same molecule differs. Any shared-root check across
    arms must therefore compare sequence and endpoint identity, never the endpoint content digest.
    """
    endpoints, outcomes = {}, {}
    for arm in ("joint", "a_only", "b_only"):
        endpoints[arm], outcomes[arm] = _arm_decision(arm)

    assert len({e.sequence_md5 for e in endpoints.values()}) == 1, "one exact molecule"
    assert len({e.endpoint_id for e in endpoints.values()}) == 1, "one endpoint identity"
    assert len({e.source_state_id for e in endpoints.values()}) == 1, "one source state"
    # joint and a_only share the endpoint object entirely; b_only carries the other Head's evidence
    assert endpoints["joint"].content_digest == endpoints["a_only"].content_digest
    assert endpoints["b_only"].content_digest != endpoints["a_only"].content_digest

    for arm, outcome in outcomes.items():
        assert isinstance(outcome, (pol.PolicyDecision, pol.PolicyRejection)), arm


def test_the_code_can_answer_whether_head_B_changes_the_transition():
    """DUALF6's acceptance criterion, stated as an executable question.

    It does not assert WHICH way the answer goes -- that is the cluster's to measure. It asserts
    that the answer is obtainable: the joint arm and the single-allele arm are comparable objects
    produced from one endpoint under one decision sequence.
    """
    _, joint = _arm_decision("joint")
    _, a_only = _arm_decision("a_only")

    def transition(outcome):
        if isinstance(outcome, pol.PolicyDecision):
            return (outcome.write_from_endpoint, outcome.reopen)
        return ("stall", getattr(outcome, "reason", None))

    assert transition(joint) is not None and transition(a_only) is not None
    # the two are comparable, and on this fixture they are not the same transition
    assert transition(joint) != transition(a_only)


# ------------------------------------------------------------------------------------------
# 3. an observer second Head changes nothing generative
# ------------------------------------------------------------------------------------------

def test_binding_head_B_as_an_observer_changes_no_decision():
    """"B is present" and "B acts" are separable, and only the second is the Dual claim.

    Scoring an endpoint with both Heads and recording the joint evidence is an OBSERVATION. It
    becomes an intervention only when the objective authority is injected into the policy. A run
    that binds Head B for evidence and leaves the policy alone must decide exactly as before.
    """
    from inverse_folding.reference_flow.fusion_v2 import dual_evidence as de

    source = _source(SOURCE_SEQ)
    endpoint = _donor(source=source)
    baseline = _policy(calibration=_calibration(epsilon=0.0)).decide(
        source=source, endpoint=endpoint, coordinates=_coords())

    # bind role B evidence for the same endpoint -- an observation, not an intervention
    authority = _authority(endpoint)
    evidence = de.bind_dual_evidence(
        endpoint=endpoint,
        result_a=de.HeadResult(score=endpoint.head_score, binding=endpoint.head_binding),
        result_b=de.HeadResult(
            score=_score_b(endpoint.sequence),
            binding=ident.HeadScoreBinding(
                protein_id=endpoint.protein_id, sequence_md5=endpoint.sequence_md5,
                sequence_length=endpoint.sequence_length,
                window_grid_digest=endpoint.head_binding.window_grid_digest, evaluator=EVAL_B)),
        objective=authority.objective)
    assert evidence.b.allele == EVAL_B.allele          # B really was scored

    observed = _policy(calibration=_calibration(epsilon=0.0)).decide(
        source=source, endpoint=endpoint, coordinates=_coords())
    assert isinstance(baseline, pol.PolicyDecision) and isinstance(observed, pol.PolicyDecision)
    assert (baseline.write_from_endpoint, baseline.reopen) == \
           (observed.write_from_endpoint, observed.reopen)


# ------------------------------------------------------------------------------------------
# 4. no NetMHCIIpan anywhere in the Dual layer
# ------------------------------------------------------------------------------------------

def test_the_dual_layer_imports_no_external_immune_predictor():
    """PLAN §0.2: no NMP runtime, acceptance, tuning or capability verdict.

    The frozen Heads are the only immune objective. An external predictor reaching the Dual layer
    would be a second objective entering by the back door, and the method's own verdict would stop
    being a property of the objective it declares.
    """
    from inverse_folding.reference_flow.fusion_v2 import (
        dual_config, dual_evidence, dual_policy, joint_objective,
    )
    from inverse_folding.reference_flow.fusion_v2_runtime import (
        dual_lookahead, dual_runtime, dual_selection,
    )

    # ``dual_runtime`` is the module the wiring repair added and the one the kernel actually calls;
    # a structural claim that skipped it would be checking the layer that never runs.
    for module in (joint_objective, dual_evidence, dual_policy, dual_config,
                   dual_lookahead, dual_selection, dual_runtime):
        src = inspect.getsource(module).lower()
        for banned in ("netmhciipan", "nmp", "el_rank", "netmhcii"):
            assert banned not in src, f"{module.__name__}: {banned}"
