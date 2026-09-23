"""Depth-zero uses generated-pool rank; later depths use the strict lineage incumbent."""

from __future__ import annotations

import dataclasses

from inverse_folding.reference_flow.fusion_v2 import policy as pol
from inverse_folding.reference_flow.fusion_v2 import reward as rw
from inverse_folding.reference_flow.fusion_v2_runtime.cycle import run_one_cycle
from tests.inverse_folding.test_fusion_v2_head_directed_policy import (
    _cycle_kwargs,
    _coords,
    _donor,
    _policy,
    _source,
    SOURCE_SEQ,
)


def _bootstrap_policy(*, head=None):
    return dataclasses.replace(
        _policy(head=head),
        depth0_incumbent_rule=rw.DEPTH0_BOOTSTRAP_RULE,
        policy_version="v2",
    )


def _bootstrap_cycle(*, policy=None, **over):
    policy = policy or _bootstrap_policy()
    return run_one_cycle(**_cycle_kwargs(
        support_policy=policy,
        declared_policy=pol.DeclaredPolicy(
            policy_id=pol.HEAD_DIRECTED_CAPPED_POLICY_ID,
            policy_version="v2",
            is_diagnostic=False,
            phase="policy_qualification",
        ),
        **over,
    ))


class _GloballyShiftedHead:
    """Keep local differences fixed while making every generated design worse than WT."""

    def __init__(self, offset: float):
        from tests.inverse_folding.test_fusion_v2_head_directed_policy import _ScriptedHead

        self._delegate = _ScriptedHead()
        self.offset = float(offset)

    def evaluator_identity(self):
        return self._delegate.evaluator_identity()

    def score(self, requests):
        shifted = []
        for row in self._delegate.score(requests):
            risk = float(row.global_risk) + self.offset
            shifted.append(dataclasses.replace(
                row,
                score=dataclasses.replace(row.score, global_risk=risk),
                global_risk=risk,
            ))
        return shifted


def test_depth_zero_rank0_is_not_required_to_beat_wt():
    head = _GloballyShiftedHead(offset=100.0)
    policy = _bootstrap_policy(head=head)
    outcome = _bootstrap_cycle(policy=policy, head_oracle=head)

    assert outcome.committed, outcome.detail
    assert outcome.selected_endpoint.head_global_risk > policy.incumbent.head_global_risk
    assert outcome.policy_evidence.donor_gate is None
    assert outcome.policy_evidence.reward_gate_kind == rw.DEPTH0_BOOTSTRAP_RULE


def test_only_rank0_can_bootstrap_depth_zero():
    outcome = _bootstrap_cycle(endpoint_rank=1)
    assert not outcome.committed
    assert rw.DEPTH0_BOOTSTRAP_RULE in outcome.detail


def test_depth_zero_donor_becomes_i1_and_future_gating_is_strict():
    policy = _bootstrap_policy()
    outcome = _bootstrap_cycle(policy=policy)
    assert outcome.committed, outcome.detail

    advanced = policy.advance_lineage_incumbent(
        donor=outcome.selected_endpoint,
        verdict=None,
        accepted_at_depth=1,
    )
    assert advanced.incumbent.source_endpoint_id == outcome.selected_endpoint.endpoint_id

    worse = _donor("Y" * len(SOURCE_SEQ), source=_source(SOURCE_SEQ), fork_index=9, fork_seed=99)
    refused = advanced.decide(
        source=_source(SOURCE_SEQ), endpoint=worse,
        coordinates=_coords(depth=1),
        runtime=pol.PolicyRuntime(source_depth=1, selected_rank=0,
                                  selected_endpoint_id=worse.endpoint_id),
    )
    assert isinstance(refused, pol.PolicyRejection)
    assert refused.decision_evidence.donor_gate is not None
    assert refused.decision_evidence.reward_gate_kind == "strict_improvement"
