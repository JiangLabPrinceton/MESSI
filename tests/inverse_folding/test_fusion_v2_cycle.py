"""V2F5-1: the one-cycle runner (PLAN task V2F5).

One complete cycle: source checkpoint -> lookaheads -> definitive selection -> projection ->
propagated checkpoint -> descendant lookaheads.  This is the first point at which the mechanism
exists end to end rather than as a set of parts.

PLAN's scientific boundary applies to every test here: "local tests validate wiring only.  Real
feedback transmission remains a cluster mechanism gate and must not be inferred from fake-oracle
success."  Nothing below claims the mechanism works -- only that each state edge is reconstructible
and every identity is explicit.

The acceptance bullet is "one ordinary and one anchored fake-oracle protein reconstruct every state
edge", so the cycle is exercised with and without hard anchors.
"""

from __future__ import annotations

import dataclasses
import pathlib
import types

import numpy as np
import pytest
import torch

from inverse_folding.reference_flow.config import (
    AmplificationConfig,
    HShuffleConfig,
    ReferenceFlowConfig,
    RemaskConfig,
    SamplerConfig,
    ScheduleConfig,
)
from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
from inverse_folding.reference_flow.fusion_v2 import identity as ident
from inverse_folding.reference_flow.fusion_v2 import policy as pol
from inverse_folding.reference_flow.fusion_v2 import state as st
from inverse_folding.reference_flow.fusion_v2_runtime.cycle import (
    CycleOutcome,
    V2CycleError,
    run_one_cycle,
)
from inverse_folding.reference_flow.sampler import PositionDependentDFMSampler
from tests.inverse_folding import _v2_fixtures as F

VOCAB = 33
C0, R1, C1 = 50, 40, 60
CANONICAL_AA20 = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET = {token: CANONICAL_AA20[index] for index, token in enumerate(sorted(F.AA))}


def _denoiser(x_t: torch.Tensor, t: float, struct) -> torch.Tensor:
    length = int(x_t.shape[0])
    logits = torch.full((length, VOCAB), float("-inf"), dtype=torch.float32)
    for position in range(length):
        for token in sorted(F.AA):
            logits[position, token] = float((position * 7 + token * 3 + int(x_t[position])) % 11)
    return logits


def _cfg():
    return ReferenceFlowConfig(
        sampler=SamplerConfig(n_steps=F.N_STEPS, seed=11, temperature=1.0,
                              n_designs_per_protein=1,
                              remask=RemaskConfig(enabled=True, fraction_scale=0.0)),
        schedule=ScheduleConfig(base_form="linear"),
        amplification=AmplificationConfig(form="constant_one", h_source="h_processed"),
        h_shuffle=HShuffleConfig(enabled=False, seed=None),
    )


def _head(sequence_md5, *, risk, length=F.L):
    """A fake Head oracle result: identity-bearing, deterministic, no model."""
    windows = F.windows(length)
    evaluator = F.safety_reference(length=length).head_binding.evaluator
    binding = ident.HeadScoreBinding(
        protein_id="5ZHV_B", sequence_md5=sequence_md5, sequence_length=length,
        window_grid_digest=ident.window_grid_digest(windows), evaluator=evaluator,
    )
    return dataclasses.replace(
        F.EndpointHeadScore(
            protein_id="5ZHV_B", sequence_md5=sequence_md5, sequence_length=length,
            allele=evaluator.allele, score_scale=evaluator.score_scale, windows=windows,
            residue_hotspot=(-0.1,) * length, global_risk=risk,
        )
    ), binding


class _FakeHeadOracle:
    """Deterministic fake Head: risk is a function of the sequence, so selection is reproducible
    and a source-blind or endpoint-blind kernel cannot accidentally look correct.

    It scores what a REAL Head scores -- the residue string -- and derives its own join key from
    that string rather than trusting the caller's.  A runner that handed it a digest in place of a
    sequence therefore fails here instead of only on the cluster.
    """

    def __init__(self, length=F.L):
        self.calls = 0
        self.length = int(length)
        self.scored_md5s = set()
        self.requests = []

    def score(self, requests):
        from inverse_folding.reference_flow.fusion.state import sequence_md5

        self.calls += 1
        rows = []
        for request in requests:
            self.requests.append(request)
            sequence_key = sequence_md5(request.sequence)
            self.scored_md5s.add(sequence_key)
            risk = -float(int(sequence_key[:6], 16) % 1000) / 100.0
            score, binding = _head(sequence_key, risk=risk, length=self.length)
            rows.append(_Row(score=score, binding=binding, sequence_md5=sequence_key,
                             global_risk=risk))
        return rows


@dataclasses.dataclass(frozen=True)
class _Row:
    score: object
    binding: ident.HeadScoreBinding
    sequence_md5: str
    global_risk: float

    @property
    def protein_id(self):
        return self.score.protein_id

    @property
    def sequence_length(self):
        return self.score.sequence_length

    @property
    def allele(self):
        return self.score.allele

    @property
    def score_scale(self):
        return self.score.score_scale

    @property
    def windows(self):
        return self.score.windows


def _structure_oracle(request):
    """Every design folds in the fake world; a real run replaces this with a refold.

    It takes the same typed request the Head does, because a refold needs the residue string too --
    the digest is only what the on-disk structure cache is keyed by.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest

    assert isinstance(request, OracleRequest), (
        f"the structure gate was handed {type(request).__name__}; a real refold cannot fold a "
        "digest"
    )
    return StructureOutcome(feasible=True, metrics={"scTM": 0.9})


def _support_policy(source, endpoint, coordinates):
    """A diagnostic explicit-probe policy chosen FROM THE STATE, not hard-coded.

    Where the endpoint actually carries new information is the positions the SOURCE had not yet
    decided: a position already resolved in the source is inherited verbatim by its own
    completions, so it can never disagree.  So the probe writes the endpoint at a source-masked
    position, reopens a resolved one, injects any future-committed identity, and carries the rest.
    Writing over a mask also drives the coupled mask-load identity with a non-zero ``a`` term.
    """
    editable = list(source.editable_positions)
    masked = [i for i in editable if source.tokens[i] == source.mask_token_id]
    resolved = [i for i in editable if source.tokens[i] != source.mask_token_id]
    future = [
        i for i in resolved
        if source.active_commit_depth_step_by_pos[i] is not None
        and source.active_commit_depth_step_by_pos[i].step >= coordinates.r_step
    ]
    write = masked[:1]
    reopen = [i for i in resolved if i not in future][:1]
    if not write or not reopen:
        return pol.PolicyRejection(
            reason=f"no admissible support: {len(masked)} masked, {len(resolved)} resolved, "
                   f"{len(future)} future-committed",
            policy=_policy_identity(),
        )
    carry = [i for i in editable if i not in write and i not in future and i not in reopen]
    reasons = {}
    reasons.update({i: pol.SupportReason.IMPROVEMENT_ASSOCIATED for i in write})
    reasons.update({i: pol.SupportReason.FUTURE_SOURCE_IDENTITY for i in future})
    reasons.update({i: pol.SupportReason.UNCERTAIN for i in reopen})
    reasons.update({i: pol.SupportReason.TEMPORALLY_VALID for i in carry})
    return pol.PolicyDecision(
        write_from_endpoint=tuple(write), inject_from_source_feedback=tuple(future),
        reopen=tuple(reopen), carry_from_source=tuple(carry),
        reason_by_pos=reasons, policy=_policy_identity(),
    )


def _policy_identity():
    return ident.ProjectionPolicyIdentity(
        policy_id="explicit_probe", policy_version="v0",
        policy_config_digest=F.digest("pcfg"), policy_spec_digest=F.digest("pspec"),
        is_diagnostic_only=True,
    )


def _wide_band_table():
    """A permissive band for the WIRING test.

    B(r) is a calibration artifact produced by ``scripts/rho_maturity_scan.py --mode step`` against
    a real cohort; the shared fixture's band is tuned to a hand-built state, not to what this toy
    denoiser actually produces at r=40.  Band calibration has its own suite
    (``test_fusion_v2_schedule``); narrowing it here would make this test fail for a reason that has
    nothing to do with the cycle.
    """
    import inverse_folding.reference_flow.fusion_v2.schedule as sch

    band = sch.make_band(
        step=R1, stratum_key=F.STRATUM, levels=sch.QuantileLevels(levels=(0.1, 0.5, 0.9)),
        rho_quantiles=(0.05, 0.5, 0.95), unresolved_quantiles=(F.L, 2, 1),
        rho_accept=sch.BandInterval(lo=0.0, hi=1.0, lo_level=0.1, hi_level=0.9),
        unresolved_accept=sch.BandInterval(lo=1.0, hi=float(F.L), lo_level=0.9, hi_level=0.1),
        combination_rule="both_axes", n_attempts=32, n_captured=32,
        n_editable_min=1, n_editable_max=F.L,
    )
    return sch.make_band_table(provenance=F.band_provenance(), bands=(band,))


def _module_meter():
    """A journal on a real (throwaway) filesystem path.

    The journal's whole value is that a ``requested`` line survives a process that dies inside an
    oracle, so an in-memory stub would test the opposite of what it is for.  Tests that READ the
    journal pass their own ``tmp_path``-backed meter.
    """
    import tempfile

    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import AttemptJournal, CostMeter

    return CostMeter(
        journal=AttemptJournal(pathlib.Path(tempfile.mkdtemp()) / "attempts.jsonl"),
        protein_id="5ZHV_B", arm="v2", gpu_clock=lambda: 0.0,
    )


def _run_kwargs(**over):
    table = _wide_band_table()
    kw = dict(
        sampler=PositionDependentDFMSampler(mask_token_id=F.MASK, vocab_size=VOCAB),
        denoiser=_denoiser, config=_cfg(), sequence_length=F.L,
        h_values=np.zeros(F.L, dtype=np.float32), residue_token_ids=F.AA,
        alphabet=ALPHABET, fixed_tokens={0: 10},
        lineage=F.lineage(), mask_token_id=F.MASK, aa_token_ids=F.AA,
        conditioning=F.conditioning(
            # The kernel refuses a band table the run's own frozen provenance does not
            # name, so the fixture binds the two instead of leaving them unrelated.
            schedule_band_calibration=table.provenance.calibration_content_digest), safety_reference=F.safety_reference(),
        c_source_step=C0, r_step=R1, c_next_step=C1,
        source_fork_seeds=(5001, 5002, 5003), descendant_fork_seeds=(6001, 6002),
        descendant_propagation_seed=9001,
        head_oracle=_FakeHeadOracle(), structure_oracle=_structure_oracle,
        support_policy=_support_policy,
        band_table=table, stratum_key=F.STRATUM, declared_band_id=F.BAND_ID,
        declared_band_digest=table.provenance.calibration_content_digest,
        origin_transition_id=F.TXN,
        safety_gate=F.safety_gate(),
        declared_policy=F.declared_policy(),
        feedback_enabled=True,
        cost_meter=_module_meter(),
    )
    kw.update(over)
    return kw


def _run(**over):
    return run_one_cycle(**_run_kwargs(**over))


# --------------------------------------------------------------------------------------------
# the cycle completes and every edge is present
# --------------------------------------------------------------------------------------------


def test_the_cycle_completes():
    outcome = _run()
    assert isinstance(outcome, CycleOutcome)
    assert outcome.committed


def test_every_state_edge_is_reconstructible():
    """The acceptance bullet: every edge of source -> endpoints -> selection -> projection ->
    propagation -> descendants must be present and linked by identity."""
    o = _run()
    assert o.source.sampler_step == C0
    assert o.endpoints
    assert o.selected_endpoint in o.endpoints
    assert o.projected.r_step == R1
    assert o.projected.lineage.parent_state_id == o.source.state_id
    assert o.projected.lineage.origin_endpoint_id == o.selected_endpoint.endpoint_id
    assert o.propagated.sampler_step == C1
    assert o.descendant_endpoints


def test_the_propagated_capture_validates_against_its_projection():
    """The trust boundary before a captured row enters the denoiser/archive graph."""
    o = _run()
    st.validate_propagated_capture(
        projected=o.projected, propagated=o.propagated,
        background_remask_events=o.segment.background_remask_events,
    )


def test_the_descendant_is_one_depth_below_the_source():
    o = _run()
    assert o.propagated.lineage.depth == o.source.lineage.depth + 1


def test_the_anchored_and_ordinary_proteins_both_complete():
    """PLAN acceptance: "one ordinary and one anchored fake-oracle protein"."""
    assert _run(fixed_tokens={0: 10}).committed
    assert _run(fixed_tokens={}).committed


# --------------------------------------------------------------------------------------------
# cost accounting
# --------------------------------------------------------------------------------------------


def test_the_screening_cost_follows_the_plan_formula():
    """PLAN §3.4: C_screen = c + K(S-c) for one source checkpoint with K lookaheads."""
    o = _run()
    k = len(o.endpoints)
    assert o.cost.screen_logical_dfe == C0 + k * (F.N_STEPS - C0)


def test_the_segment_costs_exactly_its_own_span():
    assert _run().cost.segment_logical_dfe == C1 - R1


def test_the_source_prefix_is_charged_once():
    """PLAN §3.4: "The ledger must not charge the source prefix again"."""
    o = _run()
    assert o.cost.source_prefix_logical_dfe == C0


# --------------------------------------------------------------------------------------------
# selection is definitive-only
# --------------------------------------------------------------------------------------------


def test_the_selected_endpoint_is_definitively_feasible():
    """PLAN §4.2: only a definitive endpoint may become feedback ancestry."""
    o = _run()
    assert o.selected_endpoint.feasibility_level is st.FeasibilityLevel.DEFINITIVE
    assert o.selected_endpoint.structure_outcome.feasible


def test_the_selection_is_the_best_available_endpoint():
    o = _run()
    eligible = [e for e in o.endpoints
                if e.feasibility_level is st.FeasibilityLevel.DEFINITIVE]
    assert o.selected_endpoint.head_global_risk == min(e.head_global_risk for e in eligible)


def test_a_run_where_nothing_folds_yields_a_typed_null_not_a_crash():
    """PLAN §4.5: stopping reasons are typed; "no admissible endpoint" is a normal outcome."""
    outcome = _run(structure_oracle=lambda md5: StructureOutcome(feasible=False,
                                                                 failure_reason="scTM 0.2"))
    assert not outcome.committed
    assert outcome.outcome is st.TransitionOutcome.NULL_NO_ADMISSIBLE_ENDPOINT
    assert outcome.projected is None


def test_a_policy_that_declines_yields_a_typed_null():
    outcome = _run(support_policy=lambda source, endpoint, coordinates: pol.PolicyRejection(
        reason="declined", policy=_policy_identity()))
    assert not outcome.committed
    assert outcome.outcome is st.TransitionOutcome.NULL_INVALID_POLICY_RESULT


# --------------------------------------------------------------------------------------------
# the A2 view is taken before feedback
# --------------------------------------------------------------------------------------------


def test_the_a2_view_holds_exactly_the_pre_feedback_endpoints():
    """PLAN §2.3: A2 and V2 share the pre-feedback rows; identity of the ids is the proof."""
    o = _run()
    assert set(o.a2_view.endpoint_ids) == {e.endpoint_id for e in o.endpoints}


def test_the_a2_view_excludes_the_descendants():
    o = _run()
    descendants = {e.endpoint_id for e in o.descendant_endpoints}
    assert set(o.a2_view.endpoint_ids).isdisjoint(descendants)


def test_every_endpoint_generated_is_retained_in_the_archive():
    """No endpoint is discarded, before or after feedback."""
    o = _run()
    rows = {row.endpoint_id for row in o.archive.raw_rows()}
    assert {e.endpoint_id for e in o.endpoints} <= rows
    assert {e.endpoint_id for e in o.descendant_endpoints} <= rows


# --------------------------------------------------------------------------------------------
# seeds and determinism
# --------------------------------------------------------------------------------------------


def test_the_whole_cycle_is_deterministic():
    a, b = _run(), _run()
    assert a.propagated.content_digest == b.propagated.content_digest
    assert [e.sequence for e in a.descendant_endpoints] == \
        [e.sequence for e in b.descendant_endpoints]


def test_source_and_descendant_fork_seeds_are_disjoint():
    """Reusing a source seed for a descendant would regenerate an endpoint the archive already
    holds, so the descendant pool would measure nothing new."""
    with pytest.raises(V2CycleError, match="disjoint"):
        _run(source_fork_seeds=(5001, 5002), descendant_fork_seeds=(5001, 7000))


def test_a_different_propagation_seed_changes_the_descendant():
    a = _run(descendant_propagation_seed=9001)
    b = _run(descendant_propagation_seed=9002)
    assert a.propagated.content_digest != b.propagated.content_digest


# --------------------------------------------------------------------------------------------
# positive controls: the cycle must actually read source and endpoint
# --------------------------------------------------------------------------------------------


def test_a_source_blind_policy_is_visible_in_the_projection():
    """PLAN V2F5 acceptance: "a deliberately source-blind kernel fails the deterministic
    source-dependence check".  A policy that ignores the source picks a fixed support, so its
    projection cannot depend on which positions the source actually resolved."""
    fixed_support = lambda source, endpoint, coordinates: pol.PolicyDecision(
        write_from_endpoint=(1,), inject_from_source_feedback=(),
        reopen=(2,), carry_from_source=tuple(
            i for i in source.editable_positions if i not in (1, 2)),
        reason_by_pos={i: pol.SupportReason.NO_EVIDENCE for i in source.editable_positions},
        policy=_policy_identity(),
    )
    blind = _run(support_policy=fixed_support)
    informed = _run()
    if blind.committed and informed.committed:
        assert blind.projected.support != informed.projected.support


def test_the_endpoint_bytes_reach_the_projection():
    """Endpoint dependence: the written position must hold the ENDPOINT's byte, not the source's."""
    o = _run()
    for position in o.projected.support.write_from_endpoint:
        assert o.projected.tokens[position] == \
            o.selected_endpoint.endpoint_provenance_evidence_by_pos[position].token


# --------------------------------------------------------------------------------------------
# the ledger must describe what the cycle actually spent
# --------------------------------------------------------------------------------------------


class _CountingSampler:
    """Wraps the real sampler and records every exact completion it is asked to run.

    A lookahead is the expensive object in a cycle: it runs the tail ``S - c`` for one fork.  The
    cost ledger claims ``C_screen = c + K(S - c)``, and the only way to know whether that claim is
    true is to count the completions the run actually performed rather than the ones it planned.
    """

    def __init__(self):
        self._inner = PositionDependentDFMSampler(mask_token_id=F.MASK, vocab_size=VOCAB)
        self.completion_tail_dfe = []

    def __getattr__(self, name):
        # The segment reads sampler attributes directly (vocab_size, mask_token_id); proxying them
        # keeps the wrapper transparent so the count measures the real cycle, not a stub of it.
        return getattr(self._inner, name)

    def sample(self, **kw):
        resume = kw.get("continuation_resume")
        if resume is not None:
            self.completion_tail_dfe.append(int(resume.n_steps) - int(resume.start_step))
        return self._inner.sample(**kw)


def test_the_screening_ledger_counts_every_completion_the_cycle_actually_ran():
    """PLAN §3.4: the ledger is the cost claim, so an unbilled completion is a false claim.

    An extra completion run only to read some metadata off the Head still pays a full tail of
    forward passes.  At the real unit costs a tail is the dominant term, so a ledger that omits one
    understates the method's screening cost by a whole lookahead per cycle -- and the number that
    gets compared against a baseline is the ledger, not the wall clock.
    """
    sampler = _CountingSampler()
    source_seeds, descendant_seeds = (5001, 5002, 5003), (6001, 6002)
    outcome = _run(sampler=sampler, source_fork_seeds=source_seeds,
                   descendant_fork_seeds=descendant_seeds)
    assert outcome.committed

    source_tail, descendant_tail = F.N_STEPS - C0, F.N_STEPS - C1
    expected = ([source_tail] * len(source_seeds)) + ([descendant_tail] * len(descendant_seeds))
    assert sorted(sampler.completion_tail_dfe) == sorted(expected), (
        "the cycle ran a completion the declared breadth does not account for"
    )

    realized_source_screening = sum(
        tail for tail in sampler.completion_tail_dfe if tail == source_tail)
    assert outcome.cost.screening_logical_dfe == realized_source_screening
    assert outcome.cost.screen_logical_dfe == C0 + realized_source_screening
    realized_descendant = sum(
        tail for tail in sampler.completion_tail_dfe if tail == descendant_tail)
    assert outcome.cost.descendant_screening_logical_dfe == realized_descendant


# --------------------------------------------------------------------------------------------
# ancestry eligibility is a CONJUNCTION, not the structure verdict alone
# --------------------------------------------------------------------------------------------


def test_the_whole_landscape_safety_gate_actually_runs_inside_the_cycle(monkeypatch):
    """The probe that a disconnected gate cannot survive.

    Replace the comparator with something that cannot produce an answer.  If the cycle still
    completes, the gate is not in the path -- and every "definitive" endpoint the run reported was
    labelled on the structure verdict alone, while the label is defined as a conjunction of
    structure, constraints and whole-landscape immune safety (PLAN §2.7, §4.2).
    """
    import inverse_folding.reference_flow.fusion_v2_runtime.admission as adm

    def _refuse(*args, **kwargs):
        raise AssertionError("the cumulative comparator must be reached")

    monkeypatch.setattr(adm, "measure_cumulative", _refuse)
    with pytest.raises(AssertionError, match="cumulative comparator"):
        _run()


def test_an_endpoint_that_opens_a_new_hotspot_never_becomes_ancestry():
    """A design can fold perfectly and still be inadmissible.

    Because ancestry is recursive, admitting one puts its hotspot into the parent of everything
    downstream, where the cumulative ratchet can no longer see it as new.  The structure oracle here
    passes every design, so the ONLY thing that can stop the cycle is the immune gate.
    """
    hot = _run(safety_gate=F.safety_gate(reference_z=-9.0))
    assert hot.outcome is st.TransitionOutcome.NULL_NO_ADMISSIBLE_ENDPOINT
    assert hot.selected_endpoint is None
    assert hot.projected is None
    # ...and the raw evidence survives: nothing scored is discarded (PLAN §4.1).
    assert hot.endpoints
    assert hot.archive.raw_rows()
    assert all(not hot.archive.may_become_ancestry(row.endpoint_id)
               for row in hot.archive.raw_rows())


def test_a_safe_design_still_reaches_definitive_and_projects():
    """The gate must not be satisfiable only by refusing everything."""
    safe = _run(safety_gate=F.safety_gate(reference_z=-0.7))
    assert safe.committed
    assert safe.selected_endpoint is not None
    assert safe.archive.may_become_ancestry(safe.selected_endpoint.endpoint_id)


def test_the_gate_is_a_required_argument_of_the_cycle():
    """An optional safety gate defaults to off, and a run with it off is indistinguishable in its
    outputs from a run with it on."""
    kw = _run_kwargs()
    kw.pop("safety_gate")
    with pytest.raises(TypeError, match="safety_gate"):
        run_one_cycle(**kw)


def test_a_segment_that_resolves_everything_stops_the_cycle_with_a_typed_outcome():
    """PLAN §4.5: a cycle that cannot continue is a RESULT, not an exception.

    If the propagation segment resolves its last mask, the descendant is a complete sequence -- a
    terminal endpoint, not a live partial state that can emit lookaheads.  Letting the state layer's
    "must retain at least one unresolved editable position" escape would report an ordinary
    trajectory outcome as a crash, and a cohort runner could not tell it from real corruption.
    """
    # A capture step deep enough that the toy denoiser resolves everything left.
    outcome = _run(r_step=R1, c_next_step=F.N_STEPS - 1)
    assert outcome.outcome is st.TransitionOutcome.STALLED_NO_NOVEL_DESCENDANT
    assert outcome.propagated is None
    assert outcome.segment is not None, "the paid segment must stay on the record"
    assert "terminal" in outcome.detail or "resolved" in outcome.detail
    # The work already paid for is still accounted.
    assert outcome.cost.segment_logical_dfe == (F.N_STEPS - 1) - R1


def test_one_editable_site_returns_the_best_lookahead_without_calling_the_policy():
    def policy_must_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("one-position fallback must stop before feedback projection")

    config = dataclasses.replace(
        _cfg(), sampler=dataclasses.replace(_cfg().sampler, seed=41))
    outcome = _run(
        config=config, fixed_tokens={i: 10 for i in range(F.L - 1)},
        support_policy=policy_must_not_run,
    )
    assert outcome.outcome is st.TransitionOutcome.TERMINAL_BEST_LOOKAHEAD
    assert outcome.selected_endpoint is not None
    assert outcome.segment is None and outcome.propagated is None


# --------------------------------------------------------------------------------------------
# the oracles are asked about sequences, not about digests
# --------------------------------------------------------------------------------------------


def test_the_head_is_asked_about_complete_canonical_sequences():
    """A real Head tokenizes the string it is handed.  Handing it an MD5 satisfies every fake keyed
    by identity and fails only on the cluster, after an allocation has been paid for."""
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest

    oracle = _FakeHeadOracle()
    outcome = _run(head_oracle=oracle)
    assert oracle.requests, "the Head was never called"
    produced = {endpoint.sequence for endpoint in
                outcome.endpoints + outcome.descendant_endpoints}
    for request in oracle.requests:
        assert isinstance(request, OracleRequest)
        assert request.sequence in produced
        assert set(request.sequence) <= set(CANONICAL_AA20)
        assert len(request.sequence) == F.L


def test_the_head_is_asked_once_per_distinct_sequence():
    """The Head is a function of the sequence, so a duplicate sibling is one measurement.  The
    request list must therefore be de-duplicated on the JOIN KEY while still carrying sequences."""
    oracle = _FakeHeadOracle()
    _run(head_oracle=oracle)
    keys = [request.sequence_md5 for request in oracle.requests]
    assert len(keys) == len(set(keys))


def test_the_structure_gate_is_asked_about_the_endpoint_it_decides_on():
    seen = []

    def _recording(request):
        seen.append(request)
        return StructureOutcome(feasible=True, metrics={"scTM": 0.9})

    outcome = _run(structure_oracle=_recording)
    assert seen
    by_key = {endpoint.sequence_md5: endpoint.sequence for endpoint in
              outcome.endpoints + outcome.descendant_endpoints}
    for request in seen:
        assert request.sequence == by_key[request.sequence_md5]


# --------------------------------------------------------------------------------------------
# the ledger charges what RAN, not what a full cycle would have cost
# --------------------------------------------------------------------------------------------


def test_the_a2_view_is_charged_only_for_the_work_it_actually_did():
    """A2 stops at the view: it runs no projection, no segment and no descendant pool.

    Charging it for them would hand the control arm a fictional bill in exactly the currency PLAN
    §4.3 matches the two arms on -- so an A2 that looked resource-matched would in fact have been
    given less, and the comparison would understate V2's cost advantage or overstate its benefit.
    """
    a2 = _run(feedback_enabled=False)
    assert a2.cost.segment_logical_dfe == 0
    assert a2.cost.descendant_screening_logical_dfe == 0
    assert a2.cost.screening_logical_dfe > 0


def test_a_cycle_that_inherits_its_source_pool_is_not_charged_for_screening_it_skipped():
    """A ladder rung sources from the previous rung's descendants (PLAN §3.4 forbids re-charging).

    The pool is INHERITED, so no completion is generated and no prefix is paid: both are already on
    the record from the rung that actually ran them.
    """
    first = _run()
    reused = _run(source_override=first.propagated, source_endpoints=first.descendant_endpoints,
                  c_source_step=C1, r_step=52, c_next_step=72,
                  source_fork_seeds=(), lineage=first.propagated.lineage)
    assert reused.cost.screening_logical_dfe == 0
    assert reused.cost.source_prefix_logical_dfe == 0


def test_the_source_prefix_is_charged_only_by_the_cycle_that_captured_it():
    """The ladder captures depth 0 itself and hands every rung an explicit source.

    A cycle that charged a prefix it did not run would make the ladder's total exceed its own
    execution graph by one prefix per depth.
    """
    first = _run()
    assert first.cost.source_prefix_logical_dfe == C0, "this cycle DID capture its own source"
    handed = _run(source_override=first.source)
    assert handed.cost.source_prefix_logical_dfe == 0


def test_a_cycle_that_stops_before_projecting_is_not_charged_for_the_segment():
    """PLAN §4.5 makes a typed stop a RESULT.  A stop that still billed for the segment it never
    ran would let a cohort of refusals consume a budget nothing spent."""
    outcome = _run(structure_oracle=lambda request: StructureOutcome(
        feasible=False, failure_reason="scTM 0.2"))
    assert outcome.outcome is st.TransitionOutcome.NULL_NO_ADMISSIBLE_ENDPOINT
    assert outcome.cost.segment_logical_dfe == 0
    assert outcome.cost.descendant_screening_logical_dfe == 0


def test_the_cost_fields_are_disjoint_and_sum_to_the_execution_graph():
    """The four fields must partition the work, or a total computed from them either double-counts
    the prefix or drops the descendant pool -- the two errors actually found in the ladder."""
    o = _run()
    k = len(o.endpoints)
    k_desc = len(o.descendant_endpoints)
    assert o.cost.source_prefix_logical_dfe == C0
    assert o.cost.screening_logical_dfe == k * (F.N_STEPS - C0)
    assert o.cost.segment_logical_dfe == C1 - R1
    assert o.cost.descendant_screening_logical_dfe == k_desc * (F.N_STEPS - C1)
    assert o.cost.total_logical_dfe == (
        C0 + k * (F.N_STEPS - C0) + (C1 - R1) + k_desc * (F.N_STEPS - C1))


def test_the_plan_screen_formula_is_still_expressible():
    """PLAN §3.4: C_screen = c + K(S-c).  Splitting the prefix into its own field must not lose
    the quantity the PLAN's cost model is written in."""
    o = _run()
    k = len(o.endpoints)
    assert o.cost.screen_logical_dfe == C0 + k * (F.N_STEPS - C0)


def test_the_cycle_refuses_to_default_the_feedback_flag():
    """The frozen config must be the runtime AUTHORITY, not a label on the manifest.

    ``feedback_enabled`` defaulted to ``True``, and nothing carried ``config.arm.feedback_enabled``
    into the cycle -- it reached only the manifest and ``--print-config``.  An A2 config would
    therefore have run WITH feedback while its own artifact recorded ``feedback_enabled: false``,
    labelling a treatment arm as its own control.  The same argument that makes ``safety_gate``
    required applies: an optional flag defaults to a value, and a run with it wrong is
    indistinguishable in its outputs from a run with it right.
    """
    kwargs = _run_kwargs()
    kwargs.pop("feedback_enabled")
    with pytest.raises(TypeError, match="feedback_enabled"):
        run_one_cycle(**kwargs)


# --------------------------------------------------------------------------------------------
# every oracle request is journaled before it runs (PLAN §5.4)
# --------------------------------------------------------------------------------------------


def _meter(tmp_path, **over):
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import AttemptJournal, CostMeter

    kw = dict(journal=AttemptJournal(tmp_path / "attempts.jsonl"), protein_id="5ZHV_B",
              arm="v2", gpu_clock=lambda: 0.0)
    kw.update(over)
    return CostMeter(**kw)


def _events(tmp_path):
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import events_from_journal

    return events_from_journal(tmp_path / "attempts.jsonl")


def test_the_cycle_refuses_to_run_unmetered():
    """An optional meter defaults to no journal, and PLAN §5.4's ``unknown_after_start`` can then
    never be produced by a real run: a process that dies inside a refold leaves no record that
    anything was ever started, so the cost reads as zero rather than as unknown."""
    kwargs = _run_kwargs()
    kwargs.pop("cost_meter")
    with pytest.raises(TypeError, match="cost_meter"):
        run_one_cycle(**kwargs)


def test_every_oracle_stage_of_a_cycle_reaches_the_journal(tmp_path):
    _run(cost_meter=_meter(tmp_path))
    phases = {event.phase for event in _events(tmp_path)}
    assert {"root_capture", "screen", "head", "structure", "segment",
            "descendant_screen"} <= phases


def test_the_journaled_logical_dfe_is_the_cycles_own_cost_report(tmp_path):
    """Two accounts of the same run that disagree are worse than one: the ledger is what a
    matched-compute claim is judged on, and the cost report is what the ladder sums."""
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import aggregate_v2_ledger

    outcome = _run(cost_meter=_meter(tmp_path))
    totals = aggregate_v2_ledger(_events(tmp_path))
    assert totals["logical_dfe"] == outcome.cost.total_logical_dfe


def test_a_head_that_dies_leaves_unknown_after_start_on_the_record(tmp_path):
    """PLAN §5.4: persist ``unknown_after_start`` "rather than zero or a guessed value"."""
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import UNKNOWN_AFTER_START

    class _Dies:
        def score(self, requests):
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError):
        _run(cost_meter=_meter(tmp_path), head_oracle=_Dies())
    events = _events(tmp_path)
    assert any(event.physical_cost_status == UNKNOWN_AFTER_START for event in events)
    assert any(event.phase == "head" and event.status == "failed" for event in events)


def test_an_unknown_attempt_makes_the_measured_caps_unverifiable(tmp_path):
    """The point of the record: a run that lost a measurement may not certify a measured cap."""
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import (
        aggregate_v2_ledger,
        check_caps,
    )

    class _Dies:
        def score(self, requests):
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError):
        _run(cost_meter=_meter(tmp_path), head_oracle=_Dies())
    caps = types.SimpleNamespace(
        max_logical_dfe=10 ** 9, max_head_calls=10 ** 9, max_definitive_refolds=10 ** 9,
        max_gpu_seconds=10 ** 9, max_walltime_s=10 ** 9, max_retries=10 ** 9)
    verdict = check_caps(aggregate_v2_ledger(_events(tmp_path)), caps)
    assert not verdict.within
    assert "max_gpu_seconds" in verdict.unverifiable


def test_the_head_batch_is_journaled_as_head_calls_not_as_forward_passes(tmp_path):
    """A Head call is not a DFM forward.  Charging it as one would inflate the DFE total that
    A2 matching is computed on by the size of every batch."""
    _run(cost_meter=_meter(tmp_path))
    head = [event for event in _events(tmp_path) if event.phase == "head"]
    assert head
    assert all(event.head_calls > 0 for event in head)
    assert all(event.logical_dfe == 0 for event in head)


def test_the_cycle_reports_the_policy_identity_that_actually_answered():
    """PLAN §5.3 makes the policy a load-bearing field of every feedback event.

    The identity has to be the one carried by the DECISION the kernel acted on, not one read back
    off the policy object afterwards.  The two are separate sources that can disagree -- a callable
    can name itself one thing and stamp its answers with another -- and the artifact must record
    which policy produced this transition, not which policy the runner believes it called.
    """
    outcome = _run()
    assert outcome.policy_identity == _policy_identity()


def test_a_refused_projection_still_names_the_policy_that_refused():
    """A null event is a result (PLAN §4.5).  One that could not say which policy declined would
    leave the refusal unattributable."""
    rejecting = lambda source, endpoint, coordinates: pol.PolicyRejection(  # noqa: E731
        reason="declines by construction", policy=_policy_identity())
    outcome = _run(support_policy=rejecting)
    assert outcome.outcome is st.TransitionOutcome.NULL_INVALID_POLICY_RESULT
    assert outcome.policy_identity == _policy_identity()


def test_a_cycle_that_never_reached_the_policy_reports_no_identity():
    """A2 stops before feedback, so there is no policy answer to attribute -- and inventing one
    would put a policy in the record that was never asked."""
    assert _run(feedback_enabled=False).policy_identity is None


def test_every_endpoint_names_the_cost_events_that_paid_for_it(tmp_path):
    """PLAN §5.3 lists "cost" among the load-bearing fields of ``complete endpoints``.

    ``bind_head_scores`` takes ``cost_event_ids`` and the cycle never passed any, so every endpoint
    shipped ``cost_event_ids_json = "[]"`` and no endpoint could be joined to the ledger events that
    paid for it -- the screen that forked it, the Head batch that scored it, the refold that
    validated it.  The compute ledger and the endpoint table were two unconnected artifacts.
    """
    outcome = _run(cost_meter=_meter(tmp_path))
    journaled = {event.event_id for event in _events(tmp_path)}
    assert outcome.endpoints
    for endpoint in outcome.endpoints + outcome.descendant_endpoints:
        assert endpoint.cost_event_ids, f"{endpoint.endpoint_id} names no cost event"
        assert set(endpoint.cost_event_ids) <= journaled, (
            "an endpoint names a cost event the journal never recorded")
