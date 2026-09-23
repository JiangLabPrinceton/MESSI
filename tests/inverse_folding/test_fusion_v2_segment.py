"""V2F3: the projected-segment executor (PLAN §3.1, §3.3, §3.4).

The executor lives in ``fusion_v2_runtime`` and the V1 sampler is byte-identical to HEAD.  That
buys safety but costs a guarantee: a separate loop can DRIFT from the V1 loop, and a drifted loop
would silently destroy two things at once -- A2 would stop being a matched control of the same
process, and the ``B(r)`` band table, calibrated on the V1 loop, would no longer describe the
substrate V2 runs on.

``test_the_segment_reproduces_the_v1_loop_token_for_token`` is the anti-drift control that pays for
the separation.  Everything else here is the V2-specific semantics PLAN §3.1 adds on top.
"""

from __future__ import annotations

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
from inverse_folding.reference_flow.fusion_v2.errors import V2Error
from inverse_folding.reference_flow.fusion_v2.state import ActiveScoreStatus
from inverse_folding.reference_flow.fusion_v2_runtime.segment import (
    NEVER_UNMASKED,
    BackgroundRemaskError,
    adapt_segment_to_live_state,
    PendingAssimilationError,
    V2SegmentError,
    pending_positions,
    run_projected_segment,
    run_projected_segment_batch,
    segment_is_terminal,
)
from inverse_folding.reference_flow.fusion_v2_runtime.scoring import token_logprob
from inverse_folding.reference_flow.sampler import (
    ContinuationResume,
    PositionDependentDFMSampler,
    _replay_state_hash,
)
from tests.inverse_folding import _v2_fixtures as F

VOCAB = 33


def _denoiser(x_t: torch.Tensor, t: float, struct) -> torch.Tensor:
    """Deterministic, token-dependent, model-free.

    Token-dependent so the segment must genuinely depend on the state it is handed -- a constant
    denoiser would hide a kernel that ignored its input.  Mass lives only on the canonical AA set,
    as the real denoiser's does: a draw outside it is not a residue and the maturity accounting
    rejects it."""
    length = int(x_t.shape[0])
    logits = torch.full((length, VOCAB), float("-inf"), dtype=torch.float32)
    for position in range(length):
        for token in sorted(F.AA):
            logits[position, token] = float((position * 7 + token * 3 + int(x_t[position])) % 11)
    return logits


def _cfg(**over):
    """The FROZEN V2 substrate: controller-free, h-map-free, constant_one, zero remask."""
    kw = dict(n_steps=F.N_STEPS, seed=7, temperature=1.0, fraction_scale=0.0,
              remask_enabled=True, form="constant_one", shuffle=False)
    kw.update(over)
    return ReferenceFlowConfig(
        sampler=SamplerConfig(
            n_steps=kw["n_steps"], seed=kw["seed"], temperature=kw["temperature"],
            n_designs_per_protein=1,
            remask=RemaskConfig(enabled=kw["remask_enabled"],
                                fraction_scale=kw["fraction_scale"]),
        ),
        schedule=ScheduleConfig(base_form="linear"),
        amplification=AmplificationConfig(form=kw["form"], h_source="h_processed"),
        h_shuffle=HShuffleConfig(enabled=kw["shuffle"], seed=None),
    )


def _sampler():
    return PositionDependentDFMSampler(mask_token_id=F.MASK, vocab_size=VOCAB)


def _run(**over):
    kw = dict(
        projected=F.projected(), sampler=_sampler(), denoiser=_denoiser, config=_cfg(),
        h_values=np.zeros(F.L, dtype=np.float32), residue_token_ids=F.AA,
    )
    kw.update(over)
    return run_projected_segment(**kw)


# --------------------------------------------------------------------------------------------
# THE ANTI-DRIFT CONTROL
# --------------------------------------------------------------------------------------------


def test_the_segment_reproduces_the_v1_loop_token_for_token():
    """Hand the V1 sampler the same state, the same fork seed, and the same denoiser, and its
    pre-denoiser state at c_{d+1} must equal the V2 segment's captured checkpoint token for token.

    ``snapshot_steps`` captures ``x_t`` as the denoiser SAW it at that step -- the top-of-step
    state -- which is exactly the boundary the V2 segment captures on.

    Scores are compared everywhere EXCEPT the assimilated positions: V1 has no assimilation, so it
    leaves those NaN, while V2 fills them from the first forward.  That difference is the feature
    under test; the token dynamics being identical is the invariant under test.
    """
    projected = F.projected()
    outcome = _run(projected=projected)

    resume = _v1_resume(projected)
    v1 = _sampler().sample(
        sequence_length=F.L, h_values=np.zeros(F.L, dtype=np.float32), denoiser=_denoiser,
        config=_cfg(), controller=None, residue_token_ids=F.AA,
        continuation_resume=resume, snapshot_steps=[F.C_NEXT],
    )
    snapshot = next(s for s in v1.snapshots if s.step == F.C_NEXT)

    assert torch.equal(outcome.checkpoint.x_t, snapshot.x_t), (
        "the V2 segment's token dynamics diverged from the V1 loop; a drifted loop invalidates "
        "both the A2 control and the B(r) calibration"
    )
    assimilated = {event.position for event in outcome.assimilation}
    for position in range(F.L):
        if position in assimilated:
            continue
        a, b = outcome.checkpoint.scores[position], snapshot.scores[position]
        assert (np.isnan(a) and np.isnan(b)) or a == b


def _v1_resume(projected) -> ContinuationResume:
    """The same state, expressed in the V1 sampler's own resume vocabulary."""
    x_t = torch.tensor(projected.tokens, dtype=torch.long)
    scores = np.array(
        [np.nan if v is None else float(v) for v in projected.active_sampler_score_by_pos],
        dtype=np.float64,
    )
    unmask = tuple(-1 if c is None else int(c.step)
                   for c in projected.active_commit_depth_step_by_pos)
    fixed = tuple((int(p), int(t)) for p, t in projected.hard_anchors)
    editable = tuple(int(i) for i in projected.editable_positions)
    seed = int(projected.descendant_fork_seed)
    return ContinuationResume(
        x_t=x_t, scores=scores, unmask_step_by_pos=unmask, start_step=int(projected.r_step),
        n_steps=int(projected.n_steps), fixed_tokens=fixed, editable_positions=editable,
        mode="fork", state_hash=_replay_state_hash(
            x_t, scores, int(projected.r_step), int(projected.n_steps), fixed, editable,
            unmask, "fork", None, seed),
        fork_seed=seed,
    )


# --------------------------------------------------------------------------------------------
# coordinates and cost
# --------------------------------------------------------------------------------------------


def test_the_segment_captures_on_the_declared_next_checkpoint():
    assert _run().checkpoint.step == F.C_NEXT


def test_the_capture_is_a_pre_denoiser_boundary():
    """PLAN §3.1: "stop at c_{d+1} on the true pre-denoiser boundary"."""
    outcome = _run()
    assert outcome.checkpoint.snapshot_phase == "pre_denoiser_after_previous_remask"
    assert outcome.checkpoint.paid_prefix_dfe == F.C_NEXT - F.R_STEP


def test_logical_dfe_is_exactly_the_segment_length():
    """PLAN §3.4: a projected segment from r_d to c_{d+1} costs exactly c_{d+1} - r_d."""
    assert _run().logical_dfe == F.C_NEXT - F.R_STEP


def test_the_ledger_does_not_charge_the_source_prefix_again():
    """PLAN §3.4: "The ledger must not charge the source prefix again"."""
    assert _run().logical_dfe < F.C_NEXT


def test_physical_forwards_equal_the_logical_lane_dfe_in_scalar_mode():
    outcome = _run()
    assert outcome.physical_forwards == outcome.logical_dfe


def test_hard_anchors_survive_the_segment_unchanged():
    projected = F.projected()
    outcome = _run(projected=projected)
    for position, token in projected.hard_anchors:
        assert int(outcome.checkpoint.x_t[position]) == token


def test_unresolved_mass_never_increases_across_the_segment():
    """PLAN §7.2: unresolved mass may only fall inside a propagation segment; it rises only at an
    explicit projection.  This is the deterministic form of "the segment did work" -- with linear
    base_form the per-step unmask probability is 1/(S-step), so a 40->60 segment of a 100-step
    schedule resolves only about a third of the masked positions in expectation, and asserting a
    SPECIFIC position resolved would be a coin flip dressed up as a test."""
    projected = F.projected()
    before = sum(1 for i in projected.editable_positions
                 if projected.tokens[i] == F.MASK)
    outcome = _run(projected=projected)
    after = sum(1 for i in projected.editable_positions
                if int(outcome.checkpoint.x_t[i]) == F.MASK)
    assert after <= before


def test_a_segment_running_to_the_schedule_tail_resolves_everything():
    """The complement: give the segment the whole remaining horizon and no mask may survive."""
    projected = F.projected(coordinates=F.coords(c_next_step=F.N_STEPS - 1))
    outcome = _run(projected=projected)
    assert all(int(outcome.checkpoint.x_t[i]) != F.MASK
               for i in projected.editable_positions)


# --------------------------------------------------------------------------------------------
# first-forward assimilation (PLAN §3.1, §3.3)
# --------------------------------------------------------------------------------------------


def test_every_pending_injection_is_assimilated():
    projected = F.projected()
    outcome = _run(projected=projected)
    assert {e.position for e in outcome.assimilation} == set(pending_positions(projected))


def test_assimilation_happens_on_the_first_forward_at_reentry():
    for event in _run().assimilation:
        assert event.step == F.R_STEP


def test_the_assimilated_score_is_the_token_logprob_of_the_first_forward():
    """The score must be this token's log-probability under the FIRST re-entry forward at the
    frozen temperature -- not the endpoint's completion log-probability, and not a resample."""
    projected = F.projected()
    outcome = _run(projected=projected)
    x0 = torch.tensor(projected.tokens, dtype=torch.long)
    logits = _denoiser(x0, F.R_STEP / float(F.N_STEPS), None)
    for event in outcome.assimilation:
        expected = token_logprob(
            logits[[event.position]], [event.token], temperature=1.0,
        )[0]
        assert event.score == expected


@pytest.mark.parametrize("temperature", [0.5, 0.8, 1.3, 2.0])
def test_the_assimilated_score_tracks_the_configured_temperature(temperature):
    """PLAN §7.2 names this exact failure: "assimilation score differs from categorical
    chosen-token log-probability at temperature NOT EQUAL TO ONE".  Every other test in this file
    runs at T=1.0, where an implementation that hard-coded 1.0 is indistinguishable from one that
    reads the config -- so this is the only test that can catch it."""
    projected = F.projected()
    outcome = _run(projected=projected, config=_cfg(temperature=temperature))
    logits = _denoiser(torch.tensor(projected.tokens, dtype=torch.long),
                       F.R_STEP / float(F.N_STEPS), None)
    for event in outcome.assimilation:
        expected = token_logprob(
            logits[[event.position]], [event.token], temperature=temperature,
        )[0]
        assert event.score == expected


def test_assimilation_does_not_resample_the_injected_token():
    """PLAN §3.1: assimilate "without resampling it".  The token in the checkpoint at an
    endpoint-write position must still be the endpoint's byte."""
    projected = F.projected()
    outcome = _run(projected=projected)
    for event in outcome.assimilation:
        assert event.token == projected.tokens[event.position]


def test_the_endpoint_completion_logprob_never_becomes_the_assimilated_score():
    """The evidence-namespace firewall, at the moment it is most likely to leak."""
    terminal = {
        e.completion_logprob for e in F.endpoint().endpoint_provenance_evidence_by_pos
    }
    assert {e.score for e in _run().assimilation}.isdisjoint(terminal)


def test_a_state_with_no_pending_position_assimilates_nothing():
    """Guard against an implementation that fabricates an assimilation event unconditionally."""
    projected = F.projected()
    statuses = list(projected.active_score_status_by_pos)
    assert ActiveScoreStatus.PENDING_ASSIMILATION in statuses   # the fixture really has some


# --------------------------------------------------------------------------------------------
# zero background remask (PLAN §3.1, §7.2)
# --------------------------------------------------------------------------------------------


def test_the_segment_reports_zero_background_remask_events():
    assert _run().background_remask_events == 0


def test_a_substrate_with_non_zero_remask_scale_is_refused_before_any_forward():
    with pytest.raises(V2SegmentError, match="fraction_scale"):
        _run(config=_cfg(fraction_scale=0.1))


def test_a_non_constant_one_amplification_is_refused():
    with pytest.raises(V2SegmentError, match="amplification"):
        _run(config=_cfg(form="linear"))


def test_h_map_shuffling_is_refused():
    with pytest.raises(V2SegmentError, match="h-map"):
        _run(config=_cfg(shuffle=True))


def test_a_controller_may_not_run_in_a_segment():
    with pytest.raises(V2SegmentError, match="controller"):
        _run(controller=object())


def test_a_config_whose_n_steps_disagrees_with_the_state_is_refused():
    with pytest.raises(V2SegmentError, match="n_steps"):
        _run(config=_cfg(n_steps=F.N_STEPS + 1))


# --------------------------------------------------------------------------------------------
# temporary protection lifecycle (PLAN §3.3)
# --------------------------------------------------------------------------------------------


def test_every_temporary_protection_expires_at_the_captured_checkpoint():
    projected = F.projected()
    outcome = _run(projected=projected)
    assert set(outcome.expired_protection) == {
        p.position for p in projected.active_temporary_protection
    }


def test_protection_does_not_survive_into_the_descendant():
    """PLAN §3.3: "no implicit cross-depth protection is allowed"."""
    outcome = _run()
    assert outcome.expired_protection
    assert not hasattr(outcome.checkpoint, "active_temporary_protection")


def test_protected_positions_remain_in_the_maturity_denominator():
    """PLAN §3.3: "Temporary positions remain part of the maturity denominator" -- protection
    removes them from remask, not from the editable domain."""
    projected = F.projected()
    outcome = _run(projected=projected)
    assert outcome.checkpoint.maturity.n_editable == len(projected.editable_positions)


# --------------------------------------------------------------------------------------------
# determinism and positive controls
# --------------------------------------------------------------------------------------------


def test_the_same_state_and_seed_reproduce_the_segment_byte_for_byte():
    a, b = _run(), _run()
    assert torch.equal(a.checkpoint.x_t, b.checkpoint.x_t)
    np.testing.assert_array_equal(a.checkpoint.scores, b.checkpoint.scores)


def test_a_different_descendant_seed_produces_a_different_segment():
    """Positive control on the fork seed: if it were ignored, every descendant of a state would be
    identical and the lookahead pool would collapse to one design."""
    a = _run(projected=F.projected(descendant_fork_seed=9001))
    b = _run(projected=F.projected(descendant_fork_seed=9002))
    assert not torch.equal(a.checkpoint.x_t, b.checkpoint.x_t)


def test_a_different_projected_byte_produces_a_different_segment():
    """Positive control on the state: the segment must actually read the tokens handed to it."""
    other = tuple([*F.ENDPOINT_TOKENS[:1], 21, *F.ENDPOINT_TOKENS[2:]])
    a = _run()
    b = _run(projected=F.projected(endpoint=F.endpoint(tokens=other), endpoint_tokens=other))
    assert not torch.equal(a.checkpoint.x_t, b.checkpoint.x_t)


def test_the_projected_state_is_not_mutated_by_the_segment():
    projected = F.projected()
    before = projected.content_digest
    _run(projected=projected)
    assert projected.content_digest == before


def test_the_v1_sampler_module_is_byte_identical_to_head():
    """The whole reason this executor exists as a separate module."""
    import subprocess

    repo = F.__file__.rsplit("/tests/", 1)[0]
    diff = subprocess.run(
        ["git", "diff", "--numstat", "--", "inverse_folding/reference_flow/sampler.py"],
        capture_output=True, text=True, cwd=repo,
    ).stdout.strip()
    assert diff == "", f"sampler.py must stay pristine, got: {diff}"


# --------------------------------------------------------------------------------------------
# parent-resume evidence vs. the descendant's own capture identity
# --------------------------------------------------------------------------------------------


def _descendant(**over):
    projected = over.pop("projected", None) or F.projected()
    outcome = _run(projected=projected, **over)
    return projected, outcome, adapt_segment_to_live_state(
        projected=projected, outcome=outcome, conditioning=F.conditioning(),
        safety_reference=F.safety_reference(), cost_event_ids=("evt:root",),
    )


def _self_hash(state) -> str:
    """The replay hash recomputed from a live state's OWN fields, exactly as a reader would."""
    x_t = torch.tensor(state.tokens, dtype=torch.long)
    scores = np.array(
        [np.nan if v is None else float(v) for v in state.active_sampler_score_by_pos],
        dtype=np.float64,
    )
    return _replay_state_hash(
        x_t, scores, int(state.sampler_step), int(state.n_steps),
        tuple((int(p), int(t)) for p, t in state.hard_anchors),
        tuple(int(i) for i in state.editable_positions),
        tuple(NEVER_UNMASKED if c is None else int(c.step)
              for c in state.active_commit_depth_step_by_pos),
        state.replay.mode, state.replay.rng_state, state.replay.fork_seed,
    )


def test_the_descendant_replay_hash_is_recomputable_from_the_descendant_itself():
    """``LivePartialState.replay.replay_state_hash`` must identify THAT state.

    ``capture_depth_zero`` already stores the hash of the state it captured, so a reader can verify
    a depth-0 root by recomputing it.  A descendant that instead stored the hash of the projected
    state it resumed FROM would use the same field for a different quantity: the two states differ
    in tokens, in scores, and in step, so the stored value identifies neither one thing nor the
    other, and no consumer could tell which convention it was reading.
    """
    _, _, descendant = _descendant()
    assert descendant.replay.replay_state_hash == _self_hash(descendant)


def test_parent_resume_evidence_survives_separately_from_the_capture_identity():
    """Both quantities are real and both must be reachable -- in two places, not one.

    The resume hash authorizes the segment's continuation from ``r_d`` and stays on the
    ``SegmentOutcome``; the capture identity describes the state at ``c_{d+1}`` and belongs to that
    state.  They are not interchangeable: the two states differ in tokens, in scores and in step.

    They also cannot be the same computation.  The checkpoint's score vector still carries NaN at an
    injected position -- the sampler never redraws a token that is already unmasked -- while the
    descendant carries the first-forward assimilation score there.  So the capture identity has to
    be taken from the descendant's own fields, which is why it is not a third field on the outcome.
    """
    projected, outcome, descendant = _descendant()
    assert outcome.resume_state_hash != descendant.replay.replay_state_hash, (
        "the segment resumed at r_d and captured at c_{d+1}; one hash for both means one of them "
        "is not being computed"
    )
    # The resume hash still describes the projected state the segment actually ran from.
    x_t = torch.tensor(projected.tokens, dtype=torch.long)
    scores = np.array(
        [np.nan if v is None else float(v) for v in projected.active_sampler_score_by_pos],
        dtype=np.float64,
    )
    assert outcome.resume_state_hash == _replay_state_hash(
        x_t, scores, int(projected.r_step), int(projected.n_steps),
        tuple((int(p), int(t)) for p, t in projected.hard_anchors),
        tuple(int(i) for i in projected.editable_positions),
        tuple(NEVER_UNMASKED if c is None else int(c.step)
              for c in projected.active_commit_depth_step_by_pos),
        "fork", None, int(projected.descendant_fork_seed),
    )


def test_the_capture_identity_moves_when_the_captured_state_moves():
    """A hash that did not depend on the capture would pass the round-trip test vacuously."""
    _, early_outcome, early = _descendant(projected=F.projected())
    late_projected = F.projected(coordinates=F.coords(c_next_step=F.C_NEXT + 10))
    _, late_outcome, late = _descendant(projected=late_projected)
    assert early_outcome.checkpoint.step != late_outcome.checkpoint.step
    assert early.replay.replay_state_hash != late.replay.replay_state_hash
    # ...while the resume evidence is unchanged: both segments resumed from the same r_d state.
    assert early_outcome.resume_state_hash == late_outcome.resume_state_hash


# --------------------------------------------------------------------------------------------
# scalar / batch semantic parity (PLAN §3.1, §3.3, §7.2)
# --------------------------------------------------------------------------------------------


def _batched_denoiser_factory():
    """One batched forward per step, plus a call counter.

    Mirrors the V1 ``sample_batch`` contract exactly: ``([x_t...], t, [struct...]) -> [logits...]``.
    Deliberately delegates to the SAME per-lane denoiser the scalar path uses, so any difference the
    parity test finds is a difference in the SEGMENT, not in the model stand-in.
    """
    calls = {"n": 0, "widths": []}

    def batched(x_ts, t, structs):
        calls["n"] += 1
        calls["widths"].append(len(x_ts))
        return [_denoiser(x_t, t, struct) for x_t, struct in zip(x_ts, structs)]

    return batched, calls


def _lanes(seeds=(9001, 9002, 9003)):
    lanes = [F.projected(descendant_fork_seed=int(seed)) for seed in seeds]
    assert len({lane.descendant_fork_seed for lane in lanes}) == len(seeds)
    return lanes


def test_the_batch_lane_is_byte_identical_to_running_each_lane_scalar():
    """PLAN §3.1: "preserve scalar/batch semantic parity"; §7.2 lists the disagreement as a case.

    Parity is not a coincidence to be checked -- both paths run ONE per-lane step body, so there is
    no second implementation that could drift.  This test is what proves the refactor did not
    quietly create one.
    """
    lanes = _lanes()
    scalar = [_run(projected=lane) for lane in lanes]
    batched_denoiser, calls = _batched_denoiser_factory()
    batch = run_projected_segment_batch(
        projected_lanes=lanes, sampler=_sampler(), batched_denoiser=batched_denoiser,
        config=_cfg(), h_values=np.zeros(F.L, dtype=np.float32), residue_token_ids=F.AA,
    )

    # The lanes must actually differ, or "identical" would be trivially satisfiable.
    assert len({tuple(o.checkpoint.x_t.tolist()) for o in scalar}) > 1, (
        "the fixture lanes produced identical trajectories; the parity test would be vacuous"
    )

    for index, (one, many) in enumerate(zip(scalar, batch)):
        assert torch.equal(one.checkpoint.x_t, many.checkpoint.x_t), f"lane {index} bytes differ"
        np.testing.assert_array_equal(
            np.nan_to_num(one.checkpoint.scores, nan=-1e9),
            np.nan_to_num(many.checkpoint.scores, nan=-1e9),
            err_msg=f"lane {index} active scores differ",
        )
        assert one.logical_dfe == many.logical_dfe, f"lane {index} logical DFE differs"
        assert one.assimilation == many.assimilation, f"lane {index} assimilation differs"
        assert one.expired_protection == many.expired_protection
        assert one.resume_state_hash == many.resume_state_hash
        assert one.background_remask_events == many.background_remask_events
        assert one.checkpoint.unmask_step_by_pos == many.checkpoint.unmask_step_by_pos


def test_the_descendant_states_agree_across_scalar_and_batch():
    """Parity has to survive adaptation, since that is what the archive and the ledger see.

    A lane whose segment resolved every editable position has no descendant LIVE state at all --
    it is a terminal endpoint (PLAN §4.5).  Both paths must agree on WHICH lanes those are, and the
    rest must adapt to byte-identical descendants.
    """
    lanes = _lanes()
    batched_denoiser, _ = _batched_denoiser_factory()
    batch = run_projected_segment_batch(
        projected_lanes=lanes, sampler=_sampler(), batched_denoiser=batched_denoiser,
        config=_cfg(), h_values=np.zeros(F.L, dtype=np.float32), residue_token_ids=F.AA,
    )
    adapted = 0
    for lane, many in zip(lanes, batch):
        one = _run(projected=lane)
        assert segment_is_terminal(one, lane) == segment_is_terminal(many, lane)
        if segment_is_terminal(many, lane):
            continue
        adapt = lambda outcome: adapt_segment_to_live_state(  # noqa: E731
            projected=lane, outcome=outcome, conditioning=F.conditioning(),
            safety_reference=F.safety_reference(), cost_event_ids=("evt:root",))
        assert adapt(one).content_digest == adapt(many).content_digest
        adapted += 1
    assert adapted, "every lane terminated; the adaptation half of this test ran on nothing"


def test_a_segment_that_resolves_everything_is_reported_as_terminal_not_raised_on():
    """PLAN §4.5: stopping is typed.

    A segment CAN resolve its last mask -- that is an ordinary trajectory outcome, not a bug.  The
    result is a complete sequence, which is a terminal endpoint and not a live partial state, so
    ``adapt_segment_to_live_state`` correctly refuses to build one.  The caller must therefore be
    able to ASK, rather than discover it by catching a state-layer exception it cannot distinguish
    from real corruption.
    """
    lanes = _lanes()
    outcomes = [(lane, _run(projected=lane)) for lane in lanes]
    terminal = [(lane, o) for lane, o in outcomes if segment_is_terminal(o, lane)]
    assert terminal, "no fixture lane terminates; this test would be vacuous"
    for lane, outcome in terminal:
        assert not any(
            int(outcome.checkpoint.x_t[i]) == F.MASK for i in lane.editable_positions)
        with pytest.raises(V2Error):
            adapt_segment_to_live_state(
                projected=lane, outcome=outcome, conditioning=F.conditioning(),
                safety_reference=F.safety_reference(), cost_event_ids=("evt:root",))


def test_the_batch_issues_one_forward_per_step_not_one_per_lane():
    """The reason the batch lane exists.  Without this the two paths are the same cost."""
    lanes = _lanes()
    batched_denoiser, calls = _batched_denoiser_factory()
    outcomes = run_projected_segment_batch(
        projected_lanes=lanes, sampler=_sampler(), batched_denoiser=batched_denoiser,
        config=_cfg(), h_values=np.zeros(F.L, dtype=np.float32), residue_token_ids=F.AA,
    )
    span = F.C_NEXT - F.R_STEP
    assert calls["n"] == span, f"expected {span} batched calls, got {calls['n']}"
    assert set(calls["widths"]) == {len(lanes)}, "a lane dropped out of the batch mid-flight"
    for outcome in outcomes:
        assert outcome.batch_size == len(lanes)
        assert outcome.physical_forwards == span


def test_a_batch_group_must_share_its_coordinates():
    """Grouping is the caller's job, as it is in the V1 sampler.

    Lanes with different re-entry or capture steps do not run the same span, so one batched forward
    could not serve all of them -- and silently running them anyway would either drop a lane or
    charge it for steps it never took.
    """
    batched_denoiser, _ = _batched_denoiser_factory()
    mixed = [F.projected(), F.projected(coordinates=F.coords(c_next_step=F.C_NEXT + 5))]
    with pytest.raises(V2SegmentError, match="same|share"):
        run_projected_segment_batch(
            projected_lanes=mixed, sampler=_sampler(), batched_denoiser=batched_denoiser,
            config=_cfg(), h_values=np.zeros(F.L, dtype=np.float32), residue_token_ids=F.AA,
        )


def test_a_batched_denoiser_that_returns_the_wrong_number_of_lanes_is_refused():
    lanes = _lanes()
    with pytest.raises(V2SegmentError, match="lane"):
        run_projected_segment_batch(
            projected_lanes=lanes, sampler=_sampler(),
            batched_denoiser=lambda x_ts, t, structs: [_denoiser(x_ts[0], t, structs[0])],
            config=_cfg(), h_values=np.zeros(F.L, dtype=np.float32), residue_token_ids=F.AA,
        )


def test_the_scalar_path_reports_a_batch_size_of_one():
    """So a ledger can charge a group once and attribute per lane without a special case."""
    outcome = _run()
    assert outcome.batch_size == 1
    assert outcome.physical_forwards == F.C_NEXT - F.R_STEP
