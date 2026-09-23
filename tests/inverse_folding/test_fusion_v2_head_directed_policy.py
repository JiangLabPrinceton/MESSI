"""V2F5A: the minimal capped Head-directed policy and its matched source-geometry control.

Every test here maps to a PLAN task V2F5A acceptance bullet.  The suite exists because S7 showed
that a mechanism can transmit identity perfectly and still transmit no reward: the donor was ranked
by exact complete Head risk, the written position was chosen by source mask geometry, and the two
had nothing to do with each other.  So what is checked below is never "the policy ran" -- it is
that each chosen position carries the specific evidence the PLAN requires, that the cap is a cap
rather than a quota, that the reopen count reproduces the calibrated band centre EXACTLY, and that
every way of having no evidence produces a typed stall instead of a write.

**Scientific boundary.**  A green suite proves executability and bookkeeping.  ``a_i`` here is
computed by a scripted Head whose risk is a known function of the residues, which makes the
selection law testable; it says nothing about whether a real frozen Head's contribution evidence
predicts a descendant shift.  Only the matched one-cycle cluster comparison can support that.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import pytest

from inverse_folding.reference_flow.fusion.state import sequence_md5
from inverse_folding.reference_flow.fusion.v1_admission import StructureOutcome
from inverse_folding.reference_flow.fusion_v2 import config as cfg
from inverse_folding.reference_flow.fusion_v2 import evidence as ev
from inverse_folding.reference_flow.fusion_v2 import identity as ident
from inverse_folding.reference_flow.fusion_v2 import policy as pol
from inverse_folding.reference_flow.fusion_v2 import reward as rw
from inverse_folding.reference_flow.fusion_v2 import schedule as sch
from inverse_folding.reference_flow.fusion_v2 import state as st
from tests.inverse_folding import _v2_fixtures as F

# --------------------------------------------------------------------------------------------
# a world big enough for a MULTI-position write
# --------------------------------------------------------------------------------------------
#
# The shared fixture is length 6 with one source mask, which can only ever exercise ``m_d == 1`` --
# exactly the fixed one-token cardinality PLAN Appendix A retires.  This world has 11 editable
# positions and 4 source masks, so the cap, the positive-contribution filter and the band solver
# can each bind in turn and be told apart.

L = 12
ANCHOR = 0
MASKED = (3, 5, 7, 9)
LATE = 4                       # resolved AT/AFTER r_d: future information, must be injected
R_STEP, C_SOURCE, C_NEXT = F.R_STEP, F.C_SOURCE, F.C_NEXT

CANONICAL_AA20 = "ACDEFGHIKLMNPQRSTVWY"
#: residue -> risk.  Alphabetically earlier is LOWER risk, so a design's global risk and every
#: window's z are exact, hand-computable functions of its residues -- which is what lets a test say
#: "position 5 must rank first" and mean it.
RESIDUE_RISK = {letter: index * 0.1 for index, letter in enumerate(CANONICAL_AA20)}
WINDOW_K = F.WINDOW_K


def _global_risk(sequence: str) -> float:
    return round(sum(RESIDUE_RISK[c] for c in sequence), 10)


def _windows(sequence: str):
    """One window per k-mer, z = mean residue risk inside it.

    A window therefore improves in the donor exactly when the donor lowered some residue risk
    inside it, which makes the PLAN A.3 window screen checkable by construction.
    """
    return tuple(
        F.HeadWindow(start_0b=start, end_0b=start + WINDOW_K, k=WINDOW_K,
                     z=round(sum(RESIDUE_RISK[c] for c in sequence[start:start + WINDOW_K])
                             / WINDOW_K, 10))
        for start in range(len(sequence) - WINDOW_K + 1)
    )


def _evaluator():
    return F.safety_reference(length=L).head_binding.evaluator


def _score(sequence: str, protein_id: str = "5ZHV_B"):
    evaluator = _evaluator()
    return F.EndpointHeadScore(
        protein_id=protein_id, sequence_md5=sequence_md5(sequence), sequence_length=len(sequence),
        allele=evaluator.allele, score_scale=evaluator.score_scale, windows=_windows(sequence),
        # Present but never read: PLAN §2.5 forbids cross-sequence evidence built from an
        # independently centered residue summary, and an AST test below pins that structurally.
        residue_hotspot=(0.0,) * len(sequence), global_risk=_global_risk(sequence),
    )


@dataclasses.dataclass(frozen=True)
class _Result:
    """What a Head oracle returns: the score plus the binding results are matched BY."""

    score: object
    binding: ident.HeadScoreBinding
    sequence_md5: str
    global_risk: float

    # ``bind_head_scores`` stores the RESULT itself as the endpoint's ``head_score``, so a result
    # has to satisfy ``HeadScoreLike`` -- protein, length, allele, scale and the window grid.
    @property
    def windows(self):
        return self.score.windows

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
    def residue_hotspot(self):
        return self.score.residue_hotspot


class _ScriptedHead:
    """A deterministic Head whose risk is a known function of the residues.

    It scores the SEQUENCE and derives its own join key from it, so a caller that handed it a
    digest, a truncated counterfactual or a masked string fails here rather than on the cluster.
    """

    def __init__(self, protein_id: str = "5ZHV_B"):
        self.protein_id = protein_id
        self.batches: list[list[str]] = []

    def evaluator_identity(self):
        """Same shape as ``ProductionHeadOracle``: the instrument names itself."""
        return _evaluator()

    def score(self, requests):
        self.batches.append([request.sequence for request in requests])
        rows = []
        for request in requests:
            score = _score(request.sequence, protein_id=self.protein_id)
            rows.append(_Result(
                score=score,
                binding=ident.HeadScoreBinding(
                    protein_id=self.protein_id, sequence_md5=score.sequence_md5,
                    sequence_length=len(request.sequence),
                    window_grid_digest=ident.window_grid_digest(score.windows),
                    evaluator=_evaluator(),
                ),
                sequence_md5=score.sequence_md5, global_risk=score.global_risk,
            ))
        return rows


def _scorer(head=None):
    """The pure ``scorer(protein_id, sequences)`` contract, through the real runtime adapter."""
    from inverse_folding.reference_flow.fusion_v2_runtime.contribution import (
        CounterfactualHeadScorer,
    )

    return CounterfactualHeadScorer(head_oracle=head or _ScriptedHead())


def _tokens(sequence: str) -> tuple[int, ...]:
    return tuple(CANONICAL_AA20.index(letter) + min(F.AA) for letter in sequence)


def _source(sequence: str, **over):
    """A live partial state at ``c_d`` whose masked positions are :data:`MASKED`.

    ``sequence`` supplies the RESOLVED residues; masked positions take the mask token whatever the
    letter there says, so one string can describe both the source and the sequences derived from it.
    """
    tokens = list(_tokens(sequence))
    for position in MASKED:
        tokens[position] = F.MASK
    editable = tuple(p for p in range(L) if p != ANCHOR)
    kind, ref, commit, score, status, prov = [], [], [], [], [], []
    for position in range(L):
        if position == ANCHOR:
            kind.append(st.ActiveOriginKind.HARD_ANCHOR)
            commit.append(None)
            score.append(None)
            status.append(st.ActiveScoreStatus.NOT_RANKED_ANCHOR)
            prov.append(F.provenance(tokens[position], kind=st.ActiveOriginKind.HARD_ANCHOR))
        elif position in MASKED:
            kind.append(st.ActiveOriginKind.UNRESOLVED)
            commit.append(None)
            score.append(None)
            status.append(st.ActiveScoreStatus.MASKED)
            prov.append(F.provenance(F.MASK, kind=st.ActiveOriginKind.UNRESOLVED))
        else:
            step = F.LATE_COMMIT if position == LATE else 10 + position
            kind.append(st.ActiveOriginKind.DENOISER_SAMPLE)
            commit.append(sch.history_key(0, step))
            # Distinct per position, so the "active uncertainty" conjunct of the reopen priority
            # has a real order to sort on rather than a tie broken by index.
            score.append(-0.1 * (position + 1))
            status.append(st.ActiveScoreStatus.HISTORICAL_NATURAL)
            prov.append(F.provenance(tokens[position], kind=st.ActiveOriginKind.DENOISER_SAMPLE,
                                     step=step, n_events=1 + (position % 3)))
        ref.append(st.FeedbackOriginRef.NONE)

    kw = dict(
        schema_version=ident.V2_STATE_SCHEMA_VERSION, lineage=F.lineage(),
        sampler_step=C_SOURCE, n_steps=F.N_STEPS, tokens=tuple(tokens), mask_token_id=F.MASK,
        aa_token_ids=F.AA, hard_anchors=((ANCHOR, tokens[ANCHOR]),), editable_positions=editable,
        active_origin_kind_by_pos=tuple(kind), feedback_origin_ref_by_pos=tuple(ref),
        active_commit_depth_step_by_pos=tuple(commit), origin_transition_id_by_pos=tuple([None] * L),
        active_sampler_score_by_pos=tuple(score), active_score_status_by_pos=tuple(status),
        provenance_by_pos=tuple(prov), expired_protection=(),
        replay=st.ReplayIdentity(mode="identity", rng_state={"pos": 7}, fork_seed=None,
                                 replay_state_hash=F.D),
        accumulated_lineage_dfe=C_SOURCE, conditioning=F.conditioning(),
        safety_reference=F.safety_reference(length=L), cost_event_ids=("evt:root",),
    )
    kw.update(over)
    return st.LivePartialState(**kw)


def _endpoint(sequence: str, source, *, definitive=True, fork_index=0, fork_seed=4242):
    score = _score(sequence)
    binding = ident.HeadScoreBinding(
        protein_id="5ZHV_B", sequence_md5=score.sequence_md5, sequence_length=L,
        window_grid_digest=ident.window_grid_digest(score.windows), evaluator=_evaluator(),
    )
    tokens = _tokens(sequence)
    return st.CompleteEndpoint(
        schema_version=ident.V2_STATE_SCHEMA_VERSION, lineage=source.lineage,
        protein_id="5ZHV_B", sequence=sequence, sequence_md5=score.sequence_md5, sequence_length=L,
        source_state_id=source.state_id, source_state_content_digest=source.content_digest,
        fork_index=fork_index, fork_seed=fork_seed,
        replay=st.ReplayIdentity(mode="fork", rng_state=None, fork_seed=fork_seed,
                                 replay_state_hash=F.E),
        endpoint_provenance_evidence_by_pos=tuple(
            st.EndpointPositionEvidence(
                token=token, commit=sch.history_key(0, 60 + index), completion_logprob=-2.5,
                inherited_from_source=(token == source.tokens[index]))
            for index, token in enumerate(tokens)
        ),
        head_binding=binding, head_score=score, head_global_risk=score.global_risk,
        feasibility_level=(st.FeasibilityLevel.DEFINITIVE if definitive
                           else st.FeasibilityLevel.UNVALIDATED),
        structure_outcome=(StructureOutcome(feasible=True, metrics={"scTM": 0.91})
                           if definitive else None),
        cost_event_ids=("evt:fork0",),
    )


def _band_table(*, unresolved_lo=4.0, unresolved_hi=8.0, step=R_STEP):
    """A band whose PINNED interval is exactly ``[unresolved_lo, unresolved_hi]`` at 11 editable.

    ``rho_accept`` is deliberately the whole unit interval so the absolute axis alone decides the
    pinned interval: this suite is about the CENTRE and the reopen equation, and a normalized axis
    that also bit would make an off-by-one impossible to attribute.
    """
    band = sch.make_band(
        step=step, stratum_key=F.STRATUM, levels=sch.QuantileLevels(levels=(0.1, 0.5, 0.9)),
        rho_quantiles=(0.05, 0.5, 0.95), unresolved_quantiles=(L - 1, 4, 1),
        rho_accept=sch.BandInterval(lo=0.0, hi=1.0, lo_level=0.1, hi_level=0.9),
        unresolved_accept=sch.BandInterval(lo=unresolved_lo, hi=unresolved_hi, lo_level=0.9,
                                           hi_level=0.1),
        combination_rule="both_axes", n_attempts=32, n_captured=32,
        n_editable_min=1, n_editable_max=L,
    )
    return sch.make_band_table(provenance=F.band_provenance(), bands=(band,))


def _calibration(*, fraction=0.3, epsilon=0.0, tolerance=0.0,
                 rule=sch.BandCenterRule.MIDPOINT_TIE_LOW, counterfactual_budget=64):
    return pol.HeadDirectedCalibration(
        write_cap_editable_fraction=fraction, write_cap_source_ref=F.digest("cap"),
        epsilon_r=epsilon, epsilon_source_ref=F.digest("eps"),
        local_contribution_tolerance=tolerance,
        local_contribution_source_ref=F.digest("tol"), band_center_rule=rule,
        max_counterfactual_head_calls_per_cycle=counterfactual_budget,
    )


#: The incumbent's residues.  Deliberately mid-alphabet so a donor can be better at some positions
#: and worse at others -- a uniform incumbent would make every legal candidate positive and the
#: positive filter untestable.
INCUMBENT_SEQ = "KKKKKKKKKKKK"


def _incumbent(sequence: str = INCUMBENT_SEQ, *, kind=None, safety_md5=None):
    """``I_d`` bound as the depth-0 cumulative safety reference, the rule this repo implements."""
    score = _score(sequence)
    binding = ident.SafetyReferenceBinding(
        reference_id="ref:wt", reference_label="wt_native", sequence_md5=score.sequence_md5,
        sequence_length=L, reference_content_digest=F.D,
        head_binding=ident.HeadScoreBinding(
            protein_id="5ZHV_B", sequence_md5=score.sequence_md5, sequence_length=L,
            window_grid_digest=ident.window_grid_digest(score.windows), evaluator=_evaluator()),
        head_score_digest=F.D, bound_at_depth=0, source_kind="predeclared_external",
    )
    reference = dataclasses.make_dataclass(
        "_Ref", [("binding", object), ("head_score", object)], frozen=True,
    )(binding=binding, head_score=score)
    incumbent = rw.bind_incumbent_from_safety_reference(
        reference=reference, reference_sequence=sequence, lineage_id="5ZHV_B:fam0",
        evaluator=_evaluator(), rule="cumulative_safety_reference",
    )
    if kind is not None or safety_md5 is not None:
        incumbent = dataclasses.replace(
            incumbent, kind=kind or incumbent.kind,
            safety_reference_sequence_md5=safety_md5 or incumbent.safety_reference_sequence_md5)
    return incumbent


def _policy(*, incumbent=None, calibration=None, band_table=None, head=None, **over):
    incumbent = incumbent or _incumbent()
    kw = dict(
        band_table=band_table or _band_table(), stratum_key=F.STRATUM, incumbent=incumbent,
        safety_reference_score=_score(incumbent.sequence), evaluator=_evaluator(),
        window_grid_digest=ident.window_grid_digest(_windows(incumbent.sequence)),
        calibration=calibration or _calibration(),
        incumbent_update_law="strict_improvement_by_epsilon",
        counterfactual_scorer=_scorer(head), policy_spec_digest=F.digest("pspec"),
    )
    kw.update(over)
    return pol.HeadDirectedCappedPolicy(**kw)


def _coords(**over):
    kw = dict(depth=0, r_step=R_STEP, c_source_step=C_SOURCE, c_next_step=C_NEXT,
              n_steps=F.N_STEPS, law=sch.CoordinateLaw.PROGRESSIVE_CHECKPOINT)
    kw.update(over)
    return sch.make_cycle(**kw)


#: The default scenario: source resolved at 'K' everywhere (equal to the incumbent, so no resolved
#: position is a write candidate), and a donor that is strictly better at every MASKED position.
SOURCE_SEQ = "KKKKKKKKKKKK"


def _donor(sequence: str | None = None, source=None, **over):
    source = source or _source(SOURCE_SEQ)
    # All four are below the incumbent's 'K', and their risks are DISTINCT, so the contribution
    # ranking is a fact rather than a tie broken by index (which has its own test).
    default = list(SOURCE_SEQ)
    for offset, position in enumerate(MASKED):
        default[position] = "ACDE"[offset]
    return _endpoint(sequence or "".join(default), source, **over)


def _decide(policy=None, source=None, endpoint=None, coordinates=None):
    source = source if source is not None else _source(SOURCE_SEQ)
    return (policy or _policy()).decide(
        source=source, endpoint=endpoint if endpoint is not None else _donor(source=source),
        coordinates=coordinates or _coords(),
    )


# --------------------------------------------------------------------------------------------
# the donor gate
# --------------------------------------------------------------------------------------------


def test_a_donor_that_does_not_beat_the_incumbent_yields_a_typed_stall():
    """PLAN A.2: "If no donor passes this gate, the policy emits a typed
    ``stall_no_better_donor``.  It must not fall back to an arbitrary endpoint token."""
    source = _source(SOURCE_SEQ)
    # 'Y' is the highest-risk residue: this donor is worse than the incumbent everywhere.
    worse = _donor("Y" * L, source=source)
    result = _decide(endpoint=worse, source=source)
    assert isinstance(result, pol.PolicyRejection)
    assert pol.StallReason.NO_BETTER_DONOR.value in result.reason
    assert result.decision_evidence.donor_gate.passed is False
    assert result.decision_evidence.stall_reason == pol.StallReason.NO_BETTER_DONOR.value


def test_the_donor_margin_must_exceed_the_calibrated_epsilon():
    """A win inside the frozen Head's own noise floor is not a win (PLAN §2.5)."""
    source = _source(SOURCE_SEQ)
    donor = _donor(source=source)
    margin = _global_risk(INCUMBENT_SEQ) - float(donor.head_global_risk)
    assert margin > 0.0, "the fixture donor must be better, or this test proves nothing"

    passing = _decide(policy=_policy(calibration=_calibration(epsilon=margin / 2)),
                      source=source, endpoint=donor)
    assert isinstance(passing, pol.PolicyDecision)

    refused = _decide(policy=_policy(calibration=_calibration(epsilon=margin * 2)),
                      source=source, endpoint=donor)
    assert isinstance(refused, pol.PolicyRejection)
    assert pol.StallReason.NO_BETTER_DONOR.value in refused.reason


def test_a_non_definitive_donor_cannot_supply_feedback():
    """PLAN §4.2: a provisional endpoint may never become ancestry."""
    source = _source(SOURCE_SEQ)
    donor = _donor(source=source, definitive=False)
    result = _decide(endpoint=donor, source=source)
    assert isinstance(result, pol.PolicyRejection)
    assert result.decision_evidence.donor_gate.reason is rw.DonorGateReason.DONOR_NOT_DEFINITIVE


def test_the_donor_cannot_be_the_incumbent():
    source = _source(SOURCE_SEQ)
    donor = _donor(INCUMBENT_SEQ, source=source)
    result = _decide(endpoint=donor, source=source)
    assert isinstance(result, pol.PolicyRejection)
    assert result.decision_evidence.donor_gate.reason is rw.DonorGateReason.DONOR_IS_INCUMBENT


# --------------------------------------------------------------------------------------------
# reference identities cannot alias silently
# --------------------------------------------------------------------------------------------


def test_an_accepted_endpoint_incumbent_may_not_carry_the_safety_references_bytes():
    """PLAN V2F5A: "safety reference, incumbent, and donor identities cannot alias silently".

    The depth-0 collapse ``I_0 = ybar`` is legal AND declared by ``kind``; an ACCEPTED_ENDPOINT
    incumbent that happens to be the reference is not, because "better than the incumbent" would
    silently mean "better than the reference" without the run ever saying so.
    """
    with pytest.raises(rw.IncumbentAliasError):
        _incumbent(kind=rw.LineageIncumbentKind.ACCEPTED_ENDPOINT)


def test_a_cumulative_reference_incumbent_must_actually_be_the_reference():
    with pytest.raises(rw.IncumbentAliasError):
        _incumbent(safety_md5=sequence_md5("A" * L))


def test_the_policy_refuses_a_safety_score_that_is_not_the_incumbents_own_reference():
    """The new-hotspot conjunct must be measured against the reference this lineage froze."""
    with pytest.raises(pol.V2PolicyError, match="safety reference"):
        _policy(safety_reference_score=_score("A" * L))


def test_the_incumbent_moves_only_on_a_passing_gate():
    """PLAN §3.2 of the working note: "a worse endpoint must not overwrite the lineage reference"."""
    source = _source(SOURCE_SEQ)
    incumbent = _incumbent()
    better = _donor(source=source)
    worse = _donor("Y" * L, source=source)

    passing = rw.donor_gate(donor=better, incumbent=incumbent, epsilon_r=0.0,
                            epsilon_source_ref=F.digest("eps"))
    advanced = rw.advance_incumbent(
        incumbent=incumbent, donor=better, verdict=passing, evaluator=_evaluator(),
        accepted_at_depth=1, law="strict_improvement_by_epsilon")
    assert advanced is not incumbent
    assert advanced.kind is rw.LineageIncumbentKind.ACCEPTED_ENDPOINT
    assert advanced.source_endpoint_id == better.endpoint_id

    refused = rw.donor_gate(donor=worse, incumbent=incumbent, epsilon_r=0.0,
                            epsilon_source_ref=F.digest("eps"))
    kept = rw.advance_incumbent(
        incumbent=incumbent, donor=worse, verdict=refused, evaluator=_evaluator(),
        accepted_at_depth=1, law="strict_improvement_by_epsilon")
    assert kept is incumbent, "a refused gate must return the SAME object, not a copy"


def test_an_advanced_policy_rejects_a_donor_that_only_beats_depth_zero():
    """The recursive ladder must compare depth ``d+1`` against ``I_{d+1}``, not forever ``I_0``.

    The second donor is genuinely better than the frozen depth-zero reference but worse than the
    endpoint already admitted into this lineage.  It would pass if the production policy object
    kept its construction-time incumbent, which is the exact regression this test guards.
    """
    source = _source(SOURCE_SEQ)
    policy = _policy()
    risk0 = float(policy.incumbent.head_global_risk)

    first = _donor(source=source)
    first = dataclasses.replace(
        first,
        head_score=dataclasses.replace(first.head_score, global_risk=risk0 - 1.0),
        head_global_risk=risk0 - 1.0,
    )
    first_gate = rw.donor_gate(
        donor=first, incumbent=policy.incumbent, epsilon_r=0.0,
        epsilon_source_ref=F.digest("eps"),
    )
    assert first_gate.passed

    advanced = policy.advance_lineage_incumbent(
        donor=first, verdict=first_gate, accepted_at_depth=1)
    assert advanced.incumbent.source_endpoint_id == first.endpoint_id
    assert advanced.identity().policy_config_digest != policy.identity().policy_config_digest

    second = _donor("ACDEFGHIKLMN", source=source, fork_index=1, fork_seed=99)
    second = dataclasses.replace(
        second,
        head_score=dataclasses.replace(second.head_score, global_risk=risk0 - 0.5),
        head_global_risk=risk0 - 0.5,
    )
    assert rw.donor_gate(
        donor=second, incumbent=policy.incumbent, epsilon_r=0.0,
        epsilon_source_ref=F.digest("eps"),
    ).passed, "the counterexample must beat I0"
    refused = rw.donor_gate(
        donor=second, incumbent=advanced.incumbent, epsilon_r=0.0,
        epsilon_source_ref=F.digest("eps"),
    )
    assert refused.passed is False
    assert refused.reason is rw.DonorGateReason.STALL_NO_BETTER_DONOR
    kept = advanced.advance_lineage_incumbent(
        donor=second, verdict=refused, accepted_at_depth=2)
    assert kept is advanced, "a stalled depth must not mint a new policy/incumbent identity"


def test_an_incumbent_cannot_be_advanced_on_another_endpoints_verdict():
    source = _source(SOURCE_SEQ)
    incumbent = _incumbent()
    donor = _donor(source=source)
    other = _donor("ACDEFGHIKLMN", source=source, fork_index=1, fork_seed=99)
    verdict = rw.donor_gate(donor=other, incumbent=incumbent, epsilon_r=0.0,
                            epsilon_source_ref=F.digest("eps"))
    with pytest.raises(rw.V2RewardError, match="verdict names donor"):
        rw.advance_incumbent(incumbent=incumbent, donor=donor, verdict=verdict,
                             evaluator=_evaluator(), accepted_at_depth=1,
                             law="strict_improvement_by_epsilon")


# --------------------------------------------------------------------------------------------
# the write channel
# --------------------------------------------------------------------------------------------


def test_every_write_is_legal_positive_and_in_the_deterministic_top_m():
    """PLAN V2F5A: "every chosen write is legal, changes the intended source identity/provenance,
    has positive ``a_i``, and belongs to the exact deterministic top ``m_d``"."""
    source = _source(SOURCE_SEQ)
    donor = _donor(source=source)
    decision = _decide(source=source, endpoint=donor)
    assert isinstance(decision, pol.PolicyDecision)
    record = decision.decision_evidence

    selected = [row for row in record.write_candidates if row.selected]
    assert selected, "the scenario must produce writes or it tests nothing"
    for row in selected:
        assert row.position in MASKED, "a write must land on a source-UNRESOLVED position"
        assert row.legal is True
        assert row.contribution is not None and row.contribution > 0.0
        assert row.donor_residue != row.incumbent_residue

    ranked = sorted(
        (row for row in record.write_candidates if row.contribution is not None
         and row.contribution > 0.0),
        key=lambda row: (-row.contribution, row.position),
    )
    assert [row.position for row in selected] == sorted(
        row.position for row in ranked[:record.realized_writes])


def test_ties_break_on_residue_index_only():
    """Two candidates with identical ``a_i`` must resolve by index, deterministically."""
    source = _source(SOURCE_SEQ)
    # 'A' at every masked position: all four contributions are exactly equal.
    sequence = list(SOURCE_SEQ)
    for position in MASKED:
        sequence[position] = "A"
    donor = _donor("".join(sequence), source=source)
    decision = _decide(source=source, endpoint=donor,
                       policy=_policy(calibration=_calibration(fraction=0.2)))
    assert isinstance(decision, pol.PolicyDecision)
    contributions = {row.position: row.contribution
                     for row in decision.decision_evidence.write_candidates
                     if row.contribution is not None}
    assert len({round(v, 9) for v in contributions.values()}) == 1, contributions
    assert decision.write_from_endpoint == tuple(sorted(MASKED)[:len(decision.write_from_endpoint)])


def test_a_position_identical_to_the_incumbent_is_never_a_candidate():
    source = _source(SOURCE_SEQ)
    sequence = list(SOURCE_SEQ)
    sequence[MASKED[0]] = INCUMBENT_SEQ[MASKED[0]]        # unchanged from the incumbent
    for position in MASKED[1:]:
        sequence[position] = "A"
    decision = _decide(source=source, endpoint=_donor("".join(sequence), source=source))
    row = next(r for r in decision.decision_evidence.write_candidates if r.position == MASKED[0])
    assert row.legal is False
    assert row.rejection_reason == "identical_to_incumbent"
    assert MASKED[0] not in decision.write_from_endpoint


def test_a_source_resolved_position_is_never_a_write_candidate():
    """PLAN A.3: a legal candidate is "unresolved in the live source"."""
    source = _source(SOURCE_SEQ)
    donor = _donor("A" * L, source=source)               # better EVERYWHERE, including resolved
    decision = _decide(source=source, endpoint=donor)
    assert isinstance(decision, pol.PolicyDecision)
    assert set(decision.write_from_endpoint) <= set(MASKED)
    resolved_rows = [row for row in decision.decision_evidence.write_candidates
                     if row.position not in MASKED and row.position != ANCHOR]
    assert resolved_rows and all(row.rejection_reason == "source_resolved"
                                 for row in resolved_rows)


def test_no_positive_local_write_stalls_rather_than_writing():
    """PLAN §2.5: "If ``m_d=0``, it emits ``stall_no_positive_local_write``; it may not fall back
    to the diagnostic position or silently widen to a block rule."."""
    source = _source(SOURCE_SEQ)
    donor = _donor(source=source)
    # A tolerance above every realized contribution leaves the donor globally better (so the gate
    # still passes) with no LOCAL evidence anywhere -- exactly the case the stall exists for.
    decision = _decide(source=source, endpoint=donor,
                       policy=_policy(calibration=_calibration(tolerance=10.0)))
    assert isinstance(decision, pol.PolicyRejection)
    assert pol.StallReason.NO_POSITIVE_LOCAL_WRITE.value in decision.reason
    record = decision.decision_evidence
    assert record.n_legal_write_candidates == len(MASKED)
    assert record.n_positive_contributions == 0
    assert record.realized_writes == 0
    # PLAN A.5: "A typed stall must preserve the rejected candidate counts and rejection reasons".
    assert all(row.rejection_reason == "non_positive_contribution"
               for row in record.write_candidates if row.legal)


def test_the_cap_is_a_cap_and_never_a_quota():
    """Both directions.  Under a binding cap the realized count is the cap; under a generous one it
    is the number of positive candidates, NOT the cap."""
    source = _source(SOURCE_SEQ)
    donor = _donor(source=source)

    # ceil(0.1 * 11) = 2 -> the cap binds.
    capped = _decide(source=source, endpoint=donor,
                     policy=_policy(calibration=_calibration(fraction=0.1)))
    assert len(capped.write_from_endpoint) == 2
    assert capped.decision_evidence.write_cap == 2

    # ceil(0.9 * 11) = 10 -> only the four positive candidates exist, so four is the answer.
    generous = _decide(source=source, endpoint=donor,
                       policy=_policy(calibration=_calibration(fraction=0.9)))
    assert generous.decision_evidence.write_cap == 10
    assert len(generous.write_from_endpoint) == len(MASKED)


def test_the_frozen_cap_reproduces_the_plans_own_numbers():
    """PLAN §2.5 freezes ``ceil(0.05 * N_editable)`` and states it is ~5 for 5ZHV_B and 14 for
    Q00511.  A cap computed over the sequence LENGTH instead of the editable domain would stop
    binding on a heavily anchored protein."""
    policy = _policy(calibration=_calibration(fraction=0.05))
    assert policy._write_cap(100) == 5
    assert policy._write_cap(280) == 14
    assert policy._write_cap(11) == 1


def test_the_realized_count_is_the_minimum_of_positive_cap_and_band():
    source = _source(SOURCE_SEQ)
    donor = _donor(source=source)
    decision = _decide(source=source, endpoint=donor)
    record = decision.decision_evidence
    assert record.realized_writes == min(
        record.n_positive_contributions, record.write_cap, record.m_band_max)


# --------------------------------------------------------------------------------------------
# the band centre and the exact reopen equation
# --------------------------------------------------------------------------------------------


def test_the_band_centre_is_an_explicit_integer_with_a_declared_tie_law():
    """PLAN §2.5: the calibration must materialize "one content-bound integer unresolved target
    ``u_target`` at the declared center of ``B(r_d)``, including its rounding/tie law"."""
    table = _band_table(unresolved_lo=4.0, unresolved_hi=7.0)      # even width: 4,5,6,7
    low = sch.band_center_target(table=table, step=R_STEP, stratum_key=F.STRATUM, n_editable=L - 1,
                                 rule=sch.BandCenterRule.MIDPOINT_TIE_LOW)
    high = sch.band_center_target(table=table, step=R_STEP, stratum_key=F.STRATUM, n_editable=L - 1,
                                  rule=sch.BandCenterRule.MIDPOINT_TIE_HIGH)
    assert (low.u_target, high.u_target) == (5, 6)
    assert low.pinned.min_unresolved == 4 and low.pinned.max_unresolved == 7
    # Content-bound: the target names the calibration it was read off.
    assert low.calibration_content_digest == table.provenance.calibration_content_digest


def test_the_band_centre_rule_has_no_default():
    with pytest.raises(sch.V2ScheduleError, match="BandCenterRule"):
        sch.band_center_target(table=_band_table(), step=R_STEP, stratum_key=F.STRATUM,
                               n_editable=L - 1, rule="midpoint_tie_low")


def test_the_reopen_equation_is_exact_and_returns_negatives_unclamped():
    """``m_reopen = u_target - u_src + m_d``.  A silent ``max(0, ...)`` would miss the declared
    target while still reporting its name."""
    assert sch.required_reopen_count(u_target=6, n_unresolved_source=4, n_writes=2) == 4
    assert sch.required_reopen_count(u_target=2, n_unresolved_source=9, n_writes=1) == -6


def test_the_realized_reopen_reproduces_the_band_target_exactly():
    """The projected state's unresolved mass must LAND on ``u_target``, not merely inside B(r)."""
    source = _source(SOURCE_SEQ)
    donor = _donor(source=source)
    decision = _decide(source=source, endpoint=donor)
    record = decision.decision_evidence
    assert record.required_reopen == record.u_target - record.n_unresolved_source \
        + record.realized_writes
    assert len(decision.reopen) == record.required_reopen

    projected_unresolved = (record.n_unresolved_source - record.realized_writes
                            + len(decision.reopen))
    assert projected_unresolved == record.u_target


def test_a_band_that_needs_more_writes_than_the_cap_allows_stalls_closed():
    """The cap and the band can DISAGREE, and when they do the transition fails closed.

    With ``u_target = 1`` and ``u_src = 4`` the equation permits ``m_reopen = m_d - 3 >= 0``, i.e. at
    least three writes -- but a cap of ``ceil(0.1 * 11) = 2`` allows two.  A policy that wrote two
    anyway would land on three unresolved positions while its artifact named a target of one.
    """
    source = _source(SOURCE_SEQ)
    donor = _donor(source=source)
    table = _band_table(unresolved_lo=1.0, unresolved_hi=1.0)
    result = _decide(source=source, endpoint=donor,
                     policy=_policy(band_table=table,
                                    calibration=_calibration(fraction=0.1)))
    assert isinstance(result, pol.PolicyRejection)
    assert pol.StallReason.BAND_INFEASIBLE.value in result.reason
    assert result.decision_evidence.m_band_min == 3
    assert result.decision_evidence.write_cap == 2


# --------------------------------------------------------------------------------------------
# the reopen channel
# --------------------------------------------------------------------------------------------


def test_reopen_excludes_anchors_and_names_only_source_resolved_positions():
    source = _source(SOURCE_SEQ)
    decision = _decide(source=source)
    assert ANCHOR not in decision.reopen
    assert set(decision.reopen) & set(MASKED) == set(), "a reopen must NEWLY mask something"
    assert set(decision.reopen) <= {p for p in range(L) if p != ANCHOR and p not in MASKED}


def test_reopen_priority_puts_new_hotspot_before_worsening_before_residual():
    """PLAN §2.5's frozen order.  Built directly on the sort key so the test states the LAW rather
    than a scenario that happens to realize it."""
    def row(**over):
        kw = dict(position=5, new_hotspot=None, worsening=None, residual_burden=None,
                  active_sampler_score=None, n_origin_events=1, commit_step=1,
                  selection_rank=None, selected=False, priority_reason="index")
        kw.update(over)
        return pol.ReopenCandidateEvidence(**kw)

    hotspot = row(position=9, new_hotspot=0.5)
    worsened = row(position=2, worsening=0.9)
    residual = row(position=1, residual_burden=99.0)
    uncertain = row(position=3, active_sampler_score=-9.0)
    order = sorted([residual, uncertain, worsened, hotspot], key=pol._reopen_sort_key)
    assert [r.position for r in order] == [9, 2, 1, 3]


def test_absent_evidence_ranks_below_present_evidence_and_is_recorded_as_none():
    """PLAN §2.5: "Missing or inapplicable evidence ranks below present evidence and is recorded
    explicitly rather than replaced by a numeric sentinel"."""
    def row(position, **over):
        kw = dict(position=position, new_hotspot=None, worsening=None, residual_burden=None,
                  active_sampler_score=None, n_origin_events=1, commit_step=1,
                  selection_rank=None, selected=False, priority_reason="index")
        kw.update(over)
        return pol.ReopenCandidateEvidence(**kw)

    present_zero = row(2, new_hotspot=0.0, worsening=0.0, residual_burden=0.0)
    absent = row(1)
    assert sorted([absent, present_zero], key=pol._reopen_sort_key)[0].position == 2


def test_a_previously_written_non_anchor_can_be_reopened_once_its_protection_expired():
    """PLAN §2.5: "Previously endpoint-written non-anchor positions become eligible again after
    temporary protection expires".  Soft protection is reversible; only hard anchors are not."""
    source = _source(SOURCE_SEQ)
    written = 6                                       # resolved, non-anchor, not a source mask
    expired = st.TemporaryProtection(
        position=written, expiry_step=C_SOURCE, granted_at_depth=0, granted_at_step=R_STEP,
        granted_by_transition_id=F.TXN,
        grant_reason=st.ProtectionGrantReason.ENDPOINT_INJECTION,
    )
    source = _source(SOURCE_SEQ, expired_protection=(expired,))
    decision = _decide(source=source)
    assert isinstance(decision, pol.PolicyDecision)
    candidates = {row.position for row in decision.decision_evidence.reopen_candidates}
    assert written in candidates, (
        "an EXPIRED endpoint write must be reopen-eligible again; a permanently protected "
        "non-anchor would be an anchor by another name"
    )


# --------------------------------------------------------------------------------------------
# the partition the kernel actually accepts
# --------------------------------------------------------------------------------------------


def _project(decision, *, source, endpoint, band_table):
    from inverse_folding.reference_flow.fusion_v2 import projection as proj

    return proj.source_writeback(
        source=source, endpoint=endpoint, endpoint_tokens=_tokens(endpoint.sequence),
        alphabet=F.ALPHABET, decision=decision,
        declared_policy=pol.DeclaredPolicy(
            policy_id=pol.HEAD_DIRECTED_CAPPED_POLICY_ID, policy_version="v1",
            is_diagnostic=False, phase="policy_qualification"),
        coordinates=_coords(), band_table=band_table, stratum_key=F.STRATUM,
        descendant_fork_seed=9001, origin_transition_id=F.TXN, declared_band_id=F.BAND_ID,
        declared_band_digest=band_table.provenance.calibration_content_digest,
    )


def test_the_policy_output_passes_the_exact_temporal_partition_and_band_solver():
    """PLAN V2F5A: "policy output still passes the exact temporal partition and B(r) solver"."""
    table = _band_table()
    source = _source(SOURCE_SEQ, conditioning=F.conditioning(
        schedule_band_calibration=table.provenance.calibration_content_digest,
        projection_policy_spec=F.digest("pspec")))
    donor = _donor(source=source)
    decision = _decide(source=source, endpoint=donor, policy=_policy(band_table=table))
    outcome = _project(decision, source=source, endpoint=donor, band_table=table)
    assert outcome.committed, outcome.detail

    projected = outcome.projected
    unresolved = sum(1 for p in projected.editable_positions
                     if projected.tokens[p] == projected.mask_token_id)
    assert unresolved == decision.decision_evidence.u_target
    # The temporal law, as an invariant rather than as one position: nothing carried as ordinary
    # history may have been committed at or after the re-entry boundary.
    for position in projected.support.carry_from_source:
        commit = source.active_commit_depth_step_by_pos[position]
        assert commit is None or commit.step < R_STEP


def test_a_future_committed_source_identity_is_injected_rather_than_carried():
    """Same law, in the case where it BITES: a tight band leaves the late-committed position
    unreopened, and it must then be explicitly injected rather than carried as ordinary history."""
    table = _band_table(unresolved_lo=4.0, unresolved_hi=4.0)      # u_target == u_src == 4
    source = _source(SOURCE_SEQ, conditioning=F.conditioning(
        schedule_band_calibration=table.provenance.calibration_content_digest,
        projection_policy_spec=F.digest("pspec")))
    donor = _donor(source=source)
    decision = _decide(source=source, endpoint=donor,
                       policy=_policy(band_table=table,
                                      calibration=_calibration(fraction=0.05)))
    assert isinstance(decision, pol.PolicyDecision)
    assert len(decision.write_from_endpoint) == 1 and len(decision.reopen) == 1
    assert LATE not in decision.reopen
    assert LATE in decision.inject_from_source_feedback

    outcome = _project(decision, source=source, endpoint=donor, band_table=table)
    assert outcome.committed, outcome.detail
    assert LATE in outcome.projected.support.inject_from_source_feedback


def test_the_written_bytes_are_the_donors_bytes():
    table = _band_table()
    source = _source(SOURCE_SEQ, conditioning=F.conditioning(
        schedule_band_calibration=table.provenance.calibration_content_digest,
        projection_policy_spec=F.digest("pspec")))
    donor = _donor(source=source)
    decision = _decide(source=source, endpoint=donor, policy=_policy(band_table=table))
    projected = _project(decision, source=source, endpoint=donor, band_table=table).projected
    for position in decision.write_from_endpoint:
        assert projected.tokens[position] == _tokens(donor.sequence)[position]
        assert projected.feedback_origin_ref_by_pos[position] is st.FeedbackOriginRef.SELECTED_ENDPOINT


# --------------------------------------------------------------------------------------------
# the matched source-geometry control
# --------------------------------------------------------------------------------------------


def _control(*, writes, reopens, incumbent=None):
    return pol.SourceGeometryControlPolicy(
        band_table=_band_table(), stratum_key=F.STRATUM, incumbent=incumbent or _incumbent(),
        evaluator=_evaluator(), calibration=_calibration(),
        required_writes=writes, required_reopens=reopens,
        policy_spec_digest=F.digest("pspec"),
    )


def test_the_control_matches_the_treatment_cardinalities_and_is_head_blind():
    source = _source(SOURCE_SEQ)
    donor = _donor(source=source)
    treatment = _decide(source=source, endpoint=donor)
    control = _control(writes=len(treatment.write_from_endpoint),
                       reopens=len(treatment.reopen)).decide(
        source=source, endpoint=donor, coordinates=_coords())

    assert len(control.write_from_endpoint) == len(treatment.write_from_endpoint)
    assert len(control.reopen) == len(treatment.reopen)
    assert control.decision_evidence.head_evidence_consulted is False
    assert all(row.contribution is None for row in control.decision_evidence.write_candidates)
    # The whole point: same dose, different identities.
    assert control.write_from_endpoint == tuple(sorted(MASKED)[:len(control.write_from_endpoint)])


def test_the_control_gates_the_donor_on_the_same_incumbent():
    source = _source(SOURCE_SEQ)
    worse = _donor("Y" * L, source=source)
    result = _control(writes=1, reopens=1).decide(
        source=source, endpoint=worse, coordinates=_coords())
    assert isinstance(result, pol.PolicyRejection)
    assert pol.StallReason.NO_BETTER_DONOR.value in result.reason


def test_the_control_refuses_rather_than_improvising_when_it_cannot_match():
    source = _source(SOURCE_SEQ)
    donor = _donor(source=source)
    result = _control(writes=len(MASKED) + 1, reopens=1).decide(
        source=source, endpoint=donor, coordinates=_coords())
    assert isinstance(result, pol.PolicyRejection)
    assert pol.StallReason.CONTROL_CARDINALITY_UNMATCHABLE.value in result.reason


def test_the_two_policies_have_distinct_identities_and_neither_is_diagnostic():
    treatment = _policy().identity()
    control = _control(writes=1, reopens=1).identity()
    assert treatment.policy_id != control.policy_id
    assert treatment.policy_config_digest != control.policy_config_digest
    assert treatment.is_diagnostic_only is False and control.is_diagnostic_only is False
    # They share the frozen SPEC: one rule file describes the treatment law and its matched control.
    assert treatment.policy_spec_digest == control.policy_spec_digest


# --------------------------------------------------------------------------------------------
# the cross-sequence evidence contract
# --------------------------------------------------------------------------------------------


def test_an_independently_centered_residue_summary_cannot_satisfy_the_evidence_contract():
    """PLAN V2F5A: "independently centered/clipped residue summaries cannot satisfy the
    cross-sequence evidence contract".

    Enforced STRUCTURALLY rather than by convention: the evidence module never reads
    ``residue_hotspot`` at all, so there is no code path by which a per-sequence summary could
    become a cross-sequence delta.
    """
    tree = ast.parse(pathlib.Path(ev.__file__).read_text())
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    constants = {node.value for node in ast.walk(tree)
                 if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "residue_hotspot" not in attributes
    assert not any("residue_hotspot" == value for value in constants)


def test_the_policys_whole_import_closure_cannot_reach_the_run_config():
    """PLAN §2.5, strengthened.  The original rule checked only ``policy.py``'s own imports, which
    a new dependency could satisfy while still handing the policy a path to the run config.  V2F5A
    adds two modules to that closure, so the check is now transitive over the package."""
    package = pathlib.Path(pol.__file__).parent
    seen, queue = set(), ["policy"]
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        path = package / f"{module}.py"
        if not path.exists():
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module and node.level:
                names = [node.module.split(".")[0]]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]] if node.level else []
            for name in names:
                assert name != "config", (
                    f"fusion_v2.{module} imports fusion_v2.config; a policy that can reach the run "
                    "config can read an unfrozen threshold out of it (PLAN §2.5)"
                )
                queue.append(name)


def test_the_window_evidence_refuses_two_grids_that_do_not_align():
    donor = _score("A" * L)
    shorter = _score("A" * (L - 1))
    with pytest.raises(ev.EvidenceGridMismatch):
        ev.build_window_evidence(donor_score=donor, reference_score=shorter,
                                 evaluator=_evaluator(), reference_label="lineage_incumbent")


def test_the_window_evidence_agrees_with_the_safety_comparator_on_one_fixture():
    """The alignment law is implemented twice, for two different contracts.  This pins them to the
    same verdict so the duplication cannot drift into two different landscapes."""
    from inverse_folding.reference_flow.fusion_v2 import safety as sf

    donor, reference = _score("A" * L), _score("K" * L)
    ours = ev.build_window_evidence(donor_score=donor, reference_score=reference,
                                    evaluator=_evaluator(), reference_label="ybar")
    theirs = sf.whole_landscape_new_hotspot(
        donor, reference, endpoint_id="endpoint:" + "0" * 16, head_identity=_evaluator(),
        reference_kind=sf.ReferenceKind.CUMULATIVE_DEPTH0, reference_binding_id="ref:wt")
    assert theirs.n_windows == len(ours.windows)
    assert pytest.approx(theirs.max_increase) == max(
        max(0.0, window.delta) for window in ours.windows)


def test_a_counterfactual_result_batch_that_loses_a_row_is_refused():
    """Matching is BY DIGEST; a dropped or reordered row must not silently drop a candidate."""
    def lossy(protein_id, sequences):
        return _scorer()(protein_id, list(sequences)[:-1])

    with pytest.raises(ev.EvidenceIdentityMismatch, match="no counterfactual Head result"):
        ev.score_leave_one_out(
            scorer=lossy, protein_id="5ZHV_B", donor_sequence="A" * L,
            donor_global_risk=_global_risk("A" * L), incumbent_sequence="K" * L,
            positions=(1, 2, 3), evaluator=_evaluator(),
            window_grid_digest=ident.window_grid_digest(_windows("A" * L)),
        )


def test_a_counterfactual_changes_exactly_one_position():
    donor, incumbent = "A" * L, "K" * L
    out = ev.counterfactual_sequence(donor, incumbent, 5)
    assert out[5] == "K"
    assert sum(1 for a, b in zip(out, donor) if a != b) == 1


def test_the_contribution_is_the_counterfactual_minus_the_donor():
    donor = list(SOURCE_SEQ)
    donor[3] = "A"
    donor_sequence = "".join(donor)
    out = ev.score_leave_one_out(
        scorer=_scorer(), protein_id="5ZHV_B", donor_sequence=donor_sequence,
        donor_global_risk=_global_risk(donor_sequence), incumbent_sequence=INCUMBENT_SEQ,
        positions=(3,), evaluator=_evaluator(),
        window_grid_digest=ident.window_grid_digest(_windows(donor_sequence)),
    )
    expected = RESIDUE_RISK["K"] - RESIDUE_RISK["A"]
    assert pytest.approx(out[3].contribution, abs=1e-9) == expected


# --------------------------------------------------------------------------------------------
# config: no library defaults, and the block cannot be dormant
# --------------------------------------------------------------------------------------------


def _policy_scalar(value, *, unit, kind, source_id):
    artifact = cfg.PolicyCalibrationArtifact(
        schema_version=cfg.V2_POLICY_CALIBRATION_SCHEMA_VERSION, measurement_kind=kind,
        allele="DRB1_0701", score_scale="nats", n_observations=32,
        calibration_data_digest=F.D,
    )
    return {
        "value": value, "unit": unit, "source_kind": "measured_calibration",
        "source_id": source_id,
        "source_ref": cfg.policy_calibration_source_ref(
            value=value, unit=unit, source_kind="measured_calibration", source_id=source_id,
            artifact=artifact),
        "artifact": artifact.canonical_payload(),
    }


def _head_directed_payload(**over):
    payload = {
        "write_cap_editable_fraction": _policy_scalar(
            0.05, unit="editable_fraction", kind="frozen_declared_fraction",
            source_id="tmp-fusion-v2-note-3.4"),
        "donor_improvement_epsilon": _policy_scalar(
            0.01, unit="nats", kind="frozen_head_repeatability", source_id="head-repeat-1"),
        "local_contribution_tolerance": _policy_scalar(
            0.001, unit="nats", kind="frozen_head_numeric_floor", source_id="head-floor-1"),
        "band_center_rule": "midpoint_tie_low",
        "lineage_incumbent_depth0_rule": "cumulative_safety_reference",
        "lineage_incumbent_update_law": "strict_improvement_by_epsilon",
        "write_candidate_window_rule": pol.WRITE_WINDOW_RULE,
        "reopen_count_law": pol.REOPEN_COUNT_LAW,
        "reopen_priority_law": pol.REOPEN_PRIORITY_LAW,
        "control_policy_id": pol.SOURCE_GEOMETRY_CONTROL_POLICY_ID,
        "control_policy_version": "v1",
        "max_counterfactual_head_calls_per_cycle": 64,
    }
    payload.update(over)
    return payload


def _config_payload(**over):
    from tests.inverse_folding.test_fusion_v2_config import _mapping as base_payload

    payload = base_payload()
    payload["identity"]["phase"] = "policy_qualification"
    payload["projection"]["support_policy_id"] = pol.HEAD_DIRECTED_CAPPED_POLICY_ID
    payload["projection"]["support_policy_is_diagnostic"] = False
    payload["projection"]["support_policy_version"] = "v1"
    payload["projection"]["head_directed"] = _head_directed_payload()
    for key, value in over.items():
        node = payload
        *path, leaf = key.split(".")
        for part in path:
            node = node[part]
        node[leaf] = value
    return payload


def test_the_head_directed_block_is_required_for_its_policy_and_forbidden_otherwise():
    payload = _config_payload()
    config = cfg.load_v2_config(payload)
    assert config.projection.head_directed is not None

    del payload["projection"]["head_directed"]
    with pytest.raises(cfg.V2ConfigError, match="head_directed is required"):
        cfg.load_v2_config(payload)

    dormant = _config_payload()
    dormant["identity"]["phase"] = "state_transition_canary"
    dormant["projection"]["support_policy_id"] = "state_derived_probe"
    dormant["projection"]["support_policy_is_diagnostic"] = True
    with pytest.raises(cfg.V2ConfigError, match="head_directed is forbidden"):
        cfg.load_v2_config(dormant)


def test_a_head_scale_threshold_must_be_measured_on_this_runs_head():
    payload = _config_payload()
    payload["projection"]["head_directed"]["donor_improvement_epsilon"] = _policy_scalar(
        0.01, unit="raw_logit", kind="frozen_head_repeatability", source_id="head-repeat-1")
    with pytest.raises(cfg.V2ConfigError, match="does not match Head"):
        cfg.load_v2_config(payload)


def test_the_cap_may_not_claim_to_be_a_head_measurement():
    payload = _config_payload()
    payload["projection"]["head_directed"]["write_cap_editable_fraction"] = _policy_scalar(
        0.05, unit="editable_fraction", kind="frozen_head_repeatability", source_id="x")
    with pytest.raises(cfg.V2ConfigError, match="frozen_declared_fraction"):
        cfg.load_v2_config(payload)


def test_the_production_config_freezes_the_head_directed_cap_at_five_percent():
    calibration = cfg.load_v2_config(_config_payload()).head_directed_calibration()
    assert calibration.write_cap_editable_fraction == 0.05

    payload = _config_payload()
    payload["projection"]["head_directed"]["write_cap_editable_fraction"] = _policy_scalar(
        0.25, unit="editable_fraction", kind="frozen_declared_fraction",
        source_id="tmp-fusion-v2-note-3.4")
    with pytest.raises(cfg.V2ConfigError, match="must equal the frozen V2F5A value 0.05"):
        cfg.load_v2_config(payload)


def test_an_edited_calibration_value_breaks_its_own_source_ref():
    payload = _config_payload()
    payload["projection"]["head_directed"]["donor_improvement_epsilon"]["value"] = 0.5
    with pytest.raises(cfg.V2ConfigError, match="source_ref"):
        cfg.load_v2_config(payload)


def test_the_config_resolves_the_pure_calibration_the_policy_reads():
    config = cfg.load_v2_config(_config_payload())
    calibration = config.head_directed_calibration()
    assert isinstance(calibration, pol.HeadDirectedCalibration)
    assert calibration.write_cap_editable_fraction == 0.05
    assert calibration.band_center_rule is sch.BandCenterRule.MIDPOINT_TIE_LOW
    control = config.declared_control_policy()
    assert control.policy_id == pol.SOURCE_GEOMETRY_CONTROL_POLICY_ID
    assert control.is_diagnostic is False


def test_the_depth0_incumbent_rule_has_no_default_and_is_a_closed_vocabulary():
    payload = _config_payload()
    payload["projection"]["head_directed"]["lineage_incumbent_depth0_rule"] = "best_endpoint"
    with pytest.raises(cfg.V2ConfigError):
        cfg.load_v2_config(payload)


# --------------------------------------------------------------------------------------------
# end to end: one cycle, and the matched one-cycle qualification contrast
# --------------------------------------------------------------------------------------------
#
# PLAN V2F5A's scientific boundary applies to everything below: a green run proves the two support
# laws EXECUTE under matched source, donor, cardinalities, band and seeds.  Whether the descendant
# Head distribution shifts is a cohort statistic frozen in the runbook, not something a fake-oracle
# suite can answer.


def _live_safety_gate():
    """The shared fixture's gate, re-bound so the REFERENCE is scored by this suite's Head.

    ``_v2_fixtures.safety_gate`` gives every window a flat ``z``; under that reference no window
    ever improves in a donor, so the write screen would reject everything for a reason that belongs
    to the fixture rather than to the policy.  The reference SEQUENCE is unchanged, so the frozen
    content digest the config declares still signs the same bytes.
    """
    import dataclasses as _dc

    from inverse_folding.reference_flow.fusion_v2 import safety as sf
    from inverse_folding.reference_flow.fusion_v2_runtime.admission import SafetyGate

    sequence = F.reference_sequence(L)
    content_digest = sf.reference_sequence_content_digest(sequence)
    base = F.v2_config(cumulative_value=0.10, length=L)
    config = _dc.replace(base, content=tuple(
        _dc.replace(row, expected_sha256=content_digest)
        if row.role == "complete_reference_sequence" else row
        for row in base.content
    ))
    policy = sf.bind_admission_policy(config, _evaluator())
    reference = sf.bind_cumulative_reference(
        lineage_id="5ZHV_B:fam0", protein_id="5ZHV_B", reference_label="wt_native",
        reference_sequence=sequence, reference_content_digest=content_digest, policy=policy,
        head_score=_score(sequence),
    )
    return SafetyGate(policy=policy, ledger=sf.open_lineage_ledger(reference)), reference, sequence


def _cycle_band_table(step=R_STEP):
    """Permissive on both axes, so the CYCLE's realized maturity decides nothing here.

    Band calibration has its own suite; a narrow band would make this test fail for a reason that
    has nothing to do with the support law.  The centre still comes from the band, so the reopen
    equation is exercised exactly as it would be in production.
    """
    band = sch.make_band(
        step=step, stratum_key=F.STRATUM, levels=sch.QuantileLevels(levels=(0.1, 0.5, 0.9)),
        rho_quantiles=(0.05, 0.5, 0.95), unresolved_quantiles=(L - 1, 4, 1),
        rho_accept=sch.BandInterval(lo=0.0, hi=1.0, lo_level=0.1, hi_level=0.9),
        unresolved_accept=sch.BandInterval(lo=3.0, hi=7.0, lo_level=0.9, hi_level=0.1),
        combination_rule="both_axes", n_attempts=32, n_captured=32,
        n_editable_min=1, n_editable_max=L,
    )
    return sch.make_band_table(provenance=F.band_provenance(), bands=(band,))


def _cycle_kwargs(**over):
    import numpy as np

    from inverse_folding.reference_flow.sampler import PositionDependentDFMSampler
    from tests.inverse_folding.test_fusion_v2_cycle import (
        VOCAB,
        _cfg,
        _denoiser,
        _module_meter,
        _structure_oracle,
    )

    gate, reference, reference_sequence = _live_safety_gate()
    table = _cycle_band_table()
    head = _ScriptedHead()
    incumbent = rw.bind_incumbent_from_safety_reference(
        reference=reference, reference_sequence=reference_sequence, lineage_id="5ZHV_B:fam0",
        evaluator=_evaluator(), rule="cumulative_safety_reference",
    )
    policy = _policy(
        incumbent=incumbent, band_table=table, head=head,
        safety_reference_score=_score(reference_sequence),
        calibration=_calibration(fraction=0.3),
    )
    kw = dict(
        sampler=PositionDependentDFMSampler(mask_token_id=F.MASK, vocab_size=VOCAB),
        denoiser=_denoiser, config=_cfg(), sequence_length=L,
        h_values=np.zeros(L, dtype=np.float32), residue_token_ids=F.AA,
        alphabet=F.ALPHABET, fixed_tokens={ANCHOR: min(F.AA)},
        lineage=F.lineage(), mask_token_id=F.MASK, aa_token_ids=F.AA,
        conditioning=F.conditioning(
            schedule_band_calibration=table.provenance.calibration_content_digest,
            projection_policy_spec=F.digest("pspec")),
        # The STATE's reference binding comes from the shared fixture (whose conditioning
        # declares the same placeholder digest); the GATE carries the real content-bound reference
        # the admission ratchet measures against.  Same split the cycle suite already uses.
        safety_reference=F.safety_reference(length=L),
        c_source_step=C_SOURCE, r_step=R_STEP, c_next_step=C_NEXT,
        source_fork_seeds=(5001, 5002, 5003), descendant_fork_seeds=(6001, 6002),
        descendant_propagation_seed=9001,
        head_oracle=head, structure_oracle=_structure_oracle, support_policy=policy,
        band_table=table, stratum_key=F.STRATUM, declared_band_id=F.BAND_ID,
        declared_band_digest=table.provenance.calibration_content_digest,
        origin_transition_id=F.TXN, safety_gate=gate,
        declared_policy=pol.DeclaredPolicy(
            policy_id=pol.HEAD_DIRECTED_CAPPED_POLICY_ID, policy_version="v1",
            is_diagnostic=False, phase="policy_qualification"),
        feedback_enabled=True, cost_meter=_module_meter(),
    )
    kw.update(over)
    return kw


def test_one_cycle_runs_under_the_head_directed_policy_and_records_its_evidence():
    from inverse_folding.reference_flow.fusion_v2_runtime.cycle import run_one_cycle

    outcome = run_one_cycle(**_cycle_kwargs())
    assert outcome.committed, outcome.detail
    record = outcome.policy_evidence
    assert record is not None, "the cycle must carry the policy's own decision record (PLAN §5.3)"
    assert record.policy_id == pol.HEAD_DIRECTED_CAPPED_POLICY_ID
    assert record.donor_gate.passed is True
    assert record.head_evidence_consulted is True

    selected = [row for row in record.write_candidates if row.selected]
    assert selected and all(row.contribution > 0.0 for row in selected)
    assert len(selected) == len(outcome.projected.support.write_from_endpoint)
    # The realized state landed exactly on the calibrated integer centre.
    projected = outcome.projected
    unresolved = sum(1 for p in projected.editable_positions
                     if projected.tokens[p] == projected.mask_token_id)
    assert unresolved == record.u_target


def test_the_counterfactual_head_batch_is_journaled_before_it_runs():
    """PLAN §5.4: the policy's leave-one-out batch is real Head work and must be on the ledger.

    Off-ledger it would spend against ``max_head_calls`` without the cap accounting ever seeing it.
    """
    import json
    import pathlib
    import tempfile

    from inverse_folding.reference_flow.fusion_v2_runtime.cycle import run_one_cycle
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import AttemptJournal, CostMeter

    path = pathlib.Path(tempfile.mkdtemp()) / "attempts.jsonl"
    meter = CostMeter(journal=AttemptJournal(path), protein_id="5ZHV_B", arm="v2",
                      gpu_clock=lambda: 0.0)
    outcome = run_one_cycle(**_cycle_kwargs(cost_meter=meter))
    assert outcome.committed, outcome.detail
    events = [json.loads(line) for line in path.read_text().splitlines()]
    counterfactual = [row for row in events if "counterfactual" in str(row.get("event_id", ""))]
    assert counterfactual, "the policy's Head batch never reached the journal"
    assert any(row.get("head_calls") for row in counterfactual)


def test_the_matched_qualification_contrast_runs_both_laws_at_one_dose():
    """PLAN §8.4: same donor, same realized cardinalities, same band, same seeds -- different
    support IDENTITY, which is the treatment."""
    from inverse_folding.reference_flow.fusion_v2.seeds import FeedbackPairSeedContext
    from inverse_folding.reference_flow.fusion_v2_runtime.paired import (
        run_policy_qualification_view,
    )

    kwargs = _cycle_kwargs()
    treatment_policy = kwargs.pop("support_policy")
    for name in ("c_source_step", "r_step", "c_next_step", "descendant_fork_seeds",
                 "descendant_propagation_seed"):
        kwargs.pop(name)

    context = FeedbackPairSeedContext(
        seed_schema="v2seed-1", campaign_id="v2f5a", split_role="dev", master_seed=20260807,
        protein_id="5ZHV_B", depth=0, r_step=R_STEP, c_next_step=C_NEXT, pair_ordinal=0,
        n_forks=1, n_descendant_lookaheads=2,
    )

    def control_factory(*, required_writes, required_reopens):
        return pol.SourceGeometryControlPolicy(
            band_table=treatment_policy.band_table, stratum_key=treatment_policy.stratum_key,
            incumbent=treatment_policy.incumbent, evaluator=treatment_policy.evaluator,
            calibration=treatment_policy.calibration, required_writes=required_writes,
            required_reopens=required_reopens,
            policy_spec_digest=treatment_policy.policy_spec_digest,
            policy_version=treatment_policy.policy_version,
        )

    result = run_policy_qualification_view(
        context=context, fork_index=0, c_source_step=C_SOURCE, r_step=R_STEP, c_next_step=C_NEXT,
        control_policy_factory=control_factory,
        control_declared_policy=pol.DeclaredPolicy(
            policy_id=pol.SOURCE_GEOMETRY_CONTROL_POLICY_ID, policy_version="v1",
            is_diagnostic=False, phase="policy_qualification"),
        support_policy=treatment_policy, **kwargs,
    )
    assert not result.parity_violation, result.parity_violation
    assert result.contrastable

    treatment, control = result.treatment.cycle, result.control.cycle
    # One donor, one source, one realized pool.
    assert treatment.selected_endpoint.endpoint_id == control.selected_endpoint.endpoint_id
    assert treatment.source.state_id == control.source.state_id
    # One dose.
    assert len(treatment.projected.support.write_from_endpoint) == result.realized_writes
    assert len(control.projected.support.write_from_endpoint) == result.realized_writes
    assert len(control.projected.support.reopen) == result.realized_reopens
    # Two identities -- and the artifact can PROVE which arm was Head-directed.
    assert result.treatment_writes_are_head_directed is True
    assert result.control_is_head_blind is True
    assert (treatment.policy_identity.policy_id != control.policy_identity.policy_id)
    # Matched randomness: the descendant seeds come from the pair context alone.
    assert (result.treatment.realized_propagation_seed
            == result.control.realized_propagation_seed)

    # Shared PRE-feedback evidence does not mean shared mutable post-feedback history.  Each arm
    # must admit descendants into its own fork or the treatment changes what the control sees.
    assert treatment.archive is not control.archive
    treatment_descendants = {endpoint.endpoint_id for endpoint in treatment.descendant_endpoints}
    control_descendants = {endpoint.endpoint_id for endpoint in control.descendant_endpoints}
    treatment_archive = {row.endpoint_id for row in treatment.archive.raw_rows()}
    control_archive = {row.endpoint_id for row in control.archive.raw_rows()}
    assert treatment_descendants and control_descendants
    assert treatment_descendants <= treatment_archive
    assert control_descendants <= control_archive
    assert treatment_descendants.isdisjoint(control_archive)
    assert control_descendants.isdisjoint(treatment_archive)


def test_the_qualification_contrast_cannot_be_run_through_the_mechanism_runner():
    """Its B arm re-uses one policy object; running the support-law view there would produce two
    arms under the SAME law and report a null that means nothing."""
    from inverse_folding.reference_flow.fusion_v2_runtime import paired

    with pytest.raises(paired.V2PairedError, match="run_policy_qualification_view"):
        paired.run_mechanism_views(
            context=None, fork_index=0, c_source_step=C_SOURCE, r_step=R_STEP,
            c_next_step=C_NEXT, interventions=(paired.SUPPORT_LAW_VIEW,))


# --------------------------------------------------------------------------------------------
# the offline coverage replay
# --------------------------------------------------------------------------------------------


def _replay_config():
    """A ``V2Config`` whose Head domain is THIS suite's, carrying the Head-directed block.

    The loader suite's payload declares the production Head's ``k`` domain (13-25); the toy Head
    here emits width-4 windows, and a replay run under the wrong domain rejects every window as
    off-grid -- correctly, but for a reason that has nothing to do with the support law.  Built
    from the shared fixture config so the two agree on allele, scale and grid by construction.
    """
    import dataclasses as _dc

    base = F.v2_config(length=L)
    scalar = lambda value, unit, kind, source_id: cfg.PolicyCalibratedScalar(  # noqa: E731
        value=value, unit=unit, source_kind="measured_calibration", source_id=source_id,
        source_ref=cfg.policy_calibration_source_ref(
            value=value, unit=unit, source_kind="measured_calibration", source_id=source_id,
            artifact=cfg.PolicyCalibrationArtifact(
                schema_version=cfg.V2_POLICY_CALIBRATION_SCHEMA_VERSION, measurement_kind=kind,
                allele=base.head.allele, score_scale=base.head.score_scale, n_observations=32,
                calibration_data_digest=F.D)),
        artifact=cfg.PolicyCalibrationArtifact(
            schema_version=cfg.V2_POLICY_CALIBRATION_SCHEMA_VERSION, measurement_kind=kind,
            allele=base.head.allele, score_scale=base.head.score_scale, n_observations=32,
            calibration_data_digest=F.D),
    )
    block = cfg.V2HeadDirectedConfig(
        write_cap_editable_fraction=scalar(
            0.3, "editable_fraction", "frozen_declared_fraction", "tmp-note-3.4"),
        donor_improvement_epsilon=scalar(
            0.0, base.head.score_scale, "frozen_head_repeatability", "head-repeat-1"),
        local_contribution_tolerance=scalar(
            0.0, base.head.score_scale, "frozen_head_numeric_floor", "head-floor-1"),
        band_center_rule=sch.BandCenterRule.MIDPOINT_TIE_LOW.value,
        lineage_incumbent_depth0_rule="cumulative_safety_reference",
        lineage_incumbent_update_law="strict_improvement_by_epsilon",
        write_candidate_window_rule=pol.WRITE_WINDOW_RULE,
        reopen_count_law=pol.REOPEN_COUNT_LAW,
        reopen_priority_law=pol.REOPEN_PRIORITY_LAW,
        control_policy_id=pol.SOURCE_GEOMETRY_CONTROL_POLICY_ID, control_policy_version="v1",
        max_counterfactual_head_calls_per_cycle=64,
    )
    return _dc.replace(
        base,
        identity=_dc.replace(base.identity, phase="policy_qualification"),
        projection=_dc.replace(
            base.projection, support_policy_id=pol.HEAD_DIRECTED_CAPPED_POLICY_ID,
            support_policy_is_diagnostic=False, support_policy_version="v1",
            head_directed=block),
    )


def _write_bundle(directory, outcome, *, config):
    """Persist one cycle's rows as the three tables the replay reads."""
    import pandas as pd

    from scripts.rf_fusion_v2_artifacts import (
        complete_endpoint_rows,
        feedback_event_rows,
        partial_state_rows,
    )

    states = [outcome.source] + ([outcome.projected] if outcome.projected is not None else [])
    if outcome.propagated is not None:
        states.append(outcome.propagated)
    tables = {
        "partial_states": partial_state_rows(states),
        "complete_endpoints": complete_endpoint_rows(outcome.endpoints, depth=0),
        "feedback_events": feedback_event_rows([dict(
            source=outcome.source, endpoint=outcome.selected_endpoint,
            projected=outcome.projected, propagated=outcome.propagated,
            policy=outcome.policy_identity, policy_evidence=outcome.policy_evidence,
            outcome=outcome.outcome.value, detail=outcome.detail,
            pair_id=None, arm_slot="arm_a", treatment_identity="v2")]),
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_parquet(directory / f"{name}.parquet")
    return directory


def test_the_offline_replay_reports_coverage_without_generating_descendants(tmp_path):
    """PLAN §8.4's gate: enough legal positive writes and exact band targets BEFORE any descendant.

    The replay runs the same ``select`` code path the cycle ran, so a coverage number here
    describes the policy that would actually run -- not a second implementation of its law.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.cycle import run_one_cycle
    from scripts.analysis.replay_v2_head_directed_policy import replay_bundle

    kwargs = _cycle_kwargs()
    outcome = run_one_cycle(**kwargs)
    assert outcome.committed, outcome.detail

    bundle = _write_bundle(tmp_path / "bundle", outcome, config=None)
    config = _replay_config()
    head = kwargs["head_oracle"]
    reference_sequence = F.reference_sequence(L)

    report = replay_bundle(
        bundles=[bundle], config=config, band_table=kwargs["band_table"],
        stratum_key=F.STRATUM, reference_sequence=reference_sequence,
        head_oracle=head, protein_id="5ZHV_B",
    )
    coverage = report["coverage"]
    assert coverage["n_scored"] >= 1
    assert coverage["n_with_legal_write_candidate"] >= 1
    assert coverage["n_with_positive_write"] >= 1
    assert coverage["n_committed_decisions"] >= 1
    assert coverage["head_contribution_measured"] is True
    assert coverage["realized_writes"]["n"] >= 1
    assert coverage["contribution"] is not None
    # The replay must reproduce the live decision it is replaying.
    source_row = next(row for row in report["sources"]
                      if row["state_id"] == outcome.source.state_id)
    assert source_row["outcome"] == "decision"
    assert source_row["selected_positions"] == sorted(
        outcome.projected.support.write_from_endpoint)
    assert source_row["required_reopen"] == len(outcome.projected.support.reopen)


def test_the_replay_refuses_to_fabricate_a_contribution_without_a_head(tmp_path):
    """``a_i = 0`` everywhere would read as "no position contributes" -- a measurement this replay
    has not taken.  Without a Head it must say so rather than report zeros."""
    from inverse_folding.reference_flow.fusion_v2_runtime.cycle import run_one_cycle
    from scripts.analysis.replay_v2_head_directed_policy import replay_bundle

    kwargs = _cycle_kwargs()
    outcome = run_one_cycle(**kwargs)
    bundle = _write_bundle(tmp_path / "bundle", outcome, config=None)
    config = _replay_config()

    with pytest.raises(SystemExit, match="no Head was loaded"):
        replay_bundle(
            bundles=[bundle], config=config, band_table=kwargs["band_table"],
            stratum_key=F.STRATUM, reference_sequence=F.reference_sequence(L),
            head_oracle=None, protein_id="5ZHV_B",
        )


# --------------------------------------------------------------------------------------------
# the counterfactual Head budget is DECLARED, PROJECTED and ENFORCED
# --------------------------------------------------------------------------------------------


def test_the_counterfactual_batch_is_bounded_by_the_declared_budget():
    """REGRESSION. The leave-one-out batch is one Head call per legal write candidate, so a run
    whose projection counted only the scored endpoints could pass `--dry-run` and then breach
    `max_head_calls` after paying for a model.

    Enforced BEFORE the batch runs, and by refusal rather than truncation: `a_i` is compared across
    ALL legal candidates to take the exact top `m_d`, so a partial batch would rank the best of an
    arbitrary subset while the artifact still claimed the exact rule.
    """
    source = _source(SOURCE_SEQ)
    donor = _donor(source=source)
    head = _ScriptedHead()
    result = _decide(source=source, endpoint=donor,
                     policy=_policy(head=head,
                                    calibration=_calibration(counterfactual_budget=1)))
    assert isinstance(result, pol.PolicyRejection)
    assert pol.StallReason.COUNTERFACTUAL_BUDGET_EXCEEDED.value in result.reason
    assert result.decision_evidence.n_legal_write_candidates == len(MASKED)
    assert head.batches == [], "the refusal must cost no Head call at all"


def test_the_budget_projection_charges_the_counterfactual_batch():
    """REGRESSION. `--dry-run` must see the Head calls the policy itself will make."""
    from scripts.rf_fusion_v2_preflight import project_v2_budget

    config = cfg.load_v2_config(_config_payload())
    projection = project_v2_budget(config, n_proteins=2)
    per_cycle = config.projection.head_directed.max_counterfactual_head_calls_per_cycle
    n_points = len(config.schedule.points)
    assert projection.per_protein_counterfactual_head_calls == per_cycle * n_points

    without = cfg.load_v2_config(_config_payload())
    without = dataclasses.replace(
        without, projection=dataclasses.replace(without.projection, head_directed=None))
    baseline = project_v2_budget(without, n_proteins=2)
    assert baseline.per_protein_counterfactual_head_calls == 0
    assert (projection.total_head_calls
            == baseline.total_head_calls + per_cycle * n_points * 2)


def test_a_projection_that_breaches_the_head_cap_on_counterfactuals_alone_is_infeasible():
    """The bound must be able to FAIL the launch gate, or charging it changes nothing."""
    from scripts.rf_fusion_v2_preflight import project_v2_budget

    payload = _config_payload()
    payload["projection"]["head_directed"]["max_counterfactual_head_calls_per_cycle"] = 10 ** 6
    projection = project_v2_budget(cfg.load_v2_config(payload), n_proteins=1)
    assert "max_head_calls" in projection.breached_caps
    assert not projection.feasible


# --------------------------------------------------------------------------------------------
# the production CLI path for the qualification contrast
# --------------------------------------------------------------------------------------------


def test_the_driver_selects_the_qualification_runner():
    """REGRESSION. `run_policy_qualification_view` had no caller outside the tests, so the PLAN
    §8.4 comparison could not be launched from a CLI at all."""
    import types

    from scripts.run_rf_fusion_v2 import V2DriverError, build_parser, select_runner

    assert "--qualification" in build_parser().format_help()

    chosen = select_runner(
        types.SimpleNamespace(mechanism_prefixes=4, mechanism_prefix_start=0, qualification=True),
        ladder="LADDER", mechanism=lambda **kw: kw)
    assert chosen() == {"n_prefixes": 4, "prefix_start": 0, "qualification": True}

    with pytest.raises(V2DriverError, match="--qualification needs --mechanism-prefixes"):
        select_runner(
            types.SimpleNamespace(mechanism_prefixes=0, mechanism_prefix_start=0,
                                  qualification=True),
            ladder="LADDER", mechanism=lambda **kw: kw)


def test_the_qualification_shard_refuses_a_config_with_no_head_directed_policy():
    """One law is not a contrast.  Refused before the shard pays for a model."""
    from scripts.rf_fusion_v2_cohort import V2CohortError, run_v2_mechanism_shard

    with pytest.raises(V2CohortError, match="--qualification needs a config"):
        run_v2_mechanism_shard(
            protein_id="5ZHV_B", config=F.v2_config(length=L), signature=None, out_dir=".",
            oracles_factory=lambda **kw: {}, n_prefixes=1, qualification=True)


def test_the_support_law_contrast_is_scorable_on_the_shared_free_domain():
    """The two support laws reopen DIFFERENT positions -- that is the treatment -- so their free
    domains differ by construction. Requiring equality (as the mechanism views do) would mark every
    pair unscorable and the qualification cohort would produce an empty table."""
    from scripts.rf_fusion_v2_artifacts import (
        POLICY_QUALIFICATION_CONTRAST_VIEWS,
        SCORABLE_CONTRAST_VIEWS,
        mechanism_contrast_rows,
    )
    from inverse_folding.reference_flow.fusion_v2_runtime.cycle import run_one_cycle

    assert "support_law" in SCORABLE_CONTRAST_VIEWS
    assert "support_law" in POLICY_QUALIFICATION_CONTRAST_VIEWS

    kwargs = _cycle_kwargs()
    treatment = run_one_cycle(**kwargs)
    assert treatment.committed, treatment.detail
    control_policy = pol.SourceGeometryControlPolicy(
        band_table=kwargs["band_table"], stratum_key=F.STRATUM,
        incumbent=kwargs["support_policy"].incumbent,
        evaluator=kwargs["support_policy"].evaluator,
        calibration=kwargs["support_policy"].calibration,
        required_writes=len(treatment.projected.support.write_from_endpoint),
        required_reopens=len(treatment.projected.support.reopen),
        policy_spec_digest=kwargs["support_policy"].policy_spec_digest,
    )
    control = run_one_cycle(**dict(
        kwargs, support_policy=control_policy,
        declared_policy=pol.DeclaredPolicy(
            policy_id=pol.SOURCE_GEOMETRY_CONTROL_POLICY_ID, policy_version="v1",
            is_diagnostic=False, phase="policy_qualification")))
    assert control.committed, control.detail

    rows = mechanism_contrast_rows(
        protein_id="5ZHV_B", source_index=0, source_seed=1, source_state_id=None,
        source_unresolved_editable=None, view="support_law",
        arm_a=treatment, arm_b=control)
    assert rows and any(row["analyzable"] for row in rows), [r["reason"] for r in rows]
    scored = next(row for row in rows if row["analyzable"])
    # The descendant ids are what a reader joins to `complete_endpoints.head_global_risk` for the
    # PRIMARY readout; the Hamming columns are the mechanism secondary.
    assert scored["arm_a_descendant_id"] and scored["arm_b_descendant_id"]

    # The same pair under a mechanism view is still refused: only the support-law contrast is
    # allowed to score on the intersection.
    refused = mechanism_contrast_rows(
        protein_id="5ZHV_B", source_index=0, source_seed=1, source_state_id=None,
        source_unresolved_editable=None, view="endpoint_change",
        arm_a=treatment, arm_b=control)
    if set(free_positions(treatment)) != set(free_positions(control)):
        assert not any(row["analyzable"] for row in refused)


def free_positions(cycle):
    from scripts.rf_fusion_v2_artifacts import free_domain

    return free_domain(cycle.projected)
