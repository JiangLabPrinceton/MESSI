"""The state-derived diagnostic probe policy (PLAN §2.5, frozen by the V2 runbook).

``ExplicitProbePolicy`` takes PREDECLARED support sets, and the kernel requires ``reopen`` to name
only source-RESOLVED positions -- but which positions are resolved at ``c_d`` is stochastic.  A
predeclared partition therefore yields ``null_invalid_policy_result`` on a real state, so the first
state-transition diagnostic PLAN §2.5 authorises could pay a prefix, K lookaheads, a Head batch and
K refolds per cycle and measure nothing.

This policy derives the partition FROM THE LIVE STATE under a rule frozen in the runbook:

* the lowest-indexed source-masked editable position becomes ``write_from_endpoint``;
* every other inherited mask is carried;
* resolved tokens committed BEFORE ``r_d`` are carried;
* resolved tokens committed AT OR AFTER ``r_d`` are injected as source feedback;
* hard anchors never enter any set;
* ``reopen`` cardinality is not a free parameter -- it is pinned by ``B(r_d)`` through
  ``admissible_reopen_cardinality``.

Every way the rule can fail to apply is a TYPED NULL, never a quiet fallback: a fallback would let
the canary report a transition produced under a partition nobody declared.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from inverse_folding.reference_flow.fusion_v2 import policy as pol
from inverse_folding.reference_flow.fusion_v2 import schedule as sch
from tests.inverse_folding import _v2_fixtures as F




#: Stands in for the sha256 of ``configs/v2_state_derived_probe_policy_v1.json`` -- the run's
#: DECLARED ``projection_policy_spec`` role.  Supplied, never derived: the kernel checks the
#: answering policy's spec digest against the run's conditioning, and a policy that computed its
#: own would make both sides agree by construction.
SPEC_DIGEST = "5" * 64


def _policy(**over):
    table = over.pop("band_table", None) or F.band_table()
    kw = dict(band_table=table, stratum_key=F.STRATUM, policy_spec_digest=SPEC_DIGEST)
    kw.update(over)
    return pol.StateDerivedProbePolicy(**kw)


def _decide(policy=None, source=None, **over):
    policy = policy or _policy()
    return policy.decide(source=source or F.source(), endpoint=F.endpoint(),
                         coordinates=F.coords(**over))


# --------------------------------------------------------------------------------------------
# it is a diagnostic policy, and says so in the one vocabulary that decides
# --------------------------------------------------------------------------------------------


def test_the_probe_registers_itself_in_the_diagnostic_vocabulary():
    """PLAN §2.5: a diagnostic policy "may not become a silent production default".

    The gate is keyed on ``policy_id_is_diagnostic``, so a probe absent from that vocabulary would
    be admissible in a production phase -- which is exactly the defect the vocabulary merge closed.
    """
    identity = _policy().identity()
    assert identity.is_diagnostic_only is True
    assert pol.policy_id_is_diagnostic(identity.policy_id)


def test_the_identity_binds_the_band_CONTENT_not_only_its_name():
    """The reopen count is derived from ``B(r_d)``, so two runs under different calibrations ran
    different policies even at identical coordinates.

    Two tables that share a calibration ID and differ in CONTENT are the case that matters: a
    digest keyed on the name alone would give them one policy identity while they pinned different
    cardinalities.
    """
    wider = F.band_table(bands_override=(F.band(
        unresolved_accept=sch.BandInterval(lo=1.0, hi=3.0, lo_level=0.9, hi_level=0.1)),))
    assert wider.provenance.calibration_id == F.band_table().provenance.calibration_id
    assert wider.provenance.calibration_content_digest != \
        F.band_table().provenance.calibration_content_digest
    assert _policy().identity().policy_config_digest != \
        _policy(band_table=wider).identity().policy_config_digest


# --------------------------------------------------------------------------------------------
# the frozen partition rule
# --------------------------------------------------------------------------------------------


def test_the_partition_follows_the_frozen_rule():
    """The runbook's rule, position by position, against the shared fixture's source state."""
    source = F.source()
    r_step = F.R_STEP
    decision = _decide(source=source)
    assert isinstance(decision, pol.PolicyDecision)

    editable = set(source.editable_positions)
    masked = {p for p in editable if source.tokens[p] == source.mask_token_id}
    resolved = editable - masked
    reopened = set(decision.reopen)

    assert len(decision.write_from_endpoint) == 1
    write = decision.write_from_endpoint[0]
    assert write in masked, "the write site must be a SOURCE-MASKED position"
    assert write == min(masked), "the rule is deterministic: lowest-indexed masked position"

    for position in masked - {write}:
        assert position in decision.carry_from_source, "an unwritten inherited mask is carried"

    for position in resolved - reopened:
        commit = source.active_commit_depth_step_by_pos[position]
        if commit is not None and commit.step < r_step:
            assert position in decision.carry_from_source
        else:
            assert position in decision.inject_from_source_feedback


def test_hard_anchors_never_enter_any_support_set():
    """PLAN §2.4 makes anchors a permanent class; the kernel refuses a support that names one."""
    source = F.source()
    anchors = {position for position, _ in source.hard_anchors}
    assert anchors
    assert not anchors & set(_decide(source=source).claimed_positions)


def test_the_partition_is_disjoint_and_exhaustive_over_the_editable_domain():
    """The kernel enforces this, so a policy that violated it would only ever yield typed nulls."""
    source = F.source()
    claimed = _decide(source=source).claimed_positions
    assert len(claimed) == len(set(claimed)), "the four sets overlap"
    assert set(claimed) == set(source.editable_positions)


# --------------------------------------------------------------------------------------------
# reopen cardinality is pinned by B(r_d), not chosen
# --------------------------------------------------------------------------------------------


def test_the_reopen_cardinality_is_the_one_the_band_pins():
    """``schedule.admissible_reopen_cardinality``: "reopen size is not a free parameter".

    Reading it from the band is what makes the probe instantiable only AFTER a real ``B(r)`` scan,
    which is the order the runbook fixes.
    """
    source = F.source()
    table = F.band_table()
    decision = _decide(policy=_policy(band_table=table), source=source)
    editable = set(source.editable_positions)
    masked = {p for p in editable if source.tokens[p] == source.mask_token_id}
    load = sch.admissible_reopen_cardinality(
        band=sch.lookup_band(table, step=F.R_STEP, stratum_key=F.STRATUM),
        n_editable=len(editable), n_unresolved_source=len(masked),
        n_endpoint_writes_over_masked=1,
    )
    assert load.feasible
    assert len(decision.reopen) == load.min_newly_masked


def test_a_band_pinning_more_than_one_reopen_gets_more_than_one():
    """The cardinality is READ from the band.  A policy that always reopened one position would
    satisfy the fixture's ``min_newly_masked == 1`` while ignoring the calibration entirely."""
    two = F.band_table(bands_override=(F.band(
        unresolved_accept=sch.BandInterval(lo=2.0, hi=4.0, lo_level=0.9, hi_level=0.1)),))
    assert len(_decide(policy=_policy(band_table=two)).reopen) == 2


def test_reopen_takes_the_LATEST_committed_resolved_positions_first():
    """Two reasons, and the ordering is the frozen half of the rule.

    The most recently committed identity carries the most future information relative to ``r_d``;
    and taking from that end preserves the OLDER ``carry_from_source`` class, which is the only
    class a fixed-support ablation (PLAN §2.6 intervention 2) can target.  Index order would reopen
    the oldest history instead and erode exactly the class the ablation needs.
    """
    source = F.source()
    reopen = _decide(source=source).reopen
    resolved = [p for p in source.editable_positions
                if source.tokens[p] != source.mask_token_id]

    def step(position):
        commit = source.active_commit_depth_step_by_pos[position]
        return -1 if commit is None else int(commit.step)

    assert set(reopen) == set(sorted(resolved, key=step, reverse=True)[:len(reopen)])
    assert min(step(p) for p in reopen) > max(
        [step(p) for p in resolved if p not in set(reopen)] or [-2])


def test_a_band_demanding_more_reopens_than_there_are_resolved_positions_declines():
    """Reopening fewer than the band pins would leave the projected maturity outside ``B(r_d)``
    while the run reported a committed transition.

    The refusal comes from the SCHEDULE layer, which caps the admissible cardinality at the number
    of resolved positions and names the empty interval.  The policy's job is only to turn that into
    a typed decline instead of letting it escape as a crash.
    """
    source = F.source()
    n_resolved = sum(1 for p in source.editable_positions
                     if source.tokens[p] != source.mask_token_id)
    n_editable = len(source.editable_positions)
    # The rho axis is widened so ONLY the unresolved-mass axis binds: with the default rho band the
    # envelope would come back empty first and this check would never be reached.
    greedy = F.band_table(bands_override=(F.band(
        rho_accept=sch.BandInterval(lo=0.0, hi=1.0, lo_level=0.1, hi_level=0.9),
        rho_quantiles=(0.0, 0.5, 1.0),
        unresolved_accept=sch.BandInterval(lo=float(n_editable), hi=float(n_editable),
                                           lo_level=0.9, hi_level=0.1)),))
    assert n_editable > n_resolved
    decision = _decide(policy=_policy(band_table=greedy))
    assert isinstance(decision, pol.PolicyRejection)
    # The band is what refuses, and the message must say so: an operator reading it has to know
    # whether to re-calibrate B(r) or to look at the state.
    assert "no legal reopen cardinality" in decision.reason, decision.reason


def test_every_reopened_position_was_resolved_in_the_source():
    """The kernel refuses a reopen of an already-unresolved position: relabelling an inherited mask
    newly masks nothing and would satisfy the non-empty requirement without opening any identity."""
    source = F.source()
    for position in _decide(source=source).reopen:
        assert source.tokens[position] != source.mask_token_id


# --------------------------------------------------------------------------------------------
# every inapplicable case is a TYPED NULL, never a quiet fallback
# --------------------------------------------------------------------------------------------


def test_the_state_layer_makes_a_missing_write_site_unreachable():
    """The write site is the lowest-indexed SOURCE-MASKED position, and there is always one.

    ``LivePartialState`` refuses a fully resolved state outright -- such a state can emit no
    lookahead and is a terminal endpoint -- so the policy's "no write site" decline is a BACKSTOP,
    not a reachable branch.  Recorded as such rather than left looking like a live guard: a future
    state layer that relaxed the invariant must not silently start writing the endpoint over a
    resolved position, which would change the coupled mask-load ``a`` term and move the admissible
    reopen envelope.
    """
    from inverse_folding.reference_flow.fusion_v2.errors import V2Error

    source = F.source()
    with pytest.raises(V2Error, match="unresolved"):
        F.source(tokens=tuple(7 if t == source.mask_token_id else t for t in source.tokens))


def test_a_band_that_cannot_be_satisfied_declines_with_a_typed_null():
    """Not enough resolved positions to reach the pinned cardinality is a fact about the STATE
    against the calibration, and the run must say so rather than reopening fewer."""
    narrow = sch.make_band_table(
        provenance=F.band_provenance(),
        bands=(F.band(unresolved_accept=sch.BandInterval(lo=5.0, hi=6.0, lo_level=0.9,
                                                         hi_level=0.1)),),
    )
    decision = _decide(policy=_policy(band_table=narrow))
    assert isinstance(decision, pol.PolicyRejection)


def test_the_decline_still_names_the_policy_that_declined():
    """A refusal is a policy ANSWER (PLAN §4.5), and the kernel refuses an unattributable one."""
    narrow = sch.make_band_table(
        provenance=F.band_provenance(),
        bands=(F.band(unresolved_accept=sch.BandInterval(lo=5.0, hi=6.0, lo_level=0.9,
                                                         hi_level=0.1)),),
    )
    decision = _decide(policy=_policy(band_table=narrow))
    assert isinstance(decision, pol.PolicyRejection)
    assert decision.policy == _policy(band_table=narrow).identity()


# --------------------------------------------------------------------------------------------
# the whole point: it actually drives the kernel
# --------------------------------------------------------------------------------------------


def test_the_probe_produces_a_projection_the_kernel_commits():
    """The defect this policy exists to close: ``ExplicitProbePolicy`` returns predeclared sets and
    the kernel answers ``null_invalid_policy_result`` on a realized state, so the canary measures
    only typed nulls.  A state-derived partition has to survive every kernel invariant."""
    table = F.band_table()
    probe = _policy(band_table=table)
    outcome = F.project(
        # The run's frozen conditioning must NAME the policy that answers -- the kernel binds the
        # answering spec digest to ``conditioning.projection_policy_spec``.
        source=F.source(conditioning=F.conditioning(
            projection_policy_spec=probe.identity().policy_spec_digest,
            schedule_band_calibration=table.provenance.calibration_content_digest)),
        decision=_decide(policy=probe),
        declared_policy=F.declared_policy(
            policy_id=probe.identity().policy_id, policy_version=probe.policy_version),
        band_table=table,
        declared_band_digest=table.provenance.calibration_content_digest,
    )
    assert outcome.committed, outcome.detail


def test_the_committed_partition_leaves_a_carried_resolved_position():
    """PLAN §2.6 intervention 2 -- fixed-support ablation -- exists only on a CARRIED, RESOLVED
    position: masking one in any other class makes the pinned partition illegal.  The rule keeps
    that class populated wherever the source has history older than ``r_d``, so the ablation arm
    can return a verdict instead of a typed 'no admissible target'."""
    source = F.source()
    decision = _decide(source=source)
    carried_resolved = [p for p in decision.carry_from_source
                        if source.tokens[p] != source.mask_token_id]
    assert carried_resolved


# --------------------------------------------------------------------------------------------
# the SPEC digest and the CONFIG digest are two different questions
# --------------------------------------------------------------------------------------------


def _band(calibration_id, lo):
    """A DIFFERENT cell's calibration: another id and another pinned envelope.

    Built through ``make_band_table`` rather than by replacing provenance fields, because the table
    rebinds ``calibration_content_digest`` to its own content -- a hand-set digest would not be the
    one the loader recomputes.
    """
    return F.band_table(
        bands_override=(F.band(unresolved_accept=sch.BandInterval(
            lo=lo, hi=lo + 2.0, lo_level=0.9, hi_level=0.1)),),
        calibration_id=calibration_id)


def test_the_spec_digest_is_the_declared_one_not_a_derived_stand_in():
    """The kernel compares it against `conditioning.projection_policy_spec`, which is the sha256 of
    the frozen spec FILE.  A policy that derived its own would make both sides agree by
    construction -- and a canonical dict digest could never equal a file sha256, so the check as
    previously written could not pass at all."""
    assert _policy().identity().policy_spec_digest == SPEC_DIGEST


def test_the_spec_digest_is_invariant_across_cells():
    """REGRESSION.  The spec digest used to fold in `stratum_key` and the band, so every Canary
    cell got a different one and no single frozen file could sign the four-cell campaign."""
    cell_a = _policy(stratum_key="5zhv_b_unconstrained",
                     band_table=_band("band:v2:canary:5zhv_b_unconstrained:v1", 1.0)).identity()
    cell_b = _policy(stratum_key="q00511_anchor24",
                     band_table=_band("band:v2:canary:q00511_anchor24:v1", 4.0)).identity()
    assert cell_a.policy_spec_digest == cell_b.policy_spec_digest == SPEC_DIGEST
    # ...while they remain distinguishable as CONFIGURATIONS.
    assert cell_a.policy_config_digest != cell_b.policy_config_digest


def test_the_config_digest_still_tracks_the_band_and_the_stratum():
    """The per-cell binding must NOT be lost in the split: two runs under different calibrations
    pinned different reopen cardinalities, so they ran the same rule under different config."""
    base = _policy().identity().policy_config_digest
    assert _policy(band_table=_band("band:other", 4.0)).identity().policy_config_digest != base
    assert _policy(stratum_key="another_stratum").identity().policy_config_digest != base


def test_the_two_digests_are_not_the_same_value():
    identity = _policy().identity()
    assert identity.policy_config_digest != identity.policy_spec_digest


@pytest.mark.parametrize("bad", ["", "   ", "unset:projection_policy_spec"])
def test_a_policy_with_no_usable_spec_digest_is_refused(bad):
    """PLAN §5.2's "missing content identity fails closed" -- the policy may not answer under a
    spec the run never declared."""
    with pytest.raises(Exception, match="policy_spec_digest"):
        _policy(policy_spec_digest=bad)


# --------------------------------------------------------------------------------------------
# the shipped spec file
# --------------------------------------------------------------------------------------------

SPEC_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / "inverse_folding/reference_flow/configs/v2_state_derived_probe_policy_v1.json")


def _spec():
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def test_the_shipped_spec_names_the_policy_the_code_registers():
    spec = _spec()
    assert spec["policy_id"] == pol.STATE_DERIVED_PROBE_POLICY_ID
    assert spec["policy_version"] == _policy().identity().policy_version
    assert spec["is_diagnostic_only"] is True


def test_the_shipped_spec_carries_no_per_cell_configuration():
    """A spec containing `stratum_key`, a band digest, a protein id or `r_step` would be a
    per-cell artifact wearing a campaign-wide role, and the four Canary cells could not share it."""
    blob = json.dumps(_spec())
    for banned in ('"stratum_key":', '"r_step":', '"protein_id":',
                   '"calibration_content_digest":', '"c_source_step":', '"c_next_step":'):
        assert banned not in blob, f"the frozen spec must not carry {banned}"


def test_the_shipped_spec_describes_every_support_class_the_policy_emits():
    classes = set(_spec()["state_partition"]["classes"])
    assert classes == {"write_from_endpoint", "inject_from_source_feedback",
                       "reopen", "carry_from_source"}
    assert {rule["class"] for rule in _spec()["rules"]} == classes
