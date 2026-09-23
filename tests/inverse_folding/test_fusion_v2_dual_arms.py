"""DUALF3/PLAN §7: "Arm labels name the objective and nothing else" -- so the label must CHANGE it.

The defect this file pins was the second incarnation of the same failure. The first was that the
Dual layer was never called, so joint / a_only / b_only all executed A-only V2. The wiring repair
fixed that and left the arm label inert, so all three executed the JOINT law instead. Either way
C1's three-branch comparison contrasts one objective against itself twice: every arm agrees, and the
agreement means nothing.

A test that only asserts "the three arms are constructible" would pass in both broken states. What
distinguishes them is that the arms must ORDER A POOL DIFFERENTLY when the two alleles disagree --
so the fixture below makes them disagree and asserts the orderings actually differ.
"""

from __future__ import annotations

import pytest

from inverse_folding.reference_flow.fusion_v2 import joint_objective as jo

from .test_fusion_v2_dual_execution import _calibration, _runtime, _run

ARMS = ("joint", "a_only", "b_only")


def _objective(arm):
    return jo.build_arm_objective(_calibration(), arm=arm)


# -- the law itself -----------------------------------------------------------------------------

def test_each_arm_resolves_a_different_ordering_law():
    got = {arm: _objective(arm) for arm in ARMS}
    assert isinstance(got["joint"], jo.DualObjective)
    assert not isinstance(got["joint"], jo.SingleAlleleObjective)
    assert got["a_only"].role is jo.AlleleRole.A
    assert got["b_only"].role is jo.AlleleRole.B
    assert {o.arm for o in got.values()} == set(ARMS)


def test_the_single_allele_arms_value_exactly_their_own_coordinate():
    raw_a, raw_b = -3.0, 1.0
    for arm, role in (("a_only", "u_a"), ("b_only", "u_b")):
        value = _objective(arm).evaluate(raw_a=raw_a, raw_b=raw_b)
        assert value.value == pytest.approx(getattr(value, role))
        # Both coordinates are still computed and published: an a_only arm MEASURES role B and
        # declines to steer on it, which is what makes the contrast interpretable.
        assert value.u_a is not None and value.u_b is not None


def test_the_joint_arm_is_strictly_between_the_two_single_allele_arms():
    raw_a, raw_b = -3.0, 1.0
    values = {arm: _objective(arm).evaluate(raw_a=raw_a, raw_b=raw_b).value for arm in ARMS}
    worse, better = max(values["a_only"], values["b_only"]), min(values["a_only"], values["b_only"])
    assert better < values["joint"] <= worse
    assert values["a_only"] != values["b_only"], "fixture: the two alleles must disagree"


def test_two_arms_of_one_calibration_do_not_share_an_objective_identity():
    """Two arms share a calibration and must NOT share an objective digest, or an artifact that
    stamped both could never be split after the fact."""
    digests = {_objective(arm).evaluate(raw_a=-1.0, raw_b=2.0).objective_digest for arm in ARMS}
    assert len(digests) == 3


def test_a_single_allele_arm_uses_its_own_noise_floor_not_the_worse_of_the_two():
    import dataclasses

    base = _calibration()
    scaled = dataclasses.replace(base, risk=jo.QuantityCoordinates(
        quantity="global_risk",
        a=dataclasses.replace(base.risk.a, scale=2.0, raw_noise_floor=0.02),
        b=dataclasses.replace(base.risk.b, scale=8.0, raw_noise_floor=0.40)))
    joint = jo.build_arm_objective(scaled, arm="joint").joint_margin
    a_only = jo.build_arm_objective(scaled, arm="a_only").joint_margin
    b_only = jo.build_arm_objective(scaled, arm="b_only").joint_margin
    assert a_only == pytest.approx(0.02 / 2.0)
    assert b_only == pytest.approx(0.40 / 8.0)
    assert joint == pytest.approx(max(a_only, b_only))
    assert a_only != joint, (
        "an a_only arm held to the joint floor would differ from the joint arm in its gate WIDTH "
        "as well as in its ordering law -- two treatments where the design declares one")


def test_an_undeclared_arm_names_no_law():
    with pytest.raises(jo.V2JointObjectiveError, match="names no ordering law"):
        jo.build_arm_objective(_calibration(), arm="c_only")


# -- and it reaches the running kernel ----------------------------------------------------------

def _ordering(arm):
    """The realized (J, endpoint_id) ordering of one real cycle under one arm."""
    runtime = _runtime(arm=arm)
    outcome = _run(dual=runtime)
    evidence = runtime.evidence_by_endpoint
    return [e.endpoint_id for e in sorted(
        (e for e in outcome.endpoints if str(e.endpoint_id) in evidence),
        key=lambda e: (float(evidence[str(e.endpoint_id)].risk.value), str(e.endpoint_id)))]


def test_the_arms_order_a_real_pool_differently():
    orders = {arm: _ordering(arm) for arm in ARMS}
    assert len(orders["a_only"]) >= 2, "fixture: need a pool to have an ordering at all"
    assert orders["a_only"] != orders["b_only"], (
        "the two single-allele arms ranked the pool identically, so this fixture cannot detect an "
        "inert arm label at all")
    assert orders["joint"] != orders["a_only"] or orders["joint"] != orders["b_only"], (
        "the joint arm reproduced both single-allele orderings; the arm label is inert")


def test_a_cycle_records_the_arm_it_actually_ran():
    runtime = _runtime(arm="b_only")
    outcome = _run(dual=runtime)
    assert outcome.endpoints
    evidence = next(iter(runtime.evidence_by_endpoint.values()))
    # The value published for every endpoint is the ARM's value, not the joint one.
    expected = jo.build_arm_objective(_calibration(), arm="b_only").evaluate(
        raw_a=evidence.a.raw_risk, raw_b=evidence.b.raw_risk)
    assert evidence.risk.value == pytest.approx(expected.value)
    assert evidence.risk.value == pytest.approx(evidence.risk.u_b)


# -- why the arms need no in-process matched-pair engine -------------------------------------------

def test_separate_runs_of_two_arms_share_their_pool_by_determinism_alone():
    """The arms are a MATCHED comparison without any shared-source machinery.

    Seeds derive from ``(campaign, split, master_seed, protein)`` and the arm enters none of them;
    the depth-zero root capture never sees the Dual runtime. So two runs that differ only in
    ``--dual-arm`` produce the same root, the same lookahead pool and the same role A risks, and
    diverge exactly where the design says they should -- at selection.

    This is the property that makes "run it twice and compare the artifacts" a paired design rather
    than two independent samples, so it is asserted rather than assumed. An in-process engine that
    forked one realized pool would buy compute (the shared prefix is paid once instead of twice) and
    nothing else; if this test ever fails, that stops being true and the comparison stops being
    paired.
    """
    pools = {}
    for arm in ARMS:
        runtime = _runtime(arm=arm)
        outcome = _run(dual=runtime)
        pools[arm] = {
            "ids": [str(e.endpoint_id) for e in outcome.endpoints],
            "md5": [e.sequence_md5 for e in outcome.endpoints],
            # role A's raw risk must be identical too: it is the legacy field, and Dual may never
            # overwrite it with a joint scalar in any arm.
            "raw_a": [float(e.head_global_risk) for e in outcome.endpoints],
            "selected": str(getattr(outcome.selected_endpoint, "endpoint_id", None)),
        }

    base = pools["joint"]
    assert base["ids"], "fixture: the pool must be non-empty for this to mean anything"
    for arm in ARMS:
        for field in ("ids", "md5", "raw_a"):
            assert pools[arm][field] == base[field], (
                f"{arm} produced a different {field} from joint; the two arms are then two "
                "different samples and comparing their artifacts is not a paired design")

    assert len({pools[arm]["selected"] for arm in ARMS}) > 1, (
        "every arm selected the same endpoint, so this fixture cannot tell a working arm law from "
        "an inert one")
