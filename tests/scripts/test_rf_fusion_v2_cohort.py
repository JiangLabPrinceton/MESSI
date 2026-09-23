"""V2F7: the per-protein shard stage -- arm identity, evidence graph, and compute ledger.

What this suite protects is the join between a frozen config and what a run ACTUALLY did.  The
shard is the only place those two meet, and three ways they can silently come apart are each worth
a suite of their own:

**The config must be the runtime authority.**  ``config.arm.feedback_enabled`` reaching only the
manifest means an A2 config can run WITH feedback while its own artifact records
``feedback_enabled: false`` -- a treatment arm labelled as its own control, indistinguishable from
the real control in every downstream table.

**The evidence graph must carry what PLAN §5.3 calls load-bearing.**  Endpoint depth, the projected
state at ``r_d``, the policy that produced each transition, the A2 view, the admission reason and
the terminal validation are all named there.  A shard that emits a sparse payload still exits 0, so
nothing downstream can tell "the run produced none of that" from "the writer never collected it".

**A run that spent budget must report it.**  The journal exists so that work started and never
finished is recorded as ``unknown_after_start`` rather than as free; a shard whose payload drops
the ledger throws that record away at the one boundary where it becomes an artifact.

Every test here drives the REAL ladder over fake oracles.  PLAN's scientific boundary applies
unchanged: a green result means the wiring and the identities hold, never that feedback transmits.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from inverse_folding.reference_flow.fusion_v2 import config as cfg  # noqa: E402
from inverse_folding.reference_flow.fusion_v2_runtime.capture import (  # noqa: E402
    FullyResolvedRootError,
)
from inverse_folding.reference_flow.sampler import (  # noqa: E402
    PositionDependentDFMSampler,
)
from scripts.rf_fusion_v2_cohort import (  # noqa: E402
    ShardInputs,
    V2CohortError,
    run_v2_shard,
)
from scripts.rf_fusion_v2_resume import RunSignature  # noqa: E402
from tests.inverse_folding import _v2_fixtures as F  # noqa: E402
from tests.inverse_folding.test_fusion_v2_cycle import (  # noqa: E402
    ALPHABET,
    VOCAB,
    _cfg,
    _denoiser,
    _FakeHeadOracle,
    _policy_identity,
    _structure_oracle,
    _support_policy,
)
from tests.inverse_folding.test_fusion_v2_ladder import (  # noqa: E402
    C0,
    C1,
    C2,
    LADDER_L,
    R1,
    R2,
    _ladder_band_table,
)

PROTEIN = "5ZHV_B"


# --------------------------------------------------------------------------------------------
# fixtures: one real two-depth ladder over fake oracles
# --------------------------------------------------------------------------------------------


def _v2_config(**over):
    """The shared V2 config, re-pointed at the ladder fixture's declared two-depth schedule.

    Built by REPLACING fields of the shared fixture rather than by hand, so this suite cannot drift
    into describing a config shape the validator would reject.
    """
    base = F.v2_config()
    schedule = dataclasses.replace(
        base.schedule,
        depth_cap=2,
        points=(
            cfg.DepthSchedulePoint(depth=0, r_step=R1, c_source_step=C0, c_next_step=C1,
                                   n_lookaheads=3, band_key="step40"),
            cfg.DepthSchedulePoint(depth=1, r_step=R2, c_source_step=C1, c_next_step=C2,
                                   n_lookaheads=2, band_key="step52"),
        ),
    )
    return dataclasses.replace(base, schedule=schedule, **over)


def _a2_config(**over):
    """The same science with the feedback stage OFF -- the control arm, not a second run."""
    base = _v2_config(**over)
    return dataclasses.replace(
        base, arm=dataclasses.replace(base.arm, feedback_enabled=False, arm_role="a2"))


class _NamedPolicy:
    """A support policy that can NAME itself, which is what PLAN §2.5's Protocol requires.

    The bare function the cycle suite uses is a probe; a run whose feedback events cannot say which
    policy produced them has no auditable policy column, which is exactly what this suite pins.
    """

    def identity(self):
        return _policy_identity()

    def __call__(self, source, endpoint, coordinates):
        return _support_policy(source, endpoint, coordinates)


def _cycle_kwargs(**over):
    table = _ladder_band_table()
    kwargs = dict(
        sampler=PositionDependentDFMSampler(mask_token_id=F.MASK, vocab_size=VOCAB),
        denoiser=_denoiser, config=_cfg(), sequence_length=LADDER_L,
        h_values=np.zeros(LADDER_L, dtype=np.float32), residue_token_ids=F.AA,
        alphabet=ALPHABET, fixed_tokens={0: 10}, lineage=F.lineage(),
        mask_token_id=F.MASK, aa_token_ids=F.AA,
        conditioning=F.conditioning(
            schedule_band_calibration=table.provenance.calibration_content_digest),
        safety_reference=F.safety_reference(length=LADDER_L),
        head_oracle=_FakeHeadOracle(length=LADDER_L), structure_oracle=_structure_oracle,
        support_policy=_NamedPolicy(),
        band_table=table, stratum_key=F.STRATUM, declared_band_id=F.BAND_ID,
        declared_band_digest=table.provenance.calibration_content_digest,
        safety_gate=F.safety_gate(length=LADDER_L),
    )
    kwargs.update(over)
    return kwargs


def _factory(**over):
    """An oracles factory for the fake world, with the same shape the real one must return."""

    def build(*, protein_id, config, inputs):
        del protein_id, config, inputs
        return {
            "cycle_kwargs": _cycle_kwargs(**over.pop("cycle_kwargs", {})),
            "gpu_clock": lambda: 0.0,
            **over,
        }

    return build


def _shard(tmp_path, *, config=None, **over):
    resolved = config or _v2_config()
    return run_v2_shard(
        protein_id=PROTEIN, config=resolved,
        signature=_signature(resolved, production_depth_authorized=True), out_dir=tmp_path,
        oracles_factory=_factory(**over), production_depth_authorized=True,
    )


def _signature(config=None, **over):
    config = config or _v2_config()
    fields = dict(
        config_digest=config.config_digest(), campaign_id=config.identity.campaign_id,
        split_role=config.identity.split_role, arm_role=config.arm.arm_role,
        protein_id=PROTEIN, input_signature="1" * 64,
        code_revision=config.identity.code_revision,
    )
    fields.update(over)
    return RunSignature(**fields)


_CACHE: dict = {}


def _default_shard(tmp_path_factory):
    """One realized two-depth run, shared by every read-only assertion below.

    Each ladder run costs real sampler work; re-running it per test would turn this suite into the
    slowest one in the repo without testing anything a second time.
    """
    if "default" not in _CACHE:
        out = tmp_path_factory.mktemp("shard")
        _CACHE["default"] = (out, _shard(out))
    return _CACHE["default"]


@pytest.fixture(scope="module")
def realized(tmp_path_factory):
    out, (status, payload) = _default_shard(tmp_path_factory)
    return out, status, payload


# --------------------------------------------------------------------------------------------
# the frozen config is the runtime authority (arm identity)
# --------------------------------------------------------------------------------------------


def test_the_shard_runs_the_arm_its_config_declares(tmp_path):
    """An A2 config must produce an A2 run.

    ``feedback_enabled`` reaching only the manifest is the worst kind of defect available here: the
    artifact says ``false`` while the engine ran the treatment, so the control arm's own tables are
    a record of the treatment and every contrast computed from them is between V2 and V2.
    """
    config = _a2_config()
    status, payload = _shard(tmp_path, config=config)
    assert status == "ok"
    assert payload["feedback_events"], "an A2 cycle is still an event and must be recorded"
    assert all(row["projected_state_id"] is None for row in payload["feedback_events"]), (
        "feedback ran in an arm whose config disabled it")
    assert not [row for row in payload["partial_states"] if row["layer"] == "projected"]
    assert payload["a2_views"]
    assert all(row["matching_resource"] is None for row in payload["a2_views"])
    assert all(row["matching_status"] == "declared_not_executed"
               for row in payload["a2_views"])
    assert all(row["declared_matching_resource"] == config.arm.a2_matching_resource
               for row in payload["a2_views"])


def test_a_shard_refuses_an_unbound_run_signature(tmp_path):
    with pytest.raises(V2CohortError, match="signature"):
        run_v2_shard(
            protein_id=PROTEIN, config=_v2_config(), signature=None, out_dir=tmp_path,
            oracles_factory=_factory(),
        )


def test_the_treatment_arm_still_projects(tmp_path):
    """The guard above must not be satisfiable by never projecting at all."""
    _, payload = _shard(tmp_path)
    assert any(row["projected_state_id"] is not None for row in payload["feedback_events"])


def test_depth_one_shard_needs_no_authorization_marker(tmp_path):
    """The exceptional D>1 seam must not widen or break the ordinary D=1 production surface."""
    config = _v2_config()
    config = dataclasses.replace(
        config,
        schedule=dataclasses.replace(
            config.schedule, depth_cap=1, points=(config.schedule.points[0],)),
    )
    status, payload = run_v2_shard(
        protein_id=PROTEIN, config=config, signature=_signature(config), out_dir=tmp_path,
        oracles_factory=_factory(),
    )
    assert status == "ok"
    assert payload["production_depth_authorized"] is False
    assert payload["exploratory_depth_override"] is False


def test_one_editable_position_is_a_successful_terminal_best_of_k(tmp_path):
    config = _v2_config()
    config = dataclasses.replace(
        config,
        schedule=dataclasses.replace(
            config.schedule, depth_cap=1, points=(config.schedule.points[0],)),
    )
    runtime = _cfg()
    runtime = dataclasses.replace(
        runtime, sampler=dataclasses.replace(runtime.sampler, seed=41))

    def policy_must_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("one-position fallback must stop before feedback projection")

    status, payload = run_v2_shard(
        protein_id=PROTEIN, config=config, signature=_signature(config), out_dir=tmp_path,
        oracles_factory=_factory(cycle_kwargs={
            "config": runtime,
            "fixed_tokens": {i: 10 for i in range(LADDER_L) if i != 5},
            "support_policy": policy_must_not_run,
        }),
    )

    assert status == "ok"
    assert payload["depth_reached"] == 0
    assert payload["stopping_reason"] == "terminal_best_lookahead"
    assert payload["terminal_validation"]
    assert payload["root_capture"]["attempts_used"] == 1
    assert payload["root_capture"]["n_unresolved_editable"] == 1


def test_config_max_retries_bounds_root_recapture(tmp_path):
    class AlwaysResolved:
        def sample(self, **kwargs):
            del kwargs
            raise FullyResolvedRootError("rho_edit == 1.0")

    config = _v2_config()
    config = dataclasses.replace(
        config, caps=dataclasses.replace(config.caps, max_retries=2))
    _, payload = _shard(
        tmp_path, config=config,
        cycle_kwargs={"sampler": AlwaysResolved()},
    )

    assert payload["root_capture"]["attempts_used"] == 3


@pytest.mark.parametrize("owned", ["feedback_enabled", "cost_meter"])
def test_a_factory_may_not_override_the_runs_arm_identity_or_its_journal(tmp_path, owned):
    """The factory builds oracles; it does not get to say which arm this is.

    A factory able to set ``feedback_enabled`` reintroduces exactly the defect above by another
    route, and one able to swap the ``cost_meter`` can point a run's journal at a file nobody
    aggregates.  Both are refused loudly rather than resolved by precedence, because a silent
    precedence rule is invisible in the artifact.
    """
    with pytest.raises(V2CohortError, match=owned):
        _shard(tmp_path, cycle_kwargs={owned: True})


def test_the_shard_journals_every_oracle_request_under_the_declared_out_dir(realized):
    """PLAN §5.4: requests are journaled BEFORE execution, so a process that dies inside an oracle
    still leaves a record that budget was started."""
    out, _, _ = realized
    journals = sorted(out.rglob("*.jsonl"))
    assert journals, "the shard ran real oracles and journaled nothing"
    rows = [json.loads(line) for line in journals[0].read_text().splitlines() if line.strip()]
    assert {row["record"] for row in rows} >= {"requested", "observed"}
    assert {row["arm"] for row in rows if row["record"] == "requested"} == {"v2"}


def test_a_gpu_run_that_cannot_name_its_instrument_is_refused(tmp_path, monkeypatch):
    """``check_caps`` treats GPU-seconds as a MEASURED quantity.

    A process that has initialized CUDA and reports 0.0 would certify ``max_gpu_seconds`` against a
    number nobody took.  A CPU process reporting 0.0 is making a true measurement -- the difference
    is only that it can say which instrument made it, which is why the fallback is conditioned on
    the device state rather than on convenience.
    """
    import torch

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(V2CohortError, match="gpu_clock"):
        config = _v2_config()
        run_v2_shard(
            protein_id=PROTEIN, config=config,
            signature=_signature(config, production_depth_authorized=True), out_dir=tmp_path,
            oracles_factory=_factory(gpu_clock=None), production_depth_authorized=True,
        )


def test_the_journal_path_is_a_runtime_input_not_a_hardcoded_location(tmp_path):
    """Cluster paths arrive from the CLI.  A shard that could only journal beside its out_dir would
    make the journal unroutable on a filesystem where that directory is not writable."""
    elsewhere = tmp_path / "scratch" / "journals"
    _shard(tmp_path / "out", inputs=None)  # warm the default location
    run_v2_shard(
        protein_id=PROTEIN, config=_v2_config(),
        signature=_signature(production_depth_authorized=True),
        out_dir=tmp_path / "out2",
        inputs=ShardInputs(journal_dir=elsewhere), oracles_factory=_factory(),
        production_depth_authorized=True,
    )
    assert sorted(p.name for p in elsewhere.rglob("*.jsonl")) == [f"{PROTEIN}.attempts.jsonl"]


def test_journal_rows_from_another_run_signature_are_not_aggregated(tmp_path):
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import AttemptJournal

    journal_root = tmp_path / "journals"
    stale = AttemptJournal(journal_root / f"{PROTEIN}.attempts.jsonl")
    attempt_id = stale.open_request(
        event_id="evt:stale", protein_id=PROTEIN, arm="v2", phase="screen",
        request_kind="lookahead_pool", request_digest="f" * 64, logical_dfe=999,
    )
    stale.close_observed(
        attempt_id, status="ok", physical_forwards=999, gpu_seconds=0.0, walltime_s=0.0,
    )

    signature = _signature(production_depth_authorized=True)
    _, payload = run_v2_shard(
        protein_id=PROTEIN, config=_v2_config(), signature=signature,
        out_dir=tmp_path / "out", inputs=ShardInputs(journal_dir=journal_root),
        oracles_factory=_factory(), production_depth_authorized=True,
    )
    assert "evt:stale" not in {row["event_id"] for row in payload["ledger_events"]}
    assert (journal_root / signature.value / f"{PROTEIN}.attempts.jsonl").exists()


# --------------------------------------------------------------------------------------------
# the declared substrate must be the substrate that runs (PLAN §3.2, §5.1)
# --------------------------------------------------------------------------------------------


def _runtime_config(*, sampler=None, remask=None, **over):
    """The runtime ``ReferenceFlowConfig`` the oracles factory hands the engine.

    Built by REPLACING fields of the fixture the healthy run uses, so a drift test perturbs exactly
    the one field it is about and nothing else.
    """
    base = _cfg()
    sampler_over = dict(sampler or {})
    if remask:
        sampler_over["remask"] = dataclasses.replace(base.sampler.remask, **remask)
    if sampler_over:
        base = dataclasses.replace(
            base, sampler=dataclasses.replace(base.sampler, **sampler_over))
    return dataclasses.replace(base, **over)


def _amplified(form):
    base = _cfg()
    return dataclasses.replace(
        base, amplification=dataclasses.replace(base.amplification, form=form))


#: One drift per ``config.substrate`` field that has a runtime counterpart.  Each entry is a
#: runtime config that contradicts the V2 config the run declared and the manifest publishes.
SUBSTRATE_DRIFTS = {
    "n_steps": _runtime_config(sampler={"n_steps": 140}),
    "temperature": _runtime_config(sampler={"temperature": 3.0}),
    "amplification_form": _amplified("exp_linear"),
    "remask_enabled": _runtime_config(remask={"enabled": False}),
    "remask_fraction_scale": _runtime_config(remask={"fraction_scale": 0.5}),
}


@pytest.mark.parametrize("field", sorted(SUBSTRATE_DRIFTS))
def test_a_runtime_substrate_that_contradicts_the_declared_one_is_refused(tmp_path, field):
    """PLAN §3.2: a mismatch in sampler steps or the frozen substrate "must fail before a denoiser
    call"; PLAN §5.1 makes the substrate part of the V2 config's identity.

    The declared substrate was reaching only the ``DepthPlan`` and the preflight budget projection.
    The sampler ran against the SEPARATE ``ReferenceFlowConfig`` the oracles factory returns, and
    nothing compared the two: a 140-step runtime burned 654 logical DFE under a launch gate that
    had authorized 374, and a temperature the config digest said was 1.0 silently rescaled every
    assimilated token log-probability.
    """
    with pytest.raises(V2CohortError, match=field):
        _shard(tmp_path, cycle_kwargs={"config": SUBSTRATE_DRIFTS[field]})


def test_a_runtime_substrate_that_declares_a_controller_is_refused(tmp_path):
    """The V2/A2 substrate is controller-free (PLAN §5.1, ``_assert_frozen_substrate``).

    ``ReferenceFlowConfig`` has no controller surface, so this is the one substrate field whose
    runtime counterpart is an ABSENCE.  A factory that attached one is declaring a different
    kernel, and the config that says ``controller_enabled: false`` would be describing a run that
    did not happen.
    """
    import types

    base = _cfg()
    runtime = types.SimpleNamespace(
        sampler=base.sampler, schedule=base.schedule, amplification=base.amplification,
        h_shuffle=base.h_shuffle, controller=types.SimpleNamespace(enabled=True),
    )
    with pytest.raises(V2CohortError, match="controller"):
        _shard(tmp_path, cycle_kwargs={"config": runtime})


def test_the_substrate_is_checked_before_any_denoiser_call(tmp_path):
    """PLAN §3.2 is explicit about WHEN: "must fail before a denoiser call".

    The ladder pays a root prefix of real forward passes before the first cycle, so a check that
    ran inside the engine would refuse a run only after buying the very compute the gate exists to
    authorize.
    """
    calls: list[float] = []

    def counting_denoiser(x_t, t, struct):
        calls.append(float(t))
        return _denoiser(x_t, t, struct)

    with pytest.raises(V2CohortError, match="n_steps"):
        _shard(tmp_path, cycle_kwargs={"config": SUBSTRATE_DRIFTS["n_steps"],
                                       "denoiser": counting_denoiser})
    assert calls == [], "the substrate mismatch was found only after paying for forward passes"


def test_a_factory_that_names_no_runtime_config_is_refused(tmp_path):
    """A shard with no runtime config has nothing to check the declared substrate against, and
    PLAN §5.2 fails closed on missing content identity rather than proceeding unchecked."""
    with pytest.raises(V2CohortError, match="config"):
        _shard(tmp_path, cycle_kwargs={"config": None})


def test_the_realized_substrate_digest_reaches_the_payload(realized):
    """PLAN §5.1 requires the frozen substrate identity to be recorded.

    The digest is computed by the LADDER from the config the engine really ran, so it is the run's
    one realized witness; a payload that dropped it would leave no record of the substrate at all.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.ladder import _substrate_digest

    _, _, payload = realized
    assert payload["substrate_digest"] == _substrate_digest(_cfg())


# --------------------------------------------------------------------------------------------
# the evidence graph (PLAN §5.3)
# --------------------------------------------------------------------------------------------


def test_every_endpoint_carries_the_depth_it_was_generated_at(realized):
    """A two-depth ladder produces endpoints at depths 0, 1 and 2.

    Depth 0 is the source pool; depth d's descendants ARE depth d+1's source pool, and the deepest
    rung's descendants are inherited by nobody -- so a writer that collected only ``cycle.endpoints``
    loses the last pool entirely and stamps depth 0 on every row that survives.  Every per-depth
    rate read off that table would then be computed over the wrong denominator.
    """
    _, _, payload = realized
    rows = payload["complete_endpoints"]
    assert {row["depth"] for row in rows} == {0, 1, 2}
    ids = [row["endpoint_id"] for row in rows]
    assert len(set(ids)) == len(ids), "an inherited pool was written twice"


def test_the_projected_state_at_every_reentry_reaches_the_partial_state_table(realized):
    """``q_phi``'s own output is the object the projection produced; PLAN §5.3 requires partial
    states to be reconstructible, and a table holding only the source and the propagated capture
    cannot show what the projection did between them."""
    _, _, payload = realized
    layers = [row["layer"] for row in payload["partial_states"]]
    assert layers.count("projected") == 2, "one projected state per committed rung"
    assert "live" in layers
    ids = [row["state_id"] for row in payload["partial_states"]]
    assert len(set(ids)) == len(ids), "a rung's source is the previous rung's propagated capture"
    for row in payload["partial_states"]:
        if row["layer"] == "projected":
            assert row["r_step"] in (R1, R2)


def test_every_feedback_event_names_the_policy_that_produced_it(realized):
    """PLAN §5.3 makes the policy a load-bearing field of the feedback event.

    Null policy columns make two runs under two different policies indistinguishable in the one
    table that records what the policy did.
    """
    _, _, payload = realized
    committed = [row for row in payload["feedback_events"]
                 if row["projected_state_id"] is not None]
    assert committed
    identity = _policy_identity()
    for row in committed:
        assert row["policy_id"] == identity.policy_id
        assert row["policy_version"] == identity.policy_version
        assert row["policy_config_digest"] == identity.policy_config_digest
        assert row["policy_spec_digest"] == identity.policy_spec_digest
        assert row["policy_is_diagnostic"] is True


def test_a_feedback_arm_whose_policy_answer_cannot_be_attributed_is_refused(tmp_path):
    """A committed projection with an anonymous policy has no auditable provenance.

    The identity is read off the DECISION the kernel acted on, not off the policy callable -- the
    two are separate sources that can disagree, and the artifact must record which policy produced
    this transition.  ``_support_policy`` is therefore NOT anonymous: it stamps every answer it
    returns.  A genuinely unattributable answer is one whose ``PolicyDecision.policy`` is ``None``,
    and the kernel now refuses it outright (``test_fusion_v2_projection``), so this shard-level
    check is the second line: an identity lost between the kernel and the writer still refuses
    rather than writing null policy columns and calling the run complete.
    """
    from inverse_folding.reference_flow.fusion_v2.errors import V2Error

    def _anonymous(source, endpoint, coordinates):
        decision = _support_policy(source, endpoint, coordinates)
        return dataclasses.replace(decision, policy=None)

    with pytest.raises(V2Error, match="policy|identit"):
        _shard(tmp_path, cycle_kwargs={"support_policy": _anonymous})


def test_the_a2_view_of_every_cycle_reaches_the_bundle(realized):
    """PLAN §4.3: the A2 view is the exact pre-feedback archive membership.

    An empty A2 table would make the control arm unreconstructible from the artifact, which is the
    one thing PLAN forbids reconstructing post hoc.
    """
    _, _, payload = realized
    rows = payload["a2_views"]
    assert len(rows) == 2
    assert {row["depth"] for row in rows} == {0, 1}
    assert all(json.loads(row["member_endpoint_ids_json"]) for row in rows)
    assert {row["matching_resource"] for row in rows} == {None}
    assert {row["declared_matching_resource"] for row in rows} == {
        _v2_config().arm.a2_matching_resource}
    assert {row["matching_status"] for row in rows} == {"declared_not_executed"}


def test_the_archive_records_why_each_endpoint_was_or_was_not_admissible(realized):
    """PLAN §4.2 keeps the failure explicit.  "Not eligible" without a reason cannot be audited:
    a design refused for a hotspot and one refused for a fold read identically."""
    _, _, payload = realized
    rows = payload["archive"]
    assert rows
    assert all(row["admission_reason"] for row in rows)


def test_terminal_validation_covers_every_definitive_design(realized):
    """The designs the run returns must appear with their structure and immune results SEPARATE."""
    _, _, payload = realized
    rows = payload["terminal_validation"]
    assert rows
    definitive = {row["endpoint_id"] for row in payload["archive"]
                  if row["may_become_ancestry"]}
    assert {row["endpoint_id"] for row in rows} == definitive
    for row in rows:
        assert row["structure_definitive"] is True
        assert json.loads(row["structure_metrics_json"])
        assert row["immune_evaluator"]
        assert row["immune_global_risk"] is not None
        assert row["diversity_family_id"]


def test_the_payload_writes_a_real_bundle(realized, tmp_path):
    """The end-to-end contract: what the shard collects must satisfy every table schema.

    Duplicate join keys are the failure this catches -- an inherited endpoint pool or a shared
    source/propagated state written twice silently multiplies rows in every downstream join.
    """
    from scripts.rf_fusion_v2_artifacts import (
        V2_TABLE_SCHEMAS,
        run_manifest,
        write_v2_bundle,
    )

    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    _, _, payload = realized
    write_v2_bundle(
        tmp_path,
        manifest=run_manifest(config=_v2_config(), code_revision="deadbeef",
                              content_identities={}, seed_namespaces=("v2_lookahead",)),
        tables={name: payload[name] for name in V2_TABLE_SCHEMAS},
        ledger_events=payload["ledger_events"],
    )
    for name in V2_TABLE_SCHEMAS:
        table = pq.read_table(str(tmp_path / f"{name}.parquet"))
        assert table.num_rows == len(payload[name]), name
    assert (tmp_path / "cost_ledger.jsonl").read_text().strip()


# --------------------------------------------------------------------------------------------
# the compute ledger (PLAN §5.3-5.4)
# --------------------------------------------------------------------------------------------


def test_the_shard_reports_the_compute_ledger_it_realized(realized):
    """The journal is on disk; the payload is what the cohort aggregates.

    A shard that journaled and then dropped the events discards the record at exactly the boundary
    where it becomes an artifact, and the cohort's compute total is then a number nobody measured.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import (
        V2LedgerEvent,
        aggregate_v2_ledger,
    )

    _, _, payload = realized
    events = payload["ledger_events"]
    assert events
    assert all(isinstance(row, dict) for row in events), "fragments are JSON on disk"
    json.dumps(events)                                  # must survive the fragment round trip
    totals = aggregate_v2_ledger(tuple(V2LedgerEvent(**row) for row in events))
    assert totals["logical_dfe"] == payload["total_logical_dfe"]
    assert totals["head_calls"] > 0
    assert totals["structure_attempts"] > 0
    phases = {row["phase"] for row in events}
    assert {"root_capture", "screen", "head", "structure", "segment", "descendant_screen"} == phases


def test_a_shard_that_breached_a_declared_cap_is_not_reported_ok(tmp_path):
    """Caps are HARD (PLAN §5.1).  A breach reported as ``ok`` puts an over-budget protein into the
    cohort's success count and its designs into the pooled artifact as if they were in budget."""
    config = _v2_config()
    config = dataclasses.replace(
        config, caps=dataclasses.replace(config.caps, max_logical_dfe=1))
    status, payload = _shard(tmp_path, config=config)
    assert status == "failed"
    assert "max_logical_dfe" in payload["cap_verdict"]["breached"]
    assert payload["cap_verdict"]["detail"]


def test_a_run_inside_its_caps_reports_the_verdict_that_says_so(realized):
    """The guard above must not be satisfiable by failing every run."""
    _, status, payload = realized
    assert status == "ok"
    assert payload["cap_verdict"]["breached"] == []
    assert payload["cap_verdict"]["within"] is True


def test_the_whole_landscape_measurement_reaches_the_artifact(realized):
    """PLAN §2.7 makes ``N_H^whole`` the quantity ancestry eligibility turns on, and §5.3 requires
    terminal validation to carry the independent immune results.

    Only an ``admitted`` bool and a reason string were surviving, so the number the safety decision
    was actually made on could not be re-checked from the bundle: a reviewer could see THAT a
    design was admitted and never WHAT it measured.
    """
    _out, _status, payload = realized
    rows = payload["terminal_validation"]
    assert rows
    measured = [r for r in rows if r.get("whole_landscape_max_increase") is not None]
    assert measured, "no row carries the whole-landscape measurement"
    for row in measured:
        assert row["whole_landscape_positive_count"] is not None
        assert row["whole_landscape_n_windows"] is not None
        assert row["whole_landscape_reference_binding_id"]
