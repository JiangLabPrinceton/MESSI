"""The V2 production oracle stack (PLAN §2.5, §2.7, §5.1-5.2).

**Verification boundary, stated first.**  ``build_v2_oracles`` cannot run here: it needs torch, a
DPLM checkpoint, a Head checkpoint, a refold backend and PDB inputs.  What this suite proves is the
BINDING -- that every object handed to the cycle is tied to the run's frozen config rather than to a
local default, and that a mismatch is refused before any model is touched.  Real-oracle behaviour is
a cluster check and is not claimed here.
"""

from __future__ import annotations

import types

import pytest

from scripts.rf_fusion_v2_oracles import (
    SUPPORT_POLICY_REGISTRY,
    OracleSeams,
    V2OracleError,
    resolve_support_policy,
)
from tests.inverse_folding import _v2_fixtures as F


def _config(**over):
    from inverse_folding.reference_flow.fusion_v2.config import load_v2_config
    from tests.inverse_folding.test_fusion_v2_config import _mapping

    payload = {"projection.support_policy_id": "state_derived_probe",
               "projection.support_policy_version": "v1",
               "projection.support_policy_is_diagnostic": True,
               "identity.phase": "state_transition_canary"}
    payload.update(over)
    return load_v2_config(_mapping(**payload))


# --------------------------------------------------------------------------------------------
# the support policy is RESOLVED from the config, never substituted
# --------------------------------------------------------------------------------------------


def test_the_registry_returns_a_policy_the_KERNEL_can_actually_invoke():
    """REGRESSION. `run_one_cycle` calls `support_policy(source, endpoint, coordinates)`.

    `StateDerivedProbePolicy` declared only `decide(*, source, endpoint, coordinates)` — keyword-only
    and differently named — so it satisfied the `FeedbackSupportPolicy` Protocol and was still
    unusable by the one thing that consumes it. Every cycle-level test passes a bare function or a
    fake with `__call__`, so the divergence was invisible until a real projection on the cluster
    raised `'StateDerivedProbePolicy' object is not callable` — after the structure gate had already
    admitted an endpoint, which is the most expensive place in the cycle to find out.

    The signature is checked positionally against what the kernel passes, not merely `callable()`:
    an object with a `__call__` taking different parameter names would pass a callability check and
    fail identically at the call site.
    """
    import inspect

    from inverse_folding.reference_flow.fusion_v2_runtime import cycle as kernel

    policy = resolve_support_policy(_config(), band_table=F.band_table(), stratum_key=F.STRATUM)
    assert callable(policy), type(policy)
    assert hasattr(policy, "identity")
    parameters = list(inspect.signature(type(policy).__call__).parameters)
    assert parameters == ["self", "source", "endpoint", "coordinates"], parameters
    # And the kernel really does call it positionally -- read from the source, so a future kernel
    # that switched to `.decide(...)` makes this test wrong rather than silently vacuous.
    source = inspect.getsource(kernel.run_one_cycle)
    assert "support_policy(" in source, "the kernel no longer calls the policy directly"


def test_the_declared_policy_is_the_one_that_gets_built():
    policy = resolve_support_policy(_config(), band_table=F.band_table(), stratum_key=F.STRATUM)
    assert policy.identity().policy_id == _config().projection.support_policy_id


def test_a_policy_id_with_no_authorized_implementation_is_refused_by_name():
    """A factory that quietly substituted a policy would produce transitions the run's own
    declaration does not describe, and the artifact would name the declared one.

    The refusal lists what IS implemented, because the operator's next question is always that.
    """
    config = _config(**{"projection.support_policy_id": "source_writeback_v1",
                        "projection.support_policy_is_diagnostic": False,
                        "identity.phase": "mechanism_cohort"})
    with pytest.raises(V2OracleError, match="source_writeback_v1"):
        resolve_support_policy(config, band_table=F.band_table(), stratum_key=F.STRATUM)


def test_depth0_generated_pool_rule_binds_the_wt_only_as_bootstrap_attribution_reference():
    """The production factory must assemble the policy-v2 runtime it accepts in config.

    ``best_admissible_depth0`` does not need a second external sequence: the cycle proves which
    generated endpoint is rank zero and the ladder adopts that endpoint as I1.  The already-bound
    WT object is still needed at D0 for cumulative safety and local Head attribution, but it is not
    consulted by the global donor gate.
    """
    from inverse_folding.reference_flow.fusion_v2.reward import (
        DEPTH0_BOOTSTRAP_RULE,
        LineageIncumbentKind,
    )
    from scripts.rf_fusion_v2_oracles import bind_lineage_incumbent
    from tests.inverse_folding.test_fusion_v2_head_directed_policy import (
        _evaluator,
        _live_safety_gate,
    )

    _, reference, sequence = _live_safety_gate()
    config = types.SimpleNamespace(projection=types.SimpleNamespace(
        head_directed=types.SimpleNamespace(
            lineage_incumbent_depth0_rule=DEPTH0_BOOTSTRAP_RULE,
        )
    ))

    bound = bind_lineage_incumbent(
        config=config,
        cumulative_reference=reference,
        reference_sequence=sequence,
        evaluator=_evaluator(),
        lineage_id="5ZHV_B:fam0",
    )

    assert bound.kind is LineageIncumbentKind.CUMULATIVE_SAFETY_REFERENCE
    assert bound.sequence == sequence


def test_the_predeclared_probe_is_deliberately_absent_from_the_registry():
    """``explicit_probe`` measures only typed nulls against a realized state: its predeclared reopen
    set names positions whose resolvedness is stochastic.  Leaving it registered would let a canary
    be configured with it and burn a prefix, K lookaheads, a Head batch and K refolds per cycle for
    nothing."""
    assert "explicit_probe" not in SUPPORT_POLICY_REGISTRY()


def test_a_version_the_run_did_not_declare_is_caught_before_the_prefix_is_paid():
    """The kernel would refuse every transition on this, but only after the ladder had bought a
    root prefix of real forward passes."""
    config = _config(**{"projection.support_policy_version": "v9"})
    with pytest.raises(V2OracleError, match="version"):
        resolve_support_policy(config, band_table=F.band_table(), stratum_key=F.STRATUM)


# --------------------------------------------------------------------------------------------
# the GPU instrument
# --------------------------------------------------------------------------------------------


def test_a_cuda_process_with_no_gpu_clock_is_refused_rather_than_credited_with_zero():
    """``check_caps`` treats GPU-seconds as a MEASURED quantity, so a fabricated ``0.0`` would let
    ``max_gpu_seconds`` read as satisfied on a number nobody took."""
    from scripts import rf_fusion_v2_oracles as mod

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True, is_initialized=lambda: True))
    import sys

    original = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        with pytest.raises(V2OracleError, match="gpu_clock"):
            mod._default_gpu_clock()
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original


def test_the_factory_names_a_gpu_instrument_that_a_cuda_process_can_actually_use():
    """REGRESSION. `OracleSeams.resolved()` handed back the REFUSAL sentinel.

    The refusal is right in itself, but the driver injects no seams, so the production stack had no
    instrument at all: every cluster launch reached `CostMeter`, called the clock on the first
    attempt, and died with "no GPU clock was supplied" — after the checkpoints were resident. A
    guard that no production path can satisfy does not protect the measurement; it prevents it.
    """
    import sys

    from scripts import rf_fusion_v2_oracles as mod

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True, is_initialized=lambda: True, device_count=lambda: 1))
    original = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        clock = OracleSeams().resolved()["gpu_clock"]
        first, second = clock(), clock()
        assert second >= first >= 0.0, (first, second)
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original


def test_the_gpu_clock_scales_with_the_devices_the_allocation_holds():
    """Reserved seconds, not busy seconds: two reserved devices burn two GPU-seconds per second."""
    import sys

    from scripts import rf_fusion_v2_oracles as mod

    original = sys.modules.get("torch")
    try:
        sys.modules["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_available=lambda: False, is_initialized=lambda: False, device_count=lambda: 0))
        assert mod.reserved_gpu_seconds_clock()() == 0.0

        sys.modules["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_available=lambda: True, is_initialized=lambda: True, device_count=lambda: 2))
        one, two = mod.reserved_gpu_seconds_clock(), mod.reserved_gpu_seconds_clock()
        del one
        assert two() > 0.0
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original


def test_a_process_that_never_touched_cuda_truthfully_reports_zero():
    """The guard must not be satisfiable by refusing every run: a CPU process really does burn no
    GPU seconds, and saying so is a measurement rather than a default."""
    from scripts import rf_fusion_v2_oracles as mod

    import sys

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: False, is_initialized=lambda: False))
    original = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        assert mod._default_gpu_clock() == 0.0
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original


# --------------------------------------------------------------------------------------------
# nothing here loads a model
# --------------------------------------------------------------------------------------------


def test_importing_the_oracle_stack_loads_no_torch():
    """``--print-config`` and ``--dry-run`` must not pull in a model (PLAN V2F7 acceptance), and the
    driver imports this module's package path on the execution branch only."""
    import pathlib
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import scripts.rf_fusion_v2_oracles as m; m.OracleSeams(); "
         "print('torch' in sys.modules)"],
        capture_output=True, text=True, cwd=str(pathlib.Path(__file__).parents[2]),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


# --------------------------------------------------------------------------------------------
# content identity: declared where frozen, OBSERVED where runtime-bound, never absent
# --------------------------------------------------------------------------------------------


class _Model:
    tokenizer_digest = "a" * 64
    fixed_token_policy = "unconstrained"

    def coordinate_mask_digest(self, protein_id):
        return "b" * 64


def _runtime_inputs(config, tmp_path):
    from scripts.rf_fusion_v2_cohort import ShardInputs

    paths = {}
    for row in config.content:
        if row.expected_sha256 is None:
            path = tmp_path / f"{row.role}.bin"
            path.write_text(row.role)
            paths[row.role] = str(path)
    return ShardInputs(**paths)


def test_a_runtime_bound_role_is_signed_by_its_OBSERVED_bytes(tmp_path):
    """PLAN §5.2 binds file CONTENTS, not paths.  A runtime-bound role carries no declared digest
    by construction, so the only honest identity for it is the one computed from what was actually
    supplied."""
    from scripts import rf_fusion_v2_oracles as mod

    config = _config()
    inputs = _runtime_inputs(config, tmp_path)
    first = mod._conditioning(config, model=_Model(), protein_id="5ZHV_B", inputs=inputs,
                              band_table=F.band_table())

    role = next(r.role for r in config.content if r.expected_sha256 is None)
    (tmp_path / f"{role}.bin").write_text("different bytes entirely")
    second = mod._conditioning(config, model=_Model(), protein_id="5ZHV_B", inputs=inputs,
                               band_table=F.band_table())
    assert first.digest() != second.digest(), (
        f"changing the {role!r} file did not change the run's conditioning identity")


def test_a_role_that_is_neither_frozen_nor_supplied_fails_closed(tmp_path):
    """"Missing content identity fails closed" (PLAN §5.2).  Signing a run whose inputs nobody can
    name is the one outcome that must be impossible."""
    from scripts import rf_fusion_v2_oracles as mod
    from scripts.rf_fusion_v2_cohort import ShardInputs

    with pytest.raises(V2OracleError, match="content identity"):
        mod._conditioning(_config(), model=_Model(), protein_id="5ZHV_B",
                          inputs=ShardInputs(), band_table=F.band_table())


def test_the_per_protein_coordinate_mask_overrides_any_file_level_digest(tmp_path):
    """The mask is a property of the PREPARED backbone, not of a file: two proteins under one
    coordinate-mask policy have different masks, and the conditioning must say so."""
    from scripts import rf_fusion_v2_oracles as mod

    config = _config()
    inputs = _runtime_inputs(config, tmp_path)

    class _Other(_Model):
        def coordinate_mask_digest(self, protein_id):
            return "c" * 64

    assert mod._conditioning(config, model=_Model(), protein_id="5ZHV_B",
                             inputs=inputs, band_table=F.band_table()).digest() != \
        mod._conditioning(config, model=_Other(), protein_id="5ZHV_B", inputs=inputs, band_table=F.band_table()).digest()


# --------------------------------------------------------------------------------------------
# the depth-0 reference is PER PROTEIN
# --------------------------------------------------------------------------------------------


def test_the_reference_manifest_resolves_a_sequence_per_protein(tmp_path):
    """PLAN §2.7 binds "a predeclared complete reference sequence" per lineage.

    One file for the whole cohort binds every protein's whole-landscape hotspot comparator to the
    same sequence: at different lengths the Head binding simply fails, and at equal lengths -- the
    dangerous case -- it succeeds while measuring each design against another protein's native.
    """
    from scripts.rf_fusion_v2_oracles import resolve_reference

    import hashlib
    import json

    def _write(name, sequence):
        (tmp_path / name).write_text(sequence)
        return {"path": name,
                "sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest()}

    manifest = tmp_path / "references.json"
    manifest.write_text(json.dumps({"5ZHV_B": _write("a.seq", "ACDEFG"),
                                    "9L2Q_A": _write("b.seq", "WYWYWYWY")}))
    assert resolve_reference(manifest, "5ZHV_B")[0] == "ACDEFG"
    assert resolve_reference(manifest, "9L2Q_A")[0] == "WYWYWYWY"


def test_a_reference_whose_bytes_do_not_match_the_manifest_is_refused(tmp_path):
    """The per-protein digest is what binds the comparator to its molecule; a file edited after the
    manifest was written must not be silently accepted."""
    import hashlib
    import json

    from scripts.rf_fusion_v2_oracles import resolve_reference

    (tmp_path / "a.seq").write_text("ACDEFG")
    manifest = tmp_path / "references.json"
    manifest.write_text(json.dumps({"5ZHV_B": {
        "path": "a.seq", "sha256": hashlib.sha256(b"ACDEFG").hexdigest()}}))
    assert resolve_reference(manifest, "5ZHV_B")[0] == "ACDEFG"
    (tmp_path / "a.seq").write_text("WWWWWW")
    with pytest.raises(V2OracleError, match="hashes to"):
        resolve_reference(manifest, "5ZHV_B")


def test_a_protein_absent_from_the_reference_manifest_fails_closed(tmp_path):
    """PLAN §5.2: missing content identity fails closed.  Falling back to any other protein's
    native would anchor the safety ratchet to the wrong molecule for the whole lineage."""
    from scripts.rf_fusion_v2_oracles import resolve_reference

    manifest = tmp_path / "references.json"
    import hashlib
    import json

    manifest.write_text(json.dumps({"5ZHV_B": {
        "path": "a.seq", "sha256": hashlib.sha256(b"ACDEFG").hexdigest()}}))
    (tmp_path / "a.seq").write_text("ACDEFG")
    with pytest.raises(V2OracleError, match="9L2Q_A"):
        resolve_reference(manifest, "9L2Q_A")


def test_a_single_sequence_file_is_refused_as_a_cohort_reference(tmp_path):
    """A bare sequence file is exactly the shape that silently shares one reference across the
    cohort, so it is refused by TYPE rather than accepted for a one-protein run and then reused."""
    from scripts.rf_fusion_v2_oracles import resolve_reference

    bare = tmp_path / "reference.seq"
    bare.write_text("ACDEFG")
    with pytest.raises(V2OracleError, match="manifest"):
        resolve_reference(bare, "5ZHV_B")


# --------------------------------------------------------------------------------------------
# the factory reads attributes that PreparedModel actually carries
# --------------------------------------------------------------------------------------------


def test_every_model_attribute_the_factory_reads_exists_on_prepared_model():
    """REGRESSION, twice over. `build_v2_oracles` passed `alphabet=model.alphabet`.

    The cycle's `alphabet` parameter is a `Mapping[int, str]` residue map and `PreparedModel` calls
    it `id_to_aa`, so the attribute simply does not exist -- but every fake in this suite defines
    whatever the production code asks for, which is precisely why no test saw it. The identical
    rename had already cost the hotspot calibrator one cluster allocation; it cost the Canary a
    second one.

    Read off the SOURCE rather than from a hand-maintained list: a list is a third place to forget.
    """
    import ast
    import inspect

    from scripts import rf_fusion_v2_oracles as mod
    from scripts.rf_fusion_model_factory import PreparedModel

    tree = ast.parse(inspect.getsource(mod.build_v2_oracles))
    read = sorted({node.attr for node in ast.walk(tree)
                   if isinstance(node, ast.Attribute)
                   and isinstance(node.value, ast.Name) and node.value.id == "model"})
    declared = set(dir(PreparedModel)) | set(getattr(PreparedModel, "__annotations__", {}))
    missing = [name for name in read if name not in declared]
    assert not missing, (
        f"build_v2_oracles reads model.{missing}, which PreparedModel does not carry; "
        f"it reads {read}")


# --------------------------------------------------------------------------------------------
# the constraint policy the conditioning records is the one that was ENFORCED
# --------------------------------------------------------------------------------------------


def test_an_anchored_run_does_not_record_an_unconstrained_fixed_token_policy(tmp_path):
    """``build_model_factory`` defaults ``fixed_token_policy`` to the literal ``"unconstrained"``.

    V2 was never passing it, so an anchored protein ran with its 24 hard anchors enforced while
    every artifact's conditioning row named an unconstrained policy -- and the two Canary strata,
    which differ in exactly this, declared the same one.
    """
    import hashlib

    from scripts.rf_fusion_v2_cohort import ShardInputs
    from scripts.rf_fusion_v2_oracles import fixed_token_policy_label

    manifest = tmp_path / "anchors.yaml"
    manifest.write_text("entries: []\n")
    expected = hashlib.sha256(manifest.read_bytes()).hexdigest()

    assert fixed_token_policy_label(ShardInputs()) == "unconstrained"
    assert fixed_token_policy_label(
        ShardInputs(constraint_manifest=str(manifest))) == f"manifest:{expected}"


def test_the_factory_hands_the_realized_constraint_policy_to_the_model(tmp_path):
    """The label must reach ``build_model_factory``, not merely be computable.

    ``_conditioning`` reads ``model.fixed_token_policy`` and overrides whatever a file-level digest
    said, so a label that never reaches the model is a label that never reaches the artifact.
    """
    from scripts import rf_fusion_v2_oracles as mod

    manifest = tmp_path / "anchors.yaml"
    manifest.write_text("entries: []\n")
    seen = {}

    def _factory(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after the model call")

    inputs = _runtime_inputs(_config(), tmp_path)
    inputs.paths.update({
        "base_if_checkpoint": "ckpt", "rf_sampler_config": "rf.yaml",
        "test_set_parquet": "t.parquet", "pdb_root": str(tmp_path),
        "protein_stratum_manifest": str(tmp_path / "strata.json"),
        "constraint_manifest": str(manifest),
        "schedule_band_calibration": str(tmp_path / "B_r.json"),
    })
    (tmp_path / "strata.json").write_text('{"5ZHV_B": "%s"}' % F.STRATUM)

    with pytest.raises(RuntimeError, match="stop after the model call"):
        mod.build_v2_oracles(
            protein_id="5ZHV_B", config=_config(), inputs=inputs,
            seams=OracleSeams(build_model_factory=_factory,
                              load_band_table=lambda *a, **k: F.band_table()))
    assert seen["fixed_token_policy"].startswith("manifest:"), seen.get("fixed_token_policy")


# --------------------------------------------------------------------------------------------
# the band's two digests are two different quantities
# --------------------------------------------------------------------------------------------


def test_the_conditioning_carries_the_TABLE_digest_not_the_files_sha256(tmp_path):
    """``make_band_table`` REBINDS ``calibration_content_digest`` to a canonical digest of the
    table's own content, and that is what ``declared_band_digest`` carries and what the kernel
    compares against ``conditioning.schedule_band_calibration``.

    The config role's ``expected_sha256`` is the FILE's sha256 (PLAN §5.2 signs file CONTENTS, and
    that is what the driver computes for a declared input).  Two different numbers for one field:
    left as it was, every real run would be refused by the kernel's own provenance check.

    Resolved exactly as ``coordinate_mask`` already is -- an intrinsic identity the file digest
    cannot express overrides the role digest in the conditioning, while the role digest keeps
    signing the file.
    """
    from scripts import rf_fusion_v2_oracles as mod

    config = _config()
    inputs = _runtime_inputs(config, tmp_path)
    table = F.band_table()
    conditioning = mod._conditioning(
        config, model=_Model(), protein_id="5ZHV_B", inputs=inputs, band_table=table)
    assert conditioning.schedule_band_calibration == \
        table.provenance.calibration_content_digest


def test_the_band_digest_in_the_conditioning_tracks_the_band_content(tmp_path):
    """Two calibrations under one name must not share a conditioning identity."""
    from scripts import rf_fusion_v2_oracles as mod

    config = _config()
    inputs = _runtime_inputs(config, tmp_path)
    other = F.band_table(bands_override=(F.band(
        unresolved_accept=__import__(
            "inverse_folding.reference_flow.fusion_v2.schedule",
            fromlist=["BandInterval"]).BandInterval(lo=1.0, hi=3.0, lo_level=0.9, hi_level=0.1)),))
    assert mod._conditioning(config, model=_Model(), protein_id="5ZHV_B", inputs=inputs,
                             band_table=F.band_table()).digest() != \
        mod._conditioning(config, model=_Model(), protein_id="5ZHV_B", inputs=inputs,
                          band_table=other).digest()


# --------------------------------------------------------------------------------------------
# the band stratum is the PROTEIN's, not the schedule point's
# --------------------------------------------------------------------------------------------


def test_the_stratum_comes_from_an_explicit_per_protein_manifest(tmp_path):
    """``source_writeback``'s own docstring: ``stratum_key`` "is deliberately NOT derived from
    ``DepthSchedulePoint.band_key`` -- the Interface Map states they are different keys".

    ``band_key`` names a schedule cell; ``stratum_key`` names the COHORT STRATUM the band was
    measured on.  Using one for the other gives every protein in a cohort the same stratum, so an
    anchored protein would read its reopen cardinality off a band measured on unconstrained ones.
    """
    import json

    from scripts.rf_fusion_v2_oracles import resolve_stratum

    manifest = tmp_path / "strata.json"
    manifest.write_text(json.dumps({"5ZHV_B": "len4_8_unconstrained",
                                    "9L2Q_A": "len4_8_anchored"}))
    assert resolve_stratum(manifest, "5ZHV_B") == "len4_8_unconstrained"
    assert resolve_stratum(manifest, "9L2Q_A") == "len4_8_anchored"


def test_a_protein_with_no_declared_stratum_fails_closed(tmp_path):
    """Defaulting to any other protein's stratum is how an anchored protein silently reads an
    unconstrained band."""
    import json

    from scripts.rf_fusion_v2_oracles import resolve_stratum

    manifest = tmp_path / "strata.json"
    manifest.write_text(json.dumps({"5ZHV_B": "len4_8_unconstrained"}))
    with pytest.raises(V2OracleError, match="9L2Q_A"):
        resolve_stratum(manifest, "9L2Q_A")


def test_an_anchored_protein_may_not_run_on_an_unconstrained_band():
    """The declared stratum is a LABEL; the constraint class is a FACT about the protein.

    Hard anchors remove positions from the editable domain, so at equal length an anchored protein
    carries different unresolved mass at the same step.  A band measured on an unconstrained cohort
    does not describe it, and the reopen cardinality read off that band is pinned from the wrong
    distribution -- which the artifact would record as a legitimate projection.
    """
    from scripts.rf_fusion_v2_oracles import assert_constraint_class_matches_band

    table = F.band_table()          # provenance declares constraint_stratum_id="anchored"
    assert table.provenance.constraint_stratum_id == "anchored"
    with pytest.raises(V2OracleError, match="unconstrained"):
        assert_constraint_class_matches_band(band_table=table, fixed_tokens=None,
                                             protein_id="5ZHV_B")


def test_an_anchored_protein_on_an_anchored_band_is_accepted():
    """The guard must not be satisfiable by refusing every protein."""
    from scripts.rf_fusion_v2_oracles import assert_constraint_class_matches_band

    assert assert_constraint_class_matches_band(
        band_table=F.band_table(), fixed_tokens={0: 10}, protein_id="5ZHV_B") is None


# --------------------------------------------------------------------------------------------
# the PRODUCTION Head/structure pair: the declared paths must build real oracles
# --------------------------------------------------------------------------------------------


class _Scorer:
    """An ``OnlineHeadScorer``-shaped stand-in: the fields ``ProductionHeadOracle`` reads."""

    allele, score_scale = "DRB1_0701", "raw_logit"
    window_k_min, window_k_max = 12, 25
    head_config_hash, head_checkpoint_digest = "a" * 64, "b" * 64

    def __init__(self, md5_override=None):
        self._md5_override = md5_override

    def score_batch_same_protein(self, *, protein_id, records):
        from inverse_folding.reference_flow.fusion.state import sequence_md5

        scores = tuple(
            types.SimpleNamespace(
                protein_id=protein_id,
                sequence_md5=self._md5_override or sequence_md5(sequence),
                sequence_length=len(sequence), allele=self.allele, score_scale=self.score_scale,
                windows=(types.SimpleNamespace(start_0b=0, end_0b=13, k=13, z=1.0),),
                residue_hotspot=(0.0,) * len(sequence), global_risk=-1.0)
            for _label, sequence in records
        )
        return types.SimpleNamespace(scores=scores)


def _oracle(scorer=None, **over):
    from scripts.rf_fusion_v2_oracles import ProductionHeadOracle

    kw = dict(allele="DRB1_0701", score_scale="raw_logit", window_k_min=12, window_k_max=25)
    kw.update(over)
    return ProductionHeadOracle(scorer or _Scorer(), **kw)


def _request(protein_id="5ZHV_B", sequence="ACDEFGHIKLMNPQRSTVWY"):
    from inverse_folding.reference_flow.fusion.state import sequence_md5
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest

    return OracleRequest(protein_id=protein_id, sequence=sequence,
                         sequence_md5=sequence_md5(sequence), sequence_length=len(sequence))


def test_the_head_identity_is_read_off_the_scorer_that_actually_scored():
    identity = _oracle().evaluator_identity()
    assert (identity.allele, identity.score_scale) == ("DRB1_0701", "raw_logit")
    assert (identity.window_k_min, identity.window_k_max) == (12, 25)
    assert identity.head_config_hash == "a" * 64
    assert identity.head_checkpoint_digest == "b" * 64


def test_a_scorer_whose_domain_differs_from_the_declared_one_is_refused_by_field():
    """``bind_admission_policy`` compares this exact 4-tuple and raises with two opaque tuples --
    after the checkpoints are resident.  Refusing here names the field instead."""
    with pytest.raises(V2OracleError, match="score_scale: declared 'nats' != realized 'raw_logit'"):
        _oracle(score_scale="nats")
    # 13 against the Head's real 12: this check compares the run's DECLARED domain to the scorer's
    # recorded one.  It is not the check that catches a domain the Head cannot emit -- the scorer
    # records whatever it was constructed with, so that one has to compare against the windows
    # actually returned (`calibrate_rf_fusion_v2_hotspot._assert_declared_window_domain`).
    with pytest.raises(V2OracleError, match="window_k_min"):
        _oracle(window_k_min=13)


def test_each_result_carries_the_binding_the_runtime_matches_results_by():
    from inverse_folding.reference_flow.fusion_v2.identity import HeadScoreBinding

    result = _oracle().score([_request()])[0]
    assert isinstance(result.binding, HeadScoreBinding)
    assert result.binding.sequence_md5 == result.sequence_md5
    assert result.binding.evaluator == _oracle().evaluator_identity()


def test_identical_sequences_are_scored_once_not_twice():
    """``bind_head_scores`` refuses a duplicate result for one digest: two forks that produced the
    same bytes are one measurement."""
    assert len(_oracle().score([_request(), _request()])) == 1


def test_a_result_keyed_to_another_sequence_is_a_hard_failure():
    """The digest is the archive/cache/ledger join key; a mismatched one attaches a Head result to
    a sequence it was never computed for."""
    with pytest.raises(V2OracleError, match="join key"):
        _oracle(scorer=_Scorer(md5_override="f" * 32)).score([_request()])


def test_the_head_result_is_copied_verbatim_never_rescored():
    result = _oracle().score([_request()])[0]
    assert result.global_risk == -1.0 and result.score_scale == "raw_logit"
    assert len(result.windows) == 1 and result.windows[0].z == 1.0


# --------------------------------------------------------------------------------------------
# the production STRUCTURE oracle: v0's struct_fn composed with v0's own feasibility contract
# --------------------------------------------------------------------------------------------

V0_STRUCTURE_CONFIG = "inverse_folding/reference_flow/configs/rf_refine_fusion_final_repair_beam.yaml"


class _Manifest:
    def __init__(self, anchored=()):
        self._anchored = set(anchored)

    def has_protein(self, protein_id):
        return protein_id in self._anchored


def _fake_build_oracles(*, metrics=None, raises=None, manifest=None, seen=None):
    """Stands in for ``run_rf_refine_fusion.build_oracles``: returns ``(FusionOracles, manifest)``."""
    from inverse_folding.reference_flow.fusion.oracles import FusionOracles

    def build(args, config):
        if seen is not None:
            seen.append((args, config))

        def struct_fn(protein_id, sequence):
            if raises is not None:
                raise raises
            return metrics

        return FusionOracles(head_fn=(lambda *_a, **_k: []), struct_fn=struct_fn), manifest

    return build


def _metrics(**over):
    from inverse_folding.reference_flow.refine import StructureMetrics

    kw = dict(scTM=0.95, pLDDT=88.0, scRMSD=1.2)
    kw.update(over)
    return StructureMetrics(**kw)


def _structure_oracle(**over):
    """Only the structure half is exercised; the Head half needs a real scorer in the closure."""
    from scripts.rf_fusion_v2_oracles import build_production_oracles

    kw = dict(structure_config=V0_STRUCTURE_CONFIG, head_config_dir="/nx/cfg",
              head_checkpoint="/nx/head.pt", test_set_parquet="/nx/t.parquet",
              pdb_root="/nx/pdbs", refold_cache_dir="/nx/refold", allele="DRB1_0701",
              score_scale="raw_logit", window_k_min=12, window_k_max=25, head_variant_id="LC1")
    kw.update(over)
    with pytest.raises(V2OracleError, match="closure"):
        # The fake head_fn closes over no scorer, so the Head half refuses -- which is itself the
        # guarantee that a scorer can never be silently absent.  Build the structure half directly.
        build_production_oracles(**kw)


def _structure_only(build_oracles, **over):
    """Build just the structure adapter by giving the Head half a recoverable scorer."""
    from scripts.rf_fusion_v2_oracles import build_production_oracles

    scorer = _Scorer()

    def wrapped(args, config):
        oracles, manifest = build_oracles(args, config)
        head_fn = (lambda *_a, **_k: scorer)          # noqa: ARG005 - closes over the scorer
        return types.SimpleNamespace(head_fn=head_fn, struct_fn=oracles.struct_fn), manifest

    kw = dict(structure_config=V0_STRUCTURE_CONFIG, head_config_dir="/nx/cfg",
              head_checkpoint="/nx/head.pt", test_set_parquet="/nx/t.parquet",
              pdb_root="/nx/pdbs", refold_cache_dir="/nx/refold", allele="DRB1_0701",
              score_scale="raw_logit", window_k_min=12, window_k_max=25, head_variant_id="LC1",
              build_oracles=wrapped)
    kw.update(over)
    return build_production_oracles(**kw)[1]


def test_a_passing_structure_is_an_EVALUATED_feasible_verdict():
    gate = _structure_only(_fake_build_oracles(metrics=_metrics(), manifest=_Manifest()))
    outcome = gate(_request())
    assert outcome.evaluated is True and outcome.feasible is True
    assert outcome.metrics["scTM"] == pytest.approx(0.95)


@pytest.mark.parametrize(
    ("cache_hit", "expected_status", "expected_executed"),
    [(True, "hit", False), (False, "miss", True)],
)
def test_structure_oracle_preserves_real_refold_cache_execution_status(
    cache_hit, expected_status, expected_executed,
):
    gate = _structure_only(
        _fake_build_oracles(
            metrics=_metrics(cache_hit=cache_hit, model_executed=not cache_hit),
            manifest=_Manifest(),
        )
    )
    outcome = gate(_request())
    assert outcome.cache_status == expected_status
    assert outcome.model_executed is expected_executed


def test_a_failing_scTM_is_refused_with_the_gates_own_reason():
    gate = _structure_only(_fake_build_oracles(metrics=_metrics(scTM=0.10),
                                               manifest=_Manifest()))
    outcome = gate(_request())
    assert outcome.evaluated is True and outcome.feasible is False
    assert "scTM" in outcome.failure_reason


def test_a_fold_failure_is_evaluated_and_infeasible_never_deferred():
    """The 48/64 calibration floor and Canary admission both read `evaluated AND feasible`; a
    deferred outcome would let an endpoint nothing folded count toward either."""
    gate = _structure_only(_fake_build_oracles(raises=RuntimeError("refold died"),
                                               manifest=_Manifest()))
    outcome = gate(_request())
    assert outcome.evaluated is True and outcome.feasible is False
    assert "RuntimeError" in outcome.failure_reason and "refold died" in outcome.failure_reason


def test_the_active_site_branch_fires_for_exactly_the_proteins_v0_constrains():
    """Anchored/unconstrained is read off the manifest v0 itself resolved, not a caller flag: an
    anchored protein whose active-site metrics are missing must FAIL, not pass on scTM alone."""
    metrics = _metrics()                    # carries no active-site geometry
    unconstrained = _structure_only(
        _fake_build_oracles(metrics=metrics, manifest=_Manifest()))
    anchored = _structure_only(
        _fake_build_oracles(metrics=metrics, manifest=_Manifest(anchored={"5ZHV_B"})))
    assert unconstrained(_request()).feasible is True
    assert anchored(_request()).feasible is False


def test_unset_refold_knobs_take_v0s_own_defaults_not_none():
    """REGRESSION.  These defaulted to `None`, which would have folded the V2 calibration and the
    Canary under a protocol v0 never used -- while the artifact still claimed the v0 definitive
    contract.  They are READ off v0's parser, so a change to v0's protocol propagates here instead
    of leaving two literals to drift."""
    from scripts.rf_fusion_v2_oracles import _v0_oracle_args, v0_oracle_arg_defaults

    args = _v0_oracle_args()
    for name in ("esmfold2_model", "esmfold2_num_loops", "esmfold2_num_sampling_steps",
                 "esmfold2_num_diffusion_samples", "esmfold2_seed"):
        assert getattr(args, name) is not None, f"{name} would reach load_refold_model as None"
        assert getattr(args, name) == v0_oracle_arg_defaults()[name]
    # ...and v0's declared values, so a silent protocol change is visible in the diff.
    assert args.esmfold2_model == "biohub/ESMFold2"
    assert (args.esmfold2_num_loops, args.esmfold2_num_sampling_steps) == (3, 50)
    assert (args.esmfold2_num_diffusion_samples, args.esmfold2_seed) == (1, 0)
    assert (args.head_allele_idx, args.head_window_batch_size) == (0, 64)


def test_an_explicit_knob_still_overrides_v0s_default():
    from scripts.rf_fusion_v2_oracles import _v0_oracle_args

    assert _v0_oracle_args(esmfold2_seed=7).esmfold2_seed == 7


def test_a_misspelled_oracle_argument_is_refused_not_ignored():
    from scripts.rf_fusion_v2_oracles import _v0_oracle_args

    with pytest.raises(V2OracleError, match="unknown v0 oracle argument"):
        _v0_oracle_args(esmfold_seed=7)
