"""V2F7: the V2 driver's exit-code matrix and its no-model config paths.

PLAN V2F7 acceptance names the cases directly: "fake full-driver run, crash after paid work, retry,
shard reorder, empty/partial cohort, stale input/config, duplicate fragments, and zero-success cases
all produce correct non-zero exits or exact resume behavior.  ``--print-config`` and ``--dry-run``
load no model and derive one shared config digest."

The exit code is the entire interface a cohort runner sees.  Every test below is really one
question: can a run that produced nothing usable be mistaken for one that worked?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

import yaml  # noqa: E402

from scripts.run_rf_fusion_v2 import (  # noqa: E402
    DeclaredInput,
    EXIT_FAILED,
    EXIT_OK,
    EXIT_PARTIAL,
    EXIT_UNLAUNCHABLE,
    V2DriverError,
    build_parser,
    execution_replicates,
    main,
)
from scripts.rf_fusion_v2_preflight import input_signature  # noqa: E402
from scripts.rf_fusion_v2_resume import RunSignature  # noqa: E402
from tests.inverse_folding import _v2_fixtures as F  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _config_mapping():
    """A complete, legal V2 config as a plain mapping.

    REUSED from the config suite rather than re-written here: a second definition of "a legal V2
    config" would drift from the loader's actual requirements, and this file would then be testing
    the driver against a config shape nothing else accepts.
    """
    from tests.inverse_folding.test_fusion_v2_config import _mapping

    return _mapping()


def _write_config(tmp_path, **over):
    payload = _config_mapping()
    for dotted, value in over.items():
        node = payload
        *path, leaf = dotted.split(".")
        for key in path:
            node = node[key]
        node[leaf] = value
    path = tmp_path / "v2.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return path


def _input_file(tmp_path, name="cohort.tsv", text="protein_id\nA_1\nB_2\n"):
    """A declared input file.  Content is stable across calls so a resume stays valid."""
    path = tmp_path / name
    path.write_text(text)
    return path


def _write_config_binding(tmp_path, role, digest):
    """A config whose frozen provenance row for ``role`` declares ``digest``."""
    payload = _config_mapping()
    for row in payload["content"]:
        if row["role"] == role:
            row["expected_sha256"] = digest
    path = tmp_path / "v2.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return path


def _fake_runner(status="ok", rows=1, raises=None):
    calls = []

    def runner(*, protein_id, config, signature, out_dir, inputs=None, oracles_factory=None):
        calls.append(protein_id)
        if raises is not None:
            raise raises
        return status, {
            "complete_endpoints": [{"endpoint_id": f"{protein_id}:{i}"} for i in range(rows)],
        }

    runner.calls = calls
    return runner


def _run(tmp_path, *, cohort=("A_1",), runner=None, extra=(), config_path=None, inputs=None):
    config_path = config_path or _write_config(tmp_path)
    if inputs is None:
        inputs = [f"cohort_table={_input_file(tmp_path)}"]
    argv = ["--v2-config", str(config_path), "--out-dir", str(tmp_path / "out"),
            "--cohort", *cohort, "--code-revision", "deadbeef",
            *(["--input-file", *inputs] if inputs else []), *extra]
    return main(argv, runner=runner if runner is not None else _fake_runner())


# --------------------------------------------------------------------------------------------
# --print-config / --dry-run load no model
# --------------------------------------------------------------------------------------------


def test_print_config_derives_the_one_shared_digest_and_loads_no_model(tmp_path):
    """Three code paths computing "the digest" separately is how a run reports a configuration it
    did not use."""
    config_path = _write_config(tmp_path)
    result = subprocess.run(
        [sys.executable, "scripts/run_rf_fusion_v2.py", "--v2-config", str(config_path),
         "--out-dir", str(tmp_path / "out"), "--cohort", "A_1", "--print-config"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == EXIT_OK, result.stderr
    payload = json.loads(result.stdout)

    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    assert payload["config_digest"] == load_v2_config_file(config_path).config_digest()
    assert payload["budget_projection"]["total_logical_dfe"] > 0


def test_no_torch_is_imported_by_the_config_paths(tmp_path):
    """On a cluster, a preflight that imported torch would burn a GPU allocation to find a typo."""
    config_path = _write_config(tmp_path)
    probe = (
        "import sys, runpy;\n"
        "sys.argv = ['run_rf_fusion_v2.py', '--v2-config', %r, '--out-dir', %r,"
        " '--cohort', 'A_1', '--input-file', %r, '--dry-run'];\n"
        "import contextlib, io;\n"
        "buf = io.StringIO();\n"
        "code = 0\n"
        "try:\n"
        "    with contextlib.redirect_stdout(buf):\n"
        "        runpy.run_path('scripts/run_rf_fusion_v2.py', run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    code = exc.code\n"
        "print('torch' in sys.modules)\n"
    ) % (str(config_path), str(tmp_path / "out"),
         f"cohort_table={_input_file(tmp_path)}")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            cwd=str(REPO))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_dry_run_refuses_a_schedule_that_cannot_fit_its_caps(tmp_path):
    """A cap discovered mid-run has already burned the budget it was supposed to bound."""
    config_path = _write_config(tmp_path, **{"caps.max_logical_dfe": 1})
    assert _run(tmp_path, config_path=config_path, extra=["--dry-run"]) == EXIT_UNLAUNCHABLE


def test_qualification_budget_counts_both_arms_of_every_prefix():
    args = build_parser().parse_args([
        "--v2-config", "v2.yaml", "--out-dir", "out",
        "--mechanism-prefixes", "56", "--qualification",
    ])
    assert execution_replicates(args) == 112


def test_qualification_without_a_prefix_is_refused_before_preflight():
    args = build_parser().parse_args([
        "--v2-config", "v2.yaml", "--out-dir", "out", "--qualification",
    ])
    with pytest.raises(V2DriverError, match="requires --mechanism-prefixes"):
        execution_replicates(args)


def test_exploratory_depth_override_requires_capability_phase_and_split(tmp_path):
    bad_phase = _write_config(
        tmp_path,
        **{
            "identity.split_role": "exploratory_deep_smoke",
        },
    )
    assert _run(
        tmp_path,
        config_path=bad_phase,
        extra=["--exploratory-depth-override", "--dry-run"],
    ) == EXIT_UNLAUNCHABLE

    bad_split = _write_config(
        tmp_path,
        **{
            "identity.phase": "capability_ladder",
            "projection.support_policy_id": "source_writeback_v1",
            "projection.support_policy_is_diagnostic": False,
        },
    )
    assert _run(
        tmp_path,
        config_path=bad_split,
        extra=["--exploratory-depth-override", "--dry-run"],
    ) == EXIT_UNLAUNCHABLE


def test_exploratory_depth_override_reaches_runner_and_manifest(tmp_path):
    config_path = _write_config(
        tmp_path,
        **{
            "identity.phase": "capability_ladder",
            "identity.split_role": "exploratory_deep_smoke",
            "projection.support_policy_id": "source_writeback_v1",
            "projection.support_policy_is_diagnostic": False,
        },
    )
    observed = []

    def runner(*, protein_id, config, signature, out_dir, inputs=None,
               oracles_factory=None, exploratory_depth_override=False,
               production_depth_authorized=False):
        observed.append((signature, exploratory_depth_override,
                         production_depth_authorized))
        return "ok", {
            "complete_endpoints": [{"endpoint_id": f"{protein_id}:0"}],
            "production_depth_authorized": production_depth_authorized,
            "exploratory_depth_override": exploratory_depth_override,
        }

    assert _run(
        tmp_path,
        config_path=config_path,
        runner=runner,
        extra=["--exploratory-depth-override"],
    ) == EXIT_OK
    signature, exploratory, production = observed[0]
    assert exploratory is True
    assert production is False
    assert signature.exploratory_depth_override is True
    assert signature.production_depth_authorized is False

    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["exploratory_depth_override"] is True
    assert manifest["production_depth_authorized"] is False


def test_a_malformed_config_is_refused_before_anything_runs(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"schema_version": "nope"}))
    runner = _fake_runner()
    assert _run(tmp_path, config_path=path, runner=runner) == EXIT_UNLAUNCHABLE
    assert runner.calls == [], "the runner ran despite an unusable config"


# --------------------------------------------------------------------------------------------
# the exit-code matrix
# --------------------------------------------------------------------------------------------


def test_a_complete_successful_cohort_exits_zero(tmp_path):
    assert _run(tmp_path, cohort=("A_1", "B_2")) == EXIT_OK


def test_a_cohort_where_every_protein_failed_does_not_exit_zero(tmp_path):
    """PLAN §7.2 names "all proteins fail but the driver exits zero" as an adversarial case.

    Every protein WAS processed and every fragment is valid -- so the run is "complete" in the
    resume sense.  Exiting zero here would let a cohort of total failures be consumed downstream as
    a finished run.
    """
    assert _run(tmp_path, cohort=("A_1", "B_2"),
                runner=_fake_runner(status="failed", rows=0)) == EXIT_FAILED


def test_a_partial_cohort_exits_partial_and_names_the_missing_protein(tmp_path, capsys):
    def runner(*, protein_id, config, signature, out_dir, inputs=None, oracles_factory=None):
        if protein_id == "B_2":
            raise RuntimeError("shard died")
        return "ok", {"complete_endpoints": [{"endpoint_id": protein_id}]}

    config_path = _write_config(tmp_path)
    declared = f"cohort_table={_input_file(tmp_path)}"
    with pytest.raises(RuntimeError):
        main(["--v2-config", str(config_path), "--out-dir", str(tmp_path / "out"),
              "--cohort", "A_1", "B_2", "--code-revision", "deadbeef",
              "--input-file", declared], runner=runner)
    # A_1's fragment was written before the crash; aggregating alone must report B_2 as missing.
    code = main(["--v2-config", str(config_path), "--out-dir", str(tmp_path / "out"),
                 "--cohort", "A_1", "B_2", "--code-revision", "deadbeef",
                 "--input-file", declared, "--aggregate-only"])
    assert code == EXIT_PARTIAL
    assert "missing protein: B_2" in capsys.readouterr().err


def test_an_empty_cohort_is_not_a_success(tmp_path):
    """Requesting nothing and succeeding at it is not a completed run."""
    assert _run(tmp_path, cohort=()) == EXIT_FAILED


def test_a_duplicate_protein_in_the_cohort_is_refused(tmp_path):
    assert _run(tmp_path, cohort=("A_1", "A_1")) == EXIT_UNLAUNCHABLE


# --------------------------------------------------------------------------------------------
# resume: paid work survives, stale work does not
# --------------------------------------------------------------------------------------------


def test_paid_work_is_not_recomputed_on_a_second_invocation(tmp_path):
    """The point of a content-bound resume: a re-run pays only for what is missing."""
    runner = _fake_runner()
    assert _run(tmp_path, cohort=("A_1", "B_2"), runner=runner) == EXIT_OK
    assert runner.calls == ["A_1", "B_2"]

    again = _fake_runner()
    assert _run(tmp_path, cohort=("A_1", "B_2"), runner=again) == EXIT_OK
    assert again.calls == [], "the driver re-paid for work that was already on disk"


def test_a_crash_after_paid_work_leaves_that_work_reusable(tmp_path):
    """A shard that died after finishing A_1 must not force A_1 to be recomputed."""
    def flaky(*, protein_id, config, signature, out_dir, inputs=None, oracles_factory=None):
        if protein_id == "B_2":
            raise RuntimeError("killed")
        return "ok", {"complete_endpoints": [{"endpoint_id": protein_id}]}

    config_path = _write_config(tmp_path)
    argv = ["--v2-config", str(config_path), "--out-dir", str(tmp_path / "out"),
            "--cohort", "A_1", "B_2", "--code-revision", "deadbeef",
            "--input-file", f"cohort_table={_input_file(tmp_path)}"]
    with pytest.raises(RuntimeError):
        main(argv, runner=flaky)

    retry = _fake_runner()
    assert main(argv, runner=retry) == EXIT_OK
    assert retry.calls == ["B_2"], "the retry recomputed work that had already been paid for"


def test_a_config_change_invalidates_previously_paid_work(tmp_path):
    """Reusing it would mix two experiments' results into one table."""
    runner = _fake_runner()
    assert _run(tmp_path, cohort=("A_1",), runner=runner) == EXIT_OK

    changed = _write_config(tmp_path, **{"identity.master_seed": 999})
    rerun = _fake_runner()
    assert _run(tmp_path, cohort=("A_1",), runner=rerun, config_path=changed) == EXIT_OK
    assert rerun.calls == ["A_1"], "a stale fragment was reused under a changed config"


def test_a_code_revision_change_invalidates_previously_paid_work(tmp_path):
    """Two runs of the same config under different code are not the same experiment.

    The revision is DECLARED in the config (``identity.code_revision``) rather than asserted only
    on the command line, so changing it moves the config digest too -- the fragment is refused as
    stale rather than reused across a rebuild.
    """
    inputs = [f"cohort_table={_input_file(tmp_path)}"]
    assert _run(tmp_path, inputs=inputs) == EXIT_OK

    rebuilt = _write_config(tmp_path, **{"identity.code_revision": "cafebabe"})
    rerun = _fake_runner()
    assert main(["--v2-config", str(rebuilt), "--out-dir", str(tmp_path / "out"),
                 "--cohort", "A_1", "--input-file", *inputs], runner=rerun) == EXIT_OK
    assert rerun.calls == ["A_1"], "a fragment from another code revision was reused"
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["code_revision"] == "cafebabe"


def test_a_changed_declared_input_invalidates_previously_paid_work(tmp_path):
    """The signature is bound to input CONTENT, not to a filename or an mtime."""
    data = tmp_path / "input.txt"
    data.write_text("first")
    config_path = _write_config(tmp_path)
    base = ["--v2-config", str(config_path), "--out-dir", str(tmp_path / "out"),
            "--cohort", "A_1", "--code-revision", "deadbeef",
            "--input-file", f"cohort_table={data}"]
    assert main(base, runner=_fake_runner()) == EXIT_OK

    data.write_text("second")
    rerun = _fake_runner()
    assert main(base, runner=rerun) == EXIT_OK
    assert rerun.calls == ["A_1"], "work computed from different input content was reused"


def test_swapping_input_roles_invalidates_previously_paid_work(tmp_path):
    """The same content set under different scientific roles is a different run."""
    cohort = _input_file(tmp_path, name="cohort.tsv", text="A_1\n")
    backbone = _input_file(tmp_path, name="backbone.pdb", text="ATOM\n")
    config_path = _write_config(tmp_path)
    common = [
        "--v2-config", str(config_path), "--out-dir", str(tmp_path / "out"),
        "--cohort", "A_1", "--code-revision", "deadbeef", "--input-file",
    ]
    first = common + [f"cohort_table={cohort}", f"backbone={backbone}"]
    assert main(first, runner=_fake_runner()) == EXIT_OK

    swapped = common + [f"cohort_table={backbone}", f"backbone={cohort}"]
    rerun = _fake_runner()
    assert main(swapped, runner=rerun) == EXIT_OK
    assert rerun.calls == ["A_1"], "a fragment survived a scientific input-role swap"


def test_the_bundle_records_the_requested_cohort_and_what_was_missing(tmp_path):
    """A bundle that recorded only what succeeded could not be told apart from one where nothing
    else was ever requested."""
    def runner(*, protein_id, config, signature, out_dir, inputs=None, oracles_factory=None):
        if protein_id == "B_2":
            raise __import__("inverse_folding.reference_flow.fusion_v2.errors",
                             fromlist=["V2Error"]).V2Error("no admissible endpoint")
        return "ok", {"complete_endpoints": [{"endpoint_id": protein_id}]}

    _run(tmp_path, cohort=("A_1", "B_2"), runner=runner)
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["requested_cohort"] == ["A_1", "B_2"]
    assert manifest["n_ok"] == 1
    assert manifest["missing_proteins"] == []


def test_a_failed_fragment_is_retried_rather_than_skipped_forever(tmp_path):
    """A resume may only skip work that actually SUCCEEDED.

    Skipping on admissibility alone means the first transient failure -- an OOM, a preempted node --
    freezes that protein as failed for every future invocation, and the cohort can never be
    completed by re-running it.  The failure is still recorded; it is just not a reason to stop
    trying.
    """
    config_path = _write_config(tmp_path)
    first = _fake_runner(status="failed", rows=0)
    assert _run(tmp_path, cohort=("A_1",), runner=first, config_path=config_path) == EXIT_FAILED
    assert first.calls == ["A_1"]

    retry = _fake_runner(status="ok")
    assert _run(tmp_path, cohort=("A_1",), runner=retry, config_path=config_path) == EXIT_OK
    assert retry.calls == ["A_1"], "a failed shard was skipped instead of retried"


# --------------------------------------------------------------------------------------------
# content provenance: PLAN §5.2 "Missing content identity fails closed"
# --------------------------------------------------------------------------------------------


def test_a_run_that_declares_no_inputs_is_refused_rather_than_signed_with_a_sentinel(tmp_path):
    """PLAN §5.2: the run signature binds file CONTENTS, and missing content identity fails closed.

    A sentinel signature is worse than no signature: every run that declared nothing shares it, so
    two experiments over different data resume from each other's fragments and the run manifest
    records an identity that was never observed.
    """
    runner = _fake_runner()
    assert _run(tmp_path, runner=runner, inputs=[]) == EXIT_UNLAUNCHABLE
    assert runner.calls == [], "work was paid for under an unsigned identity"


def test_a_dry_run_that_declares_no_inputs_is_refused(tmp_path):
    """The gate exists to be hit BEFORE the allocation, so it must fail in the cheap path too.

    A --dry-run that exits zero is exactly the evidence an operator uses to justify submitting.
    """
    assert _run(tmp_path, inputs=[], extra=["--dry-run"]) == EXIT_UNLAUNCHABLE


def test_print_config_stays_lenient_about_undeclared_inputs(tmp_path, capsys):
    """A DELIBERATE exception, documented in the driver's docstring.

    --print-config asserts nothing about a run: it resolves the config, echoes its identity and
    projects its budget, which is precisely what an operator does BEFORE assembling the input set.
    It signs nothing, writes nothing and reuses nothing, so there is no identity to fail closed on
    -- and it reports the missing input signature as null rather than as a value.
    """
    assert _run(tmp_path, inputs=[], extra=["--print-config"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["input_signature"] is None


def test_content_signature_binds_roles_but_not_path_aliases(tmp_path):
    first = _input_file(tmp_path, name="first.dat", text="first")
    second = _input_file(tmp_path, name="second.dat", text="second")
    first_digest = hashlib.sha256(first.read_bytes()).hexdigest()
    second_digest = hashlib.sha256(second.read_bytes()).hexdigest()
    declared = (
        DeclaredInput("cohort_table", first, first_digest),
        DeclaredInput("backbone", second, second_digest),
    )
    swapped = (
        DeclaredInput("cohort_table", second, second_digest),
        DeclaredInput("backbone", first, first_digest),
    )
    assert input_signature(declared) != input_signature(swapped)

    alias = _input_file(tmp_path, name="alias.dat", text="first")
    aliased = (
        DeclaredInput("cohort_table", alias, first_digest),
        DeclaredInput("backbone", second, second_digest),
    )
    assert input_signature(declared) == input_signature(aliased)


@pytest.mark.parametrize("declared,message", [
    (["{path}"], "ROLE=PATH"),                                     # a bare path binds no role
    (["cohort_table="], "ROLE=PATH"),                              # an empty path is not a file
    (["not_a_role={path}"], "does not declare"),                   # unknown role
    (["cohort_table={path}", "cohort_table={path}"], "twice"),     # one role, two files
])
def test_a_declared_input_must_name_exactly_one_config_role(tmp_path, capsys, declared, message):
    """An input whose ROLE is unknown cannot be checked against anything.

    PLAN §5.2 lists the provenance by role -- cohort table, backbone, constraint manifest, Head
    checkpoint -- so a file that arrives without one can be neither compared to its declared digest
    nor recorded as that role's observed identity.

    The MESSAGE is asserted, not only the exit code.  Several of these would be refused anyway by
    the next check down (a bare path is not a declared role either), and a refusal that misnames
    the cause sends an operator to fix the wrong thing.
    """
    path = _input_file(tmp_path)
    runner = _fake_runner()
    assert _run(tmp_path, runner=runner,
                inputs=[d.format(path=path) for d in declared]) == EXIT_UNLAUNCHABLE
    assert message in capsys.readouterr().err
    assert runner.calls == []


def test_a_missing_declared_input_is_refused_before_any_work(tmp_path):
    runner = _fake_runner()
    assert _run(tmp_path, runner=runner,
                inputs=[f"cohort_table={tmp_path / 'absent.tsv'}"]) == EXIT_UNLAUNCHABLE
    assert runner.calls == []


def test_the_manifest_carries_the_declared_and_observed_identity_of_every_role(tmp_path):
    """PLAN §5.3: the run manifest carries ALL content identities.

    Keyed by ROLE, because that is the vocabulary §5.2 states the requirement in -- a label is a
    human name that two roles may share.  A digest that was never observed is explicitly null: the
    empty string reads as a value, and "" would sort, compare and print as though the content had
    been identified.
    """
    data = _input_file(tmp_path)
    assert _run(tmp_path, inputs=[f"cohort_table={data}"]) == EXIT_OK
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())

    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    config = load_v2_config_file(tmp_path / "v2.yaml")
    rows = {row["role"]: row for row in manifest["content_provenance"]}
    assert set(rows) == {row.role for row in config.content}, "a declared role went unrecorded"

    observed = hashlib.sha256(data.read_bytes()).hexdigest()
    assert rows["cohort_table"]["observed_sha256"] == observed
    assert rows["cohort_table"]["observed_path"] == str(data)
    assert rows["cohort_table"]["declared_sha256"] is None, "a runtime role declares no digest"
    assert rows["cohort_table"]["declared_label"] == "cohort_table.bin"

    # No file was supplied for this role, so no digest may be invented for it.
    assert rows["backbone"]["observed_sha256"] is None
    assert rows["backbone"]["observed_path"] is None

    # A frozen role's declared digest is carried even when nothing was observed against it.
    assert rows["projection_policy_spec"]["declared_sha256"] == "a" * 64

    assert "" not in [row["declared_sha256"] for row in rows.values()]
    assert "" not in [row["observed_sha256"] for row in rows.values()]
    assert manifest["content_identities"]["cohort_table"] == observed


def test_a_declared_input_that_contradicts_its_frozen_digest_is_refused(tmp_path):
    """The config says what the content must be; the file says what it is.

    Running on content that contradicts the declaration would produce results attributed to a
    calibration or policy spec that never entered the run.
    """
    data = _input_file(tmp_path)
    config_path = _write_config_binding(tmp_path, "projection_policy_spec", "b" * 64)
    runner = _fake_runner()
    assert _run(tmp_path, runner=runner, config_path=config_path,
                inputs=[f"projection_policy_spec={data}"]) == EXIT_UNLAUNCHABLE
    assert runner.calls == []


def test_a_declared_input_that_matches_its_frozen_digest_is_accepted(tmp_path):
    data = _input_file(tmp_path)
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    config_path = _write_config_binding(tmp_path, "projection_policy_spec", digest)
    assert _run(tmp_path, config_path=config_path,
                inputs=[f"projection_policy_spec={data}"]) == EXIT_OK
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    row = next(r for r in manifest["content_provenance"] if r["role"] == "projection_policy_spec")
    assert row["declared_sha256"] == row["observed_sha256"] == digest


# --------------------------------------------------------------------------------------------
# code revision: one experiment, one revision
# --------------------------------------------------------------------------------------------


def test_a_code_revision_that_disagrees_with_the_config_is_refused_and_names_both(tmp_path,
                                                                                  capsys):
    """Two different revisions must not silently be treated as one experiment."""
    runner = _fake_runner()
    config_path = _write_config(tmp_path)
    code = main(["--v2-config", str(config_path), "--out-dir", str(tmp_path / "out"),
                 "--cohort", "A_1", "--code-revision", "cafebabe",
                 "--input-file", f"cohort_table={_input_file(tmp_path)}"], runner=runner)
    assert code == EXIT_UNLAUNCHABLE
    err = capsys.readouterr().err
    assert "cafebabe" in err and "deadbeef" in err
    assert runner.calls == []


def test_the_code_revision_defaults_to_the_config_declaration(tmp_path):
    """No sentinel, and no second place to get it wrong.

    "unknown" is already a REJECTED placeholder for this field elsewhere in V2
    (``fusion_v2.schedule._require_code_revision``), so defaulting to it made the driver the one
    surface that would sign a run with a value the rest of the codebase refuses.
    """
    config_path = _write_config(tmp_path)
    code = main(["--v2-config", str(config_path), "--out-dir", str(tmp_path / "out"),
                 "--cohort", "A_1",
                 "--input-file", f"cohort_table={_input_file(tmp_path)}"], runner=_fake_runner())
    assert code == EXIT_OK
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["code_revision"] == "deadbeef"


def test_the_parser_requires_the_scientific_arguments():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


# --------------------------------------------------------------------------------------------
# the execution stage's contract with the driver
# --------------------------------------------------------------------------------------------


def test_the_shard_refuses_to_build_oracles_implicitly(tmp_path):
    """--dry-run must never cost a GPU allocation.

    A shard that assembled the DPLM denoiser, the frozen Head and the structure gate implicitly
    would make the cheapest safety check the most expensive one.  The factory is therefore
    explicit, and its absence is a typed refusal rather than a lazy import that happens to fire.
    """
    from scripts.rf_fusion_v2_cohort import V2CohortError, run_v2_shard

    with pytest.raises(V2CohortError, match="oracles_factory"):
        run_v2_shard(protein_id="A_1", config=None, signature=None, out_dir=tmp_path)


def test_the_depth_plan_is_transcribed_from_the_config_not_inferred(tmp_path):
    """Breadth and coordinates are DECLARED (PLAN V2F6); anything inferred here would be a
    scientific decision smuggled into a driver."""
    from scripts.rf_fusion_v2_cohort import build_depth_plan
    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    config = load_v2_config_file(_write_config(tmp_path))
    plan = build_depth_plan(config)
    points = sorted(config.schedule.points, key=lambda p: p.depth)
    assert plan.depth_cap == config.schedule.depth_cap
    assert plan.cycles == tuple(
        (p.r_step, p.c_source_step, p.c_next_step) for p in points)
    assert plan.lookaheads_per_depth == tuple(p.n_lookaheads for p in points)
    assert plan.n_steps == config.substrate.n_steps


def _matching_runtime_config(config):
    """A runtime sampler config that agrees with the run's declared V2 substrate."""
    import types

    sub = config.substrate
    return types.SimpleNamespace(
        sampler=types.SimpleNamespace(
            n_steps=sub.n_steps, temperature=sub.temperature,
            remask=types.SimpleNamespace(enabled=sub.remask_enabled,
                                         fraction_scale=sub.remask_fraction_scale)),
        amplification=types.SimpleNamespace(form=sub.amplification_form),
    )


def test_a_shard_that_stops_without_a_definitive_design_is_not_a_success(tmp_path):
    """A typed stop is a real result AND a real failure for that protein.

    Counting it as a success would publish an empty table under a completed protein, and the
    cohort's success rate would be measured on the wrong denominator.
    """
    import types

    from scripts.rf_fusion_v2_cohort import run_v2_shard
    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    config = load_v2_config_file(_write_config(tmp_path))
    empty_ladder = types.SimpleNamespace(
        cycles=(), archive=_EmptyArchive(), stopping_reason=types.SimpleNamespace(value="no_ok"),
        depth_reached=0, total_logical_dfe=0, substrate_digest="d" * 64, best_definitive=None,
        production_depth_authorized=False, exploratory_depth_override=False,
    )

    import scripts.rf_fusion_v2_cohort as cohort_mod
    from inverse_folding.reference_flow.fusion_v2_runtime import ladder as ladder_mod

    original = ladder_mod.run_depth_ladder
    ladder_mod.run_depth_ladder = lambda **kw: empty_ladder
    try:
        status, payload = run_v2_shard(
            protein_id="A_1", config=config,
            signature=RunSignature(
                config_digest=config.config_digest(),
                campaign_id=config.identity.campaign_id,
                split_role=config.identity.split_role,
                arm_role=config.arm.arm_role,
                protein_id="A_1",
                input_signature="1" * 64,
                code_revision=config.identity.code_revision,
            ),
            out_dir=tmp_path,
            # A runtime config that MATCHES the declared substrate: this test is about a typed
            # stop not counting as success, and a mismatched substrate would refuse the shard for
            # an unrelated reason before the ladder ever ran.
            oracles_factory=lambda **kw: {"cycle_kwargs": {"config": _matching_runtime_config(
                config)}},
        )
    finally:
        ladder_mod.run_depth_ladder = original
    assert status == "failed"
    assert payload["depth_reached"] == 0
    assert payload["stopping_reason"] == "no_ok"


class _EmptyArchive:
    def raw_rows(self):
        return ()

    def endpoints(self):
        return ()

    def may_become_ancestry(self, endpoint_id):  # pragma: no cover - no rows to ask about
        return False


# --------------------------------------------------------------------------------------------
# the compute ledger reaches the bundle (PLAN §5.3, §5.4)
# --------------------------------------------------------------------------------------------


def _ledger_runner(events):
    def runner(*, protein_id, config, signature, out_dir, inputs=None, oracles_factory=None):
        return "ok", {
            "complete_endpoints": [{"endpoint_id": f"{protein_id}:0"}],
            "ledger_events": list(events),
        }
    return runner


def test_the_shards_compute_ledger_is_written_into_the_bundle(tmp_path):
    """PLAN §5.3 lists the compute ledger as a REQUIRED evidence object.

    ``aggregate_fragments`` already collected ``payload["ledger_events"]`` into
    ``report.ledger_events`` and the writer already accepted a ``ledger_events=`` argument -- but
    nothing joined the two, so `cost_ledger.jsonl` was never produced.  Without it a run cannot say
    what it burned, and PLAN §5.4's ``unknown_after_start`` -- the record of work started and never
    measured -- has nowhere to be read from.
    """
    from scripts.rf_artifact_io import read_cost_ledger_jsonl

    event = {
        "event_id": "evt:screen", "attempt_id": "att:1", "protein_id": "A_1", "arm": "v2",
        "phase": "screen", "status": "ok", "logical_dfe": 150, "physical_forwards": 150,
    }
    assert _run(tmp_path, runner=_ledger_runner([event])) == EXIT_OK
    ledger = tmp_path / "out" / "cost_ledger.jsonl"
    assert ledger.exists(), "the bundle carries no compute ledger"
    assert read_cost_ledger_jsonl(ledger) == [event]


def test_a_run_that_burned_nothing_measurable_still_writes_a_ledger_file(tmp_path):
    """An absent file and an empty one are different claims: "no ledger was kept" versus "the run
    kept a ledger and it is empty".  Only the second is auditable."""
    assert _run(tmp_path, runner=_ledger_runner([])) == EXIT_OK
    assert (tmp_path / "out" / "cost_ledger.jsonl").exists()


# --------------------------------------------------------------------------------------------
# runtime paths reach the shard (CLAUDE.md: cluster paths are CLI arguments, never hardcoded)
# --------------------------------------------------------------------------------------------


def test_the_driver_hands_the_shard_the_runtime_paths_it_was_given(tmp_path):
    """``ShardInputs`` is how a shard receives every cluster path -- the journal directory, the
    checkpoint, the test set, the PDB root.  The driver never passed one, so nothing declared on
    the command line could reach the execution stage and the journal could only fall back to a
    location the operator never chose.

    ``NAME=PATH`` rather than a fixed flag per path: the factory's parameter list is a property of
    the oracle stack, not of the driver, and inventing one here would be a contract this driver
    cannot honour.
    """
    seen = {}

    def runner(*, protein_id, config, signature, out_dir, inputs=None, **kw):
        seen["inputs"] = inputs
        return "ok", {"complete_endpoints": [{"endpoint_id": protein_id}]}

    journal = tmp_path / "journals"
    assert _run(tmp_path, runner=runner, extra=[
        "--shard-input", f"journal_dir={journal}", f"pdb_root={tmp_path}",
    ]) == EXIT_OK
    assert seen["inputs"] is not None, "the shard was given no runtime paths at all"
    assert seen["inputs"].require("journal_dir") == str(journal)
    assert seen["inputs"].require("pdb_root") == str(tmp_path)


def test_a_shard_input_without_a_name_is_a_typed_refusal(tmp_path):
    """A bare path cannot be routed to a parameter, and guessing which one it meant is how a
    checkpoint ends up passed as a PDB root."""
    code = _run(tmp_path, extra=["--shard-input", str(tmp_path)])
    assert code == EXIT_UNLAUNCHABLE


def test_a_duplicated_shard_input_name_is_refused(tmp_path):
    """Last-one-wins would silently pick between two paths the operator declared."""
    code = _run(tmp_path, extra=["--shard-input", f"pdb_root={tmp_path}", f"pdb_root={tmp_path}"])
    assert code == EXIT_UNLAUNCHABLE


# --------------------------------------------------------------------------------------------
# cohort-level hard caps (PLAN §5.1, §5.3)
# --------------------------------------------------------------------------------------------


def _cap_runner(per_protein_dfe):
    def runner(*, protein_id, config, signature, out_dir, inputs=None, oracles_factory=None):
        return "ok", {
            "complete_endpoints": [{"endpoint_id": f"{protein_id}:0"}],
            "ledger_events": [{
                "event_id": f"evt:{protein_id}", "attempt_id": f"att:{protein_id}",
                "protein_id": protein_id, "arm": "v2", "phase": "screen", "status": "ok",
                "logical_dfe": per_protein_dfe, "physical_forwards": per_protein_dfe,
            }],
        }
    return runner


def test_a_cohort_whose_sum_breaches_a_declared_cap_does_not_exit_zero(tmp_path):
    """PLAN §5.1 caps are COHORT-scoped, and the preflight already reads them that way
    (`total = per_protein * n_proteins`).  The only realized check was per shard, so four proteins
    each individually under the cap could breach it together and the driver would still report a
    completed run.  A cohort runner reads the exit code and nothing else.
    """
    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    config_path = _write_config(tmp_path)
    cap = load_v2_config_file(config_path).caps.max_logical_dfe
    per_protein = cap // 2                      # each shard is comfortably under the cap
    code = _run(tmp_path, cohort=("A_1", "B_2", "C_3", "D_4"),
                runner=_cap_runner(per_protein), config_path=config_path)
    assert code != EXIT_OK, (
        f"4 x {per_protein} = {4 * per_protein} logical DFE against a cap of {cap} exited 0")


def test_a_cohort_inside_its_caps_still_exits_zero(tmp_path):
    """The guard must not be satisfiable by failing every run."""
    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    config_path = _write_config(tmp_path)
    cap = load_v2_config_file(config_path).caps.max_logical_dfe
    assert _run(tmp_path, cohort=("A_1", "B_2"), runner=_cap_runner(cap // 8),
                config_path=config_path) == EXIT_OK


def test_the_breached_cap_is_named_in_the_manifest_not_only_in_the_exit_code(tmp_path):
    """An exit code says a run is unusable; it cannot say WHICH budget it blew, and the operator
    has to decide whether to re-scope the cohort or re-scope the science."""
    import json

    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    config_path = _write_config(tmp_path)
    cap = load_v2_config_file(config_path).caps.max_logical_dfe
    _run(tmp_path, cohort=("A_1", "B_2", "C_3", "D_4"), runner=_cap_runner(cap // 2),
         config_path=config_path)
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert "max_logical_dfe" in json.dumps(manifest.get("realized_caps", {}))


# --------------------------------------------------------------------------------------------
# the driver reaches a real oracle stack, and --dry-run is a STRICT gate
# --------------------------------------------------------------------------------------------


def test_the_driver_supplies_the_production_oracle_factory(tmp_path):
    """``run_v2_shard`` refuses to build oracles implicitly, so a driver that never passed one made
    every real launch fail with "no oracles_factory was supplied" after the config had already been
    resolved and signed."""
    seen = {}

    def runner(*, protein_id, config, signature, out_dir, inputs=None, oracles_factory=None):
        seen["factory"] = oracles_factory
        return "ok", {"complete_endpoints": [{"endpoint_id": protein_id}]}

    assert _run(tmp_path, runner=runner) == EXIT_OK
    assert seen["factory"] is not None, "the driver handed the shard no oracle factory"


def test_dry_run_parses_the_runtime_paths_it_will_launch_with(tmp_path):
    """``--dry-run`` returned success BEFORE ``--shard-input`` was parsed, so a malformed runtime
    path -- the thing that decides whether the job can start at all -- was discovered only on the
    GPU.  A gate that does not check the launch inputs is not a launch gate.
    """
    code = _run(tmp_path, extra=["--dry-run", "--shard-input", str(tmp_path)])
    assert code == EXIT_UNLAUNCHABLE


def test_dry_run_still_succeeds_on_well_formed_runtime_paths(tmp_path):
    """The gate must not be satisfiable by refusing every dry run."""
    assert _run(tmp_path, extra=["--dry-run", "--shard-input", f"pdb_root={tmp_path}"]) == EXIT_OK


# --------------------------------------------------------------------------------------------
# the launch gate must see the Dual overlay it would launch with
# --------------------------------------------------------------------------------------------


def _overlay_file(tmp_path, **over):
    """A signed Dual overlay on disk, authored through the loader's own serializer."""
    from inverse_folding.reference_flow.fusion_v2.dual_config import dual_overlay_payload
    from tests.inverse_folding.test_fusion_v2_dual_config import overlay

    payload = dual_overlay_payload(overlay())
    payload.update(over)
    path = tmp_path / "dual_overlay.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def test_dry_run_refuses_a_dual_overlay_that_does_not_exist(tmp_path):
    """``--dry-run`` returned EXIT_OK before ``resolve_dual`` ran at all.

    So a three-arm Dual submission whose overlay path was wrong passed its own launch gate and
    failed on the GPU -- after the allocation the gate exists to authorize had been paid for.
    """
    code = _run(tmp_path, extra=["--dry-run", "--dual-overlay",
                                 str(tmp_path / "definitely-missing.json"),
                                 "--dual-arm", "joint"])
    assert code == EXIT_UNLAUNCHABLE


def test_dry_run_refuses_an_arm_the_overlay_never_declared(tmp_path):
    code = _run(tmp_path, extra=["--dry-run", "--shard-input", f"dual_overlay={_overlay_file(tmp_path)}", "--dual-overlay", str(_overlay_file(tmp_path)),
                                 "--dual-arm", "b_only"])
    assert code == EXIT_UNLAUNCHABLE


def test_dry_run_refuses_half_a_dual_specification(tmp_path):
    assert _run(tmp_path, extra=["--dry-run", "--dual-arm", "joint"]) == EXIT_UNLAUNCHABLE
    assert _run(tmp_path, extra=["--dry-run", "--dual-overlay",
                                 str(_overlay_file(tmp_path))]) == EXIT_UNLAUNCHABLE


def test_dry_run_accepts_a_declared_overlay_and_arm(tmp_path):
    """The gate must not be satisfiable by refusing every Dual dry run."""
    assert _run(tmp_path, extra=["--dry-run", "--shard-input", f"dual_overlay={_overlay_file(tmp_path)}", "--dual-overlay", str(_overlay_file(tmp_path)),
                                 "--dual-arm", "joint"]) == EXIT_OK


def test_print_config_reports_the_resolved_dual_mode_rather_than_the_flag(tmp_path):
    """An operator reads the printed payload, not the command line, to know what was resolved."""
    config_path = _write_config(tmp_path)
    overlay_path = _overlay_file(tmp_path)
    result = subprocess.run(
        [sys.executable, "scripts/run_rf_fusion_v2.py", "--v2-config", str(config_path),
         "--out-dir", str(tmp_path / "out"), "--cohort", "A_1", "--print-config",
         "--shard-input", f"dual_overlay={overlay_path}", "--dual-overlay", str(overlay_path), "--dual-arm", "joint"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == EXIT_OK, result.stderr
    payload = json.loads(result.stdout)
    assert payload["dual_mode"] == "dual"
    assert payload["dual_arm"] == "joint"
    assert payload["dual_signature"]


def test_print_config_reports_no_dual_mode_for_a_legacy_run(tmp_path):
    config_path = _write_config(tmp_path)
    result = subprocess.run(
        [sys.executable, "scripts/run_rf_fusion_v2.py", "--v2-config", str(config_path),
         "--out-dir", str(tmp_path / "out"), "--cohort", "A_1", "--print-config"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == EXIT_OK, result.stderr
    payload = json.loads(result.stdout)
    assert payload["dual_mode"] == "none"
    assert payload["dual_signature"] == ""


def test_a_factory_that_ignores_the_resolved_dual_arm_is_refused(tmp_path):
    """The failure this guard exists for is silent by construction.

    Before the wiring, ``resolve_dual`` fed the run SIGNATURE and nothing else: the shard still ran
    one Head, so ``joint``, ``a_only`` and ``b_only`` executed identical A-only V2 under three
    different signatures.  Nothing in the artifact said so -- three arms agreed, and the agreement
    meant nothing.  So the driver's resolved arm and the factory's returned runtime are compared.
    """
    from scripts.rf_fusion_v2_cohort import V2CohortError, run_v2_shard

    def blind_factory(*, protein_id, config, inputs, dual=None):
        del dual                       # the exact mistake: accepted and dropped
        return {"gpu_clock": lambda: 0.0, "cycle_kwargs": {}}

    from scripts.rf_fusion_v2_preflight import load_v2_config_file
    from scripts.run_rf_fusion_v2 import _signatures

    config = load_v2_config_file(_write_config(tmp_path))
    path = _input_file(tmp_path)
    declared = (DeclaredInput(
        "cohort_table", str(path), hashlib.sha256(Path(path).read_bytes()).hexdigest()),)
    signature = _signatures(
        config, ("A_1",), arm_role=config.arm.arm_role, code_revision="deadbeef",
        inputs=declared, dual_signature="deadbeef")["A_1"]

    with pytest.raises(V2CohortError, match="A-only|dual"):
        run_v2_shard(
            protein_id="A_1", config=config, signature=signature,
            out_dir=tmp_path / "out", oracles_factory=blind_factory,
            dual=(object(), "joint"))


# -- only a calibration whose panel sensitivity was MEASURED may steer a run --------------------

def test_an_overlay_with_no_measured_panel_sensitivity_is_refused_at_the_gate(tmp_path):
    """Four structurally valid signed overlays sit in the calibration tree; two may be launched.

    Nothing in the launch path told them apart. The primary calibration RECORD, the deliberately
    contaminated D2 sensitivity calibration, and a 200-sequence bring-up probe whose equal-risk line
    has the OPPOSITE SIGN are all loadable, all carry a plausible C, and all share the
    ``dual_overlay*`` prefix. A shell glob, a tab-completion or an agent resolving the variable from
    the wrong line launches one of them and nothing objects.

    The discriminator needs no schema change and is the D2 gate expressed in code: AUDIT J.7 D2
    requires the overlap-inclusion sensitivity to be MEASURED and recorded before launch, and only
    the campaign artifacts carry it. Re-signing from a contaminated source also drops it, so this
    one guard closes that path too.
    """
    import dataclasses
    import json

    from inverse_folding.reference_flow.fusion_v2.dual_config import dual_overlay_payload
    from tests.inverse_folding.test_fusion_v2_dual_config import overlay as _ov

    def write(name, calibration):
        p = tmp_path / name
        p.write_text(json.dumps(dual_overlay_payload(_ov(calibration=calibration)),
                                indent=2, sort_keys=True) + "\n")
        return p

    launchable = _ov().calibration                       # fixture carries a measured shift
    assert launchable.panel.leave_overlap_out_shift is not None
    record = dataclasses.replace(
        launchable, panel=dataclasses.replace(launchable.panel, leave_overlap_out_shift=None))

    from scripts import run_rf_fusion_v2 as drv

    args = argparse.Namespace(dual_overlay=str(write("ok.json", launchable)), dual_arm="joint")
    got, arm = drv.resolve_dual(args)
    assert arm == "joint" and got is not None

    args = argparse.Namespace(dual_overlay=str(write("record.json", record)), dual_arm="joint")
    with pytest.raises(drv.V2DriverError, match="sensitivity|leave_overlap_out_shift"):
        drv.resolve_dual(args)
