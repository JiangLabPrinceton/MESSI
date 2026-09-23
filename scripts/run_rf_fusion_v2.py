"""Run config-bound Fusion V2 generation with optional Dual steering."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion_v2.errors import V2Error  # noqa: E402
from scripts.rf_fusion_v2_artifacts import (  # noqa: E402
    V2_TABLE_SCHEMAS,
    run_manifest,
    DUAL_TABLE_SCHEMAS,
    write_v2_bundle,
)
from scripts.rf_fusion_v2_preflight import (  # noqa: E402
    V2PreflightError,
    assert_launch_feasible,
    input_signature,
    load_v2_config_file,
    print_config_payload,
    project_v2_budget,
)
from scripts.rf_fusion_v2_resume import (  # noqa: E402
    RunSignature,
    aggregate_fragments,
    write_fragment,
)

EXIT_OK = 0
EXIT_FAILED = 2      # nothing usable (also argparse's usage-error code)
EXIT_PARTIAL = 3     # some proteins missing or rejected
EXIT_UNLAUNCHABLE = 4

__all__ = [
    "EXIT_OK", "EXIT_FAILED", "EXIT_PARTIAL", "EXIT_UNLAUNCHABLE",
    "V2DriverError", "DeclaredInput",
    "build_parser", "main", "decide_exit_code",
    "parse_declared_inputs", "parse_shard_inputs", "realized_cap_verdict",
    "resolve_code_revision",
    "content_provenance",
]


class V2DriverError(V2Error):
    """A run identity this driver refuses to sign, as opposed to a run that failed."""


@dataclass(frozen=True)
class DeclaredInput:
    """One PLAN §5.2 role, the file supplied for it, and that file's OBSERVED content digest."""

    role: str
    path: Path
    sha256: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_rf_fusion_v2",
        description="V2 trajectory-coupled pre-terminal feedback driver",
    )
    parser.add_argument("--v2-config", required=True,
                        help="V2 run config YAML (schema-validated; no defaults are supplied)")
    parser.add_argument("--out-dir", required=True, help="artifact bundle destination")
    parser.add_argument("--cohort", nargs="*", default=(), metavar="PROTEIN_ID",
                        help="the frozen requested cohort; a protein missing from the result is "
                             "reported, never dropped")
    parser.add_argument("--input-file", nargs="*", default=(), metavar="ROLE=PATH",
                        help="declared input files, each naming the PLAN §5.2 content role it "
                             "binds; their CONTENT signs the resume identity and is checked "
                             "against any digest the config declares for that role. A run that "
                             "declares none is refused rather than signed with a sentinel")
    parser.add_argument("--code-revision", default=None,
                        help="the code revision this run is bound to. Defaults to the config's "
                             "identity.code_revision, which is the authority; a value that "
                             "disagrees with it is refused rather than run as one experiment")
    parser.add_argument("--dual-overlay", default=None, metavar="PATH",
                        help="path to the signed dual-allele overlay JSON. Optional and off by "
                             "default: with no overlay this run is the frozen single-Head V2 in "
                             "every byte it produces, and the Dual layer is never imported")
    parser.add_argument("--dual-arm", default=None, metavar="ARM",
                        help="which arm of the overlay's declared bundle this process runs "
                             "(joint | a_only | b_only). Required with --dual-overlay and refused "
                             "without it; the overlay's arm_bundle is the authority for which arms "
                             "exist, and the executing arm enters the run signature so two arms of "
                             "one protein can never resume from each other")
    parser.add_argument("--shard-input", nargs="*", default=(), metavar="NAME=PATH",
                        help="runtime paths handed to the execution stage as ShardInputs "
                             "(e.g. journal_dir=..., pdb_root=...). Cluster paths are CLI "
                             "arguments and are never hardcoded in a module; the NAMES belong to "
                             "the oracle stack, so the driver routes them rather than fixing them")
    parser.add_argument("--fragment-dir", default=None,
                        help="per-shard fragment directory (default: <out-dir>/fragments)")
    parser.add_argument("--print-config", action="store_true",
                        help="resolve the config, derive its digest and project its budget; "
                             "loads no model")
    parser.add_argument("--dry-run", action="store_true",
                        help="everything --print-config does, plus the launch-feasibility gate; "
                             "loads no model")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="skip execution and aggregate the fragments already on disk")
    parser.add_argument("--mechanism-prefixes", type=int, default=0, metavar="N",
                        help="run the runbook §7 MECHANISM cohort instead of the depth ladder: "
                             "N independent source prefixes per protein, each carried through the "
                             "matched arms (0 = the ordinary ladder run)")
    parser.add_argument("--mechanism-prefix-start", type=int, default=0, metavar="I",
                        help="first prefix index of this BATCH (default 0).  A second batch "
                             "continues the sequence rather than repeating it: prefix seeds are "
                             "derived from the index, so restarting at 0 reproduces batch 1")
    parser.add_argument("--qualification", action="store_true",
                        help="run the V2F5A POLICY-QUALIFICATION contrast over the same prefixes "
                             "(PLAN §8.4): the declared Head-directed policy against its matched "
                             "source-geometry control, at the treatment's own realized write and "
                             "reopen cardinalities.  Requires --mechanism-prefixes and a config "
                             "declaring projection.head_directed")
    parser.add_argument(
        "--exploratory-depth-override", action="store_true",
        help="explicitly permit an exploratory real-model D>1 ladder. This is not production "
             "authorization and is accepted only for phase=capability_ladder with an "
             "exploratory* split_role; it is bound into resume and manifest identity",
    )
    return parser


def select_runner(args, *, ladder, mechanism):
    """Which experiment this invocation is.

    The two shards share a signature and the whole driver around them -- config resolution, the
    signature, oracles, fragments, resume, the ledger and the bundle -- so the choice is one
    argument rather than a second driver.  It is made HERE, once, so that no code below has to ask
    again which kind of run it is in.
    """
    n_prefixes = int(getattr(args, "mechanism_prefixes", 0) or 0)
    if n_prefixes < 0:
        raise V2DriverError(f"--mechanism-prefixes must be >= 0, got {n_prefixes}")
    start = int(getattr(args, "mechanism_prefix_start", 0) or 0)
    if start < 0:
        raise V2DriverError(f"--mechanism-prefix-start must be >= 0, got {start}")
    qualification = bool(getattr(args, "qualification", False))
    if not n_prefixes:
        if start:
            raise V2DriverError(
                "--mechanism-prefix-start is meaningless without --mechanism-prefixes; a ladder "
                "run has no prefix sequence to continue")
        if qualification:
            raise V2DriverError(
                "--qualification needs --mechanism-prefixes: the policy contrast is measured over "
                "independent source prefixes, and a ladder run has none")
        return ladder
    return functools.partial(mechanism, n_prefixes=n_prefixes, prefix_start=start,
                             qualification=qualification)


def execution_replicates(args) -> int:
    """Conservative launch-gate multiplicity for the selected execution mode.

    An ordinary ladder is one execution graph per protein.  The V2F5A qualification runs two
    descendant-generating arms for every independent source prefix.  Its support policies share
    some prefix work, but charging ``2 * N`` complete one-cycle graphs is deliberately conservative
    and, crucially, cannot let ``--dry-run`` certify a 56-prefix paired run as if it were one cycle.
    """
    n_prefixes = int(getattr(args, "mechanism_prefixes", 0) or 0)
    start = int(getattr(args, "mechanism_prefix_start", 0) or 0)
    qualification = bool(getattr(args, "qualification", False))
    if n_prefixes < 0:
        raise V2DriverError(f"--mechanism-prefixes must be >= 0, got {n_prefixes}")
    if start < 0:
        raise V2DriverError(f"--mechanism-prefix-start must be >= 0, got {start}")
    if qualification and n_prefixes < 1:
        raise V2DriverError(
            "--qualification requires --mechanism-prefixes >= 1; otherwise no paired source "
            "prefix exists to qualify"
        )
    return 2 * n_prefixes if qualification else 1


def depth_authorization(args, *, config) -> tuple[bool, bool]:
    """Resolve the two non-interchangeable ways a D>1 ladder may be opened.

    This driver exposes only the exploratory override.  Production authorization remains false
    until the authority/runbook gates are revised; keeping both booleans explicit prevents an
    exploratory artifact from being read as such a revision.
    """
    exploratory = bool(getattr(args, "exploratory_depth_override", False))
    production = False
    if not exploratory:
        return production, exploratory
    if int(getattr(args, "mechanism_prefixes", 0) or 0) or bool(
        getattr(args, "qualification", False)
    ):
        raise V2DriverError(
            "--exploratory-depth-override applies only to the ordinary capability ladder, not "
            "the mechanism or policy-qualification runners"
        )
    if int(config.schedule.depth_cap) <= 1:
        raise V2DriverError(
            "--exploratory-depth-override is only meaningful for a D>1 schedule"
        )
    if (config.identity.phase != "capability_ladder"
            or not str(config.identity.split_role).startswith("exploratory")):
        raise V2DriverError(
            "--exploratory-depth-override requires identity.phase='capability_ladder' and an "
            f"identity.split_role beginning with 'exploratory'; got "
            f"{config.identity.phase!r}/{config.identity.split_role!r}"
        )
    return production, exploratory


def decide_exit_code(report, *, requested: int) -> int:
    """Turn an aggregate report into the one number a cohort runner reads.

    Ordering matters.  "Partial" is checked before "no successes" because a partial cohort is a
    different operator action (re-run the missing shards) from a cohort that ran completely and
    produced nothing (investigate the method).
    """
    if requested == 0:
        return EXIT_FAILED
    if report.missing_proteins or report.rejected:
        # Some evidence exists but the cohort is not the one that was requested.
        return EXIT_PARTIAL if report.any_success else EXIT_FAILED
    if not report.any_success:
        # Every requested protein was processed and every one failed.  PLAN §7.2 names the zero
        # exit here as an adversarial case: it would let a cohort of total failures be consumed
        # downstream as a completed run.
        return EXIT_FAILED
    return EXIT_OK


def parse_declared_inputs(values: Sequence[Any], *, config) -> tuple[DeclaredInput, ...]:
    """Bind each declared file to the PLAN §5.2 content role it stands for, and digest it.

    A bare path is refused.  §5.2 states the provenance requirement per role -- cohort table,
    backbone, constraint manifest, Head checkpoint -- so a file that names no role can neither be
    compared against the digest its config declares nor be recorded as that role's observed
    identity; it would be provenance in name only.

    Where the config declares an expected digest, a file whose content contradicts it is refused:
    running anyway would attribute results to a calibration or policy spec that never entered the
    run.
    """
    known = {row.role: row for row in config.content}
    declared: dict[str, DeclaredInput] = {}
    for value in values:
        role, separator, raw = str(value).partition("=")
        role, raw = role.strip(), raw.strip()
        if not separator or not role or not raw:
            raise V2DriverError(
                f"--input-file {str(value)!r} must be ROLE=PATH; an input that names no content "
                f"role cannot be checked or recorded (declared roles: {sorted(known)})"
            )
        if role not in known:
            raise V2DriverError(
                f"--input-file names content role {role!r}, which this config does not declare; "
                f"declared roles: {sorted(known)}"
            )
        if role in declared:
            raise V2DriverError(
                f"content role {role!r} was declared twice ({declared[role].path} and {raw}); one "
                "role identifies one content, or the manifest cannot say which was used"
            )
        path = Path(raw)
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = known[role].expected_sha256
        if expected is not None and observed != expected:
            raise V2DriverError(
                f"declared input for role {role!r} contradicts the digest the config binds it to: "
                f"{path} has sha256 {observed}, config.content declares {expected}"
            )
        declared[role] = DeclaredInput(role=role, path=path, sha256=observed)
    return tuple(declared[role] for role in sorted(declared))


def resolve_code_revision(requested, *, config) -> str:
    """The config is the authority; the flag may only agree with it.

    ``identity.code_revision`` is required, has no default and is inside ``config_digest``.  A
    second, independently settable source for the same fact is how two revisions get treated as one
    experiment -- so omitting the flag adopts the config's value and supplying a different one is
    refused with both named.
    """
    declared = str(config.identity.code_revision)
    if requested is None:
        return declared
    if str(requested) != declared:
        raise V2DriverError(
            f"--code-revision {str(requested)!r} disagrees with the config's declared "
            f"identity.code_revision {declared!r}; two revisions are two experiments"
        )
    return declared


def content_provenance(config, declared: Sequence[DeclaredInput]) -> list[dict]:
    """One row per declared content role (PLAN §5.3: the manifest carries ALL content identities).

    Keyed by ROLE because that is the vocabulary §5.2 states the requirement in; a label is a human
    name two roles may share.  An identity that does not exist is ``None`` rather than ``""`` --
    the empty string sorts, compares and prints as though the content had been identified.
    """
    observed_by_role = {item.role: item for item in declared}
    rows = []
    for row in sorted(config.content, key=lambda entry: entry.role):
        observed = observed_by_role.get(row.role)
        rows.append({
            "role": row.role,
            "declared_label": row.label,
            "binding": row.binding,
            "declared_sha256": row.expected_sha256,
            "observed_path": None if observed is None else str(observed.path),
            "observed_sha256": None if observed is None else observed.sha256,
        })
    return rows


def _content_identities(provenance: Sequence[dict]) -> dict:
    """Role -> the digest actually established for it, omitting roles where none was.

    ``run_manifest`` coerces this mapping's values with ``str()``, so a null cannot be expressed in
    it without becoming the string ``"None"`` -- a placeholder by another name.  The complete,
    explicitly nullable table is the manifest's ``content_provenance``; this map carries only what
    is known.
    """
    identities = {}
    for row in provenance:
        digest = row["observed_sha256"] or row["declared_sha256"]
        if digest:
            identities[row["role"]] = digest
    return identities


def parse_shard_inputs(raw: Sequence[str]):
    """Turn ``NAME=PATH`` pairs into the :class:`ShardInputs` the execution stage consumes.

    A bare path is refused rather than positionally assigned: guessing which parameter it meant is
    how a checkpoint ends up passed as a PDB root.  A repeated name is refused rather than
    last-one-wins, which would silently pick between two paths the operator declared.
    """
    from scripts.rf_fusion_v2_cohort import ShardInputs

    paths: dict[str, str] = {}
    for item in raw:
        name, sep, value = str(item).partition("=")
        if not sep or not name.strip() or not value.strip():
            raise V2Error(
                f"--shard-input expects NAME=PATH, got {item!r}; a bare path cannot be routed to a "
                "parameter and guessing which one it meant is how a checkpoint is passed as a PDB "
                "root"
            )
        if name in paths:
            raise V2Error(
                f"--shard-input names {name!r} twice; last-one-wins would silently pick between "
                "two paths the run declared"
            )
        paths[name] = value
    return ShardInputs(**paths)


def realized_cap_verdict(ledger_events, *, caps):
    """Check the REALIZED cohort ledger against the run's declared hard caps (PLAN §5.1, §5.3).

    Cohort-scoped, because that is how the caps are declared and how ``project_v2_budget`` already
    reads them (``total = per_protein * n_proteins``).  The per-shard check inside ``run_v2_shard``
    cannot see this: four proteins each comfortably under ``max_logical_dfe`` can breach it
    together, and the two code paths were disagreeing about the scope of the same quantity while
    only the projection was ever evaluated cohort-wide.

    Returns ``None`` when the run kept no ledger at all -- an absent ledger is not a certificate of
    compliance and the caller reports it as such rather than reading it as "within budget".
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.ledger import (  # noqa: PLC0415
        V2LedgerEvent,
        aggregate_v2_ledger,
        check_caps,
    )

    rows = list(ledger_events or ())
    if not rows:
        return None
    try:
        events = [row if isinstance(row, V2LedgerEvent) else V2LedgerEvent(**dict(row))
                  for row in rows]
    except (TypeError, ValueError) as exc:
        raise V2Error(f"the aggregated cost ledger is not a V2 ledger: {exc}") from exc
    return check_caps(aggregate_v2_ledger(events), caps)


def resolve_dual(args, *, shard_inputs=None):
    """Resolve the optional Dual overlay and executing arm, or ``(None, None)``.

    Half a specification is refused: an overlay without an arm cannot say which comparison this
    process is running, and an arm without an overlay names a bundle nothing declared. The overlay's
    own ``arm_bundle`` is the authority for which arms exist, exactly as the config is the authority
    for ``code_revision`` -- a second independently settable source for one fact is how two things
    get treated as one experiment.

    Nothing Dual is imported unless an overlay was actually supplied, which is what keeps a legacy
    run from loading the Dual layer at all.
    """
    overlay_path = getattr(args, "dual_overlay", None)
    arm = getattr(args, "dual_arm", None)
    if overlay_path is None and arm is None:
        return None, None
    if overlay_path is None or arm is None:
        raise V2DriverError(
            "--dual-overlay and --dual-arm must be supplied together; an overlay without an arm "
            "cannot say which comparison this process is running, and an arm without an overlay "
            "names a bundle nothing declared"
        )
    from inverse_folding.reference_flow.fusion_v2 import dual_config as _dual_config

    from scripts.rf_fusion_v2_preflight import load_dual_overlay_file

    overlay = load_dual_overlay_file(overlay_path)
    if shard_inputs is not None:
        declared_path = shard_inputs.require("dual_overlay")
        declared_overlay = load_dual_overlay_file(declared_path)
        if declared_overlay.content_digest != overlay.content_digest:
            raise V2DriverError("--dual-overlay differs from the declared shard dual_overlay")
    # AUDIT J.7 D2 expressed as a gate. Four structurally valid signed overlays exist in the
    # calibration tree and only the two campaign artifacts may be launched; the primary RECORD, the
    # deliberately contaminated sensitivity calibration, and the bring-up probe are all loadable and
    # all carry a plausible C, so a glob or a mis-resolved variable reaches one of them with nothing
    # to object. What separates them needs no schema change: only an artifact whose panel
    # sensitivity has been MEASURED and recorded may steer, and re-signing from a contaminated
    # source drops that record too, which closes the same hole from the other side.
    if overlay.calibration.panel.leave_overlap_out_shift is None:
        raise V2DriverError(
            f"{overlay_path} carries no measured overlap-inclusion sensitivity "
            "(leave_overlap_out_shift is absent), so it is a calibration RECORD or a MEASUREMENT, "
            "not a launchable campaign artifact. The launchable overlays are the ones signed with "
            "--overlap-inclusion-shift; see MANIFEST.md in the calibration directory"
        )
    label = str(arm)
    if label not in overlay.arms:
        raise V2DriverError(
            f"--dual-arm {label!r} is not in the overlay's declared bundle "
            f"{list(overlay.arms)}; the overlay is the authority for which arms exist"
        )
    return overlay, label


def _dual_signature(overlay, arm: str) -> str:
    """The overlay's contribution to the run signature.

    Deliberately NOT the place the resolved Head B is proved against the overlay: nothing has been
    resolved yet.  That check belongs where the checkpoint is actually opened, and calling it here
    against the overlay's own declared digests would compare a value with itself and always pass.
    """
    from inverse_folding.reference_flow.fusion_v2.dual_config import dual_run_signature_component

    return dual_run_signature_component(overlay, arm=arm)


def _signatures(
    config, cohort, *, arm_role, code_revision, inputs,
    production_depth_authorized: bool = False,
    exploratory_depth_override: bool = False,
    dual_signature: str = "",
):
    if not inputs:
        # PLAN §5.2: "Missing content identity fails closed."  A sentinel signature would be shared
        # by every run that declared nothing, so two experiments over different data would resume
        # from each other's fragments.
        raise V2DriverError(
            "no --input-file was declared, so this run has no content identity to sign; PLAN §5.2 "
            "binds the run signature to file CONTENTS and missing content identity fails closed"
        )
    signature = input_signature(inputs)
    return {
        protein_id: RunSignature(
            config_digest=config.config_digest(),
            campaign_id=config.identity.campaign_id,
            split_role=config.identity.split_role,
            arm_role=arm_role,
            protein_id=protein_id,
            input_signature=signature,
            code_revision=code_revision,
            production_depth_authorized=bool(production_depth_authorized),
            exploratory_depth_override=bool(exploratory_depth_override),
            dual_signature=str(dual_signature),
        )
        for protein_id in cohort
    }


def main(argv=None, *, runner=None, oracles_factory=None) -> int:
    """Run, or explain why it will not.

    ``runner`` is the injection seam for the execution stage: the default is resolved lazily so the
    config paths never import torch.  Tests drive the whole driver through a fake runner, which is
    what makes the exit-code matrix verifiable without a GPU.
    """
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    fragment_dir = Path(args.fragment_dir) if args.fragment_dir else out_dir / "fragments"

    try:
        config = load_v2_config_file(args.v2_config)
    except (V2Error, OSError) as exc:
        print(f"config refused: {exc}", file=sys.stderr)
        return EXIT_UNLAUNCHABLE

    cohort = list(args.cohort)
    if len(set(cohort)) != len(cohort):
        print("duplicate protein id in the requested cohort", file=sys.stderr)
        return EXIT_UNLAUNCHABLE
    if not cohort and not (args.print_config or args.dry_run):
        # Requesting nothing and succeeding at it is not a completed run.  Reported here as a typed
        # refusal rather than left to argparse, so the message names the actual problem.
        print("the requested cohort is empty; nothing was asked for and nothing was produced",
              file=sys.stderr)
        return EXIT_FAILED

    try:
        declared = parse_declared_inputs(args.input_file, config=config)
        code_revision = resolve_code_revision(args.code_revision, config=config)
        budget_replicates = execution_replicates(args)
        production_depth_authorized, exploratory_depth_override = depth_authorization(
            args, config=config)
    except (V2Error, OSError) as exc:
        print(f"declared input refused: {exc}", file=sys.stderr)
        return EXIT_UNLAUNCHABLE
    # Parsed BEFORE the --print-config/--dry-run branch returns.  A gate that certified a launch
    # without looking at the runtime paths it would launch WITH is not a launch gate: a malformed
    # --shard-input was discovered only on the GPU, after the allocation the gate exists to
    # authorize had already been paid for.
    try:
        shard_inputs = parse_shard_inputs(args.shard_input)
    except V2Error as exc:
        print(f"shard input refused: {exc}", file=sys.stderr)
        return EXIT_UNLAUNCHABLE

    # Resolved BEFORE the --print-config/--dry-run branch returns, for the same reason the shard
    # inputs are.  --dry-run answers "may this be submitted", and a gate that never looked at the
    # Dual overlay it would launch WITH certified a launch it cannot make: a missing or malformed
    # overlay exited 0 here and failed on the GPU, after the allocation had been paid for.
    try:
        dual_overlay, dual_arm = resolve_dual(args, shard_inputs=shard_inputs)
        dual_signature = "" if dual_overlay is None else _dual_signature(dual_overlay, dual_arm)
    except (V2Error, OSError) as exc:
        print(f"dual overlay refused: {exc}", file=sys.stderr)
        return EXIT_UNLAUNCHABLE

    # A Dual run issues every Head batch twice, once per allele, over the overlay's OWN candidate
    # domain -- never the legacy per-cycle field, which is a different number by design.
    n_heads = 1 if dual_overlay is None else 2
    dual_candidate_ceiling = (
        None if dual_overlay is None
        else int(dual_overlay.max_counterfactual_sequences_per_cycle))

    # Refused at the GATE, not on the GPU. The mechanism/qualification path is a within-run matched
    # contrast between two INTERVENTIONS; allele arms are compared across runs, because the Dual arm
    # enters no seed and two runs differing only in --dual-arm share their depth-zero pool exactly.
    if dual_overlay is not None and (
            getattr(args, "mechanism_prefixes", None) or getattr(args, "dual_qualification", False)
            or getattr(args, "qualification", False)):
        print("launch gate refused: --dual-overlay cannot be combined with the mechanism or "
              "qualification path; run the cohort once per --dual-arm and compare the artifacts",
              file=sys.stderr)
        return EXIT_UNLAUNCHABLE

    if args.print_config or args.dry_run:
        try:
            payload = print_config_payload(
                config, n_proteins=len(cohort), declared_inputs=declared,
                code_revision=code_revision, execution_replicates=budget_replicates,
                n_heads=n_heads,
                counterfactual_sequences_per_cycle=dual_candidate_ceiling)
            payload["production_depth_authorized"] = production_depth_authorized
            payload["exploratory_depth_override"] = exploratory_depth_override
            # The resolved Dual mode is part of what a launch gate certifies, so it is printed
            # rather than inferred by the reader from the presence of a flag.
            payload["dual_mode"] = "none" if dual_overlay is None else "dual"
            payload["dual_arm"] = dual_arm or ""
            payload["dual_signature"] = dual_signature
        except (V2Error, OSError) as exc:
            print(f"preflight refused: {exc}", file=sys.stderr)
            return EXIT_UNLAUNCHABLE
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.dry_run:
            # --dry-run answers "may this be submitted", so it must fail on everything the run
            # would fail on.  A zero exit here is the evidence an operator submits on.
            # --print-config makes no such claim and stays lenient (see the module docstring).
            if not declared:
                print("dry run refused: no --input-file was declared, so the run has no content "
                      "identity to sign (PLAN §5.2 fails closed on missing content identity)",
                      file=sys.stderr)
                return EXIT_UNLAUNCHABLE
            try:
                assert_launch_feasible(project_v2_budget(
                    config, n_proteins=len(cohort),
                    execution_replicates=budget_replicates, n_heads=n_heads,
                    counterfactual_sequences_per_cycle=dual_candidate_ceiling))
            except V2PreflightError as exc:
                print(f"launch gate refused: {exc}", file=sys.stderr)
                return EXIT_UNLAUNCHABLE
        return EXIT_OK

    try:
        assert_launch_feasible(project_v2_budget(
            config, n_proteins=len(cohort), execution_replicates=budget_replicates,
            n_heads=n_heads,
                    counterfactual_sequences_per_cycle=dual_candidate_ceiling))
    except V2PreflightError as exc:
        print(f"launch gate refused: {exc}", file=sys.stderr)
        return EXIT_UNLAUNCHABLE

    try:
        expected = _signatures(config, cohort, arm_role=config.arm.arm_role,
                               code_revision=code_revision, inputs=declared,
                               production_depth_authorized=production_depth_authorized,
                               exploratory_depth_override=exploratory_depth_override,
                               dual_signature=dual_signature)
    except (V2Error, OSError) as exc:
        print(f"run signature refused: {exc}", file=sys.stderr)
        return EXIT_UNLAUNCHABLE

    # A Dual run's fragments carry three more tables, and ``aggregate_fragments`` collects
    # payload keys BY NAME: a name it is not given is silently dropped, which is how every Dual
    # shard's three tables were built and then discarded. A legacy run's tuple is unchanged.
    fragment_table_names = tuple(V2_TABLE_SCHEMAS) + (
        () if dual_overlay is None else tuple(DUAL_TABLE_SCHEMAS))

    if not args.aggregate_only:
        if runner is None:
            from scripts.rf_fusion_v2_cohort import (  # noqa: PLC0415
                run_v2_mechanism_shard,
                run_v2_shard,
            )

            runner = select_runner(args, ladder=run_v2_shard, mechanism=run_v2_mechanism_shard)
        if oracles_factory is None:
            # ``run_v2_shard`` refuses to build oracles implicitly, so that --dry-run can never
            # cost a GPU allocation.  The DRIVER is where the real stack is named: without this a
            # production launch failed with "no oracles_factory was supplied" after the config had
            # already been resolved, signed and budget-checked.
            from scripts.rf_fusion_v2_oracles import build_v2_oracles  # noqa: PLC0415

            oracles_factory = build_v2_oracles

        for protein_id in cohort:
            signature = expected[protein_id]
            fragment_path = fragment_dir / f"{protein_id}.json"
            if fragment_path.exists():
                # Content-bound resume: previously paid work is reused only when every scientific
                # condition still matches.  A stale fragment is REPLACED, not trusted.
                from scripts.rf_fusion_v2_resume import validate_fragment

                verdict = validate_fragment(fragment_path, expected=signature,
                                            table_names=fragment_table_names)
                # Only work that actually SUCCEEDED may be skipped.  Skipping on admissibility
                # alone would freeze a protein as failed for every future invocation -- one
                # preempted node and the cohort could never be completed by re-running it.
                if verdict.records_success:
                    continue
            try:
                runner_kwargs = dict(
                    protein_id=protein_id, config=config, signature=signature,
                    out_dir=out_dir, inputs=shard_inputs,
                    oracles_factory=oracles_factory,
                )
                if dual_overlay is not None:
                    # The overlay reached the SIGNATURE and stopped there: the shard still ran a
                    # single Head, so joint / a_only / b_only executed identical A-only V2 under
                    # three different signatures.  This is where the resolved arm reaches the
                    # engine that is supposed to differ between them.
                    runner_kwargs["dual"] = (dual_overlay, dual_arm)
                # Preserve the long-standing injected-runner surface for ordinary runs while
                # making the exceptional path explicit all the way into the cohort runner.
                if production_depth_authorized or exploratory_depth_override:
                    runner_kwargs.update(
                        production_depth_authorized=production_depth_authorized,
                        exploratory_depth_override=exploratory_depth_override,
                    )
                status, payload = runner(**runner_kwargs)
            except V2Error as exc:
                status, payload = "failed", {"error": f"{type(exc).__name__}: {exc}"}
            write_fragment(fragment_path, signature=signature, status=status, payload=payload)

    report = aggregate_fragments(
        fragment_dir, requested_cohort=cohort, expected_by_protein=expected,
        table_names=fragment_table_names,
    )

    # Realized cohort budget, computed from the ledger the run actually wrote.  A breach is a fact
    # about spend and stops the run; "unverifiable" is a fact about INSTRUMENTATION (some attempt
    # reported unknown_after_start) and is reported without failing the cohort -- PLAN §5.4 requires
    # that state to be persisted rather than resolved, and failing on it would make one preempted
    # attempt condemn an otherwise complete cohort.
    try:
        realized = realized_cap_verdict(report.ledger_events, caps=config.caps)
    except V2Error as exc:
        print(f"cost ledger refused: {exc}", file=sys.stderr)
        return EXIT_FAILED

    tables = {name: list(report.tables.get(name, [])) for name in V2_TABLE_SCHEMAS}
    # ``None`` on a legacy run, so its bundle gains no file (see DUAL_TABLE_SCHEMAS).
    dual_tables = None if dual_overlay is None else {
        name: list(report.tables.get(name, [])) for name in DUAL_TABLE_SCHEMAS}
    provenance = content_provenance(config, declared)
    write_v2_bundle(
        out_dir,
        dual_tables=dual_tables,
        manifest={
            **run_manifest(
                config=config, code_revision=code_revision,
                content_identities=_content_identities(provenance),
                seed_namespaces=("v2_depth0_root", "v2_lookahead", "a2_extra_lookahead",
                                 "matched_descendant"),
                production_depth_authorized=production_depth_authorized,
                exploratory_depth_override=exploratory_depth_override,
            ),
            # Per PLAN §5.2 ROLE, with the declared identity and the observed one side by side.
            # A digest that was never observed is null: "" would read as a value.
            "content_provenance": provenance,
            "input_signature": expected[cohort[0]].input_signature if cohort else None,
            # The cohort's own outcome belongs in the manifest: a bundle that recorded only what
            # succeeded could not be told apart from one where nothing else was ever requested.
            "requested_cohort": cohort,
            "accepted_fragments": [v.fragment_id for v in report.accepted],
            "rejected_fragments": [
                {"fragment_id": v.fragment_id, "status": v.status, "detail": v.detail}
                for v in report.rejected
            ],
            # Dual identity, published or absent -- never null. Without it a joint bundle and an
            # a_only bundle are indistinguishable on disk, and the arm is the whole experiment.
            **({} if dual_overlay is None else {
                "dual_mode": "dual",
                "dual_arm": dual_arm,
                "dual_signature": dual_signature,
                "dual_overlay_digest": dual_overlay.content_digest,
                "dual_calibration_digest": dual_overlay.calibration.content_digest,
                "dual_objective_mode": dual_overlay.calibration.law.mode.value,
                "dual_objective_tau": dual_overlay.calibration.law.tau,
                "dual_head_b_allele": dual_overlay.head_b_runtime.evaluator.allele,
                "dual_head_b_checkpoint_digest":
                    dual_overlay.head_b_runtime.evaluator.head_checkpoint_digest,
            }),
            "missing_proteins": list(report.missing_proteins),
            "n_ok": report.n_ok,
            # Recorded even when clean: an exit code says a run is unusable, it cannot say WHICH
            # budget it blew, and that decides whether the operator re-scopes the cohort or the
            # science.  ``null`` means no ledger was kept, which is NOT a compliance certificate.
            "realized_caps": (
                None if realized is None
                else {"within": bool(realized.within), "breached": list(realized.breached),
                      "unverifiable": list(realized.unverifiable), "detail": realized.detail}
            ),
        },
        tables=tables,
        # PLAN §5.3 lists the compute ledger as a REQUIRED evidence object and PLAN §5.4 routes it
        # through resume.  Aggregation already collected it and the writer already accepted it;
        # nothing joined the two, so `cost_ledger.jsonl` was never produced and a run could not say
        # what it burned -- least of all the attempts that started and were never measured.
        ledger_events=report.ledger_events,
    )

    for verdict in report.rejected:
        print(f"fragment rejected [{verdict.status}] {verdict.fragment_id}: {verdict.detail}",
              file=sys.stderr)
    for protein_id in report.missing_proteins:
        print(f"missing protein: {protein_id}", file=sys.stderr)

    if realized is not None and realized.breached:
        print(f"declared hard cap breached over the cohort: {realized.detail}", file=sys.stderr)
        return EXIT_FAILED
    if realized is not None and realized.unverifiable:
        print(f"budget could not be certified: {realized.detail}", file=sys.stderr)

    return decide_exit_code(report, requested=len(cohort))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
