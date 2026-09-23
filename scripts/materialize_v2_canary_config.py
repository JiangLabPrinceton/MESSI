"""Materialize a content-bound Fusion V2 configuration and launch arguments."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__all__ = [
    "build_parser", "main",
    "file_digest", "head_config_hash", "no_constraint_manifest_digest",
    "resolve_content_bindings", "fill_config",
]


# This is deliberately one named, closed exploratory profile rather than a bag of schedule flags.
# It is an unblinded capability run, not another way to edit the V2F5A qualification cell.  The
# phase/split identity is therefore part of the materialized config, while the separate runtime
# launch override remains required by the driver.
EXPLORATORY_PROFILES: dict[str, dict[str, Any]] = {
    "uricase_d4_k12_r40": {
        "phase": "capability_ladder",
        "split_role": "exploratory_uricase",
        "schedule": {
            "schedule_id": "exploratory-uricase-d4-k12-r40-v1",
            "coordinate_law": "progressive_checkpoint",
            "depth_cap": 4,
            "active_population_width": 1,
            "min_lookahead_tail_steps": 10,
            "points": [
                {"depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 60,
                 "n_lookaheads": 12, "band_key": "step40"},
                {"depth": 1, "r_step": 40, "c_source_step": 60, "c_next_step": 70,
                 "n_lookaheads": 12, "band_key": "step40"},
                {"depth": 2, "r_step": 40, "c_source_step": 70, "c_next_step": 80,
                 "n_lookaheads": 12, "band_key": "step40"},
                {"depth": 3, "r_step": 40, "c_source_step": 80, "c_next_step": 90,
                 "n_lookaheads": 12, "band_key": "step40"},
            ],
        },
        # Exact plan: 1,990 logical DFE and 60 definitive endpoint refolds per protein.  These are
        # close fail-closed ceilings with small operational slack, not the template's broad cohort
        # qualification caps.  Head calls remain a CLI-bound quantity because the counterfactual
        # ceiling is calibrated from the realized cohort's editable domains.
        "caps": {
            "max_logical_dfe": 2200,
            "max_definitive_refolds": 64,
        },
    },
    "highrisk_d4_k12_r40": {
        "phase": "capability_ladder",
        "split_role": "exploratory_highrisk_ceiling_v2",
        "support_policy_version": "v2",
        "schedule": {
            "schedule_id": "highrisk-d4-k12-r40-global-v2",
            "coordinate_law": "progressive_checkpoint",
            "depth_cap": 4,
            "active_population_width": 1,
            "min_lookahead_tail_steps": 10,
            "points": [
                {"depth": depth, "r_step": 40, "c_source_step": 50 + 10 * depth,
                 "c_next_step": 60 + 10 * depth, "n_lookaheads": 12,
                 "band_key": "step40"}
                for depth in range(4)
            ],
        },
        "caps": {
            "max_logical_dfe": 2200,
            "max_definitive_refolds": 64,
        },
    },
    "testset_d4_k12_r40": {
        "phase": "capability_ladder",
        "split_role": "exploratory_testset_design_v1",
        "support_policy_version": "v2",
        "schedule": {
            "schedule_id": "testset-d4-k12-r40-global-v1",
            "coordinate_law": "progressive_checkpoint",
            "depth_cap": 4,
            "active_population_width": 1,
            "min_lookahead_tail_steps": 10,
            "points": [
                {"depth": depth, "r_step": 40, "c_source_step": 50 + 10 * depth,
                 "c_next_step": 60 + 10 * depth, "n_lookaheads": 12,
                 "band_key": "step40"}
                for depth in range(4)
            ],
        },
        # The generic single-allele test-set profile freezes the algorithm proven by the
        # high-risk D4/K12 campaign but carries an honest, non-high-risk split identity.  The
        # protocol/case binds the actual cohort and full-data Head; those are content inputs, not
        # reasons to duplicate this schedule under one profile name per allele or subset.
        "caps": {
            "max_logical_dfe": 2200,
            "max_definitive_refolds": 64,
        },
    },
    "uricase_core0_pilot_d2_k6_r40": {
        "phase": "capability_ladder",
        "split_role": "exploratory_uricase_core0_pilot_v1",
        "support_policy_version": "v2",
        "schedule": {
            "schedule_id": "uricase-core0-pilot-d2-k6-r40-v1",
            "coordinate_law": "progressive_checkpoint",
            "depth_cap": 2,
            "active_population_width": 1,
            "min_lookahead_tail_steps": 10,
            "points": [
                {"depth": depth, "r_step": 40, "c_source_step": 50 + 10 * depth,
                 "c_next_step": 60 + 10 * depth, "n_lookaheads": 6,
                 "band_key": "step40"}
                for depth in range(2)
            ],
        },
        # Nominal 820 + at most 7*50 recapture DFE = 1170; 10% rounded slack -> 1290.
        "caps": {
            "max_logical_dfe": 1290,
            "max_definitive_refolds": 20,
            "max_retries": 7,
        },
    },
    "uricase_core0_pilot_d3_k12_r40": {
        "phase": "capability_ladder",
        "split_role": "exploratory_uricase_core0_pilot_v1",
        "support_policy_version": "v2",
        "schedule": {
            "schedule_id": "uricase-core0-pilot-d3-k12-r40-v1",
            "coordinate_law": "progressive_checkpoint",
            "depth_cap": 3,
            "active_population_width": 1,
            "min_lookahead_tail_steps": 10,
            "points": [
                {"depth": depth, "r_step": 40, "c_source_step": 50 + 10 * depth,
                 "c_next_step": 60 + 10 * depth, "n_lookaheads": 12,
                 "band_key": "step40"}
                for depth in range(3)
            ],
        },
        # Nominal 1820 + at most 7*50 recapture DFE = 2170; 10% rounded slack -> 2390.
        "caps": {
            "max_logical_dfe": 2390,
            "max_definitive_refolds": 53,
            "max_retries": 7,
        },
    },
    "highrisk_d4_k24_r40": {
        "phase": "capability_ladder",
        "split_role": "exploratory_highrisk_breadth_ceiling_v1",
        "support_policy_version": "v2",
        "schedule": {
            "schedule_id": "highrisk-d4-k24-r40-global-v1",
            "coordinate_law": "progressive_checkpoint",
            "depth_cap": 4,
            "active_population_width": 1,
            "min_lookahead_tail_steps": 10,
            "points": [
                {"depth": depth, "r_step": 40, "c_source_step": 50 + 10 * depth,
                 "c_next_step": 60 + 10 * depth, "n_lookaheads": 24,
                 "band_key": "step40"}
                for depth in range(4)
            ],
        },
        "caps": {
            "max_logical_dfe": 4200,
            "max_definitive_refolds": 128,
        },
    },
    "highrisk_d8_k32_r40": {
        "phase": "capability_ladder",
        "split_role": "exploratory_highrisk_ceiling_v1",
        "support_policy_version": "v2",
        "schedule": {
            "schedule_id": "highrisk-d8-k32-r40-global-v1",
            "coordinate_law": "progressive_checkpoint",
            "depth_cap": 8,
            "active_population_width": 1,
            "min_lookahead_tail_steps": 10,
            "points": [
                {"depth": depth, "r_step": 40, "c_source_step": 50 + 5 * depth,
                 "c_next_step": 55 + 5 * depth, "n_lookaheads": 32,
                 "band_key": "step40"}
                for depth in range(8)
            ],
        },
        # Engineering ceilings only. This campaign asks for capability, not matched compute.
        "caps": {
            "max_logical_dfe": 20000,
            "max_head_calls": 20000,
            "max_definitive_refolds": 1024,
            "max_gpu_seconds": 604800,
            "max_walltime_s": 604800,
        },
    },
}


class MaterializeError(RuntimeError):
    """A resolved config this producer refuses to write."""


def file_digest(path: Any) -> str:
    """SHA-256 over a file's bytes -- the same function the driver signs declared inputs with."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _structure_runtime_protocol(args) -> dict[str, Any]:
    """Validate the explicit ESMFold2 selector/protocol required by an exploratory run."""
    model = getattr(args, "esmfold2_model", None)
    if not isinstance(model, str) or not model.strip():
        raise MaterializeError(
            "--esmfold2-model is required by an exploratory profile; relying on v0's parser "
            "default would leave the executed structure model outside the run identity"
        )
    values: dict[str, int] = {}
    for field, flag, minimum in (
        ("num_loops", "--esmfold2-num-loops", 1),
        ("num_sampling_steps", "--esmfold2-num-sampling-steps", 1),
        ("num_diffusion_samples", "--esmfold2-num-diffusion-samples", 1),
        ("seed", "--esmfold2-seed", 0),
    ):
        value = getattr(args, f"esmfold2_{field}", None)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise MaterializeError(
                f"{flag} must be an integer >= {minimum} for an exploratory profile, got "
                f"{value!r}"
            )
        values[field] = int(value)
    return {"model_selector": model.strip(), "protocol": values}


def _structure_runtime_bytes(payload: dict[str, Any]) -> bytes:
    """Canonical bytes signed by both the config content row and the run input signature."""
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _structure_runtime_identity(args) -> tuple[Path, dict[str, Any], str]:
    """Bind the realized ESMFold2 selector/protocol to the supplied local model snapshot.

    The selector may be a registry id (for example ``biohub/ESMFold2``), so this producer cannot
    prove that the registry name resolves to the supplied local file.  It can, however, make both
    immutable inputs part of ONE signed content artifact.  The worker's realized metadata remains
    the runtime proof of how that selector resolved on the cluster.
    """
    declared = _structure_runtime_protocol(args)
    snapshot = Path(args.structure_backend)
    if not snapshot.is_file():
        raise MaterializeError(
            f"--structure-backend must be the local ESMFold2 model snapshot file, got {snapshot}"
        )
    payload = {
        "schema_version": "v2-esmfold2-runtime-1",
        "backend": "esmfold2_live",
        **declared,
        "local_model_snapshot_sha256": file_digest(snapshot),
    }
    raw = _structure_runtime_bytes(payload)
    path = Path(args.out).with_suffix(".structure_runtime.json")
    return path, payload, hashlib.sha256(raw).hexdigest()


def head_config_hash(config_dir: Any) -> str:
    """The Head config directory's realized identity, read from the ONE implementation.

    `OnlineHeadScorer` reports this number as `head_config_hash`, and the V2 runtime compares the
    reference's evaluator identity against every endpoint's.  A second rolling hash here that
    agreed today would be a second implementation to drift tomorrow, so the producer imports v0's.
    """
    from scripts.head_runtime import _compute_head_config_hash

    return _compute_head_config_hash(Path(config_dir))


def no_constraint_manifest_digest() -> str:
    """The project's canonical digest for "this cohort declares NO constraint manifest".

    An unconstrained protein has no manifest file, but PLAN §5.2 gives `constraint_manifest` no
    exemption: a run whose conditioning is simply silent about constraints cannot be told apart
    from one whose manifest went missing.  `rho_maturity_scan` already answered this -- it records
    a typed ABSENCE digest, and `5ZHV_B`'s own `B(r)` artifact carries it -- so the resolved config
    declares that same number rather than inventing a second convention or hashing a file written
    only to be hashed.  Being frozen, it also makes `--input-file constraint_manifest=...` refuse
    any manifest on an unconstrained cell.
    """
    from scripts.rho_maturity_scan import _canonical_digest

    return _canonical_digest({"schema": "scan-constraint-binding/1", "constraint_manifest": None})


def _band_content_digest(band_json: Any, *, r_step: int, stratum_key: str) -> str:
    """The band TABLE's `calibration_content_digest`, and proof the requested cell exists.

    Runbook §3 note 2: this role's `expected_sha256` is the table's canonical content digest, NOT
    the file's sha256.  `make_band_table` rebinds it and `q_phi` compares against it; the file
    digest would refuse every real run.
    """
    payload = json.loads(Path(band_json).read_text(encoding="utf-8"))
    provenance = payload.get("provenance") or {}
    digest = provenance.get("calibration_content_digest")
    if not digest:
        raise MaterializeError(f"{band_json} carries no provenance.calibration_content_digest")
    cells = [band for band in payload.get("bands") or ()
             if int(band.get("step", -1)) == int(r_step)
             and str(band.get("stratum_key")) == str(stratum_key)]
    if not cells:
        available = sorted({(int(b.get("step", -1)), str(b.get("stratum_key")))
                            for b in payload.get("bands") or ()})
        raise MaterializeError(
            f"{band_json} has no band at (step={r_step}, stratum_key={stratum_key!r}); it carries "
            f"{available}.  lookup_band is exact and never interpolates, so this cell would yield "
            "a typed null for every transition -- knowable now rather than after the GPU time"
        )
    return str(digest)


def _delta_new_block(hotspot_json: Any, *, head: dict) -> dict:
    """The `delta_new` block, copied VERBATIM, after checking it was measured on THIS Head domain."""
    payload = json.loads(Path(hotspot_json).read_text(encoding="utf-8"))
    block = payload.get("delta_new")
    if not isinstance(block, dict):
        raise MaterializeError(f"{hotspot_json} carries no delta_new block")
    artifact = block.get("artifact") or {}
    declared = (str(head["allele"]), str(head["score_scale"]),
                int(head["window_k_min"]), int(head["window_k_max"]))
    measured = (str(artifact.get("allele")), str(artifact.get("score_scale")),
                int(artifact.get("window_k_min", -1)), int(artifact.get("window_k_max", -1)))
    if declared != measured:
        names = ("allele", "score_scale", "window_k_min", "window_k_max")
        diff = ", ".join(f"{n}: config {d!r} != artifact {m!r}"
                         for n, d, m in zip(names, declared, measured) if d != m)
        raise MaterializeError(
            f"the hotspot threshold in {hotspot_json} was measured under a different Head domain "
            f"({diff}).  N_H^whole is a maximum over the grid these fields define, so a threshold "
            "measured under one domain does not bound designs scored under another"
        )
    return json.loads(json.dumps(block))  # a plain, YAML-safe copy


def resolve_content_bindings(
    args, *, structure_runtime_identity: tuple[Path, dict[str, Any], str] | None = None,
) -> tuple[dict[str, str | None], dict[str, str]]:
    """`role -> frozen digest or None`, plus `role -> the runtime file that supplies it`.

    A role appears in exactly one of the two.  The split is not cosmetic: a frozen role is bound
    before the run and a supplied file that contradicts it is refused, while a runtime role is
    signed by whatever bytes actually arrived.  Three roles here are frozen for a reason no file
    can express -- the Head config is a DIRECTORY rolling-hash, the band is a table digest rather
    than a file digest, and an unconstrained cell's constraint manifest is a typed absence.
    """
    constrained = bool(args.constraint_manifest)
    frozen: dict[str, str] = {
        "code_revision": str(args.code_revision),
        # THIS protein's canonical sequence file, not the manifest that resolves it.  A Canary cell
        # is one protein, and `bind_cumulative_reference` compares the digest it recomputes from
        # the reference BYTES against `policy.complete_reference_content_digest`, which is exactly
        # this field -- so a manifest digest here can never equal it and the shard dies with
        # `ReferenceRebindAttempt` after the checkpoints are resident.  The cohort-level manifest
        # is a resolution table and binds separately, as `reference_sequences`.
        "complete_reference_sequence": file_digest(args.reference_sequence),
        "projection_policy_spec": file_digest(args.projection_policy_spec),
        "schedule_band_calibration": _band_content_digest(
            args.band_json, r_step=int(args.r_step), stratum_key=str(args.stratum_key)),
        "head_config": head_config_hash(args.head_config_dir),
        "head_checkpoint": file_digest(args.head_checkpoint),
    }
    runtime: dict[str, str] = {
        "cohort_table": str(args.cohort_table),
        # Plural: the cohort's reference TABLE, which is what `complete_reference_manifest` reads.
        "reference_sequences": str(args.reference_manifest),
        "backbone": str(args.backbone),
        # The mask is DERIVED from these coordinates.  `_conditioning` replaces this digest with
        # the realized per-protein mask, so the file records where the mask came from.
        "coordinate_mask": str(args.backbone),
        "rf_sampler_config": str(args.rf_sampler_config),
        "dplm_checkpoint": str(args.dplm_checkpoint),
        # Likewise: the alphabet lives in the checkpoint, and `_conditioning` replaces this with
        # the realized tokenizer digest.
        "tokenizer": str(args.dplm_checkpoint),
        "structure_config": str(args.structure_config),
        "v0_structure_gate_config": str(args.v0_structure_gate_config),
    }
    if getattr(args, "exploratory_profile", None):
        identity = structure_runtime_identity or _structure_runtime_identity(args)
        # Unlike an ordinary Canary, the recursive sandbox refuses v0 defaults.  Its config digest
        # freezes a canonical artifact containing BOTH the local snapshot digest and the actual
        # selector/protocol handed to the worker.
        frozen["structure_backend"] = identity[2]
    else:
        # Existing D1 behavior: the supplied local weight file itself is the runtime-bound role.
        runtime["structure_backend"] = str(args.structure_backend)
    if constrained:
        runtime["constraint_manifest"] = str(args.constraint_manifest)
        runtime["fixed_token_policy"] = str(args.constraint_manifest)
    else:
        absence = no_constraint_manifest_digest()
        frozen["constraint_manifest"] = absence
        frozen["fixed_token_policy"] = absence
    return frozen, runtime


def fill_config(template: dict, *, args, frozen: dict, runtime: dict) -> dict:
    """Return the resolved config; refuse if any `REPLACE_*` placeholder would survive."""
    config = json.loads(json.dumps(template))  # deep copy without YAML aliases

    config["identity"]["code_revision"] = str(args.code_revision)
    # The campaign names the EXPERIMENT.  Two runs that share a template but answer different
    # questions -- the Canary's "does it execute", the mechanism cohort's "does it transmit" --
    # must not sign their artifacts under one campaign, or the second reads as more of the first.
    if getattr(args, "campaign_id", None):
        config["identity"]["campaign_id"] = str(args.campaign_id)
    master_seed = getattr(args, "master_seed", None)
    if master_seed is not None:
        if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
            raise MaterializeError("--master-seed must be an integer >= 0")
        config["identity"]["master_seed"] = int(master_seed)

    profile_name = getattr(args, "exploratory_profile", None)
    profile = EXPLORATORY_PROFILES.get(profile_name) if profile_name else None
    if profile_name and profile is None:
        raise MaterializeError(
            f"unknown --exploratory-profile {profile_name!r}; allowed values are "
            f"{sorted(EXPLORATORY_PROFILES)}"
        )
    if profile is None and getattr(args, "run_max_head_calls", None) is not None:
        raise MaterializeError(
            "--run-max-head-calls requires an explicit --exploratory-profile; an ordinary "
            "Canary/qualification materialization may not silently consume or ignore an "
            "exploratory run cap"
        )
    if profile is not None:
        _structure_runtime_protocol(args)
        if int(args.r_step) != 40:
            raise MaterializeError(
                f"exploratory profile {profile_name!r} requires --r-step 40, got {args.r_step}; "
                "every depth is bound to the declared B(40) cell"
            )
        substrate = config.get("substrate") or {}
        required_substrate = {
            "n_steps": 100,
            "amplification_form": "constant_one",
            "controller_enabled": False,
            # Keep the post-step lifecycle but set the remask fraction to zero: this is the
            # project's frozen no-remask spelling, not remask_enabled=false.
            "remask_enabled": True,
            "remask_fraction_scale": 0.0,
        }
        mismatch = {
            key: (substrate.get(key), value)
            for key, value in required_substrate.items() if substrate.get(key) != value
        }
        if mismatch:
            raise MaterializeError(
                f"exploratory profile {profile_name!r} requires the frozen 100-step null/no-remask "
                f"substrate, but the template disagrees: {mismatch}"
            )

    policy_calibration = getattr(args, "policy_calibration_json", None)
    if profile is not None and not policy_calibration:
        raise MaterializeError(
            f"exploratory profile {profile_name!r} requires --policy-calibration-json; "
            "head_directed_capped may not run from invented thresholds"
        )
    if policy_calibration:
        calibration_path = Path(policy_calibration)
        if not calibration_path.is_file():
            raise MaterializeError(
                f"policy calibration artifact does not exist or is not a file: "
                f"{calibration_path}"
            )
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "v2-head-policy-calibration-bundle/1":
            raise MaterializeError(
                f"{policy_calibration} is not a v2-head-policy-calibration-bundle/1 artifact"
            )
        block = payload.get("head_directed")
        if not isinstance(block, dict):
            raise MaterializeError(f"{policy_calibration} carries no head_directed config block")
        measured_head = payload.get("head")
        expected_head = {
            "allele": config["head"]["allele"],
            "score_scale": config["head"]["score_scale"],
            "window_k_min": int(config["head"]["window_k_min"]),
            "window_k_max": int(config["head"]["window_k_max"]),
            "head_config_hash": frozen["head_config"],
            "head_checkpoint_digest": frozen["head_checkpoint"],
        }
        if not isinstance(measured_head, dict):
            raise MaterializeError(
                f"{policy_calibration} carries no frozen Head evaluator identity"
            )
        mismatch = {
            key: (measured_head.get(key), value)
            for key, value in expected_head.items() if measured_head.get(key) != value
        }
        if mismatch:
            raise MaterializeError(
                f"{policy_calibration} was measured on a different Head instrument/domain: "
                f"{mismatch}"
            )
        if payload.get("policy_spec_sha256") != frozen["projection_policy_spec"]:
            raise MaterializeError(
                f"{policy_calibration} was bound to policy spec "
                f"{payload.get('policy_spec_sha256')!r}, but this cell supplies "
                f"{frozen['projection_policy_spec']!r}"
            )
        config["identity"]["phase"] = (
            str(profile["phase"]) if profile is not None else "policy_qualification"
        )
        config["identity"]["split_role"] = (
            str(profile["split_role"]) if profile is not None else "policy_qualification"
        )
        config["projection"].update({
            "support_policy_id": "head_directed_capped",
            "support_policy_version": str(
                profile.get("support_policy_version", "v1") if profile is not None else "v1"),
            "support_policy_is_diagnostic": False,
            "head_directed": json.loads(json.dumps(block)),
        })
        head_cap = (
            getattr(args, "run_max_head_calls", None)
            if profile is not None
            else getattr(args, "qualification_max_head_calls", None)
        )
        if isinstance(head_cap, bool) or not isinstance(head_cap, int) or head_cap < 1:
            flag = (
                "--run-max-head-calls" if profile is not None
                else "--qualification-max-head-calls"
            )
            raise MaterializeError(
                f"{flag} is required with --policy-calibration-json"
                + (" for an explicit exploratory run Head cap" if profile is not None else
                   "; the one-cycle paired cohort must carry an explicit cohort Head cap")
            )
        config["caps"]["max_head_calls"] = int(head_cap)

    if profile is not None:
        config["schedule"] = json.loads(json.dumps(profile["schedule"]))
        config["caps"].update(json.loads(json.dumps(profile["caps"])))
    else:
        points = config["schedule"]["points"]
        if len(points) != 1:
            raise MaterializeError(
                f"the template declares {len(points)} schedule points; a Canary cell is ONE "
                "(protein, r_step) and a multi-point template cannot be resolved into one"
            )
        points[0]["r_step"] = int(args.r_step)
        points[0]["band_key"] = f"step{int(args.r_step)}"

    config["safety"]["delta_new_cumulative"] = _delta_new_block(
        args.hotspot_json, head=config["head"])

    for row in config["content"]:
        role = row["role"]
        if role in frozen:
            row["binding"], row["expected_sha256"] = "frozen", frozen[role]
        elif role in runtime:
            row["binding"], row["expected_sha256"] = "runtime", None
        else:
            raise MaterializeError(
                f"content role {role!r} was bound neither frozen nor runtime; PLAN §5.2 fails "
                "closed on missing content identity"
            )

    leftovers = sorted(_placeholders(config))
    if leftovers:
        raise MaterializeError(
            f"unresolved placeholder(s) remain: {leftovers}.  The loader would refuse them at "
            "launch; refusing here names the field instead"
        )
    return config


def _placeholders(node: Any, path: str = "") -> list[str]:
    if isinstance(node, dict):
        return [item for key, value in node.items()
                for item in _placeholders(value, f"{path}.{key}" if path else str(key))]
    if isinstance(node, list):
        return [item for index, value in enumerate(node)
                for item in _placeholders(value, f"{path}[{index}]")]
    return [path] if isinstance(node, str) and node.startswith("REPLACE") else []


def _args_script(
    args, *, frozen: dict, runtime: dict, config_path: Path,
    structure_runtime_identity: tuple[Path, dict[str, Any], str] | None = None,
) -> str:
    """The driver argument vectors this config requires, as a `source`-able shell fragment.

    `--input-file ROLE=PATH` signs the run and is checked against any frozen digest; `--shard-input
    NAME=PATH` is what the oracle stack and the conditioning identity actually read.  A role can
    need both, under two different names (`dplm_checkpoint` / `base_if_checkpoint`), which is
    exactly the kind of thing a hand-written command gets wrong once and then copies four times.

    `schedule_band_calibration`, `head_config`, `fixed_token_policy` and an unconstrained
    `constraint_manifest` are deliberately absent from `--input-file`: their frozen digests are a
    table digest, a directory rolling-hash, a policy label and a typed absence, so the driver's
    file-sha256 check would refuse the very files that produced them.
    """
    input_files = [
        # Frozen roles whose digest IS the file's sha256 -- these are the ones the driver can
        # genuinely verify, so declaring them turns the run signature into a real check.
        f"complete_reference_sequence={args.reference_sequence}",
        f"projection_policy_spec={args.projection_policy_spec}",
        f"head_checkpoint={args.head_checkpoint}",
    ] + [f"{role}={path}" for role, path in sorted(runtime.items())
         # coordinate_mask/tokenizer alias another role's file; recording them twice in the
         # manifest would claim two content identities where one file was read.
         if role not in {"coordinate_mask", "tokenizer", "fixed_token_policy"}]

    exploratory = bool(getattr(args, "exploratory_profile", None))
    if exploratory:
        identity = structure_runtime_identity or _structure_runtime_identity(args)
        input_files.append(f"structure_backend={identity[0]}")

    shard_inputs = dict(runtime)
    shard_inputs.update({
        # Names the oracle stack owns, distinct from the role vocabulary on purpose.
        "base_if_checkpoint": str(args.dplm_checkpoint),
        "test_set_parquet": str(args.cohort_table),
        "pdb_root": str(args.pdb_root),
        "refold_cache_dir": str(args.refold_cache_dir),
        "head_config": str(args.head_config_dir),
        "head_checkpoint": str(args.head_checkpoint),
        "head_variant_id": str(args.head_variant_id),
        "esmfold2_site_packages": str(args.esmfold2_site_packages),
        "schedule_band_calibration": str(args.band_json),
        "complete_reference_manifest": str(args.reference_manifest),
        "protein_stratum_manifest": str(args.stratum_manifest),
    })
    if getattr(args, "dual_overlay", None) is not None:
        # PATHS only. ``variant_id``, ``allele_idx`` and ``window_batch_size`` are carried by the
        # SIGNED OVERLAY and read from there; requiring them here made the operator declare three
        # values that nothing read -- an inert knob is worse than a missing one, because it reads
        # as a control. A checkpoint path is a property of one filesystem and cannot live in the
        # signed overlay, which is why these two do stay here.
        required = {
            "head_b_config": args.head_b_config_dir,
            "head_b_checkpoint": args.head_b_checkpoint,
        }
        absent = sorted(name for name, value in required.items() if value is None)
        if absent:
            raise SystemExit(
                f"--dual-overlay requires {sorted(required)}; missing {absent}. Neither is "
                "defaulted: a silent Head-B binding would be a second instrument nobody declared"
            )
        shard_inputs.update({name: str(value) for name, value in required.items()})
        shard_inputs["dual_overlay"] = str(args.dual_overlay)
    if exploratory:
        identity = structure_runtime_identity or _structure_runtime_identity(args)
        protocol = _structure_runtime_protocol(args)
        shard_inputs.update({
            # The oracle consumes these exact names.  They are repeated in the signed runtime
            # identity artifact above, so changing a worker knob changes both config_digest and
            # run input_signature rather than silently falling back to v0's parser defaults.
            "structure_backend": str(identity[0]),
            "esmfold2_model": str(protocol["model_selector"]),
            "esmfold2_num_loops": str(protocol["protocol"]["num_loops"]),
            "esmfold2_num_sampling_steps": str(
                protocol["protocol"]["num_sampling_steps"]),
            "esmfold2_num_diffusion_samples": str(
                protocol["protocol"]["num_diffusion_samples"]),
            "esmfold2_seed": str(protocol["protocol"]["seed"]),
        })
    # `fixed_token_policy` stays in SHARD_INPUTS even though no oracle reads that key: it is one of
    # the eighteen roles `_conditioning` must find a digest for, and on an anchored cell it is bound
    # RUNTIME, so dropping it made the shard refuse with "no content identity for role(s)
    # ['fixed_token_policy']".  On an unconstrained cell it is frozen and absent from `runtime`, so
    # it never appears here at all.

    lines = [
        "# GENERATED by scripts/materialize_v2_canary_config.py -- do not edit by hand.",
        f"# cell: {args.protein_id} r={int(args.r_step)}  stratum={args.stratum_key}",
        f"V2_CONFIG={shlex.quote(str(config_path))}",
        f"V2_COHORT={shlex.quote(str(args.protein_id))}",
        # A named alias for the same value. The assembly preflight takes --cell CONFIG=PROTEIN_ID
        # and the launcher cross-checks the Dual mode, and both should read a variable that says
        # what it is rather than infer the protein from a filename.
        f"V2_PROTEIN_ID={shlex.quote(str(args.protein_id))}",
        # `none` for every non-Dual cell. Emitted unconditionally because the launcher runs under
        # `set -u`, so a Dual-aware launcher reading an unset variable would abort on legacy cells.
        # CAPABILITY, not arm. The arm is a LAUNCH flag: one resolved config is launched once per
        # arm, and the arms are compared across runs. Baking an arm in here would force one
        # materialized cell per arm, which is the per-arm branching this design does not have.
        f"DUAL_MODE={shlex.quote('dual' if getattr(args, 'dual_overlay', None) else 'none')}",
        "INPUT_FILES=(",
        *[f"  {shlex.quote(item)}" for item in input_files],
        ")",
        "SHARD_INPUTS=(",
        *[f"  {shlex.quote(f'{name}={path}')}" for name, path in sorted(shard_inputs.items())],
        ")",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="materialize_v2_canary_config",
        description="resolve one V2 Canary cell's config + launch arguments from measured artifacts",
    )
    parser.add_argument("--template", required=True, help="tracked V2 Canary config template")
    parser.add_argument("--out", required=True, help="resolved config destination (.yaml)")
    parser.add_argument("--protein-id", required=True)
    parser.add_argument("--r-step", required=True, type=int)
    parser.add_argument("--stratum-key", required=True,
                        help="the COHORT stratum the band was measured over; not the band_key")
    parser.add_argument("--code-revision", required=True, help="7-64 lowercase hex; 'unknown' is refused")
    parser.add_argument("--band-json", required=True, help="this stratum's B(r) artifact")
    parser.add_argument("--hotspot-json", required=True, help="this protein's hotspot calibration")
    parser.add_argument("--reference-manifest", required=True, help="{protein_id: {path, sha256}}")
    parser.add_argument("--stratum-manifest", required=True, help="{protein_id: stratum_key}")
    parser.add_argument("--reference-sequence", required=True, help="this protein's .seq")
    parser.add_argument("--projection-policy-spec", required=True)
    parser.add_argument("--head-config-dir", required=True)
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--head-variant-id", required=True)
    # Optional dual-allele overlay. Absent by default: a cell materialized without these emits the
    # same .args.sh arrays it emits today, with DUAL_MODE=none.
    parser.add_argument("--dual-overlay", default=None,
                        help="path to the signed dual-allele overlay JSON")
    # No --dual-arm here. The arm is a LAUNCH flag, so one resolved config is launched once per
    # arm and the arms are compared across runs; baking an arm in would force one cell per arm.
    # No --head-b-variant-id / --head-b-allele-idx / --head-b-window-batch-size either: those are
    # carried by the SIGNED overlay and read from there, so accepting them here would be three
    # knobs that read as controls and change nothing.
    parser.add_argument("--head-b-config-dir", default=None)
    parser.add_argument("--head-b-checkpoint", default=None)
    parser.add_argument("--structure-backend", required=True,
                        help="the structure model's weight file (its bytes ARE the backend)")
    parser.add_argument("--structure-config", required=True)
    parser.add_argument("--v0-structure-gate-config", required=True)
    parser.add_argument("--rf-sampler-config", required=True)
    parser.add_argument("--dplm-checkpoint", required=True)
    parser.add_argument("--cohort-table", required=True, help="the test-set parquet")
    parser.add_argument("--backbone", required=True, help="this protein's structure file")
    parser.add_argument("--pdb-root", required=True)
    parser.add_argument("--refold-cache-dir", required=True)
    parser.add_argument("--esmfold2-site-packages", required=True)
    parser.add_argument("--constraint-manifest", default=None,
                        help="omit for an unconstrained cell; its absence is then declared, "
                             "not left silent")
    parser.add_argument("--campaign-id", default=None,
                        help="override identity.campaign_id (default: the template's).  Use a "
                             "distinct campaign for a distinct question, e.g. the mechanism cohort")
    parser.add_argument("--master-seed", type=int, default=None,
                        help="override identity.master_seed for an independent seeded lineage")
    parser.add_argument(
        "--policy-calibration-json", default=None,
        help="optional v2-head-policy-calibration-bundle/1 artifact. When supplied, materialize "
             "a head_directed_capped config instead of the template's diagnostic policy. The "
             "default identity is policy_qualification; an explicit exploratory profile replaces "
             "it with that profile's capability identity. Calibrated values are copied verbatim "
             "and validated by the typed config loader",
    )
    parser.add_argument(
        "--qualification-max-head-calls", type=int, default=None,
        help="cohort Head-call hard cap written into a policy-qualification config; required with "
             "--policy-calibration-json and ignored otherwise",
    )
    parser.add_argument(
        "--exploratory-profile", choices=sorted(EXPLORATORY_PROFILES), default=None,
        help="explicitly materialize one closed, non-confirmatory capability-ladder profile. "
             "This only changes config identity/schedule; the driver still requires its separate "
             "exploratory D>1 launch override",
    )
    parser.add_argument(
        "--run-max-head-calls", type=int, default=None,
        help="generic whole-run Head hard cap required by an exploratory profile. It is not a "
             "qualification-arm cap and is rejected unless --exploratory-profile is supplied",
    )
    parser.add_argument("--esmfold2-model", default=None,
                        help="actual ESMFold2 model selector; required by an exploratory profile")
    parser.add_argument("--esmfold2-num-loops", type=int, default=None)
    parser.add_argument("--esmfold2-num-sampling-steps", type=int, default=None)
    parser.add_argument("--esmfold2-num-diffusion-samples", type=int, default=None)
    parser.add_argument("--esmfold2-seed", type=int, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)

    template = yaml.safe_load(Path(args.template).read_text(encoding="utf-8"))
    structure_identity = (
        _structure_runtime_identity(args) if args.exploratory_profile else None
    )
    frozen, runtime = resolve_content_bindings(
        args, structure_runtime_identity=structure_identity)
    config = fill_config(template, args=args, frozen=frozen, runtime=runtime)

    out.parent.mkdir(parents=True, exist_ok=True)
    if structure_identity is not None:
        identity_path, identity_payload, identity_digest = structure_identity
        identity_path.write_bytes(_structure_runtime_bytes(identity_payload))
        if file_digest(identity_path) != identity_digest:  # pragma: no cover - disk corruption
            raise MaterializeError(
                f"structure runtime identity changed while writing {identity_path}"
            )
    out.write_text(yaml.safe_dump(config, sort_keys=True, default_flow_style=False),
                   encoding="utf-8")

    # Written, then LOADED: a config this producer cannot itself resolve is not a config, and the
    # digest below is the one the run will sign, computed by the loader rather than predicted here.
    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    resolved = load_v2_config_file(out)

    args_path = out.with_suffix(".args.sh")
    args_path.write_text(_args_script(
        args, frozen=frozen, runtime=runtime, config_path=out,
        structure_runtime_identity=structure_identity),
                         encoding="utf-8")

    print(json.dumps({
        "config": str(out),
        "args": str(args_path),
        "protein_id": args.protein_id,
        "r_step": int(args.r_step),
        "stratum_key": args.stratum_key,
        "config_digest": resolved.config_digest(),
        "delta_new_cumulative": resolved.safety.delta_new_cumulative.value,
        "delta_new_unit": resolved.safety.delta_new_cumulative.unit,
        "frozen_roles": {role: frozen[role] for role in sorted(frozen)},
        "runtime_roles": sorted(runtime),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
