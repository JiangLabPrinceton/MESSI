"""The V2F5A materializer binds calibration to the exact Head and policy spec."""

from __future__ import annotations

import hashlib
import json
import types

import pytest

from inverse_folding.reference_flow.fusion_v2.identity import HeadEvaluatorIdentity
from inverse_folding.reference_flow.fusion_v2.config import (
    PolicyCalibrationArtifact,
    load_v2_config,
    policy_calibration_source_ref,
)
from scripts.calibrate_v2_head_policy import build_calibration_bundle
from scripts.materialize_v2_canary_config import (
    MaterializeError,
    _args_script,
    _structure_runtime_identity,
    fill_config,
)
from scripts.rf_fusion_v2_preflight import project_v2_budget
from tests.inverse_folding.test_fusion_v2_config import _mapping


D = "a" * 64


def _case(tmp_path):
    template = _mapping(**{
        "schedule.depth_cap": 1,
        "schedule.points": [{
            "depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 60,
            "n_lookaheads": 4, "band_key": "step40",
        }],
    })
    hotspot = tmp_path / "hotspot.json"
    hotspot.write_text(json.dumps({"delta_new": template["safety"]["delta_new_cumulative"]}))
    spec = tmp_path / "policy.json"
    spec.write_text(json.dumps({"policy_id": "head_directed_capped", "policy_version": "v1"}))
    spec_sha = hashlib.sha256(spec.read_bytes()).hexdigest()
    evaluator = HeadEvaluatorIdentity(
        allele=template["head"]["allele"], score_scale=template["head"]["score_scale"],
        window_k_min=template["head"]["window_k_min"],
        window_k_max=template["head"]["window_k_max"],
        head_config_hash=D, head_checkpoint_digest=D,
    )
    calibration = build_calibration_bundle(
        [{"protein_id": "P1", "sequence_md5": "b" * 32, "abs_repeat_drift": 0.01}],
        evaluator=evaluator, policy_spec=spec,
        max_counterfactual_head_calls_per_cycle=278,
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    args = types.SimpleNamespace(
        code_revision="deadbeef", campaign_id="v2f5a-test", r_step=40,
        hotspot_json=hotspot, policy_calibration_json=calibration_path,
        qualification_max_head_calls=40000,
        exploratory_profile=None, run_max_head_calls=None,
        esmfold2_model=None, esmfold2_num_loops=None,
        esmfold2_num_sampling_steps=None, esmfold2_num_diffusion_samples=None,
        esmfold2_seed=None,
    )
    frozen = {row["role"]: D for row in template["content"]}
    frozen.update({"head_config": D, "head_checkpoint": D,
                   "projection_policy_spec": spec_sha})
    return template, args, frozen


def _replace_calibrated_value(block, key, value):
    scalar = block[key]
    scalar["value"] = value
    scalar["source_ref"] = policy_calibration_source_ref(
        value=value,
        unit=scalar["unit"],
        source_kind=scalar["source_kind"],
        source_id=scalar["source_id"],
        artifact=PolicyCalibrationArtifact(**scalar["artifact"]),
    )


def test_policy_qualification_materialization_copies_only_a_matching_calibration(tmp_path):
    template, args, frozen = _case(tmp_path)
    config = fill_config(template, args=args, frozen=frozen, runtime={})
    assert config["identity"]["phase"] == "policy_qualification"
    assert config["projection"]["support_policy_id"] == "head_directed_capped"
    assert config["projection"]["head_directed"][
        "max_counterfactual_head_calls_per_cycle"] == 278
    assert config["caps"]["max_head_calls"] == 40000
    assert config["schedule"]["depth_cap"] == 1
    assert config["schedule"]["points"] == [{
        "depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 60,
        "n_lookaheads": 4, "band_key": "step40",
    }]
    assert load_v2_config(config).projection.head_directed is not None


def test_a_calibration_from_another_head_is_refused(tmp_path):
    template, args, frozen = _case(tmp_path)
    frozen["head_checkpoint"] = "f" * 64
    with pytest.raises(MaterializeError, match="different Head instrument"):
        fill_config(template, args=args, frozen=frozen, runtime={})


def test_explicit_uricase_recursive_profile_materializes_the_frozen_d4_k12_schedule(tmp_path):
    template, args, frozen = _case(tmp_path)
    args.exploratory_profile = "uricase_d4_k12_r40"
    args.run_max_head_calls = 90000
    args.esmfold2_model = "biohub/ESMFold2"
    args.esmfold2_num_loops = 3
    args.esmfold2_num_sampling_steps = 50
    args.esmfold2_num_diffusion_samples = 1
    args.esmfold2_seed = 0

    config = fill_config(template, args=args, frozen=frozen, runtime={})

    assert config["identity"]["phase"] == "capability_ladder"
    assert config["identity"]["split_role"] == "exploratory_uricase"
    assert config["schedule"]["schedule_id"] == "exploratory-uricase-d4-k12-r40-v1"
    assert config["schedule"]["depth_cap"] == 4
    assert config["schedule"]["active_population_width"] == 1
    assert config["schedule"]["points"] == [
        {"depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 60,
         "n_lookaheads": 12, "band_key": "step40"},
        {"depth": 1, "r_step": 40, "c_source_step": 60, "c_next_step": 70,
         "n_lookaheads": 12, "band_key": "step40"},
        {"depth": 2, "r_step": 40, "c_source_step": 70, "c_next_step": 80,
         "n_lookaheads": 12, "band_key": "step40"},
        {"depth": 3, "r_step": 40, "c_source_step": 80, "c_next_step": 90,
         "n_lookaheads": 12, "band_key": "step40"},
    ]
    assert config["caps"]["max_logical_dfe"] == 2200
    assert config["caps"]["max_definitive_refolds"] == 64
    assert config["caps"]["max_head_calls"] == 90000
    loaded = load_v2_config(config)
    assert loaded.identity.phase == "capability_ladder"
    assert loaded.schedule.depth_cap == 4


def test_highrisk_profile_is_single_root_d8_k32_with_five_step_rungs(tmp_path):
    template, args, frozen = _case(tmp_path)
    args.exploratory_profile = "highrisk_d8_k32_r40"
    args.run_max_head_calls = 20000
    args.esmfold2_model = "biohub/ESMFold2"
    args.esmfold2_num_loops = 3
    args.esmfold2_num_sampling_steps = 50
    args.esmfold2_num_diffusion_samples = 1
    args.esmfold2_seed = 0
    payload = json.loads(args.policy_calibration_json.read_text())
    payload["head_directed"]["lineage_incumbent_depth0_rule"] = "best_admissible_depth0"
    spec = tmp_path / "policy-v2.json"
    spec.write_text(json.dumps({"policy_id": "head_directed_capped", "policy_version": "v2"}))
    payload["policy_spec_sha256"] = hashlib.sha256(spec.read_bytes()).hexdigest()
    args.policy_calibration_json.write_text(json.dumps(payload))
    frozen["projection_policy_spec"] = payload["policy_spec_sha256"]

    config = fill_config(template, args=args, frozen=frozen, runtime={})

    assert config["projection"]["support_policy_version"] == "v2"
    assert config["schedule"]["active_population_width"] == 1
    assert config["schedule"]["depth_cap"] == 8
    assert [point["n_lookaheads"] for point in config["schedule"]["points"]] == [32] * 8
    assert [(point["c_source_step"], point["c_next_step"])
            for point in config["schedule"]["points"]] == [
        (50, 55), (55, 60), (60, 65), (65, 70),
        (70, 75), (75, 80), (80, 85), (85, 90),
    ]
    assert config["caps"]["max_logical_dfe"] == 20000
    assert load_v2_config(config).schedule.depth_cap == 8


def test_highrisk_d4_k24_profile_changes_only_breadth_identity_and_caps(tmp_path):
    template, args, frozen = _case(tmp_path)
    args.run_max_head_calls = 6000
    args.esmfold2_model = "biohub/ESMFold2"
    args.esmfold2_num_loops = 3
    args.esmfold2_num_sampling_steps = 50
    args.esmfold2_num_diffusion_samples = 1
    args.esmfold2_seed = 0

    payload = json.loads(args.policy_calibration_json.read_text())
    payload["head_directed"]["lineage_incumbent_depth0_rule"] = (
        "best_admissible_depth0"
    )
    _replace_calibrated_value(
        payload["head_directed"], "donor_improvement_epsilon", 0.005,
    )
    _replace_calibrated_value(
        payload["head_directed"], "local_contribution_tolerance",
        0.017012596130371094,
    )
    payload["head_directed"]["max_counterfactual_head_calls_per_cycle"] = 454
    spec = tmp_path / "policy-v2-k24.json"
    spec.write_text(json.dumps({
        "policy_id": "head_directed_capped", "policy_version": "v2",
    }))
    payload["policy_spec_sha256"] = hashlib.sha256(spec.read_bytes()).hexdigest()
    args.policy_calibration_json.write_text(json.dumps(payload))
    frozen["projection_policy_spec"] = payload["policy_spec_sha256"]

    args.exploratory_profile = "highrisk_d4_k12_r40"
    inherited = fill_config(template, args=args, frozen=frozen, runtime={})
    args.exploratory_profile = "highrisk_d4_k24_r40"
    config = fill_config(template, args=args, frozen=frozen, runtime={})

    assert config["identity"]["phase"] == "capability_ladder"
    assert config["identity"]["split_role"] == "exploratory_highrisk_breadth_ceiling_v1"
    assert config["projection"]["support_policy_version"] == "v2"
    assert config["schedule"] == {
        "schedule_id": "highrisk-d4-k24-r40-global-v1",
        "coordinate_law": "progressive_checkpoint",
        "depth_cap": 4,
        "active_population_width": 1,
        "min_lookahead_tail_steps": 10,
        "points": [
            {"depth": 0, "r_step": 40, "c_source_step": 50, "c_next_step": 60,
             "n_lookaheads": 24, "band_key": "step40"},
            {"depth": 1, "r_step": 40, "c_source_step": 60, "c_next_step": 70,
             "n_lookaheads": 24, "band_key": "step40"},
            {"depth": 2, "r_step": 40, "c_source_step": 70, "c_next_step": 80,
             "n_lookaheads": 24, "band_key": "step40"},
            {"depth": 3, "r_step": 40, "c_source_step": 80, "c_next_step": 90,
             "n_lookaheads": 24, "band_key": "step40"},
        ],
    }
    assert config["caps"]["max_logical_dfe"] == 4200
    assert config["caps"]["max_definitive_refolds"] == 128
    assert config["caps"]["max_head_calls"] == 6000

    # The new profile is a breadth-only extension of the existing D4/K12 high-risk method.
    algorithm_keys = set(config) - {"identity", "schedule", "caps"}
    assert {key: config[key] for key in algorithm_keys} == {
        key: inherited[key] for key in algorithm_keys
    }
    assert config["projection"]["head_directed"][
        "donor_improvement_epsilon"]["value"] == 0.005
    assert config["projection"]["head_directed"][
        "local_contribution_tolerance"]["value"] == 0.017012596130371094

    projection = project_v2_budget(load_v2_config(config), n_proteins=1)
    assert projection.per_protein_logical_dfe == 3790
    assert projection.total_definitive_refolds == 120
    assert projection.per_protein_counterfactual_head_calls == 4 * 454
    assert projection.total_head_calls == 120 + 4 * 454 == 1936
    assert projection.feasible


def test_testset_profile_reuses_frozen_d4_k12_algorithm_under_honest_identity(tmp_path):
    template, args, frozen = _case(tmp_path)
    args.run_max_head_calls = 6000
    args.esmfold2_model = "biohub/ESMFold2"
    args.esmfold2_num_loops = 3
    args.esmfold2_num_sampling_steps = 50
    args.esmfold2_num_diffusion_samples = 1
    args.esmfold2_seed = 0

    payload = json.loads(args.policy_calibration_json.read_text())
    payload["head_directed"]["lineage_incumbent_depth0_rule"] = (
        "best_admissible_depth0"
    )
    _replace_calibrated_value(
        payload["head_directed"], "donor_improvement_epsilon", 0.005,
    )
    _replace_calibrated_value(
        payload["head_directed"], "local_contribution_tolerance",
        0.017012596130371094,
    )
    spec = tmp_path / "policy-v2-testset.json"
    spec.write_text(json.dumps({
        "policy_id": "head_directed_capped", "policy_version": "v2",
    }))
    payload["policy_spec_sha256"] = hashlib.sha256(spec.read_bytes()).hexdigest()
    args.policy_calibration_json.write_text(json.dumps(payload))
    frozen["projection_policy_spec"] = payload["policy_spec_sha256"]

    args.exploratory_profile = "highrisk_d4_k12_r40"
    highrisk = fill_config(template, args=args, frozen=frozen, runtime={})
    args.exploratory_profile = "testset_d4_k12_r40"
    config = fill_config(template, args=args, frozen=frozen, runtime={})

    assert config["identity"]["phase"] == "capability_ladder"
    assert config["identity"]["split_role"] == "exploratory_testset_design_v1"
    assert config["schedule"]["schedule_id"] == "testset-d4-k12-r40-global-v1"
    assert config["schedule"]["points"] == highrisk["schedule"]["points"]
    assert config["caps"] == highrisk["caps"]
    algorithm_keys = set(config) - {"identity", "schedule", "caps"}
    assert {key: config[key] for key in algorithm_keys} == {
        key: highrisk[key] for key in algorithm_keys
    }

    projection = project_v2_budget(load_v2_config(config), n_proteins=1)
    assert projection.per_protein_logical_dfe == 1990
    assert projection.total_definitive_refolds == 60
    assert projection.feasible


@pytest.mark.parametrize(
    ("profile", "depth", "width", "logical_dfe", "max_dfe", "cap", "refolds"),
    [
        ("uricase_core0_pilot_d2_k6_r40", 2, 6, 820, 1170, 1290, 18),
        ("uricase_core0_pilot_d3_k12_r40", 3, 12, 1820, 2170, 2390, 48),
    ],
)
def test_uricase_core0_pilot_profiles_match_frozen_budget(
    tmp_path, profile, depth, width, logical_dfe, max_dfe, cap, refolds,
):
    template, args, frozen = _case(tmp_path)
    args.run_max_head_calls = 6000
    args.esmfold2_model = "biohub/ESMFold2"
    args.esmfold2_num_loops = 3
    args.esmfold2_num_sampling_steps = 50
    args.esmfold2_num_diffusion_samples = 1
    args.esmfold2_seed = 0
    payload = json.loads(args.policy_calibration_json.read_text())
    payload["head_directed"]["lineage_incumbent_depth0_rule"] = "best_admissible_depth0"
    spec = tmp_path / "policy-v2.json"
    spec.write_text(json.dumps({
        "policy_id": "head_directed_capped", "policy_version": "v2",
    }))
    payload["policy_spec_sha256"] = hashlib.sha256(spec.read_bytes()).hexdigest()
    args.policy_calibration_json.write_text(json.dumps(payload))
    frozen["projection_policy_spec"] = payload["policy_spec_sha256"]
    args.exploratory_profile = profile

    config = fill_config(template, args=args, frozen=frozen, runtime={})
    projection = project_v2_budget(load_v2_config(config), n_proteins=1)

    assert config["schedule"]["depth_cap"] == depth
    assert {point["n_lookaheads"] for point in config["schedule"]["points"]} == {width}
    assert projection.per_protein_logical_dfe == logical_dfe
    assert projection.max_total_logical_dfe == max_dfe
    assert config["caps"]["max_logical_dfe"] == cap
    assert config["caps"]["max_retries"] == 7
    assert projection.total_definitive_refolds == refolds
    assert projection.feasible


def test_recursive_profile_is_never_inferred_from_an_ordinary_calibrated_call(tmp_path):
    template, args, frozen = _case(tmp_path)

    config = fill_config(template, args=args, frozen=frozen, runtime={})

    assert config["identity"]["phase"] == "policy_qualification"
    assert config["identity"]["split_role"] == "policy_qualification"
    assert config["schedule"]["depth_cap"] == 1
    assert config["caps"]["max_head_calls"] == 40000

    args.run_max_head_calls = 90000
    with pytest.raises(MaterializeError, match="requires an explicit --exploratory-profile"):
        fill_config(template, args=args, frozen=frozen, runtime={})


def test_recursive_profile_requires_its_calibration_r40_cell_and_generic_head_cap(tmp_path):
    template, args, frozen = _case(tmp_path)
    args.exploratory_profile = "uricase_d4_k12_r40"
    args.esmfold2_model = "biohub/ESMFold2"
    args.esmfold2_num_loops = 3
    args.esmfold2_num_sampling_steps = 50
    args.esmfold2_num_diffusion_samples = 1
    args.esmfold2_seed = 0
    args.run_max_head_calls = None
    with pytest.raises(MaterializeError, match="--run-max-head-calls"):
        fill_config(template, args=args, frozen=frozen, runtime={})

    args.run_max_head_calls = 90000
    args.policy_calibration_json = None
    with pytest.raises(MaterializeError, match="--policy-calibration-json"):
        fill_config(template, args=args, frozen=frozen, runtime={})

    args.policy_calibration_json = tmp_path / "missing-policy-calibration.json"
    with pytest.raises(MaterializeError, match="does not exist"):
        fill_config(template, args=args, frozen=frozen, runtime={})

    args.policy_calibration_json = _case(tmp_path)[1].policy_calibration_json
    args.r_step = 30
    with pytest.raises(MaterializeError, match="requires --r-step 40"):
        fill_config(template, args=args, frozen=frozen, runtime={})


def test_recursive_profile_requires_an_explicit_esmfold2_selector_and_protocol(tmp_path):
    template, args, frozen = _case(tmp_path)
    args.exploratory_profile = "uricase_d4_k12_r40"
    args.run_max_head_calls = 90000

    for name, value in (
        ("esmfold2_model", "biohub/ESMFold2"),
        ("esmfold2_num_loops", 3),
        ("esmfold2_num_sampling_steps", 50),
        ("esmfold2_num_diffusion_samples", 1),
        ("esmfold2_seed", 0),
    ):
        with pytest.raises(MaterializeError, match=name.replace("_", "-")):
            fill_config(template, args=args, frozen=frozen, runtime={})
        setattr(args, name, value)

    assert fill_config(template, args=args, frozen=frozen, runtime={})[
        "identity"]["split_role"] == "exploratory_uricase"


def test_structure_runtime_identity_binds_snapshot_selector_protocol_and_shard_inputs(tmp_path):
    template, args, frozen = _case(tmp_path)
    args.exploratory_profile = "uricase_d4_k12_r40"
    args.run_max_head_calls = 90000
    args.esmfold2_model = "biohub/ESMFold2"
    args.esmfold2_num_loops = 3
    args.esmfold2_num_sampling_steps = 50
    args.esmfold2_num_diffusion_samples = 1
    args.esmfold2_seed = 7
    args.out = tmp_path / "resolved.yaml"
    args.structure_backend = tmp_path / "model.safetensors"
    args.structure_backend.write_bytes(b"frozen model snapshot")

    path, payload, digest = _structure_runtime_identity(args)
    assert path == tmp_path / "resolved.structure_runtime.json"
    assert payload["schema_version"] == "v2-esmfold2-runtime-1"
    assert payload["backend"] == "esmfold2_live"
    assert payload["model_selector"] == "biohub/ESMFold2"
    assert payload["protocol"] == {
        "num_loops": 3, "num_sampling_steps": 50,
        "num_diffusion_samples": 1, "seed": 7,
    }
    assert payload["local_model_snapshot_sha256"] == hashlib.sha256(
        b"frozen model snapshot").hexdigest()
    assert len(digest) == 64
    frozen["structure_backend"] = digest
    config = fill_config(template, args=args, frozen=frozen, runtime={})
    structure_row = next(row for row in config["content"]
                         if row["role"] == "structure_backend")
    assert structure_row == {
        "role": "structure_backend", "label": "structure_backend.bin",
        "binding": "frozen", "expected_sha256": digest,
    }

    for name in (
        "reference_sequence", "projection_policy_spec", "head_checkpoint", "dplm_checkpoint",
        "cohort_table", "pdb_root", "refold_cache_dir", "head_config_dir", "head_variant_id",
        "esmfold2_site_packages", "band_json", "reference_manifest", "stratum_manifest",
        "protein_id", "stratum_key",
    ):
        if not hasattr(args, name):
            setattr(args, name, f"/{name}")
    args.reference_sequence = "/reference.seq"
    args.projection_policy_spec = "/policy.json"
    args.head_checkpoint = "/head.pt"
    args.dplm_checkpoint = "/dplm.pt"
    args.cohort_table = "/cohort.parquet"
    args.pdb_root = "/pdbs"
    args.refold_cache_dir = "/refolds"
    args.head_config_dir = "/head-config"
    args.head_variant_id = "LC1"
    args.esmfold2_site_packages = "/site-packages"
    args.band_json = "/band.json"
    args.reference_manifest = "/references.json"
    args.stratum_manifest = "/strata.json"
    args.protein_id = "P1"
    args.r_step = 40
    args.stratum_key = "uricase"
    text = _args_script(args, frozen={}, runtime={}, config_path=tmp_path / "resolved.yaml")
    assert f"structure_backend={path}" in text
    assert "esmfold2_model=biohub/ESMFold2" in text
    assert "esmfold2_num_loops=3" in text
    assert "esmfold2_num_sampling_steps=50" in text
    assert "esmfold2_num_diffusion_samples=1" in text
    assert "esmfold2_seed=7" in text
