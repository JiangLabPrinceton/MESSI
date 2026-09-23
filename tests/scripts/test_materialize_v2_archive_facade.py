"""Strict V2 archive -> Phase-C/v0 facade materialization."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from inverse_folding.reference_flow.fusion_v2.identity import (
    HeadEvaluatorIdentity,
    window_grid_digest,
)
from scripts.materialize_v2_archive_facade import (
    V2FacadeError,
    main,
    materialize_archive_facade,
)
from scripts.materialize_v2_canary_config import no_constraint_manifest_digest


EXPECTED_MASTER_SEEDS = (20260811, 20260812, 20260813, 20260814)
DEFAULT_CAPS = {
    "max_definitive_refolds": 64,
    "max_gpu_seconds": 86400,
    "max_head_calls": 6000,
    "max_logical_dfe": 2200,
    "max_retries": 2,
    "max_walltime_s": 21600,
    "retry_scope": "per_request",
}


def _write_bundle(
    root,
    *,
    protein_id: str,
    endpoints: list[dict],
    archive_overrides: dict[str, dict] | None = None,
    manifest_overrides: dict | None = None,
    root_id: str | None = None,
):
    root.mkdir(parents=True)
    archive_overrides = archive_overrides or {}
    complete_rows = []
    archive_rows = []
    structure_rows = []
    terminal_rows = []
    root_id = root_id or root.name
    content_overrides = (manifest_overrides or {}).get("content_identities", {})
    head_config_hash = content_overrides.get("head_config", "1" * 64)
    head_checkpoint_digest = content_overrides.get("head_checkpoint", "f" * 64)
    for index, spec in enumerate(endpoints):
        endpoint_id = spec.get("endpoint_id", f"endpoint:{protein_id}:{index}")
        sequence = spec.get("sequence", "ACDEFGHIK")
        sequence_md5 = spec.get(
            "sequence_md5", hashlib.md5(sequence.encode("ascii"), usedforsecurity=False).hexdigest(),
        )
        digest = spec.get("endpoint_content_digest", f"digest:{protein_id}:{index}")
        head_global_risk = spec.get("head_global_risk", float(index))
        residue_hotspot = spec.get("residue_hotspot", [0.0] * len(sequence))
        windows = spec.get("windows", [{
            "start_0b": 0,
            "end_0b": len(sequence),
            "k": len(sequence),
            "z": head_global_risk,
        }])
        grid_digest = window_grid_digest(SimpleNamespace(**window) for window in windows)
        allele = spec.get("allele", "DRB1_0701")
        score_scale = spec.get("score_scale", "raw_logit")
        window_ks = [int(window["k"]) for window in windows]
        head_evaluator_digest = spec.get(
            "head_evaluator_digest",
            HeadEvaluatorIdentity(
                allele=allele,
                score_scale=score_scale,
                window_k_min=min(window_ks),
                window_k_max=max(window_ks),
                head_config_hash=head_config_hash,
                head_checkpoint_digest=head_checkpoint_digest,
            ).digest(),
        )
        head_score_json = spec.get("head_score_json", json.dumps({
            "allele": allele,
            "global_risk": head_global_risk,
            "protein_id": protein_id,
            "residue_hotspot": residue_hotspot,
            "score_scale": score_scale,
            "sequence_length": len(sequence),
            "sequence_md5": sequence_md5,
            "windows": windows,
        }, sort_keys=True))
        structure_evaluated = spec.get("structure_evaluated", True)
        structure_feasible = spec.get("structure_feasible", True)
        complete_rows.append({
            "endpoint_id": endpoint_id,
            "endpoint_content_digest": digest,
            "protein_id": protein_id,
            "root_id": root_id,
            "depth": spec.get("depth", index),
            "sequence": sequence,
            "sequence_md5": sequence_md5,
            "sequence_equivalence_key": spec.get("sequence_equivalence_key", sequence_md5),
            "sequence_length": len(sequence),
            "head_evaluator_digest": head_evaluator_digest,
            "head_window_grid_digest": spec.get("head_window_grid_digest", grid_digest),
            "head_global_risk": head_global_risk,
            "head_score_json": head_score_json,
            "feasibility_level": spec.get("feasibility_level", "definitive"),
            "structure_evaluated": structure_evaluated,
            "structure_feasible": structure_feasible,
        })
        archive = {
            "endpoint_id": endpoint_id,
            "endpoint_content_digest": digest,
            "protein_id": protein_id,
            "root_id": root_id,
            "sequence_equivalence_key": spec.get("sequence_equivalence_key", sequence_md5),
            "feasibility_level": spec.get(
                "archive_feasibility_level", spec.get("feasibility_level", "definitive"),
            ),
            "is_elite": spec.get("is_elite", index == 0),
            "elite_rank": 0 if spec.get("is_elite", index == 0) else None,
        }
        archive.update(archive_overrides.get(endpoint_id, {}))
        archive_rows.append(archive)
        structure_rows.append({
            "endpoint_id": endpoint_id,
            "protein_id": protein_id,
            "depth": spec.get("depth", index),
            "sequence_md5": sequence_md5,
            "evaluated": spec.get("structure_result_evaluated", True),
            "feasible": spec.get("structure_result_feasible", structure_feasible),
            "metrics_json": spec.get("metrics_json", json.dumps({
                "pLDDT": spec.get("pLDDT", 80.0),
                "scTM": spec.get("scTM", 0.8),
            }, sort_keys=True)),
            "feasibility_level": spec.get("structure_result_level", spec.get(
                "feasibility_level", "definitive")),
            "structure_backend_digest": spec.get("structure_backend_digest", "a" * 64),
            "v0_structure_gate_config_digest": spec.get(
                "v0_structure_gate_config_digest", "b" * 64,
            ),
        })
        write_terminal = spec.get(
            "write_terminal",
            bool(structure_evaluated and structure_feasible)
            and spec.get("feasibility_level", "definitive") == "definitive",
        )
        if write_terminal:
            terminal_rows.append({
                "endpoint_id": endpoint_id,
                "sequence_md5": spec.get("terminal_sequence_md5", sequence_md5),
                "structure_definitive": spec.get("terminal_structure_definitive", True),
                "structure_feasible": spec.get("terminal_structure_feasible", True),
                "structure_metrics_json": spec.get(
                    "terminal_structure_metrics_json",
                    json.dumps({
                        "pLDDT": spec.get("pLDDT", 80.0),
                        "scTM": spec.get("scTM", 0.8),
                    }, sort_keys=True),
                ),
                "immune_evaluator": spec.get(
                    "terminal_immune_evaluator",
                    head_evaluator_digest,
                ),
                "immune_global_risk": spec.get(
                    "terminal_immune_global_risk", head_global_risk,
                ),
                "immune_passed": spec.get("terminal_immune_passed", True),
            })
    pd.DataFrame(complete_rows).to_parquet(root / "complete_endpoints.parquet", index=False)
    pd.DataFrame(archive_rows).to_parquet(root / "archive.parquet", index=False)
    pd.DataFrame(structure_rows).to_parquet(
        root / "structure_evaluations.parquet", index=False,
    )
    pd.DataFrame(terminal_rows, columns=[
        "endpoint_id",
        "sequence_md5",
        "structure_definitive",
        "structure_feasible",
        "structure_metrics_json",
        "immune_evaluator",
        "immune_global_risk",
        "immune_passed",
    ]).to_parquet(root / "terminal_validation.parquet", index=False)
    no_constraints = no_constraint_manifest_digest()
    manifest = {
        "campaign_id": "fusion-v2-exploratory-uricase",
        "phase": "capability_ladder",
        "split_role": "exploratory_uricase",
        "code_revision": "deadbeef",
        "schedule_id": "exploratory-uricase-d4-k12-r40-v1",
        "coordinate_law": "progressive_checkpoint",
        "depth_cap": 4,
        "master_seed": 1000,
        "schema_version": "v2cfg-1",
        "requested_cohort": [protein_id],
        "accepted_fragments": [protein_id],
        "missing_proteins": [],
        "rejected_fragments": [],
        "n_ok": 1,
        "arm_role": "v2",
        "feedback_enabled": True,
        "active_population_width": 1,
        "a2_matching_resource": "definitive_refolds",
        "seed_schema": "v2seed-1",
        "seed_namespaces": [
            "a2_extra_lookahead",
            "matched_descendant",
            "v2_depth0_root",
            "v2_lookahead",
        ],
        "caps": dict(DEFAULT_CAPS),
        "realized_caps": {
            "within": True,
            "breached": [],
            "unverifiable": [],
            "detail": "within every declared cap",
        },
        "exploratory_depth_override": True,
        "production_depth_authorized": False,
        # These three define the common terminal structure instrument.  The backbone is
        # deliberately protein-specific and must not make two cells look like different studies.
        "content_identities": {
            "structure_backend": "a" * 64,
            "v0_structure_gate_config": "b" * 64,
            "structure_config": "c" * 64,
            "backbone": hashlib.sha256(protein_id.encode("ascii")).hexdigest(),
            "cohort_table": "d" * 64,
            "complete_reference_sequence": hashlib.sha256(
                f"reference:{protein_id}".encode("ascii")
            ).hexdigest(),
            "constraint_manifest": no_constraints,
            "dplm_checkpoint": "e" * 64,
            "fixed_token_policy": no_constraints,
            "head_checkpoint": head_checkpoint_digest,
            "head_config": head_config_hash,
            "projection_policy_spec": "2" * 64,
            "reference_sequences": "3" * 64,
            "rf_sampler_config": "4" * 64,
            "schedule_band_calibration": "5" * 64,
        },
        "config_digest": hashlib.sha256(f"config:{protein_id}".encode("ascii")).hexdigest(),
    }
    for key, value in (manifest_overrides or {}).items():
        if key == "content_identities":
            manifest[key].update(value)
        else:
            manifest[key] = value
    (root / "run_manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return root


def _write_highrisk_bundle(
    root,
    *,
    protein_id: str,
    root_index: int,
    endpoints: list[dict],
    manifest_overrides: dict | None = None,
    master_seed: int | None = None,
):
    overrides = {
        "campaign_id": "highrisk-gumbel-d4k12-four-root",
        "split_role": "exploratory_highrisk_ceiling_v2",
        "schedule_id": "highrisk-d4-k12-r40-global-v2",
        "master_seed": 20260811 + root_index if master_seed is None else master_seed,
    }
    for key, value in (manifest_overrides or {}).items():
        if key == "content_identities":
            overrides.setdefault("content_identities", {}).update(value)
        else:
            overrides[key] = value
    return _write_bundle(
        root,
        protein_id=protein_id,
        endpoints=endpoints,
        manifest_overrides=overrides,
        # The runtime's internal root_id is lineage-local and repeats across seeded campaign roots.
        root_id=f"{protein_id}:v2:d0:r0",
    )


def test_elite_mode_joins_by_endpoint_and_preserves_phase_c_and_lineage_columns(tmp_path):
    first = _write_bundle(
        tmp_path / "p1",
        protein_id="P1",
        endpoints=[
            {"endpoint_id": "endpoint:P1:best", "head_global_risk": -3.0, "depth": 2},
            {"endpoint_id": "endpoint:P1:other", "head_global_risk": -1.0,
             "is_elite": False},
        ],
    )
    second = _write_bundle(
        tmp_path / "p2", protein_id="P2",
        endpoints=[{"endpoint_id": "endpoint:P2:best", "head_global_risk": -2.0,
                    "depth": 1}],
    )

    out = materialize_archive_facade([first, second], mode="elite")

    assert out[["protein_id", "design_idx", "entry_source_id"]].to_dict("records") == [
        {"protein_id": "P1", "design_idx": 0, "entry_source_id": "endpoint:P1:best"},
        {"protein_id": "P2", "design_idx": 0, "entry_source_id": "endpoint:P2:best"},
    ]
    assert list(out["endpoint_id"]) == list(out["entry_source_id"])
    assert list(out["depth"]) == [2, 1]
    assert list(out["head_global_risk"]) == [-3.0, -2.0]


def test_top_k_is_risk_sorted_stably_and_does_not_give_duplicate_sequences_mass(tmp_path):
    bundle = _write_bundle(
        tmp_path / "bundle",
        protein_id="P1",
        endpoints=[
            {"endpoint_id": "endpoint:z", "sequence": "ACDEFGHIK",
             "head_global_risk": -5.0, "is_elite": True},
            {"endpoint_id": "endpoint:a", "sequence": "ACDEFGHIK",
             "head_global_risk": -5.0, "is_elite": False},
            {"endpoint_id": "endpoint:b", "sequence": "LMNPQRSTV",
             "head_global_risk": -5.0, "is_elite": False},
            {"endpoint_id": "endpoint:c", "sequence": "WYACDEFGH",
             "head_global_risk": -4.0, "is_elite": False},
        ],
    )

    out = materialize_archive_facade([bundle], mode="top-k", k=3, require_k=True)

    # endpoint:a is the stable representative of the two convergent logical endpoints.
    assert list(out["endpoint_id"]) == ["endpoint:a", "endpoint:b", "endpoint:c"]
    assert list(out["design_idx"]) == [0, 1, 2]


def test_top_k_require_k_fails_when_distinct_feasible_pool_is_too_small(tmp_path):
    bundle = _write_bundle(
        tmp_path / "bundle", protein_id="P1",
        endpoints=[{"endpoint_id": "endpoint:only"}],
    )
    with pytest.raises(V2FacadeError, match="requires 2.*has 1"):
        materialize_archive_facade([bundle], mode="top-k", k=2, require_k=True)


def test_feasible_immune_pareto_is_complete_deduplicated_and_input_order_invariant(tmp_path):
    first = _write_highrisk_bundle(
        tmp_path / "root0", protein_id="P1", root_index=0,
        endpoints=[
            {"endpoint_id": "p1:r0:a", "sequence": "ACDEFGHIK",
             "head_global_risk": 1.0, "residue_hotspot": [1.0] * 9},
            {"endpoint_id": "p1:r0:b", "sequence": "LMNPQRSTV",
             "head_global_risk": 0.0, "residue_hotspot": [2.0] * 9},
            {"endpoint_id": "p1:r0:dominated", "sequence": "WYACDEFGH",
             "head_global_risk": 2.0, "residue_hotspot": [2.0] * 9},
        ],
    )
    second = _write_highrisk_bundle(
        tmp_path / "root1", protein_id="P1", root_index=1,
        endpoints=[
            # Exact sequence convergence must retain both source endpoint/root identities.
            {"endpoint_id": "p1:r1:a", "sequence": "ACDEFGHIK",
             "head_global_risk": 1.0, "residue_hotspot": [1.0] * 9},
            # Equal objective values are non-dominating because neither axis is strictly better.
            {"endpoint_id": "p1:r1:tie", "sequence": "KIHGFEDCA",
             "head_global_risk": 1.0, "residue_hotspot": [1.0] * 9},
            {"endpoint_id": "p1:r1:dominated", "sequence": "CCCCCCCCC",
             "head_global_risk": 0.0, "residue_hotspot": [3.0] * 9},
        ],
    )
    third = _write_highrisk_bundle(
        tmp_path / "root2", protein_id="P1", root_index=2,
        endpoints=[{
            "endpoint_id": "p1:r2:dominated", "sequence": "GGGGGGGGG",
            "head_global_risk": 3.0, "residue_hotspot": [3.0] * 9,
        }],
    )
    fourth = _write_highrisk_bundle(
        tmp_path / "root3", protein_id="P1", root_index=3,
        endpoints=[{
            "endpoint_id": "p1:r3:dominated", "sequence": "TTTTTTTTT",
            "head_global_risk": 4.0, "residue_hotspot": [4.0] * 9,
        }],
    )

    forward = materialize_archive_facade(
        [first, second, third, fourth], mode="feasible_immune_pareto",
        expected_master_seeds=EXPECTED_MASTER_SEEDS,
    )
    reverse = materialize_archive_facade(
        [fourth, third, second, first], mode="feasible_immune_pareto",
        expected_master_seeds=tuple(reversed(EXPECTED_MASTER_SEEDS)),
    )

    pd.testing.assert_frame_equal(forward, reverse)
    assert set(forward["sequence"]) == {"ACDEFGHIK", "LMNPQRSTV", "KIHGFEDCA"}
    assert set(forward["pareto_rank"]) == {1}
    assert set(forward["selection_status"]) == {"feasible_immune_pareto"}
    assert set(forward["terminal_validated"]) == {True}
    assert set(forward["structure_feasible"]) == {True}
    converged = forward[forward["sequence"] == "ACDEFGHIK"].iloc[0]
    assert json.loads(converged["source_endpoint_ids_json"]) == ["p1:r0:a", "p1:r1:a"]
    assert json.loads(converged["source_root_ids_json"]) == [
        {"master_seed": 20260811, "root_id": "P1:v2:d0:r0"},
        {"master_seed": 20260812, "root_id": "P1:v2:d0:r0"},
    ]
    provenance = json.loads(converged["source_provenance_json"])
    assert [item["master_seed"] for item in provenance] == [20260811, 20260812]
    assert converged["n_source_endpoints"] == 2
    assert converged["n_source_roots"] == 2
    seed_grid = json.loads(converged["master_seed_grid_json"])
    assert seed_grid["master_seeds"] == list(EXPECTED_MASTER_SEEDS)
    assert seed_grid["expected_roots_per_protein"] == 4
    assert len(converged["master_seed_grid_digest"]) == 64
    assert len(converged["selection_provenance_digest"]) == 64


def test_official_multiroot_selection_uses_pareto_layers_and_explicit_candidate_count(
    tmp_path,
):
    specifications = [
        ("AAAAAAAAA", 0.0, 3.0),
        ("CCCCCCCCC", 1.0, 1.0),
        ("DDDDDDDDD", 3.0, 0.0),
        ("EEEEEEEEE", 2.0, 2.0),
        ("FFFFFFFFF", 4.0, 4.0),
    ]
    bundles = []
    for root_index in range(4):
        endpoints = []
        if root_index == 0:
            endpoints = [
                {
                    "endpoint_id": f"p1:{index}",
                    "sequence": sequence,
                    "head_global_risk": risk,
                    "residue_hotspot": [density] * 9,
                }
                for index, (sequence, risk, density) in enumerate(specifications)
            ]
        else:
            # Exact convergence across the other roots closes the four-seed grid without
            # changing the deduplicated candidate surface.
            sequence, risk, density = specifications[0]
            endpoints = [{
                "endpoint_id": f"p1:r{root_index}:a",
                "sequence": sequence,
                "head_global_risk": risk,
                "residue_hotspot": [density] * 9,
            }]
        bundles.append(_write_highrisk_bundle(
            tmp_path / f"root{root_index}",
            protein_id="P1",
            root_index=root_index,
            endpoints=endpoints,
        ))

    complete_front = materialize_archive_facade(
        bundles,
        mode="feasible_immune_pareto",
        expected_master_seeds=EXPECTED_MASTER_SEEDS,
    )
    official = materialize_archive_facade(
        bundles,
        mode="feasible_immune_pareto",
        official=True,
        final_candidates_per_protein=4,
        expected_master_seeds=EXPECTED_MASTER_SEEDS,
    )
    explicit_without_official = materialize_archive_facade(
        bundles,
        mode="feasible_immune_pareto",
        final_candidates_per_protein=4,
        expected_master_seeds=EXPECTED_MASTER_SEEDS,
    )

    assert set(complete_front["sequence"]) == {
        "AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD",
    }
    assert list(official["sequence"]) == [
        "CCCCCCCCC", "AAAAAAAAA", "DDDDDDDDD", "EEEEEEEEE",
    ]
    assert list(official["pareto_rank"]) == [1, 1, 1, 2]
    assert set(official["selection_status"]) == {
        "official_feasible_immune_pareto",
    }
    pd.testing.assert_frame_equal(
        official.drop(columns=["selection_status", "selection_provenance_digest"]),
        explicit_without_official.drop(
            columns=["selection_status", "selection_provenance_digest"]
        ),
    )
    assert set(explicit_without_official["selection_status"]) == {
        "bounded_feasible_immune_pareto",
    }


def test_official_multiroot_defaults_to_eight_and_candidate_count_must_be_positive(tmp_path):
    bundles = [
        _write_highrisk_bundle(
            tmp_path / f"root{root_index}",
            protein_id="P1",
            root_index=root_index,
            endpoints=[{
                "endpoint_id": f"p1:r{root_index}",
                "sequence": "ACDEFGHIK",
                "head_global_risk": -1.0,
            }],
        )
        for root_index in range(4)
    ]
    official = materialize_archive_facade(
        bundles,
        mode="feasible_immune_pareto",
        official=True,
        expected_master_seeds=EXPECTED_MASTER_SEEDS,
    )
    assert len(official) == 1

    with pytest.raises(V2FacadeError, match="final_candidates_per_protein"):
        materialize_archive_facade(
            bundles,
            mode="feasible_immune_pareto",
            final_candidates_per_protein=0,
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )


@pytest.mark.parametrize(
    "split_role",
    ["exploratory_testset_design_v1", "exploratory_fast_generalization_0401"],
)
def test_multiroot_testset_split_roles_use_the_same_complete_grid_validator(
    tmp_path, split_role,
):
    bundles = [
        _write_highrisk_bundle(
            tmp_path / f"root{root_index}",
            protein_id="P1",
            root_index=root_index,
            endpoints=[{
                "endpoint_id": f"p1:r{root_index}",
                "sequence": "ACDEFGHIK",
                "head_global_risk": -1.0,
            }],
            manifest_overrides={"split_role": split_role},
        )
        for root_index in range(4)
    ]

    out = materialize_archive_facade(
        bundles,
        mode="feasible_immune_pareto",
        expected_master_seeds=EXPECTED_MASTER_SEEDS,
    )

    assert len(out) == 1
    assert out.iloc[0]["protein_id"] == "P1"
    assert out.iloc[0]["n_source_roots"] == 4
    assert out.iloc[0]["selection_status"] == "feasible_immune_pareto"


def test_fallback_uses_only_all_root_failures_and_plddt_does_not_change_the_front(tmp_path):
    successful = _write_highrisk_bundle(
        tmp_path / "p1-root0", protein_id="P1", root_index=0,
        endpoints=[
            {"endpoint_id": "p1:feasible", "sequence": "ACDEFGHIK",
             "structure_feasible": True},
        ],
    )
    failed_sibling = _write_highrisk_bundle(
        tmp_path / "p1-root1", protein_id="P1", root_index=1,
        endpoints=[
            {"endpoint_id": "p1:rejected", "sequence": "LMNPQRSTV",
             "structure_evaluated": False, "structure_feasible": False,
             "structure_result_feasible": False, "head_global_risk": -10.0,
             "residue_hotspot": [0.0] * 9, "scTM": 0.6, "pLDDT": 99.0,
             "feasibility_level": "unvalidated",
             "structure_result_level": "unvalidated"},
        ],
    )
    failed_siblings = [
        failed_sibling,
        *[
            _write_highrisk_bundle(
                tmp_path / f"p1-root{root_index}", protein_id="P1",
                root_index=root_index,
                endpoints=[{
                    "endpoint_id": f"p1:rejected:{root_index}",
                    "sequence": "GGGGGGGGG" if root_index == 2 else "TTTTTTTTT",
                    "structure_evaluated": False,
                    "structure_feasible": False,
                    "structure_result_feasible": False,
                    "feasibility_level": "unvalidated",
                    "structure_result_level": "unvalidated",
                }],
            )
            for root_index in (2, 3)
        ],
    ]
    rejected_zero = _write_highrisk_bundle(
        tmp_path / "p2-root0", protein_id="P2", root_index=0,
        endpoints=[
            {"endpoint_id": "p2:a", "sequence": "AAAAAAAAA",
             "structure_evaluated": False, "structure_feasible": False,
             "structure_result_feasible": False, "residue_hotspot": [1.0] * 9,
             "scTM": 0.4, "pLDDT": 50.0, "feasibility_level": "unvalidated",
             "structure_result_level": "unvalidated"},
            {"endpoint_id": "p2:b", "sequence": "CCCCCCCCC",
             "structure_evaluated": False, "structure_feasible": False,
             "structure_result_feasible": False, "residue_hotspot": [2.0] * 9,
             "scTM": 0.5, "pLDDT": 90.0, "feasibility_level": "unvalidated",
             "structure_result_level": "unvalidated"},
        ],
    )
    rejected_one = _write_highrisk_bundle(
        tmp_path / "p2-root1", protein_id="P2", root_index=1,
        endpoints=[
            {"endpoint_id": "p2:dominated", "sequence": "DDDDDDDDD",
             "structure_evaluated": False, "structure_feasible": False,
             "structure_result_feasible": False, "residue_hotspot": [2.5] * 9,
             "scTM": 0.3, "pLDDT": 99.0, "feasibility_level": "unvalidated",
             "structure_result_level": "unvalidated"},
            # Same two Pareto axes as p2:a but higher pLDDT: both remain on the front.
            {"endpoint_id": "p2:axis-tie", "sequence": "EEEEEEEEE",
             "structure_evaluated": False, "structure_feasible": False,
             "structure_result_feasible": False, "residue_hotspot": [1.0] * 9,
             "scTM": 0.4, "pLDDT": 80.0, "feasibility_level": "unvalidated",
             "structure_result_level": "unvalidated"},
        ],
    )
    rejected_more = [
        _write_highrisk_bundle(
            tmp_path / f"p2-root{root_index}", protein_id="P2", root_index=root_index,
            endpoints=[{
                "endpoint_id": f"p2:dominated:{root_index}",
                "sequence": "FFFFFFFFF" if root_index == 2 else "GGGGGGGGG",
                "structure_evaluated": False,
                "structure_feasible": False,
                "structure_result_feasible": False,
                "residue_hotspot": [3.0] * 9,
                "scTM": 0.2,
                "pLDDT": 70.0,
                "feasibility_level": "unvalidated",
                "structure_result_level": "unvalidated",
            }],
        )
        for root_index in (2, 3)
    ]

    bundles = [successful, *failed_siblings, rejected_zero, rejected_one, *rejected_more]
    out = materialize_archive_facade(
        bundles, mode="structure_rejected_fallback",
        expected_master_seeds=EXPECTED_MASTER_SEEDS,
    )
    reverse = materialize_archive_facade(
        list(reversed(bundles)), mode="structure_rejected_fallback",
        expected_master_seeds=tuple(reversed(EXPECTED_MASTER_SEEDS)),
    )

    pd.testing.assert_frame_equal(out, reverse)
    assert set(out["protein_id"]) == {"P2"}
    assert set(out["endpoint_id"]) == {"p2:a", "p2:b", "p2:axis-tie"}
    assert list(out.loc[out["scTM"].eq(0.4), "endpoint_id"]) == [
        "p2:axis-tie", "p2:a",
    ]
    assert set(out["selection_status"]) == {"structure_rejected_fallback"}
    assert set(out["pareto_rank"]) == {1}
    assert set(out["terminal_validated"]) == {False}
    assert set(out["structure_evaluated"]) == {True}
    assert set(out["structure_feasible"]) == {False}


def test_fallback_refuses_an_omitted_successful_root_before_selecting(tmp_path):
    successful = _write_highrisk_bundle(
        tmp_path / "root0-success", protein_id="P1", root_index=0,
        endpoints=[{"endpoint_id": "p1:success"}],
    )
    failed = [
        _write_highrisk_bundle(
            tmp_path / f"root{root_index}-failed", protein_id="P1",
            root_index=root_index,
            endpoints=[{
                "endpoint_id": f"p1:failed:{root_index}",
                "structure_evaluated": False, "structure_feasible": False,
                "structure_result_feasible": False, "feasibility_level": "unvalidated",
                "structure_result_level": "unvalidated",
            }],
        )
        for root_index in (1, 2, 3)
    ]

    with pytest.raises(V2FacadeError, match="expected_master_seeds"):
        materialize_archive_facade(failed, mode="structure_rejected_fallback")
    with pytest.raises(V2FacadeError, match="master-seed grid.*missing.*20260811"):
        materialize_archive_facade(
            failed, mode="structure_rejected_fallback",
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )
    with pytest.raises(V2FacadeError, match="exactly 4 distinct master seeds"):
        materialize_archive_facade(
            failed, mode="structure_rejected_fallback",
            expected_master_seeds=EXPECTED_MASTER_SEEDS[1:],
        )
    # Supplying the omitted successful root proves the same protein is not fallback-eligible.
    with pytest.raises(V2FacadeError, match="candidate pool is empty"):
        materialize_archive_facade(
            [successful, *failed], mode="structure_rejected_fallback",
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )


def test_multiroot_completeness_refuses_duplicate_seed_and_different_seed_sets(tmp_path):
    duplicate_seed = [
        _write_highrisk_bundle(
            tmp_path / f"duplicate-{root_index}", protein_id="P1", root_index=root_index,
            master_seed=20260811 if root_index in (0, 1) else 20260810 + root_index,
            endpoints=[{"endpoint_id": f"p1:{root_index}"}],
        )
        for root_index in range(4)
    ]
    with pytest.raises(V2FacadeError, match="duplicate master_seed 20260811"):
        materialize_archive_facade(
            duplicate_seed, mode="feasible_immune_pareto",
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )

    bundles = [
        _write_highrisk_bundle(
            tmp_path / f"{protein_id}-{seed}", protein_id=protein_id, root_index=index,
            master_seed=seed,
            endpoints=[{"endpoint_id": f"{protein_id}:{seed}"}],
        )
        for protein_id, seeds in (
            ("P1", EXPECTED_MASTER_SEEDS),
            ("P2", (*EXPECTED_MASTER_SEEDS[:3], 20260815)),
        )
        for index, seed in enumerate(seeds)
    ]
    with pytest.raises(V2FacadeError, match="master-seed grid.*missing.*20260814"):
        materialize_archive_facade(
            bundles, mode="feasible_immune_pareto",
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )


@pytest.mark.parametrize(
    ("manifest_overrides", "message"),
    [
        ({"requested_cohort": ["P2"], "accepted_fragments": ["P2"]},
         "requested_cohort"),
        ({"arm_role": "v1"}, "arm_role"),
        ({"feedback_enabled": False}, "feedback_enabled"),
        ({"active_population_width": 2}, "active_population_width"),
        ({"seed_schema": "legacy"}, "seed_schema"),
        ({"seed_namespaces": ["v2_depth0_root"]}, "seed_namespaces"),
        ({"schema_version": "legacy"}, "schema_version"),
        ({"a2_matching_resource": "head_calls"}, "a2_matching_resource"),
        ({"accepted_fragments": []}, "accepted_fragments"),
        ({"caps": {}}, "caps"),
    ],
)
def test_multiroot_manifest_campaign_invariants_fail_closed(
    tmp_path, manifest_overrides, message,
):
    bundle = _write_highrisk_bundle(
        tmp_path / "root0", protein_id="P1", root_index=0,
        endpoints=[{"endpoint_id": "p1:root0"}],
        manifest_overrides=manifest_overrides,
    )
    with pytest.raises(V2FacadeError, match=message):
        materialize_archive_facade(
            [bundle], mode="feasible_immune_pareto",
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )


def test_multiroot_manifest_caps_must_match_across_seed_grid(tmp_path):
    bundles = []
    for root_index in range(4):
        overrides = None
        if root_index == 3:
            changed_caps = dict(DEFAULT_CAPS)
            changed_caps["max_head_calls"] = 5999
            overrides = {"caps": changed_caps}
        bundles.append(_write_highrisk_bundle(
            tmp_path / f"root{root_index}", protein_id="P1", root_index=root_index,
            endpoints=[{"endpoint_id": f"p1:{root_index}"}],
            manifest_overrides=overrides,
        ))

    with pytest.raises(V2FacadeError, match="caps"):
        materialize_archive_facade(
            bundles, mode="feasible_immune_pareto",
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )


@pytest.mark.parametrize(
    ("realized_caps", "message"),
    [
        ({
            "within": False,
            "breached": ["max_head_calls"],
            "unverifiable": [],
        }, "breached"),
        ({
            "within": True,
            "breached": [],
            "unverifiable": ["max_gpu_seconds"],
        }, "unverifiable"),
    ],
)
def test_multiroot_manifest_realized_caps_must_be_clean(
    tmp_path, realized_caps, message,
):
    bundle = _write_highrisk_bundle(
        tmp_path / "root0", protein_id="P1", root_index=0,
        endpoints=[{"endpoint_id": "p1:0"}],
        manifest_overrides={"realized_caps": realized_caps},
    )

    with pytest.raises(V2FacadeError, match=message):
        materialize_archive_facade(
            [bundle], mode="feasible_immune_pareto",
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )


@pytest.mark.parametrize(
    ("terminal_override", "message"),
    [
        ({"write_terminal": False}, "missing terminal-validation evidence"),
        ({"terminal_sequence_md5": "0" * 32}, "terminal sequence_md5"),
        ({"terminal_structure_definitive": False}, "structure_definitive"),
        ({"terminal_structure_feasible": False}, "terminal structure_feasible"),
        ({"terminal_immune_evaluator": "other-head"}, "immune_evaluator"),
        ({"terminal_immune_global_risk": 1.0}, "immune_global_risk"),
        ({"terminal_immune_passed": False}, "immune_passed"),
    ],
)
def test_feasible_pareto_requires_matching_terminal_evidence(
    tmp_path, terminal_override, message,
):
    bundles = []
    for root_index in range(4):
        spec = {"endpoint_id": f"p1:{root_index}", "sequence": "ACDEFGHIK"}
        if root_index == 0:
            spec.update(terminal_override)
        bundles.append(_write_highrisk_bundle(
            tmp_path / f"root{root_index}", protein_id="P1", root_index=root_index,
            endpoints=[spec],
        ))

    with pytest.raises(V2FacadeError, match=message):
        materialize_archive_facade(
            bundles, mode="feasible_immune_pareto",
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )


@pytest.mark.parametrize(
    ("structure_override", "message"),
    [
        ({"structure_backend_digest": "c" * 64}, "structure_backend_digest"),
        ({"v0_structure_gate_config_digest": "d" * 64},
         "v0_structure_gate_config_digest"),
        ({"structure_result_level": "definitive"}, "structure feasibility_level"),
    ],
)
def test_fallback_requires_manifest_bound_structure_evidence(
    tmp_path, structure_override, message,
):
    bundles = []
    for root_index in range(4):
        spec = {
            "endpoint_id": f"p1:{root_index}",
            "structure_evaluated": False,
            "structure_feasible": False,
            "structure_result_feasible": False,
            "feasibility_level": "unvalidated",
            "structure_result_level": "unvalidated",
        }
        if root_index == 0:
            spec.update(structure_override)
        bundles.append(_write_highrisk_bundle(
            tmp_path / f"root{root_index}", protein_id="P1", root_index=root_index,
            endpoints=[spec],
        ))

    with pytest.raises(V2FacadeError, match=message):
        materialize_archive_facade(
            bundles, mode="structure_rejected_fallback",
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protein_id", "P2", "head_score_json protein_id"),
        ("sequence_md5", "0" * 32, "head_score_json sequence_md5"),
        ("sequence_length", 8, "head_score_json sequence_length"),
        ("allele", "", "head_score_json allele"),
        ("score_scale", "", "head_score_json score_scale"),
        ("windows", [{"start_0b": 1, "end_0b": 9, "k": 8, "z": 0.0}],
         "head_window_grid_digest"),
    ],
)
def test_multiroot_head_payload_identity_is_verified(tmp_path, field, value, message):
    bundles = [
        _write_highrisk_bundle(
            tmp_path / f"root{root_index}", protein_id="P1", root_index=root_index,
            endpoints=[{"endpoint_id": f"p1:{root_index}"}],
        )
        for root_index in range(4)
    ]
    path = bundles[0] / "complete_endpoints.parquet"
    frame = pd.read_parquet(path)
    payload = json.loads(frame.loc[0, "head_score_json"])
    payload[field] = value
    frame.loc[0, "head_score_json"] = json.dumps(payload, sort_keys=True)
    frame.to_parquet(path, index=False)

    with pytest.raises(V2FacadeError, match=message):
        materialize_archive_facade(
            bundles, mode="feasible_immune_pareto",
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )


def _replace_all_head_payload_values(bundles, field, value):
    for bundle in bundles:
        path = bundle / "complete_endpoints.parquet"
        frame = pd.read_parquet(path)
        payload = json.loads(frame.loc[0, "head_score_json"])
        payload[field] = value
        frame.loc[0, "head_score_json"] = json.dumps(payload, sort_keys=True)
        frame.to_parquet(path, index=False)


def test_multiroot_refuses_consistently_wrong_head_payload_allele(tmp_path):
    bundles = [
        _write_highrisk_bundle(
            tmp_path / f"root{root_index}", protein_id="P1", root_index=root_index,
            endpoints=[{"endpoint_id": f"p1:{root_index}"}],
        )
        for root_index in range(4)
    ]
    _replace_all_head_payload_values(bundles, "allele", "DRB1_WRONG")

    with pytest.raises(V2FacadeError, match="head_evaluator_digest"):
        materialize_archive_facade(
            bundles, mode="feasible_immune_pareto",
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )


def test_multiroot_refuses_consistently_wrong_head_payload_score_scale(tmp_path):
    bundles = [
        _write_highrisk_bundle(
            tmp_path / f"root{root_index}", protein_id="P1", root_index=root_index,
            endpoints=[{"endpoint_id": f"p1:{root_index}"}],
        )
        for root_index in range(4)
    ]
    _replace_all_head_payload_values(bundles, "score_scale", "wrong_scale")

    with pytest.raises(V2FacadeError, match="head_evaluator_digest"):
        materialize_archive_facade(
            bundles, mode="feasible_immune_pareto",
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )


@pytest.mark.parametrize("endpoint_ids", [("p1:a", "p1:z"), ("p1:z", "p1:a")])
def test_converged_metrics_require_exact_identity_independent_of_endpoint_order(
    tmp_path, endpoint_ids,
):
    specs = [
        {
            "endpoint_id": endpoint_ids[0],
            "sequence": "ACDEFGHIK",
            "residue_hotspot": [1.0] * 9,
        },
        {
            "endpoint_id": endpoint_ids[1],
            "sequence": "ACDEFGHIK",
            "residue_hotspot": [1.0 + 5e-13] * 9,
        },
        {
            "endpoint_id": "p1:r2",
            "sequence": "GGGGGGGGG",
            "head_global_risk": 3.0,
            "residue_hotspot": [3.0] * 9,
        },
        {
            "endpoint_id": "p1:r3",
            "sequence": "TTTTTTTTT",
            "head_global_risk": 4.0,
            "residue_hotspot": [4.0] * 9,
        },
    ]
    bundles = [
        _write_highrisk_bundle(
            tmp_path / f"root{root_index}", protein_id="P1", root_index=root_index,
            endpoints=[spec],
        )
        for root_index, spec in enumerate(specs)
    ]

    with pytest.raises(V2FacadeError, match="conflicting head_positive_mass_density"):
        materialize_archive_facade(
            bundles, mode="feasible_immune_pareto",
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )


def test_converged_metrics_treat_positive_and_negative_zero_as_identical(tmp_path):
    specs = [
        {
            "endpoint_id": f"p1:{root_index}",
            "sequence": "ACDEFGHIK",
            "head_global_risk": -0.0 if root_index == 0 else 0.0,
            "residue_hotspot": [-0.0 if root_index == 0 else 0.0] * 9,
        }
        for root_index in range(4)
    ]
    bundles = [
        _write_highrisk_bundle(
            tmp_path / f"root{root_index}", protein_id="P1", root_index=root_index,
            endpoints=[spec],
        )
        for root_index, spec in enumerate(specs)
    ]

    out = materialize_archive_facade(
        bundles, mode="feasible_immune_pareto",
        expected_master_seeds=EXPECTED_MASTER_SEEDS,
    )

    assert len(out) == 1
    assert out.loc[0, "n_source_endpoints"] == 4
    assert out.loc[0, "head_positive_mass_density"] == 0.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda root: pd.concat([
            pd.read_parquet(root / "archive.parquet"),
            pd.read_parquet(root / "archive.parquet"),
        ]).to_parquet(root / "archive.parquet", index=False), "duplicate endpoint_id"),
        (lambda root: pd.DataFrame(columns=pd.read_parquet(
            root / "complete_endpoints.parquet").columns).to_parquet(
                root / "complete_endpoints.parquet", index=False), "no complete endpoint"),
        (lambda root: _replace_column(root / "complete_endpoints.parquet",
                                      "structure_feasible", False), "not definitive-feasible"),
        (lambda root: _replace_column(root / "complete_endpoints.parquet",
                                      "sequence", "ACD#"), "AA20"),
    ],
)
def test_malformed_or_ineligible_evidence_fails_closed(tmp_path, mutate, message):
    bundle = _write_bundle(
        tmp_path / "bundle", protein_id="P1",
        endpoints=[{"endpoint_id": "endpoint:only"}],
    )
    mutate(bundle)
    with pytest.raises(V2FacadeError, match=message):
        materialize_archive_facade([bundle], mode="elite")


def _replace_column(path, column, value):
    frame = pd.read_parquet(path)
    frame[column] = value
    frame.to_parquet(path, index=False)


def _drop_column(path, column):
    frame = pd.read_parquet(path).drop(columns=[column])
    frame.to_parquet(path, index=False)


def test_multiroot_modes_fail_closed_on_missing_head_or_structure_fields(tmp_path):
    feasible = _write_highrisk_bundle(
        tmp_path / "feasible", protein_id="P1", root_index=0,
        endpoints=[{"endpoint_id": "p1:ok"}],
    )
    _drop_column(feasible / "complete_endpoints.parquet", "head_score_json")
    with pytest.raises(V2FacadeError, match="head_score_json"):
        materialize_archive_facade(
            [feasible], mode="feasible_immune_pareto",
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )

    rejected = _write_highrisk_bundle(
        tmp_path / "rejected", protein_id="P2", root_index=0,
        endpoints=[{
            "endpoint_id": "p2:rejected", "structure_evaluated": False,
            "structure_feasible": False, "structure_result_feasible": False,
            "feasibility_level": "unvalidated", "structure_result_level": "unvalidated",
        }],
    )
    _drop_column(rejected / "structure_evaluations.parquet", "metrics_json")
    with pytest.raises(V2FacadeError, match="metrics_json"):
        materialize_archive_facade(
            [rejected], mode="structure_rejected_fallback",
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )


def test_multiroot_selection_validates_persisted_hard_anchors(tmp_path):
    manifest = tmp_path / "anchors.yaml"
    manifest.write_text(
        "schema_version: test-anchors-v1\n"
        "entries:\n"
        "  - protein_id: P1\n"
        "    hard_anchors:\n"
        "      - index_0b: 0\n"
        "        expected_aa: A\n"
        "        label: anchor_0\n"
        "        biological_role: test\n"
        "        source: synthetic\n"
    )
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rejected = [
        _write_highrisk_bundle(
            tmp_path / f"anchored-{root_index}", protein_id="P1", root_index=root_index,
            endpoints=[{
                "endpoint_id": f"p1:anchor-broken:{root_index}",
                "sequence": "CCCCCCCCC",
                "structure_evaluated": False,
                "structure_feasible": False,
                "structure_result_feasible": False,
                "feasibility_level": "unvalidated",
                "structure_result_level": "unvalidated",
            }],
            manifest_overrides={"content_identities": {
                "constraint_manifest": manifest_digest,
                "fixed_token_policy": manifest_digest,
            }},
        )
        for root_index in range(4)
    ]

    with pytest.raises(V2FacadeError, match="constraint manifest"):
        materialize_archive_facade(
            rejected, mode="structure_rejected_fallback",
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )
    with pytest.raises(V2FacadeError, match="hard anchor mismatch"):
        materialize_archive_facade(
            rejected, mode="structure_rejected_fallback",
            constraint_manifest=manifest,
            expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )


def test_a_protein_may_not_be_silently_merged_across_two_bundles(tmp_path):
    first = _write_bundle(tmp_path / "first", protein_id="P1",
                          endpoints=[{"endpoint_id": "endpoint:first"}])
    second = _write_bundle(tmp_path / "second", protein_id="P1",
                           endpoints=[{"endpoint_id": "endpoint:second"}])
    with pytest.raises(V2FacadeError, match="appears in more than one bundle"):
        materialize_archive_facade([first, second], mode="elite")
    with pytest.raises(V2FacadeError, match="expected_master_seeds.*only"):
        materialize_archive_facade(
            [first], mode="elite", expected_master_seeds=EXPECTED_MASTER_SEEDS,
        )


@pytest.mark.parametrize(
    ("manifest_overrides", "message"),
    [
        ({"campaign_id": "other-campaign"}, "campaign_id"),
        ({"code_revision": "cafebabe"}, "code_revision"),
        ({"schedule_id": "another-schedule"}, "schedule_id"),
        ({"coordinate_law": "stationary_checkpoint"}, "coordinate_law"),
        ({"depth_cap": 3}, "depth_cap"),
        ({"content_identities": {"structure_backend": "d" * 64}},
         "structure_backend"),
        ({"content_identities": {"v0_structure_gate_config": "e" * 64}},
         "v0_structure_gate_config"),
        ({"content_identities": {"structure_config": "f" * 64}},
         "structure_config"),
    ],
)
def test_bundles_from_different_experiments_cannot_be_merged(
    tmp_path, manifest_overrides, message,
):
    first = _write_bundle(
        tmp_path / "first", protein_id="P1",
        endpoints=[{"endpoint_id": "endpoint:first"}],
    )
    second = _write_bundle(
        tmp_path / "second", protein_id="P2",
        endpoints=[{"endpoint_id": "endpoint:second"}],
        manifest_overrides=manifest_overrides,
    )
    with pytest.raises(V2FacadeError, match=message):
        materialize_archive_facade([first, second], mode="elite")


@pytest.mark.parametrize(
    ("manifest_overrides", "message"),
    [
        ({"phase": "policy_qualification"}, "phase.*capability_ladder"),
        ({"split_role": "exploratory_other"}, "split_role.*exploratory_uricase"),
        ({"exploratory_depth_override": False}, "exploratory_depth_override"),
        ({"production_depth_authorized": True}, "production_depth_authorized"),
    ],
)
def test_every_bundle_must_be_an_exploratory_uricase_capability_run(
    tmp_path, manifest_overrides, message,
):
    bundle = _write_bundle(
        tmp_path / "bundle", protein_id="P1",
        endpoints=[{"endpoint_id": "endpoint:only"}],
        manifest_overrides=manifest_overrides,
    )
    with pytest.raises(V2FacadeError, match=message):
        materialize_archive_facade([bundle], mode="elite")


def test_missing_manifest_is_refused_before_parquet_merge(tmp_path):
    bundle = _write_bundle(
        tmp_path / "bundle", protein_id="P1",
        endpoints=[{"endpoint_id": "endpoint:only"}],
    )
    (bundle / "run_manifest.json").unlink()
    (bundle / "archive.parquet").unlink()  # manifest identity must fail before table discovery
    with pytest.raises(V2FacadeError, match="run_manifest"):
        materialize_archive_facade([bundle], mode="elite")


def test_protein_specific_content_identities_may_differ(tmp_path):
    first = _write_bundle(
        tmp_path / "first", protein_id="P1",
        endpoints=[{"endpoint_id": "endpoint:first"}],
        manifest_overrides={"content_identities": {
            "constraint_manifest": "1" * 64,
            "schedule_band_calibration": "2" * 64,
            "complete_reference_sequence": "3" * 64,
        }},
    )
    second = _write_bundle(
        tmp_path / "second", protein_id="P2",
        endpoints=[{"endpoint_id": "endpoint:second"}],
        manifest_overrides={"content_identities": {
            "constraint_manifest": "4" * 64,
            "schedule_band_calibration": "5" * 64,
            "complete_reference_sequence": "6" * 64,
        }},
    )
    assert list(materialize_archive_facade(
        [first, second], mode="elite",
    )["protein_id"]) == ["P1", "P2"]


def test_cli_output_is_accepted_by_the_existing_phase_c_loader(tmp_path):
    from scripts.evaluate_phase_c import load_generated_designs

    bundle = _write_bundle(tmp_path / "bundle", protein_id="P1",
                           endpoints=[{"endpoint_id": "endpoint:only"}])
    output = tmp_path / "generation" / "generated.parquet"
    assert main([
        "--bundle", str(bundle), "--mode", "elite", "--output", str(output),
    ]) == 0

    loaded = load_generated_designs(output)
    assert loaded.loc[0, "entry_source_id"] == "endpoint:only"
    assert loaded.loc[0, "design_id"] == "design_0000"


def test_multiroot_cli_binds_four_repeatable_master_seeds(tmp_path):
    bundles = [
        _write_highrisk_bundle(
            tmp_path / f"root{root_index}", protein_id="P1", root_index=root_index,
            endpoints=[{
                "endpoint_id": f"p1:{root_index}",
                "sequence": "ACDEFGHIK",
            }],
        )
        for root_index in range(4)
    ]
    output = tmp_path / "pareto.parquet"
    argv = ["--mode", "feasible_immune_pareto", "--output", str(output)]
    for bundle in bundles:
        argv.extend(["--bundle", str(bundle)])
    for seed in reversed(EXPECTED_MASTER_SEEDS):
        argv.extend(["--expected-master-seed", str(seed)])

    assert main(argv) == 0
    out = pd.read_parquet(output)
    assert len(out) == 1
    assert json.loads(out.loc[0, "master_seed_grid_json"])["master_seeds"] == list(
        EXPECTED_MASTER_SEEDS
    )


def test_multiroot_official_cli_writes_content_bound_selection_manifest(tmp_path):
    bundles = [
        _write_highrisk_bundle(
            tmp_path / f"root{root_index}", protein_id="P1", root_index=root_index,
            endpoints=[{
                "endpoint_id": f"p1:{root_index}",
                "sequence": "ACDEFGHIK",
            }],
        )
        for root_index in range(4)
    ]
    output = tmp_path / "official.parquet"
    argv = [
        "--mode", "feasible_immune_pareto",
        "--official",
        "--final-candidates-per-protein", "3",
        "--output", str(output),
    ]
    for bundle in bundles:
        argv.extend(["--bundle", str(bundle)])
    for seed in EXPECTED_MASTER_SEEDS:
        argv.extend(["--expected-master-seed", str(seed)])

    assert main(argv) == 0
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["official"] is True
    assert manifest["requested_candidates_per_protein"] == 3
    assert manifest["realized_counts"] == {"P1": 1}
    assert len(manifest["output_sha256"]) == 64
