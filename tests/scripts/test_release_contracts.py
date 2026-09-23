"""Release defaults and provenance survive removal of the research entry points."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from scripts.refine_rf_designs import build_arg_parser
from scripts.rf_fusion_v2_cohort import ShardInputs, _terminal_stop_label
from scripts.run_rf_fusion_v2 import resolve_dual
from inverse_folding.reference_flow.fusion_v2.errors import V2Error
from inverse_folding.reference_flow.fusion_v2.dual_config import dual_overlay_payload
from inverse_folding.reference_flow.fusion_v2.joint_objective import arm_objective_digest
from tests.inverse_folding.test_fusion_v2_dual_config import overlay
from tests.inverse_folding.test_fusion_v2_dual_arms import ARMS, _objective


def test_selected_release_constraints_and_full_data_recipes():
    import hashlib
    from inverse_folding.reference_flow.constraints import load_constraint_manifest

    root = Path(__file__).resolve().parents[2]
    entries = json.loads((root / 'examples/constraints/manifest.json').read_text())['constraints']
    assert len(entries) == 12
    for item in entries:
        path = root / item['file']
        expected = item.get('sha256', item['source_sha256'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        manifest = load_constraint_manifest(path)
        assert manifest.num_hard_anchors_total == item['n_fixed']
        entry = yaml.safe_load(path.read_text())['entries'][0]
        if 'free_positions_0b' in entry:
            seq = entry['source_sequence']
            assert hashlib.md5(seq.encode()).hexdigest() == entry['sequence_md5']
            constraint = manifest.constraint_for_protein(item['protein_id'])
            constraint.validate_against_sequence(seq)
            fixed = set(constraint.hard_anchor_indices)
            free = set(entry['free_positions_0b'])
            assert not fixed & free
            assert fixed | free == set(range(len(seq)))
    for allele, epochs in [('0701', 30), ('0401', 35), ('1501', 25)]:
        train = yaml.safe_load((root / f'epitope_head/configs/full_data/drb{allele}.yaml').read_text())['train']
        assert train['max_epochs'] == epochs
        assert train['early_stopping_patience'] > epochs
        assert train['loss']['objective_mode'] == 'mixed_margin'
        assert train['loss']['lambda_iou_rank'] == 1.0
        assert train['residue']['lambda_residue'] == 0.3


def test_official_and_family_recipes_are_not_interchangeable():
    root = Path(__file__).resolve().parents[2]
    official = json.loads((root / 'examples/messi_official.json').read_text())
    family = json.loads((root / 'examples/uricase_family_historical.json').read_text())
    template = yaml.safe_load((root / official['template']).read_text())
    assert template['caps']['max_retries'] == 0
    assert official['schedule']['depth_cap'] == 4
    assert family['schedule']['depth_cap'] == 3
    assert family['local_contribution_tolerance'] == 0
    for allele, config in official['alleles'].items():
        assert len(config['root_seeds']) == 4
        assert config['caps'] == template['caps']
        expected = 1.9073486328125e-06 if allele == '0701' else 0.017012596130371094
        assert config['head_directed']['local_contribution_tolerance']['value'] == expected


def test_public_refinement_defaults_to_head_and_exact_front():
    args = build_arg_parser().parse_args(['--allele', 'A', '--out-dir', 'out'])
    assert args.target_source == 'head'
    assert args.official is True


def test_merge_cli_runs_without_an_inherited_pythonpath():
    import sys

    root = Path(__file__).resolve().parents[2]
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    result = subprocess.run(
        [sys.executable, str(root / 'scripts/merge_refine_shards.py'), '--help'],
        cwd=root, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout.lower()


def test_dual_overlay_must_match_declared_content_not_path(tmp_path):
    expected = overlay()
    first = tmp_path / 'declared.json'
    alias = tmp_path / 'alias.json'
    wrong = tmp_path / 'wrong.json'
    first.write_text(json.dumps(dual_overlay_payload(expected)))
    alias.write_text(json.dumps(dual_overlay_payload(expected), indent=2))
    wrong.write_text(json.dumps(dual_overlay_payload(overlay(objective_spec_digest='a' * 64))))
    args = SimpleNamespace(dual_overlay=str(alias), dual_arm='joint')
    inputs = ShardInputs(dual_overlay=str(first))
    assert resolve_dual(args, shard_inputs=inputs)[0].content_digest == expected.content_digest
    args.dual_overlay = str(wrong)
    with pytest.raises(V2Error, match='differs'):
        resolve_dual(args, shard_inputs=inputs)
    with pytest.raises(V2Error, match='dual-overlay'):
        resolve_dual(args, shard_inputs=ShardInputs())


def test_all_arm_values_use_the_same_canonical_identity_constructor():
    identities = set()
    for arm in ARMS:
        objective = _objective(arm)
        value = objective.evaluate(raw_a=-1.0, raw_b=2.0)
        assert value.objective_digest == arm_objective_digest(objective.calibration, arm)
        assert value.objective_digest == objective.objective_digest
        identities.add(value.objective_digest)
    assert len(identities) == 3


def test_terminal_stall_is_not_reported_as_invalid_geometry():
    outcome = SimpleNamespace(stopping_reason=SimpleNamespace(value='invalid_projection'),
        cycles=[SimpleNamespace(cycle=SimpleNamespace(
            policy_evidence=SimpleNamespace(stall_reason='stall_no_better_donor'),
            outcome=SimpleNamespace(value='null_invalid_policy_result')))])
    assert _terminal_stop_label(outcome) == 'stall_no_better_donor'
    outcome.cycles[0].cycle.policy_evidence = None
    assert _terminal_stop_label(outcome) == 'null_invalid_policy_result'
    outcome.stopping_reason = SimpleNamespace(value='depth_cap')
    assert _terminal_stop_label(outcome) == 'depth_cap'
