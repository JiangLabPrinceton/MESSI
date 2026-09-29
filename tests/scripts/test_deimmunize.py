import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import deimmunize as entry


def args_for(tmp_path, *extra):
    return entry.build_parser().parse_args([
        '--structure', str(tmp_path / 'input.pdb'), '--chain', 'A',
        '--allele', 'HLA-DRB1*15:01', '--out-dir', str(tmp_path / 'out'), *extra])


def make_structure(tmp_path, suffix='.pdb', missing=False):
    from Bio.PDB import Atom, Chain, Model, Residue, Structure, PDBIO, MMCIFIO
    import numpy as np

    structure = Structure.Structure('test')
    model = Model.Model(0)
    structure.add(model)
    serial = 1
    for chain_name, resname in [('A', 'ALA'), ('B', 'GLY')]:
        chain = Chain.Chain(chain_name)
        model.add(chain)
        for index in range(50):
            residue = Residue.Residue((' ', index + 10, ' '), resname, '')
            chain.add(residue)
            for atom_name, element in [('N', 'N'), ('CA', 'C'), ('C', 'C'), ('O', 'O')]:
                if missing and index == 1 and atom_name == 'O' and chain_name == 'A':
                    continue
                residue.add(Atom.Atom(atom_name, np.array([index * 3.8, serial % 3, 0.0]),
                    80., 1., ' ', f'{atom_name:>4}', serial, element=element))
                serial += 1
    path = tmp_path / ('input' + suffix)
    io = MMCIFIO() if suffix != '.pdb' else PDBIO()
    io.set_structure(structure)
    io.save(str(path))
    return path


def test_defaults_and_lightweight_help(tmp_path):
    import subprocess
    import sys

    args = args_for(tmp_path)
    assert args.allele == 'DRB1_1501' and args.num_designs == 8
    assert args.master_seeds == [20260811, 20260812, 20260813, 20260814]
    assert entry.allele_name('HLA-DRB1_15_01') == args.allele
    script = "import deimmunize,sys; assert 'torch' not in sys.modules; deimmunize.build_parser().parse_args(['--help'])"
    result = subprocess.run([sys.executable, '-S', '-c', script], cwd=entry.ROOT, capture_output=True)
    assert result.returncode == 0


def test_bundled_calibration_matches_frozen_official_values():
    from scripts.materialize_v2_canary_config import head_config_hash
    from inverse_folding.reference_flow.fusion_v2.schedule import load_band_table
    from inverse_folding.reference_flow.fusion_v2.config import (
        PolicyCalibrationArtifact, policy_calibration_source_ref)
    recipe = json.loads((entry.ROOT / 'examples/messi_official.json').read_text())
    for tag, expected in recipe['alleles'].items():
        load_band_table(entry.ROOT / f'examples/calibration/drb{tag}/global_B_r40.json')
        data = json.loads((entry.ROOT / f'examples/calibration/drb{tag}/head_policy.json').read_text())
        assert data['head_directed'] == expected['head_directed']
        assert data['head']['head_checkpoint_digest'] == expected['head_checkpoint_sha256']
        assert data['head']['head_config_hash'] == head_config_hash(entry.ROOT / 'epitope_head/configs')
        for key in ('donor_improvement_epsilon', 'local_contribution_tolerance', 'write_cap_editable_fraction'):
            scalar = data['head_directed'][key]
            assert scalar['source_ref'] == policy_calibration_source_ref(
                value=scalar['value'], unit=scalar['unit'], source_kind=scalar['source_kind'],
                source_id=scalar['source_id'], artifact=PolicyCalibrationArtifact(**scalar['artifact']))


def test_settings_relative_paths_cli_precedence_and_typo_refusal(tmp_path):
    config = tmp_path / 'local.json'
    values = {key: key for key in entry.PATH_OPTIONS}
    for key in values:
        if key in {'head_config_dir', 'esmfold2_site_packages', 'refold_cache_dir'}:
            (tmp_path / key).mkdir()
        else:
            (tmp_path / key).write_text('asset')
    values.update(stratum_key='global', esmfold2_model='model')
    config.write_text(json.dumps({'common': values}))
    other = tmp_path / 'other.pt'
    other.write_text('checkpoint')
    args = args_for(tmp_path, '--settings', str(config), '--head-checkpoint', str(other))
    result = entry.read_settings(args)
    assert result['head_checkpoint'] == str(other)
    assert result['band_json'] == str(tmp_path / 'band_json')
    args.constraint_manifest = tmp_path / 'constraints.yaml'
    args.stratum_key = 'highrisk_global_unconstrained'
    with pytest.raises(ValueError, match='constraints require'):
        entry.read_settings(args)
    args.constraint_manifest = None
    args.stratum_key = None
    values['head_chekpoint'] = 'wrong'
    config.write_text(json.dumps({'common': values}))
    with pytest.raises(ValueError, match='unknown installation'):
        entry.read_settings(args)


@pytest.mark.parametrize('suffix', ['.pdb', '.cif', '.mmcif'])
def test_selected_chain_and_author_mapping(tmp_path, suffix):
    import pandas as pd

    args = args_for(tmp_path)
    args.structure = make_structure(tmp_path, suffix)
    args.chain = 'B'
    args.protein_id = 'target'
    target = tmp_path / 'inputs'
    target.mkdir()
    backbone = entry.prepare_structure(args, target)
    assert backbone.exists()
    row = pd.read_parquet(target / 'cohort.parquet').iloc[0]
    assert row['sequence'] == 'G' * 50 and row['if_chain_id'] == 'B'
    mapping = json.loads((target / 'residue_map.json').read_text())
    assert mapping[0]['author_resnum'] == 10 and mapping[-1]['index_0b'] == 49


def test_incomplete_backbone_and_wrong_chain_are_not_silently_dropped(tmp_path):
    args = args_for(tmp_path)
    args.structure = make_structure(tmp_path, missing=True)
    args.protein_id = 'target'
    target = tmp_path / 'inputs'
    target.mkdir()
    with pytest.raises(ValueError, match='incomplete backbone'):
        entry.prepare_structure(args, target)
    args.chain = 'Z'
    with pytest.raises(ValueError, match='chain_not_found'):
        entry.prepare_structure(args, target)


def test_constraint_wrong_sequence_and_anchor_fail_closed(tmp_path):
    args = args_for(tmp_path)
    args.structure = make_structure(tmp_path)
    args.protein_id = 'target'
    target = tmp_path / 'inputs'
    target.mkdir()
    args.constraint_manifest = tmp_path / 'constraints.yaml'
    data = {'schema_version': 'test', 'entries': [{'protein_id': 'target',
        'sequence_md5': '0' * 32, 'hard_anchors': [{'index_0b': 0, 'expected_aa': 'A'}]}]}
    args.constraint_manifest.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='sequence_md5'):
        entry.prepare_structure(args, target)
    del data['entries'][0]['sequence_md5']
    data['entries'][0]['hard_anchors'][0]['expected_aa'] = 'G'
    args.constraint_manifest.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='anchor mismatch'):
        entry.prepare_structure(args, target)


def test_all_preflights_precede_gpu_and_dry_run_never_generates(tmp_path, monkeypatch):
    calls = []
    bundles = [tmp_path / str(i) for i in range(4)]
    for bundle in bundles:
        bundle.mkdir()
        (bundle / 'run_manifest.json').write_text(json.dumps(dict(accepted_fragments=['ok'],
            missing_proteins=[], rejected_fragments=[],
            realized_caps=dict(within=True, breached=[], unverifiable=[]))))
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=2 if command == ['root2'] else 0)
    monkeypatch.setattr(entry.subprocess, 'run', run)
    commands = [[f'root{i}'] for i in range(4)]
    entry.execute(commands, bundles, dry_run=True)
    assert calls == [[*c, '--dry-run'] for c in commands]
    calls.clear()
    entry.execute(commands, bundles, dry_run=False)
    assert calls == [[*c, '--dry-run'] for c in commands] + commands
    (bundles[0] / 'run_manifest.json').write_text('{}')
    with pytest.raises(RuntimeError, match='incomplete'):
        entry.execute(commands, bundles, dry_run=False)


def test_failed_preflight_prevents_all_generation(tmp_path, monkeypatch):
    import subprocess
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(4, command)
    monkeypatch.setattr(entry.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        entry.execute([['root0'], ['root1']], [tmp_path/'0', tmp_path/'1'], dry_run=False)
    assert calls == [['root0', '--dry-run']]


def test_existing_output_and_missing_settings_refused_before_writes(tmp_path):
    source = make_structure(tmp_path)
    out = tmp_path / 'out'
    base = ['--structure', str(source), '--chain', 'A', '--allele', 'DRB1_1501', '--out-dir', str(out)]
    assert entry.main([*base, '--settings', str(tmp_path / 'missing.json')]) == 2
    assert not out.exists()
    out.mkdir()
    marker = out / 'keep'
    marker.write_text('unchanged')
    assert entry.main(base) == 2
    assert marker.read_text() == 'unchanged'


def test_selector_output_is_not_padded(tmp_path, monkeypatch):
    import pandas as pd
    from scripts import materialize_v2_archive_facade as facade

    args = args_for(tmp_path)
    args.out_dir.mkdir()
    args.protein_id = 'target'
    observed = []
    def select(argv):
        observed.extend(argv)
        pd.DataFrame([{'protein_id':'target', 'design_idx':0, 'sequence':'A'*50}]).to_parquet(
            args.out_dir / 'generated.parquet')
    monkeypatch.setattr(facade, 'main', select)
    assert entry.select_outputs(args, [tmp_path / str(i) for i in range(4)]) == 1
    assert observed[observed.index('--final-candidates-per-protein') + 1] == '8'
    assert observed.count('--expected-master-seed') == 4
    assert (args.out_dir / 'designs.fasta').read_text().count('>') == 1
