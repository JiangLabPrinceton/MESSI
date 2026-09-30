#!/usr/bin/env python3
"""Single-structure entry point for the existing official MESSI pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PATH_OPTIONS = (
    'dplm_checkpoint', 'head_checkpoint', 'head_config_dir', 'structure_backend',
    'esmfold2_site_packages', 'band_json', 'hotspot_json', 'policy_calibration_json',
    'refold_cache_dir',
)
SETTINGS_OPTIONS = frozenset((*PATH_OPTIONS, 'stratum_key', 'esmfold2_model'))
ALLELES = ('DRB1_0701', 'DRB1_0401', 'DRB1_1501')


def allele_name(value):
    compact = value.upper().removeprefix('HLA-').replace('*', '').replace(':', '').replace('_', '')
    normalized = 'DRB1_' + compact.removeprefix('DRB1')
    if normalized not in ALLELES:
        raise argparse.ArgumentTypeError(f'allele must be one of {ALLELES}')
    return normalized


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--structure', type=Path, required=True, help='input PDB or mmCIF')
    parser.add_argument('--chain', required=True, help='author chain ID')
    parser.add_argument('--allele', type=allele_name, required=True)
    parser.add_argument('--num-designs', type=positive_int, default=8,
                        help='maximum unique feasible outputs; not a search budget')
    parser.add_argument('--out-dir', type=Path, required=True, help='new output directory')
    parser.add_argument('--settings', type=Path,
                        default=os.environ.get('MESSI_CONFIG', ROOT / 'messi.local.json'),
                        help='one-time model and calibration configuration')
    parser.add_argument('--protein-id', help='constraint entry ID; defaults to structure filename stem')
    parser.add_argument('--model-index', type=int, default=0, help='zero-based structure model')
    parser.add_argument('--constraint-manifest', type=Path,
                        help='sequence-bound constraints indexed into the extracted chain')
    parser.add_argument('--master-seeds', type=int, nargs=4,
                        default=[20260811, 20260812, 20260813, 20260814],
                        help='four distinct nonnegative root seeds')
    parser.add_argument('--dry-run', action='store_true',
                        help='prepare and validate every root without loading models')
    parser.add_argument('--reentry-offset', type=positive_int, default=10,
                        help='re-enter this many steps before each source checkpoint')
    advanced = parser.add_argument_group('installation overrides (otherwise read from settings)')
    for name in PATH_OPTIONS:
        advanced.add_argument('--' + name.replace('_', '-'), type=Path)
    advanced.add_argument('--stratum-key')
    advanced.add_argument('--esmfold2-model', help='local model selector matching structure-backend')
    return parser


def read_settings(args):
    path = Path(args.settings).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'Installation settings not found: {path}. See docs/deimmunize.md; '
                         'set MESSI_CONFIG or pass --settings.')
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or set(data) - {'common', 'alleles'}:
        raise ValueError('settings must contain only common and alleles mappings')
    alleles = data.get('alleles', {})
    if not isinstance(alleles, dict) or set(alleles) - set(ALLELES):
        raise ValueError(f'alleles settings must be keyed by {ALLELES}')
    tag = args.allele.removeprefix('DRB1_')
    calibration = ROOT / f'examples/calibration/drb{tag}'
    assets = json.loads((ROOT / 'head-assets.json').read_text())
    head = next(a for a in assets['assets']
                if a['file'].endswith('.pt') and allele_name(a['allele']) == args.allele)
    values = dict(head_checkpoint=str(ROOT / head['file']),
        dplm_checkpoint=str(ROOT / 'dplm/checkpoints/best.ckpt'),
        band_json=str(calibration / 'global_B_r40.json'),
        hotspot_json=str(calibration / 'global_relaxed_hotspot.json'),
        policy_calibration_json=str(calibration / 'head_policy.json'),
        stratum_key='highrisk_global_unconstrained')
    for block in (data.get('common', {}), alleles.get(args.allele, {})):
        if not isinstance(block, dict) or set(block) - SETTINGS_OPTIONS:
            raise ValueError('unknown installation option; see docs/deimmunize.md')
        for key, value in block.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{key} must be a nonempty string')
            if key in PATH_OPTIONS:
                p = Path(value).expanduser()
                value = str((path.parent / p).resolve())
            values[key] = value
    for key in SETTINGS_OPTIONS:
        value = getattr(args, key, None)
        if value is not None:
            values[key] = str(value.expanduser().resolve()) if key in PATH_OPTIONS else value
    values.setdefault('head_config_dir', str(ROOT / 'epitope_head/configs'))
    values.setdefault('refold_cache_dir', str(args.out_dir.resolve() / 'refold_cache'))
    if args.constraint_manifest and (
        values['band_json'] == str(calibration / 'global_B_r40.json')
        or values['stratum_key'] == 'highrisk_global_unconstrained'
    ):
        raise ValueError('constraints require an explicit compatible --band-json and --stratum-key; '
                         'the default band is unconstrained')
    required = SETTINGS_OPTIONS - {'refold_cache_dir'}
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f'Missing installation settings: {", ".join(missing)}')
    for key in PATH_OPTIONS:
        if key == 'refold_cache_dir':
            continue
        p = Path(values[key])
        valid = p.is_dir() if key in {'head_config_dir', 'esmfold2_site_packages'} else p.is_file()
        if not valid:
            raise ValueError(f'{key} does not resolve to the required file/directory: {p}')
    return values


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def prepare_structure(args, target):
    import pandas as pd
    from scripts.build_if_ready_test_set import extract_resolved_residues, write_clean_structure
    from inverse_folding.reference_flow.constraints import load_constraint_manifest

    source = args.structure.expanduser().resolve()
    # Bio.PDB's existing helper accepts .cif; normalize .mmcif without altering input bytes.
    if source.suffix.lower() == '.mmcif':
        local = target / 'source.cif'
        local.write_bytes(source.read_bytes())
        source = local
    structure, residues, stats = extract_resolved_residues(
        source, chain_id=args.chain, model_idx=args.model_index)
    if stats['skipped_missing_backbone'] or stats['duplicate_author_residue_id']:
        raise ValueError('selected chain has incomplete backbone or duplicate residue numbering')
    sequence = ''.join(r.aa for r in residues)
    if len(sequence) < 50:
        raise ValueError('selected chain must contain at least 50 resolved residues')
    if args.constraint_manifest:
        manifest = load_constraint_manifest(args.constraint_manifest)
        entry = manifest.constraint_for_protein(args.protein_id)
        entry.validate_against_sequence(sequence)
        import yaml
        raw = yaml.safe_load(args.constraint_manifest.read_text())
        selected = next(e for e in raw['entries'] if e['protein_id'] == args.protein_id)
        if selected.get('sequence_md5') and selected['sequence_md5'] != hashlib.md5(sequence.encode()).hexdigest():
            raise ValueError('constraint sequence_md5 does not match selected chain')
        if selected.get('allele') and allele_name(selected['allele']) != args.allele:
            raise ValueError('constraint allele does not match requested allele')
    backbone = target / (args.protein_id + '.cif')
    write_clean_structure(structure=structure, clean_path=backbone,
        model_idx=args.model_index, chain_id=args.chain, resolved=residues)
    pd.DataFrame([dict(protein_id=args.protein_id, sequence=sequence,
        sequence_length=len(sequence), pdb_path=str(backbone), if_chain_id=args.chain,
        if_ready=True)]).to_parquet(target / 'cohort.parquet', index=False)
    reference = target / 'reference.seq'
    reference.write_text(sequence)
    write_json(target / 'references.json', {args.protein_id: {
        'path': str(reference), 'sha256': hashlib.sha256(reference.read_bytes()).hexdigest()}})
    write_json(target / 'residue_map.json', [dict(index_0b=i, aa=r.aa,
        author_resnum=r.author_resnum, insertion_code=r.insertion_code,
        original_resname=r.original_resname) for i, r in enumerate(residues)])
    return backbone


def prepare_commands(args, settings, target, backbone):
    import yaml
    from scripts import materialize_v2_canary_config as materializer
    from inverse_folding.reference_flow.fusion_v2.schedule import load_band_table

    load_band_table(settings['band_json'])
    recipe = json.loads((ROOT / 'examples/messi_official.json').read_text())
    official = recipe['alleles'][args.allele.removeprefix('DRB1_')]
    policy = json.loads(Path(settings['policy_calibration_json']).read_text())
    for key in ('donor_improvement_epsilon', 'local_contribution_tolerance', 'write_cap_editable_fraction'):
        if policy.get('head_directed', {}).get(key, {}).get('value') != official['head_directed'][key]['value']:
            raise ValueError(f'policy {key} differs from the official {args.allele} recipe')
    template = yaml.safe_load((ROOT / recipe['template']).read_text())
    template['head']['allele'] = args.allele
    template_path = target / 'template.yaml'
    template_path.write_text(yaml.safe_dump(template))
    write_json(target / 'strata.json', {args.protein_id: settings['stratum_key']})
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    gate = str(ROOT / 'inverse_folding/reference_flow/configs/rf_refine_fusion_highrisk_sctm070.yaml')
    common = dict(settings, template=str(template_path), protein_id=args.protein_id,
        r_step=40, code_revision=revision, reference_manifest=str(target / 'references.json'),
        stratum_manifest=str(target / 'strata.json'), reference_sequence=str(target / 'reference.seq'),
        projection_policy_spec=str(ROOT / 'inverse_folding/reference_flow/configs/v2_head_directed_capped_policy_v2.json'),
        head_variant_id='LC1', structure_config=gate, v0_structure_gate_config=gate,
        rf_sampler_config=str(ROOT / 'inverse_folding/reference_flow/configs/c1_constant_clean_no_remask.yaml'),
        cohort_table=str(target / 'cohort.parquet'), backbone=str(backbone), pdb_root=str(target),
        exploratory_profile=recipe['profile'], reentry_offset=args.reentry_offset,
        run_max_head_calls=6000,
        esmfold2_num_loops=3, esmfold2_num_sampling_steps=50, esmfold2_num_diffusion_samples=1,
        esmfold2_seed=0, campaign_id='deimmunize-' + args.out_dir.name)
    if args.constraint_manifest:
        common['constraint_manifest'] = str(args.constraint_manifest)
    commands, bundles = [], []
    for seed in args.master_seeds:
        config_path = target / f'root_{seed}.yaml'
        options = dict(common, out=str(config_path), master_seed=seed)
        argv = [word for key, value in options.items() for word in ('--' + key.replace('_', '-'), str(value))]
        materializer.main(argv)
        parsed = materializer.build_parser().parse_args(argv)
        identity = materializer._structure_runtime_identity(parsed)
        frozen, runtime = materializer.resolve_content_bindings(parsed, structure_runtime_identity=identity)
        files, inputs = materializer.build_driver_inputs(parsed, frozen=frozen, runtime=runtime,
            config_path=config_path, structure_runtime_identity=identity)
        bundle = args.out_dir / 'roots' / str(seed)
        commands.append([sys.executable, str(ROOT / 'scripts/run_rf_fusion_v2.py'),
            '--v2-config', str(config_path), '--cohort', args.protein_id,
            '--input-file', *files, '--shard-input', *[f'{k}={v}' for k,v in sorted(inputs.items())],
            '--out-dir', str(bundle), '--exploratory-depth-override'])
        bundles.append(bundle)
    return commands, bundles


def execute(commands, bundles, *, dry_run):
    for command in commands:
        subprocess.run([*command, '--dry-run'], cwd=ROOT, check=True)
    if dry_run:
        return
    for command, bundle in zip(commands, bundles, strict=True):
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode not in (0, 2):
            raise RuntimeError(f'root failed with exit {result.returncode}: {bundle}')
        # Exit 2 can be a complete typed no-design result or a runtime failure.
        manifest = json.loads((bundle / 'run_manifest.json').read_text())
        caps = manifest.get('realized_caps') or {}
        if (manifest.get('missing_proteins') or manifest.get('rejected_fragments')
                or len(manifest.get('accepted_fragments', [])) != 1
                or caps.get('within') is not True or caps.get('breached') != []
                or caps.get('unverifiable') != []):
            raise RuntimeError(f'root evidence/caps are incomplete: {bundle}')


def select_outputs(args, bundles):
    import pandas as pd
    from scripts.materialize_v2_archive_facade import main as select

    output = args.out_dir / 'generated.parquet'
    argv = ['--mode', 'feasible_immune_pareto', '--official',
            '--final-candidates-per-protein', str(args.num_designs), '--output', str(output)]
    for bundle in bundles:
        argv += ['--bundle', str(bundle)]
    for seed in args.master_seeds:
        argv += ['--expected-master-seed', str(seed)]
    if args.constraint_manifest:
        argv += ['--constraint-manifest', str(args.constraint_manifest)]
    select(argv)
    frame = pd.read_parquet(output)
    with (args.out_dir / 'designs.fasta').open('w') as handle:
        for row in frame.itertuples():
            handle.write(f'>{row.protein_id}_design_{row.design_idx}\n{row.sequence}\n')
    return len(frame)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.model_index < 0 or len(set(args.master_seeds)) != 4 or min(args.master_seeds) < 0:
            raise ValueError('model index and four distinct master seeds must be nonnegative')
        args.structure = args.structure.expanduser().resolve()
        args.out_dir = args.out_dir.expanduser().resolve()
        args.protein_id = args.protein_id or args.structure.stem
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.protein_id):
            raise ValueError('protein-id must be a filename-safe identifier; use --protein-id')
        if args.constraint_manifest:
            args.constraint_manifest = args.constraint_manifest.expanduser().resolve()
        if not args.structure.is_file() or args.structure.suffix.lower() not in {'.pdb', '.cif', '.mmcif'}:
            raise ValueError('--structure must name an existing PDB/mmCIF file')
        if args.out_dir.exists():
            raise ValueError(f'output directory already exists; choose a new --out-dir: {args.out_dir}')
        settings = read_settings(args)
        target = args.out_dir / 'inputs'
        target.mkdir(parents=True)
        backbone = prepare_structure(args, target)
        commands, bundles = prepare_commands(args, settings, target, backbone)
        write_json(args.out_dir / 'commands.json', commands)
        execute(commands, bundles, dry_run=args.dry_run)
        if args.dry_run:
            summary = dict(status='dry_run', generated=0, requested=args.num_designs)
        else:
            count = select_outputs(args, bundles)
            summary = dict(status='complete' if count == args.num_designs else 'fewer_than_requested',
                           generated=count, requested=args.num_designs)
        write_json(args.out_dir / 'summary.json', summary)
        print(json.dumps(summary))
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'deimmunize: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
