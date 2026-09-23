# MESSI

MHC-II-aware protein design with MESSI (Fusion V2), a residue-level epitope Head,
Head-guided refinement, and independent immune/structure evaluation.

## Installation And Models

Use Python 3.12 and follow [installation](docs/installation.md) for the CPU or
CUDA environment. Generation and structure prediction need the optional backends
described there. All workflows use Python; scheduling is managed by the caller.

Model packages are separate from Git. See [model artifacts](docs/artifacts.md),
[Head checksums](head-assets.json), and [DPLM checksums](dplm-assets.json).
Extract Head weights under `weights/` and preserve the packaged DPLM checkpoint
and `.hydra/config.yaml` layout. External structure-model weights and NetMHCIIpan
must be obtained separately under their own terms.

## Experiment Configurations

| Experiment | Configuration | Python entry points |
|---|---|---|
| Head scoring | `epitope_head/configs/`, allele-matched checkpoint | `scripts/score_head.py`, `scripts/precompute_h_maps.py` |
| Full-data Head training | `epitope_head/configs/full_data/drb{0701,0401,1501}.yaml` | `scripts/train_head.py` |
| Single-allele MESSI official | `examples/messi_official.json`, `examples/messi_official.template.yaml` | `scripts/materialize_v2_canary_config.py`, `scripts/run_rf_fusion_v2.py` |
| Multiroot selection | Same campaign's declared seed grid | `scripts/materialize_v2_archive_facade.py` |
| Head refinement | `examples/refinement_standard.json` | `scripts/refine_rf_designs.py`, `scripts/merge_refine_shards.py` |
| Constrained targets | `examples/constraints/manifest.json` and per-target YAMLs | Supply `--constraint-manifest` to materialization/refinement |
| Dual-allele MESSI | Signed dual overlay and allele-bound calibration artifacts | `scripts/run_rf_fusion_v2.py --dual-overlay ... --dual-arm ...` |
| Independent evaluation | Generated sequence table, reference cohort, model/backend paths | `scripts/evaluate_phase_c.py` |
| Historical uricase family experiment | `examples/uricase_family_historical.json` | Historical recipe only; its production profile is not exposed by this release's materializer |

The JSON recipes describe settings; they are not automatically loaded as runtime
configs. [Configuration details](docs/configs.md) distinguish the official D4/K12,
four-root experiment from the constrained uricase D3/K12, two-root experiment.
Do not substitute one for the other.

## Head Scoring

```bash
python scripts/score_head.py \
  --checkpoint weights/drb1501/epoch_24.pt \
  --input-fasta examples/sequence.fasta \
  --device cpu --output-dir run/head
```

Outputs include window scores, residue hotspots and global risk. The example
FASTA is synthetic, for installation checks only. The canonical model is
`a1res03` / `LC1`; positive mass density is
`sum(max(residue_hotspot, 0)) / sequence_length`.

## Full-Data Head Training

Supply the allele's manifest directory with `splits/strict/full/train_ids.txt`
and `val_ids.txt`, then run:

```bash
python scripts/train_head.py --variant-id LC1 --seed 42 --profile strict \
  --data-dir DATA_DIR --splits-subdir full --device cuda \
  --override-config epitope_head/configs/full_data/drb1501.yaml \
  --max-epochs 25 --no-early-stopping --output-root run/head_training
```

| Allele | Training epochs | Published checkpoint |
|---|---:|---|
| DRB1*07:01 | 30 | `epoch_29.pt` |
| DRB1*04:01 | 35 | `epoch_34.pt` |
| DRB1*15:01 | 25 | `epoch_24.pt` |

The training overrides preserve the full-data checkpoints' resolved parameters.
Full-data validation curves are not held-out results; use fixed-epoch checkpoints,
not `best.pt`. CV preparation/evaluation uses `scripts/build_cv_folds.py` and
`scripts/cv_select_and_aggregate.py`.

## MESSI Generation

The official single-allele recipe uses 100 RF steps, D4/K12/r40, checkpoints
50/60/70/80/90, four independent roots and a definitive scTM >= 0.70 gate.
Realized policy thresholds are allele-specific in `examples/messi_official.json`.

Use the official template with the materializer's
`--exploratory-profile highrisk_d4_k12_r40 --run-max-head-calls 6000` options.
Supply all required reference, cohort, sampler, checkpoint, policy, calibration
and structure inputs explicitly:

```bash
python scripts/materialize_v2_canary_config.py --help
python scripts/run_rf_fusion_v2.py --help
```

Materialization creates content-bound cell configs and launch arguments. Pass
the resulting `--v2-config`, `--input-file`, `--shard-input`, `--cohort` and
`--out-dir` arguments to the driver. First add `--dry-run` for validation; remove
it to execute. Preserve the explicit depth-authorization argument emitted by
the materializer. See [generation workflow](docs/workflows.md#fusion-v2).
The example template is intentionally not executable before its input identities
are resolved. Schedule-band and policy artifacts must match the actual models
and editable domain; they are not interchangeable between experiments.

## Official Selection

After all declared roots have completed, select up to eight designs per protein:

```bash
python scripts/materialize_v2_archive_facade.py \
  --bundle run/root0 --bundle run/root1 --bundle run/root2 --bundle run/root3 \
  --expected-master-seed 20260811 --expected-master-seed 20260812 \
  --expected-master-seed 20260813 --expected-master-seed 20260814 \
  --mode feasible_immune_pareto --final-candidates-per-protein 8 \
  --output run/selected/generated.parquet
```

Each bundle is one protein/root cell; repeat `--bundle` for every cell in the
campaign. Selection verifies grid completeness, feasibility and provenance.
Official output uses Pareto layers of global risk and positive mass density.
`--no-official` without a count override exports the complete first front.
Keep the original archives; the output-count limit is not a search-budget limit.

## Head Refinement And Target Constraints

`examples/refinement_standard.json` contains the standard six-round profile's
`cli_args`. Append that list to the required input/model/output arguments when
invoking `scripts/refine_rf_designs.py`; the JSON is not loaded implicitly.
The profile uses residue-local maxima above 0.15, singles, bounded double-mutant
scoring and beam accumulation. Head search does not invoke NetMHCIIpan.

```bash
python scripts/refine_rf_designs.py --help
python scripts/merge_refine_shards.py --run-dir run/refinement \
  --official --final-candidates-per-protein 8
```

Provide the seed table, reference cohort/backbones, Head checkpoint/config,
ESMFold2 worker/cache and output directory. Use `--n-shards`/`--shard-idx` for
independent workers. The standard profile has a seed-relative structure gate;
the separate protocol gate requires explicit thresholds and must not be silently
substituted. See [refinement workflow](docs/workflows.md#head-refinement).

The selected target manifests are LuxSit-i (22 fixed positions), PD1_b2 (57 fixed
positions), and DRB1501 uricases A0A9P8P4R1, A0A100I4D7, Q00511, P25689, Q6P700,
Q9RV70, O74409, P25688, P04670 and P16164. Pass the appropriate YAML from
`examples/constraints/` as `--constraint-manifest`. These masks are
sequence-specific, not templates for homologs or other alleles.

## Independent Evaluation

Head-only immune evaluation:

```bash
python scripts/evaluate_phase_c.py \
  --generated-parquet run/selected/generated.parquet \
  --test-set-parquet COHORT.parquet --allele 'HLA-DRB1*15:01' \
  --mode imm --no-nmp --imm-full --device cpu \
  --epitope-ckpt weights/drb1501/epoch_24.pt \
  --epitope-config-dir epitope_head/configs --output-root run/evaluation
```

For independent NetMHCIIpan scoring, omit `--no-nmp` and supply
`--netmhciipan-bin`. For structure evaluation, use `--mode struct` or `all`
with the required backend/cache and reference paths. Active-site metrics use
matched side-chain heavy atoms; a global C-alpha RMSD is retained separately.
Protenix/AF3 precompute and complex-interface evaluation entry points are listed
in the [script index](docs/scripts.md).

## Validation And License

```bash
python -m pip install pytest==9.0.2
python -m pytest tests -q
```

Source is distributed under [MIT](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md)
for bundled source; external tools and model weights retain their own terms.
