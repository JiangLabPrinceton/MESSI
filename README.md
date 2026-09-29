# MESSI

Protein deimmunization with discrete diffusion. MESSI redesigns protein sequences
for a target MHC-II allele while enforcing structural and optional residue constraints.

## Run MESSI

### 1. Install And Download Models

```bash
git clone https://github.com/JiangLabPrinceton/MESSI.git
cd MESSI
```

Follow the [installation guide](docs/installation.md), including the generation
dependencies and ESMFold2 backend. Download the pretrained models using the
[model setup guide](docs/artifacts.md). Retraining is not required.

### 2. Prepare Inputs And Generate Designs

You need reference sequences and structures, a target allele, model paths, and
matching calibration inputs. The standard experiment is recorded in
[the MESSI configuration](examples/messi_official.json).

| Step | Script |
|---|---|
| Prepare a run configuration | `scripts/materialize_v2_canary_config.py` |
| Generate designs | `scripts/run_rf_fusion_v2.py` |
| Select final candidates | `scripts/materialize_v2_archive_facade.py` |

Start with the [generation instructions](docs/workflows.md#fusion-v2) and
[configuration requirements](docs/configs.md#official-messi).
Materialize each run, validate its arguments with `--dry-run`, then run the same
generation command without that flag. Use `--help` on any script for its inputs.

**Input requirement:** the example configuration is a template, not a complete
runnable dataset. Reference and calibration files must be supplied and matched
to the selected models before generation.

### 3. Select And Save Candidates

After all declared roots finish, select up to eight designs per protein:

```bash
python scripts/materialize_v2_archive_facade.py \
  --bundle run/root0 --bundle run/root1 --bundle run/root2 --bundle run/root3 \
  --expected-master-seed 20260811 --expected-master-seed 20260812 \
  --expected-master-seed 20260813 --expected-master-seed 20260814 \
  --mode feasible_immune_pareto --final-candidates-per-protein 8 \
  --output run/selected/generated.parquet
```

Replace the bundle paths and seeds with your run's values. Each bundle represents
one protein/root; include every cell for multi-protein runs. The output contains
selected sequences and their scores/provenance. Keep the full run archives.

## Optional Steps

- [Refine generated designs](docs/workflows.md#head-refinement).
- [Evaluate immune and structural properties](docs/workflows.md#evaluation).
- [Apply target-specific residue constraints](docs/configs.md#selected-target-constraints).
- [All scripts and advanced usage](docs/scripts.md).

## License

[MIT](LICENSE). Bundled source and external models retain their
[respective terms](THIRD_PARTY_NOTICES.md).
