# Single-Structure Design

`deimmunize.py` prepares one selected structure chain, runs four official MESSI
roots serially, and selects up to the requested number of unique feasible designs.
It reuses the existing materializer, driver and guarded official selector. It does
not run refinement, NetMHCIIpan, automatic calibration or extra roots to fill a quota.

Re-entry follows each source checkpoint: `r = c_source - reentry_offset`.
Use `--reentry-offset` to set the rollback distance (default 10). Missing per-step
bands are recorded as linear assumptions from empirical B40, not measured calibration.

## One-Time Setup

Install the [generation dependencies and ESMFold2 worker](installation.md) and
extract the [model packages](artifacts.md) into the repository root. Default model
locations are `dplm/checkpoints/best.ckpt` (with its packaged `.hydra/config.yaml`)
and `weights/drb{0701,0401,1501}/epoch_{29,34,24}.pt`.

Create `messi.local.json` using [this example](../examples/messi.local.example.json).
Set the actual ESMFold2 snapshot file, local model directory (containing that
snapshot and its config), and isolated worker site-packages directory. This local
settings file is Git-ignored. For a different settings location use `--settings`
or set `MESSI_CONFIG`. Relative paths resolve against the settings file directory;
CLI path overrides resolve against your current directory.

Models outside the default layout can be configured in `common` or an `alleles`
mapping keyed by `DRB1_0701`, `DRB1_0401`, or `DRB1_1501`:

```json
{
  "common": {
    "dplm_checkpoint": "/path/to/dplm/checkpoints/best.ckpt",
    "structure_backend": "/path/to/ESMFold2/model.safetensors",
    "esmfold2_model": "/path/to/ESMFold2",
    "esmfold2_site_packages": "/path/to/esmfold2/site-packages"
  },
  "alleles": {
    "DRB1_1501": {"head_checkpoint": "/path/to/weights/drb1501/epoch_24.pt"}
  }
}
```

Precedence is bundled defaults, common settings, allele settings, then CLI.
Unknown settings fail rather than being ignored. Other supported settings are
`head_config_dir`, `refold_cache_dir`, `band_json`, `hotspot_json`,
`policy_calibration_json`, and `stratum_key`.

## Run

```bash
python deimmunize.py --structure input.pdb --chain A \
  --allele DRB1_1501 --num-designs 8 --out-dir results/
```

PDB, `.cif` and `.mmcif` are supported. Chain IDs are author chain IDs; the default
model index is zero. The default protein ID is the filename stem; set `--protein-id`
to match a constraint entry or replace an unsafe filename. References are the
resolved sequence of the selected chain, not an inferred full biological sequence.
Missing backbone atoms, duplicate residue numbering and chains shorter than 50
resolved residues are refused. Ligands/water are not part of the design chain.
Canonicalization and altloc selection reuse the existing structure preparation.

All four roots are preflighted before generation. To perform only preparation and
preflight, add `--dry-run` and use a separate output directory such as `checks/`.
Preflight does not load weights or validate a live GPU/worker inference call.
Existing output directories are never overwritten; use a new directory to run.

## Defaults And Options

- Official D4/K12/r40, four single-lineage roots at seeds 20260811-20260814,
  100 steps and scTM >= 0.70. ESMFold2 uses 3 loops, 50 sampling steps,
  one sample and seed zero. Search settings do not scale with `--num-designs`.
- Bundled calibration files under `examples/calibration/` are exported from the
  original official runs, with source hashes and unchanged scientific blocks.
  Allele-specific policy thresholds remain those in `examples/messi_official.json`.
  They are not universal calibration claims for arbitrary models or constraints.
  Schedule-band files preserve the original strict schema and bytes (SHA-256
  `34df15ab305028a2d7caed31b2efa60e2dc0fd9bf9661587ded7f70592658a0c`).
- `--master-seeds` accepts exactly four distinct nonnegative seeds.
- `--constraint-manifest` preserves hard anchors but requires an explicitly supplied,
  compatible `--band-json` and `--stratum-key`; the default band is unconstrained.
  Constraints index the extracted chain from zero, not PDB author residue numbers.
  Sequence hashes and allele labels are checked when supplied by the manifest.
- Model/cache/calibration paths can also be overridden directly on the CLI.
  Replacing the Head/config with a different instrument requires compatible signed
  calibration; the wrapper never manufactures or silently rebinds measured evidence.
- `python deimmunize.py --help` lists all options without importing model libraries.

## Output

`generated.parquet` and `designs.fasta` contain the selected sequences.
`generated.manifest.json` records the official selection contract; `summary.json`
records requested and realized counts. Fewer than N designs is a valid result;
the wrapper never pads, duplicates or weakens a gate. If the existing selector
refuses an empty feasible pool, the command exits nonzero and retains the evidence;
structure-rejected fallback is not silently exported as a successful design.

`inputs/` contains the cleaned chain, cohort, reference sequence, author-residue
mapping and resolved per-root configs. `commands.json` records argument arrays;
`roots/` retains every full evidence bundle. Runtime failures or incomplete/breached
root evidence stop the wrapper. No generated shell fragment is executed.
