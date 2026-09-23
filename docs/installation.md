# Installation

Use Python 3.12. Head inference, selection, and CPU metrics do not require DPLM,
ESMFold2, NetMHCIIpan, or a scheduler.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, substitute the CUDA 12.8 wheel index
`https://download.pytorch.org/whl/cu128` in the PyTorch command. Consult the
[PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/#v271)
for other platforms. Do not install the CPU and CUDA wheels together.

## Generation

The modified DPLM source is under `inverse_folding/dplm/`. It requires the
checkpoint's ByProt experiment configuration and corresponding model weights;
a standalone checkpoint without its configuration is insufficient. Keep their
relative layout and provide the checkpoint through the CLI.

Extend a separate environment with the generation dependencies:

```bash
python -m pip install -r requirements-generation.txt
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
```

For PyTorch 2.7.1 with CUDA 12.8, use the corresponding
`https://data.pyg.org/whl/torch-2.7.0+cu128.html` scatter wheel index. The runtime
loads the bundled ByProt and OpenFold Python sources. It does not require the
old cluster-specific editable installs. See the
[inverse-folding artifact layout](artifacts.md#inverse-folding-checkpoint).
The generation backend is validated separately from the CPU Head environment.

## Structure Backends

ESMFold2 uses an isolated worker. Supply its site-packages directory through
`--esmfold2-site-packages`; do not install its `esm` package over the generation
environment's `fair-esm`. Select `--refold-model esmfold2_live` for live refinement,
or `esmfold2` with `--refold-cache-dir` to evaluate an existing normalized cache.

Protenix and AlphaFold3 are optional external installations. Their precompute
CLIs accept interpreter/wrapper paths and inputs explicitly. Use the respective
upstream installation and weight-access instructions. NetMHCIIpan and Rosetta
are also external tools; supply their executable paths only when requested.

## Tests

```bash
python -m pip install pytest==9.0.2
python -m pytest tests -q
```

All paths and sharding options are supplied through Python CLIs. No SLURM
launcher, cluster module command, or local binary shim is required for Head use.
