# Official Workflows

See [Configuration Scope](configs.md) for the exact official-run recipe,
selected target constraints and checkpoint-derived full-data training configs.

## Head

Use `scripts/score_head.py` for JSON inference and `scripts/precompute_h_maps.py`
for sequence-bound residue tables. The canonical model is `a1res03`, profile
`LC1`. Production checkpoints are fixed-epoch full-data fits: DRB1*07:01 epoch
29, DRB1*04:01 epoch 34, and DRB1*15:01 epoch 24. Their training-monitor scores
are not held-out performance estimates. Use the separate CV models for that purpose.

The three inference config files retain their original bytes because existing
evaluator identities bind their content. Historical profile entries in that
identity-bearing file do not change the public LC1 default.

`scripts/train_head.py` defaults to LC1 with the canonical `a1res03` override.
For a production refit, use `--max-epochs` and `--no-early-stopping`, and supply
the appropriate full-data split. Epoch numbers in checkpoint filenames are
zero-based; epoch 29 requires 30 training epochs. `build_cv_folds.py` and
`cv_select_and_aggregate.py` retain the CV preparation and evaluation workflow.

## Fusion V2

Prepare configs with `scripts/materialize_v2_canary_config.py`. Its name is
historical; it resolves production/capability profiles as well as small probes.
It binds sampler, Head, structure, reference sequence, policy, schedule-band,
and constraint inputs, and refuses unresolved or mismatched identities.
Use `--help` for the full explicit input contract. Generated per-run files
belong outside the source tree.

Validate the materialized input arguments with `scripts/run_rf_fusion_v2.py
--dry-run`, then run the same command without that flag. Supply the materialized
`--v2-config`, `--input-file`, `--shard-input`, `--cohort`, and `--out-dir` arguments.
Recursive capability runs require the explicit depth authorization flag already
recorded by the materializer. A dry run does not load model weights.

For Dual mode, the command-line `--dual-overlay` must match the content of the
cell's `--shard-input dual_overlay=...`. Supply `--dual-arm` explicitly. Calibration
and objective identities are distinct: the objective additionally binds the arm.

## Official Selection

```bash
python scripts/materialize_v2_archive_facade.py \
  --bundle run/root0 --bundle run/root1 --bundle run/root2 --bundle run/root3 \
  --expected-master-seed 20260811 --expected-master-seed 20260812 \
  --expected-master-seed 20260813 --expected-master-seed 20260814 \
  --mode feasible_immune_pareto --final-candidates-per-protein 8 \
  --output run/selected/generated.parquet
```

Replace the seeds with the exact campaign seed grid. Every protein must have
all declared roots. Converged sequences retain all root provenance. The default
official panel ranks the complete eligible pool by Pareto layer, full-pool rank
sum, and sequence digest. `--no-official` with no count override exports the
complete first front. Structure-rejected fallback is a separate explicit mode;
it cannot become a feasible final product during evaluation.

## Head Refinement

`scripts/refine_rf_designs.py` defaults to Head targeting and official front
retention. Pass the seed table, target structures, allele-matched Head weights,
Head config directory, refold backend/cache, output directory, and gate parameters.
It does not invoke NetMHCIIpan in Head mode. `--no-official` retains admitted history.

The protocol gate requires explicit `--gate-scTM-min`,
`--gate-cat-max-scRMSD-max`, and `--gate-predicted-active-site-min-pLDDT-min`, plus
the constraint manifest. The historical seed-relative gate is an explicit
`--structure-gate-profile legacy` option with its own `--scTM-eps`; do not
substitute one gate for the other when reproducing a run.

`examples/refinement_standard.json` records the frozen six-round seed-relative
profile as a list of CLI arguments. Append its `cli_args` to the required input,
model, and output arguments when reproducing that profile; it is not a protocol
gate configuration and is not loaded implicitly by the driver.

Each parent proposes singles at residue-local maxima and scores bounded full
double mutants. Both global risk and positive hotspot mass divided by sequence
length must improve without either worsening. Hard anchors remain fixed.

For independent workers, pass `--n-shards` and `--shard-idx`, with separate
output directories. Finalize with `scripts/merge_refine_shards.py --run-dir DIR
--official --final-candidates-per-protein 8`. The complete per-seed fronts remain
available; the final panel takes at most one minimum-global-risk representative
per seed and preserves seed groups.

## Evaluation

`scripts/evaluate_phase_c.py` accepts `--generated-parquet`, `--test-set-parquet`,
`--allele`, `--mode {imm,struct,all}`, and `--output-root`. For Head-only immune
evaluation, supply `--no-nmp` and the epitope checkpoint/config arguments.
For independent NMP evaluation, supply the executable path instead. `--imm-full`
adds residue and peptide tables. Selection authority propagates to immune tables.

Structure output uses canonical v2 metrics. Active-site errors use matched
side-chain heavy atoms; a global CA RMSD is retained separately. Residue tables
contain atom counts and summed squared errors, allowing exact later aggregation
over selected indices. Missing structure/WT evidence is never filled with a
fabricated baseline. `eval_complex_gate.py` adds complex/interface metrics to
existing predictions.
