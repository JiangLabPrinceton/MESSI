# Python Entry Points And Shared APIs

Run CLI scripts from the repository root with `python scripts/NAME.py --help`.
All run-specific paths are arguments. No SLURM files are distributed.

## Head

| File | Purpose |
|---|---|
| `scripts/score_head.py` | Canonical LC1 checkpoint inference to sequence JSON |
| `scripts/train_head.py` | Canonical a1res03 training and fixed-epoch production refit |
| `scripts/precompute_h_maps.py` | Sequence-bound residue landscape tables |
| `scripts/build_cv_folds.py` | Cluster-aware cross-validation split construction |
| `scripts/cv_select_and_aggregate.py` | Select and aggregate CV evaluations |
| `scripts/benchmark_iedb_test.py` | Head and NMP comparison against epitope annotations |
| `scripts/benchmark_head_vs_nmp.py` | Independent NMP comparison on FASTA inputs |
| `scripts/score_sequences_dual_heads.py` | Score the same sequences under two explicit Heads |
| `scripts/head_runtime.py` | Shared predictor construction and config identity; import-only |

## Generation And Selection

| File | Purpose |
|---|---|
| `scripts/run_rf_fusion_v2.py` | Bound Fusion V2 generation and model-free preflight |
| `scripts/materialize_v2_canary_config.py` | Resolve profile and content identities into a runnable config |
| `scripts/preflight_v2_canary_assembly.py` | Validate assembled runtime inputs |
| `scripts/materialize_v2_archive_facade.py` | Validated multiroot official panels or complete Pareto fronts |
| `scripts/refine_rf_designs.py` | Head-guided residue refinement and exact front retention |
| `scripts/merge_refine_shards.py` | Join worker outputs and produce lineage-balanced final panels |
| `scripts/run_if_phase_c0.py` | DPLM-native inverse-folding baseline |
| `scripts/run_proteinmpnn_baseline.py` | External ProteinMPNN baseline adapter |
| `scripts/build_if_ready_test_set.py` | Prepare structure-aligned target inputs |
| `scripts/build_wt_facade.py` | Convert real WT inputs to evaluation format |
| `scripts/split_facade_shards.py` | Partition generated tables for independent workers |

## Calibration And Replay

| File | Purpose |
|---|---|
| `scripts/rho_maturity_scan.py` | Measure schedule/maturity bands on the declared substrate |
| `scripts/calibrate_rf_fusion_v2_hotspot.py` | Freeze the Head-domain hotspot admission calibration |
| `scripts/calibrate_v2_head_policy.py` | Measure Head repeatability and policy margins |
| `scripts/build_dual_calibration_panel.py` | Construct an allele-neutral calibration panel |
| `scripts/calibrate_v2_dual_objective.py` | Fit and bind Dual objective coordinates |
| `scripts/analysis/replay_v2_head_directed_policy.py` | Reproduce stored policy decisions without generation |
| `scripts/analysis/read_v2_mechanism.py` | Validate and summarize paired mechanism evidence |

## Evaluation And Prediction

| File | Purpose |
|---|---|
| `scripts/evaluate_phase_c.py` | Head/NMP/structure evaluation with selection authority preserved |
| `scripts/merge_eval_immune_shards.py` | Reconcile independent immune evaluation outputs |
| `scripts/merge_windows_cache_shards.py` | Merge bound window-score caches |
| `scripts/predict_esmfold2_gt.py` | Predict structures with the isolated ESMFold2 installation |
| `scripts/precompute_protenix_refold.py` | Protenix prediction adapter |
| `scripts/precompute_af3_refold.py` | AlphaFold3 prediction adapter |
| `scripts/build_protenix_jsons.py` | Assemble predictor inputs and MSA references |
| `scripts/build_binder_complex_inputs.py` | Construct binder-complex predictor inputs |
| `scripts/eval_complex_gate.py` | Complex confidence, structure, ligand, and interface metrics |
| `scripts/eval_tetramer_reference.py` | Metrics and optional energy scans for a reference complex |

## Internal Runtime Modules

These modules are imported by the entry points, not launched as separate jobs:

- `scripts/rf_fusion_model_factory.py`: shared model/backbone setup and token identity.
- `scripts/rf_shared_oracles.py`: Head and definitive structure oracle assembly.
- `scripts/rf_artifact_io.py`: stable typed Parquet, atomic JSON, and ledger writers.
- `scripts/rf_fusion_v2_oracles.py`: config-bound V2 oracle stack.
- `scripts/rf_fusion_v2_preflight.py`: config and budget validation.
- `scripts/rf_fusion_v2_cohort.py`: per-protein execution and aggregation.
- `scripts/rf_fusion_v2_artifacts.py`: public artifact schemas and evidence tables.
- `scripts/rf_fusion_v2_resume.py`: content-bound resume state.

Pure two-axis ordering is available from
`inverse_folding.reference_flow.official_selection.two_axis_pareto_order`.
Selective side-chain/CA/pLDDT metrics are available from
`inverse_folding.evaluation.structural_metrics_v2`.
