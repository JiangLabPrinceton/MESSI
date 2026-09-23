"""Head and definitive structure oracles shared by the official workflow."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def _require(path, what):
    if not path:
        raise SystemExit(f"missing required input: {what}")
    return path


def build_oracles(args, config):
    """Wire the real Head / structure / DPLM-repair oracles. NetMHCIIpan is never imported."""
    from inverse_folding.evaluation.refold import load_refold_model, refold
    from inverse_folding.evaluation.tmalign import run_tmalign
    from inverse_folding.reference_flow.constraints import load_constraint_manifest
    from inverse_folding.reference_flow.runtime import resolve_structure_path

    _require(args.head_checkpoint, "--head-checkpoint")
    _require(args.refold_cache_dir, "--refold-cache-dir")
    _require(args.test_set_parquet, "--test-set-parquet")
    _require(args.pdb_root, "--pdb-root")

    # Head: OnlineHeadScorer -> head_fn(protein_id, sequences) -> [HeadScore]
    from scripts.head_runtime import build_head_scorer  # namespace import
    scorer = _build_head_scorer_from_args(args, build_head_scorer)

    def head_fn(protein_id, sequences):
        records = [(f"cand{i}", s) for i, s in enumerate(sequences)]
        batch = scorer.score_batch_same_protein(protein_id=protein_id, records=records)
        return list(batch.scores)

    # Structure: TMalign provides scTM; active-site geometry is selected explicitly by config.
    backend = config.structure.backend
    backend_options = {}
    if backend == "esmfold2_live":
        backend_options = {
            "site_packages": _require(
                args.esmfold2_site_packages, "--esmfold2-site-packages"
            ),
            "model_name": args.esmfold2_model,
            "num_loops": args.esmfold2_num_loops,
            "num_sampling_steps": args.esmfold2_num_sampling_steps,
            "num_diffusion_samples": args.esmfold2_num_diffusion_samples,
            "seed": args.esmfold2_seed,
        }
    model = load_refold_model(backend, device=args.head_device, **backend_options)
    args._refold_backend = backend
    args._refold_runtime = getattr(model, "metadata", None)
    test_df = pd.read_parquet(args.test_set_parquet)
    test_lookup = {str(r["protein_id"]): r for _, r in test_df.iterrows()}
    manifest = load_constraint_manifest(args.constraint_manifest) if args.constraint_manifest else None
    ref_path_cache: dict[str, Path] = {}

    def _reference_path(pid: str) -> Path:
        if pid not in test_lookup:
            raise KeyError(f"protein {pid!r} is absent from --test-set-parquet")
        if pid not in ref_path_cache:
            ref_path_cache[pid] = Path(resolve_structure_path(test_lookup[pid], args.pdb_root))
        return ref_path_cache[pid]

    shell_indices_fn = None
    compute_ca_self_consistency_fn = None
    reference_context_fn = None
    evaluate_prediction_fn = None
    v2_metric_names = None
    if config.structure.active_site_metric == "legacy_ca_shell":
        import numpy as np

        from inverse_folding.evaluation.sc_rmsd import (
            compute_ca_self_consistency as compute_ca_self_consistency_fn,
            parse_ca_trace,
        )
        from inverse_folding.reference_flow.fusion import structure_metrics as sm

        shell_cache: dict[str, frozenset[int]] = {}

        def shell_indices_fn(pid, ref_pdb, anchor_idx):
            if pid not in shell_cache:
                ca = parse_ca_trace(ref_pdb)
                coords = np.vstack([res.coord for res in ca]).astype(float)
                shell_cache[pid] = sm.shell_indices_from_ca(
                    coords,
                    anchor_idx,
                    config.structure.active_site_shell_radius,
                )
            return shell_cache[pid]
    elif config.structure.active_site_metric == "sidechain_max_anchor":
        from inverse_folding.evaluation.structural_metrics_v2 import (
            METRIC_GLOBAL_CA_RMSD,
            METRIC_PLDDT,
            METRIC_SIDECHAIN,
            evaluate_prediction as evaluate_prediction_fn,
            prepare_reference_context,
        )

        reference_context_cache: dict[str, object] = {}
        v2_metric_names = {
            METRIC_GLOBAL_CA_RMSD,
            METRIC_PLDDT,
            METRIC_SIDECHAIN,
        }

        def reference_context_fn(pid: str):
            if pid not in reference_context_cache:
                ref_sequence = str(test_lookup[pid]["sequence"])
                if manifest is not None and manifest.has_protein(pid):
                    manifest.constraint_for_protein(pid).validate_against_sequence(ref_sequence)
                reference_context_cache[pid] = prepare_reference_context(
                    _reference_path(pid),
                    ref_sequence=ref_sequence,
                    anchor_indices=_anchor_indices(manifest, pid),
                )
            return reference_context_cache[pid]

    from inverse_folding.reference_flow.refine import StructureMetrics

    def struct_fn(protein_id, sequence):
        pred = refold(sequence, protein_id, "fusion", backend=backend,
                      cache_dir=args.refold_cache_dir, model=model)
        ref = _reference_path(protein_id)
        tm = run_tmalign(pred_pdb=str(pred["pdb_path"]), ref_pdb=str(ref), cache_dir=None)
        asr = None
        v2 = None
        anchor_idx = _anchor_indices(manifest, protein_id)
        if config.structure.active_site_metric == "legacy_ca_shell":
            if anchor_idx:
                shell = shell_indices_fn(protein_id, str(ref), anchor_idx)
                _scr, per_res = compute_ca_self_consistency_fn(
                    pred_pdb=str(pred["pdb_path"]), ref_pdb=str(ref), protein_id=protein_id,
                    design_id="fusion", design_idx=0, design_sequence=sequence,
                    ref_sequence=str(test_lookup[protein_id]["sequence"]),
                    refold_backend=backend)
                asr = sm.shell_rmsd_from_rows(per_res, shell)
        elif config.structure.active_site_metric == "sidechain_max_anchor":
            v2 = evaluate_prediction_fn(
                reference_context_fn(protein_id),
                str(pred["pdb_path"]),
                design_sequence=sequence,
                metrics=v2_metric_names,
            )
        plddt = (
            v2.predicted_global_plddt
            if v2 is not None and v2.predicted_global_plddt is not None
            else float(pred["pLDDT"])
        )
        return StructureMetrics(
            scTM=float(tm["tm_score"]),
            pLDDT=float(plddt),
            scRMSD=float(tm["rmsd"]),
            active_site_RMSD=asr,
            global_ca_RMSD=None if v2 is None else v2.global_ca_rmsd,
            active_site_sidechain_RMSD=(
                None if v2 is None else v2.active_site_sidechain_rmsd
            ),
            max_anchor_sidechain_RMSD=(
                None if v2 is None else v2.max_anchor_sidechain_rmsd
            ),
            max_anchor_atom_distance=(
                None if v2 is None else v2.max_anchor_atom_distance
            ),
            active_site_complete=None if v2 is None else v2.active_site_complete,
            active_site_min_pLDDT=(
                None if v2 is None else v2.predicted_active_site_min_plddt
            ),
            cache_hit=bool(pred.get("cache_hit", False)),
            model_executed=not bool(pred.get("cache_hit", False)),
        )

    from inverse_folding.reference_flow.fusion.oracles import FusionOracles
    return FusionOracles(head_fn=head_fn, struct_fn=struct_fn), manifest


def _anchor_indices(manifest, protein_id):
    # ConstraintManifest has NO .get(): use has_protein()/constraint_for_protein() (verified).
    if manifest is None or not manifest.has_protein(protein_id):
        return set()
    return set(manifest.constraint_for_protein(protein_id).hard_anchor_indices)


def _build_head_scorer_from_args(args, build_head_scorer):
    from pathlib import Path as _P
    from types import SimpleNamespace
    setup = SimpleNamespace(
        head_config_dir=_P(args.head_config_dir), head_checkpoint=_P(args.head_checkpoint),
        head_variant_id=args.head_variant_id, head_device=args.head_device,
        head_allele_idx=args.head_allele_idx, head_window_batch_size=args.head_window_batch_size,
        allele=args.allele, config=SimpleNamespace(head=SimpleNamespace(score_scale="raw_logit")))
    return build_head_scorer(setup, window_k_min=args.window_k_min, window_k_max=args.window_k_max)


def build_arg_parser():
    p = argparse.ArgumentParser(description="RF-Refine Fusion driver (Head-only, structure-gated).")
    p.add_argument("--fusion-config", required=True)
    p.add_argument("--run-dir", required=True, help="directory with the generated designs parquet")
    p.add_argument("--generated-parquet", default=None, help="explicit generated.parquet path")
    p.add_argument("--proteins", nargs="*", default=None, help="protein filter (default: all)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--allele", required=True)
    p.add_argument("--test-set-parquet", default=None)
    p.add_argument("--pdb-root", default=None)
    p.add_argument(
        "--refold-cache-dir",
        "--esmfold-cache-dir",
        dest="refold_cache_dir",
        default=None,
        help="shared structure cache; --esmfold-cache-dir is a deprecated alias",
    )
    p.add_argument(
        "--esmfold2-site-packages",
        default=None,
        help="Biohub esm/transformers site-packages overlay for backend=esmfold2_live",
    )
    p.add_argument("--esmfold2-model", default="biohub/ESMFold2")
    p.add_argument("--esmfold2-num-loops", type=int, default=3)
    p.add_argument("--esmfold2-num-sampling-steps", type=int, default=50)
    p.add_argument("--esmfold2-num-diffusion-samples", type=int, default=1)
    p.add_argument("--esmfold2-seed", type=int, default=0)
    p.add_argument("--constraint-manifest", default=None)
    # required only when rf_reopen / edit-repair is enabled (the H3 arm)
    p.add_argument("--base-if-checkpoint", default=None, help="base IF checkpoint for the DPLM repair sampler")
    p.add_argument("--rf-sampler-config", default=None, help="ReferenceFlow YAML for the repair sampler")
    p.add_argument("--head-checkpoint", default=None)
    p.add_argument("--head-config-dir", default=None)
    p.add_argument("--head-variant-id", default=None)
    p.add_argument("--head-allele-idx", type=int, default=0)
    p.add_argument("--head-window-batch-size", type=int, default=64)
    p.add_argument("--head-device", default="cuda")
    p.add_argument("--window-k-min", type=int, default=12)
    p.add_argument("--window-k-max", type=int, default=25)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--print-config", action="store_true")
    return p
