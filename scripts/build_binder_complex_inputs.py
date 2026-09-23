#!/usr/bin/env python
"""Stage a designed-binder cohort for a Protenix COMPLEX prediction + complex-gate evaluation.

A binder's function is its interface, so every binding decision has to be computed on a complex
(PROTOCOL/binder_interface_deimm_selection.md). This script turns one or more RF
``generated.parquet`` cohorts into the two artifacts that path needs:

  1. ``msa_cf/<pred_id>.a3m``  -- the per-design binder MSA, consumed by
     ``build_protenix_jsons.py --msa-cf``. The target partner's deep a3m is copied in under its
     own id so the same directory serves both chains.
  2. ``pred_manifest.parquet`` (pred_id, sequence_pred) for ``build_protenix_jsons.py``, and
     ``gate_manifest.parquet`` (id, name, parent, kind, allele, sequence) for
     ``eval_complex_gate.py``, including the PARENT WT row that the gate normalizes against.

MSA depth is deliberately query-only for the binder chain. A de novo binder has no homologs: the
parent's own ColabFold search returned its query and nothing else, so running a search per design
would spend GPU-hours to reproduce that and, worse, would break the protocol's requirement that
candidates and parent be predicted under an IDENTICAL protocol. ``--binder-msa`` overrides this
for a binder that does have a family (e.g. a nanobody scaffold).

Usage:
  python scripts/build_binder_complex_inputs.py \
    --cohort 'HLA-DRB1*04:01=<run>/HLA-DRB1_04_01/<run_id>/generated.parquet' \
    --cohort 'HLA-DRB1*15:01=<run>/HLA-DRB1_15_01/<run_id>/generated.parquet' \
    --parent-parquet <caseset_if_ready.parquet> --parent-id PD1_b2_BINDER \
    --partner-id PD1_ECTO_Q15116_33_150 --partner-a3m <target.a3m> \
    --out-root <run>/complex_gate --name-prefix cplx_
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The SAME id the refold cache uses, imported rather than reimplemented so complex-gate rows
# always join to the monomer structural/immune tables on `id`.
from inverse_folding.evaluation.esmfold_runner import cache_key as design_id  # noqa: E402

WT_KIND = "WT"
DESIGN_KIND = "DESIGN"


def write_query_only_a3m(path: Path, pred_id: str, sequence: str) -> None:
    """Two identical records, the shape ColabFold emits for a query with no hits."""
    path.write_text(f">{pred_id}\n{sequence}\n>{pred_id}\n{sequence}\n")


def parse_cohort(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"FATAL: --cohort must be 'ALLELE=/path/generated.parquet', got {spec!r}")
    allele, path = spec.split("=", 1)
    return allele.strip(), Path(path.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", action="append", required=True,
                    help="repeatable 'ALLELE=/path/generated.parquet'")
    ap.add_argument("--parent-parquet", required=True,
                    help="IF-ready parquet carrying the exact parent (WT) binder sequence")
    ap.add_argument("--parent-id", required=True)
    ap.add_argument("--partner-id", required=True, help="target chain id (a3m stem + msa_px dir)")
    ap.add_argument("--partner-a3m", required=True, help="the target's deep ColabFold a3m")
    ap.add_argument("--binder-msa", default=None,
                    help="optional a3m to use for EVERY binder chain instead of query-only; only "
                         "valid when the parent's own MSA is deep (a scaffolded binder)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--name-prefix", default="cplx_",
                    help="must match build_protenix_jsons.py --name-prefix")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    msa_cf = out_root / "msa_cf"
    msa_cf.mkdir(parents=True, exist_ok=True)

    partner_src = Path(args.partner_a3m)
    if not partner_src.is_file():
        raise SystemExit(f"FATAL: partner a3m absent: {partner_src}")
    shutil.copyfile(partner_src, msa_cf / f"{args.partner_id}.a3m")

    binder_msa = Path(args.binder_msa) if args.binder_msa else None
    if binder_msa is not None and not binder_msa.is_file():
        raise SystemExit(f"FATAL: --binder-msa absent: {binder_msa}")

    parent = pd.read_parquet(args.parent_parquet)
    parent = parent[parent["protein_id"].astype(str) == args.parent_id]
    if len(parent) != 1:
        raise SystemExit(
            f"FATAL: --parent-parquet must contain exactly one row for {args.parent_id!r}, "
            f"got {len(parent)}"
        )
    parent_seq = str(parent.iloc[0]["sequence"]).upper()

    # The parent WT row is not optional: eval_complex_gate.py --require-interface-wt normalizes
    # every design against a real same-protocol WT prediction and refuses to invent one.
    rows: list[dict] = []
    seen: dict[str, str] = {}

    def add(sequence: str, kind: str, allele: str | None) -> None:
        sequence = sequence.upper()
        pid = design_id(args.parent_id, sequence)
        if pid in seen:
            return  # a sequence recurring across alleles is ONE prediction, scored once
        seen[pid] = sequence
        if binder_msa is not None:
            shutil.copyfile(binder_msa, msa_cf / f"{pid}.a3m")
        else:
            write_query_only_a3m(msa_cf / f"{pid}.a3m", pid, sequence)
        rows.append({
            "id": pid,
            "name": f"{args.name_prefix}{pid}",
            "pred_id": pid,
            "sequence_pred": sequence,
            "sequence": sequence,
            "parent": args.parent_id,
            "kind": kind,
            "allele": allele,
            "n_mutations": sum(a != b for a, b in zip(parent_seq, sequence)),
        })

    add(parent_seq, WT_KIND, None)
    n_parent_rows = len(rows)

    for spec in args.cohort:
        allele, path = parse_cohort(spec)
        if not path.is_file():
            raise SystemExit(f"FATAL: cohort parquet absent: {path}")
        gen = pd.read_parquet(path)
        before = len(rows)
        for r in gen.itertuples():
            if str(r.protein_id) != args.parent_id:
                raise SystemExit(
                    f"FATAL: {path} carries protein_id={r.protein_id!r}, expected {args.parent_id!r}"
                )
            if len(str(r.sequence)) != len(parent_seq):
                raise SystemExit(
                    f"FATAL: design length {len(str(r.sequence))} != parent {len(parent_seq)}; "
                    "the complex gate assumes a fixed-length redesign"
                )
            add(str(r.sequence), DESIGN_KIND, allele)
        print(f"[cohort] {allele}: {len(gen)} designs -> {len(rows) - before} new predictions "
              f"({len(gen) - (len(rows) - before)} duplicates already staged)")

    man = pd.DataFrame(rows)
    (out_root / "pred_manifest.parquet").parent.mkdir(parents=True, exist_ok=True)
    man[["pred_id", "sequence_pred"]].to_parquet(out_root / "pred_manifest.parquet", index=False)
    man[["id", "name", "parent", "kind", "allele", "sequence", "n_mutations"]].to_parquet(
        out_root / "gate_manifest.parquet", index=False
    )
    n_designs = len(man) - n_parent_rows
    print(f"[staged] {len(man)} predictions ({n_parent_rows} WT + {n_designs} designs)")
    print(f"[staged] binder MSA = "
          f"{'shared ' + str(binder_msa) if binder_msa else 'query-only (de novo binder, no homologs)'}")
    print(f"[staged] {msa_cf}/<pred_id>.a3m + {args.partner_id}.a3m")
    print(f"[staged] pred_manifest.parquet -> build_protenix_jsons.py --manifest")
    print(f"[staged] gate_manifest.parquet -> eval_complex_gate.py --manifest")
    if n_designs and man[man.kind == DESIGN_KIND]["n_mutations"].min() == 0:
        print("[warn] at least one design is identical to the parent", file=sys.stderr)


if __name__ == "__main__":
    main()
