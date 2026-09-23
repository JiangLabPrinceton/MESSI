#!/usr/bin/env python
r"""Build the frozen natural calibration panel the Dual objective is normalized on.

Scientific authority: ``doc/Dual_Allele_Steering.md`` §2.2; panel decision in
``doc/DUAL_ALLELE_DUALF0_AUDIT.md`` Appendix C. Consumed by
``scripts/calibrate_v2_dual_objective.py``, which measures ``b_a``/``s_a`` on what this produces and
never builds a panel of its own.

The panel must be **allele-neutral**: nothing about its membership may depend on either Head's
opinion, or the coordinates would be fitted to the very landscape they are supposed to normalize.
So the pool is the deduplicated Tier-2 PDB candidate set -- single-chain, 100-500 aa, < 2.5 A,
decided upstream of every immune screen -- and every filter below is either a property of the
sequence or a property of what some model has already SEEN.

Six filters, in order, each reported:

1. deduplicate by exact sequence;
2. keep 100-500 aa (the deployment domain, and long enough to carry a window), and canonical
   AA20 only;
3. drop the union of BOTH Heads' training homology (``--head-overlap exclude``, the default; the
   overlap is measured either way and ``include`` keeps it, which is the AUDIT J.7 D2 sensitivity
   panel and not a second coordinate authority). Not the intersection: the hazard is ASYMMETRIC
   overlap, because a Head that memorized part of the panel has a shifted median and spread, and
   the normalized location difference ``b_A/s_A - b_B/s_B`` -- the equal-risk line -- is biased by
   exactly that shift;
4. drop the deployment proteins, so no protein's own natural sequence enters its own coordinate;
5. drop the evaluation proteins, so calibrating does not spend the test set; and
6. cluster by homology and keep one representative per cluster, because raw PDB determination bias
   is severe and a single over-solved family would otherwise set the exchange rate between two
   alleles for every design that follows.

**What counts as "the Head has seen it".** Train by gradient, and validation by model selection --
``best.pt`` is chosen on a validation metric, so a validation protein influenced which weights
exist. The held-out test split is not included. ``--seen-splits`` makes that choice explicit rather
than implicit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:                       # pragma: no cover - entry-point plumbing
    sys.path.insert(0, str(REPO_ROOT))

MIN_LENGTH, MAX_LENGTH = 100, 500

#: The Head scores a COMPLETE canonical design. A panel entry carrying ``X`` (unknown residue),
#: ``U`` (selenocysteine), ``O`` (pyrrolysine) or ``Z``/``B`` (ambiguous) is dropped whole rather
#: than repaired: substituting a canonical residue for an unknown one invents the very measurement
#: the coordinate is built from, and the oracle refuses the request anyway.
CANONICAL_AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: The project's homology convention, matching ``freeze_t0_dev_cohort.py`` §7.0: 30% identity over
#: 80% coverage. ``cov_mode=2`` covers the QUERY -- a short panel entry contained inside a long
#: training protein is a hit under cov-mode 2 and is missed under cov-mode 0, and that is exactly
#: the direction memorization would come from.
IDENTITY, COVERAGE, COV_MODE = 0.3, 0.8, 2


class PanelBuildError(RuntimeError):
    """A panel input or filter contract was violated."""


def read_fasta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    header, chunks = None, []

    def flush() -> None:
        if header is not None:
            out[header.split()[0]] = "".join(chunks).upper()

    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            flush()
            header, chunks = line[1:].strip(), []
        elif line.strip():
            chunks.append(line.strip())
    flush()
    if not out:
        raise PanelBuildError(f"{path} contains no sequences")
    return out


def head_seen_sequences(manifest_dir: Path, splits: tuple[str, ...]) -> dict[str, str]:
    """Every protein sequence one Head's fitting procedure touched, by id."""
    import pyarrow.parquet as pq

    frame = pq.read_table(manifest_dir / "protein_samples_strict.parquet").to_pandas()
    seen_ids: set[str] = set()
    for split in splits:
        path = manifest_dir / "splits" / f"{split}_ids.txt"
        if not path.exists():
            raise PanelBuildError(f"{path} does not exist; cannot say what this Head has seen")
        seen_ids.update(line.strip() for line in path.read_text().splitlines() if line.strip())
    rows = frame[frame["protein_id"].astype(str).isin(seen_ids)]
    missing = seen_ids - set(rows["protein_id"].astype(str))
    if missing:
        raise PanelBuildError(
            f"{len(missing)} split id(s) have no row in protein_samples_strict, e.g. "
            f"{sorted(missing)[:5]}; the manifest and the splits describe different runs")
    return {str(r.protein_id): str(r.protein_seq) for r in rows.itertuples()}


def write_fasta(path: Path, sequences: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f">{pid}\n{seq}\n" for pid, seq in sorted(sequences.items())))


def homology_hits(pool: dict[str, str], reference: dict[str, str], *, mmseqs: str,
                  tmp_dir: Path, label: str) -> set[str]:
    """Pool ids homologous to anything in ``reference``, by the project's 30%/80% convention."""
    from inverse_folding.evaluation.overlap import run_mmseqs_overlap

    pool_fasta, ref_fasta = tmp_dir / f"pool_{label}.fasta", tmp_dir / f"ref_{label}.fasta"
    write_fasta(pool_fasta, pool)
    write_fasta(ref_fasta, reference)
    # The parameter is NAMED cath_train_fasta because its first caller compared against the
    # inverse-folding model's CATH training set. It is just "the reference set", and the question
    # here is a different one: has the EPITOPE HEAD seen this sequence.
    results = run_mmseqs_overlap(
        candidate_fasta=str(pool_fasta), cath_train_fasta=str(ref_fasta), mmseqs_bin=mmseqs,
        threshold=IDENTITY, min_seq_id=IDENTITY, cov_mode=COV_MODE, coverage=COVERAGE)
    return {qid for qid, r in results.items() if r.exclude}


def build(args: argparse.Namespace) -> dict[str, Any]:
    from inverse_folding.reference_flow.fusion.state import sequence_md5

    report: dict[str, Any] = {}
    raw = read_fasta(args.candidate_fasta)
    report["raw_entries"] = len(raw)

    # 1. exact-sequence deduplication, lowest id wins so the choice is a function of the SET
    by_md5: dict[str, str] = {}
    for pid in sorted(raw):
        by_md5.setdefault(sequence_md5(raw[pid]), pid)
    pool = {pid: raw[pid] for pid in by_md5.values()}
    report["after_dedup"] = len(pool)

    # 2. the deployment length domain
    pool = {p: s for p, s in pool.items() if MIN_LENGTH <= len(s) <= MAX_LENGTH}
    report["after_length"] = len(pool)

    # 2b. canonical AA20 only
    non_canonical = {p: sorted(set(s) - CANONICAL_AA20) for p, s in pool.items()
                     if set(s) - CANONICAL_AA20}
    pool = {p: s for p, s in pool.items() if p not in non_canonical}
    report["dropped_non_canonical"] = len(non_canonical)
    report["non_canonical_characters"] = sorted(
        {ch for chars in non_canonical.values() for ch in chars})
    report["after_canonical"] = len(pool)

    # 3. the UNION of both Heads' training homology
    tmp = Path(args.work_dir); tmp.mkdir(parents=True, exist_ok=True)
    splits = tuple(s.strip() for s in args.seen_splits.split(",") if s.strip())
    report["seen_splits"] = list(splits)
    seen_by_role, hits_by_role = {}, {}
    for role, manifest in (("a", args.head_a_manifest_dir), ("b", args.head_b_manifest_dir)):
        seen = head_seen_sequences(Path(manifest), splits)
        seen_by_role[role] = len(seen)
        hits_by_role[role] = homology_hits(pool, seen, mmseqs=args.mmseqs, tmp_dir=tmp, label=role)
        print(f"[panel] head {role.upper()}: {len(seen)} seen protein(s) -> "
              f"{len(hits_by_role[role])} pool homolog(s) "
              f"({100.0 * len(hits_by_role[role]) / max(1, len(pool)):.2f}% of the pool)",
              flush=True)
    union = hits_by_role["a"] | hits_by_role["b"]
    report["head_seen_proteins"] = seen_by_role
    report["head_homology_hits"] = {r: len(h) for r, h in hits_by_role.items()}
    report["head_homology_fraction"] = {
        r: len(h) / max(1, len(pool)) for r, h in hits_by_role.items()}
    report["head_homology_union"] = len(union)
    report["head_homology_asymmetry"] = abs(
        report["head_homology_fraction"]["a"] - report["head_homology_fraction"]["b"])
    # The overlap is MEASURED in both policies and REMOVED in only one. A sensitivity panel that
    # skipped the measurement could not be compared against the primary panel on the quantity the
    # sensitivity is about, and `head_homology_fraction` is computed against the same pre-removal
    # pool in both, so the two reports' f_A/f_B are the same number measured twice, not two
    # different numbers.
    report["head_overlap_policy"] = str(args.head_overlap)
    if args.head_overlap == "exclude":
        pool = {p: s for p, s in pool.items() if p not in union}
    report["after_head_homology"] = len(pool)

    # 4. the deployment cohort -- a protein must not normalize against itself
    deployment = _ids(args.deployment_protein_ids)
    pool = {p: s for p, s in pool.items() if p not in deployment}
    report["deployment_proteins"] = len(deployment)
    report["after_deployment"] = len(pool)

    # 5. the evaluation proteins -- calibrating must not spend the test set
    evaluation = set()
    for path in args.evaluation_protein_ids or []:
        evaluation |= _ids(path)
    pool = {p: s for p, s in pool.items() if p not in evaluation}
    report["evaluation_proteins"] = len(evaluation)
    report["after_evaluation"] = len(pool)

    if not pool:
        raise PanelBuildError("every candidate was filtered out; the panel would be empty")

    # 6. one representative per homology cluster
    clusters = mmseqs_clusters(args.mmseqs, pool)
    representatives: dict[str, str] = {}
    for pid in sorted(pool):
        representatives.setdefault(clusters[pid], pid)
    panel = {pid: pool[pid] for pid in sorted(representatives.values())}
    report["n_clusters"] = len(representatives)
    report["panel_size"] = len(panel)
    return {"report": report, "panel": panel}


def _ids(path: Any) -> set[str]:
    if path is None:
        return set()
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_dual_calibration_panel",
        description="build the frozen allele-neutral natural panel for the Dual calibration")
    parser.add_argument("--candidate-fasta", required=True, type=Path)
    parser.add_argument("--head-a-manifest-dir", required=True, type=Path)
    parser.add_argument("--head-b-manifest-dir", required=True, type=Path)
    parser.add_argument("--deployment-protein-ids", required=True, type=Path)
    parser.add_argument("--evaluation-protein-ids", action="append", type=Path, default=None,
                        help="repeatable; the per-allele test-set protein id lists")
    # AUDIT J.7 D2. `include` builds the overlap-inclusion SENSITIVITY panel: identical in every
    # other filter, Head identity, objective law and scoring protocol, differing only in whether
    # step 3 removes what it measured. It is a sensitivity label on the primary panel, never a
    # competing coordinate authority -- the primary panel was selected by an outcome-independent
    # leakage rule and stays authoritative whatever this measures.
    parser.add_argument("--head-overlap", choices=("exclude", "include"), default="exclude",
                        help="exclude (default, the primary panel) or include the union of both "
                             "Heads' training homology; the overlap is measured either way")
    parser.add_argument("--seen-splits", default="train,val",
                        help="which splits count as SEEN by a Head. train fits the weights and val "
                             "selects them, so both are seen; the held-out test split is not")
    parser.add_argument("--out-fasta", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/dual_panel"))
    parser.add_argument("--mmseqs", default="mmseqs")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    built = build(args)
    write_fasta(args.out_fasta, built["panel"])
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(built["report"], indent=2, sort_keys=True) + "\n")
    for key, value in built["report"].items():
        print(f"[panel] {key} = {value}", flush=True)
    print(f"[panel] -> {args.out_fasta}", flush=True)
    return 0


MMSEQS_MIN_SEQ_ID = 0.30


MMSEQS_COVERAGE = 0.80


def mmseqs_clusters(mmseqs: str, sequences: dict[str, str]) -> dict[str, str]:
    """Cluster the WHOLE pool with ``mmseqs easy-cluster``; return protein_id -> cluster rep id.

    Clustering the whole pool (not just the eligible subset) is what lets a cluster touching a prior
    cohort be excluded wholesale. A non-zero exit, a missing TSV, or an unclustered protein is FATAL:
    §7.0 forbids falling back to exact-sequence dedup for the load-bearing holdout, because that
    would silently admit 30-99%-identity homologs as independent statistical units.
    """
    import tempfile
    import subprocess

    with tempfile.TemporaryDirectory(prefix="mmseqs_cohort_") as tmp:
        tmp = Path(tmp)
        fasta = tmp / "pool.fasta"
        fasta.write_text("".join(f">{pid}\n{seq}\n" for pid, seq in sorted(sequences.items())))
        prefix = tmp / "clu"
        cmd = [str(mmseqs), "easy-cluster", str(fasta), str(prefix), str(tmp / "work"),
               "--min-seq-id", str(MMSEQS_MIN_SEQ_ID), "-c", str(MMSEQS_COVERAGE),
               "--cov-mode", "0", "-v", "1"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        tsv = Path(f"{prefix}_cluster.tsv")
        if proc.returncode != 0 or not tsv.is_file():
            raise SystemExit(
                f"mmseqs clustering FAILED (rc={proc.returncode}); refusing to fall back to "
                f"exact-sequence dedup for a load-bearing cohort (§7.0).\n"
                f"cmd: {' '.join(cmd)}\nstderr tail:\n{proc.stderr[-2000:]}"
            )
        out: dict[str, str] = {}
        for line in tsv.read_text().splitlines():
            if not line.strip():
                continue
            rep, member = line.split("\t")[:2]
            out[member] = rep
        missing = sorted(set(sequences) - set(out))
        if missing:
            raise SystemExit(f"mmseqs left {len(missing)} protein(s) unclustered, e.g. {missing[:5]}")
        return out


if __name__ == "__main__":                                # pragma: no cover - CLI entry
    raise SystemExit(main())
