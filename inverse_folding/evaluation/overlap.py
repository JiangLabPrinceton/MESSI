"""MMseqs2-based sequence identity overlap detection.

Replaces PDB-code-level overlap checking with proper sequence identity
clustering so that structurally similar but differently-named proteins
are caught.

Frozen parameters (PLAN_DATA_SEL §L1):
  --min-seq-id 0.3  --cov-mode 0  -c 0.8
Exclusion rule: seq identity > 30% (strictly greater).
"""

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional


class MMseqs2NotFoundError(RuntimeError):
    """Raised when the mmseqs binary is not found or not executable."""


@dataclass
class OverlapResult:
    """Per-query overlap decision from MMseqs2 search."""

    query_id: str
    best_target: Optional[str]
    best_identity: float
    exclude: bool

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "best_target": self.best_target,
            "best_identity": self.best_identity,
            "exclude": self.exclude,
        }


def _materialize_cath_train_fasta(
    cath_train_source: str,
    tmp_dir: str,
) -> str:
    """Return a FASTA path for MMseqs2 from either FASTA or CATH chain_set.jsonl."""
    if cath_train_source.endswith((".fasta", ".fa", ".faa")):
        return cath_train_source

    if os.path.basename(cath_train_source) != "chain_set.jsonl":
        return cath_train_source

    splits_path = os.path.join(
        os.path.dirname(cath_train_source), "chain_set_splits.json"
    )
    if not os.path.isfile(splits_path):
        raise FileNotFoundError(
            "CATH JSONL source requires sibling chain_set_splits.json: "
            f"{splits_path}"
        )

    with open(splits_path) as f:
        splits = json.load(f)
    train_names = set(splits.get("train", []))

    fasta_path = os.path.join(tmp_dir, "cath_train_from_chain_set.fasta")
    with open(cath_train_source) as src, open(fasta_path, "w") as dst:
        for line in src:
            entry = json.loads(line)
            name = entry.get("name")
            seq = entry.get("seq")
            if name not in train_names or not seq:
                continue
            dst.write(f">{name}\n{seq}\n")

    return fasta_path


def parse_mmseqs_results(
    raw_output: str,
    threshold: float = 0.3,
) -> Dict[str, OverlapResult]:
    """Parse MMseqs2 easy-search tab-separated output into overlap decisions.

    Output format (BLAST-tab): query, target, fident, alnlen, mismatch,
    gapopen, qstart, qend, tstart, tend, evalue, bits.

    Args:
        raw_output: raw TSV string from mmseqs easy-search.
        threshold: sequence identity threshold. Candidates with any
            match **strictly above** this threshold are excluded.

    Returns:
        Dict mapping query_id → OverlapResult. Queries with no hits
        are not included (caller treats absence as pass).
    """
    best: Dict[str, tuple] = {}  # query_id → (identity, target_id)

    for line in raw_output.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        query_id = parts[0]
        target_id = parts[1]
        identity = float(parts[2])

        if query_id not in best or identity > best[query_id][0]:
            best[query_id] = (identity, target_id)

    results: Dict[str, OverlapResult] = {}
    for query_id, (identity, target_id) in best.items():
        results[query_id] = OverlapResult(
            query_id=query_id,
            best_target=target_id,
            best_identity=identity,
            exclude=identity > threshold,
        )
    return results


def _run_mmseqs_easy_search(
    candidate_fasta: str,
    cath_train_fasta: str,
    output_file: str,
    tmp_dir: str,
    mmseqs_bin: str = "mmseqs",
    min_seq_id: float = 0.3,
    cov_mode: int = 0,
    coverage: float = 0.8,
) -> str:
    """Run mmseqs easy-search and return raw output."""
    cmd = [
        mmseqs_bin, "easy-search",
        candidate_fasta,
        cath_train_fasta,
        output_file,
        tmp_dir,
        "--min-seq-id", str(min_seq_id),
        "--cov-mode", str(cov_mode),
        "-c", str(coverage),
        "--format-output", "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mmseqs easy-search failed (rc={result.returncode}): {result.stderr}"
        )
    with open(output_file) as f:
        return f.read()


def run_mmseqs_overlap(
    candidate_fasta: str,
    cath_train_fasta: str,
    mmseqs_bin: str = "mmseqs",
    threshold: float = 0.3,
    min_seq_id: float = 0.3,
    cov_mode: int = 0,
    coverage: float = 0.8,
) -> Dict[str, OverlapResult]:
    """Run MMseqs2 overlap detection end-to-end.

    Args:
        candidate_fasta: path to candidate protein sequences (FASTA).
        cath_train_fasta: path to CATH training sequences source. Accepts
            FASTA directly, or `chain_set.jsonl` with sibling
            `chain_set_splits.json` (train split only).
        mmseqs_bin: path to the mmseqs binary.
        threshold: identity threshold for exclusion (strictly greater).
        min_seq_id: mmseqs --min-seq-id parameter.
        cov_mode: mmseqs --cov-mode parameter.
        coverage: mmseqs -c parameter.

    Returns:
        Dict mapping query_id → OverlapResult.

    Raises:
        MMseqs2NotFoundError: if the mmseqs binary is not found.
    """
    # Fail fast if binary not found
    resolved = shutil.which(mmseqs_bin)
    if resolved is None and not os.path.isfile(mmseqs_bin):
        raise MMseqs2NotFoundError(
            f"MMseqs2 binary not found: {mmseqs_bin!r}. "
            "Install MMseqs2 or provide the correct path."
        )

    with tempfile.TemporaryDirectory(prefix="mmseqs_overlap_") as tmp_dir:
        output_file = os.path.join(tmp_dir, "results.tsv")
        resolved_cath_train_fasta = _materialize_cath_train_fasta(
            cath_train_source=cath_train_fasta,
            tmp_dir=tmp_dir,
        )
        raw = _run_mmseqs_easy_search(
            candidate_fasta=candidate_fasta,
            cath_train_fasta=resolved_cath_train_fasta,
            output_file=output_file,
            tmp_dir=os.path.join(tmp_dir, "tmp"),
            mmseqs_bin=mmseqs_bin,
            min_seq_id=min_seq_id,
            cov_mode=cov_mode,
            coverage=coverage,
        )
    return parse_mmseqs_results(raw, threshold=threshold)
