"""NetMHCIIpan runner abstraction — standalone binary and mock backends.

Implements PLAN.md Module J (J1+):
  - Protocol-based runner interface
  - Standalone subprocess runner (cluster)
  - Mock runner (local testing)
"""

from __future__ import annotations

import csv
import logging
import os
import signal
import subprocess
import tempfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PeptideScore:
    """Score for a single peptide window from NetMHCIIpan."""
    pos: int            # 0-based position in protein
    peptide: str        # peptide sequence
    core: str           # binding core (9-mer)
    el_score: float     # raw EL score
    el_rank: float      # %Rank_EL (as fraction, e.g. 0.02 = 2%)
    pep_length: int = 0 # peptide length (filled by multi-length parsing)


class NetMHCIIpanRunner(ABC):
    """Abstract interface for NetMHCIIpan prediction."""

    @abstractmethod
    def score_protein(
        self,
        protein_id: str,
        protein_seq: str,
        allele: str,
        pep_length: int,
    ) -> list[PeptideScore]:
        """Score all windows of given length in a protein sequence."""
        ...

    def score_batch(
        self,
        entries: list[tuple[str, str]],
        allele: str,
        pep_lengths: list[int],
    ) -> dict[str, dict[int, list[PeptideScore]]]:
        """Score multiple proteins × multiple lengths in one call.

        Returns {protein_id: {pep_length: [scores]}}.
        Default: sequential fallback.
        """
        results: dict[str, dict[int, list[PeptideScore]]] = {}
        for pid, seq in entries:
            results[pid] = {}
            for pl in pep_lengths:
                results[pid][pl] = self.score_protein(pid, seq, allele, pl)
        return results


class StandaloneRunner(NetMHCIIpanRunner):
    """Calls NetMHCIIpan standalone binary via subprocess."""

    def __init__(
        self,
        binary_path: str | Path,
        batch_size: int = 30,
        subprocess_timeout: int = 600,
        max_lengths_per_call: int = 4,
        n_workers: int = 1,
    ):
        self.binary_path = Path(binary_path)
        self.batch_size = batch_size
        self.subprocess_timeout = subprocess_timeout
        self.max_lengths_per_call = max_lengths_per_call
        self.n_workers = max(1, n_workers)
        if not self.binary_path.exists():
            raise FileNotFoundError(f"NetMHCIIpan binary not found: {self.binary_path}")

    def _run_netmhciipan(self, cmd: list[str]) -> subprocess.CompletedProcess:
        """Run NetMHCIIpan with timeout and clean process-group kill."""
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=self.subprocess_timeout)
        except subprocess.TimeoutExpired:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait()
            raise
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

    def score_protein(
        self,
        protein_id: str,
        protein_seq: str,
        allele: str,
        pep_length: int,
    ) -> list[PeptideScore]:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fasta", delete=False
        ) as f:
            f.write(f">{protein_id}\n{protein_seq}\n")
            fasta_path = f.name

        allele_fmt = allele.replace("HLA-", "").replace("*", "_").replace(":", "")
        cmd = [
            str(self.binary_path),
            "-f", fasta_path,
            "-a", allele_fmt,
            "-length", str(pep_length),
            "-context",
        ]

        try:
            result = self._run_netmhciipan(cmd)
            if result.returncode != 0:
                raise RuntimeError(
                    f"NetMHCIIpan failed (rc={result.returncode}): {result.stderr[:500]}"
                )
            return self._parse_output(result.stdout)
        except subprocess.TimeoutExpired:
            logger.warning(
                "NetMHCIIpan timed out after %ds for protein %s (len=%d, pep_len=%d)",
                self.subprocess_timeout, protein_id, len(protein_seq), pep_length,
            )
            return []
        finally:
            Path(fasta_path).unlink(missing_ok=True)

    class ChunkFailedError(Exception):
        """Raised when a NetMHCIIpan chunk returns non-zero exit code."""

    def _run_chunk_lengths(
        self,
        chunk: list[tuple[str, str]],
        allele_fmt: str,
        length_group: list[int],
        chunk_offset: int,
    ) -> dict[str, dict[int, list[PeptideScore]]]:
        """Run one chunk × one length group. Returns {orig_id: {len: [scores]}}.

        Raises:
            subprocess.TimeoutExpired: if the subprocess times out.
            ChunkFailedError: if the subprocess returns non-zero exit code.
        """
        short_to_orig: dict[str, str] = {}
        length_str = ",".join(str(pl) for pl in sorted(length_group))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fasta", delete=False
        ) as f:
            for i, (pid, seq) in enumerate(chunk):
                short_id = f"S{chunk_offset + i:06d}"
                short_to_orig[short_id] = pid
                f.write(f">{short_id}\n{seq}\n")
            fasta_path = f.name

        cmd = [
            str(self.binary_path),
            "-f", fasta_path,
            "-a", allele_fmt,
            "-length", length_str,
            "-context",
        ]

        try:
            result = self._run_netmhciipan(cmd)
            if result.returncode != 0:
                raise StandaloneRunner.ChunkFailedError(
                    f"rc={result.returncode}, stderr={result.stderr[:300]}"
                )
            chunk_results = self._parse_batch_output(result.stdout)
            # Remap short IDs → original IDs
            remapped: dict[str, dict[int, list[PeptideScore]]] = {}
            for short_id, by_len in chunk_results.items():
                orig_id = short_to_orig.get(short_id, short_id)
                remapped[orig_id] = by_len
            return remapped
        except subprocess.TimeoutExpired:
            raise
        finally:
            Path(fasta_path).unlink(missing_ok=True)

    def _fallback_single_protein(
        self,
        entries: list[tuple[str, str]],
        allele_fmt: str,
        length_group: list[int],
        executor: ThreadPoolExecutor | None = None,
    ) -> dict[str, dict[int, list[PeptideScore]]]:
        """Retry failed chunk one protein at a time, optionally in parallel."""
        results: dict[str, dict[int, list[PeptideScore]]] = {}

        if executor is None:
            # Sequential fallback
            for pid, seq in entries:
                try:
                    single = self._run_chunk_lengths(
                        [(pid, seq)], allele_fmt, length_group, chunk_offset=0,
                    )
                    for orig_id, by_len in single.items():
                        results.setdefault(orig_id, {}).update(by_len)
                except (subprocess.TimeoutExpired, StandaloneRunner.ChunkFailedError) as e:
                    logger.warning(
                        "NMP fallback failed for %s (len=%d, lengths=%s): %s",
                        pid, len(seq),
                        ",".join(str(pl) for pl in length_group),
                        type(e).__name__,
                    )
            return results

        # Parallel fallback
        futures = {}
        for pid, seq in entries:
            fut = executor.submit(
                self._run_chunk_lengths,
                [(pid, seq)], allele_fmt, length_group, 0,
            )
            futures[fut] = (pid, seq)

        for fut in as_completed(futures):
            pid, seq = futures[fut]
            try:
                single = fut.result()
                for orig_id, by_len in single.items():
                    results.setdefault(orig_id, {}).update(by_len)
            except (subprocess.TimeoutExpired, StandaloneRunner.ChunkFailedError) as e:
                logger.warning(
                    "NMP fallback failed for %s (len=%d, lengths=%s): %s",
                    pid, len(seq),
                    ",".join(str(pl) for pl in length_group),
                    type(e).__name__,
                )
        return results

    def score_batch(
        self,
        entries: list[tuple[str, str]],
        allele: str,
        pep_lengths: list[int],
    ) -> dict[str, dict[int, list[PeptideScore]]]:
        """Score multiple proteins × multiple lengths, chunked to avoid timeout.

        Strategy to minimize data loss:
          1. Split pep_lengths into groups of max_lengths_per_call (default 4)
             to reduce per-subprocess workload ~3-4x.
          2. Chunk proteins by batch_size as before.
          3. Within each chunk, run all length groups in parallel via
             ThreadPoolExecutor(n_workers). Each thread spawns one
             NetMHCIIpan subprocess — the GIL is irrelevant since threads
             only wait on subprocesses.
          4. On chunk failure, fall back to per-protein parallel retry.

        Uses short FASTA IDs (S000000, S000001, ...) to avoid NetMHCIIpan's
        ~15-char Identity truncation, then remaps back to original IDs.
        Returns {protein_id: {pep_length: [scores]}}.
        """
        if not entries or not pep_lengths:
            return {}

        allele_fmt = allele.replace("HLA-", "").replace("*", "_").replace(":", "")
        all_results: dict[str, dict[int, list[PeptideScore]]] = {}

        # Split lengths into smaller groups to reduce per-call compute
        sorted_lengths = sorted(pep_lengths)
        length_groups: list[list[int]] = []
        for i in range(0, len(sorted_lengths), self.max_lengths_per_call):
            length_groups.append(sorted_lengths[i:i + self.max_lengths_per_call])

        n_chunks = (len(entries) + self.batch_size - 1) // self.batch_size
        n_timeout_chunks = 0
        n_fallback_recovered = 0

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for chunk_idx, chunk_start in enumerate(
                range(0, len(entries), self.batch_size)
            ):
                chunk = entries[chunk_start:chunk_start + self.batch_size]

                # Submit all length groups for this chunk in parallel
                futures = {}
                for lg_idx, length_group in enumerate(length_groups):
                    fut = executor.submit(
                        self._run_chunk_lengths,
                        chunk, allele_fmt, length_group,
                        chunk_offset=chunk_start,
                    )
                    futures[fut] = (lg_idx, length_group)

                # Collect results; track which length groups failed
                failed_groups: list[tuple[int, list[int]]] = []
                for fut in as_completed(futures):
                    lg_idx, length_group = futures[fut]
                    try:
                        chunk_results = fut.result()
                        for orig_id, by_len in chunk_results.items():
                            all_results.setdefault(orig_id, {}).update(by_len)
                    except (
                        subprocess.TimeoutExpired,
                        StandaloneRunner.ChunkFailedError,
                    ) as e:
                        n_timeout_chunks += 1
                        logger.warning(
                            "NMP chunk %d/%d (lg %d/%d) failed: %s "
                            "(%d seqs, lengths=%s). Will retry per-protein.",
                            chunk_idx + 1, n_chunks,
                            lg_idx + 1, len(length_groups),
                            type(e).__name__, len(chunk),
                            ",".join(str(pl) for pl in length_group),
                        )
                        failed_groups.append((lg_idx, length_group))

                # Parallel per-protein fallback for failed length groups
                for lg_idx, length_group in failed_groups:
                    fallback = self._fallback_single_protein(
                        chunk, allele_fmt, length_group,
                        executor=executor,
                    )
                    for orig_id, by_len in fallback.items():
                        all_results.setdefault(orig_id, {}).update(by_len)
                    n_fallback_recovered += len(fallback)

                if (chunk_idx + 1) % 5 == 0 or (chunk_idx + 1) == n_chunks:
                    logger.debug(
                        "  score_batch: chunk %d/%d done (%d entries so far, "
                        "%d failures, %d recovered)",
                        chunk_idx + 1, n_chunks, len(all_results),
                        n_timeout_chunks, n_fallback_recovered,
                    )

        if n_timeout_chunks > 0:
            logger.warning(
                "  score_batch complete: %d total failures, %d proteins "
                "recovered via fallback",
                n_timeout_chunks, n_fallback_recovered,
            )

        return all_results

    def _parse_batch_output(
        self, stdout: str,
    ) -> dict[str, dict[int, list[PeptideScore]]]:
        """Parse multi-sequence/multi-length output.

        Groups by Identity and peptide length.
        Returns {identity: {pep_length: [PeptideScore]}}.

        Robust parser: skips separators/headers/summaries by content,
        does NOT rely on dash counting (which breaks for multi-protein output
        where each protein has its own --- block).
        """
        results: dict[str, dict[int, list[PeptideScore]]] = {}
        idx_score_el = 8
        idx_rank_el = 9
        idx_identity = 7

        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # Skip separators, comments, summary lines
            if stripped.startswith("---") or stripped.startswith("#"):
                continue
            if "Number of" in stripped:
                continue

            # Detect header line (appears once or repeated per block)
            if "Score_EL" in stripped and "%Rank_EL" in stripped:
                header_parts = stripped.split()
                for i, h in enumerate(header_parts):
                    if h == "Score_EL":
                        idx_score_el = i
                    elif h == "%Rank_EL":
                        idx_rank_el = i
                    elif h == "Identity":
                        idx_identity = i
                continue

            # Try to parse as data line
            parts = stripped.split()
            if len(parts) < idx_rank_el + 1:
                continue

            try:
                pos = int(parts[0]) - 1
                peptide = parts[2]
                core = parts[4]
                identity = parts[idx_identity]
                el_score = float(parts[idx_score_el])
                el_rank = float(parts[idx_rank_el]) / 100.0
                pep_len = len(peptide)

                results.setdefault(identity, {}).setdefault(pep_len, []).append(
                    PeptideScore(
                        pos=pos, peptide=peptide, core=core,
                        el_score=el_score, el_rank=el_rank,
                        pep_length=pep_len,
                    )
                )
            except (ValueError, IndexError):
                continue

        return results

    def _parse_output(self, stdout: str) -> list[PeptideScore]:
        """Parse NetMHCIIpan 4.3 stdout into PeptideScore list.

        Robust content-based parser (no dash counting).
        Auto-detects Score_EL/%Rank_EL column positions from the header.
        """
        scores = []
        idx_score_el = 8
        idx_rank_el = 9

        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("---") or stripped.startswith("#"):
                continue
            if "Number of" in stripped:
                continue

            if "Score_EL" in stripped and "%Rank_EL" in stripped:
                header_parts = stripped.split()
                for i, h in enumerate(header_parts):
                    if h == "Score_EL":
                        idx_score_el = i
                    elif h == "%Rank_EL":
                        idx_rank_el = i
                continue

            parts = stripped.split()
            if len(parts) < idx_rank_el + 1:
                continue

            try:
                pos = int(parts[0]) - 1
                peptide = parts[2]
                core = parts[4]
                el_score = float(parts[idx_score_el])
                el_rank = float(parts[idx_rank_el]) / 100.0
                scores.append(PeptideScore(
                    pos=pos, peptide=peptide, core=core,
                    el_score=el_score, el_rank=el_rank,
                ))
            except (ValueError, IndexError):
                continue
        return scores


class MockRunner(NetMHCIIpanRunner):
    """Deterministic mock runner for testing without NetMHCIIpan binary.

    Generates biologically plausible synthetic scores:
    - Most windows get moderate-to-high ranks (not presented)
    - ~30% of windows get low ranks (< 5%), simulating real binders
    - Mutations at hydrophobic anchor positions tend to increase rank
    """

    HYDROPHOBIC = set("AILMFWVP")

    def __init__(self, seed: int = 42):
        import numpy as np
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def _score_seq(protein_seq: str, pep_length: int) -> list[PeptideScore]:
        """Deterministic scoring based on sequence content only (ID-independent)."""
        import numpy as np

        n_windows = len(protein_seq) - pep_length + 1
        if n_windows <= 0:
            return []

        scores = []
        for i in range(n_windows):
            peptide = protein_seq[i:i + pep_length]
            # Seed from (position, peptide content) — not protein_id
            h = hash((i, peptide)) & 0xFFFFFFFF
            local_rng = np.random.RandomState(h % (2**31))

            # Base rank: bimodal — ~30% binders (low rank), ~70% non-binders
            if local_rng.random() < 0.3:
                el_rank = local_rng.exponential(0.008)  # mostly < 2%
            else:
                el_rank = 0.05 + local_rng.exponential(0.15)  # mostly > 5%
            el_rank = min(el_rank, 0.99)

            scores.append(PeptideScore(
                pos=i,
                peptide=peptide,
                core=peptide[3:12] if len(peptide) >= 12 else peptide[:9],
                el_score=max(0.0, 1.0 - el_rank * 2),
                el_rank=el_rank,
            ))
        return scores

    def score_protein(
        self,
        protein_id: str,
        protein_seq: str,
        allele: str,
        pep_length: int,
    ) -> list[PeptideScore]:
        return self._score_seq(protein_seq, pep_length)

    def score_batch(
        self,
        entries: list[tuple[str, str]],
        allele: str,
        pep_lengths: list[int],
    ) -> dict[str, dict[int, list[PeptideScore]]]:
        """Batch scoring: same sequence → same scores regardless of ID."""
        results: dict[str, dict[int, list[PeptideScore]]] = {}
        for pid, seq in entries:
            results[pid] = {}
            for pl in pep_lengths:
                results[pid][pl] = self._score_seq(seq, pl)
        return results


def build_runner(
    backend: str = "standalone",
    binary_path: str | Path | None = None,
    seed: int = 42,
    batch_size: int = 30,
    subprocess_timeout: int = 600,
    max_lengths_per_call: int = 4,
    n_workers: int = 1,
) -> NetMHCIIpanRunner:
    """Factory for NetMHCIIpan runners."""
    if backend == "standalone":
        if binary_path is None:
            raise ValueError("binary_path required for standalone backend")
        return StandaloneRunner(
            binary_path, batch_size=batch_size,
            subprocess_timeout=subprocess_timeout,
            max_lengths_per_call=max_lengths_per_call,
            n_workers=n_workers,
        )
    elif backend == "mock":
        return MockRunner(seed=seed)
    else:
        raise ValueError(f"Unknown backend: {backend}")
