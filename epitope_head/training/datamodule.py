"""DataModule: split-scoped protein loading with chunk-aware token-budget batching.

Implements PLAN.md Task E1:
  - Load per-split protein_id manifests + ProteinSample
  - Build chunk plans for long proteins
  - Token-budget dynamic batching
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from epitope_head.training.chunking import assign_span_to_chunk, build_chunk_plan, ChunkPlan

logger = logging.getLogger(__name__)


@dataclass
class ProteinEntry:
    """A single protein ready for training/validation."""
    protein_id: str
    protein_seq: str
    allele: str
    positives: list[dict]  # list of {start_0b, end_0b, pep_len, support_n}
    sequence_length: int
    chunk_plan: ChunkPlan | None = None
    disrupted_spans: list[dict] | None = None  # Stage J: spans disrupted by runtime mutation


@dataclass
class ChunkSample:
    """A single training sample: one chunk of one protein."""
    protein_id: str
    protein_seq: str
    allele: str
    positives: list[dict]
    sequence_length: int
    chunk_idx: int
    chunk_start: int
    chunk_end: int  # exclusive, in residue coords
    disrupted_spans: list[dict] | None = None  # Stage J: carried from ProteinEntry


def load_split_proteins(
    samples_path: Path,
    split_ids_path: Path,
) -> list[ProteinEntry]:
    """Load ProteinSample parquet filtered to split IDs."""
    split_ids = set()
    with open(split_ids_path) as f:
        for line in f:
            pid = line.strip()
            if pid:
                split_ids.add(pid)

    df = pd.read_parquet(samples_path)
    df = df[df["protein_id"].isin(split_ids)]

    # Fail-fast if IDs are missing
    found_ids = set(df["protein_id"])
    missing = split_ids - found_ids
    if missing:
        raise ValueError(f"Split IDs not found in samples: {len(missing)} missing")

    entries = []
    for _, row in df.iterrows():
        positives = json.loads(row["positives_json"])
        entries.append(ProteinEntry(
            protein_id=row["protein_id"],
            protein_seq=row["protein_seq"],
            allele=row["allele"],
            positives=positives,
            sequence_length=row["sequence_length"],
        ))

    return entries


def load_augmented_proteins(
    aug_parquet_path: Path | str,
) -> list[ProteinEntry]:
    """Load augmented ProteinSample parquet as ProteinEntry list.

    Augmented entries have AUG:: prefixed protein_ids and are train-only.
    No split ID filtering — the artifact is already train-only by construction.
    """
    path = Path(aug_parquet_path)
    if not path.exists():
        return []

    df = pd.read_parquet(path)
    entries = []
    for _, row in df.iterrows():
        positives = json.loads(row["positives_json"])
        entries.append(ProteinEntry(
            protein_id=row["protein_id"],
            protein_seq=row["protein_seq"],
            allele=row["allele"],
            positives=positives,
            sequence_length=row["sequence_length"],
        ))

    return entries


def build_chunk_samples(
    entries: list[ProteinEntry],
    context_len: int = 1022,
    stride: int = 512,
    margin: int = 32,
) -> list[ChunkSample]:
    """Convert protein entries to chunk-level samples.

    Short proteins (L <= context_len) produce one sample.
    Long proteins produce one sample per chunk.
    """
    samples = []
    for entry in entries:
        plan = build_chunk_plan(
            entry.sequence_length,
            context_len=context_len,
            stride=stride,
            margin=margin,
        )
        entry.chunk_plan = plan

        # Pre-assign each positive to exactly one chunk:
        # expanded interval (includes flanks) ⊆ trusted_interior(j) + nearest-center tie-break
        chunk_positives: dict[int, list[dict]] = {j: [] for j in range(plan.n_chunks)}
        orphaned = []
        for p in entry.positives:
            owner = assign_span_to_chunk(plan, p["start_0b"], p["end_0b"])
            if owner >= 0:
                chunk_positives[owner].append(p)
            else:
                orphaned.append(p)
        if orphaned:
            raise ValueError(
                f"Protein {entry.protein_id}: {len(orphaned)} positives fit in no chunk. "
                f"First orphan: span [{orphaned[0]['start_0b']}, {orphaned[0]['end_0b']}), "
                f"L={entry.sequence_length}, n_chunks={plan.n_chunks}. "
                f"Check context_len vs max span length."
            )

        # Assign disrupted spans to chunks (same logic as positives)
        chunk_disrupted: dict[int, list[dict]] = {j: [] for j in range(plan.n_chunks)}
        if entry.disrupted_spans:
            for ds in entry.disrupted_spans:
                owner = assign_span_to_chunk(plan, ds["start_0b"], ds["end_0b"])
                if owner >= 0:
                    chunk_disrupted[owner].append(ds)

        for j in range(plan.n_chunks):
            c_start, c_end = plan.chunk_range(j)
            ds_list = chunk_disrupted[j] if chunk_disrupted[j] else None
            samples.append(ChunkSample(
                protein_id=entry.protein_id,
                protein_seq=entry.protein_seq,
                allele=entry.allele,
                positives=chunk_positives[j],
                sequence_length=entry.sequence_length,
                chunk_idx=j,
                chunk_start=c_start,
                chunk_end=c_end,
                disrupted_spans=ds_list,
            ))

    return samples


class TokenBudgetSampler(Sampler):
    """Dynamic batch sampler that respects a max token budget.

    Groups samples into batches such that the total residues
    (sum of chunk lengths) does not exceed max_tokens.
    """

    def __init__(
        self,
        samples: list[ChunkSample],
        max_tokens: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.samples = samples
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        indices = list(range(len(self.samples)))

        if self.shuffle:
            rng = np.random.RandomState(self.seed + self._epoch)
            rng.shuffle(indices)

        batch = []
        batch_tokens = 0
        for idx in indices:
            sample_len = self.samples[idx].chunk_end - self.samples[idx].chunk_start
            if batch and batch_tokens + sample_len > self.max_tokens:
                yield batch
                batch = []
                batch_tokens = 0
            batch.append(idx)
            batch_tokens += sample_len

        if batch:
            yield batch

    def __len__(self):
        # Approximate: recalculate each time
        count = 0
        batch_tokens = 0
        for s in self.samples:
            slen = s.chunk_end - s.chunk_start
            if batch_tokens + slen > self.max_tokens and batch_tokens > 0:
                count += 1
                batch_tokens = 0
            batch_tokens += slen
        if batch_tokens > 0:
            count += 1
        return count


class EpitopeDataset(Dataset):
    """Wraps chunk samples for DataLoader access."""

    def __init__(self, samples: list[ChunkSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> ChunkSample:
        return self.samples[idx]


def make_collate_fn(
    tokenize_fn: Callable[[list[str]], dict[str, torch.Tensor]] | None = None,
):
    """Create a collate function with optional tokenizer.

    Args:
        tokenize_fn: If provided, called with list[str] chunk sequences.
            Must return dict with at least 'token_ids' (LongTensor [B, T])
            and 'attention_mask' (BoolTensor [B, T]).
            When None, batch contains raw chunk_seqs strings (for testing).
    """
    def collate_chunk_batch(batch: list[ChunkSample]) -> dict:
        """Collate chunk samples into a batch dict.

        Always returns:
          - protein_ids: list[str]
          - protein_seqs: list[str] (full protein sequences)
          - chunk_seqs: list[str] (chunk subsequences)
          - alleles: list[str]
          - positives: list[list[dict]]
          - chunk_starts: LongTensor [B]
          - chunk_ends: LongTensor [B]
          - chunk_indices: LongTensor [B]
          - sequence_lengths: LongTensor [B] (full protein lengths)

        When tokenize_fn is provided, additionally:
          - token_ids: LongTensor [B, T] (padded, with BOS/EOS)
          - attention_mask: BoolTensor [B, T]
        """
        chunk_seqs = [s.protein_seq[s.chunk_start:s.chunk_end] for s in batch]

        result = {
            "protein_ids": [s.protein_id for s in batch],
            "protein_seqs": [s.protein_seq for s in batch],
            "chunk_seqs": chunk_seqs,
            "alleles": [s.allele for s in batch],
            "positives": [s.positives for s in batch],
            "disrupted_spans": [s.disrupted_spans for s in batch],
            "chunk_starts": torch.tensor([s.chunk_start for s in batch], dtype=torch.long),
            "chunk_ends": torch.tensor([s.chunk_end for s in batch], dtype=torch.long),
            "chunk_indices": torch.tensor([s.chunk_idx for s in batch], dtype=torch.long),
            "sequence_lengths": torch.tensor([s.sequence_length for s in batch], dtype=torch.long),
        }

        if tokenize_fn is not None:
            tok_out = tokenize_fn(chunk_seqs)
            result["token_ids"] = tok_out["token_ids"]
            result["attention_mask"] = tok_out["attention_mask"]

        return result

    return collate_chunk_batch


def build_dataloader(
    samples: list[ChunkSample],
    max_tokens: int,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    tokenize_fn: Callable[[list[str]], dict[str, torch.Tensor]] | None = None,
) -> DataLoader:
    """Build a DataLoader with token-budget batching.

    Args:
        tokenize_fn: Optional tokenizer for producing encoder-ready tensors.
            See make_collate_fn for contract.
    """
    dataset = EpitopeDataset(samples)
    sampler = TokenBudgetSampler(
        samples, max_tokens=max_tokens,
        shuffle=shuffle, seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=make_collate_fn(tokenize_fn),
        num_workers=num_workers,
    )
