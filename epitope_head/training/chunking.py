"""Long-protein chunking: chunk plan generation, trusted intervals, and stitching.

Implements PLAN.md Long-Protein Chunking Policy v1.1:
  C = context_len (1022), S = stride (512), M = margin (32)
  Stitch: per_residue_stitch with deterministic center-crop
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ChunkPlan:
    """Chunk decomposition for a single protein."""
    protein_length: int
    context_len: int
    stride: int
    margin: int
    starts: list[int] = field(default_factory=list)

    @property
    def n_chunks(self) -> int:
        return len(self.starts)

    def chunk_range(self, j: int) -> tuple[int, int]:
        """Return [start, end) of chunk j in residue coordinates."""
        t = self.starts[j]
        return t, min(t + self.context_len, self.protein_length)

    def trusted_interior(self, j: int) -> tuple[int, int]:
        """Return [start, end) of trusted interior for chunk j."""
        t = self.starts[j]
        u = min(t + self.context_len, self.protein_length)

        # M_L = 0 for first chunk, else margin
        ml = 0 if t == 0 else self.margin
        # M_R = 0 for last chunk, else margin
        t_last = max(0, self.protein_length - self.context_len)
        mr = 0 if (t == t_last or u >= self.protein_length) else self.margin

        return t + ml, u - mr


def build_chunk_plan(
    protein_length: int,
    context_len: int = 1022,
    stride: int = 512,
    margin: int = 32,
) -> ChunkPlan:
    """Generate chunk starts for a protein of given length.

    For L <= context_len, returns single chunk at 0 (no chunking needed).
    """
    L = protein_length
    C = context_len

    if L <= C:
        return ChunkPlan(
            protein_length=L, context_len=C,
            stride=stride, margin=margin, starts=[0],
        )

    t_last = max(0, L - C)
    starts = list(range(0, t_last + 1, stride))
    if starts[-1] != t_last:
        starts.append(t_last)

    return ChunkPlan(
        protein_length=L, context_len=C,
        stride=stride, margin=margin, starts=starts,
    )


def assign_residue_to_chunk(plan: ChunkPlan) -> np.ndarray:
    """Assign each residue to its owning chunk index via center-crop.

    Returns array of shape (protein_length,) with chunk index per residue.
    Residues not in any trusted interior get nearest-center assignment
    from all chunks (should not happen with correct margin/stride).
    """
    L = plan.protein_length
    assignment = np.full(L, -1, dtype=np.int32)

    # For each residue, collect candidate chunks where it's in trusted interior
    # Choose chunk with minimal |i - center_j|, tie-break smallest j
    chunk_centers = []
    trusted_intervals = []
    for j in range(plan.n_chunks):
        t_start, t_end = plan.trusted_interior(j)
        c_start, c_end = plan.chunk_range(j)
        center = (c_start + c_end) / 2.0
        chunk_centers.append(center)
        trusted_intervals.append((t_start, t_end))

    for i in range(L):
        best_j = -1
        best_dist = float("inf")
        for j in range(plan.n_chunks):
            t_start, t_end = trusted_intervals[j]
            if t_start <= i < t_end:
                dist = abs(i - chunk_centers[j])
                if dist < best_dist or (dist == best_dist and (best_j == -1 or j < best_j)):
                    best_dist = dist
                    best_j = j
        if best_j == -1:
            # Fallback: assign to nearest chunk center (should not happen)
            for j in range(plan.n_chunks):
                dist = abs(i - chunk_centers[j])
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
        assignment[i] = best_j

    return assignment


def assign_span_to_chunk(plan: ChunkPlan, start: int, end: int) -> int:
    """Determine which chunk owns span [start, end).

    Rule: build an expanded interval E that includes required flanks
    (left = start-1, right = end). Among chunks whose trusted interior
    fully contains E, pick the one whose center is nearest to the span
    midpoint.
    Tie-break: smallest chunk index.

    Returns chunk index, or -1 if span fits in no chunk.
    """
    # Expand to include flanks used by the span feature builder:
    # - left flank uses index start-1 (or PAD when start==0)
    # - right flank uses index end (or PAD when end==L)
    L = plan.protein_length
    e_start = max(0, start - 1)
    e_end = min(L, end + 1)  # half-open, includes index `end` when end < L

    midpoint = (start + end) / 2.0
    best_j = -1
    best_dist = float("inf")

    for j in range(plan.n_chunks):
        t_start, t_end = plan.trusted_interior(j)
        if e_start >= t_start and e_end <= t_end:
            c_start, c_end = plan.chunk_range(j)
            center = (c_start + c_end) / 2.0
            dist = abs(midpoint - center)
            if dist < best_dist or (dist == best_dist and (best_j == -1 or j < best_j)):
                best_dist = dist
                best_j = j

    return best_j


def compute_residue_reliability(plan: ChunkPlan) -> np.ndarray:
    """Compute per-residue reliability flags.

    Returns array of shape (protein_length,) with float reliability in [0, 1].
    - Interior of trusted region: 1.0
    - True N/C termini are NOT penalized (natural boundary, not chunk seam)
    - Only chunk seam margins get reduced reliability
    """
    L = plan.protein_length
    reliability = np.zeros(L, dtype=np.float32)

    # Mark all trusted interiors as reliable
    for j in range(plan.n_chunks):
        t_start, t_end = plan.trusted_interior(j)
        reliability[t_start:t_end] = 1.0

    # For single-chunk proteins, everything is reliable
    if plan.n_chunks == 1:
        reliability[:] = 1.0

    return reliability
