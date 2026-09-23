"""Stage J: NetMHCIIpan mutation augmentation — config and core logic.

Implements PLAN.md Module J (J0-J5):
  - Augmentation config validation
  - WT verification threshold classification
  - Pilot protein selection
  - Mutation candidate enumeration and deduplication
  - Affected-span recalculation
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ── Frozen constants (PLAN.md J0) ──────────────────────────────────────────

REQUIRED_TOP_KEYS = {
    "profile", "tool", "tool_version", "tool_mode", "allele",
    "wt_rank_threshold", "wt_uncertain_upper",
    "mut_rank_threshold", "delta_rank_threshold",
    "top_n_per_protein",
    "inputs", "outputs",
}

REQUIRED_INPUT_KEYS = {"source_parquet", "train_ids", "val_ids", "test_ids"}

REQUIRED_OUTPUT_KEYS = {
    "pilot", "registry", "registry_summary",
    "aug_train_samples", "aug_train_summary",
}

VALID_TOOLS = {"netmhciipan"}
VALID_MODES = {"EL", "BA"}


# ── Config dataclass ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class AugmentationConfig:
    """Validated, immutable Stage-J augmentation configuration."""
    profile: str
    tool: str
    tool_version: str
    tool_mode: str
    allele: str
    wt_rank_threshold: float
    wt_uncertain_upper: float
    mut_rank_threshold: float
    delta_rank_threshold: float
    top_n_per_protein: int
    inputs: dict
    outputs: dict


# ── Config loader + validator ──────────────────────────────────────────────

def load_augmentation_config(path: Path | str | None = None) -> AugmentationConfig:
    """Load and validate augmentation config. Raises ValueError on violations."""
    if path is None:
        path = Path(__file__).parents[1] / "configs" / "augmentation.yaml"
    path = Path(path)

    with open(path) as f:
        cfg = yaml.safe_load(f)

    raw = cfg.get("augmentation")
    if raw is None:
        raise ValueError("Config missing top-level 'augmentation' key")

    return validate_augmentation_config(raw)


def validate_augmentation_config(raw: dict) -> AugmentationConfig:
    """Validate raw augmentation config dict. Returns AugmentationConfig or raises."""
    # Top-level keys
    missing = REQUIRED_TOP_KEYS - set(raw.keys())
    if missing:
        raise ValueError(f"Augmentation config missing required keys: {sorted(missing)}")

    # Tool validation
    if raw["tool"] not in VALID_TOOLS:
        raise ValueError(f"Unknown tool '{raw['tool']}', expected one of {sorted(VALID_TOOLS)}")
    if raw["tool_mode"] not in VALID_MODES:
        raise ValueError(f"Unknown tool_mode '{raw['tool_mode']}', expected one of {sorted(VALID_MODES)}")

    # Threshold validation
    wt = raw["wt_rank_threshold"]
    wt_upper = raw["wt_uncertain_upper"]
    mut = raw["mut_rank_threshold"]
    delta = raw["delta_rank_threshold"]

    if not (0 < wt < wt_upper < 1):
        raise ValueError(
            f"Threshold order violated: need 0 < wt_rank_threshold ({wt}) "
            f"< wt_uncertain_upper ({wt_upper}) < 1"
        )
    if not (0 < mut < 1):
        raise ValueError(f"mut_rank_threshold must be in (0, 1), got {mut}")
    if not (0 < delta < 1):
        raise ValueError(f"delta_rank_threshold must be in (0, 1), got {delta}")

    # top_n validation
    top_n = raw["top_n_per_protein"]
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError(f"top_n_per_protein must be a positive integer, got {top_n}")

    # Input paths
    inputs = raw["inputs"]
    if not isinstance(inputs, dict):
        raise ValueError("augmentation.inputs must be a dict")
    missing_in = REQUIRED_INPUT_KEYS - set(inputs.keys())
    if missing_in:
        raise ValueError(f"augmentation.inputs missing keys: {sorted(missing_in)}")

    # Output paths
    outputs = raw["outputs"]
    if not isinstance(outputs, dict):
        raise ValueError("augmentation.outputs must be a dict")
    missing_out = REQUIRED_OUTPUT_KEYS - set(outputs.keys())
    if missing_out:
        raise ValueError(f"augmentation.outputs missing keys: {sorted(missing_out)}")

    return AugmentationConfig(
        profile=raw["profile"],
        tool=raw["tool"],
        tool_version=str(raw["tool_version"]),
        tool_mode=raw["tool_mode"],
        allele=raw["allele"],
        wt_rank_threshold=float(wt),
        wt_uncertain_upper=float(wt_upper),
        mut_rank_threshold=float(mut),
        delta_rank_threshold=float(delta),
        top_n_per_protein=int(top_n),
        inputs=dict(inputs),
        outputs=dict(outputs),
    )


# ── WT classification ──────────────────────────────────────────────────────

def classify_wt_rank(
    wt_el_rank: float,
    wt_threshold: float,
    uncertain_upper: float,
) -> str:
    """Classify a WT EL_Rank into threshold buckets.

    Returns one of: 'keep_wt_confirmed', 'skip_uncertain', 'reject_wt_disagree'.
    """
    if wt_el_rank < wt_threshold:
        return "keep_wt_confirmed"
    elif wt_el_rank <= uncertain_upper:
        return "skip_uncertain"
    else:
        return "reject_wt_disagree"


def is_disrupted(
    mut_el_rank: float,
    wt_el_rank: float,
    mut_threshold: float,
    delta_threshold: float,
) -> bool:
    """Check if a mutation qualifies as a disruption."""
    delta = mut_el_rank - wt_el_rank
    return mut_el_rank > mut_threshold and delta > delta_threshold


# ── Pilot protein selection (J1) ──────────────────────────────────────────

def select_pilot_proteins(
    df: pd.DataFrame,
    train_ids: set[str],
    n_proteins: int = 8,
    seed: int = 42,
) -> list[str]:
    """Select a stratified pilot subset from strict-train proteins.

    Stratifies by positive_count and sequence_length to cover diversity.
    Returns protein_id list.
    """
    train_df = df[df["protein_id"].isin(train_ids)].copy()
    if len(train_df) == 0:
        raise ValueError("No train proteins found in parquet")

    # Stratify: bin by positive_count (low/high) and seq_length (short/long)
    med_pos = train_df["positive_count"].median()
    med_len = train_df["sequence_length"].median()
    train_df["_stratum"] = (
        (train_df["positive_count"] > med_pos).astype(int) * 2
        + (train_df["sequence_length"] > med_len).astype(int)
    )

    rng = np.random.RandomState(seed)
    selected = []
    per_stratum = max(1, n_proteins // 4)
    for stratum in sorted(train_df["_stratum"].unique()):
        group = train_df[train_df["_stratum"] == stratum]
        n_take = min(per_stratum, len(group))
        picked = group.sample(n=n_take, random_state=rng)
        selected.extend(picked["protein_id"].tolist())

    return selected[:n_proteins]


# ── WT verification (J2) ──────────────────────────────────────────────────

@dataclass
class WTVerificationResult:
    """Result of WT verification for one positive span."""
    protein_id: str
    start_0b: int
    end_0b: int
    pep_len: int
    support_n: int
    wt_el_rank: float
    classification: str  # keep_wt_confirmed / skip_uncertain / reject_wt_disagree


def verify_wt_positives(
    protein_id: str,
    protein_seq: str,
    positives: list[dict],
    runner,  # NetMHCIIpanRunner
    allele: str,
    cfg: AugmentationConfig,
) -> list[WTVerificationResult]:
    """Verify WT positives against NetMHCIIpan and classify each."""
    results = []
    # Group positives by pep_len for batched scoring
    by_length: dict[int, list[dict]] = {}
    for pos in positives:
        by_length.setdefault(pos["pep_len"], []).append(pos)

    for pep_len, spans in by_length.items():
        scores = runner.score_protein(protein_id, protein_seq, allele, pep_len)
        # Build lookup: pos -> PeptideScore
        score_by_pos = {s.pos: s for s in scores}

        for span in spans:
            ps = score_by_pos.get(span["start_0b"])
            wt_rank = ps.el_rank if ps is not None else 1.0  # missing = reject
            classification = classify_wt_rank(
                wt_rank, cfg.wt_rank_threshold, cfg.wt_uncertain_upper,
            )
            results.append(WTVerificationResult(
                protein_id=protein_id,
                start_0b=span["start_0b"],
                end_0b=span["end_0b"],
                pep_len=span["pep_len"],
                support_n=span.get("support_n", 0),
                wt_el_rank=wt_rank,
                classification=classification,
            ))

    return results


# ── Mutation enumeration (J3) ─────────────────────────────────────────────

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class MutationCandidate:
    """A single-point mutation candidate before scoring."""
    source_protein_id: str
    mut_pos_0b: int
    wt_aa: str
    mut_aa: str
    seed_spans: list[dict] = field(default_factory=list)


def enumerate_mutations(
    protein_id: str,
    protein_seq: str,
    eligible_spans: list[dict],
) -> list[MutationCandidate]:
    """Enumerate deduplicated single-point mutations within eligible spans.

    Dedup key: (protein_id, mut_pos_0b, mut_aa).
    Seed spans from multiple eligible spans are merged.
    """
    # Collect unique (pos, mut_aa) -> candidate
    candidates: dict[tuple[int, str], MutationCandidate] = {}

    for span in eligible_spans:
        start, end = span["start_0b"], span["end_0b"]
        for pos in range(start, end):
            wt_aa = protein_seq[pos]
            for mut_aa in AMINO_ACIDS:
                if mut_aa == wt_aa:
                    continue
                key = (pos, mut_aa)
                if key not in candidates:
                    candidates[key] = MutationCandidate(
                        source_protein_id=protein_id,
                        mut_pos_0b=pos,
                        wt_aa=wt_aa,
                        mut_aa=mut_aa,
                    )
                candidates[key].seed_spans.append({
                    "start_0b": start,
                    "end_0b": end,
                    "pep_len": span["pep_len"],
                })

    return sorted(candidates.values(), key=lambda c: (c.mut_pos_0b, c.mut_aa))


# ── Mutation scoring + affected-span recalculation (J3+J4) ────────────────

def _span_key(pep_len: int, start_0b: int) -> str:
    """JSON-safe key for per-span rank storage: '15:85'."""
    return f"{pep_len}:{start_0b}"


@dataclass
class ScoredMutation:
    """A scored and filtered mutation with affected spans."""
    source_protein_id: str
    mut_pos_0b: int
    wt_aa: str
    mut_aa: str
    seed_spans: list[dict]
    affected_spans: list[dict]  # spans disrupted by this mutation
    wt_ranks: dict              # {"pep_len:start_0b": wt_el_rank} per affected span
    mut_ranks: dict             # {"pep_len:start_0b": mut_el_rank} per affected span


def score_and_filter_mutations(
    protein_id: str,
    protein_seq: str,
    candidates: list[MutationCandidate],
    all_positives: list[dict],
    runner,  # NetMHCIIpanRunner
    allele: str,
    cfg: AugmentationConfig,
    wt_scores_cache: dict | None = None,
    max_candidates: int = 100,
) -> list[ScoredMutation]:
    """Score mutations and determine affected spans (J3+J4 combined).

    Uses LOCAL WINDOWS around each mutation site instead of full protein
    sequences, reducing NetMHCIIpan work by ~15x. Each batch entry is a
    short fragment (~60-80 AA) centered on the mutation.

    Per-span ranks use key 'pep_len:start_0b' to avoid collisions when
    multiple affected spans share the same pep_len.
    """
    if wt_scores_cache is None:
        wt_scores_cache = {}

    # Pre-compute: for each candidate, find overlapping positives
    cand_overlaps: list[
        tuple[MutationCandidate, list[dict], str, str, int]
    ] = []
    all_lengths_needed: set[int] = set()

    for cand in candidates:
        overlapping = [
            p for p in all_positives
            if p["start_0b"] <= cand.mut_pos_0b < p["end_0b"]
        ]
        if not overlapping:
            continue
        mut_id = f"{protein_id}__mut_{cand.mut_pos_0b}{cand.wt_aa}>{cand.mut_aa}"
        cand_overlaps.append((cand, overlapping, "", mut_id, 0))
        for p in overlapping:
            all_lengths_needed.add(p["pep_len"])

    if not cand_overlaps:
        return []

    # Cap candidates to limit worst-case runtime
    if len(cand_overlaps) > max_candidates:
        rng = np.random.RandomState(hash(protein_id) & 0x7FFFFFFF)
        indices = rng.choice(len(cand_overlaps), max_candidates, replace=False)
        indices.sort()
        cand_overlaps = [cand_overlaps[i] for i in indices]
        logger.info(
            "  %s: capped candidates from %d to %d",
            protein_id, len(candidates), max_candidates,
        )

    # Build local windows for each mutation
    max_k = max(all_lengths_needed) if all_lengths_needed else 25
    context_pad = max_k + 5
    rebuilt: list[tuple[MutationCandidate, list[dict], str, str, int]] = []
    for cand, overlapping, _, mut_id, _ in cand_overlaps:
        local_start = max(0, cand.mut_pos_0b - context_pad)
        local_end = min(len(protein_seq), cand.mut_pos_0b + context_pad + 1)
        mut_seq_local = (
            protein_seq[local_start:cand.mut_pos_0b]
            + cand.mut_aa
            + protein_seq[cand.mut_pos_0b + 1:local_end]
        )
        rebuilt.append((cand, overlapping, mut_seq_local, mut_id, local_start))
    cand_overlaps = rebuilt

    logger.info(
        "  %s: scoring %d candidates (%d lengths: %s)",
        protein_id, len(cand_overlaps), len(all_lengths_needed),
        sorted(all_lengths_needed),
    )

    # Cache WT scores (one call per pep_len for the full source protein)
    for pep_len in all_lengths_needed:
        cache_key = (protein_id, pep_len)
        if cache_key not in wt_scores_cache:
            wt_scores = runner.score_protein(
                protein_id, protein_seq, allele, pep_len,
            )
            wt_scores_cache[cache_key] = {s.pos: s.el_rank for s in wt_scores}

    # Score one pep_length at a time to keep per-call work manageable
    seen_ids: set[str] = set()
    batch_entries: list[tuple[str, str]] = []
    offset_map: dict[str, int] = {}
    for cand, overlapping, local_seq, mut_id, local_start in cand_overlaps:
        if mut_id not in seen_ids:
            seen_ids.add(mut_id)
            batch_entries.append((mut_id, local_seq))
            offset_map[mut_id] = local_start

    mut_batch_results: dict[int, dict[str, dict[int, float]]] = {}
    for pep_len in sorted(all_lengths_needed):
        partial = runner.score_batch(batch_entries, allele, [pep_len])
        for mid, by_len in partial.items():
            for pl, scores in by_len.items():
                mut_batch_results.setdefault(pl, {})[mid] = {
                    s.pos: s.el_rank for s in scores
                }

    # Check disruption per candidate using batched results
    scored = []
    for cand, overlapping, _, mut_id, local_start in cand_overlaps:
        affected = []
        wt_ranks = {}
        mut_ranks = {}

        for p in overlapping:
            pep_len = p["pep_len"]
            pos = p["start_0b"]
            local_pos = pos - local_start
            sk = _span_key(pep_len, pos)

            wt_r = wt_scores_cache.get((protein_id, pep_len), {}).get(pos, 1.0)
            mut_r = (
                mut_batch_results
                .get(pep_len, {})
                .get(mut_id, {})
                .get(local_pos, 0.0)
            )

            if is_disrupted(
                mut_r, wt_r, cfg.mut_rank_threshold, cfg.delta_rank_threshold,
            ):
                affected.append(p)
                wt_ranks[sk] = wt_r
                mut_ranks[sk] = mut_r

        if affected:
            scored.append(ScoredMutation(
                source_protein_id=protein_id,
                mut_pos_0b=cand.mut_pos_0b,
                wt_aa=cand.wt_aa,
                mut_aa=cand.mut_aa,
                seed_spans=cand.seed_spans,
                affected_spans=affected,
                wt_ranks=wt_ranks,
                mut_ranks=mut_ranks,
            ))

    logger.info(
        "  %s: %d / %d candidates are disruptions",
        protein_id, len(scored), len(cand_overlaps),
    )
    return scored


# ── Top-N selection per protein (J5) ──────────────────────────────────────

def select_top_mutations(
    mutations: list[ScoredMutation],
    top_n: int,
) -> list[ScoredMutation]:
    """Select top-N mutations per protein by max delta_rank across affected spans."""
    def max_delta(m: ScoredMutation) -> float:
        deltas = []
        for sk in m.mut_ranks:
            deltas.append(m.mut_ranks[sk] - m.wt_ranks.get(sk, 0))
        return max(deltas) if deltas else 0.0

    ranked = sorted(mutations, key=max_delta, reverse=True)
    return ranked[:top_n]


# ── Runtime augmentation (J7) ─────────────────────────────────────────────

@dataclass
class MutationRegistryIndex:
    """Pre-loaded registry index for runtime p_aug replacement.

    Groups eligible mutations by source_protein_id with precomputed
    sampling weights (linear proportional to delta_rank_seed).
    """
    # {source_protein_id: list of registry row dicts}
    by_protein: dict[str, list[dict]] = field(default_factory=dict)
    # {source_protein_id: np.ndarray of sampling weights}
    weights: dict[str, np.ndarray] = field(default_factory=dict)

    @staticmethod
    def from_parquet(registry_path: Path | str) -> "MutationRegistryIndex":
        """Load registry and build weighted index for runtime sampling."""
        path = Path(registry_path)
        if not path.exists():
            return MutationRegistryIndex()

        df = pd.read_parquet(path)
        if len(df) == 0:
            return MutationRegistryIndex()

        # Filter to runtime-eligible only
        if "runtime_train_eligible" in df.columns:
            df = df[df["runtime_train_eligible"] == True]

        idx = MutationRegistryIndex()
        for pid, group in df.groupby("source_protein_id"):
            rows = group.to_dict("records")
            idx.by_protein[pid] = rows
            # Linear proportional weights from max_delta_rank
            deltas = np.array([r["max_delta_rank"] for r in rows], dtype=np.float64)
            deltas = np.clip(deltas, 0.0, None)
            total = deltas.sum()
            if total > 0:
                idx.weights[pid] = deltas / total
            else:
                idx.weights[pid] = np.ones(len(rows)) / len(rows)

        return idx

    @property
    def n_eligible_proteins(self) -> int:
        return len(self.by_protein)

    @property
    def n_eligible_mutations(self) -> int:
        return sum(len(v) for v in self.by_protein.values())


def apply_runtime_augmentation(
    entries: list,  # list[ProteinEntry] — avoid circular import
    registry_idx: MutationRegistryIndex,
    p_aug: float,
    seed: int,
    epoch: int,
    return_stats: bool = False,
) -> list | tuple[list, dict]:
    """Build augmented epoch view by runtime p_aug replacement.

    For each base train protein with eligible registry mutations:
    - with probability (1 - p_aug): keep original
    - with probability p_aug: sample one mutation (weighted by delta_rank)
      and apply it (mutate seq + remove affected spans)

    Does NOT change list length — base protein count stays fixed.
    Deterministic: same (seed, epoch, protein_id) → same choice.

    Returns a new list (original entries are not mutated).
    When ``return_stats=True``, returns ``(entries, stats)`` where stats
    contains structured augmentation exposure counters for logging/audit.
    """
    n_base_entries = len(entries)
    n_eligible_entries = sum(1 for entry in entries if entry.protein_id in registry_idx.by_protein)

    def _stats(n_replaced: int) -> dict:
        return {
            "epoch": epoch,
            "configured_p_aug": float(p_aug),
            "n_base_entries": n_base_entries,
            "n_eligible_entries": n_eligible_entries,
            "n_replaced": n_replaced,
            "effective_aug_fraction": (
                float(n_replaced) / n_base_entries if n_base_entries > 0 else 0.0
            ),
        }

    if not registry_idx.by_protein or p_aug <= 0:
        entries_out = list(entries)
        return (entries_out, _stats(0)) if return_stats else entries_out

    result = []
    n_replaced = 0
    total_disrupted_spans = 0

    for entry in entries:
        pid = entry.protein_id
        if pid not in registry_idx.by_protein:
            result.append(entry)
            continue

        # Deterministic RNG per (seed, epoch, protein) — stable across processes
        h = int(hashlib.blake2b(
            f"{seed}:{epoch}:{pid}".encode(), digest_size=4,
        ).hexdigest(), 16)
        rng = np.random.RandomState(h)

        if rng.random() >= p_aug:
            result.append(entry)
            continue

        # Sample one mutation weighted by delta_rank
        mutations = registry_idx.by_protein[pid]
        weights = registry_idx.weights[pid]
        chosen_idx = rng.choice(len(mutations), p=weights)
        mut_row = mutations[chosen_idx]

        # Apply mutation to a COPY of the entry
        mut_pos = mut_row["mut_pos_0b"]
        mut_aa = mut_row["mut_aa"]
        mut_seq = entry.protein_seq[:mut_pos] + mut_aa + entry.protein_seq[mut_pos + 1:]

        affected = set()
        for a in json.loads(mut_row["affected_spans_json"]):
            affected.add((a["start_0b"], a["end_0b"], a["pep_len"]))

        remaining_positives = [
            p for p in entry.positives
            if (p["start_0b"], p["end_0b"], p["pep_len"]) not in affected
        ]

        # Safety: skip if no positives remain (shouldn't happen with registry filter)
        if not remaining_positives:
            result.append(entry)
            continue

        # Create mutated entry (same type as input)
        from copy import copy
        mut_entry = copy(entry)
        mut_entry.protein_seq = mut_seq
        mut_entry.positives = remaining_positives
        # Attach disrupted spans for hard negative priority in sample_negatives
        mut_entry.disrupted_spans = [
            {"start_0b": a[0], "end_0b": a[1], "pep_len": a[2]}
            for a in affected
        ]
        # Keep original protein_id so cardinality doesn't change
        result.append(mut_entry)
        n_replaced += 1
        total_disrupted_spans += len(mut_entry.disrupted_spans)

    logger.info(
        "Runtime augmentation: epoch=%d, p_aug=%.2f, replaced=%d/%d proteins, "
        "disrupted_spans=%d (avg %.1f/replaced)",
        epoch, p_aug, n_replaced, len(entries),
        total_disrupted_spans,
        total_disrupted_spans / max(n_replaced, 1),
    )
    stats = _stats(n_replaced)
    stats["total_disrupted_spans"] = total_disrupted_spans
    return (result, stats) if return_stats else result
