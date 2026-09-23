"""Pure logic for the RF refinement stage (PLAN_RF_REFINE.md).

Pure search logic for legacy NMP-core elimination and NMP-free Head-hotspot
refinement. No torch / no I/O: Head, NetMHCIIpan, and refold oracles are injected,
so each mode is unit-testable without model dependencies.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence

MHC2_POCKETS = (0, 3, 5, 8)  # P1, P4, P6, P9 offsets within the 9-mer MHC-II core
AA20 = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class EpitopeCore:
    core_start: int                       # 0-based core start in the sequence
    core_seq: str                         # the 9-mer binding core
    best_rank: float                      # min NMP rank_EL over covering strong windows
    pocket_positions: tuple[int, ...]     # absolute P1/P4/P6/P9 positions
    editable_positions: tuple[int, ...]   # positions this core may mutate (pockets minus anchors)


@dataclass(frozen=True)
class Candidate:
    desc: str                     # e.g. "D2R" or "D2R,G5W"
    seq: str                      # full mutated sequence (same length; no indels)
    positions: tuple[int, ...]    # edited positions, sorted ascending


def extract_target_cores(peptides: Iterable[dict], *, anchors,
                         strong_rank: float = 0.02) -> list[EpitopeCore]:
    """Aggregate NMP strong-binder windows into distinct 9-mer core targets.

    ``peptides``: rows with keys ``pos, peptide, core, rank_EL``. A window is a target
    iff ``rank_EL < strong_rank``. Cores are keyed by 0-based ``core_start`` (= ``pos``
    plus the offset of ``core`` within ``peptide``); the min-rank window wins per
    ``core_start``. Returned sorted by ``best_rank`` ascending (strongest first).
    """
    anchors = set(anchors)
    by_start: dict[int, dict] = {}
    for r in peptides:
        if float(r["rank_EL"]) >= strong_rank:
            continue
        offset = r["peptide"].find(r["core"])
        if offset < 0:
            continue
        cs = int(r["pos"]) + offset
        cur = by_start.get(cs)
        if cur is None or float(r["rank_EL"]) < cur["rank_EL"]:
            by_start[cs] = {"core": r["core"], "rank_EL": float(r["rank_EL"])}
    cores: list[EpitopeCore] = []
    for cs in sorted(by_start):
        pockets = tuple(cs + d for d in MHC2_POCKETS)
        cores.append(EpitopeCore(
            core_start=cs,
            core_seq=by_start[cs]["core"],
            best_rank=by_start[cs]["rank_EL"],
            pocket_positions=pockets,
            editable_positions=tuple(p for p in pockets if p not in anchors),
        ))
    return sorted(cores, key=lambda c: c.best_rank)


def editable_positions(core: EpitopeCore, *, anchors,
                       head_high_positions=frozenset()) -> tuple[int, ...]:
    """Positions of a core that may be mutated: its P1/P4/P6/P9 pockets plus any
    high-head residue inside the 9-mer span, minus frozen active-site anchors."""
    span = range(core.core_start, core.core_start + 9)
    positions = set(core.pocket_positions) | {p for p in head_high_positions if p in span}
    return tuple(sorted(positions - set(anchors)))


def hotspot_blocks(cores: Iterable[EpitopeCore], *, gap: int = 0) -> list[list[EpitopeCore]]:
    """Group cores into connected-component blocks: cores whose 9-mer spans overlap
    (or lie within ``gap`` positions) share a block. A block is the co-targeting search
    unit for overlapping epitopes (register-shift / whack-a-mole)."""
    ordered = sorted(cores, key=lambda c: c.core_start)
    blocks: list[list[EpitopeCore]] = []
    current: list[EpitopeCore] = []
    current_end: int | None = None
    for c in ordered:
        start, end = c.core_start, c.core_start + 9
        if current and current_end is not None and start <= current_end + gap:
            current.append(c)
            current_end = max(current_end, end)
        else:
            if current:
                blocks.append(current)
            current, current_end = [c], end
    if current:
        blocks.append(current)
    return blocks


def _apply(seq: str, muts) -> str:
    """Return ``seq`` with each ``(position, amino_acid)`` in ``muts`` substituted."""
    chars = list(seq)
    for pos, aa in muts:
        chars[pos] = aa
    return "".join(chars)


def enumerate_singles(seq: str, editable, *, alphabet: str = AA20) -> list[Candidate]:
    """Every single substitution over ``editable`` positions, skipping the wild-type AA
    at each position. This is the primary (exhaustive) proposer for a small block."""
    out: list[Candidate] = []
    for p in sorted({int(x) for x in editable}):
        wt = seq[p]
        for a in alphabet:
            if a != wt:
                out.append(Candidate(f"{wt}{p}{a}", _apply(seq, [(p, a)]), (p,)))
    return out


def enumerate_pairs(seq: str, single_choices, *, max_pairs: int | None = None) -> list[Candidate]:
    """Double substitutions drawn from ranked single ``(position, amino_acid)`` choices
    (best first). Same-position pairs are skipped and duplicate unordered pairs deduped;
    ``max_pairs`` caps the count. Feed the top head-ranked singles here to build the
    pair layer of block-structured enumeration."""
    out: list[Candidate] = []
    seen: set = set()
    n = len(single_choices)
    for i in range(n):
        pi, ai = single_choices[i]
        for j in range(i + 1, n):
            pj, aj = single_choices[j]
            if pi == pj:
                continue
            key = tuple(sorted([(pi, ai), (pj, aj)]))
            if key in seen:
                continue
            seen.add(key)
            muts = sorted([(pi, ai), (pj, aj)])
            out.append(Candidate(",".join(f"{seq[p]}{p}{a}" for p, a in muts),
                                 _apply(seq, muts), tuple(p for p, _ in muts)))
            if max_pairs is not None and len(out) >= max_pairs:
                return out
    return out


# --------------------------------------------------------------------------- #
# Task R3 — margin surrogate, structure gate, accept rule, beam search loop
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StructureMetrics:
    scTM: float
    pLDDT: float
    scRMSD: float | None = None            # populated when wired; None in v0
    active_site_RMSD: float | None = None  # legacy C-alpha active-site compatibility field
    global_ca_RMSD: float | None = None
    active_site_sidechain_RMSD: float | None = None
    max_anchor_sidechain_RMSD: float | None = None
    max_anchor_atom_distance: float | None = None
    active_site_complete: bool | None = None
    active_site_min_pLDDT: float | None = None
    cat_max_scRMSD: float | None = None
    predicted_active_site_min_pLDDT: float | None = None
    cache_hit: bool = False
    model_executed: bool = True
    passed: bool = True                    # set by structure_gate
    reason: str = ""


def rank_margin_mass(core_ranks, *, strong_rank: float = 0.02, margin_band: float = 0.10) -> float:
    """Continuous NMP surrogate to MINIMISE: Σ over cores with rank_EL < margin_band of
    max(strong_rank − rank_EL, 0). Deep strong cores contribute most; a core pushed to
    ≥ strong_rank contributes 0 (eliminated from the strong count)."""
    return float(sum(max(strong_rank - r, 0.0) for r in core_ranks if r < margin_band))


def structure_gate(seed: "StructureMetrics", cand: "StructureMetrics", *,
                   scTM_eps: float | None,
                   scTM_min: float | None = None,
                   cat_max_scRMSD_max: float | None = None,
                   predicted_active_site_min_pLDDT_min: float | None = None,
                   scRMSD_max: float | None = None,
                   active_site_RMSD_max: float | None = None,
                   max_anchor_sidechain_RMSD_max: float | None = None) -> tuple[bool, str]:
    """Verdict for a candidate fold against configured structure criteria.

    The protocol gate uses an absolute scTM floor, direct-functional catalytic side-chain
    maximum, and worst active-site confidence. Legacy seed-relative/global ceilings remain
    available for explicit old callers. Every configured metric is fail-closed.
    """
    if scTM_min is not None:
        if not math.isfinite(float(cand.scTM)):
            return False, f"scTM unavailable/not finite: {cand.scTM}"
        if cand.scTM < scTM_min:
            return False, f"scTM {cand.scTM:.3f} < {scTM_min:.3f}"
    if scTM_eps is not None:
        if not math.isfinite(float(cand.scTM)) or not math.isfinite(float(seed.scTM)):
            return False, f"scTM unavailable/not finite: seed={seed.scTM}, cand={cand.scTM}"
        if cand.scTM < seed.scTM - scTM_eps:
            return False, f"scTM {cand.scTM:.3f} < {seed.scTM - scTM_eps:.3f}"
    if cat_max_scRMSD_max is not None:
        value = cand.cat_max_scRMSD
        if value is None or not math.isfinite(float(value)):
            return False, f"cat_max_scRMSD unavailable/not finite: {value}"
        if value > cat_max_scRMSD_max:
            return False, f"cat_max_scRMSD {value:.2f} > {cat_max_scRMSD_max}"
    if predicted_active_site_min_pLDDT_min is not None:
        value = cand.predicted_active_site_min_pLDDT
        if value is None or not math.isfinite(float(value)):
            return False, (
                "predicted_active_site_min_pLDDT unavailable/not finite: "
                f"{value}"
            )
        if value < predicted_active_site_min_pLDDT_min:
            return False, (
                f"predicted_active_site_min_pLDDT {value:.2f} < "
                f"{predicted_active_site_min_pLDDT_min}"
            )
    if scRMSD_max is not None:
        if cand.scRMSD is None or not math.isfinite(float(cand.scRMSD)):
            return False, f"scRMSD unavailable/not finite: {cand.scRMSD}"
        if cand.scRMSD > scRMSD_max:
            return False, f"scRMSD {cand.scRMSD:.2f} > {scRMSD_max}"
    if active_site_RMSD_max is not None:
        if cand.active_site_RMSD is None or not math.isfinite(float(cand.active_site_RMSD)):
            return False, f"active_site_RMSD unavailable/not finite: {cand.active_site_RMSD}"
        if cand.active_site_RMSD > active_site_RMSD_max:
            return False, f"active_site_RMSD {cand.active_site_RMSD:.2f} > {active_site_RMSD_max}"
    if max_anchor_sidechain_RMSD_max is not None:
        if cand.active_site_complete is not True:
            return False, (
                "max_anchor_sidechain_RMSD unavailable: side-chain active-site metrics "
                "incomplete"
            )
        value = cand.max_anchor_sidechain_RMSD
        if value is None or not math.isfinite(float(value)):
            return False, f"max_anchor_sidechain_RMSD unavailable/not finite: {value}"
        if value > max_anchor_sidechain_RMSD_max:
            return False, (
                f"max_anchor_sidechain_RMSD {value:.2f} > "
                f"{max_anchor_sidechain_RMSD_max}"
            )
    return True, "ok"


def accept_refinement(*, seed_core_count: int, cand_core_count: int,
                      structure_passed: bool) -> bool:
    """Hard output-accept: strictly fewer distinct cores than the seed AND the candidate
    passed the structure gate. Structure thresholds live in ``structure_gate`` so adding
    scRMSD / active-site metrics never touches this predicate."""
    return cand_core_count < seed_core_count and structure_passed


@dataclass
class RefineResult:
    protein_id: str
    seed_seq: str
    seed_core_count: int
    best_seq: str
    best_core_count: int
    best_structure: "StructureMetrics"
    n_accepts: int
    diverged: bool
    shortlist: list = field(default_factory=list)    # per-output immune + selected structure metrics
    trace: list = field(default_factory=list)


@dataclass(frozen=True)
class HeadRefinementMetrics:
    """Validated two-axis Head objective plus its residue landscape."""

    global_risk: float
    positive_mass: float
    positive_mass_density: float
    mean_hotspot: float
    max_hotspot: float
    n_positive_hotspot_positions: int
    residue_hotspot: tuple[float, ...]
    windows: tuple[object, ...]


@dataclass(frozen=True)
class HeadResidueTargets:
    """Thresholded residue-local maxima used by the Head-only proposer.

    ``positions`` is ordered by descending hotspot value, then coordinate. The
    threshold is a proposal-localization threshold under one frozen Head identity;
    it is not represented as an absolute immunogenicity probability.
    """

    positions: tuple[int, ...]
    hotspot_values: tuple[float, ...]
    threshold: float
    n_above_threshold: int
    n_local_maxima_pre_cap: int


@dataclass
class HeadRefineResult:
    """Result of the NMP-free, two-axis Head refinement path."""

    protein_id: str
    seed_seq: str
    seed_head: HeadRefinementMetrics
    seed_structure: "StructureMetrics"
    n_accepts: int
    diverged: bool
    shortlist: list = field(default_factory=list)
    trace: list = field(default_factory=list)


def head_refinement_metrics(
    score,
    *,
    protein_id: str,
    sequence: str,
) -> HeadRefinementMetrics:
    """Validate the Head fields and derive the frozen positive-mass density axis.

    ``residue_hotspot`` supplies both the proposal landscape and positive-mass
    objective. ``windows`` remains required evaluator evidence and is validated even
    though residue targeting does not infer a window register. The two co-equal
    lower-is-better axes are ``global_risk`` and
    ``sum(max(residue_hotspot, 0)) / sequence_length``. Optional identity fields
    carried by production ``HeadScore`` objects are checked when present.
    """
    import numpy as np

    risk = float(getattr(score, "global_risk", float("nan")))
    if not math.isfinite(risk):
        raise ValueError("Head global_risk is missing or non-finite")
    if not sequence:
        raise ValueError("Head refinement requires a non-empty sequence")

    score_pid = getattr(score, "protein_id", None)
    if score_pid is not None and str(score_pid) != str(protein_id):
        raise ValueError(f"Head protein_id mismatch: {score_pid!r} != {protein_id!r}")
    score_length = getattr(score, "sequence_length", None)
    if score_length is not None and int(score_length) != len(sequence):
        raise ValueError(
            f"Head sequence_length mismatch: {score_length} != {len(sequence)}"
        )
    score_md5 = getattr(score, "sequence_md5", None)
    expected_md5 = hashlib.md5(sequence.encode("utf-8")).hexdigest()
    if score_md5 is not None and str(score_md5) != expected_md5:
        raise ValueError("Head sequence_md5 does not match the scored sequence")

    raw_hotspot = getattr(score, "residue_hotspot", None)
    if raw_hotspot is None:
        raise ValueError("Head residue_hotspot is required for Head target mode")
    hotspot = np.asarray(raw_hotspot, dtype=float)
    if hotspot.ndim != 1 or hotspot.shape[0] != len(sequence):
        raise ValueError(
            "Head residue_hotspot must be one-dimensional with one value per residue"
        )
    if not np.isfinite(hotspot).all():
        raise ValueError("Head residue_hotspot contains non-finite values")

    raw_windows = getattr(score, "windows", None)
    if raw_windows is None:
        raise ValueError("Head windows are required for Head target mode")
    windows = tuple(raw_windows)
    for index, window in enumerate(windows):
        fields = {}
        for name in ("start_0b", "end_0b", "k"):
            value = getattr(window, name, None)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Head window {index} has invalid integer {name}")
            fields[name] = int(value)
        z = float(getattr(window, "z", float("nan")))
        if not math.isfinite(z):
            raise ValueError(f"Head window {index} has non-finite z")
        start, end, k = fields["start_0b"], fields["end_0b"], fields["k"]
        if start < 0 or end > len(sequence) or end <= start or k != end - start:
            raise ValueError(
                f"Head window {index} has invalid span/k: [{start}, {end}), k={k}, "
                f"sequence_length={len(sequence)}"
            )
    if not windows:
        raise ValueError("Head windows are empty; evaluator evidence is incomplete")

    positive_mass = math.fsum(max(float(value), 0.0) for value in hotspot)
    return HeadRefinementMetrics(
        global_risk=risk,
        positive_mass=positive_mass,
        positive_mass_density=positive_mass / len(sequence),
        mean_hotspot=float(hotspot.mean()),
        max_hotspot=float(hotspot.max()),
        n_positive_hotspot_positions=int((hotspot > 0.0).sum()),
        residue_hotspot=tuple(float(value) for value in hotspot),
        windows=windows,
    )


def head_objective_dominates(
    challenger: HeadRefinementMetrics,
    incumbent: HeadRefinementMetrics,
) -> bool:
    """True when ``challenger`` Pareto-dominates ``incumbent`` on both Head axes."""
    no_worse = (
        challenger.global_risk <= incumbent.global_risk
        and challenger.positive_mass_density <= incumbent.positive_mass_density
    )
    strictly_better = (
        challenger.global_risk < incumbent.global_risk
        or challenger.positive_mass_density < incumbent.positive_mass_density
    )
    return bool(no_worse and strictly_better)


def select_head_residue_targets(
    metrics: HeadRefinementMetrics,
    *,
    anchors,
    threshold: float = 0.15,
    max_positions: int = 12,
) -> HeadResidueTargets:
    """Select capped, editable residue-local maxima above ``threshold``.

    Anchors are masked before maxima detection, so an uneditable peak cannot hide a
    nearby editable shoulder. Exact-value plateaus collapse to their lower middle
    coordinate. This reads the complete residue landscape and may return zero targets.
    """
    cutoff = float(threshold)
    if not math.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError(
            f"Head residue threshold must be finite and non-negative: {threshold}"
        )
    cap = int(max_positions)
    if cap < 1:
        raise ValueError(f"Head max target positions must be positive: {max_positions}")

    hotspot = tuple(float(value) for value in metrics.residue_hotspot)
    frozen = {int(value) for value in anchors}
    eligible = [
        index not in frozen and value >= cutoff
        for index, value in enumerate(hotspot)
    ]
    n_above = sum(eligible)
    maxima: list[int] = []
    index = 0
    while index < len(hotspot):
        if not eligible[index]:
            index += 1
            continue
        plateau_end = index
        while (
            plateau_end + 1 < len(hotspot)
            and eligible[plateau_end + 1]
            and hotspot[plateau_end + 1] == hotspot[index]
        ):
            plateau_end += 1
        left = (
            hotspot[index - 1]
            if index > 0 and index - 1 not in frozen
            else -math.inf
        )
        right = (
            hotspot[plateau_end + 1]
            if plateau_end + 1 < len(hotspot) and plateau_end + 1 not in frozen
            else -math.inf
        )
        if hotspot[index] > left and hotspot[index] > right:
            maxima.append((index + plateau_end) // 2)
        index = plateau_end + 1

    ranked = sorted(maxima, key=lambda position: (-hotspot[position], position))
    selected = tuple(ranked[:cap])
    return HeadResidueTargets(
        positions=selected,
        hotspot_values=tuple(hotspot[position] for position in selected),
        threshold=cutoff,
        n_above_threshold=int(n_above),
        n_local_maxima_pre_cap=len(maxima),
    )


def _hamming_positions(a: str, b: str) -> list[int]:
    """0-based positions where two equal-length strings differ (the point-edit set)."""
    return [i for i in range(min(len(a), len(b))) if a[i] != b[i]]


def head_first_pareto_front(metrics: Iterable[HeadRefinementMetrics]) -> tuple[int, ...]:
    """Indices on the complete first front of the two frozen Head axes.

    The two-dimensional sweep is ``O(n log n)`` and preserves exact comparisons. Points
    with identical axes remain together on the front, matching the facade selector.
    """
    rows = list(metrics)
    ordered = sorted(
        range(len(rows)),
        key=lambda idx: (
            rows[idx].global_risk,
            rows[idx].positive_mass_density,
            idx,
        ),
    )
    dominated = [False] * len(rows)
    best_prior_density = math.inf
    cursor = 0
    while cursor < len(ordered):
        group_end = cursor + 1
        group_risk = rows[ordered[cursor]].global_risk
        while (
            group_end < len(ordered)
            and rows[ordered[group_end]].global_risk == group_risk
        ):
            group_end += 1
        group = ordered[cursor:group_end]
        group_min_density = min(rows[idx].positive_mass_density for idx in group)
        for idx in group:
            density = rows[idx].positive_mass_density
            dominated[idx] = (
                best_prior_density <= density or group_min_density < density
            )
        best_prior_density = min(best_prior_density, group_min_density)
        cursor = group_end
    return tuple(idx for idx, is_dominated in enumerate(dominated) if not is_dominated)


def _cap_head_front(entries: list[dict], cap: int | None) -> list[dict]:
    """Deterministically cap a front without scalarizing its two Head axes.

    Both axis minima are retained first; any remaining slots use sequence MD5 only.
    This keeps the cap metric-neutral rather than introducing an unregistered weight.
    """
    if cap is None or len(entries) <= cap:
        return list(entries)
    cap = int(cap)
    if cap <= 0:
        return []
    ordered = sorted(
        entries,
        key=lambda row: hashlib.md5(row["seq"].encode("utf-8")).hexdigest(),
    )
    selected: list[dict] = []
    for key in (
        lambda row: row["head"].global_risk,
        lambda row: row["head"].positive_mass_density,
    ):
        extreme = min(ordered, key=lambda row: (key(row), row["seq"]))
        if extreme not in selected:
            selected.append(extreme)
        if len(selected) == cap:
            return selected
    for row in ordered:
        if row not in selected:
            selected.append(row)
        if len(selected) == cap:
            break
    return selected


def select_head_single_aa_choices(
    candidates: Sequence[Candidate],
    metrics_by_sequence: Mapping[str, HeadRefinementMetrics],
    *,
    position_order: Sequence[int],
    max_aa_per_position: int = 2,
) -> dict[int, tuple[Candidate, ...]]:
    """Keep a small Pareto-ordered AA set independently at each position.

    Every Pareto layer contributes its global-risk and positive-mass-density
    extremes first. This makes the default cap of two exactly one representative
    per Head axis when the extrema differ, without introducing a scalar weight.
    """
    cap = int(max_aa_per_position)
    if cap < 1:
        raise ValueError(f"Head AA choices per position must be positive: {cap}")
    by_position: dict[int, list[Candidate]] = {
        int(position): [] for position in position_order
    }
    for candidate in candidates:
        if len(candidate.positions) != 1:
            raise ValueError("Head single-AA selection requires single-mutant candidates")
        position = int(candidate.positions[0])
        if candidate.seq not in metrics_by_sequence:
            raise ValueError(f"missing Head metrics for single candidate {candidate.seq}")
        by_position.setdefault(position, []).append(candidate)

    selected: dict[int, tuple[Candidate, ...]] = {}
    for raw_position in position_order:
        position = int(raw_position)
        remaining = list(by_position.get(position, ()))
        ordered: list[Candidate] = []
        while remaining and len(ordered) < cap:
            front_indices = head_first_pareto_front(
                metrics_by_sequence[candidate.seq] for candidate in remaining
            )
            front = [remaining[index] for index in front_indices]
            layer: list[Candidate] = []
            for key in (
                lambda candidate: metrics_by_sequence[candidate.seq].global_risk,
                lambda candidate: metrics_by_sequence[
                    candidate.seq
                ].positive_mass_density,
            ):
                extreme = min(front, key=lambda candidate: (key(candidate), candidate.seq))
                if extreme not in layer:
                    layer.append(extreme)
            layer.extend(sorted(
                (candidate for candidate in front if candidate not in layer),
                key=lambda candidate: hashlib.md5(
                    candidate.seq.encode("utf-8")
                ).hexdigest(),
            ))
            for candidate in layer:
                if len(ordered) == cap:
                    break
                ordered.append(candidate)
            front_sequences = {candidate.seq for candidate in front}
            remaining = [
                candidate for candidate in remaining
                if candidate.seq not in front_sequences
            ]
        selected[position] = tuple(ordered)
    return selected


def round_robin_position_pairs(
    positions: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """All unordered position pairs in deterministic tournament rounds.

    Within each round every position appears at most once. Flattening these rounds
    lets a global candidate cap distribute work across positions instead of
    exhausting every AA combination for the first coordinate pair.
    """
    participants = [int(position) for position in positions]
    if len(set(participants)) != len(participants):
        raise ValueError("position round-robin requires unique positions")
    if len(participants) < 2:
        return ()
    rotation: list[int | None] = list(participants)
    if len(rotation) % 2:
        rotation.append(None)
    pairs: list[tuple[int, int]] = []
    for _round in range(len(rotation) - 1):
        for index in range(len(rotation) // 2):
            left = rotation[index]
            right = rotation[-1 - index]
            if left is None or right is None:
                continue
            pairs.append(tuple(sorted((left, right))))
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return tuple(pairs)


def enumerate_round_robin_doubles(
    seq: str,
    choices_by_position: Mapping[int, Sequence[Candidate]],
    *,
    position_order: Sequence[int],
    max_candidates: int,
) -> list[Candidate]:
    """Compose retained single AAs into capped, full double mutants.

    Combination layer zero (the first retained AA at each endpoint) is emitted for
    every position pair before layer one is considered for any pair.
    """
    cap = int(max_candidates)
    if cap < 0:
        raise ValueError(f"Head double-mutant cap must be non-negative: {cap}")
    if cap == 0:
        return []
    active_positions = [
        int(position) for position in position_order
        if choices_by_position.get(int(position))
    ]
    pair_order = round_robin_position_pairs(active_positions)
    combinations: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for left, right in pair_order:
        combinations[(left, right)] = [
            (left_candidate.seq[left], right_candidate.seq[right])
            for left_candidate in choices_by_position[left]
            for right_candidate in choices_by_position[right]
        ]
    max_layers = max((len(rows) for rows in combinations.values()), default=0)
    out: list[Candidate] = []
    seen_sequences: set[str] = set()
    for layer in range(max_layers):
        for left, right in pair_order:
            rows = combinations[(left, right)]
            if layer >= len(rows):
                continue
            left_aa, right_aa = rows[layer]
            muts = ((left, left_aa), (right, right_aa))
            candidate_seq = _apply(seq, muts)
            if candidate_seq in seen_sequences:
                continue
            seen_sequences.add(candidate_seq)
            out.append(Candidate(
                ",".join(f"{seq[position]}{position}{aa}" for position, aa in muts),
                candidate_seq,
                (left, right),
            ))
            if len(out) == cap:
                return out
    return out


def refine_sequence_head(
    protein_id,
    seed_seq,
    *,
    head_score_fn,
    struct_fn,
    anchors,
    residue_threshold=0.15,
    max_target_positions=12,
    aa_per_position=2,
    max_double_candidates=200,
    alphabet=AA20,
    scTM_eps=0.05,
    scRMSD_max=None,
    active_site_RMSD_max=None,
    scTM_min=None,
    cat_max_scRMSD_max=None,
    predicted_active_site_min_pLDDT_min=None,
    max_anchor_sidechain_RMSD_max=None,
    topB=None,
    beam_width=8,
    max_rounds=20,
    patience=3,
    max_path_mutations=8,
    refold_cap=16,
    allow_structure_unknown=False,
    log_fn=None,
):
    """NMP-free Head refinement on a complete two-axis Pareto objective.

    Residue-local maxima above a fixed Head-identity-bound threshold localize proposals.
    Every round scores all singles first, retains a small per-position AA Pareto set,
    then scores capped full double mutants emitted in position round-robin order. A
    child can advance only when it Pareto-dominates its parent on both ``global_risk``
    and positive-mass density. NetMHCIIpan is absent from this interface by construction.
    """
    import time

    def score_sequences(sequences):
        unique = list(dict.fromkeys(sequences))
        if not unique:
            return {}
        raw_scores = list(head_score_fn(protein_id, unique))
        if len(raw_scores) != len(unique):
            raise RuntimeError(
                f"Head returned {len(raw_scores)} rows for {len(unique)} sequences"
            )
        return {
            sequence: head_refinement_metrics(
                score,
                protein_id=protein_id,
                sequence=sequence,
            )
            for sequence, score in zip(unique, raw_scores)
        }

    seed_head = score_sequences([seed_seq])[seed_seq]
    seed_structure = struct_fn(protein_id, seed_seq)
    beam = [{"seq": seed_seq, "head": seed_head, "structure": seed_structure}]
    accepted: dict[str, dict] = {}
    trace: list[dict] = []
    stale = 0

    def beam_signature(states):
        return tuple(sorted(
            (
                row["head"].global_risk,
                row["head"].positive_mass_density,
                hashlib.md5(row["seq"].encode("utf-8")).hexdigest(),
            )
            for row in states
        ))

    signature = beam_signature(beam)
    for round_idx in range(int(max_rounds)):
        targets = [select_head_residue_targets(
            state["head"],
            anchors=anchors,
            threshold=residue_threshold,
            max_positions=max_target_positions,
        ) for state in beam]
        target_sizes = [
            len(target.positions) for target in targets if target.positions
        ]
        singles_by_parent = [
            enumerate_singles(state["seq"], target.positions, alphabet=alphabet)
            for state, target in zip(beam, targets)
        ]
        single_pool = [
            (parent_index, state, candidate)
            for parent_index, (state, candidates) in enumerate(
                zip(beam, singles_by_parent)
            )
            for candidate in candidates
        ]
        if not single_pool:
            break

        started = time.time()
        single_head_by_sequence = score_sequences([
            candidate.seq for _parent_index, _state, candidate in single_pool
        ])
        t_head_single = time.time() - started

        double_pool: list[tuple[int, dict, Candidate]] = []
        n_retained_single_aas = 0
        for parent_index, (state, target, singles) in enumerate(
            zip(beam, targets, singles_by_parent)
        ):
            choices = select_head_single_aa_choices(
                singles,
                single_head_by_sequence,
                position_order=target.positions,
                max_aa_per_position=aa_per_position,
            )
            n_retained_single_aas += sum(len(rows) for rows in choices.values())
            doubles = enumerate_round_robin_doubles(
                state["seq"],
                choices,
                position_order=target.positions,
                max_candidates=max_double_candidates,
            )
            double_pool.extend(
                (parent_index, state, candidate) for candidate in doubles
            )

        started = time.time()
        double_head_by_sequence = score_sequences([
            candidate.seq for _parent_index, _state, candidate in double_pool
        ])
        t_head_double = time.time() - started
        head_by_sequence = dict(single_head_by_sequence)
        head_by_sequence.update(double_head_by_sequence)
        pool = single_pool + double_pool

        improving_by_sequence: dict[str, dict] = {}
        for _parent_index, parent, candidate in pool:
            child_head = head_by_sequence[candidate.seq]
            n_mutations = len(_hamming_positions(seed_seq, candidate.seq))
            if n_mutations > int(max_path_mutations):
                continue
            if not head_objective_dominates(child_head, parent["head"]):
                continue
            improving_by_sequence.setdefault(candidate.seq, {
                "seq": candidate.seq,
                "head": child_head,
                "candidate": candidate,
                "n_mutations": n_mutations,
            })

        improving = list(improving_by_sequence.values())
        if improving:
            front_indices = head_first_pareto_front(row["head"] for row in improving)
            improving = [improving[idx] for idx in front_indices]
        improving = _cap_head_front(improving, topB)
        to_refold = _cap_head_front(improving, refold_cap)

        admitted: list[dict] = []
        t_refold = 0.0
        for row in to_refold:
            refold_started = time.time()
            metrics = struct_fn(protein_id, row["seq"])
            passed, reason = structure_gate(
                seed_structure,
                metrics,
                scTM_eps=scTM_eps,
                scTM_min=scTM_min,
                cat_max_scRMSD_max=cat_max_scRMSD_max,
                predicted_active_site_min_pLDDT_min=(
                    predicted_active_site_min_pLDDT_min
                ),
                scRMSD_max=scRMSD_max,
                active_site_RMSD_max=active_site_RMSD_max,
                max_anchor_sidechain_RMSD_max=max_anchor_sidechain_RMSD_max,
            )
            t_refold += time.time() - refold_started
            metrics = replace(metrics, passed=passed, reason=reason)
            if not (passed or allow_structure_unknown):
                continue
            state = {"seq": row["seq"], "head": row["head"], "structure": metrics}
            admitted.append(state)
            accepted[row["seq"]] = {
                **state,
                "muts": row["candidate"].desc,
                "n_mutations": row["n_mutations"],
            }

        combined_by_sequence = {row["seq"]: row for row in beam}
        combined_by_sequence.update({row["seq"]: row for row in admitted})
        combined = list(combined_by_sequence.values())
        front_indices = head_first_pareto_front(row["head"] for row in combined)
        beam = _cap_head_front([combined[idx] for idx in front_indices], int(beam_width))

        new_signature = beam_signature(beam)
        min_global = min(row["head"].global_risk for row in beam)
        min_density = min(row["head"].positive_mass_density for row in beam)
        trace.append({
            "round": round_idx,
            "n_candidates": len(pool),
            "n_head": len(head_by_sequence),
            "n_single_candidates": len(single_pool),
            "n_retained_single_aas": n_retained_single_aas,
            "n_double_candidates": len(double_pool),
            "n_head_single": len(single_head_by_sequence),
            "n_head_double": len(double_head_by_sequence),
            "n_pareto_improving": len(improving),
            "n_refold": len(to_refold),
            "n_structure_admitted": len(admitted),
            "beam_size": len(beam),
            "n_target_sets": len(target_sizes),
            "min_target_positions": min(target_sizes) if target_sizes else 0,
            "max_target_positions": max(target_sizes) if target_sizes else 0,
            "max_above_threshold_positions": max(
                (target.n_above_threshold for target in targets), default=0
            ),
            "max_local_maxima_pre_cap": max(
                (target.n_local_maxima_pre_cap for target in targets), default=0
            ),
            "head_residue_threshold": float(residue_threshold),
            "beam_min_global_risk": min_global,
            "beam_min_positive_mass_density": min_density,
            "t_head": round(t_head_single + t_head_double, 2),
            "t_head_single": round(t_head_single, 2),
            "t_head_double": round(t_head_double, 2),
            "t_refold": round(t_refold, 2),
        })
        if log_fn is not None:
            log_fn(
                f"[refine:head] {protein_id} r{round_idx}: "
                f"singles={len(single_pool)} doubles={len(double_pool)} "
                f"head={len(head_by_sequence)} pareto={len(improving)} "
                f"refold={len(to_refold)} admitted={len(admitted)} "
                f"target_positions_max={max(target_sizes) if target_sizes else 0} "
                f"min_global={min_global:.6f} min_density={min_density:.6f}"
            )
        if new_signature != signature:
            signature, stale = new_signature, 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    shortlist = sorted(
        accepted.values(),
        key=lambda row: (
            row["head"].global_risk,
            row["head"].positive_mass_density,
            hashlib.md5(row["seq"].encode("utf-8")).hexdigest(),
        ),
    )
    return HeadRefineResult(
        protein_id=protein_id,
        seed_seq=seed_seq,
        seed_head=seed_head,
        seed_structure=seed_structure,
        n_accepts=len(shortlist),
        diverged=not shortlist,
        shortlist=shortlist,
        trace=trace,
    )


def splice_nmp_rows(seed_rows, sub_rows, edited_positions, offset, *, context=3):
    """Merge a candidate's sub-sequence NMP rows over the cached seed rows.

    A window (peptide) at protein position ``pos`` of length ``pep_length`` COVERS an edit
    iff ``[pos-context, pos+pep_length+context)`` intersects ``edited_positions`` — the
    ``context`` flanks matter because NetMHCIIpan runs with ``-context`` (±3 residues), so an
    edit just OUTSIDE a peptide still changes its %Rank_EL. Covered windows are taken from
    ``sub_rows`` (``pos`` remapped by ``+offset``); every other window is copied verbatim from
    ``seed_rows`` — byte-identical to a full re-score because neither the peptide nor its
    context touches the mutation. EXACT iff each covered window keeps its full peptide+context
    inside the sub-sequence; the caller guarantees this with margin ≥ max_len + context.
    """
    edited = set(int(p) for p in edited_positions)

    def covers(pos, plen):
        return any(pos - context <= p < pos + plen + context for p in edited)

    sub_covered = {}
    for r in sub_rows:
        pos, plen = int(r["pos"]) + offset, int(r["pep_length"])
        if covers(pos, plen):
            rr = dict(r)
            rr["pos"] = pos
            sub_covered[(pos, plen)] = rr
    merged = []
    for r in seed_rows:
        pos, plen = int(r["pos"]), int(r["pep_length"])
        if covers(pos, plen):
            repl = sub_covered.get((pos, plen))
            if repl is not None:
                merged.append(repl)
            # else: a covered window absent from the sub-seq means the margin was too small;
            # the caller's span-budget fallback prevents this, so drop defensively.
        else:
            merged.append(dict(r))
    return merged


def _incremental_nmp_rows(protein_id, seed_seq, seed_rows, cand_seqs, nmp_fn, *, margin=30,
                          context=3):
    """Score candidate sequences against a cached seed via the sub-sequence splice.

    Per candidate: diff vs ``seed_seq`` → edited positions; unchanged → reuse ``seed_rows``;
    edit span ≤ ``2*margin+1`` → score only ``seq[minP-margin : maxP+margin+1]`` and splice;
    else fall back to a full-length re-score. ``margin`` must be ≥ ``max_pep_len + context``
    (30 ≥ 25 + 3) so every window whose peptide OR ``-context`` flank touches an edit keeps its
    full footprint inside the sub-sequence. All sub-sequences (+ fallbacks) go through ONE
    batched ``nmp_fn`` call so the NetMHCIIpan model-load is amortized. Returns one row-list
    per candidate, in order.
    """
    n = len(seed_seq)
    budget = 2 * margin + 1
    plans, batch = [], []
    for s in cand_seqs:
        pos = _hamming_positions(s, seed_seq)
        if not pos:
            plans.append(("hit", None, None))
        elif pos[-1] - pos[0] + 1 > budget:
            plans.append(("full", len(batch), None))
            batch.append(s)
        else:
            offset = max(0, pos[0] - margin)
            plans.append(("splice", len(batch), (offset, pos)))
            batch.append(s[offset:min(n, pos[-1] + margin + 1)])
    scored = nmp_fn(protein_id, batch) if batch else []
    out = []
    for kind, idx, extra in plans:
        if kind == "hit":
            out.append(seed_rows)
        elif kind == "full":
            out.append(scored[idx])
        else:
            offset, pos = extra
            out.append(splice_nmp_rows(seed_rows, scored[idx], pos, offset, context=context))
    return out


def refine_sequence(protein_id, seed_seq, *, propose_fn, head_fn, nmp_fn, struct_fn, anchors,
                    target_window_idx_fn=None, strong_rank=0.02, margin_band=0.10,
                    scTM_eps=0.05, scRMSD_max=None, active_site_RMSD_max=None,
                    scTM_min=None, cat_max_scRMSD_max=None,
                    predicted_active_site_min_pLDDT_min=None,
                    max_anchor_sidechain_RMSD_max=None,
                    topB=None, beam_width=8, max_rounds=20, patience=3,
                    max_path_mutations=8, refold_cap=16, allow_structure_unknown=False,
                    incremental_nmp=False, nmp_context_margin=30, log_fn=None):
    """Beam search that eliminates NMP epitope cores. ``nmp_fn`` is BATCHED:
    ``(protein_id, list[seq]) -> list[list[dict]]`` (one NMP row-list per input sequence);
    each round scores its whole candidate set in ONE ``nmp_fn`` call and duplicate sequences
    are scored once. ``head_fn`` (``(pid, seqs) -> [K, W]``) is only invoked when ``topB`` is
    finite — NMP-all (``topB=None``) discards the ranking, so the head pass is skipped.

    ``incremental_nmp`` (exact for per-window NMP): score only each candidate's mutated
    sub-sequence and splice it over the cached seed rows (a ~10x NMP-volume cut with identical
    cores/count/margin — see ``splice_nmp_rows``). Off by default; the driver enables it since
    real NetMHCIIpan scores each window against a fixed allele background.

    Structure is gated on OUTPUTS, not every step (PLAN §1): a candidate that drops the
    distinct-core count below the seed is a potential output → refold + ``structure_gate`` now
    (pass → shortlist + beam, fail → excluded from both); a margin-progress state (count not
    below the seed) enters the beam with NO refold. ``max_path_mutations`` caps stacked edits.
    ``log_fn`` (optional) receives one flushed line per round for progress visibility."""
    import time

    import numpy as np

    def _rows_to_state(seq, rows):
        cores = extract_target_cores(rows, anchors=anchors, strong_rank=strong_rank)
        margin = rank_margin_mass([c.best_rank for c in cores], strong_rank=strong_rank,
                                  margin_band=margin_band)
        return {"seq": seq, "cores": cores, "count": len(cores), "margin": margin}

    def score_states(cand_seqs):
        """NMP → per-seq state, DEDUPED (identical seqs scored once); uses the incremental
        splice vs the cached seed when enabled."""
        uniq = list(dict.fromkeys(cand_seqs))
        if not uniq:
            return {}
        rows_list = (_incremental_nmp_rows(protein_id, seed_seq, seed_rows, uniq, nmp_fn,
                                           margin=nmp_context_margin)
                     if incremental_nmp else nmp_fn(protein_id, uniq))
        return {s: _rows_to_state(s, r) for s, r in zip(uniq, rows_list)}

    seed_rows = nmp_fn(protein_id, [seed_seq])[0]          # full seed baseline for the splice
    seed = _rows_to_state(seed_seq, seed_rows)
    seed_metrics = struct_fn(protein_id, seed_seq)
    seed["structure"] = seed_metrics
    seed_count, seed_margin = seed["count"], seed["margin"]
    beam = [seed]
    best = {"seq": seed_seq, "count": seed_count, "structure": seed_metrics}
    shortlist, trace, best_key, stale = [], [], (seed_count, seed_margin), 0

    for rnd in range(max_rounds):
        if best["count"] == 0:
            break
        pool = [(st, c) for st in beam for c in propose_fn(st["seq"], st["cores"])]
        if not pool:
            break

        t0 = time.time()
        if topB is None:
            chosen_pairs = pool                            # NMP-all: the head ranking is discarded
        else:
            uniq_seqs = list(dict.fromkeys(c.seq for _, c in pool))
            head_rows = dict(zip(uniq_seqs, head_fn(protein_id, uniq_seqs)))
            tw_by_state = {}                               # hoist target windows: invariant per state
            proxy = np.empty(len(pool))
            for i, (st, c) in enumerate(pool):
                if id(st) not in tw_by_state:
                    tw_by_state[id(st)] = (target_window_idx_fn(st["cores"])
                                           if target_window_idx_fn else None)
                tw = tw_by_state[id(st)]
                row = np.asarray(head_rows[c.seq])
                proxy[i] = row[tw].max() if tw else row.max()
            chosen_pairs = [pool[int(k)] for k in np.argsort(proxy)[:topB]]
        t_head = time.time() - t0

        t0 = time.time()
        states_by_seq = score_states([c.seq for _, c in chosen_pairs])
        t_nmp = time.time() - t0

        # Split chosen into count-droppers (potential outputs -> refold, capped) and
        # margin-progress states (enter the beam with no refold).
        beam_admits, droppers = [], []
        for st, c in chosen_pairs:
            state = dict(states_by_seq[c.seq])             # copy: per-parent 'structure' must not alias
            improving = (state["count"] < st["count"]
                         or (state["count"] == st["count"] and state["margin"] < st["margin"]))
            n_mut = sum(a != b for a, b in zip(c.seq, seed_seq))
            if not improving or n_mut > max_path_mutations:
                continue                                    # no help / too many edits -> drop
            if state["count"] < seed_count:
                droppers.append((state, c, n_mut))          # potential OUTPUT -> refold below (capped)
            else:
                beam_admits.append(state)                   # margin-progress only -> beam, NO refold

        # Refold only the top-`refold_cap` count-droppers (biggest drop, then fewest edits).
        # Single mutations kill cores readily, so most candidates drop the count -> unbounded
        # refold explodes (2000+/round on multi-core seeds); the cap bounds ESMFold cost AND
        # yields the lean ranked shortlist the experiment wants (PLAN_RF_REFINE §7 cost fix).
        droppers.sort(key=lambda x: (x[0]["count"], x[2]))
        if refold_cap is not None:
            droppers = droppers[:refold_cap]
        n_refold, t_refold = 0, 0.0
        for state, c, _ in droppers:
            tr = time.time()
            metrics = struct_fn(protein_id, c.seq); n_refold += 1
            passed, reason = structure_gate(seed_metrics, metrics, scTM_eps=scTM_eps,
                                            scTM_min=scTM_min,
                                            cat_max_scRMSD_max=cat_max_scRMSD_max,
                                            predicted_active_site_min_pLDDT_min=(
                                                predicted_active_site_min_pLDDT_min
                                            ),
                                            scRMSD_max=scRMSD_max,
                                            active_site_RMSD_max=active_site_RMSD_max,
                                            max_anchor_sidechain_RMSD_max=(
                                                max_anchor_sidechain_RMSD_max
                                            ))
            t_refold += time.time() - tr
            metrics = replace(metrics, passed=passed, reason=reason)
            if not (passed or allow_structure_unknown):
                continue                                    # structure-failed output -> no beam
            state["structure"] = metrics
            beam_admits.append(state)
            shortlist.append({"seq": c.seq, "core_count": state["count"], "scTM": metrics.scTM,
                              "pLDDT": metrics.pLDDT, "scRMSD": metrics.scRMSD,
                              "active_site_RMSD": metrics.active_site_RMSD,
                              "global_ca_RMSD": metrics.global_ca_RMSD,
                              "active_site_sidechain_RMSD": metrics.active_site_sidechain_RMSD,
                              "max_anchor_sidechain_RMSD": metrics.max_anchor_sidechain_RMSD,
                              "max_anchor_atom_distance": metrics.max_anchor_atom_distance,
                              "active_site_complete": metrics.active_site_complete,
                              "active_site_min_pLDDT": metrics.active_site_min_pLDDT,
                              "cat_max_scRMSD": metrics.cat_max_scRMSD,
                              "predicted_active_site_min_pLDDT": (
                                  metrics.predicted_active_site_min_pLDDT
                              ),
                              "muts": c.desc})
            if state["count"] < best["count"]:
                best = {"seq": c.seq, "count": state["count"], "structure": metrics}
        beam = sorted(beam + beam_admits, key=lambda s: (s["count"], s["margin"]))[:beam_width]
        key = (beam[0]["count"], beam[0]["margin"])
        trace.append({"round": rnd, "n_candidates": len(pool), "n_nmp": len(states_by_seq),
                      "n_refold": n_refold, "best_count": best["count"],
                      "beam_best_count": beam[0]["count"], "beam_best_margin": beam[0]["margin"],
                      "t_head": round(t_head, 2), "t_nmp": round(t_nmp, 2),
                      "t_refold": round(t_refold, 2)})
        if log_fn is not None:
            log_fn(f"[refine] {protein_id} r{rnd}: pool={len(pool)} nmp={len(states_by_seq)} "
                   f"refold={n_refold} best={best['count']} beam={beam[0]['count']}/"
                   f"{beam[0]['margin']:.3f} t_nmp={t_nmp:.1f}s t_head={t_head:.1f}s "
                   f"t_refold={t_refold:.1f}s")
        if key < best_key:
            best_key, stale = key, 0
        else:
            stale += 1
            if stale >= patience:
                break

    return RefineResult(protein_id, seed_seq, seed_count, best["seq"], best["count"],
                        best["structure"], len(shortlist), best["count"] == seed_count,
                        shortlist, trace)
