"""Immutable Fusion state objects with fail-fast validation (PLAN §1.1).

Pure Python: no torch, no I/O, no model calls. Every inherited ``ParticleState`` is a
complete canonical AA20 sequence carrying a definitive structure verdict and a Head value;
IDs are content-derived (process-independent — never Python ``hash()``, which is salted and
unstable across processes); a ``PopulationState`` has normalized weights; the ``EliteState``
holds a feasible best-so-far.

The structure result is duck-typed (``.scTM``, selected active-site fields, ``.passed`` ...); the
concrete container is ``reference_flow.refine.StructureMetrics`` (imported only under
TYPE_CHECKING so this module stays torch-free).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from inverse_folding.reference_flow.refine import StructureMetrics

CANONICAL_AA20 = "ACDEFGHIKLMNPQRSTVWY"
_AA20 = frozenset(CANONICAL_AA20)
MOVE_FAMILIES = frozenset({"explicit_edit", "rf_reopen", "edit_repair"})
_WEIGHT_SUM_TOL = 1e-6


class FusionStateError(ValueError):
    """Raised when a Fusion state object violates its §1.1 invariant."""


# --------------------------------------------------------------------------- #
# content-derived identity (process-independent; never Python hash())
# --------------------------------------------------------------------------- #
def sequence_md5(sequence: str) -> str:
    return hashlib.md5(sequence.encode("ascii")).hexdigest()


def make_particle_id(protein_id: str, round_idx: int, slot_idx: int,
                     sequence_md5_hex: str) -> str:
    return f"{protein_id}:r{int(round_idx)}:s{int(slot_idx)}:{sequence_md5_hex[:12]}"


def make_proposal_id(parent_particle_id: str, move_family: str,
                     sequence_md5_hex: str) -> str:
    return f"{parent_particle_id}:{move_family}:{sequence_md5_hex[:12]}"


def make_elite_id(protein_id: str, round_idx: int, sequence_md5_hex: str) -> str:
    """Distinct id scheme for the elite ARCHIVE particle — never collides with a population
    ``make_particle_id`` (which prevents a beam elite-copy from becoming its own parent)."""
    return f"{protein_id}:elite:r{int(round_idx)}:{sequence_md5_hex[:12]}"


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _require_canonical(sequence: str, field: str = "sequence") -> None:
    if not isinstance(sequence, str) or not sequence:
        raise FusionStateError(f"{field} must be a non-empty string")
    bad = sorted(set(sequence) - _AA20)
    if bad:
        raise FusionStateError(f"{field} has non-canonical residue(s) {bad}; only AA20 allowed")


def _require_finite(value: float, field: str) -> None:
    if value is None or not math.isfinite(float(value)):
        raise FusionStateError(f"{field} must be a finite number, got {value!r}")


# --------------------------------------------------------------------------- #
# ParticleState — an inherited complete state (§1.1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParticleState:
    particle_id: str
    protein_id: str
    round_idx: int
    slot_idx: int
    sequence: str
    sequence_md5: str
    weight: float
    parent_particle_id: str | None
    source_proposal_id: str | None
    head_global_risk: float | None
    structure: Any
    feasible: bool
    lineage_seed: int
    #: The entry-stage facade row this particle's round-0 ancestor came from (V1 only).
    #:
    #: PLAN_RF_REFINE_FUSION_V1 §2.11:531 -- "Do not reconstruct the mapping from sequence alone.
    #: Distinct roots may converge to the same complete sequence." Without this the only edge from a
    #: round-0 particle back to its root is ``sequence_md5``, a DERIVED key: it happens to be unique
    #: because the handoff refuses duplicate sequences, but the PLAN requires the mapping to be
    #: PERSISTED, not re-derived. ``None`` for a standalone v0 run, whose generated parquet carries
    #: no entry lineage at all.
    entry_source_id: str | None = None

    def __post_init__(self) -> None:
        _require_canonical(self.sequence)
        if self.sequence_md5 != sequence_md5(self.sequence):
            raise FusionStateError("sequence_md5 does not match sequence")
        if self.round_idx < 0 or self.slot_idx < 0:
            raise FusionStateError("round_idx and slot_idx must be non-negative")
        if not self.particle_id or not self.protein_id:
            raise FusionStateError("particle_id and protein_id must be non-empty")
        # an inherited state has a definitive structure verdict + Head value (no missing eval)
        if self.structure is None:
            raise FusionStateError("inherited ParticleState requires a definitive structure verdict")
        if self.head_global_risk is None:
            raise FusionStateError("inherited ParticleState requires a Head global_risk value")
        _require_finite(self.head_global_risk, "head_global_risk")
        _require_finite(self.weight, "weight")
        if self.weight < 0.0:
            raise FusionStateError("weight must be non-negative")


# --------------------------------------------------------------------------- #
# Proposal — a move output whose identity survives evaluation (§1.1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    parent_particle_id: str
    move_family: str
    sequence: str
    edited_positions: tuple[int, ...]
    target_start_0b: int
    target_end_0b: int
    halo_start_0b: int
    halo_end_0b: int
    proposal_seed: int
    diagnostics: "Mapping[str, Any] | None" = None

    def __post_init__(self) -> None:
        if self.move_family not in MOVE_FAMILIES:
            raise FusionStateError(
                f"move_family {self.move_family!r} not in {sorted(MOVE_FAMILIES)}")
        _require_canonical(self.sequence)
        n = len(self.sequence)
        for lo, hi, name in ((self.target_start_0b, self.target_end_0b, "target"),
                             (self.halo_start_0b, self.halo_end_0b, "halo")):
            if not (0 <= lo <= hi <= n):
                raise FusionStateError(f"{name} span [{lo},{hi}) invalid for length {n}")
        # halo must contain the target (repair regenerates halo context around the target)
        if not (self.halo_start_0b <= self.target_start_0b
                and self.target_end_0b <= self.halo_end_0b):
            raise FusionStateError("halo must contain the target register")
        for pos in self.edited_positions:
            if not (self.target_start_0b <= pos < self.target_end_0b):
                raise FusionStateError(
                    f"edited position {pos} outside target register "
                    f"[{self.target_start_0b},{self.target_end_0b})")


# --------------------------------------------------------------------------- #
# CandidateEvaluation — the exact evaluated sequence that may be inherited (§1.1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CandidateEvaluation:
    sequence_md5: str
    parent_particle_id: str
    proposal_id: str
    target_start_0b: int
    target_end_0b: int
    head_global_risk: float
    aligned_windows: tuple
    local_target_delta: float
    new_hotspot_max: float
    new_hotspot_mass: float
    new_hotspot_count: int
    structure: Any | None
    structure_evaluated: bool
    feasible: bool
    reason: str = ""

    def __post_init__(self) -> None:
        _require_finite(self.head_global_risk, "head_global_risk")
        if self.structure_evaluated and self.structure is None:
            raise FusionStateError("structure_evaluated=True requires a structure result")
        if self.feasible and not self.structure_evaluated:
            raise FusionStateError("a feasible candidate must have been structure-evaluated")


# --------------------------------------------------------------------------- #
# EliteState — feasible best-so-far, monotone (§1.1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EliteState:
    particle: ParticleState
    first_round_seen: int
    last_round_seen: int

    def __post_init__(self) -> None:
        if not self.particle.feasible:
            raise FusionStateError("elite particle must be feasible")
        if self.first_round_seen > self.last_round_seen:
            raise FusionStateError("elite first_round_seen must be <= last_round_seen")


# --------------------------------------------------------------------------- #
# PopulationState — exactly N slots with normalized weights (§1.1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PopulationState:
    round_idx: int
    particles: tuple[ParticleState, ...]
    elite: EliteState
    #: ADDITIVE AUDIT SEAM (PLAN_RF_REFINE_FUSION_V1 §2.11: "The v0 admission path may receive an
    #: additive audit seam, but its decisions remain..."). One dict per facade row the round-0 scan
    #: examined, admitted or rejected, with the reason. Without it a row v0 drops leaves no trace
    #: anywhere -- the ENTRY stage cannot cover it, because its own gate defers and the definitive
    #: verdict exists only here. Empty for a population not built by ``build_initial_population``.
    admission_attempts: tuple = ()
    #: Definitive structure evaluations actually executed during round-0 admission (cache misses).
    #: The entry ledger books 0 structure requests for a deferred attempt precisely because v0
    #: charges it; v0 counts refolds only from round 1, so without this the round-0 spend is
    #: charged in no artifact at all.
    initial_refolds: int = 0

    def __post_init__(self) -> None:
        if not self.particles:
            raise FusionStateError("population must have at least one particle")
        object.__setattr__(self, "particles", tuple(self.particles))
        proteins = {p.protein_id for p in self.particles}
        if len(proteins) != 1:
            raise FusionStateError(f"population spans multiple proteins: {sorted(proteins)}")
        for p in self.particles:
            if p.round_idx != self.round_idx:
                raise FusionStateError(
                    f"particle round_idx {p.round_idx} != population round_idx {self.round_idx}")
        total = math.fsum(p.weight for p in self.particles)
        if abs(total - 1.0) > _WEIGHT_SUM_TOL:
            raise FusionStateError(f"population weights must sum to 1, got {total}")
        ids = [p.particle_id for p in self.particles]
        if len(set(ids)) != len(ids):
            raise FusionStateError("population has duplicate particle_id")
        if sorted(p.slot_idx for p in self.particles) != list(range(len(self.particles))):
            raise FusionStateError("population slot_idx must be exactly 0..N-1")


def uniform_population(*, round_idx: int, particles: "Sequence[ParticleState]",
                      elite: EliteState) -> PopulationState:
    """Build a population with weights reset to ``1/N`` (post-resample / control reset)."""
    particles = list(particles)
    if not particles:
        raise FusionStateError("uniform_population requires at least one particle")
    w = 1.0 / len(particles)
    reset = tuple(replace(p, weight=w, round_idx=round_idx) for p in particles)
    return PopulationState(round_idx=round_idx, particles=reset, elite=elite)


# --- Public aliases for fusion_v2 reuse (additive; zero behavior change) ---------------------
# require_canonical_aa20 is the AA20 firewall PLAN section 2.7 relies on; a private copy would be a
# second place a masked or non-canonical residue can reach the Head.
require_canonical_aa20 = _require_canonical
require_finite = _require_finite
