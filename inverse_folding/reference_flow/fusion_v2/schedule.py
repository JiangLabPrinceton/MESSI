"""V2 cycle coordinates, depth schedule, calibrated maturity band, and mask-load contract.

Name collision, deliberate: ``inverse_folding/reference_flow/schedule.py`` already exists and owns
the DFM base kappa schedule (``base_schedule_value``, ``positionwise_unmask_probabilities``). This
module is a different concept -- feedback depth/step geometry -- and keeps the PLAN's own vocabulary
("coordinate law and ordered ``(r_d, c_d, c_{d+1})`` schedule"). The two must never be merged, this
module must never import that one, and consumers must import the fully qualified
``fusion_v2.schedule`` path.

This module decides no science. Every number arrives from a frozen calibration artifact or the run
config; every illegal coordinate or maturity combination fails closed before any sampler or
projection code runs. In particular it never estimates a quantile, never picks a tolerance, and
never infers maturity from a coordinate label.

The typed JSON loader is deliberately the only I/O boundary in this module. It recomputes the
artifact's canonical digest before returning a table; callers cannot turn an arbitrary mapping or
non-empty label into trusted calibration evidence.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
import re
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import V2Error

__all__ = [
    "SCHEDULE_BAND_SCHEMA_VERSION", "SCHEDULE_BAND_SCOPE", "SCHEDULE_BAND_SEED_SCHEMA",
    "V2ScheduleError", "MissingScheduleBandError", "EmptyBandIntersectionError",
    "CoordinateLaw", "HistoryKey", "CycleCoordinates", "DepthSchedule",
    "QuantileLevels", "BandInterval", "ScheduleBand", "BandProvenance", "ScheduleBandTable",
    "MaturityRecord", "BandVerdictReason", "BandVerdict", "UnresolvedInterval",
    "AdmissibleMaskLoad", "BandCenterRule", "BandCenterTarget",
    "history_key", "make_cycle", "make_depth_schedule", "make_band", "make_band_table",
    "bind_band_table", "band_table_content_digest", "band_table_payload", "load_band_table",
    "lookup_band", "observe_maturity", "gate_projected_maturity", "pinned_unresolved_interval",
    "admissible_reopen_cardinality", "validate_realized_mask_load",
    "band_center_target", "required_reopen_count",
    "segment_cost", "screening_cost",
]


SCHEDULE_BAND_SCHEMA_VERSION = "rf_fusion_v2_schedule_band/1"
SCHEDULE_BAND_SCOPE = "step_indexed_maturity_band"
SCHEDULE_BAND_SEED_SCHEMA = "rho-maturity-scan-2"
_SCHEDULE_BAND_PRODUCER = "scripts/rho_maturity_scan.py --mode step"
_CONTENT_DIGEST_RE = re.compile(r"(?:[a-z][a-z0-9_-]*-)?[0-9a-f]{32,64}")
_CODE_REVISION_RE = re.compile(r"[0-9a-f]{7,64}")
_PLACEHOLDER_TEXT = frozenset({"unknown", "unset", "none", "null", "placeholder", "na", "n/a"})


class V2ScheduleError(V2Error):
    """A coordinate, schedule, band, or mask-load contract was violated."""


class MissingScheduleBandError(V2ScheduleError):
    """No band was calibrated for this ``(step, stratum_key)``. There is no interpolation."""


class EmptyBandIntersectionError(V2ScheduleError):
    """The normalized and absolute maturity intervals do not overlap."""


class CoordinateLaw(str, enum.Enum):
    """Which ordering law a cycle asserts (PLAN §2.1). Values are the literal config strings."""

    PROGRESSIVE_CHECKPOINT = "progressive_checkpoint"
    STATIONARY_CHECKPOINT = "stationary_checkpoint"


def _require_index(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise V2ScheduleError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise V2ScheduleError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_text(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise V2ScheduleError(f"{name} must be a non-empty str, got {value!r}")
    return value


def _require_content_digest(value: object, name: str) -> str:
    text = _require_text(value, name).strip().lower()
    if text in _PLACEHOLDER_TEXT or text.startswith(("unset-", "unknown-", "placeholder-")):
        raise V2ScheduleError(f"{name} may not be a placeholder, got {value!r}")
    if _CONTENT_DIGEST_RE.fullmatch(text) is None:
        raise V2ScheduleError(
            f"{name} must be a content digest (optional typed prefix plus 32-64 lowercase hex), "
            f"got {value!r}"
        )
    hex_part = text.rsplit("-", 1)[-1]
    if len(set(hex_part)) == 1:
        raise V2ScheduleError(f"{name} may not be a repeated-character placeholder")
    return text


def _require_code_revision(value: object) -> str:
    text = _require_text(value, "code_revision").strip().lower()
    if text in _PLACEHOLDER_TEXT or text.startswith(("unset", "unknown", "placeholder")):
        raise V2ScheduleError(f"code_revision may not be a placeholder, got {value!r}")
    if _CODE_REVISION_RE.fullmatch(text) is None:
        raise V2ScheduleError(
            "code_revision must be an explicit 7-64 character lowercase hexadecimal revision"
        )
    return text


def _require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V2ScheduleError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise V2ScheduleError(f"{name} must be finite, got {out!r}")
    return out


# --------------------------------------------------------------------------------------------
# Coordinates
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class HistoryKey:
    """``(depth, step)`` -- the only legal key for any per-event history or lineage record.

    Equality requires both components, so a stationary schedule that revisits sampler step ``s`` at
    depth 1 and depth 2 produces two distinct keys and cannot alias two events (PLAN §2.1, §2.4).
    """

    depth: int
    step: int

    def __post_init__(self) -> None:
        _require_index(self.depth, "depth")
        _require_index(self.step, "step")


def history_key(depth: int, step: int) -> HistoryKey:
    """The single constructor for the joint history key."""
    return HistoryKey(depth=depth, step=step)


@dataclass(frozen=True)
class CycleCoordinates:
    """One feedback cycle's geometry.

    Direct construction and :func:`make_cycle` obey the same validation law. The dataclass remains
    a public typed API, but it is no longer a bypass around the factory.
    """

    depth: int
    r_step: int
    c_source_step: int
    c_next_step: int
    n_steps: int
    law: CoordinateLaw

    def __post_init__(self) -> None:
        _validate_cycle_coordinates(
            depth=self.depth,
            r_step=self.r_step,
            c_source_step=self.c_source_step,
            c_next_step=self.c_next_step,
            n_steps=self.n_steps,
            law=self.law,
        )

    @property
    def segment_lane_dfe(self) -> int:
        """``c_{d+1} - r_d`` -- the only cost this propagation segment may charge (PLAN §3.4)."""
        return self.c_next_step - self.r_step

    @property
    def lookahead_tail(self) -> int:
        """``S - c_d`` -- the per-lookahead tail in ``C_screen = c + K(S - c)``."""
        return self.n_steps - self.c_source_step

    @property
    def source_key(self) -> HistoryKey:
        return history_key(self.depth, self.c_source_step)

    @property
    def reentry_key(self) -> HistoryKey:
        """Where feedback-injected identities are committed: depth advances, time rolls back."""
        return history_key(self.depth + 1, self.r_step)

    @property
    def commit_key(self) -> HistoryKey:
        return history_key(self.depth + 1, self.c_next_step)


def _validate_cycle_coordinates(
    *, depth: int, r_step: int, c_source_step: int, c_next_step: int, n_steps: int,
    law: CoordinateLaw,
) -> None:
    """Validate one cycle under the PLAN §2.1 ordering laws.

    The full rule is ``0 <= r_step < c_source_step <= c_next_step < n_steps``. The upper bound is
    strict: checkpoints are top-of-step pre-denoiser states, the sampler refuses ``at_step`` outside
    ``[start_step, n_steps)``, and a state at ``n_steps`` necessarily has no unresolved editable
    position, so it is a terminal endpoint rather than a live partial state.
    """
    _require_index(depth, "depth")
    _require_index(r_step, "r_step")
    _require_index(c_source_step, "c_source_step")
    _require_index(c_next_step, "c_next_step")
    _require_index(n_steps, "n_steps", minimum=1)
    if not isinstance(law, CoordinateLaw):
        raise V2ScheduleError(f"law must be a CoordinateLaw, got {law!r}")
    if r_step >= c_source_step:
        raise V2ScheduleError(
            f"re-entry must precede the source checkpoint: r_step={r_step} >= "
            f"c_source_step={c_source_step} (PLAN §2.1)"
        )
    if c_next_step < c_source_step:
        raise V2ScheduleError(
            f"c_next_step={c_next_step} < c_source_step={c_source_step}; the committed checkpoint "
            "never moves backwards (PLAN §2.1)"
        )
    for name, value in (("c_source_step", c_source_step), ("c_next_step", c_next_step)):
        if value >= n_steps:
            raise V2ScheduleError(
                f"{name}={value} must be < n_steps={n_steps}; a capture at n_steps is not a live "
                "partial state"
            )
    if law is CoordinateLaw.PROGRESSIVE_CHECKPOINT and c_next_step == c_source_step:
        raise V2ScheduleError(
            "progressive_checkpoint requires c_next_step > c_source_step; declare "
            "stationary_checkpoint to reuse a sampler step (PLAN §2.1)"
        )
    if law is CoordinateLaw.STATIONARY_CHECKPOINT and c_next_step != c_source_step:
        raise V2ScheduleError(
            f"stationary_checkpoint requires c_next_step == c_source_step, got "
            f"{c_next_step} != {c_source_step}"
        )


def make_cycle(
    *, depth: int, r_step: int, c_source_step: int, c_next_step: int, n_steps: int,
    law: CoordinateLaw,
) -> CycleCoordinates:
    """Construct one validated cycle under the PLAN §2.1 ordering laws."""
    return CycleCoordinates(
        depth=depth, r_step=r_step, c_source_step=c_source_step, c_next_step=c_next_step,
        n_steps=n_steps, law=law,
    )


@dataclass(frozen=True)
class DepthSchedule:
    """An ordered, contiguous, chained sequence of cycles sharing one ``S`` and one law."""

    schedule_id: str
    law: CoordinateLaw
    n_steps: int
    cycles: tuple[CycleCoordinates, ...]
    depth_cap: int


def make_depth_schedule(
    *, schedule_id: str, law: CoordinateLaw, n_steps: int, depth_cap: int,
    cycles: Sequence[CycleCoordinates],
) -> DepthSchedule:
    """Validate depth contiguity, chaining, one shared ``S``, one shared law, and ``depth_cap``.

    Chaining matters for more than tidiness: without ``cycles[d].c_next_step ==
    cycles[d+1].c_source_step`` the schedule silently drops a propagation segment and the ledger's
    ``c_next - r`` accounting stops summing to the real trajectory (PLAN §2.1, §3.4).
    """
    _require_text(schedule_id, "schedule_id")
    _require_index(n_steps, "n_steps", minimum=1)
    _require_index(depth_cap, "depth_cap", minimum=1)
    if not isinstance(law, CoordinateLaw):
        raise V2ScheduleError(f"law must be a CoordinateLaw, got {law!r}")
    ordered = tuple(cycles)
    if not ordered:
        raise V2ScheduleError("a depth schedule needs at least one cycle")
    if depth_cap != len(ordered):
        raise V2ScheduleError(
            f"depth_cap={depth_cap} disagrees with {len(ordered)} declared cycles"
        )
    for index, cycle in enumerate(ordered):
        if not isinstance(cycle, CycleCoordinates):
            raise V2ScheduleError(f"cycles[{index}] must be a CycleCoordinates")
        if cycle.depth != index:
            raise V2ScheduleError(
                f"cycle depths must be exactly 0..{len(ordered) - 1} in order; cycles[{index}] "
                f"declares depth {cycle.depth}"
            )
        if cycle.n_steps != n_steps:
            raise V2ScheduleError(
                f"cycles[{index}].n_steps={cycle.n_steps} != schedule n_steps={n_steps}; S is "
                "locked across a resume, so a per-depth S is unrepresentable"
            )
        if cycle.law is not law:
            raise V2ScheduleError(
                f"cycles[{index}] declares {cycle.law.value!r} but the schedule declares "
                f"{law.value!r}; one law per schedule"
            )
    for index in range(len(ordered) - 1):
        if ordered[index].c_next_step != ordered[index + 1].c_source_step:
            raise V2ScheduleError(
                f"schedule is not chained: cycles[{index}].c_next_step="
                f"{ordered[index].c_next_step} != cycles[{index + 1}].c_source_step="
                f"{ordered[index + 1].c_source_step}"
            )
    return DepthSchedule(
        schedule_id=schedule_id, law=law, n_steps=n_steps, cycles=ordered, depth_cap=depth_cap,
    )


def segment_cost(cycle: CycleCoordinates) -> int:
    """``c_{d+1} - r_d`` logical lane-DFE (PLAN §3.4)."""
    return cycle.segment_lane_dfe


def screening_cost(*, c_source_step: int, k_lookaheads: int, n_steps: int) -> int:
    """``C_screen = c + K(S - c)`` -- the prefix is paid once and the K tails are paid each."""
    _require_index(c_source_step, "c_source_step")
    _require_index(k_lookaheads, "k_lookaheads", minimum=1)
    _require_index(n_steps, "n_steps", minimum=1)
    if c_source_step >= n_steps:
        raise V2ScheduleError(f"c_source_step={c_source_step} must be < n_steps={n_steps}")
    return c_source_step + k_lookaheads * (n_steps - c_source_step)


# --------------------------------------------------------------------------------------------
# Realized maturity
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MaturityRecord:
    """Torch-free mirror of ``sampler.MaturityTelemetry``'s scalar fields.

    Field names are reused verbatim so a scan artifact and a V2 state row are directly comparable
    and the two definitions of ``rho_edit`` can never drift. Built only from measured counts.
    """

    length_total: int
    n_fixed: int
    n_editable: int
    n_resolved_editable: int
    n_unresolved_editable: int
    rho_edit: float
    rho_known_sequence_identity: float


def observe_maturity(
    *, length_total: int, n_fixed: int, n_editable: int, n_unresolved_editable: int,
) -> MaturityRecord:
    """Build a maturity record from measured byte-derived counts, re-deriving every ratio.

    PLAN §2.1: realized editable maturity is measured from bytes, never inferred from a coordinate
    label. There is deliberately no code path here that takes ``r`` as an input.
    """
    _require_index(length_total, "length_total", minimum=1)
    _require_index(n_fixed, "n_fixed")
    _require_index(n_editable, "n_editable", minimum=1)
    _require_index(n_unresolved_editable, "n_unresolved_editable")
    if n_fixed + n_editable != length_total:
        raise V2ScheduleError(
            f"n_fixed + n_editable ({n_fixed} + {n_editable}) != length_total ({length_total})"
        )
    if n_unresolved_editable > n_editable:
        raise V2ScheduleError(
            f"n_unresolved_editable={n_unresolved_editable} exceeds n_editable={n_editable}"
        )
    n_resolved = n_editable - n_unresolved_editable
    return MaturityRecord(
        length_total=length_total,
        n_fixed=n_fixed,
        n_editable=n_editable,
        n_resolved_editable=n_resolved,
        n_unresolved_editable=n_unresolved_editable,
        rho_edit=n_resolved / n_editable,
        rho_known_sequence_identity=(n_resolved + n_fixed) / length_total,
    )


# --------------------------------------------------------------------------------------------
# The calibrated band B(r)
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantileLevels:
    """Strictly increasing quantile levels in ``(0, 1)``. No default: PLAN §3.5 defers the choice."""

    levels: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.levels:
            raise V2ScheduleError("quantile levels must be non-empty")
        previous = None
        for level in self.levels:
            value = _require_finite(level, "quantile level")
            if not 0.0 < value < 1.0:
                raise V2ScheduleError(f"quantile level must be in (0, 1), got {value}")
            if previous is not None and value <= previous:
                raise V2ScheduleError("quantile levels must be strictly increasing")
            previous = value


@dataclass(frozen=True)
class BandInterval:
    """A frozen accept interval and the quantile levels that produced its endpoints."""

    lo: float
    hi: float
    lo_level: float
    hi_level: float

    def __post_init__(self) -> None:
        lo = _require_finite(self.lo, "lo")
        hi = _require_finite(self.hi, "hi")
        _require_finite(self.lo_level, "lo_level")
        _require_finite(self.hi_level, "hi_level")
        if lo > hi:
            raise V2ScheduleError(f"band interval lo={lo} exceeds hi={hi}")

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi


@dataclass(frozen=True)
class BandProvenance:
    """Content provenance of a whole calibration run (PLAN §3.5, §5.2)."""

    schema_version: str
    calibration_scope: str
    calibration_id: str
    calibration_content_digest: str
    sampler_config_digest: str
    tokenizer_digest: str
    backbone_digest: str
    coordinate_mask_policy_digest: str
    constraint_manifest_digest: str
    constraint_stratum_id: str
    cohort_digest: str
    raw_attempts_digest: str
    attempted_seed_digest: str
    seed_schema: str
    code_revision: str
    produced_by: str
    n_steps: int
    base_form: str
    amplification_form: str
    remask_fraction_scale: float
    head_free: bool
    n_attempted_seeds: int
    n_failed_captures: int

    def __post_init__(self) -> None:
        for name in (
            "schema_version", "calibration_scope", "calibration_id", "constraint_stratum_id",
            "seed_schema", "code_revision", "produced_by", "base_form", "amplification_form",
        ):
            _require_text(getattr(self, name), name)
        for name in (
            "calibration_content_digest", "sampler_config_digest", "tokenizer_digest",
            "backbone_digest", "coordinate_mask_policy_digest", "constraint_manifest_digest",
            "cohort_digest", "raw_attempts_digest", "attempted_seed_digest",
        ):
            _require_content_digest(getattr(self, name), name)
        if self.schema_version != SCHEDULE_BAND_SCHEMA_VERSION:
            raise V2ScheduleError(
                f"schema_version {self.schema_version!r} != {SCHEDULE_BAND_SCHEMA_VERSION!r}"
            )
        if self.calibration_scope != SCHEDULE_BAND_SCOPE:
            raise V2ScheduleError(
                f"calibration_scope must be {SCHEDULE_BAND_SCOPE!r}, got "
                f"{self.calibration_scope!r}"
            )
        if self.seed_schema != SCHEDULE_BAND_SEED_SCHEMA:
            raise V2ScheduleError(
                f"seed_schema must be {SCHEDULE_BAND_SEED_SCHEMA!r}, got {self.seed_schema!r}"
            )
        _require_code_revision(self.code_revision)
        if self.produced_by != _SCHEDULE_BAND_PRODUCER:
            raise V2ScheduleError(
                f"produced_by must be {_SCHEDULE_BAND_PRODUCER!r}, got {self.produced_by!r}"
            )
        if self.base_form != "linear":
            raise V2ScheduleError(
                f"base_form must be 'linear' for the frozen V2 substrate, got {self.base_form!r}"
            )
        if self.amplification_form != "constant_one":
            raise V2ScheduleError(
                "amplification_form must be 'constant_one' for the frozen V2 substrate, got "
                f"{self.amplification_form!r}"
            )
        if self.constraint_stratum_id not in {"unconstrained", "anchored", "mixed"}:
            raise V2ScheduleError(
                "constraint_stratum_id must be derived as 'unconstrained', 'anchored', or "
                f"'mixed', got {self.constraint_stratum_id!r}"
            )
        _require_index(self.n_steps, "n_steps", minimum=1)
        _require_index(self.n_attempted_seeds, "n_attempted_seeds", minimum=1)
        _require_index(self.n_failed_captures, "n_failed_captures")
        if self.n_failed_captures > self.n_attempted_seeds:
            raise V2ScheduleError(
                f"n_failed_captures={self.n_failed_captures} exceeds "
                f"n_attempted_seeds={self.n_attempted_seeds}"
            )
        if not isinstance(self.head_free, bool):
            raise V2ScheduleError("head_free must be an explicit bool")
        if not self.head_free:
            raise V2ScheduleError(
                "the schedule-band calibration must be Head-free; an outcome-dependent band would "
                "make coordinate choice a post-hoc decision (PLAN §3.5)"
            )
        scale = _require_finite(self.remask_fraction_scale, "remask_fraction_scale")
        if scale != 0.0:
            raise V2ScheduleError(
                f"remask_fraction_scale must be 0.0 for a V2/A2 band, got {scale}; the remask-on "
                "c1_null crossings may not be reused to select V2 coordinates (PLAN §2.1)"
            )


@dataclass(frozen=True)
class ScheduleBand:
    """The realized ``B(r)`` for one step within one cohort stratum.

    Self-validating.  ``make_band`` is not the only door into this type -- it is exported, it sits
    inside every band table, and a deserializer would construct it directly.  Keeping the invariants
    in a factory would mean the TYPE guarantees nothing: a band declaring zero attempts would claim
    a calibration that never ran, and every downstream gate would trust it because it type-checks.
    """

    step: int
    stratum_key: str
    levels: QuantileLevels
    rho_quantiles: tuple[float, ...]
    unresolved_quantiles: tuple[int, ...]
    rho_accept: BandInterval
    unresolved_accept: BandInterval
    combination_rule: str
    n_attempts: int
    n_captured: int
    n_editable_min: int
    n_editable_max: int

    def __post_init__(self) -> None:
        _require_index(self.step, "step")
        _require_text(self.stratum_key, "stratum_key")
        _require_text(self.combination_rule, "combination_rule")
        _require_index(self.n_attempts, "n_attempts", minimum=1)
        _require_index(self.n_captured, "n_captured", minimum=1)
        _require_index(self.n_editable_min, "n_editable_min", minimum=1)
        _require_index(self.n_editable_max, "n_editable_max", minimum=1)
        if self.combination_rule != "both_axes":
            raise V2ScheduleError(
                f"combination_rule must be 'both_axes', got {self.combination_rule!r}; normalized "
                "maturity and absolute unresolved mass are jointly load-bearing (PLAN §3.5)"
            )
        if not isinstance(self.levels, QuantileLevels):
            raise V2ScheduleError("levels must be a QuantileLevels")
        for name in ("rho_accept", "unresolved_accept"):
            if not isinstance(getattr(self, name), BandInterval):
                raise V2ScheduleError(f"{name} must be a BandInterval")
        if self.n_captured > self.n_attempts:
            raise V2ScheduleError(
                f"n_captured={self.n_captured} exceeds n_attempts={self.n_attempts}")
        if self.n_editable_min > self.n_editable_max:
            raise V2ScheduleError("n_editable_min exceeds n_editable_max")

        rho = tuple(_require_finite(v, "rho quantile") for v in self.rho_quantiles)
        unresolved = tuple(
            _require_index(v, "unresolved quantile") for v in self.unresolved_quantiles)
        width = len(self.levels.levels)
        if len(rho) != width or len(unresolved) != width:
            raise V2ScheduleError(
                f"quantile vectors must have one entry per level ({width}); got "
                f"{len(rho)} rho and {len(unresolved)} unresolved"
            )
        for value in rho:
            if not 0.0 <= value <= 1.0:
                raise V2ScheduleError(f"rho quantile must be in [0, 1], got {value}")
        if any(b < a for a, b in zip(rho, rho[1:])):
            raise V2ScheduleError("rho quantiles must be non-decreasing across increasing levels")
        if any(b > a for a, b in zip(unresolved, unresolved[1:])):
            raise V2ScheduleError(
                "unresolved-mass quantiles must be non-increasing across increasing levels; mass "
                "falls as maturity rises"
            )
        for value in unresolved:
            if value > self.n_editable_max:
                raise V2ScheduleError(
                    f"unresolved quantile {value} exceeds n_editable_max={self.n_editable_max}"
                )
        # Normalize AFTER validating, so a list and a tuple of the same numbers are one band and the
        # canonical band payload cannot depend on which container the caller happened to pass.
        object.__setattr__(self, "rho_quantiles", rho)
        object.__setattr__(self, "unresolved_quantiles", unresolved)


def make_band(
    *, step: int, stratum_key: str, levels: QuantileLevels,
    rho_quantiles: Sequence[float], unresolved_quantiles: Sequence[int],
    rho_accept: BandInterval, unresolved_accept: BandInterval, combination_rule: str,
    n_attempts: int, n_captured: int, n_editable_min: int, n_editable_max: int,
) -> ScheduleBand:
    """Construct one band from already-computed quantiles. This never estimates a quantile.

    Every scientific field is a required argument: omitting one is a ``TypeError``, not a fallback
    (PLAN §5.1). ``stratum_key`` and ``combination_rule`` in particular have no library default --
    the stratification law and the two-axis combination rule are open decisions (PLAN §3.5).

    The validation lives in :class:`ScheduleBand` itself; this factory exists for the keyword-only
    call shape and adds nothing the type does not already guarantee.
    """
    return ScheduleBand(
        step=step, stratum_key=stratum_key, levels=levels, rho_quantiles=tuple(rho_quantiles),
        unresolved_quantiles=tuple(unresolved_quantiles), rho_accept=rho_accept,
        unresolved_accept=unresolved_accept, combination_rule=combination_rule,
        n_attempts=n_attempts, n_captured=n_captured, n_editable_min=n_editable_min,
        n_editable_max=n_editable_max,
    )


@dataclass(frozen=True)
class ScheduleBandTable:
    """The frozen calibration artifact: one provenance, many ``(step, stratum)`` bands."""

    provenance: BandProvenance
    bands: tuple[ScheduleBand, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.bands, tuple):
            raise V2ScheduleError("bands must be an immutable tuple")
        _validate_band_table(self.provenance, self.bands, verify_content_digest=True)


def _provenance_payload(
    provenance: BandProvenance, *, include_content_digest: bool,
) -> dict[str, Any]:
    payload = {field.name: getattr(provenance, field.name) for field in fields(BandProvenance)}
    if not include_content_digest:
        payload.pop("calibration_content_digest")
    return payload


def _interval_payload(interval: BandInterval) -> dict[str, float]:
    return {
        "lo": interval.lo,
        "hi": interval.hi,
        "lo_level": interval.lo_level,
        "hi_level": interval.hi_level,
    }


def _band_payload(band: ScheduleBand) -> dict[str, Any]:
    return {
        "step": band.step,
        "stratum_key": band.stratum_key,
        "levels": list(band.levels.levels),
        "rho_quantiles": list(band.rho_quantiles),
        "unresolved_quantiles": list(band.unresolved_quantiles),
        "rho_accept": _interval_payload(band.rho_accept),
        "unresolved_accept": _interval_payload(band.unresolved_accept),
        "combination_rule": band.combination_rule,
        "n_attempts": band.n_attempts,
        "n_captured": band.n_captured,
        "n_editable_min": band.n_editable_min,
        "n_editable_max": band.n_editable_max,
    }


def band_table_content_digest(
    *, provenance: BandProvenance, bands: Sequence[ScheduleBand],
) -> str:
    """Canonical SHA-256 over every artifact field except the digest itself.

    Excluding only ``calibration_content_digest`` avoids a circular self-hash while binding the
    schema, substrate/reference provenance, seed evidence, and all numeric bands.
    """
    payload = {
        "schema_version": SCHEDULE_BAND_SCHEMA_VERSION,
        "provenance": _provenance_payload(provenance, include_content_digest=False),
        "bands": [_band_payload(band) for band in bands],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_band_table(
    provenance: BandProvenance,
    bands: Sequence[ScheduleBand],
    *,
    verify_content_digest: bool,
) -> tuple[ScheduleBand, ...]:
    if not isinstance(provenance, BandProvenance):
        raise V2ScheduleError("provenance must be a BandProvenance")
    ordered = tuple(bands)
    if not ordered:
        raise V2ScheduleError("a band table needs at least one band")
    canonical_order = tuple(sorted(ordered, key=lambda band: (band.step, band.stratum_key)))
    if ordered != canonical_order:
        raise V2ScheduleError("bands must be ordered canonically by (step, stratum_key)")
    seen: set[tuple[int, str]] = set()
    for band in ordered:
        if not isinstance(band, ScheduleBand):
            raise V2ScheduleError("every band must be a ScheduleBand")
        if band.step >= provenance.n_steps:
            raise V2ScheduleError(
                f"band step {band.step} is outside the calibrated n_steps={provenance.n_steps}"
            )
        key = (band.step, band.stratum_key)
        if key in seen:
            raise V2ScheduleError(
                f"duplicate band for step={band.step} stratum={band.stratum_key!r}"
            )
        seen.add(key)
    if verify_content_digest:
        expected = band_table_content_digest(provenance=provenance, bands=ordered)
        if provenance.calibration_content_digest != expected:
            raise V2ScheduleError(
                "calibration_content_digest does not match the canonical typed payload: "
                f"declared={provenance.calibration_content_digest}, computed={expected}"
            )
    return ordered


def make_band_table(
    *, provenance: BandProvenance, bands: Sequence[ScheduleBand],
) -> ScheduleBandTable:
    """Canonicalize rows and bind their full typed content into provenance."""
    materialized = tuple(bands)
    if any(not isinstance(band, ScheduleBand) for band in materialized):
        raise V2ScheduleError("every band must be a ScheduleBand")
    ordered = tuple(sorted(materialized, key=lambda band: (band.step, band.stratum_key)))
    _validate_band_table(provenance, ordered, verify_content_digest=False)
    digest = band_table_content_digest(provenance=provenance, bands=ordered)
    bound = replace(provenance, calibration_content_digest=digest)
    return ScheduleBandTable(provenance=bound, bands=ordered)


def bind_band_table(
    *, provenance: BandProvenance, bands: Sequence[ScheduleBand],
) -> ScheduleBandTable:
    """Explicit spelling of :func:`make_band_table` for producers binding a new artifact."""
    return make_band_table(provenance=provenance, bands=bands)


def band_table_payload(table: ScheduleBandTable) -> dict[str, Any]:
    """Return the strict JSON-compatible artifact payload."""
    if not isinstance(table, ScheduleBandTable):
        raise V2ScheduleError("table must be a ScheduleBandTable")
    return {
        "schema_version": SCHEDULE_BAND_SCHEMA_VERSION,
        "provenance": _provenance_payload(table.provenance, include_content_digest=True),
        "bands": [_band_payload(band) for band in table.bands],
    }


def _require_exact_keys(mapping: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise V2ScheduleError(f"{path} keys mismatch: missing={missing}, unknown={unknown}")


def _load_interval(value: object, path: str) -> BandInterval:
    if not isinstance(value, Mapping):
        raise V2ScheduleError(f"{path} must be a mapping")
    expected = {field.name for field in fields(BandInterval)}
    _require_exact_keys(value, expected, path)
    try:
        return BandInterval(**value)
    except (TypeError, ValueError) as exc:
        raise V2ScheduleError(f"invalid {path}: {exc}") from exc


def _load_band(value: object, index: int) -> ScheduleBand:
    path = f"bands[{index}]"
    if not isinstance(value, Mapping):
        raise V2ScheduleError(f"{path} must be a mapping")
    expected = {field.name for field in fields(ScheduleBand)}
    _require_exact_keys(value, expected, path)
    try:
        return make_band(
            step=value["step"],
            stratum_key=value["stratum_key"],
            levels=QuantileLevels(tuple(value["levels"])),
            rho_quantiles=tuple(value["rho_quantiles"]),
            unresolved_quantiles=tuple(value["unresolved_quantiles"]),
            rho_accept=_load_interval(value["rho_accept"], f"{path}.rho_accept"),
            unresolved_accept=_load_interval(
                value["unresolved_accept"], f"{path}.unresolved_accept",
            ),
            combination_rule=value["combination_rule"],
            n_attempts=value["n_attempts"],
            n_captured=value["n_captured"],
            n_editable_min=value["n_editable_min"],
            n_editable_max=value["n_editable_max"],
        )
    except (TypeError, ValueError) as exc:
        raise V2ScheduleError(f"invalid {path}: {exc}") from exc


def load_band_table(
    path: str | Path, *, expected_content_digest: str | None = None,
) -> ScheduleBandTable:
    """Load, strictly type-check, and content-verify a schedule-band artifact."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V2ScheduleError(f"cannot load schedule-band artifact {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise V2ScheduleError("schedule-band artifact must be a top-level mapping")
    _require_exact_keys(payload, {"schema_version", "provenance", "bands"}, "artifact")
    if payload["schema_version"] != SCHEDULE_BAND_SCHEMA_VERSION:
        raise V2ScheduleError(
            f"artifact schema_version {payload['schema_version']!r} != "
            f"{SCHEDULE_BAND_SCHEMA_VERSION!r}"
        )
    raw_provenance = payload["provenance"]
    if not isinstance(raw_provenance, Mapping):
        raise V2ScheduleError("artifact.provenance must be a mapping")
    _require_exact_keys(
        raw_provenance, {field.name for field in fields(BandProvenance)}, "artifact.provenance",
    )
    try:
        provenance = BandProvenance(**raw_provenance)
    except (TypeError, ValueError) as exc:
        raise V2ScheduleError(f"invalid artifact.provenance: {exc}") from exc
    raw_bands = payload["bands"]
    if not isinstance(raw_bands, list):
        raise V2ScheduleError("artifact.bands must be a list")
    table = ScheduleBandTable(
        provenance=provenance,
        bands=tuple(_load_band(value, index) for index, value in enumerate(raw_bands)),
    )
    if expected_content_digest is not None:
        expected = _require_content_digest(expected_content_digest, "expected_content_digest")
        if table.provenance.calibration_content_digest != expected:
            raise V2ScheduleError(
                "loaded calibration digest does not match the runtime-declared digest: "
                f"loaded={table.provenance.calibration_content_digest}, expected={expected}"
            )
    return table


def lookup_band(table: ScheduleBandTable, *, step: int, stratum_key: str) -> ScheduleBand:
    """Exact lookup. No nearest-neighbour, no interpolation.

    A projected state at an uncalibrated ``(step, stratum)`` cannot proceed: PLAN §3.5 allows a
    production state to consume only a frozen artifact with matching identity, and inventing a band
    between two calibrated steps would be exactly the hardcoded tolerance PLAN §2.1 forbids.
    """
    _require_index(step, "step")
    _require_text(stratum_key, "stratum_key")
    for band in table.bands:
        if band.step == step and band.stratum_key == stratum_key:
            return band
    raise MissingScheduleBandError(
        f"no calibrated band for step={step} stratum={stratum_key!r} in calibration "
        f"{table.provenance.calibration_id!r}; bands are exact and are never interpolated"
    )


# --------------------------------------------------------------------------------------------
# The band gate
# --------------------------------------------------------------------------------------------

class BandVerdictReason(str, enum.Enum):
    ACCEPTED = "accepted"
    RHO_BELOW = "rho_below"
    RHO_ABOVE = "rho_above"
    UNRESOLVED_BELOW = "unresolved_below"
    UNRESOLVED_ABOVE = "unresolved_above"
    NO_UNRESOLVED_POSITION = "no_unresolved_position"
    NO_BAND_FOR_STEP = "no_band_for_step"
    STRATUM_MISMATCH = "stratum_mismatch"
    CALIBRATION_ID_MISMATCH = "calibration_id_mismatch"
    CALIBRATION_DIGEST_MISMATCH = "calibration_digest_mismatch"
    INCONSISTENT_OBSERVATION = "inconsistent_observation"
    MASK_LOAD_BELOW = "mask_load_below"
    MASK_LOAD_ABOVE = "mask_load_above"
    MASK_LOAD_INFEASIBLE = "mask_load_infeasible"


@dataclass(frozen=True)
class BandVerdict:
    """A gate outcome. A rejection is a value, not an exception.

    The caller needs to persist an explicit null/stalled feedback event rather than crash
    (PLAN §2.4 last bullet), so the schedule layer never raises on a legitimate out-of-band state.
    """

    accepted: bool
    reason: BandVerdictReason
    step: int
    stratum_key: str
    calibration_id: str
    observed: MaturityRecord | None
    rho_accept: BandInterval | None
    unresolved_accept: BandInterval | None
    detail: str = ""


def gate_projected_maturity(
    *, table: ScheduleBandTable, step: int, observed: MaturityRecord, stratum_key: str,
    declared_calibration_id: str, declared_calibration_digest: str,
) -> BandVerdict:
    """The ``B(r_d)`` gate: is this projected state's realized maturity legal at ``r_d``?

    The state-declared calibration identity is checked against the content-verified typed table
    before numeric bands are read. A caller cannot combine a foreign ID with locally convenient
    band values and obtain an accepted verdict.

    Rejects a fully resolved state regardless of the band: PLAN §2.4 requires the projected state
    to retain at least one unresolved editable position, or there is nothing left for the segment
    to denoise and no lookahead can fork from the descendant.
    """
    if not isinstance(table, ScheduleBandTable):
        raise V2ScheduleError("table must be a content-verified ScheduleBandTable")
    _require_index(step, "step")
    _require_text(stratum_key, "stratum_key")
    _require_text(declared_calibration_id, "declared_calibration_id")
    _require_content_digest(declared_calibration_digest, "declared_calibration_digest")
    if not isinstance(observed, MaturityRecord):
        raise V2ScheduleError("observed must be a MaturityRecord built from measured counts")

    provenance = table.provenance

    def identity_verdict(reason: BandVerdictReason, detail: str) -> BandVerdict:
        return BandVerdict(
            accepted=False,
            reason=reason,
            step=step,
            stratum_key=stratum_key,
            calibration_id=provenance.calibration_id,
            observed=observed,
            rho_accept=None,
            unresolved_accept=None,
            detail=detail,
        )

    if declared_calibration_id != provenance.calibration_id:
        return identity_verdict(
            BandVerdictReason.CALIBRATION_ID_MISMATCH,
            f"state declares {declared_calibration_id!r}, table is "
            f"{provenance.calibration_id!r}",
        )
    if declared_calibration_digest != provenance.calibration_content_digest:
        return identity_verdict(
            BandVerdictReason.CALIBRATION_DIGEST_MISMATCH,
            f"state declares {declared_calibration_digest}, table is "
            f"{provenance.calibration_content_digest}",
        )
    try:
        band = lookup_band(table, step=step, stratum_key=stratum_key)
    except MissingScheduleBandError as exc:
        return identity_verdict(BandVerdictReason.NO_BAND_FOR_STEP, str(exc))

    def verdict(accepted: bool, reason: BandVerdictReason, detail: str = "") -> BandVerdict:
        return BandVerdict(
            accepted=accepted, reason=reason, step=band.step, stratum_key=stratum_key,
            calibration_id=provenance.calibration_id, observed=observed,
            rho_accept=band.rho_accept,
            unresolved_accept=band.unresolved_accept, detail=detail,
        )

    if stratum_key != band.stratum_key:
        return verdict(
            False, BandVerdictReason.STRATUM_MISMATCH,
            f"band was calibrated for stratum {band.stratum_key!r}, state is in {stratum_key!r}",
        )
    if observed.n_unresolved_editable == 0:
        return verdict(
            False, BandVerdictReason.NO_UNRESOLVED_POSITION,
            "a projected state must retain >=1 unresolved editable position (PLAN §2.4)",
        )
    if observed.rho_edit < band.rho_accept.lo:
        return verdict(False, BandVerdictReason.RHO_BELOW)
    if observed.rho_edit > band.rho_accept.hi:
        return verdict(False, BandVerdictReason.RHO_ABOVE)
    if observed.n_unresolved_editable < band.unresolved_accept.lo:
        return verdict(False, BandVerdictReason.UNRESOLVED_BELOW)
    if observed.n_unresolved_editable > band.unresolved_accept.hi:
        return verdict(False, BandVerdictReason.UNRESOLVED_ABOVE)
    return verdict(True, BandVerdictReason.ACCEPTED)


# --------------------------------------------------------------------------------------------
# The coupled mask-load contract
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class UnresolvedInterval:
    """The integer count interval the projected state's unresolved mass is pinned to."""

    min_unresolved: int
    max_unresolved: int


@dataclass(frozen=True)
class AdmissibleMaskLoad:
    """How many previously-resolved positions the policy must newly mask -- cardinality only.

    *Which* residues occupy that cardinality (reward-responsible, uncertain, inconsistent,
    source-implicated) is entirely a ``FeedbackSupportPolicy`` decision inside this range. PLAN §2.5
    forbids freezing any reopen rule here.
    """

    pinned: UnresolvedInterval
    n_editable: int
    n_unresolved_source: int
    n_endpoint_writes_over_masked: int
    min_newly_masked: int
    max_newly_masked: int
    feasible: bool
    infeasible_reason: str | None


def pinned_unresolved_interval(
    *, band: ScheduleBand, n_editable: int,
) -> UnresolvedInterval:
    """Intersect the normalized band with the absolute band, in units of positions.

    The normalized interval is converted to counts with ``ceil`` on the low end and ``floor`` on the
    high end, so a rounded interval can only be narrower than the measured one -- never wider. The
    low end is clamped to >=1 because a projected state with no unresolved editable position is
    illegal whatever the band says.
    """
    _require_index(n_editable, "n_editable", minimum=1)
    rho_lo = math.ceil(band.rho_accept.lo * n_editable)
    rho_hi = math.floor(band.rho_accept.hi * n_editable)
    # rho is the RESOLVED fraction, so the unresolved interval is its complement, flipped.
    lo_from_rho = n_editable - rho_hi
    hi_from_rho = n_editable - rho_lo
    lo = max(1, lo_from_rho, math.ceil(band.unresolved_accept.lo))
    hi = min(n_editable, hi_from_rho, math.floor(band.unresolved_accept.hi))
    if lo > hi:
        raise EmptyBandIntersectionError(
            f"normalized and absolute maturity intervals do not overlap at step {band.step} for "
            f"n_editable={n_editable}: normalized gives [{lo_from_rho}, {hi_from_rho}], absolute "
            f"gives [{band.unresolved_accept.lo}, {band.unresolved_accept.hi}]"
        )
    return UnresolvedInterval(min_unresolved=lo, max_unresolved=hi)


def admissible_reopen_cardinality(
    *, band: ScheduleBand, n_editable: int, n_unresolved_source: int,
    n_endpoint_writes_over_masked: int,
) -> AdmissibleMaskLoad:
    """How many previously-resolved editable positions the projection must newly mask.

    Post-projection unresolved mass obeys the byte-level identity

    ``u_proj = u_src - a + b_new``

    where ``a`` counts endpoint writes landing on source-masked positions (they resolve a mask) and
    ``b_new`` counts previously-resolved editable positions that end up masked.
    ``inject_from_source_feedback`` and ``carry_from_source`` are byte-neutral on the count:
    injection requires a resolved source token, and a masked source position under carry stays
    masked. So ``b_new`` is pinned by the band once ``r_d`` is chosen -- reopen size is not a free
    parameter, which is the concrete form of PLAN Appendix B's "re-entry coordinate, maturity band,
    and total mask/reopen cardinality are coupled".

    Zero is legal when endpoint writes already land inside the calibrated band. An empty range is
    reported as ``feasible=False`` rather than clamped: it is a fact about the schedule, not
    something to silently repair.
    """
    _require_index(n_editable, "n_editable", minimum=1)
    _require_index(n_unresolved_source, "n_unresolved_source")
    _require_index(n_endpoint_writes_over_masked, "n_endpoint_writes_over_masked")
    if n_unresolved_source > n_editable:
        raise V2ScheduleError(
            f"n_unresolved_source={n_unresolved_source} exceeds n_editable={n_editable}"
        )
    if n_endpoint_writes_over_masked > n_unresolved_source:
        raise V2ScheduleError(
            f"n_endpoint_writes_over_masked={n_endpoint_writes_over_masked} exceeds the "
            f"{n_unresolved_source} masked source positions available to write over"
        )
    pinned = pinned_unresolved_interval(band=band, n_editable=n_editable)
    a = n_endpoint_writes_over_masked
    lo = max(0, pinned.min_unresolved - n_unresolved_source + a)
    # b_new counts previously-RESOLVED positions, so it cannot exceed how many there are.
    hi = min(n_editable - n_unresolved_source, pinned.max_unresolved - n_unresolved_source + a)
    feasible = lo <= hi
    reason = None
    if not feasible:
        reason = (
            f"no legal reopen cardinality: the band pins post-projection unresolved mass to "
            f"[{pinned.min_unresolved}, {pinned.max_unresolved}] but the source carries "
            f"{n_unresolved_source} unresolved of {n_editable} editable positions, so b_new would "
            f"have to lie in [{lo}, {hi}]"
        )
    return AdmissibleMaskLoad(
        pinned=pinned, n_editable=n_editable, n_unresolved_source=n_unresolved_source,
        n_endpoint_writes_over_masked=a, min_newly_masked=lo, max_newly_masked=hi,
        feasible=feasible, infeasible_reason=reason,
    )


class BandCenterRule(str, enum.Enum):
    """How the integer unresolved target is read off the pinned interval.

    PLAN §2.5 requires the calibration to "materialize one content-bound integer unresolved target
    ``u_target`` at the declared center of ``B(r_d)``, including its rounding/tie law".  The interval
    is over integer position counts, so an even-width interval has no integer midpoint and the
    rounding direction is a scientific choice: it decides whether the projected state sits one
    position more or less mature than the band's centre, and therefore how many positions the
    reopen equation demands.  It is declared, never defaulted.
    """

    #: ``floor((lo + hi) / 2)`` -- an even-width interval resolves toward the LOWER unresolved
    #: count, i.e. the MORE mature side.
    MIDPOINT_TIE_LOW = "midpoint_tie_low"
    #: ``ceil((lo + hi) / 2)`` -- an even-width interval resolves toward the HIGHER unresolved
    #: count, i.e. the LESS mature side.
    MIDPOINT_TIE_HIGH = "midpoint_tie_high"


@dataclass(frozen=True)
class BandCenterTarget:
    """The single integer unresolved mass a projection at ``r_d`` must realize.

    Content-bound, not merely computed: the record carries the calibration identity and digest of
    the table the interval came from, so a target quoted in an artifact can be checked against the
    band it claims to come from rather than taken on trust.  ``n_editable`` is part of the identity
    because the pinned interval is a function of it -- two proteins under one band have two targets.
    """

    u_target: int
    pinned: UnresolvedInterval
    rule: BandCenterRule
    step: int
    stratum_key: str
    n_editable: int
    calibration_id: str
    calibration_content_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.rule, BandCenterRule):
            raise V2ScheduleError("rule must be a BandCenterRule")
        if not isinstance(self.pinned, UnresolvedInterval):
            raise V2ScheduleError("pinned must be an UnresolvedInterval")
        _require_index(self.u_target, "u_target", minimum=1)
        if not (self.pinned.min_unresolved <= self.u_target <= self.pinned.max_unresolved):
            raise V2ScheduleError(
                f"u_target={self.u_target} is outside the pinned interval "
                f"[{self.pinned.min_unresolved}, {self.pinned.max_unresolved}]; a target off its "
                "own band would put every projection outside B(r_d) while claiming the band's name"
            )


def band_center_target(
    *, table: ScheduleBandTable, step: int, stratum_key: str, n_editable: int,
    rule: BandCenterRule,
) -> BandCenterTarget:
    """The declared integer centre of ``B(r_d)``, bound to the calibration it was read from.

    Derived from the content-verified table rather than stored as a fourth quantile: the table's
    digest already signs the interval this is a pure function of, and adding a field would create a
    second number that can disagree with the interval it summarizes.  The rounding/tie law is
    supplied by the caller from frozen config, so the derivation is total and the choice is visible.
    """
    if not isinstance(table, ScheduleBandTable):
        raise V2ScheduleError("table must be a content-verified ScheduleBandTable")
    if not isinstance(rule, BandCenterRule):
        raise V2ScheduleError(
            "rule must be a BandCenterRule; the rounding/tie law is declared by the run, and an "
            "implicit one would silently move the target by one position (PLAN §2.5)"
        )
    band = lookup_band(table, step=step, stratum_key=stratum_key)
    pinned = pinned_unresolved_interval(band=band, n_editable=n_editable)
    total = pinned.min_unresolved + pinned.max_unresolved
    if rule is BandCenterRule.MIDPOINT_TIE_LOW:
        target = total // 2
    else:
        target = -((-total) // 2)
    return BandCenterTarget(
        u_target=int(target), pinned=pinned, rule=rule, step=band.step, stratum_key=stratum_key,
        n_editable=int(n_editable), calibration_id=table.provenance.calibration_id,
        calibration_content_digest=table.provenance.calibration_content_digest,
    )


def required_reopen_count(*, u_target: int, n_unresolved_source: int, n_writes: int) -> int:
    """``m_reopen = u_target - u_src + m_d`` (PLAN §2.5), as an exact integer identity.

    A separate function rather than an expression at the call site because it is the one place the
    equation exists: the projection's own byte-level identity is ``u_proj = u_src - a + b_new``, and
    with every endpoint write landing on a source-masked position (``a == m_d``) this is the unique
    ``b_new`` that lands exactly on the target.  A negative result is returned AS a negative number
    rather than clamped -- the caller must fail closed on it, and a silent ``max(0, ...)`` would
    produce a projection that misses the declared target while reporting the target's name.
    """
    _require_index(u_target, "u_target", minimum=1)
    _require_index(n_unresolved_source, "n_unresolved_source")
    _require_index(n_writes, "n_writes")
    return int(u_target) - int(n_unresolved_source) + int(n_writes)


def validate_realized_mask_load(
    *, load: AdmissibleMaskLoad, n_newly_masked: int,
) -> BandVerdict:
    """Check that a policy's realized partition landed inside the admissible range.

    Runs before the segment, so a policy that ignores the cardinality contract fails closed rather
    than producing an off-schedule descendant that would then be relabelled as an ordinary state.
    """
    _require_index(n_newly_masked, "n_newly_masked")

    def verdict(accepted: bool, reason: BandVerdictReason, detail: str = "") -> BandVerdict:
        return BandVerdict(
            accepted=accepted, reason=reason, step=-1, stratum_key="", calibration_id="",
            observed=None, rho_accept=None, unresolved_accept=None, detail=detail,
        )

    if not load.feasible:
        return verdict(False, BandVerdictReason.MASK_LOAD_INFEASIBLE, load.infeasible_reason or "")
    if n_newly_masked < load.min_newly_masked:
        return verdict(
            False, BandVerdictReason.MASK_LOAD_BELOW,
            f"{n_newly_masked} newly masked < minimum {load.min_newly_masked}",
        )
    if n_newly_masked > load.max_newly_masked:
        return verdict(
            False, BandVerdictReason.MASK_LOAD_ABOVE,
            f"{n_newly_masked} newly masked > maximum {load.max_newly_masked}",
        )
    return verdict(True, BandVerdictReason.ACCEPTED)
