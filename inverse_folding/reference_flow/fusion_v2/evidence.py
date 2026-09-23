"""V2F5A: cross-sequence local Head evidence on ONE common raw-window scale (PLAN §2.5, A.3).

The S7 null is what this module exists to answer.  That experiment ranked DONORS by exact complete
Head risk and then copied one token chosen by source mask geometry, so the applied dose was never
ordered by anything the Head said about that position.  PLAN §2.5 therefore requires the write
selector to consider only positions that "fall in an aligned Head window improved in the donor",
scored by an identity-bound frozen-Head counterfactual.  This module builds both halves and nothing
else: it decides no cardinality, reads no config, and selects no position.

**Two prohibitions are structural here, not conventional.**

1. **Never ``residue_hotspot``.**  PLAN §2.5: "Cross-sequence local evidence must be reconstructed
   from raw aligned-window Head outputs on one common scale, not by subtracting independently
   centered or clipped residue-hotspot summaries."  A residue-hotspot vector is a per-sequence
   summary -- whatever centering or clipping produced it was computed within that sequence, so
   ``h_i(Y) - h_i(I)`` across two sequences subtracts two different transforms and the sign it
   reports is an artifact of the summary.  The raw aligned window ``z`` is the same measurement on
   both sides.  :mod:`tests.inverse_folding.test_fusion_v2_head_directed_policy` asserts by AST that
   this module never reads the attribute.

2. **Never the run config.**  Same rule as :mod:`fusion_v2.policy`: a module the policy imports must
   not be able to reach an unfrozen threshold.  Enforced transitively by the same AST test.

**A second alignment law, deliberately.**  :func:`fusion_v2.safety.whole_landscape_new_hotspot`
aligns a design against the IMMUTABLE reference to decide ADMISSION, and an unusable comparison
there is a hard refusal.  This aligns a donor against the lineage INCUMBENT to decide SUPPORT, and
an unusable comparison here is a typed policy stall.  The two share the coordinate law -- imported
from the same ``fusion.objective.window_coord`` v0 primitive both sides use -- and a test pins them
to the same verdict on a shared fixture so the duplication cannot drift.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from inverse_folding.reference_flow.fusion.objective import window_coord
from inverse_folding.reference_flow.fusion.state import sequence_md5

from .errors import V2Error
from .identity import HeadEvaluatorIdentity, canonical_digest

__all__ = [
    "V2EvidenceError",
    "EvidenceIdentityMismatch",
    "EvidenceGridMismatch",
    "EmptyEvidenceGrid",
    "NonFiniteEvidence",
    "WindowDelta",
    "AlignedWindowEvidence",
    "build_window_evidence",
    "LeaveOneOutContribution",
    "counterfactual_sequence",
    "score_leave_one_out",
]


class V2EvidenceError(V2Error):
    """Cross-sequence local evidence could not be built from what was supplied."""


class EvidenceIdentityMismatch(V2EvidenceError):
    """The two scores were not produced by one evaluator, or describe different molecules."""


class EvidenceGridMismatch(V2EvidenceError):
    """The two window grids are not the same coordinate set, so there is no common scale."""


class EmptyEvidenceGrid(V2EvidenceError):
    """A score carries no windows; reporting "no local evidence" as "no difference" fails open."""


class NonFiniteEvidence(V2EvidenceError):
    """A window risk or a global risk is non-finite, and would compare false against every gate."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NonFiniteEvidence(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise NonFiniteEvidence(f"{name} must be finite, got {out!r}")
    return out


def _text(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise V2EvidenceError(f"{name} must be a non-empty str, got {value!r}")
    return value


@dataclass(frozen=True)
class WindowDelta:
    """One aligned window, both raw risks, and their difference on that one scale.

    ``delta = z_donor - z_incumbent``.  Lower Head risk is better everywhere in V2 (the archive
    ranks endpoints by ascending ``head_global_risk``), so a NEGATIVE delta is a donor improvement
    and a positive delta is a worsening.  Both signs are retained: the write channel needs the
    improved windows and the reopen channel needs the worsened ones, and a single unsigned magnitude
    could serve neither.
    """

    start_0b: int
    end_0b: int
    k: int
    z_donor: float
    z_incumbent: float

    @property
    def coord(self) -> tuple[int, int, int]:
        return (self.start_0b, self.end_0b, self.k)

    @property
    def delta(self) -> float:
        return self.z_donor - self.z_incumbent

    @property
    def improves(self) -> bool:
        return self.delta < 0.0

    @property
    def worsens(self) -> bool:
        return self.delta > 0.0

    def contains(self, position: int) -> bool:
        return self.start_0b <= int(position) < self.end_0b


@dataclass(frozen=True)
class AlignedWindowEvidence:
    """Every aligned window of one (donor, reference) pair, indexed by position.

    ``reference`` is whichever complete sequence the donor is being compared AGAINST -- the lineage
    incumbent for the write/worsening channel, the immutable cumulative safety reference for the
    new-hotspot channel.  The type does not know which; the caller names it in ``reference_label``
    so an artifact can never confuse the two comparisons.
    """

    reference_label: str
    donor_sequence_md5: str
    reference_sequence_md5: str
    sequence_length: int
    head_identity_digest: str
    windows: tuple[WindowDelta, ...]
    #: position -> indices into ``windows``.  Precomputed because every candidate asks for it.
    _by_position: Mapping[int, tuple[int, ...]]

    def windows_at(self, position: int) -> tuple[WindowDelta, ...]:
        return tuple(self.windows[index] for index in self._by_position.get(int(position), ()))

    def covered(self, position: int) -> bool:
        """Whether ANY window contains this position.

        A position no window covers has no local evidence at all.  It is reported as uncovered
        rather than as "no improvement", because the two call for different operator actions: the
        first says the Head grid does not reach here, the second says the Head looked and disagreed.
        """
        return bool(self._by_position.get(int(position)))

    def min_delta_at(self, position: int) -> float | None:
        """The most-improved aligned window containing ``position`` (``None`` if uncovered)."""
        windows = self.windows_at(position)
        return min(window.delta for window in windows) if windows else None

    def max_delta_at(self, position: int) -> float | None:
        """The most-worsened aligned window containing ``position`` (``None`` if uncovered)."""
        windows = self.windows_at(position)
        return max(window.delta for window in windows) if windows else None

    def improves_at(self, position: int) -> bool:
        """PLAN A.3's window screen: does SOME aligned window containing ``i`` improve?

        Strict: a window that is exactly equal in the two sequences is not an improvement.  The
        calibrated local tolerance applies to the counterfactual contribution ``a_i``, which is the
        quantity the policy actually ranks on; applying a second tolerance here would make the
        screen's strictness depend on a number calibrated for a different measurement.
        """
        delta = self.min_delta_at(position)
        return delta is not None and delta < 0.0

    def worsening_at(self, position: int) -> float:
        """``[max_w delta_w]_+`` over windows containing ``i``: local worsening evidence, 0 if none."""
        delta = self.max_delta_at(position)
        return max(0.0, delta) if delta is not None else 0.0

    def residual_burden_at(self, position: int) -> float | None:
        """``max_w z_w(donor)`` over windows containing ``i``: how much burden the DONOR still has.

        Read off the donor's own raw window risks rather than off the delta, because a position can
        be an improvement over the incumbent and still carry the highest absolute burden left in the
        design -- which is exactly the position the reopen channel should target.
        """
        windows = self.windows_at(position)
        return max(window.z_donor for window in windows) if windows else None

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "reference_label": self.reference_label,
            "donor_sequence_md5": self.donor_sequence_md5,
            "reference_sequence_md5": self.reference_sequence_md5,
            "sequence_length": self.sequence_length,
            "head_identity_digest": self.head_identity_digest,
            "windows": [
                [w.start_0b, w.end_0b, w.k, w.z_donor, w.z_incumbent] for w in self.windows
            ],
        }

    @property
    def evidence_digest(self) -> str:
        """Content identity of this comparison, for the PLAN §5.3 "raw-window evidence identity"."""
        return canonical_digest(self.canonical_payload())


def _grid(score: Any, label: str, evaluator: HeadEvaluatorIdentity) -> dict[tuple[int, int, int], float]:
    if getattr(score, "allele", None) != evaluator.allele:
        raise EvidenceIdentityMismatch(
            f"{label} was scored for allele {getattr(score, 'allele', None)!r}, evaluator declares "
            f"{evaluator.allele!r}; two alleles are two instruments and their z values do not share "
            "a scale"
        )
    if getattr(score, "score_scale", None) != evaluator.score_scale:
        raise EvidenceIdentityMismatch(
            f"{label} uses score scale {getattr(score, 'score_scale', None)!r}, evaluator declares "
            f"{evaluator.score_scale!r}; subtracting across two scales is not a local improvement"
        )
    out: dict[tuple[int, int, int], float] = {}
    for window in tuple(getattr(score, "windows", ())):
        coord = window_coord(window)
        if coord in out:
            raise EvidenceGridMismatch(f"{label} grid carries duplicate coordinate {coord}")
        if not evaluator.window_k_min <= coord[2] <= evaluator.window_k_max:
            raise EvidenceIdentityMismatch(
                f"{label} window k={coord[2]} is outside the evaluator domain "
                f"[{evaluator.window_k_min}, {evaluator.window_k_max}]"
            )
        out[coord] = _finite(getattr(window, "z", None), f"{label} window {coord} z")
    if not out:
        raise EmptyEvidenceGrid(
            f"{label} carries no Head windows; an empty grid reported as 'no local difference' "
            "would let a policy write on evidence it never had"
        )
    return out


def build_window_evidence(
    *,
    donor_score: Any,
    reference_score: Any,
    evaluator: HeadEvaluatorIdentity,
    reference_label: str,
) -> AlignedWindowEvidence:
    """Align two complete-sequence Head scores exactly and keep both raw risks per window.

    Every failure is a refusal, in a deterministic order, so a caller that sees a grid mismatch can
    trust that the coordinate sets really differ rather than that a length problem surfaced late.
    """
    if not isinstance(evaluator, HeadEvaluatorIdentity):
        raise EvidenceIdentityMismatch("evaluator must be a HeadEvaluatorIdentity")
    _text(reference_label, "reference_label")
    donor_protein = getattr(donor_score, "protein_id", None)
    reference_protein = getattr(reference_score, "protein_id", None)
    if donor_protein != reference_protein:
        raise EvidenceIdentityMismatch(
            f"donor protein {donor_protein!r} != {reference_label} protein {reference_protein!r}"
        )
    donor_length = getattr(donor_score, "sequence_length", None)
    reference_length = getattr(reference_score, "sequence_length", None)
    if donor_length != reference_length:
        raise EvidenceGridMismatch(
            f"donor length {donor_length} != {reference_label} length {reference_length}; for a "
            "fixed k-range the window grid is determined by length, so there is no alignment"
        )
    donor_grid = _grid(donor_score, "donor", evaluator)
    reference_grid = _grid(reference_score, reference_label, evaluator)
    if set(donor_grid) != set(reference_grid):
        only_donor = sorted(set(donor_grid) - set(reference_grid))
        only_reference = sorted(set(reference_grid) - set(donor_grid))
        raise EvidenceGridMismatch(
            f"window coordinate sets differ; donor-only {only_donor[:3]}, "
            f"{reference_label}-only {only_reference[:3]}"
        )

    windows = tuple(
        WindowDelta(start_0b=coord[0], end_0b=coord[1], k=coord[2],
                    z_donor=donor_grid[coord], z_incumbent=reference_grid[coord])
        for coord in sorted(donor_grid)
    )
    by_position: dict[int, list[int]] = {}
    for index, window in enumerate(windows):
        for position in range(window.start_0b, window.end_0b):
            by_position.setdefault(position, []).append(index)
    return AlignedWindowEvidence(
        reference_label=reference_label,
        donor_sequence_md5=str(getattr(donor_score, "sequence_md5", "")),
        reference_sequence_md5=str(getattr(reference_score, "sequence_md5", "")),
        sequence_length=int(donor_length),
        head_identity_digest=evaluator.digest(),
        windows=windows,
        _by_position={key: tuple(value) for key, value in sorted(by_position.items())},
    )


# --------------------------------------------------------------------------------------------
# the frozen-Head leave-one-out counterfactual
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LeaveOneOutContribution:
    r"""``a_i = R_H(Y*_{i <- I_i}) - R_H(Y*)`` for one position (PLAN §2.5).

    Positive ``a_i`` means REVERTING the donor identity to the incumbent's makes the donor worse
    under the frozen Head -- i.e. the donor identity at ``i`` is carrying some of the donor's
    advantage.  PLAN §2.5 names what this is and is not: "exact frozen-Head contribution evidence in
    the donor context, not biological causality".  The whole record is kept, including the
    counterfactual's own digest, so the artifact can be re-derived from the same Head without
    re-running the policy.
    """

    position: int
    donor_residue: str
    incumbent_residue: str
    counterfactual_sequence_md5: str
    donor_global_risk: float
    counterfactual_global_risk: float

    @property
    def contribution(self) -> float:
        return self.counterfactual_global_risk - self.donor_global_risk

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "donor_residue": self.donor_residue,
            "incumbent_residue": self.incumbent_residue,
            "counterfactual_sequence_md5": self.counterfactual_sequence_md5,
            "donor_global_risk": self.donor_global_risk,
            "counterfactual_global_risk": self.counterfactual_global_risk,
            "contribution": self.contribution,
        }


def counterfactual_sequence(donor_sequence: str, incumbent_sequence: str, position: int) -> str:
    """The donor with exactly one identity reverted to the incumbent's.

    A builder rather than a slice at the call site so "only position ``i`` differs from the donor"
    holds by construction: a counterfactual that changed two positions would attribute two
    identities' contribution to one.
    """
    index = int(position)
    if not 0 <= index < len(donor_sequence):
        raise V2EvidenceError(f"position {index} is outside the donor sequence")
    if len(donor_sequence) != len(incumbent_sequence):
        raise EvidenceGridMismatch(
            f"donor length {len(donor_sequence)} != incumbent length {len(incumbent_sequence)}; a "
            "leave-one-out revert is only defined between aligned complete sequences of one length"
        )
    return donor_sequence[:index] + incumbent_sequence[index] + donor_sequence[index + 1:]


def score_leave_one_out(
    *,
    scorer: Any,
    protein_id: str,
    donor_sequence: str,
    donor_global_risk: float,
    incumbent_sequence: str,
    positions: Sequence[int],
    evaluator: HeadEvaluatorIdentity,
    window_grid_digest: str,
) -> dict[int, LeaveOneOutContribution]:
    """Score every shortlisted counterfactual through ONE frozen-Head batch.

    ``scorer`` is injected with the shape ``scorer(protein_id, sequences) -> results``, where each
    result carries ``sequence_md5``, ``binding`` and ``global_risk``.  It is a callable rather than
    the Head object itself so this module stays free of the runtime's request type, of torch and of
    any cost journal -- the runtime adapter owns those (see
    ``fusion_v2_runtime.contribution.CounterfactualHeadScorer``).

    Results are matched BY SEQUENCE DIGEST, never by position in the returned list, for the same
    reason ``bind_head_scores`` does: a reordered batch would attach one counterfactual's risk to
    another position and the resulting ranking would be noise that looks entirely plausible.

    A position whose donor and incumbent residues are equal is refused rather than scored: its
    counterfactual IS the donor, so ``a_i`` is identically zero and asking the Head is a wasted call
    that would then be ranked as if it were evidence.
    """
    _text(protein_id, "protein_id")
    if not isinstance(evaluator, HeadEvaluatorIdentity):
        raise EvidenceIdentityMismatch("evaluator must be a HeadEvaluatorIdentity")
    donor_risk = _finite(donor_global_risk, "donor_global_risk")
    if len(donor_sequence) != len(incumbent_sequence):
        raise EvidenceGridMismatch(
            f"donor length {len(donor_sequence)} != incumbent length {len(incumbent_sequence)}"
        )
    wanted: list[int] = []
    for raw in positions:
        position = int(raw)
        if position in wanted:
            continue
        if donor_sequence[position] == incumbent_sequence[position]:
            raise V2EvidenceError(
                f"position {position} carries the same residue in donor and incumbent, so its "
                "leave-one-out counterfactual IS the donor; a candidate must differ from the "
                "incumbent before it can have a contribution (PLAN §2.5)"
            )
        wanted.append(position)
    if not wanted:
        return {}

    sequences = [counterfactual_sequence(donor_sequence, incumbent_sequence, p) for p in wanted]
    by_position = {p: sequence_md5(sequence) for p, sequence in zip(wanted, sequences)}
    results = scorer(protein_id, sequences)

    by_md5: dict[str, Any] = {}
    for result in results:
        md5 = getattr(result, "sequence_md5", None)
        if md5 is None:
            raise EvidenceIdentityMismatch(
                "every counterfactual Head result must carry its own sequence_md5; without it the "
                "batch can only be matched by position and a reordered batch is undetectable"
            )
        if md5 in by_md5:
            raise EvidenceIdentityMismatch(f"duplicate counterfactual Head result for {md5}")
        by_md5[str(md5)] = result

    out: dict[int, LeaveOneOutContribution] = {}
    for position in wanted:
        md5 = by_position[position]
        result = by_md5.get(md5)
        if result is None:
            raise EvidenceIdentityMismatch(
                f"no counterfactual Head result for position {position} (sequence {md5}); a missing "
                "score would silently drop a candidate from the ranking"
            )
        binding = getattr(result, "binding", None)
        if binding is None or getattr(binding, "evaluator", None) != evaluator:
            raise EvidenceIdentityMismatch(
                f"the counterfactual result for position {position} was produced by a different "
                "evaluator identity than the donor; a_i would subtract two instruments"
            )
        if getattr(binding, "window_grid_digest", None) != window_grid_digest:
            raise EvidenceGridMismatch(
                f"the counterfactual result for position {position} declares window grid "
                f"{getattr(binding, 'window_grid_digest', None)}, the donor pool declares "
                f"{window_grid_digest}"
            )
        out[position] = LeaveOneOutContribution(
            position=position,
            donor_residue=donor_sequence[position],
            incumbent_residue=incumbent_sequence[position],
            counterfactual_sequence_md5=md5,
            donor_global_risk=donor_risk,
            counterfactual_global_risk=_finite(
                getattr(result, "global_risk", None), f"counterfactual global_risk at {position}"),
        )
    return out
