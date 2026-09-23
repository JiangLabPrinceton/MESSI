r"""DUALF1: the frozen two-allele objective, its calibration record, and its rank key.

Scientific authority: ``doc/Dual_Allele_Steering.md`` §§2.2-2.3. Implementation contract:
``PLAN_RF_FUSION_V2_DUAL_ALLELE.md`` §2.1.

Two independently trained Heads produce risks on incomparable raw scales, so each allele is mapped
through a frozen affine coordinate and the pair is reduced by a bounded smooth worst-residual
scalar:

.. math::

    u_a(y) = \frac{R_a(y) - b_a}{s_a},
    \qquad
    J_\tau(y) = \tau\log\frac{e^{u_A/\tau} + e^{u_B/\tau}}{2}.

Three properties are why this scalar and not another. It is bounded below the hard maximum by
exactly ``tau*log 2``, so an arbitrarily favourable non-worst allele cannot buy unbounded credit
against the worse one -- the failure mode of any weighted sum. It is strictly increasing in each
coordinate, so a Pareto-dominated endpoint can never outrank its dominator. And it is continuous
through active-worst switches, so donor selection and leave-one-out attribution can share one stable
total order instead of needing a separate tie rule.

**Purity.** stdlib and :mod:`fusion_v2.identity` only -- no torch, no config import, no I/O. Same
contract, and same reason, as :mod:`fusion_v2.reward`: a law that can reach an unfrozen threshold is
not frozen. Everything measured arrives as a validated value on :class:`DualCalibration`, which is
constructed once from a signed artifact by the calibration producer and never from a runtime batch.
This module deliberately exposes no verb that would fit coordinates to an observed candidate cloud.

**What is calibrated, and what that decides.** For every decision taken inside one protein and one
lineage a shift common to both alleles cancels, because ``J(u_A + c, u_B + c) = J(u_A, u_B) + c``.
The scientific content of the locations is therefore the NORMALIZED difference
``b_A/s_A - b_B/s_B``: it declares the equal-risk line ``R_A/s_A - R_B/s_B = b_A/s_A - b_B/s_B``.
Not the raw ``b_A - b_B`` -- adding one raw constant to both locations leaves that unchanged while
moving the boundary by ``delta*(1/s_A - 1/s_B)``. The scales set the exchange rate, and because
``s_A != s_B`` in general, ``tau``
is declared in normalized units and never in a raw score scale -- a raw-unit declaration would mean
two different credits for the two alleles and reintroduce the asymmetry the normalization exists to
remove.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Any

from .errors import V2Error
from .identity import HeadEvaluatorIdentity, canonical_digest, require_digest

__all__ = [
    "V2JointObjectiveError",
    "LOG2",
    "MEASURED_QUANTITIES",
    "QUANTITY_ATTRIBUTES",
    "AlleleRole",
    "ObjectiveMode",
    "smooth_max",
    "hard_max",
    "AlleleCoordinate",
    "QuantityCoordinates",
    "DualObjectiveLaw",
    "PanelBinding",
    "DualCalibration",
    "JointValue",
    "DualObjective",
    "SingleAlleleObjective",
    "ARM_STEERING_ROLE",
    "build_arm_objective",
]

LOG2 = math.log(2.0)

#: The two complete-sequence quantities a frozen Head reports that the Dual layer may reduce.
#: ``global_risk`` is the runtime steering objective. ``positive_mass_density`` exists only for the
#: return-boundary front (scientific document §3.2.1) and never enters runtime steering.
MEASURED_QUANTITIES = frozenset({"global_risk", "positive_mass_density", "window_z"})

#: Which attribute of a Head score carries each quantity. ``window_z`` has no entry: it is a
#: PER-WINDOW quantity read off the aligned window evidence, not an endpoint-level scalar, so it
#: cannot be resolved from a score object the way the other two can.
QUANTITY_ATTRIBUTES = {
    "global_risk": "global_risk",
    "positive_mass_density": "positive_mass_density",
}

#: The unit in which ``tau`` must be declared. Anything else fails closed.
NORMALIZED_UNITS = "normalized"


class V2JointObjectiveError(V2Error):
    """A Dual objective, calibration, or evidence-identity contract was violated."""


class AlleleRole(str, enum.Enum):
    """Which side of the objective an allele occupies. Roles are positional, not biological."""

    A = "A"
    B = "B"


class ObjectiveMode(str, enum.Enum):
    """The reduction law. ``HARD_MAX`` is the matched strict control, not a second architecture."""

    SMOOTH_MAX = "smooth_max"
    HARD_MAX = "hard_max"


# ----------------------------------------------------------------------------------------------
# validators
# ----------------------------------------------------------------------------------------------

def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V2JointObjectiveError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise V2JointObjectiveError(f"{name} must be finite, got {out!r}")
    return out


def _text(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise V2JointObjectiveError(f"{name} must be a non-empty str, got {value!r}")
    return value


def _quantity(value: Any, name: str) -> str:
    text = _text(value, name)
    if text not in MEASURED_QUANTITIES:
        raise V2JointObjectiveError(
            f"{name}={text!r} is not a declared measured quantity; available: "
            f"{sorted(MEASURED_QUANTITIES)}"
        )
    return text


# ----------------------------------------------------------------------------------------------
# the scalar
# ----------------------------------------------------------------------------------------------

def smooth_max(u_a: float, u_b: float, tau: float) -> float:
    r"""``tau * log((exp(u_A/tau) + exp(u_B/tau)) / 2)``, evaluated without a direct exponential.

    Written as ``m + tau * (log1p(exp(-d/tau)) - log 2)`` with ``m = max(u_A, u_B)`` and
    ``d = |u_A - u_B| >= 0``. The exponent is never positive, so nothing overflows however far apart
    the two coordinates are; and at ``d = 0`` the bracket is exactly zero, so ``J(u, u) = u`` holds
    to the last bit rather than to a tolerance. Evaluating the defining formula directly would
    overflow for ``u/tau`` beyond about 709 -- reachable at any realistic ``tau`` -- and would lose
    the diagonal identity to cancellation.
    """
    left = _finite(u_a, "u_a")
    right = _finite(u_b, "u_b")
    width = _finite(tau, "tau")
    if width <= 0.0:
        raise V2JointObjectiveError(
            f"tau must be strictly positive, got {width}; tau is the width of the region in which "
            "the near-worst allele may influence the decision, and a non-positive width is the "
            "hard maximum, which is a separately declared mode"
        )
    peak = left if left >= right else right
    gap = abs(left - right)
    return peak + width * (math.log1p(math.exp(-gap / width)) - LOG2)


def hard_max(u_a: float, u_b: float) -> float:
    """The strict normalized maximum: the ``tau -> 0`` limit of :func:`smooth_max`."""
    left = _finite(u_a, "u_a")
    right = _finite(u_b, "u_b")
    return left if left >= right else right


# ----------------------------------------------------------------------------------------------
# calibration records
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AlleleCoordinate:
    """``(b_a, s_a)`` for ONE allele on ONE measured quantity, bound to the Head that produced it.

    ``raw_noise_floor`` is that allele's matched-difference repeatability floor **in its own raw
    score scale**. It is carried here rather than alongside the objective because the propagation
    into the joint margin divides by this allele's own ``s_a``; a floor detached from its scale is
    not a margin.
    """

    role: AlleleRole
    quantity: str
    location: float
    scale: float
    raw_noise_floor: float
    evaluator: HeadEvaluatorIdentity
    source_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, AlleleRole):
            raise V2JointObjectiveError("role must be an AlleleRole")
        _quantity(self.quantity, "quantity")
        _finite(self.location, "location")
        scale = _finite(self.scale, "scale")
        floor = _finite(self.raw_noise_floor, "raw_noise_floor")
        if scale <= 0.0:
            raise V2JointObjectiveError(f"scale must be strictly positive, got {scale}")
        if floor < 0.0:
            raise V2JointObjectiveError(f"raw_noise_floor must be non-negative, got {floor}")
        if scale <= floor:
            raise V2JointObjectiveError(
                f"scale ({scale}) does not exceed this allele's own raw noise floor ({floor}); a "
                "coordinate whose unit is no larger than the instrument's repeatability measures "
                "noise, and inventing a floor here would hide that (PLAN §2.1: fail rather than "
                "invent a floor)"
            )
        if not isinstance(self.evaluator, HeadEvaluatorIdentity):
            raise V2JointObjectiveError("evaluator must be a HeadEvaluatorIdentity")
        require_digest(self.source_ref, "source_ref")

    def normalize(self, raw: float) -> float:
        """``u_a = (R_a - b_a) / s_a``."""
        return (_finite(raw, "raw") - self.location) / self.scale

    @property
    def normalized_noise_floor(self) -> float:
        """This allele's raw floor expressed in the shared coordinate."""
        return self.raw_noise_floor / self.scale

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "quantity": self.quantity,
            "location": self.location,
            "scale": self.scale,
            "raw_noise_floor": self.raw_noise_floor,
            "evaluator": self.evaluator.canonical_payload(),
            "source_ref": self.source_ref,
        }


@dataclass(frozen=True)
class QuantityCoordinates:
    """The A/B coordinate pair for one measured quantity, with the pairing laws enforced."""

    quantity: str
    a: AlleleCoordinate
    b: AlleleCoordinate

    def __post_init__(self) -> None:
        _quantity(self.quantity, "quantity")
        for slot, coordinate in (("a", self.a), ("b", self.b)):
            if not isinstance(coordinate, AlleleCoordinate):
                raise V2JointObjectiveError(f"{slot} must be an AlleleCoordinate")
        if self.a.role is not AlleleRole.A or self.b.role is not AlleleRole.B:
            raise V2JointObjectiveError(
                f"slot a carries role {self.a.role.value!r} and slot b carries "
                f"{self.b.role.value!r}; the two slots are positional and must carry roles A and B "
                "respectively, so a swapped pair cannot silently relabel which allele is which"
            )
        for slot, coordinate in (("a", self.a), ("b", self.b)):
            if coordinate.quantity != self.quantity:
                raise V2JointObjectiveError(
                    f"slot {slot} calibrates {coordinate.quantity!r} but the pair declares "
                    f"{self.quantity!r}; one pair reduces exactly one quantity"
                )
        left, right = self.a.evaluator, self.b.evaluator
        if left.head_checkpoint_digest == right.head_checkpoint_digest:
            raise V2JointObjectiveError(
                "both roles are bound to the same Head checkpoint digest; the two production Heads "
                "share a config directory and an identical checkpoint metadata config_hash, so the "
                "checkpoint digest is the only bit that discriminates them and an equal digest "
                "means one Head has been bound twice"
            )
        if left.allele == right.allele:
            raise V2JointObjectiveError(
                f"both roles declare allele {left.allele!r}; the objective would reduce one allele "
                "against itself"
            )
        if left.score_scale != right.score_scale:
            raise V2JointObjectiveError(
                f"role A scores on {left.score_scale!r} and role B on {right.score_scale!r}; the "
                "normalization removes scale, not units, and two score scales are two instruments"
            )
        if (left.window_k_min, left.window_k_max) != (right.window_k_min, right.window_k_max):
            raise V2JointObjectiveError(
                f"role A uses window k in [{left.window_k_min},{left.window_k_max}] and role B "
                f"[{right.window_k_min},{right.window_k_max}]; the window grid must be identical "
                "for the two Heads to describe the same coordinate system on one sequence"
            )

    def normalize_level(self, role: AlleleRole, value: float) -> float:
        """An absolute reading on this allele's raw scale, mapped to the shared coordinate."""
        return self.coordinate(role).normalize(value)

    def normalize_difference(self, role: AlleleRole, value: float) -> float:
        """A DIFFERENCE of two readings on this allele's raw scale.

        Divided by the scale only. Subtracting the location as well would subtract it twice: the
        location cancels in any difference, and re-applying it would turn a zero difference into a
        non-zero coordinate.
        """
        return _finite(value, "difference") / self.coordinate(role).scale

    @property
    def joint_margin(self) -> float:
        r"""``max_a eps_a_raw / s_a``.

        The gradient of the smooth maximum with respect to ``u`` is the softmax weight vector: it is
        non-negative and sums to one, so ``J`` is 1-Lipschitz in the supremum norm on ``u`` and
        ``|dJ| <= max_a |du_a| = max_a |dR_a| / s_a``. Taking the worse allele is therefore the
        tightest bound that holds however the active worst switches, and it is derived from the
        objective's own geometry rather than declared.
        """
        return max(self.a.normalized_noise_floor, self.b.normalized_noise_floor)

    def coordinate(self, role: AlleleRole) -> AlleleCoordinate:
        return self.a if role is AlleleRole.A else self.b

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "quantity": self.quantity,
            "a": self.a.canonical_payload(),
            "b": self.b.canonical_payload(),
            "joint_margin": self.joint_margin,
        }


@dataclass(frozen=True)
class DualObjectiveLaw:
    """The reduction law: which mode, how wide its credit band, and how precise a panel it demands.

    Exactly one number here is chosen by a human: ``tau``. Everything else in the Dual calibration
    is measured in the panel pass or derived from the objective's own geometry.
    """

    mode: ObjectiveMode
    tau: float
    tau_units: str
    version: str
    #: The human decision this law was frozen from: ``c_u`` in NORMALIZED risk units, the most the
    #: non-worst allele may buy against the worse one. ``tau`` is derived as ``c_u / log 2`` and
    #: both are stored, so the loader can prove they are one decision rather than two numbers that
    #: happen to be near each other. ``None`` only for a law whose spec predates the field.
    declared_credit_normalized: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ObjectiveMode):
            raise V2JointObjectiveError("mode must be an ObjectiveMode")
        tau = _finite(self.tau, "tau")
        if tau <= 0.0:
            raise V2JointObjectiveError(f"tau must be strictly positive, got {tau}")
        if not isinstance(self.tau_units, str) or self.tau_units != NORMALIZED_UNITS:
            raise V2JointObjectiveError(
                f"tau_units must be exactly {NORMALIZED_UNITS!r}, got {self.tau_units!r}; because "
                "s_A != s_B in general, one normalized credit corresponds to two different raw "
                "credits, so a raw-scale tau would reintroduce the very asymmetry the "
                "normalization exists to remove"
            )
        _text(self.version, "version")
        if self.declared_credit_normalized is not None:
            declared = _finite(self.declared_credit_normalized, "declared_credit_normalized")
            if declared <= 0.0:
                raise V2JointObjectiveError(
                    f"declared_credit_normalized must be strictly positive, got {declared}")
            # Exact float equality would fail on a spec that wrote the credit and the tau
            # independently at full precision; a tolerance this tight still catches a real
            # disagreement (0.10 vs 0.11 is 1e-2, twelve orders above it).
            if abs(self.credit - declared) > 1e-12:
                raise V2JointObjectiveError(
                    f"the spec declares a credit of {declared} but tau={self.tau} gives "
                    f"tau*log2={self.credit}; one human decision has been stored two inconsistent "
                    "ways, and nothing downstream could say which one the run obeyed"
                )

    @property
    def credit(self) -> float:
        r"""``c_u = tau * log 2``: the most the non-worst allele can buy, in normalized units.

        Zero for the hard maximum, which gives the non-worst allele no credit at all until it
        reaches the active maximum.
        """
        return 0.0 if self.mode is ObjectiveMode.HARD_MAX else self.tau * LOG2

    def reduce(self, u_a: float, u_b: float) -> float:
        if self.mode is ObjectiveMode.HARD_MAX:
            return hard_max(u_a, u_b)
        return smooth_max(u_a, u_b, self.tau)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "tau": self.tau,
            "tau_units": self.tau_units,
            "credit": self.credit,
            "declared_credit_normalized": self.declared_credit_normalized,
            "version": self.version,
        }


#: Arm label -> the allele whose coordinate that arm STEERS on. ``joint`` is absent because it
#: steers on the reduction of both, which is the whole point of the comparison.
ARM_STEERING_ROLE = {"a_only": AlleleRole.A, "b_only": AlleleRole.B}


@dataclass(frozen=True)
class PanelBinding:
    """Identity of the fixed natural calibration panel plus the diagnostics that qualify it.

    ``equal_risk_line_stderr`` and ``leave_overlap_out_shift`` are both expressed in normalized
    units, because that is the space the credit lives in and the space the comparison must happen in.
    ``cross_allele_pearson`` is carried as a first-class output rather than a note: the argument that
    panel composition effects cancel is worth exactly as much as the two Heads are correlated, so a
    result that relies on that cancellation must state the number it relied on.
    """

    panel_id: str
    panel_digest: str
    n_proteins: int
    #: Reported, never gated. These are outputs of the one calibration pass, not knobs: the two
    #: overlap fractions and the leave-overlap-out shift say how much of the panel each Head had
    #: seen in training and how far removing it moves the equal-risk line; the standard error says
    #: how precisely the panel places that line; the correlation says how much of the
    #: common-mode cancellation argument the two Heads actually support. A run that did not measure
    #: one leaves it absent rather than zero.
    overlap_fraction_a: float | None = None
    overlap_fraction_b: float | None = None
    equal_risk_line_stderr: float | None = None
    leave_overlap_out_shift: float | None = None
    cross_allele_pearson: float | None = None

    def __post_init__(self) -> None:
        _text(self.panel_id, "panel_id")
        require_digest(self.panel_digest, "panel_digest")
        if isinstance(self.n_proteins, bool) or not isinstance(self.n_proteins, int) \
                or self.n_proteins < 1:
            raise V2JointObjectiveError(
                f"n_proteins must be a positive int, got {self.n_proteins!r}"
            )
        for name in ("overlap_fraction_a", "overlap_fraction_b"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= _finite(value, name) <= 1.0:
                raise V2JointObjectiveError(f"{name} must lie in [0, 1], got {value}")
        for name in ("equal_risk_line_stderr", "leave_overlap_out_shift"):
            value = getattr(self, name)
            if value is not None and _finite(value, name) < 0.0:
                raise V2JointObjectiveError(f"{name} must be non-negative, got {value}")
        if self.cross_allele_pearson is not None \
                and not -1.0 <= _finite(self.cross_allele_pearson, "cross_allele_pearson") <= 1.0:
            raise V2JointObjectiveError(
                f"cross_allele_pearson must lie in [-1, 1], got {self.cross_allele_pearson}"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "panel_digest": self.panel_digest,
            "n_proteins": self.n_proteins,
            "overlap_fraction_a": self.overlap_fraction_a,
            "overlap_fraction_b": self.overlap_fraction_b,
            "equal_risk_line_stderr": self.equal_risk_line_stderr,
            "leave_overlap_out_shift": self.leave_overlap_out_shift,
            "cross_allele_pearson": self.cross_allele_pearson,
        }


@dataclass(frozen=True)
class DualCalibration:
    """Everything frozen before a run: panel, coordinates, law. The single source of ``b_a, s_a``."""

    panel: PanelBinding
    risk: QuantityCoordinates
    density: QuantityCoordinates | None
    law: DualObjectiveLaw
    version: str
    #: Per-WINDOW coordinates, used only by the reopen union reducer. Required there and nowhere
    #: else, because the reopen conjuncts are per-window raw readings while ``risk`` is calibrated
    #: on the log-mean-exp AGGREGATE of those windows -- two different distributions on one raw
    #: scale, so the aggregate's location and scale do not transfer to the windows they summarize.
    window: QuantityCoordinates | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.panel, PanelBinding):
            raise V2JointObjectiveError("panel must be a PanelBinding")
        if not isinstance(self.risk, QuantityCoordinates):
            raise V2JointObjectiveError("risk must be a QuantityCoordinates")
        if self.risk.quantity != "global_risk":
            raise V2JointObjectiveError(
                f"the runtime steering coordinates must calibrate 'global_risk', got "
                f"{self.risk.quantity!r}"
            )
        if self.density is not None:
            if not isinstance(self.density, QuantityCoordinates):
                raise V2JointObjectiveError("density must be a QuantityCoordinates or None")
            if self.density.quantity != "positive_mass_density":
                raise V2JointObjectiveError(
                    f"the return-boundary coordinates must calibrate 'positive_mass_density', got "
                    f"{self.density.quantity!r}"
                )
            self._require_same_heads(self.risk, self.density)
        if self.window is not None:
            if not isinstance(self.window, QuantityCoordinates):
                raise V2JointObjectiveError("window must be a QuantityCoordinates or None")
            if self.window.quantity != "window_z":
                raise V2JointObjectiveError(
                    f"the reopen-reducer coordinates must calibrate 'window_z', got "
                    f"{self.window.quantity!r}"
                )
            self._require_same_heads(self.risk, self.window)
        if not isinstance(self.law, DualObjectiveLaw):
            raise V2JointObjectiveError("law must be a DualObjectiveLaw")
        _text(self.version, "version")

    @staticmethod
    def _require_same_heads(left: QuantityCoordinates, right: QuantityCoordinates) -> None:
        for role in (AlleleRole.A, AlleleRole.B):
            if left.coordinate(role).evaluator != right.coordinate(role).evaluator:
                raise V2JointObjectiveError(
                    f"role {role.value} is bound to different Head evaluators for "
                    f"{left.quantity!r} and {right.quantity!r}; both quantities are read off the "
                    "same score object and must come from the same instrument"
                )

    def coordinates(self, quantity: str) -> QuantityCoordinates:
        wanted = _quantity(quantity, "quantity")
        if wanted == "global_risk":
            return self.risk
        if wanted == "window_z":
            if self.window is None:
                raise V2JointObjectiveError(
                    "window_z was requested but the calibration carries no per-window coordinates; "
                    "the reopen union reducer compares per-window readings across two alleles and "
                    "cannot borrow the aggregate risk scale to do it"
                )
            return self.window
        if self.density is None:
            raise V2JointObjectiveError(
                "positive_mass_density was requested but the calibration carries no coordinates "
                "for it; the return-boundary front needs its own frozen pair, produced by the same "
                "calibration pass"
            )
        return self.density

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "panel": self.panel.canonical_payload(),
            "risk": self.risk.canonical_payload(),
            "density": None if self.density is None else self.density.canonical_payload(),
            "window": None if self.window is None else self.window.canonical_payload(),
            "law": self.law.canonical_payload(),
        }

    @property
    def content_digest(self) -> str:
        return canonical_digest(self.canonical_payload())


# ----------------------------------------------------------------------------------------------
# the resolver
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class JointValue:
    """One resolved objective value with the coordinates and the law identity that produced it."""

    quantity: str
    u_a: float
    u_b: float
    value: float
    mode: ObjectiveMode
    active_worst: AlleleRole
    objective_digest: str

    def rank_key(self, endpoint_id: str) -> tuple[float, str]:
        """``(J, endpoint_id)``: a total order in which exact ties are closed by identity.

        Endpoint identity, not row order, decides a tie -- so replaying the same pool in a different
        order selects the same endpoint.
        """
        return (self.value, _text(endpoint_id, "endpoint_id"))

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "quantity": self.quantity,
            "u_a": self.u_a,
            "u_b": self.u_b,
            "value": self.value,
            "mode": self.mode.value,
            "active_worst": self.active_worst.value,
            "objective_digest": self.objective_digest,
        }


class DualObjective:
    """Resolves one frozen calibration into joint values. Stateless apart from the calibration."""

    def __init__(self, calibration: DualCalibration, quantity: str = "global_risk") -> None:
        if not isinstance(calibration, DualCalibration):
            raise V2JointObjectiveError("calibration must be a DualCalibration")
        self._calibration = calibration
        self._quantity = _quantity(quantity, "quantity")
        self._coordinates = calibration.coordinates(self._quantity)

    @property
    def calibration(self) -> DualCalibration:
        return self._calibration

    @property
    def objective_digest(self) -> str:
        """Bind the ordering arm as well as the shared calibration."""
        return arm_objective_digest(self.calibration, self.arm)

    @property
    def quantity(self) -> str:
        return self._quantity

    @property
    def coordinates(self) -> QuantityCoordinates:
        return self._coordinates

    @property
    def joint_margin(self) -> float:
        """The 1-Lipschitz bound on ONE measurement's drift: ``max_a e_a / s_a``."""
        return self._coordinates.joint_margin

    @property
    def decision_margin(self) -> float:
        r"""The floor a DIFFERENCE of two joint values must clear: ``2 * max_a e_a / s_a``.

        The donor gate compares ``J(I_d)`` against ``J(Y*_d)`` and the write filter compares
        ``J(y^{(-i)})`` against ``J(y)``. Each side is its own measurement and carries its own
        drift ``d_a = e_a / s_a``, so their difference can drift by ``2 d_a`` -- the single
        measurement bound is too permissive by exactly a factor of two, which is the difference
        between a gate that admits instrument noise and one that does not.
        """
        return 2.0 * self.joint_margin

    def evaluate(self, *, raw_a: float, raw_b: float) -> JointValue:
        """Reduce one pair of raw same-sequence risks to a joint value."""
        u_a = self._coordinates.a.normalize(raw_a)
        u_b = self._coordinates.b.normalize(raw_b)
        value = self._calibration.law.reduce(u_a, u_b)
        return JointValue(
            quantity=self._quantity, u_a=u_a, u_b=u_b, value=value,
            mode=self._calibration.law.mode,
            active_worst=AlleleRole.A if u_a >= u_b else AlleleRole.B,
            objective_digest=self.objective_digest,
        )

    @property
    def arm(self) -> str:
        """Which arm's ordering law this objective IS. The joint objective is ``joint``."""
        return "joint"

    def evaluate_scores(self, *, score_a: Any, score_b: Any) -> JointValue:
        """Reduce two Head scores after proving they describe the same exact sequence.

        The two Heads must have scored one molecule on one window grid, otherwise the difference
        being reduced is between two objects rather than between two alleles.
        """
        self._require_same_sequence(score_a, score_b)
        self._require_role_identity(score_a, AlleleRole.A)
        self._require_role_identity(score_b, AlleleRole.B)
        attribute = QUANTITY_ATTRIBUTES[self._quantity]
        raw_a = getattr(score_a, attribute, None)
        raw_b = getattr(score_b, attribute, None)
        if raw_a is None or raw_b is None:
            raise V2JointObjectiveError(
                f"a Head score carries no {attribute!r}; the objective cannot be resolved from a "
                "score that does not report the quantity it declares"
            )
        return self.evaluate(raw_a=raw_a, raw_b=raw_b)

    @staticmethod
    def _require_same_sequence(score_a: Any, score_b: Any) -> None:
        for field in ("protein_id", "sequence_md5", "sequence_length", "window_grid_digest"):
            left = getattr(score_a, field, None)
            right = getattr(score_b, field, None)
            if left is None or right is None:
                raise V2JointObjectiveError(
                    f"both Head scores must carry {field!r} before they can be paired"
                )
            if left != right:
                raise V2JointObjectiveError(
                    f"the two Head scores disagree on {field!r} ({left!r} vs {right!r}); a joint "
                    "value is defined only for two alleles on ONE exact sequence and ONE window "
                    "grid, and pairing across sequences would compare two molecules"
                )

    def _require_role_identity(self, score: Any, role: AlleleRole) -> None:
        evaluator = self._coordinates.coordinate(role).evaluator
        allele = getattr(score, "allele", None)
        if allele != evaluator.allele:
            raise V2JointObjectiveError(
                f"the score supplied for role {role.value} declares allele {allele!r} but that "
                f"role is calibrated for {evaluator.allele!r}; a swapped pair would apply each "
                "allele's coordinate to the other allele's risk"
            )
        scale = getattr(score, "score_scale", None)
        if scale != evaluator.score_scale:
            raise V2JointObjectiveError(
                f"the score supplied for role {role.value} is on scale {scale!r} but the "
                f"coordinate was calibrated on {evaluator.score_scale!r}"
            )


# ----------------------------------------------------------------------------------------------
# the arms
# ----------------------------------------------------------------------------------------------

class SingleAlleleObjective(DualObjective):
    """The ordering law of an ``a_only`` / ``b_only`` arm: ONE allele's normalized coordinate.

    PLAN §7: "Arm labels name the objective and nothing else." Without this class the arm label was
    inert -- ``joint``, ``a_only`` and ``b_only`` all resolved the same ``DualObjective``, so C1's
    three-branch comparison contrasted the joint law against itself twice and every arm agreed for a
    reason that had nothing to do with the science.

    A subclass rather than a sibling, deliberately: every consumer (the support authority, the
    runtime, the rank key, the evidence binder) type-checks ``DualObjective`` and must keep working
    unchanged, because the arm is meant to be the ONLY difference between the branches.

    Both Heads are still scored, bound and published in every arm. An ``a_only`` arm is not a
    single-Head run: it measures role B and declines to steer on it, which is exactly what makes the
    contrast interpretable -- the counterfactual "what would role B have said" is recorded rather
    than unavailable.
    """

    def __init__(self, calibration: DualCalibration, role: AlleleRole,
                 quantity: str = "global_risk") -> None:
        super().__init__(calibration, quantity)
        if not isinstance(role, AlleleRole):
            raise V2JointObjectiveError("role must be an AlleleRole")
        self._role = role
        self._arm = "a_only" if role is AlleleRole.A else "b_only"

    @property
    def role(self) -> AlleleRole:
        return self._role

    @property
    def arm(self) -> str:
        return self._arm

    @property
    def joint_margin(self) -> float:
        """This arm's OWN propagated floor: ``eps_role_raw / s_role``, not the worse of the two.

        ``decision_margin`` doubles it for the same reason it doubles for the joint arm: the gates
        compare a difference of two measurements.

        An ``a_only`` arm decides on ``u_A``, so the noise it must clear is role A's alone. Taking
        the max over both alleles here would hold the single-allele arms to a threshold set by an
        instrument they do not steer on, and the three arms would differ in their gate WIDTH as well
        as in their ordering law -- two treatments where the design declares one.
        """
        return self.coordinates.coordinate(self._role).normalized_noise_floor

    def evaluate(self, *, raw_a: float, raw_b: float) -> JointValue:
        """Both coordinates are computed and published; only one of them orders."""
        u_a = self.coordinates.a.normalize(raw_a)
        u_b = self.coordinates.b.normalize(raw_b)
        value = u_a if self._role is AlleleRole.A else u_b
        return JointValue(
            quantity=self.quantity, u_a=u_a, u_b=u_b, value=value,
            mode=self.calibration.law.mode,
            # The allele that set the value IS the steering allele here -- there is no contest.
            active_worst=self._role,
            objective_digest=self.objective_digest,
        )


def arm_objective_digest(calibration: DualCalibration, arm: str) -> str:
    """The objective identity of one arm.

    The arm is folded in because two arms of one matched comparison share a calibration and must NOT
    share an objective identity: a joint value and an a_only value are two different measurements,
    and an artifact that stamped them with the same digest could not be split after the fact.
    """
    return canonical_digest({"calibration": calibration.content_digest, "arm": _text(arm, "arm")})


def build_arm_objective(calibration: DualCalibration, *, arm: str,
                        quantity: str = "global_risk") -> DualObjective:
    """Resolve the ordering law an arm label names. This is where the label stops being a label."""
    label = _text(arm, "arm")
    if label == "joint":
        return DualObjective(calibration, quantity)
    role = ARM_STEERING_ROLE.get(label)
    if role is None:
        raise V2JointObjectiveError(
            f"arm={label!r} names no ordering law; declared arms are "
            f"{sorted(['joint', *ARM_STEERING_ROLE])}"
        )
    return SingleAlleleObjective(calibration, role, quantity)
