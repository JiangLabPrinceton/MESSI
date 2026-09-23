r"""DUALF5: the strict optional Dual overlay - the single carrier of Dual identity.

Implementation contract: ``PLAN_RF_FUSION_V2_DUAL_ALLELE.md`` §3.1, §3.2.

Everything Dual knows about itself lives here and nowhere else. That is not tidiness; it is the only
shape in which the compatibility rule can hold. ``V2Config.canonical_payload()`` emits the Head block
as ``vars(self.head)``, so a single new field on ``V2HeadConfig`` -- optional, ``None``-defaulted,
anything -- moves the canonical digest of every legacy config, and with it every run signature and
every resume fragment, with no test failing anywhere near the change. The content-role vocabulary is
likewise closed and every role is mandatory, so a new role makes every legacy config fail to load and
moves every ``endpoint_id``. Neither is touched. The overlay is a separate signed record whose digest
becomes one optional run-signature component.

**Paths are not identity.** The overlay binds Head B by CONTENT -- allele, checkpoint digest, config
hash, score scale, window k-range -- and never by filesystem path. Runtime paths arrive through the
CLI and are verified against these digests, which is the discipline the rest of V2 already uses and
the reason a cluster path never appears in a module.

**Purity.** stdlib plus the ``fusion_v2`` leaves. No torch, no config import, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import V2Error
from .identity import HeadEvaluatorIdentity, canonical_digest, require_digest
from .joint_objective import (
    AlleleCoordinate,
    AlleleRole,
    DualCalibration,
    DualObjectiveLaw,
    ObjectiveMode,
    PanelBinding,
    QuantityCoordinates,
)

__all__ = [
    "V2DualConfigError",
    "DUAL_OVERLAY_SCHEMA_VERSION",
    "ARM_LABELS",
    "MATCHED_ARM_BUNDLES",
    "HeadRuntimeBinding",
    "DualOverlay",
    "load_dual_overlay",
    "dual_overlay_from_mapping",
    "dual_overlay_payload",
    "dual_run_signature_component",
]

#: Bumped whenever the overlay's canonical payload changes, so an old signature can never silently
#: validate against a new schema.
DUAL_OVERLAY_SCHEMA_VERSION = "dualcfg-2"

#: The frozen arm vocabulary. Lowercase, identical in both experiments, and load-bearing: these
#: strings appear in the matched-arm bundle, the run signature, the resume fragment key and every
#: Dual artifact. An arm label names the OBJECTIVE and nothing else -- which safety contract is in
#: force is a separately declared property of the run, not a suffix on the arm.
ARM_LABELS = ("joint", "a_only", "b_only")

#: The only arm bundles a run may declare. A one-cycle directionality experiment contrasts the joint
#: law against the incumbent single-allele law; a recursive capability comparison adds the other
#: single-allele arm so that "the method merely switched which allele it optimizes" is separable
#: from "the joint law is better than both".
MATCHED_ARM_BUNDLES = (
    ("joint", "a_only"),
    ("joint", "a_only", "b_only"),
)


class V2DualConfigError(V2Error):
    """A Dual overlay contract was violated."""


def _text(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise V2DualConfigError(f"{name} must be a non-empty str, got {value!r}")
    return value


def _index(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise V2DualConfigError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


@dataclass(frozen=True)
class HeadRuntimeBinding:
    """How role B is IDENTIFIED, plus the two runtime knobs a Head scorer needs.

    ``allele_idx`` and ``window_batch_size`` are declared rather than defaulted. Both are read from
    shard inputs today with silent fallbacks of 0 and 64, which means a run's Head configuration can
    differ from the one its manifest appears to describe without anything noticing. Requiring them
    here makes the value a signed part of the overlay.
    """

    evaluator: HeadEvaluatorIdentity
    variant_id: str
    allele_idx: int
    window_batch_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator, HeadEvaluatorIdentity):
            raise V2DualConfigError("evaluator must be a HeadEvaluatorIdentity")
        _text(self.variant_id, "variant_id")
        _index(self.allele_idx, "allele_idx", minimum=0)
        _index(self.window_batch_size, "window_batch_size", minimum=1)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "evaluator": self.evaluator.canonical_payload(),
            "variant_id": self.variant_id,
            "allele_idx": self.allele_idx,
            "window_batch_size": self.window_batch_size,
        }


@dataclass(frozen=True)
class DualOverlay:
    """The one signed record that turns a legacy V2 run into a Dual run."""

    schema_version: str
    calibration: DualCalibration
    head_b_runtime: HeadRuntimeBinding
    arm_bundle: tuple[str, ...]
    #: The per-cycle CANDIDATE/SEQUENCE ceiling for a Dual cycle, in editable positions.
    #:
    #: Declared here rather than reusing the legacy ``max_counterfactual_head_calls_per_cycle``,
    #: whose name says Head calls and whose gate counts candidate POSITIONS -- identical numbers
    #: under one Head and different ones under two. Charging that legacy field in Head calls would
    #: halve the editable candidate domain because a second Head exists, which is a scientific
    #: change nobody asked for: the domain a cycle may consider is a property of the design, not of
    #: how many instruments watch it. The logical Head-call budget is ``2 * C`` and is projected by
    #: the preflight, never by re-reading this number.
    #:
    #: All three arms bill ``2 * C``: every arm scores both Heads and executes joint safety, so the
    #: cost and the candidate domain are identical across the comparison.
    max_counterfactual_sequences_per_cycle: int
    #: Digest of the tracked objective-law spec file this overlay was resolved from, so a
    #: hand-edited spec cannot be presented as the signed one.
    objective_spec_digest: str
    #: Digest of the measured calibration artifact, likewise.
    calibration_artifact_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != DUAL_OVERLAY_SCHEMA_VERSION:
            raise V2DualConfigError(
                f"schema_version must be {DUAL_OVERLAY_SCHEMA_VERSION!r}, got "
                f"{self.schema_version!r}; an overlay written against another schema cannot be "
                "validated against this one"
            )
        if not isinstance(self.calibration, DualCalibration):
            raise V2DualConfigError("calibration must be a DualCalibration")
        if not isinstance(self.head_b_runtime, HeadRuntimeBinding):
            raise V2DualConfigError("head_b_runtime must be a HeadRuntimeBinding")

        bundle = tuple(self.arm_bundle)
        if bundle not in MATCHED_ARM_BUNDLES:
            raise V2DualConfigError(
                f"arm_bundle {bundle!r} is not a declared matched bundle; available: "
                f"{[list(b) for b in MATCHED_ARM_BUNDLES]}. An ad hoc bundle would make the "
                "comparison's own denominator a run-time choice"
            )

        role_b = self.calibration.risk.coordinate(AlleleRole.B).evaluator
        if self.head_b_runtime.evaluator != role_b:
            raise V2DualConfigError(
                f"the runtime Head B binding names {self.head_b_runtime.evaluator.allele!r}/"
                f"{self.head_b_runtime.evaluator.head_checkpoint_digest[:12]} but the calibration "
                f"was measured on {role_b.allele!r}/{role_b.head_checkpoint_digest[:12]}; the "
                "coordinates would be applied to a Head that did not produce them"
            )
        role_a = self.calibration.risk.coordinate(AlleleRole.A).evaluator
        if role_a.head_checkpoint_digest == role_b.head_checkpoint_digest:
            raise V2DualConfigError("both roles carry the same Head checkpoint digest")

        require_digest(self.objective_spec_digest, "objective_spec_digest")
        require_digest(self.calibration_artifact_digest, "calibration_artifact_digest")

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(self.arm_bundle)

    @property
    def is_recursive_bundle(self) -> bool:
        return len(self.arm_bundle) == 3

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calibration": self.calibration.canonical_payload(),
            "calibration_digest": self.calibration.content_digest,
            "head_b_runtime": self.head_b_runtime.canonical_payload(),
            "arm_bundle": list(self.arm_bundle),
            "max_counterfactual_sequences_per_cycle":
                int(self.max_counterfactual_sequences_per_cycle),
            "objective_spec_digest": self.objective_spec_digest,
            "calibration_artifact_digest": self.calibration_artifact_digest,
        }

    @property
    def content_digest(self) -> str:
        """The single value that carries Dual identity into the run signature."""
        return canonical_digest(self.canonical_payload())

    def assert_observed_head_b(self, *, checkpoint_digest: str, config_hash: str) -> None:
        """Verify a runtime-resolved Head B against the identity the overlay signed.

        Paths arrive from the CLI; this is where a path is proved to be the object the overlay
        names. The checkpoint digest is the ONLY discriminator between the two production Heads --
        they share a config directory and their checkpoint metadata config hashes are identical --
        so it is checked first and its failure is the loud one.
        """
        expected = self.head_b_runtime.evaluator
        if str(checkpoint_digest) != expected.head_checkpoint_digest:
            raise V2DualConfigError(
                f"the resolved Head B checkpoint digests to {str(checkpoint_digest)[:12]} but the "
                f"overlay signed {expected.head_checkpoint_digest[:12]}; the checkpoint digest is "
                "the only bit that distinguishes the two production Heads, so this is a Head mixup "
                "rather than a bookkeeping difference"
            )
        if str(config_hash) != expected.head_config_hash:
            raise V2DualConfigError(
                f"the resolved Head B config hashes to {str(config_hash)[:12]} but the overlay "
                f"signed {expected.head_config_hash[:12]}"
            )


def load_dual_overlay(node: Mapping[str, Any]) -> DualOverlay:
    """Strict loader. Unknown keys are refused; nothing has a scientific default."""
    if not isinstance(node, Mapping):
        raise V2DualConfigError("the Dual overlay must be a mapping")
    allowed = {"schema_version", "calibration", "head_b_runtime", "arm_bundle",
               "max_counterfactual_sequences_per_cycle",
               "objective_spec_digest", "calibration_artifact_digest"}
    unknown = sorted(set(node) - allowed)
    if unknown:
        raise V2DualConfigError(
            f"unknown Dual overlay key(s) {unknown}; a silently ignored key is a declared "
            "scientific choice that never took effect"
        )
    missing = sorted(allowed - set(node))
    if missing:
        raise V2DualConfigError(f"missing Dual overlay key(s) {missing}")
    return DualOverlay(
        max_counterfactual_sequences_per_cycle=_index(
            node["max_counterfactual_sequences_per_cycle"],
            "max_counterfactual_sequences_per_cycle", minimum=1),
        schema_version=_text(node["schema_version"], "schema_version"),
        calibration=node["calibration"],
        head_b_runtime=node["head_b_runtime"],
        arm_bundle=tuple(node["arm_bundle"]),
        objective_spec_digest=_text(node["objective_spec_digest"], "objective_spec_digest"),
        calibration_artifact_digest=_text(
            node["calibration_artifact_digest"], "calibration_artifact_digest"),
    )


# ----------------------------------------------------------------------------------------------
# from-mapping construction
# ----------------------------------------------------------------------------------------------
#
# The overlay is authored as JSON so it can be signed, diffed and shipped, but this module performs
# no I/O: it takes an already-parsed mapping and returns typed objects. The file read lives in the
# script layer, which is where every other "read a file, load no model" step in V2 already lives.
#
# Every builder is strict in both directions -- unknown keys are refused, missing keys are refused --
# because a silently ignored key is a declared scientific choice that never took effect, and a
# silently defaulted one is a choice nobody made.


def _section(node: Mapping[str, Any], name: str, path: str) -> Mapping[str, Any]:
    if name not in node:
        raise V2DualConfigError(f"{path}.{name} is required")
    value = node[name]
    if not isinstance(value, Mapping):
        raise V2DualConfigError(f"{path}.{name} must be a mapping, got {type(value).__name__}")
    return value


def _exact_keys(node: Mapping[str, Any], keys: tuple[str, ...], path: str) -> None:
    unknown = sorted(set(node) - set(keys))
    if unknown:
        raise V2DualConfigError(f"unknown key(s) {unknown} in {path}")
    missing = sorted(set(keys) - set(node))
    if missing:
        raise V2DualConfigError(f"missing key(s) {missing} in {path}")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V2DualConfigError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


_EVALUATOR_KEYS = ("allele", "score_scale", "window_k_min", "window_k_max",
                   "head_config_hash", "head_checkpoint_digest")


def _evaluator_from(node: Mapping[str, Any], path: str) -> HeadEvaluatorIdentity:
    _exact_keys(node, _EVALUATOR_KEYS, path)
    return HeadEvaluatorIdentity(
        allele=_text(node["allele"], f"{path}.allele"),
        score_scale=_text(node["score_scale"], f"{path}.score_scale"),
        window_k_min=_index(node["window_k_min"], f"{path}.window_k_min", minimum=1),
        window_k_max=_index(node["window_k_max"], f"{path}.window_k_max", minimum=1),
        head_config_hash=_text(node["head_config_hash"], f"{path}.head_config_hash"),
        head_checkpoint_digest=_text(node["head_checkpoint_digest"],
                                     f"{path}.head_checkpoint_digest"),
    )


_COORDINATE_KEYS = ("location", "scale", "raw_noise_floor", "evaluator", "source_ref")


def _coordinate_from(node: Mapping[str, Any], *, role: AlleleRole, quantity: str,
                     path: str) -> AlleleCoordinate:
    _exact_keys(node, _COORDINATE_KEYS, path)
    return AlleleCoordinate(
        role=role, quantity=quantity,
        location=_number(node["location"], f"{path}.location"),
        scale=_number(node["scale"], f"{path}.scale"),
        raw_noise_floor=_number(node["raw_noise_floor"], f"{path}.raw_noise_floor"),
        evaluator=_evaluator_from(_section(node, "evaluator", path), f"{path}.evaluator"),
        source_ref=_text(node["source_ref"], f"{path}.source_ref"),
    )


def _quantity_from(node: Mapping[str, Any], *, quantity: str, path: str) -> QuantityCoordinates:
    _exact_keys(node, ("a", "b"), path)
    return QuantityCoordinates(
        quantity=quantity,
        a=_coordinate_from(_section(node, "a", path), role=AlleleRole.A, quantity=quantity,
                           path=f"{path}.a"),
        b=_coordinate_from(_section(node, "b", path), role=AlleleRole.B, quantity=quantity,
                           path=f"{path}.b"),
    )


_PANEL_REQUIRED = ("panel_id", "panel_digest", "n_proteins")
#: Measured in the calibration pass and reported; a run that did not measure one omits it.
_PANEL_DIAGNOSTICS = ("overlap_fraction_a", "overlap_fraction_b", "equal_risk_line_stderr",
                      "leave_overlap_out_shift", "cross_allele_pearson")

#: ``declared_credit_normalized`` is OPTIONAL on read so an overlay written before the field
#: existed still loads, but it is always written -- without it the law's credit cross-check is
#: silently skipped on every round trip, which is a guard that exists only where it is never needed.
_LAW_KEYS = ("mode", "tau", "tau_units", "version", "declared_credit_normalized")
_LAW_REQUIRED = ("mode", "tau", "tau_units", "version")

_CALIBRATION_KEYS = ("version", "panel", "risk", "density", "window", "law")


def _calibration_from(node: Mapping[str, Any], path: str) -> DualCalibration:
    _exact_keys(node, _CALIBRATION_KEYS, path)
    panel_node = _section(node, "panel", path)
    unknown = sorted(set(panel_node) - set(_PANEL_REQUIRED) - set(_PANEL_DIAGNOSTICS))
    if unknown:
        raise V2DualConfigError(f"unknown key(s) {unknown} in {path}.panel")
    missing = sorted(set(_PANEL_REQUIRED) - set(panel_node))
    if missing:
        raise V2DualConfigError(f"missing key(s) {missing} in {path}.panel")
    law_node = _section(node, "law", path)
    unknown_law = sorted(set(law_node) - set(_LAW_KEYS))
    if unknown_law:
        raise V2DualConfigError(f"unknown key(s) {unknown_law} in {path}.law")
    missing_law = sorted(set(_LAW_REQUIRED) - set(law_node))
    if missing_law:
        raise V2DualConfigError(f"missing key(s) {missing_law} in {path}.law")
    mode = _text(law_node["mode"], f"{path}.law.mode")
    if mode not in {m.value for m in ObjectiveMode}:
        raise V2DualConfigError(
            f"{path}.law.mode must be one of {sorted(m.value for m in ObjectiveMode)}, got {mode!r}"
        )
    optional = {}
    for name, quantity in (("density", "positive_mass_density"), ("window", "window_z")):
        raw = node[name]
        optional[name] = (None if raw is None
                          else _quantity_from(raw, quantity=quantity, path=f"{path}.{name}"))
    return DualCalibration(
        version=_text(node["version"], f"{path}.version"),
        panel=PanelBinding(
            panel_id=_text(panel_node["panel_id"], f"{path}.panel.panel_id"),
            panel_digest=_text(panel_node["panel_digest"], f"{path}.panel.panel_digest"),
            n_proteins=_index(panel_node["n_proteins"], f"{path}.panel.n_proteins", minimum=1),
            **{name: (None if panel_node.get(name) is None
                      else _number(panel_node[name], name))
               for name in _PANEL_DIAGNOSTICS},
        ),
        risk=_quantity_from(_section(node, "risk", path), quantity="global_risk",
                            path=f"{path}.risk"),
        density=optional["density"],
        window=optional["window"],
        law=DualObjectiveLaw(
            mode=ObjectiveMode(mode),
            tau=_number(law_node["tau"], f"{path}.law.tau"),
            tau_units=_text(law_node["tau_units"], f"{path}.law.tau_units"),
            version=_text(law_node["version"], f"{path}.law.version"),
            declared_credit_normalized=(
                None if law_node.get("declared_credit_normalized") is None
                else _number(law_node["declared_credit_normalized"],
                             f"{path}.law.declared_credit_normalized")),
        ),
    )


_RUNTIME_KEYS = ("evaluator", "variant_id", "allele_idx", "window_batch_size")


def dual_overlay_from_mapping(node: Mapping[str, Any]) -> DualOverlay:
    """Build a fully typed overlay from a parsed mapping, refusing anything under-specified."""
    if not isinstance(node, Mapping):
        raise V2DualConfigError("the Dual overlay must be a mapping")
    _exact_keys(node, ("schema_version", "calibration", "head_b_runtime", "arm_bundle",
                       "max_counterfactual_sequences_per_cycle",
                       "objective_spec_digest", "calibration_artifact_digest"), "dual")
    runtime_node = _section(node, "head_b_runtime", "dual")
    _exact_keys(runtime_node, _RUNTIME_KEYS, "dual.head_b_runtime")
    bundle = node["arm_bundle"]
    if not isinstance(bundle, (list, tuple)):
        raise V2DualConfigError("dual.arm_bundle must be a list")
    return DualOverlay(
        schema_version=_text(node["schema_version"], "dual.schema_version"),
        calibration=_calibration_from(_section(node, "calibration", "dual"), "dual.calibration"),
        head_b_runtime=HeadRuntimeBinding(
            evaluator=_evaluator_from(_section(runtime_node, "evaluator", "dual.head_b_runtime"),
                                      "dual.head_b_runtime.evaluator"),
            variant_id=_text(runtime_node["variant_id"], "dual.head_b_runtime.variant_id"),
            allele_idx=_index(runtime_node["allele_idx"], "dual.head_b_runtime.allele_idx"),
            window_batch_size=_index(runtime_node["window_batch_size"],
                                     "dual.head_b_runtime.window_batch_size", minimum=1),
        ),
        arm_bundle=tuple(str(label) for label in bundle),
        max_counterfactual_sequences_per_cycle=_index(
            node["max_counterfactual_sequences_per_cycle"],
            "dual.max_counterfactual_sequences_per_cycle", minimum=1),
        objective_spec_digest=_text(node["objective_spec_digest"], "dual.objective_spec_digest"),
        calibration_artifact_digest=_text(node["calibration_artifact_digest"],
                                          "dual.calibration_artifact_digest"),
    )


def dual_overlay_payload(overlay: DualOverlay) -> dict[str, Any]:
    """Serialize an overlay into exactly what :func:`dual_overlay_from_mapping` accepts.

    Distinct from ``canonical_payload``, which exists to be DIGESTED and therefore carries derived
    quantities (``joint_margin``, ``credit``, the calibration digest).  Feeding that back to the
    strict authoring loader is refused, and rightly so -- a derived value present in a hand-written
    file is a second, editable source for something the objective computes.

    Without this the overlay had no round trip at all: the runbook asks an operator to produce the
    file, and nothing could check that what they produced is what the loader would read back.
    """
    if not isinstance(overlay, DualOverlay):
        raise V2DualConfigError("dual_overlay_payload takes a DualOverlay")

    def evaluator(value: HeadEvaluatorIdentity) -> dict[str, Any]:
        return {name: getattr(value, name) for name in _EVALUATOR_KEYS}

    def coordinate(value: AlleleCoordinate) -> dict[str, Any]:
        return {
            "location": float(value.location), "scale": float(value.scale),
            "raw_noise_floor": float(value.raw_noise_floor),
            "evaluator": evaluator(value.evaluator), "source_ref": value.source_ref,
        }

    def quantity(value: QuantityCoordinates | None) -> dict[str, Any] | None:
        return None if value is None else {"a": coordinate(value.a), "b": coordinate(value.b)}

    panel = overlay.calibration.panel
    panel_node: dict[str, Any] = {name: getattr(panel, name) for name in _PANEL_REQUIRED}
    for name in _PANEL_DIAGNOSTICS:
        measured = getattr(panel, name, None)
        # Omitted rather than nulled: "not measured" and "measured as zero" are different claims.
        if measured is not None:
            panel_node[name] = measured
    law = overlay.calibration.law
    return {
        "schema_version": overlay.schema_version,
        "calibration": {
            "version": overlay.calibration.version,
            "panel": panel_node,
            "risk": quantity(overlay.calibration.risk),
            "density": quantity(overlay.calibration.density),
            "window": quantity(overlay.calibration.window),
            "law": {"mode": law.mode.value, "tau": float(law.tau),
                    "tau_units": law.tau_units, "version": law.version,
                    # Written ALWAYS. Dropped, the law's credit cross-check is skipped on every
                    # round trip and exists only in the producer that never needed it.
                    "declared_credit_normalized": (
                        None if law.declared_credit_normalized is None
                        else float(law.declared_credit_normalized))},
        },
        "head_b_runtime": {
            "evaluator": evaluator(overlay.head_b_runtime.evaluator),
            "variant_id": overlay.head_b_runtime.variant_id,
            "allele_idx": int(overlay.head_b_runtime.allele_idx),
            "window_batch_size": int(overlay.head_b_runtime.window_batch_size),
        },
        "arm_bundle": list(overlay.arm_bundle),
        "max_counterfactual_sequences_per_cycle":
            int(overlay.max_counterfactual_sequences_per_cycle),
        "objective_spec_digest": overlay.objective_spec_digest,
        "calibration_artifact_digest": overlay.calibration_artifact_digest,
    }


def dual_run_signature_component(overlay: DualOverlay, *, arm: str) -> str:
    """The single opaque string the run signature carries for a Dual run.

    The EXECUTING arm is folded in, not just the overlay. ``DualOverlay.content_digest`` digests the
    arm *bundle*, so it is constant across the arms of one matched comparison -- and two arms of one
    protein that shared a signature would be indistinguishable to resume, which is exactly the
    collision a matched comparison must not have.
    """
    if not isinstance(overlay, DualOverlay):
        raise V2DualConfigError("overlay must be a DualOverlay")
    label = _text(arm, "arm")
    if label not in overlay.arm_bundle:
        raise V2DualConfigError(
            f"arm {label!r} is not in this overlay's declared bundle {list(overlay.arm_bundle)}; "
            "an arm outside the bundle would be a comparison the run never declared"
        )
    return canonical_digest({
        "schema": DUAL_OVERLAY_SCHEMA_VERSION,
        "overlay": overlay.content_digest,
        "arm": label,
    })
