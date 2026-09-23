"""DUALF6 evidence item 3, pinned early: with no Dual overlay, nothing about V2 changed.

PLAN §0.3 makes Dual opt-in and requires that a run without the overlay keep byte- or
value-equivalent config digests, endpoint identities, orderings and decisions. Those guarantees are
easy to state and easy to break silently -- ``V2Config.canonical_payload()`` serializes the Head
block as ``vars(self.head)``, so ANY field added to ``V2HeadConfig``, even optional and defaulting to
None, moves the digest of every legacy config and therefore every run signature and every resume
fragment, with no test failing anywhere near the change.

So the digests are pinned to literals here. A literal is the only form of this assertion that
cannot be satisfied by the same mistake on both sides.
"""

from __future__ import annotations

import pytest

from inverse_folding.reference_flow.fusion_v2 import reward as rw
from inverse_folding.reference_flow.fusion_v2_runtime import archive as arch

from . import _v2_fixtures as fx

#: Pinned 2026-08-21, before any Dual work touched the config schema.
FROZEN_CONFIG_DIGEST = "dfc5cd3ac845552de9d5aff4d05a7590d10137b862f1ecffe021cb45261f13e4"
FROZEN_ENDPOINT_ID = "5ZHV_B:fam0:live:d0:c50:d89df5c86bc5:endpoint:k0:6dfa8bc5a2c9"
FROZEN_ENDPOINT_CONTENT_DIGEST = "25ae9a5216650d01e415bb42b3184b8ec70d4987753c2b2cf133c66836c5905b"


def test_the_legacy_config_digest_has_not_moved():
    assert fx.v2_config().config_digest() == FROZEN_CONFIG_DIGEST


def test_the_legacy_endpoint_identity_has_not_moved():
    endpoint = fx.endpoint()
    assert endpoint.endpoint_id == FROZEN_ENDPOINT_ID
    assert endpoint.content_digest == FROZEN_ENDPOINT_CONTENT_DIGEST


def test_the_head_config_block_still_carries_exactly_its_four_frozen_fields():
    # vars(self.head) IS the canonical payload of the head block, so this set is the digest.
    assert sorted(vars(fx.v2_config().head)) == [
        "allele", "score_scale", "window_k_max", "window_k_min"]


def test_the_default_archive_ordering_is_still_the_raw_single_head_risk():
    endpoint = fx.endpoint()
    assert arch.legacy_rank_key(endpoint) == (
        float(endpoint.head_global_risk), endpoint.endpoint_id)


def test_a_donor_gate_called_without_a_joint_view_records_no_joint_view():
    endpoint = fx.endpoint()
    incumbent = rw.bind_incumbent_from_endpoint(
        endpoint=endpoint, lineage_id=endpoint.lineage.family_id,
        evaluator=endpoint.head_binding.evaluator,
        safety_reference_sequence_md5="e" * 32, accepted_at_depth=0)
    verdict = rw.donor_gate(donor=endpoint, incumbent=incumbent, epsilon_r=0.005,
                            epsilon_source_ref=fx.digest("eps"))
    assert verdict.joint is None
    # ABSENT, not null. The payload is digested, so a key that appears on every legacy verdict --
    # even carrying None -- is a new digest for every run that predates Dual.
    assert "joint" not in verdict.canonical_payload()
    # the donor IS the incumbent here, so the gate refuses -- unchanged behaviour
    assert verdict.reason is rw.DonorGateReason.DONOR_IS_INCUMBENT


# ------------------------------------------------------------------------------------------
# the run signature gained one self-omitting field
# ------------------------------------------------------------------------------------------

def _signature(**over):
    from scripts.rf_fusion_v2_resume import RunSignature

    kw = dict(config_digest="c", campaign_id="m", split_role="s", arm_role="a",
              protein_id="p", input_signature="i", code_revision="abc1234")
    kw.update(over)
    return RunSignature(**kw)


#: Pinned 2026-08-22 against the pre-Dual field set. A run with no Dual overlay must hash to this.
FROZEN_LEGACY_RUN_SIGNATURE = \
    "5496d5e473401008c99578162eded5ba3a0db25e36ada4929e35459abbf33686"


def test_a_run_with_no_dual_overlay_hashes_exactly_as_before():
    assert _signature().value == FROZEN_LEGACY_RUN_SIGNATURE


def test_the_dual_field_is_omitted_from_the_payload_when_empty():
    assert "dual_signature" not in _signature().canonical_payload()
    assert "dual_signature" in _signature(dual_signature="deadbeef").canonical_payload()
    assert _signature(dual_signature="deadbeef").value != FROZEN_LEGACY_RUN_SIGNATURE


def test_a_fragment_written_before_the_field_existed_is_still_reusable():
    legacy_stored = _signature().canonical_payload()
    assert "dual_signature" not in legacy_stored
    assert _signature().mismatch_status(legacy_stored) is None


def test_a_dual_run_may_not_reuse_a_legacy_fragment_and_the_reverse():
    legacy_stored = _signature().canonical_payload()
    dual_stored = _signature(dual_signature="deadbeef").canonical_payload()
    assert _signature(dual_signature="deadbeef").mismatch_status(legacy_stored) == "foreign_run"
    assert _signature().mismatch_status(dual_stored) == "foreign_run"


def test_two_arms_of_one_bundle_do_not_share_a_signature():
    """The executing arm is folded into the component, not just the overlay.

    ``DualOverlay.content_digest`` digests the arm BUNDLE, so it is constant across the arms of one
    matched comparison. Without the arm, two arms of the same protein would be indistinguishable to
    resume -- which is exactly the collision a matched comparison must not have.
    """
    from inverse_folding.reference_flow.fusion_v2 import dual_config as dc
    from .test_fusion_v2_dual_config import overlay

    over = overlay(arm_bundle=("joint", "a_only", "b_only"))
    components = {arm: dc.dual_run_signature_component(over, arm=arm) for arm in over.arms}
    assert len(set(components.values())) == 3
    assert len({_signature(dual_signature=v).value for v in components.values()}) == 3
    with pytest.raises(dc.V2DualConfigError):
        dc.dual_run_signature_component(over, arm="not_an_arm")


def _authoring_mapping(over):
    """The shape a real overlay JSON is AUTHORED in.

    Deliberately not ``canonical_payload()``: that projection carries DERIVED values -- the credit
    tau*log2, the propagated joint margin -- which belong in the digest so a change in the
    derivation is visible, but must never be accepted back as inputs. A hand-edited credit
    inconsistent with its own tau would otherwise be indistinguishable from a signed one.
    """
    def coordinate(c):
        return {"location": c.location, "scale": c.scale, "raw_noise_floor": c.raw_noise_floor,
                "evaluator": dict(c.evaluator.canonical_payload(), **{}), "source_ref": c.source_ref}

    def pair(q):
        if q is None:
            return None
        return {"a": coordinate(q.a), "b": coordinate(q.b)}

    def evaluator(e):
        node = e.canonical_payload()
        node.pop("schema", None)
        return node

    def coordinate_clean(c):
        node = coordinate(c)
        node["evaluator"] = evaluator(c.evaluator)
        return node

    def pair_clean(q):
        return None if q is None else {"a": coordinate_clean(q.a), "b": coordinate_clean(q.b)}

    cal = over.calibration
    return {
        "schema_version": over.schema_version,
        "calibration": {
            "version": cal.version,
            "panel": cal.panel.canonical_payload(),
            "risk": pair_clean(cal.risk),
            "density": pair_clean(cal.density),
            "window": pair_clean(cal.window),
            "law": {"mode": cal.law.mode.value, "tau": cal.law.tau,
                    "tau_units": cal.law.tau_units, "version": cal.law.version},
        },
        "head_b_runtime": {
            "evaluator": evaluator(over.head_b_runtime.evaluator),
            "variant_id": over.head_b_runtime.variant_id,
            "allele_idx": over.head_b_runtime.allele_idx,
            "window_batch_size": over.head_b_runtime.window_batch_size,
        },
        "arm_bundle": list(over.arm_bundle),
        "max_counterfactual_sequences_per_cycle":
            over.max_counterfactual_sequences_per_cycle,
        "objective_spec_digest": over.objective_spec_digest,
        "calibration_artifact_digest": over.calibration_artifact_digest,
    }


def test_the_overlay_round_trips_through_a_plain_mapping():
    """The overlay is authored as JSON, so a from-mapping path must exist and be strict."""
    import json

    from inverse_folding.reference_flow.fusion_v2 import dual_config as dc
    from .test_fusion_v2_dual_config import overlay

    original = overlay()
    node = json.loads(json.dumps(_authoring_mapping(original)))
    assert dc.dual_overlay_from_mapping(node).content_digest == original.content_digest


def test_a_derived_value_may_not_be_authored_back_in():
    """The digest carries the credit and the joint margin; the author may not supply them.

    Accepting them would let a hand-edited credit that contradicts its own tau be presented as a
    signed overlay, and the digest would agree with the file rather than with the law.
    """
    from inverse_folding.reference_flow.fusion_v2 import dual_config as dc
    from .test_fusion_v2_dual_config import overlay

    node = _authoring_mapping(overlay())
    node["calibration"]["law"]["credit"] = 999.0
    with pytest.raises(dc.V2DualConfigError, match="unknown key"):
        dc.dual_overlay_from_mapping(node)

    node = _authoring_mapping(overlay())
    node["calibration"]["risk"]["joint_margin"] = 0.0
    with pytest.raises(dc.V2DualConfigError, match="unknown key"):
        dc.dual_overlay_from_mapping(node)


def test_the_from_mapping_builder_refuses_an_underspecified_overlay():
    from inverse_folding.reference_flow.fusion_v2 import dual_config as dc

    with pytest.raises(dc.V2DualConfigError):
        dc.dual_overlay_from_mapping({"schema_version": dc.DUAL_OVERLAY_SCHEMA_VERSION})
