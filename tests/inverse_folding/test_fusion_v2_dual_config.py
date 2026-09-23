"""DUALF5: the strict optional Dual overlay - the single carrier of Dual identity."""

from __future__ import annotations

import dataclasses

import pytest

from inverse_folding.reference_flow.fusion_v2 import dual_config as dc
from inverse_folding.reference_flow.fusion_v2 import joint_objective as jo

from . import _v2_fixtures as fx
from .test_fusion_v2_dual_evidence import EVAL_A, EVAL_B, calibration


def head_b(**over):
    kwargs = dict(evaluator=EVAL_B, variant_id="LC1", allele_idx=0, window_batch_size=64)
    kwargs.update(over)
    return dc.HeadRuntimeBinding(**kwargs)


def overlay(**over):
    kwargs = dict(
        schema_version=dc.DUAL_OVERLAY_SCHEMA_VERSION, calibration=calibration(),
        head_b_runtime=head_b(), arm_bundle=("joint", "a_only"),
        max_counterfactual_sequences_per_cycle=278,
        objective_spec_digest=fx.digest("objective-spec"),
        calibration_artifact_digest=fx.digest("calibration-artifact"))
    kwargs.update(over)
    return dc.DualOverlay(**kwargs)


# ------------------------------------------------------------------------------------------
# identity
# ------------------------------------------------------------------------------------------

def test_a_coherent_overlay_carries_one_digest_for_the_whole_dual_identity():
    one, same = overlay(), overlay()
    assert one.content_digest == same.content_digest
    assert one.arms == ("joint", "a_only")
    assert not one.is_recursive_bundle
    assert overlay(arm_bundle=("joint", "a_only", "b_only")).is_recursive_bundle


@pytest.mark.parametrize("field,value", [
    ("objective_spec_digest", fx.digest("edited-spec")),
    ("calibration_artifact_digest", fx.digest("edited-artifact")),
])
def test_hand_editing_the_spec_or_the_calibration_moves_the_overlay_digest(field, value):
    assert overlay(**{field: value}).content_digest != overlay().content_digest


def test_changing_tau_moves_the_overlay_digest():
    wider = dataclasses.replace(
        calibration(), law=dataclasses.replace(calibration().law, tau=0.20))
    assert overlay(calibration=wider).content_digest != overlay().content_digest


def test_an_overlay_from_another_schema_version_is_refused():
    with pytest.raises(dc.V2DualConfigError):
        overlay(schema_version="dualcfg-0")


# ------------------------------------------------------------------------------------------
# the arm vocabulary is frozen
# ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bundle", [
    (),
    ("joint",),
    ("a_only", "b_only"),
    ("joint", "b_only"),
    ("joint", "a_only", "a_only"),
    ("joint", "A_only"),
    ("joint", "a_only_with_joint_safety"),
    ("a_only", "joint"),
])
def test_an_undeclared_arm_bundle_is_refused(bundle):
    with pytest.raises(dc.V2DualConfigError):
        overlay(arm_bundle=bundle)


def test_the_two_declared_bundles_are_the_directionality_pair_and_the_capability_triple():
    assert dc.MATCHED_ARM_BUNDLES == (("joint", "a_only"), ("joint", "a_only", "b_only"))
    assert dc.ARM_LABELS == ("joint", "a_only", "b_only")
    # the retired labels are gone from the vocabulary entirely
    assert not any("safety" in label for label in dc.ARM_LABELS)


# ------------------------------------------------------------------------------------------
# Head B is bound by content, never by path
# ------------------------------------------------------------------------------------------

def test_a_runtime_head_B_that_is_not_the_calibrated_one_is_refused():
    with pytest.raises(dc.V2DualConfigError):
        overlay(head_b_runtime=head_b(evaluator=EVAL_A))


def test_a_resolved_checkpoint_is_proved_against_the_signed_digest():
    over = overlay()
    over.assert_observed_head_b(
        checkpoint_digest=EVAL_B.head_checkpoint_digest, config_hash=EVAL_B.head_config_hash)
    # the checkpoint digest is the ONLY discriminator between the two production Heads
    with pytest.raises(dc.V2DualConfigError, match="Head mixup"):
        over.assert_observed_head_b(
            checkpoint_digest=EVAL_A.head_checkpoint_digest,
            config_hash=EVAL_A.head_config_hash)


def test_a_shared_config_directory_does_not_by_itself_authorize_a_head():
    # Both production Heads share a config dir and an identical checkpoint-metadata config hash, so
    # a matching config hash with a wrong checkpoint must still fail.
    over = overlay()
    with pytest.raises(dc.V2DualConfigError):
        over.assert_observed_head_b(
            checkpoint_digest=fx.digest("some-other-checkpoint"),
            config_hash=EVAL_B.head_config_hash)


def test_binding_the_same_head_to_both_roles_is_refused_before_an_overlay_can_exist():
    # The coordinate pair itself refuses, so a same-Head overlay is unconstructible rather than
    # merely rejected later.
    with pytest.raises(jo.V2JointObjectiveError, match="same Head checkpoint digest"):
        jo.QuantityCoordinates(
            quantity="global_risk", a=calibration().risk.a,
            b=dataclasses.replace(calibration().risk.b, evaluator=EVAL_A))


def test_the_runtime_knobs_are_declared_rather_than_defaulted():
    # read from shard inputs today with silent fallbacks of 0 and 64
    for bad in (dict(allele_idx=-1), dict(window_batch_size=0), dict(variant_id=""),
                dict(window_batch_size=True)):
        with pytest.raises(dc.V2DualConfigError):
            head_b(**bad)


# ------------------------------------------------------------------------------------------
# the loader is strict
# ------------------------------------------------------------------------------------------

def payload():
    return dict(
        schema_version=dc.DUAL_OVERLAY_SCHEMA_VERSION, calibration=calibration(),
        head_b_runtime=head_b(), arm_bundle=["joint", "a_only"],
        max_counterfactual_sequences_per_cycle=278,
        objective_spec_digest=fx.digest("objective-spec"),
        calibration_artifact_digest=fx.digest("calibration-artifact"))


def test_the_loader_round_trips_a_declared_overlay():
    assert dc.load_dual_overlay(payload()).content_digest == overlay().content_digest


def test_an_unknown_overlay_key_is_refused_rather_than_ignored():
    node = payload()
    node["tau"] = 0.15
    with pytest.raises(dc.V2DualConfigError, match="unknown"):
        dc.load_dual_overlay(node)


@pytest.mark.parametrize("missing", [
    "schema_version", "calibration", "head_b_runtime", "arm_bundle",
    "max_counterfactual_sequences_per_cycle",
    "objective_spec_digest", "calibration_artifact_digest",
])
def test_every_overlay_key_is_required(missing):
    node = payload()
    node.pop(missing)
    with pytest.raises(dc.V2DualConfigError, match="missing"):
        dc.load_dual_overlay(node)


def test_the_module_holds_no_cluster_path_and_no_scientific_default():
    src = open(dc.__file__, encoding="utf-8").read()
    for banned in ("/scratch/", "/home/", "DRB1_0701", "DRB1_0401", "tau=0.", "= 0.15"):
        assert banned not in src, banned
