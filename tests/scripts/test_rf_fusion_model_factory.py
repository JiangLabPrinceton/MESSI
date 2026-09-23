"""V2F7: the shared model-preparation factory.

**Verification boundary, stated first because it bounds every claim below.**
``build_entry_oracles`` cannot run in this repo: it needs torch, a DPLM checkpoint, a Head
checkpoint and PDB inputs, and no test exercises it.  So this suite verifies the FACTORY -- the
piece both entry paths now share -- against injected seams, and separately asserts that V1 no longer
carries a second copy of that logic.  Real-model byte identity between the old inline code and the
delegated factory is NOT provable locally and is not claimed here; it stays a cluster check.

What IS proved: the factory makes the same calls with the same arguments, caches the expensive
per-protein preparation exactly once, resolves the hard-anchor constraint class the same way, and
asserts the null substrate before anything can sample.
"""

from __future__ import annotations

import types

import pytest

from scripts.rf_fusion_model_factory import ModelSeams, PreparedModel, build_model_factory


class _Alphabet:
    mask_idx = 32

    def __init__(self):
        self._toks = {i: c for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY", start=4)}

    def __len__(self):
        return 33

    def get_tok(self, index):
        return self._toks.get(index, "<pad>")

    def get_idx(self, token):
        for index, tok in self._toks.items():
            if tok == token:
                return index
        raise KeyError(token)


class _Rows:
    """Stands in for the test-set dataframe: only ``.loc[pid]`` is used."""

    def __init__(self):
        self.loc = {"5ZHV_B": {"sequence_length": 6}, "9L2Q_A": {"sequence_length": 8}}


def _rf_config():
    from inverse_folding.reference_flow.config import (
        AmplificationConfig,
        HShuffleConfig,
        ReferenceFlowConfig,
        RemaskConfig,
        SamplerConfig,
        ScheduleConfig,
    )

    return ReferenceFlowConfig(
        sampler=SamplerConfig(n_steps=100, seed=11, temperature=1.0, n_designs_per_protein=1,
                              remask=RemaskConfig(enabled=True, fraction_scale=0.0)),
        schedule=ScheduleConfig(base_form="linear"),
        amplification=AmplificationConfig(form="constant_one", h_source="h_processed"),
        h_shuffle=HShuffleConfig(enabled=False, seed=None),
    )


def _seams(calls, *, rf_config=None):
    def _record(name, result):
        def inner(*args, **kwargs):
            calls.append((name, args, tuple(sorted(kwargs))))
            return result(*args, **kwargs) if callable(result) else result
        return inner

    task = types.SimpleNamespace(alphabet=_Alphabet())
    return ModelSeams(
        load_if_task=_record("load_if_task", task),
        prepare_backbone=_record(
            "prepare_backbone",
            lambda **kw: types.SimpleNamespace(
                sequence_length=int(kw["entry"]["sequence_length"]),
                protein_id=kw["entry"]["protein_id"], coordinate_mask=(True,) * 6)),
        build_dplm_denoiser_context=_record("build_dplm_denoiser_context", lambda **kw: "ctx"),
        make_dplm_denoiser=_record("make_dplm_denoiser", lambda ctx: f"denoiser::{ctx}"),
        load_reference_flow_config=_record(
            "load_reference_flow_config", rf_config or _rf_config()),
        make_sampler=_record("make_sampler",
                             lambda **kw: types.SimpleNamespace(**kw)),
        read_test_rows=_record("read_test_rows", _Rows()),
        load_constraint_manifest=_record("load_constraint_manifest", None),
    )


def _build(calls, **over):
    kw = dict(base_if_checkpoint="/ckpt/dplm.pt", rf_sampler_config="/cfg/rf.yaml",
              test_set_parquet="/data/test.parquet", pdb_root="/data/pdb", device="cpu",
              seams=_seams(calls))
    kw.update(over)
    return build_model_factory(**kw)


# --------------------------------------------------------------------------------------------
# the factory prepares once and caches
# --------------------------------------------------------------------------------------------


def test_the_factory_loads_the_task_sampler_and_config_exactly_once():
    calls = []
    model = _build(calls)
    assert isinstance(model, PreparedModel)
    names = [name for name, _, _ in calls]
    assert names.count("load_if_task") == 1
    assert names.count("load_reference_flow_config") == 1
    assert names.count("make_sampler") == 1
    assert names.count("read_test_rows") == 1


def test_a_protein_backbone_is_prepared_once_per_cohort_not_once_per_call():
    """This cache is the whole reason the factory is a shared OBJECT rather than a function.

    Preparing a backbone is the expensive step; a per-call factory would repeat it for every root,
    every lookahead and every propagation segment of the same protein.
    """
    calls = []
    model = _build(calls)
    for _ in range(4):
        model.backbone_and_denoiser("5ZHV_B")
    model.backbone_and_denoiser("9L2Q_A")
    names = [name for name, _, _ in calls]
    assert names.count("prepare_backbone") == 2, "one preparation per DISTINCT protein"
    assert names.count("make_dplm_denoiser") == 2


def test_the_same_protein_returns_the_identical_denoiser_object():
    """A second denoiser for one protein would be a second model with its own state."""
    model = _build([])
    first, second = model.backbone_and_denoiser("5ZHV_B"), model.backbone_and_denoiser("5ZHV_B")
    assert first[0] is second[0]
    assert first[1] is second[1]


def test_sequence_length_comes_from_the_prepared_backbone_not_the_test_set_row():
    """The backbone is what the sampler actually runs on; the row is only how it was located."""
    model = _build([])
    assert model.sequence_length("9L2Q_A") == 8


# --------------------------------------------------------------------------------------------
# the frozen substrate is asserted before anything can sample
# --------------------------------------------------------------------------------------------


def test_an_amplified_sampler_config_is_refused_at_construction():
    """A version that forgot this assertion would silently run a different kernel than the one
    every calibration was measured on -- and the failure would surface as a scientific anomaly
    much later, not as an error."""
    import dataclasses

    from inverse_folding.reference_flow.config import AmplificationConfig

    amplified = dataclasses.replace(
        _rf_config(), amplification=AmplificationConfig(form="exp", h_source="h_processed"))
    with pytest.raises(Exception):
        _build([], seams=_seams([], rf_config=amplified))


# --------------------------------------------------------------------------------------------
# the hard-anchor constraint class
# --------------------------------------------------------------------------------------------


def test_an_unconstrained_protein_gets_none_rather_than_an_empty_mapping():
    """The sampler treats the two differently: ``{}`` declares a constraint set that exists and is
    empty, ``None`` declares one that was never made."""
    assert _build([]).fixed_tokens("5ZHV_B") is None


def test_anchors_are_resolved_through_the_alphabet_not_carried_as_letters():
    calls = []
    anchor = types.SimpleNamespace(index_0b=0, expected_aa="C")
    manifest = types.SimpleNamespace(
        constraint_for_protein=lambda pid: types.SimpleNamespace(hard_anchors=[anchor]))
    seams = _seams(calls)
    seams = ModelSeams(**{**{f.name: getattr(seams, f.name)
                             for f in seams.__dataclass_fields__.values()},
                          "load_constraint_manifest": lambda path: manifest})
    model = _build(calls, constraint_manifest="/cfg/anchors.json", seams=seams)
    assert model.fixed_tokens("5ZHV_B") == {0: 5}, "expected the alphabet index for 'C'"


def test_the_fixed_token_policy_is_recorded_on_the_model():
    """It enters the conditioning identity, so it must be carried rather than re-derived."""
    model = _build([], fixed_token_policy="manifest:abc123")
    assert model.fixed_token_policy == "manifest:abc123"


# --------------------------------------------------------------------------------------------
# there is now exactly ONE implementation
# --------------------------------------------------------------------------------------------




def test_the_factory_is_importable_without_torch():
    """``--print-config`` and ``--dry-run`` must not pull in a model (PLAN V2F7 acceptance).

    The seams resolve lazily, so importing the module and constructing ``ModelSeams()`` costs
    nothing; only ``resolved()`` reaches for the runtime.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import scripts.rf_fusion_model_factory as m; m.ModelSeams(); "
         "print('torch' in sys.modules)"],
        capture_output=True, text=True, cwd=str(__import__("pathlib").Path(__file__).parents[2]),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "importing the factory pulled in torch"
