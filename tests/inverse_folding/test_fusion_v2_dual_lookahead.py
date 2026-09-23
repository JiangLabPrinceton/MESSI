"""DUALF2 runtime: the paired binder, its ledger namespacing, and Dual-off equivalence."""

from __future__ import annotations

import dataclasses

import pytest

from inverse_folding.reference_flow.fusion_v2 import dual_evidence as de
from inverse_folding.reference_flow.fusion_v2 import identity as ident
from inverse_folding.reference_flow.fusion_v2 import joint_objective as jo
from inverse_folding.reference_flow.fusion_v2 import state as st
from inverse_folding.reference_flow.fusion_v2 import schedule as sch
from inverse_folding.reference_flow.fusion_v2_runtime import dual_lookahead as dl
from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import RawCompletion, bind_head_scores

from . import _v2_fixtures as fx
from .test_fusion_v2_dual_evidence import EVAL_A, EVAL_B, calibration, objective

WINDOWS = fx.WINDOWS
GRID = ident.window_grid_digest(WINDOWS)


@dataclasses.dataclass(frozen=True)
class Result:
    protein_id: str
    sequence_md5: str
    sequence_length: int
    allele: str
    score_scale: str
    windows: tuple
    binding: ident.HeadScoreBinding
    global_risk: float
    residue_hotspot: tuple[float, ...]


def completions(n=2):
    lin = fx.lineage()
    src = fx.source(lineage=lin)
    rows = []
    for index in range(n):
        tokens = (10 + index, 20, 12, 13, 14, 15)
        seq = "".join(fx.ALPHABET[int(t)] for t in tokens)
        rows.append(RawCompletion(
            fork_index=index, fork_seed=4242 + index, protein_id=lin.protein_id, lineage=lin,
            source_state_id=src.state_id, source_state_content_digest=src.content_digest,
            tokens=tokens, sequence=seq, sequence_md5=fx.sequence_md5(seq),
            evidence_by_pos=tuple(
                st.EndpointPositionEvidence(
                    token=t, commit=sch.history_key(0, 60 + i), completion_logprob=-2.5 - i,
                    inherited_from_source=False)
                for i, t in enumerate(tokens)),
            logical_dfe=10,
            replay=st.ReplayIdentity(mode="fork", rng_state=None, fork_seed=4242 + index,
                                     replay_state_hash=fx.E),
        ))
    return tuple(rows)


def results(comps, evaluator, *, base=-9.0):
    return tuple(
        Result(protein_id=c.protein_id, sequence_md5=c.sequence_md5, sequence_length=len(c.tokens),
               allele=evaluator.allele, score_scale=evaluator.score_scale, windows=WINDOWS,
               binding=ident.HeadScoreBinding(
                   protein_id=c.protein_id, sequence_md5=c.sequence_md5,
                   sequence_length=len(c.tokens), window_grid_digest=GRID, evaluator=evaluator),
               global_risk=base - index, residue_hotspot=(-0.1,) * len(c.tokens))
        for index, c in enumerate(comps))


def bind(**over):
    comps = over.pop("completions", None) or completions()
    kw = dict(completions=comps, results_a=results(comps, EVAL_A),
              results_b=results(comps, EVAL_B, base=-4.0),
              evaluator_a=EVAL_A, evaluator_b=EVAL_B, window_grid_digest=GRID,
              objective=objective())
    kw.update(over)
    return dl.bind_paired_head_scores(**kw)


def test_the_ledger_event_is_namespaced_per_allele_and_never_reuses_the_legacy_spelling():
    prefix = "5ZHV_B:fam0:d0"
    a = dl.dual_head_event_id(prefix, jo.AlleleRole.A)
    b = dl.dual_head_event_id(prefix, jo.AlleleRole.B)
    assert a != b
    assert a != f"{prefix}:head" and b != f"{prefix}:head"
    assert a.startswith(f"{prefix}:head:") and b.startswith(f"{prefix}:head:")


def test_the_paired_request_digest_is_a_set_not_a_sequence():
    assert dl.paired_request_digest(["b", "a"]) == dl.paired_request_digest(["a", "b", "a"])
    assert dl.paired_request_digest(["a"]) != dl.paired_request_digest(["a", "b"])
    with pytest.raises(de.V2DualEvidenceError):
        dl.paired_request_digest([])


def test_the_endpoints_a_dual_run_produces_are_the_ones_a_single_head_run_would_produce():
    comps = completions()
    rows_a = results(comps, EVAL_A)
    legacy = bind_head_scores(completions=comps, results=rows_a, evaluator=EVAL_A,
                              window_grid_digest=GRID)
    endpoints, _ = bind(completions=comps, results_a=rows_a)
    assert [e.endpoint_id for e in endpoints] == [e.endpoint_id for e in legacy]
    assert [e.content_digest for e in endpoints] == [e.content_digest for e in legacy]
    assert [e.head_global_risk for e in endpoints] == [e.head_global_risk for e in legacy]


def test_every_endpoint_receives_exactly_one_piece_of_dual_evidence():
    endpoints, evidence = bind()
    assert len(evidence) == len(endpoints)
    assert [e.endpoint_id for e in evidence] == [e.endpoint_id for e in endpoints]


def test_the_two_heads_must_be_asked_for_the_same_sequence_set():
    comps = completions(2)
    with pytest.raises(de.V2DualEvidenceError):
        bind(completions=comps, results_b=results(comps, EVAL_B, base=-4.0)[:1])


def test_binding_the_same_evaluator_to_both_roles_is_refused():
    comps = completions()
    with pytest.raises(de.V2DualEvidenceError):
        bind(completions=comps, evaluator_b=EVAL_A, results_b=results(comps, EVAL_A))


def test_a_role_B_result_on_another_window_grid_is_refused():
    comps = completions()
    rows_b = results(comps, EVAL_B, base=-4.0)
    # the grid digest hashes window COORDINATES only, so a different z is the same grid --
    # a shorter sequence's grid is a genuinely different coordinate set.
    other = ident.window_grid_digest(fx.windows(fx.L - 1))
    assert other != GRID
    bad = tuple(dataclasses.replace(r, binding=dataclasses.replace(r.binding,
                                                                   window_grid_digest=other))
                for r in rows_b)
    with pytest.raises(de.V2DualEvidenceError):
        bind(completions=comps, results_b=bad)


def test_the_density_axis_is_opt_in_and_computed_from_the_head_s_own_hotspot_vector():
    comps = completions()
    shared = calibration(density=True)
    _, evidence = bind(completions=comps, objective=objective(cal=shared),
                       density_objective=objective(quantity="positive_mass_density", cal=shared))
    assert all(e.density is not None for e in evidence)
    # all hotspot entries are negative in this fixture, so the positive mass is exactly zero
    assert all(e.a.raw_density == pytest.approx(0.0) for e in evidence)
    # and without a density objective the axis is absent rather than defaulted
    _, plain = bind(completions=comps)
    assert all(e.density is None for e in plain)


def test_the_density_helper_reproduces_the_documented_formula():
    assert de.positive_mass_density((-0.1, 0.4, 0.0, 0.2), 4) == pytest.approx((0.4 + 0.2) / 4)
    assert de.positive_mass_density((-1.0, -2.0), 2) == pytest.approx(0.0)
    with pytest.raises(de.V2DualEvidenceError):
        de.positive_mass_density((0.1, 0.2), 3)
