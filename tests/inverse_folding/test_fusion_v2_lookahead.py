"""V2F4-1: the exact lookahead pool and endpoint scoring (PLAN §4.1).

PLAN §4.1: "All lookaheads at a depth fork from the same committed live-state bytes under
persisted, disjoint seed namespaces.  Complete endpoint scoring uses one identity-bearing batch
Head interface and rejects missing, duplicate, extra, or reordered results.  The runner stores the
exact sequence and per-position completion evidence before any ranking.  Endpoint values may
support deterministic selection, but no scalar aggregate replaces the endpoint or becomes an
inherited parent."

Two failure modes drive the design.  A Head batch that silently reorders would attach one design's
immunogenicity to another design's sequence -- the resulting "improvement" would be pure noise and
nothing downstream could detect it.  And an endpoint discarded before its raw evidence is stored
would make the archive a biased sample of what was actually generated, so any frontier computed
from it overstates what the method found.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from inverse_folding.reference_flow.config import (
    AmplificationConfig,
    HShuffleConfig,
    ReferenceFlowConfig,
    RemaskConfig,
    SamplerConfig,
    ScheduleConfig,
)
from inverse_folding.reference_flow.fusion_v2 import identity as ident
from inverse_folding.reference_flow.fusion_v2_runtime.capture import capture_depth_zero
from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import (
    HeadBatchError,
    RawCompletion,
    V2LookaheadError,
    bind_head_scores,
    generate_lookaheads,
)
from inverse_folding.reference_flow.sampler import PositionDependentDFMSampler
from tests.inverse_folding import _v2_fixtures as F

VOCAB = 33
C0 = 50
K = 4
FORK_SEEDS = (5001, 5002, 5003, 5004)
#: A token-id -> residue-letter map for the toy vocabulary.  The letters must be genuinely
#: canonical AA20 -- CompleteEndpoint refuses a sequence containing anything else, which is what
#: stops a masked or out-of-alphabet completion from reaching the Head or the archive.
CANONICAL_AA20 = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET = {token: CANONICAL_AA20[index] for index, token in enumerate(sorted(F.AA))}


def _denoiser(x_t: torch.Tensor, t: float, struct) -> torch.Tensor:
    length = int(x_t.shape[0])
    logits = torch.full((length, VOCAB), float("-inf"), dtype=torch.float32)
    for position in range(length):
        for token in sorted(F.AA):
            logits[position, token] = float((position * 7 + token * 3 + int(x_t[position])) % 11)
    return logits


def _cfg(**over):
    kw = dict(n_steps=F.N_STEPS, seed=11, temperature=1.0)
    kw.update(over)
    return ReferenceFlowConfig(
        sampler=SamplerConfig(
            n_steps=kw["n_steps"], seed=kw["seed"], temperature=kw["temperature"],
            n_designs_per_protein=1,
            remask=RemaskConfig(enabled=True, fraction_scale=0.0),
        ),
        schedule=ScheduleConfig(base_form="linear"),
        amplification=AmplificationConfig(form="constant_one", h_source="h_processed"),
        h_shuffle=HShuffleConfig(enabled=False, seed=None),
    )


ANCHORS = {0: 10}


def _sampler():
    return PositionDependentDFMSampler(mask_token_id=F.MASK, vocab_size=VOCAB)


def _source():
    return capture_depth_zero(
        sampler=_sampler(), denoiser=_denoiser, config=_cfg(), sequence_length=F.L,
        h_values=np.zeros(F.L, dtype=np.float32), residue_token_ids=F.AA,
        at_step=C0, fixed_tokens=ANCHORS, lineage=F.lineage(), mask_token_id=F.MASK,
        aa_token_ids=F.AA, conditioning=F.conditioning(),
        safety_reference=F.safety_reference(), cost_event_ids=("evt:root",),
    )


def _generate(**over):
    kw = dict(
        source=_source(), sampler=_sampler(), denoiser=_denoiser, config=_cfg(),
        h_values=np.zeros(F.L, dtype=np.float32), residue_token_ids=F.AA,
        fork_seeds=FORK_SEEDS, alphabet=ALPHABET,
    )
    kw.update(over)
    return generate_lookaheads(**kw)


# --------------------------------------------------------------------------------------------
# the pool forks from ONE committed state under disjoint seeds
# --------------------------------------------------------------------------------------------


def test_the_pool_has_one_completion_per_requested_fork():
    completions = _generate()
    assert len(completions) == K
    assert [c.fork_index for c in completions] == list(range(K))


def test_every_fork_starts_from_the_same_committed_source_bytes():
    source = _source()
    for completion in _generate(source=source):
        assert completion.source_state_id == source.state_id
        assert completion.source_state_content_digest == source.content_digest


def test_disjoint_seeds_produce_distinct_completions():
    """If the seed were ignored, the pool would be K copies of one design and the whole breadth
    argument would be vacuous."""
    sequences = {c.sequence for c in _generate()}
    assert len(sequences) > 1


def test_the_same_seed_reproduces_the_same_completion():
    a = _generate(fork_seeds=(7777,))[0]
    b = _generate(fork_seeds=(7777,))[0]
    assert a.sequence == b.sequence
    assert a.tokens == b.tokens


def test_a_duplicate_fork_seed_is_refused():
    """Two forks on one seed are the same draw counted twice; PLAN §4.1 requires DISJOINT seed
    namespaces, and a duplicate would let one design buy two archive slots."""
    with pytest.raises(V2LookaheadError, match="disjoint"):
        _generate(fork_seeds=(5001, 5002, 5001))


def test_the_source_state_is_not_mutated_by_generation():
    source = _source()
    before = source.content_digest
    _generate(source=source)
    assert source.content_digest == before


# --------------------------------------------------------------------------------------------
# every completion is exact and complete
# --------------------------------------------------------------------------------------------


def test_every_completion_is_terminal():
    """A lookahead is an EXACT complete endpoint: no mask may survive."""
    for completion in _generate():
        assert F.MASK not in completion.tokens


def test_every_completion_preserves_the_hard_anchors():
    source = _source()
    for completion in _generate(source=source):
        for position, token in source.hard_anchors:
            assert completion.tokens[position] == token


def test_the_sequence_is_the_decoded_token_vector():
    for completion in _generate():
        assert completion.sequence == "".join(ALPHABET[t] for t in completion.tokens)


def test_per_position_completion_evidence_covers_every_position():
    """PLAN §4.1: "stores the exact sequence and per-position completion evidence"."""
    for completion in _generate():
        assert len(completion.evidence_by_pos) == F.L
        for position, evidence in enumerate(completion.evidence_by_pos):
            assert evidence.token == completion.tokens[position]


def test_evidence_marks_which_positions_were_inherited_from_the_source():
    source = _source()
    completion = _generate(source=source)[0]
    for position, evidence in enumerate(completion.evidence_by_pos):
        inherited = source.tokens[position] != F.MASK
        assert evidence.inherited_from_source is inherited


def test_each_lookahead_pays_its_own_tail():
    """PLAN §3.4: C_screen = c + K(S-c); each lookahead pays S-c and the source prefix is charged
    once, not K times."""
    for completion in _generate():
        assert completion.logical_dfe == F.N_STEPS - C0


def test_an_alphabet_missing_a_produced_token_fails_closed():
    """A silently unmapped token would become a wrong residue in the scored sequence."""
    broken = {t: ALPHABET[t] for t in sorted(F.AA)[:2]}
    with pytest.raises(V2LookaheadError):
        _generate(alphabet=broken)


# --------------------------------------------------------------------------------------------
# the identity-bearing batch Head interface
# --------------------------------------------------------------------------------------------


def _binding(completion):
    return ident.HeadScoreBinding(
        protein_id=completion.protein_id, sequence_md5=completion.sequence_md5,
        sequence_length=F.L, window_grid_digest=ident.window_grid_digest(_WINDOWS),
        evaluator=F.safety_reference().head_binding.evaluator,
    )


_WINDOWS = (F.HeadWindow(start_0b=0, end_0b=F.L, k=F.L, z=-0.7),)


@dataclasses.dataclass(frozen=True)
class _Result:
    """What a batch Head hands back.

    It satisfies ``identity.HeadScoreLike`` (protein_id / sequence_md5 / sequence_length / allele /
    score_scale / windows) and additionally carries its OWN ``binding``, so a reordered batch is
    caught by comparison rather than trusted by position.
    """

    protein_id: str
    sequence_md5: str
    sequence_length: int
    allele: str
    score_scale: str
    windows: tuple
    binding: ident.HeadScoreBinding
    global_risk: float
    residue_hotspot: tuple[float, ...]


def _results(completions, **over):
    evaluator = F.safety_reference().head_binding.evaluator
    rows = [
        _Result(
            protein_id=c.protein_id, sequence_md5=c.sequence_md5, sequence_length=F.L,
            allele=evaluator.allele, score_scale=evaluator.score_scale, windows=_WINDOWS,
            binding=_binding(c), global_risk=-9.0 - index, residue_hotspot=(-0.1,) * F.L,
        )
        for index, c in enumerate(completions)
    ]
    return over.get("rows", rows)


def _bind(completions=None, rows=None, **over):
    completions = completions if completions is not None else _generate()
    rows = rows if rows is not None else _results(completions)
    kw = dict(completions=completions, results=rows,
              evaluator=F.safety_reference().head_binding.evaluator,
              window_grid_digest=ident.window_grid_digest(_WINDOWS))
    kw.update(over)
    return bind_head_scores(**kw)


def test_binding_returns_one_scored_row_per_completion():
    completions = _generate()
    bound = _bind(completions)
    assert len(bound) == len(completions)


def test_a_missing_head_result_is_refused():
    completions = _generate()
    with pytest.raises(HeadBatchError, match="missing"):
        _bind(completions, rows=_results(completions)[:-1])


def test_an_extra_head_result_is_refused():
    """A result for a sequence nobody asked about.  Appending a COPY of an existing row would be a
    duplicate, not an extra -- a different failure with a different guard -- so the extra row here
    carries a novel md5."""
    completions = _generate()
    rows = _results(completions)
    stray = dataclasses.replace(rows[0], sequence_md5="f" * 32)
    with pytest.raises(HeadBatchError, match="extra|unexpected"):
        _bind(completions, rows=rows + [stray])


def test_a_duplicate_head_result_is_refused():
    completions = _generate()
    rows = _results(completions)
    rows[1] = rows[0]
    with pytest.raises(HeadBatchError, match="duplicate|missing"):
        _bind(completions, rows=rows)


def test_a_reordered_head_batch_is_rebound_by_identity_not_rejected():
    """The most dangerous silent failure in the pipeline: a shuffled batch attaching one design's
    immunogenicity to another design's sequence.

    The requirement is IDENTITY BINDING, which is strictly stronger than rejection.  Each result
    carries its own sequence, so a reordered batch is not an error -- it must simply be rebound
    correctly.  Asserting "rejected OR rebound" would be satisfied by a bind-by-position
    implementation too (it rejects on the cross-check), so this test demands success.

    The swap must involve two GENUINELY DISTINCT sequences: forks often collide -- a duplicate
    sibling is legal -- and swapping two identical rows is a no-op.
    """
    completions = _generate()
    rows = _results(completions)
    distinct = [
        (i, j)
        for i in range(len(completions)) for j in range(i + 1, len(completions))
        if completions[i].sequence_md5 != completions[j].sequence_md5
    ]
    assert distinct, "the fixture produced no two distinct sequences; the swap would be a no-op"
    i, j = distinct[0]
    shuffled = list(rows)
    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

    bound = _bind(completions, rows=shuffled)

    for endpoint, completion in zip(bound, completions):
        assert endpoint.sequence_md5 == completion.sequence_md5
        assert endpoint.head_score.sequence_md5 == completion.sequence_md5
        assert endpoint.head_binding.sequence_md5 == completion.sequence_md5


def test_a_completely_reversed_head_batch_is_still_rebound_correctly():
    """Belt and braces on the same invariant: order carries no information at all."""
    completions = _generate()
    bound = _bind(completions, rows=list(reversed(_results(completions))))
    for endpoint, completion in zip(bound, completions):
        assert endpoint.head_score.sequence_md5 == completion.sequence_md5


def test_a_head_result_for_an_unknown_sequence_is_refused():
    completions = _generate()
    rows = _results(completions)
    rows[0] = dataclasses.replace(rows[0], sequence_md5="0" * 32)
    with pytest.raises(HeadBatchError):
        _bind(completions, rows=rows)


def test_a_different_head_evaluator_identity_is_refused():
    """Two evaluators are two measuring instruments; mixing them makes the comparison meaningless."""
    other = dataclasses.replace(
        F.safety_reference().head_binding.evaluator, window_k_max=99)
    with pytest.raises(HeadBatchError, match="evaluator|identity"):
        _bind(evaluator=other)


def test_a_window_grid_mismatch_is_refused():
    with pytest.raises(HeadBatchError, match="window"):
        _bind(window_grid_digest=ident.window_grid_digest(
            (F.HeadWindow(start_0b=0, end_0b=F.L, k=F.L - 1, z=-0.7),)))


# --------------------------------------------------------------------------------------------
# raw evidence survives scoring
# --------------------------------------------------------------------------------------------


def test_no_scored_endpoint_is_discarded():
    """PLAN §4.1 acceptance: "no scored endpoint is discarded from raw evidence"."""
    completions = _generate()
    bound = _bind(completions)
    assert {e.sequence_md5 for e in bound} == {c.sequence_md5 for c in completions}


def test_the_exact_sequence_survives_into_the_endpoint():
    completions = _generate()
    for endpoint, completion in zip(_bind(completions), completions):
        assert endpoint.sequence == completion.sequence


def test_the_per_position_evidence_survives_into_the_endpoint():
    completions = _generate()
    for endpoint, completion in zip(_bind(completions), completions):
        assert endpoint.endpoint_provenance_evidence_by_pos == completion.evidence_by_pos


def test_a_scored_endpoint_starts_unvalidated():
    """PLAN §4.2: "Endpoints may enter the archive as unvalidated", and must become definitive
    before they can become feedback ancestry.  Scoring alone does not confer feasibility."""
    from inverse_folding.reference_flow.fusion_v2.state import FeasibilityLevel

    for endpoint in _bind():
        assert endpoint.feasibility_level is FeasibilityLevel.UNVALIDATED
        assert endpoint.structure_outcome is None


def test_the_endpoint_keeps_its_own_fork_identity():
    completions = _generate()
    for endpoint, completion in zip(_bind(completions), completions):
        assert endpoint.fork_index == completion.fork_index
        assert endpoint.fork_seed == completion.fork_seed


# --------------------------------------------------------------------------------------------
# the oracle request: what a model is actually asked about
# --------------------------------------------------------------------------------------------


def test_an_oracle_request_carries_the_sequence_and_keeps_the_digest_as_a_key():
    """The Head and the structure gate are functions of the SEQUENCE.

    An MD5 handed to a real scorer cannot be tokenized and cannot be folded, but it type-checks and
    satisfies every fake -- so the two roles are separated by TYPE here rather than by convention.
    """
    from inverse_folding.reference_flow.fusion.state import sequence_md5
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest

    sequence = "ACDEFG"
    request = OracleRequest(
        protein_id="5ZHV_B", sequence=sequence, sequence_md5=sequence_md5(sequence),
        sequence_length=len(sequence),
    )
    assert request.sequence == sequence
    assert request.sequence_md5 == sequence_md5(sequence)


def test_a_request_whose_digest_does_not_key_its_own_sequence_is_refused():
    """The digest is a JOIN KEY.  One that names a different sequence would let a cached refold or
    a Head result be attached to a design it was never computed for."""
    from inverse_folding.reference_flow.fusion.state import sequence_md5
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest

    with pytest.raises(V2LookaheadError):
        OracleRequest(protein_id="5ZHV_B", sequence="ACDEFG",
                      sequence_md5=sequence_md5("GFEDCA"), sequence_length=6)


def test_a_request_that_is_not_complete_canonical_aa20_is_refused():
    """PLAN §4.1 endpoints are COMPLETE.  A mask character, a gap or a lowercase residue would be
    scored as some arbitrary token by whichever backend happened to accept it."""
    from inverse_folding.reference_flow.fusion.state import sequence_md5
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest

    for bad in ("ACDEF#", "acdefg", "ACDEFX", "ACDEF-"):
        with pytest.raises(V2LookaheadError):
            OracleRequest(protein_id="5ZHV_B", sequence=bad,
                          sequence_md5=sequence_md5(bad), sequence_length=6)


def test_a_request_whose_declared_length_disagrees_with_its_sequence_is_refused():
    from inverse_folding.reference_flow.fusion.state import sequence_md5
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest

    with pytest.raises(V2LookaheadError):
        OracleRequest(protein_id="5ZHV_B", sequence="ACDEFG",
                      sequence_md5=sequence_md5("ACDEFG"), sequence_length=7)


def test_a_completion_builds_the_request_that_describes_it():
    """The runner must not assemble the request by hand at each call site; a builder is what makes
    "the oracle sees the completion's own bytes" structural."""
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import oracle_request

    completion = _generate()[0]
    request = oracle_request(completion)
    assert request.sequence == completion.sequence
    assert request.sequence_md5 == completion.sequence_md5
    assert request.protein_id == completion.protein_id
    assert request.sequence_length == len(completion.tokens)


def test_the_per_position_commit_names_the_step_the_position_actually_resolved():
    """PLAN §4.1 stores "per-position completion evidence"; the commit coordinate is provenance.

    It was derived from the SOURCE's unmask vector -- which the completion never updates -- so
    ``max(unmask[pos], start_step)`` collapsed to ``c_d`` for every position resolved during the
    tail, regardless of when in the tail it resolved.  The sampler reports the real per-position
    step (``SamplerOutput.unmask_step_by_pos``), so the value was fabricated where a measurement
    was available, and an audit replaying the endpoint would place every tail commit at ``c_d``.
    """
    completions = _generate()
    source = _source()
    start = int(source.sampler_step)
    tail_commits = {
        evidence.commit.step
        for completion in completions
        for position, evidence in enumerate(completion.evidence_by_pos)
        if not evidence.inherited_from_source
    }
    assert tail_commits, "the fixture resolved nothing during the tail"
    assert tail_commits != {start}, (
        f"every tail-resolved position was stamped at the source checkpoint {start}")
    assert all(step > start for step in tail_commits)
