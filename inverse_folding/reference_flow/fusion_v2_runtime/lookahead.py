"""V2F4-1: the exact lookahead pool and identity-bound endpoint scoring (PLAN §4.1).

At a source checkpoint ``c_d``, K exact complete futures are forked from the SAME committed
live-state bytes under disjoint seeds, run to termination, and scored by one identity-bearing batch
Head interface.

Two silent failure modes shape this module:

**A reordered Head batch** would attach one design's immunogenicity to another design's sequence.
The resulting "improvement" would be pure noise, and nothing downstream could detect it -- the
numbers would look entirely plausible.  So :func:`bind_head_scores` matches results to completions
by SEQUENCE IDENTITY, never by position, and refuses a batch that is missing, duplicated, extra, or
unrecognised.

**A discarded endpoint** would make the archive a biased sample of what was actually generated, so
any frontier computed from it overstates what the method found.  So the raw completion -- exact
sequence and per-position evidence -- is materialized BEFORE any Head call, and scoring only
attaches to it.

PLAN §4.1 also forbids a scalar shortcut: "Endpoint values may support deterministic selection, but
no scalar aggregate replaces the endpoint or becomes an inherited parent."  Nothing here returns a
score in place of an endpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from ..fusion.state import CANONICAL_AA20
from ..fusion.state import sequence_md5 as _sequence_md5
from ..fusion_v2.errors import V2Error
from ..fusion_v2.identity import (
    V2_STATE_SCHEMA_VERSION,
    HeadEvaluatorIdentity,
    HeadScoreBinding,
)
from ..fusion_v2.schedule import history_key
from ..fusion_v2.state import (
    CompleteEndpoint,
    EndpointPositionEvidence,
    FeasibilityLevel,
    LivePartialState,
    ReplayIdentity,
)
from ..sampler import ContinuationResume, _replay_state_hash

__all__ = [
    "V2LookaheadError",
    "HeadBatchError",
    "OracleRequest",
    "RawCompletion",
    "generate_lookaheads",
    "oracle_request",
    "oracle_request_for_endpoint",
    "bind_head_scores",
]

_CANONICAL_AA20 = frozenset(CANONICAL_AA20)


class V2LookaheadError(V2Error):
    """A lookahead-pool contract violation."""


class HeadBatchError(V2LookaheadError):
    """The Head batch does not correspond one-to-one with the completions it was asked to score."""


@dataclass(frozen=True)
class OracleRequest:
    """One complete design in the form an oracle actually consumes (PLAN §4.1, §4.2).

    Both the Head and the structure gate are functions of the SEQUENCE: the Head tokenizes the
    residue string and the structure gate folds it.  ``sequence_md5`` travels alongside as a CACHE
    AND JOIN KEY only -- it is what an on-disk refold cache is keyed by and what binds a result back
    to the completion that produced it -- but it is never what a model is asked about.

    The two are separated by TYPE rather than by convention because substituting one for the other
    is invisible everywhere it can be tested cheaply.  A digest passed where a sequence belongs is
    still a ``str``, so it satisfies every signature and every fake keyed by identity; it fails only
    on a real backend, i.e. only after a cluster allocation has been paid for.

    ``sequence_md5`` is verified against ``sequence`` here, so a request cannot carry a key that
    names some other design -- which is how a cached refold gets attached to a sequence it was never
    computed for.
    """

    protein_id: str
    sequence: str
    sequence_md5: str
    sequence_length: int

    def __post_init__(self) -> None:
        for name in ("protein_id", "sequence", "sequence_md5"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise V2LookaheadError(f"{name} must be a non-empty str, got {value!r}")
        stray = sorted(set(self.sequence) - _CANONICAL_AA20)
        if stray:
            raise V2LookaheadError(
                f"sequence carries non-canonical character(s) {stray}; an oracle request describes "
                "a COMPLETE canonical AA20 design, and a mask, gap or digest character would be "
                "mapped to some arbitrary token by whichever backend happened to accept it"
            )
        if isinstance(self.sequence_length, bool) or not isinstance(self.sequence_length, int):
            raise V2LookaheadError("sequence_length must be an int")
        if self.sequence_length != len(self.sequence):
            raise V2LookaheadError(
                f"sequence_length={self.sequence_length} disagrees with the {len(self.sequence)} "
                "residues actually carried"
            )
        if self.sequence_md5 != _sequence_md5(self.sequence):
            raise V2LookaheadError(
                "sequence_md5 does not key the sequence it is stored with; the digest is a cache "
                "and join key, so one naming a different design would attach a cached refold or a "
                "Head result to a sequence it was never computed for"
            )


def oracle_request(completion: RawCompletion) -> OracleRequest:
    """Build the request that describes one raw completion.

    A builder rather than a call-site literal: "the oracle is asked about the completion's own
    bytes" then holds by construction at every call site instead of at the ones that remembered.
    """
    return OracleRequest(
        protein_id=completion.protein_id, sequence=completion.sequence,
        sequence_md5=completion.sequence_md5, sequence_length=len(completion.tokens),
    )


def oracle_request_for_endpoint(endpoint: Any) -> OracleRequest:
    """Build the request that describes one scored endpoint (the structure gate's input)."""
    return OracleRequest(
        protein_id=endpoint.protein_id, sequence=endpoint.sequence,
        sequence_md5=endpoint.sequence_md5, sequence_length=int(endpoint.sequence_length),
    )


@dataclass(frozen=True)
class RawCompletion:
    """One exact complete future, recorded BEFORE any ranking or Head call.

    This is the evidence PLAN §4.1 requires the runner to store: the exact sequence and the
    per-position completion evidence.  It exists as its own type so that "the endpoint was
    generated" and "the endpoint was scored" are separate, independently auditable facts.
    """

    fork_index: int
    fork_seed: int
    protein_id: str
    lineage: Any
    source_state_id: str
    source_state_content_digest: str
    tokens: tuple[int, ...]
    sequence: str
    sequence_md5: str
    evidence_by_pos: tuple[EndpointPositionEvidence, ...]
    logical_dfe: int
    replay: ReplayIdentity


def _decode(tokens: Sequence[int], alphabet: Mapping[int, str]) -> str:
    letters = []
    for position, token in enumerate(tokens):
        letter = alphabet.get(int(token))
        if letter is None:
            raise V2LookaheadError(
                f"position {position} holds token {token}, which the alphabet does not map; an "
                "unmapped token would silently become the wrong residue in the scored sequence"
            )
        letters.append(letter)
    return "".join(letters)


def generate_lookaheads(
    *,
    source: LivePartialState,
    sampler: Any,
    denoiser: Callable[[torch.Tensor, float, Any], torch.Tensor],
    config: Any,
    h_values: np.ndarray | Sequence[float],
    residue_token_ids: frozenset[int] | Sequence[int],
    fork_seeds: Sequence[int],
    alphabet: Mapping[int, str],
    struct: Any = None,
) -> tuple[RawCompletion, ...]:
    """Fork K exact complete futures from one committed source state.

    Every fork resumes from the SAME bytes -- ``source.tokens``, its anchors, its editable domain,
    its unmask history -- and differs only in its seed.  That is what makes the pool a measurement
    of the source state rather than of K unrelated trajectories.
    """
    if not isinstance(source, LivePartialState):
        raise V2LookaheadError("source must be a LivePartialState")

    seeds = [int(s) for s in fork_seeds]
    if len(set(seeds)) != len(seeds):
        raise V2LookaheadError(
            f"fork seeds must be disjoint, got {seeds}; two forks on one seed are the same draw "
            "counted twice and would let one design buy two archive slots (PLAN §4.1)"
        )
    if not seeds:
        raise V2LookaheadError("at least one fork seed is required")

    length = len(source.tokens)
    x_t = torch.tensor(source.tokens, dtype=torch.long)
    scores = np.array(
        [np.nan if v is None else float(v) for v in source.active_sampler_score_by_pos],
        dtype=np.float64,
    )
    unmask = tuple(-1 if c is None else int(c.step)
                   for c in source.active_commit_depth_step_by_pos)
    anchors = tuple((int(p), int(t)) for p, t in source.hard_anchors)
    editable = tuple(int(i) for i in source.editable_positions)
    start_step = int(source.sampler_step)
    n_steps = int(source.n_steps)

    completions: list[RawCompletion] = []
    for fork_index, seed in enumerate(seeds):
        resume = ContinuationResume(
            x_t=x_t.detach().clone(), scores=scores.copy(), unmask_step_by_pos=unmask,
            start_step=start_step, n_steps=n_steps, fixed_tokens=anchors,
            editable_positions=editable, mode="fork",
            state_hash=_replay_state_hash(
                x_t, scores, start_step, n_steps, anchors, editable, unmask, "fork", None, seed),
            fork_seed=seed,
        )
        output = sampler.sample(
            sequence_length=length, h_values=h_values, denoiser=denoiser, config=config,
            struct=struct, controller=None, residue_token_ids=residue_token_ids,
            continuation_resume=resume,
        )
        tokens = tuple(int(v) for v in output.tokens.detach().cpu().tolist())
        if source.mask_token_id in tokens:
            # A backstop, not a reachable branch: the V1 sampler completes any residual mask at the
            # end of its loop, so this cannot fire through the normal path.  Mutation testing
            # confirms no test can distinguish its removal.  It is kept because "exact and
            # complete" is the defining property of a lookahead, and a future sampler entry that
            # stopped early must not silently produce a partial "endpoint".
            raise V2LookaheadError(
                f"fork {fork_index} did not terminate: the mask token survives in the completion"
            )
        final_scores = np.asarray(output.final_scores, dtype=np.float64)
        # The REALIZED per-position resolve step, reported by the sampler for this completion.
        # Deriving it from the SOURCE's unmask vector -- which a completion never updates --
        # collapsed every tail-resolved position onto ``c_d``, so the immutable provenance PLAN
        # §4.1 requires recorded a coordinate no position actually committed at, while the real
        # value was sitting in the sampler's own output.
        realized_unmask = tuple(int(v) for v in output.unmask_step_by_pos)
        evidence = tuple(
            EndpointPositionEvidence(
                token=int(token),
                commit=history_key(source.lineage.depth,
                                   int(max(realized_unmask[position], unmask[position]))),
                completion_logprob=(
                    None if not np.isfinite(final_scores[position])
                    else float(final_scores[position])
                ),
                # A position already resolved in the source was carried, not completed here.
                inherited_from_source=source.tokens[position] != source.mask_token_id,
            )
            for position, token in enumerate(tokens)
        )
        sequence = _decode(tokens, alphabet)
        completions.append(RawCompletion(
            fork_index=fork_index, fork_seed=seed, protein_id=source.lineage.protein_id,
            lineage=source.lineage,
            source_state_id=source.state_id,
            source_state_content_digest=source.content_digest,
            tokens=tokens, sequence=sequence, sequence_md5=_sequence_md5(sequence),
            evidence_by_pos=evidence,
            logical_dfe=n_steps - start_step,
            replay=ReplayIdentity(mode="fork", rng_state=None, fork_seed=seed,
                                  replay_state_hash=resume.state_hash),
        ))
    return tuple(completions)


def bind_head_scores(
    *,
    completions: Sequence[RawCompletion],
    results: Sequence[Any],
    evaluator: HeadEvaluatorIdentity,
    window_grid_digest: str,
    cost_event_ids: Sequence[str] = (),
) -> tuple[CompleteEndpoint, ...]:
    """Attach one Head result to each completion, BY IDENTITY.

    Every result must carry its own ``sequence_md5`` and ``binding``.  Matching by position would
    make a reordered batch undetectable, which is the one failure in this pipeline that produces
    plausible-looking numbers with no signal in them at all.

    Returns endpoints in the order of ``completions``, each at ``FeasibilityLevel.UNVALIDATED``:
    PLAN §4.2 requires a separate promotion to ``definitive`` before an endpoint may become
    feedback ancestry, support the feasible frontier, or be returned as a final design.
    """
    if not isinstance(evaluator, HeadEvaluatorIdentity):
        raise HeadBatchError("evaluator must be a HeadEvaluatorIdentity")

    # The Head is a function of the SEQUENCE, so the batch is keyed by sequence identity and one
    # result serves every completion carrying that sequence.  Two forks producing identical bytes
    # is a legal duplicate sibling (PLAN §4.4 governs what it may buy, not whether it may exist),
    # and scoring it twice would be the same measurement counted twice.
    by_md5: dict[str, Any] = {}
    for result in results:
        md5 = getattr(result, "sequence_md5", None)
        if md5 is None:
            raise HeadBatchError(
                "every Head result must carry its own sequence_md5; without it the batch can only "
                "be matched by position, and a reordered batch would be undetectable"
            )
        if md5 in by_md5:
            raise HeadBatchError(f"duplicate Head result for sequence {md5}")
        by_md5[md5] = result

    expected = {completion.sequence_md5 for completion in completions}
    unexpected = set(by_md5) - expected
    if unexpected:
        raise HeadBatchError(
            f"extra/unexpected Head result(s) for {sorted(unexpected)}; the batch does not "
            "correspond to the completions it was asked to score"
        )

    endpoints: list[CompleteEndpoint] = []
    for completion in completions:
        result = by_md5.get(completion.sequence_md5)
        if result is None:
            raise HeadBatchError(
                f"missing Head result for fork {completion.fork_index} "
                f"(sequence {completion.sequence_md5}); a silently dropped endpoint biases the "
                "archive toward whatever survived"
            )
        binding = getattr(result, "binding", None)
        if not isinstance(binding, HeadScoreBinding):
            raise HeadBatchError("every Head result must carry a HeadScoreBinding")
        if binding.evaluator != evaluator:
            raise HeadBatchError(
                f"Head result for fork {completion.fork_index} was produced by a different "
                "evaluator identity; two evaluators are two measuring instruments and mixing them "
                "makes the comparison meaningless"
            )
        if binding.window_grid_digest != window_grid_digest:
            raise HeadBatchError(
                f"window grid mismatch for fork {completion.fork_index}: result declares "
                f"{binding.window_grid_digest}, the pool declares {window_grid_digest}"
            )
        if binding.sequence_md5 != completion.sequence_md5:
            raise HeadBatchError(
                f"the binding on the result for sequence {completion.sequence_md5} describes "
                f"{binding.sequence_md5} instead"
            )

        endpoints.append(CompleteEndpoint(
            schema_version=V2_STATE_SCHEMA_VERSION,
            # The endpoint inherits the lineage of the state it forked from; fabricating one here
            # would decouple it from the source it is evidence about.
            lineage=completion.lineage,
            protein_id=completion.protein_id,
            sequence=completion.sequence,
            sequence_md5=completion.sequence_md5,
            sequence_length=len(completion.tokens),
            source_state_id=completion.source_state_id,
            source_state_content_digest=completion.source_state_content_digest,
            fork_index=completion.fork_index,
            fork_seed=completion.fork_seed,
            replay=completion.replay,
            endpoint_provenance_evidence_by_pos=completion.evidence_by_pos,
            head_binding=binding,
            head_score=result,
            head_global_risk=float(getattr(result, "global_risk")),
            # Scoring alone confers no feasibility (PLAN §4.2).
            feasibility_level=FeasibilityLevel.UNVALIDATED,
            structure_outcome=None,
            cost_event_ids=tuple(str(v) for v in cost_event_ids),
        ))
    return tuple(endpoints)
