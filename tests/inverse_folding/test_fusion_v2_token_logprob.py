"""V2F3a: the sampler-neutral arbitrary-token log-probability primitive.

PLAN task V2F3 requires a primitive that scores an ARBITRARY token under the same normalization the
categorical sampler uses for its own chosen token.  V2's projection injects tokens that the sampler
did not draw (an endpoint byte, a carried source identity); those positions sit at
``ActiveScoreStatus.PENDING_ASSIMILATION`` with no active score until the first forward pass scores
them.  If that score were computed with a different normalization, an injected identity would
compete for survival on a scale the natural tokens were never measured on.

PLAN §7.2 names the exact failure this suite must prevent:

    "assimilation score differs from categorical chosen-token log-probability at temperature not
    equal to one"

so the temperature-not-one case is the point, not an edge case.

The primitive lives in ``fusion_v2_runtime``, NOT in ``sampler.py``: the V1 sampler is byte-identical
to HEAD and stays that way, because V1 reference-flow results are already published and PLAN V2F3
requires "V1 continuation tests remain byte-identical".  Agreement with the V1 categorical path is
therefore proven by comparison here rather than guaranteed by shared code.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from inverse_folding.reference_flow.fusion_v2_runtime.scoring import token_logprob
from inverse_folding.reference_flow.sampler import _sample_categorical  # read-only: V1 is pristine

VOCAB = 33


def _logits(rows: int = 4, *, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, VOCAB, generator=generator, dtype=torch.float32)


# --------------------------------------------------------------------------------------------
# agreement with the categorical sampler -- the whole point of the primitive
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("temperature", [1.0, 0.5, 0.7, 1.3, 2.0])
def test_the_primitive_reproduces_the_categorical_chosen_token_score(temperature):
    """The categorical path divides logits by T and then log-softmaxes; so must the primitive."""
    logits = _logits()
    chosen, chosen_logp = _sample_categorical(
        logits / float(temperature), np.random.default_rng(11)
    )
    recovered = token_logprob(logits, chosen.numpy(), temperature=temperature)
    np.testing.assert_array_equal(recovered, chosen_logp)


def test_agreement_holds_across_many_random_draws():
    """One lucky row proves nothing; sweep seeds so a subtly different normalization shows up."""
    for seed in range(25):
        logits = _logits(8, seed=seed)
        chosen, chosen_logp = _sample_categorical(
            logits / 0.85, np.random.default_rng(seed)
        )
        np.testing.assert_array_equal(
            token_logprob(logits, chosen.numpy(), temperature=0.85), chosen_logp
        )


def test_temperature_actually_changes_the_score():
    """A primitive that ignored temperature would still pass a T=1.0-only comparison."""
    logits = _logits()
    tokens = np.array([3, 4, 5, 6])
    assert not np.allclose(
        token_logprob(logits, tokens, temperature=1.0),
        token_logprob(logits, tokens, temperature=0.5),
    )


# --------------------------------------------------------------------------------------------
# arbitrary tokens -- not merely the one the sampler happened to draw
# --------------------------------------------------------------------------------------------


def test_an_arbitrary_token_is_scored_on_the_same_scale():
    logits = _logits()
    expected = torch.log_softmax(logits / 0.9, dim=-1).numpy()
    for token in (0, 7, VOCAB - 1):
        tokens = np.full(logits.shape[0], token, dtype=np.int64)
        np.testing.assert_array_equal(
            token_logprob(logits, tokens, temperature=0.9), expected[:, token]
        )


def test_scores_of_all_tokens_in_a_row_sum_to_one_in_probability():
    logits = _logits(1)
    every = np.array([
        token_logprob(logits, np.array([t]), temperature=1.1)[0] for t in range(VOCAB)
    ])
    # float32 logits, so the achievable tolerance is ~1e-7, not 1e-9.
    assert np.exp(every).sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("bad", [VOCAB, VOCAB + 5, -1, -VOCAB])
def test_a_token_outside_the_vocabulary_fails_closed(bad):
    """Pass the RIGHT number of tokens so the row-count check cannot fire first and make this
    test pass for the wrong reason.  A negative id is the dangerous case: numpy would silently
    index from the end of the vocabulary and return a real-looking score for the wrong token."""
    tokens = np.array([1, 2, 3, bad])
    with pytest.raises(ValueError, match="must lie in"):
        token_logprob(_logits(4), tokens, temperature=1.0)


def test_the_token_count_must_match_the_row_count():
    with pytest.raises(ValueError, match="one token is scored per row"):
        token_logprob(_logits(4), np.array([1, 2]), temperature=1.0)


# --------------------------------------------------------------------------------------------
# purity: the primitive may not perturb V1 replay
# --------------------------------------------------------------------------------------------


def test_the_primitive_takes_no_generator_at_all():
    """Structural: a primitive with no rng parameter cannot consume a draw the V1 path would."""
    import inspect
    parameters = inspect.signature(token_logprob).parameters
    assert not any("rng" in name or "generator" in name for name in parameters)


def test_the_primitive_does_not_touch_the_legacy_global_numpy_rng():
    """The V1 sampler threads an explicit Generator, but a stray np.random.* call in the primitive
    would still be a hidden global side effect.  Pin the global state across a call."""
    before = np.random.get_state()
    token_logprob(_logits(), np.array([1, 2, 3, 4]), temperature=1.0)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_calling_it_mid_stream_does_not_shift_a_generator(monkeypatch):
    """End-to-end purity: two identical streams stay identical when one is interrupted by a call."""
    untouched = np.random.default_rng(5)
    interrupted = np.random.default_rng(5)
    _ = untouched.integers(0, 1000, size=3)
    _ = interrupted.integers(0, 1000, size=3)
    token_logprob(_logits(), np.array([1, 2, 3, 4]), temperature=0.8)
    np.testing.assert_array_equal(
        untouched.integers(0, 1000, size=5), interrupted.integers(0, 1000, size=5)
    )


def test_the_primitive_does_not_mutate_its_input_logits():
    logits = _logits()
    snapshot = logits.clone()
    token_logprob(logits, np.array([1, 2, 3, 4]), temperature=0.6)
    assert torch.equal(logits, snapshot)


# --------------------------------------------------------------------------------------------
# the temperature contract
# --------------------------------------------------------------------------------------------


def test_temperature_has_no_default():
    """CLAUDE.md forbids scientific defaults: the caller must state the temperature it ran at."""
    with pytest.raises(TypeError):
        token_logprob(_logits(), np.array([1, 2, 3, 4]))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_a_non_positive_or_non_finite_temperature_fails_closed(temperature):
    with pytest.raises(ValueError):
        token_logprob(_logits(), np.array([1, 2, 3, 4]), temperature=temperature)


# --------------------------------------------------------------------------------------------
# dtype and shape, so an assimilated score is storable beside a natural one
# --------------------------------------------------------------------------------------------


def test_the_score_is_float64_like_the_sampler_stores():
    out = token_logprob(_logits(), np.array([1, 2, 3, 4]), temperature=1.0)
    assert out.dtype == np.float64
    assert out.shape == (4,)


def test_a_single_row_is_accepted_and_returns_one_score():
    assert token_logprob(_logits(1), np.array([9]), temperature=1.0).shape == (1,)


def test_a_python_int_sequence_is_accepted_as_well_as_an_array():
    logits = _logits()
    np.testing.assert_array_equal(
        token_logprob(logits, [1, 2, 3, 4], temperature=0.75),
        token_logprob(logits, np.array([1, 2, 3, 4]), temperature=0.75),
    )
