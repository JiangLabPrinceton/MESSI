"""V2 seed namespaces and the matched-pair derivation law.

Two jobs. The ordinary namespaces (depth-0 root, per-depth lookahead fork, A2 extra lookahead) must
be provably disjoint from each other and from every V1 namespace. The matched-pair namespace of
PLAN §2.6 must be provably independent of anything that differs between the two arms of a pair.

That second property is the load-bearing one. Every paired control in V2F5 -- change the endpoint,
change or ablate the source, shuffle the source -- rests on the two arms sharing a descendant random
stream so the only difference is the intervention. If a treatment digest could reach its own seed,
the two arms would silently compare different random streams instead of different interventions, and
nothing would fail. So the exclusion is structural rather than conventional:

* every input to :class:`FeedbackPairSeedContext` is declared config plus a schedule coordinate plus
  an ordinal, so :meth:`~FeedbackPairSeedContext.enumerate_pair_seeds` runs at ``--dry-run`` with no
  model loaded. A value computable before the arms exist cannot depend on them;
* the only per-call arguments in the paired path are bounded exact ints -- there is no ``str``,
  ``bytes`` or ``Mapping`` parameter for a digest to travel through;
* the contexts are ``frozen``/``slots`` with no free-form field, and ``pair_id`` is derived rather
  than accepted; and
* treatment content lives only on :class:`MatchedPairSeedRecord`, which no derivation function ever
  accepts. Because the record re-derives its seeds from the context alone, an implementation that
  folded a treatment field into a seed fails at record construction on every real run.

The two contexts deliberately share no base class: a field later added to :class:`V2SeedContext`
must not be able to leak into the paired identity.

This module derives seeds only. It never constructs an RNG, touches a sampler, or sees a state,
endpoint, projected byte, or policy result. Purity: no torch, no numpy, no I/O, no hashing of its
own -- it reuses the audited V1 derivation unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from inverse_folding.reference_flow.fusion.v1_seeds import (
    assert_no_seed_collisions,
    derive_seed,
    exact_index,
)

from .errors import V2Error

__all__ = [
    "V2SeedError", "MatchedPairSeedError", "V2_SEED_ENCODING_VERSION", "V2_SEED_NAMESPACES",
    "ROOT_CAPTURE_MAX_ATTEMPTS",
    "PAIR_ARM_SLOTS", "V2SeedContext", "FeedbackPairSeedContext", "MatchedPairSeedRecord",
    "derive_pair_id", "assert_matched_pair_seeds", "realized_v2_seed_manifest",
]

#: Field 0 of every V2 hashed tuple. Deliberately a different *value* from V1's
#: ``SEED_ENCODING_VERSION`` so the two encodings bump independently and no V2 tuple can ever
#: positionally align with a V1 one.
V2_SEED_ENCODING_VERSION = "v2seed-enc-1"

#: The closed set of V2 namespaces. Each tag is hard-wired inside exactly one derivation method and
#: is never a parameter, so a caller cannot request an arbitrary namespace.
V2_SEED_NAMESPACES = frozenset({
    "v2_depth0_root", "v2_lookahead", "a2_extra_lookahead", "matched_descendant",
})

# ponytail: eight makes an m=1 capture miss <0.4%; make this config-driven only if one fixed
# ceiling proves inadequate across editable-domain strata.
ROOT_CAPTURE_MAX_ATTEMPTS = 8

#: Reporting labels for the two rows of a pair. Never hashed.
PAIR_ARM_SLOTS = frozenset({"arm_a", "arm_b"})

_NS_DEPTH0_ROOT = "v2_depth0_root"
_NS_LOOKAHEAD = "v2_lookahead"
_NS_A2_EXTRA = "a2_extra_lookahead"
_NS_MATCHED = "matched_descendant"


class V2SeedError(V2Error):
    """A seed identity or derivation contract was violated."""


class MatchedPairSeedError(V2SeedError):
    """The PLAN §2.6 matched-pair law was violated.

    Separate from :class:`V2SeedError` so a runner can catch it and record a typed null/stalled
    outcome instead of dying (PLAN §2.4, §4.5).
    """


def _require_text(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise V2SeedError(f"{name} must be a non-empty str, got {value!r}")
    return value


def _require_exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    try:
        out = exact_index(value, name)
    except (TypeError, ValueError) as exc:  # v1_seeds raises the repo's own exact-int errors
        raise V2SeedError(f"{name} must be an exact int, got {value!r}") from exc
    if out < minimum:
        raise V2SeedError(f"{name} must be >= {minimum}, got {out}")
    return out


def derive_pair_id(
    *, protein_id: str, depth: int, r_step: int, c_next_step: int, pair_ordinal: int,
) -> str:
    """Render the human-readable pair label from pre-arm components only.

    Callable at preflight, before either arm exists. The label is for reporting; the *hashed* tuple
    carries the typed components rather than this rendered string, so no separator ambiguity can
    change a seed.
    """
    _require_text(protein_id, "protein_id")
    return (
        f"{protein_id}:d{_require_exact_int(depth, 'depth')}"
        f":r{_require_exact_int(r_step, 'r_step')}"
        f"-c{_require_exact_int(c_next_step, 'c_next_step')}"
        f":p{_require_exact_int(pair_ordinal, 'pair_ordinal')}"
    )


@dataclass(frozen=True, slots=True)
class V2SeedContext:
    """Run-level seed identity for the ordinary V2 namespaces.

    There is deliberately no ``arm``/``view`` field. A2 and V2 share one pre-feedback pool
    (PLAN §2.3, §4.3), so they must be structurally unable to draw different lookahead seeds. This
    is the mirror image of V1's decision to hash the entry arm, where the two arms are genuinely
    separate pools.
    """

    seed_schema: str
    campaign_id: str
    split_role: str
    master_seed: int
    protein_id: str

    def __post_init__(self) -> None:
        for name in ("seed_schema", "campaign_id", "split_role", "protein_id"):
            _require_text(getattr(self, name), name)
        _require_exact_int(self.master_seed, "master_seed")

    def _prefix(self, namespace: str) -> tuple[object, ...]:
        return (
            V2_SEED_ENCODING_VERSION, self.seed_schema, self.campaign_id, self.split_role,
            self.master_seed, self.protein_id, namespace,
        )

    def depth0_root_seed(self, *, checkpoint_step: int, root_index: int) -> int:
        """Seed for the ``root_index``-th depth-0 fresh-root capture attempt at ``c_0``."""
        return derive_seed(
            *self._prefix(_NS_DEPTH0_ROOT),
            _require_exact_int(checkpoint_step, "checkpoint_step"),
            _require_exact_int(root_index, "root_index"),
        )

    def lookahead_seed(self, *, depth: int, source_state_id: str, lookahead_index: int) -> int:
        """Seed for one exact lookahead forked from a named committed live state.

        Bound to both ``depth`` and ``source_state_id``: stationary recurrence repeats a sampler
        step across depths, and two active-population members share a depth, so neither alone
        identifies a stream.
        """
        return derive_seed(
            *self._prefix(_NS_LOOKAHEAD),
            _require_exact_int(depth, "depth"),
            _require_text(source_state_id, "source_state_id"),
            _require_exact_int(lookahead_index, "lookahead_index"),
        )

    def a2_extra_lookahead_seed(
        self, *, depth: int, source_state_id: str, extra_index: int,
    ) -> int:
        """Seed for one *additional* A2 exact future reallocated from post-feedback work.

        A separate namespace from ``v2_lookahead`` so an extra A2 future can never repeat a draw
        from the shared pre-feedback pool (PLAN §4.3).
        """
        return derive_seed(
            *self._prefix(_NS_A2_EXTRA),
            _require_exact_int(depth, "depth"),
            _require_text(source_state_id, "source_state_id"),
            _require_exact_int(extra_index, "extra_index"),
        )


@dataclass(frozen=True, slots=True)
class FeedbackPairSeedContext:
    """The complete, closed seed identity of one matched pair (PLAN §2.6).

    The hashed identity is exactly: encoding version, seed schema, campaign, split, master seed,
    protein, the ``matched_descendant`` namespace, depth, the shared ``r_d`` and ``c_{d+1}``, and
    the pair ordinal. ``n_forks`` and ``n_descendant_lookaheads`` are bounds, not hashed content.

    Nothing here can carry an arm label, an intervention label, an endpoint/source/state digest,
    projected bytes, a transition ID, or a policy digest -- those are treatment identities, and
    including one would silently unpair the downstream randomness.
    """

    seed_schema: str
    campaign_id: str
    split_role: str
    master_seed: int
    protein_id: str
    depth: int
    r_step: int
    c_next_step: int
    pair_ordinal: int
    n_forks: int
    n_descendant_lookaheads: int

    def __post_init__(self) -> None:
        for name in ("seed_schema", "campaign_id", "split_role", "protein_id"):
            _require_text(getattr(self, name), name)
        _require_exact_int(self.master_seed, "master_seed")
        _require_exact_int(self.depth, "depth")
        _require_exact_int(self.r_step, "r_step")
        _require_exact_int(self.c_next_step, "c_next_step")
        _require_exact_int(self.pair_ordinal, "pair_ordinal")
        _require_exact_int(self.n_forks, "n_forks", minimum=1)
        _require_exact_int(self.n_descendant_lookaheads, "n_descendant_lookaheads", minimum=1)
        if self.r_step >= self.c_next_step:
            raise V2SeedError(
                f"r_step={self.r_step} must precede c_next_step={self.c_next_step}; the full "
                "(r_d, c_d, c_{d+1}) ordering is checked by the schedule layer, which owns c_d"
            )

    @property
    def pair_id(self) -> str:
        """Derived, never an input (PLAN §2.6: 'frozen before either arm exists')."""
        return derive_pair_id(
            protein_id=self.protein_id, depth=self.depth, r_step=self.r_step,
            c_next_step=self.c_next_step, pair_ordinal=self.pair_ordinal,
        )

    def _prefix(self) -> tuple[object, ...]:
        return (
            V2_SEED_ENCODING_VERSION, self.seed_schema, self.campaign_id, self.split_role,
            self.master_seed, self.protein_id, _NS_MATCHED, self.depth, self.r_step,
            self.c_next_step, self.pair_ordinal,
        )

    def _fork(self, fork_index: int) -> int:
        index = _require_exact_int(fork_index, "fork_index")
        if index >= self.n_forks:
            raise V2SeedError(
                f"fork_index={index} is outside the frozen fork count n_forks={self.n_forks}; the "
                "bound is also the anti-smuggling guard, since a content digest coerced to an int "
                "cannot land inside a small declared range"
            )
        return index

    def matched_descendant_seed(self, fork_index: int) -> int:
        """The propagation fork seed for one matched descendant, ``r_d -> c_{d+1}``.

        Both arms of a pair call this with identical arguments and therefore receive the identical
        seed. That equality is the control.
        """
        return derive_seed(*self._prefix(), "propagation", self._fork(fork_index))

    def matched_descendant_lookahead_seed(self, fork_index: int, lookahead_index: int) -> int:
        """The seed for one descendant lookahead forked from the captured ``c_{d+1}`` state."""
        index = _require_exact_int(lookahead_index, "lookahead_index")
        if index >= self.n_descendant_lookaheads:
            raise V2SeedError(
                f"lookahead_index={index} is outside the frozen breadth "
                f"n_descendant_lookaheads={self.n_descendant_lookaheads}"
            )
        return derive_seed(*self._prefix(), "lookahead", self._fork(fork_index), index)

    def enumerate_pair_seeds(self) -> dict[str, int]:
        """Every seed of this pair, computable before either arm exists.

        This is the executable proof of the exclusion law: it runs at ``--print-config`` /
        ``--dry-run`` with no model loaded, so a single test -- "the dry-run table equals the
        realized table" -- closes the whole PLAN §2.6 exclusion list at once.
        """
        table: dict[str, int] = {}
        for fork in range(self.n_forks):
            table[f"fork{fork}:propagation"] = self.matched_descendant_seed(fork)
            for index in range(self.n_descendant_lookaheads):
                table[f"fork{fork}:lookahead{index}"] = self.matched_descendant_lookahead_seed(
                    fork, index
                )
        return table


@dataclass(frozen=True, slots=True)
class MatchedPairSeedRecord:
    """One arm's persisted row of the PLAN §2.6 artifact.

    Treatment content lives here and only here. No derivation function accepts a record, and the
    realized seeds are re-derived from ``context`` alone at construction -- so a seed that is not
    the derived seed cannot be written, on every real run rather than only under test.
    """

    context: FeedbackPairSeedContext
    fork_index: int
    arm_slot: str
    intervention_kind: str
    treatment_identity: str
    shared_match_identity: str
    c_source_step: int
    realized_propagation_seed: int
    realized_lookahead_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.context, FeedbackPairSeedContext):
            raise MatchedPairSeedError("context must be a FeedbackPairSeedContext")
        fork = _require_exact_int(self.fork_index, "fork_index")
        if fork >= self.context.n_forks:
            raise MatchedPairSeedError(
                f"fork_index={fork} outside n_forks={self.context.n_forks}"
            )
        if self.arm_slot not in PAIR_ARM_SLOTS:
            raise MatchedPairSeedError(
                f"arm_slot must be one of {sorted(PAIR_ARM_SLOTS)}, got {self.arm_slot!r}"
            )
        for name in ("intervention_kind", "treatment_identity", "shared_match_identity"):
            _require_text(getattr(self, name), name)
        _require_exact_int(self.c_source_step, "c_source_step")

        expected = self.context.matched_descendant_seed(fork)
        if self.realized_propagation_seed != expected:
            raise MatchedPairSeedError(
                f"realized_propagation_seed {self.realized_propagation_seed} is not the derived "
                f"seed {expected} for fork {fork}; a treatment identity must never reach a matched "
                "descendant seed (PLAN §2.6)"
            )
        realized = tuple(self.realized_lookahead_seeds)
        if len(realized) > self.context.n_descendant_lookaheads:
            raise MatchedPairSeedError(
                f"{len(realized)} realized lookahead seeds exceed the frozen breadth "
                f"{self.context.n_descendant_lookaheads}"
            )
        for index, value in enumerate(realized):
            want = self.context.matched_descendant_lookahead_seed(fork, index)
            if value != want:
                raise MatchedPairSeedError(
                    f"realized_lookahead_seeds[{index}] = {value} is not the derived seed {want}"
                )


def assert_matched_pair_seeds(records: Sequence[MatchedPairSeedRecord]) -> None:
    """Enforce the PLAN §2.6 pairing law over a set of persisted rows.

    Per ``(pair_id, fork_index)`` group: exactly two rows, the two contexts equal, the two arm slots
    the two distinct members of :data:`PAIR_ARM_SLOTS`, identical realized seeds of identical
    length, identical ``intervention_kind`` / ``shared_match_identity`` / ``c_source_step``, and
    **distinct** ``treatment_identity``.

    The distinctness check is not pedantry: it catches a control that silently equals its treatment,
    which would read as a null effect rather than as a broken design.
    """
    groups: dict[tuple[str, int], list[MatchedPairSeedRecord]] = {}
    for row in records:
        if not isinstance(row, MatchedPairSeedRecord):
            raise MatchedPairSeedError("every element must be a MatchedPairSeedRecord")
        groups.setdefault((row.context.pair_id, row.fork_index), []).append(row)

    for (pair_id, fork), rows in sorted(groups.items()):
        where = f"pair {pair_id!r} fork {fork}"
        if len(rows) != 2:
            raise MatchedPairSeedError(f"{where}: expected exactly 2 arms, got {len(rows)}")
        a, b = rows
        if a.context != b.context:
            raise MatchedPairSeedError(f"{where}: the two arms declare different seed contexts")
        if {a.arm_slot, b.arm_slot} != PAIR_ARM_SLOTS:
            raise MatchedPairSeedError(
                f"{where}: arm slots must be {sorted(PAIR_ARM_SLOTS)}, got "
                f"{sorted({a.arm_slot, b.arm_slot})}"
            )
        for name in ("intervention_kind", "shared_match_identity", "c_source_step"):
            if getattr(a, name) != getattr(b, name):
                raise MatchedPairSeedError(
                    f"{where}: {name} differs across the pair "
                    f"({getattr(a, name)!r} vs {getattr(b, name)!r}); it is held fixed by design"
                )
        if a.treatment_identity == b.treatment_identity:
            raise MatchedPairSeedError(
                f"{where}: both arms carry treatment identity {a.treatment_identity!r}; a control "
                "that equals its treatment is not a paired intervention"
            )
        if a.realized_propagation_seed != b.realized_propagation_seed:
            raise MatchedPairSeedError(f"{where}: propagation seeds differ across the pair")
        if a.realized_lookahead_seeds != b.realized_lookahead_seeds:
            raise MatchedPairSeedError(
                f"{where}: lookahead seed vectors differ across the pair "
                f"({len(a.realized_lookahead_seeds)} vs {len(b.realized_lookahead_seeds)} entries)"
            )


def realized_v2_seed_manifest(
    context: V2SeedContext,
    *,
    depth0_root_draws: Sequence[tuple[int, int]],
    lookahead_draws: Sequence[tuple[int, str, int]] = (),
    a2_extra_draws: Sequence[tuple[int, str, int]] = (),
    pair_records: Sequence[MatchedPairSeedRecord] = (),
) -> dict[str, int]:
    """Enumerate every *realized* seed of a run into one named map, failing closed on collision.

    Takes explicit realized draw tuples rather than counts, so the table cannot over-claim draws
    that never happened. Each pair's two arms collapse to one entry per fork before the collision
    check -- a naive per-arm map would false-positive on the intended within-pair equality.
    """
    if not isinstance(context, V2SeedContext):
        raise V2SeedError("context must be a V2SeedContext")
    assert_matched_pair_seeds(pair_records)

    manifest: dict[str, int] = {}

    def put(key: str, value: int) -> None:
        if key in manifest:
            raise V2SeedError(
                f"duplicate seed manifest key {key!r}; a repeated realized draw or an ambiguous "
                "source_state_id is a hard error, never a silent overwrite"
            )
        manifest[key] = value

    for checkpoint_step, root_index in depth0_root_draws:
        put(
            f"v2_depth0_root:c{checkpoint_step}:r{root_index}",
            context.depth0_root_seed(checkpoint_step=checkpoint_step, root_index=root_index),
        )
    for depth, source_state_id, index in lookahead_draws:
        put(
            f"v2_lookahead:d{depth}:{source_state_id}:k{index}",
            context.lookahead_seed(
                depth=depth, source_state_id=source_state_id, lookahead_index=index
            ),
        )
    for depth, source_state_id, index in a2_extra_draws:
        put(
            f"a2_extra_lookahead:d{depth}:{source_state_id}:k{index}",
            context.a2_extra_lookahead_seed(
                depth=depth, source_state_id=source_state_id, extra_index=index
            ),
        )
    for row in pair_records:
        base = f"matched_descendant:{row.context.pair_id}:f{row.fork_index}"
        if f"{base}:propagation" not in manifest:
            put(f"{base}:propagation", row.realized_propagation_seed)
            for index, value in enumerate(row.realized_lookahead_seeds):
                put(f"{base}:lookahead{index}", value)

    assert_no_seed_collisions(manifest)
    return manifest
