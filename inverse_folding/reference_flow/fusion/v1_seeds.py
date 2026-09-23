"""V1F3 seed derivation for the pre-terminal continuation-value entry layer.

Process-independent, length-delimited SHA-256 seed derivation (PLAN_RF_REFINE_FUSION_V1 §2.6).
Pure Python: no torch, no numpy, no I/O. Never Python ``hash()`` (salted / process-unstable)
and never a raw separator-join (ambiguous: ``("a","bc")`` vs ``("ab","c")`` would collide).

Seed namespaces (root / est / eval / final / random-control) are made disjoint by encoding a
distinct ``set_tag`` and field tuple into the same canonical hash, so no namespace can draw a
seed already used by another.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

# Bump when the canonical encoding changes so old seeds are never silently reproduced.
SEED_ENCODING_VERSION = "v1seed-enc-1"

_CONTINUATION_SET_TAGS = frozenset({"est", "eval", "final"})


def _encode_field(field: object) -> bytes:
    """Type-tagged, length-delimited encoding of a single seed field.

    ``int`` -> ``i<repr>``; ``str`` -> ``s<value>``. ``bool`` and ``float`` are rejected: a
    float must be canonicalized to a stable string (e.g. ``rho_id="rho0.850"``) by the caller
    so platform float rendering never enters the hash, and ``bool`` (an ``int`` subclass) would
    be ambiguous.
    """
    if isinstance(field, bool):
        raise TypeError("bool seed fields are ambiguous; pass an explicit int or str")
    if isinstance(field, float):
        raise TypeError(
            "float seed fields are non-canonical; convert to a stable str first "
            "(e.g. a canonical rho_id) so platform float rendering never enters the hash"
        )
    if isinstance(field, int):
        token = f"i{field}"
    elif isinstance(field, str):
        token = f"s{field}"
    else:
        raise TypeError(f"seed field must be int or str, got {type(field).__name__}")
    raw = token.encode("utf-8")
    return f"{len(raw)}:".encode("ascii") + raw


def derive_seed(*fields: object) -> int:
    """Deterministic, process-independent 64-bit seed from typed fields.

    The fields are encoded length-delimited and hashed with SHA-256; the first 8 bytes become
    an unsigned 64-bit integer suitable for ``numpy.random.default_rng``. Identical field
    tuples always give the same seed on any machine / interpreter hash seed; different tuples
    (including int-vs-str of the same characters, or different arities) never collide by
    construction of the length-delimited encoding.
    """
    payload = b"\x1e".join(_encode_field(f) for f in fields)
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def assert_no_seed_collisions(seeds: Mapping[str, int]) -> None:
    """Fail-fast if any two named seeds share a value (PLAN §2.6: a derived-seed collision is a
    hard error, never silently tolerated)."""
    by_value: dict[int, str] = {}
    for name, value in seeds.items():
        if value in by_value:
            raise ValueError(
                f"seed collision: {name!r} and {by_value[value]!r} both derived {value}"
            )
        by_value[value] = name


#: The V1-A entry arms. The arm is part of every seed prefix, so this enum is what makes the
#: shared-pool law STRUCTURAL: a policy label (selected_partial / random_partial) can never be
#: passed where an arm belongs, and therefore can never fork its own root pool or eval table.
_ENTRY_ARMS = frozenset({"terminal", "preterminal"})


def _exact_index(value, name: str) -> int:
    """A stream index must be an EXACT int. ``int(1.9)``, ``int("1")`` and ``int(True)`` all
    collapse to 1, which would silently alias three different callers onto one seed stream."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an exact int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


@dataclass(frozen=True)
class SeedContext:
    """Campaign/phase/split-scoped seed factory. All streams share this prefix and differ only
    in their set-tag and per-stream fields, so root / est / eval / final / terminal-complete /
    independent-full namespaces are disjoint (PLAN §2.6). ``rho_id`` is a pre-canonicalized
    string, never a float, so the same maturity always hashes identically.

    The T0 policy views (``selected_partial`` / ``random_partial``) are MEMBERSHIP views over the
    same pre-terminal root pool and the same held-out evaluation table. They share this context
    verbatim; only :meth:`random_membership_seed` distinguishes them, and it decides WHICH roots
    the random view holds -- never a fresh continuation draw.
    """

    seed_schema: str
    campaign_id: str
    phase: str
    split_role: str
    master_seed: int
    entry_arm: str
    protein_id: str
    rho_id: str
    #: The T0 config's declared ``random_membership_seed``. It enters ONLY the membership draw, so
    #: re-drawing the random control as a robustness replicate leaves every continuation seed --
    #: and therefore the selected view and the shared eval table -- byte-identical. Without it the
    #: config field would be inert: mandatory, validated, digest-bearing, and unable to change
    #: anything the run does. Named distinctly from :meth:`random_membership_seed` so the declared
    #: INPUT and the derived seed can never be confused for one another.
    declared_membership_seed: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.rho_id, float):
            raise TypeError("rho_id must be a canonical str, not a float")
        if self.phase not in ("t0", "p1"):
            raise ValueError(f"phase must be 't0' or 'p1', got {self.phase!r}")
        if self.entry_arm not in _ENTRY_ARMS:
            raise ValueError(
                f"entry_arm must be one of {sorted(_ENTRY_ARMS)}, got {self.entry_arm!r} "
                "(a policy label is not an arm: selected/random are membership views over ONE "
                "shared root pool and ONE shared evaluation table)"
            )

    def _prefix(self) -> tuple[object, ...]:
        return (
            SEED_ENCODING_VERSION,
            self.seed_schema,
            self.campaign_id,
            self.phase,
            self.split_role,
            int(self.master_seed),
            self.entry_arm,
            self.protein_id,
            self.rho_id,
        )

    def _require_preterminal(self, stream: str) -> None:
        if self.entry_arm != "preterminal":
            raise ValueError(
                f"{stream} is a pre-terminal stream; entry_arm={self.entry_arm!r} has no root "
                "prefix and no estimator set"
            )

    def root_seed(self, root_index: int) -> int:
        """Seed for the ``root_index``-th root-prefix attempt (lineage, pre-collapse)."""
        self._require_preterminal("root_seed")
        return derive_seed(*self._prefix(), "root", _exact_index(root_index, "root_index"))

    def continuation_seed(
        self, root_equivalence_hash: str, set_tag: str, replicate_index: int
    ) -> int:
        """Seed for an est / eval / final continuation of a unique root (bound to its
        equivalence hash, so the representative choice or input order cannot change draws)."""
        if set_tag not in _CONTINUATION_SET_TAGS:
            raise ValueError(
                f"set_tag must be one of {sorted(_CONTINUATION_SET_TAGS)}, got {set_tag!r}"
            )
        self._require_preterminal("continuation_seed")
        return derive_seed(
            *self._prefix(), str(root_equivalence_hash), set_tag,
            _exact_index(replicate_index, "replicate_index"),
        )

    def random_membership_seed(self) -> int:
        """The SINGLE membership draw that decides which roots the ``random_partial`` view holds.

        It is deliberately un-indexed and creates no continuation stream: the random view reads the
        same per-root evaluation table as the selected view, so switching policy label must leave
        every continuation seed byte-identical (PLAN §2.9-§2.10).

        The config's declared value participates so the control can be RE-DRAWN without disturbing
        anything else. ``None`` (no declared value) keeps the pure-namespace derivation, which is
        what the P1 arms and older T0 configs resolve to."""
        parts = [*self._prefix(), "policy_membership", "random_partial"]
        if self.declared_membership_seed is not None:
            parts.append(_exact_index(self.declared_membership_seed, "declared_membership_seed"))
        return derive_seed(*parts)

    def terminal_complete_seed(self, replicate_index: int) -> int:
        """Seed for one independent complete trajectory of the TERMINAL arm. The terminal arm has
        no maturity, so this stream is only valid in a terminal context."""
        if self.entry_arm != "terminal":
            raise ValueError(
                f"terminal_complete_seed requires entry_arm='terminal', got {self.entry_arm!r}"
            )
        return derive_seed(*self._prefix(), "terminal_complete",
                           _exact_index(replicate_index, "replicate_index"))

    def independent_full_seed(self, replicate_index: int) -> int:
        """Seed for one compute-matched independent full trajectory (the T0 ``independent_full``
        policy). ``control_id`` substitutes for the arm in the namespace, since this control is
        not a pre-terminal-prefixed stream (PLAN §2.6)."""
        prefix = self._prefix()
        control_prefix = prefix[:6] + ("independent_full",) + prefix[7:]
        return derive_seed(*control_prefix, "full_trajectory",
                           _exact_index(replicate_index, "replicate_index"))


def realized_seed_manifest(
    ctx: SeedContext,
    *,
    root_count: int,
    unique_root_hashes: "Sequence[str]",
    k_est: int,
    k_eval: int,
    final_count: int,
    #: The roots that ACTUALLY ran a final completion (the ranked beam). Defaults to every unique
    #: root only for callers that materialize all of them.
    final_root_hashes: "Sequence[str] | None" = None,
    independent_full_count: int = 0,
    terminal_count: int = 0,
    include_random_membership: bool = False,
) -> dict[str, int]:
    """Enumerate EVERY realized seed for a run into a named manifest and fail-fast on any
    collision (PLAN §2.6: the resolved derivation and every realized seed are persisted, and a
    derived-seed collision is a hard error)."""
    seeds: dict[str, int] = {}
    for i in range(int(root_count)):
        seeds[f"root:{i}"] = ctx.root_seed(i)
    for root_hash in unique_root_hashes:
        for k in range(int(k_est)):
            seeds[f"est:{root_hash}:{k}"] = ctx.continuation_seed(root_hash, "est", k)
        for k in range(int(k_eval)):
            seeds[f"eval:{root_hash}:{k}"] = ctx.continuation_seed(root_hash, "eval", k)
    # `final` continuations are drawn ONLY for the ranked beam -- the unique-root pool is a minimum
    # coverage requirement, not a truncation, so it is normally larger. Enumerating a final seed per
    # unique root would claim draws that never happened and contradict the run's own
    # `root_selection` / `continuations`; this table is named for what was REALIZED.
    for root_hash in (unique_root_hashes if final_root_hashes is None else final_root_hashes):
        for k in range(int(final_count)):
            seeds[f"final:{root_hash}:{k}"] = ctx.continuation_seed(root_hash, "final", k)
    if include_random_membership:
        # exactly ONE membership draw, never a per-replicate stream
        seeds["random_membership"] = ctx.random_membership_seed()
    for i in range(int(independent_full_count)):
        seeds[f"independent_full:{i}"] = ctx.independent_full_seed(i)
    for i in range(int(terminal_count)):
        seeds[f"terminal_complete:{i}"] = ctx.terminal_complete_seed(i)
    assert_no_seed_collisions(seeds)
    return seeds


# --- Public aliases for fusion_v2 reuse (additive; zero behavior change) ---------------------
# exact_index rejects bool and non-int, preventing int(1.9) / int("1") / int(True) from aliasing
# three callers onto one random stream.
exact_index = _exact_index
