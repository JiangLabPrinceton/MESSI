"""V2F4-2: the monotone exact archive and family-aware selection (PLAN §2.3, §4.2, §4.4).

The archive is the run's memory.  Three properties make it trustworthy, and each one exists to stop
a specific way the science could go wrong:

**Monotone elite.**  PLAN §2.3: the archive "must preserve the previous best definitively feasible
endpoint even if every new descendant regresses".  Without this, one bad generation erases the best
result the run ever found and the reported frontier understates the method.

**Every raw row retained.**  Selection narrows what may become ancestry; it never deletes evidence.
An archive that dropped rows would be a biased sample of what was generated, and any frontier
computed from it would overstate what the method found.

**Multiplicity buys nothing.**  PLAN §2.3: "Duplicate sequence or sibling multiplicity may not buy
extra ancestry mass".  Sibling counts are sampling variance; letting them vote would make selection
a measurement of luck.

And PLAN §4.2 gates promotion: an endpoint may enter as ``unvalidated``, but must be ``definitive``
-- with an actually evaluated, feasible structure -- before it can become feedback ancestry, support
the reported frontier, or be returned as a final design.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..fusion_v2.errors import V2Error
from ..fusion_v2.state import (
    ArchiveEntry,
    ArchiveMembership,
    CompleteEndpoint,
    FeasibilityLevel,
    advance_archive_entry,
)

__all__ = ["V2ArchiveError", "ExactArchive", "legacy_rank_key", "select_family_representatives"]

#: Ordered weakest to strongest.  Promotion may only move right (PLAN §4.2).
_ORDER = {
    FeasibilityLevel.UNVALIDATED: 0,
    FeasibilityLevel.PROVISIONAL: 1,
    FeasibilityLevel.DEFINITIVE: 2,
}


class V2ArchiveError(V2Error):
    """An archive contract violation."""


def _is_feasible_definitive(level: FeasibilityLevel, structure: Any) -> bool:
    """Elite-eligible: definitively evaluated AND actually feasible.

    ``definitive`` alone is not enough -- an endpoint whose structure was evaluated and found
    infeasible is definitively BAD, and must never lead the frontier or become ancestry.
    """
    if level is not FeasibilityLevel.DEFINITIVE:
        return False
    if structure is None or not getattr(structure, "evaluated", False):
        return False
    return bool(getattr(structure, "feasible", False))


@dataclass
class _Row:
    """One raw endpoint row plus its mutable archive bookkeeping."""

    endpoint: CompleteEndpoint
    entry: ArchiveEntry
    structure_outcome: Any


def legacy_rank_key(endpoint: CompleteEndpoint) -> tuple[float, str]:
    """``(head_global_risk, endpoint_id)`` -- the single-Head ordering, unchanged.

    Lower risk is better; the endpoint id resolves exact ties so the ordering is a function of the
    SET of endpoints rather than their arrival order.
    """
    return (float(endpoint.head_global_risk), endpoint.endpoint_id)


class ExactArchive:
    """A monotone archive of exact endpoint records.

    Deliberately NOT a dataclass: it has identity and accumulates state across a run, and making it
    look like a value would invite copies that silently diverge from the one true archive.
    """

    def __init__(self, *, rank_key: Any = None) -> None:
        """``rank_key`` selects the ordering the elite is maintained under.

        Defaults to :func:`legacy_rank_key`, so an archive constructed exactly as before behaves
        exactly as before. An alternative objective is INJECTED rather than detected: the archive
        must not decide for itself which scalar the run is optimizing, because a hidden fallback to
        the raw single-Head risk would silently make a joint run's elite single-allele while every
        artifact continued to claim otherwise.
        """
        self._rows: dict[str, _Row] = {}
        self._elite_id: str | None = None
        self._rank_key = legacy_rank_key if rank_key is None else rank_key

    # -- admission -----------------------------------------------------------------------------

    def admit(self, endpoint: CompleteEndpoint, *, depth: int) -> None:
        """Record an endpoint exactly as generated.

        Two forks producing identical bytes are two DISTINCT logical rows (PLAN §2.3: "Distinct
        logical endpoints remain visible even when sequences converge").  What they may not do is
        buy extra ancestry mass -- that is enforced at selection, not here.
        """
        if not isinstance(endpoint, CompleteEndpoint):
            raise V2ArchiveError("only a CompleteEndpoint may be admitted")
        endpoint_id = endpoint.endpoint_id
        if endpoint_id in self._rows:
            raise V2ArchiveError(
                f"endpoint {endpoint_id} is already in the archive; re-admitting the same id is a "
                "bookkeeping error, not a duplicate sibling"
            )
        entry = ArchiveEntry(
            endpoint_id=endpoint_id,
            endpoint_content_digest=endpoint.content_digest,
            lineage=endpoint.lineage,
            sequence_equivalence_key=endpoint.sequence_md5,
            feasibility_level=endpoint.feasibility_level,
            first_depth_seen=int(depth),
            last_depth_seen=int(depth),
            membership=ArchiveMembership(is_elite=False, elite_rank=None,
                                         is_diversity_frontier=False),
        )
        self._rows[endpoint_id] = _Row(
            endpoint=endpoint, entry=entry, structure_outcome=endpoint.structure_outcome,
        )
        self._reconsider_elite(endpoint_id)

    # -- promotion -----------------------------------------------------------------------------

    def promote(
        self, endpoint_id: str, *, feasibility_level: FeasibilityLevel,
        structure_outcome: Any, depth: int,
    ) -> None:
        """Advance an endpoint's feasibility.  Never regresses (PLAN §4.2)."""
        row = self._rows.get(endpoint_id)
        if row is None:
            raise V2ArchiveError(f"unknown endpoint {endpoint_id}: not in the archive")
        if _ORDER[feasibility_level] < _ORDER[row.entry.feasibility_level]:
            raise V2ArchiveError(
                f"feasibility may only advance, never regress: {row.entry.feasibility_level.value}"
                f" -> {feasibility_level.value}"
            )
        if feasibility_level is FeasibilityLevel.DEFINITIVE:
            if structure_outcome is None or not getattr(structure_outcome, "evaluated", False):
                raise V2ArchiveError(
                    "promotion to definitive requires an EVALUATED structure outcome; "
                    "'definitive' without a measurement behind it is a claim, not a result"
                )
        row.entry = advance_archive_entry(
            row.entry, feasibility_level=feasibility_level, depth=int(depth),
        )
        if structure_outcome is not None:
            row.structure_outcome = structure_outcome
        if _is_feasible_definitive(feasibility_level, structure_outcome):
            # Keep the STORED ENDPOINT consistent with its row.  Leaving the original object in
            # place would make ``elite()`` hand back a record whose own feasibility_level still
            # says "unvalidated" -- a trap for every consumer that reads the endpoint rather than
            # the row, and one no type check would catch.
            row.endpoint = replace(
                row.endpoint, feasibility_level=feasibility_level,
                structure_outcome=structure_outcome, endpoint_id=None,
            )
            # The digest is a JOIN KEY between the ``archive`` and ``complete_endpoints`` tables,
            # and ``CompleteEndpoint.canonical_payload`` includes both feasibility_level and the
            # structure outcome -- so promotion CHANGES the endpoint's content.  Computing it once
            # at admit time left the two tables naming different content for the same endpoint,
            # and the join was silently wrong for exactly the rows that reached definitive
            # feasibility, i.e. the ones every downstream result is read off.
            row.entry = replace(row.entry,
                                endpoint_content_digest=row.endpoint.content_digest)
        self._reconsider_elite(endpoint_id)

    # -- queries -------------------------------------------------------------------------------

    def raw_rows(self) -> tuple[ArchiveEntry, ...]:
        """Every admitted row, in admission order.  Never filtered -- this is the audit trail."""
        return tuple(row.entry for row in self._rows.values())

    def endpoints(self) -> tuple[CompleteEndpoint, ...]:
        """The exact endpoint records, in admission order."""
        return tuple(row.endpoint for row in self._rows.values())

    def structure_outcome(self, endpoint_id: str) -> Any:
        row = self._rows.get(endpoint_id)
        if row is None:
            raise V2ArchiveError(f"unknown endpoint {endpoint_id}: not in the archive")
        return row.structure_outcome

    def feasibility_level(self, endpoint_id: str) -> FeasibilityLevel:
        row = self._rows.get(endpoint_id)
        if row is None:
            raise V2ArchiveError(f"unknown endpoint {endpoint_id}: not in the archive")
        return row.entry.feasibility_level

    def may_become_ancestry(self, endpoint_id: str) -> bool:
        """PLAN §4.2: only a definitively feasible endpoint may become feedback ancestry."""
        row = self._rows.get(endpoint_id)
        if row is None:
            raise V2ArchiveError(f"unknown endpoint {endpoint_id}: not in the archive")
        return _is_feasible_definitive(row.entry.feasibility_level, row.structure_outcome)

    def fork_view(self) -> "ExactArchive":
        """A separate archive holding exactly the rows this one holds right now.

        Two treatment arms defined over ONE realized pre-feedback pool still generate their own
        descendants, and those must not appear in each other's history: PLAN §2.3 defines the A2
        view as the archive "immediately before feedback", so an arm that saw the other arm's
        post-feedback descendants would not be reading a control at all.

        Sharing the pre-feedback ROWS while separating what happens after is exactly the split the
        design calls for, and it is only safe because every stored object is frozen -- the fork
        copies the bookkeeping, never the evidence.
        """
        # The ORDERING LAW travels with the fork.  A default archive here silently reverted a
        # joint-ordered archive to ``head_global_risk``, so under Dual every matched arm would fork
        # an elite chosen by role A while its donor gate compared J -- selection under one law and
        # gating under another, with no artifact recording that it had happened.
        forked = ExactArchive(rank_key=self._rank_key)
        forked._rows = {
            endpoint_id: _Row(endpoint=row.endpoint, entry=row.entry,
                              structure_outcome=row.structure_outcome)
            for endpoint_id, row in self._rows.items()
        }
        forked._elite_id = self._elite_id
        return forked

    def elite(self) -> CompleteEndpoint | None:
        """The best definitively feasible endpoint seen so far, or ``None`` if there is none yet."""
        if self._elite_id is None:
            return None
        return self._rows[self._elite_id].endpoint

    # -- internals -----------------------------------------------------------------------------

    def _reconsider_elite(self, endpoint_id: str) -> None:
        """Monotone under the stable ordering ``(risk, endpoint_id)``.

        The endpoint id resolves exact-risk ties, so the elite is a function of the SET of
        endpoints seen rather than their arrival order.
        """
        row = self._rows[endpoint_id]
        if not _is_feasible_definitive(row.entry.feasibility_level, row.structure_outcome):
            return
        if self._elite_id is None:
            self._promote_elite(endpoint_id)
            return
        # Lower is better under whichever ordering the run declared.
        if self._rank_key(row.endpoint) < self._rank_key(self._rows[self._elite_id].endpoint):
            self._promote_elite(endpoint_id)

    def _promote_elite(self, endpoint_id: str) -> None:
        if self._elite_id is not None:
            previous = self._rows[self._elite_id]
            previous.entry = replace(
                previous.entry,
                membership=replace(previous.entry.membership, is_elite=False, elite_rank=None),
            )
        row = self._rows[endpoint_id]
        row.entry = replace(
            row.entry, membership=replace(row.entry.membership, is_elite=True, elite_rank=0),
        )
        self._elite_id = endpoint_id


def select_family_representatives(
    endpoints: Sequence[CompleteEndpoint],
    *,
    rank_key: Any = None,
) -> tuple[CompleteEndpoint, ...]:
    """One representative per sequence-equivalence class, best first.

    PLAN §2.3: "selection must operate on declared lineage families or sequence-equivalence classes
    while preserving all raw endpoint rows for audit", and "sibling multiplicity may not buy extra
    ancestry mass".  So a class contributes exactly one candidate no matter how many identical
    siblings it emitted -- three copies of a mediocre design cannot out-vote one copy of a better
    one.

    Only endpoints that may become ancestry are considered (PLAN §4.2).  An empty result is a
    normal, typed outcome, not an error (PLAN §4.5).

    Order-independent: the input order never changes the output.
    """
    key_of = legacy_rank_key if rank_key is None else rank_key
    eligible = [
        endpoint for endpoint in endpoints
        if _is_feasible_definitive(endpoint.feasibility_level, endpoint.structure_outcome)
    ]
    best_by_class: dict[str, CompleteEndpoint] = {}
    for endpoint in eligible:
        key = endpoint.sequence_md5
        incumbent = best_by_class.get(key)
        if incumbent is None:
            best_by_class[key] = endpoint
            continue
        # Same sequence => same Head score; break the tie on a stable identity so the choice does
        # not depend on which sibling happened to arrive first.
        if key_of(endpoint) < key_of(incumbent):
            best_by_class[key] = endpoint
    return tuple(sorted(best_by_class.values(), key=key_of))
