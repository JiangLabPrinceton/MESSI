"""V2F7: the compute ledger, hard caps, and the append-only attempt journal (PLAN §3.4, §5.3-5.4).

PLAN §5.4 requires three quantities to stay in separate fields, and the reason is that they answer
different questions and only one of them is a measurement:

**Logical assigned work** -- what the method decided to spend.  Counted ONCE per logical event, so a
retry cannot inflate the number a matched-compute claim is compared on.

**Observed physical work** -- what the machine actually burned and reported.  Summed per distinct
attempt, so re-reading a fragment during resume does not double-charge it.

**Unknown physical work** -- what a backend burned but could not report.  PLAN §5.4: persist
``unknown_after_start`` "rather than zero or a guessed value".  A zero would read as "this attempt
was free", which is the one thing it certainly was not, and it would make an over-budget run look
compliant.

The writer is reused: :func:`scripts.rf_fusion_v1_artifacts.write_cost_ledger_jsonl` serializes any
dataclass through ``asdict``, so V2 gets the audited append-safe JSONL format without a second
implementation.  What V2 does *not* reuse is ``fusion.v1_ledger.LedgerEvent`` itself: it has no
GPU-seconds field and no way to say "unknown", and V1's ledger schema sits behind published results.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..fusion_v2.errors import V2Error
from ..fusion_v2.identity import canonical_digest

__all__ = [
    "UNKNOWN_AFTER_START",
    "OBSERVED",
    "LEDGER_PHASES",
    "LEDGER_STATUSES",
    "V2LedgerError",
    "CapsExceeded",
    "V2LedgerEvent",
    "CapVerdict",
    "AttemptJournal",
    "AttemptReceipt",
    "CostMeter",
    "aggregate_v2_ledger",
    "check_caps",
    "assert_within_caps",
    "events_from_journal",
]

#: The sentinel PLAN §5.4 requires when a backend cannot report what it burned before failing.
UNKNOWN_AFTER_START = "unknown_after_start"
OBSERVED = "observed"
_PHYSICAL_STATUSES = frozenset({OBSERVED, UNKNOWN_AFTER_START})

#: Closed sets.  An open vocabulary would let a typo create a phase nothing aggregates.
LEDGER_PHASES = frozenset({
    "root_capture", "screen", "head", "structure", "projection", "segment", "descendant_screen",
})
LEDGER_STATUSES = frozenset({"ok", "failed", "retried", "deferred"})

#: Counted once per ``event_id``: what the method ASSIGNED.
_LOGICAL_FIELDS = ("logical_dfe", "head_calls", "structure_attempts")
#: Summed per distinct ``attempt_id``: what the machine REPORTED.
_PHYSICAL_FIELDS = ("physical_forwards", "structure_cache_hits", "head_cache_hits",
                    "gpu_seconds", "walltime_s")
_LOGICAL_IDENTITY = ("event_id", "protein_id", "arm", "phase")


class V2LedgerError(V2Error):
    """A ledger or journal contract violation."""


class CapsExceeded(V2LedgerError):
    """A hard cap was breached, or could not be verified.

    Separate from the generic error so a driver can exit on a budget breach with its own code
    rather than treating it as corruption.
    """


@dataclass(frozen=True)
class V2LedgerEvent:
    """One row of the V2 compute ledger (PLAN §5.3).

    ``event_id`` is the LOGICAL identity and ``attempt_id`` the PHYSICAL one.  They are separate
    because one decision can be executed several times: a retried oracle call is one logical event
    and two physical attempts.
    """

    event_id: str
    attempt_id: str
    protein_id: str
    arm: str
    phase: str
    status: str

    # -- logical assigned work ------------------------------------------------------------------
    logical_dfe: int = 0
    head_calls: int = 0
    structure_attempts: int = 0

    # -- observed physical work -----------------------------------------------------------------
    physical_forwards: int = 0
    #: How many lanes shared each of those forwards (PLAN §3.1 "physical batched forwards").
    batch_size: int = 1
    #: Whether THIS row carries the group's shared physical cost.  Exactly one lane of a batch
    #: does; the others declare their logical spend and nothing physical, so a batched forward is
    #: charged once rather than once per lane.
    physical_forwards_charged: bool = True
    structure_cache_hits: int = 0
    head_cache_hits: int = 0
    gpu_seconds: float = 0.0
    walltime_s: float = 0.0

    # -- unknown physical work ------------------------------------------------------------------
    physical_cost_status: str = OBSERVED

    #: Descriptive only, never summed into a total: what happened, independent of what was charged.
    outcome: str | None = None

    def __post_init__(self) -> None:
        for name in _LOGICAL_IDENTITY + ("attempt_id",):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise V2LedgerError(f"{name} must be a non-empty str, got {value!r}")
        if self.phase not in LEDGER_PHASES:
            raise V2LedgerError(
                f"phase must be one of {sorted(LEDGER_PHASES)}, got {self.phase!r}")
        if self.status not in LEDGER_STATUSES:
            raise V2LedgerError(
                f"status must be one of {sorted(LEDGER_STATUSES)}, got {self.status!r}")
        if self.physical_cost_status not in _PHYSICAL_STATUSES:
            raise V2LedgerError(
                f"physical_cost_status must be one of {sorted(_PHYSICAL_STATUSES)}, got "
                f"{self.physical_cost_status!r}"
            )
        for name in _LOGICAL_FIELDS + ("physical_forwards", "structure_cache_hits",
                                       "head_cache_hits", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise V2LedgerError(f"{name} must be a non-negative int, got {value!r}")
        if self.batch_size < 1:
            raise V2LedgerError(f"batch_size must be >= 1, got {self.batch_size}")
        for name in ("gpu_seconds", "walltime_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise V2LedgerError(f"{name} must be a number, got {value!r}")
            if float(value) < 0.0 or float(value) != float(value):
                raise V2LedgerError(f"{name} must be finite and >= 0, got {value!r}")
        if not isinstance(self.physical_forwards_charged, bool):
            raise V2LedgerError("physical_forwards_charged must be an explicit bool")
        if self.physical_cost_status == UNKNOWN_AFTER_START:
            reported = [
                name for name in ("physical_forwards", "structure_cache_hits", "head_cache_hits")
                if getattr(self, name)
            ] + [name for name in ("gpu_seconds", "walltime_s") if float(getattr(self, name))]
            if reported:
                raise V2LedgerError(
                    f"physical_cost_status is {UNKNOWN_AFTER_START!r} but the row also reports "
                    f"{reported}; a cost is either measured or unknown, and a row claiming both "
                    "would let a guessed number be read as a measurement"
                )


def _logical_identity(event: V2LedgerEvent) -> tuple:
    return tuple(getattr(event, name) for name in _LOGICAL_IDENTITY)


def _physical_signature(event: V2LedgerEvent) -> tuple:
    return (
        event.physical_cost_status, event.physical_forwards_charged, event.batch_size,
        event.physical_forwards, event.structure_cache_hits, event.head_cache_hits,
        round(float(event.gpu_seconds), 9), round(float(event.walltime_s), 9),
    )


def aggregate_v2_ledger(events: Sequence[V2LedgerEvent]) -> dict:
    """Idempotently aggregate an append-only V2 ledger.

    Logical fields are counted once per assigned ``event_id``; physical fields sum once per
    distinct ``attempt_id``.  Both rules exist so that RESUME -- which re-reads fragments it has
    already seen -- produces the same totals as a clean run.  A conflicting second row is an error
    rather than a last-write-wins overwrite: one physical execution has one cost, and silently
    keeping the newer row would let a bogus count replace a measured one.
    """
    assigned: dict[str, V2LedgerEvent] = {}
    physical_by_attempt: dict[str, V2LedgerEvent] = {}
    attempt_owner: dict[str, tuple] = {}
    event_owner: dict[str, tuple] = {}
    unknown_attempts: set[str] = set()
    attempts_by_event: dict[str, set[str]] = {}
    seen_rows: set[tuple] = set()
    uncharged_lanes: list[V2LedgerEvent] = []

    for event in events:
        if not isinstance(event, V2LedgerEvent):
            raise V2LedgerError("every ledger row must be a V2LedgerEvent")
        identity = _logical_identity(event)

        owner = event_owner.get(event.event_id)
        if owner is not None and owner != identity:
            raise V2LedgerError(
                f"event_id {event.event_id!r} used for two different logical events: {owner!r} "
                f"vs {identity!r}; the same id under a different protein or arm would merge two "
                "runs' method cost"
            )
        event_owner[event.event_id] = identity

        row_key = (event.event_id, event.attempt_id)
        if row_key in seen_rows:
            # An exact replay of a row already aggregated.  Idempotent by construction.
            pass
        else:
            seen_rows.add(row_key)

        if event.physical_forwards_charged:
            bound = attempt_owner.get(event.attempt_id)
            prior = physical_by_attempt.get(event.attempt_id)
            if prior is not None and _physical_signature(prior) != _physical_signature(event):
                raise V2LedgerError(
                    f"conflicting physical values for attempt {event.attempt_id!r}; one physical "
                    "execution has one cost"
                )
            attempt_owner[event.attempt_id] = bound or identity
            physical_by_attempt[event.attempt_id] = event
            if event.physical_cost_status == UNKNOWN_AFTER_START:
                unknown_attempts.add(event.attempt_id)
        else:
            # Deferred to a SECOND pass.  Checking here compared the lane against whatever had
            # already arrived, so the same row set raised when the charged row came first and
            # silently accepted a contradictory lane when it came second -- dropping that lane's
            # number without a word.  Resume re-reads fragments in scheduler order, so an
            # aggregate that depends on arrival order is not a function of the evidence.
            uncharged_lanes.append(event)

        if event.status != "deferred":
            attempts_by_event.setdefault(event.event_id, set()).add(event.attempt_id)
            prior_assignment = assigned.get(event.event_id)
            if prior_assignment is not None and any(
                getattr(prior_assignment, name) != getattr(event, name)
                for name in _LOGICAL_FIELDS
            ):
                raise V2LedgerError(
                    f"conflicting assigned logical values for event {event.event_id!r}")
            assigned[event.event_id] = prior_assignment or event

    # Second pass: every uncharged batch lane is now compared against the CHARGED row for its
    # attempt, whichever order they arrived in.  A lane naming an attempt nothing ever charged is
    # itself a fault: the shared physical cost would be unrecorded.
    for event in uncharged_lanes:
        charged = physical_by_attempt.get(event.attempt_id)
        if charged is None:
            raise V2LedgerError(
                f"lane {event.event_id!r} declares it is not charged for attempt "
                f"{event.attempt_id!r}, but no row carries that attempt's physical cost"
            )
        if _physical_signature(charged) != _physical_signature(
            type(event)(**{**asdict(event), "physical_forwards_charged": True})
        ):
            raise V2LedgerError(
                f"lane {event.event_id!r} is not charged for attempt {event.attempt_id!r} but "
                "reports a different physical cost than the lane that is"
            )

    totals: dict[str, Any] = {}
    for name in _LOGICAL_FIELDS:
        totals[name] = sum(getattr(row, name) for row in assigned.values())
    for name in _PHYSICAL_FIELDS:
        value = sum(getattr(row, name) for row in physical_by_attempt.values())
        totals[name] = float(value) if isinstance(value, float) else value
    totals["n_logical_events"] = len(assigned)
    totals["n_physical_attempts"] = len(physical_by_attempt)
    totals["n_retries"] = sum(
        max(0, len(attempt_ids) - 1) for attempt_ids in attempts_by_event.values()
    )
    totals["n_attempts_unknown_physical"] = len(unknown_attempts)
    #: False whenever ANY attempt burned an unmeasured amount.  Every physical total above is then
    #: a lower bound, and the cap check must refuse to certify a budget from it.
    totals["physical_cost_complete"] = not unknown_attempts

    by_phase: dict[str, dict] = {}
    for row in assigned.values():
        bucket = by_phase.setdefault(row.phase, {name: 0 for name in _LOGICAL_FIELDS})
        for name in _LOGICAL_FIELDS:
            bucket[name] += getattr(row, name)
    totals["by_phase"] = by_phase
    return totals


@dataclass(frozen=True)
class CapVerdict:
    """Whether a run is inside its declared caps, and which ones could not be checked."""

    within: bool
    breached: tuple[str, ...]
    unverifiable: tuple[str, ...]
    detail: str


#: Cap -> the aggregate field it bounds, and whether that field is an assigned or a measured
#: quantity.  Only the measured ones become unverifiable when a backend fails to report.
_CAP_FIELDS = (
    ("max_logical_dfe", "logical_dfe", "assigned"),
    ("max_head_calls", "head_calls", "assigned"),
    ("max_definitive_refolds", "structure_attempts", "assigned"),
    ("max_gpu_seconds", "gpu_seconds", "measured"),
    ("max_walltime_s", "walltime_s", "measured"),
    ("max_retries", "n_retries", "assigned"),
)


def check_caps(totals: Mapping[str, Any], caps: Any) -> CapVerdict:
    """Compare aggregate totals against the config's hard caps.

    A cap over a MEASURED quantity becomes ``unverifiable`` -- not "within" -- as soon as any
    attempt reported ``unknown_after_start``.  The observed totals are then a lower bound, so
    declaring the run inside budget would assert a measurement nobody made.  Caps over ASSIGNED
    quantities stay verifiable: the method decided them, so they are known whether or not the
    machine reported back.
    """
    breached: list[str] = []
    unverifiable: list[str] = []
    complete = bool(totals.get("physical_cost_complete", True))
    for cap_name, field_name, kind in _CAP_FIELDS:
        limit = getattr(caps, cap_name, None)
        if limit is None:
            raise V2LedgerError(f"caps object has no {cap_name}")
        observed = totals.get(field_name, 0)
        if kind == "measured" and not complete:
            unverifiable.append(cap_name)
            continue
        if float(observed) > float(limit):
            breached.append(cap_name)
    detail_parts = []
    if breached:
        detail_parts.append("breached: " + ", ".join(
            f"{name}={totals.get(dict((c, f) for c, f, _ in _CAP_FIELDS)[name], 0)} > "
            f"{getattr(caps, name)}" for name in breached))
    if unverifiable:
        detail_parts.append(
            f"unverifiable ({totals.get('n_attempts_unknown_physical', 0)} attempt(s) reported "
            f"{UNKNOWN_AFTER_START}): " + ", ".join(unverifiable))
    return CapVerdict(
        within=not breached and not unverifiable,
        breached=tuple(breached), unverifiable=tuple(unverifiable),
        detail="; ".join(detail_parts) or "within every declared cap",
    )


def assert_within_caps(totals: Mapping[str, Any], caps: Any) -> None:
    """Hard caps: a breach or an unverifiable budget stops the run."""
    verdict = check_caps(totals, caps)
    if not verdict.within:
        raise CapsExceeded(verdict.detail)
    return None


# --------------------------------------------------------------------------------------------
# the append-only attempt journal
# --------------------------------------------------------------------------------------------


def _append_line(path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON line and flush it to the OS before returning.

    The journal's entire value is that the ``requested`` line survives a process that dies inside
    the oracle.  A buffered write that never reached the file would leave exactly the run whose
    cost is unknown looking like a run that never started.
    """
    with open(path, "a") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class AttemptJournal:
    """Append-only journal of oracle requests and their outcomes (PLAN §5.4).

    The ordering is the contract: a request is written BEFORE execution and its outcome AFTER.  A
    journal written only on success records nothing about the runs that failed, which are precisely
    the runs whose cost is in question.

    Deliberately not a dataclass: it owns a file and accumulates state, and making it look like a
    value would invite copies that disagree with the file on disk.
    """

    def __init__(self, path: Any) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._open: dict[str, dict] = {}
        self._closed: set[str] = set()
        if self.path.exists():
            self._replay()

    def _replay(self) -> None:
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            attempt_id = row["attempt_id"]
            if row["record"] == "requested":
                self._open[attempt_id] = row
            else:
                self._open.pop(attempt_id, None)
                self._closed.add(attempt_id)

    def open_attempts(self) -> tuple[str, ...]:
        """Attempts that were requested and never closed -- i.e. work of unknown cost."""
        return tuple(sorted(self._open))

    def open_request(
        self, *, event_id: str, protein_id: str, arm: str, phase: str,
        request_kind: str, request_digest: str,
        logical_dfe: int = 0, head_calls: int = 0, structure_attempts: int = 0,
    ) -> str:
        if phase not in LEDGER_PHASES:
            raise V2LedgerError(f"phase must be one of {sorted(LEDGER_PHASES)}, got {phase!r}")
        attempt_id = "att:" + canonical_digest({
            "event_id": event_id, "protein_id": protein_id, "arm": arm, "phase": phase,
            "request_kind": request_kind, "request_digest": request_digest,
            # Two identical requests under one event are two distinct attempts (a retry is exactly
            # that), so the ordinal is part of the identity.
            "ordinal": len(self._open) + len(self._closed),
        })[:16]
        row = {
            "record": "requested", "attempt_id": attempt_id, "event_id": event_id,
            "protein_id": protein_id, "arm": arm, "phase": phase,
            "request_kind": request_kind, "request_digest": request_digest,
            "logical_dfe": int(logical_dfe), "head_calls": int(head_calls),
            "structure_attempts": int(structure_attempts),
        }
        _append_line(self.path, row)
        self._open[attempt_id] = row
        return attempt_id

    def _require_open(self, attempt_id: str) -> dict:
        if attempt_id in self._closed:
            raise V2LedgerError(f"attempt {attempt_id!r} is already closed")
        row = self._open.get(attempt_id)
        if row is None:
            raise V2LedgerError(
                f"attempt {attempt_id!r} was never opened; an outcome without a request means the "
                "journal cannot say what was started"
            )
        return row

    def close_observed(
        self, attempt_id: str, *, status: str, physical_forwards: int, gpu_seconds: float,
        walltime_s: float, head_calls: int = 0, batch_size: int = 1,
        structure_cache_hits: int = 0, head_cache_hits: int = 0, outcome: str | None = None,
    ) -> None:
        self._require_open(attempt_id)
        _append_line(self.path, {
            "record": "observed", "attempt_id": attempt_id, "status": status,
            "physical_cost_status": OBSERVED,
            "physical_forwards": int(physical_forwards), "batch_size": int(batch_size),
            "gpu_seconds": float(gpu_seconds), "walltime_s": float(walltime_s),
            "head_calls": int(head_calls), "structure_cache_hits": int(structure_cache_hits),
            "head_cache_hits": int(head_cache_hits), "outcome": outcome,
        })
        self._open.pop(attempt_id, None)
        self._closed.add(attempt_id)

    def close_unknown(self, attempt_id: str, *, status: str, reason: str) -> None:
        """Record that the attempt burned an amount the backend could not report (PLAN §5.4)."""
        self._require_open(attempt_id)
        _append_line(self.path, {
            "record": "abandoned", "attempt_id": attempt_id, "status": status,
            "physical_cost_status": UNKNOWN_AFTER_START, "reason": str(reason),
        })
        self._open.pop(attempt_id, None)
        self._closed.add(attempt_id)


@dataclass
class AttemptReceipt:
    """What one journaled attempt reports back.

    Mutable and deliberately empty on entry: the body must SAY what it burned.  A receipt that
    defaulted to zero would make forgotten instrumentation indistinguishable from a call that was
    genuinely free, which is the one reading PLAN §5.4 forbids.
    """

    attempt_id: str
    _observed: dict | None = None

    def observe(
        self, *, physical_forwards: int, head_calls: int = 0, batch_size: int = 1,
        structure_cache_hits: int = 0, head_cache_hits: int = 0, outcome: str | None = None,
    ) -> None:
        """Report the measured physical cost of this attempt.

        ``gpu_seconds`` and ``walltime_s`` are NOT accepted here: the meter takes them from its own
        instruments, so the two fields nothing else can cross-check cannot become free parameters.
        """
        if self._observed is not None:
            raise V2LedgerError(
                f"attempt {self.attempt_id!r} reported its cost twice; one physical execution has "
                "one cost"
            )
        self._observed = {
            "physical_forwards": int(physical_forwards), "head_calls": int(head_calls),
            "batch_size": int(batch_size), "structure_cache_hits": int(structure_cache_hits),
            "head_cache_hits": int(head_cache_hits), "outcome": outcome,
        }


class CostMeter:
    """Binds an :class:`AttemptJournal` to one shard's identity and its GPU instrument.

    This is the piece that was missing.  The journal existed and was tested, but nothing called it:
    every oracle request in a cycle ran unjournaled, so a process that died inside a refold left no
    record that anything had been started -- and PLAN §5.4's ``unknown_after_start``, the whole
    point of journaling requests BEFORE execution, could never be produced by a real run.

    ``gpu_clock`` is required rather than defaulted.  ``check_caps`` treats GPU-seconds as a
    MEASURED quantity, so a fabricated ``0.0`` would let a GPU cap read as satisfied on a number
    nobody took.  A CPU run passing ``lambda: 0.0`` is making a true measurement -- the difference
    is that it had to state which instrument made it.
    """

    def __init__(self, *, journal: AttemptJournal, protein_id: str, arm: str,
                 gpu_clock: Any) -> None:
        if not isinstance(journal, AttemptJournal):
            raise V2LedgerError("journal must be an AttemptJournal")
        for name, value in (("protein_id", protein_id), ("arm", arm)):
            if not isinstance(value, str) or not value.strip():
                raise V2LedgerError(f"{name} must be a non-empty str, got {value!r}")
        if not callable(gpu_clock):
            raise V2LedgerError(
                "gpu_clock must be a callable returning cumulative GPU seconds; a run that cannot "
                "name its GPU instrument cannot certify a GPU cap"
            )
        self.journal = journal
        self.protein_id = str(protein_id)
        self.arm = str(arm)
        self.gpu_clock = gpu_clock

    @contextmanager
    def attempt(
        self, *, event_id: str, phase: str, request_kind: str, request_digest: str,
        logical_dfe: int = 0, head_calls: int = 0, structure_attempts: int = 0,
    ):
        """Journal one oracle request, run it, and journal what it cost.

        Three exits, all recorded: the body reports and the attempt closes OBSERVED; the body
        raises and it closes ``unknown_after_start`` while the exception still propagates; or the
        body returns without reporting, which is a wiring fault and also closes unknown -- loudly,
        because a silent free close is how an unmetered stage stays unmetered.
        """
        attempt_id = self.journal.open_request(
            event_id=event_id, protein_id=self.protein_id, arm=self.arm, phase=phase,
            request_kind=request_kind, request_digest=request_digest,
            logical_dfe=logical_dfe, head_calls=head_calls,
            structure_attempts=structure_attempts,
        )
        receipt = AttemptReceipt(attempt_id=attempt_id)
        started_wall = time.monotonic()
        started_gpu = float(self.gpu_clock())
        try:
            yield receipt
        except BaseException as exc:                        # noqa: BLE001 - re-raised below
            self.journal.close_unknown(
                attempt_id, status="failed", reason=f"{type(exc).__name__}: {exc}")
            raise
        if receipt._observed is None:
            self.journal.close_unknown(
                attempt_id, status="failed",
                reason="the attempt completed without reporting a physical cost")
            raise V2LedgerError(
                f"attempt {attempt_id!r} ({phase}) finished without calling receipt.observe(); an "
                "unreported cost is journaled as unknown rather than as free, but a stage that "
                "never reports is a wiring fault and is raised here"
            )
        self.journal.close_observed(
            attempt_id, status="ok",
            gpu_seconds=max(0.0, float(self.gpu_clock()) - started_gpu),
            walltime_s=max(0.0, time.monotonic() - started_wall),
            **receipt._observed,
        )


def events_from_journal(path: Any) -> tuple[V2LedgerEvent, ...]:
    """Reconstruct ledger events from a journal, including the attempts that never closed.

    An attempt with a ``requested`` line and no outcome is the crash case: the process died inside
    the oracle, so it burned budget nobody measured.  It becomes a FAILED event with
    ``unknown_after_start`` -- never a free one, and never silently dropped.
    """
    requests: dict[str, dict] = {}
    outcomes: dict[str, dict] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["record"] == "requested":
            requests[row["attempt_id"]] = row
        else:
            outcomes[row["attempt_id"]] = row

    events: list[V2LedgerEvent] = []
    for attempt_id, request in requests.items():
        outcome = outcomes.get(attempt_id)
        base = dict(
            event_id=request["event_id"], attempt_id=attempt_id,
            protein_id=request["protein_id"], arm=request["arm"], phase=request["phase"],
            logical_dfe=int(request.get("logical_dfe", 0)),
            structure_attempts=int(request.get("structure_attempts", 0)),
        )
        if outcome is None or outcome.get("physical_cost_status") == UNKNOWN_AFTER_START:
            events.append(V2LedgerEvent(
                **base,
                head_calls=int(request.get("head_calls", 0)),
                status=(outcome or {}).get("status", "failed"),
                physical_cost_status=UNKNOWN_AFTER_START,
                outcome=(outcome or {}).get("reason", "process ended before the outcome was "
                                                      "journaled"),
            ))
            continue
        events.append(V2LedgerEvent(
            **base,
            head_calls=int(outcome.get("head_calls", request.get("head_calls", 0))),
            status=outcome["status"],
            physical_forwards=int(outcome.get("physical_forwards", 0)),
            batch_size=int(outcome.get("batch_size", 1)),
            structure_cache_hits=int(outcome.get("structure_cache_hits", 0)),
            head_cache_hits=int(outcome.get("head_cache_hits", 0)),
            gpu_seconds=float(outcome.get("gpu_seconds", 0.0)),
            walltime_s=float(outcome.get("walltime_s", 0.0)),
            outcome=outcome.get("outcome"),
        ))
    return tuple(events)
