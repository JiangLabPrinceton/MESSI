"""V2F7: content-bound resume and fragment aggregation (PLAN §5.4).

A resume is a claim: "this work was already done, under exactly these scientific conditions, so I
may skip it."  Everything here exists to make that claim checkable.

**Exact run-signature equality.**  A fragment is accepted only when the config digest, the split,
the arm, the protein, the input content and the code revision all match the run asking to resume.
Anything else is a fragment from a different experiment, and reusing it would silently mix two runs'
results into one table.

**Content-bound, not merely condition-bound.**  ``run_sig`` digests the run's CONDITIONS and says
nothing about the result, so a fragment is additionally bound to its own content: ``result_sig``
digests the status and the payload together.  Without it, an endpoint table can be edited to say
anything at all and the fragment still validates, and a ``failed`` flipped to ``ok`` is the cheapest
possible way to manufacture a cohort success rate.  The format version is validated for the same
reason: a fragment written by a different writer may lay its payload out differently, and reading it
under this version's assumptions is how a table silently changes meaning between runs.

**Admissible and successful are different questions.**  A ``failed`` fragment is valid evidence --
of a failure.  A driver that skipped every ``accepted`` fragment would never retry a failed shard,
so the verdict reports the fragment's own outcome separately from its admissibility.

**Every fragment is validated, and every rejection is REPORTED.**  The V1 aggregator skips a stale
checkpoint with a bare ``continue``; V2 must not, because a silently skipped fragment is precisely
how a partial cohort masquerades as a complete one.  A rejected fragment produces a typed verdict
that reaches the driver's exit code.

**Order independence.**  Shards finish in whatever order the scheduler gives them.  The aggregate
must be a function of the SET of fragments, never of their arrival sequence, or a re-run would
produce a different artifact from the same evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion_v2.errors import V2Error  # noqa: E402

__all__ = [
    "V2ResumeError",
    "RunSignature",
    "FragmentVerdict",
    "AggregateReport",
    "FRAGMENT_SCHEMA",
    "FRAGMENT_STATUSES",
    "FRAGMENT_RESULT_STATUSES",
    "FRAGMENT_SUCCESS_STATUSES",
    "write_fragment",
    "read_fragment",
    "validate_fragment",
    "scan_fragments",
    "aggregate_fragments",
]


class V2ResumeError(V2Error):
    """A resume contract violation the caller got wrong -- not a rejected fragment."""


#: The fragment file format this reader understands.  Stamped by :func:`write_fragment` AND
#: validated by :func:`validate_fragment`: a version that is written but never read is not a
#: contract, it is a comment.
FRAGMENT_SCHEMA = "v2frag-1"

#: Closed vocabulary for a fragment's OWN outcome -- what the shard reported, not whether the
#: fragment may be reused.  Exactly the two outcomes ``rf_fusion_v2_cohort.run_v2_shard`` and the
#: driver's typed-error path can produce; nothing else is writable.  ``running`` is deliberately
#: absent: no fragment is written before its shard finishes, and an in-progress status would be
#: accepted by a resume and skip that protein forever without any process having completed it.
FRAGMENT_RESULT_STATUSES = frozenset({"ok", "failed"})

#: Which of those outcomes counts as work that need not be re-paid.  A ``failed`` fragment is valid
#: evidence and must be REPORTED, but it is not a reason to skip the protein on a retry.
FRAGMENT_SUCCESS_STATUSES = frozenset({"ok"})

#: Closed vocabulary.  Every reason a fragment can fail to be reusable has its own name, because
#: "skipped" tells an operator nothing about whether to re-run, re-configure, or investigate.
FRAGMENT_STATUSES = frozenset({
    "accepted",
    "stale_config",       # the scientific config changed since this fragment was written
    "stale_input",        # the input content changed; the fragment describes different data
    "foreign_run",        # a different campaign/split/arm/protein entirely
    "stale_code",         # the same config under a different code revision
    # unreadable, missing its own signature, or unable to prove what it contains: an unknown
    # ``fragment_schema``, an out-of-vocabulary result status, a payload that is not table-shaped,
    # or content that disagrees with its stored digest.  These share ONE status deliberately --
    # they name different causes but exactly one operator action ("this fragment cannot be
    # reused; re-run the shard"), and a status whose action duplicates another's would make the
    # vocabulary longer without making any decision different.  The cause is in ``detail``.
    "corrupt",
    "duplicate_conflict",  # two fragments claim the same id with different content
})


@dataclass(frozen=True)
class RunSignature:
    """Everything that must be identical for previously-paid work to be reusable.

    ``code_revision`` is included deliberately.  Two runs of the same config under different code
    are not the same experiment, and a resume that ignored it would let a bug fix be applied to half
    a cohort while the other half kept results from before it.
    """

    config_digest: str
    campaign_id: str
    split_role: str
    arm_role: str
    protein_id: str
    input_signature: str
    code_revision: str
    production_depth_authorized: bool = False
    exploratory_depth_override: bool = False
    #: The Dual overlay's content digest folded with the EXECUTING arm, or "" for a legacy run.
    #: Self-omitting: an empty value never reaches the canonical payload, so a run with no Dual
    #: overlay produces the byte-identical signature it produced before this field existed. The
    #: alternative -- emitting it unconditionally as null -- would move the signature of every
    #: fragment ever written, and nothing downstream would notice until a resume silently refused.
    dual_signature: str = ""

    def __post_init__(self) -> None:
        for name in (
            "config_digest", "campaign_id", "split_role", "arm_role", "protein_id",
            "input_signature", "code_revision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise V2ResumeError(f"{name} must be a non-empty str, got {value!r}")
        for name in ("production_depth_authorized", "exploratory_depth_override"):
            if not isinstance(getattr(self, name), bool):
                raise V2ResumeError(f"{name} must be an explicit bool")
        if self.production_depth_authorized and self.exploratory_depth_override:
            raise V2ResumeError(
                "production_depth_authorized and exploratory_depth_override are mutually "
                "exclusive execution identities"
            )
        # Checked separately from the loop above because "" is legal here and nowhere else.
        if not isinstance(self.dual_signature, str):
            raise V2ResumeError(
                f"dual_signature must be a str (\"\" for a run with no Dual overlay), got "
                f"{self.dual_signature!r}"
            )

    def canonical_payload(self) -> dict:
        payload = {
            "config_digest": self.config_digest,
            "campaign_id": self.campaign_id,
            "split_role": self.split_role,
            "arm_role": self.arm_role,
            "protein_id": self.protein_id,
            "input_signature": self.input_signature,
            "code_revision": self.code_revision,
            "production_depth_authorized": self.production_depth_authorized,
            "exploratory_depth_override": self.exploratory_depth_override,
        }
        if self.dual_signature:
            payload["dual_signature"] = self.dual_signature
        return payload

    @property
    def value(self) -> str:
        payload = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def mismatch_status(self, other: Mapping[str, Any]) -> str | None:
        """Name the FIRST field that differs, in the order an operator should act on.

        The order matters: a foreign run needs a different command, a stale config needs a decision
        about the science, and a stale code revision needs a rebuild.  Reporting only "mismatch"
        would leave all three looking the same.
        """
        for name in (
            "campaign_id", "split_role", "arm_role", "protein_id",
            "production_depth_authorized", "exploratory_depth_override",
        ):
            if other.get(name) != getattr(self, name):
                return "foreign_run"
        if other.get("config_digest") != self.config_digest:
            return "stale_config"
        if other.get("input_signature") != self.input_signature:
            return "stale_input"
        if other.get("code_revision") != self.code_revision:
            return "stale_code"
        # Defaulted to "" so a fragment written before this field existed compares equal to a
        # legacy run and is still reusable, while a Dual run's fragment can never be handed to a
        # legacy run or to another arm of the same bundle.
        if other.get("dual_signature", "") != self.dual_signature:
            return "foreign_run"
        return None


@dataclass(frozen=True)
class FragmentVerdict:
    """One fragment's admissibility, with the reason attached.

    ``result_status`` is the fragment's OWN outcome and is populated only when the fragment was
    admitted.  Admissibility and success are different questions: a ``failed`` fragment is
    admissible evidence of a failure, and a caller that read only :attr:`accepted` would skip that
    protein forever instead of retrying it.
    """

    path: str
    fragment_id: str
    status: str
    detail: str = ""
    result_status: str | None = None

    def __post_init__(self) -> None:
        if self.status not in FRAGMENT_STATUSES:
            raise V2ResumeError(
                f"status must be one of {sorted(FRAGMENT_STATUSES)}, got {self.status!r}")
        if self.result_status is not None and self.result_status not in FRAGMENT_RESULT_STATUSES:
            raise V2ResumeError(
                f"result status must be one of {sorted(FRAGMENT_RESULT_STATUSES)}, got "
                f"{self.result_status!r}")
        if self.result_status is not None and not self.accepted:
            # Nothing may be inferred about the outcome of work that was not admitted: reporting
            # one would let a rejected fragment still contribute to a success count.
            raise V2ResumeError(
                f"a {self.status!r} fragment may not also report a result status")

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    @property
    def records_success(self) -> bool:
        """True only for admitted work that actually succeeded -- the resume-skip condition."""
        return self.accepted and self.result_status in FRAGMENT_SUCCESS_STATUSES


@dataclass(frozen=True)
class AggregateReport:
    """What aggregation found, in enough detail for a driver to choose an exit code."""

    accepted: tuple[FragmentVerdict, ...] = ()
    rejected: tuple[FragmentVerdict, ...] = ()
    missing_proteins: tuple[str, ...] = ()
    tables: Mapping[str, list] = field(default_factory=dict)
    ledger_events: tuple = ()
    n_ok: int = 0

    @property
    def complete(self) -> bool:
        """True only when every requested protein contributed an accepted fragment."""
        return not self.missing_proteins and not self.rejected

    @property
    def any_success(self) -> bool:
        return self.n_ok > 0


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write via a same-directory temp file and ``os.replace``.

    A shard can be killed mid-write.  A partially written fragment that a later resume happily
    parsed would be worse than no fragment at all, because it would look like completed work.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True))
    os.replace(tmp, path)


def _result_digest(status: str, payload: Any) -> str:
    """Digest a fragment's RESULT -- its status and payload together.

    Both, not just the payload: a status flipped from ``failed`` to ``ok`` looks exactly as
    reusable as an edited table, and is a cheaper way to manufacture a cohort success rate.

    The digest is taken over the JSON ROUND-TRIPPED form so that a tuple written as a JSON array
    digests identically before and after persistence -- otherwise every fragment containing one
    would reject itself on the next read.
    """
    canonical = json.dumps({"status": status, "payload": payload}, sort_keys=True,
                           separators=(",", ":"))
    normalized = json.dumps(json.loads(canonical), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _table_shape_problem(payload: Any, *, table_names: Sequence[str] = ()) -> str | None:
    """Name the reason a payload could not be aggregated, or ``None`` if it can.

    :func:`aggregate_fragments` EXTENDS its tables from the payload's lists, so a malformed payload
    does not fail -- it publishes.  A positional row lands under whatever column happens to align,
    and a str under a table name is spread one character per row.

    Whether a key is a TABLE is knowable only from the declared table names, so that half of the
    check runs only where those names are known rather than being guessed from the value's type.
    """
    if not isinstance(payload, Mapping):
        return f"payload must be a mapping, got {type(payload).__name__}"
    for key, value in payload.items():
        if not isinstance(key, str):
            return f"payload key {key!r} is not a str"
        if key in tuple(table_names) and not isinstance(value, (list, tuple)):
            return (f"payload[{key!r}] is declared a table but holds {type(value).__name__}; "
                    "aggregation would spread it row by row")
        if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
            continue
        if isinstance(value, Sequence):
            for index, row in enumerate(value):
                if not isinstance(row, Mapping):
                    return (f"payload[{key!r}][{index}] is {type(row).__name__}, not a table row "
                            "(a mapping)")
    return None


def write_fragment(
    path: Any, *, signature: RunSignature, status: str, payload: Mapping[str, Any],
) -> None:
    """Persist one shard's result, content-bound to the run AND to the result it reports.

    Both closed vocabularies are enforced HERE, at the boundary where the fragment comes into
    existence, so a malformed or in-progress fragment can never reach a resume at all -- and so the
    error names the runner that produced it rather than a file nobody can explain later.
    """
    status = str(status)
    if status not in FRAGMENT_RESULT_STATUSES:
        raise V2ResumeError(
            f"result status must be one of {sorted(FRAGMENT_RESULT_STATUSES)}, got {status!r}; "
            "an in-progress or free-text status would be accepted by a resume and that protein "
            "would be skipped forever without any process having finished it"
        )
    problem = _table_shape_problem(payload)
    if problem is not None:
        raise V2ResumeError(f"refusing to persist a payload that cannot be aggregated: {problem}")
    body = dict(payload)
    _atomic_write_json(Path(path), {
        "fragment_schema": FRAGMENT_SCHEMA,
        "signature": signature.canonical_payload(),
        "run_sig": signature.value,
        "status": status,
        "payload": body,
        "result_sig": _result_digest(status, body),
    })


def read_fragment(path: Any) -> dict:
    return json.loads(Path(path).read_text())


def validate_fragment(
    path: Any, *, expected: RunSignature, table_names: Sequence[str] = (),
) -> FragmentVerdict:
    """Decide whether one fragment may be reused, and say why if not.

    ``table_names`` are the payload keys the caller will aggregate as tables.  Supplying them
    enables the stricter shape check; omitting them still validates everything else.
    """
    path = Path(path)
    fragment_id = path.stem
    try:
        data = read_fragment(path)
    except (json.JSONDecodeError, OSError) as exc:
        return FragmentVerdict(str(path), fragment_id, "corrupt", f"unreadable: {exc}")
    if not isinstance(data, dict) or "signature" not in data or "run_sig" not in data:
        return FragmentVerdict(
            str(path), fragment_id, "corrupt",
            "fragment carries no run signature; work that cannot name its own conditions may "
            "never be reused",
        )
    schema = data.get("fragment_schema")
    if schema != FRAGMENT_SCHEMA:
        return FragmentVerdict(
            str(path), fragment_id, "corrupt",
            f"fragment_schema {schema!r} is not {FRAGMENT_SCHEMA!r}; a fragment written by a "
            "different writer version may lay its payload out differently, and reading it under "
            "this version's assumptions is how a table silently changes meaning between runs",
        )
    signature = data["signature"]
    if not isinstance(signature, dict):
        return FragmentVerdict(str(path), fragment_id, "corrupt", "signature is not an object")
    mismatch = expected.mismatch_status(signature)
    if mismatch is not None:
        return FragmentVerdict(
            str(path), fragment_id, mismatch,
            f"fragment signature does not match the requested run ({mismatch})",
        )
    # The stored digest must also agree with the stored fields: an edited fragment that kept a
    # matching field set but a stale digest -- or the reverse -- is tampering, not staleness.
    if data["run_sig"] != RunSignature(**signature).value:
        return FragmentVerdict(
            str(path), fragment_id, "corrupt",
            "run_sig does not digest the signature fields it is stored with",
        )
    status = data.get("status")
    if status not in FRAGMENT_RESULT_STATUSES:
        return FragmentVerdict(
            str(path), fragment_id, "corrupt",
            f"result status {status!r} is not one of {sorted(FRAGMENT_RESULT_STATUSES)}; an "
            "unreadable outcome is not a reusable outcome",
        )
    payload = data.get("payload")
    problem = _table_shape_problem(payload, table_names=table_names)
    if problem is not None:
        return FragmentVerdict(str(path), fragment_id, "corrupt", problem)
    # ``run_sig`` binds the run's CONDITIONS and says nothing about the result.  Without this,
    # an endpoint table can be edited to say anything at all and the fragment still validates.
    stored = data.get("result_sig")
    if not isinstance(stored, str) or stored != _result_digest(status, payload):
        return FragmentVerdict(
            str(path), fragment_id, "corrupt",
            "the fragment's status and payload do not match the result digest stored with them",
        )
    return FragmentVerdict(str(path), fragment_id, "accepted", result_status=status)


def scan_fragments(
    fragment_dir: Any, *, expected_by_protein: Mapping[str, RunSignature],
    table_names: Sequence[str] = (),
) -> tuple[FragmentVerdict, ...]:
    """Validate every fragment on disk, in a canonical order.

    Sorting by path is what makes the result a function of the SET of fragments rather than of the
    order the filesystem happened to return them in.
    """
    directory = Path(fragment_dir)
    if not directory.exists():
        return ()
    verdicts: list[FragmentVerdict] = []
    by_id: dict[str, tuple[str, str]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            data = read_fragment(path)
            protein_id = data.get("signature", {}).get("protein_id")
        except (json.JSONDecodeError, OSError):
            verdicts.append(FragmentVerdict(str(path), path.stem, "corrupt", "unreadable"))
            continue
        expected = expected_by_protein.get(protein_id)
        if expected is None:
            verdicts.append(FragmentVerdict(
                str(path), path.stem, "foreign_run",
                f"fragment names protein {protein_id!r}, which is not in the requested cohort",
            ))
            continue
        verdict = validate_fragment(path, expected=expected, table_names=table_names)
        if verdict.accepted:
            content = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
            previous = by_id.get(protein_id)
            if previous is not None and previous[1] != content:
                # Two accepted fragments for one protein that disagree.  Keeping either would be a
                # coin flip about which run's result the cohort reports.
                verdict = FragmentVerdict(
                    str(path), path.stem, "duplicate_conflict",
                    f"protein {protein_id!r} already has an accepted fragment at {previous[0]} "
                    "with different content",
                )
            elif previous is not None:
                # A byte-identical re-emission (a shard re-ran and wrote the same result).
                # Idempotent: keep one, and do not report a problem that is not one.
                continue
            else:
                by_id[protein_id] = (str(path), content)
        verdicts.append(verdict)
    return tuple(verdicts)


def aggregate_fragments(
    fragment_dir: Any, *, requested_cohort: Sequence[str],
    expected_by_protein: Mapping[str, RunSignature],
    table_names: Sequence[str] = (),
    ok_statuses: frozenset[str] = FRAGMENT_SUCCESS_STATUSES,
) -> AggregateReport:
    """Merge every ACCEPTED fragment into one artifact set and report what was left out.

    Missing proteins are reported rather than dropped.  A cohort that quietly aggregated only the
    shards that happened to finish would publish a partial result under a whole cohort's name --
    and every per-cohort rate computed from it would be conditioned on success.
    """
    requested = list(requested_cohort)
    if len(set(requested)) != len(requested):
        raise V2ResumeError("duplicate protein id in the requested cohort")

    verdicts = scan_fragments(fragment_dir, expected_by_protein=expected_by_protein,
                              table_names=table_names)
    accepted = tuple(v for v in verdicts if v.accepted)
    rejected = tuple(v for v in verdicts if not v.accepted)

    tables: dict[str, list] = {name: [] for name in table_names}
    ledger_events: list = []
    contributed: set[str] = set()
    n_ok = 0
    for verdict in accepted:
        data = read_fragment(verdict.path)
        contributed.add(data["signature"]["protein_id"])
        # The outcome comes from the VERDICT, which read it under the closed vocabulary and the
        # result digest; re-reading the raw field here would count a status no validator approved.
        if verdict.result_status in ok_statuses:
            n_ok += 1
        payload = data.get("payload", {})
        for name in table_names:
            tables[name].extend(payload.get(name, []) or [])
        ledger_events.extend(payload.get("ledger_events", []) or [])

    # Canonical order, so a re-aggregation of the same fragment set is byte-stable regardless of
    # which shard finished first.
    for name, rows in tables.items():
        rows.sort(key=lambda row: json.dumps(row, sort_keys=True))

    return AggregateReport(
        accepted=accepted, rejected=rejected,
        missing_proteins=tuple(p for p in requested if p not in contributed),
        tables=tables, ledger_events=tuple(ledger_events), n_ok=n_ok,
    )
