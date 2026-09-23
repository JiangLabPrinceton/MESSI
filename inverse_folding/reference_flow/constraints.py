"""Active-site constraint manifest loader for uricase enzyme mode v0.

Parses a hard-anchor / monitored-shell constraint manifest keyed by
``protein_id`` (PLAN_URICASE_ENZYME_MODE Task U1). The loader is intentionally
pure: it validates structure, computes a content-stable hash, and validates
anchor identity against a resolved sequence. Amino-acid -> model token-id
conversion is **not** done here; it happens at the script/runtime boundary via
the DPLM alphabet (``task.alphabet.get_idx``), keeping this module free of model
dependencies and trivially testable.

Manifest shape (YAML, matching the existing ``controller_config`` style)::

    schema_version: uricase_active_site_v0
    description: Q00511 v0 hard-anchor constraint
    entries:
      - protein_id: Q00511
        sequence_md5: optional
        hard_anchors:
          - {index_0b: 10, expected_aa: K, label: Lys11, biological_role: ..., source: ...}
        monitored_shell:
          - {index_0b: 227, expected_aa: V, label: Val228, enforcement: monitored_posthoc, rationale: ...}
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Canonical 20 amino acids (DPLM alphabet subset); expected_aa must be one of these.
VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Code-surface version for the enzyme-mode hard-anchor constraint layer (v0).
CONSTRAINT_SURFACE_VERSION = "enzyme_mode_v0"


@dataclass(frozen=True)
class HardAnchor:
    """A residue whose identity is permanently fixed during generation."""

    index_0b: int
    expected_aa: str
    label: str = ""
    biological_role: str = ""
    source: str = ""


@dataclass(frozen=True)
class MonitoredShellResidue:
    """A residue tracked post-hoc but NOT hard-fixed in v0."""

    index_0b: int
    expected_aa: str
    label: str = ""
    enforcement: str = "monitored_posthoc"
    rationale: str = ""


@dataclass(frozen=True)
class ActiveSiteConstraint:
    """Per-protein active-site constraint (one manifest entry)."""

    protein_id: str
    hard_anchors: tuple[HardAnchor, ...]
    monitored_shell: tuple[MonitoredShellResidue, ...] = ()
    sequence_md5: str | None = None
    source_sequence: str | None = None
    unconstrained_control: bool = False

    @property
    def hard_anchor_indices(self) -> tuple[int, ...]:
        return tuple(a.index_0b for a in self.hard_anchors)

    def validate_against_sequence(self, sequence: str) -> None:
        """Fail-fast: assert every hard anchor is in bounds and matches ``expected_aa``.

        Raises ``ValueError`` on the first out-of-range index or AA mismatch. This
        is the three-way numbering safety net (doc/Uricases_Design.md §7.2): a
        single mislabeled index relative to the resolved parquet sequence aborts
        the run before any tokens are generated.
        """
        n = len(sequence)
        for anchor in self.hard_anchors:
            if not (0 <= anchor.index_0b < n):
                raise ValueError(
                    f"hard anchor index {anchor.index_0b} (label={anchor.label!r}) "
                    f"out of range [0, {n}) for protein {self.protein_id!r} "
                    f"(sequence length {n})"
                )
            actual = sequence[anchor.index_0b]
            if actual != anchor.expected_aa:
                raise ValueError(
                    f"hard anchor mismatch for protein {self.protein_id!r} at "
                    f"index_0b={anchor.index_0b} (label={anchor.label!r}): "
                    f"expected_aa={anchor.expected_aa!r} but resolved sequence has "
                    f"{actual!r}"
                )


@dataclass(frozen=True)
class ConstraintManifest:
    """A parsed constraint manifest: protein_id -> ActiveSiteConstraint + a stable hash."""

    schema_version: str
    description: str
    entries: Mapping[str, ActiveSiteConstraint]
    manifest_hash: str
    source_path: str | None = None

    @property
    def constrained_protein_ids(self) -> frozenset[str]:
        return frozenset(self.entries)

    def has_protein(self, protein_id: str) -> bool:
        return protein_id in self.entries

    def constraint_for_protein(self, protein_id: str) -> ActiveSiteConstraint:
        """Return the constraint for ``protein_id`` or fail-fast if absent."""
        try:
            return self.entries[protein_id]
        except KeyError as exc:
            raise KeyError(
                f"constraint manifest has no entry for protein_id={protein_id!r}; "
                f"known protein_ids={sorted(self.entries)}"
            ) from exc

    @property
    def num_hard_anchors_total(self) -> int:
        return sum(len(c.hard_anchors) for c in self.entries.values())


def _canonical_hash(raw: Any) -> str:
    """Content hash that is stable across key order / whitespace in the source file."""
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_hard_anchors(entry: dict, protein_id: str) -> tuple[HardAnchor, ...]:
    raw = entry.get("hard_anchors") or []
    anchors: list[HardAnchor] = []
    for item in raw:
        expected_aa = str(item["expected_aa"]).upper()
        if expected_aa not in VALID_AA:
            raise ValueError(
                f"protein {protein_id!r} hard anchor at index "
                f"{item.get('index_0b')!r} has invalid expected_aa={expected_aa!r}"
            )
        anchors.append(
            HardAnchor(
                index_0b=int(item["index_0b"]),
                expected_aa=expected_aa,
                label=str(item.get("label", "")),
                biological_role=str(item.get("biological_role", "")),
                source=str(item.get("source", "")),
            )
        )
    indices = [a.index_0b for a in anchors]
    if len(set(indices)) != len(indices):
        dupes = sorted({i for i in indices if indices.count(i) > 1})
        raise ValueError(
            f"protein {protein_id!r} has duplicate hard-anchor indices {dupes}"
        )
    return tuple(anchors)


def _parse_monitored_shell(entry: dict) -> tuple[MonitoredShellResidue, ...]:
    raw = entry.get("monitored_shell") or []
    return tuple(
        MonitoredShellResidue(
            index_0b=int(item["index_0b"]),
            expected_aa=str(item["expected_aa"]).upper(),
            label=str(item.get("label", "")),
            enforcement=str(item.get("enforcement", "monitored_posthoc")),
            rationale=str(item.get("rationale", "")),
        )
        for item in raw
    )


def load_constraint_manifest(path: str | Path) -> ConstraintManifest:
    """Load and validate a constraint manifest from YAML.

    Validation: unique ``protein_id``, valid amino acids, no duplicate hard-anchor
    indices, and empty hard-anchor lists allowed only when explicitly flagged
    ``unconstrained_control: true``. Raises ``ValueError`` on any violation.
    """
    path = Path(path)
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"constraint manifest {path} must be a mapping at top level")

    entries_raw = raw.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise ValueError(
            f"constraint manifest {path} must have a non-empty 'entries' list"
        )

    entries: dict[str, ActiveSiteConstraint] = {}
    for entry in entries_raw:
        protein_id = entry.get("protein_id")
        if not protein_id:
            raise ValueError("constraint manifest entry missing 'protein_id'")
        protein_id = str(protein_id)
        if protein_id in entries:
            raise ValueError(
                f"duplicate protein_id {protein_id!r} in constraint manifest {path}"
            )

        hard_anchors = _parse_hard_anchors(entry, protein_id)
        unconstrained = bool(entry.get("unconstrained_control", False))
        if not hard_anchors and not unconstrained:
            raise ValueError(
                f"protein {protein_id!r} has empty hard_anchors but is not marked "
                "'unconstrained_control: true'; empty anchor sets are allowed only "
                "for explicit unconstrained controls"
            )

        entries[protein_id] = ActiveSiteConstraint(
            protein_id=protein_id,
            hard_anchors=hard_anchors,
            monitored_shell=_parse_monitored_shell(entry),
            sequence_md5=(
                str(entry["sequence_md5"]) if entry.get("sequence_md5") else None
            ),
            source_sequence=(
                str(entry["source_sequence"]) if entry.get("source_sequence") else None
            ),
            unconstrained_control=unconstrained,
        )

    return ConstraintManifest(
        schema_version=str(raw.get("schema_version", "")),
        description=str(raw.get("description", "")),
        entries=entries,
        manifest_hash=_canonical_hash(raw),
        source_path=str(path),
    )


# ---------------------------------------------------------------------------
# Script / runtime boundary helpers (Task U3): AA -> token id conversion happens
# HERE (via an injected accessor), never inside the manifest parser.
# ---------------------------------------------------------------------------


def build_fixed_token_map(
    constraint: ActiveSiteConstraint,
    sequence: str,
    aa_to_token: Callable[[str], int],
) -> dict[int, int]:
    """Validate ``constraint`` against the resolved ``sequence``, then map every
    hard-anchor ``index_0b`` to a model token id via ``aa_to_token``.

    Fail-fast: ``validate_against_sequence`` raises before any token is emitted, so
    a mislabeled anchor aborts the run prior to generation.
    """
    constraint.validate_against_sequence(sequence)
    return {
        anchor.index_0b: int(aa_to_token(anchor.expected_aa))
        for anchor in constraint.hard_anchors
    }


def build_run_constraints(
    manifest: ConstraintManifest,
    sequences_by_protein: Mapping[str, str],
    aa_to_token: Callable[[str], int],
) -> dict[str, dict[int, int]]:
    """Build per-protein fixed-token maps for every constrained protein present in
    this run.

    Proteins in the manifest but absent from ``sequences_by_protein`` are skipped
    (the caller may log them); unconstrained controls / empty anchor sets are also
    skipped. Validation is fail-fast for proteins that ARE present.
    """
    run_constraints: dict[str, dict[int, int]] = {}
    for protein_id in sorted(manifest.constrained_protein_ids):
        if protein_id not in sequences_by_protein:
            continue
        constraint = manifest.constraint_for_protein(protein_id)
        if constraint.unconstrained_control or not constraint.hard_anchors:
            continue
        run_constraints[protein_id] = build_fixed_token_map(
            constraint, sequences_by_protein[protein_id], aa_to_token
        )
    return run_constraints


def constraint_manifest_provenance(
    manifest: ConstraintManifest,
    *,
    num_constrained_proteins: int,
    num_hard_anchors_total: int,
    constraints_applied_path: str | None = None,
) -> dict[str, Any]:
    """Flat run-manifest provenance block, emitted only when constraints are enabled."""
    block: dict[str, Any] = {
        "enzyme_mode_enabled": True,
        "constraint_manifest_path": manifest.source_path,
        "constraint_manifest_hash": manifest.manifest_hash,
        "constraint_schema_version": manifest.schema_version,
        "constraint_surface_version": CONSTRAINT_SURFACE_VERSION,
        "num_constrained_proteins": int(num_constrained_proteins),
        "num_hard_anchors_total": int(num_hard_anchors_total),
    }
    if constraints_applied_path is not None:
        block["constraints_applied_path"] = constraints_applied_path
    return block


# ---------------------------------------------------------------------------
# Hard-constraint telemetry + gate (Task U4). v0 invariant: every hard anchor in
# every generated design must equal its expected AA. A mismatch is an impl/manifest
# bug (the sampler guarantees preservation), not a weak metric — it fails the run.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstraintApplicationReport:
    """Per-(design, anchor) preservation rows plus a compact summary."""

    rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def build_constraint_application_report(
    generated_designs: Iterable[Mapping[str, Any]],
    manifest: ConstraintManifest,
    aa_to_token: Callable[[str], int],
) -> ConstraintApplicationReport:
    """Compare each constrained design's generated AA at every hard anchor against
    the expected AA. ``generated_designs`` items must carry ``protein_id``,
    ``design_idx`` and the decoded ``sequence``.
    """
    rows: list[dict[str, Any]] = []
    constrained_designs = 0
    for design in generated_designs:
        protein_id = str(design["protein_id"])
        if not manifest.has_protein(protein_id):
            continue
        constraint = manifest.constraint_for_protein(protein_id)
        if not constraint.hard_anchors:
            continue
        sequence = str(design["sequence"])
        design_idx = int(design["design_idx"])
        constrained_designs += 1
        for anchor in constraint.hard_anchors:
            in_range = 0 <= anchor.index_0b < len(sequence)
            generated_aa = sequence[anchor.index_0b] if in_range else ""
            preserved = bool(in_range and generated_aa == anchor.expected_aa)
            if generated_aa in VALID_AA:
                generated_token_id = int(aa_to_token(generated_aa))
            else:
                generated_token_id = -1
            rows.append(
                {
                    "protein_id": protein_id,
                    "design_idx": design_idx,
                    "anchor_index_0b": anchor.index_0b,
                    "expected_aa": anchor.expected_aa,
                    "expected_token_id": int(aa_to_token(anchor.expected_aa)),
                    "generated_aa": generated_aa,
                    "generated_token_id": generated_token_id,
                    "preserved": preserved,
                    "label": anchor.label,
                    "biological_role": anchor.biological_role,
                    "constraint_manifest_hash": manifest.manifest_hash,
                }
            )

    mismatches = [r for r in rows if not r["preserved"]]
    summary = {
        "num_designs": constrained_designs,
        "num_anchor_rows": len(rows),
        "num_anchor_mismatches": len(mismatches),
        "all_anchors_preserved": len(mismatches) == 0,
        "mismatch_examples": [
            {
                key: m[key]
                for key in (
                    "protein_id",
                    "design_idx",
                    "anchor_index_0b",
                    "expected_aa",
                    "generated_aa",
                )
            }
            for m in mismatches[:10]
        ],
    }
    return ConstraintApplicationReport(rows=tuple(rows), summary=summary)
