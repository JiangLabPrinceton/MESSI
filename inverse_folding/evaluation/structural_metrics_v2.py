"""Standalone structural benchmark metrics with selective, lightweight evaluation.

The module deliberately imports no scientific or model stack at import time. Structure
parsing uses the standard library; NumPy is imported lazily only when a requested metric
needs rigid-body alignment. Benchmark callers request ``ALL_METRICS`` and receive a
self-contained summary plus residue-indexable side-chain rows. Refinement callers cache a
``ReferenceContext`` and request only the metrics used by their gate.
"""

from __future__ import annotations

import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


METRIC_GLOBAL_CA_RMSD = "global_ca_rmsd"
METRIC_PLDDT = "plddt"
METRIC_SIDECHAIN = "sidechain"
METRIC_RESIDUE_METRICS = "residue_metrics"
ALL_METRICS = frozenset(
    {
        METRIC_GLOBAL_CA_RMSD,
        METRIC_PLDDT,
        METRIC_SIDECHAIN,
        METRIC_RESIDUE_METRICS,
    }
)

_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})
_EXPECTED_SIDECHAIN_ATOMS: Mapping[str, frozenset[str]] = {
    "A": frozenset({"CB"}),
    "R": frozenset({"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"}),
    "N": frozenset({"CB", "CG", "OD1", "ND2"}),
    "D": frozenset({"CB", "CG", "OD1", "OD2"}),
    "C": frozenset({"CB", "SG"}),
    "Q": frozenset({"CB", "CG", "CD", "OE1", "NE2"}),
    "E": frozenset({"CB", "CG", "CD", "OE1", "OE2"}),
    "G": frozenset(),
    "H": frozenset({"CB", "CG", "ND1", "CD2", "CE1", "NE2"}),
    "I": frozenset({"CB", "CG1", "CG2", "CD1"}),
    "L": frozenset({"CB", "CG", "CD1", "CD2"}),
    "K": frozenset({"CB", "CG", "CD", "CE", "NZ"}),
    "M": frozenset({"CB", "CG", "SD", "CE"}),
    "F": frozenset({"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"}),
    "P": frozenset({"CB", "CG", "CD"}),
    "S": frozenset({"CB", "OG"}),
    "T": frozenset({"CB", "OG1", "CG2"}),
    "W": frozenset({"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"}),
    "Y": frozenset({"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"}),
    "V": frozenset({"CB", "CG1", "CG2"}),
}
_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

# AlphaFold/OpenFold atom-renaming ambiguities. Each residue's pairs are swapped
# together and the lower-error assignment is retained.
_SYMMETRIC_SWAPS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "ASP": (("OD1", "OD2"),),
    "GLU": (("OE1", "OE2"),),
    "PHE": (("CD1", "CD2"), ("CE1", "CE2")),
    "TYR": (("CD1", "CD2"), ("CE1", "CE2")),
}

_VALID_SIDECHAIN_MATCH_STATUSES = frozenset({"ok", "no_sidechain"})


class StructuralMetricError(ValueError):
    """Raised when structures cannot satisfy the metric correspondence contract."""


@dataclass(frozen=True)
class AtomRecord:
    name: str
    element: str
    coord: tuple[float, float, float]
    b_factor: float | None


@dataclass(frozen=True)
class ResidueRecord:
    residue_idx: int
    residue_name: str
    aa: str
    chain_id: str
    resseq: str
    icode: str
    atoms: tuple[AtomRecord, ...]

    def atom_map(self) -> dict[str, AtomRecord]:
        return {atom.name: atom for atom in self.atoms}


@dataclass(frozen=True)
class ReferenceContext:
    path: str
    sequence: str
    residues: tuple[ResidueRecord, ...]
    anchor_indices: tuple[int, ...]


@dataclass(frozen=True)
class ResidueSidechainMetric:
    residue_idx: int
    residue_idx_1based: int
    ref_chain_id: str
    ref_resseq: str
    ref_icode: str
    ref_aa: str
    design_aa: str
    is_anchor: bool
    match_status: str
    symmetry_swap_applied: bool
    reference_sidechain_atom_count: int
    predicted_sidechain_atom_count: int
    sidechain_atom_count: int
    sidechain_sq_error_sum: float | None
    sidechain_rmsd: float | None
    sidechain_max_atom_distance: float | None
    predicted_plddt: float | None
    reference_plddt: float | None


@dataclass(frozen=True)
class StructuralMetricResult:
    computed_metrics: frozenset[str]
    global_ca_rmsd: float | None
    predicted_global_plddt: float | None
    reference_global_plddt: float | None
    predicted_active_site_mean_plddt: float | None
    predicted_active_site_min_plddt: float | None
    reference_active_site_mean_plddt: float | None
    reference_active_site_min_plddt: float | None
    active_site_sidechain_rmsd: float | None
    max_anchor_sidechain_rmsd: float | None
    max_anchor_atom_distance: float | None
    active_site_sidechain_atom_count: int
    anchor_count: int
    matched_anchor_count: int
    active_site_complete: bool | None
    residue_metrics: tuple[ResidueSidechainMetric, ...]


@dataclass(frozen=True)
class SidechainAggregate:
    residue_count: int
    sidechain_atom_count: int
    pooled_sidechain_rmsd: float
    max_residue_sidechain_rmsd: float
    max_atom_distance: float


def prepare_reference_context(
    ref_structure: str | Path,
    *,
    ref_sequence: str,
    anchor_indices: Iterable[int] = (),
) -> ReferenceContext:
    """Parse and validate a reference once for reuse across many candidate structures."""

    sequence = str(ref_sequence).upper()
    residues = tuple(parse_structure(ref_structure))
    _validate_trace_sequence(residues, sequence, label="reference")
    anchors = tuple(sorted({int(index) for index in anchor_indices}))
    for index in anchors:
        if index < 0 or index >= len(sequence):
            raise StructuralMetricError(
                f"anchor index {index} out of range [0, {len(sequence)})"
            )
    return ReferenceContext(
        path=str(ref_structure),
        sequence=sequence,
        residues=residues,
        anchor_indices=anchors,
    )


def evaluate_prediction(
    context: ReferenceContext,
    pred_structure: str | Path,
    *,
    design_sequence: str,
    metrics: Iterable[str] | None = None,
) -> StructuralMetricResult:
    """Compute a selected metric subset against a cached reference context.

    Side-chain comparisons are made only between identical residue identities with
    complete, identical side-chain atom sets. Active-site mismatches raise immediately;
    non-anchor mismatches remain explicit residue-row statuses rather than being compared
    over a misleading common-atom subset.
    """

    requested = frozenset(ALL_METRICS if metrics is None else metrics)
    unknown = requested - ALL_METRICS
    if unknown:
        raise StructuralMetricError(f"unknown structural metrics: {sorted(unknown)}")

    sequence = str(design_sequence).upper()
    if len(sequence) != len(context.sequence):
        raise StructuralMetricError(
            f"design/reference sequence length mismatch: {len(sequence)} != {len(context.sequence)}"
        )
    for index in context.anchor_indices:
        if sequence[index] != context.sequence[index]:
            raise StructuralMetricError(
                f"anchor identity mismatch at index {index}: "
                f"design={sequence[index]} reference={context.sequence[index]}"
            )

    predicted = tuple(parse_structure(pred_structure))
    _validate_trace_sequence(predicted, sequence, label="predicted")

    pred_global_plddt = None
    ref_global_plddt = None
    pred_site_mean = None
    pred_site_min = None
    ref_site_mean = None
    ref_site_min = None
    if METRIC_PLDDT in requested:
        pred_ca_plddt = _ca_plddt_values(predicted)
        ref_ca_plddt = _ca_plddt_values(context.residues)
        pred_global_plddt = _mean_or_none(pred_ca_plddt)
        ref_global_plddt = _mean_or_none(ref_ca_plddt)
        if context.anchor_indices:
            pred_site = [pred_ca_plddt[index] for index in context.anchor_indices]
            ref_site = [ref_ca_plddt[index] for index in context.anchor_indices]
            pred_site_mean, pred_site_min = _mean_or_none(pred_site), _min_or_none(pred_site)
            ref_site_mean, ref_site_min = _mean_or_none(ref_site), _min_or_none(ref_site)

    needs_alignment = bool(
        requested
        & {METRIC_GLOBAL_CA_RMSD, METRIC_SIDECHAIN, METRIC_RESIDUE_METRICS}
    )
    transform = None
    global_ca_rmsd = None
    if needs_alignment:
        transform, ca_distances = _global_ca_alignment(context.residues, predicted)
        if METRIC_GLOBAL_CA_RMSD in requested:
            global_ca_rmsd = math.sqrt(
                math.fsum(distance * distance for distance in ca_distances)
                / len(ca_distances)
            )

    residue_rows: list[ResidueSidechainMetric] = []
    anchor_rows: list[ResidueSidechainMetric] = []
    if requested & {METRIC_SIDECHAIN, METRIC_RESIDUE_METRICS}:
        indices: Iterable[int]
        if METRIC_RESIDUE_METRICS in requested:
            indices = range(len(context.residues))
        else:
            indices = context.anchor_indices
        anchor_set = frozenset(context.anchor_indices)
        for index in indices:
            metric = _residue_sidechain_metric(
                reference=context.residues[index],
                predicted=predicted[index],
                design_aa=sequence[index],
                is_anchor=index in anchor_set,
                transform=transform,
            )
            if METRIC_RESIDUE_METRICS in requested:
                residue_rows.append(metric)
            if metric.is_anchor:
                anchor_rows.append(metric)

    active_rmsd = None
    max_anchor_rmsd = None
    max_anchor_atom = None
    active_atom_count = 0
    matched_anchor_count = 0
    active_complete: bool | None = None
    if METRIC_SIDECHAIN in requested and context.anchor_indices:
        matched = [row for row in anchor_rows if row.match_status == "ok"]
        matched_anchor_count = len(matched)
        active_complete = all(
            row.match_status in _VALID_SIDECHAIN_MATCH_STATUSES
            for row in anchor_rows
        )
        if active_complete:
            active_atom_count = sum(row.sidechain_atom_count for row in matched)
            total_sq = math.fsum(float(row.sidechain_sq_error_sum) for row in matched)
            if active_atom_count <= 0:
                active_complete = False
            else:
                active_rmsd = math.sqrt(total_sq / active_atom_count)
                max_anchor_rmsd = max(float(row.sidechain_rmsd) for row in matched)
                max_anchor_atom = max(
                    float(row.sidechain_max_atom_distance) for row in matched
                )

    return StructuralMetricResult(
        computed_metrics=requested,
        global_ca_rmsd=global_ca_rmsd,
        predicted_global_plddt=pred_global_plddt,
        reference_global_plddt=ref_global_plddt,
        predicted_active_site_mean_plddt=pred_site_mean,
        predicted_active_site_min_plddt=pred_site_min,
        reference_active_site_mean_plddt=ref_site_mean,
        reference_active_site_min_plddt=ref_site_min,
        active_site_sidechain_rmsd=active_rmsd,
        max_anchor_sidechain_rmsd=max_anchor_rmsd,
        max_anchor_atom_distance=max_anchor_atom,
        active_site_sidechain_atom_count=active_atom_count,
        anchor_count=len(context.anchor_indices),
        matched_anchor_count=matched_anchor_count,
        active_site_complete=active_complete,
        residue_metrics=tuple(residue_rows),
    )


def build_benchmark_summary_row(
    result: StructuralMetricResult,
    *,
    protein_id: str,
    design_id: str,
    design_idx: int,
    sequence: str,
    sc_tm: float,
    recovery: float,
    refold_backend: str,
) -> dict[str, Any]:
    """Build one complete v2 benchmark row without consulting legacy artifacts."""

    sc_tm_value = float(sc_tm)
    return {
        "protein_id": str(protein_id),
        "design_id": str(design_id),
        "design_idx": int(design_idx),
        "sequence": str(sequence),
        "scTM": sc_tm_value,
        "global_ca_RMSD": result.global_ca_rmsd,
        "pLDDT": result.predicted_global_plddt,
        "reference_pLDDT": result.reference_global_plddt,
        "predicted_active_site_mean_pLDDT": result.predicted_active_site_mean_plddt,
        "predicted_active_site_min_pLDDT": result.predicted_active_site_min_plddt,
        "reference_active_site_mean_pLDDT": result.reference_active_site_mean_plddt,
        "reference_active_site_min_pLDDT": result.reference_active_site_min_plddt,
        "active_site_sidechain_RMSD": result.active_site_sidechain_rmsd,
        "max_anchor_sidechain_RMSD": result.max_anchor_sidechain_rmsd,
        "max_anchor_atom_distance": result.max_anchor_atom_distance,
        "active_site_sidechain_atom_count": result.active_site_sidechain_atom_count,
        "anchor_count": result.anchor_count,
        "matched_anchor_count": result.matched_anchor_count,
        "active_site_complete": result.active_site_complete,
        "recovery": float(recovery),
        "foldability": bool(math.isfinite(sc_tm_value) and sc_tm_value > 0.5),
        "refold_backend": str(refold_backend),
    }


def build_benchmark_residue_rows(
    result: StructuralMetricResult,
    *,
    protein_id: str,
    design_id: str,
    design_idx: int,
    refold_backend: str,
) -> list[dict[str, Any]]:
    """Serialize compact residue rows that exactly support arbitrary index-set RMSD."""

    rows: list[dict[str, Any]] = []
    for metric in result.residue_metrics:
        rows.append(
            {
                "protein_id": str(protein_id),
                "design_id": str(design_id),
                "design_idx": int(design_idx),
                "residue_idx": metric.residue_idx,
                "residue_idx_1based": metric.residue_idx_1based,
                "ref_chain_id": metric.ref_chain_id,
                "ref_resseq": metric.ref_resseq,
                "ref_icode": metric.ref_icode,
                "ref_aa": metric.ref_aa,
                "design_aa": metric.design_aa,
                "is_anchor": metric.is_anchor,
                "match_status": metric.match_status,
                "symmetry_swap_applied": metric.symmetry_swap_applied,
                "reference_sidechain_atom_count": metric.reference_sidechain_atom_count,
                "predicted_sidechain_atom_count": metric.predicted_sidechain_atom_count,
                "sidechain_atom_count": metric.sidechain_atom_count,
                "sidechain_sq_error_sum": metric.sidechain_sq_error_sum,
                "sidechain_RMSD": metric.sidechain_rmsd,
                "sidechain_max_atom_distance": metric.sidechain_max_atom_distance,
                "predicted_pLDDT": metric.predicted_plddt,
                "reference_pLDDT": metric.reference_plddt,
                "refold_backend": str(refold_backend),
            }
        )
    return rows


def aggregate_residue_sidechain_metrics(
    residue_metrics: Iterable[ResidueSidechainMetric | Mapping[str, Any]],
    residue_indices: Iterable[int],
) -> SidechainAggregate:
    """Exactly aggregate any residue-index subset from compact v2 residue rows.

    The helper fails closed when an index is absent or its side-chain correspondence is
    not exact. ``sidechain_sq_error_sum`` and ``sidechain_atom_count`` preserve enough
    information to reproduce the atom-pooled RMSD without retaining a large atom table.
    """

    requested = tuple(sorted({int(index) for index in residue_indices}))
    if not requested:
        raise StructuralMetricError("residue_indices must be non-empty")
    by_index = {
        int(_residue_metric_value(metric, "residue_idx")): metric
        for metric in residue_metrics
    }
    missing = [index for index in requested if index not in by_index]
    if missing:
        raise StructuralMetricError(f"residue metrics missing requested indices: {missing}")
    selected = [by_index[index] for index in requested]
    invalid = [
        (
            int(_residue_metric_value(metric, "residue_idx")),
            str(_residue_metric_value(metric, "match_status")),
        )
        for metric in selected
        if _residue_metric_value(metric, "match_status")
        not in _VALID_SIDECHAIN_MATCH_STATUSES
    ]
    if invalid:
        raise StructuralMetricError(
            f"requested side-chain metrics are not exactly comparable: {invalid}"
        )
    matched = [
        metric
        for metric in selected
        if _residue_metric_value(metric, "match_status") == "ok"
    ]
    atom_count = sum(
        int(_residue_metric_value(metric, "sidechain_atom_count"))
        for metric in matched
    )
    if atom_count <= 0:
        raise StructuralMetricError("requested residues contain no comparable side-chain atoms")
    total_sq = math.fsum(
        float(_residue_metric_value(metric, "sidechain_sq_error_sum"))
        for metric in matched
    )
    return SidechainAggregate(
        residue_count=len(selected),
        sidechain_atom_count=atom_count,
        pooled_sidechain_rmsd=math.sqrt(total_sq / atom_count),
        max_residue_sidechain_rmsd=max(
            float(_residue_metric_value(metric, "sidechain_rmsd", "sidechain_RMSD"))
            for metric in matched
        ),
        max_atom_distance=max(
            float(_residue_metric_value(metric, "sidechain_max_atom_distance"))
            for metric in matched
        ),
    )


def _residue_metric_value(
    metric: ResidueSidechainMetric | Mapping[str, Any],
    attribute: str,
    serialized_column: str | None = None,
) -> Any:
    if isinstance(metric, Mapping):
        column = serialized_column or attribute
        if column not in metric:
            raise StructuralMetricError(f"serialized residue metric lacks column {column!r}")
        return metric[column]
    return getattr(metric, attribute)


def parse_structure(path: str | Path) -> list[ResidueRecord]:
    """Parse a single-chain PDB or mmCIF into trace-ordered all-heavy-atom residues."""

    path = Path(path)
    if path.suffix.lower() in {".cif", ".mmcif"}:
        residues = _parse_mmcif(path)
    else:
        residues = _parse_pdb(path)
    if not residues:
        raise StructuralMetricError(f"{path}: no residues with C-alpha atoms found")
    chains = {residue.chain_id for residue in residues}
    if len(chains) != 1:
        raise StructuralMetricError(f"{path}: expected one chain, found {sorted(chains)}")
    return residues


def _validate_trace_sequence(
    residues: tuple[ResidueRecord, ...] | list[ResidueRecord],
    sequence: str,
    *,
    label: str,
) -> None:
    if len(residues) != len(sequence):
        raise StructuralMetricError(
            f"{label} structure/sequence length mismatch: {len(residues)} != {len(sequence)}"
        )
    mismatches = [
        (index, residue.aa, sequence[index])
        for index, residue in enumerate(residues)
        if residue.aa != sequence[index]
    ]
    if mismatches:
        index, structure_aa, sequence_aa = mismatches[0]
        raise StructuralMetricError(
            f"{label} structure sequence mismatch at index {index}: "
            f"structure={structure_aa} sequence={sequence_aa}"
        )


def _global_ca_alignment(reference, predicted):
    import numpy as np

    ref_ca = np.asarray([_atom(residue, "CA").coord for residue in reference], dtype=float)
    pred_ca = np.asarray([_atom(residue, "CA").coord for residue in predicted], dtype=float)
    mobile_center = pred_ca.mean(axis=0)
    target_center = ref_ca.mean(axis=0)
    mobile_zero = pred_ca - mobile_center
    target_zero = ref_ca - target_center
    u_mat, _singular_values, vt_mat = np.linalg.svd(mobile_zero.T @ target_zero)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u_mat @ vt_mat)) or 1.0
    rotation = u_mat @ correction @ vt_mat
    aligned = mobile_zero @ rotation + target_center
    distances = np.linalg.norm(aligned - ref_ca, axis=1)
    return (rotation, mobile_center, target_center), tuple(float(x) for x in distances)


def _residue_sidechain_metric(
    *,
    reference: ResidueRecord,
    predicted: ResidueRecord,
    design_aa: str,
    is_anchor: bool,
    transform,
) -> ResidueSidechainMetric:
    ref_atoms = reference.atom_map()
    pred_atoms = predicted.atom_map()
    ref_sidechain = {name: atom for name, atom in ref_atoms.items() if name not in _BACKBONE_ATOMS}
    pred_sidechain = {name: atom for name, atom in pred_atoms.items() if name not in _BACKBONE_ATOMS}
    ref_plddt = ref_atoms["CA"].b_factor
    pred_plddt = pred_atoms["CA"].b_factor

    common = set(ref_sidechain) & set(pred_sidechain)
    base = {
        "residue_idx": reference.residue_idx,
        "residue_idx_1based": reference.residue_idx + 1,
        "ref_chain_id": reference.chain_id,
        "ref_resseq": reference.resseq,
        "ref_icode": reference.icode,
        "ref_aa": reference.aa,
        "design_aa": design_aa,
        "is_anchor": bool(is_anchor),
        "reference_sidechain_atom_count": len(ref_sidechain),
        "predicted_sidechain_atom_count": len(pred_sidechain),
        "sidechain_atom_count": len(common),
        "predicted_plddt": pred_plddt,
        "reference_plddt": ref_plddt,
    }
    empty = {
        "symmetry_swap_applied": False,
        "sidechain_sq_error_sum": None,
        "sidechain_rmsd": None,
        "sidechain_max_atom_distance": None,
    }
    if reference.aa != design_aa:
        return ResidueSidechainMetric(match_status="residue_identity_mismatch", **base, **empty)
    expected_atoms = _EXPECTED_SIDECHAIN_ATOMS.get(reference.aa)
    if expected_atoms is None:
        return ResidueSidechainMetric(match_status="unsupported_residue", **base, **empty)
    if not expected_atoms:
        return ResidueSidechainMetric(match_status="no_sidechain", **base, **empty)
    if set(ref_sidechain) != expected_atoms or set(pred_sidechain) != expected_atoms:
        return ResidueSidechainMetric(match_status="atom_set_mismatch", **base, **empty)

    identity = {name: name for name in ref_sidechain}
    assignments = [(False, identity)]
    pairs = _SYMMETRIC_SWAPS.get(reference.residue_name, ())
    if pairs and all(left in ref_sidechain and right in ref_sidechain for left, right in pairs):
        swapped = dict(identity)
        for left, right in pairs:
            swapped[left] = right
            swapped[right] = left
        assignments.append((True, swapped))

    best = None
    for swapped, assignment in assignments:
        distances = []
        for pred_name in sorted(pred_sidechain):
            aligned = _apply_transform(pred_sidechain[pred_name].coord, transform)
            target = ref_sidechain[assignment[pred_name]].coord
            distance = math.dist(aligned, target)
            distances.append(distance)
        sum_sq = math.fsum(distance * distance for distance in distances)
        candidate = (sum_sq, swapped, distances)
        if best is None or candidate[0] < best[0]:
            best = candidate

    sum_sq, swapped, distances = best
    count = len(distances)
    return ResidueSidechainMetric(
        match_status="ok",
        symmetry_swap_applied=bool(swapped),
        sidechain_sq_error_sum=float(sum_sq),
        sidechain_rmsd=math.sqrt(sum_sq / count),
        sidechain_max_atom_distance=max(distances),
        **base,
    )


def _apply_transform(coord, transform) -> tuple[float, float, float]:
    import numpy as np

    rotation, mobile_center, target_center = transform
    aligned = (np.asarray(coord, dtype=float) - mobile_center) @ rotation + target_center
    return (float(aligned[0]), float(aligned[1]), float(aligned[2]))


def _ca_plddt_values(residues) -> list[float | None]:
    return [_atom(residue, "CA").b_factor for residue in residues]


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return math.fsum(finite) / len(finite) if finite else None


def _min_or_none(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return min(finite) if finite else None


def _atom(residue: ResidueRecord, name: str) -> AtomRecord:
    try:
        return residue.atom_map()[name]
    except KeyError as exc:
        raise StructuralMetricError(
            f"residue {residue.residue_idx} {residue.residue_name} lacks atom {name}"
        ) from exc


def _parse_pdb(path: Path) -> list[ResidueRecord]:
    residue_order: list[tuple[str, str, str]] = []
    residue_data: dict[tuple[str, str, str], tuple[str, dict[str, AtomRecord]]] = {}
    in_first_model = True
    saw_model = False
    with open(path) as handle:
        for line in handle:
            record_type = line[:6].strip()
            if record_type == "MODEL":
                if saw_model:
                    in_first_model = False
                saw_model = True
                continue
            if record_type == "ENDMDL":
                if saw_model:
                    break
                continue
            if not in_first_model or record_type not in {"ATOM", "HETATM"} or len(line) < 54:
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            atom_name = line[12:16].strip()
            element = _pdb_element(line, atom_name)
            if element == "H":
                continue
            chain_id = line[21].strip() or "?"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain_id, resseq, icode)
            if key not in residue_data:
                residue_order.append(key)
                residue_data[key] = (line[17:20].strip().upper(), {})
            residue_name, atoms = residue_data[key]
            if residue_name == "MSE" and atom_name == "SE":
                atom_name = "SD"
            if atom_name in atoms:
                continue
            try:
                coord = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except ValueError:
                continue
            b_factor = _float_or_none(line[60:66] if len(line) >= 66 else None)
            atoms[atom_name] = AtomRecord(
                name=atom_name,
                element=element,
                coord=coord,
                b_factor=b_factor,
            )

    return _materialize_residues(residue_order, residue_data)


def _parse_mmcif(path: Path) -> list[ResidueRecord]:
    tokens = list(_iter_cif_tokens(path))
    idx = 0
    while idx < len(tokens):
        if tokens[idx] != "loop_":
            idx += 1
            continue
        idx += 1
        headers: list[str] = []
        while idx < len(tokens) and tokens[idx].startswith("_"):
            headers.append(tokens[idx])
            idx += 1
        if not headers:
            continue
        width = len(headers)
        if not any(header.startswith("_atom_site.") for header in headers):
            while idx < len(tokens) and tokens[idx] != "loop_" and not tokens[idx].startswith("_"):
                idx += width
            continue

        residue_order: list[tuple[str, str, str]] = []
        residue_data: dict[tuple[str, str, str], tuple[str, dict[str, AtomRecord]]] = {}
        while idx + width <= len(tokens):
            if tokens[idx] == "loop_" or tokens[idx].startswith("_"):
                break
            row = dict(zip(headers, tokens[idx : idx + width], strict=True))
            idx += width
            model_num = _clean_cif(row.get("_atom_site.pdbx_PDB_model_num", "1"))
            if model_num not in {"", "1"}:
                continue
            altloc = _clean_cif(row.get("_atom_site.label_alt_id", ""))
            if altloc not in {"", "A"}:
                continue
            atom_name = _clean_cif(
                row.get("_atom_site.label_atom_id")
                or row.get("_atom_site.auth_atom_id")
                or ""
            )
            element = _clean_cif(row.get("_atom_site.type_symbol", "")).upper()
            if not element:
                element = _infer_element(atom_name)
            if element == "H":
                continue
            residue_name = _clean_cif(
                row.get("_atom_site.label_comp_id")
                or row.get("_atom_site.auth_comp_id")
                or ""
            ).upper()
            if residue_name == "MSE" and atom_name == "SE":
                atom_name = "SD"
            chain_id = _clean_cif(
                row.get("_atom_site.auth_asym_id")
                or row.get("_atom_site.label_asym_id")
                or "?"
            ) or "?"
            resseq = _clean_cif(
                row.get("_atom_site.auth_seq_id")
                or row.get("_atom_site.label_seq_id")
                or ""
            )
            icode = _clean_cif(row.get("_atom_site.pdbx_PDB_ins_code", ""))
            key = (chain_id, resseq, icode)
            if key not in residue_data:
                residue_order.append(key)
                residue_data[key] = (residue_name, {})
            _stored_name, atoms = residue_data[key]
            if atom_name in atoms:
                continue
            try:
                coord = (
                    float(_clean_cif(row["_atom_site.Cartn_x"])),
                    float(_clean_cif(row["_atom_site.Cartn_y"])),
                    float(_clean_cif(row["_atom_site.Cartn_z"])),
                )
            except (KeyError, ValueError):
                continue
            atoms[atom_name] = AtomRecord(
                name=atom_name,
                element=element,
                coord=coord,
                b_factor=_float_or_none(row.get("_atom_site.B_iso_or_equiv")),
            )
        return _materialize_residues(residue_order, residue_data)
    return []


def _materialize_residues(residue_order, residue_data) -> list[ResidueRecord]:
    residues: list[ResidueRecord] = []
    for key in residue_order:
        residue_name, atoms = residue_data[key]
        if "CA" not in atoms:
            continue
        chain_id, resseq, icode = key
        residues.append(
            ResidueRecord(
                residue_idx=len(residues),
                residue_name=residue_name,
                aa=_THREE_TO_ONE.get(residue_name, "X"),
                chain_id=chain_id,
                resseq=resseq,
                icode=icode,
                atoms=tuple(atoms.values()),
            )
        )
    return residues


def _iter_cif_tokens(path: Path):
    in_text_block = False
    with open(path) as handle:
        for raw_line in handle:
            if raw_line.startswith(";"):
                in_text_block = not in_text_block
                continue
            if in_text_block:
                continue
            line = raw_line.strip()
            if not line or line == "#":
                continue
            yield from shlex.split(line, comments=True, posix=True)


def _clean_cif(value: Any) -> str:
    text = str(value).strip().strip("'\"")
    return "" if text in {".", "?"} else text


def _pdb_element(line: str, atom_name: str) -> str:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    return element or _infer_element(atom_name)


def _infer_element(atom_name: str) -> str:
    letters = "".join(char for char in atom_name if char.isalpha())
    return letters[:1].upper()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(_clean_cif(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
