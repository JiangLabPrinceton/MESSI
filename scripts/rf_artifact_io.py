"""Stable typed tables, atomic manifests, and cost-ledger serialization."""
from __future__ import annotations
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

@dataclass(frozen=True)
class TableSchema:
    columns: tuple[str, ...]
    sort_by: tuple[str, ...]


_UINT_COLS = frozenset({"seed"})


_INT_COLS = frozenset({
    "attempt_index", "paid_prefix_dfe", "step", "n_steps", "n_unresolved_editable", "length",
    "n_fixed", "n_editable", "n_resolved_editable", "n_unresolved", "window_start", "window_end",
    "replicate_index", "dfe", "convergence_multiplicity", "rank", "expected_k", "observed_k",
    "subset_rank", "design_idx", "entry_rank", "attempt_order", "slot_idx",
    "representative_design_idx",
})


_FLOAT_COLS = frozenset({
    "t", "target_maturity", "actual_maturity", "rho_edit", "known_identity_fraction",
    "coordinate_valid_fraction", "unresolved_head_relevant_mass", "anchor_preservation",
    "window_risk", "global_risk", "mean", "value", "source_value", "terminal_global_risk",
    "target_backbone_metric", "walltime_s",
})


_BOOL_COLS = frozenset({
    "cond_controller_enabled", "cond_h_maps_present",
    "valid", "selected", "structure_feasible", "requested", "gate_pass", "model_executed",
    "fresh_materialization_authorized", "head_independent_sampling",
    "selected_partial_member", "random_partial_member", "independent_full_member", "subset_key",
})


def _arrow_type(column: str):
    if column in _UINT_COLS:
        return pa.uint64()  # seeds span the full uint64 range and would overflow int64
    if column in _INT_COLS:
        return pa.int64()
    if column in _FLOAT_COLS:
        return pa.float64()
    if column in _BOOL_COLS:
        return pa.bool_()
    return pa.string()


def write_stable_parquet(
    path, rows: Sequence[Mapping], *, columns: Sequence[str], sort_by: Sequence[str],
    types: Mapping[str, str] | None = None,
) -> None:
    """Write ``rows`` as parquet with the frozen ``columns`` (present even when empty, in declared
    order) under an EXPLICIT per-column Arrow schema, sorted by ``sort_by`` with a stable sort. The
    schema is identical whether or not there are rows, so a downstream reader never sees an
    all-null empty table drift into typed columns. An undeclared column, or a ``sort_by`` key not
    in ``columns``, is a hard error rather than a silent drop."""
    columns = list(columns)
    col_set = set(columns)
    missing_sort = [k for k in sort_by if k not in col_set]
    if missing_sort:
        raise ValueError(f"sort_by keys not in columns: {missing_sort}")
    for row in rows:
        extra = set(row) - col_set
        if extra:
            raise ValueError(f"undeclared columns in row: {sorted(extra)}")

    norm = [{c: row.get(c) for c in columns} for row in rows]
    if sort_by and norm:
        norm.sort(key=lambda r: tuple(r[k] for k in sort_by))  # join/order keys are non-null
    # ``types`` lets a caller with its own column vocabulary declare Arrow types explicitly
    # instead of registering every name in this module's global sets.  Omitting it reproduces the
    # previous behaviour exactly, so V1 output is unchanged.
    declared = dict(types or {})
    unknown = sorted(set(declared) - col_set)
    if unknown:
        raise ValueError(f"declared types for undeclared columns: {unknown}")
    _BY_NAME = {"int": pa.int64(), "uint": pa.uint64(), "float": pa.float64(),
                "bool": pa.bool_(), "str": pa.string()}
    bad = sorted({v for v in declared.values() if v not in _BY_NAME})
    if bad:
        raise ValueError(f"unknown column type name(s) {bad}; expected {sorted(_BY_NAME)}")
    schema = pa.schema([
        (c, _BY_NAME[declared[c]] if c in declared else _arrow_type(c)) for c in columns
    ])
    table = pa.Table.from_arrays(
        [pa.array([r[c] for r in norm], type=schema.field(c).type) for c in columns],
        schema=schema,
    )
    pq.write_table(table, str(Path(path)))


def write_cost_ledger_jsonl(path, events: Sequence) -> None:
    """One JSON line per logical/physical ledger event, in emission order (append-safe §3.3).

    Accepts a dataclass OR an already-plain mapping.  The widening is for V2, whose ledger is
    routed through resume (PLAN_RF_REFINE_FUSION_V2 §5.4): a fragment round-trips through JSON, so
    by the time the events reach a writer they are mappings, and a dataclass-only writer made a
    PLAN-required evidence object unwritable for exactly the path it has to survive.  Purely
    additive -- V1 passes dataclasses and its output is unchanged -- and it keeps ONE ledger writer
    rather than a second implementation for V2.

    Anything that is neither is refused: a bare ``str`` would serialize to a JSON string and read
    back as a row nothing can aggregate.
    """
    with open(Path(path), "w") as handle:
        for event in events:
            if is_dataclass(event) and not isinstance(event, type):
                row = asdict(event)
            elif isinstance(event, Mapping):
                row = dict(event)
            else:
                raise TypeError(
                    f"a ledger event must be a dataclass or a mapping, got "
                    f"{type(event).__name__}"
                )
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_cost_ledger_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_manifest(path, manifest: Mapping) -> None:
    _write_json_atomic(Path(path), dict(manifest))


def _write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)
