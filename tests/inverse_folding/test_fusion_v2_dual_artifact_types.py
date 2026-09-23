"""DUALF5: a Dual table must be able to hold the numbers it declares.

``write_v2_bundle`` writes every declared table even when EMPTY, which is the right contract -- an
absent table and an empty one are different claims.  It is also what makes an inferred column type
fatal: with no rows to infer from, a numeric column is written as string, and the first shard that
actually measures something cannot append to it.

So the type of every Dual column is DECLARED, and the declaration is asserted complete at import.
The round trip below is the part that matters: an empty bundle and a populated one must agree on
the schema, or a cohort's shards cannot be read as one dataset.
"""

from __future__ import annotations

import pathlib

import pytest

import scripts.rf_fusion_v2_artifacts as artifacts

pq = pytest.importorskip("pyarrow.parquet")

EMPTY_V2 = {name: [] for name in artifacts.V2_TABLE_SCHEMAS}

NUMERIC = {
    "raw_risk_a": -1.2, "raw_risk_b": 0.4, "u_a": -0.5, "u_b": 0.3,
    "joint_risk": 0.25, "hotspot_b_positive_count": 3, "sequence_length": 12,
}


def _row(schema, **over):
    row = {column: "" for column in schema.columns}
    for column in schema.columns:
        kind = artifacts.DUAL_COLUMN_TYPES[column]
        row[column] = {"int": 0, "float": 0.0, "bool": False, "str": ""}[kind]
    row.update(over)
    return row


def _dual_rows(**over):
    rows = {}
    for name, schema in artifacts.DUAL_TABLE_SCHEMAS.items():
        rows[name] = [_row(schema, **{k: v for k, v in over.items() if k in schema.columns})]
    return rows


def _write(tmp_path, dual_tables):
    artifacts.write_v2_bundle(
        tmp_path, manifest={"schema_version": "v2run-1"}, tables=EMPTY_V2,
        dual_tables=dual_tables)
    return {p.stem: pq.read_table(p).schema
            for p in pathlib.Path(tmp_path).glob("dual_*.parquet")}


def test_every_declared_dual_column_has_a_declared_arrow_type():
    declared = {c for s in artifacts.DUAL_TABLE_SCHEMAS.values() for c in s.columns}
    assert declared - set(artifacts.DUAL_COLUMN_TYPES) == set()


def test_an_empty_dual_bundle_types_its_numeric_columns_numerically(tmp_path):
    schemas = _write(tmp_path, {n: [] for n in artifacts.DUAL_TABLE_SCHEMAS})
    field = schemas["dual_endpoint_evidence"].field("raw_risk_a")
    assert "string" not in str(field.type), (
        "an empty shard typed a measured risk as text; the first shard with a real value could "
        "not be appended to it")


def test_a_populated_dual_bundle_round_trips_its_measurements(tmp_path):
    _write(tmp_path, _dual_rows(**NUMERIC))
    table = pq.read_table(tmp_path / "dual_endpoint_evidence.parquet")
    got = table.to_pylist()[0]
    for column, value in NUMERIC.items():
        if column in table.schema.names:
            assert got[column] == pytest.approx(value), column


def test_an_empty_and_a_populated_dual_shard_share_one_schema(tmp_path):
    empty = _write(tmp_path / "empty", {n: [] for n in artifacts.DUAL_TABLE_SCHEMAS})
    full = _write(tmp_path / "full", _dual_rows(**NUMERIC))
    for name in empty:
        assert empty[name] == full[name], (
            f"{name}: two shards of one cohort disagree on the schema, so they cannot be read as "
            "one dataset")


def test_the_dual_registry_does_not_retype_a_legacy_column():
    shared = set(artifacts.DUAL_COLUMN_TYPES) & set(artifacts.V2_COLUMN_TYPES)
    for column in shared:
        assert artifacts.DUAL_COLUMN_TYPES[column] == artifacts.V2_COLUMN_TYPES[column], column
