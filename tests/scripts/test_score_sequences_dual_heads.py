"""The refusals that keep a two-allele comparison from being wrong while looking right."""
from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inverse_folding.reference_flow.fusion_v2.identity import HeadEvaluatorIdentity  # noqa: E402
from inverse_folding.reference_flow.fusion_v2.joint_objective import (  # noqa: E402
    AlleleCoordinate, AlleleRole, DualCalibration, DualObjectiveLaw, DualObjective, ObjectiveMode,
    PanelBinding, QuantityCoordinates,
)
from scripts.score_sequences_dual_heads import (  # noqa: E402
    DualScoringError, assert_overlay_binds, read_sequences, score_table,
)

TAU = 0.10 / 0.6931471805599453


def _evaluator(allele="DRB1_0701", digest="a" * 64):
    return HeadEvaluatorIdentity(
        allele=allele, score_scale="raw_logit", window_k_min=12, window_k_max=25,
        head_config_hash="c" * 64, head_checkpoint_digest=digest)


def _coordinates(quantity="global_risk", digest_a="a" * 64, digest_b="b" * 64):
    return QuantityCoordinates(
        quantity=quantity,
        a=AlleleCoordinate(role=AlleleRole.A, quantity=quantity, location=-6.0, scale=6.0,
                           raw_noise_floor=1e-6, evaluator=_evaluator("DRB1_0701", digest_a),
                           source_ref="d" * 64),
        b=AlleleCoordinate(role=AlleleRole.B, quantity=quantity, location=-2.0, scale=4.0,
                           raw_noise_floor=1e-6, evaluator=_evaluator("DRB1_0401", digest_b),
                           source_ref="e" * 64))


def _calibration(digest_a="a" * 64, digest_b="b" * 64):
    return DualCalibration(
        version="dual-obj-1",
        law=DualObjectiveLaw(mode=ObjectiveMode.SMOOTH_MAX, tau=TAU, tau_units="normalized",
                             version="dual-obj-1", declared_credit_normalized=0.10),
        risk=_coordinates("global_risk", digest_a, digest_b),
        density=_coordinates("positive_mass_density", digest_a, digest_b),
        window=_coordinates("window_z", digest_a, digest_b),
        panel=PanelBinding(panel_id="p", panel_digest="f" * 64, n_proteins=10,
                           cross_allele_pearson=0.1, equal_risk_line_stderr=0.02,
                           overlap_fraction_a=0.1, overlap_fraction_b=0.1,
                           leave_overlap_out_shift=1e-4))


class _Overlay:
    def __init__(self, calibration):
        self.calibration = calibration


class _Head:
    """Scores by a fixed per-sequence table, keyed by md5, and declares one identity.

    Shaped like the REAL ``ProductionHeadOracle``: the window-grid digest lives on ``.binding``,
    not on the result.  An earlier fake put it on the result itself, every test passed, and the
    first cluster run died on the real object -- a fake that is easier to satisfy than the thing it
    stands for tests nothing.
    """

    def __init__(self, identity, values, *, grid="w" * 64):
        self.identity, self.values, self.grid = identity, values, grid

    def evaluator_identity(self):
        return self.identity

    def score(self, requests):
        out = []
        for request in requests:
            out.append(types.SimpleNamespace(
                sequence_md5=request.sequence_md5, sequence_length=request.sequence_length,
                protein_id=request.protein_id, allele=self.identity.allele,
                score_scale=self.identity.score_scale,
                binding=types.SimpleNamespace(window_grid_digest=self.grid),
                evaluator=self.identity,
                global_risk=self.values[request.sequence_md5]))
        return out


def _table(sequences):
    rows = [{"protein_id": f"P{i}", "design_idx": 0, "sequence": s, "method": "wt",
             "source": "/src"} for i, s in enumerate(sequences)]
    return pd.DataFrame(rows)


def _values(sequences, base):
    return {hashlib.md5(s.encode()).hexdigest(): base + 0.1 * i
            for i, s in enumerate(sequences)}


def test_an_overlay_calibrated_on_other_checkpoints_is_refused():
    overlay = _Overlay(_calibration(digest_a="a" * 64, digest_b="b" * 64))
    with pytest.raises(DualScoringError, match="never measured on"):
        assert_overlay_binds(overlay, evaluator_a=_evaluator("DRB1_0701", "9" * 64),
                             evaluator_b=_evaluator("DRB1_0401", "b" * 64))


def test_an_overlay_that_binds_these_exact_heads_is_accepted():
    overlay = _Overlay(_calibration())
    assert_overlay_binds(overlay, evaluator_a=_evaluator("DRB1_0701", "a" * 64),
                         evaluator_b=_evaluator("DRB1_0401", "b" * 64)) is None


def test_a_changed_window_grid_is_refused_even_with_the_right_checkpoint():
    overlay = _Overlay(_calibration())
    other = HeadEvaluatorIdentity(
        allele="DRB1_0701", score_scale="raw_logit", window_k_min=13, window_k_max=25,
        head_config_hash="c" * 64, head_checkpoint_digest="a" * 64)
    with pytest.raises(DualScoringError, match="never measured on"):
        assert_overlay_binds(overlay, evaluator_a=other,
                             evaluator_b=_evaluator("DRB1_0401", "b" * 64))


def test_both_raw_risks_come_from_one_sequence_and_j_is_the_overlay_law():
    sequences = ["ACDEFGHIKL", "MNPQRSTVWY"]
    objective = DualObjective(_calibration(), quantity="global_risk")
    rows = score_table(_table(sequences),
                       head_a=_Head(_evaluator("DRB1_0701", "a" * 64), _values(sequences, -7.0)),
                       head_b=_Head(_evaluator("DRB1_0401", "b" * 64), _values(sequences, -3.0)),
                       objective=objective)
    assert len(rows) == 2
    for row, sequence in zip(rows, sequences):
        assert row["sequence"] == sequence
        assert row["sequence_md5"] == hashlib.md5(sequence.encode()).hexdigest()
        expected = objective.evaluate(raw_a=row["R_A"], raw_b=row["R_B"])
        assert row["u_a"] == pytest.approx(expected.u_a)
        assert row["u_b"] == pytest.approx(expected.u_b)
        assert row["J"] == pytest.approx(expected.value)
        assert row["active_worst"] == expected.active_worst.value


def test_a_non_finite_risk_fails_instead_of_shrinking_the_cohort():
    sequences = ["ACDEFGHIKL"]
    values_a = {hashlib.md5(sequences[0].encode()).hexdigest(): float("nan")}
    with pytest.raises(DualScoringError, match="non-finite"):
        score_table(_table(sequences),
                    head_a=_Head(_evaluator("DRB1_0701", "a" * 64), values_a),
                    head_b=_Head(_evaluator("DRB1_0401", "b" * 64), _values(sequences, -3.0)),
                    objective=DualObjective(_calibration(), quantity="global_risk"))


def test_duplicate_join_keys_are_refused(tmp_path):
    path = tmp_path / "dup.parquet"
    pd.DataFrame([
        {"protein_id": "P1", "design_idx": 0, "sequence": "AAAA"},
        {"protein_id": "P1", "design_idx": 0, "sequence": "CCCC"},
    ]).to_parquet(path, index=False)
    with pytest.raises(DualScoringError, match="duplicated"):
        read_sequences([f"wt={path}"])


def test_labels_are_carried_as_the_method_column(tmp_path):
    for name, sequence in (("wt", "AAAA"), ("proteinmpnn", "CCCC")):
        pd.DataFrame([{"protein_id": "P1", "design_idx": 0, "sequence": sequence}]).to_parquet(
            tmp_path / f"{name}.parquet", index=False)
    table = read_sequences([f"wt={tmp_path/'wt.parquet'}",
                            f"proteinmpnn={tmp_path/'proteinmpnn.parquet'}"])
    assert sorted(table["method"]) == ["proteinmpnn", "wt"]
    assert len(table) == 2


def test_a_sequence_table_missing_a_required_column_is_refused(tmp_path):
    path = tmp_path / "bad.parquet"
    pd.DataFrame([{"protein_id": "P1", "sequence": "AAAA"}]).to_parquet(path, index=False)
    with pytest.raises(DualScoringError, match="design_idx"):
        read_sequences([f"wt={path}"])


def test_the_window_grid_digest_is_read_off_the_binding_like_the_real_oracle():
    """`ProductionHeadOracle` returns a V2HeadResult carrying its grid digest on `.binding`.

    `DualObjective.evaluate_scores` reads it off the score itself, so the pairing needs a view.
    """
    from scripts.score_sequences_dual_heads import _ScoreView
    result = types.SimpleNamespace(
        protein_id="P1", sequence_md5="a" * 32, sequence_length=10, allele="DRB1_0701",
        score_scale="raw_logit", global_risk=-7.0,
        binding=types.SimpleNamespace(window_grid_digest="g" * 64))
    assert _ScoreView(result).window_grid_digest == "g" * 64


def test_a_result_with_no_grid_digest_anywhere_is_refused():
    from scripts.score_sequences_dual_heads import _ScoreView
    result = types.SimpleNamespace(
        protein_id="P1", sequence_md5="a" * 32, sequence_length=10, allele="DRB1_0701",
        score_scale="raw_logit", global_risk=-7.0, binding=types.SimpleNamespace())
    with pytest.raises(DualScoringError, match="window\ngrid digest|window grid digest"):
        _ScoreView(result)


def test_two_heads_on_different_window_grids_cannot_be_paired():
    sequences = ["ACDEFGHIKL"]
    with pytest.raises(Exception, match="one window|window_grid_digest"):
        score_table(_table(sequences),
                    head_a=_Head(_evaluator("DRB1_0701", "a" * 64), _values(sequences, -7.0),
                                 grid="1" * 64),
                    head_b=_Head(_evaluator("DRB1_0401", "b" * 64), _values(sequences, -3.0),
                                 grid="2" * 64),
                    objective=DualObjective(_calibration(), quantity="global_risk"))
