"""Unit tests for inverse_folding/reference_flow/refine.py (pure refinement logic).

Loaded by file path to avoid the heavy reference_flow package __init__ (which imports
the torch-backed sampler); refine.py itself has no torch / I/O dependency.

Correctness-critical: a mis-mapped core would fix/mutate the wrong residue during
refinement, and a wrong block grouping would break whack-a-mole co-targeting.
"""
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_PATH = (pathlib.Path(__file__).resolve().parents[2]
         / "inverse_folding" / "reference_flow" / "refine.py")
_spec = importlib.util.spec_from_file_location("rf_refine", _PATH)
refine = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = refine  # dataclass needs the module resolvable via __module__
_spec.loader.exec_module(refine)

EpitopeCore = refine.EpitopeCore
extract_target_cores = refine.extract_target_cores
editable_positions = refine.editable_positions
hotspot_blocks = refine.hotspot_blocks
Candidate = refine.Candidate
enumerate_singles = refine.enumerate_singles
enumerate_pairs = refine.enumerate_pairs


def _row(pos, peptide, core, rank, pep_length=None):
    return {"pos": pos, "pep_length": pep_length or len(peptide),
            "peptide": peptide, "core": core, "rank_EL": rank}


def test_extract_keeps_only_strong_and_dedups_by_core_start():
    rows = [
        _row(10, "ABCDEFGHI", "ABCDEFGHI", 0.005),   # strong
        _row(10, "ABCDEFGHI", "ABCDEFGHI", 0.018),   # same core_start, weaker -> loses to 0.005
        _row(40, "KLMNPQRST", "KLMNPQRST", 0.10),     # weak (>=2%) -> dropped
    ]
    cores = extract_target_cores(rows, anchors=set(), strong_rank=0.02)
    assert [c.core_start for c in cores] == [10]
    assert cores[0].core_seq == "ABCDEFGHI"
    assert cores[0].best_rank == 0.005
    assert cores[0].pocket_positions == (10, 13, 15, 18)   # P1/P4/P6/P9


def test_pocket_within_15mer_uses_core_offset():
    # core sits at offset +3 inside a 15-mer peptide -> pockets shift by 3
    row = _row(10, "XYZABCDEFGHIJKL", "ABCDEFGHI", 0.001, pep_length=15)
    cores = extract_target_cores([row], anchors=set())
    assert cores[0].core_start == 13
    assert cores[0].pocket_positions == (13, 16, 18, 21)


def test_editable_excludes_anchors_and_adds_high_head_in_span():
    core = EpitopeCore(core_start=10, core_seq="ABCDEFGHI", best_rank=0.005,
                       pocket_positions=(10, 13, 15, 18), editable_positions=())
    ed = editable_positions(core, anchors={13}, head_high_positions={12, 99})
    assert set(ed) == {10, 12, 15, 18}   # 13 dropped (anchor); 99 dropped (outside span)


def test_hotspot_blocks_merge_overlap_separate_distant():
    cores = extract_target_cores(
        [_row(10, "AAAAAAAAA", "AAAAAAAAA", 0.001),
         _row(15, "CCCCCCCCC", "CCCCCCCCC", 0.001),   # span [15,24) overlaps [10,19) -> same block
         _row(91, "GGGGGGGGG", "GGGGGGGGG", 0.001)],  # distant -> own block
        anchors=set())
    blocks = hotspot_blocks(cores, gap=0)
    assert [[c.core_start for c in b] for b in blocks] == [[10, 15], [91]]


def test_singles_cover_editable_times_19_and_skip_wildtype():
    seq = "ACDEFGHIKL"          # positions 0..9
    ed = (2, 5)                 # seq[2]='D', seq[5]='G'
    singles = enumerate_singles(seq, ed)
    assert len(singles) == 2 * 19    # 19 non-wild-type AAs per editable position
    for c in singles:
        diffs = [i for i in range(len(seq)) if seq[i] != c.seq[i]]
        assert len(diffs) == 1 and diffs[0] in ed
        assert c.positions == (diffs[0],)
        assert c.seq[c.positions[0]] != seq[c.positions[0]]   # never the wild-type


def test_pairs_skip_same_position_dedup_and_respect_max():
    seq = "ACDEFGHIKL"
    choices = [(2, "R"), (5, "W"), (2, "K")]     # ranked single (pos, aa) choices
    pairs = enumerate_pairs(seq, choices)
    # cross-position only: (2,R)+(5,W) and (2,K)+(5,W); (2,R)+(2,K) skipped (same pos)
    assert len(pairs) == 2
    for c in pairs:
        assert len(c.positions) == 2 and c.positions[0] != c.positions[1]
        diffs = [i for i in range(len(seq)) if seq[i] != c.seq[i]]
        assert set(diffs) == set(c.positions)
    assert len(enumerate_pairs(seq, choices, max_pairs=1)) == 1


def _window(start, end, z):
    return SimpleNamespace(start_0b=start, end_0b=end, k=end - start, z=z)


def test_head_metrics_use_fusion_positive_mass_density():
    metrics = refine.head_refinement_metrics(
        SimpleNamespace(
            global_risk=1.25,
            residue_hotspot=(-1.0, 1.0, 2.0, 0.0),
            sequence_length=4,
            windows=(_window(0, 3, 1.0), _window(1, 4, 2.0)),
        ),
        protein_id="P",
        sequence="ACDE",
    )
    assert metrics.positive_mass == 3.0
    assert metrics.positive_mass_density == 0.75
    assert metrics.n_positive_hotspot_positions == 2


def test_head_residue_targets_apply_inclusive_threshold_local_maxima_anchor_and_cap():
    metrics = refine.head_refinement_metrics(
        SimpleNamespace(
            global_risk=1.0,
            residue_hotspot=(0.15, 0.30, 0.10, 0.40, 0.20, 0.10, 0.35, 0.10),
            sequence_length=8,
            windows=(_window(0, 8, 1.0),),
        ),
        protein_id="P",
        sequence="ACDEFGHI",
    )
    target = refine.select_head_residue_targets(
        metrics,
        anchors={3},
        threshold=0.15,
        max_positions=2,
    )
    # The anchored 0.40 peak is masked before maxima detection, so its editable
    # 0.20 neighbour remains discoverable. The cap keeps the two strongest peaks.
    assert target.positions == (6, 1)
    assert target.hotspot_values == (0.35, 0.30)
    assert target.n_above_threshold == 4
    assert target.n_local_maxima_pre_cap == 3


def test_head_residue_targets_collapse_plateau_and_allow_zero_targets():
    plateau = refine.HeadRefinementMetrics(
        global_risk=0.0,
        positive_mass=0.8,
        positive_mass_density=0.16,
        mean_hotspot=0.16,
        max_hotspot=0.20,
        n_positive_hotspot_positions=5,
        residue_hotspot=(0.10, 0.20, 0.20, 0.20, 0.10),
        windows=(_window(0, 5, 0.0),),
    )
    target = refine.select_head_residue_targets(
        plateau, anchors=set(), threshold=0.15, max_positions=12
    )
    assert target.positions == (2,)

    flat = refine.replace(
        plateau,
        positive_mass=0.56,
        positive_mass_density=0.112,
        mean_hotspot=0.112,
        max_hotspot=0.14,
        residue_hotspot=(0.14,) * 5,
    )
    target = refine.select_head_residue_targets(
        flat, anchors=set(), threshold=0.15, max_positions=12
    )
    assert target.positions == ()
    assert target.n_above_threshold == 0
    assert target.n_local_maxima_pre_cap == 0


def test_head_target_mode_fails_closed_without_valid_window_table():
    base = dict(global_risk=1.0, residue_hotspot=(1.0,) * 4, sequence_length=4)
    with pytest.raises(ValueError, match="windows are required"):
        refine.head_refinement_metrics(
            SimpleNamespace(**base), protein_id="P", sequence="ACDE"
        )
    with pytest.raises(ValueError, match="invalid span/k"):
        refine.head_refinement_metrics(
            SimpleNamespace(**base, windows=(_window(0, 5, 1.0),)),
            protein_id="P",
            sequence="ACDE",
        )


def _head_metrics(global_risk, density, *, length=4):
    return refine.HeadRefinementMetrics(
        global_risk=global_risk,
        positive_mass=density * length,
        positive_mass_density=density,
        mean_hotspot=0.0,
        max_hotspot=0.0,
        n_positive_hotspot_positions=0,
        residue_hotspot=(0.0,) * length,
        windows=(_window(0, length, 0.0),),
    )


def test_head_single_aa_selection_keeps_each_axis_extreme_per_position():
    seed = "AAAA"
    candidates = [
        Candidate("A0C", "CAAA", (0,)),
        Candidate("A0D", "DAAA", (0,)),
        Candidate("A0E", "EAAA", (0,)),
        Candidate("A2C", "AACA", (2,)),
        Candidate("A2D", "AADA", (2,)),
    ]
    metrics = {
        "CAAA": _head_metrics(0.70, 0.50),
        "DAAA": _head_metrics(0.90, 0.30),
        "EAAA": _head_metrics(0.80, 0.40),
        "AACA": _head_metrics(0.60, 0.60),
        "AADA": _head_metrics(0.65, 0.55),
    }
    selected = refine.select_head_single_aa_choices(
        candidates,
        metrics,
        position_order=(0, 2),
        max_aa_per_position=2,
    )
    assert [row.seq for row in selected[0]] == ["CAAA", "DAAA"]
    # One candidate dominates the other on both axes, so the first Pareto layer
    # contributes one AA and the second layer fills the explicit cap.
    assert [row.seq for row in selected[2]] == ["AACA", "AADA"]


def test_position_pair_round_robin_covers_each_position_once_per_round():
    pairs = refine.round_robin_position_pairs((9, 3, 7, 1))
    assert len(pairs) == 6
    assert len(set(pairs)) == 6
    assert set(pairs) == {
        (1, 3), (1, 7), (1, 9), (3, 7), (3, 9), (7, 9)
    }
    for offset in range(0, 6, 2):
        assert sorted(position for pair in pairs[offset:offset + 2] for position in pair) \
            == [1, 3, 7, 9]


@pytest.mark.parametrize("n_positions", range(2, 10))
def test_position_pair_round_robin_is_complete_for_odd_and_even_counts(n_positions):
    positions = tuple(range(n_positions))
    pairs = refine.round_robin_position_pairs(positions)
    expected = {
        (left, right)
        for left in positions
        for right in positions
        if left < right
    }
    assert set(pairs) == expected
    assert len(pairs) == len(expected)
    pairs_per_round = n_positions // 2
    for offset in range(0, len(pairs), pairs_per_round):
        round_pairs = pairs[offset:offset + pairs_per_round]
        flattened = [position for pair in round_pairs for position in pair]
        assert len(flattened) == len(set(flattened))


def test_round_robin_doubles_cover_position_pairs_before_second_aa_combo():
    seed = "AAAA"
    choices = {
        0: (Candidate("A0C", "CAAA", (0,)), Candidate("A0D", "DAAA", (0,))),
        1: (Candidate("A1C", "ACAA", (1,)), Candidate("A1D", "ADAA", (1,))),
        2: (Candidate("A2C", "AACA", (2,)), Candidate("A2D", "AADA", (2,))),
        3: (Candidate("A3C", "AAAC", (3,)), Candidate("A3D", "AAAD", (3,))),
    }
    doubles = refine.enumerate_round_robin_doubles(
        seed,
        choices,
        position_order=(0, 1, 2, 3),
        max_candidates=6,
    )
    assert len(doubles) == 6
    assert {candidate.positions for candidate in doubles} == {
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
    }
    assert all(candidate.seq[p] == "C" for candidate in doubles for p in candidate.positions)


def test_default_head_double_cap_covers_every_twelve_position_pair_three_times():
    seed = "A" * 12
    choices = {}
    for position in range(12):
        first = seed[:position] + "C" + seed[position + 1:]
        second = seed[:position] + "D" + seed[position + 1:]
        choices[position] = (
            Candidate(f"A{position}C", first, (position,)),
            Candidate(f"A{position}D", second, (position,)),
        )
    doubles = refine.enumerate_round_robin_doubles(
        seed,
        choices,
        position_order=tuple(range(12)),
        max_candidates=200,
    )
    pair_counts = {}
    for candidate in doubles:
        pair_counts[candidate.positions] = pair_counts.get(candidate.positions, 0) + 1
    assert len(doubles) == 200
    assert len(pair_counts) == 66
    assert min(pair_counts.values()) == 3
    assert max(pair_counts.values()) == 4


def test_head_pareto_dominance_keeps_axes_coequal():
    def metrics(global_risk, density):
        return refine.HeadRefinementMetrics(
            global_risk=global_risk,
            positive_mass=density * 10,
            positive_mass_density=density,
            mean_hotspot=0.0,
            max_hotspot=0.0,
            n_positive_hotspot_positions=0,
            residue_hotspot=(0.0,) * 10,
            windows=(_window(0, 10, 0.0),),
        )

    incumbent = metrics(1.0, 0.5)
    tradeoff = metrics(0.8, 0.6)
    dominant = metrics(0.9, 0.5)
    assert refine.head_objective_dominates(tradeoff, incumbent) is False
    assert refine.head_objective_dominates(dominant, incumbent) is True
    assert refine.head_first_pareto_front([incumbent, tradeoff, dominant]) == (1, 2)


def test_head_refinement_rejects_global_gain_that_worsens_positive_mass_density():
    seed = "AA"
    structure_calls = []

    def head_score(_protein_id, sequences):
        return [
            SimpleNamespace(
                global_risk=1.0 if sequence == seed else 0.9,
                residue_hotspot=(1.0, 0.0) if sequence == seed else (2.0, 0.0),
                sequence_length=2,
                windows=(_window(0, 2, 1.0),),
            )
            for sequence in sequences
        ]

    def structure(_protein_id, sequence):
        structure_calls.append(sequence)
        return _OK

    result = refine.refine_sequence_head(
        "P",
        seed,
        head_score_fn=head_score,
        struct_fn=structure,
        anchors=set(),
        residue_threshold=0.15,
        max_target_positions=12,
        aa_per_position=1,
        max_double_candidates=0,
        alphabet="AC",
        max_rounds=2,
        patience=1,
        beam_width=2,
        refold_cap=2,
    )
    assert result.shortlist == []
    assert result.diverged is True
    assert structure_calls == [seed]


def test_head_refinement_scores_singles_then_full_double_and_admits_synergy():
    seed = "AAAA"
    head_batches = []
    structure_calls = []

    def head_score(_protein_id, sequences):
        head_batches.append(tuple(sequences))
        rows = []
        for sequence in sequences:
            n_mutations = sum(a != b for a, b in zip(seed, sequence))
            if n_mutations == 0:
                risk, hotspot = 1.0, (0.30, 0.0, 0.25, 0.0)
            elif n_mutations == 1:
                # Each single is a tradeoff and cannot advance the beam alone.
                risk, hotspot = 0.9, (0.35, 0.0, 0.25, 0.0)
            else:
                risk, hotspot = 0.8, (0.20, 0.0, 0.20, 0.0)
            rows.append(SimpleNamespace(
                global_risk=risk,
                residue_hotspot=hotspot,
                sequence_length=4,
                windows=(_window(0, 4, risk),),
            ))
        return rows

    def structure(_protein_id, sequence):
        structure_calls.append(sequence)
        return _OK

    result = refine.refine_sequence_head(
        "P",
        seed,
        head_score_fn=head_score,
        struct_fn=structure,
        anchors=set(),
        residue_threshold=0.15,
        max_target_positions=12,
        aa_per_position=1,
        max_double_candidates=1,
        alphabet="AC",
        max_rounds=1,
        beam_width=2,
        refold_cap=2,
    )
    assert len(result.shortlist) == 1
    assert result.shortlist[0]["seq"] == "CACA"
    assert head_batches == [(seed,), ("CAAA", "AACA"), ("CACA",)]
    assert structure_calls == [seed, "CACA"]
    assert result.trace[0]["n_single_candidates"] == 2
    assert result.trace[0]["n_retained_single_aas"] == 2
    assert result.trace[0]["n_double_candidates"] == 1


def test_head_beam_accumulates_single_edits_across_rounds():
    seed = "AAAA"

    def head_score(_protein_id, sequences):
        rows = []
        for sequence in sequences:
            changed = {position for position in (0, 2) if sequence[position] != "A"}
            hotspot = tuple(
                0.10 if position in changed else 0.30 if position in (0, 2) else 0.0
                for position in range(4)
            )
            rows.append(SimpleNamespace(
                global_risk=float(2 - len(changed)),
                residue_hotspot=hotspot,
                sequence_length=4,
                windows=(_window(0, 4, float(2 - len(changed))),),
            ))
        return rows

    result = refine.refine_sequence_head(
        "P",
        seed,
        head_score_fn=head_score,
        struct_fn=lambda _protein_id, _sequence: _OK,
        anchors=set(),
        residue_threshold=0.15,
        max_target_positions=12,
        aa_per_position=1,
        max_double_candidates=0,
        alphabet="AC",
        max_rounds=2,
        beam_width=4,
        refold_cap=4,
        max_path_mutations=2,
    )
    assert "CACA" in {row["seq"] for row in result.shortlist}
    assert [row["n_single_candidates"] for row in result.trace] == [2, 2]
    assert all(row["n_double_candidates"] == 0 for row in result.trace)


def test_head_refinement_low_landscape_returns_no_targets_without_candidate_scoring():
    seed = "AAAA"
    head_batches = []
    structure_calls = []

    def head_score(_protein_id, sequences):
        head_batches.append(tuple(sequences))
        return [SimpleNamespace(
            global_risk=-9.0,
            residue_hotspot=(0.14,) * 4,
            sequence_length=4,
            windows=(_window(0, 4, -9.0),),
        ) for _sequence in sequences]

    def structure(_protein_id, sequence):
        structure_calls.append(sequence)
        return _OK

    result = refine.refine_sequence_head(
        "P",
        seed,
        head_score_fn=head_score,
        struct_fn=structure,
        anchors=set(),
        residue_threshold=0.15,
        max_target_positions=12,
        aa_per_position=2,
        max_double_candidates=200,
        max_rounds=3,
    )
    assert result.shortlist == []
    assert result.trace == []
    assert head_batches == [(seed,)]
    assert structure_calls == [seed]


# --------------------------------------------------------------------------- #
# Task R3 — margin, structure gate, accept rule, beam search loop
# --------------------------------------------------------------------------- #
rank_margin_mass = refine.rank_margin_mass
structure_gate = refine.structure_gate
accept_refinement = refine.accept_refinement
refine_sequence = refine.refine_sequence
StructureMetrics = refine.StructureMetrics

_OK = StructureMetrics(scTM=0.90, pLDDT=90.0)          # passes any reasonable floor


def _propose_singles(seq, cores):
    ed = sorted({p for c in cores for p in c.editable_positions})
    return refine.enumerate_singles(seq, ed)


def test_rank_margin_mass_rewards_pushing_toward_threshold():
    assert round(rank_margin_mass([0.002], strong_rank=0.02, margin_band=0.10), 6) == 0.018  # deep = most mass
    assert round(rank_margin_mass([0.018], strong_rank=0.02, margin_band=0.10), 3) == 0.002
    assert rank_margin_mass([0.05], strong_rank=0.02, margin_band=0.10) == 0.0      # outside band


def test_structure_gate_enforces_scTM_floor():
    seed = StructureMetrics(scTM=0.80, pLDDT=90.0)
    ok, _ = structure_gate(seed, StructureMetrics(scTM=0.78, pLDDT=88.0), scTM_eps=0.05)
    bad, reason = structure_gate(seed, StructureMetrics(scTM=0.70, pLDDT=88.0), scTM_eps=0.05)
    assert ok is True and bad is False and "scTM" in reason


def test_structure_gate_enforces_protocol_metrics_conjunctively():
    seed = StructureMetrics(scTM=0.96, pLDDT=92.0)
    good = StructureMetrics(
        scTM=0.95,
        pLDDT=91.0,
        cat_max_scRMSD=1.2,
        predicted_active_site_min_pLDDT=88.0,
    )
    thresholds = {
        "scTM_min": 0.94,
        "cat_max_scRMSD_max": 1.5,
        "predicted_active_site_min_pLDDT_min": 85.0,
    }

    assert structure_gate(seed, good, scTM_eps=None, **thresholds) == (True, "ok")

    bad_cases = [
        (refine.replace(good, scTM=0.93), "scTM"),
        (refine.replace(good, cat_max_scRMSD=1.6), "cat_max_scRMSD"),
        (
            refine.replace(good, predicted_active_site_min_pLDDT=84.0),
            "predicted_active_site_min_pLDDT",
        ),
    ]
    for candidate, metric_name in bad_cases:
        passed, reason = structure_gate(seed, candidate, scTM_eps=None, **thresholds)
        assert passed is False and metric_name in reason


def test_structure_gate_protocol_metrics_are_fail_closed_when_missing_or_nonfinite():
    seed = StructureMetrics(scTM=0.96, pLDDT=92.0)
    thresholds = {
        "scTM_min": 0.94,
        "cat_max_scRMSD_max": 1.5,
        "predicted_active_site_min_pLDDT_min": 85.0,
    }
    candidates = [
        StructureMetrics(
            scTM=float("nan"),
            pLDDT=91.0,
            cat_max_scRMSD=1.2,
            predicted_active_site_min_pLDDT=88.0,
        ),
        StructureMetrics(
            scTM=0.95,
            pLDDT=91.0,
            cat_max_scRMSD=None,
            predicted_active_site_min_pLDDT=88.0,
        ),
        StructureMetrics(
            scTM=0.95,
            pLDDT=91.0,
            cat_max_scRMSD=1.2,
            predicted_active_site_min_pLDDT=float("nan"),
        ),
    ]

    for candidate in candidates:
        passed, reason = structure_gate(seed, candidate, scTM_eps=None, **thresholds)
        assert passed is False and "unavailable/not finite" in reason


def test_structure_gate_sidechain_max_is_fail_closed():
    seed = StructureMetrics(scTM=0.90, pLDDT=90.0)
    good = StructureMetrics(
        scTM=0.90,
        pLDDT=88.0,
        max_anchor_sidechain_RMSD=1.2,
        active_site_complete=True,
    )
    bad = StructureMetrics(
        scTM=0.90,
        pLDDT=88.0,
        max_anchor_sidechain_RMSD=1.8,
        active_site_complete=True,
    )
    missing = StructureMetrics(scTM=0.90, pLDDT=88.0)

    assert structure_gate(
        seed,
        good,
        scTM_eps=0.05,
        max_anchor_sidechain_RMSD_max=1.5,
    ) == (True, "ok")
    passed, reason = structure_gate(
        seed,
        bad,
        scTM_eps=0.05,
        max_anchor_sidechain_RMSD_max=1.5,
    )
    assert passed is False and "max_anchor_sidechain_RMSD" in reason
    passed, reason = structure_gate(
        seed,
        missing,
        scTM_eps=0.05,
        max_anchor_sidechain_RMSD_max=1.5,
    )
    assert passed is False and "unavailable" in reason


def test_legacy_active_site_gate_missing_value_is_fail_closed():
    seed = StructureMetrics(scTM=0.90, pLDDT=90.0)
    passed, reason = structure_gate(
        seed,
        StructureMetrics(scTM=0.90, pLDDT=88.0),
        scTM_eps=0.05,
        active_site_RMSD_max=2.0,
    )
    assert passed is False and "unavailable" in reason


def test_accept_requires_count_drop_and_structure_pass():
    assert accept_refinement(seed_core_count=3, cand_core_count=2, structure_passed=True) is True
    assert accept_refinement(seed_core_count=3, cand_core_count=3, structure_passed=True) is False
    assert accept_refinement(seed_core_count=3, cand_core_count=2, structure_passed=False) is False


def test_refine_drives_count_to_zero_single_step():
    seq = "A" * 30
    def nmp(pid, seqs):
        def one(s):
            return [] if s[10] != "A" else [
                {"pos": 10, "pep_length": 9, "peptide": s[10:19], "core": s[10:19], "rank_EL": 0.001}]
        return [one(s) for s in seqs]
    def head(pid, seqs):
        return np.array([[0.0 if s[10] != "A" else 9.0] for s in seqs])
    def struct(pid, s):
        return _OK
    res = refine_sequence("P", seq, propose_fn=_propose_singles, head_fn=head, nmp_fn=nmp,
                          struct_fn=struct, anchors=set(), topB=None, beam_width=4,
                          max_rounds=5, scTM_eps=0.05, target_window_idx_fn=lambda cores: [0])
    assert res.best_core_count == 0 and res.best_seq[10] != "A" and res.n_accepts >= 1


def test_soft_beam_enables_two_step_elimination():
    # the core needs TWO edits (positions 10 and 13) to cross 2%; each single edit only lowers
    # the rank (count stays 1). Only a soft beam that carries the margin-improved 1-edit state
    # (which must PASS the structure gate to be admitted) can reach 0 on the next round.
    seq = "A" * 30
    def nmut(s):
        return sum(1 for p in (10, 13) if s[p] != "A")
    def nmp(pid, seqs):
        def one(s):
            rank = {0: 0.001, 1: 0.015}.get(nmut(s), 0.05)
            return [] if rank >= 0.02 else [
                {"pos": 10, "pep_length": 9, "peptide": s[10:19], "core": s[10:19], "rank_EL": rank}]
        return [one(s) for s in seqs]
    def head(pid, seqs):
        return np.array([[-nmut(s)] for s in seqs])
    def struct(pid, s):
        return _OK
    res = refine_sequence("P", seq, propose_fn=_propose_singles, head_fn=head, nmp_fn=nmp,
                          struct_fn=struct, anchors=set(), topB=None, beam_width=4,
                          max_rounds=6, patience=3, scTM_eps=0.05,
                          target_window_idx_fn=lambda cores: [0])
    assert res.best_core_count == 0 and res.n_accepts >= 1


def test_margin_progress_states_enter_beam_without_refold():
    # New v0 policy (PLAN §1 "structure gated on outputs"): a 1-edit margin-progress state
    # (count stays 1) enters the beam WITHOUT a refold; only the count-dropping 2-edit
    # output is refolded. struct_fn records every sequence it is asked to fold.
    seq = "A" * 30
    folded = []
    def nmut(s):
        return sum(1 for p in (10, 13) if s[p] != "A")
    def nmp(pid, seqs):
        def one(s):
            rank = {0: 0.001, 1: 0.015}.get(nmut(s), 0.05)
            return [] if rank >= 0.02 else [
                {"pos": 10, "pep_length": 9, "peptide": s[10:19], "core": s[10:19], "rank_EL": rank}]
        return [one(s) for s in seqs]
    def head(pid, seqs):
        return np.array([[-nmut(s)] for s in seqs])
    def struct(pid, s):
        folded.append(s)
        return _OK
    res = refine_sequence("P", seq, propose_fn=_propose_singles, head_fn=head, nmp_fn=nmp,
                          struct_fn=struct, anchors=set(), topB=None, beam_width=4,
                          max_rounds=6, patience=3, scTM_eps=0.05,
                          target_window_idx_fn=lambda cores: [0])
    assert res.best_core_count == 0
    # Only the seed and count-dropping (>=2-edit) candidates are refolded — never a
    # 1-edit margin-only intermediate.
    assert all(nmut(s) >= 2 for s in folded if s != seq)


def test_max_path_mutations_caps_deep_paths():
    # Reaching count 0 needs 2 edits (pos 10 & 13); with max_path_mutations=1 the 2-edit
    # output is dropped before refold, so no elimination is accepted.
    seq = "A" * 30
    def nmut(s):
        return sum(1 for p in (10, 13) if s[p] != "A")
    def nmp(pid, seqs):
        def one(s):
            rank = {0: 0.001, 1: 0.015}.get(nmut(s), 0.05)
            return [] if rank >= 0.02 else [
                {"pos": 10, "pep_length": 9, "peptide": s[10:19], "core": s[10:19], "rank_EL": rank}]
        return [one(s) for s in seqs]
    def head(pid, seqs):
        return np.array([[-nmut(s)] for s in seqs])
    def struct(pid, s):
        return _OK
    res = refine_sequence("P", seq, propose_fn=_propose_singles, head_fn=head, nmp_fn=nmp,
                          struct_fn=struct, anchors=set(), topB=None, beam_width=4,
                          max_rounds=6, patience=3, scTM_eps=0.05, max_path_mutations=1,
                          target_window_idx_fn=lambda cores: [0])
    assert res.best_core_count == 1 and res.n_accepts == 0


def test_incremental_nmp_splice_matches_full_scoring():
    # A per-window NMP that scores each peptide purely by its substring content
    # (position-INDEPENDENT, like real NetMHCIIpan's fixed-background %Rank). The
    # sub-sequence splice must reproduce a full re-score exactly for point mutations.
    LENGTHS = (9, 12, 15)

    def _full_rows(seq):
        rows = []
        for L in LENGTHS:
            for pos in range(0, len(seq) - L + 1):
                pep = seq[pos:pos + L]
                rows.append({"pos": pos, "pep_length": L, "peptide": pep,
                             "core": pep[:9], "rank_EL": 0.001 if "W" in pep else 0.5})
        return rows

    def nmp(pid, seqs):
        return [_full_rows(s) for s in seqs]

    seed = "ACDEFGHIK" * 14                       # 126 aa, no 'W'
    single = seed[:63] + "W" + seed[64:]          # one edit at position 63 (far from termini)
    pair = single[:30] + "W" + single[31:]        # a second edit at 30 (span 33 <= 61 budget)

    def _norm(rows):
        return sorted((r["pos"], r["pep_length"], r["peptide"], r["core"], r["rank_EL"])
                      for r in rows)

    seed_rows = nmp("P", [seed])[0]
    spliced = refine._incremental_nmp_rows("P", seed, seed_rows, [single, pair, seed], nmp,
                                           margin=30)
    assert _norm(spliced[0]) == _norm(nmp("P", [single])[0])   # splice reproduces full
    assert _norm(spliced[1]) == _norm(nmp("P", [pair])[0])     # multi-edit within budget
    assert spliced[2] is seed_rows                             # unchanged seq reuses the cache


def test_incremental_refine_equals_full_refine():
    # End-to-end: incremental_nmp=True must give the SAME best_core_count / n_accepts as the
    # full path under a position-independent per-window NMP.
    def _rows(seq):
        rows = []
        for pos in range(0, len(seq) - 9 + 1):
            pep = seq[pos:pos + 9]
            rows.append({"pos": pos, "pep_length": 9, "peptide": pep, "core": pep,
                         "rank_EL": 0.001 if "W" in pep else 0.5})
        return rows

    def nmp(pid, seqs):
        return [_rows(s) for s in seqs]

    def head(pid, seqs):
        return np.array([[0.0] for _ in seqs])

    def struct(pid, s):
        return _OK

    seed = "ACDEFGHIK" * 8 + "W" + "ACDEFGHIK" * 5    # one strong hotspot (the 'W' windows)
    kw = dict(propose_fn=_propose_singles, head_fn=head, nmp_fn=nmp, struct_fn=struct,
              anchors=set(), topB=None, beam_width=4, max_rounds=5, scTM_eps=0.05,
              target_window_idx_fn=lambda cores: [0])
    full = refine_sequence("P", seed, incremental_nmp=False, **kw)
    inc = refine_sequence("P", seed, incremental_nmp=True, **kw)
    assert full.best_core_count == inc.best_core_count
    assert full.n_accepts == inc.n_accepts


def test_structure_floor_blocks_fold_breaking_elimination_and_beam():
    seq = "A" * 30
    def nmp(pid, seqs):
        def one(s):
            return [] if s[10] != "A" else [
                {"pos": 10, "pep_length": 9, "peptide": s[10:19], "core": s[10:19], "rank_EL": 0.001}]
        return [one(s) for s in seqs]
    def head(pid, seqs):
        return np.array([[0.0 if s[10] != "A" else 9.0] for s in seqs])
    def struct(pid, s):    # the only count-dropping edit also breaks the fold
        return StructureMetrics(scTM=0.50, pLDDT=40.0) if s[10] != "A" else _OK
    res = refine_sequence("P", seq, propose_fn=_propose_singles, head_fn=head, nmp_fn=nmp,
                          struct_fn=struct, anchors=set(), topB=None, beam_width=4,
                          max_rounds=5, scTM_eps=0.05, target_window_idx_fn=lambda cores: [0])
    assert res.best_core_count == 1 and res.n_accepts == 0    # rejected; never admitted to the beam
