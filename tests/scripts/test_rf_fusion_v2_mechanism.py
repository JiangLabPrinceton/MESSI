"""The mechanism cohort's readout: does a projected endpoint reach the completed descendant?

Runbook §7.  The Canary proved the feedback machinery EXECUTES; `source_dependence_verdict` proves
the kernel is not blind to what it is handed.  Neither says the injected identity survives
propagation, and that is the whole scientific claim of V2 -- so it needs its own statistic.

The statistic is a normalized Hamming distance between two matched descendants, and everything that
matters about it is the DOMAIN it is computed on.  Two facts about the projection kernel force that
domain:

* ``inject_from_source_feedback`` writes the SOURCE's own token (``projection.py``), and
  ``carry_from_source`` copies source rows verbatim.  Under ``source_shuffle`` -- which permutes the
  source's resolved bytes -- those two classes differ between the arms BY CONSTRUCTION, with no
  mechanism involved at all.  On the three executed Canary coordinates that is 46/101, 90/277 and
  119/277 positions, so a whole-editable Hamming would read 0.33-0.46 for a kernel that transmits
  nothing.  A gate at 0.02 on that domain is not a gate.
* ``write_from_endpoint`` is the single position the endpoint's identity is copied to, so it differs
  between two different endpoints by construction too.

What is left -- editable, still unresolved at re-entry -- is exactly the set the arms share
byte-for-byte at ``r_d`` and that only forward propagation can decide.  A difference there is
transmission or it is nothing.  Hard anchors are outside ``editable_positions`` and the written
position is resolved, so ONE predicate excludes all three classes and no hand-maintained exclusion
list can fall out of date with the kernel.
"""

from __future__ import annotations

import json
import types

import pytest

from scripts.rf_fusion_v2_artifacts import (
    V2ArtifactError,
    V2_TABLE_SCHEMAS,
    free_domain,
    mechanism_contrast_rows,
)
from tests.inverse_folding import _v2_fixtures as F


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def _arm(projected, descendants):
    """A duck-typed CycleOutcome: the row builder reads only these two fields."""
    return types.SimpleNamespace(projected=projected, descendant_endpoints=tuple(descendants))


def _seq_from(projected, overrides=None):
    """A COMPLETE descendant: every editable position resolved, anchors kept."""
    tokens = list(projected.tokens)
    for position in projected.editable_positions:
        if tokens[position] == projected.mask_token_id:
            tokens[position] = 10          # any residue token; the tests override what matters
    for position, token in (overrides or {}).items():
        tokens[position] = token
    return "".join(F.ALPHABET[token] for token in tokens)


def _descendant(projected, overrides=None, endpoint_id="d0"):
    seq = _seq_from(projected, overrides)
    return types.SimpleNamespace(sequence=seq, endpoint_id=endpoint_id,
                                 sequence_md5=F.sequence_md5(seq) if hasattr(F, "sequence_md5")
                                 else "0" * 32)


def _rows(arm_a, arm_b, **over):
    kw = dict(protein_id="5ZHV_B", source_index=0, source_seed=7, source_state_id="s0",
              source_unresolved_editable=1, view="endpoint_change", arm_a=arm_a, arm_b=arm_b)
    kw.update(over)
    return mechanism_contrast_rows(**kw)


# --------------------------------------------------------------------------------------------
# the domain
# --------------------------------------------------------------------------------------------


def test_free_domain_is_the_unresolved_editable_positions_of_the_projection():
    projected = F.projected()
    expected = tuple(p for p in projected.editable_positions
                     if projected.tokens[p] == projected.mask_token_id)
    assert free_domain(projected) == expected
    assert expected, "the fixture must reopen something or the contrast has nowhere to land"


def test_free_domain_excludes_hard_anchors():
    projected = F.projected()
    anchors = {position for position, _ in projected.hard_anchors}
    assert anchors, "the fixture must carry an anchor for this to prove anything"
    assert not (set(free_domain(projected)) & anchors)


def test_free_domain_excludes_the_endpoint_written_position():
    """The one position the endpoint is copied to differs between endpoints by construction."""
    projected = F.projected()
    written = set(projected.support.write_from_endpoint)
    assert written, "a committed projection must write at least one position"
    assert not (set(free_domain(projected)) & written)


def test_free_domain_includes_every_reopened_position():
    projected = F.projected()
    assert set(projected.support.reopen) <= set(free_domain(projected))


# --------------------------------------------------------------------------------------------
# the statistic -- what must NOT count
# --------------------------------------------------------------------------------------------


def test_a_difference_at_a_carried_position_scores_zero():
    """The anti-vacuity property.

    ``source_shuffle`` rewrites the source's resolved bytes, and ``carry_from_source`` copies them
    verbatim into both projections.  If those positions counted, the positive control would pass at
    ~0.4 on a kernel that transmits nothing.
    """
    projected = F.projected()
    carried = [p for p in projected.support.carry_from_source
               if projected.tokens[p] != projected.mask_token_id]
    assert carried, "the fixture must carry a RESOLVED position for this to prove anything"

    arm_a = _arm(projected, [_descendant(projected)])
    arm_b = _arm(projected, [_descendant(projected, {carried[0]: 21})])

    row, = _rows(arm_a, arm_b)
    assert row["analyzable"] is True
    assert row["n_diff_free"] == 0
    assert row["hamming_free"] == 0.0
    # ... and the whole-editable reading, which the free domain exists to replace, does see it.
    assert row["n_diff_editable"] == 1


def test_identical_descendants_score_zero_on_both_domains():
    projected = F.projected()
    arm = _arm(projected, [_descendant(projected)])
    row, = _rows(arm, _arm(projected, [_descendant(projected)]))
    assert (row["n_diff_free"], row["n_diff_editable"]) == (0, 0)
    assert (row["hamming_free"], row["hamming_editable"]) == (0.0, 0.0)


# --------------------------------------------------------------------------------------------
# the statistic -- what MUST count
# --------------------------------------------------------------------------------------------


def test_a_difference_at_a_free_position_scores_on_the_free_domain():
    projected = F.projected()
    free = free_domain(projected)
    arm_a = _arm(projected, [_descendant(projected)])
    arm_b = _arm(projected, [_descendant(projected, {free[0]: 21})])

    row, = _rows(arm_a, arm_b)
    assert row["n_diff_free"] == 1
    assert row["hamming_free"] == pytest.approx(1.0 / len(free))
    assert row["n_free"] == len(free)


def test_both_denominators_are_recorded_so_delta_can_be_frozen_on_either():
    projected = F.projected()
    row, = _rows(_arm(projected, [_descendant(projected)]),
                 _arm(projected, [_descendant(projected)]))
    written = set(projected.support.write_from_endpoint)
    assert row["n_free"] == len(free_domain(projected))
    assert row["n_editable_scored"] == len(set(projected.editable_positions) - written)


# --------------------------------------------------------------------------------------------
# pairing discipline
# --------------------------------------------------------------------------------------------


def test_every_matched_fork_gets_its_own_row():
    projected = F.projected()
    arm_a = _arm(projected, [_descendant(projected, endpoint_id=f"a{k}") for k in range(3)])
    arm_b = _arm(projected, [_descendant(projected, endpoint_id=f"b{k}") for k in range(3)])
    rows = _rows(arm_a, arm_b)
    assert [row["fork_index"] for row in rows] == [0, 1, 2]
    assert [row["arm_b_descendant_id"] for row in rows] == ["b0", "b1", "b2"]


def test_unequal_descendant_counts_are_refused_rather_than_zipped():
    """``zip`` would silently drop the tail and report a matched contrast that is not one."""
    projected = F.projected()
    arm_a = _arm(projected, [_descendant(projected), _descendant(projected)])
    arm_b = _arm(projected, [_descendant(projected)])
    rows = _rows(arm_a, arm_b)
    assert len(rows) == 1
    assert rows[0]["analyzable"] is False
    assert "descendant" in rows[0]["reason"]


def test_a_null_arm_is_recorded_not_dropped():
    """A prefix that produced no projection is a RESULT; shrinking the sample would hide it."""
    projected = F.projected()
    rows = _rows(_arm(None, []), _arm(projected, [_descendant(projected)]))
    assert len(rows) == 1
    assert rows[0]["analyzable"] is False
    assert rows[0]["n_free"] is None


def test_arms_whose_free_domains_disagree_are_refused():
    """Support parity is enforced upstream; if it ever fails, the contrast is not attributable."""
    projected = F.projected()
    # Resolve one of arm B's free positions, so the two arms no longer share a measurement domain.
    other = F.projected()
    tokens = list(other.tokens)
    tokens[free_domain(other)[0]] = 10
    object.__setattr__(other, "tokens", tuple(tokens))

    rows = _rows(_arm(projected, [_descendant(projected)]),
                 _arm(other, [_descendant(other)]))
    assert rows[0]["analyzable"] is False
    assert "free domain" in rows[0]["reason"]


# --------------------------------------------------------------------------------------------
# provenance the analysis needs
# --------------------------------------------------------------------------------------------


def test_contrastable_says_whether_the_intervention_was_real():
    """Two endpoints that agree at the written position make the arms byte-identical inputs.

    Such a pair is a structural zero, not evidence of no transmission, and the analysis has to be
    able to exclude it by a pre-registered rule rather than by inspecting the outcome.
    """
    projected = F.projected()
    written = projected.support.write_from_endpoint[0]
    same = _rows(_arm(projected, [_descendant(projected)]),
                 _arm(projected, [_descendant(projected)]))[0]
    assert same["contrastable"] is False
    assert same["n_input_diff"] == 0

    other_tokens = list(projected.tokens)
    other_tokens[written] = projected.tokens[written] + 1
    perturbed = F.projected()
    object.__setattr__(perturbed, "tokens", tuple(other_tokens))
    differing = _rows(_arm(projected, [_descendant(projected)]),
                      _arm(perturbed, [_descendant(perturbed)]))[0]
    assert differing["contrastable"] is True
    assert differing["n_input_diff"] == 1


def test_rows_carry_the_view_and_the_source_so_prefixes_can_be_the_sampling_unit():
    projected = F.projected()
    row, = _rows(_arm(projected, [_descendant(projected)]),
                 _arm(projected, [_descendant(projected)]),
                 view="source_shuffle", source_index=11, source_seed=4242,
                 source_state_id="src-11")
    assert row["view"] == "source_shuffle"
    assert (row["source_index"], row["source_seed"]) == (11, 4242)
    assert row["source_state_id"] == "src-11"


def test_parity_violation_is_carried_and_makes_the_row_unanalyzable():
    projected = F.projected()
    row, = _rows(_arm(projected, [_descendant(projected)]),
                 _arm(projected, [_descendant(projected)]),
                 parity_violation="the arms changed different numbers of positions")
    assert row["analyzable"] is False
    assert "different numbers" in row["parity_violation"]


# --------------------------------------------------------------------------------------------
# the table is part of the bundle contract
# --------------------------------------------------------------------------------------------


def test_the_contrast_table_has_a_declared_schema():
    assert "mechanism_contrasts" in V2_TABLE_SCHEMAS


def test_the_declared_schema_matches_what_the_builder_emits():
    projected = F.projected()
    row, = _rows(_arm(projected, [_descendant(projected)]),
                 _arm(projected, [_descendant(projected)]))
    assert set(row) == set(V2_TABLE_SCHEMAS["mechanism_contrasts"].columns)


def test_an_unknown_view_is_refused():
    projected = F.projected()
    with pytest.raises(V2ArtifactError):
        _rows(_arm(projected, [_descendant(projected)]),
              _arm(projected, [_descendant(projected)]), view="wishful_thinking")


# --------------------------------------------------------------------------------------------
# the driver chooses which experiment it is, once
# --------------------------------------------------------------------------------------------


def test_the_driver_runs_the_ladder_by_default():
    from scripts.run_rf_fusion_v2 import select_runner

    args = types.SimpleNamespace(mechanism_prefixes=0)
    assert select_runner(args, ladder="LADDER", mechanism="MECH") == "LADDER"


def test_the_flag_selects_the_mechanism_shard_and_binds_the_prefix_count():
    from scripts.run_rf_fusion_v2 import select_runner

    def mechanism(**kw):
        return kw

    chosen = select_runner(types.SimpleNamespace(mechanism_prefixes=16, mechanism_prefix_start=0),
                           ladder="LADDER", mechanism=mechanism)
    assert chosen() == {"n_prefixes": 16, "prefix_start": 0, "qualification": False}


def test_a_negative_prefix_count_is_refused():
    from scripts.run_rf_fusion_v2 import V2DriverError, select_runner

    with pytest.raises(V2DriverError):
        select_runner(types.SimpleNamespace(mechanism_prefixes=-1),
                      ladder="LADDER", mechanism="MECH")


def test_the_ladder_payload_declares_the_contrast_table_empty():
    """A ladder run scores no contrast, and an ABSENT table is a different claim from an empty one."""
    from scripts.rf_fusion_v2_artifacts import V2_TABLE_SCHEMAS

    assert "mechanism_contrasts" in V2_TABLE_SCHEMAS
    source = (__import__("pathlib").Path("scripts/rf_fusion_v2_cohort.py")).read_text()
    assert '"mechanism_contrasts": [],' in source


def test_paired_arms_do_not_collide_on_the_feedback_event_join_key():
    """Both arms of a contrast share one transition id; the key has to separate them."""
    from scripts.rf_fusion_v2_artifacts import V2_TABLE_SCHEMAS

    key = V2_TABLE_SCHEMAS["feedback_events"].sort_by
    assert "arm_slot" in key and "treatment_identity" in key


# --------------------------------------------------------------------------------------------
# the readout (runbook §7)
# --------------------------------------------------------------------------------------------

pd = pytest.importorskip("pandas")


def _frame(spec, view="endpoint_change", protein_id="5ZHV_B", n_free=55):
    """`spec` maps source_index -> list of per-fork free-domain Hamming values."""
    rows = []
    for source_index, values in spec.items():
        for fork_index, value in enumerate(values):
            rows.append({
                "protein_id": protein_id, "view": view, "source_index": source_index,
                "fork_index": fork_index, "analyzable": True, "contrastable": True,
                "reason": "", "n_free": n_free, "hamming_free": value,
                "hamming_editable": value / 2.0,
            })
    return pd.DataFrame(rows)


def test_the_sizing_formula_reproduces_the_reference_table():
    """sigma_d -> pairs, at delta = 0.02.  One-sample paired: no factor of 2."""
    from scripts.analysis.read_v2_mechanism import _Z_SUM_SQUARED

    import math
    for sigma, expected in [(0.03, 18), (0.04, 32), (0.05, 50), (0.06, 71), (0.08, 126)]:
        assert math.ceil(_Z_SUM_SQUARED * sigma**2 / 0.02**2) == expected


def test_forks_are_averaged_within_a_prefix_before_prefixes_are_counted():
    """The clustering correction.  Four forks of one prefix are ONE observation, not four."""
    from scripts.analysis.read_v2_mechanism import prefix_means, summarize

    frame = _frame({0: [0.00, 0.10, 0.20, 0.30], 1: [0.15, 0.15, 0.15, 0.15]})
    means = prefix_means(frame, column="hamming_free")
    assert list(means) == pytest.approx([0.15, 0.15])
    # Both prefixes average to the same value, so the between-prefix spread is zero even though
    # the raw fork values range over 0.00-0.30.  Counting forks would report spread that is not
    # there at the level the sample is drawn at.
    assert summarize(frame, column="hamming_free", delta=0.036, floor=32,
                     cap=128)["sd"] == pytest.approx(0.0)


def test_a_structural_zero_is_excluded_and_counted():
    """Two arms handed byte-identical inputs cannot inform a transmission claim."""
    from scripts.analysis.read_v2_mechanism import read_mechanism

    frame = _frame({0: [0.05], 1: [0.05], 2: [0.0]})
    frame.loc[frame["source_index"] == 2, "contrastable"] = False
    report = read_mechanism(frame, delta=0.036, floor=32, cap=128)
    block = report["proteins"]["5ZHV_B"]["endpoint_change"]
    assert block["n_prefixes_seen"] == 3
    assert block["n_prefixes_scored"] == 2
    assert block["contrastable_fraction"] == pytest.approx(2 / 3)


def test_the_pilot_emits_no_verdict_but_does_emit_a_pair_count():
    from scripts.analysis.read_v2_mechanism import read_mechanism, required_pairs

    frame = _frame({i: [0.04 + 0.01 * (i % 3)] for i in range(16)})
    report = read_mechanism(frame, delta=0.036, floor=32, cap=128)
    assert "verdict" not in report
    sizing = required_pairs(report)
    assert sizing["resolved"] is True
    assert sizing["sized_on_view"] == "endpoint_change"
    assert sizing["n_pairs_required"] >= 32          # the floor always applies


def test_sizing_is_driven_by_the_primary_not_by_the_positive_control():
    """`source_shuffle` perturbs tens of tokens; its spread must not size the one-token test."""
    from scripts.analysis.read_v2_mechanism import read_mechanism, required_pairs

    primary = _frame({i: [0.04] if i % 2 else [0.045] for i in range(12)})
    control = _frame({i: [0.10 * (1 + i)] for i in range(12)}, view="source_shuffle")
    report = read_mechanism(pd.concat([primary, control], ignore_index=True),
                            delta=0.036, floor=32, cap=128)
    sizing = required_pairs(report)
    assert sizing["sized_on_view"] == "endpoint_change"
    assert sizing["sigma_d"] == pytest.approx(
        report["proteins"]["5ZHV_B"]["endpoint_change"]["free"]["sd"])


def test_a_spread_past_the_cap_is_declared_unresolved_rather_than_shrunk():
    from scripts.analysis.read_v2_mechanism import read_mechanism, required_pairs

    frame = _frame({i: [0.5 * (i % 2)] for i in range(12)})     # sd ~= 0.26
    sizing = required_pairs(read_mechanism(frame, delta=0.036, floor=32, cap=128))
    assert sizing["verdict"] == "underpowered_unresolved"
    assert sizing["n_pairs_required"] == 128


def test_the_confirmatory_gate_is_a_margin_test_not_a_point_estimate():
    """A sample mean above delta whose interval does not clear it must NOT pass."""
    from scripts.analysis.read_v2_mechanism import confirmatory_verdict, read_mechanism

    # Mean 0.05 > delta 0.036, but the spread is wide enough that the CI lower bound is below it.
    primary = _frame({i: [0.05 + (0.06 if i % 2 else -0.06)] for i in range(8)})
    control = _frame({i: [0.30] for i in range(8)}, view="source_shuffle")
    report = read_mechanism(pd.concat([primary, control], ignore_index=True),
                            delta=0.036, floor=32, cap=128)
    block = report["proteins"]["5ZHV_B"]["endpoint_change"]["free"]
    assert block["mean"] > 0.036 and block["ci95_lower"] < 0.036
    verdict = confirmatory_verdict(report, delta=0.036)
    assert verdict["passed"] is False
    assert verdict["reading"] == "transmission_not_demonstrated"


def test_a_silent_positive_control_reads_as_assay_failure_not_as_no_transmission():
    """If the readout cannot see a many-token perturbation, a one-token zero means nothing."""
    from scripts.analysis.read_v2_mechanism import confirmatory_verdict, read_mechanism

    primary = _frame({i: [0.0] for i in range(8)})
    control = _frame({i: [0.0] for i in range(8)}, view="source_shuffle")
    report = read_mechanism(pd.concat([primary, control], ignore_index=True),
                            delta=0.036, floor=32, cap=128)
    verdict = confirmatory_verdict(report, delta=0.036)
    assert verdict["reading"] == "assay_failure"


def test_a_clean_pass_needs_both_the_margin_and_a_live_control():
    from scripts.analysis.read_v2_mechanism import confirmatory_verdict, read_mechanism

    primary = _frame({i: [0.12 + 0.005 * (i % 3)] for i in range(12)})
    control = _frame({i: [0.30 + 0.01 * (i % 3)] for i in range(12)}, view="source_shuffle")
    report = read_mechanism(pd.concat([primary, control], ignore_index=True),
                            delta=0.036, floor=32, cap=128)
    verdict = confirmatory_verdict(report, delta=0.036)
    assert verdict["passed"] is True
    assert verdict["reading"] == "transmission_demonstrated"


def test_a_null_arm_carries_the_cycles_own_reason():
    """The commonest null is `endpoint_rank=1` on a one-endpoint pool -- a coverage fact.

    Reporting it as a bare "no projection" would make it indistinguishable from a band refusal or
    a policy rejection, and those call for opposite next actions.
    """
    projected = F.projected()
    dead = types.SimpleNamespace(
        projected=None, descendant_endpoints=(),
        outcome=types.SimpleNamespace(value="null_no_admissible_endpoint"),
        detail="endpoint_rank=1 but only 1 admissible endpoint(s) exist")
    row, = _rows(_arm(projected, [_descendant(projected)]), dead)
    assert row["analyzable"] is False
    assert "null_no_admissible_endpoint" in row["reason"]
    assert "only 1 admissible" in row["reason"]


# --------------------------------------------------------------------------------------------
# the kwargs contract, read out of production source
# --------------------------------------------------------------------------------------------


def _call_keywords(path, callee):
    """Keyword names supplied at every call site of `callee`, including `dict(**base, k=v)` splats."""
    import ast

    tree = ast.parse(__import__("pathlib").Path(path).read_text())
    names, seen = set(), False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == callee):
            continue
        seen = True
        names |= {kw.arg for kw in node.keywords if kw.arg}
        for kw in node.keywords:
            if kw.arg is None and isinstance(kw.value, ast.Call) and \
                    getattr(kw.value.func, "id", "") == "dict":
                names |= {inner.arg for inner in kw.value.keywords if inner.arg}
    assert seen, f"no call to {callee}() found in {path}"
    return names


def _dict_literal_keys(path):
    """Every `dict(..., k=v)` key in a module -- how the paired executor builds its arm kwargs."""
    import ast

    tree = ast.parse(__import__("pathlib").Path(path).read_text())
    return {kw.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "dict"
            for kw in node.keywords if kw.arg}


def _popped_from_cycle_kwargs(path):
    """Names the ladder pops out of `cycle_kwargs` -- i.e. what the ORACLE FACTORY provides."""
    import ast

    tree = ast.parse(__import__("pathlib").Path(path).read_text())
    return {node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "pop"
            and getattr(node.func.value, "id", "") == "cycle_kwargs"
            and node.args and isinstance(node.args[0], ast.Constant)}


LADDER = "inverse_folding/reference_flow/fusion_v2_runtime/ladder.py"
PAIRED = "inverse_folding/reference_flow/fusion_v2_runtime/paired.py"
COHORT = "scripts/rf_fusion_v2_cohort.py"


def test_the_mechanism_shard_supplies_every_kwarg_the_ladder_does():
    """The defect this exists for cost a GPU allocation to discover.

    `run_one_cycle` takes 31 required keyword-only arguments.  The oracle factory supplies most of
    them and the LADDER supplies the rest at its own call site, so a second caller that only
    forwards `cycle_kwargs` is missing exactly the ladder's extras.  Locally invisible, because the
    paired-executor tests pass every argument explicitly; on the cluster it is a TypeError after
    both checkpoints are resident (measured: job 12098131, `origin_transition_id`).

    Every term is read from production source, so the check cannot go stale: what the ladder
    passes, what the paired executor adds per arm, and what the factory provides -- the last being
    exactly the names the ladder pops back out of `cycle_kwargs`.
    """
    required = _call_keywords(LADDER, "run_one_cycle")
    # BOTH runners plus the shared ``dict(...)`` literal the shard now builds once and spreads into
    # each of them.  Reading only one call site would let a name move into the literal -- or into
    # the qualification branch alone -- and silently stop being checked.
    shard = (_call_keywords(COHORT, "run_mechanism_views")
             | _call_keywords(COHORT, "run_policy_qualification_view")
             | _dict_literal_keys(COHORT))
    executor = _dict_literal_keys(PAIRED)
    from_factory = _popped_from_cycle_kwargs(LADDER)

    assert "lineage" in from_factory, "the ladder no longer reads lineage off the factory"
    missing = required - shard - executor - from_factory
    assert not missing, (
        f"the mechanism shard forwards cycle_kwargs but never supplies {sorted(missing)}, "
        "which the ladder passes explicitly and run_one_cycle requires"
    )


def test_the_executor_credit_is_real_and_not_a_blanket_exemption():
    """The exemption above is only sound if the executor genuinely sets these per arm."""
    executor = _dict_literal_keys(PAIRED)
    for name in ("descendant_propagation_seed", "descendant_fork_seeds", "source_override",
                 "source_endpoints", "archive", "feedback_enabled"):
        assert name in executor, f"run_mechanism_views no longer supplies {name}"


def test_the_shard_holds_rows_not_archives():
    """A 16-prefix run forks tens of archives; retaining them peaked ~39 GiB on ONE prefix."""
    source = (__import__("pathlib").Path("scripts/rf_fusion_v2_cohort.py")).read_text()
    body = source.split("def run_v2_mechanism_shard")[1]
    assert "archives.append" not in body
    assert "archive_rows_by_id.setdefault" in body


def test_batch_two_continues_the_prefix_sequence_instead_of_repeating_it():
    """Prefix seeds are derived from the INDEX, so restarting at 0 reproduces batch 1 exactly.

    Worse than the wasted compute: the two bundles would then collide on `source_index`, and the
    reader groups on it -- two different prefixes would be averaged into one sampling unit.
    """
    from scripts.run_rf_fusion_v2 import select_runner

    def mechanism(**kw):
        return kw

    chosen = select_runner(
        types.SimpleNamespace(mechanism_prefixes=31, mechanism_prefix_start=16),
        ladder="LADDER", mechanism=mechanism)
    assert chosen() == {"n_prefixes": 31, "prefix_start": 16, "qualification": False}


def test_a_prefix_start_without_a_mechanism_run_is_refused():
    """It would silently do nothing, and reading the flag back would suggest it had."""
    from scripts.run_rf_fusion_v2 import V2DriverError, select_runner

    with pytest.raises(V2DriverError):
        select_runner(types.SimpleNamespace(mechanism_prefixes=0, mechanism_prefix_start=16),
                      ladder="LADDER", mechanism="MECH")


def test_the_shard_loops_the_offset_range():
    source = (__import__("pathlib").Path("scripts/rf_fusion_v2_cohort.py")).read_text()
    assert "range(prefix_start, prefix_start + n_prefixes)" in source


# --------------------------------------------------------------------------------------------
# merging batches is a provenance claim
# --------------------------------------------------------------------------------------------


def _bundle(tmp_path, name, frame, *, code_revision="rev1"):
    d = tmp_path / name
    d.mkdir()
    frame.to_parquet(d / "mechanism_contrasts.parquet")
    (d / "run_manifest.json").write_text(json.dumps(
        {"code_revision": code_revision, "config_digest": "cfg-" + code_revision}))
    return d


def test_a_prefix_index_reused_across_batches_is_refused(tmp_path):
    """The reader groups on source_index; a collision averages two prefixes into one unit."""
    from scripts.analysis.read_v2_mechanism import load_contrasts

    b1 = _bundle(tmp_path, "b1", _frame({0: [0.05], 1: [0.05]}))
    b2 = _bundle(tmp_path, "b2", _frame({1: [0.09], 2: [0.09]}))   # 1 overlaps
    with pytest.raises(SystemExit) as exc:
        load_contrasts([str(b1), str(b2)])
    assert "5ZHV_B:1" in str(exc.value)


def test_disjoint_batches_merge_into_one_sample(tmp_path):
    from scripts.analysis.read_v2_mechanism import load_contrasts, prefix_means

    b1 = _bundle(tmp_path, "b1", _frame({0: [0.05], 1: [0.05]}))
    b2 = _bundle(tmp_path, "b2", _frame({16: [0.09], 17: [0.09]}))
    merged = load_contrasts([str(b1), str(b2)])
    assert prefix_means(merged, column="hamming_free").size == 4


def test_a_merge_across_code_revisions_is_visible_not_silent(tmp_path):
    """Legitimate only when the generative path is the same -- which must be stated, not assumed."""
    from scripts.analysis.read_v2_mechanism import load_contrasts, provenance

    b1 = _bundle(tmp_path, "b1", _frame({0: [0.05]}), code_revision="aaa")
    b2 = _bundle(tmp_path, "b2", _frame({16: [0.05]}), code_revision="bbb")
    prov = provenance(load_contrasts([str(b1), str(b2)]))
    assert prov["single_code_revision"] is False
    assert prov["code_revisions"] == ["aaa", "bbb"]

    same = provenance(load_contrasts([str(b1)]))
    assert same["single_code_revision"] is True


def test_a_single_unscorable_prefix_does_not_change_how_the_rest_are_counted():
    """One null row turns `contrastable` from bool into OBJECT dtype across a batch merge.

    Object masks still index correctly, but `~mask` on them is integer negation -- it yields
    -1/-2 rather than the complement.  Measured on the real merged 5ZHV_B sample, where two rows
    are typed nulls from a band refusal.  The counts must not depend on which operation the code
    happens to reach for.
    """
    from scripts.analysis.read_v2_mechanism import _flag, read_mechanism

    clean = _frame({i: [0.05] for i in range(4)})
    dirty = pd.concat([clean, pd.DataFrame([{
        "protein_id": "5ZHV_B", "view": "endpoint_change", "source_index": 9, "fork_index": 0,
        "analyzable": None, "contrastable": None, "reason": "band refused",
        "n_free": None, "hamming_free": None, "hamming_editable": None,
    }])], ignore_index=True)
    assert dirty["contrastable"].dtype == object, "the fixture must reproduce the dtype collapse"

    mask = _flag(dirty, "contrastable")
    assert mask.dtype == bool
    assert list(~mask) == [False, False, False, False, True]      # a real complement

    clean_block = read_mechanism(clean, delta=0.036, floor=32,
                                 cap=128)["proteins"]["5ZHV_B"]["endpoint_change"]
    dirty_block = read_mechanism(dirty, delta=0.036, floor=32,
                                 cap=128)["proteins"]["5ZHV_B"]["endpoint_change"]
    assert dirty_block["n_prefixes_scored"] == clean_block["n_prefixes_scored"] == 4
    assert dirty_block["free"]["mean"] == pytest.approx(clean_block["free"]["mean"])
    assert dirty_block["unanalyzable_reasons"] == {"band refused": 1}
