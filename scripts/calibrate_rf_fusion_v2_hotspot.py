"""Produce the V2 whole-landscape hotspot calibration artifact, PER PROTEIN (runbook §2.1).

**Scope, first, because the artifact is easy to over-read.**  This authorizes the WIRING of the
two-protein Canary only.  It is not a production immune-safety threshold and may not be carried into
a mechanism cohort, policy qualification, capability ladder, or holdout.

The calibration law is FROZEN by the runbook, not chosen here.  Implemented exactly as §2.1 states:

1. generate ``--n-completions`` complete trajectories with feedback disabled under the frozen V2
   substrate (controller-free, h-map-free, ``constant_one``, background remask ``0.0``), drawing
   from the disjoint seed namespace ``v2_hotspot_calibration_1``;
2. score every candidate AND that protein's own native reference with the production Head, and
   compute ``N_H^whole(y; ybar_p) = max_w [z_w(y) - z_w(ybar_p)]_+`` through the ADMISSION GATE's
   own comparator;
3. run the exact definitive structure gate the Canary uses and record its verdict and metrics in
   full -- as a DIAGNOSTIC.  It does **not** select the calibration population;
4. require at least ``--min-head-valid`` endpoints with a valid Head measurement, and below that
   floor write NO artifact;
5. take the empirical ``Q0.90`` over ALL head-valid endpoints by the **higher** order statistic --
   the value at one-indexed rank ``ceil(0.90*n)`` -- and preserve its full floating-point value.

**Why the population is head-valid rather than structure-feasible.**  The Head gate and the
structure gate answer independent safety questions: whether a sequence creates a new immune hotspot,
and whether it preserves the target backbone and active-site geometry.  Conditioning the Head null
distribution on the structure verdict couples two axes the science keeps apart, and it does so at
exactly the wrong moment -- a low structure-pass rate shrinks the sample the threshold is estimated
from, so the estimate is least stable precisely when structure is hardest.  The executed Canary made
this concrete: ``Q00511`` preserved hard anchors 64/64 and passed scTM 64/64, and was rejected only
because a predicted side-chain RMSD exceeded an absolute band whose own native baseline is 1.791 A.
Nothing in that verdict makes the sequence's Head hotspot unmeasurable.  The structure rate is
reported beside the threshold as ``structure_operability``, never inside it.

**Per protein, never pooled.**  Sequence length, window multiplicity and the anchor domain all
change the distribution of a whole-landscape maximum, so two proteins get two thresholds and two
artifacts.  There is no cohort-level maximum and no pooling.

**One source of truth for the statistic.**  ``--threshold-statistic`` is a closed enum with a single
admissible value; there is deliberately no ``--quantile`` or ``--quantile-method`` override, because
two ways to say the same thing is two ways for them to disagree.

Every model-touching step goes through the SHARED seams -- ``scripts.rf_fusion_model_factory`` for
the sampler/denoiser/backbone/anchors, the production Head batch scorer, and the v0 definitive
structure gate -- so the calibration measures the same objects the Canary will.

**Verification boundary.**  ``main`` cannot run in this repo: it needs torch, a DPLM checkpoint, a
Head checkpoint, a refold backend and PDBs.  The frozen LAW -- the order statistic, the floor, the
definitive-verdict rule, the artifact shape -- is unit-tested against injected seams; that the real
oracles behave is a cluster check.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion_v2.errors import V2Error  # noqa: E402

__all__ = [
    "V2HotspotCalibrationError",
    "THRESHOLD_STATISTIC",
    "THRESHOLD_STATISTIC_BY_POPULATION",
    "RESUMED_SAMPLING_UNIT",
    "Draw",
    "full_trajectory_draws",
    "resumed_draws",
    "source_variance",
    "SEED_NAMESPACE",
    "CalibrationSeams",
    "q90_higher_order_statistic",
    "source_id_for",
    "summarize",
    "build_parser",
    "main",
]

#: The ONE admissible value PER POPULATION.  A closed map rather than a free-form label: the
#: producer must interpret each exactly as runbook §2.1 step 5 and reject anything else.
#:
#: There are two because there are two populations, and the first Canary MEASURED that they differ.
#: `full_trajectory` draws one complete de novo trajectory per replicate; `resumed` captures a
#: source prefix at `c_source` and forks feedback-off completions from it -- which is what a
#: mechanism run's endpoints actually are. On the structure axis the two differ by ~0.05 scTM at the
#: median, so a threshold measured on one is not automatically a threshold for the other. The
#: statistic name carries the population so a config can never silently mix them.
THRESHOLD_STATISTIC_BY_POPULATION = {
    "full_trajectory": "per_protein_feedback_disabled_head_valid_q90_higher",
    "resumed": "per_protein_resumed_feedback_off_head_valid_q90_higher",
}

#: Back-compatible alias: the original frozen statistic, which is the `full_trajectory` one.
THRESHOLD_STATISTIC = THRESHOLD_STATISTIC_BY_POPULATION["full_trajectory"]

#: In the `resumed` population the SOURCE PREFIX is the sampling unit. Sixty-four siblings of one
#: prefix are one sample of the prefix distribution, not sixty-four samples: within a Canary cell
#: the four lookaheads share fifty committed steps, which is why "no admissible endpoint" behaved
#: like a per-cell coin flip rather than the 4th power of a per-endpoint rate.
RESUMED_SAMPLING_UNIT = "source_prefix"

#: Disjoint from every sampling namespace the Canary itself draws from, so calibration draws and
#: run draws can never collide (PLAN §2.6's exclusion law applies to this producer too).
SEED_NAMESPACE = "v2_hotspot_calibration_1"

#: The depth-0 reference the whole-landscape comparator measures against, spelled exactly as
#: ``config.safety.cumulative_reference_{kind,label}`` spell it.  The emitted block is pasted into
#: the config verbatim, so a producer that used its own wording would put two names for one
#: reference into a single resolved config.
CUMULATIVE_REFERENCE_KIND = "native_wt"
CUMULATIVE_REFERENCE_LABEL = "wt_native"


class V2HotspotCalibrationError(V2Error):
    """The calibration could not be produced under the frozen law."""


class V2HotspotFloorNotMet(V2HotspotCalibrationError):
    """Too few endpoints yielded a valid Head measurement.

    Carries the rows so the caller can still persist them.  §2.1's "write no calibration artifact"
    withholds the THRESHOLD, not the evidence: the raw table is the only place the head-valid rate,
    the structure-operability rate and the per-kind failure counts survive, and the floor is checked
    against exactly that first rate -- so discarding the table at the moment it fails is discarding
    the one record that says WHY.
    """

    def __init__(self, message: str, *, rows, failures) -> None:
        super().__init__(message)
        self.rows = list(rows)
        self.failures = dict(failures)


# --------------------------------------------------------------------------------------------
# the frozen statistic
# --------------------------------------------------------------------------------------------


def q90_higher_order_statistic(values: Sequence[float]) -> tuple[float, int]:
    """Empirical ``Q0.90`` by the HIGHER order statistic: rank ``ceil(0.90*n)``, one-indexed.

    Returns ``(value, rank)``.  Deliberately NOT ``numpy.quantile``: that interpolates linearly
    between order statistics and returns a number no endpoint actually produced.  A threshold is a
    bound on measured designs, so it has to BE one of them.

    The value is returned at full precision.  Rounding before ``source_ref`` is computed would make
    the digest describe a different number than the one enforced.
    """
    if not values:
        raise V2HotspotCalibrationError(
            "no retained endpoints: a threshold reduced from an empty distribution would be an "
            "invented number wearing a measurement's provenance"
        )
    ordered = sorted(float(v) for v in values)
    if not all(math.isfinite(v) for v in ordered):
        raise V2HotspotCalibrationError("a retained N_H measurement is not finite")
    rank = math.ceil(0.90 * len(ordered))
    rank = max(1, min(rank, len(ordered)))
    return ordered[rank - 1], rank


def source_id_for(protein_id: str, *, population: str = "full_trajectory") -> str:
    """Protein-specific because the threshold is, and POPULATION-specific because it is too.

    The two nulls measure different distributions of the same quantity, so a config that named only
    the protein could carry one population's threshold under the other's name and nothing would
    contradict it.
    """
    if population == "full_trajectory":
        return f"v2-canary-hotspot-head-valid-q90-higher-v1:{protein_id}"
    return f"v2-{population}-hotspot-head-valid-q90-higher-v1:{protein_id}"


def summarize(values: Sequence[float]) -> dict:
    """The distribution the scalar came from.  A producer that emitted only the chosen number would
    leave a reviewer unable to see whether it sat in a tail or in a cliff."""
    import numpy as np

    array = np.asarray(sorted(float(v) for v in values), dtype=np.float64)
    chosen, rank = q90_higher_order_statistic(array.tolist())
    return {
        "q50": float(np.quantile(array, 0.50)), "q90_higher": chosen,
        "q95": float(np.quantile(array, 0.95)), "max": float(array.max()),
        "order_statistic_rank": rank, "n_retained": int(array.size),
    }


# --------------------------------------------------------------------------------------------
# injectable seams
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationSeams:
    """Every model-touching entry point, injected so the frozen LAW is testable without a GPU."""

    build_model_factory: Any = None
    head_scorer: Any = None
    structure_gate: Any = None
    generate_completion: Any = None
    derive_seed: Any = None
    build_production_oracles: Any = None

    def resolved(self) -> dict:
        from inverse_folding.reference_flow.fusion_v2.seeds import derive_seed

        from scripts.rf_fusion_model_factory import build_model_factory
        from scripts.rf_fusion_v2_oracles import build_production_oracles

        return {
            "build_model_factory": self.build_model_factory or build_model_factory,
            # Left as ``None`` on purpose: ``main`` builds the REAL pair from the declared paths
            # when a caller injected neither.  Defaulting them here would need the CLI paths this
            # dataclass cannot see, and returning ``None`` unconditionally is what made the
            # producer die on ``NoneType.evaluator_identity`` after the DPLM checkpoint had already
            # been loaded onto the GPU.
            "head_scorer": self.head_scorer,
            "structure_gate": self.structure_gate,
            "generate_completion": self.generate_completion or _generate_completion,
            "derive_seed": self.derive_seed or derive_seed,
            "build_production_oracles": (
                self.build_production_oracles or build_production_oracles),
        }


def _oracle_request(protein_id: str, sequence: str):
    """The typed request BOTH oracles consume -- the same one the Canary's runtime builds.

    Not a bare ``(protein_id, sequence)`` pair: ``OracleRequest`` re-derives the digest from the
    bytes and refuses a non-canonical residue, so a masked or mis-keyed candidate cannot reach a
    backend that would map it to some arbitrary token.
    """
    from inverse_folding.reference_flow.fusion.state import sequence_md5
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest

    return OracleRequest(protein_id=str(protein_id), sequence=str(sequence),
                         sequence_md5=sequence_md5(sequence), sequence_length=len(sequence))


def _score_one(head_oracle: Any, protein_id: str, sequence: str):
    """One sequence through the SAME batch contract the Canary scores every endpoint through."""
    results = list(head_oracle.score([_oracle_request(protein_id, sequence)]))
    if len(results) != 1:
        raise V2HotspotCalibrationError(
            f"the Head returned {len(results)} result(s) for one request; the calibration cannot "
            "attribute a score it cannot match to the sequence it asked about"
        )
    return results[0]


def _generate_completion(*, model, protein_id: str, seed: int) -> str:
    """One complete trajectory with feedback disabled, under the frozen substrate.

    Feedback-disabled and complete by construction: this is the NULL distribution the threshold is
    measured against, so nothing here may consult a Head, a structure verdict or a projection.
    """
    import dataclasses

    from scripts.rf_fusion_model_factory import decode_tokens_to_aa

    prepared, denoiser = model.backbone_and_denoiser(protein_id)
    length = model.sequence_length(protein_id)
    config = dataclasses.replace(
        model.rf_config,
        sampler=dataclasses.replace(model.rf_config.sampler, seed=int(seed)))
    output = model.sampler.sample(
        sequence_length=length, h_values=model.null_h_values(length), denoiser=denoiser,
        config=config, controller=None, struct=None,
        residue_token_ids=model.aa_token_ids, fixed_tokens=model.fixed_tokens(protein_id),
    )
    del prepared
    # ``id_to_aa``, not ``alphabet``: that is the mapping ``PreparedModel`` actually carries, and
    # it is the same one the V1 entry path decodes through.
    return decode_tokens_to_aa(output.tokens, model.id_to_aa)


# --------------------------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------------------------


def _assert_declared_window_domain(score: Any, head_identity: Any, *, protein_id: str) -> None:
    """Check the DECLARED window domain against the one the Head actually emits, on row zero.

    The declared ``window_k_min/max`` are metadata: ``OnlineHeadScorer`` stores them and echoes
    them into its cache identity, but the window grid comes from the Head's own inference config,
    so passing a narrower domain does not narrow anything -- it only makes
    ``whole_landscape_new_hotspot`` reject every window outside it.  A mismatch is therefore not a
    flaky failure but a guaranteed 0-of-N, and without this it is discovered one endpoint at a time
    after a GPU allocation has been paid for.  Checked once, on the reference, before any
    completion is drawn.
    """
    observed = sorted({int(window.k) for window in getattr(score, "windows", ()) or ()})
    if not observed:
        raise V2HotspotCalibrationError(
            f"{protein_id}: the Head returned no windows for the reference; N_H^whole is a maximum "
            "over the window grid, and an empty grid makes it vacuous"
        )
    low, high = int(head_identity.window_k_min), int(head_identity.window_k_max)
    stray = [k for k in observed if not low <= k <= high]
    if stray:
        raise V2HotspotCalibrationError(
            f"{protein_id}: the run declares window domain [{low}, {high}] but the frozen Head "
            f"emits k in [{observed[0]}, {observed[-1]}] (outside: {stray}).  The declared bounds "
            "are metadata -- the grid comes from the Head's inference config -- so this would "
            "reject EVERY endpoint, not some of them.  Declare the domain the evaluator actually "
            "has, or change the evaluator; do not narrow the declaration and call the result a "
            "calibration over the narrower grid"
        )


def _definitive_feasible(outcome: Any) -> bool:
    """A REAL definitive-feasible verdict: ``evaluated`` and ``feasible`` must BOTH hold.

    Recorded as a structure-operability DIAGNOSTIC.  It no longer selects the calibration
    population -- see :func:`calibrate_protein`.
    """
    return bool(getattr(outcome, "evaluated", False)) and bool(getattr(outcome, "feasible", False))


# --------------------------------------------------------------------------------------------
# the two populations
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Draw:
    """One attempted endpoint: its labels, and either a sequence or the reason there is none.

    Generation is separated from scoring so the two POPULATIONS differ in exactly one place. The
    scoring law -- Head validity, the anchor check, `N_H^whole`, the structure verdict, the floor
    and the Q0.90 -- is then provably the same for both, rather than the same by inspection.
    """

    labels: dict
    sequence: str | None = None
    error: str | None = None


def full_trajectory_draws(*, protein_id, n_completions, master_seed, model, seams):
    """One complete de novo trajectory per replicate -- the original frozen law, unchanged."""
    draws = []
    for replicate in range(int(n_completions)):
        seed = int(seams["derive_seed"](
            SEED_NAMESPACE, protein_id, str(master_seed), str(replicate)))
        labels = {"replicate_index": replicate, "seed": seed,
                  "source_index": None, "fork_index": None, "source_state_id": None,
                  "source_unresolved_editable": None}
        try:
            draws.append(Draw(labels, sequence=seams["generate_completion"](
                model=model, protein_id=protein_id, seed=seed)))
        except Exception as exc:                                # noqa: BLE001 - recorded, not lost
            draws.append(Draw(labels, error=str(exc)[:200]))
    return draws


def resumed_draws(*, protein_id, n_sources, completions_per_source, c_source, master_seed,
                  cycle_kwargs, seams):
    """`n_sources` independent prefixes captured at `c_source`, each forked `k` ways.

    This is the population a mechanism run's depth-0 endpoints ACTUALLY come from, and it is built
    through the same two runtime functions the cycle uses -- `capture_depth_zero` and
    `generate_lookaheads` -- so it cannot drift into being a different generative process that
    merely resembles one.

    A capture that fails costs its whole fork group, and every lost endpoint is recorded as a draw
    with a reason rather than silently shrinking the sample.
    """
    from inverse_folding.reference_flow.fusion_v2_runtime.capture import capture_depth_zero
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import generate_lookaheads

    draws, replicate = [], 0
    for source_index in range(int(n_sources)):
        source_seed = int(seams["derive_seed"](
            SEED_NAMESPACE, protein_id, str(master_seed), "source", str(source_index)))
        fork_seeds = [int(seams["derive_seed"](
            SEED_NAMESPACE, protein_id, str(master_seed), "fork", str(source_index), str(k)))
            for k in range(int(completions_per_source))]
        # The SOURCE seed varies per prefix; the sampler config is otherwise the run's own.
        source_config = dataclasses.replace(
            cycle_kwargs["config"],
            sampler=dataclasses.replace(cycle_kwargs["config"].sampler, seed=source_seed))
        labels_base = {"source_index": source_index, "source_seed": source_seed}
        try:
            source = capture_depth_zero(
                sampler=cycle_kwargs["sampler"], denoiser=cycle_kwargs["denoiser"],
                config=source_config, sequence_length=cycle_kwargs["sequence_length"],
                h_values=cycle_kwargs["h_values"],
                residue_token_ids=cycle_kwargs["residue_token_ids"], at_step=int(c_source),
                fixed_tokens=cycle_kwargs["fixed_tokens"], lineage=cycle_kwargs["lineage"],
                mask_token_id=cycle_kwargs["mask_token_id"],
                aa_token_ids=cycle_kwargs["aa_token_ids"],
                conditioning=cycle_kwargs["conditioning"],
                safety_reference=cycle_kwargs["safety_reference"],
                cost_event_ids=(f"resumed-null:{protein_id}:source{source_index}",))
        except Exception as exc:                                # noqa: BLE001
            for k in range(int(completions_per_source)):
                draws.append(Draw({**labels_base, "replicate_index": replicate, "fork_index": k,
                                   "seed": fork_seeds[k], "source_state_id": None,
                                   "source_unresolved_editable": None},
                                  error=f"capture failed: {str(exc)[:180]}"))
                replicate += 1
            continue
        # ``realized_maturity``, not ``maturity``: a LivePartialState derives its maturity from its
        # own tokens and stores none. The continuation CHECKPOINT has a ``.maturity``, the adapted
        # state does not, and reading the wrong one costs a whole GPU allocation to discover.
        common = {**labels_base, "source_state_id": source.state_id,
                  "source_unresolved_editable":
                      int(source.realized_maturity.n_unresolved_editable)}
        try:
            completions = generate_lookaheads(
                source=source, sampler=cycle_kwargs["sampler"],
                denoiser=cycle_kwargs["denoiser"], config=cycle_kwargs["config"],
                h_values=cycle_kwargs["h_values"],
                residue_token_ids=cycle_kwargs["residue_token_ids"],
                fork_seeds=fork_seeds, alphabet=cycle_kwargs["alphabet"])
        except Exception as exc:                                # noqa: BLE001
            for k in range(int(completions_per_source)):
                draws.append(Draw({**common, "replicate_index": replicate, "fork_index": k,
                                   "seed": fork_seeds[k]},
                                  error=f"fork failed: {str(exc)[:180]}"))
                replicate += 1
            continue
        for k, completion in enumerate(completions):
            draws.append(Draw({**common, "replicate_index": replicate, "fork_index": k,
                               "seed": fork_seeds[k]}, sequence=completion.sequence))
            replicate += 1
    return draws


def source_variance(rows) -> dict:
    """Within-source vs between-source spread of `N_H^whole`, on the head-valid rows.

    The number that says whether 64 endpoints are 64 samples or 32. If between-source variance
    dominates, the effective sample size is the number of PREFIXES and any interval computed as
    though the endpoints were independent is too narrow.
    """
    import statistics

    groups: dict[Any, list[float]] = {}
    for row in rows:
        if row.get("status") != "head_valid" or row.get("source_index") is None:
            continue
        groups.setdefault(int(row["source_index"]), []).append(float(row["n_h_whole"]))
    usable = {key: values for key, values in groups.items() if len(values) >= 2}
    if len(groups) < 2:
        return {"n_sources": len(groups), "note": "fewer than two sources; not estimable"}
    means = [statistics.fmean(values) for values in groups.values()]
    between = statistics.variance(means) if len(means) >= 2 else 0.0
    within = (statistics.fmean([statistics.variance(v) for v in usable.values()])
              if usable else 0.0)
    total = between + within
    return {
        "n_sources": len(groups),
        "n_endpoints": sum(len(v) for v in groups.values()),
        "between_source_variance": float(between),
        "within_source_variance": float(within),
        # Intraclass correlation: 1.0 means every endpoint of a prefix is the same measurement.
        "icc_between_over_total": (float(between / total) if total > 0 else None),
        "effective_sample_size_note": (
            "if ICC is near 1 the effective n is the number of SOURCES, not endpoints"),
    }


def calibrate_protein(
    *, protein_id: str, min_head_valid: int,
    reference_sequence: str, reference_digest: str, head_identity: Any, model: Any, seams: dict,
    draws: Sequence["Draw"] | None = None,
    n_completions: int | None = None, master_seed: Any = None,
    population: str = "full_trajectory", native_structure: Any = None,
) -> tuple[list[dict], dict]:
    """Run the frozen law for ONE protein and return ``(rows, summary)``.

    **The calibration population is every HEAD-VALID endpoint, not the structure-feasible subset.**
    The two gates answer independent safety questions -- the Head asks whether a sequence creates a
    new immune hotspot, the structure gate asks whether it keeps the target backbone and active-site
    geometry -- and conditioning the Head null distribution on the structure verdict couples them
    for no reason the science requires.  ``Q00511`` is the case that makes it concrete: hard anchors
    preserved 64/64 and scTM passing 64/64, rejected only because a predicted side-chain RMSD
    exceeded an absolute band whose own native baseline is 1.791 A.  Nothing about that verdict
    makes the sequence's Head hotspot unmeasurable.

    The structure verdict and its metrics are still evaluated and recorded in full -- they are the
    structure-OPERABILITY diagnostic, reported beside the threshold and never filtering it.

    Raises rather than returning a partial artifact when the head-valid floor is not met: runbook
    §2.1 still says "write no calibration artifact and do not relax the floor".
    """
    from inverse_folding.reference_flow.fusion.state import sequence_md5
    from inverse_folding.reference_flow.fusion_v2.identity import window_grid_digest
    from inverse_folding.reference_flow.fusion_v2.safety import (
        ReferenceKind,
        whole_landscape_new_hotspot,
    )

    if draws is None:
        # The original law's convenience path: `full_trajectory` draws are fully determined by
        # (protein, master_seed, n_completions), so a caller that names those has named the
        # population. `resumed` has no such shorthand -- it needs a captured source stack -- and
        # must hand its draws in.
        if n_completions is None:
            raise V2HotspotCalibrationError(
                "pass either `draws` or `n_completions`; a calibration over an unnamed population "
                "is not a calibration")
        draws = full_trajectory_draws(
            protein_id=protein_id, n_completions=n_completions, master_seed=master_seed,
            model=model, seams=seams)
    draws = list(draws)

    reference_score = _score_one(seams["head_scorer"], protein_id, reference_sequence)
    _assert_declared_window_domain(reference_score, head_identity, protein_id=protein_id)
    binding_id = f"ref:calib:{protein_id}:{reference_digest[:12]}"

    rows: list[dict] = []
    failures: dict[str, int] = {}

    def _fail(kind: str) -> None:
        failures[kind] = failures.get(kind, 0) + 1

    for draw in draws:
        replicate = int(draw.labels["replicate_index"])
        row: dict[str, Any] = {
            "protein_id": protein_id, "population": population,
            **{key: draw.labels.get(key) for key in (
                "replicate_index", "seed", "source_index", "source_seed", "fork_index",
                "source_state_id", "source_unresolved_editable")},
            "reference_digest": reference_digest,
            "head_evaluator_digest": head_identity.digest(),
        }
        if draw.sequence is None:
            _fail("generation")
            rows.append({**row, "status": "generation_failed", "failure": draw.error})
            continue
        sequence = draw.sequence
        row["sequence_md5"] = sequence_md5(sequence)

        anchors = model.fixed_tokens(protein_id) or {}
        alphabet = model.id_to_aa
        anchor_ok = all(alphabet.get(int(token)) == sequence[int(position)]
                        for position, token in anchors.items())
        row["anchor_verdict"] = bool(anchor_ok)
        if not anchor_ok:
            _fail("anchor")
            rows.append({**row, "status": "anchor_mismatch"})
            continue

        try:
            design_score = _score_one(seams["head_scorer"], protein_id, sequence)
            evidence = whole_landscape_new_hotspot(
                design_score, reference_score, endpoint_id=f"endpoint:{protein_id}:{replicate}",
                head_identity=head_identity, reference_kind=ReferenceKind.CUMULATIVE_DEPTH0,
                reference_binding_id=binding_id)
        except Exception as exc:                                # noqa: BLE001
            _fail("head")
            rows.append({**row, "status": "head_failed", "failure": str(exc)[:200]})
            continue
        row["n_h_whole"] = float(evidence.max_increase)
        row["window_grid_digest"] = window_grid_digest(design_score.windows)

        # Evaluated and recorded, but NOT a filter: a structure rejection is a fact about geometry,
        # not a reason the Head measurement did not happen.
        outcome = seams["structure_gate"](_oracle_request(protein_id, sequence))
        row["structure_evaluated"] = bool(getattr(outcome, "evaluated", False))
        row["structure_feasible"] = bool(getattr(outcome, "feasible", False))
        row["structure_definitive_feasible"] = _definitive_feasible(outcome)
        row["structure_metrics_json"] = json.dumps(
            dict(getattr(outcome, "metrics", None) or {}), sort_keys=True)
        rows.append({**row, "status": "head_valid"})

    head_valid = [row["n_h_whole"] for row in rows if row["status"] == "head_valid"]
    if len(head_valid) < int(min_head_valid):
        by_status: dict[str, int] = {}
        for row in rows:
            by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        raise V2HotspotFloorNotMet(
            f"{protein_id}: only {len(head_valid)} of {len(draws)} endpoints yielded a valid "
            f"Head measurement, below the frozen floor of {min_head_valid}.  Runbook §2.1: write no "
            "calibration artifact and do not relax the floor -- a Q0.90 over a thin sample is close "
            "to its own second-largest value and is not a distribution.  "
            f"status counts: {dict(sorted(by_status.items()))}; failure counts: "
            f"{dict(sorted(failures.items()))}",
            rows=rows, failures=failures,
        )
    summary = summarize(head_valid)
    # The structure verdict is reported BESIDE the threshold, never inside it.  Keeping the rate
    # visible is what stops "the Head null was measured over every endpoint" from being read as
    # "structure was not checked" -- it was checked on all of them, and here is how it went.
    n_definitive = sum(1 for row in rows if row.get("structure_definitive_feasible"))
    summary.update({
        "population": population,
        "sampling_unit": (RESUMED_SAMPLING_UNIT if population == "resumed" else "trajectory"),
        "n_attempted": len(draws),
        "n_head_valid": len(head_valid),
        "failure_counts": dict(sorted(failures.items())),
        "structure_operability": {
            "n_definitive_feasible": n_definitive,
            "rate": (n_definitive / len(head_valid)) if head_valid else 0.0,
            "note": "DIAGNOSTIC ONLY -- an independent admission gate, not a filter on this "
                    "threshold's population",
        },
    })
    if population == "resumed":
        summary["source_variance"] = source_variance(rows)
        summary["mechanism_stage_structure"] = mechanism_stage_structure_gates(
            rows, native_structure=native_structure)
    return rows, summary


def _metric_values(rows, name: str) -> list[float]:
    values = []
    for row in rows:
        if row.get("status") != "head_valid":
            continue
        metrics = json.loads(row.get("structure_metrics_json") or "{}")
        if metrics.get(name) is not None:
            values.append(float(metrics[name]))
    return values


def mechanism_stage_structure_gates(rows, *, native_structure: Any = None) -> dict:
    """The mechanism-stage structure gates, measured on THIS null (owner's ruling, 2026-08-06).

    Two changes from the v0 contract, and one thing that does not change.

    The anchor gate stops being an absolute number. `Q00511`'s native scores 1.791 A under this
    same backend, so of a 2.0 A band only 0.209 A is anything but backend and rotamer error --
    the gate was mostly measuring the predictor. Admission moves to the native-relative excess
    `Delta_anchor(y) = RMSD_anchor(y) - RMSD_anchor(native)`, thresholded at the upper tail of the
    feedback-off null; raw absolute RMSD stays reported in full so nothing is hidden by the
    subtraction. The ordinary protein's `scTM_min` comes from the LOWER tail of the same null,
    because a floor calibrated on a population that folds better rejects most of this one.

    **Hard-anchor residue IDENTITY is untouched and remains an absolute hard gate.** This
    recalibrates a geometric tolerance, never the requirement that an anchored residue is the WT
    residue.

    These are MECHANISM-OPERABILITY gates for the transmission experiment only. They may not be
    carried into capability or holdout: a threshold set at the 10th percentile of a null is chosen
    so the experiment can run, not so the product is safe.
    """
    native_metrics = dict(getattr(native_structure, "metrics", None) or {})
    sctm = _metric_values(rows, "scTM")
    anchor = _metric_values(rows, "max_anchor_sidechain_RMSD")
    native_anchor = native_metrics.get("max_anchor_sidechain_RMSD")
    gates: dict[str, Any] = {
        "authority": "mechanism_operability_only -- NOT capability, NOT holdout",
        "hard_anchor_identity": "absolute hard gate, unchanged and not recalibrated here",
        "native_metrics": {k: float(v) for k, v in native_metrics.items()},
        "scTM": ({"n": len(sctm), "q10_lower": _quantile_lower(sctm, 0.10),
                  "q50": _quantile_lower(sctm, 0.50), "min": min(sctm), "max": max(sctm),
                  "proposed_scTM_min": _quantile_lower(sctm, 0.10)} if sctm else None),
    }
    if anchor and native_anchor is not None:
        excess = [value - float(native_anchor) for value in anchor]
        gates["delta_anchor"] = {
            "definition": "RMSD_anchor(y) - RMSD_anchor(native), same backend and protocol",
            "native_absolute": float(native_anchor),
            "n": len(excess),
            "q50": _quantile_lower(excess, 0.50),
            "q90_higher": q90_higher_order_statistic(excess)[0],
            "max": max(excess),
            "absolute_rmsd": {"q50": _quantile_lower(anchor, 0.50), "min": min(anchor),
                              "max": max(anchor)},
            "proposed_delta_anchor_max": q90_higher_order_statistic(excess)[0],
        }
    elif anchor:
        gates["delta_anchor"] = {
            "unmeasurable": "the native was not evaluated under this backend, so the "
                            "native-relative excess has no baseline; absolute RMSD only",
            "absolute_rmsd": {"n": len(anchor), "q50": _quantile_lower(anchor, 0.50),
                              "min": min(anchor), "max": max(anchor)},
        }
    return gates


def _quantile_lower(values: Sequence[float], level: float) -> float | None:
    """The order statistic at or BELOW `level` -- the mirror of the `higher` convention.

    A floor and a ceiling must round in opposite directions, or one of them silently admits a
    sample it was meant to exclude.
    """
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    rank = max(1, math.floor(level * len(ordered)))
    return float(ordered[rank - 1])


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate_rf_fusion_v2_hotspot",
        description="Per-protein V2 whole-landscape hotspot calibration (runbook §2.1)",
    )
    parser.add_argument("--protein-id", required=True)
    parser.add_argument("--n-completions", type=int, required=True)
    parser.add_argument("--master-seed", type=int, required=True)
    parser.add_argument("--seed-namespace", required=True, choices=(SEED_NAMESPACE,),
                        help="disjoint from every namespace the Canary itself draws from")
    parser.add_argument("--threshold-statistic", required=True,
                        choices=tuple(sorted(THRESHOLD_STATISTIC_BY_POPULATION.values())),
                        help="a closed enum, not a label: the producer interprets it exactly as "
                             "runbook §2.1 step 5 and rejects any other value.  There is "
                             "deliberately no --quantile override -- two ways to say this is two "
                             "ways for them to disagree")
    parser.add_argument("--min-head-valid", type=int, required=True,
                        help="floor on endpoints yielding a VALID HEAD MEASUREMENT. Not a structure\n"
                             "floor: the structure verdict is recorded in full but does not select\n"
                             "this threshold's population -- the two gates answer independent safety\n"
                             "questions and conditioning one on the other couples them for no reason"),
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rf-config", required=True)
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--pdb-root", required=True)
    parser.add_argument("--complete-reference-manifest", required=True)
    parser.add_argument("--head-config-dir", required=True,
                        help="directory holding model.yaml / model_ablation.yaml / inference.yaml")
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--structure-config", required=True,
                        help="the v0 Fusion config defining the DEFINITIVE structure contract "
                             "(backend, scTM_min, active-site metric and thresholds). Not a PDB "
                             "root and not a refold checkpoint; the Canary must name the same file")
    # --- the Head DOMAIN -----------------------------------------------------------------------
    # Required, and not defaulted, because these four fields ARE the artifact: the emitted
    # HotspotCalibrationArtifact records them, and ``bind_admission_policy`` refuses a run whose
    # ``config.head`` 4-tuple differs.  They must be transcribed from the config that will consume
    # this threshold -- N_H^whole is a maximum over the window grid they define, so a threshold
    # measured under one domain does not bound designs scored under another.
    parser.add_argument("--allele", required=True,
                        help="must equal config.head.allele VERBATIM")
    parser.add_argument("--score-scale", required=True,
                        help="must equal config.head.score_scale; it is the threshold's unit")
    parser.add_argument("--window-k-min", type=int, required=True)
    parser.add_argument("--window-k-max", type=int, required=True)
    parser.add_argument("--head-variant-id", required=True)
    parser.add_argument("--head-allele-idx", type=int, default=0)
    parser.add_argument("--head-window-batch-size", type=int, default=64)
    parser.add_argument("--refold-cache-dir", required=True,
                        help="the v0 on-disk refold cache identity the definitive gate folds into")
    for name in ("--esmfold2-site-packages", "--esmfold2-model"):
        parser.add_argument(name, default=None)
    for name in ("--esmfold2-num-loops", "--esmfold2-num-sampling-steps",
                 "--esmfold2-num-diffusion-samples", "--esmfold2-seed"):
        parser.add_argument(name, type=int, default=None)
    parser.add_argument("--constraint-manifest", default=None,
                        help="omit for an unconstrained protein; supply the exact manifest for an "
                             "anchored one")
    parser.add_argument("--device", default="cuda")
    # The population is a SCIENTIFIC choice with no default: `full_trajectory` is the original
    # frozen law, `resumed` is the population a mechanism run's endpoints actually come from, and
    # the first Canary measured that the two differ.
    parser.add_argument("--population", required=True,
                        choices=sorted(THRESHOLD_STATISTIC_BY_POPULATION),
                        help="which null to measure; the statistic name carries it")
    parser.add_argument("--v2-config", default=None,
                        help="resumed only: the resolved cell config the oracle stack is built "
                             "from, so the null runs on the declared substrate")
    parser.add_argument("--shard-input", nargs="*", default=(), metavar="NAME=PATH",
                        help="resumed only: the same shard inputs the Canary launcher passes")
    parser.add_argument("--c-source", type=int, default=None,
                        help="resumed only: the step each source prefix is captured at")
    parser.add_argument("--n-sources", type=int, default=None,
                        help="resumed only: INDEPENDENT source prefixes -- the sampling unit")
    parser.add_argument("--completions-per-source", type=int, default=None,
                        help="resumed only: feedback-off completions forked from each prefix")
    parser.add_argument("--out-rows", required=True, help="canonical raw table (parquet)")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--code-revision", required=True)
    return parser


class _ModelView:
    """The two things `calibrate_protein` reads off a model, taken from the cycle kwargs.

    Not a general model: `resumed` generation happens through `build_v2_oracles`, which already
    built one, and the anchor check needs only the constraint class and the residue map. Building a
    second `PreparedModel` to supply them would put two DPLM checkpoints on one allocation.
    """

    def __init__(self, cycle_kwargs: dict) -> None:
        self.id_to_aa = cycle_kwargs["alphabet"]
        self._fixed = cycle_kwargs["fixed_tokens"]

    def fixed_tokens(self, protein_id: str):        # noqa: ARG002 - one protein per invocation
        return self._fixed


def main(argv=None, *, seams: CalibrationSeams | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Argument-only checks FIRST, before a path is opened or a model is touched.  A gate that
    # refuses after the oracles are built has already spent the allocation it exists to protect.
    expected_statistic = THRESHOLD_STATISTIC_BY_POPULATION[args.population]
    if args.threshold_statistic != expected_statistic:
        print(f"--threshold-statistic {args.threshold_statistic!r} does not name the "
              f"--population {args.population!r} law, which is {expected_statistic!r}; two "
              "independently settable names for one choice is how a resumed threshold ends up "
              "wearing the full-trajectory label", file=sys.stderr)
        return 2
    if args.population == "resumed":
        missing = [name for name, value in (
            ("--v2-config", args.v2_config), ("--c-source", args.c_source),
            ("--n-sources", args.n_sources),
            ("--completions-per-source", args.completions_per_source)) if value is None]
        if missing or not args.shard_input:
            print(f"the resumed population needs {missing or ['--shard-input']}: it is generated "
                  "through the cycle's own capture and fork, which need the declared substrate",
                  file=sys.stderr)
            return 2

    resolved = (seams or CalibrationSeams()).resolved()

    from inverse_folding.reference_flow.fusion_v2.config import (
        HOTSPOT_GATE_KIND,
        HOTSPOT_WINDOW_DOMAIN,
        V2_HOTSPOT_CALIBRATION_SCHEMA_VERSION,
        HotspotCalibrationArtifact,
        calibration_source_ref,
    )
    from inverse_folding.reference_flow.fusion_v2.identity import (
        HeadEvaluatorIdentity,
        canonical_digest,
    )
    from scripts.rf_fusion_v2_oracles import resolve_reference

    reference_sequence, reference_digest = resolve_reference(
        args.complete_reference_manifest, args.protein_id)

    # The Head and the definitive structure gate come from the DECLARED paths.  Built before the
    # sampler stack so a bad Head path fails in seconds instead of after DPLM is on the GPU.
    if resolved["head_scorer"] is None or resolved["structure_gate"] is None:
        esmfold2 = {name: getattr(args, name) for name in (
            "esmfold2_site_packages", "esmfold2_model", "esmfold2_num_loops",
            "esmfold2_num_sampling_steps", "esmfold2_num_diffusion_samples", "esmfold2_seed",
        ) if getattr(args, name) is not None}
        produced_head, produced_structure = resolved["build_production_oracles"](
            structure_config=args.structure_config, head_config_dir=args.head_config_dir,
            head_checkpoint=args.head_checkpoint, test_set_parquet=args.test_set,
            pdb_root=args.pdb_root, refold_cache_dir=args.refold_cache_dir,
            allele=args.allele, score_scale=args.score_scale,
            window_k_min=args.window_k_min, window_k_max=args.window_k_max,
            head_variant_id=args.head_variant_id, head_allele_idx=args.head_allele_idx,
            head_window_batch_size=args.head_window_batch_size,
            constraint_manifest=args.constraint_manifest, device=args.device,
            esmfold2=(esmfold2 or None),
        )
        resolved["head_scorer"] = resolved["head_scorer"] or produced_head
        resolved["structure_gate"] = resolved["structure_gate"] or produced_structure

    # In `resumed` mode the model comes from the ORACLE STACK, below, and is not built twice: two
    # `PreparedModel`s mean two DPLM checkpoints resident at once, and one already peaks at ~30 GB
    # RSS against a 40 GB request. `calibrate_protein` needs exactly two things off it, and the
    # cycle kwargs carry both.
    model = None if args.population == "resumed" else resolved["build_model_factory"](
        base_if_checkpoint=args.checkpoint, rf_sampler_config=args.rf_config,
        test_set_parquet=args.test_set, pdb_root=args.pdb_root, device=args.device,
        constraint_manifest=args.constraint_manifest,
    )
    scorer = resolved["head_scorer"]
    head_identity: HeadEvaluatorIdentity = scorer.evaluator_identity()

    native_structure = None
    if args.population == "resumed":
        # The resumed population is generated through the CYCLE's own two functions, reached via
        # the production oracle factory, so the null cannot drift into a generative process that
        # merely resembles the one a mechanism run uses.
        from scripts.rf_fusion_v2_cohort import ShardInputs
        from scripts.rf_fusion_v2_oracles import build_v2_oracles
        from scripts.rf_fusion_v2_preflight import load_v2_config_file

        shard_inputs = ShardInputs(**dict(
            pair.split("=", 1) for pair in args.shard_input))
        oracles = build_v2_oracles(
            protein_id=args.protein_id, config=load_v2_config_file(args.v2_config),
            inputs=shard_inputs)
        model = _ModelView(oracles["cycle_kwargs"])
        draws = resumed_draws(
            protein_id=args.protein_id, n_sources=args.n_sources,
            completions_per_source=args.completions_per_source, c_source=args.c_source,
            master_seed=args.master_seed, cycle_kwargs=oracles["cycle_kwargs"], seams=resolved)
        # The native under the SAME backend and protocol: without it the native-relative anchor
        # excess has no baseline, and an absolute band is exactly what the ruling replaced.
        native_structure = resolved["structure_gate"](
            _oracle_request(args.protein_id, reference_sequence))
    else:
        draws = full_trajectory_draws(
            protein_id=args.protein_id, n_completions=args.n_completions,
            master_seed=args.master_seed, model=model, seams=resolved)

    try:
        rows, summary = calibrate_protein(
            protein_id=args.protein_id, draws=draws,
            min_head_valid=args.min_head_valid,
            reference_sequence=reference_sequence, reference_digest=reference_digest,
            head_identity=head_identity, model=model, seams=resolved,
            population=args.population, native_structure=native_structure,
        )
    except V2HotspotFloorNotMet as exc:
        # The THRESHOLD is withheld; the EVIDENCE is not.  Without this the only record of why the
        # floor was missed is a one-line stderr message, and a below-floor run -- the case that most
        # needs diagnosing -- would be the one case that leaves nothing to diagnose.
        _write_rows(args.out_rows, exc.rows)
        print(f"calibration refused: {exc}", file=sys.stderr)
        print(f"[calibrate_rf_fusion_v2_hotspot] no artifact written; the {len(exc.rows)} attempted "
              f"row(s) are in {args.out_rows} so the failure is diagnosable", file=sys.stderr)
        return 2
    except V2HotspotCalibrationError as exc:
        print(f"calibration refused: {exc}", file=sys.stderr)
        return 2

    _write_rows(args.out_rows, rows)

    # Signed over the CANONICAL ROW PROJECTION, not the file: a re-serialization or a re-ordering
    # of the same measurements must not look like a different calibration.
    data_digest = canonical_digest({
        "schema": "v2-hotspot-calibration-rows/2",
        "statistic": THRESHOLD_STATISTIC,
        "rows": sorted(rows, key=lambda row: row["replicate_index"]),
    })
    # ``allele`` / ``score_scale`` / ``window_k_*`` come from the REALIZED Head identity, never
    # from a literal here: they are the domain the threshold was measured over, and
    # ``bind_admission_policy`` compares the same 4-tuple against ``config.head``.  In particular
    # ``score_scale`` is whatever the frozen evaluator actually emits (``raw_logit``) rather than a
    # label someone preferred -- an uncalibrated classifier logit is not a nat, and a threshold
    # wearing the wrong unit cannot be re-derived from the measurements that produced it.
    artifact = HotspotCalibrationArtifact(
        schema_version=V2_HOTSPOT_CALIBRATION_SCHEMA_VERSION, gate_kind=HOTSPOT_GATE_KIND,
        scope="cumulative_depth0", reference_kind=CUMULATIVE_REFERENCE_KIND,
        reference_label=CUMULATIVE_REFERENCE_LABEL,
        window_domain=HOTSPOT_WINDOW_DOMAIN, allele=head_identity.allele,
        score_scale=head_identity.score_scale, window_k_min=head_identity.window_k_min,
        window_k_max=head_identity.window_k_max, calibration_data_digest=data_digest,
    )
    value = summary["q90_higher"]
    statistic = THRESHOLD_STATISTIC_BY_POPULATION[args.population]
    source_id = source_id_for(args.protein_id, population=args.population)
    delta_new = {
        "value": value, "unit": head_identity.score_scale,
        "source_kind": "measured_calibration", "source_id": source_id,
        "source_ref": calibration_source_ref(
            value=value, unit=head_identity.score_scale, source_kind="measured_calibration",
            source_id=source_id, artifact=artifact),
        "artifact": artifact.canonical_payload(),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "delta_new": delta_new, "protein_id": args.protein_id,
        "code_revision": args.code_revision, "threshold_statistic": statistic,
        "seed_namespace": SEED_NAMESPACE, "master_seed": args.master_seed,
        "min_head_valid": args.min_head_valid,
        "rows_path": str(args.out_rows), **summary,
    }, indent=2, sort_keys=True))
    structure = summary["structure_operability"]
    print(f"[calibrate_rf_fusion_v2_hotspot] {args.protein_id}: "
          f"{summary['n_head_valid']}/{summary['n_attempted']} head-valid, "
          f"Q0.90(higher) at rank {summary['order_statistic_rank']} -> {value!r} -> {out}")
    print(f"[calibrate_rf_fusion_v2_hotspot] structure-operability DIAGNOSTIC (not a filter): "
          f"{structure['n_definitive_feasible']}/{summary['n_head_valid']} definitive feasible "
          f"({structure['rate']:.1%})")
    print("copy the 'delta_new' block into config.safety.delta_new_cumulative VERBATIM; the loader "
          "recomputes source_ref and refuses an edited one")
    return 0


def _write_rows(path: Any, rows: Sequence[dict]) -> None:
    """The canonical raw table, every attempt included.

    Failures are rows, not omissions: a table holding only the retained endpoints would make the
    definitive-feasible rate unrecoverable, and that rate is what the floor is checked against.
    """
    import pandas as pd

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows)).sort_values("replicate_index").reset_index(drop=True)
    frame.to_parquet(target, index=False)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
