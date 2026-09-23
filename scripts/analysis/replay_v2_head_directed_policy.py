"""Offline replay: would the Head-directed capped policy FIRE on an existing V2 cohort?

PLAN §8.4 puts this before any descendant is generated: "the offline replay must establish that the
frozen cohort contains enough legal positive local writes and exact band-target transitions across
the realized capped cardinalities to support the planned comparison.  A low-coverage result returns
to policy design; it may not be rescued by substituting arbitrary support or treating the cap as a
quota."

**What it is.**  An executability and coverage gate, run on the S7 bundles.  For every source state
in the bundle it replays :class:`~fusion_v2.policy.HeadDirectedCappedPolicy` -- literally the same
``select`` code path a live cycle runs, over a
:class:`~fusion_v2.policy.SourceView` rebuilt from the recorded per-position vectors -- and reports:

* the fraction of sources with at least one legal write candidate, and with at least one POSITIVE
  frozen-Head contribution;
* the realized ``m_d`` distribution and what bound it (positives, the cap, or the band);
* the contribution distribution;
* the required and legal reopen counts, and whether the band target is exactly reproducible; and
* the overlap between the Head-directed writes and the diagnostic write S7 actually made.

**What it is not.**  It generates no descendants, loads no denoiser and no structure backend, and
says nothing about whether a positive local dose shifts the descendant Head distribution.  Only the
matched one-cycle cluster comparison can answer that.

**The frozen Head is required and explicit.**  This qualification gate must measure the exact
leave-one-out contribution that defines a positive local dose; candidate geometry alone cannot
qualify the policy.  The replay loads ONLY the Head -- never the denoiser or structure backend --
and scores the counterfactuals exactly as the live policy scores them.

Cluster paths are CLI arguments.  Exit ``0`` when the replay completes, non-zero otherwise.
Registered in ``doc/SCRIPTS.md``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverse_folding.reference_flow.fusion.state import sequence_md5  # noqa: E402
from inverse_folding.reference_flow.fusion_v2 import identity as ident  # noqa: E402
from inverse_folding.reference_flow.fusion_v2 import policy as pol  # noqa: E402
from inverse_folding.reference_flow.fusion_v2 import reward as rw  # noqa: E402
from inverse_folding.reference_flow.fusion_v2.config import load_v2_config  # noqa: E402
from inverse_folding.reference_flow.fusion_v2.schedule import load_band_table  # noqa: E402
from inverse_folding.reference_flow.fusion_v2.state import FeasibilityLevel  # noqa: E402

__all__ = ["ReplayDonor", "source_views", "replay_bundle", "main"]


# --------------------------------------------------------------------------------------------
# the recorded run, read back as the law's inputs
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Window:
    start_0b: int
    end_0b: int
    k: int
    z: float


@dataclass(frozen=True)
class _Score:
    """A recorded Head score, rebuilt from ``complete_endpoints.head_score_json``.

    The window grid is the load-bearing part: PLAN §2.5 requires cross-sequence local evidence to
    come from RAW aligned windows, and those are exactly what the bundle persisted.  The residue
    hotspot column is deliberately not read -- see ``fusion_v2.evidence``.
    """

    protein_id: str
    sequence_md5: str
    sequence_length: int
    allele: str
    score_scale: str
    windows: tuple[_Window, ...]
    global_risk: float


@dataclass(frozen=True)
class ReplayDonor:
    """The donor as the law sees it: sequence, exact Head evidence, feasibility, identity.

    Duck-typed against what ``donor_gate`` and the write selector read off a
    :class:`~fusion_v2.state.CompleteEndpoint`.  Rebuilding a full endpoint offline would need the
    source state's replay identity and conditioning, which the bundle records only as digests --
    and none of it is input to the support law.
    """

    endpoint_id: str
    protein_id: str
    sequence: str
    sequence_md5: str
    sequence_length: int
    head_global_risk: float
    head_score: _Score
    head_binding: Any
    feasibility_level: FeasibilityLevel


def _score_from_json(payload: str, *, evaluator) -> _Score:
    data = json.loads(payload)
    return _Score(
        protein_id=str(data["protein_id"]), sequence_md5=str(data["sequence_md5"]),
        sequence_length=int(data["sequence_length"]), allele=str(data["allele"]),
        score_scale=str(data["score_scale"]),
        windows=tuple(
            _Window(start_0b=int(row["start_0b"]), end_0b=int(row["end_0b"]), k=int(row["k"]),
                    z=float(row["z"]))
            for row in data["windows"]
        ),
        global_risk=float(data["global_risk"]),
    )


def source_views(states: pd.DataFrame) -> dict[str, pol.SourceView]:
    """Rebuild one :class:`~fusion_v2.policy.SourceView` per recorded LIVE partial state.

    Only ``layer == 'live'`` rows: a projected state is the OUTPUT of a transition, and replaying
    the policy on one would ask what the law would do to its own result.
    """
    views: dict[str, pol.SourceView] = {}
    for _, row in states[states["layer"] == "live"].iterrows():
        tokens = json.loads(row["tokens_json"])
        editable = tuple(int(p) for p in json.loads(row["editable_positions_json"]))
        mask_token = int(row["mask_token_id"])
        commits = json.loads(row["active_commit_json"])
        scores = json.loads(row["active_sampler_score_json"])
        provenance = json.loads(row["provenance_json"])
        views[str(row["state_id"])] = pol.SourceView(
            editable_positions=editable,
            masked_positions=tuple(p for p in editable if int(tokens[p]) == mask_token),
            resolved_positions=tuple(p for p in editable if int(tokens[p]) != mask_token),
            commit_step_by_pos={
                p: (None if commits[p] is None else int(commits[p][1])) for p in editable},
            active_sampler_score_by_pos={
                p: (None if scores[p] is None else float(scores[p])) for p in editable},
            n_origin_events_by_pos={
                p: int(provenance[p]["n_origin_events"]) for p in editable},
        )
    return views


def _donors(endpoints: pd.DataFrame, *, evaluator, binding_of) -> dict[str, list[ReplayDonor]]:
    """Definitively feasible endpoints per source state, best (lowest Head risk) first.

    Ranked exactly as ``select_family_representatives`` ranks them, so "the donor" here is the
    donor the live runner would have selected at ``endpoint_rank=0``.
    """
    out: dict[str, list[ReplayDonor]] = {}
    for _, row in endpoints.iterrows():
        if str(row["feasibility_level"]).lower() != "definitive":
            continue
        score = _score_from_json(row["head_score_json"], evaluator=evaluator)
        out.setdefault(str(row["source_state_id"]), []).append(ReplayDonor(
            endpoint_id=str(row["endpoint_id"]), protein_id=str(row["protein_id"]),
            sequence=str(row["sequence"]), sequence_md5=str(row["sequence_md5"]),
            sequence_length=int(row["sequence_length"]),
            head_global_risk=float(row["head_global_risk"]), head_score=score,
            head_binding=binding_of(score), feasibility_level=FeasibilityLevel.DEFINITIVE,
        ))
    for state_id in out:
        out[state_id].sort(key=lambda donor: (donor.head_global_risk, donor.endpoint_id))
    return out


def _diagnostic_writes(events: pd.DataFrame) -> dict[str, set[int]]:
    """What S7's Head-blind probe actually wrote, per source state -- the overlap baseline."""
    out: dict[str, set[int]] = {}
    for _, row in events.iterrows():
        payload = row.get("write_from_endpoint_json")
        if not payload:
            continue
        out.setdefault(str(row["source_state_id"]), set()).update(
            int(position) for position in json.loads(payload))
    return out


# --------------------------------------------------------------------------------------------
# the replay
# --------------------------------------------------------------------------------------------


def replay_bundle(
    *, bundles: list[Path], config, band_table, stratum_key: str, reference_sequence: str,
    head_oracle: Any | None, protein_id: str | None = None,
) -> dict:
    """Replay the policy over every recorded source state and summarize its coverage."""
    frames = {name: [] for name in ("partial_states", "complete_endpoints", "feedback_events")}
    for bundle in bundles:
        for name in frames:
            table = Path(bundle) / f"{name}.parquet"
            if not table.exists():
                raise SystemExit(f"{bundle} has no {name}.parquet")
            frames[name].append(pd.read_parquet(table))
    states = pd.concat(frames["partial_states"], ignore_index=True).drop_duplicates("state_id")
    endpoints = pd.concat(frames["complete_endpoints"],
                          ignore_index=True).drop_duplicates("endpoint_id")
    events = pd.concat(frames["feedback_events"], ignore_index=True)
    if protein_id:
        states = states[states["protein_id"] == protein_id]
        endpoints = endpoints[endpoints["protein_id"] == protein_id]
        events = events[events["protein_id"] == protein_id]

    head_config_hash, head_config_source = _content_digest(config, "head_config")
    head_checkpoint_digest, head_checkpoint_source = _content_digest(config, "head_checkpoint")
    policy_spec_digest, policy_spec_source = _content_digest(config, "projection_policy_spec")
    stand_ins = {
        role: source for role, source in (
            ("head_config", head_config_source),
            ("head_checkpoint", head_checkpoint_source),
            ("projection_policy_spec", policy_spec_source),
        ) if source != "config"
    }
    declared_evaluator = ident.HeadEvaluatorIdentity(
        allele=config.head.allele, score_scale=config.head.score_scale,
        window_k_min=config.head.window_k_min, window_k_max=config.head.window_k_max,
        head_config_hash=head_config_hash, head_checkpoint_digest=head_checkpoint_digest,
    )
    # The INSTRUMENT defines the identity when one is loaded.  ``a_i`` is a difference of two
    # complete-sequence risks, and the counterfactual results carry the loaded Head's own identity;
    # a replay that ranked them under a config-derived identity would refuse its own measurements.
    # Where the config froze the digests they must AGREE -- a loaded Head that is not the run's is
    # a different instrument and its contributions are not this cohort's.
    realized = getattr(head_oracle, "evaluator_identity", None)
    evaluator = realized() if callable(realized) else declared_evaluator
    for name in ("allele", "score_scale", "window_k_min", "window_k_max"):
        if getattr(evaluator, name) != getattr(declared_evaluator, name):
            raise SystemExit(
                f"the loaded Head declares {name}={getattr(evaluator, name)!r} but the run's config "
                f"declares {getattr(declared_evaluator, name)!r}; the window grid the local "
                "evidence is built on would not be the grid the cohort was scored on"
            )
    for name, source in (("head_config_hash", head_config_source),
                         ("head_checkpoint_digest", head_checkpoint_source)):
        if source == "config" and getattr(evaluator, name) != getattr(declared_evaluator, name):
            raise SystemExit(
                f"the loaded Head's {name} is not the one this run froze; its contributions would "
                "be measured by a different instrument than the cohort was"
            )

    def binding_of(score: _Score):
        return ident.HeadScoreBinding(
            protein_id=score.protein_id, sequence_md5=score.sequence_md5,
            sequence_length=score.sequence_length,
            window_grid_digest=ident.window_grid_digest(score.windows), evaluator=evaluator,
        )

    reference_score = _reference_score(
        reference_sequence, head_oracle=head_oracle, endpoints=endpoints, evaluator=evaluator,
        protein_id=str(endpoints["protein_id"].iloc[0]) if len(endpoints) else (protein_id or ""),
    )
    incumbent = rw.LineageIncumbent(
        kind=rw.LineageIncumbentKind.CUMULATIVE_SAFETY_REFERENCE,
        incumbent_id="incumbent:" + ident.canonical_digest(
            {"replay": True, "sequence_md5": reference_score.sequence_md5})[:16],
        protein_id=reference_score.protein_id, lineage_id=f"{reference_score.protein_id}:fam0",
        sequence=reference_sequence, sequence_md5=reference_score.sequence_md5,
        sequence_length=len(reference_sequence), head_score=reference_score,
        head_global_risk=reference_score.global_risk,
        head_identity_digest=evaluator.digest(),
        head_score_digest=ident.canonical_digest({
            "sequence_md5": reference_score.sequence_md5,
            "head_identity_digest": evaluator.digest()}),
        bound_at_depth=0, accepted_at_depth=0,
        safety_reference_sequence_md5=reference_score.sequence_md5,
    )

    calibration = config.head_directed_calibration()
    if calibration is None:
        raise SystemExit(
            "--config declares no projection.head_directed block, so there is no cap, no "
            "epsilon_R and no tie law to replay under"
        )
    policy = pol.HeadDirectedCappedPolicy(
        band_table=band_table, stratum_key=stratum_key, incumbent=incumbent,
        safety_reference_score=reference_score, evaluator=evaluator,
        window_grid_digest=ident.window_grid_digest(reference_score.windows),
        calibration=calibration,
        incumbent_update_law=config.projection.head_directed.lineage_incumbent_update_law,
        counterfactual_scorer=_replay_scorer(head_oracle),
        policy_spec_digest=policy_spec_digest,
        policy_version=config.projection.support_policy_version,
    )

    views = source_views(states)
    donors = _donors(endpoints, evaluator=evaluator, binding_of=binding_of)
    diagnostic = _diagnostic_writes(events)
    r_step_by_state = _r_step_by_state(events, config)

    rows: list[dict] = []
    for state_id, view in sorted(views.items()):
        pool = donors.get(state_id, [])
        if not pool:
            rows.append({"state_id": state_id, "outcome": "no_definitive_donor"})
            continue
        donor = pool[0]
        r_step = r_step_by_state.get(state_id)
        if r_step is None:
            rows.append({"state_id": state_id, "outcome": "no_recorded_r_step"})
            continue
        result = policy.select(source_view=view, donor=donor, r_step=int(r_step))
        record = getattr(result, "decision_evidence", None)
        payload = record.canonical_payload() if record is not None else {}
        contributions = [row["contribution"] for row in payload.get("write_candidates", [])
                         if row.get("contribution") is not None]
        selected = {row["position"] for row in payload.get("write_candidates", [])
                    if row.get("selected")}
        probe = diagnostic.get(state_id, set())
        rows.append({
            "state_id": state_id,
            "outcome": ("decision" if isinstance(result, pol.PolicyDecision)
                        else payload.get("stall_reason", "rejection")),
            "donor_endpoint_id": donor.endpoint_id,
            "r_step": int(r_step),
            "n_editable": payload.get("n_editable"),
            "n_unresolved_source": payload.get("n_unresolved_source"),
            "u_target": payload.get("u_target"),
            "write_cap": payload.get("write_cap"),
            "n_legal_write_candidates": payload.get("n_legal_write_candidates"),
            "n_positive_contributions": payload.get("n_positive_contributions"),
            "m_band_min": payload.get("m_band_min"),
            "m_band_max": payload.get("m_band_max"),
            "realized_writes": payload.get("realized_writes"),
            "required_reopen": payload.get("required_reopen"),
            "n_legal_reopen_candidates": payload.get("n_legal_reopen_candidates"),
            "donor_gate_passed": (payload.get("donor_gate") or {}).get("passed"),
            "donor_gate_margin": (payload.get("donor_gate") or {}).get("margin"),
            "contributions": contributions,
            "selected_positions": sorted(selected),
            "diagnostic_write_positions": sorted(probe),
            "overlap_with_diagnostic_write": len(selected & probe),
        })

    return {
        "provenance": {
            "bundles": [str(bundle) for bundle in bundles],
            "config_digest": config.config_digest(),
            "band_calibration_id": band_table.provenance.calibration_id,
            "band_calibration_digest": band_table.provenance.calibration_content_digest,
            "stratum_key": stratum_key,
            "policy_id": policy.identity().policy_id,
            "policy_config_digest": policy.identity().policy_config_digest,
            "incumbent_id": incumbent.incumbent_id,
            "incumbent_sequence_md5": incumbent.sequence_md5,
            "head_loaded": head_oracle is not None,
            # Which content roles the replay could NOT bind to the run's own frozen digest.  Empty
            # is the good case: the replayed policy identity is then the run's own.
            "replay_local_content_roles": sorted(stand_ins),
            "calibration": calibration.canonical_payload(),
        },
        "coverage": _coverage(rows, head_loaded=head_oracle is not None),
        "sources": rows,
    }


def _coverage(rows: list[dict], *, head_loaded: bool) -> dict:
    """The numbers PLAN §8.4 reads before authorizing a descendant-generating run."""
    scored = [row for row in rows if "n_legal_write_candidates" in row]
    decisions = [row for row in scored if row["outcome"] == "decision"]
    contributions = [value for row in scored for value in row.get("contributions", [])]
    realized = [row["realized_writes"] for row in decisions]
    stalls: dict[str, int] = {}
    for row in rows:
        if row["outcome"] != "decision":
            stalls[row["outcome"]] = stalls.get(row["outcome"], 0) + 1
    bound_by = {"positives": 0, "cap": 0, "band": 0}
    for row in decisions:
        limits = {
            "positives": row["n_positive_contributions"], "cap": row["write_cap"],
            "band": row["m_band_max"],
        }
        for name, value in limits.items():
            if value == row["realized_writes"]:
                bound_by[name] += 1
    return {
        "n_sources": len(rows),
        "n_scored": len(scored),
        "n_with_legal_write_candidate": sum(
            1 for row in scored if (row.get("n_legal_write_candidates") or 0) > 0),
        "n_with_positive_write": sum(
            1 for row in scored if (row.get("n_positive_contributions") or 0) > 0),
        "n_committed_decisions": len(decisions),
        "stalls": stalls,
        "realized_writes": _describe(realized),
        "required_reopen": _describe([row["required_reopen"] for row in decisions]),
        "contribution": _describe(contributions) if head_loaded else None,
        "realized_write_bound_by": bound_by,
        "mean_overlap_with_diagnostic_write": (
            statistics.fmean([row["overlap_with_diagnostic_write"] for row in decisions])
            if decisions else None),
        # PLAN §8.4: a low-coverage result "returns to policy design" -- it is never rescued by
        # substituting arbitrary support.  This reader states the fact; the decision is the
        # runbook's.
        "head_contribution_measured": head_loaded,
    }


def _describe(values: list) -> dict | None:
    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return None
    numbers.sort()
    return {
        "n": len(numbers), "min": numbers[0], "max": numbers[-1],
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "q10": numbers[max(0, int(0.10 * (len(numbers) - 1)))],
        "q90": numbers[min(len(numbers) - 1, int(0.90 * (len(numbers) - 1)))],
    }


def _r_step_by_state(events: pd.DataFrame, config) -> dict[str, int]:
    """The re-entry coordinate each source was actually projected from.

    Read off the recorded transition where one exists; a source that never reached a projection
    falls back to the schedule's declared depth-0 ``r_step``, because the replay's question is what
    the policy WOULD have done at this run's own coordinate.
    """
    declared = int(config.schedule.points[0].r_step)
    out: dict[str, int] = {}
    for _, row in events.iterrows():
        value = row.get("r_step")
        if value is None or (isinstance(value, float) and value != value):
            continue
        out[str(row["source_state_id"])] = int(value)
    return _Defaulted(out, declared)


class _Defaulted(dict):
    """A mapping that answers the DECLARED coordinate for a source with no recorded transition."""

    def __init__(self, values: dict, default: int) -> None:
        super().__init__(values)
        self._default = int(default)

    def get(self, key, default=None):  # noqa: D102 - dict contract
        return super().get(key, self._default)


def _replay_scorer(head_oracle: Any | None):
    """The pure ``scorer(protein_id, sequences)`` contract, or an explicit refusal without a Head.

    Refusing rather than returning fabricated zeros: ``a_i = 0`` everywhere would read as "no
    position contributes", which is a MEASUREMENT, and this replay has taken none.
    """
    if head_oracle is None:
        def refuse(protein_id, sequences):
            raise RuntimeError(
                "no frozen Head was loaded, so this qualification replay cannot measure the "
                "leave-one-out contribution that defines a positive local dose; pass the required "
                "--head-checkpoint and --head-config arguments"
            )
        return refuse

    from inverse_folding.reference_flow.fusion_v2_runtime.contribution import (
        CounterfactualHeadScorer,
    )

    return CounterfactualHeadScorer(head_oracle=head_oracle)


def _reference_score(sequence: str, *, head_oracle, endpoints: pd.DataFrame, evaluator,
                     protein_id: str) -> _Score:
    """The incumbent's own Head evidence: scored live if a Head is loaded, else read from the run.

    The bundle already contains a scored row for the reference whenever the run's safety reference
    was scored through the same batch -- reusing it keeps the replay's incumbent byte-identical to
    the one the run gated against.  A digest mismatch is a hard failure, never a silent rescore.
    """
    digest = sequence_md5(sequence)
    recorded = endpoints[endpoints["sequence_md5"] == digest]
    if len(recorded):
        return _score_from_json(recorded.iloc[0]["head_score_json"], evaluator=evaluator)
    if head_oracle is None:
        raise SystemExit(
            "the reference sequence is not among the bundle's scored endpoints and no Head was "
            "loaded, so the lineage incumbent has no exact Head evidence.  Pass "
            "--head-checkpoint/--head-config, or point --reference-sequence at the run's own "
            "complete reference"
        )
    from inverse_folding.reference_flow.fusion_v2_runtime.lookahead import OracleRequest

    result = head_oracle.score([OracleRequest(
        protein_id=protein_id, sequence=sequence, sequence_md5=digest,
        sequence_length=len(sequence))])[0]
    return _Score(
        protein_id=str(result.protein_id), sequence_md5=str(result.sequence_md5),
        sequence_length=int(result.sequence_length), allele=str(result.allele),
        score_scale=str(result.score_scale),
        windows=tuple(_Window(start_0b=int(w.start_0b), end_0b=int(w.end_0b), k=int(w.k),
                              z=float(w.z)) for w in result.windows),
        global_risk=float(result.global_risk),
    )


def _content_digest(config, role: str) -> tuple[str, str]:
    """``(digest, source)`` for one content role: the run's frozen one, or a replay-local stand-in.

    Several roles a real run binds at RUNTIME carry no ``expected_sha256`` in the config -- the
    driver observes them from the files it was handed.  The replay has no such files, so it derives
    a stand-in from the config digest and REPORTS which roles it stood in for.  Inside one replay
    every score is produced under one evaluator identity, so the comparison stays internally
    consistent; what a stand-in costs is only that the replay's policy identity may not equal the
    run's, and a provenance field nobody can miss is the honest way to say so.
    """
    for row in config.content:
        if row.role == role and row.expected_sha256:
            return row.expected_sha256, "config"
    return ident.canonical_digest({"replay_local_role": role,
                                   "config_digest": config.config_digest()}), "replay_local"


def _load_head(args):
    """Load ONLY the frozen Head.

    PLAN V2F7: the offline replay "may load the frozen Head for explicitly requested counterfactual
    scoring, but it must not load the denoiser or structure backend or generate descendants".  So
    this reaches the Head builder directly rather than through v0's ``build_oracles``, which would
    also construct a refold model.
    """
    if not args.head_checkpoint:
        return None
    from types import SimpleNamespace

    from scripts.rf_fusion_v2_oracles import ProductionHeadOracle
    from scripts.rf_shared_oracles import _build_head_scorer_from_args
    from scripts.head_runtime import build_head_scorer

    # The window grid belongs on this namespace too: ``_build_head_scorer_from_args`` passes
    # ``args.window_k_min``/``window_k_max`` straight through to ``build_head_scorer``, so omitting
    # them is an AttributeError at Head-construction time -- i.e. after a GPU has been allocated.
    # ``main`` fills both from ``config.head`` when the flags are absent, and
    # ``calibrate_v2_head_policy.py`` requires them, so they are always resolved by the time we get
    # here.  Guarded by ``test_load_head_namespace_covers_every_field_the_builder_reads``.
    scorer = _build_head_scorer_from_args(SimpleNamespace(
        head_config_dir=args.head_config, head_checkpoint=args.head_checkpoint,
        head_variant_id=args.head_variant_id, head_device=args.device,
        head_allele_idx=args.head_allele_idx, head_window_batch_size=args.head_window_batch_size,
        allele=args.allele,
        window_k_min=args.window_k_min, window_k_max=args.window_k_max,
    ), build_head_scorer)
    return ProductionHeadOracle(
        scorer, allele=args.allele, score_scale=args.score_scale,
        window_k_min=args.window_k_min, window_k_max=args.window_k_max)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bundle", action="append", required=True, type=Path,
                        help="a V2 artifact bundle directory (repeatable)")
    parser.add_argument("--config", required=True, type=Path,
                        help="the V2 run config declaring projection.head_directed")
    parser.add_argument("--band-table", required=True, type=Path,
                        help="the frozen schedule-band calibration artifact")
    parser.add_argument("--stratum-key", required=True)
    parser.add_argument("--reference-sequence", required=True, type=Path,
                        help="the run's complete reference sequence (raw ASCII bytes)")
    parser.add_argument("--protein-id", default=None)
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    head = parser.add_argument_group(
        "frozen Head (required for the leave-one-out qualification gate)")
    head.add_argument("--head-checkpoint", type=Path, required=True)
    head.add_argument("--head-config", type=Path, required=True)
    head.add_argument("--head-variant-id", default=None)
    head.add_argument("--head-allele-idx", type=int, default=0)
    head.add_argument("--head-window-batch-size", type=int, default=64)
    head.add_argument("--allele", default=None)
    head.add_argument("--score-scale", default="raw_logit")
    head.add_argument("--window-k-min", type=int, default=None)
    head.add_argument("--window-k-max", type=int, default=None)
    head.add_argument("--device", default="cuda")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_v2_config(yaml.safe_load(Path(args.config).read_text()))
    for name, value in (("allele", config.head.allele),
                        ("window_k_min", config.head.window_k_min),
                        ("window_k_max", config.head.window_k_max)):
        if getattr(args, name) is None:
            setattr(args, name, value)
    band_table = load_band_table(args.band_table)
    reference_sequence = Path(args.reference_sequence).read_text(encoding="ascii")
    report = replay_bundle(
        bundles=list(args.bundle), config=config, band_table=band_table,
        stratum_key=args.stratum_key, reference_sequence=reference_sequence,
        head_oracle=_load_head(args), protein_id=args.protein_id,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    coverage = report["coverage"]
    print(json.dumps({"provenance": report["provenance"], "coverage": coverage},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":                                    # pragma: no cover - CLI
    raise SystemExit(main())
