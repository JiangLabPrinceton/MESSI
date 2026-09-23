"""Tier-specific candidate validation logic.

Used by scripts/validate_tier1.py and scripts/validate_tier3.py
to check manually curated candidate JSON files before proceeding
with downstream pipeline steps.
"""

import json
from typing import Any, Dict, List, Optional


# ── Tier 1 ───────────────────────────────────────────────────────────────────

_TIER1_REQUIRED_KEYS = {
    "protein_id", "uniprot_id", "pdb_id", "chain",
    "sequence_length", "resolution", "experimental_epitopes",
    "selection_reason",
}

_MIN_LENGTH = 100
_MAX_LENGTH = 500
_MAX_RESOLUTION = 2.5
_MIN_EPITOPES = 2


def validate_tier1_candidate(candidate: dict) -> List[str]:
    """Validate a single Tier 1 candidate entry.

    Returns a list of error strings. Empty list means valid.
    """
    errors: List[str] = []

    # Check required keys
    missing = _TIER1_REQUIRED_KEYS - set(candidate.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")
        return errors  # can't validate further

    # Sequence length
    seq_len = candidate["sequence_length"]
    if seq_len < _MIN_LENGTH or seq_len > _MAX_LENGTH:
        errors.append(
            f"sequence_length {seq_len} outside [{_MIN_LENGTH}, {_MAX_LENGTH}]"
        )

    # Resolution
    res = candidate["resolution"]
    if res is not None and res > _MAX_RESOLUTION:
        errors.append(
            f"resolution {res} exceeds max {_MAX_RESOLUTION}"
        )

    # Experimental epitopes
    epitopes = candidate["experimental_epitopes"]
    if not isinstance(epitopes, list) or len(epitopes) < _MIN_EPITOPES:
        n = len(epitopes) if isinstance(epitopes, list) else 0
        errors.append(
            f"Need >= {_MIN_EPITOPES} experimental epitopes, got {n}"
        )

    return errors


def validate_tier1_candidates(
    json_path: str,
    overlap_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate all candidates in a Tier 1 JSON file.

    Args:
        json_path: path to tier1_candidates.json.
        overlap_results: optional dict from MMseqs2 overlap detection,
            mapping protein_id → OverlapResult (or dict with 'exclude' key).
            When provided, overlapping candidates are flagged and excluded.

    Returns a summary dict with n_valid, n_invalid, overlap_excluded,
    and per-candidate errors.
    """
    with open(json_path) as f:
        candidates = json.load(f)

    valid = []
    invalid = []
    overlap_excluded = []
    all_errors: Dict[str, List[str]] = {}

    for c in candidates:
        pid = c.get("protein_id", "<unknown>")
        errs = validate_tier1_candidate(c)

        # MMseqs2 overlap check (if results provided)
        if overlap_results is not None and pid in overlap_results:
            ov = overlap_results[pid]
            exclude = ov.exclude if hasattr(ov, "exclude") else ov.get("exclude", False)
            if exclude:
                identity = ov.best_identity if hasattr(ov, "best_identity") else ov.get("best_identity", "?")
                target = ov.best_target if hasattr(ov, "best_target") else ov.get("best_target", "?")
                errs.append(
                    f"CATH overlap excluded: {identity:.1%} identity with {target}"
                )
                overlap_excluded.append(pid)

        if errs:
            invalid.append(pid)
            all_errors[pid] = errs
        else:
            valid.append(pid)

    return {
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "n_overlap_excluded": len(overlap_excluded),
        "valid": valid,
        "invalid": invalid,
        "overlap_excluded": overlap_excluded,
        "errors": all_errors,
    }


# ── Tier 3 ───────────────────────────────────────────────────────────────────

_TIER3_REQUIRED_KEYS = {
    "protein_id", "name", "pdb_id", "chain",
    "sequence_length", "resolution", "structure_source",
    "literature_evidence", "selection_reason",
}

_TIER3_MIN_NMP_WINDOWS = 3


def validate_tier3_candidate(
    candidate: dict,
    nmp_scores: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Validate a single Tier 3 candidate entry.

    Args:
        candidate: candidate dict with required fields.
        nmp_scores: optional dict mapping protein_id → NMP aggregation dict
            (with 'n_strong_binders' key, using %Rank < 5% threshold).
            When provided, enforces minimum NMP window count.

    Returns a list of error strings. Empty list means valid.
    """
    errors: List[str] = []

    missing = _TIER3_REQUIRED_KEYS - set(candidate.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")
        return errors

    # Literature evidence must be non-empty
    lit = candidate["literature_evidence"]
    if not lit or not str(lit).strip():
        errors.append("literature_evidence must not be empty")

    # Sequence length
    seq_len = candidate["sequence_length"]
    if seq_len < _MIN_LENGTH or seq_len > _MAX_LENGTH:
        errors.append(
            f"sequence_length {seq_len} outside [{_MIN_LENGTH}, {_MAX_LENGTH}]"
        )

    # NMP signal check (when scores provided)
    pid = candidate.get("protein_id", "")
    if nmp_scores is not None:
        if pid not in nmp_scores:
            errors.append(
                f"nmp_scores missing for {pid}; "
                f"run NetMHCIIpan before validation"
            )
        else:
            n_windows = nmp_scores[pid].get("n_strong_binders", 0)
            if n_windows < _TIER3_MIN_NMP_WINDOWS:
                errors.append(
                    f"NMP signal insufficient: {n_windows} windows < "
                    f"{_TIER3_MIN_NMP_WINDOWS} required (%Rank < 5%)"
                )

    return errors


def validate_tier3_candidates(
    json_path: str,
    nmp_scores: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate all candidates in a Tier 3 JSON file.

    Args:
        json_path: path to tier3_candidates.json.
        nmp_scores: optional dict mapping protein_id → NMP aggregation dict.
            When provided, enforces minimum NMP window count per candidate.
    """
    with open(json_path) as f:
        candidates = json.load(f)

    valid = []
    invalid = []
    all_errors: Dict[str, List[str]] = {}

    for c in candidates:
        pid = c.get("protein_id", "<unknown>")
        errs = validate_tier3_candidate(c, nmp_scores=nmp_scores)
        if errs:
            invalid.append(pid)
            all_errors[pid] = errs
        else:
            valid.append(pid)

    return {
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "valid": valid,
        "invalid": invalid,
        "errors": all_errors,
    }
