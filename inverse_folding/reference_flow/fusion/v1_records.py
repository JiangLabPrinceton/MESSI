"""V1F3 torch-free partial-root records, canonical hashes, and content provenance.

Pure Python: no torch, no numpy, no model calls. A :class:`PartialRootPayload` is the
serialized (torch-free) form of a sampler ``ContinuationCheckpoint`` plus its conditioning
digests. Three distinct identities (PLAN_RF_REFINE_FUSION_V1 §2.4):

- ``root_id`` -- lineage identity, unique even when two trajectories converge;
- ``root_equivalence_hash`` -- canonical hash of the state that determines fresh-seed
  continuation behavior (visible tokens, scores, step, conditioning, fixed tokens); it EXCLUDES
  lineage identity, ``unmask_step_by_pos`` telemetry, and the uninterrupted-replay RNG stream;
- ``snapshot_payload_hash`` -- integrity hash of the complete serialized payload, including
  identity RNG state, lineage id, and unmask telemetry.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "v1root-1"
CONTINUATION_PHASE = "pre_denoiser_after_previous_remask"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj: Any) -> bytes:
    """Deterministic, key-sorted, whitespace-free JSON bytes for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def content_digest(path: Any) -> str:
    """SHA-256 of a file's BYTES (not path / size / mtime), so a same-size content edit
    invalidates any resume bound to it (PLAN §4.2)."""
    with open(path, "rb") as handle:
        return _sha256_hex(handle.read())


def verify_payload_integrity(
    payload: "PartialRootPayload", expected_snapshot_payload_hash: str
) -> None:
    """Fail-fast if a payload's recomputed integrity hash does not match the expected one, so a
    tampered persisted root never enters replay (PLAN §2.4/§4.2)."""
    actual = payload.snapshot_payload_hash
    if actual != expected_snapshot_payload_hash:
        raise ValueError(
            f"payload integrity mismatch: {actual} != {expected_snapshot_payload_hash}"
        )


def make_root_id(protein_id: str, arm_id: str, rho_id: str, root_index: int) -> str:
    """Lineage identity for the ``root_index``-th root-prefix attempt (unique per attempt even
    when two attempts converge to the same visible state)."""
    return f"{protein_id}:{arm_id}:{rho_id}:r{int(root_index)}"


@dataclass(frozen=True)
class ConditioningDigest:
    """Content digests of the inputs that determine continuation behavior. Every field is a
    required non-empty string; the driver fills them from real content (§4.2)."""

    dplm_checkpoint: str
    tokenizer: str
    backbone_row: str
    coordinate_mask: str
    entry_config: str
    fixed_token_policy: str
    #: The frozen V1-A null entry runtime. These are part of the conditioning identity because a
    #: run with a controller or an h-map is a DIFFERENT method whose roots must never collide with
    #: a null-runtime root (runbook §2). There is deliberately no h-map DIGEST field: V1-A consumes
    #: no h-map at all, so there is nothing to hash and no placeholder to fake.
    controller_enabled: bool
    h_maps_present: bool

    def __post_init__(self) -> None:
        for name in ("dplm_checkpoint", "tokenizer", "backbone_row", "coordinate_mask",
                     "entry_config", "fixed_token_policy"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"conditioning digest {name!r} must be a non-empty str")
            if value.startswith("unset:"):
                raise ValueError(
                    f"conditioning digest {name!r} is a placeholder ({value!r}); every digest must "
                    "come from real content (PLAN §4.2)"
                )
        for name in ("controller_enabled", "h_maps_present"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be an explicit bool")
        if self.controller_enabled or self.h_maps_present:
            raise ValueError(
                "V1-A conditioning must declare controller_enabled=False and h_maps_present=False; "
                "controller/h-map steering is a deferred track"
            )

    def canonical(self) -> dict[str, str]:
        return {
            "dplm_checkpoint": self.dplm_checkpoint,
            "tokenizer": self.tokenizer,
            "backbone_row": self.backbone_row,
            "coordinate_mask": self.coordinate_mask,
            "entry_config": self.entry_config,
            "fixed_token_policy": self.fixed_token_policy,
            "controller_enabled": self.controller_enabled,
            "h_maps_present": self.h_maps_present,
        }


def _require_int_token(value: Any, field_name: str) -> int:
    """An x_t / fixed-token entry must be an exact integer (PLAN §2.4). A bool or a
    non-integral float is a hard failure; an integral float is coerced."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} token must be an int, not bool")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field_name} token {value!r} is not an integer")
        return int(value)
    return int(value)


def _norm_score(value: Any) -> float:
    """Canonicalize a committed-token score: coerce to float, reject NaN, and normalize -0.0 to
    0.0 so numerically-identical scores hash identically (-inf for masked positions is kept)."""
    number = float(value)
    if number != number:  # NaN
        raise ValueError("scores must not be NaN")
    return 0.0 if number == 0.0 else number


@dataclass(frozen=True)
class PartialRootPayload:
    schema_version: str
    root_id: str
    protein_id: str
    arm_id: str
    rho_id: str
    x_t: tuple[int, ...]
    scores: tuple[float, ...]
    unmask_step_by_pos: tuple[int, ...]
    step: int
    n_steps: int
    t: float
    snapshot_phase: str
    fixed_tokens: tuple[tuple[int, int], ...]
    editable_positions: tuple[int, ...]
    n_unresolved_editable: int
    mask_token_id: int
    paid_prefix_dfe: int
    rng_state: Mapping[str, Any]
    conditioning: ConditioningDigest

    def __post_init__(self) -> None:
        # Canonicalize numeric TYPE and set ORDER before anything enters a content hash, so the
        # two hashes are stable across to_dict/from_dict and numerically-identical or
        # differently-ordered convergent roots collapse to one basin (PLAN §2.4).
        object.__setattr__(self, "x_t", tuple(_require_int_token(v, "x_t") for v in self.x_t))
        object.__setattr__(self, "scores", tuple(_norm_score(v) for v in self.scores))
        object.__setattr__(
            self, "unmask_step_by_pos", tuple(int(v) for v in self.unmask_step_by_pos)
        )
        object.__setattr__(
            self, "editable_positions", tuple(sorted(int(v) for v in self.editable_positions))
        )
        object.__setattr__(
            self, "fixed_tokens",
            tuple(
                sorted(
                    (int(pos), _require_int_token(tok, "fixed_tokens"))
                    for pos, tok in self.fixed_tokens
                )
            ),
        )
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "n_steps", int(self.n_steps))
        object.__setattr__(self, "t", float(self.t))
        object.__setattr__(self, "paid_prefix_dfe", int(self.paid_prefix_dfe))
        object.__setattr__(self, "mask_token_id", _require_int_token(self.mask_token_id, "mask_token_id"))
        object.__setattr__(self, "n_unresolved_editable", int(self.n_unresolved_editable))

        length = len(self.x_t)
        if length == 0:
            raise ValueError("x_t must be non-empty")
        if len(self.scores) != length or len(self.unmask_step_by_pos) != length:
            raise ValueError("scores and unmask_step_by_pos must match len(x_t)")
        if not (0 <= self.step <= self.n_steps):
            raise ValueError(f"step {self.step} out of [0, {self.n_steps}]")
        if self.paid_prefix_dfe != self.step:
            raise ValueError(
                f"paid_prefix_dfe {self.paid_prefix_dfe} must equal step {self.step} "
                "for a fresh root prefix"
            )
        fixed_positions = [p for p, _ in self.fixed_tokens]
        if len(set(fixed_positions)) != len(fixed_positions):
            raise ValueError("duplicate fixed_tokens positions")
        for pos in fixed_positions:
            if not (0 <= pos < length):
                raise ValueError(f"fixed position {pos} out of range [0, {length})")
        if len(set(self.editable_positions)) != len(self.editable_positions):
            raise ValueError("duplicate editable_positions")
        for pos in self.editable_positions:
            if not (0 <= pos < length):
                raise ValueError(f"editable position {pos} out of range [0, {length})")
        if set(self.editable_positions) & set(fixed_positions):
            raise ValueError("editable_positions and fixed positions overlap")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {self.schema_version!r} != {SCHEMA_VERSION!r}"
            )
        if self.snapshot_phase != CONTINUATION_PHASE:
            raise ValueError(
                f"snapshot_phase {self.snapshot_phase!r} != {CONTINUATION_PHASE!r}"
            )
        if set(fixed_positions) | set(self.editable_positions) != set(range(length)):
            raise ValueError(
                "fixed and editable positions must partition [0, length) exactly"
            )
        # A fixed anchor is a RESOLVED residue: its token cannot be the mask, and x_t at that
        # position must equal the declared fixed token. This keeps the record self-consistent
        # rather than trusting an upstream normalizer.
        for pos, tok in self.fixed_tokens:
            if tok == self.mask_token_id:
                raise ValueError(f"fixed anchor at {pos} holds the mask token {tok}")
            if self.x_t[pos] != tok:
                raise ValueError(
                    f"x_t[{pos}]={self.x_t[pos]} != declared fixed token {tok}"
                )
        # Derive the unresolved-editable count from x_t (never trust the declared value): an
        # unresolved editable residue is an editable position still holding the mask token, so a
        # fully-resolved payload cannot claim rho_edit < 1 (PLAN §2.3). Runs AFTER the range/
        # partition checks so an out-of-range editable index fails as a ValueError, not IndexError.
        derived_unresolved = sum(
            1 for i in self.editable_positions if self.x_t[i] == self.mask_token_id
        )
        if self.n_unresolved_editable != derived_unresolved:
            raise ValueError(
                f"n_unresolved_editable {self.n_unresolved_editable} != masked-editable count "
                f"{derived_unresolved} derived from x_t"
            )
        if derived_unresolved < 1:
            raise ValueError(
                "pre-terminal root must have >=1 unresolved editable residue "
                "(rho_edit == 1.0 is not a valid partial root)"
            )

    # --- canonical projections used for the two content hashes --- #
    def _equivalence_obj(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protein_id": self.protein_id,
            "arm_id": self.arm_id,
            "rho_id": self.rho_id,
            "x_t": list(self.x_t),
            "scores": list(self.scores),
            "step": self.step,
            "n_steps": self.n_steps,
            "snapshot_phase": self.snapshot_phase,
            "fixed_tokens": [list(ft) for ft in self.fixed_tokens],
            "editable_positions": list(self.editable_positions),
            "mask_token_id": self.mask_token_id,
            "conditioning": self.conditioning.canonical(),
        }

    def _full_obj(self) -> dict[str, Any]:
        obj = self._equivalence_obj()
        obj.update(
            {
                "root_id": self.root_id,
                "unmask_step_by_pos": list(self.unmask_step_by_pos),
                "n_unresolved_editable": self.n_unresolved_editable,
                "t": self.t,
                "paid_prefix_dfe": self.paid_prefix_dfe,
                "rng_state": json.loads(_canonical_json(dict(self.rng_state)).decode("utf-8")),
            }
        )
        return obj

    @property
    def actual_rho_edit(self) -> float:
        """Actual editable maturity of THIS root state: resolved editable fraction derived from
        x_t (masked editable positions are the unresolved complement), never the target rho."""
        n_editable = len(self.editable_positions)
        return (n_editable - self.n_unresolved_editable) / n_editable

    @property
    def root_equivalence_hash(self) -> str:
        return _sha256_hex(_canonical_json(self._equivalence_obj()))

    @property
    def snapshot_payload_hash(self) -> str:
        return _sha256_hex(_canonical_json(self._full_obj()))

    # --- lossless round-trip --- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "root_id": self.root_id,
            "protein_id": self.protein_id,
            "arm_id": self.arm_id,
            "rho_id": self.rho_id,
            "x_t": list(self.x_t),
            "scores": list(self.scores),
            "unmask_step_by_pos": list(self.unmask_step_by_pos),
            "step": self.step,
            "n_steps": self.n_steps,
            "t": self.t,
            "snapshot_phase": self.snapshot_phase,
            "fixed_tokens": [list(ft) for ft in self.fixed_tokens],
            "editable_positions": list(self.editable_positions),
            "n_unresolved_editable": self.n_unresolved_editable,
            "mask_token_id": self.mask_token_id,
            "paid_prefix_dfe": self.paid_prefix_dfe,
            "rng_state": dict(self.rng_state),
            "conditioning": self.conditioning.canonical(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PartialRootPayload":
        return cls(
            schema_version=data["schema_version"],
            root_id=data["root_id"],
            protein_id=data["protein_id"],
            arm_id=data["arm_id"],
            rho_id=data["rho_id"],
            # Pass raw JSON values through: __post_init__ canonicalizes and REJECTS a
            # non-integral token (e.g. x_t=[1.5]) rather than silently truncating it.
            x_t=tuple(data["x_t"]),
            scores=tuple(data["scores"]),
            unmask_step_by_pos=tuple(data["unmask_step_by_pos"]),
            step=data["step"],
            n_steps=data["n_steps"],
            t=data["t"],
            snapshot_phase=data["snapshot_phase"],
            fixed_tokens=tuple((p, tok) for p, tok in data["fixed_tokens"]),
            editable_positions=tuple(data["editable_positions"]),
            n_unresolved_editable=data["n_unresolved_editable"],
            mask_token_id=data["mask_token_id"],
            paid_prefix_dfe=data["paid_prefix_dfe"],
            rng_state=dict(data["rng_state"]),
            conditioning=ConditioningDigest(**data["conditioning"]),
        )


@dataclass(frozen=True)
class RootCollapse:
    """Result of collapsing lineage roots onto unique equivalence classes. ``unique`` holds one
    representative per class (the lex-smallest ``root_id``), sorted deterministically;
    ``converged`` maps each equivalence hash to ALL member ``root_id`` s (retained telemetry --
    converged siblings never masquerade as independent basin slots)."""

    unique: tuple[PartialRootPayload, ...]
    converged: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def n_unique(self) -> int:
        return len(self.unique)


def collapse_roots(payloads: Sequence[PartialRootPayload]) -> RootCollapse:
    """Group roots by ``root_equivalence_hash``; keep the lex-smallest ``root_id`` as the
    canonical representative of each class. Deterministic and input-order independent.
    Est/eval/final work is allocated only over ``unique``; ``converged`` keeps every lineage
    visible (PLAN §2.4)."""
    representatives: dict[str, PartialRootPayload] = {}
    groups: dict[str, list[str]] = {}
    for payload in payloads:
        equivalence = payload.root_equivalence_hash
        groups.setdefault(equivalence, []).append(payload.root_id)
        current = representatives.get(equivalence)
        if current is None or payload.root_id < current.root_id:
            representatives[equivalence] = payload
    unique = tuple(sorted(representatives.values(), key=lambda p: p.root_id))
    converged = {equivalence: tuple(sorted(ids)) for equivalence, ids in groups.items()}
    return RootCollapse(unique=unique, converged=converged)


# --- Public aliases for fusion_v2 reuse (additive; zero behavior change) ---------------------
# fusion_v2 must not re-derive these: a second canonical-JSON encoder or hash recipe would digest
# the same payload to different bytes, so V1 and V2 digests of shared payloads would disagree.
# See doc/FUSION_V2_Interface_Map.md section 5 and tests/inverse_folding/test_fusion_v2_reuse_boundary.py.
canonical_json_bytes = _canonical_json
sha256_hex = _sha256_hex
require_int_token = _require_int_token
normalize_score = _norm_score
