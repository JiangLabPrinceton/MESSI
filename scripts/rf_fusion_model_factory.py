"""Shared DPLM model setup, backbone caching, and token identity."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__all__ = ["PreparedModel", "ModelSeams", "build_model_factory"]


@dataclass(frozen=True)
class ModelSeams:
    """The runtime entry points the factory calls.

    Injectable so the factory's wiring is testable without torch.  Defaults resolve the real
    implementations lazily, so importing this module costs nothing.
    """

    load_if_task: Callable[..., Any] | None = None
    prepare_backbone: Callable[..., Any] | None = None
    build_dplm_denoiser_context: Callable[..., Any] | None = None
    make_dplm_denoiser: Callable[..., Any] | None = None
    load_reference_flow_config: Callable[..., Any] | None = None
    make_sampler: Callable[..., Any] | None = None
    read_test_rows: Callable[..., Any] | None = None
    load_constraint_manifest: Callable[..., Any] | None = None

    def resolved(self) -> "ModelSeams":
        """Fill every unset seam with its real implementation, importing torch only now."""
        if all(getattr(self, f.name) is not None for f in self.__dataclass_fields__.values()):
            return self
        from inverse_folding.reference_flow.config import load_reference_flow_config
        from inverse_folding.reference_flow.constraints import load_constraint_manifest
        from inverse_folding.reference_flow.runtime import (
            build_dplm_denoiser_context,
            load_if_task,
            make_dplm_denoiser,
            prepare_backbone,
        )
        from inverse_folding.reference_flow.sampler import PositionDependentDFMSampler

        def _read_test_rows(path):
            import pandas as pd

            return pd.read_parquet(path).set_index("protein_id")

        return ModelSeams(
            load_if_task=self.load_if_task or load_if_task,
            prepare_backbone=self.prepare_backbone or prepare_backbone,
            build_dplm_denoiser_context=(
                self.build_dplm_denoiser_context or build_dplm_denoiser_context),
            make_dplm_denoiser=self.make_dplm_denoiser or make_dplm_denoiser,
            load_reference_flow_config=(
                self.load_reference_flow_config or load_reference_flow_config),
            make_sampler=self.make_sampler or (
                lambda *, mask_token_id, vocab_size: PositionDependentDFMSampler(
                    mask_token_id=mask_token_id, vocab_size=vocab_size)),
            read_test_rows=self.read_test_rows or _read_test_rows,
            load_constraint_manifest=self.load_constraint_manifest or load_constraint_manifest,
        )


@dataclass
class PreparedModel:
    """One cohort's model context: everything below the version-specific oracles.

    Deliberately mutable and NOT frozen: it owns two caches whose whole purpose is to accumulate
    across a cohort.  A frozen value that copied itself would prepare the same backbone twice.
    """

    task: Any
    sampler: Any
    rf_config: Any
    id_to_aa: Mapping[int, str]
    aa_token_ids: frozenset
    mask_token_id: int
    vocab_size: int
    tokenizer_digest: str
    fixed_token_policy: str
    device: str
    pdb_root: Any
    _seams: ModelSeams
    _test_rows: Any = None
    _manifest: Any = None
    _prepared_cache: dict = field(default_factory=dict)
    _denoiser_cache: dict = field(default_factory=dict)

    def backbone_and_denoiser(self, protein_id: str):
        """Prepare a protein's backbone and denoiser ONCE per cohort.

        The cache is the reason this is a shared object rather than a function: preparing a
        backbone is the expensive step, and a per-call factory would repeat it for every root,
        every lookahead and every segment of the same protein.
        """
        if protein_id not in self._prepared_cache:
            row = dict(self._test_rows.loc[protein_id])
            row["protein_id"] = protein_id
            prepared = self._seams.prepare_backbone(
                task=self.task, entry=row, pdb_root=self.pdb_root, device=self.device)
            ctx = self._seams.build_dplm_denoiser_context(
                task=self.task, prepared=prepared, use_draft_seq_override=False)
            self._prepared_cache[protein_id] = prepared
            self._denoiser_cache[protein_id] = self._seams.make_dplm_denoiser(ctx)
        return self._prepared_cache[protein_id], self._denoiser_cache[protein_id]

    def sequence_length(self, protein_id: str) -> int:
        prepared, _ = self.backbone_and_denoiser(protein_id)
        return int(prepared.sequence_length)

    def fixed_tokens(self, protein_id: str) -> dict | None:
        """The hard-anchor constraint class for one protein, as sampler ``fixed_tokens``.

        ``None`` rather than ``{}`` for an unconstrained protein: the sampler treats the two
        differently, and an empty dict would declare a constraint set that exists and is empty
        rather than one that was never declared.
        """
        if self._manifest is None:
            return None
        constraint = self._manifest.constraint_for_protein(protein_id)
        fixed = {
            int(anchor.index_0b): int(self.task.alphabet.get_idx(str(anchor.expected_aa)))
            for anchor in constraint.hard_anchors
        }
        return fixed or None

    def coordinate_mask_digest(self, protein_id: str) -> str:
        """PER-PROTEIN, computed after the backbone is prepared.

        A cohort-level constant would claim a binding that does not exist: the coordinate-valid
        mask is a property of one structure, not of the run.
        """
        from scripts.rf_fusion_model_factory import coordinate_mask_digest

        prepared, _ = self.backbone_and_denoiser(protein_id)
        return coordinate_mask_digest(prepared)

    def null_h_values(self, length: int):
        from scripts.rf_fusion_model_factory import null_h_values

        return null_h_values(length)


def build_model_factory(
    *,
    base_if_checkpoint: Any,
    rf_sampler_config: Any,
    test_set_parquet: Any,
    pdb_root: Any,
    device: str = "cuda",
    constraint_manifest: Any = None,
    fixed_token_policy: str = "unconstrained",
    seams: ModelSeams | None = None,
) -> PreparedModel:
    """Load the frozen task, sampler and sampler config once for a whole cohort.

    The null-amplification assertion runs HERE rather than in each caller: it is the property that
    makes the substrate the frozen one, and a version that forgot to assert it would silently run a
    different kernel than the one every calibration was measured on.
    """
    from scripts.rf_fusion_model_factory import (
        _canonical_id_to_aa,
        assert_null_amplification,
        tokenizer_digest,
    )

    resolved = (seams or ModelSeams()).resolved()
    task = resolved.load_if_task(base_if_checkpoint, device=device)
    rf_config = resolved.load_reference_flow_config(rf_sampler_config)
    assert_null_amplification(rf_config)

    id_to_aa = _canonical_id_to_aa(task.alphabet)
    model = PreparedModel(
        task=task,
        sampler=resolved.make_sampler(
            mask_token_id=int(task.alphabet.mask_idx), vocab_size=len(task.alphabet)),
        rf_config=rf_config,
        id_to_aa=id_to_aa,
        aa_token_ids=frozenset(id_to_aa),
        mask_token_id=int(task.alphabet.mask_idx),
        vocab_size=len(task.alphabet),
        tokenizer_digest=tokenizer_digest(task.alphabet),
        fixed_token_policy=fixed_token_policy,
        device=device,
        pdb_root=pdb_root,
        _seams=resolved,
    )
    model._test_rows = resolved.read_test_rows(test_set_parquet)
    if constraint_manifest:
        model._manifest = resolved.load_constraint_manifest(constraint_manifest)
    return model


def payload_from_checkpoint(
    checkpoint,
    *,
    root_id: str,
    protein_id: str,
    arm_id: str,
    rho_id: str,
    mask_token_id: int,
    conditioning: ConditioningDigest,
) -> PartialRootPayload:
    """Convert a sampler ``ContinuationCheckpoint`` (torch/numpy state) into a torch-free
    ``PartialRootPayload``. Purely transformational and deterministic: tensors/arrays become tuples,
    the unresolved count comes from the checkpoint's recomputed maturity (and is re-derived from
    x_t by the payload itself), and the content-addressed conditioning digests are attached."""
    import numpy as np
    from inverse_folding.reference_flow.fusion.v1_records import SCHEMA_VERSION, PartialRootPayload

    x_t = tuple(int(v) for v in checkpoint.x_t.detach().cpu().tolist())
    scores = tuple(float(v) for v in np.asarray(checkpoint.scores, dtype=float).tolist())
    unmask = tuple(int(v) for v in checkpoint.unmask_step_by_pos)
    fixed_tokens = tuple((int(pos), int(tok)) for pos, tok in checkpoint.fixed_tokens)
    editable_positions = tuple(int(i) for i in checkpoint.editable_positions)
    return PartialRootPayload(
        schema_version=SCHEMA_VERSION, root_id=root_id, protein_id=protein_id, arm_id=arm_id,
        rho_id=rho_id, x_t=x_t, scores=scores, unmask_step_by_pos=unmask, step=int(checkpoint.step),
        n_steps=int(checkpoint.n_steps), t=float(checkpoint.t),
        snapshot_phase=checkpoint.snapshot_phase, fixed_tokens=fixed_tokens,
        editable_positions=editable_positions,
        n_unresolved_editable=int(checkpoint.maturity.n_unresolved_editable),
        mask_token_id=int(mask_token_id), paid_prefix_dfe=int(checkpoint.paid_prefix_dfe),
        rng_state=dict(checkpoint.rng_state), conditioning=conditioning,
    )


def decode_tokens_to_aa(tokens: "torch.Tensor", id_to_aa: Mapping[int, str]) -> str:
    """Decode a complete token vector to an AA string via the tokenizer's id->AA map."""
    return "".join(id_to_aa[int(t)] for t in tokens.detach().cpu().tolist())


def resume_from_payload(payload: PartialRootPayload, *, fork_seed: int):
    """Rebuild a fork ``ContinuationResume`` from a torch-free ``PartialRootPayload`` so the
    completer can materialize a fresh completion of that exact root under ``fork_seed``. The
    ``state_hash`` is computed with the sampler's own ``_replay_state_hash`` so the resume passes
    the tamper gate before any denoiser call (PLAN §2.6)."""
    from inverse_folding.reference_flow.sampler import ContinuationResume, _replay_state_hash

    import torch
    import numpy as np

    x_t = torch.tensor(payload.x_t, dtype=torch.long)
    scores = np.asarray(payload.scores, dtype=np.float64)
    unmask = tuple(int(v) for v in payload.unmask_step_by_pos)
    fixed_tokens = tuple((int(p), int(t)) for p, t in payload.fixed_tokens)
    editable_positions = tuple(int(i) for i in payload.editable_positions)
    seed = int(fork_seed)
    state_hash = _replay_state_hash(
        x_t, scores, payload.step, payload.n_steps, fixed_tokens, editable_positions, unmask,
        "fork", None, seed,
    )
    return ContinuationResume(
        x_t=x_t, scores=scores, unmask_step_by_pos=unmask, start_step=int(payload.step),
        n_steps=int(payload.n_steps), fixed_tokens=fixed_tokens,
        editable_positions=editable_positions, mode="fork", state_hash=state_hash, fork_seed=seed,
    )


def tokenizer_digest(alphabet) -> str:
    """Digest of the RESOLVED token list, so a tokenizer change invalidates root identity (a
    literal name like "byprot_alphabet" would not)."""
    import hashlib

    tokens = [str(alphabet.get_tok(i)) for i in range(len(alphabet))]
    return "tok-" + hashlib.sha256("\x1e".join(tokens).encode("utf-8")).hexdigest()[:32]


def coordinate_mask_digest(prepared) -> str:
    """Digest of the ACTUAL per-protein coordinate-valid mask.

    Backbone conditioning differs protein to protein and residue to residue, so neither a constant
    literal nor a length-only hash is an identity: two proteins of equal length with different
    missing-density patterns must not share a conditioning digest. Hashes the real per-residue
    coordinate-validity bits from the prepared batch."""
    import hashlib

    import numpy as np

    batch = prepared.batch
    mask = batch.get("coord_mask")
    if mask is None:
        raise ValueError(
            "prepared backbone has no 'coord_mask': the coordinate-valid mask is part of the "
            "conditioning identity and must not be substituted by a length-only placeholder"
        )
    bits = np.asarray(mask.detach().cpu().numpy(), dtype=bool).reshape(-1)
    payload = bits.tobytes() + f"|len={int(prepared.sequence_length)}".encode("ascii")
    return "cm-" + hashlib.sha256(payload).hexdigest()[:32]


_NULL_AMPLIFICATION_FORM = "constant_one"


def assert_null_amplification(rf_config) -> None:
    """Second, TYPED gate on the loaded sampler config.

    ``assert_null_entry_kernel`` refuses on the raw YAML before torch is imported; this re-checks
    the value the sampler will actually consume, so a divergence between the parsed config and the
    file (a loader default, an in-process override) still cannot reach the denoiser loop.
    """
    form = str(rf_config.amplification.form)
    if form != _NULL_AMPLIFICATION_FORM:
        raise ValueError(
            f"resolved entry amplification.form={form!r}; V1-A executes only "
            f"{_NULL_AMPLIFICATION_FORM!r} (g == 1 at every position)"
        )


def null_h_values(sequence_length: int) -> "np.ndarray":
    """The h vector handed to the sampler under the frozen null kernel: ZEROS.

    Matching the C1 driver's absent-h convention, ``amplification_factor(constant_one, h)`` returns
    1 for any h -- so the schedule is position-INDEPENDENT by construction, not by the value of h.
    A ones vector would be wrong here: it reads as a real h and would produce a different g under
    any other form, hiding a firewall breach instead of exposing it.
    """
    import numpy as np

    return np.zeros(int(sequence_length), dtype=np.float32)


def _canonical_id_to_aa(alphabet) -> dict:
    aa20 = set("ACDEFGHIKLMNPQRSTVWY")
    out = {}
    for i in range(len(alphabet)):
        tok = alphabet.get_tok(i)
        if isinstance(tok, str) and tok in aa20:
            out[i] = tok
    return out
