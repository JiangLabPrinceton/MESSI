"""Runtime helpers shared by Phase C scripts."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BYPROT_SRC = PROJECT_ROOT / "inverse_folding" / "dplm" / "src"
if str(BYPROT_SRC) not in sys.path:
    sys.path.insert(0, str(BYPROT_SRC))

from dataclasses import dataclass

# Structure-path resolution lives in a torch-free module so a model-free launch gate can resolve
# EXACTLY the file this runtime will later load; a validator with its own copy of the candidate
# order would certify a path the run does not use. Re-exported here, so existing imports and the
# monkeypatch target ``reference_flow.runtime.resolve_structure_path`` are unchanged.
from inverse_folding.reference_flow.structure_paths import (  # noqa: E402
    infer_chain_id,
    resolve_structure_path,
    safe_file_id,
)

_BYPROT_IMPORTS: dict[str, Any] | None = None
CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class PreparedBackbone:
    batch: dict[str, Any]
    structure_path: Path
    sequence_length: int


@dataclass(frozen=True)
class DPLMDenoiserContext:
    task: Any
    encoder_out: dict[str, Any]
    template_prev_tokens: torch.Tensor
    residue_mask: torch.Tensor
    tokens_template: torch.Tensor
    sequence_length: int


@dataclass(frozen=True)
class BatchedDPLMDenoiserContext:
    task: Any
    encoder_out: dict[str, Any]
    template_prev_tokens: torch.Tensor
    residue_mask: torch.Tensor
    tokens_template: torch.Tensor
    sequence_lengths: tuple[int, ...]


def safe_allele_tag(allele: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", ".")) else "_" for c in allele)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_sha(root: Path | None = None) -> str:
    repo_root = root or PROJECT_ROOT
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                text=True,
            )
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def checkpoint_digest(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_if_task(checkpoint_path: str | Path, device: str) -> Any:
    imports = _load_byprot_imports()
    checkpoint_path = Path(checkpoint_path).resolve()
    experiment_dir = checkpoint_path.parent.parent
    task, _ = imports["load_from_experiment"](str(experiment_dir), ckpt=checkpoint_path.name)
    task = task.eval()
    task = task.to(torch.device(device))
    return task


def load_test_entries(test_set_parquet: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(test_set_parquet).copy()
    required = {"protein_id", "sequence", "sequence_length", "pdb_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"test-set parquet missing required columns: {sorted(missing)}")
    if df["protein_id"].duplicated().any():
        raise ValueError("test-set parquet contains duplicate protein_id values")
    return df




def _coords_from_entry_or_structure(
    row: dict[str, Any],
    *,
    pdb_root: str | Path,
) -> tuple[Any, Path]:
    """Return backbone coords from an in-memory CATH row or a PDB/CIF path."""
    protein_id = str(row["protein_id"])
    sequence_length = int(row["sequence_length"])

    coords = row.get("coords")
    if coords is not None:
        if isinstance(coords, dict):
            try:
                coord_len = len(next(iter(coords.values())))
            except StopIteration as exc:
                raise ValueError(f"{protein_id}: empty coords dict") from exc
        else:
            coord_len = len(coords)
        if int(coord_len) != sequence_length:
            raise ValueError(
                f"{protein_id}: in-memory coord length {coord_len} "
                f"!= sequence_length {sequence_length}"
            )
        return coords, Path("CATH") / safe_file_id(protein_id)

    structure_path = resolve_structure_path(row, pdb_root)
    chain_id = str(row.get("if_chain_id") or row.get("chain") or "").strip()
    if not chain_id:
        chain_id = infer_chain_id(protein_id)
    imports = _load_byprot_imports()
    coords, structure_sequence = imports["load_coords"](
        str(structure_path), chain=chain_id
    )
    if len(structure_sequence) != sequence_length:
        raise ValueError(
            f"{protein_id}: structure length {len(structure_sequence)} "
            f"!= sequence_length {sequence_length}"
        )
    return coords, structure_path


def prepare_backbone_batch(
    *,
    task: Any,
    entries: list[pd.Series | dict[str, Any]],
    pdb_root: str | Path,
    device: str,
    skip_invalid: bool = True,
) -> tuple[
    dict[str, Any] | None,
    list[int],
    list[Path],
    list[dict[str, Any]],
    list[int],
]:
    """Featurize a list of test-set rows into a single DPLM batch (B>1).

    Partial-safe by default: each entry is loaded inside its own ``try``,
    failed entries are skipped and accumulated in ``prep_failures``, and
    the returned batch is built only from the surviving rows. This
    prevents a single broken PDB / coord-length mismatch from killing
    an entire length-bucket.

    Returns
    -------
    ``(batch, sequence_lengths, structure_paths, prep_failures, kept_indices)``

    - ``batch``: featurized DPLM batch (None if no entry survived)
    - ``sequence_lengths``: per-surviving-row int list, aligned with the
      first dim of ``batch`` tensors
    - ``structure_paths``: per-surviving-row resolved PDB/CIF path
    - ``prep_failures``: ``[{"entry": row_dict, "reason": str}, ...]``
      for every entry that failed before / during featurization
    - ``kept_indices``: positions in the input ``entries`` list that
      survived (so callers can map row_i → original entry / protein_id /
      design_idx without keeping a parallel list)

    With ``skip_invalid=False`` the first per-row failure raises, mirroring
    the pre-partial-safe contract (used only by legacy callers / tests).
    """
    if not entries:
        raise ValueError("prepare_backbone_batch requires at least one entry")

    items: list[dict[str, Any]] = []
    sequence_lengths: list[int] = []
    structure_paths: list[Path] = []
    prep_failures: list[dict[str, Any]] = []
    kept_indices: list[int] = []

    for idx, entry in enumerate(entries):
        row = dict(entry)
        try:
            protein_id = str(row["protein_id"])
            sequence = str(row["sequence"]).upper()
            sequence_length = int(row["sequence_length"])
            if len(sequence) != sequence_length:
                raise ValueError(
                    f"{protein_id}: len(sequence)={len(sequence)} != "
                    f"sequence_length={sequence_length}"
                )
            coords, structure_path = _coords_from_entry_or_structure(
                row, pdb_root=pdb_root
            )
        except Exception as exc:  # noqa: BLE001 - per-entry isolation
            if not skip_invalid:
                raise
            prep_failures.append(
                {
                    "entry": row,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "stage": "prepare_backbone_batch",
                }
            )
            continue

        items.append({"name": protein_id, "seq": sequence, "coords": coords})
        sequence_lengths.append(sequence_length)
        structure_paths.append(structure_path)
        kept_indices.append(idx)

    if not items:
        return None, [], [], prep_failures, []

    featurizer = task.alphabet.featurizer
    try:
        batch = featurizer(items)
    except Exception as exc:  # noqa: BLE001
        if not skip_invalid:
            raise
        # The featurizer rarely fails after per-entry load succeeded; if
        # it does, all surviving entries get marked as failed so the
        # runner can fall back to per-protein retry.
        for kept_i, entry in zip(kept_indices, items):
            prep_failures.append(
                {
                    "entry": {"protein_id": entry["name"], "seq": entry["seq"]},
                    "reason": f"featurizer error: {type(exc).__name__}: {exc}",
                    "stage": "prepare_backbone_batch.featurizer",
                }
            )
        return None, [], [], prep_failures, []

    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device)
    return batch, sequence_lengths, structure_paths, prep_failures, kept_indices


def prepare_backbone(
    *,
    task: Any,
    entry: pd.Series | dict[str, Any],
    pdb_root: str | Path,
    device: str,
) -> PreparedBackbone:
    row = dict(entry)
    protein_id = str(row["protein_id"])
    sequence = str(row["sequence"]).upper()
    sequence_length = int(row["sequence_length"])
    if len(sequence) != sequence_length:
        raise ValueError(
            f"{protein_id}: len(sequence)={len(sequence)} != sequence_length={sequence_length}"
        )

    coords, structure_path = _coords_from_entry_or_structure(row, pdb_root=pdb_root)

    featurizer = task.alphabet.featurizer
    batch = featurizer([{"name": protein_id, "seq": sequence, "coords": coords}])
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device)

    return PreparedBackbone(
        batch=batch,
        structure_path=structure_path,
        sequence_length=sequence_length,
    )


def generate_native_sequences_batched(
    *,
    task: Any,
    batch: dict[str, Any],
    sequence_lengths: list[int],
    max_iter: int,
    temperature: float,
    seed: int,
    sampling_strategy: str = "argmax",
    logit_processor: Any = None,
) -> list[str]:
    """Batched B>1 version of ``generate_native_sequence``.

    Runs DPLM's ``generate`` once on a featurizer-batched input, then
    splits the output tokens back into per-row sequences. The same
    seed is applied once before the batched generate call; per-row
    determinism within a batch is therefore not bit-equivalent to a
    sequence of B=1 generations, but reproducibility ACROSS runs with
    the same ``(batch composition, seed)`` is preserved.

    ``logit_processor`` (if provided) is applied at every decoder step
    on the full ``[B, L, V]`` logits, exactly as in the B=1 path.
    """
    if not sequence_lengths:
        raise ValueError("generate_native_sequences_batched needs >=1 row")

    _seed_all(seed)
    batch = clone_batch(batch)
    tokens = batch["tokens"]
    coord_mask = batch["coord_mask"]
    prev_tokens, prev_token_mask = task.inject_noise(
        tokens, coord_mask, noise="full_mask"
    )
    batch["prev_tokens"] = prev_tokens
    batch["prev_token_mask"] = prev_token_mask

    generate_kwargs: dict[str, Any] = dict(
        batch=batch,
        max_iter=max_iter,
        sampling_strategy=sampling_strategy,
        temperature=temperature,
        use_draft_seq=bool(task.hparams.generator.use_draft_seq),
    )
    if logit_processor is not None:
        generate_kwargs["logit_processor"] = logit_processor
    output_tokens, _ = task.model.generate(**generate_kwargs)

    special_sym_mask = (
        tokens.eq(task.alphabet.padding_idx)
        | tokens.eq(task.alphabet.cls_idx)
        | tokens.eq(task.alphabet.eos_idx)
    )
    output_tokens.masked_scatter_(special_sym_mask, tokens[special_sym_mask])
    residue_mask = coord_mask & ~special_sym_mask  # [B, L]

    sequences: list[str] = []
    for b, expected_len in enumerate(sequence_lengths):
        row_mask = residue_mask[b]
        row_tokens = output_tokens[b, row_mask].cpu()
        if int(row_tokens.numel()) != int(expected_len):
            raise RuntimeError(
                f"batched generate: row {b} produced {row_tokens.numel()} "
                f"residue tokens but expected {expected_len}; check the "
                f"featurizer / coord_mask alignment"
            )
        sequences.append(decode_residue_tokens(task, row_tokens))
    return sequences


def generate_native_sequence(
    *,
    task: Any,
    prepared: PreparedBackbone,
    max_iter: int,
    temperature: float,
    seed: int,
    sampling_strategy: str = "argmax",
    logit_processor: Any = None,
) -> str:
    _seed_all(seed)
    batch = clone_batch(prepared.batch)
    tokens = batch["tokens"]
    coord_mask = batch["coord_mask"]
    prev_tokens, prev_token_mask = task.inject_noise(tokens, coord_mask, noise="full_mask")
    batch["prev_tokens"] = prev_tokens
    batch["prev_token_mask"] = prev_token_mask

    generate_kwargs: dict[str, Any] = dict(
        batch=batch,
        max_iter=max_iter,
        sampling_strategy=sampling_strategy,
        temperature=temperature,
        use_draft_seq=bool(task.hparams.generator.use_draft_seq),
    )
    if logit_processor is not None:
        generate_kwargs["logit_processor"] = logit_processor
    output_tokens, _ = task.model.generate(**generate_kwargs)
    special_sym_mask = (
        tokens.eq(task.alphabet.padding_idx)
        | tokens.eq(task.alphabet.cls_idx)
        | tokens.eq(task.alphabet.eos_idx)
    )
    output_tokens.masked_scatter_(special_sym_mask, tokens[special_sym_mask])
    residue_mask = coord_mask & ~special_sym_mask
    return decode_residue_tokens(task, output_tokens[0, residue_mask[0]].cpu())


def _resolve_use_draft_seq(task: Any, override: bool | None) -> bool:
    """Tri-state resolver: ``None`` falls back to the checkpoint config (byte-identical to the
    legacy callers); ``True``/``False`` force the value. Fusion passes ``False`` to guarantee a
    backbone-only encoder context (no structure-derived draft ``init_pred``) regardless of the
    checkpoint's ``generator.use_draft_seq``."""
    if override is None:
        return bool(task.hparams.generator.use_draft_seq)
    return bool(override)


def build_dplm_denoiser_context(
    *,
    task: Any,
    prepared: PreparedBackbone,
    use_draft_seq_override: bool | None = None,
) -> DPLMDenoiserContext:
    batch = clone_batch(prepared.batch)
    tokens = batch["tokens"]
    coord_mask = batch["coord_mask"]
    prev_tokens, prev_token_mask = task.inject_noise(tokens, coord_mask, noise="full_mask")
    batch["prev_tokens"] = prev_tokens
    batch["prev_token_mask"] = prev_token_mask
    encoder_out = task.model.forward_encoder(
        batch,
        use_draft_seq=_resolve_use_draft_seq(task, use_draft_seq_override),
    )
    special_sym_mask = (
        tokens.eq(task.alphabet.padding_idx)
        | tokens.eq(task.alphabet.cls_idx)
        | tokens.eq(task.alphabet.eos_idx)
    )
    residue_mask = coord_mask & ~special_sym_mask
    sequence_length = int(residue_mask.sum().item())
    return DPLMDenoiserContext(
        task=task,
        encoder_out=encoder_out,
        template_prev_tokens=prev_tokens[0].clone(),
        residue_mask=residue_mask[0].clone(),
        tokens_template=tokens[0].clone(),
        sequence_length=sequence_length,
    )


def build_batched_dplm_denoiser_context(
    *,
    task: Any,
    batch: dict[str, Any],
    sequence_lengths: list[int] | tuple[int, ...] | None = None,
    use_draft_seq_override: bool | None = None,
) -> BatchedDPLMDenoiserContext:
    """Build one frozen DPLM encoder context for a batch of RF lanes."""

    batch = clone_batch(batch)
    tokens = batch["tokens"]
    coord_mask = batch["coord_mask"]
    prev_tokens, prev_token_mask = task.inject_noise(tokens, coord_mask, noise="full_mask")
    batch["prev_tokens"] = prev_tokens
    batch["prev_token_mask"] = prev_token_mask
    encoder_out = task.model.forward_encoder(
        batch,
        use_draft_seq=_resolve_use_draft_seq(task, use_draft_seq_override),
    )
    special_sym_mask = (
        tokens.eq(task.alphabet.padding_idx)
        | tokens.eq(task.alphabet.cls_idx)
        | tokens.eq(task.alphabet.eos_idx)
    )
    residue_mask = coord_mask & ~special_sym_mask
    inferred_lengths = tuple(int(v) for v in residue_mask.sum(dim=1).detach().cpu().tolist())
    if sequence_lengths is not None:
        expected = tuple(int(v) for v in sequence_lengths)
        if expected != inferred_lengths:
            raise ValueError(
                "sequence_lengths do not match batched residue masks: "
                f"expected={expected} inferred={inferred_lengths}"
            )
        inferred_lengths = expected
    return BatchedDPLMDenoiserContext(
        task=task,
        encoder_out=encoder_out,
        template_prev_tokens=prev_tokens.clone(),
        residue_mask=residue_mask.clone(),
        tokens_template=tokens.clone(),
        sequence_lengths=inferred_lengths,
    )


def make_dplm_denoiser(context: DPLMDenoiserContext):
    """Wrap the frozen DPLM decoder as ``denoiser(x_t, t, struct)``."""

    residue_positions = torch.nonzero(context.residue_mask, as_tuple=False).flatten()

    def _denoiser(x_t: torch.Tensor, t: float, struct: Any = None) -> torch.Tensor:
        del t, struct
        if x_t.shape != (context.sequence_length,):
            raise ValueError(
                f"expected x_t shape ({context.sequence_length},), got {tuple(x_t.shape)}"
            )
        prev_tokens = context.template_prev_tokens.clone()
        prev_tokens[residue_positions] = x_t.to(prev_tokens.device)
        esm_out = context.task.model.decoder(
            batch={"prev_tokens": prev_tokens.unsqueeze(0)},
            encoder_out=context.encoder_out,
            need_head_weights=False,
        )
        logits = esm_out["logits"][0, residue_positions].detach()
        _mask_invalid_decoder_logits(logits, context.task)
        return logits.cpu()

    return _denoiser


def make_batched_dplm_denoiser(context: BatchedDPLMDenoiserContext):
    """Wrap the frozen DPLM decoder as one batched RF denoiser call per step."""

    residue_positions = [
        torch.nonzero(mask, as_tuple=False).flatten()
        for mask in context.residue_mask
    ]

    def _denoiser(
        x_ts: list[torch.Tensor],
        t: float,
        structs: list[Any] | None = None,
    ) -> list[torch.Tensor]:
        del t, structs
        if len(x_ts) != len(context.sequence_lengths):
            raise ValueError(
                f"expected {len(context.sequence_lengths)} x_t tensors, got {len(x_ts)}"
            )
        prev_tokens = context.template_prev_tokens.clone()
        for row_i, (x_t, expected_len, positions) in enumerate(
            zip(x_ts, context.sequence_lengths, residue_positions)
        ):
            if x_t.shape != (expected_len,):
                raise ValueError(
                    f"lane {row_i}: expected x_t shape ({expected_len},), "
                    f"got {tuple(x_t.shape)}"
                )
            prev_tokens[row_i, positions] = x_t.to(prev_tokens.device)
        esm_out = context.task.model.decoder(
            batch={"prev_tokens": prev_tokens},
            encoder_out=context.encoder_out,
            need_head_weights=False,
        )
        batched_logits = esm_out["logits"].detach()
        outputs: list[torch.Tensor] = []
        for row_i, positions in enumerate(residue_positions):
            logits = batched_logits[row_i, positions]
            _mask_invalid_decoder_logits(logits, context.task)
            outputs.append(logits.cpu())
        return outputs

    return _denoiser


def _mask_invalid_decoder_logits(logits: torch.Tensor, task: Any) -> None:
    invalid = [
        index for index in range(logits.shape[-1])
        if task.alphabet.get_tok(index) not in CANONICAL_AA
    ]
    logits[..., invalid] = -torch.inf


def decode_residue_tokens(task: Any, residue_tokens: torch.Tensor) -> str:
    tokens = [task.alphabet.get_tok(int(tok)) for tok in residue_tokens]
    invalid = sorted({tok for tok in tokens if tok not in CANONICAL_AA})
    if invalid:
        raise RuntimeError(
            "decoded non-canonical residue tokens from sampler output: "
            f"{invalid}"
        )
    return "".join(tokens)


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_byprot_imports() -> dict[str, Any]:
    global _BYPROT_IMPORTS
    if _BYPROT_IMPORTS is None:
        openfold_source = PROJECT_ROOT / "inverse_folding" / "dplm" / "vendor" / "openfold"
        if str(openfold_source) not in sys.path:
            sys.path.insert(0, str(openfold_source))
        from byprot.utils import load_from_experiment
        from byprot.utils.io import load_coords

        _BYPROT_IMPORTS = {
            "load_coords": load_coords,
            "load_from_experiment": load_from_experiment,
        }
    return _BYPROT_IMPORTS
