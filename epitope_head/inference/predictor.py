"""Inference predictor: checkpoint init, encoding, window scoring, aggregation."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from epitope_head.training.chunking import (
    assign_residue_to_chunk,
    build_chunk_plan,
    compute_residue_reliability,
)
from epitope_head.training.model import EpitopeScorer
from epitope_head.training.trainer import load_checkpoint


logger = logging.getLogger(__name__)

REQUIRED_INFERENCE_METADATA_KEYS = frozenset(
    {"manifest_version", "config_hash", "diff_ids_applied"},
)


def build_epitope_scorer_from_config(model_cfg: dict, encoder: torch.nn.Module) -> EpitopeScorer:
    """Construct EpitopeScorer from resolved model config."""
    required = [
        "d_enc",
        "d_proj",
        "length_embedding_dim",
        "allele_embedding_dim",
        "scorer_hidden_dim",
        "scorer_activation",
    ]
    missing = [k for k in required if k not in model_cfg]
    if missing:
        raise ValueError(f"Model config missing required keys: {missing}")

    return EpitopeScorer(
        encoder=encoder,
        d_enc=int(model_cfg["d_enc"]),
        d_proj=int(model_cfg["d_proj"]),
        length_emb_dim=int(model_cfg["length_embedding_dim"]),
        allele_emb_dim=int(model_cfg["allele_embedding_dim"]),
        min_k=int(model_cfg.get("min_k", 12)),
        max_k=int(model_cfg.get("max_k", 25)),
        n_alleles=int(model_cfg.get("n_alleles", 1)),
        scorer_hidden_dim=int(model_cfg["scorer_hidden_dim"]),
        scorer_activation=str(model_cfg["scorer_activation"]),
        pad_left_init=str(model_cfg.get("pad_left_init", "zeros")),
        pad_right_init=str(model_cfg.get("pad_right_init", "zeros")),
    )


class InferencePredictor:
    """Checkpoint-backed predictor with short/long sequence encoding paths."""

    def __init__(
        self,
        model: EpitopeScorer,
        inference_cfg: dict,
        tokenize_fn: Callable[[list[str]], dict[str, torch.Tensor]],
        checkpoint_metadata: dict | None = None,
    ):
        self.model = model
        self.inference_cfg = inference_cfg
        self.tokenize_fn = tokenize_fn
        self.checkpoint_metadata = dict(checkpoint_metadata or {})

        self.device = torch.device(inference_cfg.get("device", "cpu"))
        self.model.to(self.device)
        self.model.eval()
        # Keep frozen encoder path in eval mode.
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "esm"):
            self.model.encoder.esm.eval()

        chunking = inference_cfg["chunking"]
        self.chunking_enabled = bool(chunking["enabled"])
        self.context_len = int(chunking["context_len"])
        self.stride = int(chunking["stride"])
        self.margin = int(chunking["margin"])
        self.stitch_mode = str(chunking["stitch_mode"])
        self.enable_reliability = bool(chunking["enable_reliability"])
        if self.stitch_mode != "per_residue_stitch":
            raise ValueError(
                f"Unsupported stitch_mode '{self.stitch_mode}' for InferencePredictor "
                "(implemented: per_residue_stitch)",
            )

        # Resolve min_k/max_k: inference_cfg overrides within model-supported range.
        model_min_k = self.model.span_features.min_k
        model_max_k = self.model.span_features.max_k
        cfg_min_k = inference_cfg.get("min_k")
        cfg_max_k = inference_cfg.get("max_k")

        if cfg_min_k is not None or cfg_max_k is not None:
            self.min_k = int(cfg_min_k) if cfg_min_k is not None else model_min_k
            self.max_k = int(cfg_max_k) if cfg_max_k is not None else model_max_k
            if self.min_k < model_min_k or self.max_k > model_max_k:
                raise ValueError(
                    f"inference_cfg min_k/max_k [{self.min_k}, {self.max_k}] "
                    f"exceeds model-supported range [{model_min_k}, {model_max_k}]"
                )
            if self.min_k != model_min_k or self.max_k != model_max_k:
                logger.info(
                    "Inference k-range [%d, %d] narrowed from model range [%d, %d]",
                    self.min_k, self.max_k, model_min_k, model_max_k,
                )
        else:
            self.min_k = model_min_k
            self.max_k = model_max_k

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        model: EpitopeScorer,
        inference_cfg: dict,
        tokenize_fn: Callable[[list[str]], dict[str, torch.Tensor]],
    ) -> "InferencePredictor":
        """Load checkpoint with metadata+shape guards and build predictor."""
        ckpt = load_checkpoint(Path(checkpoint_path))
        metadata = ckpt.get("metadata", {})
        missing = REQUIRED_INFERENCE_METADATA_KEYS - set(metadata.keys())
        if missing:
            raise ValueError(
                "Checkpoint missing required inference metadata keys: "
                f"{sorted(missing)}",
            )

        state_dict = ckpt.get("model_state_dict")
        if state_dict is None:
            raise ValueError("Checkpoint missing 'model_state_dict'")

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise ValueError(f"Failed to load model state_dict: {exc}") from exc

        return cls(
            model=model,
            inference_cfg=inference_cfg,
            tokenize_fn=tokenize_fn,
            checkpoint_metadata=metadata,
        )

    def _encode_chunk(self, seq: str) -> torch.Tensor:
        toks = self.tokenize_fn([seq])
        token_ids = toks["token_ids"].to(self.device)
        attention_mask = toks["attention_mask"].to(self.device)
        with torch.no_grad():
            g, lengths = self.model.encode_and_project(token_ids, attention_mask)
        chunk_len = int(lengths[0].item())
        return g[0, :chunk_len]

    def encode_sequence(self, seq: str) -> tuple[torch.Tensor, dict]:
        """Encode one protein sequence into stitched residue embeddings."""
        if not isinstance(seq, str) or len(seq) == 0:
            raise ValueError("sequence must be a non-empty string")

        protein_len = len(seq)
        if not self.chunking_enabled or protein_len <= self.context_len:
            emb = self._encode_chunk(seq)
            debug = {
                "n_chunks": 1,
                "chunk_starts": [0],
                "residue_owner_chunk": [0] * protein_len,
                "residue_reliability": [1.0] * protein_len,
            }
            return emb.detach().cpu(), debug

        plan = build_chunk_plan(
            protein_length=protein_len,
            context_len=self.context_len,
            stride=self.stride,
            margin=self.margin,
        )
        owner = assign_residue_to_chunk(plan)
        if self.enable_reliability:
            reliability = compute_residue_reliability(plan)
        else:
            reliability = torch.ones(protein_len, dtype=torch.float32).numpy()

        # Encode each chunk once, then stitch by deterministic residue ownership.
        by_chunk: dict[int, torch.Tensor] = {}
        for j, start in enumerate(plan.starts):
            end = min(start + self.context_len, protein_len)
            by_chunk[j] = self._encode_chunk(seq[start:end])

        d_proj = int(self.model.projection.linear.out_features)
        stitched = torch.zeros(protein_len, d_proj, device=self.device)
        for i in range(protein_len):
            j = int(owner[i])
            local_idx = i - plan.starts[j]
            stitched[i] = by_chunk[j][local_idx]

        debug = {
            "n_chunks": plan.n_chunks,
            "chunk_starts": list(plan.starts),
            "residue_owner_chunk": owner.tolist(),
            "residue_reliability": [float(x) for x in reliability.tolist()],
        }
        return stitched.detach().cpu(), debug

    def enumerate_and_score(
        self,
        G: torch.Tensor,
        protein_len: int,
        min_k: int,
        max_k: int,
        allele_idx: int = 0,
        window_batch_size: int = 4096,
        return_exact: bool = False,
    ) -> tuple[list[dict], torch.Tensor] | tuple[list[dict], torch.Tensor, torch.Tensor]:
        """Enumerate all valid windows and score them (F3).

        Args:
            G: [L, D_proj] projected embeddings on CPU.
            protein_len: actual protein length.
            min_k: minimum peptide length.
            max_k: maximum peptide length.
            allele_idx: allele index (default 0).
            window_batch_size: batch size for scoring.
            return_exact: when True AND the dual-head is enabled, additionally
                compute the exact readout ``z_exact = z_region + boundary_head``,
                attach ``z_exact`` to each entry, and return a 3-tuple. The region
                score ``z`` (and hence the residue landscape) is untouched.

        Returns:
            ``(window_entries, z_tensor)`` (default), or
            ``(window_entries, z_tensor, z_exact_tensor)`` when ``return_exact``.
        """
        want_exact = return_exact and getattr(self.model, "enable_boundary_head", False)

        # Enumerate all (start, k) pairs, ordered by (start, k).
        all_spans = []
        for s in range(protein_len):
            for k in range(min_k, max_k + 1):
                e = s + k
                if e <= protein_len:
                    all_spans.append((s, e, k))

        if len(all_spans) == 0:
            empty = torch.tensor([], dtype=torch.float32)
            return ([], empty, empty) if return_exact else ([], empty)

        G_dev = G.to(self.device)
        all_logits = []
        all_exact: list[torch.Tensor] = []

        for batch_start in range(0, len(all_spans), window_batch_size):
            batch = all_spans[batch_start : batch_start + window_batch_size]
            spans_t = torch.tensor(
                [[s, e] for s, e, _ in batch], dtype=torch.long, device=self.device,
            )
            allele_t = torch.full(
                (len(batch),), allele_idx, dtype=torch.long, device=self.device,
            )
            with torch.no_grad():
                phi = self.model.span_features(G_dev, spans_t, protein_len, allele_t)
                z = self.model.scorer(phi)
                all_logits.append(z.cpu())
                if want_exact:
                    feats = self.model._extract_boundary_features(phi)
                    all_exact.append((z + self.model.boundary_head(feats)).cpu())

        z_tensor = torch.cat(all_logits, dim=0)
        z_exact_tensor = torch.cat(all_exact, dim=0) if want_exact else None

        window_entries = []
        for i, (s, e, k) in enumerate(all_spans):
            entry = {"start_0b": s, "end_0b": e, "k": k, "z": float(z_tensor[i].item())}
            if want_exact:
                entry["z_exact"] = float(z_exact_tensor[i].item())
            window_entries.append(entry)

        if return_exact:
            # When the head is disabled, mirror z so callers always get a tensor.
            return window_entries, z_tensor, (z_exact_tensor if want_exact else z_tensor)
        return window_entries, z_tensor

    def aggregate_hotspot_and_risk(
        self,
        window_entries: list[dict],
        z_tensor: torch.Tensor,
        protein_len: int,
        center_method: str = "median",
        clamp_method: str = "none",
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Aggregate window logits into per-residue hotspot and global risk (F4).

        Args:
            window_entries: list of dicts from enumerate_and_score.
            z_tensor: [N_windows] raw logits.
            protein_len: protein length.
            center_method: "median", "mean", or "none".
            clamp_method: "softplus", "relu", or "none".

        Returns:
            h_raw: [L] raw per-residue hotspot scores.
            h_processed: [L] post-processed hotspot scores.
            R: global risk scalar.
        """
        # Build per-residue covering window indices.
        residue_windows: list[list[int]] = [[] for _ in range(protein_len)]
        for i, entry in enumerate(window_entries):
            s = entry["start_0b"]
            e = entry["end_0b"]
            for r in range(s, e):
                residue_windows[r].append(i)

        # Compute h_raw: log-mean-exp of covering logits per residue.
        h_raw = torch.zeros(protein_len, dtype=torch.float32)
        for i in range(protein_len):
            indices = residue_windows[i]
            if len(indices) == 0:
                h_raw[i] = float("-inf")
            else:
                z_cover = z_tensor[indices]
                h_raw[i] = torch.logsumexp(z_cover, dim=0) - math.log(len(indices))

        # Global risk: log-mean-exp over all windows.
        n_windows = z_tensor.shape[0]
        if n_windows == 0:
            R = float("-inf")
        else:
            R = float((torch.logsumexp(z_tensor, dim=0) - math.log(n_windows)).item())

        # Post-processing: center only on finite residues to avoid -inf - (-inf) → NaN.
        h_processed = h_raw.clone()
        finite_mask = torch.isfinite(h_processed)

        if center_method in ("median", "mean") and finite_mask.any():
            finite_vals = h_processed[finite_mask]
            if center_method == "median":
                center = torch.median(finite_vals)
            else:
                center = torch.mean(finite_vals)
            h_processed[finite_mask] = h_processed[finite_mask] - center

        if clamp_method == "softplus":
            h_processed = F.softplus(h_processed)
        elif clamp_method == "relu":
            h_processed = F.relu(h_processed)

        return h_raw, h_processed, R

    def predict_protein(
        self,
        seq: str,
        allele_idx: int = 0,
        window_batch_size: int | None = None,
        return_exact: bool = False,
    ) -> dict:
        """Full protein prediction: encode → enumerate → score → aggregate (F3+F4).

        Args:
            seq: amino acid string.
            allele_idx: allele index (default 0).
            window_batch_size: optional batch size for span scoring. ``None``
                inherits ``enumerate_and_score``'s default (4096).
            return_exact: when True AND the dual-head is enabled, each entry in
                ``window_logits`` additionally carries ``z_exact``. The
                ``residue_hotspot`` landscape is always from the region score and
                is unaffected.

        Returns:
            dict with keys: window_logits, residue_hotspot, global_risk, meta, debug.
        """
        min_k = self.min_k
        max_k = self.max_k
        center_method = self.inference_cfg.get("hotspot_center_method", "median")
        clamp_method = self.inference_cfg.get("hotspot_clamp", "none")

        G, encode_debug = self.encode_sequence(seq)
        protein_len = len(seq)

        enumerate_kwargs: dict = {}
        if window_batch_size is not None:
            enumerate_kwargs["window_batch_size"] = int(window_batch_size)
        if return_exact:
            window_entries, z_tensor, _ = self.enumerate_and_score(
                G, protein_len, min_k, max_k, allele_idx, return_exact=True,
                **enumerate_kwargs,
            )
        else:
            window_entries, z_tensor = self.enumerate_and_score(
                G, protein_len, min_k, max_k, allele_idx, **enumerate_kwargs,
            )

        if len(window_entries) == 0:
            h_raw = torch.zeros(protein_len, dtype=torch.float32)
            h_processed = torch.zeros(protein_len, dtype=torch.float32)
            R = float("-inf")
        else:
            h_raw, h_processed, R = self.aggregate_hotspot_and_risk(
                window_entries, z_tensor, protein_len, center_method, clamp_method,
            )

        return {
            "window_logits": window_entries,
            "residue_hotspot": h_processed,
            "global_risk": R,
            "meta": {
                "protein_len": protein_len,
                "n_windows": len(window_entries),
                "min_k": min_k,
                "max_k": max_k,
                "center_method": center_method,
                "clamp_method": clamp_method,
            },
            "debug": {
                "encode": encode_debug,
                "h_raw": h_raw,
                "z_tensor": z_tensor,
            },
        }

    def _can_batch_encode_records(self, records: list[tuple[str, str]]) -> bool:
        if len(records) <= 1:
            return False
        for _, seq in records:
            if not isinstance(seq, str) or len(seq) == 0:
                return False
            if self.chunking_enabled and len(seq) > self.context_len:
                return False
        return True

    @staticmethod
    def _enumerate_spans(
        protein_len: int, min_k: int, max_k: int
    ) -> list[tuple[int, int, int]]:
        """Enumerate scoring spans in the canonical (start, then k) order."""
        all_spans: list[tuple[int, int, int]] = []
        for s in range(int(protein_len)):
            for k in range(int(min_k), int(max_k) + 1):
                e = s + k
                if e <= int(protein_len):
                    all_spans.append((s, e, k))
        return all_spans

    def _batched_span_score(
        self,
        *,
        g_batch_dev: torch.Tensor,
        protein_len: int,
        min_k: int,
        max_k: int,
        allele_idx: int,
        window_batch_size: int,
    ) -> tuple[list[tuple[int, int, int]], torch.Tensor] | None:
        """Score K same-length candidates, returning only ``[K, W]`` logits.

        This is the compact D2 path: it avoids hotspot/global aggregation and
        avoids materializing one dict/object per window when callers only need
        local risk over selected window indices. The returned tensor stays on
        ``self.device`` so the full prediction path can aggregate on device.
        """
        K = int(g_batch_dev.shape[0])
        L = int(protein_len)
        all_spans = self._enumerate_spans(L, min_k, max_k)
        W = len(all_spans)
        if W == 0:
            return None

        spans_t = torch.tensor(
            [[s, e] for s, e, _ in all_spans], dtype=torch.long, device=self.device,
        )
        allele_t = torch.full((W,), allele_idx, dtype=torch.long, device=self.device)
        G = g_batch_dev[:, :L]  # [K, L, D_proj]

        with torch.no_grad():
            phi = self.model.span_features.forward_batched(G, spans_t, L, allele_t)
            phi_flat = phi.reshape(K * W, phi.shape[-1])
            z_chunks = []
            for start in range(0, K * W, int(window_batch_size)):
                z_chunks.append(
                    self.model.scorer(phi_flat[start : start + int(window_batch_size)])
                )
            z_batch = torch.cat(z_chunks, dim=0).reshape(K, W)  # [K, W]
        return all_spans, z_batch

    def _batched_span_score_and_aggregate(
        self,
        *,
        g_batch_dev: torch.Tensor,
        protein_len: int,
        min_k: int,
        max_k: int,
        allele_idx: int,
        center_method: str,
        clamp_method: str,
        window_batch_size: int,
    ) -> tuple | None:
        """Score + aggregate K same-length candidates in one batched tensor pass.

        ``g_batch_dev`` is ``[K, L_pad, D_proj]`` on ``self.device`` (the batched
        encoder output, not yet moved to CPU). Because the span template
        (``all_spans``, covering indices, ``n_windows``) depends only on
        ``(protein_len, min_k, max_k)`` — not on the candidate sequence — it is
        enumerated once and reused across all K candidates. Span features, the
        scorer MLP, and the hotspot/risk aggregation all carry the candidate axis
        ``K`` and stay on ``self.device``; only the returned tensors are moved to
        CPU. Returns ``None`` when there are no valid windows
        (``protein_len < min_k``) so the caller falls back to the per-candidate
        zero-window path.

        Returns ``(all_spans, z_batch_cpu[K,W], h_raw_cpu[K,L], h_processed_cpu[K,L],
        R_cpu[K])``; numerically identical to the per-candidate
        ``enumerate_and_score`` + ``aggregate_hotspot_and_risk`` path up to float
        reduction reorder.
        """
        L = int(protein_len)
        scored = self._batched_span_score(
            g_batch_dev=g_batch_dev,
            protein_len=L,
            min_k=min_k,
            max_k=max_k,
            allele_idx=allele_idx,
            window_batch_size=window_batch_size,
        )
        if scored is None:
            return None
        all_spans, z_batch = scored
        W = len(all_spans)

        # Global risk: log-mean-exp over all windows, per candidate.
        R = torch.logsumexp(z_batch, dim=1) - math.log(W)  # [K]

        # Per-residue covering-window structure — shared across all K candidates.
        cover_lists: list[list[int]] = [[] for _ in range(L)]
        for i, (s, e, _k) in enumerate(all_spans):
            for r in range(s, e):
                cover_lists[r].append(i)
        cover_count = torch.tensor(
            [len(c) for c in cover_lists], dtype=torch.long, device=self.device,
        )  # [L]
        max_cover = int(cover_count.max().item())
        cover_index = torch.zeros(L, max_cover, dtype=torch.long, device=self.device)
        cover_mask = torch.zeros(L, max_cover, dtype=torch.bool, device=self.device)
        for r, lst in enumerate(cover_lists):
            if lst:
                cover_index[r, : len(lst)] = torch.tensor(
                    lst, dtype=torch.long, device=self.device,
                )
                cover_mask[r, : len(lst)] = True

        # h_raw[k, r] = log-mean-exp of the logits of windows covering residue r.
        z_cover = z_batch[:, cover_index]  # [K, L, max_cover]
        z_cover = z_cover.masked_fill(~cover_mask.unsqueeze(0), float("-inf"))
        h_raw = torch.logsumexp(z_cover, dim=2) - torch.log(
            cover_count.clamp(min=1).to(z_batch.dtype)
        )  # [K, L]
        zero_cover = cover_count == 0  # [L]
        if bool(zero_cover.any()):
            h_raw[:, zero_cover] = float("-inf")

        # Center on finite residues (shared finite set across K since cover_count
        # is sequence-independent), then clamp — matching aggregate_hotspot_and_risk.
        h_processed = h_raw.clone()
        finite_cols = cover_count > 0  # [L]; ~zero_cover
        if center_method in ("median", "mean") and bool(finite_cols.any()):
            finite_vals = h_processed[:, finite_cols]  # [K, n_finite]
            if center_method == "median":
                center = torch.median(finite_vals, dim=1).values  # [K]
            else:
                center = torch.mean(finite_vals, dim=1)  # [K]
            h_processed[:, finite_cols] = h_processed[:, finite_cols] - center.unsqueeze(1)
        if clamp_method == "softplus":
            h_processed = F.softplus(h_processed)
        elif clamp_method == "relu":
            h_processed = F.relu(h_processed)

        return (
            all_spans,
            z_batch.detach().cpu(),
            h_raw.detach().cpu(),
            h_processed.detach().cpu(),
            R.detach().cpu(),
        )

    def _predict_proteins_batched_encode(
        self,
        records: list[tuple[str, str]],
        allele_idx: int = 0,
        window_batch_size: int | None = None,
    ) -> list[dict]:
        """Predict short proteins with one encoder forward, preserving output order.

        Long proteins that require chunk stitching stay on the serial path in
        ``predict_proteins``. This keeps the optimization local to the common RF
        candidate case where every record is a same-protein short sequence.

        When every record has the SAME length, span scoring and hotspot/risk
        aggregation are also batched across candidates via
        :meth:`_batched_span_score_and_aggregate` (one scorer pass over all K*W
        spans, computed on ``self.device``). Mixed-length records fall back to the
        per-candidate ``enumerate_and_score`` path.
        """
        min_k = self.min_k
        max_k = self.max_k
        center_method = self.inference_cfg.get("hotspot_center_method", "median")
        clamp_method = self.inference_cfg.get("hotspot_clamp", "none")
        protein_ids = [protein_id for protein_id, _ in records]
        seqs = [seq for _, seq in records]

        toks = self.tokenize_fn(seqs)
        token_ids = toks["token_ids"].to(self.device)
        attention_mask = toks["attention_mask"].to(self.device)
        with torch.no_grad():
            g_batch, lengths = self.model.encode_and_project(token_ids, attention_mask)
        # Keep g_batch on self.device; the same-length path scores + aggregates on
        # device and only moves the results to CPU at the end.
        lengths_cpu = lengths.detach().cpu()

        protein_lens = [len(seq) for seq in seqs]
        for row_idx, (protein_id, _seq) in enumerate(zip(protein_ids, seqs)):
            encoded_len = int(lengths_cpu[row_idx].item())
            if encoded_len != protein_lens[row_idx]:
                raise RuntimeError(
                    f"batched head encode length mismatch for {protein_id}: "
                    f"{encoded_len} != {protein_lens[row_idx]}"
                )

        wbs = int(window_batch_size) if window_batch_size is not None else 4096

        # Same-length fast path: span scoring + hotspot/risk aggregation batched
        # across candidates on self.device (one scorer pass over all K*W spans).
        if len(set(protein_lens)) == 1:
            batched = self._batched_span_score_and_aggregate(
                g_batch_dev=g_batch,
                protein_len=protein_lens[0],
                min_k=min_k,
                max_k=max_k,
                allele_idx=allele_idx,
                center_method=center_method,
                clamp_method=clamp_method,
                window_batch_size=wbs,
            )
            if batched is not None:
                all_spans, z_batch, h_raw_b, h_processed_b, R_b = batched
                outputs: list[dict] = []
                for row_idx, protein_id in enumerate(protein_ids):
                    z_row = z_batch[row_idx]
                    window_entries = [
                        {"start_0b": s, "end_0b": e, "k": k, "z": float(z_row[i].item())}
                        for i, (s, e, k) in enumerate(all_spans)
                    ]
                    outputs.append({
                        "protein_id": protein_id,
                        "prediction": self._short_prediction_dict(
                            protein_len=protein_lens[row_idx],
                            window_entries=window_entries,
                            # Clone the per-candidate row slices so each prediction
                            # owns a fresh tensor (matching the serial path's
                            # per-protein torch.cat) instead of a view that keeps
                            # the shared [K, .] batch tensor alive.
                            h_processed=h_processed_b[row_idx].clone(),
                            R=float(R_b[row_idx].item()),
                            h_raw=h_raw_b[row_idx].clone(),
                            z_tensor=z_row.clone(),
                            min_k=min_k,
                            max_k=max_k,
                            center_method=center_method,
                            clamp_method=clamp_method,
                        ),
                    })
                return outputs

        # Fallback: mixed-length (or zero-window) records score per candidate.
        g_batch = g_batch.detach().cpu()
        enumerate_kwargs: dict = {}
        if window_batch_size is not None:
            enumerate_kwargs["window_batch_size"] = int(window_batch_size)

        outputs = []
        for row_idx, (protein_id, _seq) in enumerate(zip(protein_ids, seqs)):
            protein_len = protein_lens[row_idx]
            G = g_batch[row_idx, :protein_len]
            window_entries, z_tensor = self.enumerate_and_score(
                G, protein_len, min_k, max_k, allele_idx, **enumerate_kwargs,
            )
            if len(window_entries) == 0:
                h_raw = torch.zeros(protein_len, dtype=torch.float32)
                h_processed = torch.zeros(protein_len, dtype=torch.float32)
                R = float("-inf")
            else:
                h_raw, h_processed, R = self.aggregate_hotspot_and_risk(
                    window_entries, z_tensor, protein_len, center_method, clamp_method,
                )
            outputs.append({
                "protein_id": protein_id,
                "prediction": self._short_prediction_dict(
                    protein_len=protein_len,
                    window_entries=window_entries,
                    h_processed=h_processed,
                    R=R,
                    h_raw=h_raw,
                    z_tensor=z_tensor,
                    min_k=min_k,
                    max_k=max_k,
                    center_method=center_method,
                    clamp_method=clamp_method,
                ),
            })
        return outputs

    @staticmethod
    def _short_prediction_dict(
        *,
        protein_len: int,
        window_entries: list[dict],
        h_processed: torch.Tensor,
        R: float,
        h_raw: torch.Tensor,
        z_tensor: torch.Tensor,
        min_k: int,
        max_k: int,
        center_method: str,
        clamp_method: str,
    ) -> dict:
        """Assemble the single-chunk prediction dict shared by both short paths."""
        return {
            "window_logits": window_entries,
            "residue_hotspot": h_processed,
            "global_risk": R,
            "meta": {
                "protein_len": protein_len,
                "n_windows": len(window_entries),
                "min_k": min_k,
                "max_k": max_k,
                "center_method": center_method,
                "clamp_method": clamp_method,
            },
            "debug": {
                "encode": {
                    "n_chunks": 1,
                    "chunk_starts": [0],
                    "residue_owner_chunk": [0] * protein_len,
                    "residue_reliability": [1.0] * protein_len,
                },
                "h_raw": h_raw,
                "z_tensor": z_tensor,
            },
        }

    def predict_proteins_window_logits(
        self,
        records: list[tuple[str, str]],
        allele_idx: int = 0,
        window_batch_size: int | None = None,
    ) -> list[dict]:
        """Return only per-window logits for a batch of records.

        This is an array-first companion to :meth:`predict_proteins` for D2
        counterfactual scoring, where callers only need the window risk matrix.
        Same-length short records use the batched encoder + batched span scorer
        and skip hotspot/global aggregation plus ``window_logits`` dict creation.
        Mixed-length, long, and singleton inputs fall back to the canonical
        prediction path and are converted to the compact shape.
        """
        if not records:
            return []

        min_k = self.min_k
        max_k = self.max_k
        wbs = int(window_batch_size) if window_batch_size is not None else 4096

        if self._can_batch_encode_records(records):
            protein_ids = [protein_id for protein_id, _ in records]
            seqs = [seq for _, seq in records]
            protein_lens = [len(seq) for seq in seqs]

            toks = self.tokenize_fn(seqs)
            token_ids = toks["token_ids"].to(self.device)
            attention_mask = toks["attention_mask"].to(self.device)
            with torch.no_grad():
                g_batch, lengths = self.model.encode_and_project(token_ids, attention_mask)
            lengths_cpu = lengths.detach().cpu()
            for row_idx, (protein_id, _seq) in enumerate(zip(protein_ids, seqs)):
                encoded_len = int(lengths_cpu[row_idx].item())
                if encoded_len != protein_lens[row_idx]:
                    raise RuntimeError(
                        f"batched head encode length mismatch for {protein_id}: "
                        f"{encoded_len} != {protein_lens[row_idx]}"
                    )

            if len(set(protein_lens)) == 1:
                scored = self._batched_span_score(
                    g_batch_dev=g_batch,
                    protein_len=protein_lens[0],
                    min_k=min_k,
                    max_k=max_k,
                    allele_idx=allele_idx,
                    window_batch_size=wbs,
                )
                if scored is None:
                    empty = torch.tensor([], dtype=torch.float32)
                    return [
                        {
                            "protein_id": protein_id,
                            "protein_len": protein_lens[row_idx],
                            "window_spans": (),
                            "z_tensor": empty.clone(),
                        }
                        for row_idx, protein_id in enumerate(protein_ids)
                    ]
                all_spans, z_batch = scored
                z_batch = z_batch.detach().cpu()
                spans = tuple(all_spans)
                return [
                    {
                        "protein_id": protein_id,
                        "protein_len": protein_lens[row_idx],
                        "window_spans": spans,
                        "z_tensor": z_batch[row_idx].clone(),
                    }
                    for row_idx, protein_id in enumerate(protein_ids)
                ]

        rows: list[dict] = []
        for pred_row in self.predict_proteins(
            records, allele_idx=allele_idx, window_batch_size=window_batch_size,
        ):
            prediction = pred_row["prediction"]
            entries = prediction["window_logits"]
            rows.append({
                "protein_id": pred_row["protein_id"],
                "protein_len": int(prediction["meta"]["protein_len"]),
                "window_spans": tuple(
                    (int(w["start_0b"]), int(w["end_0b"]), int(w["k"]))
                    for w in entries
                ),
                "z_tensor": prediction["debug"]["z_tensor"].detach().cpu().clone(),
            })
        return rows

    def predict_proteins(
        self,
        records: list[tuple[str, str]],
        allele_idx: int = 0,
        window_batch_size: int | None = None,
    ) -> list[dict]:
        """Ordered batch facade over ``predict_protein``.

        The underlying model is constructed once on the instance and reused
        across all records (the facade is an instance method by design, so
        head weights and tokenizer state never reload between records).

        Args:
            records: list of ``(protein_id, sequence)`` pairs.
            allele_idx: allele index applied to every record.
            window_batch_size: optional window batch size forwarded to each
                ``predict_protein`` call.

        Returns:
            One ``{"protein_id": str, "prediction": dict}`` per input record,
            in input order.
        """
        if self._can_batch_encode_records(records):
            return self._predict_proteins_batched_encode(
                records, allele_idx=allele_idx, window_batch_size=window_batch_size
            )
        outputs: list[dict] = []
        for protein_id, seq in records:
            prediction = self.predict_protein(
                seq, allele_idx=allele_idx, window_batch_size=window_batch_size
            )
            outputs.append({"protein_id": protein_id, "prediction": prediction})
        return outputs
