"""Model path: Frozen ESM-2 encoder → Projection → Span Features → Scorer.

Implements PLAN.md Task E3 and codemap §6:
  - FrozenESMEncoder: frozen ESM-2 forward, strip BOS/EOS
  - ESMTokenizer: batch tokenization for collate_fn
  - ProjectionHead: Linear D_enc → D_proj
  - SpanFeatureBuilder: prefix-sum pool + endpoints + flanks + embeddings
  - ScorerMLP: 2-layer MLP, D_phi → scorer_hidden → 1
  - EpitopeScorer: full forward path (chunk-level, no stitching in training)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── ESM-2 Tokenizer (for datamodule collate_fn) ─────────────────────────────

class ESMTokenizer:
    """Wraps ESM-2 alphabet for batch tokenization.

    Produces token_ids and attention_mask tensors compatible with
    make_collate_fn(tokenize_fn=...) contract.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.cls_idx = alphabet.cls_idx    # BOS = 0
        self.eos_idx = alphabet.eos_idx    # EOS = 2
        self.pad_idx = alphabet.padding_idx  # PAD = 1

    def __call__(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize batch of AA sequences.

        Args:
            sequences: list of AA strings (no BOS/EOS).

        Returns:
            dict with:
              token_ids: LongTensor [B, T] (BOS + seq + EOS + PAD)
              attention_mask: BoolTensor [B, T]
        """
        max_len = max(len(s) for s in sequences) + 2  # +BOS +EOS
        B = len(sequences)
        token_ids = torch.full((B, max_len), self.pad_idx, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)

        for i, seq in enumerate(sequences):
            tokens = [self.cls_idx]
            for ch in seq:
                idx = self.alphabet.get_idx(ch)
                tokens.append(idx)
            tokens.append(self.eos_idx)
            t = len(tokens)
            token_ids[i, :t] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[i, :t] = True

        return {"token_ids": token_ids, "attention_mask": attention_mask}


# ── Frozen ESM-2 Encoder ────────────────────────────────────────────────────

class FrozenESMEncoder(nn.Module):
    """Frozen ESM-2 wrapper that returns residue-level embeddings.

    Strips BOS/EOS from hidden states. Runs under torch.no_grad().
    Input: token_ids [B, T] with BOS+seq+EOS+PAD, attention_mask [B, T].
    Output: embeddings [B, L_max, D_enc] and lengths [B] (residue counts).
    """

    def __init__(self, esm_model, d_enc: int = 1280):
        super().__init__()
        self.esm = esm_model
        self.d_enc = d_enc
        # Freeze all parameters
        for param in self.esm.parameters():
            param.requires_grad = False
        # Lock in eval mode to disable dropout/batchnorm shifts
        self.esm.eval()

    def train(self, mode: bool = True):
        """Override: keep ESM always in eval mode regardless of parent .train() calls."""
        # Set self.training flag normally, but never propagate to self.esm
        self.training = mode
        for name, child in self.named_children():
            if name != "esm":
                child.train(mode)
        return self

    @torch.no_grad()
    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through frozen ESM-2.

        Args:
            token_ids: [B, T] with BOS/EOS/PAD tokens
            attention_mask: [B, T] bool mask (True = real token)

        Returns:
            embeddings: [B, L_max, D_enc] residue embeddings (BOS/EOS stripped)
            lengths: [B] int tensor of residue counts per sample
        """
        # ESM-2 forward: returns dict with 'logits' and 'representations'
        results = self.esm(token_ids, repr_layers=[self.esm.num_layers])
        # Get last layer representations: [B, T, D_enc]
        hidden = results["representations"][self.esm.num_layers]

        # Strip BOS (position 0) and EOS; compute residue lengths
        # attention_mask counts: BOS + residues + EOS = real token count
        real_counts = attention_mask.sum(dim=1)  # [B]
        residue_lengths = real_counts - 2  # subtract BOS and EOS

        # Extract residue embeddings (positions 1..L for each sample)
        B, T, D = hidden.shape
        L_max = int(residue_lengths.max().item())
        embeddings = torch.zeros(B, L_max, D, device=hidden.device, dtype=hidden.dtype)
        for i in range(B):
            L_i = int(residue_lengths[i].item())
            embeddings[i, :L_i] = hidden[i, 1:1 + L_i]  # skip BOS at 0

        return embeddings, residue_lengths


# ── Projection Head ─────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """Linear projection D_enc → D_proj with optional LayerNorm."""

    def __init__(self, d_enc: int, d_proj: int, layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(d_enc, d_proj)
        self.norm = nn.LayerNorm(d_proj) if layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder embeddings.

        Args:
            x: [B, L, D_enc]
        Returns:
            [B, L, D_proj]
        """
        return self.norm(self.linear(x))


# ── Span Feature Builder ────────────────────────────────────────────────────

class SpanFeatureBuilder(nn.Module):
    """Build span feature vector phi from projected embeddings.

    Codemap §6.3 concatenation order:
      1. Mean-pooled interior via prefix-sum: [D_proj]
      2. In-span endpoints [G[start], G[end-1]]: [2 * D_proj]
      3. Boundary flanks [G[start-1], G[end]]: [2 * D_proj]
      4. Length embedding: [length_emb_dim]
      5. Allele embedding: [allele_emb_dim]

    D_phi = 5 * D_proj + length_emb_dim + allele_emb_dim
    """

    def __init__(
        self,
        d_proj: int,
        length_emb_dim: int,
        allele_emb_dim: int,
        min_k: int = 12,
        max_k: int = 25,
        n_alleles: int = 1,
        pad_left_init: str = "zeros",
        pad_right_init: str = "zeros",
        use_core_scorer: bool = False,
    ):
        super().__init__()
        self.d_proj = d_proj
        self.min_k = min_k
        self.max_k = max_k

        # Learnable boundary pad vectors
        self.pad_left = nn.Parameter(torch.zeros(d_proj))
        self.pad_right = nn.Parameter(torch.zeros(d_proj))
        if pad_left_init == "normal":
            nn.init.normal_(self.pad_left, std=0.02)
        if pad_right_init == "normal":
            nn.init.normal_(self.pad_right, std=0.02)

        # Length embedding: k in [min_k, max_k]
        n_lengths = max_k - min_k + 1
        self.length_embedding = nn.Embedding(n_lengths, length_emb_dim)
        self.length_offset = min_k

        # Allele embedding (v0: single entry)
        self.allele_embedding = nn.Embedding(n_alleles, allele_emb_dim)

        self.d_phi = 5 * d_proj + length_emb_dim + allele_emb_dim

        # Wave-4 core-aware scorer (default off -> bit-for-bit legacy). A learned
        # per-residue core logit, pooled over a window's 9-mer sub-windows by
        # logsumexp, appended to phi so the strongest ~9-mer binding core can
        # drive the window score instead of the diluting mean-pool. Constructed
        # LAST so disabled-mode RNG/state_dict are identical to legacy.
        self.use_core_scorer = use_core_scorer
        if use_core_scorer:
            self.core_scorer = nn.Linear(d_proj, 1)
            self.d_phi += 1

    def _pool_9mer_cores(
        self,
        prefix_c: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        chunk_len: int,
    ) -> torch.Tensor:
        """Pool a per-residue core prefix-sum to one core feature per span.

        For each span ``[s, e)`` take the sum of the per-residue core logit over
        every length-9 sub-window and ``logsumexp`` them (soft-max of the 9-mer
        core sums). ``prefix_c`` is ``[chunk_len+1]`` (cumulative sum, prefix_c[0]=0);
        ``starts``/``ends`` are ``[N]``; returns ``[N]``. Vectorized over the
        ``W = max_k-8 <= 17`` candidate cores. Spans with ``k<9`` (unreachable in
        the frozen k in [12,25] config) fall back to the full-span core sum.
        """
        if starts.shape[-1] == 0:                                       # empty span set
            return prefix_c.new_zeros(starts.shape)
        lens = ends - starts                                            # [N]
        W = max(1, int(lens.max().item()) - 8)
        offsets = torch.arange(W, device=starts.device)                 # [W]
        win_start = starts.unsqueeze(-1) + offsets                      # [N, W]
        valid = offsets.view(1, W) < (lens - 8).clamp(min=0).unsqueeze(-1)  # [N, W]
        win_start_c = win_start.clamp(max=chunk_len - 9)                # keep gather in range
        win_sum = prefix_c[win_start_c + 9] - prefix_c[win_start_c]     # [N, W]
        masked = win_sum.masked_fill(~valid, float("-inf"))
        pooled = torch.logsumexp(masked, dim=-1)                        # [N]
        full_span = prefix_c[ends] - prefix_c[starts]                   # [N]
        return torch.where(valid.any(dim=-1), pooled, full_span)

    def forward(
        self,
        G: torch.Tensor,
        spans: torch.Tensor,
        chunk_len: int,
        allele_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Build feature vectors for a set of spans within one chunk.

        Args:
            G: [L, D_proj] projected embeddings for one chunk (no batch dim).
            spans: [N, 2] int tensor of (start, end) in chunk-local 0b half-open coords.
            chunk_len: actual residue count in this chunk (may be < L if padded).
            allele_idx: [N] int tensor of allele indices (v0: all zeros).

        Returns:
            phi: [N, D_phi] feature vectors.
        """
        N = spans.shape[0]
        device = G.device

        starts = spans[:, 0]  # [N]
        ends = spans[:, 1]    # [N]
        pep_lens = ends - starts  # [N]

        # 1. Mean-pooled interior via prefix-sum
        # S[i] = sum(G[0:i]) so S[0] = 0, S[L] = sum(G[0:L])
        prefix_sum = torch.zeros(chunk_len + 1, self.d_proj, device=device, dtype=G.dtype)
        prefix_sum[1:chunk_len + 1] = torch.cumsum(G[:chunk_len], dim=0)

        # mean_pool[n] = (S[end_n] - S[start_n]) / k_n
        sum_interior = prefix_sum[ends] - prefix_sum[starts]  # [N, D_proj]
        mean_pool = sum_interior / pep_lens.unsqueeze(1).float().clamp(min=1)

        # 2. In-span endpoints: [G[start], G[end-1]]
        ep_left = G[starts]        # [N, D_proj]
        ep_right = G[ends - 1]     # [N, D_proj]

        # 3. Boundary flanks: [G[start-1], G[end]]
        # Handle boundary: start==0 → pad_left, end==chunk_len → pad_right
        fl_left_indices = starts - 1  # may be -1
        fl_right_indices = ends       # may be chunk_len

        # Build flank-left: pad_left for start==0, else G[start-1]
        fl_left = torch.zeros(N, self.d_proj, device=device, dtype=G.dtype)
        at_left_boundary = (starts == 0)
        if at_left_boundary.any():
            fl_left[at_left_boundary] = self.pad_left.unsqueeze(0)
        interior_left = ~at_left_boundary
        if interior_left.any():
            fl_left[interior_left] = G[fl_left_indices[interior_left]]

        # Build flank-right: pad_right for end==chunk_len, else G[end]
        fl_right = torch.zeros(N, self.d_proj, device=device, dtype=G.dtype)
        at_right_boundary = (ends == chunk_len)
        if at_right_boundary.any():
            fl_right[at_right_boundary] = self.pad_right.unsqueeze(0)
        interior_right = ~at_right_boundary
        if interior_right.any():
            fl_right[interior_right] = G[fl_right_indices[interior_right]]

        # 4. Length embedding (with debug guard for illegal lengths)
        raw_len_idx = pep_lens - self.length_offset
        if torch.any((raw_len_idx < 0) | (raw_len_idx > self.max_k - self.min_k)):
            bad = pep_lens[(raw_len_idx < 0) | (raw_len_idx > self.max_k - self.min_k)]
            raise ValueError(
                f"Span lengths outside [{self.min_k}, {self.max_k}]: {bad.tolist()}"
            )
        len_emb = self.length_embedding(raw_len_idx)  # [N, length_emb_dim]

        # 5. Allele embedding
        allele_emb = self.allele_embedding(allele_idx)  # [N, allele_emb_dim]

        # Concatenate: codemap §6.3 order
        phi = torch.cat([
            mean_pool,    # [N, D_proj]
            ep_left,      # [N, D_proj]
            ep_right,     # [N, D_proj]
            fl_left,      # [N, D_proj]
            fl_right,     # [N, D_proj]
            len_emb,      # [N, length_emb_dim]
            allele_emb,   # [N, allele_emb_dim]
        ], dim=1)  # [N, D_phi]

        if self.use_core_scorer:
            # Per-residue core logit -> prefix-sum -> logsumexp over the span's
            # 9-mer sub-windows -> one appended feature. Lets the dominant ~9-mer
            # binding core drive the window score instead of the mean over the span.
            c = self.core_scorer(G[:chunk_len]).squeeze(-1)  # [chunk_len]
            prefix_c = torch.zeros(chunk_len + 1, device=G.device, dtype=G.dtype)
            prefix_c[1:] = torch.cumsum(c, dim=0)
            core_feat = self._pool_9mer_cores(prefix_c, starts, ends, chunk_len)
            phi = torch.cat([phi, core_feat.unsqueeze(-1)], dim=1)

        return phi

    def forward_batched(
        self,
        G: torch.Tensor,
        spans: torch.Tensor,
        chunk_len: int,
        allele_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Batched ``forward`` over K candidates sharing one span template.

        Numerically identical to running :meth:`forward` once per candidate (up
        to float reduction reorder), but adds a leading candidate axis ``K`` so
        the prefix-sum pool, endpoint/flank gathers, and embeddings are computed
        once for all K. ``spans``/``chunk_len``/``allele_idx`` are shared across
        the K candidates (the RF same-length candidate case), so only ``G`` carries
        the ``K`` dimension.

        Args:
            G: [K, L, D_proj] projected embeddings for K candidates of one length.
            spans: [N, 2] int tensor of (start, end), 0b half-open, shared by all K.
            chunk_len: residue count shared by all K candidates.
            allele_idx: [N] int tensor of allele indices, shared by all K.

        Returns:
            phi: [K, N, D_phi] feature vectors.
        """
        if G.dim() != 3:
            raise ValueError(f"forward_batched expects G [K, L, D_proj], got {tuple(G.shape)}")
        K = G.shape[0]
        N = spans.shape[0]
        device = G.device

        starts = spans[:, 0]            # [N]
        ends = spans[:, 1]             # [N]
        pep_lens = ends - starts        # [N]

        # 1. Mean-pooled interior via batched prefix-sum over the L axis.
        prefix_sum = torch.zeros(K, chunk_len + 1, self.d_proj, device=device, dtype=G.dtype)
        prefix_sum[:, 1:chunk_len + 1] = torch.cumsum(G[:, :chunk_len], dim=1)
        sum_interior = prefix_sum[:, ends] - prefix_sum[:, starts]  # [K, N, D_proj]
        mean_pool = sum_interior / pep_lens.view(1, N, 1).float().clamp(min=1)

        # 2. In-span endpoints.
        ep_left = G[:, starts]          # [K, N, D_proj]
        ep_right = G[:, ends - 1]       # [K, N, D_proj]

        # 3. Boundary flanks. The boundary masks are span-only (shared across K);
        # clamp the gather index for boundary spans and overwrite them with the
        # learnable pad vectors so the result matches the per-candidate forward.
        at_left_boundary = (starts == 0)        # [N]
        at_right_boundary = (ends == chunk_len)  # [N]
        fl_left = G[:, (starts - 1).clamp(min=0)]          # [K, N, D_proj]
        fl_right = G[:, ends.clamp(max=chunk_len - 1)]     # [K, N, D_proj]
        if at_left_boundary.any():
            fl_left[:, at_left_boundary] = self.pad_left
        if at_right_boundary.any():
            fl_right[:, at_right_boundary] = self.pad_right

        # 4. Length embedding (same validity guard as forward).
        raw_len_idx = pep_lens - self.length_offset
        if torch.any((raw_len_idx < 0) | (raw_len_idx > self.max_k - self.min_k)):
            bad = pep_lens[(raw_len_idx < 0) | (raw_len_idx > self.max_k - self.min_k)]
            raise ValueError(
                f"Span lengths outside [{self.min_k}, {self.max_k}]: {bad.tolist()}"
            )
        len_emb = self.length_embedding(raw_len_idx)        # [N, length_emb_dim]
        len_emb = len_emb.unsqueeze(0).expand(K, -1, -1)    # [K, N, length_emb_dim]

        # 5. Allele embedding.
        allele_emb = self.allele_embedding(allele_idx)      # [N, allele_emb_dim]
        allele_emb = allele_emb.unsqueeze(0).expand(K, -1, -1)  # [K, N, allele_emb_dim]

        phi = torch.cat([
            mean_pool, ep_left, ep_right, fl_left, fl_right, len_emb, allele_emb,
        ], dim=2)  # [K, N, D_phi]
        return phi


# ── Scorer MLP ──────────────────────────────────────────────────────────────

class ScorerMLP(nn.Module):
    """Span scorer with cosine similarity and learnable logit scale.

    Architecture: D_phi → hidden → L2-normalize → cosine(h, prototype) × logit_scale.
    Logit output is bounded to [-logit_scale, logit_scale], preventing
    unbounded logit growth that causes val-loss explosion in InfoNCE.
    """

    def __init__(
        self,
        d_phi: int,
        hidden_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        logit_scale_init: float = 10.0,
        logit_scale_max: float = 20.0,
    ):
        super().__init__()
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.hidden = nn.Sequential(
            nn.Linear(d_phi, hidden_dim),
            act_fn,
            nn.Dropout(dropout),
        )
        self.prototype = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.log_logit_scale = nn.Parameter(
            torch.tensor(float(logit_scale_init)).log()
        )
        self.logit_scale_max = logit_scale_max

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """Score span features.

        Args:
            phi: [N, D_phi]
        Returns:
            z: [N] logits in [-logit_scale, logit_scale]
        """
        h = self.hidden(phi)
        h_norm = F.normalize(h, dim=-1)
        w_norm = F.normalize(self.prototype, dim=0)
        similarity = h_norm @ w_norm
        logit_scale = self.log_logit_scale.exp().clamp(max=self.logit_scale_max)
        return logit_scale * similarity


# ── Boundary (exact) head ────────────────────────────────────────────────────

class BoundaryHeadMLP(nn.Module):
    """Small per-window head producing an exactness correction b(s, e).

    Wave-4 dual-head: ``z_exact = stopgrad(z_region) + b_boundary``. The input is
    a DETACHED subset of the span feature phi (boundary flanks + length), so the
    exact objective's gradient reaches only these parameters — never the shared
    (trainable) projection / span-feature / scorer trunk that defines the
    per-residue landscape the downstream Reference Flow consumes.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats).squeeze(-1)


# ── Full Model ──────────────────────────────────────────────────────────────

class EpitopeScorer(nn.Module):
    """Full epitope scoring model: encoder → projection → span features → scorer.

    In training, operates at chunk level (no full-protein stitching).
    Each chunk is processed independently through the pipeline.
    """

    def __init__(
        self,
        encoder: FrozenESMEncoder | nn.Module,
        d_enc: int = 1280,
        d_proj: int = 128,
        length_emb_dim: int = 16,
        allele_emb_dim: int = 16,
        min_k: int = 12,
        max_k: int = 25,
        n_alleles: int = 1,
        scorer_hidden_dim: int = 256,
        scorer_activation: str = "gelu",
        scorer_dropout: float = 0.3,
        logit_scale_init: float = 10.0,
        logit_scale_max: float = 20.0,
        projection_layer_norm: bool = True,
        pad_left_init: str = "zeros",
        pad_right_init: str = "zeros",
        enable_boundary_head: bool = False,
        boundary_head_hidden_dim: int = 64,
        boundary_head_dropout: float = 0.1,
        use_core_scorer: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.projection = ProjectionHead(d_enc, d_proj, layer_norm=projection_layer_norm)
        self.span_features = SpanFeatureBuilder(
            d_proj=d_proj,
            length_emb_dim=length_emb_dim,
            allele_emb_dim=allele_emb_dim,
            min_k=min_k,
            max_k=max_k,
            n_alleles=n_alleles,
            pad_left_init=pad_left_init,
            pad_right_init=pad_right_init,
            use_core_scorer=use_core_scorer,
        )
        self.scorer = ScorerMLP(
            d_phi=self.span_features.d_phi,
            hidden_dim=scorer_hidden_dim,
            activation=scorer_activation,
            dropout=scorer_dropout,
            logit_scale_init=logit_scale_init,
            logit_scale_max=logit_scale_max,
        )
        # Wave-4 dual-head: constructed ONLY when enabled, so a default model is
        # bit-for-bit legacy (no extra state_dict keys, no extra init RNG draws).
        self._d_proj = d_proj
        self._length_emb_dim = length_emb_dim
        self.enable_boundary_head = enable_boundary_head
        if enable_boundary_head:
            self.boundary_head = BoundaryHeadMLP(
                input_dim=2 * d_proj + length_emb_dim,
                hidden_dim=boundary_head_hidden_dim,
                dropout=boundary_head_dropout,
            )

    @property
    def d_phi(self) -> int:
        return self.span_features.d_phi

    def encode_and_project(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and project a batch of chunks.

        Args:
            token_ids: [B, T]
            attention_mask: [B, T]

        Returns:
            G: [B, L_max, D_proj] projected residue embeddings
            lengths: [B] residue counts
        """
        H, lengths = self.encoder(token_ids, attention_mask)
        G = self.projection(H)
        return G, lengths

    def _extract_boundary_features(self, phi: torch.Tensor) -> torch.Tensor:
        """Slice the boundary-flank + length features from phi and DETACH them.

        phi concat order (SpanFeatureBuilder): [mean_pool, ep_left, ep_right,
        fl_left, fl_right, len_emb, allele_emb]. We take fl_left, fl_right,
        len_emb. Detaching the input is load-bearing: it blocks the exact loss
        from leaking into the (trainable) projection / span-feature trunk.
        """
        d = self._d_proj
        le = self._length_emb_dim
        fl_left = phi[:, 3 * d:4 * d]
        fl_right = phi[:, 4 * d:5 * d]
        len_emb = phi[:, 5 * d:5 * d + le]
        return torch.cat([fl_left, fl_right, len_emb], dim=-1).detach()

    def score_spans(
        self,
        G: torch.Tensor,
        chunk_len: int,
        spans: torch.Tensor,
        allele_idx: torch.Tensor,
        return_dual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Score spans for a single chunk.

        Args:
            G: [L, D_proj] projected embeddings (single chunk, no batch dim).
            chunk_len: actual residue count.
            spans: [N, 2] (start, end) in chunk-local coords.
            allele_idx: [N] allele indices.
            return_dual: when True AND the boundary head is enabled, also return
                the exact readout ``z_exact = stopgrad(z_region) + b_boundary``.

        Returns:
            ``z_region: [N]`` (default / legacy), or ``(z_region, z_exact)`` when
            ``return_dual`` and the boundary head is enabled.
        """
        phi = self.span_features(G, spans, chunk_len, allele_idx)
        z_region = self.scorer(phi)
        if return_dual and self.enable_boundary_head:
            feats = self._extract_boundary_features(phi)  # already detached
            z_exact = z_region.detach() + self.boundary_head(feats)
            return z_region, z_exact
        return z_region

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        spans_list: list[torch.Tensor],
        allele_idx_list: list[torch.Tensor],
        chunk_lengths: torch.Tensor,
        return_dual: bool = False,
    ) -> list[torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]]:
        """Full forward: encode batch of chunks, score per-chunk spans.

        Args:
            token_ids: [B, T]
            attention_mask: [B, T]
            spans_list: list of B tensors, each [N_i, 2] chunk-local spans.
            allele_idx_list: list of B tensors, each [N_i] allele indices.
            chunk_lengths: [B] residue counts per chunk.
            return_dual: thread to ``score_spans`` — when True and the boundary
                head is enabled, each element is ``(z_region, z_exact)``.

        Returns:
            list of B ``[N_i]`` logits, or list of B ``(z_region, z_exact)`` pairs
            when ``return_dual`` and the boundary head is enabled.
        """
        G, lengths = self.encode_and_project(token_ids, attention_mask)
        B = G.shape[0]

        # Cross-validate encoder-returned lengths vs external chunk_lengths
        if not torch.equal(lengths, chunk_lengths):
            raise RuntimeError(
                f"Encoder lengths {lengths.tolist()} != chunk_lengths {chunk_lengths.tolist()}. "
                "Tokenization/chunking mismatch."
            )

        logits_list = []
        for i in range(B):
            L_i = int(chunk_lengths[i].item())
            G_i = G[i]  # [L_max, D_proj] — only first L_i are valid
            logits_i = self.score_spans(
                G_i, L_i, spans_list[i], allele_idx_list[i], return_dual=return_dual,
            )
            logits_list.append(logits_i)

        return logits_list
