"""Encoder factory and alternative encoder implementations for Module H ablation.

Provides:
  - AATokenizer: lightweight amino-acid tokenizer for non-ESM encoders
  - DilatedCNNEncoder (E1): local-context encoder with ~121 AA receptive field
  - ShallowTransformerEncoder (E2): 4-layer global-context encoder
  - build_encoder(): factory that returns (encoder, tokenizer) by encoder_type
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── AA Tokenizer (for non-ESM encoders) ─────────────────────────────────────

class AATokenizer:
    """Lightweight amino-acid tokenizer with BOS/EOS/PAD semantics.

    Token layout matches ESMTokenizer contract:
      0 = BOS (CLS)
      1 = PAD
      2 = EOS
      3 = UNK
      4..23 = standard 20 amino acids
    """

    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
    VOCAB_SIZE = 24  # BOS, PAD, EOS, UNK + 20 AA

    def __init__(self):
        self.cls_idx = 0   # BOS
        self.pad_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        self._aa_map: dict[str, int] = {aa: i + 4 for i, aa in enumerate(self.AA_ORDER)}

    def __call__(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize batch of AA sequences.

        Returns dict with token_ids [B, T] and attention_mask [B, T].
        """
        max_len = max(len(s) for s in sequences) + 2  # +BOS +EOS
        B = len(sequences)
        token_ids = torch.full((B, max_len), self.pad_idx, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.bool)

        for i, seq in enumerate(sequences):
            tokens = [self.cls_idx]
            for ch in seq:
                tokens.append(self._aa_map.get(ch, self.unk_idx))
            tokens.append(self.eos_idx)
            t = len(tokens)
            token_ids[i, :t] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[i, :t] = True

        return {"token_ids": token_ids, "attention_mask": attention_mask}


# ── E1: Dilated CNN Encoder ──────────────────────────────────────────────────

class _ResidualDilatedBlock(nn.Module):
    """Single residual block: Conv1D (dilated) → BN → GELU → Dropout + skip."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal-style left padding
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation, padding=0,  # manual padding for mask compat
        )
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self._padding = padding

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L] feature tensor
            mask: [B, L] bool mask (True = valid)
        Returns:
            [B, C, L] with residual connection
        """
        # Zero out padded positions before convolution
        x_masked = x * mask.unsqueeze(1).float()

        # Symmetric padding to keep length unchanged
        pad_left = self._padding // 2
        pad_right = self._padding - pad_left
        h = F.pad(x_masked, (pad_left, pad_right), value=0.0)

        h = self.conv(h)
        h = self.bn(h)
        h = F.gelu(h)
        h = self.dropout(h)

        # Zero out padded positions in output
        h = h * mask.unsqueeze(1).float()

        return x + h  # residual


class DilatedCNNEncoder(nn.Module):
    """E1: Dilated 1D CNN encoder for local-context epitope prediction.

    Token embedding → stack of residual dilated conv blocks → output.
    Effective receptive field = 1 + sum((kernel_size - 1) * dilation_i).
    With k=5, dilations=[1,2,4,8,8,4,2,1]: RF = 1 + 8*4 = 121 AA.
    """

    def __init__(
        self,
        d_enc: int = 256,
        token_emb_dim: int = 256,
        n_blocks: int = 8,
        kernel_size: int = 5,
        dilations: list[int] | None = None,
        hidden_channels: int = 256,
        block_dropout: float = 0.1,
        vocab_size: int = AATokenizer.VOCAB_SIZE,
    ):
        super().__init__()
        self.d_enc = d_enc

        if dilations is None:
            dilations = [1, 2, 4, 8, 8, 4, 2, 1]
        assert len(dilations) == n_blocks

        self.token_embedding = nn.Embedding(vocab_size, token_emb_dim, padding_idx=1)

        # Project embedding dim to hidden channels if needed
        self.input_proj = (
            nn.Linear(token_emb_dim, hidden_channels)
            if token_emb_dim != hidden_channels
            else nn.Identity()
        )

        self.blocks = nn.ModuleList([
            _ResidualDilatedBlock(hidden_channels, kernel_size, d, block_dropout)
            for d in dilations
        ])

        # Project to output d_enc if needed
        self.output_proj = (
            nn.Linear(hidden_channels, d_enc)
            if hidden_channels != d_enc
            else nn.Identity()
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            token_ids: [B, T] with BOS/EOS/PAD
            attention_mask: [B, T] bool mask

        Returns:
            embeddings: [B, L_max, d_enc] residue embeddings (BOS/EOS stripped)
            lengths: [B] residue counts
        """
        # Compute residue lengths (strip BOS + EOS)
        real_counts = attention_mask.sum(dim=1)  # [B]
        residue_lengths = real_counts - 2

        # Extract residue tokens (skip BOS at position 0, skip EOS)
        B, T = token_ids.shape
        L_max = int(residue_lengths.max().item())

        # Build residue-only token_ids and mask
        residue_ids = torch.full((B, L_max), 1, dtype=torch.long, device=token_ids.device)  # PAD
        residue_mask = torch.zeros(B, L_max, dtype=torch.bool, device=token_ids.device)
        for i in range(B):
            L_i = int(residue_lengths[i].item())
            residue_ids[i, :L_i] = token_ids[i, 1:1 + L_i]  # skip BOS
            residue_mask[i, :L_i] = True

        # Embed → project → conv blocks
        x = self.token_embedding(residue_ids)  # [B, L_max, emb_dim]
        x = self.input_proj(x)                 # [B, L_max, hidden]
        x = x.transpose(1, 2)                  # [B, hidden, L_max] for conv

        for block in self.blocks:
            x = block(x, residue_mask)

        x = x.transpose(1, 2)                  # [B, L_max, hidden]
        embeddings = self.output_proj(x)        # [B, L_max, d_enc]

        # Zero out padded positions
        embeddings = embeddings * residue_mask.unsqueeze(-1).float()

        return embeddings, residue_lengths


# ── C1: Multi-Scale Dilated CNN Encoder ──────────────────────────────────────

class _MultiScaleResidualBlock(nn.Module):
    """Residual block with parallel multi-scale conv branches.

    Parallel branches with kernel sizes {3, 5, 9} (configurable), each with
    per-branch dilation → BN → GELU → concat → 1x1 fusion → residual.
    """

    def __init__(
        self,
        channels: int,
        branch_channels: int,
        branch_kernels: list[int],
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        n_branches = len(branch_kernels)
        self.branches = nn.ModuleList()
        self._paddings = []
        for k in branch_kernels:
            padding = (k - 1) * dilation
            self._paddings.append(padding)
            self.branches.append(nn.Sequential(
                nn.Conv1d(channels, branch_channels, k, dilation=dilation, padding=0),
                nn.BatchNorm1d(branch_channels),
                nn.GELU(),
            ))
        # 1x1 fusion: concat of all branches → back to channels
        self.fusion = nn.Conv1d(n_branches * branch_channels, channels, 1)
        self.bn_out = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L] feature tensor
            mask: [B, L] bool mask (True = valid)
        Returns:
            [B, C, L] with residual connection
        """
        x_masked = x * mask.unsqueeze(1).float()
        branch_outs = []
        for branch, padding in zip(self.branches, self._paddings):
            pad_left = padding // 2
            pad_right = padding - pad_left
            h = F.pad(x_masked, (pad_left, pad_right), value=0.0)
            branch_outs.append(branch(h))

        # Concat along channel dim → fuse
        h = torch.cat(branch_outs, dim=1)  # [B, n_branches * branch_channels, L]
        h = self.fusion(h)                  # [B, channels, L]
        h = self.bn_out(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = h * mask.unsqueeze(1).float()

        return x + h  # residual


class MultiScaleDilatedCNNEncoder(nn.Module):
    """C1: Multi-scale dilated CNN encoder with parallel conv branches.

    Same IO contract as DilatedCNNEncoder — BOS/EOS stripping,
    forward(token_ids, attention_mask) → (embeddings, lengths).
    Uses _MultiScaleResidualBlock with parallel {3, 5, 9} kernel branches.
    """

    def __init__(
        self,
        d_enc: int = 256,
        token_emb_dim: int = 256,
        n_blocks: int = 8,
        dilations: list[int] | None = None,
        hidden_channels: int = 256,
        branch_channels: int = 64,
        branch_kernels: list[int] | None = None,
        block_dropout: float = 0.1,
        vocab_size: int = AATokenizer.VOCAB_SIZE,
    ):
        super().__init__()
        self.d_enc = d_enc

        if dilations is None:
            dilations = [1, 2, 4, 8, 8, 4, 2, 1]
        assert len(dilations) == n_blocks

        if branch_kernels is None:
            branch_kernels = [3, 5, 9]

        self.token_embedding = nn.Embedding(vocab_size, token_emb_dim, padding_idx=1)

        self.input_proj = (
            nn.Linear(token_emb_dim, hidden_channels)
            if token_emb_dim != hidden_channels
            else nn.Identity()
        )

        self.blocks = nn.ModuleList([
            _MultiScaleResidualBlock(hidden_channels, branch_channels, branch_kernels, d, block_dropout)
            for d in dilations
        ])

        self.output_proj = (
            nn.Linear(hidden_channels, d_enc)
            if hidden_channels != d_enc
            else nn.Identity()
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass — same contract as DilatedCNNEncoder.

        Args:
            token_ids: [B, T] with BOS/EOS/PAD
            attention_mask: [B, T] bool mask

        Returns:
            embeddings: [B, L_max, d_enc] residue embeddings (BOS/EOS stripped)
            lengths: [B] residue counts
        """
        # Compute residue lengths (strip BOS + EOS)
        real_counts = attention_mask.sum(dim=1)
        residue_lengths = real_counts - 2

        B, T = token_ids.shape
        L_max = int(residue_lengths.max().item())

        # Build residue-only token_ids and mask
        residue_ids = torch.full((B, L_max), 1, dtype=torch.long, device=token_ids.device)
        residue_mask = torch.zeros(B, L_max, dtype=torch.bool, device=token_ids.device)
        for i in range(B):
            L_i = int(residue_lengths[i].item())
            residue_ids[i, :L_i] = token_ids[i, 1:1 + L_i]
            residue_mask[i, :L_i] = True

        # Embed → project → conv blocks
        x = self.token_embedding(residue_ids)
        x = self.input_proj(x)
        x = x.transpose(1, 2)  # [B, hidden, L_max]

        for block in self.blocks:
            x = block(x, residue_mask)

        x = x.transpose(1, 2)  # [B, L_max, hidden]
        embeddings = self.output_proj(x)

        # Zero out padded positions
        embeddings = embeddings * residue_mask.unsqueeze(-1).float()

        return embeddings, residue_lengths


# ── E2: Shallow Transformer Encoder ─────────────────────────────────────────

class ShallowTransformerEncoder(nn.Module):
    """E2: 4-layer Transformer encoder with absolute positional embeddings.

    Token embedding + positional embedding → Transformer encoder → output.
    Uses key-padding mask to handle variable-length sequences.
    """

    def __init__(
        self,
        d_enc: int = 256,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1022,
        vocab_size: int = AATokenizer.VOCAB_SIZE,
    ):
        super().__init__()
        self.d_enc = d_enc

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stable training from scratch
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        # Project to output d_enc if needed
        self.output_proj = (
            nn.Linear(d_model, d_enc)
            if d_model != d_enc
            else nn.Identity()
        )

        self._max_seq_len = max_seq_len

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            token_ids: [B, T] with BOS/EOS/PAD
            attention_mask: [B, T] bool mask

        Returns:
            embeddings: [B, L_max, d_enc] residue embeddings (BOS/EOS stripped)
            lengths: [B] residue counts
        """
        # Compute residue lengths
        real_counts = attention_mask.sum(dim=1)  # [B]
        residue_lengths = real_counts - 2

        B, T = token_ids.shape
        L_max = int(residue_lengths.max().item())

        # Build residue-only tokens and mask
        residue_ids = torch.full((B, L_max), 1, dtype=torch.long, device=token_ids.device)
        residue_mask = torch.zeros(B, L_max, dtype=torch.bool, device=token_ids.device)
        for i in range(B):
            L_i = int(residue_lengths[i].item())
            residue_ids[i, :L_i] = token_ids[i, 1:1 + L_i]
            residue_mask[i, :L_i] = True

        # Embed tokens + positions
        tok_emb = self.token_embedding(residue_ids)  # [B, L_max, d_model]
        positions = torch.arange(L_max, device=token_ids.device).unsqueeze(0)  # [1, L_max]
        pos_emb = self.pos_embedding(positions)       # [1, L_max, d_model]
        x = self.emb_dropout(tok_emb + pos_emb)

        # Transformer expects src_key_padding_mask: True = IGNORE
        padding_mask = ~residue_mask  # [B, L_max], True where padded
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        embeddings = self.output_proj(x)  # [B, L_max, d_enc]

        # Zero out padded positions
        embeddings = embeddings * residue_mask.unsqueeze(-1).float()

        return embeddings, residue_lengths


# ── Encoder Factory ──────────────────────────────────────────────────────────

def build_encoder(
    encoder_type: str,
    d_enc: int,
    encoder_cfg: dict,
) -> tuple[nn.Module, AATokenizer]:
    """Build encoder + tokenizer by type.

    Args:
        encoder_type: one of 'esm2_frozen', 'dilated_cnn', 'shallow_transformer'
        d_enc: encoder output dimension
        encoder_cfg: type-specific config dict

    Returns:
        (encoder, tokenizer) tuple

    Note: 'esm2_frozen' is NOT handled here — it requires the esm package
    and is constructed directly in the training script. This factory covers
    the lightweight non-ESM encoders only.
    """
    tokenizer = AATokenizer()

    if encoder_type == "dilated_cnn":
        encoder = DilatedCNNEncoder(
            d_enc=d_enc,
            token_emb_dim=encoder_cfg["token_emb_dim"],
            n_blocks=encoder_cfg["n_blocks"],
            kernel_size=encoder_cfg["kernel_size"],
            dilations=encoder_cfg["dilations"],
            hidden_channels=encoder_cfg["hidden_channels"],
            block_dropout=encoder_cfg["block_dropout"],
        )
    elif encoder_type == "multiscale_cnn":
        encoder = MultiScaleDilatedCNNEncoder(
            d_enc=d_enc,
            token_emb_dim=encoder_cfg["token_emb_dim"],
            n_blocks=encoder_cfg["n_blocks"],
            dilations=encoder_cfg["dilations"],
            hidden_channels=encoder_cfg["hidden_channels"],
            branch_channels=encoder_cfg["branch_channels"],
            branch_kernels=encoder_cfg.get("branch_kernels"),
            block_dropout=encoder_cfg["block_dropout"],
        )
    elif encoder_type == "shallow_transformer":
        encoder = ShallowTransformerEncoder(
            d_enc=d_enc,
            d_model=encoder_cfg["d_model"],
            n_layers=encoder_cfg["n_layers"],
            n_heads=encoder_cfg["n_heads"],
            ffn_dim=encoder_cfg["ffn_dim"],
            dropout=encoder_cfg["dropout"],
            max_seq_len=encoder_cfg["max_seq_len"],
        )
    elif encoder_type == "esm2_frozen":
        raise ValueError(
            "esm2_frozen encoder must be constructed directly with FrozenESMEncoder. "
            "Use build_encoder() only for lightweight non-ESM encoders."
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return encoder, tokenizer
