import math
import warnings
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# -----------------------------------------------------------------------------
# Simple MIL baselines ---------------------------------------------------------
# -----------------------------------------------------------------------------

class PassthroughCox(nn.Module):
    """Mean‑pooling MIL baseline followed by a linear Cox head."""

    def __init__(self, d: int):
        super().__init__()
        self.beta = nn.Linear(d, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:  # x: (B, d)
        return self.beta(x) # (B,)

class MeanPoolCox(nn.Module):
    """Mean‑pooling MIL baseline followed by a linear Cox head."""

    def __init__(self, d: int):
        super().__init__()
        self.beta = nn.Linear(d, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:  # x: (B, N, d)
        slide_vec = x.mean(dim=1)            # (B, d)
        return self.beta(slide_vec).squeeze(-1)  # (B,)


class MaxPoolCox(nn.Module):
    """Max‑pooling MIL baseline followed by a linear Cox head."""

    def __init__(self, d: int):
        super().__init__()
        self.beta = nn.Linear(d, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:  # x: (B, N, d)
        slide_vec = x.max(dim=1).values      # (B, d)
        return self.beta(slide_vec).squeeze(-1)


class AttnMILCox(nn.Module):
    """Gated‑attention MIL (Ilse et al., 2018) with a Cox head."""

    def __init__(self, d: int, h: int = 256):
        super().__init__()
        self.V = nn.Linear(d, h)
        self.U = nn.Linear(d, h)
        self.w = nn.Linear(h, 1)
        self.beta = nn.Linear(d, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:  # x: (B, N, d)
        att = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x)))  # (B,N,1)
        att = torch.softmax(att, dim=1)
        slide_vec = (att * x).sum(dim=1)     # (B, d)
        return self.beta(slide_vec).squeeze(-1)

# -----------------------------------------------------------------------------
# Pyramid Positional Encoding Generator (PPEG) --------------------------------
# -----------------------------------------------------------------------------

class PPEG(nn.Module):
    """Depth‑wise 7/5/3‑kernel convs used by TransMIL to inject position."""

    def __init__(self, d: int):
        super().__init__()
        self.k7 = nn.Conv2d(d, d, 7, padding=3, groups=d)
        self.k5 = nn.Conv2d(d, d, 5, padding=2, groups=d)
        self.k3 = nn.Conv2d(d, d, 3, padding=1, groups=d)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:  # x: (B,S+1,d)
        cls, feat = x[:, :1], x[:, 1:]                       # (B,1,d) + (B,S,d)
        B, _, d = feat.shape
        feat = feat.transpose(1, 2).view(B, d, H, W)
        pos = feat + self.k7(feat) + self.k5(feat) + self.k3(feat)
        pos = pos.flatten(2).transpose(1, 2)                # back to (B,S,d)
        return torch.cat([cls, pos], dim=1)

# -----------------------------------------------------------------------------
# TransMIL with top‑k correlation & rectangular padding -----------------------
# -----------------------------------------------------------------------------

class TransMILCox(nn.Module):
    """Full TransMIL aggregator (PPEG + top‑k correlation) with a Cox head.

    Parameters
    ----------
    d : int
        Patch embedding dimension (e.g. 768 for CTransPath, 1536 for UNI).
    depth : int, optional
        Number of Transformer encoder layers (default = 2).
    ff : int, optional
        Feed‑forward network width inside each encoder layer (default = 1024).
    topk : int, optional
        Top‑k tokens to keep in the correlation attention (default = 256).
    """

    def __init__(self, d: int, depth: int = 2, ff: int = 512, topk: int = 128):
        super().__init__()
        # choose n_heads ≤16 that divides d
        try:
            heads = next(h for h in range(min(16, d), 0, -1) if d % h == 0)
        except StopIteration:
            warnings.warn(f"embed dim {d} not divisible by any head ≤16; using 1 head")
            heads = 1
        self.topk = topk

        self.cls_tok = nn.Parameter(torch.zeros(1, 1, d))
        self.ppeg = PPEG(d)
        self.pre_norm = nn.LayerNorm(d)

        # Low‑rank projections for correlation block
        self.q_proj = nn.Linear(d, 256)
        self.k_proj = nn.Linear(d, 256)
        self.corr_norm = nn.LayerNorm(d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=ff,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.ModuleList([encoder_layer for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1, bias=False))
        nn.init.trunc_normal_(self.cls_tok, std=0.02)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _topk_corr(self, x: Tensor) -> Tensor:  # (B,S,d)
        q, k = self.q_proj(x), self.k_proj(x)    # (B,S,256)
        sim = torch.einsum('bsd,btd->bst', q, k) / math.sqrt(q.size(-1))
        k_sel = min(self.topk, sim.size(-1))
        top_indices = sim.topk(k_sel, dim=-1).indices
        mask = torch.full_like(sim, -torch.inf)
        sim_sparse = mask.scatter(-1, top_indices, sim.gather(-1, top_indices))
        attn = torch.softmax(sim_sparse, dim=-1)
        return self.corr_norm(x + attn @ x)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:  # x: (B,N,d) padded
        B, S, d = x.shape
        # compute rectangle dims common to the batch (bags are padded to same S)
        H = math.ceil(math.sqrt(S))
        W = math.ceil(S / H)
    
        # --- Start of Fix ---
        # Calculate the padding required
        num_patches_padded = H * W
        pad_len = num_patches_padded - S
    
        # Pad the feature tensor 'x'.
        # F.pad format is (pad_left, pad_right, pad_top, pad_bottom, ...)
        # We are padding the sequence dimension (dim=1) only on the right.
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        # --- End of Fix ---
    
        # prepend CLS & add positional encoding
        x = torch.cat([self.cls_tok.expand(B, -1, -1), x], dim=1)  # (B,S_padded+1,d)
        x = self.ppeg(x, H, W)
        x = self.pre_norm(x)
    
        cls_tok, feat_tok = x[:, :1], x[:, 1:]
        feat_tok = self._topk_corr(feat_tok)
        x = torch.cat([cls_tok, feat_tok], dim=1)
    
    
        for layer in self.encoder:
            x = layer(x)
    
        return self.head(x[:, 0]).squeeze(-1)
