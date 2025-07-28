import math
import warnings
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# -----------------------------------------------------------------------------
# Simple MIL baselines (Unchanged)
# -----------------------------------------------------------------------------

class MeanPoolCox(nn.Module):
    """Mean-pooling MIL baseline followed by a linear Cox head."""
    def __init__(self, d: int):
        super().__init__()
        self.beta = nn.Linear(d, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        slide_vec = x.mean(dim=1)
        return self.beta(slide_vec).squeeze(-1)

class MaxPoolCox(nn.Module):
    """Max-pooling MIL baseline followed by a linear Cox head."""
    def __init__(self, d: int):
        super().__init__()
        self.beta = nn.Linear(d, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        slide_vec = x.max(dim=1).values
        return self.beta(slide_vec).squeeze(-1)

class AttnMILCox(nn.Module):
    """Gated-attention MIL (Ilse et al., 2018) with a Cox head."""
    def __init__(self, d: int, h: int = 256):
        super().__init__()
        self.V = nn.Linear(d, h)
        self.U = nn.Linear(d, h)
        self.w = nn.Linear(h, 1)
        self.beta = nn.Linear(d, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        att = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x)))
        att = torch.softmax(att, dim=1)
        slide_vec = (att * x).sum(dim=1)
        return self.beta(slide_vec).squeeze(-1)

class AttnMILNewCox(nn.Module):
    def __init__(self, d: int, h: int = 256, dropout: float = 0.25):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.V = nn.Linear(d, h)
        self.U = nn.Linear(d, h)
        self.w = nn.Linear(h, 1)
        self.beta = nn.Linear(d, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)
        att = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x)))
        att = torch.softmax(att, dim=1)
        slide_vec = (att * x).sum(dim=1)
        return self.beta(slide_vec).squeeze(-1)

# -----------------------------------------------------------------------------
# Multimodal Components
# -----------------------------------------------------------------------------

class AttnMILNewBase(nn.Module):
    """
    The AttnMILNew model adapted to return the aggregated slide-level
    feature vector, to be used as a component in a larger model.
    """
    def __init__(self, d: int, h: int = 256, dropout: float = 0.25):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.V = nn.Linear(d, h)
        self.U = nn.Linear(d, h)
        self.w = nn.Linear(h, 1)

    def forward(self, x: Tensor) -> Tensor:  # x: (B, N, d)
        # Pass features through the extractor first
        x = self.feature_extractor(x)  # Shape remains (B, N, d)

        # Gated-attention mechanism
        att = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))) # (B,N,1)
        att = torch.softmax(att, dim=1)
        
        # Aggregate patch features into a single slide vector
        slide_vec = (att * x).sum(dim=1)   # (B, d)
        
        return slide_vec

class MultimodalCox(nn.Module):
    """
    A late-fusion model for combining pathology slides and RNA-seq data.
    - Pathology Stream: Aggregates patch embeddings using AttnMILNewBase.
    - Genomic Stream: Processes a fixed-size RNA-seq latent vector with an MLP.
    - Fusion: Concatenates both feature vectors for a final survival prediction.
    """
    def __init__(self, d_path: int, d_rna: int = 640, dropout: float = 0.25):
        super().__init__()
        
        # 1. Pathology Stream (using the new base model)
        self.path_agg = AttnMILNewBase(d=d_path, dropout=dropout)
        
        # 2. Genomic Stream (a simple MLP for RNA features)
        self.rna_mlp = nn.Sequential(
            nn.Linear(d_rna, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(p=dropout)
        )
        
        # 3. Fusion Head for final prediction
        # The input dimension is the sum of the two stream output dimensions
        fusion_dim = d_path + 128
        self.fusion_head = nn.Linear(fusion_dim, 1, bias=False)

    def forward(self, x_path: Tensor, x_rna: Tensor) -> Tensor:
        # Process pathology stream -> (B, d_path)
        slide_vec = self.path_agg(x_path)
        
        # Process genomic stream -> (B, 128)
        rna_vec = self.rna_mlp(x_rna)
        
        # Concatenate features for late fusion -> (B, d_path + 128)
        fused_vec = torch.cat([slide_vec, rna_vec], dim=1)
        
        # Final prediction
        risk = self.fusion_head(fused_vec) # (B, 1)
        
        return risk.squeeze(-1) # (B,)


# -----------------------------------------------------------------------------
# Pyramid Positional Encoding Generator (PPEG) 
# -----------------------------------------------------------------------------

class PPEG(nn.Module):
    """Depth-wise 7/5/3-kernel convs used by TransMIL to inject position."""
    def __init__(self, d: int):
        super().__init__()
        self.k7 = nn.Conv2d(d, d, 7, padding=3, groups=d)
        self.k5 = nn.Conv2d(d, d, 5, padding=2, groups=d)
        self.k3 = nn.Conv2d(d, d, 3, padding=1, groups=d)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        cls, feat = x[:, :1], x[:, 1:]
        B, _, d = feat.shape
        feat = feat.transpose(1, 2).view(B, d, H, W)
        pos = feat + self.k7(feat) + self.k5(feat) + self.k3(feat)
        pos = pos.flatten(2).transpose(1, 2)
        return torch.cat([cls, pos], dim=1)

# -----------------------------------------------------------------------------
# TransMIL with top-k correlation & rectangular padding 
# -----------------------------------------------------------------------------

class TransMILCox(nn.Module):
    """Full TransMIL aggregator (PPEG + top-k correlation) with a Cox head."""
    def __init__(self, d: int, depth: int = 2, ff: int = 512, topk: int = 128):
        super().__init__()
        try:
            heads = next(h for h in range(min(16, d), 0, -1) if d % h == 0)
        except StopIteration:
            warnings.warn(f"embed dim {d} not divisible by any head <=16; using 1 head")
            heads = 1
        self.topk = topk
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, d))
        self.ppeg = PPEG(d)
        self.pre_norm = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, 256)
        self.k_proj = nn.Linear(d, 256)
        self.corr_norm = nn.LayerNorm(d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ff, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.ModuleList([encoder_layer for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1, bias=False))
        nn.init.trunc_normal_(self.cls_tok, std=0.02)

    def _topk_corr(self, x: Tensor) -> Tensor:
        q, k = self.q_proj(x), self.k_proj(x)
        sim = torch.einsum('bsd,btd->bst', q, k) / math.sqrt(q.size(-1))
        k_sel = min(self.topk, sim.size(-1))
        top_indices = sim.topk(k_sel, dim=-1).indices
        mask = torch.full_like(sim, -torch.inf)
        sim_sparse = mask.scatter(-1, top_indices, sim.gather(-1, top_indices))
        attn = torch.softmax(sim_sparse, dim=-1)
        return self.corr_norm(x + attn @ x)

    def forward(self, x: Tensor) -> Tensor:
        B, S, d = x.shape
        H = math.ceil(math.sqrt(S))
        W = math.ceil(S / H)
        num_patches_padded = H * W
        pad_len = num_patches_padded - S
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        x = torch.cat([self.cls_tok.expand(B, -1, -1), x], dim=1)
        x = self.ppeg(x, H, W)
        x = self.pre_norm(x)
        cls_tok, feat_tok = x[:, :1], x[:, 1:]
        feat_tok = self._topk_corr(feat_tok)
        x = torch.cat([cls_tok, feat_tok], dim=1)
        for layer in self.encoder:
            x = layer(x)
        return self.head(x[:, 0]).squeeze(-1)