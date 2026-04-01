"""
AIFI_SEFFN: Attention-based Intra-scale Feature Interaction with SEFFN.

Replaces the standard FFN in the RT-DETR AIFI encoder module with
SpectralEnhancedFFN (SEFFN), enabling global frequency-domain feature
enhancement at negligible additional cost.

This module adds 2D sine-cosine positional encoding and wraps multi-head
self-attention + SEFFN into a single transformer encoder layer.
"""

import torch
import torch.nn as nn

from .seffn import SpectralEnhancedFFN

__all__ = ['TransformerEncoderLayer_SEFFN', 'AIFI_SEFFN']


class TransformerEncoderLayer_SEFFN(nn.Module):
    """Transformer encoder layer with SEFFN replacing the standard MLP FFN."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0,
                 act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        self.ffn = SpectralEnhancedFFN(c1, 2.0, False)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        B, C, H, W = src.size()
        src = src.flatten(2).permute(0, 2, 1)
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask,
                       key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.ffn(
            src2.permute(0, 2, 1).view([B, C, H, W]).contiguous()
        ).flatten(2).permute(0, 2, 1)
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        B, C, H, W = src.size()
        src2 = self.norm1(src.flatten(2).permute(0, 2, 1))
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask,
                       key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.ffn(
            src2.permute(0, 2, 1).view([B, C, H, W]).contiguous()
        ).flatten(2).permute(0, 2, 1)
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI_SEFFN(TransformerEncoderLayer_SEFFN):
    """AIFI module with SEFFN: adds 2D sine-cosine positional encoding."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0,
                 act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        x = super().forward(x, pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat(
            [torch.sin(out_w), torch.cos(out_w),
             torch.sin(out_h), torch.cos(out_h)], 1
        )[None]
