"""
SEFFN: Spectral Enhanced Feed-Forward Network.

An FFT-based feed-forward network that replaces the standard MLP in transformer
encoder layers. It combines dilated depthwise convolution with frequency-domain
modulation via learnable FFT weights to capture global spatial structure at
near-zero additional parameter cost.

References:
    - Sun et al., "Retinexformer: One-stage Retinex-based Transformer for
      Low-light Image Enhancement," ICCV 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SpectralEnhancedFFN']


class SpectralEnhancedFFN(nn.Module):
    """
    Spectral Enhanced Feed-Forward Network.

    Pipeline: 1x1 proj_in -> dilated DW conv -> FFT modulation -> SiLU gate -> 1x1 proj_out

    Args:
        dim: Input/output channel dimension.
        ffn_expansion_factor: Channel expansion ratio (default: 2.0).
        bias: Whether to use bias in convolutions.
    """

    def __init__(self, dim, ffn_expansion_factor=2.0, bias=False):
        super(SpectralEnhancedFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3,
            stride=1, padding=2, groups=hidden_features * 2, bias=bias, dilation=2
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # Learnable frequency-domain modulation parameters
        self.fft_channel_weight = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))
        self.fft_channel_bias = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        hw = x.shape[-1]
        return x[..., t_pad[0]:hw - t_pad[1]]

    def forward(self, x):
        x_dtype = x.dtype
        x = self.project_in(x)
        x = self.dwconv(x)

        # FFT frequency-domain modulation
        x, pad_w = self.pad(x, 2)
        x = torch.fft.rfft2(x.float())
        x = self.fft_channel_weight * x + self.fft_channel_bias
        x = torch.fft.irfft2(x)
        x = self.unpad(x, pad_w)

        # SiLU gating
        x1, x2 = x.chunk(2, dim=1)
        x = F.silu(x1) * x2

        x = self.project_out(x.to(x_dtype))
        return x
