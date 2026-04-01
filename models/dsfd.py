"""
DSFD: Dimension-aware Selective Fusion Decoder (also referred to as
Dimension-Aware Selective Integration with Focusing Diffusion in the paper).

A three-input multi-scale feature fusion module that replaces conventional
concatenation/addition with channel-grouped adaptive weighted fusion via a
soft-gating mechanism. Deployed at feature pyramid fusion nodes to form a
"focusing-diffusion" information flow pattern.

The module divides features into 4 channel groups and uses the Bag (Boundary-
Aware Guided) attention to fuse high-resolution and low-resolution features
with current-scale features acting as soft gates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Bag', 'DSFD']


class Bag(nn.Module):
    """Boundary-Aware Guided fusion: sigmoid soft-gating between two feature sources."""

    def __init__(self):
        super(Bag, self).__init__()

    def forward(self, p, i, d):
        """
        Args:
            p: Features from one scale (e.g., low-resolution).
            i: Features from another scale (e.g., high-resolution).
            d: Current-scale features used as soft gate (after sigmoid).
        Returns:
            Adaptively fused features: sigmoid(d) * p + (1 - sigmoid(d)) * i
        """
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i


class DSFD(nn.Module):
    """
    Dimension-aware Selective Fusion Decoder.

    Takes three feature maps from different scales (low, current, high),
    projects them to uniform resolution and channels, divides into 4 groups
    along the channel dimension, and applies Bag attention fusion per group.

    Args:
        in_features: List of input channel dimensions [low_ch, current_ch, high_ch].
        out_features: Output channel dimension.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.bag = Bag()
        self.tail_conv = nn.Conv2d(out_features, out_features, 1)
        self.conv = nn.Conv2d(out_features // 2, out_features // 4, 1)
        self.bns = nn.BatchNorm2d(out_features)

        # Projection convolutions for each input scale
        self.skips = nn.Conv2d(in_features[1], out_features, 1)      # current scale
        self.skips_2 = nn.Conv2d(in_features[0], out_features, 1)    # low-resolution (upsample)
        self.skips_3 = nn.Conv2d(in_features[2], out_features,       # high-resolution (downsample)
                                 kernel_size=3, stride=2, dilation=2, padding=2)
        self.silu = nn.SiLU()

    def forward(self, x_list):
        x_low, x, x_high = x_list

        if x_high is not None:
            x_high = self.skips_3(x_high)
            x_high = torch.chunk(x_high, 4, dim=1)

        if x_low is not None:
            x_low = self.skips_2(x_low)
            x_low = F.interpolate(x_low, size=[x.size(2), x.size(3)],
                                  mode='bilinear', align_corners=True)
            x_low = torch.chunk(x_low, 4, dim=1)

        x = self.skips(x)
        x_skip = x
        x = torch.chunk(x, 4, dim=1)

        if x_high is None:
            x0 = self.conv(torch.cat((x[0], x_low[0]), dim=1))
            x1 = self.conv(torch.cat((x[1], x_low[1]), dim=1))
            x2 = self.conv(torch.cat((x[2], x_low[2]), dim=1))
            x3 = self.conv(torch.cat((x[3], x_low[3]), dim=1))
        elif x_low is None:
            x0 = self.conv(torch.cat((x[0], x_high[0]), dim=1))
            x1 = self.conv(torch.cat((x[0], x_high[1]), dim=1))
            x2 = self.conv(torch.cat((x[0], x_high[2]), dim=1))
            x3 = self.conv(torch.cat((x[0], x_high[3]), dim=1))
        else:
            # Full three-input fusion via Bag attention
            x0 = self.bag(x_low[0], x_high[0], x[0])
            x1 = self.bag(x_low[1], x_high[1], x[1])
            x2 = self.bag(x_low[2], x_high[2], x[2])
            x3 = self.bag(x_low[3], x_high[3], x[3])

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        x += x_skip
        x = self.bns(x)
        x = self.silu(x)
        return x
