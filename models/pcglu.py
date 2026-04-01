"""
PCGLU: Partial Conv + Convolutional GLU backbone module for RT-DETR.

Replaces the standard BasicBlock in the RT-DETR-R18 backbone with a combined
"PConv spatial mixing + CGLU gated channel mixing" design, reducing parameters
while improving joint spatial-channel feature representation.

References:
    - PConv: Chen et al., "Run, Don't Walk: Chasing Higher FLOPS for Faster
      Neural Networks," CVPR 2023.
    - ConvolutionalGLU: Rao et al., "Hornet: Efficient High-Order Spatial
      Interactions with Recursive Gated Convolutions," NeurIPS 2022.
"""

import torch
import torch.nn as nn
from timm.layers import DropPath

__all__ = ['Partial_conv3', 'ConvolutionalGLU', 'Faster_Block_CGLU', 'PCGLU']


class Partial_conv3(nn.Module):
    """Partial Convolution that applies 3x3 conv to only 1/n_div of the input channels."""

    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class ConvolutionalGLU(nn.Module):
    """Convolutional Gated Linear Unit for channel mixing."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1,
                      padding=1, bias=True, groups=hidden_features),
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x


class Faster_Block_CGLU(nn.Module):
    """Combined PConv spatial mixing + CGLU channel mixing block with DropPath."""

    def __init__(self, inc, dim, n_div=4, mlp_ratio=2, drop_path=0.1,
                 layer_scale_init_value=0.0, pconv_fw_type='split_cat'):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        self.mlp = ConvolutionalGLU(dim)
        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type)

        self.adjust_channel = None
        if inc != dim:
            # Uses ultralytics Conv (BN+act) — replace with your own if needed
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(inc, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.SiLU(inplace=True)
            )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class PCGLU(nn.Module):
    """
    PCGLU backbone block: extends the standard ResNet BasicBlock by replacing
    the second 3x3 conv (branch2b) with Faster_Block_CGLU.

    To integrate into ultralytics, this class should inherit from BasicBlock
    (ultralytics.nn.modules.block.BasicBlock). See docs/integration.md.

    Args:
        ch_in: Input channels.
        ch_out: Output channels.
        stride: Convolution stride for downsampling.
        shortcut: Whether to use residual shortcut.
        act: Activation type (default: 'relu').
        variant: BasicBlock variant (default: 'd').
    """
    expansion = 1  # Must match BasicBlock.expansion for Blocks wrapper

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()
        # NOTE: In the full integration, this inherits from BasicBlock which
        # provides branch1 (shortcut) and branch2a (first 3x3 conv).
        # This standalone version documents the key innovation: branch2b replacement.
        #
        # The complete integration requires:
        #   class PCGLU(BasicBlock):
        #       def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        #           super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        #           self.branch2b = Faster_Block_CGLU(ch_out, ch_out)
        self.branch2b = Faster_Block_CGLU(ch_out, ch_out)

    def forward(self, x):
        # In integrated version, BasicBlock.forward handles branch1 + branch2a,
        # then calls self.branch2b. See docs/integration.md for details.
        return self.branch2b(x)
