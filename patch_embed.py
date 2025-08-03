import torch.nn as nn

from typing import Tuple


class PatchEmbedding(nn.Module):
    """
    Extracts patches with a single learned linear projection named Conv_0.
    Expects input:
      - 5D (B, C, T, H, W) for videos
      - 4D (B, C, H, W) for images
    """

    def __init__(self, patch_size: Tuple[int, int] | Tuple[int, int, int], num_features: int, input_channels: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_features = num_features

        if len(patch_size) == 3:
            self.Conv_0 = nn.Conv3d(
                in_channels=input_channels,
                out_channels=num_features,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0  # VALID
            )
        elif len(patch_size) == 2:
            self.Conv_0 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=num_features,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0
            )
        else:
            raise ValueError("patch_size must be 2D or 3D")

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, T, H, W) or (B, C, H, W)
        Returns:
            Tensor of shape (B, D, t, h, w) or (B, D, h, w)
        """
        return self.Conv_0(x)