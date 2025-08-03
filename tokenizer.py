import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange

from patch_embed import PatchEmbedding


class Tokenizer(nn.Module):
    """Tokenizes the input images and adds positional encodings."""

    def __init__(
        self,
        input_size: Tuple[int, int, int, int],
        patch_size: Tuple[int, int, int],
        num_features: int,
    ):
        super().__init__()
        c, t, h, w = input_size
        patch_t, patch_h, patch_w = patch_size

        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            num_features=num_features,
            input_channels=c
        )
        (
            num_patches_t,
            num_patches_h,
            num_patches_w
        ) = (
            t // patch_t,
            h // patch_h,
            w // patch_w,
        )
        self.posenc = nn.Parameter(
            torch.empty(
                (num_features, num_patches_t, num_patches_h, num_patches_w)
            )
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (*B, T, H, W, C)
        tokens = self.patch_embedding(images)  # (*B, T, D, h, w)
        tokens = tokens + self.posenc[None, ...]  # (*B, T, D, h, w)
        tokens = rearrange(
            tokens,
            "... D T h w -> ... (T h w) D"
        )
        return tokens  # (*B, N, D)