import torch
import math
import torch.nn as nn
from typing import Tuple
from einops import rearrange
from pdb import set_trace as st

from patch_embed import PatchEmbedding


class Tokenizer(nn.Module):
    """Tokenizes the input images and adds positional encodings."""

    def __init__(
        self,
        input_size: Tuple[int, int, int, int],
        patch_size: Tuple[int, int, int],
        num_features: int,
        # interpolate_antialias: bool = True,
        # interpolate_offset: float = 0.0, # the default value in dinov2
    ):
        # https://github.com/facebookresearch/vjepa2/blob/53925fd16be56fcedd93b1950c95bd2339c69dd6/src/models/vision_transformer.py#L215
        super().__init__()
        c, t, h, w = input_size  # input_size format: (C, T, H, W)
        patch_t, patch_h, patch_w = patch_size

        self.patch_size = patch_size
        self.is_video = True
        # self.interpolate_offset = interpolate_offset
        # self.interpolate_antialias = interpolate_antialias

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
    # supports diverse input size by interpolating the positional encodings.
    # https://github.com/facebookresearch/dinov2/blob/b8931f7bf91576930313be2c6d6af376033b35f0/dinov2/models/vision_transformer.py#L204

    def interpolate_pos_encoding(self, x, t, h, w):
        """
        Interpolate positional encoding for different input sizes.
        Supports 3D spatiotemporal interpolation for B C T H W input format.
        
        Args:
            x: Input tensor with shape (*B, T, D, h, w) where h, w are patch dimensions
            t: Target temporal dimension
            h: Target height dimension  
            w: Target width dimension
        
        Returns:
            Interpolated positional encoding with shape (D, T_patches, H_patches, W_patches)
        """
        previous_dtype = x.dtype
        patch_t, patch_h, patch_w = self.patch_size
        
        # Calculate target patch dimensions
        t_patches = t // patch_t
        h_patches = h // patch_h  
        w_patches = w // patch_w
        
        # Get original positional encoding dimensions
        orig_d, orig_t_patches, orig_h_patches, orig_w_patches = self.posenc.shape
        
        # If dimensions match, return original encoding
        if (t_patches == orig_t_patches and 
            h_patches == orig_h_patches and 
            w_patches == orig_w_patches):
            return self.posenc
        
        # Convert to float for interpolation
        pos_embed = self.posenc.float()
        
        # Prepare for 3D interpolation: (D, T, H, W) -> (1, D, T, H, W)
        pos_embed = pos_embed.unsqueeze(0)
        
        # Use trilinear interpolation for 3D spatiotemporal data
        pos_embed_interp = nn.functional.interpolate(
            pos_embed,
            mode="trilinear",
            align_corners=False,
            # scale_factor=scale_factor,
            size=(t_patches, h_patches, w_patches),
        )
        
        # Remove batch dimension: (1, D, T, H, W) -> (D, T, H, W)
        pos_embed_interp = pos_embed_interp.squeeze(0)
        
        return pos_embed_interp.to(previous_dtype)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (*B, C, T, H, W)
        B, C, T, H, W = images.shape

        tokens = self.patch_embedding(images)  # (*B, T, D, h, w)
        pos_encoding = self.interpolate_pos_encoding(tokens, T, H, W)
        tokens = tokens + pos_encoding[None, ...]  # (*B, T, D, h, w)

        # ! shall be compatible with the original code.
        # tokens = tokens + self.posenc[None, ...]  # (*B, T, D, h, w)

        tokens = rearrange(
            tokens,
            "... D T h w -> ... (T h w) D"
        )
        return tokens  # (*B, N, D)