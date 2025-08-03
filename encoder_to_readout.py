from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class EncoderToReadout(nn.Module):
    def __init__(
        self,
        embedding_shape: tuple[int, int, int],
        readout_depth: float,
        num_input_frames: int,
        sampling_mode: Literal[
            "nearest",
            "linear",
            "bilinear",
            "bicubic",
            "trilinear",
            "area",
            "nearest-exact",
        ] = "bicubic"
    ):
        super(EncoderToReadout, self).__init__()
        self.embedding_shape = embedding_shape
        self.readout_depth = readout_depth
        self.num_input_frames = num_input_frames
        self.sampling_mode = sampling_mode

    def forward(self, all_features: list[torch.Tensor]) -> torch.Tensor:
        # Select feature map according to readout depth
        readout_id = int(len(all_features) * self.readout_depth) - 1
        T, H, W = self.embedding_shape
        S = H * W  # spatial size
        # Get features
        f = all_features[readout_id]  # (B, T*H*W, C) expected
        # Reshape (B, T*H*W, C) → (B, T, S, C)
        f = rearrange(
            f,
            'b (t s) c -> b t s c',
            t=T,
            s=S
        )
        # (B, T, S, C) → (B, C, T, S)
        f = rearrange(f, 'b t s c -> b c t s')
        # Interpolate over time axis: T → num_input_frames
        f = F.interpolate(
            f,
            size=(self.num_input_frames, S),
            mode=self.sampling_mode,
        )
        # (B, C, T', S) → (B, T', S, C)
        f = rearrange(f, 'b c t s -> b t s c')
        return f