import torch
import torch.nn.functional as F
import torch.nn as nn

from typing import Optional, Tuple, Union
from einops import rearrange, reduce, repeat
from layer_norm import LayerNorm

import torch.nn as nn
from typing import Optional


class ReadoutMLP(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_hidden_layers: int = 1
    ) -> None:
        output_size = output_size or input_dim

        super().__init__(
            *(
                [
                    nn.Linear(input_dim, hidden_size),
                    nn.GELU(approximate="tanh")
                ]
                + [
                    layer
                    for _ in range(num_hidden_layers - 1)
                    for layer in (
                        nn.Linear(hidden_size, hidden_size),
                        nn.GELU(approximate="tanh")
                    )
                ]
                + [nn.Linear(hidden_size, output_size)]
            )
        )

class AttentionReadout(nn.Module):
    """Cross-attention readout with learnable latent queries."""

    def __init__(
        self,
        num_classes: int,
        num_params: int,
        num_heads: int,
        num_queries: int = 1,
        match_vjepa_implementation: bool = True,
        add_temporal_posenc: bool = True,
        num_test_clips: int = 1,
        num_input_frames: int = 16,
        hidden_size: int = 768,
        dropout_rate: float = 0.0,
        output_shape: Optional[
            Union[
                Tuple[int, int, int],
                Tuple[int, int, int, int]
            ]
        ] = None,
        decoding_patch_size: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_params = num_params
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.match_vjepa_implementation = match_vjepa_implementation
        self.add_temporal_posenc = add_temporal_posenc
        self.num_test_clips = num_test_clips
        self.dropout_rate = dropout_rate
        self.output_shape = output_shape
        self.decoding_patch_size = decoding_patch_size
        # Validate num_params is divisible by num_heads
        self.num_params_per_head = num_params // num_heads
        if self.num_params_per_head * num_heads != num_params:
            raise ValueError(
                f'num_params ({num_params}) must be a multiple of num_heads'
                f' ({num_heads}).'
            )
        # Initialize components
        if self.match_vjepa_implementation:
            self.input_norm = LayerNorm(hidden_size, eps=0.000001)
            use_bias = True
        else:
            use_bias = False
        # Temporal positional encoding
        if self.add_temporal_posenc:
            # Will be initialized dynamically based on input shape
            self.temporal_posenc = nn.Parameter(
                torch.zeros(num_input_frames, hidden_size)
            )
        # Learnable query parameters
        self.queries = nn.Parameter(
            torch.empty(
                num_queries,
                num_heads,
                self.num_params_per_head
            )
        )
        # # Query projection for external queries
        # self.query_projection = nn.Linear(
        #     num_params,
        #     num_heads * self.num_params_per_head
        # )
        # Key and value projections
        self.key_projection = nn.Linear(
            hidden_size,
            num_heads * self.num_params_per_head,
            bias=use_bias
        )
        self.value_projection = nn.Linear(
            hidden_size,
            num_heads * self.num_params_per_head,
            bias=use_bias
        )
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        # VJEPA implementation specific components
        if self.match_vjepa_implementation:
            self.residual_projection = nn.Linear(num_params, num_params)
            self.mlp_norm = LayerNorm(num_params, eps=0.000001)
            self.mlp = ReadoutMLP(num_params, num_params * 4, num_hidden_layers=1)
        # Final classification layer
        self.out_projection = nn.Linear(num_params, num_classes)

    def forward(
        self,
        inputs: torch.Tensor,  # Shape: (B, T, N, C)
        queries: Optional[torch.Tensor] = None  # Shape: (B, Q, D)
    ) -> torch.Tensor:

        feats = inputs
        b, t, n, c = feats.shape
        if self.match_vjepa_implementation:
            # Normalize input features first
            feats = self.input_norm(feats)
        # Add temporal positional encoding
        if self.add_temporal_posenc:
            temb = repeat(
                self.temporal_posenc,
                "t c -> b t n c",
                b=b,
                n=n
            )
            feats = feats + temb
        # Reshape: (B, T, N, C) -> (B, T*N, C)
        B, T, N, C = feats.shape
        feats = rearrange(feats, 'B T N C -> B (T N) C')
        # Prepare queries
        if queries is None:
            # Use learnable queries
            num_queries = self.num_queries
            query = self.queries.unsqueeze(0).expand(B, -1, -1, -1)
        else:
            # Use provided queries
            num_queries = queries.shape[1]
            # query = self.query_projection(queries)
            query = rearrange(
                query,
                'B Q (h n) -> B Q h n',
                h=self.num_heads
            )

        # Prepare keys and values
        key = self.key_projection(feats)
        value = self.value_projection(feats)

        # Reshape for multi-head attention
        key = rearrange(key, 'B L (h d) -> B h L d', h=self.num_heads)
        value = rearrange(value, 'B L (h d) -> B h L d', h=self.num_heads)
        query = rearrange(query, 'B Q h d -> B h Q d')

        # Scaled dot-product attention
        token = F.scaled_dot_product_attention(
            query=query,  # (B, h, Q, d)
            key=key,      # (B, h, L, d)
            value=value,  # (B, h, L, d)
        )

        # Apply dropout
        token = self.dropout(token)

        # Reshape back
        token = rearrange(token, 'B h Q d -> B Q (h d)')

        if self.match_vjepa_implementation:
            # Extra MLP layer with residual connection
            query_reshaped = rearrange(
                self.queries.unsqueeze(0).expand(B, -1, -1, -1),
                'B Q h d -> B Q (h d)'
            )
            token = query_reshaped + self.residual_projection(token)
            residual = token
            token = self.mlp_norm(token)
            token = self.mlp(token)
            token = token + residual

        # Squeeze num_queries dimension if it's 1
        if num_queries == 1:
            token = token.squeeze(1)  # Remove Q dimension

        # Final classification
        out = self.out_projection(token)

        # Handle output reshaping for decoding tasks
        if self.output_shape is not None and self.decoding_patch_size is not None:
            channel_dim = (
                self.output_shape[-1]
                if len(self.output_shape) == 4 else 1
            )
            # Rearrange output to match desired shape
            patch_size0, patch_size1, patch_size2 = self.decoding_patch_size
            n_pixels_patch0 = self.output_shape[0] // patch_size0
            n_pixels_patch1 = self.output_shape[1] // patch_size1
            n_pixels_patch2 = self.output_shape[2] // patch_size2

            out = rearrange(
                out,
                'B (n_p0 n_p1 n_p2) (ps0 ps1 ps2 c) -> B (n_p0 ps0) (n_p1 ps1) (n_p2 ps2) c',
                ps0=patch_size0, ps1=patch_size1, ps2=patch_size2,
                n_p0=n_pixels_patch0, n_p1=n_pixels_patch1, n_p2=n_pixels_patch2,
                c=channel_dim
            )

        # Multi-clip evaluation
        if self.num_test_clips > 1 and not self.training:
            out = F.softmax(out, dim=-1)
            out = reduce(
                out,
                '(b n) ... -> b ...',
                'mean',
                n=self.num_test_clips
            )

        return out