from typing import Optional
import torch
import torch.nn as nn

from attention import MultiHeadAttention
from layer_norm import LayerNorm
from mlp import TransformerMLP


class PreNormBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        normalize_qk: bool = False,
        use_bias: bool = True,
        dropout_p: float = 0.0
    ):
        super(PreNormBlock, self).__init__()
        self.attention_norm = LayerNorm(hidden_size, eps=0.000001)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            input_dim_q=hidden_size,
            input_dim_kv=hidden_size,
            qk_dim=hidden_size,
            v_dim=hidden_size,
            out_dim=hidden_size,
            normalize_qk=normalize_qk,
            use_bias=use_bias,
            dropout_p=dropout_p
        )
        self.mlp_norm = LayerNorm(hidden_size, eps=0.000001)
        self.mlp = TransformerMLP(
            input_dim=hidden_size,
            hidden_size=4*hidden_size
        )

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm_tokens = self.attention_norm(tokens)
        tokens += self.attention(
            inputs_q=norm_tokens,
            inputs_k=norm_tokens,
            inputs_v=norm_tokens,
            attn_mask=mask,
        )
        norm_tokens = self.mlp_norm(tokens)
        return tokens + self.mlp(norm_tokens)