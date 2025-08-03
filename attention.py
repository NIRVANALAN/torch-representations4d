
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        input_dim_q: int,
        input_dim_kv: int,
        qk_dim: int,
        v_dim: int,
        out_dim: int,
        normalize_qk: bool = False,
        use_bias: bool = True,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        input_dim_kv = input_dim_kv or input_dim_q
        qk_dim = qk_dim or input_dim_q
        v_dim = v_dim or qk_dim
        out_dim = out_dim or input_dim_q

        if qk_dim % num_heads != 0:
            raise ValueError(f"{qk_dim=} not divisible by {num_heads=}")
        if v_dim % num_heads != 0:
            raise ValueError(f"{v_dim=} not divisible by {num_heads=}")

        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.out_dim = out_dim
        self.normalize_qk = normalize_qk
        self.use_bias = use_bias
        self.dropout_p = dropout_p

        self.head_dim_qk = qk_dim // num_heads
        self.head_dim_v = v_dim // num_heads

        self.query = nn.Linear(input_dim_q, qk_dim, bias=use_bias)
        self.key = nn.Linear(input_dim_kv, qk_dim, bias=use_bias)
        self.value = nn.Linear(input_dim_kv, v_dim, bias=use_bias)
        self.out = nn.Linear(v_dim, out_dim, bias=use_bias)

        if self.normalize_qk:
            self.q_norm = nn.LayerNorm(qk_dim)
            self.k_norm = nn.LayerNorm(qk_dim)

    def forward(
        self,
        inputs_q,
        inputs_k,
        inputs_v,
        attn_mask=None,
        is_causal=False,
    ):
        q = self.query(inputs_q)
        k = self.key(inputs_k)
        v = self.value(inputs_v)

        if self.normalize_qk:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )  # (B, num_heads, Q_len, head_dim_v)

        attn_out = rearrange(attn_out, "b h l d -> b l (h d)")
        out = self.out(attn_out)

        return out