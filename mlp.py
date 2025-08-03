import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class TransformerMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: Optional[int] = None,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = (
            lambda x: F.gelu(x, approximate="tanh")
        )
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size or 4 * input_dim
        self.activation_fn = activation_fn

        self.dense_in = nn.Linear(input_dim, self.hidden_size)
        self.dense_out = nn.Linear(self.hidden_size, input_dim)

        # Initialize weights with Xavier uniform and biases with zeros
        nn.init.xavier_uniform_(self.dense_in.weight)
        nn.init.zeros_(self.dense_in.bias)
        nn.init.xavier_uniform_(self.dense_out.weight)
        nn.init.zeros_(self.dense_out.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.dense_in(inputs)
        h = self.activation_fn(h)
        out = self.dense_out(h)
        return out