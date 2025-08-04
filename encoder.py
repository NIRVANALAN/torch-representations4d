from typing import Optional, Tuple
import torch
import torch.nn as nn

from generalized_transformer import GeneralizedTransformer
from tokenizer import Tokenizer


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int, int, int],
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        hidden_size: int,
        mlp_size: Optional[int] = None,
        n_iter: int = 1
    ):
        super(Encoder, self).__init__()
        self.tokenizer = Tokenizer(
            input_size=input_size,
            patch_size=patch_size,
            num_features=hidden_size
        )
        self.processor = GeneralizedTransformer(
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_size=hidden_size,
            mlp_size=mlp_size,
            n_iter=n_iter
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(video)
        x = self.processor(x)
        return x