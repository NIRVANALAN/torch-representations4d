import torch
import torch.nn as nn
from typing import Tuple


class LearnablePosEnc(nn.Module):
    """
    Learnable positional encoding module.

    This module creates a learnable positional embedding tensor of a given shape.
    The embeddings are initialized from a normal distribution with a specified standard deviation.

    Args:
        emb_shape (Tuple[int, ...]): Shape of the learnable positional embedding tensor.
        dtype (torch.dtype): Data type of the embedding tensor. Default is torch.float32.
        init_std (float): Standard deviation used to initialize the embeddings. Default is 0.02.

    Example:
        >>> pos_enc = LearnablePosEnc((64, 128))
        >>> pos = pos_enc()  # Shape: (1, 64, 128)
    """

    def __init__(
        self,
        emb_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        init_std: float = 0.02
    ) -> None:
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty(emb_shape, dtype=dtype))
        nn.init.normal_(self.embeddings, std=init_std)

    def forward(self) -> torch.Tensor:
        """
        Returns the learnable positional embeddings with a batch dimension.

        Returns:
            torch.Tensor: Positional embeddings of shape (1, *emb_shape)
        """
        return self.embeddings[None, ...]
