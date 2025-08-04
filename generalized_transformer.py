from typing import List, Optional
import torch
import torch.nn as nn

from model import PreNormBlock


class GeneralizedTransformer(nn.Module):
    """
    Generalized ViT model.

    Inputs:
        - tokens: Tensor of shape (B, N, D)

    Outputs:
        - aux: List of intermediate states
    """

    def __init__(
        self,
        num_heads: int,
        num_layers: int,
        hidden_size: int,
        mlp_size: Optional[int] = None,
        n_iter: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            PreNormBlock(
                num_heads=num_heads,
                hidden_size=hidden_size,
                mlp_size=mlp_size
            )
            for _ in range(num_layers)
        ])
        self.n_iter = n_iter

    def forward(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            tokens: Input tokens of shape [batch_size, num_tokens, d_model]

        Returns:
            List of intermediate features including initial tokens and features 
            from each layer in the last iteration
        """
        # Store auxiliary outputs
        # equivalent to jnp.reshape(tokens, Shape('*B N D'))
        aux = [tokens.clone()]
        latent_state = tokens

        for h in range(self.n_iter):
            if h > 0:
                # Concatenate along sequence dimension (axis=-2 in JAX becomes axis=1 in PyTorch)
                latent_state = torch.cat([latent_state, tokens], dim=1)

            # Self attention layers
            for layer in self.layers:
                if h == self.n_iter - 1:
                    # Store intermediate features on last iteration
                    aux.append(latent_state.clone())

                latent_state = layer(latent_state)

        return aux