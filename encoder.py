from typing import Optional, Tuple
from pdb import set_trace as st
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

    def random_masking(self, x, mask_ratio):
        # https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py#L123C1-L148C43
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward(self, video: torch.Tensor, mask_ratio: float = 0, return_mask: bool = False) -> torch.Tensor:
        x = self.tokenizer(video)
        if mask_ratio > 0:
            # masking: length -> length * (1-mask_ratio)
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        # st()
        x = self.processor(x)
        if return_mask:
            return x, (mask, ids_keep, ids_restore)
        else:
                return x