import torch.nn as nn


class LayerNorm(nn.LayerNorm):
    def forward(self, input):
        orig_dtype = input.dtype
        return super().forward(input.float()).to(orig_dtype)
