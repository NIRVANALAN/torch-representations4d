# test_convert.py
from pdb import set_trace as st
import unittest
import torch
import numpy as np

import os

from convert import (
   convert_with_readout,
   convert_without_readout
)
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
from kauldron.modules import pos_embeddings
from kauldron.modules import vit as kd_vit
import mediapy
from representations4d.models import model as model_lib
from representations4d.models import readout
import numpy as np
from representations4d.utils import checkpoint_utils
from einops import rearrange
from flax import linen as nn
import jax
import jax.numpy as jnp
from kauldron import typing as kd_typing


typechecked = kd_typing.typechecked
Float = kd_typing.Float
Shape = kd_typing.Shape
Dim = kd_typing.Dim
Initializer = kd_typing.Initializer
check_type = kd_typing.check_type

class EncoderToReadout(nn.Module):
  """Encoder to readout."""

  embedding_shape: tuple[int, int, int]
  readout_depth: float
  num_input_frames: int
  mode: str = "cubic"

  @nn.compact
  @typechecked
  def __call__(self, all_features: list[Float['...']]) -> Float['...']:
    readout_id = int(len(all_features) * self.readout_depth) - 1
    features = all_features[readout_id]
    readout_features = jnp.reshape(
        features,
        (features.shape[0],)  # batch
        + (self.embedding_shape[0],)  # time
        + (self.embedding_shape[1] * self.embedding_shape[2],)  # space
        + features.shape[-1:],  # channels
    )
    out_shape = (
        (readout_features.shape[0],)
        + (self.num_input_frames,)
        + (
            self.embedding_shape[0]
            * self.embedding_shape[1]
            * self.embedding_shape[2]
            // self.embedding_shape[0],
        )
        + (readout_features.shape[3],)
    )
    readout_features = jax.image.resize(
        readout_features, out_shape, self.mode
    )
    return readout_features


# JAX_DEPTH_CHECKPOINT_PATH = "representations4d/scaling4d_dist_b_depth.npz"
# JAX_CHECKPOINT_PATH = "representations4d/scaling4d_dist_b.npz"

JAX_DEPTH_CHECKPOINT_PATH = "checkpoints/scaling4d_dist_b_depth.npz"
JAX_CHECKPOINT_PATH = "checkpoints/scaling4d_dist_b.npz"

TEST_VIDEO_PATH = "representations4d/horsejump-high.mp4"

model_patch_size = (2, 16, 16)
im_size = (224, 224)
# im_size = (256, 256)
model_size = "B"
dtype = jnp.float32
model_output_patch_size = (2, 8, 8)
n_pixels_patch = (
  model_output_patch_size[0]
  * model_output_patch_size[1]
  * model_output_patch_size[2]
)

num_input_frames = 16
# num_input_frames = 32 # test PE
n_pixels_video = num_input_frames * im_size[0] * im_size[1]

embedding_shape = (
  num_input_frames // model_patch_size[0],
  im_size[0] // model_patch_size[1],
  im_size[1] // model_patch_size[2],
)
num_tokens = embedding_shape[0] * embedding_shape[1] * embedding_shape[2]

def get_jax_encoder_model():
  return model_lib.Model(
      encoder=model_lib.Tokenizer(
          patch_embedding=model_lib.PatchEmbedding(
              patch_size=model_patch_size,
              num_features=kd_vit.VIT_SIZES[model_size][0],
          ),
          posenc=pos_embeddings.LearnedEmbedding(dtype=dtype),
          posenc_axes=(-4, -3, -2),
      ),
      processor=model_lib.GeneralizedTransformer.from_variant_str(
          variant_str=model_size,
          dtype=dtype,
      ),
  )

def get_torch_encoder_model(ckpt):
  from encoder import Encoder

  encoder_state_dict = ckpt["encoder_state_dict"]
  torch_encoder = Encoder(
      input_size=(3, 16, 224, 224), # defines orig PE size.
      patch_size=(2, 16, 16),
      num_heads=12,
      num_layers=12,
      hidden_size=768,
      n_iter=1
  )
  torch_encoder.load_state_dict(encoder_state_dict)   

  return torch_encoder

def get_jax_depth_model():
  from flax import linen as nn

  encoder = model_lib.Model(
      encoder=model_lib.Tokenizer(
          patch_embedding=model_lib.PatchEmbedding(
              patch_size=model_patch_size,
              num_features=kd_vit.VIT_SIZES[model_size][0],
          ),
          posenc=pos_embeddings.LearnedEmbedding(dtype=dtype),
          posenc_axes=(-4, -3, -2),
      ),
      processor=model_lib.GeneralizedTransformer.from_variant_str(
          variant_str=model_size,
          dtype=dtype,
      ),
  )
  encoder2readout = EncoderToReadout(
      embedding_shape=(
          num_input_frames // model_patch_size[0],
          im_size[0] // model_patch_size[1],
          im_size[1] // model_patch_size[2],
      ),
      readout_depth=0.95,
      num_input_frames=num_input_frames,
      mode="linear"
  )
  readout_head = readout.AttentionReadout(
      num_classes=n_pixels_patch,
      num_params=1024,
      num_heads=16,
      num_queries=n_pixels_video // n_pixels_patch,
      output_shape=(
          num_input_frames,
          im_size[0],
          im_size[1],
          1,
      ),
      decoding_patch_size=model_output_patch_size,
  )

  return nn.Sequential([encoder, encoder2readout, readout_head])

def get_torch_depth_model(ckpt):
  from encoder import Encoder
  from encoder_to_readout import EncoderToReadout
  from readout import AttentionReadout
  import torch.nn as nn

  encoder_state_dict = ckpt["encoder_state_dict"]
  readout_state_dict = ckpt["readout_state_dict"]

  torch_encoder = Encoder(
      input_size=(3, 16, 224, 224),
      patch_size=(2, 16, 16),
      num_heads=12,
      num_layers=12,
      hidden_size=768,
      n_iter=1
  )
  torch_encoder.load_state_dict(encoder_state_dict)   
  torch_encoder2readout = EncoderToReadout(
      embedding_shape=embedding_shape,
      readout_depth=0.95,
      num_input_frames=num_input_frames,
      sampling_mode="bilinear"
  )
  torch_attn_readout = AttentionReadout(
      num_classes=n_pixels_patch,
      num_params=1024,
      num_heads=16,
      num_queries=n_pixels_video // n_pixels_patch,
      output_shape=(
          num_input_frames,
          im_size[0],
          im_size[1],
          1,
      ),
      decoding_patch_size=model_output_patch_size,
  )
  torch_attn_readout.load_state_dict(readout_state_dict)
  torch_depth_model = nn.Sequential(
      torch_encoder,
      torch_encoder2readout,
      torch_attn_readout
  )
  return torch_depth_model

class TestConvert(unittest.TestCase):

    def test_convert_base_encoder(self):
      # Load ckpts
      jax_ckpt = np.load(JAX_CHECKPOINT_PATH)
      torch_ckpt = convert_without_readout(jax_ckpt)
      # Load models 
      torch_depth_model = get_torch_encoder_model(torch_ckpt)
      # Load input
      video = mediapy.read_video(TEST_VIDEO_PATH)
      video = mediapy.resize_video(video, im_size) / 255.0
      video = video[jnp.newaxis, :num_input_frames].astype(jnp.float32)
      
      video_torch = rearrange(
          torch.from_numpy(jax.device_get(video).copy()),
          "b t h w c -> b c t h w"
      )
      mask_ratio = 0.95 # as in Rep4D paper

      output_torch_1 = torch_depth_model(video_torch, mask_ratio=mask_ratio)
      output_torch_2 = torch_depth_model(video_torch, mask_ratio=mask_ratio)

      for layer_idx, (out_torch_1, out_torch_2) in enumerate(zip(output_torch_1, output_torch_2)):
            print(f"layer {layer_idx} does not match, error: {torch.abs(out_torch_1 - out_torch_2).max()}")

    #   st()  



if __name__ == '__main__':
    unittest.main()