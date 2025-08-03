# PyTorch implementation of Scaling 4D Representations

**Scaling 4D Representations**: https://arxiv.org/abs/2412.15212

## Installation

1. Clone the original repository.

```bash
git clone https://github.com/google-deepmind/representations4d.git
cd representations4d
```

2. Download `scaling4d_dist_b_depth.npz` and place it in the root of `representations4d` folder.

3. Replace `EncoderToReadout` in `model.py` with

```python
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

```

4. [Optional] Set up the environment. We provide Pipfile.

Because sampling modes are implemented differently in JAX and PyTorch, the patch above allows us to explicitly select the sampling mode used in the original JAX model's readout, which helps ensure consistency during unit testing.

## Convert to PyTorch

```bash
python convert.py --ckpt_path "representations4d/scaling4d_dist_b_depth.npz" --out_dir checkpoints
```

This will create `checkpoints` folder and produce two checkpoints, one for the encoder and one for the readout model.

Optionally, verify the correctness of the implementation by running

```bash
python test_convert.py
```

Optionally, check the `scaling4d_depth_demo_torch.ipynb`.

## License and Attribution

This repository is a **PyTorch port** of the [Scaling 4D Representations](https://github.com/google-deepmind/representations4d), originally developed by DeepMind under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
