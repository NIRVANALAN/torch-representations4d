from pathlib import Path
import numpy as np
import torch
import argparse

from einops import rearrange


def convert_readout_mlp(state_dict):
    x = state_dict
    x = {
        "0.bias": x["Dense_0/bias"],
        "0.weight": x["Dense_0/kernel"].T,
        "2.bias": x["Dense_1/bias"],
        "2.weight": x["Dense_1/kernel"].T
    }
    return x


def convert_layer_norm(state_dict):
    x = state_dict
    x = {
        (
            k
            .replace("/", ".")
            .replace("scale", "weight")
        ): v
        for (k, v) in state_dict.items()
    }
    return x


def convert_mlp(state_dict):
    x = state_dict
    x = {
        (
            k
            .replace("/", ".")
            .replace("kernel", "weight")
        ): v
        for (k, v) in x.items()
    }
    x["dense_in.weight"] = x["dense_in.weight"].T
    x["dense_out.weight"] = x["dense_out.weight"].T
    return x


def convert_attention(state_dict):
    x = state_dict
    x = {
        (
            k
            .replace("/", ".")
            .replace("kernel", "weight")
        ): v
        for (k, v) in state_dict.items()
    }

    for key in x.keys():
        if "key" in key or "value" in key or "query" in key:
            if "weight" in key:
                x[key] = rearrange(
                    x[key],
                    "d_in n_h d_h -> (n_h d_h) d_in"
                )
            else:
                x[key] = rearrange(
                    x[key],
                    "n_h d_h -> (n_h d_h)"
                )
    x["out.weight"] = rearrange(
        x["out.weight"],
        "n_h d_h d_out -> d_out (n_h d_h)"
    )

    return x


def convert_block(state_dict):
    x = state_dict

    attention_ckpt = {
        k.replace("attention/", ""): v
        for (k, v) in x.items()
        if "attention/" in k
    }
    attention_state_dict = convert_attention(attention_ckpt)
    attention_norm_ckpt = {
        k.replace("attention_norm/", ""): v
        for (k, v) in x.items()
        if "attention_norm/" in k
    }
    attention_norm_state_dict = convert_layer_norm(attention_norm_ckpt)
    mlp_norm_ckpt = {
        k.replace("mlp_norm/", ""): v
        for (k, v) in x.items()
        if "mlp_norm/" in k
    }
    mlp_norm_state_dict = convert_layer_norm(mlp_norm_ckpt)
    mlp_ckpt = {
        k.replace("mlp/", ""): v
        for (k, v) in x.items()
        if "mlp/" in k
    }
    mlp_state_dict = convert_mlp(mlp_ckpt)
    return {
        **{
            f"attention_norm.{k}": v
            for (k, v) in attention_norm_state_dict.items()
        },
        **{
            f"attention.{k}": v
            for (k, v) in attention_state_dict.items()
        },
        **{
            f"mlp_norm.{k}": v
            for (k, v) in mlp_norm_state_dict.items()
        },
        **{
            f"mlp.{k}": v
            for (k, v) in mlp_state_dict.items()
        },
    }


def convert_transformer(state_dict):
    x = {}
    for (k, v) in sorted(state_dict.items()):
        layer_idx, *_ = k.split("/")
        if layer_idx not in x:
            x[layer_idx] = {}
        x[layer_idx][k.replace(f"{layer_idx}/", "")] = v
    out = {}
    for layer_idx, state_dict in x.items():
        for k, v in convert_block(state_dict).items():
            out[f"{layer_idx.replace('_', '.')}.{k}"] = v
    return out


def convert_tokenizer(state_dict):
    x = state_dict
    x = {
        (
            k
            .replace("kernel", "weight")
            .replace("/", ".")
        ): v
        for (k, v) in x.items()
    }
    x["patch_embedding.Conv_0.weight"] = rearrange(
        x["patch_embedding.Conv_0.weight"],
        "kt kh kw cin cout -> cout cin kt kh kw"
    )
    x["posenc"] = rearrange(
        x["posenc"],
        "t h w d -> d t h w"
    )
    return x


def convert_readout(readout_ckpt):
    return {
        "temporal_posenc": torch.from_numpy(readout_ckpt["temporal_posenc/embeddings"]).squeeze(1),
        "queries": torch.from_numpy(readout_ckpt["query"]),  # [n, h, d]
        "key_projection.bias": torch.from_numpy(
            rearrange(
                readout_ckpt["key_embedding/bias"],
                "n_h d_h -> (n_h d_h)"
            )
        ),
        "key_projection.weight": torch.from_numpy(
            rearrange(
                readout_ckpt["key_embedding/kernel"],
                "d_in n_h d_h -> (n_h d_h) d_in"
            )
        ),
        "value_projection.bias": torch.from_numpy(
            rearrange(
                readout_ckpt["value_embedding/bias"],
                "n_h d_h -> (n_h d_h)"
            )
        ),
        "value_projection.weight": torch.from_numpy(
            rearrange(
                readout_ckpt["value_embedding/kernel"],
                "d_in n_h d_h -> (n_h d_h) d_in"
            )
        ),
        "residual_projection.bias": torch.from_numpy(readout_ckpt["Dense_0/bias"]),
        "residual_projection.weight": torch.from_numpy(readout_ckpt["Dense_0/kernel"].T),
        "out_projection.bias": torch.from_numpy(readout_ckpt["Dense_1/bias"]),
        "out_projection.weight": torch.from_numpy(readout_ckpt["Dense_1/kernel"].T),
        "input_norm.bias": torch.from_numpy(readout_ckpt["LayerNorm_0/bias"]),
        "input_norm.weight": torch.from_numpy(readout_ckpt["LayerNorm_0/scale"]),
        "mlp_norm.bias": torch.from_numpy(readout_ckpt["LayerNorm_1/bias"]),
        "mlp_norm.weight": torch.from_numpy(readout_ckpt["LayerNorm_1/scale"]),
        **{
            f"mlp.{k}": v
            for (k, v) in convert_readout_mlp({
                k.replace("MLP_0/", ""): torch.from_numpy(v)
                for (k, v) in readout_ckpt.items()
                if "MLP" in k
            }).items()
        }
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--ckpt_path',
      type=str,
      default=None,
      help='Path to checkpoint'
    )
    parser.add_argument(
      '--out_dir',
      type=str,
      default=None,
      help='Path to checkpoint'
    )
    return parser.parse_args()


def convert(jax_ckpt):
  readout_ckpt = {
      k.replace("params/layers_2/", ""): v
      for (k, v) in jax_ckpt.items()
      if "params/layers_2" in k
  }
  patch_embedding_ckpt = {
      k.replace("params/layers_0/encoder/patch_embedding/", "patch_embedding/"): torch.from_numpy(v)
      for (k, v) in jax_ckpt.items() if "Conv_0" in k
  }
  embed_ckpt = {
      "posenc": torch.from_numpy(jax_ckpt["params/layers_0/encoder/posenc/embeddings"])
  }
  tokenizer_ckpt = {**patch_embedding_ckpt, **embed_ckpt}
  processor_ckpt = {
      (
          k
          .replace("params/layers_0/processor/", "")
      ): torch.from_numpy(v)
      for (k, v) in jax_ckpt.items()
      if "params/layers_0/processor/layers_" in k and "encoder" not in k
  }
  readout_ckpt = {
      k.replace("params/layers_2/", ""): v
      for (k, v) in jax_ckpt.items()
      if "params/layers_2" in k
  }
  tokenizer_state_dict = convert_tokenizer(tokenizer_ckpt)
  processor_state_dict = convert_transformer(processor_ckpt)
  encoder_state_dict = {
      **{
          f"tokenizer.{k}": v
          for (k, v) in tokenizer_state_dict.items()
      },
      **{
          f"processor.{k}": v
          for (k, v) in processor_state_dict.items()
      }
  }
  readout_state_dict = convert_readout(readout_ckpt)

  return {
      "encoder_state_dict": encoder_state_dict,
      "readout_state_dict": readout_state_dict
  }

def main(ckpt_path: str, out_dir: str):
  ckpt = np.load(ckpt_path)
  
  out_folder = Path(out_dir)
  out_folder.mkdir(exist_ok=True, parents=True)

  torch_ckpt = convert(ckpt)
  torch.save(
      torch_ckpt["encoder_state_dict"],
      str(out_folder / "encoder.pth")
  )
  torch.save(
      torch_ckpt["readout_state_dict"],
      str(out_folder / "readout.pth")
  )

if __name__ == "__main__":
    args = parse_args()

    main(args.ckpt_path, args.out_dir)