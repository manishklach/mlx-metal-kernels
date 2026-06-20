# Quantized Package Format

## Overview

The quantized package format is a JSON-based metadata format for describing a checkpoint that has been converted to the repo's native q4/q8 quantized tensor layout. It captures model configuration, quantization parameters, and per-layer tensor shapes without storing the actual tensor binary data.

## JSON Example

```json
{
  "format_version": "0.1.0",
  "model_type": "llama_like",
  "config": {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_hidden_layers": 2,
    "max_position_embeddings": 64,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-5,
    "model_type": "llama_like_tiny_gqa_debug"
  },
  "quantization": {
    "bits": 4,
    "group_size": 32,
    "symmetric": true,
    "with_zeros": false
  },
  "layers": [
    {
      "layer_idx": 0,
      "tensors": {
        "input_layernorm": {
          "name": "layers.0.input_layernorm",
          "role": "norm",
          "bits": 0,
          "group_size": 0,
          "original_shape": [64],
          "packed_shape": [64],
          "scales_shape": [0]
        },
        "qkv": {
          "name": "layers.0.qkv",
          "role": "qkv",
          "bits": 4,
          "group_size": 32,
          "original_shape": [128, 64],
          "packed_shape": [128, 32],
          "scales_shape": [128, 2]
        },
        "o_proj": {
          "name": "layers.0.o_proj",
          "role": "o_proj",
          "bits": 4,
          "group_size": 32,
          "original_shape": [64, 128],
          "packed_shape": [64, 64],
          "scales_shape": [64, 4]
        }
      }
    }
  ],
  "global_tensors": {},
  "metadata": {}
}
```

## Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `format_version` | string | Package format version (currently `"0.1.0"`) |
| `model_type` | string | Model architecture identifier (e.g. `"llama_like"`) |
| `config` | dict | Model configuration (see below) |
| `quantization` | dict | Quantization parameters (see below) |
| `layers` | list | Per-layer tensor descriptors |
| `global_tensors` | dict | Non-layer tensors (embedding, lm_head) |
| `metadata` | dict | Arbitrary converter metadata |

## Config fields

Standard `LlamaLikeConfig` fields:
- `hidden_size`
- `intermediate_size`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `num_hidden_layers`
- `max_position_embeddings`
- `rope_theta`
- `rms_norm_eps`
- `model_type`
- `vocab_size` (optional)

## Quantization fields

| Field | Type | Description |
|-------|------|-------------|
| `bits` | int | Quantization bit width (4 or 8) |
| `group_size` | int | Group size for groupwise quantization |
| `symmetric` | bool | Whether quantization is symmetric |
| `with_zeros` | bool | Whether zero-point metadata is present |

## Tensor metadata fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Logical tensor name |
| `role` | string | Tensor role (qkv, o_proj, gate_proj, up_proj, down_proj, norm, etc.) |
| `bits` | int | Bit width for quantized tensors, 0 for non-quantized |
| `group_size` | int | Group size for quantized tensors, 0 for non-quantized |
| `original_shape` | list[int] | Original tensor shape before quantization |
| `packed_shape` | list[int] | Packed tensor shape (q4: half the last dim, q8: same) |
| `scales_shape` | list[int] | Scales tensor shape `[out_dim, num_groups]` |
| `zeros_shape` | list[int] or null | Optional zero-point shape |
| `dtype` | string | Original tensor dtype |
| `data_file` | string or null | Binary data file reference (future) |
| `checksum` | string or null | Data checksum (future) |

## Tensor role values

- `qkv` – fused QKV projection
- `o_proj` – output projection
- `gate_proj` – gate projection (MLP)
- `up_proj` – up projection (MLP)
- `down_proj` – down projection (MLP)
- `embedding` – token embedding
- `lm_head` – language model head
- `norm` – normalization weight (non-quantized)
- `final_norm` – final normalization
- `other` – other tensors

## Layer tensor keys

Each layer may contain:
- `input_layernorm` – input RMSNorm weight
- `post_attention_layernorm` – post-attention RMSNorm weight
- `qkv` – fused QKV weight metadata
- `o_proj` – output projection
- `gate_proj` – MLP gate projection
- `up_proj` – MLP up projection
- `down_proj` – MLP down projection
