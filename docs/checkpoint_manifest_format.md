# Checkpoint Manifest Format

The checkpoint scaffold uses a JSON manifest to describe tensor names, shapes, dtypes, and optional source metadata without requiring a binary checkpoint reader.

## Dict form

```json
{
  "model_type": "llama_like",
  "metadata": {
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_hidden_layers": 1
  },
  "tensors": {
    "model.layers.0.self_attn.q_proj.weight": {
      "shape": [64, 64],
      "dtype": "float16",
      "source": "mock"
    },
    "model.layers.0.self_attn.k_proj.weight": {
      "shape": [32, 64],
      "dtype": "float16",
      "source": "mock"
    }
  }
}
```

This form is convenient when tensor names are already unique keys.

## List form

```json
{
  "model_type": "llama_like",
  "metadata": {
    "hidden_size": 64
  },
  "tensors": [
    {
      "name": "model.layers.0.self_attn.q_proj.weight",
      "shape": [64, 64],
      "dtype": "float16"
    },
    {
      "name": "model.layers.0.self_attn.k_proj.weight",
      "shape": [32, 64],
      "dtype": "float16"
    }
  ]
}
```

This form is useful when manifests are generated from pipelines that naturally produce tensor records as lists.

## Required top-level fields

- `model_type`: non-empty string
- `tensors`: dict or list of tensor records

## Tensor fields

Each tensor may include:

- `name`
- `shape`
- `dtype`
- `source`
- `offset`
- `nbytes`
- `metadata`

Only `name`, `shape`, and `dtype` are required.

## Validation rules

- tensor names must be non-empty strings
- shape dimensions must be positive integers
- dtype must be a non-empty string
- duplicate tensor names are rejected

## Why JSON manifests

The purpose of the manifest is to make checkpoint inspection and layout validation possible without introducing:

- `safetensors`
- `transformers`
- `huggingface_hub`
- network access

This keeps the repo lightweight while still preparing it for future checkpoint adapter work.
