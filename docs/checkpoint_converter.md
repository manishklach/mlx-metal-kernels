# Checkpoint Converter Scaffold

## Purpose

The checkpoint converter connects the existing checkpoint manifest, adapter, and quantizer infrastructure into a practical CLI for packaging local layer tensors as q4/q8 package metadata. The output is a JSON package that can be inspected, validated, and loaded for synthetic runtime tests.

## Relationship to other components

| Component | Role |
|-----------|------|
| `CheckpointManifest` | Describes tensor names, shapes, and dtypes from a JSON manifest |
| `TensorStore` | Provides tensor data (in-memory, manifest-only, or safetensors) |
| `CheckpointAdapter` | Maps manifest tensor names to logical layer names and shapes |
| `CheckpointQuantizer` | Quantizes loaded tensors to q4/q8 using groupwise quantization |
| `QuantizedLlamaLayerPackage` | Holds per-layer quantized weights and norm tensors |
| `QuantizedCheckpointPackage` | Metadata-only package describing all quantized layers |
| `CheckpointConverter` | Orchestrates adapter → quantizer → package pipeline |

## Converter flow

1. Load checkpoint using a `CheckpointAdapter`.
2. Create a `CheckpointConverter` with the adapter and converter config.
3. Call `convert()` to quantize all requested layers and build a package.
4. Save the package as JSON.

## Synthetic conversion mode

The CLI supports `--synthetic-demo` mode that creates tiny random weights and converts them. This mode works without any real checkpoint files.

```bash
python scripts/convert_checkpoint.py --synthetic-demo --bits 4 --group-size 32 --output /tmp/mlx_quant_package.json
```

## Manifest dry-run mode

When a JSON manifest is provided via `--manifest`, the converter loads shape information only. Since `ManifestTensorStore` cannot load actual tensor data, the converter:

- Validates shapes and configuration
- Writes a conversion plan JSON (if `--output` is provided)
- Does not attempt quantization

Use `--dry-run` to inspect the plan. Without `--dry-run`, the manifest-only path produces a plan and exits cleanly.

```bash
python scripts/convert_checkpoint.py --manifest /path/to/manifest.json --output /tmp/conversion_plan.json
```

## Quantized package metadata

The output JSON uses the format described in `quantized_package_format.md`. It includes:

- Model configuration (hidden size, heads, layers)
- Quantization parameters (bits, group size, symmetric)
- Per-layer tensor metadata (original shapes, packed shapes, scales shapes)
- Global tensor metadata (embedding, lm_head)

## Limitations

- No tensor data is saved in the current format (metadata only).
- No safetensors or binary output is supported yet.
- No calibrated quantization (GPTQ, AWQ, SmoothQuant).
- No tokenizer or model serving support.

## Future work

- Binary tensor data writer (`save_tensor_data=True`)
- Safetensors-backed tensor data storage
- Full checkpoint conversion with real model weights
- Calibrated quantization integration
