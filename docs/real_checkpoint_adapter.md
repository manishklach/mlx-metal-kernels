# Real Checkpoint Adapter Scaffold

## Purpose

This scaffold connects local tensor stores and JSON checkpoint manifests to the repo's Llama-like config, GQA-aware layer shapes, fused QKV helpers, and quantized packaging specs.

It is intentionally shape-first and layer-first. It does not claim production model loading or full runtime support.

## TensorStore abstraction

`TensorStore` provides a small interface:

- `keys()`
- `has(name)`
- `get_shape(name)`
- `get_dtype(name)`
- `load(name)`

## InMemoryTensorStore

`InMemoryTensorStore` is the simplest path for tests and demos. It accepts already-loaded tensor objects and exposes their names, shapes, dtypes, and values.

## ManifestTensorStore

`ManifestTensorStore` is shape-only. It uses `CheckpointManifest` metadata for validation and reporting, but `load(name)` raises `NotImplementedError`.

## Optional SafeTensorsTensorStore

`SafeTensorsTensorStore` is optional and only supports local `.safetensors` files.

- no network access
- lazy import
- clear `ImportError` if `safetensors` is not installed

## CheckpointAdapter

`CheckpointAdapter` validates a `TensorStore` against `LlamaLikeConfig` and exposes:

- per-layer tensor names
- expected layer shapes
- fused QKV shape derivation
- actual fused QKV creation for loadable stores
- quantized q4/q8 packaging specs
- adapter-level validation reports

## LayerWeightAdapter

`LayerWeightAdapter` exposes the per-layer weights that match the repo's kernel-facing layout expectations.

It can:

- report required names
- provide shape summaries
- load a layer's tensors
- optionally derive fused QKV

## GQA-aware QKV fusion

The adapter reuses the existing GQA-aware config:

- `q_proj`: `Hq * D`
- `k_proj`: `Hkv * D`
- `v_proj`: `Hkv * D`
- fused QKV: `Hq * D + 2 * Hkv * D`

## Quantized packaging specs

The adapter exposes existing q4/q8 packaging specs via `llama_quantized_layer_specs(...)`.

This PR does not implement real quantization or calibration.

## What is intentionally out of scope

- model downloads
- tokenizer integration
- sampling loop
- production Llama or Mistral runtime support
- end-to-end serving
- checkpoint quantization pipelines

## Future work

- deeper safetensors loading support
- checkpoint-to-quantized packaging
- tokenizer and sampling demo
- single-layer real checkpoint decode demo
- multi-layer runtime scaffold
