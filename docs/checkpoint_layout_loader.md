# Checkpoint Layout Loader Scaffold

## Purpose

This repo now includes a dependency-light scaffold for describing and validating Llama-like checkpoint tensor layouts before any real binary checkpoint loader is added.

The goal is to define stable contracts for:

- tensor naming
- tensor shapes
- fused QKV layout derivation
- q4/q8 packaging expectations

without adding heavyweight runtime dependencies such as `safetensors`, `transformers`, or network-bound model download tooling.

## What a manifest is

A manifest is a small JSON description of checkpoint tensors. It records tensor names, shapes, dtypes, and optional source metadata without loading the actual tensor data.

The scaffold represents this with:

- `TensorInfo`
- `CheckpointManifest`

This makes it possible to inspect a model layout, check it against a `LlamaLikeConfig`, and derive kernel-facing shapes before implementing real checkpoint I/O.

## Llama-style tensor names

The scaffold currently focuses on Llama-like naming patterns such as:

- `model.layers.{i}.self_attn.q_proj.weight`
- `model.layers.{i}.self_attn.k_proj.weight`
- `model.layers.{i}.self_attn.v_proj.weight`
- `model.layers.{i}.self_attn.o_proj.weight`
- `model.layers.{i}.mlp.gate_proj.weight`
- `model.layers.{i}.mlp.up_proj.weight`
- `model.layers.{i}.mlp.down_proj.weight`
- `model.layers.{i}.input_layernorm.weight`
- `model.layers.{i}.post_attention_layernorm.weight`

Model-level names such as `model.norm.weight`, `lm_head.weight`, and `model.embed_tokens.weight` are also recognized, but they are not required for layer-only validation by default.

## Shape validation

`validate_llama_layer_shapes(...)` and `validate_llama_checkpoint_shapes(...)` compare manifest tensor shapes against a `LlamaLikeConfig`.

This includes GQA-aware expectations:

- `q_proj`: `[Hq * D, hidden_size]`
- `k_proj`: `[Hkv * D, hidden_size]`
- `v_proj`: `[Hkv * D, hidden_size]`
- `o_proj`: `[hidden_size, Hq * D]`

Validation returns a `ValidationReport` with explicit issues rather than immediately assuming full loader support.

## Fused QKV derivation

The scaffold also defines how separate Q, K, and V projections map into the repo’s fused layout:

```text
[Q ; K ; V]
```

For standard MHA:

```text
fused_qkv = [3 * hidden_size, hidden_size]
```

For GQA or MQA:

```text
fused_qkv = [q_output_dim + 2 * kv_output_dim, hidden_size]
```

This is important because grouped-query models do not have symmetric Q/K/V output dimensions.

## Quantized packaging specs

This PR does not implement production quantization. Instead, it defines packaging specs that describe how float checkpoint weights would need to map into the repo’s q4/q8 kernels.

The scaffold covers:

- q4 packed weight shapes
- q8 packed weight shapes
- per-group scales shapes
- optional zero-point shapes

This is enough to validate layout contracts before a future quantization/export pipeline exists.

## What is intentionally out of scope

This scaffold does not yet include:

- reading real `safetensors` files
- downloading checkpoints
- tokenizer integration
- full checkpoint-to-runtime execution
- production model loading

## Future work

Near-term follow-on work is expected to include:

- real checkpoint adapter infrastructure
- optional binary checkpoint readers
- tokenizer and sampling demos
- end-to-end layer execution from validated manifests
