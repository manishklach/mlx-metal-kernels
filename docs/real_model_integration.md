# Real Model Integration Scaffold

PR #17 adds a scaffold that bridges the existing toy decode-layer benchmark toward real Llama/Mistral-style model conventions.

## What This PR Adds

- a `LlamaLikeConfig` dataclass with validation and approximate presets
- RoPE table generation compatible with the repo's existing `apply_rope` expectations
- weight-layout specs for common Llama-style linear names
- a kernel adapter that reuses the repo's existing cache, RoPE, quantized decode, and autotune helpers
- small examples for shape inspection and random decode-layer execution

## What This PR Does Not Add

- full checkpoint loading
- Hugging Face integration
- `safetensors`
- tokenizer or sampling
- production Llama inference

This is intentionally scaffolding, not a complete model runtime.

## Relationship To The Toy Transformer Layer

The toy transformer decode benchmark already composes:

- RMSNorm
- fused attention-style decode
- residual add
- SwiGLU MLP

This PR re-expresses that composition in terms closer to real model structure:

- explicit model config
- named weight layouts
- cache initialization per layer
- fused-QKV expectations that future checkpoint mapping can target

## Current Supported Path

- fused QKV layout
- `num_key_value_heads == num_attention_heads`
- fp16/bf16 activations where the underlying kernels already support them
- q4/q8 quantized weights through the existing decode matvec kernels

## Current Unsupported Path

The adapter raises a clear `NotImplementedError` for GQA/MQA-style configs where:

```text
num_key_value_heads != num_attention_heads
```

That keeps the current scope explicit until the cache/update/decode path supports KV-head expansion or native grouped-query handling.

## Future Work

- real checkpoint loader scaffold
- `safetensors` support
- GQA support
- embedding and LM-head scaffolding
- multi-layer decode loop
- tokenizer and sampling integration
