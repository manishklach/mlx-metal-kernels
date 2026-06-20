# Multi-layer decode stack

## Purpose

This scaffold extends the earlier single-layer decode experiment into a small composed multi-layer stack.

It is designed for correctness-first synthetic experiments, explicit cache handling, and future model-integration work. It is not a production inference runtime.

## Relationship to the single-layer decode experiment

The single-layer decode scaffold validated the building blocks for one Llama-like layer. This multi-layer stack keeps those layer primitives intact and composes them sequentially.

## Stack formula

The stack follows the simple structure:

1. embedding or incoming token hidden state
2. for each layer:
   - `h = layer_i(h, cache_i)`
3. final RMSNorm
4. optional `lm_head`
5. optional sampling and generation loop

## One KV-cache per layer

Each layer owns its own KV-cache state.

This keeps the cache boundary explicit and testable, and matches how real multi-layer transformer decode works conceptually.

## GQA cache shapes

For each layer, the contiguous cache layout is:

- `K_cache [B, max_seq_len, num_key_value_heads, head_dim]`
- `V_cache [B, max_seq_len, num_key_value_heads, head_dim]`

The same per-layer structure is repeated across the stack.

## Backend presets

The stack scaffold currently routes through the existing single-layer backend presets:

- `reference`
- `metal`
- `tiled`
- `fused_experimental`

In dependency-light environments, the scaffold falls back to a small NumPy reference-style path so the cache and generation plumbing can still be tested.

## Synthetic generation demo

The multi-layer generation demo uses:

- toy character tokenization
- synthetic random quantized layer weights
- synthetic embedding and `lm_head`
- a simple token-by-token decode loop

The output is not meaningful language. The point is plumbing, shape validation, and cache behavior.

## What this does not claim

- production checkpoint execution
- meaningful text quality from random weights
- optimized prefill stack
- full multi-layer Llama or Mistral runtime support
- universal performance claims

## Benchmark commands

```bash
python benchmarks/bench_llama_stack_decode.py --bits 4 --B 1 --T 16 --num-layers 2 --hidden-size 512 --intermediate-size 2048 --num-heads 8 --num-kv-heads 2 --head-dim 64 --MAX_S 128 --dtype float16 --backend-preset all --validate
python benchmarks/bench_llama_stack_decode.py --bits 4 --B 1 --T 16 --num-layers 4 --hidden-size 512 --intermediate-size 2048 --num-heads 8 --num-kv-heads 2 --head-dim 64 --MAX_S 128 --dtype float16 --backend-preset fused_experimental
```

## Future work

- real checkpoint package to stack weights
- real tokenizer adapter
- optimized prefill stack
- real multi-layer model execution
- sampling with actual trained weights
