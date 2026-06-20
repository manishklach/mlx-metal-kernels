# Toy Transformer Decode Benchmark

PR #14 adds an end-to-end toy transformer layer decode benchmark built from the repo's existing correctness-first primitives.

## Layer Structure

The benchmark composes:

- RMSNorm before attention
- quantized QKV projection plus decode attention plus quantized output projection
- residual add
- RMSNorm before MLP
- quantized gate, up, and down projections
- SwiGLU activation
- final residual add

This is intentionally a toy single-layer decode path rather than a full model runtime. The goal is to measure how the current kernel set interacts in a more realistic decode stack.

## Why This Matters

Single-kernel benchmarks are useful, but they do not fully show how memory traffic and launch overhead stack up inside a transformer decode layer. This benchmark gives the repo a small end-to-end decode composition that can be compared across backend presets.

## Backends

The benchmark supports:

- contiguous KV cache
- paged KV cache
- reference preset
- metal preset
- parallel preset
- tiled preset

Before timing, the script validates the chosen optimized composition against the pure reference path unless `--skip-validate` is passed.

## Future Work

- multi-layer decode loops
- toy attention plus MLP stack with separate layer norms
- report integration across Apple Silicon generations
- decode-only throughput tables for common hidden sizes
