# Specialized Kernels

Shape specialization matters because transformer workloads often repeat a small set of head dimensions. Fixing `D` at compile time makes the kernels easier to reason about today and creates a better base for future unrolling and reduction work.

## First targets

This repo starts with `D=64` and `D=128`, which are common transformer head sizes and already fit the project’s `head_dim <= 128` scope.

## Specialized decode attention

`decode_attention(...)` now supports:

- `backend="metal_d64"`
- `backend="metal_d128"`

These paths use fixed-size `HEAD_DIM` local storage while preserving the same correctness-first behavior as the generic decode kernel.

## Specialized paged decode attention

`paged_decode_attention(...)` also supports:

- `backend="metal_d64"`
- `backend="metal_d128"`

The kernel still streams logical positions through the block table and updates online softmax state row by row.

## Specialized full attention

`fast_attention(...)` supports:

- `backend="baseline_d64"`
- `backend="baseline_d128"`

These are specialized baseline kernels for full or prefill attention. They are still experimental and correctness-first.

## Current status

The v1 specialized kernels are intentionally simple:

- one thread per output row
- float accumulation
- online softmax updates
- fixed `D=64` or `D=128`

`backend="auto"` stays conservative by default. If you want to opt into automatic routing for supported shapes during local experiments, set `MLX_METAL_USE_SPECIALIZED=1`.

They may not outperform the generic kernels yet. Performance claims should come only from Apple Silicon benchmarks after correctness validation passes.

## Future optimization ideas

- manual loop unrolling
- threadgroup reductions
- simdgroup reductions
- D-specific memory tiling
- chip-specific tuning for M1, M2, M3, and M4 GPUs
