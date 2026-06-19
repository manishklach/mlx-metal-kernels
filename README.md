# MLX Metal Kernels

Experimental high-performance custom Metal kernels for Apple Silicon using Apple’s MLX framework.

This repository explores GPU kernels for Mac-based machine learning workloads, starting with FlashAttention-style attention and expanding toward a broader library of Apple GPU inference primitives. The goal is to build correctness-first MLX custom kernels, validate them against pure MLX reference implementations, and then progressively optimize them using Apple Silicon GPU features such as threadgroup memory, SIMD-group reductions, tiled memory access, and specialized kernels for common transformer shapes.

The first kernel family focuses on fused attention:

- streaming softmax attention without materializing the full attention matrix
- causal and non-causal attention
- fp16 and bf16 support
- backend dispatch for reference, baseline, and experimental optimized kernels
- future decode and KV-cache kernels for LLM inference

Longer term, this repo is intended to become an experimental kernel lab for MLX on Mac, covering attention, decode, KV-cache operations, reductions, normalization, activation functions, quantization/dequantization, and other inference-oriented primitives.

## Project Goal

MLX makes Apple Silicon a serious local machine learning platform, but many high-performance model operations still benefit from custom fused kernels. This project investigates how far MLX custom Metal kernels can be pushed for transformer inference workloads on Mac.

The initial focus is attention. Standard attention materializes or conceptually computes the full `QK^T` score matrix before applying softmax and multiplying by `V`. FlashAttention-style kernels avoid materializing the full attention matrix by streaming over keys and values while maintaining online softmax statistics. This reduces memory traffic and creates a better foundation for long-context inference.

This repository starts with a correctness-first implementation and then adds progressively more optimized backends:

1. Pure MLX reference implementation
2. Baseline custom Metal streaming attention kernel
3. Row-parallel attention kernel
4. Tiled K/V attention kernel
5. Specialized D=64 and D=128 kernels
6. Decode attention for single-token inference
7. KV-cache and paged-KV kernels
8. Additional MLX custom kernels for transformer inference

The design philosophy is simple:

- correctness first
- every optimized backend must match the reference implementation
- no performance claims without benchmarks
- keep experimental kernels behind explicit backend flags
- make Apple Silicon GPU behavior visible and measurable

The goal is to compute:

```text
O = softmax(Q K^T * scale) V
```

without materializing the full `[S, S]` attention matrix.

## Current status

`v0.1` keeps a stable correctness-first baseline, adds a reference backend,
and includes experimental optimized backends:

- MLX Python wrapper around custom Metal kernels
- BSHD layout: `[batch, sequence, heads, head_dim]`
- fp16 and bf16 input/output
- causal and non-causal attention
- configurable scale
- `head_dim <= 128`
- reference MLX implementation for correctness tests
- benchmark CLI with backend selection and matrix mode

Backends:

- `reference`: pure MLX materialized reference implementation used for
  correctness validation.
- `baseline`: stable correctness-first kernel. One Metal thread computes one
  full attention row and streams over K/V in three passes.
- `row_parallel`: experimental kernel. One Metal threadgroup cooperates on one
  attention row, parallelizes row max and denominator reductions, and splits
  output accumulation across `D`.
- `tiled_kv`: experimental kernel. One threadgroup stages K/V tiles in
  threadgroup memory and streams over KV blocks with online softmax state.
- `auto`: currently aliases to `baseline` until the experimental path is
  consistently validated on Apple Silicon.

This is **not yet** a heavily optimized tiled/threadgroup-memory or
simdgroup-matrix FlashAttention kernel. The baseline path remains the default
stable backend, and the row-parallel path should be treated as experimental
until it passes tests and benchmarks on Apple Silicon.

## Install

```bash
pip install mlx pytest
```

Use editable mode from the repo root:

```bash
pip install -e .
pytest tests -q
python examples/run_basic.py
python benchmarks/bench_attention.py --backend all --S 128 --H 8 --D 64 --dtype float16
```

## Benchmark

```bash
python benchmarks/bench_attention.py --backend all --S 128 --H 8 --D 64 --dtype float16
python benchmarks/bench_attention.py --backend baseline --S 64 --H 4 --D 32 --dtype float16
python benchmarks/bench_attention.py --backend baseline --S 128 --H 8 --D 64 --dtype float16
python benchmarks/bench_attention.py --backend row_parallel --S 128 --H 8 --D 64 --dtype float16
python benchmarks/bench_attention.py --backend tiled_kv --S 128 --H 8 --D 64 --dtype float16
python benchmarks/bench_attention.py --backend baseline --matrix --H 8 --dtype float16
python benchmarks/bench_attention.py --backend row_parallel --matrix --H 8 --dtype float16
python benchmarks/bench_attention.py --backend tiled_kv --matrix --H 8 --dtype float16
python benchmarks/bench_attention.py --backend reference --matrix --H 8 --dtype float16
```

## API

```python
import mlx.core as mx
from ops.attention_ops import decode_attention, fast_attention

Q = mx.random.normal((1, 128, 8, 64)).astype(mx.float16)
K = mx.random.normal((1, 128, 8, 64)).astype(mx.float16)
V = mx.random.normal((1, 128, 8, 64)).astype(mx.float16)

O = fast_attention(Q, K, V, causal=True, backend="auto")
O_exp = fast_attention(Q, K, V, causal=True, backend="row_parallel")
O_tiled = fast_attention(Q, K, V, causal=True, backend="tiled_kv")

q = mx.random.normal((1, 1, 8, 64)).astype(mx.float16)
O_decode = decode_attention(q, K, V, backend="auto")
```

## Roadmap

1. Baseline streaming row kernel. **Stable default path.**
2. Row-parallel threadgroup kernel. **Experimental path.**
3. Tiled K/V threadgroup-memory kernel.
4. Simdgroup reduction kernel.
5. `simdgroup_matrix` QK/PV kernel.
6. Specialized `D=64` and `D=128` kernels.
7. Decode / paged-KV path.

## What this project is not claiming yet

This project does not yet claim to outperform MLX native attention or to match
CUDA/HIP FlashAttention kernels. Any performance claims should come only from
benchmarks run on Apple Silicon after both correctness paths pass validation.

## Verification status

The benchmark script validates non-reference backends against
`reference_attention` before timing them. The Metal kernels and MLX runtime
behavior still must be verified on Apple Silicon.
If you run this repo on a non-Apple host without `mlx`, code structure can be
updated but runtime correctness and performance remain unverified.
