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

## Kernel Families

- Attention: reference, baseline, row-parallel, and tiled-K/V fused attention backends.
- RMSNorm: correctness-first row-wise normalization with a pure MLX path and a Metal backend.
- RoPE: rotary embedding application for transformer attention inputs.
- SwiGLU: fused SiLU gate times up-projection activation.
- KV-cache update: correctness-first cache write path for single-token K/V updates.
- Decode Attention: single-token attention over KV cache tensors.
- Layout and fused helpers: QKV split, split+RoPE, cache-update fusion, residual add, and RMSNorm+residual.
- Quantization: q4/q8 dequantization and correctness-first decode matvec kernels.
- Paged KV-cache: paged cache allocation, updates, and paged decode attention scaffolds.
- Future: paged KV, quantized matvec, and tiled attention kernels.

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
python benchmarks/bench_rms_norm.py --B 2 --S 8 --D 1024 --dtype float16 --backend metal
python benchmarks/bench_rope.py --B 2 --S 16 --H 8 --D 128 --dtype float16 --backend metal
python benchmarks/bench_swiglu.py --B 2 --S 16 --D 256 --dtype float16 --backend metal
python benchmarks/bench_kv_cache_update.py --B 2 --MAX_S 128 --H 8 --D 64 --dtype float16 --backend metal
python benchmarks/bench_decode_attention.py --B 2 --MAX_S 32 --H 8 --D 64 --length 32 --dtype float16 --backend metal
python benchmarks/bench_decode_loop.py --B 2 --MAX_S 64 --T 16 --H 8 --D 64 --dtype float16 --backend metal
python benchmarks/bench_qkv_split.py --B 2 --S 16 --H 8 --D 64 --dtype float16 --backend all
python benchmarks/bench_fused_qkv_rope_cache.py --B 2 --MAX_S 128 --H 8 --D 64 --dtype float16 --backend all
python benchmarks/bench_residual_norm.py --B 2 --S 16 --D 1024 --dtype float16 --backend all
python benchmarks/bench_dequant.py --bits 4 --M 4096 --K 4096 --dtype float16 --backend all
python benchmarks/bench_quant_matvec_decode.py --bits 4 --B 1 --K 4096 --N 4096 --dtype float16 --backend all
python benchmarks/bench_paged_kv_cache_update.py --B 2 --MAX_S 128 --PAGE_SIZE 16 --H 8 --D 64 --dtype float16 --backend all
python benchmarks/bench_paged_decode_attention.py --B 2 --MAX_S 128 --PAGE_SIZE 16 --H 8 --D 64 --length 128 --dtype float16 --backend all
python benchmarks/bench_paged_decode_loop.py --B 2 --MAX_S 128 --PAGE_SIZE 16 --T 32 --H 8 --D 64 --dtype float16 --backend all
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
python benchmarks/bench_kv_cache_update.py --B 2 --MAX_S 128 --H 8 --D 64 --dtype float16 --backend metal
python benchmarks/bench_decode_attention.py --B 2 --MAX_S 32 --H 8 --D 64 --length 32 --dtype float16 --backend all
python benchmarks/bench_decode_loop.py --B 2 --MAX_S 64 --T 16 --H 8 --D 64 --dtype float16 --backend metal
python benchmarks/bench_qkv_split.py --B 2 --S 16 --H 8 --D 64 --dtype float16 --layout packed --backend all
python benchmarks/bench_fused_qkv_rope_cache.py --B 2 --MAX_S 128 --H 8 --D 64 --dtype float16 --backend all
python benchmarks/bench_residual_norm.py --B 2 --S 16 --D 1024 --dtype float16 --backend all
```

## API

```python
import mlx.core as mx
from ops.activation_ops import swiglu
from ops.attention_ops import fast_attention
from ops.decode_ops import decode_attention, decode_step
from ops.fused_ops import fused_decode_step_from_qkv, qkv_rope_cache_update, residual_add, rmsnorm_residual
from ops.kv_cache_ops import kv_cache_update
from ops.layout_ops import qkv_split, qkv_split_rope
from ops.norm_ops import rms_norm
from ops.paged_kv_ops import allocate_paged_kv_cache, paged_decode_attention, paged_decode_step, paged_kv_cache_update
from ops.quant_ops import dequant_q4, dequant_q8, pack_q4, q4_matvec_decode, q8_matvec_decode
from ops.rope_ops import apply_rope

Q = mx.random.normal((1, 128, 8, 64)).astype(mx.float16)
K = mx.random.normal((1, 128, 8, 64)).astype(mx.float16)
V = mx.random.normal((1, 128, 8, 64)).astype(mx.float16)

O = fast_attention(Q, K, V, causal=True, backend="auto")
O_exp = fast_attention(Q, K, V, causal=True, backend="row_parallel")
O_tiled = fast_attention(Q, K, V, causal=True, backend="tiled_kv")

x = mx.random.normal((2, 8, 1024)).astype(mx.float16)
weight = mx.ones((1024,), dtype=mx.float16)
y_norm = rms_norm(x, weight, backend="auto")

rope_inp = mx.random.normal((1, 16, 8, 128)).astype(mx.float16)
cos = mx.random.normal((32, 64)).astype(mx.float32)
sin = mx.random.normal((32, 64)).astype(mx.float32)
y_rope = apply_rope(rope_inp, cos, sin, backend="auto")

gate = mx.random.normal((2, 16, 256)).astype(mx.float16)
up = mx.random.normal((2, 16, 256)).astype(mx.float16)
y_swiglu = swiglu(gate, up, backend="auto")

MAX_S = 16
T = 8
K_cache = mx.zeros((1, MAX_S, 8, 64), dtype=mx.float16)
V_cache = mx.zeros((1, MAX_S, 8, 64), dtype=mx.float16)
q = mx.random.normal((1, 1, 8, 64)).astype(mx.float16)
k_new = mx.random.normal((1, 1, 8, 64)).astype(mx.float16)
v_new = mx.random.normal((1, 1, 8, 64)).astype(mx.float16)

K_cache, V_cache = kv_cache_update(K_cache, V_cache, k_new, v_new, 0)
O_decode = decode_attention(q, K_cache, V_cache, lengths=1, backend="auto")
O_step, K_cache, V_cache = decode_step(q, k_new, v_new, K_cache, V_cache, 1, backend="auto")

packed_qkv = mx.random.normal((1, 1, 3 * 8 * 64)).astype(mx.float16)
q_tok, k_tok, v_tok = qkv_split(packed_qkv, H=8, D=64, backend="auto")
q_rope, k_rope, v_tok = qkv_split_rope(packed_qkv, cos, sin, H=8, D=64, position_offset=0, backend="auto")
q_only, K_cache, V_cache = qkv_rope_cache_update(packed_qkv, K_cache, V_cache, cos, sin, 2, H=8, D=64, backend="auto")
y_add = residual_add(x[:, :1, :64], x[:, :1, :64], backend="auto")
y_norm_res, z_res = rmsnorm_residual(x, x, weight, return_residual=True, backend="auto")
out_fused, K_cache, V_cache = fused_decode_step_from_qkv(packed_qkv, K_cache, V_cache, cos, sin, 3, H=8, D=64, backend="auto")

q4_vals = (mx.random.uniform((32, 64)) * 16).astype(mx.uint8)
packed_w = pack_q4(q4_vals)
scales = mx.ones((32, 2), dtype=mx.float32)
W_deq = dequant_q4(packed_w, scales, group_size=32, backend="auto")
y_q4 = q4_matvec_decode(mx.random.normal((1, 64)).astype(mx.float16), packed_w, scales, group_size=32, backend="auto")

q8_vals = (mx.random.uniform((32, 64)) * 255).astype(mx.uint8)
y_q8 = q8_matvec_decode(mx.random.normal((1, 64)).astype(mx.float16), q8_vals, scales, group_size=32, backend="auto")

PAGE_SIZE = 4
K_pages, V_pages, block_table = allocate_paged_kv_cache(1, MAX_S, 8, 64, PAGE_SIZE, dtype=mx.float16)
K_pages, V_pages = paged_kv_cache_update(K_pages, V_pages, k_new, v_new, block_table, 0)
out_paged = paged_decode_attention(q, K_pages, V_pages, block_table, lengths=1, backend="auto")
out_step, K_pages, V_pages = paged_decode_step(q, k_new, v_new, K_pages, V_pages, block_table, 1, backend="auto")
```

## Transformer Primitive Benchmarks

```bash
python benchmarks/bench_rms_norm.py --B 2 --S 8 --D 1024 --dtype float16 --backend metal
python benchmarks/bench_rope.py --B 2 --S 16 --H 8 --D 128 --dtype float16 --backend metal
python benchmarks/bench_swiglu.py --B 2 --S 16 --D 256 --dtype float16 --backend metal
python benchmarks/bench_kv_cache_update.py --B 2 --MAX_S 128 --H 8 --D 64 --dtype float16 --backend metal
python benchmarks/bench_decode_attention.py --B 2 --MAX_S 32 --H 8 --D 64 --length 32 --dtype float16 --backend metal
python benchmarks/bench_decode_loop.py --B 2 --MAX_S 64 --T 16 --H 8 --D 64 --dtype float16 --backend metal
python benchmarks/bench_qkv_split.py --B 2 --S 16 --H 8 --D 64 --dtype float16 --backend all
python benchmarks/bench_fused_qkv_rope_cache.py --B 2 --MAX_S 128 --H 8 --D 64 --dtype float16 --backend all
python benchmarks/bench_residual_norm.py --B 2 --S 16 --D 1024 --dtype float16 --backend all
python benchmarks/bench_dequant.py --bits 4 --M 4096 --K 4096 --dtype float16 --backend all
python benchmarks/bench_quant_matvec_decode.py --bits 4 --B 1 --K 4096 --N 4096 --dtype float16 --backend all
python benchmarks/bench_paged_kv_cache_update.py --B 2 --MAX_S 128 --PAGE_SIZE 16 --H 8 --D 64 --dtype float16 --backend all
python benchmarks/bench_paged_decode_attention.py --B 2 --MAX_S 128 --PAGE_SIZE 16 --H 8 --D 64 --length 128 --dtype float16 --backend all
python benchmarks/bench_paged_decode_loop.py --B 2 --MAX_S 128 --PAGE_SIZE 16 --T 32 --H 8 --D 64 --dtype float16 --backend all
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
