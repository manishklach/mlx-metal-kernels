# Transformer Kernels

## Attention

- Purpose: fused softmax attention without materializing the full attention matrix.
- Formula: `softmax(QK^T * scale) V`
- Shapes: `Q/K/V [B, S, H, D]`
- Backend status: `reference`, `baseline`, `row_parallel`, `tiled_kv`
- Test status: covered by `tests/test_correctness.py`
- Benchmark command: `python benchmarks/bench_attention.py --backend all --S 128 --H 8 --D 64 --dtype float16`

## RMSNorm

- Purpose: row-wise RMS normalization with learned scale.
- Formula: `x * rsqrt(mean(x^2) + eps) * weight`
- Shapes: `x [B, S, D]`, `weight [D]`
- Backend status: `reference`, `metal`
- Test status: covered by `tests/test_rms_norm.py`
- Benchmark command: `python benchmarks/bench_rms_norm.py --B 2 --S 8 --D 1024 --dtype float16 --backend metal`

## RoPE

- Purpose: rotary positional embedding application for transformer attention inputs.
- Formula: even/odd pair rotation using `cos` and `sin`
- Shapes: `x [B, S, H, D]`, `cos/sin [S_total, D/2]`
- Backend status: `reference`, `metal`
- Test status: covered by `tests/test_rope.py`
- Benchmark command: `python benchmarks/bench_rope.py --B 2 --S 16 --H 8 --D 128 --dtype float16 --backend metal`

## SwiGLU

- Purpose: fused SiLU gate with multiplicative up projection.
- Formula: `silu(gate) * up`
- Shapes: `gate/up [B, S, D]`
- Backend status: `reference`, `metal`
- Test status: covered by `tests/test_swiglu.py`
- Benchmark command: `python benchmarks/bench_swiglu.py --B 2 --S 16 --D 256 --dtype float16 --backend metal`

## Decode Attention

- Purpose: single-token decode attention over cached keys and values.
- Formula: `softmax(q K_cache^T * scale) V_cache`
- Shapes: `q [B, 1, H, D]`, `K_cache/V_cache [B, S, H, D]`
- Backend status: `reference`, `metal`
- Test status: covered by `tests/test_decode_attention.py`
- Benchmark command: `python benchmarks/bench_decode_attention.py --B 2 --S 32 --H 8 --D 64 --dtype float16 --backend metal`
