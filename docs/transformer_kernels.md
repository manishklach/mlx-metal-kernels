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
- Test status: covered by `tests/test_decode_attention_optimized.py`
- Benchmark command: `python benchmarks/bench_decode_attention.py --B 2 --MAX_S 32 --H 8 --D 64 --length 32 --dtype float16 --backend metal`

## KV-cache Update

- Purpose: write one new token of K/V state into a flat BSHD cache.
- Formula: replace `K_cache[b, pos]` and `V_cache[b, pos]` with `k_new[b]` / `v_new[b]`
- Shapes: `K_cache/V_cache [B, MAX_S, H, D]`, `k_new/v_new [B, 1, H, D]`
- Backend status: `reference`, `metal`
- Test status: covered by `tests/test_kv_cache_update.py`
- Benchmark command: `python benchmarks/bench_kv_cache_update.py --B 2 --MAX_S 128 --H 8 --D 64 --dtype float16 --backend metal`

## Decode Loop

- Purpose: compose cache update plus single-token decode for autoregressive generation.
- Formula: `kv_cache_update` then `decode_attention(..., lengths=position + 1)`
- Shapes: token inputs plus caches in BSHD layout
- Backend status: helper built on `reference` and `metal` backends
- Test status: covered by `tests/test_decode_loop.py`
- Benchmark command: `python benchmarks/bench_decode_loop.py --B 2 --MAX_S 64 --T 16 --H 8 --D 64 --dtype float16 --backend metal`
