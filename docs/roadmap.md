# Roadmap

## v0.1: Correctness baseline

- Body-only MLX custom Metal kernel source.
- Pass `scale` and `causal` from Python to Metal.
- Use `ELEM_TYPE` so fp16 and bf16 both work.
- Avoid `preprocess_v` shape mismatch by making it a no-op until real K/V repack is implemented.
- Test against materialized MLX reference attention.

## Preserved optimization roadmap

1. v0.1 attention baseline
2. v0.2 transformer primitives: RMSNorm, RoPE, SwiGLU, decode scaffold
3. v0.3 KV-cache update + optimized decode attention
4. v0.4 row-parallel / tiled attention
5. v0.5 paged KV cache
6. v0.6 quantized decode matvec
7. v0.7 fused decode block
8. v1.0 stable experimental kernel suite

## v0.2: Transformer primitives

- Add correctness-first `RMSNorm`, `RoPE`, `SwiGLU`, and `decode_attention`.
- Give each primitive a pure MLX reference path plus a Metal backend.
- Add dedicated tests and small benchmark scripts for each primitive.

## v0.3: KV-cache update + optimized decode attention

- Add `kv_cache_update` with reference and Metal backends.
- Add decode-specific `decode_attention` with prefix-length support.
- Add `decode_step` helper that composes cache update plus decode.
- Keep the path correctness-first and explicit about backend status.

## v0.4: Row-parallel streaming kernel

The v0.1 kernel assigns one Metal thread to one query row. This is simple but
leaves too much parallelism unused. The next step is to split one query row
across a threadgroup:

- one threadgroup per `(b, h, q)`
- lanes collaboratively compute Q·K dot products
- threadgroup reduction for row max
- threadgroup reduction for denominator
- split output accumulation across `D`

## v0.5: Tiled K/V

- Stage K and/or V tiles into threadgroup memory.
- Use online softmax tile merge:

```text
m_new = max(m_old, max(scores_tile))
l_new = exp(m_old - m_new) * l_old + sum(exp(scores_tile - m_new))
o_new = exp(m_old - m_new) * o_old + exp(scores_tile - m_new) @ V_tile
```

- Keep `baseline` and `row_parallel` available as separate backends while the
  tiled path matures.

## v0.6: Quantized decode matvec

- Add q4 dequantization helpers and matvec kernels for decode workloads.

## v0.7: Fused decode block

- Fuse cache update, decode attention, and adjacent decode primitives where practical.

## v1.0: Stable package

- Verified backend defaults.
- Apple Silicon benchmark coverage.
- Clear guarantees around supported backends and head dimensions.
