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
4. v0.4 Layout + fused transformer block helpers
5. v0.5 Quantization + decode matvec kernels
6. v0.6 Paged KV-cache + paged decode attention
7. v0.7 Fused decode block
8. v0.8 Shape-specialized attention/decode kernels
9. v0.9 Parallel q4/q8 matvec optimization
10. v0.10 Apple-chip benchmark suite + report generator
11. v0.11 fused quantized decode block
12. v0.12 SIMD/threadgroup optimized attention v2
13. v0.13 Multi-output q4/q8 matvec tiling
14. v0.14 End-to-end toy transformer layer decode benchmark
15. v0.15 simdgroup_matrix attention experiments
16. v0.16 Chip-specific autotuning + backend selection registry
17. v0.17 Real model integration scaffold
18. v0.18 q4 MLP fused block
19. v0.19 GQA/MQA support
20. v0.20 real checkpoint loader scaffold
21. v1.0 stable experimental kernel suite

## v0.2: Transformer primitives

- Add correctness-first `RMSNorm`, `RoPE`, `SwiGLU`, and `decode_attention`.
- Give each primitive a pure MLX reference path plus a Metal backend.
- Add dedicated tests and small benchmark scripts for each primitive.

## v0.3: KV-cache update + optimized decode attention

- Add `kv_cache_update` with reference and Metal backends.
- Add decode-specific `decode_attention` with prefix-length support.
- Add `decode_step` helper that composes cache update plus decode.
- Keep the path correctness-first and explicit about backend status.

## v0.4: Layout + fused transformer block helpers

- Add packed and explicit QKV layout split helpers.
- Add split+RoPE and split+RoPE+cache-update helpers for decode composition.
- Add residual add and RMSNorm+residual helpers.
- Keep the fused decode helper as composition, not a monolithic fused block yet.

## v0.5: Quantization + decode matvec kernels

- Add q4 and q8 dequantization helpers.
- Add correctness-first decode matvec kernels that dequantize on the fly.
- Keep the first implementation simple and reference-validated.

## v0.6: Paged KV-cache + paged decode attention

- Add paged cache metadata and update paths.
- Introduce block-table aware decode helpers.

## v0.7: Fused decode block

- Add composition-first contiguous decode block helpers from projected QKV tokens.
- Add paged decode block helpers on top of block-table aware cache/update paths.
- Reuse residual plus RMSNorm block helpers without duplicating the underlying math kernel.

## v0.8: Shape-specialized attention/decode kernels

- Add D=64 / D=128 specialized attention and decode kernels.
- Keep specialized dispatch explicit and conservative by default.
- Add generic-vs-specialized benchmark coverage for decode, paged decode, and full attention.

## v0.9: Optimized q4/q8 matvec reductions

- Parallelize reductions over K.
- Add explicit `metal_parallel` backends for q4 and q8 decode matvec.
- Keep the original one-thread-per-output-element kernel as the conservative default.

## v0.10: Apple-chip benchmark suite + report generator

- Add a unified benchmark runner that records JSON and CSV output.
- Capture system metadata needed to compare Apple Silicon results.
- Generate Markdown benchmark reports and comparison tables from saved runs.

## v0.11: Fused quantized decode block

- Add composition-first q4/q8 QKV projection helpers on top of the decode matvec kernels.
- Add contiguous and paged quantized decode blocks by reusing the existing decode block helpers.
- Keep every quantized decode path reference-validated before benchmarking.

## v0.12: SIMD/threadgroup optimized attention v2

- Add experimental threadgroup decode attention for contiguous KV caches.
- Add experimental threadgroup paged decode attention over block tables.
- Add experimental threadgroup full-attention prefill backend.
- Keep `auto` conservative and gate threadgroup routing behind an explicit environment variable.

## v0.13: Multi-output q4/q8 matvec tiling

- Add experimental `metal_tiled` q4 and q8 decode matvec backends.
- Reuse one activation load across a small tile of output channels per threadgroup.
- Keep the first tiled version simple and reference-validated before performance claims.

## v0.14: End-to-end toy transformer layer decode benchmark

- Add a correctness-first toy single-layer decode composition built from existing repo primitives.
- Benchmark contiguous and paged decode-layer paths on top of quantized attention and SwiGLU.
- Keep benchmark validation anchored to the pure reference composition before timing optimized presets.

## v0.15: simdgroup_matrix attention experiments

- Add an explicit `simdgroup_d64` prefill attention backend for `D=64` and `mx.float16`.
- Keep simdgroup experiments off the default and `auto` paths.
- Treat compilation/runtime availability as platform-specific and report failures clearly.

## v0.16: Chip-specific autotuning + backend selection registry

- Add a backend registry for operations with multiple candidate kernels.
- Add an opt-in local autotune cache keyed by operation, shape, dtype, and machine information.
- Keep runtime selection conservative unless a tuned local result is available.

## v0.17: Real model integration scaffold

- Add Llama-like config, weight-layout specs, and adapter helpers around the current decode-layer kernels.
- Keep the first scaffold explicit, lightweight, and focused on future checkpoint integration.

## v0.18: q4 MLP fused block

- Extend quantized transformer composition beyond attention into MLP-heavy decode blocks.
- Reuse validated q4 matvec primitives before adding new fused kernels.

## v0.19: GQA/MQA support

- Add grouped-query and multi-query cache/update/decode support.
- Extend the model adapter once KV-head expansion or native grouped-query handling is available.

## v0.20: real checkpoint loader scaffold

- Add a lightweight checkpoint-loading bridge once the layout contracts stabilize.
- Keep production model loading, tokenizer integration, and full inference loops out of scope until the scaffold is validated.

## Historical notes

The items below remain important background for the attention kernel family.

## Row-parallel streaming kernel

The v0.1 kernel assigns one Metal thread to one query row. This is simple but
leaves too much parallelism unused. The next step is to split one query row
across a threadgroup:

- one threadgroup per `(b, h, q)`
- lanes collaboratively compute Q·K dot products
- threadgroup reduction for row max
- threadgroup reduction for denominator
- split output accumulation across `D`

## Tiled K/V

- Stage K and/or V tiles into threadgroup memory.
- Use online softmax tile merge:

```text
m_new = max(m_old, max(scores_tile))
l_new = exp(m_old - m_new) * l_old + sum(exp(scores_tile - m_new))
o_new = exp(m_old - m_new) * o_old + exp(scores_tile - m_new) @ V_tile
```

- Keep `baseline` and `row_parallel` available as separate backends while the
  tiled path matures.

## v1.0: Stable package

- Verified backend defaults.
- Apple Silicon benchmark coverage.
- Clear guarantees around supported backends and head dimensions.
