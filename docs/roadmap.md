# Roadmap

## v0.1: Correctness baseline

- Body-only MLX custom Metal kernel source.
- Pass `scale` and `causal` from Python to Metal.
- Use `ELEM_TYPE` so fp16 and bf16 both work.
- Avoid `preprocess_v` shape mismatch by making it a no-op until real K/V repack is implemented.
- Test against materialized MLX reference attention.

## Preserved optimization roadmap

1. v0.1 baseline streaming kernel
2. v0.2 row_parallel backend
3. v0.3 tiled K/V backend
4. v0.4 specialized `D=64` / `D=128` kernels
5. v0.5 decode attention
6. v0.6 paged KV cache
7. v1.0 stable package

## v0.2: Row-parallel streaming kernel

The v0.1 kernel assigns one Metal thread to one query row. This is simple but
leaves too much parallelism unused. The next step is to split one query row
across a threadgroup:

- one threadgroup per `(b, h, q)`
- lanes collaboratively compute Q·K dot products
- threadgroup reduction for row max
- threadgroup reduction for denominator
- split output accumulation across `D`

## v0.3: Tiled K/V

- Stage K and/or V tiles into threadgroup memory.
- Use online softmax tile merge:

```text
m_new = max(m_old, max(scores_tile))
l_new = exp(m_old - m_new) * l_old + sum(exp(scores_tile - m_new))
o_new = exp(m_old - m_new) * o_old + exp(scores_tile - m_new) @ V_tile
```

- Keep `baseline` and `row_parallel` available as separate backends while the
  tiled path matures.

## v0.4: Apple GPU specialization

- Separate D=64 and D=128 kernels.
- Explore `simdgroup_matrix` for QK and PV sub-blocks.
- Benchmark on M1/M2/M3/M4 and Pro/Max/Ultra variants.

## v0.5: Decode path

- Single-token query decode.
- Reference decode scaffold in Python first.

## v0.6: Paged KV cache

- Paged KV cache support.
- Optional split-KV merge.

## v1.0: Stable package

- Verified backend defaults.
- Apple Silicon benchmark coverage.
- Clear guarantees around supported backends and head dimensions.
