# GQA Prefill Attention

## Why GQA prefill matters

Grouped-query attention matters during prompt processing just as much as during decode. Prefill attention handles full query sequences instead of a single-token query.

## Decode vs prefill

Decode uses:

- `q [B, 1, Hq, D]`

Prefill uses:

- `Q [B, S, Hq, D]`
- `K [B, S, Hkv, D]`
- `V [B, S, Hkv, D]`

## Head mapping

For each query head:

- `group = Hq / Hkv`
- `hkv = hq // group`

This lets multiple query heads attend to the same KV head without explicitly expanding KV to `Hq`.

## Why avoiding KV expansion matters

Reference paths can validate correctness by expanding KV heads, but optimized paths should avoid that extra memory traffic and extra storage.

## Backends

- `reference`
- `metal_gqa`
- `metal_gqa_threadgroup`

## Causal vs non-causal support

Both causal and non-causal prefill are supported.

Current limitation:

- causal mode currently requires `Sq == Sk`

## Benchmark commands

```bash
python benchmarks/bench_gqa_attention.py --B 1 --S 128 --Hq 32 --Hkv 8 --D 128 --dtype float16 --causal --backend all --validate
python benchmarks/bench_gqa_attention.py --B 1 --S 256 --Hq 32 --Hkv 8 --D 128 --dtype float16 --causal --backend all --validate
```

## Current limitations

- `D <= 128` for Metal backends
- explicit-only backend selection
- no simdgroup GQA prefill backend yet
- no full prompt runtime yet

## Future

- shape-specialized GQA `D=64` and `D=128` prefill kernels
- simdgroup GQA prefill experiments
- fused QKV split + RoPE + GQA attention
- full prompt prefill layer benchmark
