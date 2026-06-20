# Simdgroup Attention Experiments

PR #15 adds an experimental `simdgroup_d64` backend for `fast_attention`.

## What Metal Simdgroups Are

Metal simdgroups are the subgroup-sized execution units inside a threadgroup. Threads inside the same simdgroup can cooperate with lower-latency collective operations such as reductions, broadcasts, and lane shuffles.

For attention kernels, simdgroup-style cooperation is attractive because a single query row naturally decomposes into:

- collaborative `Q·K` score computation
- online softmax state updates
- collaborative accumulation over `V`

## Why Simdgroup Matrix Work Is Interesting

Apple GPUs expose simdgroup features that can reduce scalar reduction overhead and, on supporting toolchains, may provide a path toward matrix-style tile operations. A future attention kernel could use that machinery to make `QK` and possibly `P@V` tile updates more efficient than a purely scalar threadgroup reduction path.

This PR intentionally takes the narrower working step first:

- `D=64`
- `mx.float16`
- prefill attention only
- explicit backend name only

## Current Experimental Scope

Backend:

- `fast_attention(..., backend="simdgroup_d64")`

Current limitations:

- only `D == 64`
- only `mx.float16`
- not selected by `backend="auto"`
- intended as an experiment, not a stable default

The first version uses simdgroup cooperation and lane-partitioned accumulation as a correctness-first precursor to a broader simdgroup-matrix path. That keeps the implementation narrow enough to test without disturbing the existing stable backends.

## Limitations

- not production-ready
- may not compile on all Apple GPU / Metal toolchain combinations
- may fail at runtime if simdgroup support differs from the assumptions in this kernel
- may not outperform the threadgroup backend yet
- decode attention simdgroup backend is not included in PR #15

If the backend cannot compile or execute, the Python wrapper raises a clear experimental-backend availability error so tests and benchmarks can report or skip it explicitly.

## Benchmarking

Use:

```bash
python benchmarks/bench_simdgroup_attention.py --mode prefill --B 1 --S 128 --H 8 --D 64 --dtype float16 --backend all
```

The benchmark validates the backend against `reference_attention` before timing. It reports failures instead of assuming the experimental kernel is available everywhere.

## Future Work

- true `simdgroup_matrix`-style `QK` tile experiments
- causal tuning and wider validation coverage
- `D=128` support
- `P@V` simdgroup path
- K/V threadgroup tiling combined with simdgroup reductions
- multi-query-row tile processing
- shared simdgroup reduction utilities
- chip-specific tuning across M-series generations
