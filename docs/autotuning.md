# Autotuning

PR #16 adds an opt-in backend registry plus local autotune cache for operations that already have multiple Metal backends.

## Why Autotuning Exists

Apple Silicon performance is shape-dependent and machine-dependent. The best backend for one operation can change with:

- chip generation
- GPU size
- memory bandwidth
- head dimension
- sequence length
- cache layout

Because of that, this repo does not assume one backend is universally fastest. Autotuning records local measurements and lets runtime selection stay conservative unless a machine has been tuned.

## Backend Registry

The registry lives in `benchmarks/backend_registry.py`.

It maps an operation to:

- the callable path
- its reference backend
- candidate tuned backends

It also applies simple shape and dtype filtering, for example:

- `d64` backends require `D == 64`
- `d128` backends require `D == 128`
- `simdgroup_d64` currently requires `D == 64` and `float16`

## Local Cache

Default cache path:

```text
~/.cache/mlx-metal-kernels/autotune_results.json
```

The cache stores:

- op name
- normalized shape
- dtype
- system information
- best backend
- all collected timings
- timestamp
- status

These results are local machine hints, not universal performance claims.

## Quick vs Full

Quick autotune:

- smaller representative shapes
- good for first-pass local setup

Full autotune:

- broader shapes per operation
- better when you want a more useful local cache

## Commands

```bash
python benchmarks/autotune.py --op all --quick --dtype float16 --write-cache
python benchmarks/autotune.py --op decode_attention --full --dtype float16 --include-experimental --write-cache
```

## Runtime Use

```python
from ops.autotune_ops import select_backend

backend = select_backend(
    "decode_attention",
    {"B": 1, "MAX_S": 128, "H": 8, "D": 64, "length": 128},
    "float16",
    default_backend="metal",
)
```

If no tuned result is found, `select_backend` returns a conservative default unless `require_tuned=True` is requested.

## Unified Benchmark Runner

The unified runner can optionally use saved autotune results:

```bash
python benchmarks/run_all_benchmarks.py --quick --use-autotune
```

This does not make autotuning mandatory. Without the flag, the runner keeps its normal explicit backend coverage.

## Deleting the Cache

Delete the JSON file at the cache path above to force a fresh local autotune run.
