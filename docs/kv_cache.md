# KV Cache

## BSHD cache layout

The repo currently uses a simple cache layout:

- `K_cache [B, MAX_S, H, D]`
- `V_cache [B, MAX_S, H, D]`

This matches the project-wide BSHD convention and keeps the first KV-cache path
easy to validate against pure MLX reference code.

## `kv_cache_update`

`kv_cache_update(K_cache, V_cache, k_new, v_new, positions, backend="auto")`
updates one token position per batch item and returns new cache tensors.

Supported position forms:

- Python `int`
- MLX scalar
- per-batch `mx.array [B]`

The current implementation is correctness-first. Because MLX kernels are used
through value-returning semantics here, the function returns updated cache
arrays instead of assuming in-place mutation.

## Future paged KV direction

The next step after flat BSHD cache update is paged KV cache support:

- logical token positions mapped through a block table
- fixed page size
- physical cache pages stored separately from logical sequence order
- gather/scatter style kernels for decode paths
