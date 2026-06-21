# Flash/NAND KV Offload Tier Scaffold

## Purpose

This module provides a **correctness-first scaffold** for treating cold KV-cache blocks as offloadable objects that may live outside hot unified memory. The first implementation is a local simulated/offline tier, not a production flash/DMA runtime.

## Why KV offload matters

As sequence lengths grow (128K, 1M+), the KV cache can exceed available unified memory on Apple Silicon. Offloading cold blocks to slower storage (SSD, NAND) while keeping hot blocks resident enables longer-context inference without OOM.

## Hot vs cold KV blocks

| Status | Location | Behavior |
|--------|----------|----------|
| Resident | MLX array in unified memory | Can be attended to immediately |
| Offloaded | In-memory store or `.npy` files | Must be prefetched before attention |
| Non-resident | Unknown | Runtime error if attention tries to read |

## Block metadata

Each KV block is described by `KVBlockMetadata`:

- `block_id` — layer, batch, block index (plus optional KV-head range)
- `start_token` / `end_token` — absolute token range in the sequence
- `num_tokens`, `num_kv_heads`, `head_dim`, `dtype` — shape info
- `resident` / `offloaded` — residency status
- `store_uri`, `checksum` — offload location
- `last_access_step` — for LRU-like policies

**Shape convention:** `[1, num_tokens, num_kv_heads, head_dim]` (preserves batch dim).

## Residency map

`KVResidencyMap` tracks all blocks across layers:

```python
rmap = KVResidencyMap()
rmap.add_block(meta)
rmap.mark_offloaded(block_id, store_uri="memory://...")
rmap.mark_resident(block_id)
rmap.resident_blocks()    # list of resident meta
rmap.offloaded_blocks()   # list of offloaded meta
rmap.blocks_for_token_range(layer, batch, start, end)
rmap.blocks_for_sparse_positions(layer, batch, positions)
rmap.summary()
```

## Offload stores

### InMemoryKVOffloadStore

Stores blocks in a Python dict. Used for tests and demos.

### FileKVOffloadStore

Stores each block as:
```
root_dir/L0_B0_BLK0/k.npy
root_dir/L0_B0_BLK0/v.npy
root_dir/L0_B0_BLK0/meta.json
```

- `format="npy"` (default, only supported format)
- `overwrite=False` by default (raises `FileExistsError`)

## Offload policy

`KVOffloadPolicyConfig`:

| Field | Default | Description |
|-------|---------|-------------|
| `block_size` | 128 | Token range per block |
| `keep_recent_blocks` | 4 | Blocks near current position to keep resident |
| `keep_sink_blocks` | 1 | Initial blocks to always keep resident |
| `max_resident_blocks` | None | Hard limit on resident blocks |
| `offload_enabled` | True | Master enable switch |
| `simulated_latency_ms` | 0.0 | Optional artificial latency for benchmarking |

### `plan_offload_blocks`

Returns `KVOffloadPlan` with:
- `keep_resident` — blocks to keep in cache
- `offload` — blocks to move to store
- `prefetch` — blocks to load back

### `plan_prefetch_for_sparse_attention`

Given needed token positions (from `sparse_positions_for_decode`), identifies which required blocks are offloaded and need prefetching.

## Offload operations (`ops/kv_offload_ops.py`)

| Function | Description |
|----------|-------------|
| `extract_kv_block(cache, start, end)` | Extract a slice of the layer cache |
| `insert_kv_block(cache, start, k, v)` | Insert a block into the cache (returns new cache) |
| `offload_kv_block(cache, meta, store)` | Extract + store + mark offloaded |
| `prefetch_kv_block(cache, meta, store)` | Load + insert + mark resident |
| `apply_offload_plan(caches, rmap, store, plan)` | Execute an offload/prefetch plan |

## Sparse attention integration

### `ensure_sparse_blocks_resident`

Raises `RuntimeError` if any needed block for sparse attention is offloaded.

### `sparse_positions_for_decode(length, pattern)`

Returns token positions that a query at position `length-1` will attend to under the given attention pattern (dense, sliding_window, sliding_window_sink).

## Prefix-cache relationship

`clone_residency_map()` in `ops/kv_cache_reuse_ops.py` provides a deep-ish copy of a `KVResidencyMap` that can be stored alongside a `PrefixCacheEntry` for future reuse of offload metadata.

## Benchmark commands

```bash
# Quick in-memory benchmark
python benchmarks/bench_kv_offload_tier.py \
    --seq-len 4096 --block-size 128 \
    --num-layers 2 --num-kv-heads 8 --head-dim 128 \
    --store memory --offload-ratio 0.75 \
    --window-size 512 --sink-tokens 4

# With simulated latency
python benchmarks/bench_kv_offload_tier.py \
    --seq-len 4096 --block-size 128 \
    --num-layers 2 --num-kv-heads 8 --head-dim 128 \
    --store memory --offload-ratio 0.75 \
    --window-size 512 --sink-tokens 4 \
    --simulated-latency-ms 5.0

# File-backed store
python benchmarks/bench_kv_offload_tier.py \
    --seq-len 1024 --block-size 128 \
    --store file --offload-ratio 0.5
```

## Current limitations

- scaffold only
- local file/in-memory store only
- no real async IO
- no DMA
- no production flash scheduler
- no automatic runtime offload (must be called explicitly)
- contiguous cache first (paged cache offload not yet supported)
- B=1 focused initially

## Future work

- async prefetch
- mmap-backed KV blocks
- Metal/CPU overlap scheduling
- flash/NAND latency-aware sparse attention
- quantized KV offload
- integration with speculative decoding
- page-level offload for paged KV-cache
- automatic LRU offload during generation
