from __future__ import annotations

import argparse
import time

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None


def _import_scheduler():
    from models.kv_prefetch_scheduler import KVPrefetchScheduler, KVPrefetchSchedulerConfig
    return KVPrefetchScheduler, KVPrefetchSchedulerConfig


def _import_offload():
    from models.kv_offload import KVResidencyMap, KVBlockId, partition_sequence_into_blocks
    from models.kv_offload_store import InMemoryKVOffloadStore, FileKVOffloadStore
    return KVResidencyMap, KVBlockId, partition_sequence_into_blocks, InMemoryKVOffloadStore, FileKVOffloadStore


def _import_ops():
    from ops.kv_prefetch_ops import prefetch_blocks_for_sparse_decode
    return prefetch_blocks_for_sparse_decode


def time_fn(fn, warmup=1, iters=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.mean(times), np.std(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--lookahead-tokens", type=int, default=8)
    parser.add_argument("--draft-length", type=int, default=4)
    parser.add_argument("--max-in-flight", type=int, default=4)
    parser.add_argument("--simulated-latency-steps", type=int, default=2)
    parser.add_argument("--simulated-latency-ms", type=float, default=0.0)
    parser.add_argument("--store", choices=["memory", "file"], default="memory")
    parser.add_argument("--offload-ratio", type=float, default=0.75)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    KVPrefetchScheduler, KVPrefetchSchedulerConfig = _import_scheduler()
    KVResidencyMap, KVBlockId, partition_sequence_into_blocks, InMemoryKVOffloadStore, FileKVOffloadStore = _import_offload()
    prefetch_blocks_for_sparse_decode = _import_ops()

    rmap = KVResidencyMap()
    for layer_idx in range(args.num_layers):
        blocks = partition_sequence_into_blocks(
            layer_idx=layer_idx, batch_idx=0, seq_len=args.seq_len,
            block_size=args.block_size,
            num_kv_heads=args.num_kv_heads, head_dim=args.head_dim, dtype="float16",
        )
        for meta in blocks:
            rmap.add_block(meta)

    total_blocks = len(rmap.blocks)
    num_offload = int(total_blocks * args.offload_ratio)

    if args.store == "memory":
        store = InMemoryKVOffloadStore()
    else:
        import tempfile
        store = FileKVOffloadStore(tempfile.mkdtemp())

    all_metas = sorted(rmap.blocks.values(), key=lambda m: m.block_id.block_idx)
    offload_targets = all_metas[:num_offload] if args.offload_ratio < 1.0 else all_metas
    for meta in offload_targets:
        bid = meta.block_id
        K = np.zeros((1, args.block_size, args.num_kv_heads, args.head_dim), dtype=np.float16)
        V = np.zeros((1, args.block_size, args.num_kv_heads, args.head_dim), dtype=np.float16)
        store.put_block(bid, K, V)
        meta.resident = False
        meta.offloaded = True

    blocks_resident_after_offload = sum(1 for m in rmap.blocks.values() if m.resident)
    blocks_offloaded = total_blocks - blocks_resident_after_offload

    sparse_pattern = {
        "pattern": "sliding_window_sink",
        "window_size": args.window_size,
        "sink_tokens": args.sink_tokens,
        "causal": True,
    }

    layer_caches = [
        (np.zeros((1, args.seq_len, args.num_kv_heads, args.head_dim), dtype=np.float32),
         np.zeros((1, args.seq_len, args.num_kv_heads, args.head_dim), dtype=np.float32))
        for _ in range(args.num_layers)
    ]

    decode_positions = list(range(128, min(args.seq_len, 2048), 64))

    def run_synchronous():
        from ops.long_context_ops import needed_positions_for_sparse_decode, ensure_blocks_ready_for_attention
        from models.kv_offload import token_positions_to_block_ids
        from ops.kv_offload_ops import prefetch_kv_block
        total_wait = 0
        for pos in decode_positions:
            needed = needed_positions_for_sparse_decode(length=pos, sparse_pattern=sparse_pattern)
            if not needed:
                continue
            for layer_idx in range(args.num_layers):
                block_ids = token_positions_to_block_ids(needed, layer_idx=layer_idx, batch_idx=0, block_size=args.block_size)
                for bid in block_ids:
                    meta = rmap.get(bid)
                    if meta is not None and meta.offloaded:
                        updated_cache, _ = prefetch_kv_block(layer_caches[layer_idx], meta, store)
                        layer_caches[layer_idx] = updated_cache
                        total_wait += 1
        return total_wait

    def run_scheduled():
        sched = KVPrefetchScheduler(
            store, rmap,
            config=KVPrefetchSchedulerConfig(
                max_in_flight=args.max_in_flight,
                simulated_latency_steps=args.simulated_latency_steps,
            ),
        )
        total_wait = 0
        for pos in decode_positions:
            for layer_idx in range(args.num_layers):
                prefetch_blocks_for_sparse_decode(
                    scheduler=sched, residency_map=rmap,
                    layer_idx=layer_idx, batch_idx=0,
                    current_length=pos, sparse_pattern=sparse_pattern,
                    block_size=args.block_size,
                    lookahead_tokens=args.lookahead_tokens,
                )
            sched.advance_step(layer_caches=layer_caches)
            results = sched.poll_ready(layer_caches=layer_caches)
            total_wait += len(results)
        return total_wait

    sync_wait, sync_std = time_fn(run_synchronous, warmup=1, iters=args.iters)
    sched_wait, sched_std = time_fn(run_scheduled, warmup=1, iters=args.iters)

    sync_total = run_synchronous()
    sched_total = run_scheduled()

    print("=== KV Prefetch Scheduler Benchmark ===")
    print(f"seq_len={args.seq_len}")
    print(f"block_size={args.block_size}")
    print(f"window_size={args.window_size}")
    print(f"lookahead_tokens={args.lookahead_tokens}")
    print(f"max_in_flight={args.max_in_flight}")
    print(f"simulated_latency_steps={args.simulated_latency_steps}")
    print(f"offload_ratio={args.offload_ratio}")
    print(f"total_blocks={total_blocks} offloaded={blocks_offloaded} resident={blocks_resident_after_offload}")
    print(f"decode_positions={len(decode_positions)}")
    print()
    print(f"synchronous_total_prefetches={sync_total}")
    print(f"scheduled_total_prefetches={sched_total}")
    print(f"synchronous_mean_ms={sync_wait:.3f} std={sync_std:.3f}")
    print(f"scheduled_mean_ms={sched_wait:.3f} std={sched_std:.3f}")
    print()
    print("NOTE: Simulated latency metrics. Not production async IO or DMA.")
    print(f"NOTE: Scheduled prefetch uses simulated_latency_steps={args.simulated_latency_steps}.")


if __name__ == "__main__":
    main()
