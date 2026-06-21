"""
KV Prefetch Scheduler Demo

Demonstrates a deterministic step-based KV block prefetch scheduler
for long-context offload experiments.

WARNING: This is a deterministic simulated prefetch scheduler,
not production async flash/DMA.
"""

import numpy as np

from models.kv_offload import (
    KVBlockMetadata,
    KVResidencyMap,
    KVBlockId,
    partition_sequence_into_blocks,
)
from models.kv_offload_store import InMemoryKVOffloadStore
from models.kv_prefetch_scheduler import KVPrefetchScheduler, KVPrefetchSchedulerConfig
from ops.kv_prefetch_ops import prefetch_blocks_for_sparse_decode


def main():
    print("=== KV Prefetch Scheduler Demo ===")
    print("WARNING: Deterministic simulated prefetch, not production async flash/DMA.\n")

    seq_len = 64
    block_size = 8
    num_layers = 1
    window_size = 16
    sink_tokens = 2
    lookahead = 3

    rmap = KVResidencyMap()
    blocks = partition_sequence_into_blocks(
        layer_idx=0, batch_idx=0, seq_len=seq_len,
        block_size=block_size, num_kv_heads=2, head_dim=16, dtype="float16",
    )
    for meta in blocks:
        rmap.add_block(meta)

    store = InMemoryKVOffloadStore()

    for meta in blocks:
        bid = meta.block_id
        K = np.zeros((1, block_size, 2, 16), dtype=np.float16)
        V = np.zeros((1, block_size, 2, 16), dtype=np.float16)
        store.put_block(bid, K, V)
        meta.resident = False
        meta.offloaded = True

    sched = KVPrefetchScheduler(
        store, rmap,
        config=KVPrefetchSchedulerConfig(
            max_in_flight=2,
            simulated_latency_steps=2,
            deduplicate_requests=True,
        ),
    )

    print(f"Initial blocks: {len(rmap.blocks)} total, "
          f"{sum(1 for m in rmap.blocks.values() if m.resident)} resident, "
          f"{sum(1 for m in rmap.blocks.values() if m.offloaded)} offloaded\n")

    sparse_pattern = {
        "pattern": "sliding_window_sink",
        "window_size": window_size,
        "sink_tokens": sink_tokens,
        "causal": True,
    }

    layer_caches = [
        (np.zeros((1, seq_len, 2, 16), dtype=np.float32),
         np.zeros((1, seq_len, 2, 16), dtype=np.float32))
    ]

    for current_length in [16, 24, 32, 40]:
        print(f"--- Decode position {current_length} ---")
        requests = prefetch_blocks_for_sparse_decode(
            scheduler=sched,
            residency_map=rmap,
            layer_idx=0,
            batch_idx=0,
            current_length=current_length,
            sparse_pattern=sparse_pattern,
            block_size=block_size,
            lookahead_tokens=lookahead,
        )

        if requests:
            print(f"  Submitted {len(requests)} prefetch request(s)")
            for req in requests:
                print(f"    {req.request_id.to_string()}: status={req.status}, reason={req.reason}")
        else:
            print("  No blocks to prefetch")

        sched.advance_step(layer_caches=layer_caches)
        print(f"  Scheduler step {sched.current_step}")
        stats = sched.stats_dict()
        print(f"  Stats: {stats['queued']} queued, {stats['in_flight_count']} in-flight, "
              f"{stats['completed_count']} completed, {stats['failed_count']} failed")

        results = sched.poll_ready(layer_caches=layer_caches)
        for r in results:
            print(f"  Poll result: {r.block_id.to_string()} -> {r.status} ({r.message})")

        resident_blocks = sum(1 for m in rmap.blocks.values() if m.resident)
        offloaded_blocks = sum(1 for m in rmap.blocks.values() if m.offloaded)
        print(f"  Residency: {resident_blocks} resident, {offloaded_blocks} offloaded\n")

    print("=== Final Summary ===")
    print(sched.describe())
    print()
    print("NOTE: This is a simulated, deterministic scheduler.")
    print("No production async IO, DMA, or SSD streaming is performed.")


if __name__ == "__main__":
    main()
