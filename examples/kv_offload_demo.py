"""Demonstrate KV block offload via the offload tier scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.kv_offload import KVResidencyMap, partition_sequence_into_blocks
from models.kv_offload_policy import KVOffloadPolicyConfig, plan_offload_blocks
from models.kv_offload_store import InMemoryKVOffloadStore
from ops.kv_offload_ops import apply_offload_plan


def main():
    print("=== KV Offload Tier Demo ===")
    print("Note: This is a simulated KV offload scaffold, not production flash/DMA streaming.\n")

    rng = np.random.default_rng(42)
    seq_len = 512
    block_size = 128
    num_layers = 1
    num_kv_heads = 4
    head_dim = 64

    caches = [(rng.normal(0, 0.02, (1, seq_len, num_kv_heads, head_dim)).astype(np.float32),
               rng.normal(0, 0.02, (1, seq_len, num_kv_heads, head_dim)).astype(np.float32))]

    rmap = KVResidencyMap()
    blocks = partition_sequence_into_blocks(
        layer_idx=0, batch_idx=0,
        seq_len=seq_len, block_size=block_size,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
        dtype="float32",
    )
    for b in blocks:
        rmap.add_block(b)

    print(f"Created {len(blocks)} KV blocks")
    print(f"Resident blocks: {len(rmap.resident_blocks())}")
    print(f"Offloaded blocks: {len(rmap.offloaded_blocks())}")
    print(f"Summary: {rmap.summary()}")

    store = InMemoryKVOffloadStore()
    policy = KVOffloadPolicyConfig(
        block_size=128,
        keep_sink_blocks=1,
        keep_recent_blocks=2,
        max_resident_blocks=2,
    ).validate()

    plan = plan_offload_blocks(rmap, current_position=384, policy_config=policy)
    print(f"\nOffload plan: {len(plan.offload)} blocks to offload, {len(plan.keep_resident)} keep resident")
    for bid in plan.offload:
        print(f"  Offloading: {bid.to_string()}")

    updated_caches, rmap, result = apply_offload_plan(caches, rmap, store, plan)
    print(f"\nAfter offload: {len(rmap.resident_blocks())} resident, {len(rmap.offloaded_blocks())} offloaded")
    print(f"Store stats: {store.stats()}")

    from models.kv_offload_policy import plan_prefetch_for_sparse_attention
    prefetch_plan = plan_prefetch_for_sparse_attention(
        rmap, layer_idx=0, batch_idx=0,
        needed_positions=[400],
        block_size=128,
    )
    print(f"\nPrefetch plan: {len(prefetch_plan.prefetch)} blocks to prefetch")
    for bid in prefetch_plan.prefetch:
        print(f"  Prefetching: {bid.to_string()}")

    updated_caches, rmap, _ = apply_offload_plan(caches, rmap, store, prefetch_plan)
    print(f"\nAfter prefetch: {len(rmap.resident_blocks())} resident, {len(rmap.offloaded_blocks())} offloaded")


if __name__ == "__main__":
    main()
