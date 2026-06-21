"""Demonstrate sparse attention integration with KV offload tier."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.kv_offload import KVResidencyMap, partition_sequence_into_blocks
from models.kv_offload_policy import KVOffloadPolicyConfig, plan_offload_blocks, plan_prefetch_for_sparse_attention
from models.kv_offload_store import InMemoryKVOffloadStore
from ops.kv_offload_ops import apply_offload_plan, prefetch_kv_block
from ops.sparse_attention_ops import (
    SparseAttentionPattern,
    ensure_sparse_blocks_resident,
    sparse_positions_for_decode,
)


def main():
    print("=== Sparse KV Offload Integration Demo ===")
    print("Note: This is a simulated KV offload scaffold, not production flash/DMA streaming.\n")

    seq_len = 512
    block_size = 128
    num_kv_heads = 4
    head_dim = 64

    rng = np.random.default_rng(42)
    cache = (
        rng.normal(0, 0.02, (1, seq_len, num_kv_heads, head_dim)).astype(np.float32),
        rng.normal(0, 0.02, (1, seq_len, num_kv_heads, head_dim)).astype(np.float32),
    )

    rmap = KVResidencyMap()
    blocks = partition_sequence_into_blocks(
        layer_idx=0, batch_idx=0,
        seq_len=seq_len, block_size=block_size,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
        dtype="float32",
    )
    for b in blocks:
        rmap.add_block(b)

    print("Created sliding-window + sink attention pattern...")
    pattern = SparseAttentionPattern(
        pattern="sliding_window_sink",
        window_size=128,
        sink_tokens=4,
    ).validate()

    needed = sparse_positions_for_decode(seq_len, pattern)
    print(f"For decode at position {seq_len - 1}, need {len(needed)} KV positions")
    print(f"  Sink positions: {[p for p in needed if p < 4]}")
    print(f"  Window positions: [{min(needed)}, {max(needed)}]")

    store = InMemoryKVOffloadStore()
    policy = KVOffloadPolicyConfig(
        block_size=128,
        keep_sink_blocks=1,
        keep_recent_blocks=2,
        max_resident_blocks=2,
    ).validate()

    plan = plan_offload_blocks(rmap, current_position=seq_len - 1, policy_config=policy)
    apply_offload_plan([cache], rmap, store, plan)

    print(f"\nBlocks offloaded: {len(rmap.offloaded_blocks())}")
    print(f"Blocks resident: {len(rmap.resident_blocks())}")

    print("\nChecking sparse attention residency guard...")
    try:
        ensure_sparse_blocks_resident(
            rmap,
            layer_idx=0, batch_idx=0,
            needed_positions=needed,
            block_size=block_size,
        )
        print("  PASSED: all needed blocks are resident")
    except RuntimeError as e:
        print(f"  FAILED (expected for demo): {e}")

    prefetch_plan = plan_prefetch_for_sparse_attention(
        rmap, layer_idx=0, batch_idx=0,
        needed_positions=needed,
        block_size=block_size,
    )
    print(f"\nPrefetch plan: {len(prefetch_plan.prefetch)} blocks to prefetch")
    for bid in prefetch_plan.prefetch:
        print(f"  Need to prefetch: {bid.to_string()}")

    apply_offload_plan([cache], rmap, store, prefetch_plan)

    print("\nAfter prefetch, checking residency guard again...")
    ensure_sparse_blocks_resident(
        rmap,
        layer_idx=0, batch_idx=0,
        needed_positions=needed,
        block_size=block_size,
    )
    print("  PASSED: all needed blocks now resident.")


if __name__ == "__main__":
    main()
