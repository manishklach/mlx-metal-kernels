"""Measure KV block offload and prefetch throughput for the offload tier scaffold."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.kv_offload import KVBlockId, KVBlockMetadata, KVResidencyMap, partition_sequence_into_blocks
from models.kv_offload_policy import KVOffloadPolicyConfig, plan_offload_blocks, plan_prefetch_for_sparse_attention
from models.kv_offload_store import FileKVOffloadStore, InMemoryKVOffloadStore
from ops.kv_offload_ops import apply_offload_plan


def _make_synthetic_cache(seq_len, num_layers, num_kv_heads, head_dim):
    rng = np.random.default_rng(0)
    caches = []
    for layer in range(num_layers):
        K = rng.normal(0, 0.02, size=(1, seq_len, num_kv_heads, head_dim)).astype(np.float32)
        V = rng.normal(0, 0.02, size=(1, seq_len, num_kv_heads, head_dim)).astype(np.float32)
        caches.append((K, V))
    return caches


def _time_fn(fn, iters=5):
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1e3
        samples.append(elapsed)
    return statistics.fmean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV offload tier")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--store", choices=["memory", "file"], default="memory")
    parser.add_argument("--offload-ratio", type=float, default=0.75)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--simulated-latency-ms", type=float, default=0.0)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    num_blocks = (args.seq_len + args.block_size - 1) // args.block_size
    num_offload = int(num_blocks * args.offload_ratio)

    print(f"Creating synthetic KV cache: seq_len={args.seq_len}, layers={args.num_layers}, "
          f"heads={args.num_kv_heads}, dim={args.head_dim}")
    caches = _make_synthetic_cache(args.seq_len, args.num_layers, args.num_kv_heads, args.head_dim)

    print("Partitioning into blocks...")
    rmap = KVResidencyMap()
    for layer in range(args.num_layers):
        blocks = partition_sequence_into_blocks(
            layer_idx=layer, batch_idx=0,
            seq_len=args.seq_len, block_size=args.block_size,
            num_kv_heads=args.num_kv_heads, head_dim=args.head_dim,
            dtype=args.dtype,
        )
        for b in blocks:
            rmap.add_block(b)

    if args.store == "file":
        import tempfile
        tmpdir = tempfile.mkdtemp()
        store = FileKVOffloadStore(tmpdir)
    else:
        store = InMemoryKVOffloadStore()

    policy = KVOffloadPolicyConfig(
        block_size=args.block_size,
        keep_sink_blocks=max(1, args.sink_tokens // args.block_size) if args.sink_tokens > 0 else 0,
        keep_recent_blocks=max(1, args.window_size // args.block_size),
        max_resident_blocks=num_blocks - num_offload,
        simulated_latency_ms=args.simulated_latency_ms,
    ).validate()

    plan = plan_offload_blocks(rmap, current_position=args.seq_len - 1, policy_config=policy)
    print(f"Offload plan: {len(plan.offload)} blocks to offload, "
          f"{len(plan.keep_resident)} keep resident")

    offload_mean_ms, offload_std_ms = _time_fn(
        lambda: apply_offload_plan(caches, rmap, store, plan),
        iters=args.iters,
    )

    window_blocks = (args.window_size + args.block_size - 1) // args.block_size
    needed_positions = list(range(args.seq_len - args.window_size, args.seq_len))
    if args.sink_tokens > 0:
        needed_positions = list(range(args.sink_tokens)) + needed_positions

    prefetch_plan = plan_prefetch_for_sparse_attention(
        rmap, layer_idx=args.num_layers - 1, batch_idx=0,
        needed_positions=needed_positions,
        block_size=args.block_size,
    )

    prefetch_mean_ms, prefetch_std_ms = _time_fn(
        lambda: apply_offload_plan(caches, rmap, store, prefetch_plan),
        iters=args.iters,
    )

    bytes_per_block = 2 * args.block_size * args.num_kv_heads * args.head_dim * 2  # *2 for K+V, *2 for float16 bytes
    total_offload_bytes = len(plan.offload) * bytes_per_block
    total_prefetch_bytes = len(prefetch_plan.prefetch) * bytes_per_block

    print()
    print("--- KV Offload Tier Benchmark Results ---")
    print(f"seq_len: {args.seq_len}")
    print(f"block_size: {args.block_size}")
    print(f"num_layers: {args.num_layers}")
    print(f"num_kv_heads: {args.num_kv_heads}")
    print(f"head_dim: {args.head_dim}")
    print(f"store: {args.store}")
    print(f"total_blocks: {len(rmap.blocks)}")
    print(f"offloaded_blocks: {len(plan.offload)}")
    print(f"resident_blocks: {len(rmap.blocks) - len(plan.offload)}")
    print(f"bytes_offloaded: {total_offload_bytes}")
    print(f"offload_mean_ms: {offload_mean_ms:.2f}")
    print(f"offload_std_ms: {offload_std_ms:.2f}")
    print(f"prefetch_blocks: {len(prefetch_plan.prefetch)}")
    print(f"bytes_prefetch: {total_prefetch_bytes}")
    print(f"prefetch_mean_ms: {prefetch_mean_ms:.2f}")
    print(f"prefetch_std_ms: {prefetch_std_ms:.2f}")
    print(f"window_size: {args.window_size}")
    print(f"sink_tokens: {args.sink_tokens}")
    print(f"simulated_latency_ms: {args.simulated_latency_ms}")
    if args.simulated_latency_ms > 0:
        print("WARNING: Simulated latency is active. These are NOT real flash/DMA timings.")


if __name__ == "__main__":
    main()
