from __future__ import annotations

from dataclasses import dataclass

import pytest

try:
    import mlx.core as mx  # noqa: F401
except ImportError:
    pytest.skip("sparse_attention_ops require mlx (not available in this environment)", allow_module_level=True)

from models.kv_offload import (
    KVBlockId,
    KVBlockMetadata,
    KVResidencyMap,
    partition_sequence_into_blocks,
)
from ops.sparse_attention_ops import (
    SparseAttentionPattern,
    ensure_sparse_blocks_resident,
    sparse_positions_for_decode,
)


def _make_residency_map_with_offloaded(seq_len=512, block_size=128):
    rmap = KVResidencyMap()
    blocks = partition_sequence_into_blocks(
        layer_idx=0, batch_idx=0,
        seq_len=seq_len, block_size=block_size,
        num_kv_heads=4, head_dim=64, dtype="float16",
    )
    for b in blocks:
        rmap.add_block(b)
    for key, meta in rmap.blocks.items():
        if meta.block_id.block_idx == 2:
            meta.resident = False
            meta.offloaded = True
            meta.store_uri = "memory://test"
        if meta.block_id.block_idx == 3:
            meta.resident = False
            meta.offloaded = True
            meta.store_uri = "memory://test"
    return rmap


class TestSparsePositionsForDecode:
    def test_dense(self):
        pattern = SparseAttentionPattern(pattern="dense", causal=True)
        positions = sparse_positions_for_decode(128, pattern)
        assert len(positions) == 128

    def test_sliding_window(self):
        pattern = SparseAttentionPattern(pattern="sliding_window", window_size=64)
        positions = sparse_positions_for_decode(256, pattern)
        assert len(positions) == 64
        assert positions[0] == 193

    def test_sliding_window_sink(self):
        pattern = SparseAttentionPattern(pattern="sliding_window_sink", window_size=64, sink_tokens=4)
        positions = sparse_positions_for_decode(256, pattern)
        assert positions[0] == 0
        assert positions[1] == 1
        assert positions[2] == 2
        assert positions[3] == 3
        assert len(positions) == 4 + 64

    def test_short_sequence(self):
        pattern = SparseAttentionPattern(pattern="sliding_window", window_size=64)
        positions = sparse_positions_for_decode(32, pattern)
        assert len(positions) == 32

    def test_zero_length(self):
        pattern = SparseAttentionPattern(pattern="dense")
        positions = sparse_positions_for_decode(0, pattern)
        assert positions == []


class TestEnsureSparseBlocksResident:
    def test_passes_with_resident_blocks(self):
        rmap = _make_residency_map_with_offloaded(seq_len=512, block_size=128)
        ensure_sparse_blocks_resident(
            rmap,
            layer_idx=0, batch_idx=0,
            needed_positions=[0, 1, 100],
            block_size=128,
        )

    def test_raises_with_offloaded_needed_block(self):
        rmap = _make_residency_map_with_offloaded(seq_len=512, block_size=128)
        with pytest.raises(RuntimeError, match="offloaded"):
            ensure_sparse_blocks_resident(
                rmap,
                layer_idx=0, batch_idx=0,
                needed_positions=[256, 300],
                block_size=128,
            )

    def test_raises_with_multiple_offloaded(self):
        rmap = _make_residency_map_with_offloaded(seq_len=512, block_size=128)
        with pytest.raises(RuntimeError) as excinfo:
            ensure_sparse_blocks_resident(
                rmap,
                layer_idx=0, batch_idx=0,
                needed_positions=[256, 400],
                block_size=128,
            )
        msg = str(excinfo.value)
        assert "L0_B0_BLK2" in msg or "L0_B0_BLK3" in msg


class TestPrefetchIntegration:
    def test_prefetch_plan_includes_offloaded(self):
        from models.kv_offload_policy import plan_prefetch_for_sparse_attention

        rmap = _make_residency_map_with_offloaded(seq_len=512, block_size=128)
        plan = plan_prefetch_for_sparse_attention(
            rmap,
            layer_idx=0, batch_idx=0,
            needed_positions=[256],
            block_size=128,
        )
        prefetched_idxs = {bid.block_idx for bid in plan.prefetch}
        assert 2 in prefetched_idxs

    def test_after_prefetch_guard_passes(self):
        from models.kv_offload_policy import plan_prefetch_for_sparse_attention
        from models.kv_offload_store import InMemoryKVOffloadStore
        from ops.kv_offload_ops import prefetch_kv_block

        import numpy as np

        rmap = _make_residency_map_with_offloaded(seq_len=512, block_size=128)
        store = InMemoryKVOffloadStore()
        cache = (
            np.random.default_rng(0).normal(0, 1, (1, 512, 4, 64)).astype(np.float32),
            np.random.default_rng(1).normal(0, 1, (1, 512, 4, 64)).astype(np.float32),
        )

        for key, meta in rmap.blocks.items():
            if meta.offloaded:
                k_block = cache[0][:, meta.start_token:meta.end_token]
                v_block = cache[1][:, meta.start_token:meta.end_token]
                store.put_block(meta.block_id, k_block, v_block)

        plan = plan_prefetch_for_sparse_attention(
            rmap, layer_idx=0, batch_idx=0,
            needed_positions=[256], block_size=128,
        )
        for bid in plan.prefetch:
            meta = rmap.get(bid)
            if meta and meta.offloaded:
                cache, _ = prefetch_kv_block(cache, meta, store)

        ensure_sparse_blocks_resident(
            rmap, layer_idx=0, batch_idx=0,
            needed_positions=[256], block_size=128,
        )
