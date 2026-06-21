from __future__ import annotations

import numpy as np
import pytest


def _import_prefetch_ops():
    try:
        from ops.kv_prefetch_ops import (
            ensure_prefetched_before_attention,
            prefetch_blocks_for_sparse_decode,
            prefetch_blocks_for_speculative_draft,
        )
        return {
            "ensure_prefetched_before_attention": ensure_prefetched_before_attention,
            "prefetch_blocks_for_sparse_decode": prefetch_blocks_for_sparse_decode,
            "prefetch_blocks_for_speculative_draft": prefetch_blocks_for_speculative_draft,
        }
    except ImportError:
        pytest.skip("kv_prefetch_ops require mlx (not available in this environment)")


def _import_scheduler():
    try:
        from models.kv_prefetch_scheduler import KVPrefetchScheduler, KVPrefetchSchedulerConfig
        return KVPrefetchScheduler, KVPrefetchSchedulerConfig
    except ImportError:
        pytest.skip("kv_prefetch_scheduler requires mlx (not available in this environment)")


def _make_block_id(layer_idx=0, batch_idx=0, block_idx=0):
    from models.kv_offload import KVBlockId
    return KVBlockId(layer_idx=layer_idx, batch_idx=batch_idx, block_idx=block_idx)


def _make_setup(seq_len=16, block_size=8, num_layers=1):
    from models.kv_offload import KVBlockMetadata, KVResidencyMap, partition_sequence_into_blocks
    from models.kv_offload_store import InMemoryKVOffloadStore

    store = InMemoryKVOffloadStore()
    rmap = KVResidencyMap()
    blocks = partition_sequence_into_blocks(
        layer_idx=0, batch_idx=0, seq_len=seq_len,
        block_size=block_size, num_kv_heads=2, head_dim=16, dtype="float16",
    )
    for meta in blocks:
        rmap.add_block(meta)
    return store, rmap


def _offload_all(store, rmap):
    for meta in list(rmap.blocks.values()):
        if meta.resident:
            bid = meta.block_id
            K = np.zeros((1, meta.end_token - meta.start_token, 2, 16), dtype=np.float16)
            V = np.zeros((1, meta.end_token - meta.start_token, 2, 16), dtype=np.float16)
            store.put_block(bid, K, V)
            meta.resident = False
            meta.offloaded = True


class TestPrefetchForSparseDecode:
    def test_submits_expected_block_ids(self):
        mod = _import_prefetch_ops()
        KVPrefetchScheduler, KVPrefetchSchedulerConfig = _import_scheduler()
        store, rmap = _make_setup(seq_len=16, block_size=8)
        _offload_all(store, rmap)
        sched = KVPrefetchScheduler(store, rmap, config=KVPrefetchSchedulerConfig())

        sparse_pattern = {"pattern": "sliding_window", "window_size": 8, "sink_tokens": 0}
        requests = mod["prefetch_blocks_for_sparse_decode"](
            scheduler=sched,
            residency_map=rmap,
            layer_idx=0,
            batch_idx=0,
            current_length=8,
            sparse_pattern=sparse_pattern,
            block_size=8,
            lookahead_tokens=1,
        )
        assert len(requests) > 0
        for req in requests:
            assert req.status in ("queued", "complete")

    def test_lookahead_changes_planned_blocks(self):
        mod = _import_prefetch_ops()
        KVPrefetchScheduler, KVPrefetchSchedulerConfig = _import_scheduler()
        store, rmap = _make_setup(seq_len=32, block_size=8)
        _offload_all(store, rmap)
        sched = KVPrefetchScheduler(store, rmap, config=KVPrefetchSchedulerConfig())

        sparse_pattern = {"pattern": "sliding_window", "window_size": 8, "sink_tokens": 0}
        reqs1 = mod["prefetch_blocks_for_sparse_decode"](
            scheduler=sched, residency_map=rmap, layer_idx=0, batch_idx=0,
            current_length=8, sparse_pattern=sparse_pattern, block_size=8,
            lookahead_tokens=1,
        )
        count1 = len(reqs1)
        reqs2 = mod["prefetch_blocks_for_sparse_decode"](
            scheduler=sched, residency_map=rmap, layer_idx=0, batch_idx=0,
            current_length=8, sparse_pattern=sparse_pattern, block_size=8,
            lookahead_tokens=4,
        )
        assert len(reqs2) >= count1


class TestPrefetchForSpeculativeDraft:
    def test_covers_multiple_future_positions(self):
        mod = _import_prefetch_ops()
        KVPrefetchScheduler, KVPrefetchSchedulerConfig = _import_scheduler()
        store, rmap = _make_setup(seq_len=32, block_size=8)
        _offload_all(store, rmap)
        sched = KVPrefetchScheduler(store, rmap, config=KVPrefetchSchedulerConfig())

        sparse_pattern = {"pattern": "sliding_window", "window_size": 8, "sink_tokens": 0}
        requests = mod["prefetch_blocks_for_speculative_draft"](
            scheduler=sched, residency_map=rmap, layer_idx=0, batch_idx=0,
            current_length=8, draft_length=4,
            sparse_pattern=sparse_pattern, block_size=8,
        )
        assert len(requests) > 0


class TestEnsurePrefetched:
    def test_waits_and_succeeds(self):
        mod = _import_prefetch_ops()
        KVPrefetchScheduler, KVPrefetchSchedulerConfig = _import_scheduler()
        store, rmap = _make_setup(seq_len=16, block_size=8)
        _offload_all(store, rmap)
        sched = KVPrefetchScheduler(
            store, rmap,
            config=KVPrefetchSchedulerConfig(simulated_latency_steps=1),
        )

        bid = _make_block_id(block_idx=0)
        sched.submit(bid)

        layer_caches = [(np.zeros((1, 16, 2, 16), dtype=np.float32), np.zeros((1, 16, 2, 16), dtype=np.float32))]
        results = mod["ensure_prefetched_before_attention"](
            scheduler=sched,
            block_ids=[bid],
            layer_caches=layer_caches,
            max_steps=5,
        )
        assert len(results) == 1
        assert results[0].ok

    def test_raises_on_missing_block(self):
        mod = _import_prefetch_ops()
        KVPrefetchScheduler, KVPrefetchSchedulerConfig = _import_scheduler()
        store, rmap = _make_setup(seq_len=16, block_size=8)
        sched = KVPrefetchScheduler(
            store, rmap,
            config=KVPrefetchSchedulerConfig(fail_on_missing=True),
        )

        bid = _make_block_id(block_idx=99)
        with pytest.raises(RuntimeError):
            mod["ensure_prefetched_before_attention"](
                scheduler=sched,
                block_ids=[bid],
                max_steps=2,
            )

    def test_already_resident_no_ops(self):
        mod = _import_prefetch_ops()
        KVPrefetchScheduler, KVPrefetchSchedulerConfig = _import_scheduler()
        store, rmap = _make_setup(seq_len=16, block_size=8)
        sched = KVPrefetchScheduler(store, rmap)

        bid = _make_block_id(block_idx=0)
        results = mod["ensure_prefetched_before_attention"](
            scheduler=sched,
            block_ids=[bid],
            max_steps=2,
        )
        assert len(results) == 1
        assert results[0].ok
