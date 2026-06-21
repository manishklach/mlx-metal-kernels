from __future__ import annotations

import numpy as np
import pytest


def _import_scheduler():
    try:
        from models.kv_prefetch_scheduler import (
            KVPrefetchRequest,
            KVPrefetchRequestId,
            KVPrefetchResult,
            KVPrefetchScheduler,
            KVPrefetchSchedulerConfig,
            KVPrefetchSchedulerStats,
        )
        return {
            "KVPrefetchRequest": KVPrefetchRequest,
            "KVPrefetchRequestId": KVPrefetchRequestId,
            "KVPrefetchResult": KVPrefetchResult,
            "KVPrefetchScheduler": KVPrefetchScheduler,
            "KVPrefetchSchedulerConfig": KVPrefetchSchedulerConfig,
            "KVPrefetchSchedulerStats": KVPrefetchSchedulerStats,
        }
    except ImportError:
        pytest.skip("kv_prefetch_scheduler requires mlx (not available in this environment)")


def _make_block_id(layer_idx=0, batch_idx=0, block_idx=0):
    from models.kv_offload import KVBlockId
    return KVBlockId(layer_idx=layer_idx, batch_idx=batch_idx, block_idx=block_idx)


def _make_store_and_map(block_ids):
    from models.kv_offload import KVBlockMetadata, KVResidencyMap, partition_sequence_into_blocks
    from models.kv_offload_store import InMemoryKVOffloadStore

    store = InMemoryKVOffloadStore()
    rmap = KVResidencyMap()

    for bid in block_ids:
        block_size = 4
        start = bid.block_idx * block_size
        end = start + block_size
        meta = KVBlockMetadata(
            block_id=bid,
            layer_idx=bid.layer_idx,
            batch_idx=bid.batch_idx,
            block_idx=bid.block_idx,
            start_token=start,
            end_token=end,
            num_kv_heads=2,
            head_dim=16,
            dtype="float16",
            resident=True,
            offloaded=False,
        )
        rmap.add_block(meta)

    return store, rmap


def _offload_block(store, rmap, bid):
    meta = rmap.get(bid)
    if meta is None:
        return
    K = np.zeros((1, 4, 2, 16), dtype=np.float16)
    V = np.zeros((1, 4, 2, 16), dtype=np.float16)
    store.put_block(bid, K, V)
    meta.resident = False
    meta.offloaded = True


class TestConfig:
    def test_defaults(self):
        mod = _import_scheduler()
        cfg = mod["KVPrefetchSchedulerConfig"]()
        assert cfg.mode == "step"
        assert cfg.max_in_flight == 4
        assert cfg.simulated_latency_steps == 1
        cfg.validate()

    def test_invalid_mode(self):
        mod = _import_scheduler()
        with pytest.raises((ValueError, AssertionError)):
            mod["KVPrefetchSchedulerConfig"](mode="invalid").validate()

    def test_invalid_max_in_flight(self):
        mod = _import_scheduler()
        with pytest.raises((ValueError, AssertionError)):
            mod["KVPrefetchSchedulerConfig"](max_in_flight=0).validate()

    def test_invalid_latency_steps(self):
        mod = _import_scheduler()
        with pytest.raises((ValueError, AssertionError)):
            mod["KVPrefetchSchedulerConfig"](simulated_latency_steps=-1).validate()


class TestSubmit:
    def test_resident_block_returns_complete(self):
        mod = _import_scheduler()
        bid = _make_block_id(block_idx=0)
        store, rmap = _make_store_and_map([bid])
        sched = mod["KVPrefetchScheduler"](store, rmap)
        req = sched.submit(bid)
        assert req.status == "complete"
        assert req.metadata.get("resident_already") is True

    def test_offloaded_block_queues(self):
        mod = _import_scheduler()
        bid = _make_block_id(block_idx=0)
        store, rmap = _make_store_and_map([bid])
        _offload_block(store, rmap, bid)
        sched = mod["KVPrefetchScheduler"](store, rmap)
        req = sched.submit(bid)
        assert req.status == "queued"

    def test_deduplication(self):
        mod = _import_scheduler()
        bid = _make_block_id(block_idx=0)
        store, rmap = _make_store_and_map([bid])
        _offload_block(store, rmap, bid)
        sched = mod["KVPrefetchScheduler"](store, rmap, config=mod["KVPrefetchSchedulerConfig"](deduplicate_requests=True))
        req1 = sched.submit(bid)
        req2 = sched.submit(bid)
        assert req1.request_id == req2.request_id

    def test_priority_ordering(self):
        mod = _import_scheduler()
        bids = [_make_block_id(block_idx=i) for i in range(4)]
        store, rmap = _make_store_and_map(bids)
        for b in bids:
            _offload_block(store, rmap, b)
        sched = mod["KVPrefetchScheduler"](store, rmap, config=mod["KVPrefetchSchedulerConfig"](max_in_flight=2))
        sched.submit(bids[0], priority=0)
        sched.submit(bids[1], priority=10)
        sched.submit(bids[2], priority=5)
        sched.advance_step()
        in_flight_ids = [sched._block_key(r.block_id) for r in sched._in_flight.values()]
        high_key = sched._block_key(bids[1])
        mid_key = sched._block_key(bids[2])
        assert high_key in in_flight_ids
        assert mid_key in in_flight_ids


class TestAdvance:
    def test_queue_to_in_flight(self):
        mod = _import_scheduler()
        bid = _make_block_id(block_idx=0)
        store, rmap = _make_store_and_map([bid])
        _offload_block(store, rmap, bid)
        sched = mod["KVPrefetchScheduler"](store, rmap)
        sched.submit(bid)
        assert len(sched._queue) == 1
        sched.advance_step()
        assert len(sched._queue) == 0
        assert len(sched._in_flight) == 1

    def test_simulated_latency_controls_completion(self):
        mod = _import_scheduler()
        bid = _make_block_id(block_idx=0)
        store, rmap = _make_store_and_map([bid])
        _offload_block(store, rmap, bid)
        sched = mod["KVPrefetchScheduler"](
            store, rmap,
            config=mod["KVPrefetchSchedulerConfig"](simulated_latency_steps=2),
        )
        sched.submit(bid)
        sched.advance_step()
        assert len(sched._completed) == 0
        sched.advance_step()
        assert len(sched._completed) == 1
        assert bid.to_string() in sched._completed

    def test_missing_block_fails(self):
        mod = _import_scheduler()
        bid = _make_block_id(block_idx=99)
        store, rmap = _make_store_and_map([])
        sched = mod["KVPrefetchScheduler"](store, rmap, config=mod["KVPrefetchSchedulerConfig"](fail_on_missing=True))
        req = sched.submit(bid)
        sched.advance_step()
        assert req.status == "failed"


class TestWaitFor:
    def test_completes_within_max_steps(self):
        mod = _import_scheduler()
        bid = _make_block_id(block_idx=0)
        store, rmap = _make_store_and_map([bid])
        _offload_block(store, rmap, bid)
        sched = mod["KVPrefetchScheduler"](
            store, rmap,
            config=mod["KVPrefetchSchedulerConfig"](simulated_latency_steps=2),
        )
        sched.submit(bid)
        results = sched.wait_for([bid], max_steps=5)
        assert len(results) == 1
        assert results[0].ok
        assert results[0].status == "complete"

    def test_timeout_produces_failure(self):
        mod = _import_scheduler()
        bid = _make_block_id(block_idx=0)
        store, rmap = _make_store_and_map([bid])
        _offload_block(store, rmap, bid)
        sched = mod["KVPrefetchScheduler"](
            store, rmap,
            config=mod["KVPrefetchSchedulerConfig"](simulated_latency_steps=10),
        )
        sched.submit(bid)
        results = sched.wait_for([bid], max_steps=3)
        assert any(not r.ok for r in results)


class TestStats:
    def test_stats_update(self):
        mod = _import_scheduler()
        bid = _make_block_id(block_idx=0)
        store, rmap = _make_store_and_map([bid])
        _offload_block(store, rmap, bid)
        sched = mod["KVPrefetchScheduler"](store, rmap)
        sched.submit(bid)
        sched.advance_step()
        stats = sched.stats_dict()
        assert stats["issued"] > 0
        assert stats["queued"] == 0
        assert stats["in_flight_count"] == 0
        assert stats["completed_count"] >= 0

    def test_cancel_queued(self):
        mod = _import_scheduler()
        bid = _make_block_id(block_idx=0)
        store, rmap = _make_store_and_map([bid])
        _offload_block(store, rmap, bid)
        sched = mod["KVPrefetchScheduler"](store, rmap)
        req = sched.submit(bid)
        cancelled = sched.cancel(req.request_id)
        assert cancelled
        assert req.status == "cancelled"

    def test_clear_completed(self):
        mod = _import_scheduler()
        bid = _make_block_id(block_idx=0)
        store, rmap = _make_store_and_map([bid])
        sched = mod["KVPrefetchScheduler"](store, rmap)
        req = sched.submit(bid)
        assert req.status == "complete"
        count = sched.clear_completed()
        assert count >= 1

    def test_pending_block_ids(self):
        mod = _import_scheduler()
        bid = _make_block_id(block_idx=0)
        store, rmap = _make_store_and_map([bid])
        _offload_block(store, rmap, bid)
        sched = mod["KVPrefetchScheduler"](store, rmap)
        sched.submit(bid)
        pending = sched.pending_block_ids()
        assert len(pending) == 1

    def test_submit_many(self):
        mod = _import_scheduler()
        bids = [_make_block_id(block_idx=i) for i in range(3)]
        store, rmap = _make_store_and_map(bids)
        for b in bids:
            _offload_block(store, rmap, b)
        sched = mod["KVPrefetchScheduler"](store, rmap)
        reqs = sched.submit_many(bids)
        assert len(reqs) == 3
        assert all(r.status == "queued" for r in reqs)

    def test_request_id_build(self):
        mod = _import_scheduler()
        rid = mod["KVPrefetchRequestId"].build(layer_idx=1, batch_idx=0, block_idx=5, seq=42)
        assert "L1_B0_BLK5_SEQ42" in rid.to_string()

    def test_request_to_dict(self):
        mod = _import_scheduler()
        bid = _make_block_id(block_idx=0)
        rid = mod["KVPrefetchRequestId"].build(layer_idx=0, batch_idx=0, block_idx=0)
        req = mod["KVPrefetchRequest"](request_id=rid, block_id=bid, status="queued")
        d = req.to_dict()
        assert d["status"] == "queued"
        assert d["is_done"] is False

    def test_stats_to_dict(self):
        mod = _import_scheduler()
        stats = mod["KVPrefetchSchedulerStats"](queued=1, in_flight=2)
        d = stats.to_dict()
        assert d["queued"] == 1
        assert d["in_flight"] == 2
