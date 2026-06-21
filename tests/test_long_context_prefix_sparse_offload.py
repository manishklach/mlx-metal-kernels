from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from models.kv_offload import KVResidencyMap, partition_sequence_into_blocks, KVBlockId
from models.kv_offload_policy import KVOffloadPolicyConfig, plan_offload_blocks
from models.kv_offload_store import InMemoryKVOffloadStore
from models.long_context_runtime import (
    LongContextRuntimeConfig,
    LongContextRuntimeState,
    LongContextEvent,
    LongContextRuntimeReport,
    create_long_context_runtime_state,
)


def _load_long_context_ops():
    """Load ops.long_context_ops directly, bypassing ops/__init__.py's eager mlx imports."""
    spec = importlib.util.spec_from_file_location(
        "ops.long_context_ops",
        _ROOT / "ops" / "long_context_ops.py",
    )
    mod = importlib.util.module_from_spec(spec)
    ops_mod = type(sys)("ops")
    ops_mod.__path__ = [str(_ROOT / "ops")]
    sys.modules.setdefault("ops", ops_mod)
    spec.loader.exec_module(mod)
    return mod


def _make_residency_map(num_layers=2, seq_len=32, block_size=8, batch_idx=0):
    map = KVResidencyMap()
    for layer_idx in range(num_layers):
        blocks = partition_sequence_into_blocks(
            layer_idx=layer_idx,
            batch_idx=batch_idx,
            seq_len=seq_len,
            block_size=block_size,
            num_kv_heads=2,
            head_dim=16,
            dtype="float16",
        )
        for meta in blocks:
            map.add_block(meta)
    return map


def _make_offload_store():
    return InMemoryKVOffloadStore()


def _make_synthetic_layer_cache(B=1, MAX_S=32, Hkv=2, D=16, dtype="float16"):
    try:
        import numpy as np
        return (np.zeros((B, MAX_S, Hkv, D), dtype=dtype), np.zeros((B, MAX_S, Hkv, D), dtype=dtype))
    except ImportError:
        pass
    try:
        import mlx.core as mx
        return (mx.zeros((B, MAX_S, Hkv, D)), mx.zeros((B, MAX_S, Hkv, D)))
    except ImportError:
        return None


class TestNeededPositions:
    def test_sliding_window(self):
        mod = _load_long_context_ops()
        pattern = {"pattern": "sliding_window", "window_size": 8, "sink_tokens": 0, "causal": True}
        positions = mod.needed_positions_for_sparse_decode(length=20, sparse_pattern=pattern)
        assert len(positions) == 8
        assert positions == list(range(12, 20))

    def test_sliding_window_sink(self):
        mod = _load_long_context_ops()
        pattern = {"pattern": "sliding_window_sink", "window_size": 8, "sink_tokens": 2, "causal": True}
        positions = mod.needed_positions_for_sparse_decode(length=20, sparse_pattern=pattern)
        assert len(positions) == 10
        assert positions == list(range(0, 2)) + list(range(12, 20))

    def test_short_sequence(self):
        mod = _load_long_context_ops()
        pattern = {"pattern": "sliding_window", "window_size": 8, "sink_tokens": 0, "causal": True}
        positions = mod.needed_positions_for_sparse_decode(length=3, sparse_pattern=pattern)
        assert sorted(positions) == [0, 1, 2]

    def test_dict_pattern(self):
        mod = _load_long_context_ops()
        positions = mod.needed_positions_for_sparse_decode(length=20, sparse_pattern={"pattern": "sliding_window", "window_size": 4, "sink_tokens": 0, "causal": True})
        assert len(positions) == 4


class TestNeededBlocks:
    def test_basic(self):
        mod = _load_long_context_ops()
        blocks = mod.needed_blocks_for_positions(
            [0, 1, 2, 3, 8, 15],
            layer_idx=0,
            batch_idx=0,
            block_size=8,
        )
        assert len(blocks) == 2
        assert blocks[0].block_idx == 0
        assert blocks[1].block_idx == 1

    def test_unique(self):
        mod = _load_long_context_ops()
        blocks = mod.needed_blocks_for_positions(
            [5, 6, 7],
            layer_idx=0,
            batch_idx=0,
            block_size=8,
        )
        assert len(blocks) == 1

    def test_negative_position_raises(self):
        mod = _load_long_context_ops()
        with pytest.raises(ValueError, match="must be >= 0"):
            mod.needed_blocks_for_positions(
                [-1],
                layer_idx=0,
                batch_idx=0,
                block_size=8,
            )


class TestEnsureBlocksReady:
    def test_all_resident(self):
        mod = _load_long_context_ops()
        residency = _make_residency_map(num_layers=1, seq_len=32, block_size=8)
        store = _make_offload_store()
        layer_cache = _make_synthetic_layer_cache(B=1, MAX_S=32)
        if layer_cache is None:
            pytest.skip("MLX or numpy not available")
        report = LongContextRuntimeReport(ok=True, events=[])
        updated = mod.ensure_blocks_ready_for_attention(
            layer_idx=0,
            batch_idx=0,
            needed_positions=[0, 1, 2],
            residency_map=residency,
            offload_store=store,
            layer_cache=layer_cache,
            block_size=8,
            report=report,
        )
        assert updated is not None
        assert report.blocks_prefetched == 0

    def test_prefetch_offloaded_block(self):
        mod = _load_long_context_ops()
        residency = _make_residency_map(num_layers=1, seq_len=32, block_size=8)
        store = _make_offload_store()
        layer_cache = _make_synthetic_layer_cache(B=1, MAX_S=32)
        if layer_cache is None:
            pytest.skip("MLX or numpy not available")
        bid = KVBlockId(layer_idx=0, batch_idx=0, block_idx=0)
        meta = residency.get(bid)
        meta.resident = False
        meta.offloaded = True
        store.put_block(bid, layer_cache[0][:, :8], layer_cache[1][:, :8])
        report = LongContextRuntimeReport(ok=True, events=[])
        updated = mod.ensure_blocks_ready_for_attention(
            layer_idx=0,
            batch_idx=0,
            needed_positions=[0, 1, 2],
            residency_map=residency,
            offload_store=store,
            layer_cache=layer_cache,
            block_size=8,
            report=report,
        )
        assert updated is not None
        assert report.blocks_prefetched == 1

    def test_missing_block_raises(self):
        mod = _load_long_context_ops()
        residency = KVResidencyMap()
        store = _make_offload_store()
        layer_cache = _make_synthetic_layer_cache(B=1, MAX_S=32)
        if layer_cache is None:
            pytest.skip("MLX or numpy not available")
        with pytest.raises(RuntimeError, match="no residency metadata"):
            mod.ensure_blocks_ready_for_attention(
                layer_idx=0,
                batch_idx=0,
                needed_positions=[0, 1],
                residency_map=residency,
                offload_store=store,
                layer_cache=layer_cache,
                block_size=8,
            )
