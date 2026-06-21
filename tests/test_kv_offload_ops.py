from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from models.kv_offload import KVBlockId, KVBlockMetadata, KVResidencyMap


def _load_offload_ops():
    """Load ops.kv_offload_ops directly, bypassing ops/__init__.py's eager mlx imports."""
    spec = importlib.util.spec_from_file_location(
        "ops.kv_offload_ops",
        _ROOT / "ops" / "kv_offload_ops.py",
    )
    mod = importlib.util.module_from_spec(spec)
    ops_mod = type(sys)("ops")
    ops_mod.__path__ = [str(_ROOT / "ops")]
    sys.modules.setdefault("ops", ops_mod)
    spec.loader.exec_module(mod)
    return mod


def _make_layer_cache(tokens=64, heads=4, dim=16):
    K = np.random.default_rng(0).normal(0, 1, (1, tokens, heads, dim)).astype(np.float32)
    V = np.random.default_rng(1).normal(0, 1, (1, tokens, heads, dim)).astype(np.float32)
    return (K, V)


def _block_meta(block_idx=0, start=0, end=64, tokens=64):
    return KVBlockMetadata(
        block_id=KVBlockId(layer_idx=0, batch_idx=0, block_idx=block_idx),
        start_token=start, end_token=end, num_tokens=tokens,
        num_kv_heads=4, head_dim=16, dtype="float32",
    )


def _make_store():
    from models.kv_offload_store import InMemoryKVOffloadStore
    return InMemoryKVOffloadStore()


class TestExtractKVBlock:
    def _get_ops(self):
        return _load_offload_ops()

    def test_extract_full_block(self):
        mod = self._get_ops()
        cache = _make_layer_cache(tokens=128)
        K_block, V_block = mod.extract_kv_block(cache, 0, 64)
        assert K_block.shape == (1, 64, 4, 16)
        np.testing.assert_array_equal(K_block, cache[0][:, 0:64])

    def test_extract_partial(self):
        mod = self._get_ops()
        cache = _make_layer_cache(tokens=128)
        K_block, V_block = mod.extract_kv_block(cache, 32, 96)
        assert K_block.shape == (1, 64, 4, 16)

    def test_invalid_range_raises(self):
        mod = self._get_ops()
        cache = _make_layer_cache(tokens=64)
        with pytest.raises(ValueError, match="token range"):
            mod.extract_kv_block(cache, -1, 32)
        with pytest.raises(ValueError, match="token range"):
            mod.extract_kv_block(cache, 32, 32)
        with pytest.raises(ValueError, match="token range"):
            mod.extract_kv_block(cache, 60, 128)


class TestInsertKVBlock:
    def _get_ops(self):
        return _load_offload_ops()

    def test_insert_into_cache(self):
        mod = self._get_ops()
        cache = _make_layer_cache(tokens=128)
        new_K = np.ones((1, 32, 4, 16), dtype=np.float32) * 99
        new_V = np.ones((1, 32, 4, 16), dtype=np.float32) * 99
        updated = mod.insert_kv_block(cache, 64, new_K, new_V)
        np.testing.assert_array_equal(updated[0][:, 64:96], new_K)
        np.testing.assert_array_equal(updated[1][:, 64:96], new_V)
        assert updated[0].shape == cache[0].shape

    def test_insert_does_not_mutate_original(self):
        mod = self._get_ops()
        cache = _make_layer_cache(tokens=64)
        cache_copy = (cache[0].copy(), cache[1].copy())
        new_K = np.ones((1, 16, 4, 16), dtype=np.float32) * 99
        new_V = np.ones((1, 16, 4, 16), dtype=np.float32) * 99
        mod.insert_kv_block(cache, 0, new_K, new_V)
        np.testing.assert_array_equal(cache[0], cache_copy[0])
        np.testing.assert_array_equal(cache[1], cache_copy[1])

    def test_invalid_insert_raises(self):
        mod = self._get_ops()
        cache = _make_layer_cache(tokens=64)
        new_K = np.ones((1, 32, 4, 16), dtype=np.float32)
        new_V = np.ones((1, 32, 4, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="insert"):
            mod.insert_kv_block(cache, 50, new_K, new_V)


class TestOffloadKVBlock:
    def _get_ops(self):
        return _load_offload_ops()

    def test_offload_writes_to_store(self):
        mod = self._get_ops()
        cache = _make_layer_cache(tokens=128)
        meta = _block_meta(block_idx=0, start=0, end=64)
        store = _make_store()
        updated_cache, result = mod.offload_kv_block(cache, meta, store)
        assert meta.offloaded is True
        assert meta.resident is False
        assert meta.store_uri is not None
        assert meta.checksum is not None
        assert store.has_block(meta.block_id)

    def test_offload_zero_hot(self):
        mod = self._get_ops()
        cache = _make_layer_cache(tokens=128)
        meta = _block_meta(block_idx=0, start=0, end=64)
        store = _make_store()
        updated_cache, _ = mod.offload_kv_block(cache, meta, store, zero_hot=True)
        assert np.allclose(updated_cache[0][:, 0:64], 0)

    def test_offload_default_no_zero(self):
        mod = self._get_ops()
        cache = _make_layer_cache(tokens=128)
        meta = _block_meta(block_idx=0, start=0, end=64)
        store = _make_store()
        updated_cache, _ = mod.offload_kv_block(cache, meta, store, zero_hot=False)
        np.testing.assert_array_equal(updated_cache[0], cache[0])


class TestPrefetchKVBlock:
    def _get_ops(self):
        return _load_offload_ops()

    def test_prefetch_restores_block(self):
        mod = self._get_ops()
        cache = _make_layer_cache(tokens=128)
        meta = _block_meta(block_idx=0, start=0, end=64)
        store = _make_store()
        original_K = cache[0].copy()
        mod.offload_kv_block(cache, meta, store, zero_hot=False)
        updated, result = mod.prefetch_kv_block(cache, meta, store)
        assert meta.resident is True
        assert meta.offloaded is False
        np.testing.assert_array_equal(updated[0][:, 0:64], original_K[:, 0:64])

    def test_prefetch_resident_does_nothing(self):
        mod = self._get_ops()
        cache = _make_layer_cache(tokens=128)
        meta = _block_meta(block_idx=0, start=0, end=64)
        store = _make_store()
        updated, result = mod.prefetch_kv_block(cache, meta, store)
        assert result["already_resident"] is True


class TestApplyOffloadPlan:
    def _get_ops(self):
        return _load_offload_ops()

    def test_offloads_and_prefetches(self):
        mod = self._get_ops()
        from models.kv_offload_policy import KVOffloadPlan

        caches = [_make_layer_cache(tokens=128)]
        rmap = KVResidencyMap()
        for i in range(4):
            meta = KVBlockMetadata(
                block_id=KVBlockId(0, 0, i),
                start_token=i * 32, end_token=(i + 1) * 32,
                num_tokens=32, num_kv_heads=4, head_dim=16,
                dtype="float32", resident=True,
            )
            rmap.add_block(meta)

        store = _make_store()

        plan = KVOffloadPlan(
            offload=[KVBlockId(0, 0, 0), KVBlockId(0, 0, 1)],
            prefetch=[KVBlockId(0, 0, 0)],
            reason="test",
        )
        updated, _, summary = mod.apply_offload_plan(caches, rmap, store, plan)
        assert summary["num_offloaded"] == 2
        assert summary["num_prefetched"] == 1
