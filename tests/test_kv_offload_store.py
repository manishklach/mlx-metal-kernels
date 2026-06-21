from __future__ import annotations

import numpy as np
import pytest
import tempfile
import os

from models.kv_offload import KVBlockId
from models.kv_offload_store import InMemoryKVOffloadStore, FileKVOffloadStore


def _block_id(n: int) -> KVBlockId:
    return KVBlockId(layer_idx=0, batch_idx=0, block_idx=n)


def _k_block(block_idx: int = 0, tokens: int = 64, heads: int = 8, dim: int = 128) -> np.ndarray:
    return np.ones((1, tokens, heads, dim), dtype=np.float32) * (block_idx + 1)


def _v_block(block_idx: int = 0, tokens: int = 64, heads: int = 8, dim: int = 128) -> np.ndarray:
    return np.ones((1, tokens, heads, dim), dtype=np.float32) * (-(block_idx + 1))


# ---------------------------------------------------------------------------
# InMemoryKVOffloadStore
# ---------------------------------------------------------------------------

class TestInMemoryKVOffloadStore:
    def test_put_and_get(self):
        store = InMemoryKVOffloadStore()
        bid = _block_id(0)
        k, v = _k_block(), _v_block()
        uri = store.put_block(bid, k, v)
        assert isinstance(uri, str)
        assert uri.startswith("memory://")
        k_out, v_out = store.get_block(bid)
        np.testing.assert_array_equal(k_out, k)
        np.testing.assert_array_equal(v_out, v)

    def test_get_missing_raises(self):
        store = InMemoryKVOffloadStore()
        with pytest.raises(KeyError, match="not found"):
            store.get_block(_block_id(99))

    def test_has_block(self):
        store = InMemoryKVOffloadStore()
        assert not store.has_block(_block_id(0))
        store.put_block(_block_id(0), _k_block(), _v_block())
        assert store.has_block(_block_id(0))

    def test_delete_block(self):
        store = InMemoryKVOffloadStore()
        store.put_block(_block_id(0), _k_block(), _v_block())
        store.delete_block(_block_id(0))
        assert not store.has_block(_block_id(0))

    def test_delete_missing_raises(self):
        store = InMemoryKVOffloadStore()
        with pytest.raises(KeyError, match="not found"):
            store.delete_block(_block_id(99))

    def test_stats(self):
        store = InMemoryKVOffloadStore()
        assert store.stats()["blocks"] == 0
        store.put_block(_block_id(0), _k_block(0, 64, 2, 16), _v_block(0, 64, 2, 16))
        store.put_block(_block_id(1), _k_block(1, 64, 2, 16), _v_block(1, 64, 2, 16))
        stats = store.stats()
        assert stats["blocks"] == 2
        assert stats["puts"] == 2
        store.get_block(_block_id(0))
        assert store.stats()["gets"] == 1
        store.delete_block(_block_id(0))
        assert store.stats()["deletes"] == 1

    def test_clear(self):
        store = InMemoryKVOffloadStore()
        store.put_block(_block_id(0), _k_block(), _v_block())
        store.clear()
        assert store.stats()["blocks"] == 0
        assert store.stats()["puts"] == 0


# ---------------------------------------------------------------------------
# FileKVOffloadStore
# ---------------------------------------------------------------------------

class TestFileKVOffloadStore:
    def test_put_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileKVOffloadStore(tmpdir)
            bid = _block_id(0)
            k, v = _k_block(0, 16, 2, 8), _v_block(0, 16, 2, 8)
            store.put_block(bid, k, v)
            assert store.has_block(bid)
            k_out, v_out = store.get_block(bid)
            np.testing.assert_array_equal(k_out, k)
            np.testing.assert_array_equal(v_out, v)

    def test_has_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileKVOffloadStore(tmpdir)
            assert not store.has_block(_block_id(0))
            store.put_block(_block_id(0), _k_block(0, 16, 2, 8), _v_block(0, 16, 2, 8))
            assert store.has_block(_block_id(0))

    def test_get_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileKVOffloadStore(tmpdir)
            with pytest.raises((FileNotFoundError, KeyError)):
                store.get_block(_block_id(99))

    def test_delete_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileKVOffloadStore(tmpdir)
            store.put_block(_block_id(0), _k_block(0, 16, 2, 8), _v_block(0, 16, 2, 8))
            store.delete_block(_block_id(0))
            assert not store.has_block(_block_id(0))

    def test_delete_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileKVOffloadStore(tmpdir)
            with pytest.raises((KeyError, FileNotFoundError)):
                store.delete_block(_block_id(99))

    def test_put_creates_meta_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileKVOffloadStore(tmpdir)
            bid = _block_id(0)
            store.put_block(bid, _k_block(0, 16, 2, 8), _v_block(0, 16, 2, 8))
            block_dir = store._block_dir(bid)
            assert (block_dir / "meta.json").exists()

    def test_overwrite_default_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileKVOffloadStore(tmpdir, overwrite=False)
            bid = _block_id(0)
            store.put_block(bid, _k_block(0, 16, 2, 8), _v_block(0, 16, 2, 8))
            with pytest.raises(FileExistsError):
                store.put_block(bid, _k_block(1, 16, 2, 8), _v_block(1, 16, 2, 8))

    def test_overwrite_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileKVOffloadStore(tmpdir, overwrite=True)
            bid = _block_id(0)
            store.put_block(bid, _k_block(0, 16, 2, 8), _v_block(0, 16, 2, 8))
            store.put_block(bid, _k_block(1, 16, 2, 8), _v_block(1, 16, 2, 8))
            k_out, _ = store.get_block(bid)
            assert np.allclose(k_out, _k_block(1, 16, 2, 8))

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileKVOffloadStore(tmpdir)
            bid = _block_id(0)
            store.put_block(bid, _k_block(0, 16, 2, 8), _v_block(0, 16, 2, 8))
            stats = store.stats()
            assert stats["blocks"] >= 1
            assert stats["puts"] == 1
            assert stats["gets"] == 0
            assert stats["root_dir"] == os.path.abspath(tmpdir)
