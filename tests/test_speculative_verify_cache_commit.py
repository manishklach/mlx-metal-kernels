from __future__ import annotations

import numpy as np
import pytest


def _import_cache_ops():
    try:
        from ops.speculative_cache_ops import commit_accepted_cache
        return commit_accepted_cache
    except ImportError:
        pytest.skip("Speculative cache ops require mlx (not available in this environment)")


class TestStagedCache:
    def _make_cache(self, seq_len=8, num_layers=2, Hkv=2, D=16):
        try:
            from ops.llama_stack_ops import LlamaStackCache
        except ImportError:
            pytest.skip("llama_stack_ops require mlx")
        layer_caches = []
        for _ in range(num_layers):
            K = np.zeros((1, seq_len, Hkv, D), dtype=np.float32)
            V = np.zeros((1, seq_len, Hkv, D), dtype=np.float32)
            layer_caches.append((K, V))
        return LlamaStackCache(
            layer_caches=layer_caches,
            cache_layout="contiguous",
            max_seq_len=seq_len,
        )

    def test_staged_and_committed_differ(self):
        committed = self._make_cache(seq_len=8)
        staged = self._make_cache(seq_len=8)

        for lc in staged.layer_caches:
            K, V = lc
            K[0, 0, :, :] = 1.0
            V[0, 0, :, :] = 2.0

        assert committed.layer_caches[0][0][0, 0, 0, 0] != staged.layer_caches[0][0][0, 0, 0, 0]

    def test_accepted_zero_copies_nothing(self):
        commit_accepted_cache = _import_cache_ops()
        committed = self._make_cache(seq_len=8)
        staged = self._make_cache(seq_len=8)
        for lc in staged.layer_caches:
            lc[0][0, 0, :, :] = 42.0

        updated = commit_accepted_cache(staged, committed, accepted_count=0)
        for lc in updated.layer_caches:
            assert lc[0][0, 0, 0, 0] == 0.0

    def test_accepted_two_copies_two_positions(self):
        commit_accepted_cache = _import_cache_ops()
        committed = self._make_cache(seq_len=8)
        staged = self._make_cache(seq_len=8)
        for lc in staged.layer_caches:
            lc[0][0, :2, :, :] = 42.0

        updated = commit_accepted_cache(staged, committed, accepted_count=2)
        for lc in updated.layer_caches:
            assert lc[0][0, 0, 0, 0] == 42.0
            assert lc[0][0, 1, 0, 0] == 42.0
            assert lc[0][0, 2, 0, 0] == 0.0

    def test_commit_beyond_range_copies_nothing(self):
        commit_accepted_cache = _import_cache_ops()
        committed = self._make_cache(seq_len=8)
        staged = self._make_cache(seq_len=8)
        for lc in staged.layer_caches:
            lc[0][0, 0, :, :] = 42.0

        updated = commit_accepted_cache(staged, committed, accepted_count=0)
        for lc in updated.layer_caches:
            assert lc[0][0, 0, 0, 0] == 0.0

    def test_outside_range_unchanged(self):
        commit_accepted_cache = _import_cache_ops()
        committed = self._make_cache(seq_len=8)
        staged = self._make_cache(seq_len=8)
        for lc in staged.layer_caches:
            lc[0][0, 0, :, :] = 42.0
            lc[0][0, 7, :, :] = 99.0

        updated = commit_accepted_cache(staged, committed, accepted_count=1)
        for lc in updated.layer_caches:
            assert lc[0][0, 0, 0, 0] == 42.0
            assert lc[0][0, 7, 0, 0] == 0.0

    def test_paged_raises(self):
        commit_accepted_cache = _import_cache_ops()
        try:
            from ops.llama_stack_ops import LlamaStackCache
        except ImportError:
            pytest.skip("llama_stack_ops require mlx")

        if not hasattr(LlamaStackCache, "cache_layout"):
            pytest.skip("LlamaStackCache has no cache_layout")
        staged = self._make_cache(seq_len=8)
        staged.cache_layout = "paged"
        committed = self._make_cache(seq_len=8)
        committed.cache_layout = "paged"
        with pytest.raises((NotImplementedError, ValueError)):
            commit_accepted_cache(staged, committed, accepted_count=1)


def test_commit_parallel_verification_cache():
    try:
        from ops.speculative_verify_ops import commit_parallel_verification_cache
        from ops.llama_stack_ops import LlamaStackCache
    except ImportError:
        pytest.skip("Speculative verify ops require mlx (not available in this environment)")

    committed = LlamaStackCache(
        layer_caches=[(np.zeros((1, 8, 2, 16), dtype=np.float32), np.zeros((1, 8, 2, 16), dtype=np.float32))],
        cache_layout="contiguous",
        max_seq_len=8,
    )
    staged = LlamaStackCache(
        layer_caches=[(np.zeros((1, 8, 2, 16), dtype=np.float32), np.zeros((1, 8, 2, 16), dtype=np.float32))],
        cache_layout="contiguous",
        max_seq_len=8,
    )
    staged.layer_caches[0][0][0, 0, :, :] = 42.0
    staged.layer_caches[0][1][0, 0, :, :] = 42.0

    updated = commit_parallel_verification_cache(
        committed, staged,
        start_position=0,
        accepted_count=1,
        include_replacement=False,
    )
    assert updated.layer_caches[0][0][0, 0, 0, 0] == 42.0
