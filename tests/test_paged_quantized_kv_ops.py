from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def _load_module():
    try:
        spec = importlib.util.spec_from_file_location(
            "ops.paged_quantized_kv_ops",
            _ROOT / "ops" / "paged_quantized_kv_ops.py",
        )
        mod = importlib.util.module_from_spec(spec)
        ops_mod = type(sys)("ops")
        ops_mod.__path__ = [str(_ROOT / "ops")]
        sys.modules.setdefault("ops", ops_mod)
        spec.loader.exec_module(mod)
        return mod
    except ImportError:
        pytest.skip("mlx not available")


class TestPagedQuantizedKVConfig:
    def test_default(self):
        mod = _load_module()
        cfg = mod.PagedQuantizedKVConfig().validate()
        assert cfg.bits == 8
        assert cfg.page_size == 16
        assert cfg.group_size == 32

    def test_bits_4(self):
        mod = _load_module()
        cfg = mod.PagedQuantizedKVConfig(bits=4).validate()
        assert cfg.bits == 4

    def test_bits_invalid(self):
        mod = _load_module()
        with pytest.raises(ValueError, match="bits"):
            mod.PagedQuantizedKVConfig(bits=3).validate()

    def test_page_size_zero(self):
        mod = _load_module()
        with pytest.raises(ValueError, match="page_size"):
            mod.PagedQuantizedKVConfig(page_size=0).validate()

    def test_invalid_layout(self):
        mod = _load_module()
        with pytest.raises((ValueError, NotImplementedError)):
            mod.PagedQuantizedKVConfig(layout="contiguous").validate()


class TestPagedQuantizedKVCache:
    def _get_mx(self):
        try:
            import mlx.core as _mx
            return _mx
        except ImportError:
            pytest.skip("mlx not available")

    def _get_mod(self):
        return _load_module()

    def test_shapes(self):
        mx = self._get_mx()
        mod = self._get_mod()
        pqv = mod.PagedQuantizedKVCache(
            k_pages_q=mx.ones((4, 8, 2, 16), dtype=mx.uint8),
            v_pages_q=mx.ones((4, 8, 2, 16), dtype=mx.uint8),
            k_scales=mx.ones((4, 8, 2, 1), dtype=mx.float16),
            v_scales=mx.ones((4, 8, 2, 1), dtype=mx.float16),
            block_table=mx.ones((1, 4), dtype=mx.int32),
            lengths=mx.array([8], dtype=mx.int32),
            original_page_shape=(4, 8, 2, 16),
        )
        s = pqv.shapes()
        assert s["k_pages_q"] == (4, 8, 2, 16)
        assert s["original_page_shape"] == (4, 8, 2, 16)

    def test_validate_ok(self):
        mx = self._get_mx()
        mod = self._get_mod()
        pqv = mod.PagedQuantizedKVCache(
            k_pages_q=mx.ones((4, 8, 2, 16), dtype=mx.uint8),
            v_pages_q=mx.ones((4, 8, 2, 16), dtype=mx.uint8),
            k_scales=mx.ones((4, 8, 2, 1), dtype=mx.float16),
            v_scales=mx.ones((4, 8, 2, 1), dtype=mx.float16),
            block_table=mx.ones((1, 4), dtype=mx.int32),
            lengths=mx.array([8], dtype=mx.int32),
            original_page_shape=(4, 8, 2, 16),
        )
        assert pqv.validate() is pqv

    def test_num_pages(self):
        mx = self._get_mx()
        mod = self._get_mod()
        pqv = mod.PagedQuantizedKVCache(
            k_pages_q=mx.ones((4, 8, 2, 16), dtype=mx.uint8),
            v_pages_q=mx.ones((4, 8, 2, 16), dtype=mx.uint8),
            k_scales=mx.ones((4, 8, 2, 1), dtype=mx.float16),
            v_scales=mx.ones((4, 8, 2, 1), dtype=mx.float16),
            block_table=mx.ones((1, 4), dtype=mx.int32),
            lengths=mx.array([8], dtype=mx.int32),
        )
        assert pqv.num_pages() == 4

    def test_memory_bytes(self):
        mx = self._get_mx()
        mod = self._get_mod()
        pqv = mod.PagedQuantizedKVCache(
            k_pages_q=mx.ones((2, 4, 2, 16), dtype=mx.uint8),
            v_pages_q=mx.ones((2, 4, 2, 16), dtype=mx.uint8),
            k_scales=mx.ones((2, 4, 2, 1), dtype=mx.float16),
            v_scales=mx.ones((2, 4, 2, 1), dtype=mx.float16),
            block_table=mx.ones((1, 4), dtype=mx.int32),
            lengths=mx.array([4], dtype=mx.int32),
            original_page_shape=(2, 4, 2, 16),
        )
        mb = pqv.memory_bytes()
        assert mb > 0

    def test_compression_ratio(self):
        mx = self._get_mx()
        mod = self._get_mod()
        pqv = mod.PagedQuantizedKVCache(
            k_pages_q=mx.ones((2, 4, 2, 16), dtype=mx.uint8),
            v_pages_q=mx.ones((2, 4, 2, 16), dtype=mx.uint8),
            k_scales=mx.ones((2, 4, 2, 1), dtype=mx.float16),
            v_scales=mx.ones((2, 4, 2, 1), dtype=mx.float16),
            block_table=mx.ones((1, 4), dtype=mx.int32),
            lengths=mx.array([4], dtype=mx.int32),
            original_page_shape=(2, 4, 2, 16),
        )
        cr = pqv.compression_ratio(2)
        assert cr is not None
        assert cr > 0.0

    def test_no_original_shape(self):
        mx = self._get_mx()
        mod = self._get_mod()
        pqv = mod.PagedQuantizedKVCache(
            k_pages_q=mx.ones((2, 4, 2, 16), dtype=mx.uint8),
            v_pages_q=mx.ones((2, 4, 2, 16), dtype=mx.uint8),
            k_scales=mx.ones((2, 4, 2, 1), dtype=mx.float16),
            v_scales=mx.ones((2, 4, 2, 1), dtype=mx.float16),
            block_table=mx.ones((1, 4), dtype=mx.int32),
            lengths=mx.array([4], dtype=mx.int32),
        )
        assert pqv.compression_ratio() is None


class TestQuantizeDequantPagesQ8:
    def _get_mx(self):
        try:
            import mlx.core as _mx
            return _mx
        except ImportError:
            pytest.skip("mlx not available")

    def _get_mod(self):
        return _load_module()

    def test_quantize_dequant_shape(self):
        mx = self._get_mx()
        mod = self._get_mod()
        mx.random.seed(10)
        NUM_PAGES, PAGE_SIZE, Hkv, D = 4, 8, 2, 16
        K_pages = mx.random.normal((NUM_PAGES, PAGE_SIZE, Hkv, D)).astype(mx.float16)
        V_pages = mx.random.normal((NUM_PAGES, PAGE_SIZE, Hkv, D)).astype(mx.float16)
        block_table = mx.arange(NUM_PAGES, dtype=mx.int32).reshape(1, NUM_PAGES)
        lengths = mx.array([32], dtype=mx.int32)
        cfg = mod.PagedQuantizedKVConfig(bits=8, page_size=PAGE_SIZE, group_size=16)
        pqv = mod.quantize_kv_pages(K_pages, V_pages, block_table, lengths, cfg)
        assert pqv.k_pages_q.shape == (NUM_PAGES, PAGE_SIZE, Hkv, D)
        assert pqv.v_pages_q.shape == (NUM_PAGES, PAGE_SIZE, Hkv, D)
        assert pqv.k_scales.shape[-1] == 1
        K_deq, V_deq = mod.dequantize_kv_pages(pqv)
        assert K_deq.shape == K_pages.shape
        err = mod.paged_quantized_kv_error(K_pages, V_pages, pqv)
        assert err["k_rmse"] < 1.0
        assert err["v_rmse"] < 1.0

    def test_scales_shape(self):
        mx = self._get_mx()
        mod = self._get_mod()
        mx.random.seed(11)
        NUM_PAGES, PAGE_SIZE, Hkv, D = 2, 4, 2, 32
        K_pages = mx.random.normal((NUM_PAGES, PAGE_SIZE, Hkv, D)).astype(mx.float16)
        V_pages = mx.random.normal((NUM_PAGES, PAGE_SIZE, Hkv, D)).astype(mx.float16)
        block_table = mx.arange(NUM_PAGES, dtype=mx.int32).reshape(1, NUM_PAGES)
        lengths = mx.array([8], dtype=mx.int32)
        cfg = mod.PagedQuantizedKVConfig(bits=8, page_size=PAGE_SIZE, group_size=16)
        pqv = mod.quantize_kv_pages(K_pages, V_pages, block_table, lengths, cfg)
        num_group = (D + 15) // 16
        assert pqv.k_scales.shape == (NUM_PAGES, PAGE_SIZE, Hkv, num_group)
        assert pqv.v_scales.shape == (NUM_PAGES, PAGE_SIZE, Hkv, num_group)


class TestQuantizeDequantPagesQ4:
    def _get_mx(self):
        try:
            import mlx.core as _mx
            return _mx
        except ImportError:
            pytest.skip("mlx not available")

    def _get_mod(self):
        return _load_module()

    def test_quantize_dequant_shape(self):
        mx = self._get_mx()
        mod = self._get_mod()
        mx.random.seed(12)
        NUM_PAGES, PAGE_SIZE, Hkv, D = 4, 8, 2, 16
        K_pages = mx.random.normal((NUM_PAGES, PAGE_SIZE, Hkv, D)).astype(mx.float16)
        V_pages = mx.random.normal((NUM_PAGES, PAGE_SIZE, Hkv, D)).astype(mx.float16)
        block_table = mx.arange(NUM_PAGES, dtype=mx.int32).reshape(1, NUM_PAGES)
        lengths = mx.array([32], dtype=mx.int32)
        cfg = mod.PagedQuantizedKVConfig(bits=4, page_size=PAGE_SIZE, group_size=16)
        pqv = mod.quantize_kv_pages(K_pages, V_pages, block_table, lengths, cfg)
        D_packed = (16 + 1) // 2
        assert pqv.k_pages_q.shape == (NUM_PAGES, PAGE_SIZE, Hkv, D_packed)
        K_deq, V_deq = mod.dequantize_kv_pages(pqv)
        assert K_deq.shape == K_pages.shape
        err = mod.paged_quantized_kv_error(K_pages, V_pages, pqv)
        assert err["k_rmse"] < 1.0
        assert err["v_rmse"] < 1.0

    def test_odd_d_q4(self):
        mx = self._get_mx()
        mod = self._get_mod()
        mx.random.seed(13)
        NUM_PAGES, PAGE_SIZE, Hkv, D = 2, 4, 2, 17
        K_pages = mx.random.normal((NUM_PAGES, PAGE_SIZE, Hkv, D)).astype(mx.float16)
        V_pages = mx.random.normal((NUM_PAGES, PAGE_SIZE, Hkv, D)).astype(mx.float16)
        block_table = mx.arange(NUM_PAGES, dtype=mx.int32).reshape(1, NUM_PAGES)
        lengths = mx.array([8], dtype=mx.int32)
        cfg = mod.PagedQuantizedKVConfig(bits=4, page_size=PAGE_SIZE, group_size=16)
        pqv = mod.quantize_kv_pages(K_pages, V_pages, block_table, lengths, cfg)
        D_packed = (17 + 1) // 2
        assert pqv.k_pages_q.shape[-1] == D_packed
        K_deq, V_deq = mod.dequantize_kv_pages(pqv)
        assert K_deq.shape[-1] == 17

    def test_error_finite(self):
        mx = self._get_mx()
        mod = self._get_mod()
        mx.random.seed(14)
        K_pages = mx.random.normal((2, 4, 2, 16)).astype(mx.float16)
        V_pages = mx.random.normal((2, 4, 2, 16)).astype(mx.float16)
        block_table = mx.arange(2, dtype=mx.int32).reshape(1, 2)
        lengths = mx.array([8], dtype=mx.int32)
        cfg = mod.PagedQuantizedKVConfig(bits=4, page_size=4, group_size=16)
        pqv = mod.quantize_kv_pages(K_pages, V_pages, block_table, lengths, cfg)
        err = mod.paged_quantized_kv_error(K_pages, V_pages, pqv)
        import math
        assert math.isfinite(err["k_max_abs_error"])
        assert err["compression_ratio"] > 0


class TestContiguousToPages:
    def _get_mx(self):
        try:
            import mlx.core as _mx
            return _mx
        except ImportError:
            pytest.skip("mlx not available")

    def _get_mod(self):
        return _load_module()

    def test_roundtrip(self):
        mx = self._get_mx()
        mod = self._get_mod()
        mx.random.seed(15)
        B, MAX_S, Hkv, D = 2, 8, 2, 16
        K_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)
        V_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)
        lengths = [6, 8]
        page_size = 4
        K_pages, V_pages, block_table, lengths_arr = mod.contiguous_kv_to_pages(
            K_cache, V_cache, lengths, page_size=page_size,
        )
        K_restored, V_restored = mod.pages_to_contiguous_kv(
            K_pages, V_pages, block_table, lengths_arr, max_seq_len=MAX_S,
        )
        for b in range(B):
            valid = lengths[b]
            assert mx.allclose(K_restored[b, :valid], K_cache[b, :valid], atol=1e-6, rtol=1e-6).item()
            assert mx.allclose(V_restored[b, :valid], V_cache[b, :valid], atol=1e-6, rtol=1e-6).item()

    def test_block_table_ids(self):
        mx = self._get_mx()
        mod = self._get_mod()
        K_cache = mx.zeros((1, 8, 2, 16), dtype=mx.float16)
        V_cache = mx.zeros((1, 8, 2, 16), dtype=mx.float16)
        _, _, block_table, _ = mod.contiguous_kv_to_pages(K_cache, V_cache, [8], page_size=4)
        assert block_table.shape == (1, 2)
        assert block_table[0, 0].item() == 0
        assert block_table[0, 1].item() == 1

    def test_lengths_preserved(self):
        mx = self._get_mx()
        mod = self._get_mod()
        K_cache = mx.zeros((2, 16, 2, 16), dtype=mx.float16)
        V_cache = mx.zeros((2, 16, 2, 16), dtype=mx.float16)
        _, _, _, lengths_arr = mod.contiguous_kv_to_pages(K_cache, V_cache, [4, 12], page_size=8)
        assert lengths_arr.tolist() == [4, 12]


class TestInvalidBits:
    def _get_mx(self):
        try:
            import mlx.core as _mx
            return _mx
        except ImportError:
            pytest.skip("mlx not available")

    def _get_mod(self):
        return _load_module()

    def test_invalid_bits_raises(self):
        mx = self._get_mx()
        mod = self._get_mod()
        K_pages = mx.zeros((2, 4, 2, 16), dtype=mx.float16)
        V_pages = mx.zeros((2, 4, 2, 16), dtype=mx.float16)
        block_table = mx.ones((1, 2), dtype=mx.int32)
        lengths = mx.array([4], dtype=mx.int32)
        with pytest.raises((ValueError, NotImplementedError)):
            mod.quantize_kv_pages(K_pages, V_pages, block_table, lengths,
                                  mod.PagedQuantizedKVConfig(bits=3))
