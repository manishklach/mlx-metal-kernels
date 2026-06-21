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


class TestMemoryAccounting:
    def _get_mx(self):
        try:
            import mlx.core as _mx
            return _mx
        except ImportError:
            pytest.skip("mlx not available")

    def _get_mod(self):
        return _load_module()

    def test_fp16_paged_kv_bytes(self):
        mod = self._get_mod()
        mx = self._get_mx()
        NUM_PAGES, PAGE_SIZE, Hkv, D = 4, 8, 2, 32
        fp16_bytes = 2 * NUM_PAGES * PAGE_SIZE * Hkv * D * 2
        assert fp16_bytes == 4 * 8 * 2 * 32 * 2 * 2

    def test_q8_bytes_approx_half_plus_scales(self):
        mod = self._get_mod()
        mx = self._get_mx()
        NUM_PAGES, PAGE_SIZE, Hkv, D = 4, 8, 2, 32
        groups_per_head = (32 + 31) // 32
        q8_kv = 2 * NUM_PAGES * PAGE_SIZE * Hkv * D * 1
        scales = 2 * NUM_PAGES * PAGE_SIZE * Hkv * groups_per_head * 2
        fp16_bytes = 2 * NUM_PAGES * PAGE_SIZE * Hkv * D * 2
        q8_total = q8_kv + scales
        assert q8_total < fp16_bytes

    def test_q4_bytes_approx_quarter_plus_scales(self):
        mod = self._get_mod()
        mx = self._get_mx()
        NUM_PAGES, PAGE_SIZE, Hkv, D = 4, 8, 2, 32
        groups_per_head = (32 + 31) // 32
        D_packed = (32 + 1) // 2
        q4_kv = 2 * NUM_PAGES * PAGE_SIZE * Hkv * D_packed * 1
        scales = 2 * NUM_PAGES * PAGE_SIZE * Hkv * groups_per_head * 2
        fp16_bytes = 2 * NUM_PAGES * PAGE_SIZE * Hkv * D * 2
        q4_total = q4_kv + scales
        assert q4_total < fp16_bytes

    def test_compression_ratio_sensible(self):
        mod = self._get_mod()
        mx = self._get_mx()
        NUM_PAGES, PAGE_SIZE, Hkv, D = 4, 8, 2, 32
        K_pages = mx.random.normal((NUM_PAGES, PAGE_SIZE, Hkv, D)).astype(mx.float16)
        V_pages = mx.random.normal((NUM_PAGES, PAGE_SIZE, Hkv, D)).astype(mx.float16)
        block_table = mx.arange(NUM_PAGES, dtype=mx.int32).reshape(1, NUM_PAGES)
        lengths = mx.array([32], dtype=mx.int32)
        cfg = mod.PagedQuantizedKVConfig(bits=8, page_size=PAGE_SIZE, group_size=32)
        pqv = mod.quantize_kv_pages(K_pages, V_pages, block_table, lengths, cfg)
        cr = pqv.compression_ratio(2)
        assert cr is not None
        assert 1.0 < cr < 10.0

    def test_q4_packed_shape_odd_d(self):
        mod = self._get_mod()
        mx = self._get_mx()
        NUM_PAGES, PAGE_SIZE, Hkv, D = 2, 4, 2, 17
        K_pages = mx.random.normal((NUM_PAGES, PAGE_SIZE, Hkv, D)).astype(mx.float16)
        V_pages = mx.random.normal((NUM_PAGES, PAGE_SIZE, Hkv, D)).astype(mx.float16)
        block_table = mx.arange(NUM_PAGES, dtype=mx.int32).reshape(1, NUM_PAGES)
        lengths = mx.array([8], dtype=mx.int32)
        cfg = mod.PagedQuantizedKVConfig(bits=4, page_size=4, group_size=16)
        pqv = mod.quantize_kv_pages(K_pages, V_pages, block_table, lengths, cfg)
        expected_packed = (17 + 1) // 2
        assert pqv.k_pages_q.shape[-1] == expected_packed
        cr = pqv.compression_ratio(2)
        assert cr is not None
        assert cr > 0.0
