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


def _setup_qkv(mx, mod, B, MAX_S, Hq, Hkv, D, page_size, bits=8, group_size=32):
    mx.random.seed(42)
    K_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)
    V_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)
    q = mx.random.normal((B, 1, Hq, D)).astype(mx.float16)
    lengths = [MAX_S] * B
    K_pages, V_pages, block_table, lengths_arr = mod.contiguous_kv_to_pages(
        K_cache, V_cache, lengths, page_size=page_size,
    )
    cfg = mod.PagedQuantizedKVConfig(bits=bits, page_size=page_size, group_size=group_size)
    pqv = mod.quantize_kv_pages(K_pages, V_pages, block_table, lengths_arr, cfg)
    return q, pqv, lengths_arr


class TestMetalDecodeQ8:
    TOL_ATOL = 5e-2
    TOL_RTOL = 5e-2

    def _get_mx(self):
        try:
            import mlx.core as _mx
            return _mx
        except ImportError:
            pytest.skip("mlx not available")

    def _get_mod(self):
        return _load_module()

    def test_q8_basic(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, lengths = _setup_qkv(mx, mod, 1, 16, 4, 2, 16, 4, bits=8)
        ref = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        got = mod.paged_quantized_kv_gqa_decode_attention(q, pqv, backend="metal_q8")
        mx.eval(ref, got)
        assert mx.allclose(got, ref, atol=self.TOL_ATOL, rtol=self.TOL_RTOL).item()

    def test_mqa(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, lengths = _setup_qkv(mx, mod, 1, 16, 4, 1, 16, 4, bits=8)
        ref = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        got = mod.paged_quantized_kv_gqa_decode_attention(q, pqv, backend="metal_q8")
        mx.eval(ref, got)
        assert mx.allclose(got, ref, atol=self.TOL_ATOL, rtol=self.TOL_RTOL).item()

    def test_mha(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, lengths = _setup_qkv(mx, mod, 1, 16, 4, 4, 16, 4, bits=8)
        ref = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        got = mod.paged_quantized_kv_gqa_decode_attention(q, pqv, backend="metal_q8")
        mx.eval(ref, got)
        assert mx.allclose(got, ref, atol=self.TOL_ATOL, rtol=self.TOL_RTOL).item()

    def test_batch_2(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, lengths = _setup_qkv(mx, mod, 2, 16, 4, 2, 16, 4, bits=8)
        ref = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        got = mod.paged_quantized_kv_gqa_decode_attention(q, pqv, backend="metal_q8")
        mx.eval(ref, got)
        assert mx.allclose(got, ref, atol=self.TOL_ATOL, rtol=self.TOL_RTOL).item()

    def test_d_64(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, lengths = _setup_qkv(mx, mod, 1, 16, 4, 2, 64, 4, bits=8)
        ref = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        got = mod.paged_quantized_kv_gqa_decode_attention(q, pqv, backend="metal_q8")
        mx.eval(ref, got)
        assert mx.allclose(got, ref, atol=self.TOL_ATOL, rtol=self.TOL_RTOL).item()

    def test_d_128(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, lengths = _setup_qkv(mx, mod, 1, 16, 4, 2, 128, 4, bits=8)
        ref = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        got = mod.paged_quantized_kv_gqa_decode_attention(q, pqv, backend="metal_q8")
        mx.eval(ref, got)
        assert mx.allclose(got, ref, atol=self.TOL_ATOL, rtol=self.TOL_RTOL).item()


class TestMetalDecodeQ4:
    TOL_ATOL = 7e-2
    TOL_RTOL = 7e-2

    def _get_mx(self):
        try:
            import mlx.core as _mx
            return _mx
        except ImportError:
            pytest.skip("mlx not available")

    def _get_mod(self):
        return _load_module()

    def test_q4_basic(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, lengths = _setup_qkv(mx, mod, 1, 16, 4, 2, 16, 4, bits=4)
        ref = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        got = mod.paged_quantized_kv_gqa_decode_attention(q, pqv, backend="metal_q4")
        mx.eval(ref, got)
        assert mx.allclose(got, ref, atol=self.TOL_ATOL, rtol=self.TOL_RTOL).item()

    def test_mqa(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, lengths = _setup_qkv(mx, mod, 1, 16, 4, 1, 16, 4, bits=4)
        ref = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        got = mod.paged_quantized_kv_gqa_decode_attention(q, pqv, backend="metal_q4")
        mx.eval(ref, got)
        assert mx.allclose(got, ref, atol=self.TOL_ATOL, rtol=self.TOL_RTOL).item()

    def test_mha(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, lengths = _setup_qkv(mx, mod, 1, 16, 4, 4, 16, 4, bits=4)
        ref = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        got = mod.paged_quantized_kv_gqa_decode_attention(q, pqv, backend="metal_q4")
        mx.eval(ref, got)
        assert mx.allclose(got, ref, atol=self.TOL_ATOL, rtol=self.TOL_RTOL).item()

    def test_d_64(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, lengths = _setup_qkv(mx, mod, 1, 16, 4, 2, 64, 4, bits=4)
        ref = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        got = mod.paged_quantized_kv_gqa_decode_attention(q, pqv, backend="metal_q4")
        mx.eval(ref, got)
        assert mx.allclose(got, ref, atol=self.TOL_ATOL, rtol=self.TOL_RTOL).item()

    def test_d_128(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, lengths = _setup_qkv(mx, mod, 1, 16, 4, 2, 128, 4, bits=4)
        ref = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        got = mod.paged_quantized_kv_gqa_decode_attention(q, pqv, backend="metal_q4")
        mx.eval(ref, got)
        assert mx.allclose(got, ref, atol=self.TOL_ATOL, rtol=self.TOL_RTOL).item()
