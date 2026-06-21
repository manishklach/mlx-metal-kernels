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
    return q, pqv, K_cache, V_cache, lengths_arr


class TestReferenceDecodeQ8:
    TOL = 1.5e-1

    def _get_mx(self):
        try:
            import mlx.core as _mx
            return _mx
        except ImportError:
            pytest.skip("mlx not available")

    def _get_mod(self):
        return _load_module()

    def test_q8_vs_fp16(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, K_cache, V_cache, lengths = _setup_qkv(mx, mod, 1, 16, 4, 2, 16, 4, bits=8)
        out_q = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        from ops.gqa_ops import reference_gqa_decode_attention
        ref = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=lengths)
        mx.eval(out_q, ref)
        assert mx.allclose(out_q, ref, atol=self.TOL, rtol=self.TOL).item()

    def test_gqa(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, K_cache, V_cache, lengths = _setup_qkv(mx, mod, 1, 16, 4, 2, 16, 4, bits=8)
        out_q = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        from ops.gqa_ops import reference_gqa_decode_attention
        ref = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=lengths)
        mx.eval(out_q, ref)
        assert mx.allclose(out_q, ref, atol=self.TOL, rtol=self.TOL).item()

    def test_mqa(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, K_cache, V_cache, lengths = _setup_qkv(mx, mod, 1, 16, 4, 1, 16, 4, bits=8)
        out_q = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        from ops.gqa_ops import reference_gqa_decode_attention
        ref = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=lengths)
        mx.eval(out_q, ref)
        assert mx.allclose(out_q, ref, atol=self.TOL, rtol=self.TOL).item()

    def test_mha(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, K_cache, V_cache, lengths = _setup_qkv(mx, mod, 1, 16, 4, 4, 16, 4, bits=8)
        out_q = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        from ops.gqa_ops import reference_gqa_decode_attention
        ref = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=lengths)
        mx.eval(out_q, ref)
        assert mx.allclose(out_q, ref, atol=self.TOL, rtol=self.TOL).item()

    def test_batch_2(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, K_cache, V_cache, lengths = _setup_qkv(mx, mod, 2, 16, 4, 2, 16, 4, bits=8)
        out_q = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        from ops.gqa_ops import reference_gqa_decode_attention
        ref = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=lengths)
        mx.eval(out_q, ref)
        assert mx.allclose(out_q, ref, atol=self.TOL, rtol=self.TOL).item()


class TestReferenceDecodeQ4:
    TOL = 3.0e-1

    def _get_mx(self):
        try:
            import mlx.core as _mx
            return _mx
        except ImportError:
            pytest.skip("mlx not available")

    def _get_mod(self):
        return _load_module()

    def test_q4_vs_fp16(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, K_cache, V_cache, lengths = _setup_qkv(mx, mod, 1, 16, 4, 2, 16, 4, bits=4)
        out_q = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        from ops.gqa_ops import reference_gqa_decode_attention
        ref = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=lengths)
        mx.eval(out_q, ref)
        assert mx.allclose(out_q, ref, atol=self.TOL, rtol=self.TOL).item()

    def test_gqa(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, K_cache, V_cache, lengths = _setup_qkv(mx, mod, 1, 16, 4, 2, 16, 4, bits=4)
        out_q = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        from ops.gqa_ops import reference_gqa_decode_attention
        ref = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=lengths)
        mx.eval(out_q, ref)
        assert mx.allclose(out_q, ref, atol=self.TOL, rtol=self.TOL).item()

    def test_mqa(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, K_cache, V_cache, lengths = _setup_qkv(mx, mod, 1, 16, 4, 1, 16, 4, bits=4)
        out_q = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        from ops.gqa_ops import reference_gqa_decode_attention
        ref = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=lengths)
        mx.eval(out_q, ref)
        assert mx.allclose(out_q, ref, atol=self.TOL, rtol=self.TOL).item()

    def test_mha(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, K_cache, V_cache, lengths = _setup_qkv(mx, mod, 1, 16, 4, 4, 16, 4, bits=4)
        out_q = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        from ops.gqa_ops import reference_gqa_decode_attention
        ref = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=lengths)
        mx.eval(out_q, ref)
        assert mx.allclose(out_q, ref, atol=self.TOL, rtol=self.TOL).item()

    def test_batch_2(self):
        mx = self._get_mx()
        mod = self._get_mod()
        q, pqv, K_cache, V_cache, lengths = _setup_qkv(mx, mod, 2, 16, 4, 2, 16, 4, bits=4)
        out_q = mod.reference_paged_quantized_kv_gqa_decode_attention(q, pqv)
        from ops.gqa_ops import reference_gqa_decode_attention
        ref = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=lengths)
        mx.eval(out_q, ref)
        assert mx.allclose(out_q, ref, atol=self.TOL, rtol=self.TOL).item()
