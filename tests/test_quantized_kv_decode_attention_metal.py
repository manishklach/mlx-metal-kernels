from __future__ import annotations

import pytest


def _mx():
    try:
        import mlx.core as _mx
        return _mx
    except ImportError:
        pytest.skip("mlx not available in this environment")


def _test_metal_vs_reference(backend, bits, Hq, Hkv, B, MAX_S, length, D, group_size, atol, rtol):
    mx = _mx()
    from ops.quantized_kv_cache_ops import (
        QuantizedKVCacheConfig,
        quantize_kv_cache,
        quantized_kv_gqa_decode_attention,
        reference_quantized_kv_gqa_decode_attention,
    )

    mx.random.seed(42)
    q = mx.random.normal((B, 1, Hq, D)).astype(mx.float16)
    K_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)
    V_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)

    cfg = QuantizedKVCacheConfig(bits=bits, group_size=group_size)
    qkv = quantize_kv_cache(K_cache, V_cache, cfg)

    try:
        ref = reference_quantized_kv_gqa_decode_attention(q, qkv, lengths=length)
        metal = quantized_kv_gqa_decode_attention(q, qkv, lengths=length, backend=backend)
        mx.eval(ref, metal)
        assert metal.shape == ref.shape
        assert mx.allclose(metal, ref, atol=atol, rtol=rtol).item(), (
            f"{backend}: max_diff={mx.max(mx.abs(metal - ref)).item():.6f}"
        )
    except Exception as e:
        if "No module named 'mlx'" in str(e) or "Metal" in str(e) or "not compiled with MLX" in str(e):
            pytest.skip(f"Metal unavailable: {e}")
        raise


class TestMetalQ8Decode:
    def test_gqa_basic(self):
        _test_metal_vs_reference("metal_q8", 8, Hq=4, Hkv=2, B=1, MAX_S=16, length=16, D=16, group_size=16, atol=5e-2, rtol=5e-2)

    def test_mqa(self):
        _test_metal_vs_reference("metal_q8", 8, Hq=4, Hkv=1, B=1, MAX_S=16, length=16, D=16, group_size=16, atol=5e-2, rtol=5e-2)

    def test_mha(self):
        _test_metal_vs_reference("metal_q8", 8, Hq=4, Hkv=4, B=1, MAX_S=16, length=16, D=16, group_size=16, atol=5e-2, rtol=5e-2)

    def test_batch2(self):
        _test_metal_vs_reference("metal_q8", 8, Hq=4, Hkv=2, B=2, MAX_S=16, length=[8, 12], D=16, group_size=16, atol=5e-2, rtol=5e-2)

    def test_d64(self):
        _test_metal_vs_reference("metal_q8", 8, Hq=4, Hkv=2, B=1, MAX_S=8, length=8, D=64, group_size=32, atol=5e-2, rtol=5e-2)

    def test_d128(self):
        _test_metal_vs_reference("metal_q8", 8, Hq=4, Hkv=2, B=1, MAX_S=8, length=8, D=128, group_size=32, atol=5e-2, rtol=5e-2)


class TestMetalQ4Decode:
    def test_gqa_basic(self):
        _test_metal_vs_reference("metal_q4", 4, Hq=4, Hkv=2, B=1, MAX_S=16, length=16, D=16, group_size=16, atol=7e-2, rtol=7e-2)

    def test_mqa(self):
        _test_metal_vs_reference("metal_q4", 4, Hq=4, Hkv=1, B=1, MAX_S=16, length=16, D=16, group_size=16, atol=7e-2, rtol=7e-2)

    def test_mha(self):
        _test_metal_vs_reference("metal_q4", 4, Hq=4, Hkv=4, B=1, MAX_S=16, length=16, D=16, group_size=16, atol=7e-2, rtol=7e-2)

    def test_batch2(self):
        _test_metal_vs_reference("metal_q4", 4, Hq=4, Hkv=2, B=2, MAX_S=16, length=[8, 12], D=16, group_size=16, atol=7e-2, rtol=7e-2)

    def test_d64(self):
        _test_metal_vs_reference("metal_q4", 4, Hq=4, Hkv=2, B=1, MAX_S=8, length=8, D=64, group_size=32, atol=7e-2, rtol=7e-2)
