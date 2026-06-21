from __future__ import annotations

import pytest


def _mx():
    try:
        import mlx.core as _mx
        return _mx
    except ImportError:
        pytest.skip("mlx not available in this environment")


def _test_quantized_decode_matches_fp16(Hq, Hkv, B, MAX_S, length, D, bits, group_size, atol, rtol):
    mx = _mx()
    from ops.gqa_ops import reference_gqa_decode_attention
    from ops.quantized_kv_cache_ops import (
        QuantizedKVCacheConfig,
        quantize_kv_cache,
        reference_quantized_kv_gqa_decode_attention,
    )

    mx.random.seed(42)
    q = mx.random.normal((B, 1, Hq, D)).astype(mx.float16)
    K_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)
    V_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)

    cfg = QuantizedKVCacheConfig(bits=bits, group_size=group_size)
    qkv = quantize_kv_cache(K_cache, V_cache, cfg)

    ref_out = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=length)
    quant_out = reference_quantized_kv_gqa_decode_attention(q, qkv, lengths=length)

    mx.eval(ref_out, quant_out)
    assert quant_out.shape == ref_out.shape
    assert mx.allclose(quant_out, ref_out, atol=atol, rtol=rtol).item(), (
        f"max_diff={mx.max(mx.abs(quant_out - ref_out)).item():.6f}"
    )


class TestQ8QuantizedDecode:
    def test_gqa(self):
        _test_quantized_decode_matches_fp16(Hq=4, Hkv=2, B=1, MAX_S=16, length=16, D=16, bits=8, group_size=16, atol=1.5e-1, rtol=1.5e-1)

    def test_mqa(self):
        _test_quantized_decode_matches_fp16(Hq=4, Hkv=1, B=1, MAX_S=16, length=16, D=16, bits=8, group_size=16, atol=1.5e-1, rtol=1.5e-1)

    def test_mha(self):
        _test_quantized_decode_matches_fp16(Hq=4, Hkv=4, B=1, MAX_S=16, length=16, D=16, bits=8, group_size=16, atol=1.5e-1, rtol=1.5e-1)

    def test_batch2(self):
        _test_quantized_decode_matches_fp16(Hq=4, Hkv=2, B=2, MAX_S=16, length=[8, 12], D=16, bits=8, group_size=16, atol=1.5e-1, rtol=1.5e-1)

    def test_larger_d(self):
        _test_quantized_decode_matches_fp16(Hq=4, Hkv=2, B=1, MAX_S=8, length=8, D=64, bits=8, group_size=32, atol=2e-1, rtol=2e-1)


class TestQ4QuantizedDecode:
    def test_gqa(self):
        _test_quantized_decode_matches_fp16(Hq=4, Hkv=2, B=1, MAX_S=16, length=16, D=16, bits=4, group_size=16, atol=3e-1, rtol=3e-1)

    def test_mqa(self):
        _test_quantized_decode_matches_fp16(Hq=4, Hkv=1, B=1, MAX_S=16, length=16, D=16, bits=4, group_size=16, atol=3e-1, rtol=3e-1)

    def test_mha(self):
        _test_quantized_decode_matches_fp16(Hq=4, Hkv=4, B=1, MAX_S=16, length=16, D=16, bits=4, group_size=16, atol=3e-1, rtol=3e-1)

    def test_batch2(self):
        _test_quantized_decode_matches_fp16(Hq=4, Hkv=2, B=2, MAX_S=16, length=[8, 12], D=16, bits=4, group_size=16, atol=3e-1, rtol=3e-1)
