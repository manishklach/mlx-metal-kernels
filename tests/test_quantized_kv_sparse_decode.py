from __future__ import annotations

import pytest


def _mx():
    try:
        import mlx.core as _mx
        return _mx
    except ImportError:
        pytest.skip("mlx not available in this environment")


class TestSparseQuantizedDecodeReference:
    def _run(self, bits, group_size, atol, rtol):
        mx = _mx()
        from ops.gqa_ops import reference_gqa_decode_attention
        from ops.quantized_kv_cache_ops import (
            QuantizedKVCacheConfig,
            quantize_kv_cache,
            reference_quantized_kv_sparse_gqa_decode_attention,
        )
        from ops.sparse_attention_ops import SparseAttentionPattern

        mx.random.seed(42)
        B, MAX_S, Hq, Hkv, D = 1, 32, 4, 2, 16
        length = 32
        q = mx.random.normal((B, 1, Hq, D)).astype(mx.float16)
        K_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)
        V_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)

        cfg = QuantizedKVCacheConfig(bits=bits, group_size=group_size)
        qkv = quantize_kv_cache(K_cache, V_cache, cfg)

        pattern = SparseAttentionPattern(
            pattern="sliding_window_sink",
            window_size=8,
            sink_tokens=2,
        )

        fp16_out = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=length)
        quant_out = reference_quantized_kv_sparse_gqa_decode_attention(q, qkv, length, pattern)
        mx.eval(fp16_out, quant_out)
        assert quant_out.shape == fp16_out.shape
        assert mx.allclose(quant_out, fp16_out, atol=atol, rtol=rtol).item(), (
            f"bits={bits}: max_diff={mx.max(mx.abs(quant_out - fp16_out)).item():.6f}"
        )

    def test_q8_sliding_window_sink(self):
        self._run(bits=8, group_size=16, atol=1.5e-1, rtol=1.5e-1)

    def test_q4_sliding_window_sink(self):
        self._run(bits=4, group_size=16, atol=3e-1, rtol=3e-1)

    def test_sliding_window_no_sink(self):
        mx = _mx()
        from ops.quantized_kv_cache_ops import (
            QuantizedKVCacheConfig,
            quantize_kv_cache,
            reference_quantized_kv_sparse_gqa_decode_attention,
        )
        from ops.sparse_attention_ops import SparseAttentionPattern

        mx.random.seed(44)
        B, MAX_S, Hq, Hkv, D = 1, 16, 4, 2, 16
        length = 16
        q = mx.random.normal((B, 1, Hq, D)).astype(mx.float16)
        K_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)
        V_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)

        cfg = QuantizedKVCacheConfig(bits=8, group_size=16)
        qkv = quantize_kv_cache(K_cache, V_cache, cfg)

        pattern = SparseAttentionPattern(pattern="sliding_window", window_size=6)
        quant_out = reference_quantized_kv_sparse_gqa_decode_attention(q, qkv, length, pattern)
        assert quant_out.shape == (B, 1, Hq, D)
