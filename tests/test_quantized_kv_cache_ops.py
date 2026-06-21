from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def _load_module():
    """Load ops.quantized_kv_cache_ops directly, bypassing ops/__init__.py's eager mlx imports."""
    try:
        spec = importlib.util.spec_from_file_location(
            "ops.quantized_kv_cache_ops",
            _ROOT / "ops" / "quantized_kv_cache_ops.py",
        )
        mod = importlib.util.module_from_spec(spec)
        ops_mod = type(sys)("ops")
        ops_mod.__path__ = [str(_ROOT / "ops")]
        sys.modules.setdefault("ops", ops_mod)
        spec.loader.exec_module(mod)
        return mod
    except ImportError:
        pytest.skip("mlx not available")


class TestQuantizedKVCacheConfig:
    def test_default(self):
        mod = _load_module()
        cfg = mod.QuantizedKVCacheConfig().validate()
        assert cfg.bits == 8
        assert cfg.group_size == 32

    def test_bits_4(self):
        mod = _load_module()
        cfg = mod.QuantizedKVCacheConfig(bits=4).validate()
        assert cfg.bits == 4

    def test_bits_invalid(self):
        mod = _load_module()
        with pytest.raises(ValueError, match="bits"):
            mod.QuantizedKVCacheConfig(bits=3).validate()

    def test_group_size_zero(self):
        mod = _load_module()
        with pytest.raises(ValueError, match="group_size"):
            mod.QuantizedKVCacheConfig(group_size=0).validate()


class TestQuantizedKVCacheDataclass:
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
        qkv = mod.QuantizedKVCache(
            k_q=mx.ones((1, 8, 2, 16), dtype=mx.uint8),
            v_q=mx.ones((1, 8, 2, 16), dtype=mx.uint8),
            k_scales=mx.ones((1, 8, 2, 1), dtype=mx.float16),
            v_scales=mx.ones((1, 8, 2, 1), dtype=mx.float16),
            original_shape=(1, 8, 2, 16),
        )
        shapes = qkv.shapes()
        assert shapes["k_q"] == (1, 8, 2, 16)
        assert shapes["original"] == (1, 8, 2, 16)

    def test_validate_ok(self):
        mx = self._get_mx()
        mod = self._get_mod()
        qkv = mod.QuantizedKVCache(
            k_q=mx.ones((1, 8, 2, 16), dtype=mx.uint8),
            v_q=mx.ones((1, 8, 2, 16), dtype=mx.uint8),
            k_scales=mx.ones((1, 8, 2, 1), dtype=mx.float16),
            v_scales=mx.ones((1, 8, 2, 1), dtype=mx.float16),
            original_shape=(1, 8, 2, 16),
        )
        assert qkv.validate() is qkv

    def test_memory_bytes(self):
        mx = self._get_mx()
        mod = self._get_mod()
        qkv = mod.QuantizedKVCache(
            k_q=mx.ones((1, 4, 2, 16), dtype=mx.uint8),
            v_q=mx.ones((1, 4, 2, 16), dtype=mx.uint8),
            k_scales=mx.ones((1, 4, 2, 1), dtype=mx.float16),
            v_scales=mx.ones((1, 4, 2, 1), dtype=mx.float16),
            original_shape=(1, 4, 2, 16),
        )
        expected = 1 * 4 * 2 * 16 * 1 + 1 * 4 * 2 * 16 * 1 + 1 * 4 * 2 * 1 * 2 + 1 * 4 * 2 * 1 * 2
        assert qkv.memory_bytes() == expected

    def test_compression_ratio(self):
        mx = self._get_mx()
        mod = self._get_mod()
        qkv = mod.QuantizedKVCache(
            k_q=mx.ones((1, 4, 2, 16), dtype=mx.uint8),
            v_q=mx.ones((1, 4, 2, 16), dtype=mx.uint8),
            k_scales=mx.ones((1, 4, 2, 1), dtype=mx.float16),
            v_scales=mx.ones((1, 4, 2, 1), dtype=mx.float16),
            original_shape=(1, 4, 2, 16),
        )
        cr = qkv.compression_ratio(2)
        assert cr is not None
        assert cr > 1.0

    def test_no_original_shape(self):
        mx = self._get_mx()
        mod = self._get_mod()
        qkv = mod.QuantizedKVCache(
            k_q=mx.ones((1, 4, 2, 16), dtype=mx.uint8),
            v_q=mx.ones((1, 4, 2, 16), dtype=mx.uint8),
            k_scales=mx.ones((1, 4, 2, 1), dtype=mx.float16),
            v_scales=mx.ones((1, 4, 2, 1), dtype=mx.float16),
        )
        assert qkv.compression_ratio() is None


class TestQuantizeDequantQ8:
    def _get_mx(self):
        try:
            import mlx.core as _mx
            return _mx
        except ImportError:
            pytest.skip("mlx not available")

    def _get_mod(self):
        return _load_module()

    def test_roundtrip_shape(self):
        mx = self._get_mx()
        mod = self._get_mod()
        mx.random.seed(0)
        K = mx.random.normal((1, 8, 2, 16)).astype(mx.float16)
        V = mx.random.normal((1, 8, 2, 16)).astype(mx.float16)
        cfg = mod.QuantizedKVCacheConfig(bits=8, group_size=16)
        qkv = mod.quantize_kv_cache(K, V, cfg)
        assert qkv.k_q.shape == (1, 8, 2, 16)
        assert qkv.v_q.shape == (1, 8, 2, 16)
        assert qkv.k_scales.shape[-1] == 1

    def test_roundtrip_value(self):
        mx = self._get_mx()
        mod = self._get_mod()
        mx.random.seed(1)
        K = mx.random.normal((1, 4, 2, 8)).astype(mx.float16) * 2
        V = mx.random.normal((1, 4, 2, 8)).astype(mx.float16) * 2
        cfg = mod.QuantizedKVCacheConfig(bits=8, group_size=8)
        qkv = mod.quantize_kv_cache(K, V, cfg)
        K_deq, V_deq = mod.dequantize_kv_cache(qkv)
        assert K_deq.shape == K.shape
        err = mod.quantized_kv_error(K, V, qkv)
        assert err["k_rmse"] < 1.0
        assert err["v_rmse"] < 1.0

    def test_error_finite(self):
        mx = self._get_mx()
        mod = self._get_mod()
        mx.random.seed(2)
        K = mx.random.normal((1, 4, 2, 16)).astype(mx.float16)
        V = mx.random.normal((1, 4, 2, 16)).astype(mx.float16)
        cfg = mod.QuantizedKVCacheConfig(bits=8, group_size=16)
        qkv = mod.quantize_kv_cache(K, V, cfg)
        err = mod.quantized_kv_error(K, V, qkv)
        assert math.isfinite(err["k_max_abs_error"])
        assert math.isfinite(err["k_rmse"])
        assert err["compression_ratio"] > 0


class TestQuantizeDequantQ4:
    def _get_mx(self):
        try:
            import mlx.core as _mx
            return _mx
        except ImportError:
            pytest.skip("mlx not available")

    def _get_mod(self):
        return _load_module()

    def test_roundtrip_shape(self):
        mx = self._get_mx()
        mod = self._get_mod()
        mx.random.seed(3)
        K = mx.random.normal((1, 8, 2, 16)).astype(mx.float16)
        V = mx.random.normal((1, 8, 2, 16)).astype(mx.float16)
        cfg = mod.QuantizedKVCacheConfig(bits=4, group_size=16)
        qkv = mod.quantize_kv_cache(K, V, cfg)
        assert qkv.k_q.shape == (1, 8, 2, 8)
        assert qkv.v_q.shape == (1, 8, 2, 8)
        assert qkv.k_scales.shape[-1] == 1

    def test_odd_d(self):
        mx = self._get_mx()
        mod = self._get_mod()
        mx.random.seed(4)
        K = mx.random.normal((1, 4, 2, 17)).astype(mx.float16)
        V = mx.random.normal((1, 4, 2, 17)).astype(mx.float16)
        cfg = mod.QuantizedKVCacheConfig(bits=4, group_size=16)
        qkv = mod.quantize_kv_cache(K, V, cfg)
        assert qkv.k_q.shape == (1, 4, 2, 9)
        K_deq, V_deq = mod.dequantize_kv_cache(qkv)
        assert K_deq.shape[-1] == 17

    def test_compression_ratio(self):
        mx = self._get_mx()
        mod = self._get_mod()
        mx.random.seed(5)
        K = mx.random.normal((1, 8, 2, 32)).astype(mx.float16)
        V = mx.random.normal((1, 8, 2, 32)).astype(mx.float16)
        cfg = mod.QuantizedKVCacheConfig(bits=4, group_size=32)
        qkv = mod.quantize_kv_cache(K, V, cfg)
        cr = qkv.compression_ratio(2)
        assert cr is not None
        assert cr > 1.0


class TestQuantizeWithLengths:
    def _get_mx(self):
        try:
            import mlx.core as _mx
            return _mx
        except ImportError:
            pytest.skip("mlx not available")

    def _get_mod(self):
        return _load_module()

    def test_lengths_argument(self):
        mx = self._get_mx()
        mod = self._get_mod()
        mx.random.seed(6)
        K = mx.random.normal((2, 8, 2, 16)).astype(mx.float16)
        V = mx.random.normal((2, 8, 2, 16)).astype(mx.float16)
        cfg = mod.QuantizedKVCacheConfig(bits=8, group_size=16)
        qkv = mod.quantize_kv_cache(K, V, cfg, lengths=[4, 6])
        assert qkv.k_q.shape == (2, 8, 2, 16)
        assert qkv.v_q.shape == (2, 8, 2, 16)
