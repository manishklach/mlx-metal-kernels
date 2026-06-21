from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from models.long_context_runtime import LongContextRuntimeConfig


class TestQuantizedKVConfig:
    def test_quantized_kv_default_config(self):
        cfg = LongContextRuntimeConfig(use_quantized_kv=True, quantized_kv_config=None).validate()
        assert cfg.quantized_kv_config is not None
        assert cfg.quantized_kv_config["bits"] == 8
        assert cfg.quantized_kv_config["group_size"] == 32

    def test_quantized_kv_custom_config(self):
        try:
            from ops.quantized_kv_cache_ops import QuantizedKVCacheConfig
            qcfg = QuantizedKVCacheConfig(bits=4, group_size=16).validate()
            cfg = LongContextRuntimeConfig(use_quantized_kv=True, quantized_kv_config=qcfg).validate()
            assert cfg.quantized_kv_config.bits == 4
        except ImportError:
            pytest.skip("quantized_kv_cache_ops not available (likely MLX not installed)")

    def test_quantized_kv_disabled_by_default(self):
        cfg = LongContextRuntimeConfig().validate()
        assert cfg.use_quantized_kv is False

    def test_quantized_kv_with_sparse_supported(self):
        cfg = LongContextRuntimeConfig(
            use_quantized_kv=True,
            use_sparse_attention=True,
        ).validate()
        assert cfg.use_quantized_kv is True
        assert cfg.use_sparse_attention is True


class TestQuantizedKVReport:
    def test_report_quantized_flag(self):
        from models.long_context_runtime import LongContextRuntimeReport, LongContextEvent
        report = LongContextRuntimeReport(
            ok=True,
            events=[
                LongContextEvent(kind="quantized_kv_enabled", message="test", metadata={"bits": 8}),
            ],
            quantized_kv_enabled=True,
        )
        assert report.quantized_kv_enabled is True
        assert len(report.events) == 1
        assert report.events[0].kind == "quantized_kv_enabled"
