from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from models.long_context_runtime import (
    LongContextRuntimeConfig,
    LongContextEvent,
    LongContextRuntimeReport,
    LongContextRuntimeState,
    create_long_context_runtime_state,
)


class TestLongContextRuntimeConfig:
    def test_defaults(self):
        cfg = LongContextRuntimeConfig().validate()
        assert cfg.use_prefix_cache is True
        assert cfg.use_sparse_attention is True
        assert cfg.use_kv_offload is True
        assert cfg.use_quantized_kv is False
        assert cfg.cache_layout == "contiguous"
        assert cfg.sparse_pattern is not None
        assert cfg.offload_policy is not None

    def test_invalid_cache_layout(self):
        with pytest.raises(NotImplementedError, match="cache_layout"):
            LongContextRuntimeConfig(cache_layout="paged").validate()

    def test_sparse_default_pattern_created(self):
        cfg = LongContextRuntimeConfig(use_sparse_attention=True, sparse_pattern=None).validate()
        assert cfg.sparse_pattern is not None
        assert cfg.sparse_pattern["pattern"] == "sliding_window"

    def test_offload_default_policy_created(self):
        cfg = LongContextRuntimeConfig(use_kv_offload=True, offload_policy=None).validate()
        assert cfg.offload_policy is not None
        assert cfg.offload_policy.block_size == 128

    def test_quantized_kv_default_config_created(self):
        cfg = LongContextRuntimeConfig(use_quantized_kv=True, quantized_kv_config=None).validate()
        assert cfg.quantized_kv_config is not None
        assert cfg.quantized_kv_config["bits"] == 8

    def test_paged_offload_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not implemented"):
            LongContextRuntimeConfig(cache_layout="paged", use_kv_offload=True, use_prefix_cache=False, use_sparse_attention=False).validate()

    def test_paged_quantized_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not implemented"):
            LongContextRuntimeConfig(cache_layout="paged", use_quantized_kv=True, use_prefix_cache=False, use_sparse_attention=False, use_kv_offload=False).validate()

    def test_to_dict(self):
        cfg = LongContextRuntimeConfig(model_id="test-model", tokenizer_id="test-tok").validate()
        d = cfg.to_dict()
        assert d["model_id"] == "test-model"
        assert d["tokenizer_id"] == "test-tok"
        assert "sparse_pattern" in d
        assert "offload_policy" in d


class TestLongContextEvent:
    def test_create(self):
        ev = LongContextEvent(kind="prefix_cache_hit", message="matched 10 tokens", metadata={"length": 10})
        assert ev.kind == "prefix_cache_hit"
        assert ev.metadata["length"] == 10

    def test_default_metadata(self):
        ev = LongContextEvent(kind="warning", message="test")
        assert ev.metadata == {}


class TestLongContextRuntimeReport:
    def test_create(self):
        report = LongContextRuntimeReport(ok=True, events=[])
        assert report.ok is True
        assert report.errors() == []
        assert report.warnings() == []

    def test_errors(self):
        report = LongContextRuntimeReport(
            ok=False,
            events=[
                LongContextEvent(kind="error", message="err1"),
                LongContextEvent(kind="warning", message="warn"),
                LongContextEvent(kind="error", message="err2"),
            ],
        )
        assert len(report.errors()) == 2
        assert len(report.warnings()) == 1

    def test_summary(self):
        report = LongContextRuntimeReport(
            ok=True,
            events=[LongContextEvent(kind="prefix_cache_hit", message="hit")],
            prefix_cache_hit=True,
            matched_prefix_length=10,
        )
        s = report.summary()
        assert s["ok"] is True
        assert s["prefix_cache_hit"] is True
        assert s["matched_prefix_length"] == 10
        assert len(s["events"]) == 1

    def test_to_dict(self):
        report = LongContextRuntimeReport(ok=True, events=[], metadata={"key": "val"})
        d = report.to_dict()
        assert d["metadata"]["key"] == "val"

    def test_pretty_print(self):
        report = LongContextRuntimeReport(
            ok=True,
            events=[LongContextEvent(kind="info", message="test event")],
        )
        text = report.pretty_print()
        assert "LongContextRuntimeReport" in text
        assert "test event" in text


class TestLongContextRuntimeState:
    def test_create(self):
        state = LongContextRuntimeState(stack_cache=None)
        assert state.position == 0
        assert state.prefix_cache is None

    def test_describe_empty(self):
        state = LongContextRuntimeState(stack_cache=None)
        desc = state.describe()
        assert desc["position"] == 0
        assert desc["has_prefix_cache"] is False


class TestCreateState:
    def test_errored_without_mlx(self):
        _ = self
        try:
            from models.generation import ToyGenerationState
            _ = ToyGenerationState
        except ImportError:
            with pytest.raises(RuntimeError, match="MLX"):
                create_long_context_runtime_state(
                    config=None,
                    stack_weights=None,
                    runtime_config=LongContextRuntimeConfig(use_prefix_cache=False, use_sparse_attention=False, use_kv_offload=False),
                    B=1,
                    max_seq_len=64,
                )
