from __future__ import annotations

from models import (
    GenerationConfig,
    InMemoryPrefixCache,
)
from models.prefix_cache import prefill_with_prefix_reuse

import pytest


def _mlx_available():
    try:
        import mlx.core as mx  # noqa: F401
        return True
    except ImportError:
        return False


def _create_model(seed=42):
    try:
        from models import create_synthetic_stack_generation_model
    except ImportError:
        pytest.skip("Model creation requires mlx (not available in this environment)")
    return create_synthetic_stack_generation_model(seed=seed)


def _pipeline(**overrides):
    try:
        from models import TinyGenerationPipeline, TinyGenerationPipelineConfig
    except ImportError:
        pytest.skip("Pipeline requires mlx (not available in this environment)")
    kwargs = dict(
        hidden_size=32, intermediate_size=64,
        num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, num_hidden_layers=2,
        max_position_embeddings=32, vocab_size=128,
        bits=4, backend_preset="reference",
        use_prefill=True,
    )
    kwargs.update(overrides)
    return TinyGenerationPipeline(config=TinyGenerationPipelineConfig(**kwargs).validate())


@pytest.mark.skipif(not _mlx_available(), reason="mlx is not available in this environment")
def test_pipeline_with_prefix_cache_creates_cache():
    pipeline = _pipeline(use_prefix_cache=True)
    assert pipeline.prefix_cache is not None
    assert pipeline.prefix_cache.size == 0


def test_pipeline_without_prefix_cache():
    pipeline = _pipeline(use_prefix_cache=False)
    assert pipeline.prefix_cache is None


@pytest.mark.skipif(not _mlx_available(), reason="mlx is not available in this environment")
def test_prefix_cache_accumulates_entries():
    pipeline = _pipeline(use_prefix_cache=True)
    pipeline.generate("Hello", max_new_tokens=2, greedy=True)
    assert pipeline.prefix_cache.size >= 1


@pytest.mark.skipif(not _mlx_available(), reason="mlx is not available in this environment")
def test_prefix_cache_hit_produces_same_result():
    pipeline = _pipeline(use_prefix_cache=True)
    result_first = pipeline.generate("Test", max_new_tokens=2, greedy=True)
    result_second = pipeline.generate("Test", max_new_tokens=2, greedy=True)
    assert result_first.all_ids == result_second.all_ids


@pytest.mark.skipif(not _mlx_available(), reason="mlx is not available in this environment")
def test_prefix_cache_longer_prefix_reused():
    pipeline = _pipeline(use_prefix_cache=True)
    pipeline.generate("Hello World", max_new_tokens=2, greedy=True)
    result = pipeline.generate("Hello World Again", max_new_tokens=2, greedy=True)
    assert len(result.all_ids) > len("Hello World")


@pytest.mark.skipif(not _mlx_available(), reason="mlx is not available in this environment")
def test_prefix_cache_different_prompt_no_reuse():
    pipeline = _pipeline(use_prefix_cache=True)
    pipeline.generate("Hello", max_new_tokens=2, greedy=True)
    result = pipeline.generate("World", max_new_tokens=2, greedy=True)
    assert len(result.generated_ids) == 2


@pytest.mark.skipif(not _mlx_available(), reason="mlx is not available in this environment")
def test_prefix_cache_vs_no_cache_same_output():
    pipeline_a = _pipeline(use_prefix_cache=False)
    pipeline_b = _pipeline(use_prefix_cache=True)
    result_a = pipeline_a.generate("Compare", max_new_tokens=2, greedy=True)
    result_b = pipeline_b.generate("Compare", max_new_tokens=2, greedy=True)
    assert result_a.all_ids == result_b.all_ids


@pytest.mark.skipif(not _mlx_available(), reason="mlx is not available in this environment")
def test_prefix_cache_exact_match():
    model = _create_model()
    cache = InMemoryPrefixCache(max_size=16)
    gcfg = GenerationConfig(max_new_tokens=2, backend_preset="reference")
    prefill_with_prefix_reuse([10, 20, 30], model, prefix_cache=cache, generation_config=gcfg)
    _, state, meta = prefill_with_prefix_reuse([10, 20, 30], model, prefix_cache=cache, generation_config=gcfg)
    assert meta["prefix_cache_hit"]
    assert meta["suffix_mode"] == "replay_last_token"
    assert state.generated_ids == [10, 20, 30]


@pytest.mark.skipif(not _mlx_available(), reason="mlx is not available in this environment")
def test_prefix_cache_partial_match():
    model = _create_model()
    cache = InMemoryPrefixCache(max_size=16)
    gcfg = GenerationConfig(max_new_tokens=2, backend_preset="reference")
    prefill_with_prefix_reuse([10, 20, 30], model, prefix_cache=cache, generation_config=gcfg)
    _, state, meta = prefill_with_prefix_reuse([10, 20, 30, 40], model, prefix_cache=cache, generation_config=gcfg)
    assert meta["prefix_cache_hit"]
    assert meta["matched_length"] == 3
    assert meta["suffix_mode"] == "decode_suffix"
    assert state.position == 4


@pytest.mark.skipif(not _mlx_available(), reason="mlx is not available in this environment")
def test_prefix_cache_no_match_on_different_config():
    cache = InMemoryPrefixCache(max_size=16)
    model_a = _create_model(seed=42)
    model_b = _create_model(seed=99)
    gcfg = GenerationConfig(max_new_tokens=2, backend_preset="reference")
    prefill_with_prefix_reuse([10, 20, 30], model_a, prefix_cache=cache, generation_config=gcfg)
    _, _, meta = prefill_with_prefix_reuse([10, 20, 30], model_b, prefix_cache=cache, generation_config=gcfg)
    assert not meta["prefix_cache_hit"]


@pytest.mark.skipif(not _mlx_available(), reason="mlx is not available in this environment")
def test_ignore_prefix_cache():
    pipeline = _pipeline(use_prefix_cache=True)
    pipeline.generate("Hello", max_new_tokens=2, greedy=True)
    result = pipeline.generate_ids(
        pipeline.encode("Hello"),
        generation_config=GenerationConfig(max_new_tokens=2, backend_preset="reference"),
        ignore_prefix_cache=True,
    )
    assert len(result) > 0
