from __future__ import annotations

import pytest

from models import GenerationConfig, TinyGenerationPipeline, TinyGenerationPipelineConfig

np = pytest.importorskip("numpy")


def _tiny_pipeline_config(**overrides) -> TinyGenerationPipelineConfig:
    kwargs = dict(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=1,
        max_position_embeddings=32,
        vocab_size=128,
        bits=4,
        group_size=32,
        backend_preset="reference",
    )
    kwargs.update(overrides)
    return TinyGenerationPipelineConfig(**kwargs).validate()


def test_pipeline_validate_alignment_returns_ok():
    pipeline = TinyGenerationPipeline(config=_tiny_pipeline_config())
    report = pipeline.validate_alignment()
    assert report.ok
    assert report.summary["has_stack_weights"] is True


def test_generate_validate_alignment_true_works_for_valid_pipeline():
    pipeline = TinyGenerationPipeline(
        config=_tiny_pipeline_config(),
        generation_config=GenerationConfig(max_new_tokens=3, eos_token_id=-1, backend_preset="reference"),
    )
    result = pipeline.generate("Hi", max_new_tokens=3, greedy=True, validate_alignment=True)
    assert len(result.generated_ids) == 3


def test_pipeline_validate_alignment_returns_error_for_bad_lm_head_shape():
    pipeline = TinyGenerationPipeline(config=_tiny_pipeline_config())
    bad_lm_head = np.zeros((pipeline.vocab_size + 1, pipeline.llama_config.hidden_size), dtype=np.float32)
    pipeline.stack_weights.lm_head = bad_lm_head
    pipeline.model.lm_head = bad_lm_head
    report = pipeline.validate_alignment()
    assert not report.ok
    assert any(issue.code == "VOCAB_SIZE_MISMATCH" for issue in report.errors())


def test_generate_validate_alignment_true_raises_on_bad_alignment():
    pipeline = TinyGenerationPipeline(
        config=_tiny_pipeline_config(),
        generation_config=GenerationConfig(max_new_tokens=2, eos_token_id=-1, backend_preset="reference"),
    )
    bad_lm_head = np.zeros((pipeline.vocab_size + 1, pipeline.llama_config.hidden_size), dtype=np.float32)
    pipeline.stack_weights.lm_head = bad_lm_head
    pipeline.model.lm_head = bad_lm_head
    with pytest.raises(ValueError, match="Alignment status: error"):
        pipeline.generate("Hi", max_new_tokens=2, greedy=True, validate_alignment=True)


def test_generate_validate_alignment_false_preserves_old_behavior():
    pipeline = TinyGenerationPipeline(
        config=_tiny_pipeline_config(),
        generation_config=GenerationConfig(max_new_tokens=2, eos_token_id=-1, backend_preset="reference"),
    )
    bad_lm_head = np.zeros((pipeline.vocab_size + 1, pipeline.llama_config.hidden_size), dtype=np.float32)
    pipeline.stack_weights.lm_head = bad_lm_head
    pipeline.model.lm_head = bad_lm_head
    result = pipeline.generate("Hi", max_new_tokens=2, greedy=True, validate_alignment=False)
    assert len(result.generated_ids) == 2
