from __future__ import annotations

import pytest

from models import (
    CharTokenizer,
    GenerationConfig,
    QuantizedCheckpointPackage,
    TinyGenerationPipeline,
    TinyGenerationPipelineConfig,
    create_pipeline_from_quantized_package,
)


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


def test_tiny_generation_pipeline_config_validates():
    config = _tiny_pipeline_config()
    llama_config = config.to_llama_config()
    assert llama_config.hidden_size == 32
    assert config.to_dict()["bits"] == 4


def test_invalid_tiny_generation_pipeline_config_raises():
    with pytest.raises(ValueError, match="hidden_size must equal"):
        TinyGenerationPipelineConfig(hidden_size=30, num_attention_heads=2, head_dim=16).validate()


def test_pipeline_describe_returns_expected_fields():
    pipeline = TinyGenerationPipeline(config=_tiny_pipeline_config())
    description = pipeline.describe()
    assert description["pipeline"] == "TinyGenerationPipeline"
    assert description["synthetic_weights"] is True
    assert description["num_hidden_layers"] == 1


def test_encode_decode_works_with_char_tokenizer():
    tokenizer = CharTokenizer()
    pipeline = TinyGenerationPipeline(config=_tiny_pipeline_config(), tokenizer=tokenizer)
    token_ids = pipeline.encode("Hi")
    text = pipeline.decode(token_ids)
    assert token_ids
    assert isinstance(text, str)


def test_generate_ids_returns_expected_length_with_greedy():
    pipeline = TinyGenerationPipeline(config=_tiny_pipeline_config(), generation_config=GenerationConfig(max_new_tokens=4, eos_token_id=-1, backend_preset="reference"))
    prompt_ids = pipeline.encode("Hi")
    output_ids = pipeline.generate_ids(prompt_ids, GenerationConfig(max_new_tokens=4, eos_token_id=-1, backend_preset="reference"))
    assert len(output_ids) == len(prompt_ids) + 4


def test_generate_returns_generation_result():
    pipeline = TinyGenerationPipeline(config=_tiny_pipeline_config(), generation_config=GenerationConfig(max_new_tokens=4, eos_token_id=-1, backend_preset="reference"))
    result = pipeline.generate("Hi", max_new_tokens=4, greedy=True)
    assert result.prompt == "Hi"
    assert len(result.generated_ids) == 4
    assert result.all_ids[: len(result.prompt_ids)] == result.prompt_ids


def test_generation_result_to_dict_roundtrip_shape():
    pipeline = TinyGenerationPipeline(config=_tiny_pipeline_config(), generation_config=GenerationConfig(max_new_tokens=2, eos_token_id=-1, backend_preset="reference"))
    result = pipeline.generate("Hi", max_new_tokens=2, greedy=True)
    payload = result.to_dict()
    assert payload["prompt"] == "Hi"
    assert payload["backend_preset"] == "reference"
    assert isinstance(payload["metadata"], dict)


def test_metadata_marks_synthetic_weights():
    pipeline = TinyGenerationPipeline(config=_tiny_pipeline_config(), generation_config=GenerationConfig(max_new_tokens=2, eos_token_id=-1, backend_preset="reference"))
    result = pipeline.generate("Hello", max_new_tokens=2, greedy=True)
    assert result.metadata["synthetic_weights"] is True


def test_package_metadata_only_path_raises_not_implemented():
    package = QuantizedCheckpointPackage(
        config=_tiny_pipeline_config().to_llama_config().to_dict(),
        quantization={"bits": 4, "group_size": 32},
        layers=[],
        metadata={"note": "metadata only"},
    )
    with pytest.raises(NotImplementedError, match="metadata does not contain tensor data"):
        create_pipeline_from_quantized_package(package)


def test_bits_8_pipeline_config_works():
    pipeline = TinyGenerationPipeline(config=_tiny_pipeline_config(bits=8), generation_config=GenerationConfig(max_new_tokens=2, eos_token_id=-1, backend_preset="reference"))
    result = pipeline.generate("Q8", max_new_tokens=2, greedy=True)
    assert len(result.generated_ids) == 2
