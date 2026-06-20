from __future__ import annotations

from models import GenerationConfig, TinyGenerationPipeline, TinyGenerationPipelineConfig


def _pipeline_config(*, use_prefill: bool) -> TinyGenerationPipelineConfig:
    return TinyGenerationPipelineConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=32,
        vocab_size=128,
        bits=4,
        backend_preset="reference",
        use_prefill=use_prefill,
    ).validate()


def test_prefill_then_decode_pipeline_smoke():
    generation_config = GenerationConfig(max_new_tokens=2, eos_token_id=-1, backend_preset="reference")
    pipeline = TinyGenerationPipeline(config=_pipeline_config(use_prefill=True), generation_config=generation_config)
    prompt_ids = pipeline.encode("Test")
    prefill = pipeline.prefill_prompt(prompt_ids, generation_config=generation_config)
    assert prefill.next_position == len(prompt_ids)
    assert prefill.prompt_length == len(prompt_ids)
    result = pipeline.generate("Test", max_new_tokens=2, greedy=True)
    assert len(result.generated_ids) == 2
    assert result.all_ids[: len(prompt_ids)] == prompt_ids


def test_prefill_and_decode_ingest_both_work():
    generation_config = GenerationConfig(max_new_tokens=2, eos_token_id=-1, backend_preset="reference")
    prefill_pipeline = TinyGenerationPipeline(config=_pipeline_config(use_prefill=True), generation_config=generation_config)
    decode_pipeline = TinyGenerationPipeline(config=_pipeline_config(use_prefill=False), generation_config=generation_config)
    prefill_result = prefill_pipeline.generate("Tiny", max_new_tokens=2, greedy=True)
    decode_result = decode_pipeline.generate("Tiny", max_new_tokens=2, greedy=True)
    assert len(prefill_result.generated_ids) == 2
    assert len(decode_result.generated_ids) == 2
    assert len(prefill_result.all_ids) == len(decode_result.all_ids)
