from __future__ import annotations

import pytest

from models.tiny_generation_pipeline import TinyGenerationPipeline, TinyGenerationPipelineConfig


def _make_pipeline(config=None):
    if config is None:
        config = TinyGenerationPipelineConfig(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            num_hidden_layers=1,
            max_position_embeddings=64,
            vocab_size=64,
            bits=4,
            group_size=32,
            backend_preset="reference",
            cache_layout="contiguous",
            use_prefill=False,
        ).validate()
    try:
        pipe = TinyGenerationPipeline(config=config)
    except RuntimeError as e:
        if "not available" in str(e).lower() or "mlx" in str(e).lower():
            pytest.skip(f"Skipping: {e}")
        raise
    return pipe


def _get_spec_config(draft_length=2, max_new_tokens=8, **kw):
    from models.speculative_decoding import SpeculativeConfig
    return SpeculativeConfig(
        draft_length=draft_length,
        max_new_tokens=max_new_tokens,
        backend_preset="reference",
        seed=0,
        **kw,
    ).validate()


class TestPipelineSpeculativeMode:
    def test_generate_speculative_fixed(self):
        pipe = _make_pipeline()
        cfg = _get_spec_config(draft_length=2, max_new_tokens=4)
        from models.speculative_decoding import FixedDraftProposer, SpeculativeGenerator
        proposer = FixedDraftProposer([10, 20, 30, 40, 50])
        gen = SpeculativeGenerator(pipe, draft_proposer=proposer, config=cfg)
        result = gen.generate_text("hello")
        assert len(result.generated_ids) > 0
        assert isinstance(result.text, str)
        assert result.metadata["num_steps"] >= 1

    def test_generate_speculative_random(self):
        pipe = _make_pipeline()
        cfg = _get_spec_config(draft_length=2, max_new_tokens=4)
        from models.speculative_decoding import RandomDraftProposer, SpeculativeGenerator
        proposer = RandomDraftProposer(pipe.vocab_size, seed=0)
        gen = SpeculativeGenerator(pipe, draft_proposer=proposer, config=cfg)
        result = gen.generate_text("test")
        assert len(result.generated_ids) > 0

    def test_pipeline_generate_speculative_method(self):
        pipe = _make_pipeline()
        result = pipe.generate_speculative(
            "hello",
            max_new_tokens=4,
            draft_length=2,
            draft_mode="fixed",
        )
        assert len(result.generated_ids) > 0

    def test_pipeline_generate_speculative_random_mode(self):
        pipe = _make_pipeline()
        result = pipe.generate_speculative(
            "test",
            max_new_tokens=4,
            draft_length=3,
            draft_mode="random",
        )
        assert len(result.generated_ids) > 0

    def test_metadata_in_result(self):
        pipe = _make_pipeline()
        result = pipe.generate_speculative(
            "hello",
            max_new_tokens=4,
            draft_length=2,
            draft_mode="fixed",
        )
        assert "num_steps" in result.metadata
        assert "total_proposed" in result.metadata
        assert "total_accepted" in result.metadata
        assert "acceptance_rate" in result.metadata
        assert result.to_dict()["num_steps"] == result.metadata["num_steps"]

    def test_generate_speculative_no_eos(self):
        pipe = _make_pipeline()
        result = pipe.generate_speculative(
            "hello",
            max_new_tokens=8,
            draft_length=4,
            draft_mode="fixed",
        )
        assert len(result.generated_ids) == 8

    def test_generate_speculative_invalid_draft_mode(self):
        pipe = _make_pipeline()
        with pytest.raises(ValueError, match="draft_mode"):
            pipe.generate_speculative("hello", max_new_tokens=2, draft_mode="unknown")

    def test_generate_speculative_acceptance_rate(self):
        pipe = _make_pipeline()
        for draft_mode in ("fixed", "random"):
            result = pipe.generate_speculative(
                "test",
                max_new_tokens=6,
                draft_length=2,
                draft_mode=draft_mode,
            )
            rate = result.acceptance_rate()
            assert 0.0 <= rate <= 1.0

    def test_generate_speculative_tokens_per_step(self):
        pipe = _make_pipeline()
        result = pipe.generate_speculative(
            "test", max_new_tokens=4, draft_length=2, draft_mode="fixed",
        )
        tps = result.tokens_per_step()
        assert tps > 0.0

    def test_generate_speculative_default_not_enabled(self):
        pipe = _make_pipeline()
        result = pipe.generate("hello", max_new_tokens=4)
        assert len(result.generated_ids) == 4
        assert result.text is not None
