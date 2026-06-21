from __future__ import annotations

import pytest

from models.speculative_decoding import (
    FixedDraftProposer,
    RandomDraftProposer,
    SpeculativeConfig,
    SpeculativeGenerationResult,
    SpeculativeGenerator,
    SpeculativeStepResult,
    VerificationResult,
)


class _DummyPipeline:
    def __init__(self, vocab_size=64):
        self.vocab_size = vocab_size
        self.config = _DummyConfig()
        self.generation_config = _DummyGenConfig()

    def generate_ids(self, input_ids, generation_config=None, **kwargs):
        last = input_ids[-1] if input_ids else 0
        n = generation_config.max_new_tokens if generation_config else 1
        return list(input_ids) + [(last + 1 + i) % self.vocab_size for i in range(n)]

    def encode(self, prompt):
        return [10, 20, 30]

    def decode(self, token_ids):
        return " ".join(str(t) for t in token_ids)


class _DummyConfig:
    hidden_size = 32
    intermediate_size = 64
    num_attention_heads = 2
    num_key_value_heads = 1
    head_dim = 16
    num_hidden_layers = 1
    max_position_embeddings = 64
    vocab_size = 64
    bits = 4
    group_size = 32
    dtype = "float16"
    backend_preset = "reference"
    cache_layout = "contiguous"
    use_prefill = True
    use_prefix_cache = False


class _DummyGenConfig:
    max_new_tokens = 16
    temperature = 1.0
    top_k = None
    top_p = None
    eos_token_id = None
    seed = 0
    backend_preset = "reference"
    repetition_penalty = 1.0

    def validate(self):
        return self


class TestSpeculativeFullFlow:
    def test_fixed_proposer_deterministic(self):
        pipeline = _DummyPipeline()
        ids = [5, 6, 7, 8]
        proposer = FixedDraftProposer(ids)
        cfg = SpeculativeConfig(draft_length=4, max_new_tokens=8, seed=0)
        gen = SpeculativeGenerator(pipeline, draft_proposer=proposer, config=cfg)
        result_a = gen.generate_ids([0])
        result_b = gen.generate_ids([0])
        assert result_a.generated_ids == result_b.generated_ids

    def test_random_proposer_seed_control(self):
        pipeline = _DummyPipeline()
        proposer = RandomDraftProposer(64, seed=99)
        cfg = SpeculativeConfig(draft_length=3, max_new_tokens=6, seed=99)
        gen = SpeculativeGenerator(pipeline, draft_proposer=proposer, config=cfg)
        result = gen.generate_ids([1, 2])
        assert len(result.generated_ids) == 6

    def test_multiple_steps(self):
        pipeline = _DummyPipeline()
        ids = list(range(50))
        proposer = FixedDraftProposer(ids)
        cfg = SpeculativeConfig(draft_length=3, max_new_tokens=10)
        gen = SpeculativeGenerator(pipeline, draft_proposer=proposer, config=cfg)
        result = gen.generate_ids([5])
        assert result.metadata["num_steps"] >= 1
        for step in result.steps:
            assert isinstance(step, SpeculativeStepResult)

    def test_to_dict_output(self):
        pipeline = _DummyPipeline()
        proposer = FixedDraftProposer([42, 43])
        cfg = SpeculativeConfig(draft_length=2, max_new_tokens=4)
        gen = SpeculativeGenerator(pipeline, draft_proposer=proposer, config=cfg)
        result = gen.generate_ids([0])
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["num_steps"] == len(result.steps)
        assert d["acceptance_rate"] == result.acceptance_rate()
        assert d["tokens_per_step"] == result.tokens_per_step()

    def test_prompt_in_result(self):
        pipeline = _DummyPipeline()
        proposer = FixedDraftProposer([10])
        gen = SpeculativeGenerator(pipeline, draft_proposer=proposer)
        result = gen.generate_text("hello world")
        assert result.prompt == "hello world"

    def test_acceptance_bounds(self):
        pipeline = _DummyPipeline()
        for draft_mode, cls in [("fixed", FixedDraftProposer), ("random", RandomDraftProposer)]:
            if draft_mode == "fixed":
                proposer = cls([5, 6, 7])
            else:
                proposer = cls(64, seed=0)
            cfg = SpeculativeConfig(draft_length=3, max_new_tokens=5)
            gen = SpeculativeGenerator(pipeline, draft_proposer=proposer, config=cfg)
            result = gen.generate_ids([0])
            assert 0.0 <= result.acceptance_rate() <= 1.0

    def test_all_result_fields(self):
        pipeline = _DummyPipeline()
        proposer = FixedDraftProposer([7, 8])
        gen = SpeculativeGenerator(pipeline, draft_proposer=proposer)
        result = gen.generate_ids([1, 2, 3])
        assert hasattr(result, "prompt")
        assert hasattr(result, "prompt_ids")
        assert hasattr(result, "generated_ids")
        assert hasattr(result, "all_ids")
        assert hasattr(result, "text")
        assert hasattr(result, "steps")
        assert hasattr(result, "metadata")
