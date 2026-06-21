from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def _has_mlx():
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def _try_create_pipeline():
    from models.tiny_generation_pipeline import TinyGenerationPipeline, TinyGenerationPipelineConfig

    cfg = TinyGenerationPipelineConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=1,
        max_position_embeddings=32,
        vocab_size=64,
        bits=8,
        group_size=16,
        dtype="float16" if _has_mlx() else "float32",
        backend_preset="fused_experimental",
        cache_layout="contiguous",
        use_prefill=True,
        use_prefix_cache=False,
    )
    pipe = TinyGenerationPipeline(config=cfg)
    return pipe, cfg


@pytest.mark.slow
class TestSpeculativeVerifyPipeline:
    def test_normal_generate_unchanged(self):
        pipe, _ = _try_create_pipeline()
        result = pipe.generate("hello", max_new_tokens=2, greedy=True)
        assert len(result.generated_ids) <= 2

    def test_sequential_speculative_runs(self):
        pipe, _ = _try_create_pipeline()
        result = pipe.generate_speculative(
            "hello",
            max_new_tokens=2,
            draft_length=2,
            draft_mode="fixed",
            verifier_mode="sequential",
        )
        assert hasattr(result, "generated_ids")
        assert hasattr(result, "steps")

    def test_parallel_speculative_runs(self):
        pipe, _ = _try_create_pipeline()
        result = pipe.generate_speculative(
            "hello",
            max_new_tokens=2,
            draft_length=2,
            draft_mode="fixed",
            verifier_mode="parallel",
        )
        assert hasattr(result, "generated_ids")
        assert hasattr(result, "steps")

    def test_both_modes_return_same_type(self):
        pipe, _ = _try_create_pipeline()
        seq_result = pipe.generate_speculative(
            "test",
            max_new_tokens=2,
            draft_length=2,
            draft_mode="fixed",
            verifier_mode="sequential",
        )
        par_result = pipe.generate_speculative(
            "test",
            max_new_tokens=2,
            draft_length=2,
            draft_mode="fixed",
            verifier_mode="parallel",
        )
        assert type(seq_result) == type(par_result)

    def test_parallel_metadata_includes_verifier_mode(self):
        pipe, _ = _try_create_pipeline()
        result = pipe.generate_speculative(
            "test",
            max_new_tokens=2,
            draft_length=2,
            draft_mode="fixed",
            verifier_mode="parallel",
        )
        assert result.metadata.get("verifier_mode") == "parallel"
        assert "average_accepted" in result.metadata

    def test_generated_length_limited(self):
        pipe, _ = _try_create_pipeline()
        result = pipe.generate_speculative(
            "hi",
            max_new_tokens=4,
            draft_length=2,
            draft_mode="fixed",
            verifier_mode="parallel",
        )
        assert len(result.generated_ids) <= 4

    def test_sequential_metadata_includes_verifier_mode(self):
        pipe, _ = _try_create_pipeline()
        result = pipe.generate_speculative(
            "test",
            max_new_tokens=2,
            draft_length=2,
            draft_mode="fixed",
            verifier_mode="sequential",
        )
        assert result.metadata.get("verifier_mode") == "sequential"
