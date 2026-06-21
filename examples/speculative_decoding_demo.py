"""Demonstrate speculative decoding with Fixed and Random draft proposers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import (
    FixedDraftProposer,
    RandomDraftProposer,
    SpeculativeConfig,
    SpeculativeGenerator,
)
from models.tiny_generation_pipeline import TinyGenerationPipeline, TinyGenerationPipelineConfig


def _make_pipeline():
    config = TinyGenerationPipelineConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=1,
        max_position_embeddings=128,
        vocab_size=64,
        bits=4,
        group_size=32,
        backend_preset="reference",
        cache_layout="contiguous",
        use_prefill=False,
    ).validate()
    return TinyGenerationPipeline(config=config)


def demo_fixed_proposer(pipeline):
    print("=== Fixed Draft Proposer Demo ===")
    proposer = FixedDraftProposer([10, 20, 30, 40])
    cfg = SpeculativeConfig(
        draft_length=4,
        max_new_tokens=8,
        temperature=1.0,
        greedy_verify=True,
        seed=0,
        backend_preset="reference",
    ).validate()
    gen = SpeculativeGenerator(pipeline, draft_proposer=proposer, config=cfg)
    result = gen.generate_text("hello")
    print(f"  Prompt: 'hello'")
    print(f"  Generated tokens: {result.generated_ids}")
    print(f"  Text: {result.text!r}")
    print(f"  Acceptance rate: {result.acceptance_rate():.2f}")
    print(f"  Tokens per step: {result.tokens_per_step():.2f}")
    print(f"  Steps: {len(result.steps)}")
    print()


def demo_random_proposer(pipeline):
    print("=== Random Draft Proposer Demo ===")
    proposer = RandomDraftProposer(pipeline.vocab_size, seed=42)
    cfg = SpeculativeConfig(
        draft_length=4,
        max_new_tokens=8,
        temperature=1.0,
        greedy_verify=True,
        seed=42,
        backend_preset="reference",
    ).validate()
    gen = SpeculativeGenerator(pipeline, draft_proposer=proposer, config=cfg)
    result = gen.generate_text("world")
    print(f"  Prompt: 'world'")
    print(f"  Generated tokens: {result.generated_ids}")
    print(f"  Text: {result.text!r}")
    print(f"  Acceptance rate: {result.acceptance_rate():.2f}")
    print(f"  Tokens per step: {result.tokens_per_step():.2f}")
    print(f"  Steps: {len(result.steps)}")
    print()


def demo_pipeline_method(pipeline):
    print("=== Pipeline.generate_speculative() Demo ===")
    result = pipeline.generate_speculative(
        "demo",
        max_new_tokens=6,
        draft_length=3,
        draft_mode="random",
        seed=0,
    )
    print(f"  Prompt: 'demo'")
    print(f"  Generated tokens: {result.generated_ids}")
    print(f"  Text: {result.text!r}")
    print(f"  Acceptance rate: {result.acceptance_rate():.2f}")
    print(f"  Tokens per step: {result.tokens_per_step():.2f}")
    print(f"  Metadata: {result.metadata}")
    print()


def main():
    try:
        pipeline = _make_pipeline()
    except RuntimeError as e:
        print(f"Skipping demo: {e}")
        return
    demo_fixed_proposer(pipeline)
    demo_random_proposer(pipeline)
    demo_pipeline_method(pipeline)


if __name__ == "__main__":
    main()
