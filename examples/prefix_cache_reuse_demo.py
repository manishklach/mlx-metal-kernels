from __future__ import annotations

from models import (
    GenerationConfig,
    InMemoryPrefixCache,
    TinyGenerationPipeline,
    TinyGenerationPipelineConfig,
    compute_fingerprint,
    create_synthetic_stack_generation_model,
)
from models.prefix_cache import prefill_with_prefix_reuse
from ops.kv_cache_reuse_ops import clone_stack_cache, cache_prefix_equal


def demo_basic_reuse():
    print("=== Basic prefix cache reuse ===")
    model = create_synthetic_stack_generation_model(seed=42)
    gen_config = GenerationConfig(max_new_tokens=2, backend_preset="reference")
    cache = InMemoryPrefixCache(max_size=16)
    fp = compute_fingerprint(model.config, getattr(model, "tokenizer", None))
    prompt = [10, 20, 30, 40, 50]
    reused = [10, 20, 30]
    extended = [10, 20, 30, 40, 50, 60]
    # First run: full prefill, stores result
    logits, state, meta = prefill_with_prefix_reuse(prompt, model, prefix_cache=cache, generation_config=gen_config)
    print(f"  First run (full prefill): cache_hit={meta['prefix_cache_hit']}, position={state.position}")
    # Second run: exact match
    logits, state, meta = prefill_with_prefix_reuse(prompt, model, prefix_cache=cache, generation_config=gen_config)
    print(f"  Exact match: cache_hit={meta['prefix_cache_hit']}, matched_length={meta['matched_length']}, position={state.position}")
    # Partial match: reuse prefix, decode suffix
    logits, state, meta = prefill_with_prefix_reuse(extended, model, prefix_cache=cache, generation_config=gen_config)
    print(f"  Prefix match (5/6): cache_hit={meta['prefix_cache_hit']}, matched_length={meta['matched_length']}, position={state.position}")
    # No match
    logits, state, meta = prefill_with_prefix_reuse([99, 98, 97], model, prefix_cache=cache, generation_config=gen_config)
    print(f"  No match: cache_hit={meta['prefix_cache_hit']}, position={state.position}")
    print(f"  Cache size: {cache.size}")
    print(f"  Cache stats: {cache.stats()}\n")


def demo_pipeline_integration():
    print("=== Pipeline integration ===")
    config = TinyGenerationPipelineConfig(
        hidden_size=32, intermediate_size=64,
        num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, num_hidden_layers=2,
        max_position_embeddings=32, vocab_size=128,
        bits=4, backend_preset="reference",
        use_prefill=True, use_prefix_cache=True,
    ).validate()
    pipeline = TinyGenerationPipeline(config=config)
    gen_cfg = GenerationConfig(max_new_tokens=4, eos_token_id=-1, backend_preset="reference")
    print(f"  Pipeline prefix_cache created: {pipeline.prefix_cache is not None}")
    result = pipeline.generate("Hi", max_new_tokens=4, greedy=True)
    print(f"  First generation: {result.text!r}")
    print(f"  Cache size after first: {pipeline.prefix_cache.size}")
    result = pipeline.generate("Hi", max_new_tokens=4, greedy=True)
    print(f"  Second generation (same prompt): {result.text!r}")
    print(f"  Cache size after second: {pipeline.prefix_cache.size}")
    result = pipeline.generate("Hi there", max_new_tokens=4, greedy=True)
    print(f"  Third generation (extended prompt): {result.text!r}")
    print(f"  Cache size after third: {pipeline.prefix_cache.size}\n")


def demo_compare_vs_no_cache():
    print("=== Comparison: with vs without prefix cache ===")
    config_kwargs = dict(
        hidden_size=32, intermediate_size=64,
        num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, num_hidden_layers=2,
        max_position_embeddings=32, vocab_size=128,
        bits=4, backend_preset="reference",
        use_prefill=True,
    )
    pipe_no_cache = TinyGenerationPipeline(config=TinyGenerationPipelineConfig(use_prefix_cache=False, **config_kwargs))
    pipe_with_cache = TinyGenerationPipeline(config=TinyGenerationPipelineConfig(use_prefix_cache=True, **config_kwargs))
    r1 = pipe_no_cache.generate("Compare", max_new_tokens=3, greedy=True)
    r2 = pipe_with_cache.generate("Compare", max_new_tokens=3, greedy=True)
    print(f"  No cache:  {r1.text!r}")
    print(f"  With cache: {r2.text!r}")
    print(f"  Outputs match: {r1.all_ids == r2.all_ids}\n")


def main():
    demo_basic_reuse()
    demo_pipeline_integration()
    demo_compare_vs_no_cache()


if __name__ == "__main__":
    main()
