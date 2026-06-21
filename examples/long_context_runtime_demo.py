from __future__ import annotations

from models.llama_config import LlamaLikeConfig
from models.long_context_runtime import (
    LongContextRuntime,
    LongContextRuntimeConfig,
)


def main():
    print("=" * 60)
    print("Long-Context Runtime Integration Demo")
    print("=" * 60)
    print()
    print("WARNING: This demo uses synthetic random weights.")
    print("It demonstrates runtime plumbing, not meaningful generation")
    print("or production flash streaming.")
    print()

    try:
        import mlx.core as mx
    except ImportError:
        print("ERROR: mlx is required for this demo. Install with: pip install mlx")
        return

    model_config = LlamaLikeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=256,
        vocab_size=128,
        model_type="demo_long_context",
    ).validate()

    from ops.llama_stack_ops import create_random_quantized_llama_stack_weights
    weights = create_random_quantized_llama_stack_weights(
        model_config,
        vocab_size=128,
        bits=4,
        group_size=32,
        dtype=mx.float16,
        seed=0,
        include_embedding=True,
        include_lm_head=True,
    )

    from ops.sparse_attention_ops import SparseAttentionPattern
    sparse_pattern = SparseAttentionPattern(
        pattern="sliding_window_sink",
        window_size=32,
        sink_tokens=4,
        causal=True,
    )

    from models.kv_offload_policy import KVOffloadPolicyConfig
    offload_policy = KVOffloadPolicyConfig(
        block_size=16,
        keep_recent_blocks=2,
        keep_sink_blocks=1,
    ).validate()

    runtime_config = LongContextRuntimeConfig(
        use_prefix_cache=True,
        use_sparse_attention=True,
        use_kv_offload=True,
        use_quantized_kv=False,
        sparse_pattern=sparse_pattern,
        offload_policy=offload_policy,
        backend_preset="fused_experimental",
        seed=0,
    ).validate()

    runtime = LongContextRuntime(
        model_config=model_config,
        stack_weights=weights,
        embedding=weights.embedding,
        lm_head=weights.lm_head,
        runtime_config=runtime_config,
    )

    prompt_a_tokens = list(range(64))
    prompt_b_tokens = list(range(48)) + list(range(48, 64))

    print("--- Prompt A (first sequence) ---")
    state_a, report_a = runtime.prefill(prompt_a_tokens)
    print(report_a.pretty_print())
    print()

    print("--- Prompt B (shared prefix) ---")
    state_b, report_b = runtime.prefill(prompt_b_tokens)
    print(report_b.pretty_print())
    print()

    print("--- Generate 4 tokens ---")
    result = runtime.generate(prompt_b_tokens, max_new_tokens=4)
    print(f"Generated {len(result['generated_ids'])} tokens")
    print(f"Prefix cache hit: {result['prefix_cache_hit']}")
    print(f"Total errors: {result['total_errors']}")
    print(f"Total warnings: {result['total_warnings']}")
    print(f"Total prefetched: {result['total_prefetched']}")
    print(f"Total offloaded: {result['total_offloaded']}")
    print()

    print("--- Runtime describe ---")
    print(state_b.describe())
    print()

    print("=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
