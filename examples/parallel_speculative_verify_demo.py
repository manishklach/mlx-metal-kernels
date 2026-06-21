from __future__ import annotations

"""
Parallel speculative verification demo.

Creates a tiny synthetic pipeline with random weights and demonstrates:
  - sequential verifier (baseline)
  - parallel/staged verifier (staged decode loop)
"""


def _has_mlx():
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    from models.tiny_generation_pipeline import TinyGenerationPipeline, TinyGenerationPipelineConfig

    dtype = "float16" if _has_mlx() else "float32"
    cfg = TinyGenerationPipelineConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=1,
        max_position_embeddings=64,
        vocab_size=128,
        bits=8,
        group_size=16,
        dtype=dtype,
        backend_preset="fused_experimental",
        cache_layout="contiguous",
        use_prefill=True,
        use_prefix_cache=False,
    )
    pipe = TinyGenerationPipeline(config=cfg)

    prompt = "hello world"
    draft_len = 4
    max_new = 8

    print("=== Sequential verifier ===")
    seq_result = pipe.generate_speculative(
        prompt,
        max_new_tokens=max_new,
        draft_length=draft_len,
        draft_mode="fixed",
        verifier_mode="sequential",
    )
    for step in seq_result.steps:
        print(f"  proposed={step.proposal.token_ids} target={step.verification.target_token_ids} "
              f"accept={step.verification.accept_mask} accepted={step.accepted_count}")
    print(f"  generated_ids={seq_result.generated_ids}")
    print(f"  acceptance_rate={seq_result.acceptance_rate():.3f}")
    print(f"  tokens_per_step={seq_result.tokens_per_step():.2f}")

    print()
    print("=== Parallel/Staged verifier ===")
    par_result = pipe.generate_speculative(
        prompt,
        max_new_tokens=max_new,
        draft_length=draft_len,
        draft_mode="fixed",
        verifier_mode="parallel",
    )
    for step in par_result.steps:
        print(f"  proposed={step.proposal.token_ids} target={step.verification.target_token_ids} "
              f"accept={step.verification.accept_mask} accepted={step.accepted_count}")
    print(f"  generated_ids={par_result.generated_ids}")
    print(f"  acceptance_rate={par_result.acceptance_rate():.3f}")
    print(f"  tokens_per_step={par_result.tokens_per_step():.2f}")

    print()
    print("WARNING: Synthetic/random weights. This demonstrates speculative verification")
    print("plumbing, not production speed or model quality.")


if __name__ == "__main__":
    main()
