from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np


def _has_mlx():
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def _get_dtype():
    return "float16" if _has_mlx() else "float32"


def create_synthetic_pipeline(
    *,
    num_layers=1,
    hidden_size=64,
    intermediate_size=128,
    num_heads=4,
    num_kv_heads=2,
    head_dim=16,
    vocab_size=128,
    bits=4,
    max_seq_len=128,
    backend_preset="fused_experimental",
):
    from models.tiny_generation_pipeline import TinyGenerationPipeline, TinyGenerationPipelineConfig

    cfg = TinyGenerationPipelineConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        num_hidden_layers=num_layers,
        max_position_embeddings=max_seq_len,
        vocab_size=vocab_size,
        bits=bits,
        group_size=32,
        dtype=_get_dtype(),
        backend_preset=backend_preset,
        cache_layout="contiguous",
        use_prefill=True,
        use_prefix_cache=False,
    )
    return TinyGenerationPipeline(config=cfg)


def run_bench(pipeline, prompt_ids, max_new_tokens, draft_length, draft_mode, verifier, seed, iters):
    prompt_text = " ".join(str(tid) for tid in prompt_ids)
    timings: list[float] = []
    results: list[dict[str, Any]] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = pipeline.generate_speculative(
            prompt_text,
            max_new_tokens=max_new_tokens,
            draft_length=draft_length,
            draft_mode=draft_mode,
            verifier_mode=verifier,
            seed=seed,
        )
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)
        results.append(result)
    mean_ms = float(np.mean(timings))
    total_proposed = sum(s.proposal.length() for s in results[-1].steps)
    total_accepted = sum(s.accepted_count for s in results[-1].steps)
    acceptance_rate = total_accepted / total_proposed if total_proposed > 0 else 0.0
    avg_tokens_per_step = float(
        sum(len(s.committed_token_ids) for s in results[-1].steps) / max(len(results[-1].steps), 1)
    )
    return mean_ms, total_proposed, total_accepted, acceptance_rate, avg_tokens_per_step


def main():
    parser = argparse.ArgumentParser(description="Benchmark parallel speculative verification")
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--draft-length", type=int, default=4)
    parser.add_argument("--draft-mode", default="fixed", choices=["fixed", "random", "self", "mtp"])
    parser.add_argument("--verifier", default="both", choices=["sequential", "parallel", "both"])
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--backend-preset", default="fused_experimental", choices=["reference", "metal", "tiled", "fused_experimental"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    pipeline = create_synthetic_pipeline(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        vocab_size=args.vocab_size,
        bits=args.bits,
        max_seq_len=args.prompt_len + args.max_new_tokens + 16,
        backend_preset=args.backend_preset,
    )
    prompt_ids = list(range(min(args.prompt_len, args.vocab_size)))
    prompt_text = " ".join(str(tid) for tid in prompt_ids)

    if args.verifier in ("sequential", "both"):
        seq_ms, seq_proposed, seq_accepted, seq_ar, seq_tps = run_bench(
            pipeline, prompt_ids, args.max_new_tokens, args.draft_length, args.draft_mode, "sequential", args.seed, args.iters,
        )
        print(f"verifier=sequential mean_ms={seq_ms:.2f} acceptance_rate={seq_ar:.3f} tokens_per_step={seq_tps:.2f}")

    if args.verifier in ("parallel", "both"):
        par_ms, par_proposed, par_accepted, par_ar, par_tps = run_bench(
            pipeline, prompt_ids, args.max_new_tokens, args.draft_length, args.draft_mode, "parallel", args.seed, args.iters,
        )
        print(f"verifier=parallel    mean_ms={par_ms:.2f} acceptance_rate={par_ar:.3f} tokens_per_step={par_tps:.2f}")
        if args.verifier == "both":
            speedup = seq_ms / par_ms if par_ms > 0 else 0.0
            print(f"speedup_vs_sequential_verifier={speedup:.2f}x")
            print("parallel verifier currently uses staged decode loop; true batched prefill verification is future work.")

    metrics = {
        "prompt_len": args.prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "draft_length": args.draft_length,
        "draft_mode": args.draft_mode,
        "verifier": args.verifier,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "bits": args.bits,
        "backend_preset": args.backend_preset,
        "iters": args.iters,
    }
    print(f"metrics={metrics}")


if __name__ == "__main__":
    main()
