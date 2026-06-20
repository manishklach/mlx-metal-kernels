from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import TinyGenerationPipeline, TinyGenerationPipelineConfig


def _summarize(samples: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "iters": len(samples),
    }


def _make_prompt(length: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(alphabet[idx % len(alphabet)] for idx in range(length))


def _run_once(args) -> dict:
    config = TinyGenerationPipelineConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        num_hidden_layers=args.num_layers,
        max_position_embeddings=max(args.prompt_len + args.max_new_tokens + 8, 32),
        vocab_size=args.vocab_size,
        bits=args.bits,
        backend_preset=args.backend_preset,
    ).validate()
    pipeline = TinyGenerationPipeline(config=config)
    prompt = _make_prompt(args.prompt_len)
    start = time.perf_counter()
    result = pipeline.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        greedy=args.greedy,
        top_k=None if args.greedy else 8,
        temperature=1.0 if args.greedy else 0.9,
    )
    elapsed_ms = (time.perf_counter() - start) * 1e3
    return {
        "elapsed_ms": elapsed_ms,
        "prompt_tokens": len(result.prompt_ids),
        "generated_tokens": len(result.generated_ids),
        "total_tokens_processed": len(result.all_ids),
    }


def _benchmark_backend(args, backend: str) -> None:
    samples = []
    latest = None
    for _ in range(args.iters):
        run_args = argparse.Namespace(**vars(args))
        run_args.backend_preset = backend
        latest = _run_once(run_args)
        samples.append(latest["elapsed_ms"])
    timing = _summarize(samples)
    generated_tokens = max(latest["generated_tokens"], 1)
    print(
        {
            "prompt_len": args.prompt_len,
            "max_new_tokens": args.max_new_tokens,
            "total_tokens_processed": latest["total_tokens_processed"],
            "bits": args.bits,
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "vocab_size": args.vocab_size,
            "backend_preset": backend,
            "total_ms": timing["mean_ms"],
            "ms_per_generated_token": timing["mean_ms"] / generated_tokens,
            "tokens_per_second": generated_tokens * 1000.0 / timing["mean_ms"],
            "iters": args.iters,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--backend-preset", choices=["reference", "metal", "tiled", "fused_experimental", "all"], default="fused_experimental")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    backends = ["reference", "metal", "tiled", "fused_experimental"] if args.backend_preset == "all" else [args.backend_preset]
    for backend in backends:
        _benchmark_backend(args, backend)


if __name__ == "__main__":
    main()
