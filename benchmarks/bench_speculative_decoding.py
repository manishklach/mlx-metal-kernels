"""Measure speculative decoding plumbing overhead vs baseline greedy generation."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import FixedDraftProposer, RandomDraftProposer, SpeculativeConfig, SpeculativeGenerator
from models.tiny_generation_pipeline import TinyGenerationPipeline, TinyGenerationPipelineConfig


def _make_prompt(length: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(alphabet[idx % len(alphabet)] for idx in range(length))


def _run_baseline(args) -> dict:
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
    result = pipeline.generate(prompt, max_new_tokens=args.max_new_tokens, seed=args.seed, greedy=True)
    elapsed_ms = (time.perf_counter() - start) * 1e3
    return {
        "elapsed_ms": elapsed_ms,
        "prompt_tokens": len(result.prompt_ids),
        "generated_tokens": len(result.generated_ids),
    }


def _run_speculative(args, draft_mode: str) -> dict:
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
    if draft_mode == "fixed":
        proposer = FixedDraftProposer(list(range(pipeline.vocab_size))[:args.draft_length])
    elif draft_mode == "random":
        proposer = RandomDraftProposer(pipeline.vocab_size, seed=args.seed)
    else:
        raise ValueError(f"Unknown draft_mode: {draft_mode!r}")
    spec_cfg = SpeculativeConfig(
        draft_length=args.draft_length,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        greedy_verify=True,
        seed=args.seed,
        backend_preset=args.backend_preset,
    ).validate()
    gen = SpeculativeGenerator(pipeline, draft_proposer=proposer, config=spec_cfg)
    prompt = _make_prompt(args.prompt_len)
    start = time.perf_counter()
    result = gen.generate_text(prompt)
    elapsed_ms = (time.perf_counter() - start) * 1e3
    return {
        "elapsed_ms": elapsed_ms,
        "prompt_tokens": len(result.prompt_ids),
        "generated_tokens": len(result.generated_ids),
        "acceptance_rate": result.acceptance_rate(),
        "tokens_per_step": result.tokens_per_step(),
        "num_steps": len(result.steps),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding overhead")
    parser.add_argument("--draft-mode", choices=["fixed", "random"], default="fixed", help="Draft proposer mode")
    parser.add_argument("--iters", type=int, default=5, help="Number of benchmark iterations")
    parser.add_argument("--prompt-len", type=int, default=8, help="Prompt length")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Max tokens to generate")
    parser.add_argument("--draft-length", type=int, default=4, help="Draft length")
    parser.add_argument("--backend-preset", default="reference", help="Backend preset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-size", choices=["tiny", "small"], default="tiny", help="Model size preset")
    args = parser.parse_args()

    if args.model_size == "tiny":
        args.hidden_size = 32
        args.intermediate_size = 64
        args.num_heads = 2
        args.num_kv_heads = 1
        args.head_dim = 16
        args.num_layers = 1
        args.vocab_size = 64
        args.bits = 4
    else:
        args.hidden_size = 64
        args.intermediate_size = 128
        args.num_heads = 4
        args.num_kv_heads = 2
        args.head_dim = 16
        args.num_layers = 2
        args.vocab_size = 128
        args.bits = 4

    print(f"Benchmarking speculative decoding (draft_mode={args.draft_mode}, draft_length={args.draft_length})...")

    baseline_samples = []
    for i in range(args.iters):
        b = _run_baseline(args)
        baseline_samples.append(b["elapsed_ms"])
        print(f"  baseline iter {i+1}: {b['elapsed_ms']:.2f}ms ({b['generated_tokens']} tokens)")

    spec_samples = []
    for i in range(args.iters):
        s = _run_speculative(args, args.draft_mode)
        spec_samples.append(s["elapsed_ms"])
        print(f"  speculative iter {i+1}: {s['elapsed_ms']:.2f}ms ({s['generated_tokens']} tokens, "
              f"accept_rate={s['acceptance_rate']:.2f}, tps={s['tokens_per_step']:.2f})")

    def _summary(samples):
        return {
            "mean_ms": statistics.fmean(samples),
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
            "iters": len(samples),
        }

    bl_summary = _summary(baseline_samples)
    sp_summary = _summary(spec_samples)

    print("\n--- Results ---")
    print(f"Baseline:     mean={bl_summary['mean_ms']:.2f}ms  median={bl_summary['median_ms']:.2f}ms")
    print(f"Speculative:  mean={sp_summary['mean_ms']:.2f}ms  median={sp_summary['median_ms']:.2f}ms")
    ratio = sp_summary["mean_ms"] / max(bl_summary["mean_ms"], 0.001)
    print(f"Overhead ratio: {ratio:.2f}x")


if __name__ == "__main__":
    main()
