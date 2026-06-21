from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    import mlx.core as mx
except ImportError:
    mx = None
import numpy as np

from models import (
    GenerationConfig,
    InMemoryPrefixCache,
    TinyGenerationPipeline,
    TinyGenerationPipelineConfig,
    compute_fingerprint,
    create_synthetic_stack_generation_model,
)
from models.prefix_cache import prefill_with_prefix_reuse
from ops.kv_cache_reuse_ops import clone_stack_cache


def _benchmark_prefill(model, token_ids, gen_config, prefix_cache=None):
    start = time.perf_counter()
    logits, state, meta = prefill_with_prefix_reuse(
        token_ids, model, prefix_cache=prefix_cache, generation_config=gen_config,
    )
    elapsed = time.perf_counter() - start
    return logits, state, meta, elapsed


def _assert_logits_close(a, b, *, atol: float = 1e-4) -> None:
    if a is None or b is None:
        raise AssertionError("Expected logits from both reference and cache-backed prefill paths")
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    if a_np.shape != b_np.shape:
        raise AssertionError(f"logit shape mismatch: {a_np.shape} vs {b_np.shape}")
    if not np.allclose(a_np, b_np, atol=atol, rtol=0.0):
        raise AssertionError("cache-backed logits diverged from the reference prefill path")


def run_benchmark(
    prompt_tokens: int = 8,
    reused_tokens: int = 6,
    max_new_tokens: int = 4,
    iters: int = 5,
    validate: bool = False,
):
    model = create_synthetic_stack_generation_model(seed=42)
    gen_config = GenerationConfig(max_new_tokens=max_new_tokens, backend_preset="reference")
    token_ids = list(range(prompt_tokens))
    prefix_cache = InMemoryPrefixCache(max_size=16)
    reused_ids = token_ids[:reused_tokens]
    suffix_ids = token_ids[reused_tokens:]

    _benchmark_prefill(model, token_ids, gen_config)

    if validate:
        ref_logits, ref_state, _, _ = _benchmark_prefill(model, token_ids, gen_config, prefix_cache=None)
        cache_logits, cache_state, cache_meta, _ = _benchmark_prefill(model, token_ids, gen_config, prefix_cache=prefix_cache)
        _assert_logits_close(ref_logits, cache_logits)
        if ref_state.position != cache_state.position:
            raise AssertionError(f"state position mismatch: {ref_state.position} vs {cache_state.position}")
        if list(ref_state.generated_ids) != list(cache_state.generated_ids):
            raise AssertionError("generated_ids mismatch between reference and cache-backed prefill")
        print(f"  Validate: cache_hit={cache_meta['prefix_cache_hit']}, matched_length={cache_meta['matched_length']}")
        if cache_meta["prefix_cache_hit"]:
            print("  Cache hit - reuse working correctly")
        results = []
        # cold
        _ = _benchmark_prefill(model, token_ids, gen_config, prefix_cache=None)
        times_cold = []
        for _ in range(iters):
            _, _, _, t = _benchmark_prefill(model, token_ids, gen_config, prefix_cache=None)
            times_cold.append(t)
        avg_cold = sum(times_cold) / len(times_cold)
        results.append(("cold_prefill", avg_cold, prompt_tokens))
        # warm
        _benchmark_prefill(model, reused_ids, gen_config)
        times_warm = []
        for _ in range(iters):
            _, _, _, t = _benchmark_prefill(model, token_ids, gen_config, prefix_cache=prefix_cache)
            times_warm.append(t)
        avg_warm = sum(times_warm) / len(times_warm)
        results.append(("warm_reuse", avg_warm, reused_tokens))
        # exact match
        _benchmark_prefill(model, reused_ids, gen_config)
        times_exact = []
        for _ in range(iters):
            _, _, _, t = _benchmark_prefill(model, reused_ids, gen_config, prefix_cache=prefix_cache)
            times_exact.append(t)
        avg_exact = sum(times_exact) / len(times_exact)
        results.append(("exact_match", avg_exact, reused_tokens))
        print(f"  Cold prefill ({prompt_tokens} tokens):  {avg_cold*1000:.2f}ms")
        print(f"  Warm reuse  ({reused_tokens}/{prompt_tokens} tokens): {avg_warm*1000:.2f}ms")
        print(f"  Exact match ({reused_tokens} tokens):     {avg_exact*1000:.2f}ms")
        if avg_cold > 0:
            speedup = (avg_cold - avg_warm) / avg_cold * 100
            print(f"  Speedup: {speedup:.1f}%")
        return results
    times_cold = []
    for _ in range(iters):
        _, _, _, t = _benchmark_prefill(model, token_ids, gen_config, prefix_cache=None)
        times_cold.append(t)
    avg_cold = sum(times_cold) / len(times_cold)
    _benchmark_prefill(model, reused_ids, gen_config)
    times_warm = []
    for _ in range(iters):
        _, _, _, t = _benchmark_prefill(model, token_ids, gen_config, prefix_cache=prefix_cache)
        times_warm.append(t)
    avg_warm = sum(times_warm) / len(times_warm)
    _benchmark_prefill(model, reused_ids, gen_config)
    times_exact = []
    for _ in range(iters):
        _, _, _, t = _benchmark_prefill(model, reused_ids, gen_config, prefix_cache=prefix_cache)
        times_exact.append(t)
    avg_exact = sum(times_exact) / len(times_exact)
    results = {
        "cold_prefill_ms": round(avg_cold * 1000, 3),
        "warm_reuse_ms": round(avg_warm * 1000, 3),
        "exact_match_ms": round(avg_exact * 1000, 3),
        "prompt_tokens": prompt_tokens,
        "reused_tokens": reused_tokens,
        "iters": iters,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark prefix KV-cache reuse")
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--reused-tokens", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    print(f"Benchmark: prefix KV-cache reuse (prompt={args.prompt_tokens}, reuse={args.reused_tokens}, iters={args.iters})")
    results = run_benchmark(
        prompt_tokens=args.prompt_tokens,
        reused_tokens=args.reused_tokens,
        max_new_tokens=args.max_new_tokens,
        iters=args.iters,
        validate=args.validate,
    )
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote results to {args.output}")
    elif not args.validate:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
