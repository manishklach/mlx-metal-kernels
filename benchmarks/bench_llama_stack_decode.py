from __future__ import annotations

import argparse

import mlx.core as mx

from benchmark_utils import time_fn
from models.llama_config import LlamaLikeConfig
from ops.llama_stack_ops import (
    create_random_quantized_llama_stack_weights,
    init_llama_stack_cache,
    llama_stack_decode_loop,
    reference_llama_stack_decode_loop,
)


def _make_config(args):
    return LlamaLikeConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        num_hidden_layers=args.num_layers,
        max_position_embeddings=args.MAX_S,
        vocab_size=args.vocab_size if args.with_lm_head else None,
        model_type="llama_stack_bench",
    ).validate()


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 2.5e-1, 2.5e-1
    return 2e-1, 2e-1


def _run_backend(args, config, dtype, backend):
    weights = create_random_quantized_llama_stack_weights(
        config,
        vocab_size=args.vocab_size,
        bits=args.bits,
        group_size=32,
        dtype=dtype,
        seed=args.seed,
        include_embedding=True,
        include_lm_head=args.with_lm_head,
    )
    cache = init_llama_stack_cache(config, args.B, args.MAX_S, cache_layout=args.cache, dtype=dtype, page_size=args.PAGE_SIZE)
    inputs = mx.random.normal((args.B, args.T, args.hidden_size)).astype(dtype)
    cos = mx.random.normal((args.MAX_S + 4, args.head_dim // 2)).astype(mx.float32)
    sin = mx.random.normal((args.MAX_S + 4, args.head_dim // 2)).astype(mx.float32)
    fn = lambda: llama_stack_decode_loop(inputs, weights, cache, cos, sin, config, backend_preset=backend, cache_layout=args.cache, return_logits=args.with_lm_head)
    timing = time_fn(fn, warmup=3, iters=args.iters)
    return timing, weights, inputs, cos, sin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--MAX_S", type=int, default=128)
    parser.add_argument("--PAGE_SIZE", type=int, default=16)
    parser.add_argument("--cache", choices=["contiguous", "paged"], default="contiguous")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--backend-preset", choices=["reference", "metal", "tiled", "fused_experimental", "all"], default="all")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--with-lm-head", action="store_true")
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(args.seed)
    config = _make_config(args)
    backends = ["reference", "metal", "tiled", "fused_experimental"] if args.backend_preset == "all" else [args.backend_preset]
    results = {}
    for backend in backends:
        timing, weights, inputs, cos, sin = _run_backend(args, config, dtype, backend)
        results[backend] = timing
        if args.validate and backend != "reference":
            ref_cache = init_llama_stack_cache(config, args.B, args.MAX_S, cache_layout=args.cache, dtype=dtype, page_size=args.PAGE_SIZE)
            opt_cache = init_llama_stack_cache(config, args.B, args.MAX_S, cache_layout=args.cache, dtype=dtype, page_size=args.PAGE_SIZE)
            ref, _ = reference_llama_stack_decode_loop(inputs, weights, ref_cache, cos, sin, config, return_logits=args.with_lm_head)
            got, _ = llama_stack_decode_loop(inputs, weights, opt_cache, cos, sin, config, backend_preset=backend, cache_layout=args.cache, return_logits=args.with_lm_head)
            mx.eval(ref, got)
            atol, rtol = _tol(dtype)
            if not mx.allclose(ref, got, atol=atol, rtol=rtol).item():
                raise AssertionError(f"Validation failed for backend {backend}")

    ref_ms = results["reference"]["mean_ms"] if "reference" in results else None
    tiled_ms = results["tiled"]["mean_ms"] if "tiled" in results else None
    for backend in backends:
        total_ms = results[backend]["mean_ms"]
        tokens_per_second = (args.B * args.T * 1000.0) / total_ms
        print(
            {
                "bits": args.bits,
                "cache": args.cache,
                "B": args.B,
                "T": args.T,
                "num_layers": args.num_layers,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "num_heads": args.num_heads,
                "num_kv_heads": args.num_kv_heads,
                "head_dim": args.head_dim,
                "vocab_size": args.vocab_size,
                "max_seq_len": args.MAX_S,
                "dtype": args.dtype,
                "backend_preset": backend,
                "total_ms": total_ms,
                "ms_per_token": total_ms / max(args.T, 1),
                "ms_per_layer_token": total_ms / max(args.T * args.num_layers, 1),
                "tokens_per_second": tokens_per_second,
                "speedup_vs_reference": (ref_ms / total_ms) if ref_ms is not None else None,
                "speedup_vs_tiled": (tiled_ms / total_ms) if tiled_ms is not None and backend != "tiled" else None,
            }
        )


if __name__ == "__main__":
    main()
