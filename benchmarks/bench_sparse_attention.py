from __future__ import annotations

import argparse
import math

import mlx.core as mx

from benchmark_utils import time_fn
from ops.gqa_ops import reference_gqa_attention
from ops.sparse_attention_ops import SparseAttentionPattern, reference_sparse_gqa_attention, sparse_gqa_attention


def _pattern_from_args(args, backend_name: str) -> SparseAttentionPattern:
    sink_tokens = args.sink_tokens if backend_name != "metal_sliding_window" else 0
    return SparseAttentionPattern(
        pattern="sliding_window_sink" if sink_tokens > 0 else "sliding_window",
        window_size=args.window_size,
        sink_tokens=sink_tokens,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=512)
    parser.add_argument("--Hq", type=int, default=32)
    parser.add_argument("--Hkv", type=int, default=8)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--backend", choices=["reference", "metal_sliding_window", "metal_sliding_window_sink", "all"], default="all")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(950)
    Q = mx.random.normal((args.B, args.S, args.Hq, args.D)).astype(dtype)
    K = mx.random.normal((args.B, args.S, args.Hkv, args.D)).astype(dtype)
    V = mx.random.normal((args.B, args.S, args.Hkv, args.D)).astype(dtype)
    dense_ref_timing = time_fn(lambda: reference_gqa_attention(Q, K, V, causal=True, scale=1.0 / math.sqrt(args.D)), warmup=1, iters=max(1, min(args.iters, 5)))
    backends = ["reference", "metal_sliding_window", "metal_sliding_window_sink"] if args.backend == "all" else [args.backend]
    for backend_name in backends:
        pattern = _pattern_from_args(args, backend_name)
        sparse_ref_timing = time_fn(lambda p=pattern: reference_sparse_gqa_attention(Q, K, V, p), warmup=1, iters=max(1, min(args.iters, 5)))
        if backend_name == "reference":
            fn = lambda p=pattern: reference_sparse_gqa_attention(Q, K, V, p)
        else:
            fn = lambda b=backend_name, p=pattern: sparse_gqa_attention(Q, K, V, p, backend=b)
        if args.validate and backend_name != "reference":
            got = sparse_gqa_attention(Q, K, V, pattern, backend=backend_name)
            ref = reference_sparse_gqa_attention(Q, K, V, pattern)
            mx.eval(got, ref)
            atol = 6e-2 if args.dtype == "bfloat16" else 4e-2
            if not mx.allclose(got, ref, atol=atol, rtol=atol).item():
                raise AssertionError(f"{backend_name} failed validation")
        timing = time_fn(fn, warmup=3, iters=args.iters)
        tokens_per_second = (args.B * args.S * 1000.0) / timing["mean_ms"]
        print(
            f"B={args.B} S={args.S} Hq={args.Hq} Hkv={args.Hkv} D={args.D} "
            f"window_size={args.window_size} sink_tokens={pattern.sink_tokens} dtype={args.dtype} backend={backend_name} "
            f"mean_ms={timing['mean_ms']:.3f} tokens_per_second={tokens_per_second:.3f} "
            f"speedup_vs_dense_reference={dense_ref_timing['mean_ms'] / timing['mean_ms']:.3f} "
            f"speedup_vs_sparse_reference={sparse_ref_timing['mean_ms'] / timing['mean_ms']:.3f}"
        )


if __name__ == "__main__":
    main()
