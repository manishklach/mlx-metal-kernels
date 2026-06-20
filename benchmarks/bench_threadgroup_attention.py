import argparse

import mlx.core as mx

from benchmark_utils import time_fn
from ops.attention_ops import fast_attention
from ops.decode_ops import decode_attention
from ops.paged_kv_ops import allocate_paged_kv_cache, paged_decode_attention


def _dtype(name):
    return mx.float16 if name == "float16" else mx.bfloat16


def _backends(mode, generic, threadgroup):
    if mode == "generic":
        return [generic]
    if mode == "threadgroup":
        return [threadgroup]
    return ["reference", generic, threadgroup]


def _print_result(mode, backend, args, timing, generic_ms=None, reference_ms=None):
    mean_ms = timing["mean_ms"]
    speedup_generic = ""
    speedup_reference = ""
    if generic_ms is not None and backend != "reference":
        speedup_generic = f" speedup_vs_generic={generic_ms / mean_ms:.3f}"
    if reference_ms is not None and backend != "reference":
        speedup_reference = f" speedup_vs_reference={reference_ms / mean_ms:.3f}"
    print(
        f"mode={mode} backend={backend} B={args.B} S={args.S} MAX_S={args.MAX_S} PAGE_SIZE={args.PAGE_SIZE} "
        f"H={args.H} D={args.D} length={args.length} dtype={args.dtype} causal={args.causal} "
        f"mean_ms={mean_ms:.3f}{speedup_generic}{speedup_reference}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["decode", "paged_decode", "prefill"], default="decode")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=128)
    parser.add_argument("--MAX_S", type=int, default=128)
    parser.add_argument("--PAGE_SIZE", type=int, default=16)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--backend", choices=["generic", "threadgroup", "all"], default="all")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    dtype = _dtype(args.dtype)
    mx.random.seed(209)

    timings = {}
    if args.mode == "decode":
        q = mx.random.normal((args.B, 1, args.H, args.D)).astype(dtype)
        K_cache = mx.random.normal((args.B, args.MAX_S, args.H, args.D)).astype(dtype)
        V_cache = mx.random.normal((args.B, args.MAX_S, args.H, args.D)).astype(dtype)
        for backend in _backends(args.backend, "metal", "metal_threadgroup"):
            timings[backend] = time_fn(
                lambda b=backend: decode_attention(q, K_cache, V_cache, lengths=args.length, backend=b),
                warmup=3,
                iters=args.iters,
            )
            _print_result(args.mode, backend, args, timings[backend], timings.get("metal", {}).get("mean_ms"), timings.get("reference", {}).get("mean_ms"))
    elif args.mode == "paged_decode":
        K_pages, V_pages, block_table = allocate_paged_kv_cache(args.B, args.MAX_S, args.H, args.D, args.PAGE_SIZE, dtype)
        K_pages = mx.random.normal(K_pages.shape).astype(dtype)
        V_pages = mx.random.normal(V_pages.shape).astype(dtype)
        q = mx.random.normal((args.B, 1, args.H, args.D)).astype(dtype)
        for backend in _backends(args.backend, "metal", "metal_threadgroup"):
            timings[backend] = time_fn(
                lambda b=backend: paged_decode_attention(q, K_pages, V_pages, block_table, args.length, backend=b),
                warmup=3,
                iters=args.iters,
            )
            _print_result(args.mode, backend, args, timings[backend], timings.get("metal", {}).get("mean_ms"), timings.get("reference", {}).get("mean_ms"))
    else:
        Q = mx.random.normal((args.B, args.S, args.H, args.D)).astype(dtype)
        K = mx.random.normal((args.B, args.S, args.H, args.D)).astype(dtype)
        V = mx.random.normal((args.B, args.S, args.H, args.D)).astype(dtype)
        for backend in _backends(args.backend, "baseline", "threadgroup"):
            timings[backend] = time_fn(
                lambda b=backend: fast_attention(Q, K, V, causal=args.causal, backend=b),
                warmup=3,
                iters=args.iters,
            )
            _print_result(args.mode, backend, args, timings[backend], timings.get("baseline", {}).get("mean_ms"), timings.get("reference", {}).get("mean_ms"))


if __name__ == "__main__":
    main()
