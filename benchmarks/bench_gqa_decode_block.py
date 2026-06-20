import argparse
import time

import mlx.core as mx

from benchmark_utils import summarize_times, sync
from ops.gqa_ops import gqa_decode_block_from_qkv, paged_gqa_decode_block_from_qkv
from ops.paged_kv_ops import allocate_paged_kv_cache


def _time_case(fn, warmup=3, iters=10):
    for _ in range(warmup):
        sync(fn())
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sync(fn())
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e3)
    return summarize_times(samples)


def _run_loop(args, dtype):
    cos = mx.random.normal((args.MAX_S + 4, args.D // 2)).astype(mx.float32)
    sin = mx.random.normal((args.MAX_S + 4, args.D // 2)).astype(mx.float32)
    if args.cache == "contiguous":
        K_cache = mx.zeros((args.B, args.MAX_S, args.Hkv, args.D), dtype=dtype)
        V_cache = mx.zeros((args.B, args.MAX_S, args.Hkv, args.D), dtype=dtype)

        def step():
            nonlocal K_cache, V_cache
            out = None
            for pos in range(args.T):
                qkv = mx.random.normal((args.B, 1, args.Hq * args.D + 2 * args.Hkv * args.D)).astype(dtype)
                out, K_cache, V_cache = gqa_decode_block_from_qkv(
                    qkv,
                    K_cache,
                    V_cache,
                    cos,
                    sin,
                    pos,
                    num_attention_heads=args.Hq,
                    num_key_value_heads=args.Hkv,
                    head_dim=args.D,
                    backend="reference",
                )
            return out, K_cache, V_cache

        return _time_case(step, iters=args.iters)

    K_pages, V_pages, block_table = allocate_paged_kv_cache(args.B, args.MAX_S, args.Hkv, args.D, args.PAGE_SIZE, dtype)

    def step():
        nonlocal K_pages, V_pages
        out = None
        for pos in range(args.T):
            qkv = mx.random.normal((args.B, 1, args.Hq * args.D + 2 * args.Hkv * args.D)).astype(dtype)
            out, K_pages, V_pages = paged_gqa_decode_block_from_qkv(
                qkv,
                K_pages,
                V_pages,
                block_table,
                cos,
                sin,
                pos,
                num_attention_heads=args.Hq,
                num_key_value_heads=args.Hkv,
                head_dim=args.D,
                backend="reference",
            )
        return out, K_pages, V_pages

    return _time_case(step, iters=args.iters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--MAX_S", type=int, default=128)
    parser.add_argument("--PAGE_SIZE", type=int, default=16)
    parser.add_argument("--Hq", type=int, default=32)
    parser.add_argument("--Hkv", type=int, default=8)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--cache", choices=["contiguous", "paged"], default="contiguous")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    timing = _run_loop(args, dtype)
    avg_s = timing["mean_ms"] / 1e3
    print(
        f"cache={args.cache} B={args.B} MAX_S={args.MAX_S} PAGE_SIZE={args.PAGE_SIZE if args.cache == 'paged' else 'n/a'} "
        f"Hq={args.Hq} Hkv={args.Hkv} D={args.D} T={args.T} dtype={args.dtype} "
        f"ms_per_step={timing['mean_ms'] / args.T:.3f} tokens_per_second={(args.B * args.T) / avg_s:.3f}"
    )


if __name__ == "__main__":
    main()
