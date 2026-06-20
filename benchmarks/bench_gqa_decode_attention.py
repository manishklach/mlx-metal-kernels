import argparse

import mlx.core as mx

from benchmark_utils import summarize_times, sync
from ops.gqa_ops import reference_gqa_decode_attention, reference_paged_gqa_decode_attention
from ops.paged_kv_ops import allocate_paged_kv_cache


def _time_case(fn, warmup=3, iters=10):
    import time

    for _ in range(warmup):
        sync(fn())
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sync(fn())
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e3)
    return summarize_times(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--MAX_S", type=int, default=128)
    parser.add_argument("--PAGE_SIZE", type=int, default=16)
    parser.add_argument("--Hq", type=int, default=32)
    parser.add_argument("--Hkv", type=int, default=8)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--cache", choices=["contiguous", "paged"], default="contiguous")
    parser.add_argument("--backend", choices=["reference", "metal_gqa", "all"], default="all")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    q = mx.random.normal((args.B, 1, args.Hq, args.D)).astype(dtype)
    backends = ["reference"] if args.backend != "all" else ["reference", "metal_gqa"]

    if args.cache == "contiguous":
        K_cache = mx.random.normal((args.B, args.MAX_S, args.Hkv, args.D)).astype(dtype)
        V_cache = mx.random.normal((args.B, args.MAX_S, args.Hkv, args.D)).astype(dtype)
        for backend in backends:
            if backend == "metal_gqa":
                print(f"cache=contiguous B={args.B} MAX_S={args.MAX_S} PAGE_SIZE=n/a Hq={args.Hq} Hkv={args.Hkv} D={args.D} length={args.length} dtype={args.dtype} backend={backend} status=skipped reason=no_metal_gqa_kernel")
                continue
            timing = _time_case(lambda: reference_gqa_decode_attention(q, K_cache, V_cache, lengths=args.length), iters=args.iters)
            print(f"cache=contiguous B={args.B} MAX_S={args.MAX_S} PAGE_SIZE=n/a Hq={args.Hq} Hkv={args.Hkv} D={args.D} length={args.length} dtype={args.dtype} backend={backend} mean_ms={timing['mean_ms']:.3f}")
        return

    K_pages, V_pages, block_table = allocate_paged_kv_cache(args.B, args.MAX_S, args.Hkv, args.D, args.PAGE_SIZE, dtype)
    K_pages = mx.random.normal(K_pages.shape).astype(dtype)
    V_pages = mx.random.normal(V_pages.shape).astype(dtype)
    for backend in backends:
        if backend == "metal_gqa":
            print(f"cache=paged B={args.B} MAX_S={args.MAX_S} PAGE_SIZE={args.PAGE_SIZE} Hq={args.Hq} Hkv={args.Hkv} D={args.D} length={args.length} dtype={args.dtype} backend={backend} status=skipped reason=no_metal_gqa_kernel")
            continue
        timing = _time_case(lambda: reference_paged_gqa_decode_attention(q, K_pages, V_pages, block_table, lengths=args.length), iters=args.iters)
        print(f"cache=paged B={args.B} MAX_S={args.MAX_S} PAGE_SIZE={args.PAGE_SIZE} Hq={args.Hq} Hkv={args.Hkv} D={args.D} length={args.length} dtype={args.dtype} backend={backend} mean_ms={timing['mean_ms']:.3f}")


if __name__ == "__main__":
    main()
