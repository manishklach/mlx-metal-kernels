import argparse
import time

import mlx.core as mx

from ops.kv_cache_ops import kv_cache_update


def time_fn(fn, warmup=5, iters=20):
    for _ in range(warmup):
        y = fn()
        mx.eval(*y)
    start = time.perf_counter()
    for _ in range(iters):
        y = fn()
        mx.eval(*y)
    end = time.perf_counter()
    return (end - start) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--MAX_S", type=int, default=128)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--backend", choices=["reference", "metal", "auto"], default="auto")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(20)
    K_cache = mx.random.normal((args.B, args.MAX_S, args.H, args.D)).astype(dtype)
    V_cache = mx.random.normal((args.B, args.MAX_S, args.H, args.D)).astype(dtype)
    k_new = mx.random.normal((args.B, 1, args.H, args.D)).astype(dtype)
    v_new = mx.random.normal((args.B, 1, args.H, args.D)).astype(dtype)
    positions = mx.arange(args.B, dtype=mx.int32) % args.MAX_S
    ms = time_fn(
        lambda: kv_cache_update(K_cache, V_cache, k_new, v_new, positions, backend=args.backend),
        iters=args.iters,
    ) * 1e3
    print(
        f"backend={args.backend} B={args.B} MAX_S={args.MAX_S} H={args.H} "
        f"D={args.D} dtype={args.dtype} ms={ms:.3f}"
    )


if __name__ == "__main__":
    main()
