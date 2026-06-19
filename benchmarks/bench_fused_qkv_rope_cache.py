import argparse
import time

import mlx.core as mx

from ops.fused_ops import qkv_rope_cache_update


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
    parser.add_argument("--backend", choices=["reference", "metal", "all"], default="all")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(51)
    qkv = mx.random.normal((args.B, 1, 3 * args.H * args.D)).astype(dtype)
    K_cache = mx.random.normal((args.B, args.MAX_S, args.H, args.D)).astype(dtype)
    V_cache = mx.random.normal((args.B, args.MAX_S, args.H, args.D)).astype(dtype)
    cos = mx.random.normal((args.MAX_S + 4, args.D // 2)).astype(mx.float32)
    sin = mx.random.normal((args.MAX_S + 4, args.D // 2)).astype(mx.float32)
    positions = mx.arange(args.B, dtype=mx.int32) % args.MAX_S
    backends = ["reference", "metal"] if args.backend == "all" else [args.backend]
    for backend in backends:
        ms = time_fn(lambda: qkv_rope_cache_update(qkv, K_cache, V_cache, cos, sin, positions, H=args.H, D=args.D, backend=backend), iters=args.iters) * 1e3
        print(f"backend={backend} B={args.B} MAX_S={args.MAX_S} H={args.H} D={args.D} dtype={args.dtype} ms={ms:.3f}")


if __name__ == "__main__":
    main()
