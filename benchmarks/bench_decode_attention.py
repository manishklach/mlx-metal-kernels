import argparse
import time

import mlx.core as mx

from ops.decode_ops import decode_attention


def time_fn(fn, warmup=5, iters=20):
    for _ in range(warmup):
        y = fn()
        mx.eval(y)
    start = time.perf_counter()
    for _ in range(iters):
        y = fn()
        mx.eval(y)
    end = time.perf_counter()
    return (end - start) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--S", type=int, default=32)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--backend", choices=["reference", "metal", "auto"], default="auto")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(3)
    q = mx.random.normal((args.B, 1, args.H, args.D)).astype(dtype)
    K_cache = mx.random.normal((args.B, args.S, args.H, args.D)).astype(dtype)
    V_cache = mx.random.normal((args.B, args.S, args.H, args.D)).astype(dtype)
    ms = time_fn(lambda: decode_attention(q, K_cache, V_cache, backend=args.backend), iters=args.iters) * 1e3
    print(f"backend={args.backend} B={args.B} S={args.S} H={args.H} D={args.D} dtype={args.dtype} ms={ms:.3f}")


if __name__ == "__main__":
    main()
