import argparse
import time

import mlx.core as mx

from ops.activation_ops import swiglu


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
    parser.add_argument("--S", type=int, default=16)
    parser.add_argument("--D", type=int, default=256)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--backend", choices=["reference", "metal", "auto"], default="auto")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(2)
    gate = mx.random.normal((args.B, args.S, args.D)).astype(dtype)
    up = mx.random.normal((args.B, args.S, args.D)).astype(dtype)
    ms = time_fn(lambda: swiglu(gate, up, backend=args.backend), iters=args.iters) * 1e3
    print(f"backend={args.backend} B={args.B} S={args.S} D={args.D} dtype={args.dtype} ms={ms:.3f}")


if __name__ == "__main__":
    main()
