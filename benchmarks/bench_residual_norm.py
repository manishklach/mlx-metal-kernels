import argparse
import time

import mlx.core as mx

from ops.fused_ops import residual_add, rmsnorm_residual


def time_fn(fn, warmup=5, iters=20):
    for _ in range(warmup):
        y = fn()
        if isinstance(y, tuple):
            mx.eval(*y)
        else:
            mx.eval(y)
    start = time.perf_counter()
    for _ in range(iters):
        y = fn()
        if isinstance(y, tuple):
            mx.eval(*y)
        else:
            mx.eval(y)
    end = time.perf_counter()
    return (end - start) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--S", type=int, default=16)
    parser.add_argument("--D", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--backend", choices=["reference", "metal", "all"], default="all")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(52)
    x = mx.random.normal((args.B, args.S, args.D)).astype(dtype)
    residual = mx.random.normal((args.B, args.S, args.D)).astype(dtype)
    weight = mx.random.normal((args.D,)).astype(dtype)
    backends = ["reference", "metal"] if args.backend == "all" else [args.backend]
    for backend in backends:
        add_ms = time_fn(lambda: residual_add(x, residual, backend=backend), iters=args.iters) * 1e3
        norm_ms = time_fn(lambda: rmsnorm_residual(x, residual, weight, backend=backend), iters=args.iters) * 1e3
        print(f"backend={backend} B={args.B} S={args.S} D={args.D} dtype={args.dtype} residual_add_ms={add_ms:.3f} rmsnorm_residual_ms={norm_ms:.3f}")


if __name__ == "__main__":
    main()
