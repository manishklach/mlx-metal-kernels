import argparse
import time

import mlx.core as mx

from ops.layout_ops import qkv_split, reference_qkv_split


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
    parser.add_argument("--S", type=int, default=16)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--layout", choices=["packed", "explicit"], default="packed")
    parser.add_argument("--backend", choices=["reference", "metal", "all"], default="all")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(50)
    qkv = mx.random.normal((args.B, args.S, 3 * args.H * args.D)).astype(dtype) if args.layout == "packed" else mx.random.normal((args.B, args.S, 3, args.H, args.D)).astype(dtype)
    ref_ms = time_fn(lambda: reference_qkv_split(qkv, H=args.H if args.layout == "packed" else None, D=args.D if args.layout == "packed" else None), iters=args.iters) * 1e3
    backends = ["reference", "metal"] if args.backend == "all" else [args.backend]
    for backend in backends:
        cur_ms = time_fn(lambda: qkv_split(qkv, H=args.H if args.layout == "packed" else None, D=args.D if args.layout == "packed" else None, backend=backend), iters=args.iters) * 1e3
        speedup = ref_ms / cur_ms if cur_ms > 0 else float("inf")
        print(f"backend={backend} B={args.B} S={args.S} H={args.H} D={args.D} layout={args.layout} dtype={args.dtype} ms={cur_ms:.3f} speedup_vs_reference={speedup:.2f}x")


if __name__ == "__main__":
    main()
