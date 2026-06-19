import argparse
import time

import mlx.core as mx

from ops.rope_ops import apply_rope


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
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=16)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--position-offset", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--backend", choices=["reference", "metal", "auto"], default="auto")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(1)
    x = mx.random.normal((args.B, args.S, args.H, args.D)).astype(dtype)
    rows = args.S + args.position_offset + 4
    cos = mx.random.normal((rows, args.D // 2)).astype(mx.float32)
    sin = mx.random.normal((rows, args.D // 2)).astype(mx.float32)
    ms = time_fn(
        lambda: apply_rope(x, cos, sin, position_offset=args.position_offset, backend=args.backend),
        iters=args.iters,
    ) * 1e3
    print(
        f"backend={args.backend} B={args.B} S={args.S} H={args.H} D={args.D} "
        f"offset={args.position_offset} dtype={args.dtype} ms={ms:.3f}"
    )


if __name__ == "__main__":
    main()
