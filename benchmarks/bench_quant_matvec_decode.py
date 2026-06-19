import argparse
import time

import mlx.core as mx

from ops.quant_ops import (
    pack_q4,
    q4_matvec_decode,
    q8_matvec_decode,
    reference_q4_matvec_decode,
    reference_q8_matvec_decode,
)


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
    parser.add_argument("--bits", type=int, choices=[4, 8], required=True)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--backend", choices=["reference", "metal", "all"], default="all")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(71)
    x = mx.random.normal((args.B, args.K)).astype(dtype)
    scales = mx.random.normal((args.N, args.K // args.group_size)).astype(mx.float32)
    if args.bits == 4:
        q = (mx.random.uniform((args.N, args.K)) * 16).astype(mx.uint8)
        w = pack_q4(q)
        ref_fn = lambda: reference_q4_matvec_decode(x, w, scales, group_size=args.group_size)
        metal_fn = lambda: q4_matvec_decode(x, w, scales, group_size=args.group_size, backend="metal")
    else:
        w = (mx.random.uniform((args.N, args.K)) * 255).astype(mx.uint8)
        ref_fn = lambda: reference_q8_matvec_decode(x, w, scales, group_size=args.group_size)
        metal_fn = lambda: q8_matvec_decode(x, w, scales, group_size=args.group_size, backend="metal")

    ref_ms = time_fn(ref_fn, iters=args.iters) * 1e3
    backends = ["reference", "metal"] if args.backend == "all" else [args.backend]
    for backend in backends:
        cur_ms = ref_ms if backend == "reference" else time_fn(metal_fn, iters=args.iters) * 1e3
        speedup = ref_ms / cur_ms if cur_ms > 0 else float("inf")
        print(f"backend={backend} bits={args.bits} B={args.B} K={args.K} N={args.N} group_size={args.group_size} dtype={args.dtype} milliseconds={cur_ms:.3f} speedup_vs_reference={speedup:.2f}x")


if __name__ == "__main__":
    main()
