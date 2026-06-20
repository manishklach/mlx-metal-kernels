import argparse
import time

import mlx.core as mx

from ops.quant_ops import (
    pack_q4,
    q4_matvec_decode,
    q8_matvec_decode,
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
    parser.add_argument("--zeros", action="store_true")
    parser.add_argument("--backend", choices=["reference", "metal", "metal_parallel", "all"], default="all")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(123)
    x = mx.random.normal((args.B, args.K)).astype(dtype)
    groups = (args.K + args.group_size - 1) // args.group_size
    scales = mx.random.normal((args.N, groups)).astype(mx.float32)
    zeros = (mx.random.uniform((args.N, groups)) * 8).astype(mx.float32) if args.zeros else None

    if args.bits == 4:
        q = (mx.random.uniform((args.N, args.K)) * 16).astype(mx.uint8)
        w = pack_q4(q)

        def run_backend(name):
            return q4_matvec_decode(x, w, scales, zeros, group_size=args.group_size, backend=name)
    else:
        w = (mx.random.uniform((args.N, args.K)) * 255).astype(mx.uint8)

        def run_backend(name):
            return q8_matvec_decode(x, w, scales, zeros, group_size=args.group_size, backend=name)

    backends = ["reference", "metal", "metal_parallel"] if args.backend == "all" else [args.backend]
    timings = {}
    for backend in backends:
        timings[backend] = time_fn(lambda b=backend: run_backend(b), iters=args.iters) * 1e3
        speed_ref = timings["reference"] / timings[backend] if backend != "reference" and "reference" in timings else 1.0
        speed_metal = timings["metal"] / timings[backend] if backend != "metal" and "metal" in timings else 1.0
        print(
            f"backend={backend} bits={args.bits} B={args.B} K={args.K} N={args.N} "
            f"group_size={args.group_size} dtype={args.dtype} zeros={args.zeros} "
            f"milliseconds={timings[backend]:.3f} speedup_vs_reference={speed_ref:.2f}x "
            f"speedup_vs_metal={speed_metal:.2f}x"
        )


if __name__ == "__main__":
    main()
