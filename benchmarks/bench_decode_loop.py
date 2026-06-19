import argparse
import time

import mlx.core as mx

from ops.decode_ops import decode_step


def time_fn(fn, warmup=3, iters=10):
    for _ in range(warmup):
        y = fn()
        mx.eval(*y)
    start = time.perf_counter()
    for _ in range(iters):
        y = fn()
        mx.eval(*y)
    end = time.perf_counter()
    return (end - start) / iters


def run_loop(B, MAX_S, T, H, D, dtype, backend):
    K_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
    V_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
    out = None
    for pos in range(T):
        q = mx.random.normal((B, 1, H, D)).astype(dtype)
        k_new = mx.random.normal((B, 1, H, D)).astype(dtype)
        v_new = mx.random.normal((B, 1, H, D)).astype(dtype)
        out, K_cache, V_cache = decode_step(q, k_new, v_new, K_cache, V_cache, pos, backend=backend)
    return out, K_cache, V_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--MAX_S", type=int, default=64)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--backend", choices=["reference", "metal", "auto"], default="auto")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(22)
    avg_s = time_fn(lambda: run_loop(args.B, args.MAX_S, args.T, args.H, args.D, dtype, args.backend), iters=args.iters)
    ms_per_step = avg_s * 1e3 / args.T
    tokens_per_second = (args.B * args.T) / avg_s
    print(
        f"backend={args.backend} B={args.B} MAX_S={args.MAX_S} T={args.T} "
        f"H={args.H} D={args.D} dtype={args.dtype} ms_per_step={ms_per_step:.3f} "
        f"tokens_per_second={tokens_per_second:.3f}"
    )


if __name__ == "__main__":
    main()
