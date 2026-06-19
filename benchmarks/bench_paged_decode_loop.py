import argparse
import time

import mlx.core as mx

from ops.paged_kv_ops import allocate_paged_kv_cache, paged_decode_step


def run_loop(B, MAX_S, PAGE_SIZE, T, H, D, dtype, backend):
    K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, PAGE_SIZE, dtype)
    out = None
    for pos in range(T):
        q = mx.random.normal((B, 1, H, D)).astype(dtype)
        k_new = mx.random.normal((B, 1, H, D)).astype(dtype)
        v_new = mx.random.normal((B, 1, H, D)).astype(dtype)
        out, K_pages, V_pages = paged_decode_step(q, k_new, v_new, K_pages, V_pages, block_table, pos, backend=backend)
    return out, K_pages, V_pages


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--MAX_S", type=int, default=128)
    p.add_argument("--PAGE_SIZE", type=int, default=16)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--backend", choices=["reference", "metal", "all"], default="all")
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()
    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    backends = ["reference", "metal"] if args.backend == "all" else [args.backend]
    for backend in backends:
        avg_s = time_fn(lambda: run_loop(args.B, args.MAX_S, args.PAGE_SIZE, args.T, args.H, args.D, dtype, backend), iters=args.iters)
        ms_per_step = avg_s * 1e3 / args.T
        tps = (args.B * args.T) / avg_s
        print(f"backend={backend} B={args.B} MAX_S={args.MAX_S} PAGE_SIZE={args.PAGE_SIZE} T={args.T} H={args.H} D={args.D} dtype={args.dtype} ms_per_step={ms_per_step:.3f} tokens_per_second={tps:.3f}")


if __name__ == "__main__":
    main()
