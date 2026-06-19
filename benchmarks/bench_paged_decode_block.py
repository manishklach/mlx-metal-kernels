import argparse
import time

import mlx.core as mx

from ops.decode_block_ops import paged_decode_block_from_qkv
from ops.paged_kv_ops import allocate_paged_kv_cache


def _make_qkv(B, H, D, dtype, layout):
    if layout == "packed":
        return mx.random.normal((B, 1, 3 * H * D)).astype(dtype)
    return mx.random.normal((B, 1, 3, H, D)).astype(dtype)


def run_loop(B, MAX_S, PAGE_SIZE, T, H, D, dtype, layout, backend):
    K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, PAGE_SIZE, dtype)
    cos = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    sin = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    out = None
    for pos in range(T):
        qkv = _make_qkv(B, H, D, dtype, layout)
        out, K_pages, V_pages = paged_decode_block_from_qkv(
            qkv, K_pages, V_pages, block_table, cos, sin, pos, H=H, D=D, backend=backend
        )
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--MAX_S", type=int, default=128)
    parser.add_argument("--PAGE_SIZE", type=int, default=16)
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--layout", choices=["packed", "explicit"], default="packed")
    parser.add_argument("--backend", choices=["reference", "metal", "all"], default="all")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(95)
    backends = ["reference", "metal"] if args.backend == "all" else [args.backend]
    timings = {}
    for backend in backends:
        avg_s = time_fn(
            lambda: run_loop(args.B, args.MAX_S, args.PAGE_SIZE, args.T, args.H, args.D, dtype, args.layout, backend),
            iters=args.iters,
        )
        timings[backend] = avg_s
        ms_per_step = avg_s * 1e3 / args.T
        tps = (args.B * args.T) / avg_s
        speedup = ""
        if args.backend == "all" and backend != "reference" and "reference" in timings:
            speedup = f" speedup_vs_reference={timings['reference'] / avg_s:.3f}"
        print(
            f"backend={backend} B={args.B} MAX_S={args.MAX_S} PAGE_SIZE={args.PAGE_SIZE} T={args.T} "
            f"H={args.H} D={args.D} dtype={args.dtype} layout={args.layout} "
            f"ms_per_step={ms_per_step:.3f} tokens_per_second={tps:.3f}{speedup}"
        )


if __name__ == "__main__":
    main()
