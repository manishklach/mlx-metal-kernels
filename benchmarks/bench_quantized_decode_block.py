import argparse
import time

import mlx.core as mx

from ops.paged_kv_ops import allocate_paged_kv_cache
from ops.quant_ops import pack_q4
from ops.quantized_decode_block_ops import paged_quantized_decode_block, quantized_decode_block


def _groups(K, group_size):
    return (K + group_size - 1) // group_size


def _make_quantized_weights(bits, out_dim, in_dim, group_size):
    groups = _groups(in_dim, group_size)
    scales = mx.random.normal((out_dim, groups)).astype(mx.float32)
    if bits == 4:
        q = (mx.random.uniform((out_dim, in_dim)) * 16).astype(mx.uint8)
        return pack_q4(q), scales
    q = (mx.random.uniform((out_dim, in_dim)) * 255).astype(mx.uint8)
    return q, scales


def _resolve_backends(backend_preset, matvec_backend, block_backend):
    if backend_preset is None:
        return matvec_backend, block_backend
    mapping = {
        "reference": ("reference", "reference"),
        "metal": ("metal", "metal"),
        "parallel": ("metal_parallel", "metal"),
    }
    return mapping[backend_preset]


def run_loop(bits, B, K, H, D, MAX_S, PAGE_SIZE, T, group_size, dtype, cache, matvec_backend, block_backend):
    qkv_w, qkv_scales = _make_quantized_weights(bits, 3 * H * D, K, group_size)
    out_w, out_scales = _make_quantized_weights(bits, K, H * D, group_size)
    cos = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    sin = mx.random.normal((MAX_S + 4, D // 2)).astype(mx.float32)
    out = None

    if cache == "contiguous":
        K_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
        V_cache = mx.zeros((B, MAX_S, H, D), dtype=dtype)
        for pos in range(T):
            x = mx.random.normal((B, 1, K)).astype(dtype)
            out, K_cache, V_cache = quantized_decode_block(
                x,
                qkv_w,
                qkv_scales,
                out_w,
                out_scales,
                K_cache,
                V_cache,
                cos,
                sin,
                pos,
                bits=bits,
                group_size=group_size,
                H=H,
                D=D,
                matvec_backend=matvec_backend,
                block_backend=block_backend,
            )
        return out, K_cache, V_cache

    K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, H, D, PAGE_SIZE, dtype)
    for pos in range(T):
        x = mx.random.normal((B, 1, K)).astype(dtype)
        out, K_pages, V_pages = paged_quantized_decode_block(
            x,
            qkv_w,
            qkv_scales,
            out_w,
            out_scales,
            K_pages,
            V_pages,
            block_table,
            cos,
            sin,
            pos,
            bits=bits,
            group_size=group_size,
            H=H,
            D=D,
            matvec_backend=matvec_backend,
            block_backend=block_backend,
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
    parser.add_argument("--bits", choices=[4, 8], type=int, default=4)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--MAX_S", type=int, default=128)
    parser.add_argument("--PAGE_SIZE", type=int, default=16)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--cache", choices=["contiguous", "paged"], default="contiguous")
    parser.add_argument("--matvec-backend", choices=["reference", "metal", "metal_parallel"], default="metal_parallel")
    parser.add_argument("--block-backend", choices=["reference", "metal", "auto"], default="metal")
    parser.add_argument("--backend-preset", choices=["reference", "metal", "parallel"], default=None)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(205)
    matvec_backend, block_backend = _resolve_backends(args.backend_preset, args.matvec_backend, args.block_backend)
    avg_s = time_fn(
        lambda: run_loop(
            args.bits,
            args.B,
            args.K,
            args.H,
            args.D,
            args.MAX_S,
            args.PAGE_SIZE,
            args.T,
            args.group_size,
            dtype,
            args.cache,
            matvec_backend,
            block_backend,
        ),
        iters=args.iters,
    )
    ms_per_step = avg_s * 1e3 / args.T
    tokens_per_second = (args.B * args.T) / avg_s
    print(
        f"bits={args.bits} cache={args.cache} B={args.B} K={args.K} H={args.H} D={args.D} "
        f"MAX_S={args.MAX_S} PAGE_SIZE={args.PAGE_SIZE if args.cache == 'paged' else 'n/a'} T={args.T} "
        f"group_size={args.group_size} dtype={args.dtype} matvec_backend={matvec_backend} "
        f"block_backend={block_backend} ms_per_step={ms_per_step:.3f} tokens_per_second={tokens_per_second:.3f}"
    )


if __name__ == "__main__":
    main()
