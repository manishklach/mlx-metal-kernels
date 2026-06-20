import argparse
import time

import mlx.core as mx

from benchmark_utils import time_fn
from ops.paged_kv_ops import allocate_paged_kv_cache
from ops.toy_transformer_ops import (
    make_toy_layer_weights,
    paged_toy_transformer_decode_layer,
    reference_paged_toy_transformer_decode_layer,
    reference_toy_transformer_decode_layer,
    toy_transformer_decode_layer,
)


def _resolve_backends(backend_preset, matvec_backend, block_backend, norm_backend, activation_backend, residual_backend):
    if backend_preset is None:
        return matvec_backend, block_backend, norm_backend, activation_backend, residual_backend
    mapping = {
        "reference": ("reference", "reference", "reference", "reference", "reference"),
        "metal": ("metal", "metal", "metal", "metal", "metal"),
        "parallel": ("metal_parallel", "metal", "metal", "metal", "metal"),
        "tiled": ("metal_tiled", "metal", "metal", "metal", "metal"),
    }
    return mapping[backend_preset]


def _tol(dtype):
    if dtype == mx.bfloat16:
        return 8e-2, 8e-2
    return 6e-2, 6e-2


def _validate(cache, x_seq, ref_seq, dtype):
    atol, rtol = _tol(dtype)
    for got, ref in zip(x_seq, ref_seq):
        mx.eval(got, ref)
        if not mx.allclose(got, ref, atol=atol, rtol=rtol).item():
            raise AssertionError("Toy transformer decode output validation failed")
    if cache[0] is not None:
        for got, ref in zip(cache[0], cache[1]):
            mx.eval(got, ref)
            if not mx.allclose(got, ref, atol=atol, rtol=rtol).item():
                raise AssertionError("Toy transformer cache validation failed")


def run_loop(args, dtype, weights, backends):
    matvec_backend, block_backend, norm_backend, activation_backend, residual_backend = backends
    cos = mx.random.normal((args.MAX_S + 4, args.D // 2)).astype(mx.float32)
    sin = mx.random.normal((args.MAX_S + 4, args.D // 2)).astype(mx.float32)

    if args.cache == "contiguous":
        def _run_one(reference: bool):
            K_cache = mx.zeros((args.B, args.MAX_S, args.H, args.D), dtype=dtype)
            V_cache = mx.zeros((args.B, args.MAX_S, args.H, args.D), dtype=dtype)
            outs = []
            for pos in range(args.T):
                x = mx.random.normal((args.B, 1, args.K)).astype(dtype)
                common = dict(
                    x=x,
                    attn_norm_weight=weights["attn_norm_weight"].astype(dtype),
                    ffn_norm_weight=weights["ffn_norm_weight"].astype(dtype),
                    qkv_w=weights["qkv_w"],
                    qkv_scales=weights["qkv_scales"],
                    out_w=weights["out_w"],
                    out_scales=weights["out_scales"],
                    gate_w=weights["gate_w"],
                    gate_scales=weights["gate_scales"],
                    up_w=weights["up_w"],
                    up_scales=weights["up_scales"],
                    down_w=weights["down_w"],
                    down_scales=weights["down_scales"],
                    K_cache=K_cache,
                    V_cache=V_cache,
                    cos=cos,
                    sin=sin,
                    position=pos,
                    bits=args.bits,
                    group_size=args.group_size,
                    H=args.H,
                    D=args.D,
                )
                if reference:
                    out, K_cache, V_cache = reference_toy_transformer_decode_layer(**common)
                else:
                    out, K_cache, V_cache = toy_transformer_decode_layer(
                        **common,
                        matvec_backend=matvec_backend,
                        block_backend=block_backend,
                        norm_backend=norm_backend,
                        activation_backend=activation_backend,
                        residual_backend=residual_backend,
                    )
                outs.append(out)
            return outs, (K_cache, V_cache)
    else:
        def _run_one(reference: bool):
            K_pages, V_pages, block_table = allocate_paged_kv_cache(args.B, args.MAX_S, args.H, args.D, args.PAGE_SIZE, dtype)
            outs = []
            for pos in range(args.T):
                x = mx.random.normal((args.B, 1, args.K)).astype(dtype)
                common = dict(
                    x=x,
                    attn_norm_weight=weights["attn_norm_weight"].astype(dtype),
                    ffn_norm_weight=weights["ffn_norm_weight"].astype(dtype),
                    qkv_w=weights["qkv_w"],
                    qkv_scales=weights["qkv_scales"],
                    out_w=weights["out_w"],
                    out_scales=weights["out_scales"],
                    gate_w=weights["gate_w"],
                    gate_scales=weights["gate_scales"],
                    up_w=weights["up_w"],
                    up_scales=weights["up_scales"],
                    down_w=weights["down_w"],
                    down_scales=weights["down_scales"],
                    K_pages=K_pages,
                    V_pages=V_pages,
                    block_table=block_table,
                    cos=cos,
                    sin=sin,
                    position=pos,
                    bits=args.bits,
                    group_size=args.group_size,
                    H=args.H,
                    D=args.D,
                )
                if reference:
                    out, K_pages, V_pages = reference_paged_toy_transformer_decode_layer(**common)
                else:
                    out, K_pages, V_pages = paged_toy_transformer_decode_layer(
                        **common,
                        matvec_backend=matvec_backend,
                        block_backend=block_backend,
                        norm_backend=norm_backend,
                        activation_backend=activation_backend,
                        residual_backend=residual_backend,
                    )
                outs.append(out)
            return outs, (K_pages, V_pages)

    if not args.skip_validate:
        ref_outs, ref_cache = _run_one(True)
        got_outs, got_cache = _run_one(False)
        _validate((got_cache, ref_cache), got_outs, ref_outs, dtype)

    return time_fn(lambda: _run_one(False), warmup=3, iters=args.iters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", choices=["contiguous", "paged"], default="contiguous")
    parser.add_argument("--bits", choices=[4, 8], type=int, default=4)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--INTERMEDIATE", type=int, default=1024)
    parser.add_argument("--MAX_S", type=int, default=128)
    parser.add_argument("--PAGE_SIZE", type=int, default=16)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--matvec-backend", choices=["reference", "metal", "metal_parallel", "metal_tiled"], default="metal_parallel")
    parser.add_argument("--block-backend", choices=["reference", "metal", "metal_threadgroup", "auto"], default="metal")
    parser.add_argument("--norm-backend", choices=["reference", "metal"], default="metal")
    parser.add_argument("--activation-backend", choices=["reference", "metal"], default="metal")
    parser.add_argument("--residual-backend", choices=["reference", "metal"], default="metal")
    parser.add_argument("--backend-preset", choices=["reference", "metal", "parallel", "tiled"], default="parallel")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(215)
    weights = make_toy_layer_weights(args.K, args.INTERMEDIATE, bits=args.bits, group_size=args.group_size)
    backends = _resolve_backends(
        args.backend_preset,
        args.matvec_backend,
        args.block_backend,
        args.norm_backend,
        args.activation_backend,
        args.residual_backend,
    )
    timing = run_loop(args, dtype, weights, backends)
    mean_ms = timing["mean_ms"]
    ms_per_step = mean_ms / args.T
    tokens_per_second = (args.B * args.T * 1000.0) / mean_ms
    print(
        f"cache={args.cache} bits={args.bits} B={args.B} K={args.K} H={args.H} D={args.D} "
        f"INTERMEDIATE={args.INTERMEDIATE} MAX_S={args.MAX_S} PAGE_SIZE={args.PAGE_SIZE} T={args.T} "
        f"group_size={args.group_size} dtype={args.dtype} matvec_backend={backends[0]} block_backend={backends[1]} "
        f"norm_backend={backends[2]} activation_backend={backends[3]} residual_backend={backends[4]} "
        f"mean_ms={mean_ms:.3f} ms_per_step={ms_per_step:.3f} tokens_per_second={tokens_per_second:.3f}"
    )


if __name__ == "__main__":
    main()
