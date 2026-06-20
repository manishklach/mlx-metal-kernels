import argparse
import time

import mlx.core as mx

from benchmark_utils import summarize_times, sync
from ops.mlp_block_ops import quantized_mlp_block
from ops.quant_ops import pack_q4


def _groups(k, group_size):
    return (k + group_size - 1) // group_size


def _make_quantized_weights(bits, out_dim, in_dim, group_size):
    groups = _groups(in_dim, group_size)
    scales = mx.random.normal((out_dim, groups)).astype(mx.float32)
    if bits == 4:
        q = (mx.random.uniform((out_dim, in_dim)) * 16).astype(mx.uint8)
        return pack_q4(q), scales
    q = (mx.random.uniform((out_dim, in_dim)) * 255).astype(mx.uint8)
    return q, scales


def _backends_for_preset(preset):
    mapping = {
        "reference": ("reference", "reference", "reference", "reference"),
        "metal": ("metal", "metal", "metal", "metal"),
        "parallel": ("metal", "metal_parallel", "metal", "metal"),
        "tiled": ("metal", "metal_tiled", "metal", "metal"),
    }
    if preset == "all":
        return ["reference", "metal", "parallel", "tiled"]
    if preset not in mapping:
        raise ValueError(f"Unsupported backend preset: {preset}")
    return [preset]


def _preset_kwargs(preset):
    mapping = {
        "reference": ("reference", "reference", "reference", "reference"),
        "metal": ("metal", "metal", "metal", "metal"),
        "parallel": ("metal", "metal_parallel", "metal", "metal"),
        "tiled": ("metal", "metal_tiled", "metal", "metal"),
    }
    norm_backend, matvec_backend, activation_backend, residual_backend = mapping[preset]
    return {
        "norm_backend": norm_backend,
        "matvec_backend": matvec_backend,
        "activation_backend": activation_backend,
        "residual_backend": residual_backend,
    }


def _time_preset(fn, warmup, iters):
    for _ in range(warmup):
        sync(fn())
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sync(fn())
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e3)
    return summarize_times(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", choices=[4, 8], type=int, default=4)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=11008)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--backend-preset", choices=["reference", "metal", "parallel", "tiled", "all"], default="all")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(args.seed)

    x = mx.random.normal((args.B, args.S, args.hidden_size)).astype(dtype)
    residual = mx.random.normal((args.B, args.S, args.hidden_size)).astype(dtype)
    norm_weight = mx.random.normal((args.hidden_size,)).astype(dtype)
    gate_w, gate_scales = _make_quantized_weights(args.bits, args.intermediate_size, args.hidden_size, args.group_size)
    up_w, up_scales = _make_quantized_weights(args.bits, args.intermediate_size, args.hidden_size, args.group_size)
    down_w, down_scales = _make_quantized_weights(args.bits, args.hidden_size, args.intermediate_size, args.group_size)

    results = {}
    for preset in _backends_for_preset(args.backend_preset):
        kwargs = _preset_kwargs(preset)
        timing = _time_preset(
            lambda p=kwargs: quantized_mlp_block(
                x,
                residual,
                norm_weight,
                gate_w,
                gate_scales,
                up_w,
                up_scales,
                down_w,
                down_scales,
                bits=args.bits,
                group_size=args.group_size,
                **p,
            ),
            warmup=3,
            iters=args.iters,
        )
        results[preset] = timing

    reference_ms = results.get("reference", {}).get("mean_ms")
    metal_ms = results.get("metal", {}).get("mean_ms")
    rows = args.B * args.S
    for preset, timing in results.items():
        mean_ms = timing["mean_ms"]
        rows_per_second = rows / (mean_ms / 1e3)
        line = (
            f"bits={args.bits} B={args.B} S={args.S} hidden_size={args.hidden_size} "
            f"intermediate_size={args.intermediate_size} group_size={args.group_size} "
            f"dtype={args.dtype} backend_preset={preset} mean_ms={mean_ms:.3f} rows_per_second={rows_per_second:.3f}"
        )
        if reference_ms is not None and preset != "reference":
            line += f" speedup_vs_reference={reference_ms / mean_ms:.3f}"
        if metal_ms is not None and preset not in ("reference", "metal"):
            line += f" speedup_vs_metal={metal_ms / mean_ms:.3f}"
        print(line)


if __name__ == "__main__":
    main()
