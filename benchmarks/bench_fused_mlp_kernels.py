import argparse

import mlx.core as mx

from benchmark_utils import dtype_from_string, time_fn
from ops.mlp_block_ops import quantized_mlp_block, reference_quantized_mlp_block
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


def _presets(selection):
    if selection == "all":
        return ["reference", "tiled", "fused_experimental"]
    return [selection]


def _run_preset(preset, x, residual, norm_weight, gate_w, gate_scales, up_w, up_scales, down_w, down_scales, *, bits, group_size, iters):
    if preset == "reference":
        fn = lambda: reference_quantized_mlp_block(  # noqa: E731
            x,
            residual,
            norm_weight,
            gate_w,
            gate_scales,
            up_w,
            up_scales,
            down_w,
            down_scales,
            bits=bits,
            group_size=group_size,
        )
    else:
        fn = lambda: quantized_mlp_block(  # noqa: E731
            x,
            residual,
            norm_weight,
            gate_w,
            gate_scales,
            up_w,
            up_scales,
            down_w,
            down_scales,
            bits=bits,
            group_size=group_size,
            backend_preset=preset,
        )
    return time_fn(fn, warmup=3, iters=iters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", choices=[4, 8], type=int, default=4)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=11008)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--backend-preset", choices=["reference", "tiled", "fused_experimental", "all"], default="all")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    dtype = dtype_from_string(args.dtype)
    mx.random.seed(args.seed)

    x = mx.random.normal((args.B, args.S, args.hidden_size)).astype(dtype)
    residual = mx.random.normal((args.B, args.S, args.hidden_size)).astype(dtype)
    norm_weight = mx.random.normal((args.hidden_size,)).astype(dtype)
    gate_w, gate_scales = _make_quantized_weights(args.bits, args.intermediate_size, args.hidden_size, args.group_size)
    up_w, up_scales = _make_quantized_weights(args.bits, args.intermediate_size, args.hidden_size, args.group_size)
    down_w, down_scales = _make_quantized_weights(args.bits, args.hidden_size, args.intermediate_size, args.group_size)

    reference_out = None
    if args.validate:
        reference_out = reference_quantized_mlp_block(
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
        )
        mx.eval(reference_out)

    results = {}
    for preset in _presets(args.backend_preset):
        if args.validate and preset != "reference":
            got = quantized_mlp_block(
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
                backend_preset=preset,
            )
            mx.eval(got)
            if not mx.allclose(got, reference_out, atol=1e-1, rtol=1e-1).item():
                raise AssertionError(f"{preset} failed validation against reference_quantized_mlp_block")
        results[preset] = _run_preset(
            preset,
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
            iters=args.iters,
        )

    reference_ms = results.get("reference", {}).get("mean_ms")
    tiled_ms = results.get("tiled", {}).get("mean_ms")
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
        if tiled_ms is not None and preset in ("reference", "fused_experimental"):
            line += f" speedup_vs_tiled={tiled_ms / mean_ms:.3f}"
        print(line)


if __name__ == "__main__":
    main()
