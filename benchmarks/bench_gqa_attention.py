import argparse

import mlx.core as mx

from benchmark_utils import dtype_from_string, time_fn
from ops.gqa_ops import gqa_attention, reference_gqa_attention


def _backends(selection):
    if selection == "all":
        return ["reference", "metal_gqa", "metal_gqa_threadgroup"]
    return [selection]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=128)
    parser.add_argument("--Sq", type=int, default=None)
    parser.add_argument("--Sk", type=int, default=None)
    parser.add_argument("--Hq", type=int, default=32)
    parser.add_argument("--Hkv", type=int, default=8)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--backend", choices=["reference", "metal_gqa", "metal_gqa_threadgroup", "all"], default="all")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    dtype = dtype_from_string(args.dtype)
    mx.random.seed(args.seed)
    Sq = args.S if args.Sq is None else args.Sq
    Sk = args.S if args.Sk is None else args.Sk
    Q = mx.random.normal((args.B, Sq, args.Hq, args.D)).astype(dtype)
    K = mx.random.normal((args.B, Sk, args.Hkv, args.D)).astype(dtype)
    V = mx.random.normal((args.B, Sk, args.Hkv, args.D)).astype(dtype)

    reference_out = None
    if args.validate:
        reference_out = reference_gqa_attention(Q, K, V, causal=args.causal)
        mx.eval(reference_out)

    results = {}
    for backend in _backends(args.backend):
        if args.validate and backend != "reference":
            got = gqa_attention(Q, K, V, backend=backend, causal=args.causal)
            mx.eval(got)
            if not mx.allclose(got, reference_out, atol=3e-2, rtol=3e-2).item():
                raise AssertionError(f"{backend} failed validation against reference_gqa_attention")
        fn = (
            lambda: reference_gqa_attention(Q, K, V, causal=args.causal)
            if backend == "reference"
            else lambda: gqa_attention(Q, K, V, backend=backend, causal=args.causal)
        )
        if backend == "reference":
            results[backend] = time_fn(fn, warmup=3, iters=args.iters)
        else:
            results[backend] = time_fn(lambda b=backend: gqa_attention(Q, K, V, backend=b, causal=args.causal), warmup=3, iters=args.iters)

    reference_ms = results.get("reference", {}).get("mean_ms")
    metal_ms = results.get("metal_gqa", {}).get("mean_ms")
    for backend, timing in results.items():
        mean_ms = timing["mean_ms"]
        line = (
            f"B={args.B} Sq={Sq} Sk={Sk} Hq={args.Hq} Hkv={args.Hkv} group_size={args.Hq // args.Hkv} "
            f"D={args.D} dtype={args.dtype} causal={args.causal} backend={backend} mean_ms={mean_ms:.3f}"
        )
        if reference_ms is not None and backend != "reference":
            line += f" speedup_vs_reference={reference_ms / mean_ms:.3f}"
        if metal_ms is not None and backend == "metal_gqa_threadgroup":
            line += f" speedup_vs_metal_gqa={metal_ms / mean_ms:.3f}"
        print(line)


if __name__ == "__main__":
    main()
