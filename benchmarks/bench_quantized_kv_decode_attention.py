from __future__ import annotations

import argparse
import math
import time

import mlx.core as mx

from benchmark_utils import summarize_times, sync
from ops.decode_ops import reference_decode_attention
from ops.gqa_ops import expand_kv_heads_reference, reference_gqa_decode_attention
from ops.quantized_kv_cache_ops import (
    QuantizedKVCacheConfig,
    quantize_kv_cache,
    quantized_kv_gqa_decode_attention,
    reference_quantized_kv_gqa_decode_attention,
)


def _time_fn(fn, warmup=3, iters=10):
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
    parser = argparse.ArgumentParser(description="Benchmark quantized KV-cache decode attention")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--MAX_S", type=int, default=128)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--Hq", type=int, default=32)
    parser.add_argument("--Hkv", type=int, default=8)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--backend", choices=["reference", "metal_q8", "metal_q4", "all"], default="all")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--compare-fp16", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--sink-tokens", type=int, default=4)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(args.seed)

    q = mx.random.normal((args.B, 1, args.Hq, args.D)).astype(dtype)
    K_cache = mx.random.normal((args.B, args.MAX_S, args.Hkv, args.D)).astype(dtype)
    V_cache = mx.random.normal((args.B, args.MAX_S, args.Hkv, args.D)).astype(dtype)

    fp16_kv_bytes = 2 * args.B * args.MAX_S * args.Hkv * args.D * 2  # K+V, 2 bytes per value

    backends = ["reference", "metal_q8", "metal_q4"] if args.backend == "all" else [args.backend]

    # Quantize to both q8 and q4
    qkv_8 = quantize_kv_cache(K_cache, V_cache, QuantizedKVCacheConfig(bits=8, group_size=args.group_size))
    qkv_4 = quantize_kv_cache(K_cache, V_cache, QuantizedKVCacheConfig(bits=4, group_size=args.group_size))

    quant_configs = {
        "metal_q8": qkv_8,
        "metal_q4": qkv_4,
    }

    def _get_qkv(backend):
        if backend == "metal_q8":
            return quant_configs["metal_q8"]
        if backend == "metal_q4":
            return quant_configs["metal_q4"]
        return None

    # FP16 dense decode timing
    fp16_timing = _time_fn(lambda: reference_gqa_decode_attention(q, K_cache, V_cache, lengths=args.length), iters=args.iters)
    print(f"bits=fp16 B={args.B} MAX_S={args.MAX_S} length={args.length} Hq={args.Hq} Hkv={args.Hkv} D={args.D} "
          f"backend=fp16_dense mean_ms={fp16_timing['mean_ms']:.3f}")

    for backend in backends:
        if backend == "reference":
            fn = lambda: reference_quantized_kv_gqa_decode_attention(q, qkv_8, lengths=args.length)
            label = "reference"
            qkv_bytes = qkv_8.memory_bytes()
        elif backend in quant_configs:
            qkv = quant_configs[backend]
            fn = lambda b=backend: quantized_kv_gqa_decode_attention(q, quant_configs[b], lengths=args.length, backend=b)
            label = backend
            qkv_bytes = qkv.memory_bytes()
        else:
            continue

        try:
            timing = _time_fn(fn, iters=args.iters)
            cr = qkv_8.compression_ratio(2) if "q8" in label else qkv_4.compression_ratio(2)
            speedup = fp16_timing["mean_ms"] / timing["mean_ms"] if timing["mean_ms"] > 0 else 0.0
            print(f"bits={args.bits} B={args.B} MAX_S={args.MAX_S} length={args.length} Hq={args.Hq} Hkv={args.Hkv} D={args.D} "
                  f"group_size={args.group_size} backend={label} mean_ms={timing['mean_ms']:.3f} "
                  f"fp16_kv_bytes={fp16_kv_bytes} quantized_kv_bytes={qkv_bytes} "
                  f"compression_ratio={cr:.2f} speedup_vs_fp16_dense={speedup:.3f}")

            if args.validate and backend in quant_configs:
                ref = reference_quantized_kv_gqa_decode_attention(q, quant_configs[backend], lengths=args.length)
                got = quantized_kv_gqa_decode_attention(q, quant_configs[backend], lengths=args.length, backend=backend)
                mx.eval(ref, got)
                max_diff = mx.max(mx.abs(got - ref)).item()
                print(f"  validation: max_diff={max_diff:.6f}")

            if args.compare_fp16:
                ref_fp16 = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=args.length)
                got = quantized_kv_gqa_decode_attention(q, quant_configs.get(backend, qkv_8), lengths=args.length, backend=backend if backend != "reference" else "reference")
                mx.eval(ref_fp16, got)
                err = mx.max(mx.abs(got - ref_fp16)).item()
                print(f"  error_vs_fp16={err:.6f}")

        except Exception as e:
            print(f"  status=skipped reason={e}")


if __name__ == "__main__":
    main()
