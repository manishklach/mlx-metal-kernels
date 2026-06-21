from __future__ import annotations

import argparse
import math
import time

import mlx.core as mx

from benchmark_utils import summarize_times, sync
from ops.gqa_ops import reference_gqa_decode_attention
from ops.paged_kv_ops import allocate_paged_kv_cache, reference_paged_decode_attention
from ops.paged_quantized_kv_ops import (
    PagedQuantizedKVConfig,
    contiguous_kv_to_pages,
    paged_quantized_kv_gqa_decode_attention,
    reference_paged_quantized_kv_gqa_decode_attention,
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
    parser = argparse.ArgumentParser(description="Benchmark paged quantized KV-cache decode attention")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--MAX_S", type=int, default=128)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--PAGE_SIZE", type=int, default=16)
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
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    mx.random.seed(args.seed)

    B, MAX_S, PAGE_SIZE, Hq, Hkv, D = args.B, args.MAX_S, args.PAGE_SIZE, args.Hq, args.Hkv, args.D
    length = min(args.length, MAX_S)
    max_blocks = (MAX_S + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = B * max_blocks

    q = mx.random.normal((B, 1, Hq, D)).astype(dtype)
    K_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(dtype)
    V_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(dtype)

    K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, Hkv, D, PAGE_SIZE, dtype)
    K_pages = mx.random.normal(K_pages.shape).astype(dtype)
    V_pages = mx.random.normal(V_pages.shape).astype(dtype)

    lengths = [length] * B
    K_pages_q, V_pages_q, bt_q, lengths_arr = contiguous_kv_to_pages(K_cache, V_cache, lengths, page_size=PAGE_SIZE)

    cfg = PagedQuantizedKVConfig(bits=args.bits, page_size=PAGE_SIZE, group_size=args.group_size)
    from ops.paged_quantized_kv_ops import quantize_kv_pages
    pqv = quantize_kv_pages(K_pages_q, V_pages_q, bt_q, lengths_arr, cfg)

    fp16_paged_kv_bytes = 2 * num_pages * PAGE_SIZE * Hkv * D * 2
    quantized_bytes = pqv.memory_bytes()
    cr = pqv.compression_ratio(2)

    backends = ["reference", "metal_q8", "metal_q4"] if args.backend == "all" else [args.backend]

    # FP16 paged decode timing
    fp16_timing = _time_fn(
        lambda: reference_paged_decode_attention(q, K_pages, V_pages, block_table, lengths, scale=1.0 / math.sqrt(D)),
        iters=args.iters,
    )
    print(f"backend=fp16_paged B={B} MAX_S={MAX_S} length={length} Hq={Hq} Hkv={Hkv} D={D} "
          f"PAGE_SIZE={PAGE_SIZE} mean_ms={fp16_timing['mean_ms']:.3f}")

    for backend in backends:
        try:
            if backend == "reference":
                fn = lambda: reference_paged_quantized_kv_gqa_decode_attention(q, pqv, scale=1.0 / math.sqrt(D))
            else:
                fn = lambda b=backend: paged_quantized_kv_gqa_decode_attention(
                    q, pqv, scale=1.0 / math.sqrt(D), backend=b,
                )
            timing = _time_fn(fn, iters=args.iters)
            speedup = fp16_timing["mean_ms"] / timing["mean_ms"] if timing["mean_ms"] > 0 else 0.0
            print(f"bits={args.bits} B={B} MAX_S={MAX_S} length={length} Hq={Hq} Hkv={Hkv} D={D} "
                  f"PAGE_SIZE={PAGE_SIZE} group_size={args.group_size} backend={backend} "
                  f"mean_ms={timing['mean_ms']:.3f} "
                  f"fp16_paged_kv_bytes={fp16_paged_kv_bytes} "
                  f"quantized_paged_kv_bytes={quantized_bytes} "
                  f"compression_ratio={cr:.2f} "
                  f"speedup_vs_fp16_paged={speedup:.3f}")

            if args.validate:
                ref = reference_paged_quantized_kv_gqa_decode_attention(q, pqv, scale=1.0 / math.sqrt(D))
                if backend == "reference":
                    got = ref
                else:
                    got = paged_quantized_kv_gqa_decode_attention(q, pqv, scale=1.0 / math.sqrt(D), backend=backend)
                mx.eval(ref, got)
                max_diff = float(mx.max(mx.abs(got - ref)).item())
                print(f"  validation: max_diff={max_diff:.6f}")

            if args.compare_fp16:
                ref_fp16 = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=lengths, scale=1.0 / math.sqrt(D))
                got = paged_quantized_kv_gqa_decode_attention(q, pqv, scale=1.0 / math.sqrt(D),
                                                              backend=backend if backend != "reference" else "reference")
                mx.eval(ref_fp16, got)
                err = float(mx.max(mx.abs(got - ref_fp16)).item())
                print(f"  error_vs_fp16={err:.6f}")

        except Exception as e:
            print(f"  status=skipped reason={e}")


if __name__ == "__main__":
    main()
