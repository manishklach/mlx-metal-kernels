from __future__ import annotations

import math

import mlx.core as mx

from ops.gqa_ops import reference_gqa_decode_attention
from ops.paged_quantized_kv_ops import (
    PagedQuantizedKVConfig,
    contiguous_kv_to_pages,
    dequantize_kv_pages,
    paged_quantized_kv_gqa_decode_attention,
    quantize_kv_pages,
    reference_paged_quantized_kv_gqa_decode_attention,
)


def main():
    mx.random.seed(42)
    print("=" * 60)
    print("Paged quantized KV-cache demo (experimental)")
    print("=" * 60)

    B, MAX_S, Hq, Hkv, D = 1, 32, 4, 2, 64
    PAGE_SIZE = 8
    length = 32

    q = mx.random.normal((B, 1, Hq, D)).astype(mx.float16)
    K_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)
    V_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)

    K_pages, V_pages, block_table, lengths_arr = contiguous_kv_to_pages(
        K_cache, V_cache, [length], page_size=PAGE_SIZE,
    )
    num_pages = K_pages.shape[0]
    max_blocks = block_table.shape[1]

    print(f"\nCache setup:")
    print(f"  B={B}, MAX_S={MAX_S}, Hq={Hq}, Hkv={Hkv}, D={D}")
    print(f"  PAGE_SIZE={PAGE_SIZE}, num_pages={num_pages}, max_blocks={max_blocks}")
    print(f"  fp16 paged KV bytes: {2 * num_pages * PAGE_SIZE * Hkv * D * 2}")

    # Q8
    cfg_q8 = PagedQuantizedKVConfig(bits=8, page_size=PAGE_SIZE, group_size=32)
    pqv_q8 = quantize_kv_pages(K_pages, V_pages, block_table, lengths_arr, cfg_q8)
    q8_bytes = pqv_q8.memory_bytes()
    q8_cr = pqv_q8.compression_ratio(2)

    # Q4
    cfg_q4 = PagedQuantizedKVConfig(bits=4, page_size=PAGE_SIZE, group_size=32)
    pqv_q4 = quantize_kv_pages(K_pages, V_pages, block_table, lengths_arr, cfg_q4)
    q4_bytes = pqv_q4.memory_bytes()
    q4_cr = pqv_q4.compression_ratio(2)

    print(f"\nMemory:")
    print(f"  q8 paged KV bytes: {q8_bytes}, ratio: {q8_cr:.2f}x")
    print(f"  q4 paged KV bytes: {q4_bytes}, ratio: {q4_cr:.2f}x")

    # Decode
    scale = 1.0 / math.sqrt(D)
    fp16_out = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=length, scale=scale)
    ref_q8_out = reference_paged_quantized_kv_gqa_decode_attention(q, pqv_q8, scale=scale)
    ref_q4_out = reference_paged_quantized_kv_gqa_decode_attention(q, pqv_q4, scale=scale)

    err_q8 = float(mx.max(mx.abs(ref_q8_out - fp16_out)).item())
    err_q4 = float(mx.max(mx.abs(ref_q4_out - fp16_out)).item())

    print(f"\nOutput error vs fp16:")
    print(f"  q8 vs fp16 paged: max_abs={err_q8:.6f}")
    print(f"  q4 vs fp16 paged: max_abs={err_q4:.6f}")

    # Try Metal if available
    for bits, pqv, label in [(8, pqv_q8, "metal_q8"), (4, pqv_q4, "metal_q4")]:
        try:
            metal_out = paged_quantized_kv_gqa_decode_attention(q, pqv, scale=scale, backend=label)
            ref_out = reference_paged_quantized_kv_gqa_decode_attention(q, pqv, scale=scale)
            mx.eval(metal_out, ref_out)
            diff = float(mx.max(mx.abs(metal_out - ref_out)).item())
            print(f"  {label} vs q{bits} reference: max_abs={diff:.6f}")
        except Exception as e:
            print(f"  {label}: skipped ({e})")

    print("\n" + "=" * 60)
    print("Warning: Synthetic/random tensors. Experimental paged quantized KV-cache demo.")
    print("=" * 60)


if __name__ == "__main__":
    main()
