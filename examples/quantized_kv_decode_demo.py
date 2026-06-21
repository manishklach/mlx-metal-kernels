from __future__ import annotations

import mlx.core as mx

from ops.gqa_ops import reference_gqa_decode_attention
from ops.quantized_kv_cache_ops import (
    QuantizedKVCacheConfig,
    QuantizedKVCache,
    quantize_kv_cache,
    quantized_kv_error,
    quantized_kv_gqa_decode_attention,
    reference_quantized_kv_gqa_decode_attention,
)


def main():
    print("=" * 60)
    print("Quantized KV-cache decode attention demo")
    print("=" * 60)
    print("WARNING: Synthetic/random tensors.")
    print("This is an experimental quantized KV-cache attention demo.\n")

    mx.random.seed(42)
    B, MAX_S, Hq, Hkv, D = 1, 32, 4, 2, 16
    length = 32

    q = mx.random.normal((B, 1, Hq, D)).astype(mx.float16)
    K_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)
    V_cache = mx.random.normal((B, MAX_S, Hkv, D)).astype(mx.float16)

    fp16_bytes = 2 * B * MAX_S * Hkv * D * 2

    # q8
    qkv_8 = quantize_kv_cache(K_cache, V_cache, QuantizedKVCacheConfig(bits=8, group_size=16))
    q8_bytes = qkv_8.memory_bytes()
    err_8 = quantized_kv_error(K_cache, V_cache, qkv_8)

    # q4
    qkv_4 = quantize_kv_cache(K_cache, V_cache, QuantizedKVCacheConfig(bits=4, group_size=16))
    q4_bytes = qkv_4.memory_bytes()
    err_4 = quantized_kv_error(K_cache, V_cache, qkv_4)

    print(f"KV-cache shape: [{B}, {MAX_S}, {Hkv}, {D}]")
    print(f"FP16  KV bytes: {fp16_bytes}")
    print(f"Q8    KV bytes: {q8_bytes}  (compression: {qkv_8.compression_ratio(2):.2f}x)")
    print(f"Q4    KV bytes: {q4_bytes}  (compression: {qkv_4.compression_ratio(2):.2f}x)\n")

    print("Quantization error vs FP16:")
    print(f"  Q8: K_RMSE={err_8['k_rmse']:.6f}  V_RMSE={err_8['v_rmse']:.6f}  "
          f"K_max={err_8['k_max_abs_error']:.6f}  V_max={err_8['v_max_abs_error']:.6f}")
    print(f"  Q4: K_RMSE={err_4['k_rmse']:.6f}  V_RMSE={err_4['v_rmse']:.6f}  "
          f"K_max={err_4['k_max_abs_error']:.6f}  V_max={err_4['v_max_abs_error']:.6f}\n")

    fp16_out = reference_gqa_decode_attention(q, K_cache, V_cache, lengths=length)
    q8_ref_out = reference_quantized_kv_gqa_decode_attention(q, qkv_8, lengths=length)
    q4_ref_out = reference_quantized_kv_gqa_decode_attention(q, qkv_4, lengths=length)

    mx.eval(fp16_out, q8_ref_out, q4_ref_out)

    print("Decode attention output error vs FP16 dense:")
    q8_err = mx.max(mx.abs(q8_ref_out - fp16_out)).item()
    q4_err = mx.max(mx.abs(q4_ref_out - fp16_out)).item()
    print(f"  Q8 reference vs FP16: max_diff = {q8_err:.6f}")
    print(f"  Q4 reference vs FP16: max_diff = {q4_err:.6f}")

    # Try Metal backends
    for backend, qkv, label in [
        ("metal_q8", qkv_8, "Q8 Metal"),
        ("metal_q4", qkv_4, "Q4 Metal"),
    ]:
        try:
            metal_out = quantized_kv_gqa_decode_attention(q, qkv, lengths=length, backend=backend)
            mx.eval(metal_out)
            ref = reference_quantized_kv_gqa_decode_attention(q, qkv, lengths=length)
            mx.eval(ref)
            diff = mx.max(mx.abs(metal_out - ref)).item()
            print(f"  {label} vs reference: max_diff = {diff:.6f}")
        except Exception as e:
            print(f"  {label}: skipped ({e})")


if __name__ == "__main__":
    main()
