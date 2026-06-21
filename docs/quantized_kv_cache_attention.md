# Quantized KV-Cache Attention

## 1. Purpose

This module adds correctness-first experiments for storing KV-cache tensors in q8/q4 form and dequantizing them inside decode attention. The central goal is to reduce KV-cache memory pressure by storing keys and values in quantized form with per-token/head/group scales, then dequantizing on the fly during GQA/MQA/MHA decode attention.

## 2. Why quantize KV-cache?

KV-cache memory grows linearly with batch size, sequence length, number of layers, and head dimension. For long-context inference, the KV-cache can dominate total memory usage. Quantizing the KV-cache from fp16 (2 bytes per value) to q8 (1 byte) or q4 (0.5 bytes) reduces memory by 2x or 4x respectively, at the cost of some accuracy in the attention computation.

## 3. FP16 vs q8 vs q4 KV-cache memory

| Format | Bytes per value | KV cache (B=1, L=32, Hkv=8, D=128, S=4096) |
|--------|----------------|----------------------------------------------|
| fp16   | 2              | 2 * 2 * 32 * 8 * 128 * 4096 = 512 MB        |
| q8     | 1 + scales     | ~256 MB + scales overhead                    |
| q4     | 0.5 + scales   | ~128 MB + scales overhead                    |

## 4. Quantization layout

- Per-token, per-head, per-group quantization along the D dimension
- Symmetric quantization: scale = max_abs / max_q, q = round(value / scale), stored as unsigned with bias
- q8: q_unsigned = clip(round(value / scale), -127, 127) + 128, stored as uint8
- q4: q_unsigned = clip(round(value / scale), -7, 7) + 8, two values packed per byte

## 5. Per-token/head/group scales

Scales shape: `[B, MAX_S, Hkv, ceil(D / group_size)]`

Each group of `group_size` consecutive values in the D dimension shares a single scale. The default group_size is 32, giving 4 groups when D=128. Scales are stored in float16.

## 6. GQA/MQA head mapping

Quantized KV-cache decode attention maps query heads to KV heads using GQA conventions:
- `group = Hq / Hkv`
- `hkv = hq / group`

This is the same mapping used by the existing `reference_gqa_decode_attention`.

## 7. Decode attention with dequantized K/V

The quantized decode loop:
1. For each KV position, dequantize K and V values on the fly using per-group scales
2. For q8: `k_val = (uint8_value - 128) * scale[group]`
3. For q4: unpack nibble, `k_val = (nibble - 8) * scale[group]`
4. Compute attention scores with online softmax (same as fp16 decode)
5. Accumulate dequantized V values weighted by attention probabilities

## 8. Sparse + quantized KV-cache

The reference sparse quantized decode attention dequantizes the full cache and applies a sparse attention mask. The Metal sparse+quantized backend is not implemented yet (future work).

## 9. Benchmark commands

```bash
# q8 benchmark with validation and fp16 comparison
python benchmarks/bench_quantized_kv_decode_attention.py \
  --bits 8 --B 1 --MAX_S 4096 --length 4096 --Hq 32 --Hkv 8 --D 128 \
  --group-size 32 --backend all --validate --compare-fp16

# q4 benchmark
python benchmarks/bench_quantized_kv_decode_attention.py \
  --bits 4 --B 1 --MAX_S 4096 --length 4096 --Hq 32 --Hkv 8 --D 128 \
  --group-size 32 --backend all --validate --compare-fp16
```

## 10. Current limitations

- decode attention only (no prefill)
- contiguous cache only (no paged quantized KV-cache)
- q8/q4 symmetric quantization only (no asymmetric, no zero-point)
- no production quality guarantee
- no automatic runtime policy
- sparse quantized Metal backend is future work
- GPU-tensor quantized KV-cache update not implemented

## 11. Future work

- paged quantized KV-cache
- quantized KV-cache update kernel
- sparse + quantized + offloaded KV integration
- q4 KV-cache with better error control
- mixed precision KV by layer/head
- activation-aware KV quantization
- quantized KV prefill
