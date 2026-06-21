# Paged quantized KV-cache

## 1. Purpose

Extend the existing KV-cache quantization work (PR40) from contiguous layout to paged layout. This allows quantized (q8/q4) K/V pages to be used with the existing paged decode attention infrastructure, supporting longer sequences with reduced memory footprint.

## 2. Why combine paging and KV quantization

- Paged KV-cache already supports variable-length sequences and memory-efficient storage by splitting K/V into fixed-size pages.
- KV quantization (q8/q4) reduces per-value storage from 2 bytes (fp16) to 1 byte (q8) or 0.5 bytes (q4).
- Combining them gives both the flexibility of paging and the memory savings of quantization.

## 3. Paged KV layout

The existing paged KV-cache stores K/V pages as fp16 tensors:

```
K_pages [NUM_PAGES, PAGE_SIZE, Hkv, D]
V_pages [NUM_PAGES, PAGE_SIZE, Hkv, D]
block_table [B, MAX_BLOCKS]  # maps (batch, block_idx) -> page_id
lengths [B]                   # valid tokens per batch
```

## 4. q8 paged KV layout

For q8 quantization, each fp16 value is stored as a uint8 with per-group scale:

```
k_pages_q [NUM_PAGES, PAGE_SIZE, Hkv, D]       # uint8
v_pages_q [NUM_PAGES, PAGE_SIZE, Hkv, D]       # uint8
k_scales  [NUM_PAGES, PAGE_SIZE, Hkv, num_groups]  # float16
v_scales  [NUM_PAGES, PAGE_SIZE, Hkv, num_groups]  # float16
```

Dequantization: `value_fp32 = (uint8 - 128) * scale[group]`

## 5. q4 packed paged KV layout

For q4 quantization, two values are packed per byte:

```
k_pages_q [NUM_PAGES, PAGE_SIZE, Hkv, ceil(D/2)]  # uint8 packed
v_pages_q [NUM_PAGES, PAGE_SIZE, Hkv, ceil(D/2)]  # uint8 packed
k_scales  [NUM_PAGES, PAGE_SIZE, Hkv, num_groups]  # float16
v_scales  [NUM_PAGES, PAGE_SIZE, Hkv, num_groups]  # float16
```

Nibble unpacking:
- even d: low 4 bits
- odd d: high 4 bits
- `nibble_signed = nibble - 8`
- `value_fp32 = nibble_signed * scale[group]`

## 6. Scales per page/token/head/group

Scales are stored with shape `[NUM_PAGES, PAGE_SIZE, Hkv, num_groups]`. Each group covers `group_size` values along the D dimension. This fine-grained scaling preserves accuracy better than per-token or per-head scaling.

## 7. GQA/MQA head mapping

The decode attention follows the same GQA convention as the existing paged decode: for a query head `hq`, the key-value head is `hkv = hq // (Hq / Hkv)`. This supports:
- GQA: Hq > Hkv, Hq % Hkv == 0
- MQA: Hkv == 1
- MHA: Hq == Hkv

## 8. Decode attention over paged quantized KV

The decode attention kernel iterates over valid token positions (0..length-1). For each position, it:

1. Computes `block_idx = pos / PAGE_SIZE`, `offset = pos % PAGE_SIZE`
2. Looks up `page_id = block_table[b, block_idx]`
3. Loads q8/q4 K value from `k_pages_q[page_id, offset, hkv, :]`
4. Dequantizes using `k_scales[page_id, offset, hkv, group]`
5. Computes dot product with query
6. Applies online softmax
7. Accumulates dequantized V

## 9. Memory accounting

- `memory_bytes()`: total bytes of all quantized tensors + block_table + lengths.
- `compression_ratio(fp_bytes_per_value=2)`: ratio of fp16 storage to quantized storage.
- q8 is approximately 2x compression (plus scale overhead).
- q4 is approximately 4x compression (plus scale overhead).

## 10. Benchmark commands

```
python benchmarks/bench_paged_quantized_kv_decode.py --bits 8 --B 1 --MAX_S 4096 --length 4096 --PAGE_SIZE 16 --Hq 32 --Hkv 8 --D 128 --group-size 32 --backend all --validate --compare-fp16

python benchmarks/bench_paged_quantized_kv_decode.py --bits 4 --B 1 --MAX_S 4096 --length 4096 --PAGE_SIZE 16 --Hq 32 --Hkv 8 --D 128 --group-size 32 --backend all --validate --compare-fp16
```

## 11. Current limitations

- Decode attention only (no prefill or update kernels for paged quantized KV).
- q8/q4 symmetric quantization only (no asymmetric or zero-point).
- `contiguous_kv_to_pages` helper is test/demo only, not a production cache allocator.
- No paged sparse quantized Metal backend yet.
- No automatic runtime policy (must explicitly use paged quantized backend).
- No model-quality claims.

## 12. Future work

- Paged quantized KV-cache update kernel.
- Sparse + paged + quantized KV attention.
- Paged quantized KV offload.
- Async prefetch over quantized pages.
- Mixed precision pages (some pages fp16, some q8, some q4).
- Runtime cache allocator with page-level quantization decisions.
