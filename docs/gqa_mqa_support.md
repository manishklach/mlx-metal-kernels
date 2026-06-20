# GQA and MQA Support

This repo now includes correctness-first grouped-query attention support for decode-time paths.

## Terms

- MHA: `Hq == Hkv`
- GQA: `1 < Hkv < Hq`
- MQA: `Hkv == 1`

The head mapping is:

```text
hkv = hq // (Hq / Hkv)
```

## Fused QKV layout

For GQA/MQA fused layouts, the packed projection output is:

```text
[Q ; K ; V]
```

with:

- `Q` dim = `Hq * D`
- `K` dim = `Hkv * D`
- `V` dim = `Hkv * D`

This means fused GQA output is not necessarily `3 * hidden_size`.

## Cache layout

- contiguous cache: `K_cache`, `V_cache` are `[B, MAX_S, Hkv, D]`
- paged cache: `K_pages`, `V_pages` are `[NUM_PAGES, PAGE_SIZE, Hkv, D]`
- query/output shape stays `[B, 1, Hq, D]`

## Current support

- reference GQA/MQA decode attention
- reference paged GQA/MQA decode attention
- composed contiguous GQA decode block
- composed paged GQA decode block
- quantized decode block routing for fused GQA QKV weights
- model adapter support for GQA cache shapes and quantized decode routing

## Current limits

- specialized Metal attention kernels still assume `Hq == Hkv`
- prefill GQA attention kernels are not added in this PR
- `decode_attention` and `paged_decode_attention` still require matching query/cache head counts; use `ops.gqa_ops` for grouped-query decode paths

## Why this matters

Llama and Mistral style models commonly use GQA or MQA. Supporting grouped-query layouts is a necessary step from toy decode kernels toward real checkpoint-aligned transformer inference on Apple Silicon.
