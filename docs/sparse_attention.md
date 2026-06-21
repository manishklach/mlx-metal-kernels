# Sparse Attention

## Purpose

This repo now includes correctness-first sparse GQA/MQA-aware attention paths for long-context inference experiments on Apple Silicon.

## Dense attention vs sparse attention

Dense attention exposes every query to every visible key. Sparse attention narrows that visibility pattern explicitly and is intended to reduce work for long contexts when the sparse policy is acceptable for the model/runtime.

## Sliding-window attention

The first sparse pattern is causal sliding-window attention. Each query can only attend to a trailing local window of keys.

## Sink tokens

The second pattern adds sink tokens. A small prefix of keys remains globally visible while the rest of attention is restricted to the local sliding window.

## Block-sparse scaffold

This PR also adds a block-sparse mask scaffold. The reference path can build block masks, but the Metal backend remains explicit-only and reference-first for now.

## GQA/MQA mapping

Sparse attention follows the repo's existing grouped-query mapping:

- `Hq >= Hkv`
- `Hq % Hkv == 0`
- `hkv = hq // (Hq / Hkv)`

This supports MHA, GQA, and MQA.

## Prefill sparse attention

`sparse_gqa_attention` supports `Q [B,Sq,Hq,D]`, `K/V [B,Sk,Hkv,D]`, explicit sparse patterns, and reference plus sliding-window Metal backends.

## Decode sparse attention

`sparse_gqa_decode_attention` supports `q [B,1,Hq,D]`, `K/V cache [B,MAX_S,Hkv,D]`, runtime lengths, and explicit sliding-window decode backends.

## KV-cache bandwidth implications

Sparse decode attention narrows the visible cache region and is a natural fit for future experiments around long-context cache bandwidth, paged KV, and quantized KV-cache layouts.

## Benchmarks

Benchmarks are explicit and local-machine only:

```bash
python benchmarks/bench_sparse_attention.py --B 1 --S 512 --Hq 32 --Hkv 8 --D 128 --window-size 128 --sink-tokens 4 --dtype float16 --backend all --validate
python benchmarks/bench_sparse_decode_attention.py --B 1 --MAX_S 4096 --length 4096 --Hq 32 --Hkv 8 --D 128 --window-size 512 --sink-tokens 4 --dtype float16 --backend all --validate
```

## Current limitations

- explicit backend only
- block-sparse Metal backend is not implemented yet
- no automatic sparse policy
- no production long-context runtime claims
- no tokenizer/model-quality claims

## Future work

- sparse prefill stack integration
- block-sparse Metal kernels
- sparse + paged KV-cache
- sparse + quantized KV-cache
- sparse + flash/NAND offload
- learned or dynamic sparsity policies
