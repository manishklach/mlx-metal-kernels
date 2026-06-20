# Optimized Prefill Stack

## Purpose

This scaffold adds a real multi-layer prompt prefill path for the synthetic Llama-like runtime experiments in the repo.

Instead of ingesting prompt tokens one by one through decode, prefill runs the prompt as a sequence, fills the KV-cache for every layer, and then hands off generation to decode from the final prompt position.

## Decode ingestion vs prefill

Token-by-token decode ingestion is useful for correctness plumbing, but it does not reflect the usual prompt path of a transformer runtime.

The prefill path processes:

1. token ids
2. embeddings `[B,S,H]`
3. multi-layer prefill through the stack
4. final RMSNorm
5. optional logits for the last prompt token
6. one filled KV-cache per layer
7. decode continuation from position `S`

## Multi-layer prefill flow

Each layer runs:

- input RMSNorm
- quantized QKV projection
- GQA/MQA-aware QKV split
- RoPE over all prompt positions
- causal sequence attention over the prompt segment
- output projection
- residual connection
- post-attention RMSNorm
- quantized MLP block
- final residual

The stack then applies final RMSNorm and optional `lm_head`.

## GQA/MQA prefill attention

The prefill path reuses the existing GQA/MQA prefill attention helpers already present in the repo. This keeps GQA and MQA behavior aligned with the rest of the correctness-first transformer scaffolding.

## KV-cache filling

For PR32 the implemented path focuses on contiguous KV-cache filling from `start_position=0`.

Current behavior:

- prompt prefill from an empty cache is supported
- one cache is maintained per layer
- decode can continue from the prompt length
- continuation prefill with `start_position > 0` is intentionally conservative and may raise `NotImplementedError`

## Prefill then decode continuation

`TinyGenerationPipeline` now supports `use_prefill=True`.

That path:

1. encodes the prompt
2. runs a single prompt prefill call
3. samples from the last prompt-position logits
4. continues with decode for new-token generation

The old token-by-token decode-ingest path remains available through `use_prefill=False`.

## Backend presets

The prefill scaffold exposes:

- `reference`
- `metal`
- `tiled`
- `fused_experimental`

These presets map norm, quantized matvec, GQA attention, and MLP composition to existing repo backends rather than introducing a new monolithic kernel.

## Benchmarks

Useful commands:

```bash
python benchmarks/bench_llama_prefill_stack.py --bits 4 --B 1 --S 64 --num-layers 2 --hidden-size 512 --intermediate-size 2048 --num-heads 8 --num-kv-heads 2 --head-dim 64 --MAX_S 128 --dtype float16 --backend-preset all --validate
python benchmarks/bench_llama_prefill_stack.py --bits 4 --B 1 --S 128 --num-layers 4 --hidden-size 512 --intermediate-size 2048 --num-heads 8 --num-kv-heads 2 --head-dim 64 --MAX_S 256 --dtype float16 --backend-preset fused_experimental --compare-decode-ingest
```

## Current limitations

- synthetic/random weights
- contiguous cache first
- no production model runtime
- continuation prefill with `start_position > 0` may be unsupported
- no optimized paged prefill in this scaffold
- no chat-template or tokenizer correctness claims

## Future work

- paged prefill
- optimized fused QKV split plus RoPE plus GQA attention
- real quantized package tensor-data loading
- local real-model smoke test
- prompt batching
