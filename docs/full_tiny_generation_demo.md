# Full Tiny-Model Generation Demo

## Purpose

This demo connects the repo's tokenizer, synthetic embeddings, multi-layer Llama-like decode stack, final RMSNorm, `lm_head`, sampling helpers, and text decode path into one end-to-end flow.

It is a plumbing and validation scaffold, not a trained model runtime.

## End-to-end pipeline

The demo exercises:

1. prompt text
2. tokenizer encode
3. token ids
4. embedding lookup
5. multi-layer decode stack
6. final norm and `lm_head`
7. greedy or sampled token selection
8. generated token ids
9. decoded output text

## Synthetic weights warning

All weights in this demo are synthetic random q4/q8 weights. The generated output is intentionally nonsense-like and should not be treated as meaningful language generation.

## Relationship to checkpoint converter and tokenizer adapter

- The default path uses `CharTokenizer` with no optional dependencies.
- Optional tokenizer adapters remain separate and local-file only.
- The quantized package format currently stores metadata, shapes, and packaging information, but not tensor payloads.
- Because of that, the package-loading helper currently raises a clear `NotImplementedError` and the package demo falls back to the synthetic pipeline.

## GenerationResult

`GenerationResult` returns:

- `prompt`
- `prompt_ids`
- `generated_ids`
- `all_ids`
- `text`
- `backend_preset`
- `metadata`

Metadata includes prompt length, generated length, model size hints, backend choice, and `synthetic_weights=True`.

## CLI demo

```bash
python examples/full_tiny_generation_demo.py --prompt "Hello" --max-new-tokens 8 --greedy
python examples/full_tiny_generation_with_package_demo.py
```

## Benchmark

```bash
python benchmarks/bench_tiny_generation_pipeline.py --prompt-len 8 --max-new-tokens 16 --num-layers 2 --hidden-size 512 --intermediate-size 2048 --num-heads 8 --num-kv-heads 2 --head-dim 64 --bits 4 --backend-preset fused_experimental --greedy
```

The benchmark is intended for local Apple Silicon measurements. It reports total runtime, per-generated-token time, and tokens per second for synthetic runs.

## Current limitations

- random weights only
- no trained-model quality
- no real checkpoint tensor-data loading yet
- no production tokenizer or chat template handling
- no optimized prompt prefill stack yet
- small synthetic runtime only

## Future work

- load quantized packages with tensor payloads
- align real tokenizer metadata with package/config metadata
- add optimized prefill stack paths
- run local real-model smoke tests once tensor loading is wired
- add prompt and chat-template helpers for local experiments
