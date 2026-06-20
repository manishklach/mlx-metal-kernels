# Llama-Like Decode Layer Experiment

## Purpose

This experiment connects the repo's quantized attention path, optimized GQA decode attention, fused quantized MLP path, residual helpers, normalization, KV-cache handling, and backend presets into a realistic single-layer decode loop.

## What this experiment does

- runs a synthetic one-layer decode loop
- supports MHA, GQA, and MQA layouts
- uses q4 or q8 packed weights
- compares explicit backend presets against a reference path

## What it does not do

- tokenizer integration
- sampling
- multi-layer generation
- model downloads
- full production checkpoint execution

## Layer formula

The experiment follows a Llama-like residual structure:

`h = x + o_proj(attn(rms_norm(x)))`

`y = h + down_proj(swiglu(gate_proj(rms_norm(h)), up_proj(rms_norm(h))))`

## GQA support

- query heads use `num_attention_heads`
- cache stores `num_key_value_heads`
- output attention still flattens back to `hidden_size`

## Quantized weights

- q4 or q8 packed weights
- groupwise scales
- optional zeros fields reserved in the weight container

## Backend presets

- `reference`
- `metal`
- `tiled`
- `fused_experimental`

## Benchmark commands

```bash
python benchmarks/bench_llama_layer_decode.py --bits 4 --B 1 --T 16 --hidden-size 512 --intermediate-size 2048 --num-heads 8 --num-kv-heads 2 --head-dim 64 --MAX_S 128 --dtype float16 --backend-preset all --validate
python examples/llama_layer_decode_demo.py
```

## Current limitations

- one layer only
- synthetic or random packed weights
- no tokenizer
- no sampling
- no full checkpoint execution path yet

## Future

- multi-layer stack
- checkpoint-to-quantized packaging
- tokenizer and sampling demo
- full decode loop experiments
