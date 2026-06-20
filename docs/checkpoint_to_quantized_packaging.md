# Checkpoint-to-quantized packaging

## Purpose

This scaffold connects local floating-point checkpoint tensors to the q4/q8 packed layouts already used by the repo's quantized matvec, decode-block, MLP-block, and synthetic Llama-like decode-layer experiments.

It is correctness-first and deterministic. It is not a production quantization or calibration pipeline.

## Why packaging is separate from inference kernels

The inference kernels in this repo expect fixed packed layouts, scale tensors, and optional zero-point tensors. Packaging is the bridge that prepares those tensors from ordinary floating-point weights.

Keeping packaging separate makes it easier to:

- validate quantization layouts independently from inference
- test dequantized reconstruction error on synthetic tensors
- stage future real-checkpoint support without coupling it to runtime code

## q4/q8 expected kernel layouts

The current kernel path expects:

- q4 weights packed two 4-bit values per byte
- q8 weights stored as unsigned bytes
- groupwise scales with shape `[out_dim, ceil(in_dim / group_size)]`
- optional zero-point tensors with the same shape as scales

For this scaffold, symmetric quantization stores unsigned q4/q8 values and materializes fixed zero-point offsets when converting to kernel-facing weights. That keeps the packaging simple while staying compatible with the existing dequant and matvec helpers.

## Groupwise quantization

Quantization is performed per output row and per contiguous input group.

For symmetric q4:

- compute `max_abs` for the group
- set `scale = max(max_abs / 7, eps)`
- round and clip signed values to `[-7, 7]`
- shift into unsigned storage for packing

For symmetric q8:

- compute `max_abs` for the group
- set `scale = max(max_abs / 127, eps)`
- round and clip signed values to `[-127, 127]`
- shift into unsigned storage

## Fused QKV packaging

The checkpoint adapter already knows how to fuse `q_proj`, `k_proj`, and `v_proj` into a single row-stacked tensor. The quantizer reuses that path so the resulting packed QKV weight matches the fused layout expected by the quantized decode codepath.

## GQA-aware QKV dimensions

For GQA and MQA layouts, fused QKV rows are:

`q_output_dim + 2 * kv_output_dim`

That keeps packaging aligned with the repo's split-and-RoPE helpers and with the synthetic Llama-like decode-layer experiment.

## QuantizedLlamaLayerPackage

`QuantizedLlamaLayerPackage` groups:

- preserved input and post-attention norm weights
- quantized fused QKV
- quantized output projection
- quantized gate, up, and down projections

This package is intended for tests, demos, and future checkpoint integration work.

## Conversion to LlamaLayerKernelWeights

`QuantizedLlamaLayerPackage.to_kernel_weights(config)` converts the package into `LlamaLayerKernelWeights`, which is the kernel-facing dataclass already used by the synthetic decode-layer experiment.

This conversion step also materializes fixed symmetric zero-point tensors when needed so the existing q4/q8 decode helpers can consume the packaged weights without changing their public APIs.

## Limitations

- no calibration-aware quantization
- no GPTQ, AWQ, or SmoothQuant
- no model-quality claims
- no production checkpoint conversion CLI
- no tokenizer or sampling runtime
- no Apple Silicon benchmark claims from this packaging layer alone

## Future work

- production safetensors checkpoint converter
- calibration-aware quantization
- AWQ/GPTQ-compatible import
- per-channel or richer per-group metadata export
- multi-layer quantized checkpoint package
