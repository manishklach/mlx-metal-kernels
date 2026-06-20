# Quantized Decode Block

The quantized decode block composes existing MLX helpers into a correctness-first single-token decode path for local transformer inference on Apple Silicon.

The flow is:

`x -> quantized QKV projection -> split/RoPE/cache update -> decode attention -> quantized output projection`

This PR keeps the design composition-first:

- `quantized_qkv_projection()` reuses `q4_matvec_decode()` or `q8_matvec_decode()`
- `decode_block_from_qkv()` and `paged_decode_block_from_qkv()` handle RoPE, cache update, and decode attention
- `quantized_output_projection()` reuses the existing quantized decode matvec helpers again

No new monolithic default kernel is introduced here. That keeps the path easier to validate against pure MLX references while still exercising the existing Metal kernels in the hot projection and decode-attention pieces.

## Contiguous And Paged Variants

- `quantized_decode_block()` targets contiguous KV caches shaped `[B, MAX_S, H, D]`
- `paged_quantized_decode_block()` targets paged caches plus a block table

Both variants expose separate backend knobs:

- `matvec_backend`: controls q4/q8 projection backends such as `reference`, `metal`, and `metal_parallel`
- `block_backend`: controls the decode block backend such as `reference`, `metal`, or `auto`

That split makes it easy to validate the matvec path and decode-attention path independently.

## Why This Matters

Transformer decode on Mac often spends a large share of time in linear projections and KV-cache attention. A quantized decode block gives the repo a more realistic inference-oriented building block without overclaiming performance before benchmarks are run on real Apple Silicon hardware.

## Future Work

- fused q4 QKV projection plus split/RoPE/cache update
- fused output projection plus residual add
- q4 MLP block helpers
- end-to-end single-layer decode benchmark
- benchmark-report integration across Apple Silicon generations
