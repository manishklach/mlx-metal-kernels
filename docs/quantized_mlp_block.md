# Quantized MLP Block

This repo now includes a composition-first quantized MLP block for Llama-like transformer layers:

1. residual add
2. RMSNorm
3. quantized gate projection
4. quantized up projection
5. SwiGLU
6. quantized down projection
7. residual add back into the layer stream

The implementation lives in [ops/mlp_block_ops.py](/C:/Users/ManishKL/Documents/Playground/mlx-flash-attention-metal/ops/mlp_block_ops.py) and deliberately reuses existing repo primitives instead of introducing a new monolithic fused kernel.

## Backends

`quantized_mlp_decode_step(..., backend_preset=...)` supports:

- `reference`
- `metal`
- `parallel`
- `tiled`

These presets map to existing RMSNorm, q4/q8 decode matvec, SwiGLU, and residual helpers.

## Quantized projections

The gate, up, and down projections use the same q4/q8 conventions as the decode matvec kernels:

- q4 weights: packed `[OUT_DIM, ceil(IN_DIM / 2)]`
- q8 weights: `[OUT_DIM, IN_DIM]`
- scales: `[OUT_DIM, ceil(IN_DIM / group_size)]`
- zeros: optional `[OUT_DIM, ceil(IN_DIM / group_size)]`

## Why this matters

Attention is not the only decode-time hotspot. In Llama-like models, the MLP block is also a major share of token latency, especially at larger hidden and intermediate dimensions. This PR extends the repo from quantized attention composition into quantized feed-forward composition while keeping correctness validation front and center.

## Scope

This PR is composition-first:

- reference path exists for correctness checks
- optimized path reuses validated kernels
- no monolithic fused q4 MLP kernel is claimed here
- no performance claims should be made without benchmark data

## Future work

- fused gate/up quantized matvec helper
- fused SwiGLU plus down projection helper or kernel
- MLP-specific multi-output tiling
- dedicated q4 MLP block kernel
- deeper integration into real checkpoint loading and model execution
