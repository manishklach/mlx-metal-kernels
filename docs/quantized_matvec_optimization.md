# Quantized Matvec Optimization

Decode-time matvec or GEMV is one of the most important kernels in quantized transformer inference. After attention, the model still spends substantial time applying quantized projection weights to a single or small batch of activations.

## Starting point

PR #5 added correctness-first q4 and q8 decode matvec kernels:

- dequantize one weight element at a time
- accumulate one output element per thread
- keep the implementation simple and easy to validate

That path remains available as `backend="metal"`.

## New parallel backend

PR #9 adds `backend="metal_parallel"` for:

- `q4_matvec_decode(...)`
- `q8_matvec_decode(...)`

The design is intentionally simple:

- one threadgroup computes one output element `y[b, n]`
- threads split the `K` dimension with strided work
- each thread accumulates a partial sum in `float`
- a threadgroup reduction combines partial sums
- thread `0` writes the final output value

## q4 unpacking

q4 weights stay packed as two 4-bit values per byte:

- low nibble: `byte & 0x0F`
- high nibble: `(byte >> 4) & 0x0F`

The parallel kernel unpacks values on the fly while applying groupwise scale and optional zero-point correction.

## Groupwise scales and zero-points

Both q4 and q8 paths support:

- groupwise scales
- optional zero-points
- fp16 or bf16 activations

If zero-points are omitted, the Python wrapper passes a dummy zeros tensor and sets a `has_zero` metadata flag to keep the kernel signature stable.

## Current limitations

The v1 parallel kernel is still correctness-first:

- not tiled across `N`
- not simdgroup optimized
- not vectorized for packed q4 loads
- not fused with activation or projection bias

## Future work

- multiple outputs per threadgroup
- simdgroup reductions
- vectorized q4 unpack
- blockwise `K` tiling
- fused dequant + matvec + bias
- q4/q8 MLP block kernels
- chip-specific tuning across Apple Silicon generations
