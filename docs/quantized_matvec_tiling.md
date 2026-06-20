# Quantized Matvec Tiling

PR #9 added a correctness-first parallel decode matvec path where one threadgroup computes one output element `y[b, n]`.

PR #13 adds a new experimental tiled backend:

- one threadgroup computes one batch item plus a small tile of output channels
- the kernel reuses `x[b, k]` across multiple output channels
- partial sums are accumulated in float and reduced in threadgroup memory

## Initial Design

- `N_TILE = 4`
- q4 and q8 both use the same multi-output tiling structure
- optional zero-points are supported through the existing stable signature
- wrappers keep `backend="auto"` conservative unless `MLX_METAL_USE_TILED_MATVEC=1`

## q4 Details

q4 weights remain packed two values per byte. The tiled kernel unpacks the required nibble on the fly, applies groupwise scale and optional zero-point correction, and accumulates the result for each tile output.

## Current Limitations

- not yet vectorized for packed q4 loads
- not yet staging K tiles in threadgroup memory
- not yet using simdgroup reductions
- not yet tuned per Apple Silicon generation

## Future Work

- `N_TILE` autotuning
- simdgroup reductions
- vectorized nibble unpack
- fused qkv projection
- fused q4/q8 MLP matvec
- chip-specific tuning
