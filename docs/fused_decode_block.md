# Fused Decode Block

PR #7 adds a composition-first decode block helper for MLX custom Metal kernels on Apple Silicon.

## Decode block flow

The decode block starts from a single-token QKV projection output and then composes the existing helper kernels:

1. split `qkv` into `q`, `k`, and `v`
2. apply RoPE to `q` and `k` at the decode position
3. update the KV cache
4. run decode attention over the valid prefix
5. optionally apply residual add plus RMSNorm

This keeps the implementation correctness-first while still matching the structure of a transformer decode step.

## Contiguous and paged variants

The contiguous path uses:

- `decode_block_from_qkv(...)`
- contiguous cache layout `[B, MAX_S, H, D]`

The paged path uses:

- `paged_decode_block_from_qkv(...)`
- paged cache layout `K_pages` and `V_pages` with shape `[NUM_PAGES, PAGE_SIZE, H, D]`
- `block_table [B, MAX_BLOCKS]`

Both paths validate against pure MLX reference implementations before benchmarking.

## Residual plus norm helper

`residual_rmsnorm_block(...)` reuses the existing `rmsnorm_residual(...)` helper rather than introducing a duplicate implementation. That keeps the decode-block naming readable without creating a second kernel path for the same math.

## Composition-first fusion

This PR intentionally keeps the default execution path as helper composition:

- contiguous: split/RoPE/cache-update + decode attention
- paged: split/RoPE + paged cache update + paged decode attention

That makes correctness easier to validate and keeps each stage independently testable.

## Future fused kernels

The repo now includes experimental scaffold files for future one-launch kernels:

- `kernels/fused_qkv_rope_decode.metal`
- `kernels/fused_qkv_rope_paged_decode.metal`

Potential next steps:

- contiguous one-launch `qkv -> rope -> cache update -> decode`
- paged one-launch decode block using block-table lookup
- quantized projection plus decode block fusion
- output projection fusion

Those paths should stay experimental until they pass correctness validation and are benchmarked on Apple Silicon.
