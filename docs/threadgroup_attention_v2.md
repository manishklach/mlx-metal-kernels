# Threadgroup Attention v2

This stage adds a second generation of correctness-first attention kernels built around cooperative threadgroup reductions.

## Motivation

The original baseline attention kernels keep the implementation easy to validate, but they leave a lot of GPU parallelism unused because a single thread handles a full attention row. Threadgroup Attention v2 moves to one threadgroup per attention row and lets threads collaborate on the score computation.

## Design

- one threadgroup per attention row
- threads split work across the `D` dimension
- dot products are reduced in threadgroup memory
- online softmax state is maintained in float
- outputs are written back as `ELEM_TYPE`

The same basic idea is used in three places:

- contiguous decode attention
- paged decode attention
- full prefill attention

## Contiguous Decode

`decode_attention(..., backend="metal_threadgroup")` uses one threadgroup per `[b, h]` row. The kernel streams over cache positions, performs a cooperative `q·k` reduction, updates online softmax state, and accumulates `V`.

## Paged Decode

`paged_decode_attention(..., backend="metal_threadgroup")` keeps the same reduction pattern while reading K/V through the block table and page metadata.

## Prefill / Full Attention

`fast_attention(..., backend="threadgroup")` applies the same cooperative reduction scheme to full attention rows `[b, s_q, h]`, with optional causal masking.

## Current Limitations

- not yet using `simdgroup_matrix`
- not yet staging K/V tiles into threadgroup memory for the full path
- not yet running multiple query rows per threadgroup
- may not outperform optimized MLX or native kernels yet

## Future Work

- simdgroup reductions
- vectorized loads
- multiple query rows per threadgroup
- K/V tiling
- shape-specific D=64 / D=128 threadgroup kernels
