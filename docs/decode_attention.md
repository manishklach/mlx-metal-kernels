# Decode Attention

## Prefill vs decode

Prefill attention processes many query tokens at once. Decode attention is the
single-token case used during autoregressive generation, where the model attends
from one fresh query token into an existing KV cache.

## Why decode attention is single-query attention

For decode, the query tensor has shape `[B, 1, H, D]` while the cache tensors
have shape `[B, MAX_S, H, D]`. Only a prefix of the cache is valid for each
batch item, which is why the decode path accepts `lengths`.

## Online softmax

The decode kernel uses online softmax state while streaming over the valid KV
prefix:

- `m`: running max score
- `l`: running normalization term
- `acc`: running weighted value sum

This avoids materializing the full attention score vector.

## Lengths

`lengths` can be a scalar or per-batch vector. Only positions `j < lengths[b]`
participate in attention for batch item `b`.

## Backend status

- `reference`: pure MLX reference implementation
- `metal`: correctness-first decode-specific Metal kernel
- `auto`: currently aliases to `metal`
