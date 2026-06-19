# Layout Kernels

## QKV layouts

The repo currently supports two QKV input layouts for split helpers:

- packed: `[B, S, 3 * H * D]`
- explicit: `[B, S, 3, H, D]`

Both forms produce BSHD outputs:

- `q [B, S, H, D]`
- `k [B, S, H, D]`
- `v [B, S, H, D]`

## `qkv_split`

`qkv_split` converts packed or explicit projection output into BSHD tensors.
This is a correctness-first helper used to make fused decode paths easier to
compose and benchmark.

## `qkv_split_rope`

`qkv_split_rope` performs split plus RoPE application on `q` and `k`, while
leaving `v` unchanged.

## When layout kernels matter

Layout helpers become more important when projection outputs feed directly into
decode/update kernels, because they reduce Python-side tensor surgery and make
future fused decode blocks easier to express.
