# Fused Decode Path

## Decode path in this PR

The decode path modeled here is:

1. QKV projection output for one token
2. `qkv_rope_cache_update`
3. `decode_attention`

This is not yet one monolithic fused transformer block. It is a correctness-
first composition that matches the shape of the future fused decode path while
keeping individual helpers easy to validate.

## Future direction

- tighter fusion of `qkv_rope_cache_update` and `decode_attention`
- paged KV cache layouts
- quantized matvec decode paths
- fused residual and normalization helpers around decode blocks
