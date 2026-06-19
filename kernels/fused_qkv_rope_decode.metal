// Experimental scaffold for a future monolithic contiguous decode block kernel.
// Current Python default composes qkv split + RoPE + KV cache update + decode attention.
// Keep this source body-only for mx.fast.metal_kernel integration when the fused
// path is enabled and validated on Apple Silicon.
