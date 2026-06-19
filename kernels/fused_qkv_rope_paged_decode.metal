// Experimental scaffold for a future monolithic paged decode block kernel.
// Target flow: qkv split -> RoPE -> block-table aware paged KV update -> paged decode.
// This file intentionally stays off the default path until a correctness-validated
// one-launch implementation is ready.
