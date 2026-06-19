# Paged KV Cache

Paged KV-cache exists to decouple logical token positions from physical cache
storage, which becomes useful for long-running decode sessions, reuse, and
future page-level allocation strategies.

## Contiguous vs paged KV

Contiguous cache:

- `K_cache [B, MAX_S, H, D]`
- `V_cache [B, MAX_S, H, D]`

Paged cache:

- `K_pages [NUM_PAGES, PAGE_SIZE, H, D]`
- `V_pages [NUM_PAGES, PAGE_SIZE, H, D]`
- `block_table [B, MAX_BLOCKS]`

## Logical to physical mapping

- `block_id = position // PAGE_SIZE`
- `offset = position % PAGE_SIZE`
- `page_id = block_table[b, block_id]`

The physical cache slot is then:

- `K_pages[page_id, offset, h, d]`
- `V_pages[page_id, offset, h, d]`

## APIs

- `paged_kv_cache_update`
- `paged_decode_attention`
- `paged_decode_step`

This v1 path is correctness-first and mirrors the flat-cache helpers already in
the repo.

## Future direction

- page allocator
- block reuse
- prefix sharing
- sparse update kernels
- optimized threadgroup paged decode
- integration with fused decode blocks
