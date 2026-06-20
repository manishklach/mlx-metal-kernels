import mlx.core as mx

from ops.gqa_ops import reference_paged_gqa_decode_attention
from ops.kv_cache_ops import reference_kv_cache_update
from ops.paged_kv_ops import allocate_paged_kv_cache, reference_paged_kv_cache_update


def test_reference_paged_gqa_decode_attention_shape():
    mx.random.seed(242)
    B, MAX_S, PAGE_SIZE, Hq, Hkv, D = 1, 8, 4, 4, 2, 16
    K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, Hkv, D, PAGE_SIZE, mx.float16)
    K_pages = mx.random.normal(K_pages.shape).astype(mx.float16)
    V_pages = mx.random.normal(V_pages.shape).astype(mx.float16)
    q = mx.random.normal((B, 1, Hq, D)).astype(mx.float16)
    out = reference_paged_gqa_decode_attention(q, K_pages, V_pages, block_table, lengths=MAX_S)
    mx.eval(out)
    assert out.shape == (B, 1, Hq, D)


def test_reference_paged_gqa_decode_attention_batch_shape():
    mx.random.seed(243)
    B, MAX_S, PAGE_SIZE, Hq, Hkv, D = 2, 16, 4, 8, 2, 16
    K_pages, V_pages, block_table = allocate_paged_kv_cache(B, MAX_S, Hkv, D, PAGE_SIZE, mx.float16)
    K_pages = mx.random.normal(K_pages.shape).astype(mx.float16)
    V_pages = mx.random.normal(V_pages.shape).astype(mx.float16)
    q = mx.random.normal((B, 1, Hq, D)).astype(mx.float16)
    out = reference_paged_gqa_decode_attention(q, K_pages, V_pages, block_table, lengths=[8, 12])
    mx.eval(out)
    assert out.shape == (B, 1, Hq, D)
