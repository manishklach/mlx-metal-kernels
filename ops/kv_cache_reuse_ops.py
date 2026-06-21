from __future__ import annotations

from typing import Any

try:
    import mlx.core as mx
except ImportError:
    mx = None

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError("numpy is required for kv_cache_reuse_ops") from exc

def _llama_stack_cache_class():
    from ops.llama_stack_ops import LlamaStackCache as LSC

    return LSC


def _is_mlx_array(value: Any) -> bool:
    return mx is not None and type(value).__module__.startswith("mlx")


def _as_contiguous_copy(x: Any) -> Any:
    if _is_mlx_array(x):
        return mx.array(x)
    return np.array(x, copy=True)


def clone_layer_cache(layer_cache: tuple[Any, Any]) -> tuple[Any, Any]:
    K, V = layer_cache
    return (_as_contiguous_copy(K), _as_contiguous_copy(V))


def clone_stack_cache(stack_cache) -> Any:
    LSC = _llama_stack_cache_class()
    cloned_layer_caches = [clone_layer_cache(lc) for lc in stack_cache.layer_caches]
    return LSC(
        layer_caches=cloned_layer_caches,
        cache_layout=stack_cache.cache_layout,
        max_seq_len=stack_cache.max_seq_len,
        page_size=stack_cache.page_size,
    )


def slice_layer_cache(layer_cache: tuple[Any, Any], length: int) -> tuple[Any, Any]:
    K, V = layer_cache
    K_sliced = K[:, :length, :, :]
    V_sliced = V[:, :length, :, :]
    if _is_mlx_array(K):
        return (mx.array(K_sliced), mx.array(V_sliced))
    return (np.array(K_sliced, copy=True), np.array(V_sliced, copy=True))


def copy_prefix_cache_into(
    src_cache,
    dst_cache,
    length: int,
):
    if src_cache.cache_layout != dst_cache.cache_layout:
        raise ValueError(
            f"cache_layout mismatch: src={src_cache.cache_layout}, dst={dst_cache.cache_layout}"
        )
    if len(src_cache.layer_caches) != len(dst_cache.layer_caches):
        raise ValueError(
            f"layer count mismatch: src={len(src_cache.layer_caches)}, dst={len(dst_cache.layer_caches)}"
        )
    new_layer_caches = []
    for src_lc, dst_lc in zip(src_cache.layer_caches, dst_cache.layer_caches):
        src_K, src_V = src_lc
        dst_K, dst_V = dst_lc
        new_K = _as_contiguous_copy(dst_K)
        new_V = _as_contiguous_copy(dst_V)
        if _is_mlx_array(new_K):
            new_K[:, :length] = src_K[:, :length]
            new_V[:, :length] = src_V[:, :length]
        else:
            new_K[:, :length] = src_K[:, :length]
            new_V[:, :length] = src_V[:, :length]
        new_layer_caches.append((new_K, new_V))
    LSC = _llama_stack_cache_class()
    return LSC(
        layer_caches=new_layer_caches,
        cache_layout=src_cache.cache_layout,
        max_seq_len=dst_cache.max_seq_len,
        page_size=dst_cache.page_size,
    )


def cache_prefix_equal(
    cache_a,
    cache_b,
    length: int | None = None,
) -> bool:
    if len(cache_a.layer_caches) != len(cache_b.layer_caches):
        return False
    if cache_a.cache_layout != cache_b.cache_layout:
        return False
    for lc_a, lc_b in zip(cache_a.layer_caches, cache_b.layer_caches):
        K_a, V_a = lc_a
        K_b, V_b = lc_b
        if K_a.shape != K_b.shape or V_a.shape != V_b.shape:
            return False
        if length is not None:
            if length > K_a.shape[1] or length > K_b.shape[1]:
                return False
            cmp_K = K_a[:, :length, :, :]
            cmp_V = V_a[:, :length, :, :]
            ref_K = K_b[:, :length, :, :]
            ref_V = V_b[:, :length, :, :]
        else:
            cmp_K, cmp_V = K_a, V_a
            ref_K, ref_V = K_b, V_b
        if _is_mlx_array(cmp_K):
            if not mx.all(mx.equal(cmp_K, ref_K)):
                return False
            if not mx.all(mx.equal(cmp_V, ref_V)):
                return False
        else:
            if not np.array_equal(cmp_K, ref_K):
                return False
            if not np.array_equal(cmp_V, ref_V):
                return False
    return True


def paged_cache_reuse_not_implemented():
    raise NotImplementedError(
        "Prefix KV-cache reuse currently supports only contiguous cache layout. "
        "Paged cache reuse is reserved for future work."
    )
