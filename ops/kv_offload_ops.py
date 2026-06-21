from __future__ import annotations

from typing import Any

try:
    import mlx.core as mx
except ImportError:
    mx = None

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError("numpy is required for kv_offload_ops") from exc


def _is_mlx_array(value: Any) -> bool:
    return mx is not None and type(value).__module__.startswith("mlx")


def _as_contiguous_copy(x: Any) -> Any:
    if _is_mlx_array(x):
        return mx.array(x)
    return np.array(x, copy=True)


# ---------------------------------------------------------------------------
# Extract a KV block from a layer cache
# ---------------------------------------------------------------------------

def extract_kv_block(layer_cache: tuple[Any, Any], start_token: int, end_token: int) -> tuple[Any, Any]:
    K_cache, V_cache = layer_cache
    if start_token < 0 or end_token > K_cache.shape[1] or start_token >= end_token:
        raise ValueError(
            f"Invalid token range [{start_token}, {end_token}) for cache width {K_cache.shape[1]}"
        )
    K_block = K_cache[:, start_token:end_token, :, :]
    V_block = V_cache[:, start_token:end_token, :, :]
    if _is_mlx_array(K_block):
        return (mx.array(K_block), mx.array(V_block))
    return (np.array(K_block, copy=True), np.array(V_block, copy=True))


# ---------------------------------------------------------------------------
# Insert a KV block into a layer cache
# ---------------------------------------------------------------------------

def insert_kv_block(layer_cache: tuple[Any, Any], start_token: int, k_block: Any, v_block: Any) -> tuple[Any, Any]:
    K_cache, V_cache = layer_cache
    num_tokens = k_block.shape[1]
    if start_token < 0 or start_token + num_tokens > K_cache.shape[1]:
        raise ValueError(
            f"Cannot insert {num_tokens} tokens at position {start_token} "
            f"into cache of width {K_cache.shape[1]}"
        )
    new_K = _as_contiguous_copy(K_cache)
    new_V = _as_contiguous_copy(V_cache)
    new_K[:, start_token:start_token + num_tokens, :, :] = k_block
    new_V[:, start_token:start_token + num_tokens, :, :] = v_block
    if _is_mlx_array(new_K):
        return (mx.array(new_K), mx.array(new_V))
    return (new_K, new_V)


# ---------------------------------------------------------------------------
# Offload a KV block to a store
# ---------------------------------------------------------------------------

def offload_kv_block(
    layer_cache: tuple[Any, Any],
    block_meta,
    store,
    *,
    zero_hot: bool = False,
) -> tuple[tuple[Any, Any], dict[str, Any]]:
    K_block, V_block = extract_kv_block(
        layer_cache,
        block_meta.start_token,
        block_meta.end_token,
    )
    store_uri = store.put_block(block_meta.block_id, K_block, V_block)
    block_meta.resident = False
    block_meta.offloaded = True
    block_meta.store_uri = store_uri

    try:
        import hashlib
        k_arr = K_block if isinstance(K_block, np.ndarray) else np.asarray(K_block)
        v_arr = V_block if isinstance(V_block, np.ndarray) else np.asarray(V_block)
        h = hashlib.sha256()
        h.update(k_arr.tobytes())
        h.update(v_arr.tobytes())
        block_meta.checksum = h.hexdigest()[:16]
    except Exception:
        block_meta.checksum = None

    updated_cache = layer_cache
    if zero_hot:
        updated_cache = _zero_block_in_cache(layer_cache, block_meta.start_token, block_meta.end_token)

    return updated_cache, {"block_id": block_meta.block_id.to_string(), "store_uri": store_uri}


# ---------------------------------------------------------------------------
# Prefetch a KV block from a store
# ---------------------------------------------------------------------------

def prefetch_kv_block(
    layer_cache: tuple[Any, Any],
    block_meta,
    store,
) -> tuple[tuple[Any, Any], dict[str, Any]]:
    if not block_meta.offloaded:
        return layer_cache, {"block_id": block_meta.block_id.to_string(), "already_resident": True}

    K_block, V_block = store.get_block(block_meta.block_id)
    updated_cache = insert_kv_block(layer_cache, block_meta.start_token, K_block, V_block)
    block_meta.resident = True
    block_meta.offloaded = False

    return updated_cache, {"block_id": block_meta.block_id.to_string(), "prefetched": True}


# ---------------------------------------------------------------------------
# Apply an offload plan
# ---------------------------------------------------------------------------

def apply_offload_plan(
    layer_caches: list[tuple[Any, Any]],
    residency_map,
    store,
    plan,
    *,
    zero_hot: bool = False,
):
    updated_caches = list(layer_caches)
    offload_results: list[dict[str, Any]] = []
    prefetch_results: list[dict[str, Any]] = []

    for bid in plan.offload:
        layer = bid.layer_idx
        meta = residency_map.get(bid)
        if meta is None or not meta.resident:
            continue
        if layer < len(updated_caches):
            cache, result = offload_kv_block(
                updated_caches[layer], meta, store, zero_hot=zero_hot,
            )
            updated_caches[layer] = cache
            offload_results.append(result)

    for bid in plan.prefetch:
        layer = bid.layer_idx
        meta = residency_map.get(bid)
        if meta is None or not meta.offloaded:
            continue
        if layer < len(updated_caches):
            cache, result = prefetch_kv_block(updated_caches[layer], meta, store)
            updated_caches[layer] = cache
            prefetch_results.append(result)

    return updated_caches, residency_map, {
        "offloaded": offload_results,
        "prefetched": prefetch_results,
        "num_offloaded": len(offload_results),
        "num_prefetched": len(prefetch_results),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zero_block_in_cache(layer_cache: tuple[Any, Any], start: int, end: int) -> tuple[Any, Any]:
    K_cache, V_cache = layer_cache
    new_K = _as_contiguous_copy(K_cache)
    new_V = _as_contiguous_copy(V_cache)
    if _is_mlx_array(new_K):
        new_K[:, start:end, :, :] = mx.zeros((1, end - start, K_cache.shape[2], K_cache.shape[3]), dtype=K_cache.dtype)
        new_V[:, start:end, :, :] = mx.zeros((1, end - start, V_cache.shape[2], V_cache.shape[3]), dtype=V_cache.dtype)
    else:
        new_K[:, start:end, :, :] = 0
        new_V[:, start:end, :, :] = 0
    return (new_K, new_V)
