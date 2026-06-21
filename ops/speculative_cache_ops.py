from __future__ import annotations

from typing import Any

try:
    import mlx.core as mx
except ImportError:
    mx = None

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError("numpy is required for speculative_cache_ops") from exc


def _is_mlx_array(value: Any) -> bool:
    return mx is not None and type(value).__module__.startswith("mlx")


def _as_contiguous_copy(x: Any) -> Any:
    if _is_mlx_array(x):
        return mx.array(x)
    return np.array(x, copy=True)


def _clone_stack_cache(stack_cache):
    try:
        from ops.kv_cache_reuse_ops import clone_stack_cache as _csc
    except ImportError:
        raise RuntimeError("clone_stack_cache not available from kv_cache_reuse_ops")
    return _csc(stack_cache)


def commit_accepted_cache(
    draft_cache,
    committed_cache,
    accepted_count: int,
) -> Any:
    if draft_cache.cache_layout != committed_cache.cache_layout:
        raise ValueError(
            f"cache_layout mismatch: draft={draft_cache.cache_layout}, "
            f"committed={committed_cache.cache_layout}"
        )
    if len(draft_cache.layer_caches) != len(committed_cache.layer_caches):
        raise ValueError(
            f"layer count mismatch: draft={len(draft_cache.layer_caches)}, "
            f"committed={len(committed_cache.layer_caches)}"
        )
    new_layer_caches = []
    for draft_lc, commit_lc in zip(draft_cache.layer_caches, committed_cache.layer_caches):
        draft_K, draft_V = draft_lc
        commit_K, commit_V = commit_lc
        new_K = _as_contiguous_copy(commit_K)
        new_V = _as_contiguous_copy(commit_V)
        if accepted_count == 0:
            pass
        elif accepted_count <= new_K.shape[1]:
            if _is_mlx_array(new_K):
                new_K[:, :accepted_count] = draft_K[:, :accepted_count]
                new_V[:, :accepted_count] = draft_V[:, :accepted_count]
            else:
                new_K[:, :accepted_count] = draft_K[:, :accepted_count]
                new_V[:, :accepted_count] = draft_V[:, :accepted_count]
        else:
            raise ValueError(
                f"accepted_count={accepted_count} exceeds cache width {new_K.shape[1]}"
            )
        new_layer_caches.append((new_K, new_V))
    try:
        from ops.llama_stack_ops import LlamaStackCache as LSC
    except ImportError:
        raise RuntimeError("LlamaStackCache not available")
    return LSC(
        layer_caches=new_layer_caches,
        cache_layout=committed_cache.cache_layout,
        max_seq_len=committed_cache.max_seq_len,
        page_size=committed_cache.page_size,
    )


def discard_suffix(draft_cache, suffix_start: int) -> Any:
    if draft_cache.cache_layout == "paged":
        raise NotImplementedError("discard_suffix for paged cache is not implemented yet")
    return _slice_cache(draft_cache, suffix_start)


def _slice_cache(cache, length: int) -> Any:
    new_layer_caches = []
    for K, V in cache.layer_caches:
        K_sliced = K[:, :length, :, :]
        V_sliced = V[:, :length, :, :]
        if _is_mlx_array(K_sliced):
            new_layer_caches.append((mx.array(K_sliced), mx.array(V_sliced)))
        else:
            new_layer_caches.append((np.array(K_sliced, copy=True), np.array(V_sliced, copy=True)))
    try:
        from ops.llama_stack_ops import LlamaStackCache as LSC
    except ImportError:
        raise RuntimeError("LlamaStackCache not available")
    return LSC(
        layer_caches=new_layer_caches,
        cache_layout=cache.cache_layout,
        max_seq_len=cache.max_seq_len,
        page_size=cache.page_size,
    )
