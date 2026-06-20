from __future__ import annotations

from typing import Optional

import mlx.core as mx

from .decode_ops import decode_attention, reference_decode_attention
from .fused_ops import (
    qkv_rope_cache_update,
    reference_qkv_rope_cache_update,
    reference_rmsnorm_residual,
    rmsnorm_residual,
)
from .gqa_ops import (
    gqa_decode_block_from_qkv,
    paged_gqa_decode_block_from_qkv,
    reference_gqa_decode_block_from_qkv,
    reference_paged_gqa_decode_block_from_qkv,
)
from .kv_cache_ops import normalize_positions
from .layout_ops import qkv_split_rope, reference_qkv_split_rope
from .paged_kv_ops import (
    paged_decode_attention,
    paged_kv_cache_update,
    reference_paged_decode_attention,
    reference_paged_kv_cache_update,
)


def _validate_contiguous_cache(K_cache: mx.array, V_cache: mx.array) -> tuple[int, int, int, int]:
    if K_cache.ndim != 4 or V_cache.ndim != 4 or K_cache.shape != V_cache.shape:
        raise ValueError(f"K_cache and V_cache must be matching [B,MAX_S,H,D], got {K_cache.shape}, {V_cache.shape}")
    return K_cache.shape


def _validate_paged_cache(K_pages: mx.array, V_pages: mx.array, block_table: mx.array) -> tuple[int, int, int, int, int, int]:
    if K_pages.ndim != 4 or V_pages.ndim != 4 or K_pages.shape != V_pages.shape:
        raise ValueError(
            f"K_pages and V_pages must be matching [NUM_PAGES,PAGE_SIZE,H,D], got {K_pages.shape}, {V_pages.shape}"
        )
    if block_table.ndim != 2:
        raise ValueError(f"block_table must be [B,MAX_BLOCKS], got {block_table.shape}")
    num_pages, page_size, heads, head_dim = K_pages.shape
    batch, max_blocks = block_table.shape
    return num_pages, page_size, heads, head_dim, batch, max_blocks


def _normalize_decode_positions(position, B: int, max_positions: int) -> mx.array:
    return normalize_positions(position, B, max_positions)


def _decode_lengths(position, B: int, max_positions: int):
    if isinstance(position, int):
        return position + 1
    return _normalize_decode_positions(position, B, max_positions) + 1


def _rope_decode_qkv(
    qkv: mx.array,
    cos: mx.array,
    sin: mx.array,
    position,
    *,
    H: int | None,
    D: int | None,
    backend: str,
    reference: bool,
):
    split_fn = reference_qkv_split_rope if reference else qkv_split_rope
    B = qkv.shape[0]
    if isinstance(position, int):
        return split_fn(qkv, cos, sin, H=H, D=D, position_offset=position, backend=backend) if not reference else split_fn(
            qkv, cos, sin, H=H, D=D, position_offset=position
        )

    pos_arr = _normalize_decode_positions(position, B, cos.shape[0])
    q_rows = []
    k_rows = []
    v_rows = []
    for b in range(B):
        pos_b = int(pos_arr[b].item())
        q_b, k_b, v_b = (
            split_fn(qkv[b:b + 1], cos, sin, H=H, D=D, position_offset=pos_b, backend=backend)
            if not reference
            else split_fn(qkv[b:b + 1], cos, sin, H=H, D=D, position_offset=pos_b)
        )
        q_rows.append(q_b)
        k_rows.append(k_b)
        v_rows.append(v_b)
    if B == 1:
        return q_rows[0], k_rows[0], v_rows[0]
    return mx.concatenate(q_rows, axis=0), mx.concatenate(k_rows, axis=0), mx.concatenate(v_rows, axis=0)


def reference_decode_block_from_qkv(
    qkv: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    cos: mx.array,
    sin: mx.array,
    position,
    *,
    H: int | None = None,
    D: int | None = None,
    scale: Optional[float] = None,
):
    _, max_s, _, _ = _validate_contiguous_cache(K_cache, V_cache)
    q_rope, updated_K, updated_V = reference_qkv_rope_cache_update(
        qkv, K_cache, V_cache, cos, sin, position, H=H, D=D
    )
    lengths = _decode_lengths(position, updated_K.shape[0], max_s)
    out = reference_decode_attention(q_rope, updated_K, updated_V, lengths=lengths, scale=scale, causal=False)
    return out, updated_K, updated_V


def decode_block_from_qkv(
    qkv: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    cos: mx.array,
    sin: mx.array,
    position,
    *,
    H: int | None = None,
    D: int | None = None,
    scale: Optional[float] = None,
    backend: str = "auto",
):
    B, max_s, _, _ = _validate_contiguous_cache(K_cache, V_cache)
    backend_name = backend.lower()
    if H is not None and K_cache.shape[2] != H:
        if backend_name == "reference":
            return reference_gqa_decode_block_from_qkv(
                qkv,
                K_cache,
                V_cache,
                cos,
                sin,
                position,
                num_attention_heads=H,
                num_key_value_heads=K_cache.shape[2],
                head_dim=D if D is not None else K_cache.shape[3],
                scale=scale,
            )
        return gqa_decode_block_from_qkv(
            qkv,
            K_cache,
            V_cache,
            cos,
            sin,
            position,
            num_attention_heads=H,
            num_key_value_heads=K_cache.shape[2],
            head_dim=D if D is not None else K_cache.shape[3],
            scale=scale,
            backend=backend_name,
        )
    if backend_name == "reference":
        return reference_decode_block_from_qkv(qkv, K_cache, V_cache, cos, sin, position, H=H, D=D, scale=scale)

    q_rope, updated_K, updated_V = qkv_rope_cache_update(
        qkv, K_cache, V_cache, cos, sin, position, H=H, D=D, backend=backend_name
    )
    lengths = _decode_lengths(position, B, max_s)
    out = decode_attention(q_rope, updated_K, updated_V, lengths=lengths, scale=scale, causal=False, backend=backend_name)
    return out, updated_K, updated_V


def reference_paged_decode_block_from_qkv(
    qkv: mx.array,
    K_pages: mx.array,
    V_pages: mx.array,
    block_table: mx.array,
    cos: mx.array,
    sin: mx.array,
    position,
    *,
    H: int | None = None,
    D: int | None = None,
    scale: Optional[float] = None,
):
    _, page_size, _, _, batch, max_blocks = _validate_paged_cache(K_pages, V_pages, block_table)
    q_rope, k_rope, v = _rope_decode_qkv(
        qkv, cos, sin, position, H=H, D=D, backend="reference", reference=True
    )
    updated_K, updated_V = reference_paged_kv_cache_update(K_pages, V_pages, k_rope, v, block_table, position)
    lengths = _decode_lengths(position, batch, max_blocks * page_size)
    out = reference_paged_decode_attention(q_rope, updated_K, updated_V, block_table, lengths, scale=scale)
    return out, updated_K, updated_V


def paged_decode_block_from_qkv(
    qkv: mx.array,
    K_pages: mx.array,
    V_pages: mx.array,
    block_table: mx.array,
    cos: mx.array,
    sin: mx.array,
    position,
    *,
    H: int | None = None,
    D: int | None = None,
    scale: Optional[float] = None,
    backend: str = "auto",
):
    _, page_size, _, _, batch, max_blocks = _validate_paged_cache(K_pages, V_pages, block_table)
    backend_name = backend.lower()
    if H is not None and K_pages.shape[2] != H:
        if backend_name == "reference":
            return reference_paged_gqa_decode_block_from_qkv(
                qkv,
                K_pages,
                V_pages,
                block_table,
                cos,
                sin,
                position,
                num_attention_heads=H,
                num_key_value_heads=K_pages.shape[2],
                head_dim=D if D is not None else K_pages.shape[3],
                scale=scale,
            )
        return paged_gqa_decode_block_from_qkv(
            qkv,
            K_pages,
            V_pages,
            block_table,
            cos,
            sin,
            position,
            num_attention_heads=H,
            num_key_value_heads=K_pages.shape[2],
            head_dim=D if D is not None else K_pages.shape[3],
            scale=scale,
            backend=backend_name,
        )
    if backend_name == "reference":
        return reference_paged_decode_block_from_qkv(
            qkv, K_pages, V_pages, block_table, cos, sin, position, H=H, D=D, scale=scale
        )

    q_rope, k_rope, v = _rope_decode_qkv(
        qkv, cos, sin, position, H=H, D=D, backend=backend_name, reference=False
    )
    updated_K, updated_V = paged_kv_cache_update(K_pages, V_pages, k_rope, v, block_table, position, backend=backend_name)
    lengths = _decode_lengths(position, batch, max_blocks * page_size)
    out = paged_decode_attention(q_rope, updated_K, updated_V, block_table, lengths, scale=scale, backend=backend_name)
    return out, updated_K, updated_V


def residual_rmsnorm_block(
    x: mx.array,
    residual: mx.array,
    weight: mx.array,
    eps: float = 1.0e-5,
    *,
    return_residual: bool = False,
    backend: str = "auto",
):
    return rmsnorm_residual(
        x, residual, weight, eps=eps, return_residual=return_residual, backend=backend
    )


def reference_residual_rmsnorm_block(
    x: mx.array,
    residual: mx.array,
    weight: mx.array,
    eps: float = 1.0e-5,
    *,
    return_residual: bool = False,
):
    return reference_rmsnorm_residual(x, residual, weight, eps=eps, return_residual=return_residual)


def decode_block_with_residual_norm(
    qkv: mx.array,
    residual: mx.array,
    norm_weight: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    cos: mx.array,
    sin: mx.array,
    position,
    *,
    H: int | None = None,
    D: int | None = None,
    scale: Optional[float] = None,
    eps: float = 1.0e-5,
    backend: str = "auto",
):
    out, updated_K, updated_V = decode_block_from_qkv(
        qkv, K_cache, V_cache, cos, sin, position, H=H, D=D, scale=scale, backend=backend
    )
    if residual.shape == out.shape:
        y = residual_rmsnorm_block(out, residual, norm_weight, eps=eps, backend=backend)
        return y, updated_K, updated_V
    if residual.ndim == 3 and residual.shape[:2] == out.shape[:2] and residual.shape[-1] == out.shape[2] * out.shape[3]:
        flat_out = out.reshape(out.shape[0], out.shape[1], out.shape[2] * out.shape[3])
        y = residual_rmsnorm_block(flat_out, residual, norm_weight, eps=eps, backend=backend)
        return y, updated_K, updated_V
    raise ValueError(
        "residual must either match decode output [B,1,H,D] or flattened output [B,1,H*D]. "
        f"Got residual={residual.shape}, out={out.shape}."
    )
