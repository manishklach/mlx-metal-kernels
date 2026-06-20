from __future__ import annotations

from functools import lru_cache
import math
from typing import Optional

import mlx.core as mx

from .attention_ops import reference_attention
from .decode_ops import decode_attention, reference_decode_attention
from .kernel_utils import KERNEL_DIR, make_metal_header, load_metal_source
from .kv_cache_ops import kv_cache_update, normalize_positions, reference_kv_cache_update
from .paged_kv_ops import (
    paged_kv_cache_update,
    reference_block_table_lookup,
    reference_paged_kv_cache_update,
)
from .rope_ops import apply_rope, reference_apply_rope

_KERNEL_DIR = KERNEL_DIR
_GQA_ATTENTION_KERNEL = _KERNEL_DIR / "gqa_attention.metal"
_GQA_ATTENTION_THREADGROUP_KERNEL = _KERNEL_DIR / "gqa_attention_threadgroup.metal"
_GQA_MAX_HEAD_DIM = 128
_GQA_THREADGROUP_THREADS = 128


def _make_header(dtype: mx.Dtype) -> str:
    return make_metal_header(dtype, MAX_HEAD_DIM=_GQA_MAX_HEAD_DIM, TG_THREADS=_GQA_THREADGROUP_THREADS)


@lru_cache(maxsize=8)
def _get_gqa_attention_kernel(kernel_name: str, dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name=kernel_name,
        input_names=["Q", "K", "V", "meta", "scale"],
        output_names=["O"],
        source=source,
        header=header,
    )


def validate_gqa_heads(num_attention_heads: int, num_key_value_heads: int) -> None:
    if num_key_value_heads < 1:
        raise ValueError(f"num_key_value_heads must be >= 1, got {num_key_value_heads}")
    if num_attention_heads < num_key_value_heads:
        raise ValueError(
            "num_attention_heads must be >= num_key_value_heads, "
            f"got {num_attention_heads}, {num_key_value_heads}"
        )
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            "num_attention_heads must be divisible by num_key_value_heads, "
            f"got {num_attention_heads}, {num_key_value_heads}"
        )


def gqa_group_size(num_attention_heads: int, num_key_value_heads: int) -> int:
    validate_gqa_heads(num_attention_heads, num_key_value_heads)
    return num_attention_heads // num_key_value_heads


def q_head_to_kv_head(q_head: int, num_attention_heads: int, num_key_value_heads: int) -> int:
    group = gqa_group_size(num_attention_heads, num_key_value_heads)
    if q_head < 0 or q_head >= num_attention_heads:
        raise ValueError(f"q_head must be in [0, {num_attention_heads}), got {q_head}")
    return q_head // group


def expand_kv_heads_reference(kv: mx.array, num_attention_heads: int) -> mx.array:
    if kv.ndim != 4:
        raise ValueError(f"kv must have shape [B,S,Hkv,D], got {kv.shape}")
    _, _, num_key_value_heads, _ = kv.shape
    group = gqa_group_size(num_attention_heads, num_key_value_heads)
    expanded = mx.repeat(kv, repeats=group, axis=2)
    if expanded.shape[2] != num_attention_heads:
        raise ValueError(
            f"expanded KV heads must equal num_attention_heads={num_attention_heads}, got {expanded.shape[2]}"
        )
    return expanded


def maybe_expand_kv_heads_reference(k: mx.array, v: mx.array, num_attention_heads: int) -> tuple[mx.array, mx.array]:
    if k.shape != v.shape:
        raise ValueError(f"k and v must have matching shapes, got {k.shape}, {v.shape}")
    if k.shape[2] == num_attention_heads:
        return k, v
    return expand_kv_heads_reference(k, num_attention_heads), expand_kv_heads_reference(v, num_attention_heads)


def _validate_gqa_attention_inputs(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    *,
    causal: bool,
    require_metal_supported: bool = False,
) -> tuple[int, int, int, int, int, int]:
    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        raise ValueError(f"Q, K, and V must be rank-4 [B,S,H,D], got {Q.shape}, {K.shape}, {V.shape}")
    if K.shape != V.shape:
        raise ValueError(f"K and V must have identical shapes, got {K.shape}, {V.shape}")
    B, Sq, Hq, D = Q.shape
    Kb, Sk, Hkv, Dk = K.shape
    if B != Kb or D != Dk:
        raise ValueError(f"Q, K, and V must agree on batch and head_dim, got {Q.shape}, {K.shape}, {V.shape}")
    validate_gqa_heads(Hq, Hkv)
    if causal and Sq != Sk:
        raise ValueError(f"causal GQA prefill currently requires Sq == Sk, got Sq={Sq}, Sk={Sk}")
    if Q.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"Q dtype must be float16 or bfloat16, got {Q.dtype}")
    if K.dtype not in (mx.float16, mx.bfloat16) or V.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"K/V dtype must be float16 or bfloat16, got {K.dtype}, {V.dtype}")
    if require_metal_supported and D > _GQA_MAX_HEAD_DIM:
        raise ValueError(f"Metal GQA prefill currently supports D <= {_GQA_MAX_HEAD_DIM}, got D={D}")
    return B, Sq, Sk, Hq, Hkv, D


def reference_gqa_qkv_split(
    qkv,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
):
    validate_gqa_heads(num_attention_heads, num_key_value_heads)
    q_dim = num_attention_heads * head_dim
    kv_dim = num_key_value_heads * head_dim
    if qkv.ndim == 3:
        expected = q_dim + 2 * kv_dim
        if qkv.shape[-1] != expected:
            raise ValueError(f"packed gqa qkv last dimension must be {expected}, got {qkv.shape[-1]}")
        q = qkv[:, :, :q_dim].reshape(qkv.shape[0], qkv.shape[1], num_attention_heads, head_dim)
        k = qkv[:, :, q_dim:q_dim + kv_dim].reshape(qkv.shape[0], qkv.shape[1], num_key_value_heads, head_dim)
        v = qkv[:, :, q_dim + kv_dim:].reshape(qkv.shape[0], qkv.shape[1], num_key_value_heads, head_dim)
        return q.astype(qkv.dtype), k.astype(qkv.dtype), v.astype(qkv.dtype)
    if qkv.ndim == 4:
        if qkv.shape[2] != num_attention_heads + 2 * num_key_value_heads or qkv.shape[3] != head_dim:
            raise ValueError(
                "explicit gqa qkv layout must have shape "
                f"[B,S,{num_attention_heads + 2 * num_key_value_heads},{head_dim}], got {qkv.shape}"
            )
        q = qkv[:, :, :num_attention_heads, :]
        k = qkv[:, :, num_attention_heads:num_attention_heads + num_key_value_heads, :]
        v = qkv[:, :, num_attention_heads + num_key_value_heads:, :]
        return q.astype(qkv.dtype), k.astype(qkv.dtype), v.astype(qkv.dtype)
    raise ValueError(f"qkv must have shape [B,S,Q+K+V] or [B,S,Hq+Hkv+Hkv,D], got {qkv.shape}")


def reference_gqa_qkv_split_rope(
    qkv,
    cos,
    sin,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    position_offset=0,
):
    q, k, v = reference_gqa_qkv_split(qkv, num_attention_heads, num_key_value_heads, head_dim)
    q_rope = reference_apply_rope(q, cos, sin, position_offset=position_offset)
    k_rope = reference_apply_rope(k, cos, sin, position_offset=position_offset)
    return q_rope, k_rope, v


def reference_gqa_attention_via_expansion(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    *,
    causal: bool = False,
    scale: Optional[float] = None,
) -> mx.array:
    _, _, _, Hq, _, D = _validate_gqa_attention_inputs(Q, K, V, causal=causal)
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    K_exp, V_exp = maybe_expand_kv_heads_reference(K, V, Hq)
    return reference_attention(Q, K_exp, V_exp, scale=scale, causal=causal)


def reference_gqa_attention(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    *,
    causal: bool = False,
    scale: Optional[float] = None,
) -> mx.array:
    B, Sq, Sk, Hq, Hkv, D = _validate_gqa_attention_inputs(Q, K, V, causal=causal)
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    Qf = Q.astype(mx.float32)
    Kf = K.astype(mx.float32)
    Vf = V.astype(mx.float32)
    group = gqa_group_size(Hq, Hkv)
    outputs = []
    for hq in range(Hq):
        hkv = hq // group
        q_head = Qf[:, :, hq:hq + 1, :]
        k_head = Kf[:, :, hkv:hkv + 1, :]
        v_head = Vf[:, :, hkv:hkv + 1, :]
        scores = mx.matmul(q_head.transpose(0, 2, 1, 3), k_head.transpose(0, 2, 3, 1)) * float(scale)
        if causal:
            i = mx.arange(Sq)[:, None]
            j = mx.arange(Sk)[None, :]
            scores = mx.where(j > i, mx.array(-1.0e9, dtype=scores.dtype), scores)
        probs = mx.softmax(scores, axis=-1)
        out_head = mx.matmul(probs, v_head.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        outputs.append(out_head)
    return mx.concatenate(outputs, axis=2).astype(Q.dtype)


def gqa_attention(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    *,
    causal: bool = False,
    scale: Optional[float] = None,
    backend: str = "reference",
) -> mx.array:
    B, Sq, Sk, Hq, Hkv, D = _validate_gqa_attention_inputs(
        Q,
        K,
        V,
        causal=causal,
        require_metal_supported=backend != "reference",
    )
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    backend_name = backend.lower()
    if backend_name == "reference":
        return reference_gqa_attention(Q, K, V, causal=causal, scale=scale)
    if backend_name not in ("metal_gqa", "metal_gqa_threadgroup"):
        raise ValueError("backend must be one of 'reference', 'metal_gqa', 'metal_gqa_threadgroup'")

    kernel_path = _GQA_ATTENTION_KERNEL if backend_name == "metal_gqa" else _GQA_ATTENTION_THREADGROUP_KERNEL
    kernel_name = "gqa_attention_forward" if backend_name == "metal_gqa" else "gqa_attention_threadgroup_forward"
    threadgroup = (1, 1, 1) if backend_name == "metal_gqa" else (_GQA_THREADGROUP_THREADS, 1, 1)
    total_rows = B * Sq * Hq
    grid_x = total_rows if backend_name == "metal_gqa" else total_rows * _GQA_THREADGROUP_THREADS
    dtype = Q.dtype
    meta = mx.array([B, Sq, Sk, Hq, Hkv, D, int(causal)], dtype=mx.int32)
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    source = load_metal_source(kernel_path)
    header = _make_header(dtype)
    kernel = _get_gqa_attention_kernel(kernel_name, str(dtype), source, header)
    return kernel(
        inputs=[Q.astype(dtype), K.astype(dtype), V.astype(dtype), meta, scale_arr],
        output_shapes=[(B, Sq, Hq, D)],
        output_dtypes=[dtype],
        grid=(grid_x, 1, 1),
        threadgroup=threadgroup,
    )[0]


def _validate_gqa_decode_inputs(
    q: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    lengths,
) -> tuple[mx.array, int, int, int, int, int]:
    if q.ndim != 4 or K_cache.ndim != 4 or V_cache.ndim != 4:
        raise ValueError(f"q, K_cache, and V_cache must be 4-D, got {q.shape}, {K_cache.shape}, {V_cache.shape}")
    if q.shape[1] != 1:
        raise ValueError(f"q must have shape [B,1,Hq,D], got {q.shape}")
    if K_cache.shape != V_cache.shape:
        raise ValueError(f"K_cache and V_cache must match, got {K_cache.shape}, {V_cache.shape}")
    B, _, Hq, D = q.shape
    cache_B, max_s, Hkv, cache_D = K_cache.shape
    if B != cache_B or D != cache_D:
        raise ValueError(
            f"q and caches must agree on batch/head_dim, got q={q.shape}, K_cache={K_cache.shape}, V_cache={V_cache.shape}"
        )
    validate_gqa_heads(Hq, Hkv)
    lengths_arr = normalize_positions(lengths if lengths is not None else max_s, B, max_s + 1)
    lengths_arr = mx.minimum(lengths_arr, mx.array(max_s, dtype=mx.int32))
    return lengths_arr, B, max_s, Hq, Hkv, D


def reference_gqa_decode_attention(
    q: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    lengths=None,
    scale: Optional[float] = None,
) -> mx.array:
    lengths_arr, B, max_s, Hq, Hkv, D = _validate_gqa_decode_inputs(q, K_cache, V_cache, lengths)
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    qf = q.astype(mx.float32)
    Kf = K_cache.astype(mx.float32)
    Vf = V_cache.astype(mx.float32)
    outputs = []
    positions = mx.arange(max_s).reshape(1, max_s)
    for hq in range(Hq):
        hkv = q_head_to_kv_head(hq, Hq, Hkv)
        q_head = qf[:, :, hq:hq + 1, :]
        k_head = Kf[:, :, hkv:hkv + 1, :]
        v_head = Vf[:, :, hkv:hkv + 1, :]
        scores = mx.matmul(q_head.transpose(0, 2, 1, 3), k_head.transpose(0, 2, 3, 1)) * float(scale)
        valid_mask = positions.reshape(1, 1, 1, max_s) < lengths_arr.reshape(B, 1, 1, 1)
        scores = mx.where(valid_mask, scores, mx.array(-1.0e9, dtype=scores.dtype))
        probs = mx.softmax(scores, axis=-1)
        out_head = mx.matmul(probs, v_head.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        outputs.append(out_head)
    return mx.concatenate(outputs, axis=2).astype(q.dtype)


def reference_paged_gqa_decode_attention(
    q: mx.array,
    K_pages: mx.array,
    V_pages: mx.array,
    block_table: mx.array,
    lengths=None,
    scale: Optional[float] = None,
) -> mx.array:
    if K_pages.ndim != 4 or V_pages.ndim != 4 or K_pages.shape != V_pages.shape:
        raise ValueError(f"K_pages and V_pages must match [NUM_PAGES,PAGE_SIZE,Hkv,D], got {K_pages.shape}, {V_pages.shape}")
    if block_table.ndim != 2:
        raise ValueError(f"block_table must be [B,MAX_BLOCKS], got {block_table.shape}")
    B = block_table.shape[0]
    max_s = block_table.shape[1] * K_pages.shape[1]
    lengths_arr, _, _, Hq, Hkv, D = _validate_gqa_decode_inputs(
        q,
        mx.zeros((B, max_s, K_pages.shape[2], K_pages.shape[3]), dtype=q.dtype),
        mx.zeros((B, max_s, V_pages.shape[2], V_pages.shape[3]), dtype=q.dtype),
        lengths,
    )
    page_size = K_pages.shape[1]
    gathered_K = []
    gathered_V = []
    for b in range(B):
        rows_k = []
        rows_v = []
        for j in range(max_s):
            page_id = int(block_table[b, j // page_size].item())
            offset = j % page_size
            rows_k.append(K_pages[page_id:page_id + 1, offset:offset + 1, :, :])
            rows_v.append(V_pages[page_id:page_id + 1, offset:offset + 1, :, :])
        gathered_K.append(mx.concatenate(rows_k, axis=1))
        gathered_V.append(mx.concatenate(rows_v, axis=1))
    K_contig = mx.concatenate(gathered_K, axis=0)
    V_contig = mx.concatenate(gathered_V, axis=0)
    return reference_gqa_decode_attention(q, K_contig, V_contig, lengths=lengths_arr, scale=scale)


def reference_gqa_decode_block_from_qkv(
    qkv,
    K_cache,
    V_cache,
    cos,
    sin,
    position,
    *,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    scale=None,
):
    q_rope, k_rope, v = reference_gqa_qkv_split_rope(
        qkv,
        cos,
        sin,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        position_offset=0 if not isinstance(position, int) else position,
    )
    if not isinstance(position, int):
        rows_q = []
        rows_k = []
        pos_arr = normalize_positions(position, qkv.shape[0], cos.shape[0])
        for b in range(qkv.shape[0]):
            q_b, k_b, _ = reference_gqa_qkv_split_rope(
                qkv[b:b + 1],
                cos,
                sin,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                position_offset=int(pos_arr[b].item()),
            )
            rows_q.append(q_b)
            rows_k.append(k_b)
        q_rope = mx.concatenate(rows_q, axis=0) if qkv.shape[0] > 1 else rows_q[0]
        k_rope = mx.concatenate(rows_k, axis=0) if qkv.shape[0] > 1 else rows_k[0]
    updated_K, updated_V = reference_kv_cache_update(K_cache, V_cache, k_rope, v, position)
    lengths = position + 1 if isinstance(position, int) else normalize_positions(position, qkv.shape[0], K_cache.shape[1]) + 1
    out = reference_gqa_decode_attention(q_rope, updated_K, updated_V, lengths=lengths, scale=scale)
    return out, updated_K, updated_V


def gqa_decode_block_from_qkv(
    qkv,
    K_cache,
    V_cache,
    cos,
    sin,
    position,
    *,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    scale=None,
    backend="auto",
):
    backend_name = backend.lower()
    if backend_name not in ("auto", "reference", "metal"):
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")
    q_rope, k_rope, v = reference_gqa_qkv_split_rope(
        qkv,
        cos,
        sin,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        position_offset=0 if not isinstance(position, int) else position,
    )
    if not isinstance(position, int):
        rows_q = []
        rows_k = []
        pos_arr = normalize_positions(position, qkv.shape[0], cos.shape[0])
        for b in range(qkv.shape[0]):
            q_b, k_b, _ = reference_gqa_qkv_split_rope(
                qkv[b:b + 1],
                cos,
                sin,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                position_offset=int(pos_arr[b].item()),
            )
            rows_q.append(q_b)
            rows_k.append(k_b)
        q_rope = mx.concatenate(rows_q, axis=0) if qkv.shape[0] > 1 else rows_q[0]
        k_rope = mx.concatenate(rows_k, axis=0) if qkv.shape[0] > 1 else rows_k[0]
    updated_K, updated_V = (
        reference_kv_cache_update(K_cache, V_cache, k_rope, v, position)
        if backend_name == "reference"
        else kv_cache_update(K_cache, V_cache, k_rope, v, position)
    )
    lengths = position + 1 if isinstance(position, int) else normalize_positions(position, qkv.shape[0], K_cache.shape[1]) + 1
    return reference_gqa_decode_attention(q_rope, updated_K, updated_V, lengths=lengths, scale=scale), updated_K, updated_V


def reference_paged_gqa_decode_block_from_qkv(
    qkv,
    K_pages,
    V_pages,
    block_table,
    cos,
    sin,
    position,
    *,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    scale=None,
):
    q_rope, k_rope, v = reference_gqa_qkv_split_rope(
        qkv,
        cos,
        sin,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        position_offset=0 if not isinstance(position, int) else position,
    )
    if not isinstance(position, int):
        pos_arr = normalize_positions(position, qkv.shape[0], cos.shape[0])
        q_rows = []
        k_rows = []
        for b in range(qkv.shape[0]):
            q_b, k_b, _ = reference_gqa_qkv_split_rope(
                qkv[b:b + 1],
                cos,
                sin,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                position_offset=int(pos_arr[b].item()),
            )
            q_rows.append(q_b)
            k_rows.append(k_b)
        q_rope = mx.concatenate(q_rows, axis=0) if qkv.shape[0] > 1 else q_rows[0]
        k_rope = mx.concatenate(k_rows, axis=0) if qkv.shape[0] > 1 else k_rows[0]
    updated_K, updated_V = reference_paged_kv_cache_update(K_pages, V_pages, k_rope, v, block_table, position)
    lengths = position + 1 if isinstance(position, int) else normalize_positions(position, qkv.shape[0], block_table.shape[1] * K_pages.shape[1]) + 1
    out = reference_paged_gqa_decode_attention(q_rope, updated_K, updated_V, block_table, lengths=lengths, scale=scale)
    return out, updated_K, updated_V


def paged_gqa_decode_block_from_qkv(
    qkv,
    K_pages,
    V_pages,
    block_table,
    cos,
    sin,
    position,
    *,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    scale=None,
    backend="auto",
):
    backend_name = backend.lower()
    if backend_name not in ("auto", "reference", "metal"):
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")
    q_rope, k_rope, v = reference_gqa_qkv_split_rope(
        qkv,
        cos,
        sin,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        position_offset=0 if not isinstance(position, int) else position,
    )
    if not isinstance(position, int):
        pos_arr = normalize_positions(position, qkv.shape[0], cos.shape[0])
        q_rows = []
        k_rows = []
        for b in range(qkv.shape[0]):
            q_b, k_b, _ = reference_gqa_qkv_split_rope(
                qkv[b:b + 1],
                cos,
                sin,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                position_offset=int(pos_arr[b].item()),
            )
            q_rows.append(q_b)
            k_rows.append(k_b)
        q_rope = mx.concatenate(q_rows, axis=0) if qkv.shape[0] > 1 else q_rows[0]
        k_rope = mx.concatenate(k_rows, axis=0) if qkv.shape[0] > 1 else k_rows[0]
    updated_K, updated_V = (
        reference_paged_kv_cache_update(K_pages, V_pages, k_rope, v, block_table, position)
        if backend_name == "reference"
        else paged_kv_cache_update(K_pages, V_pages, k_rope, v, block_table, position)
    )
    lengths = position + 1 if isinstance(position, int) else normalize_positions(position, qkv.shape[0], block_table.shape[1] * K_pages.shape[1]) + 1
    out = reference_paged_gqa_decode_attention(q_rope, updated_K, updated_V, block_table, lengths=lengths, scale=scale)
    return out, updated_K, updated_V
