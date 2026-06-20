from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import mlx.core as mx

from .kv_cache_ops import _normalize_token_shape, normalize_positions

_KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernels"
_PAGED_UPDATE_KERNEL = _KERNEL_DIR / "paged_kv_cache_update.metal"
_PAGED_DECODE_KERNEL = _KERNEL_DIR / "paged_decode_attention.metal"
_PAGED_DECODE_THREADGROUP_KERNEL = _KERNEL_DIR / "paged_decode_attention_threadgroup.metal"
_SPECIALIZED_PAGED_DECODE_KERNELS = {
    "metal_d64": _KERNEL_DIR / "paged_decode_attention_d64.metal",
    "metal_d128": _KERNEL_DIR / "paged_decode_attention_d128.metal",
}
_BLOCK_LOOKUP_KERNEL = _KERNEL_DIR / "block_table_lookup.metal"
_THREADS = 256
_THREADGROUP_THREADS = 128


def _make_header(dtype: mx.Dtype, *, max_head_dim: int = 128, fixed_head_dim: int | None = None) -> str:
    if dtype == mx.bfloat16:
        elem_type = "bfloat"
    elif dtype == mx.float16:
        elem_type = "half"
    else:
        raise TypeError(f"paged_kv_ops support only float16/bfloat16 caches, got {dtype}")
    fixed_dim_line = f"#define HEAD_DIM {fixed_head_dim}" if fixed_head_dim is not None else ""
    return f"""
#include <metal_stdlib>
using namespace metal;
#define ELEM_TYPE {elem_type}
#define MAX_HEAD_DIM {max_head_dim}
#define TG_THREADS {_THREADGROUP_THREADS}
{fixed_dim_line}
"""


def _load_source(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing Metal kernel source: {path}")
    return path.read_text()


@lru_cache(maxsize=8)
def _get_block_lookup_kernel(source: str):
    return mx.fast.metal_kernel(
        name="block_table_lookup_forward",
        input_names=["block_table", "positions", "meta"],
        output_names=["page_ids", "offsets"],
        source=source,
        header="""
#include <metal_stdlib>
using namespace metal;
""",
    )


@lru_cache(maxsize=8)
def _get_paged_update_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="paged_kv_cache_update_forward",
        input_names=["K_pages", "V_pages", "k_new", "v_new", "block_table", "positions", "meta"],
        output_names=["updated_K_pages", "updated_V_pages"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=8)
def _get_paged_decode_kernel(dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name="paged_decode_attention_forward",
        input_names=["q", "K_pages", "V_pages", "block_table", "lengths", "meta", "scale"],
        output_names=["out"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=8)
def _get_named_paged_decode_kernel(kernel_name: str, dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name=kernel_name,
        input_names=["q", "K_pages", "V_pages", "block_table", "lengths", "meta", "scale"],
        output_names=["out"],
        source=source,
        header=header,
    )


def _resolve_backend(backend_name: str, D: int) -> str:
    if backend_name == "auto":
        if os.environ.get("MLX_METAL_USE_THREADGROUP_ATTENTION", "0") == "1":
            return "metal_threadgroup"
        if os.environ.get("MLX_METAL_USE_SPECIALIZED", "0") == "1":
            if D == 64:
                return "metal_d64"
            if D == 128:
                return "metal_d128"
        return "metal"
    if backend_name == "metal_d64" and D != 64:
        raise ValueError(f"backend='metal_d64' requires D == 64, got D={D}")
    if backend_name == "metal_d128" and D != 128:
        raise ValueError(f"backend='metal_d128' requires D == 128, got D={D}")
    return backend_name


def make_identity_block_table(B: int, MAX_S: int, PAGE_SIZE: int) -> mx.array:
    if B <= 0 or MAX_S <= 0 or PAGE_SIZE <= 0:
        raise ValueError(f"B, MAX_S, and PAGE_SIZE must be positive, got {B}, {MAX_S}, {PAGE_SIZE}")
    max_blocks = math.ceil(MAX_S / PAGE_SIZE)
    base = mx.arange(B, dtype=mx.int32).reshape(B, 1) * max_blocks
    block_ids = mx.arange(max_blocks, dtype=mx.int32).reshape(1, max_blocks)
    return base + block_ids


def allocate_paged_kv_cache(B: int, MAX_S: int, H: int, D: int, PAGE_SIZE: int, dtype):
    if dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"dtype must be float16 or bfloat16, got {dtype}")
    block_table = make_identity_block_table(B, MAX_S, PAGE_SIZE)
    num_pages = B * math.ceil(MAX_S / PAGE_SIZE)
    K_pages = mx.zeros((num_pages, PAGE_SIZE, H, D), dtype=dtype)
    V_pages = mx.zeros((num_pages, PAGE_SIZE, H, D), dtype=dtype)
    return K_pages, V_pages, block_table


def _validate_page_tensors(K_pages: mx.array, V_pages: mx.array, block_table: mx.array):
    if K_pages.ndim != 4 or V_pages.ndim != 4:
        raise ValueError(f"K_pages and V_pages must be 4-D [NUM_PAGES,PAGE_SIZE,H,D], got {K_pages.shape}, {V_pages.shape}")
    if K_pages.shape != V_pages.shape:
        raise ValueError(f"K_pages and V_pages must match, got {K_pages.shape}, {V_pages.shape}")
    if block_table.ndim != 2:
        raise ValueError(f"block_table must be 2-D [B,MAX_BLOCKS], got {block_table.shape}")
    NUM_PAGES, PAGE_SIZE, H, D = K_pages.shape
    B, MAX_BLOCKS = block_table.shape
    if K_pages.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"K_pages dtype must be float16 or bfloat16, got {K_pages.dtype}")
    if V_pages.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"V_pages dtype must be float16 or bfloat16, got {V_pages.dtype}")
    return NUM_PAGES, PAGE_SIZE, H, D, B, MAX_BLOCKS


def _validate_positions_against_blocks(positions, B: int, max_positions: int):
    return normalize_positions(positions, B, max_positions)


def reference_block_table_lookup(block_table: mx.array, positions, PAGE_SIZE: int):
    if PAGE_SIZE <= 0:
        raise ValueError("PAGE_SIZE must be positive")
    if block_table.ndim != 2:
        raise ValueError(f"block_table must be 2-D [B,MAX_BLOCKS], got {block_table.shape}")
    B, MAX_BLOCKS = block_table.shape
    positions_arr = _validate_positions_against_blocks(positions, B, MAX_BLOCKS * PAGE_SIZE)
    block_ids = positions_arr // PAGE_SIZE
    offsets = positions_arr % PAGE_SIZE
    page_ids_rows = []
    for b in range(B):
        page_ids_rows.append(block_table[b:b + 1, block_ids[b]:block_ids[b] + 1])
    page_ids = mx.concatenate(page_ids_rows, axis=0).reshape(B)
    return page_ids.astype(mx.int32), offsets.astype(mx.int32)


def block_table_lookup(block_table: mx.array, positions, PAGE_SIZE: int, *, backend: str = "auto"):
    if PAGE_SIZE <= 0:
        raise ValueError("PAGE_SIZE must be positive")
    if block_table.ndim != 2:
        raise ValueError(f"block_table must be 2-D [B,MAX_BLOCKS], got {block_table.shape}")
    B, MAX_BLOCKS = block_table.shape
    positions_arr = _validate_positions_against_blocks(positions, B, MAX_BLOCKS * PAGE_SIZE)
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if backend_name == "reference":
        return reference_block_table_lookup(block_table, positions_arr, PAGE_SIZE)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")

    source = _load_source(_BLOCK_LOOKUP_KERNEL)
    kernel = _get_block_lookup_kernel(source)
    meta = mx.array([B, MAX_BLOCKS, PAGE_SIZE], dtype=mx.int32)
    outputs = kernel(
        inputs=[block_table.astype(mx.int32), positions_arr.astype(mx.int32), meta],
        output_shapes=[(B,), (B,)],
        output_dtypes=[mx.int32, mx.int32],
        grid=(B, 1, 1),
        threadgroup=(1, 1, 1),
    )
    return outputs[0], outputs[1]


def reference_paged_kv_cache_update(
    K_pages: mx.array,
    V_pages: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    block_table: mx.array,
    positions,
):
    k_new = _normalize_token_shape(k_new, "k_new")
    v_new = _normalize_token_shape(v_new, "v_new")
    NUM_PAGES, PAGE_SIZE, H, D, B, MAX_BLOCKS = _validate_page_tensors(K_pages, V_pages, block_table)
    if k_new.shape != (B, 1, H, D) or v_new.shape != (B, 1, H, D):
        raise ValueError(
            f"k_new/v_new must normalize to [B,1,H,D]={B,1,H,D}, got {k_new.shape}, {v_new.shape}"
        )
    positions_arr = _validate_positions_against_blocks(positions, B, MAX_BLOCKS * PAGE_SIZE)
    page_ids, offsets = reference_block_table_lookup(block_table, positions_arr, PAGE_SIZE)

    page_idx = mx.arange(NUM_PAGES, dtype=mx.int32).reshape(NUM_PAGES, 1, 1, 1)
    offset_idx = mx.arange(PAGE_SIZE, dtype=mx.int32).reshape(1, PAGE_SIZE, 1, 1)
    K_updated = K_pages
    V_updated = V_pages
    for b in range(B):
        match = (page_idx == page_ids[b]) & (offset_idx == offsets[b])
        k_fill = mx.broadcast_to(k_new[b:b + 1].astype(K_pages.dtype), (NUM_PAGES, PAGE_SIZE, H, D))
        v_fill = mx.broadcast_to(v_new[b:b + 1].astype(V_pages.dtype), (NUM_PAGES, PAGE_SIZE, H, D))
        K_updated = mx.where(match, k_fill, K_updated)
        V_updated = mx.where(match, v_fill, V_updated)
    return K_updated, V_updated


def paged_kv_cache_update(
    K_pages: mx.array,
    V_pages: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    block_table: mx.array,
    positions,
    *,
    backend: str = "auto",
):
    k_new = _normalize_token_shape(k_new, "k_new")
    v_new = _normalize_token_shape(v_new, "v_new")
    NUM_PAGES, PAGE_SIZE, H, D, B, MAX_BLOCKS = _validate_page_tensors(K_pages, V_pages, block_table)
    if k_new.shape != (B, 1, H, D) or v_new.shape != (B, 1, H, D):
        raise ValueError(
            f"k_new/v_new must normalize to [B,1,H,D]={B,1,H,D}, got {k_new.shape}, {v_new.shape}"
        )
    positions_arr = _validate_positions_against_blocks(positions, B, MAX_BLOCKS * PAGE_SIZE)
    backend_name = backend.lower()
    if backend_name == "auto":
        backend_name = "metal"
    if backend_name == "reference":
        return reference_paged_kv_cache_update(K_pages, V_pages, k_new, v_new, block_table, positions_arr)
    if backend_name != "metal":
        raise ValueError("backend must be one of 'reference', 'metal', 'auto'")
    dtype = K_pages.dtype
    source = _load_source(_PAGED_UPDATE_KERNEL)
    header = _make_header(dtype)
    kernel = _get_paged_update_kernel(str(dtype), source, header)
    meta = mx.array([NUM_PAGES, PAGE_SIZE, H, D, B, MAX_BLOCKS], dtype=mx.int32)
    outputs = kernel(
        inputs=[K_pages, V_pages, k_new.astype(dtype), v_new.astype(dtype), block_table.astype(mx.int32), positions_arr.astype(mx.int32), meta],
        output_shapes=[K_pages.shape, V_pages.shape],
        output_dtypes=[dtype, dtype],
        grid=(NUM_PAGES * PAGE_SIZE * H * D, 1, 1),
        threadgroup=(_THREADS, 1, 1),
    )
    return outputs[0], outputs[1]


def reference_paged_decode_attention(
    q: mx.array,
    K_pages: mx.array,
    V_pages: mx.array,
    block_table: mx.array,
    lengths,
    *,
    scale: Optional[float] = None,
):
    NUM_PAGES, PAGE_SIZE, H, D, B, MAX_BLOCKS = _validate_page_tensors(K_pages, V_pages, block_table)
    if q.ndim != 4 or q.shape != (B, 1, H, D):
        raise ValueError(
            f"q must have shape [B,1,H,D]={B,1,H,D}, got {q.shape}. "
            "For GQA/MQA paged decode with Hq != Hkv, use ops.gqa_ops.reference_paged_gqa_decode_attention."
        )
    lengths_arr = _validate_positions_against_blocks(lengths if lengths is not None else MAX_BLOCKS * PAGE_SIZE, B, MAX_BLOCKS * PAGE_SIZE + 1)
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    gathered_K = []
    gathered_V = []
    for b in range(B):
        rows_k = []
        rows_v = []
        for j in range(MAX_BLOCKS * PAGE_SIZE):
            block_id = j // PAGE_SIZE
            offset = j % PAGE_SIZE
            page_id = int(block_table[b, block_id].item())
            rows_k.append(K_pages[page_id:page_id + 1, offset:offset + 1, :, :])
            rows_v.append(V_pages[page_id:page_id + 1, offset:offset + 1, :, :])
        k_b = mx.concatenate(rows_k, axis=1)
        v_b = mx.concatenate(rows_v, axis=1)
        gathered_K.append(k_b)
        gathered_V.append(v_b)
    K_contig = mx.concatenate(gathered_K, axis=0)
    V_contig = mx.concatenate(gathered_V, axis=0)

    qf = q.astype(mx.float32)
    Kf = K_contig.astype(mx.float32)
    Vf = V_contig.astype(mx.float32)
    scores = mx.matmul(qf.transpose(0, 2, 1, 3), Kf.transpose(0, 2, 3, 1)) * float(scale)
    positions = mx.arange(MAX_BLOCKS * PAGE_SIZE).reshape(1, 1, 1, MAX_BLOCKS * PAGE_SIZE)
    valid_mask = positions < lengths_arr.reshape(B, 1, 1, 1)
    scores = mx.where(valid_mask, scores, mx.array(-1.0e9, dtype=scores.dtype))
    probs = mx.softmax(scores, axis=-1)
    out = mx.matmul(probs, Vf.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    return out.astype(q.dtype)


def paged_decode_attention(
    q: mx.array,
    K_pages: mx.array,
    V_pages: mx.array,
    block_table: mx.array,
    lengths,
    *,
    scale: Optional[float] = None,
    backend: str = "auto",
):
    NUM_PAGES, PAGE_SIZE, H, D, B, MAX_BLOCKS = _validate_page_tensors(K_pages, V_pages, block_table)
    if q.ndim != 4 or q.shape != (B, 1, H, D):
        raise ValueError(
            f"q must have shape [B,1,H,D]={B,1,H,D}, got {q.shape}. "
            "For GQA/MQA paged decode with Hq != Hkv, use ops.gqa_ops.reference_paged_gqa_decode_attention."
        )
    if D > 128:
        raise ValueError(f"paged_decode_attention currently supports D <= 128, got {D}")
    lengths_arr = _validate_positions_against_blocks(lengths if lengths is not None else MAX_BLOCKS * PAGE_SIZE, B, MAX_BLOCKS * PAGE_SIZE + 1)
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    backend_name = _resolve_backend(backend.lower(), D)
    if backend_name == "reference":
        return reference_paged_decode_attention(q, K_pages, V_pages, block_table, lengths_arr, scale=scale)
    if backend_name not in ("metal", "metal_threadgroup", "metal_d64", "metal_d128"):
        raise ValueError("backend must be one of 'reference', 'metal', 'metal_threadgroup', 'metal_d64', 'metal_d128', 'auto'")
    dtype = q.dtype
    if backend_name == "metal":
        source = _load_source(_PAGED_DECODE_KERNEL)
        header = _make_header(dtype)
        kernel = _get_paged_decode_kernel(str(dtype), source, header)
        grid = (B * H, 1, 1)
        threadgroup = (1, 1, 1)
    elif backend_name == "metal_threadgroup":
        source = _load_source(_PAGED_DECODE_THREADGROUP_KERNEL)
        header = _make_header(dtype)
        kernel = _get_named_paged_decode_kernel("paged_decode_attention_threadgroup_forward", str(dtype), source, header)
        grid = (B * H * _THREADGROUP_THREADS, 1, 1)
        threadgroup = (_THREADGROUP_THREADS, 1, 1)
    else:
        source = _load_source(_SPECIALIZED_PAGED_DECODE_KERNELS[backend_name])
        header = _make_header(dtype, fixed_head_dim=D)
        kernel = _get_named_paged_decode_kernel(f"paged_decode_attention_{D}_forward", str(dtype), source, header)
        grid = (B * H, 1, 1)
        threadgroup = (1, 1, 1)
    meta = mx.array([B, NUM_PAGES, PAGE_SIZE, H, D, MAX_BLOCKS], dtype=mx.int32)
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    return kernel(
        inputs=[q.astype(dtype), K_pages.astype(dtype), V_pages.astype(dtype), block_table.astype(mx.int32), lengths_arr.astype(mx.int32), meta, scale_arr],
        output_shapes=[q.shape],
        output_dtypes=[dtype],
        grid=grid,
        threadgroup=threadgroup,
    )[0]


def reference_paged_decode_step(
    q: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    K_pages: mx.array,
    V_pages: mx.array,
    block_table: mx.array,
    position,
    *,
    scale: Optional[float] = None,
):
    updated_K, updated_V = reference_paged_kv_cache_update(K_pages, V_pages, k_new, v_new, block_table, position)
    if isinstance(position, int):
        lengths = position + 1
    else:
        lengths = _validate_positions_against_blocks(position, block_table.shape[0], block_table.shape[1] * K_pages.shape[1]) + 1
    out = reference_paged_decode_attention(q, updated_K, updated_V, block_table, lengths, scale=scale)
    return out, updated_K, updated_V


def paged_decode_step(
    q: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    K_pages: mx.array,
    V_pages: mx.array,
    block_table: mx.array,
    position,
    *,
    scale: Optional[float] = None,
    backend: str = "auto",
):
    backend_name = backend.lower()
    if backend_name == "reference":
        return reference_paged_decode_step(q, k_new, v_new, K_pages, V_pages, block_table, position, scale=scale)
    updated_K, updated_V = paged_kv_cache_update(K_pages, V_pages, k_new, v_new, block_table, position, backend=backend_name)
    if isinstance(position, int):
        lengths = position + 1
    else:
        lengths = _validate_positions_against_blocks(position, block_table.shape[0], block_table.shape[1] * K_pages.shape[1]) + 1
    out = paged_decode_attention(q, updated_K, updated_V, block_table, lengths, scale=scale, backend=backend_name)
    return out, updated_K, updated_V
