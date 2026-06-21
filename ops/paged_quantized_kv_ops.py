from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import mlx.core as mx

from .decode_ops import normalize_positions
from .gqa_ops import q_head_to_kv_head, validate_gqa_heads
from .kernel_utils import KERNEL_DIR, make_metal_header, load_metal_source
from .quant_ops import pack_q4, unpack_q4_reference

_Q8_PAGED_GQA_DECODE_KERNEL = KERNEL_DIR / "q8_paged_gqa_decode_attention.metal"
_Q4_PAGED_GQA_DECODE_KERNEL = KERNEL_DIR / "q4_paged_gqa_decode_attention.metal"
_MAX_HEAD_DIM = 128


def _dtype_itemsize(dtype: Any) -> int:
    dtype_str = str(dtype)
    if "uint8" in dtype_str or "int8" in dtype_str:
        return 1
    if "uint16" in dtype_str or "int16" in dtype_str or "float16" in dtype_str or "bfloat16" in dtype_str:
        return 2
    if "uint32" in dtype_str or "int32" in dtype_str or "float32" in dtype_str:
        return 4
    if "uint64" in dtype_str or "int64" in dtype_str or "float64" in dtype_str:
        return 8
    raise ValueError(f"Unsupported dtype: {dtype}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PagedQuantizedKVConfig:
    bits: int = 8
    page_size: int = 16
    group_size: int = 32
    symmetric: bool = True
    with_zeros: bool = False
    scale_dtype: str = "float16"
    layout: str = "paged"

    def validate(self) -> PagedQuantizedKVConfig:
        if self.bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {self.bits}")
        if self.page_size <= 0:
            raise ValueError(f"page_size must be > 0, got {self.page_size}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {self.group_size}")
        if not self.symmetric:
            raise NotImplementedError("asymmetric quantization is not implemented yet")
        if self.with_zeros:
            raise NotImplementedError("zero-point quantization is not implemented yet")
        if self.layout != "paged":
            raise ValueError(f"layout must be 'paged', got {self.layout!r}")
        return self


# ---------------------------------------------------------------------------
# Paged quantized KV-cache dataclass
# ---------------------------------------------------------------------------


@dataclass
class PagedQuantizedKVCache:
    k_pages_q: Any
    v_pages_q: Any
    k_scales: Any
    v_scales: Any
    block_table: Any
    lengths: Any
    k_zeros: Any | None = None
    v_zeros: Any | None = None
    bits: int = 8
    page_size: int = 16
    group_size: int = 32
    original_page_shape: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "k_pages_q": tuple(self.k_pages_q.shape),
            "v_pages_q": tuple(self.v_pages_q.shape),
            "k_scales": tuple(self.k_scales.shape),
            "v_scales": tuple(self.v_scales.shape),
            "block_table": tuple(self.block_table.shape),
            "lengths": tuple(self.lengths.shape),
            "original_page_shape": None if self.original_page_shape is None else tuple(self.original_page_shape),
        }

    def validate(self) -> PagedQuantizedKVCache:
        if self.bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {self.bits}")
        if self.k_pages_q.shape != self.v_pages_q.shape:
            raise ValueError(f"k_pages_q and v_pages_q shapes must match, got {self.k_pages_q.shape} != {self.v_pages_q.shape}")
        if self.k_scales.shape != self.v_scales.shape:
            raise ValueError(f"k_scales and v_scales shapes must match, got {self.k_scales.shape} != {self.v_scales.shape}")
        if self.block_table.ndim != 2:
            raise ValueError(f"block_table must be 2-D [B,MAX_BLOCKS], got {self.block_table.shape}")
        if self.lengths.ndim != 1:
            raise ValueError(f"lengths must be 1-D [B], got {self.lengths.shape}")
        B = self.block_table.shape[0]
        if self.lengths.shape[0] != B:
            raise ValueError(f"lengths batch {self.lengths.shape[0]} != block_table batch {B}")
        if self.original_page_shape is not None and len(self.original_page_shape) != 4:
            raise ValueError(f"original_page_shape must be 4-D [NUM_PAGES,PAGE_SIZE,Hkv,D], got {self.original_page_shape}")
        return self

    def num_pages(self) -> int:
        return self.k_pages_q.shape[0]

    def memory_bytes(self) -> int:
        total = 0
        for arr in (self.k_pages_q, self.v_pages_q, self.k_scales, self.v_scales, self.block_table, self.lengths):
            total += arr.size * _dtype_itemsize(arr.dtype)
        if self.k_zeros is not None:
            total += self.k_zeros.size * _dtype_itemsize(self.k_zeros.dtype)
        if self.v_zeros is not None:
            total += self.v_zeros.size * _dtype_itemsize(self.v_zeros.dtype)
        return total

    def compression_ratio(self, fp_bytes_per_value: int = 2) -> float | None:
        if self.original_page_shape is not None:
            NUM_PAGES, PAGE_SIZE, Hkv, D = self.original_page_shape
            num_values = 2 * NUM_PAGES * PAGE_SIZE * Hkv * D
            fp_bytes = num_values * fp_bytes_per_value
            q_bytes = 0
            for arr in (self.k_pages_q, self.v_pages_q, self.k_scales, self.v_scales):
                q_bytes += arr.size * _dtype_itemsize(arr.dtype)
            if q_bytes == 0:
                return None
            return fp_bytes / q_bytes
        return None


# ---------------------------------------------------------------------------
# Quantization helpers (page-level)
# ---------------------------------------------------------------------------


def _quantize_group_symmetric_8(values: mx.array, eps: float = 1e-8) -> tuple[mx.array, mx.array]:
    max_abs = mx.max(mx.abs(values), axis=-1, keepdims=True)
    max_abs = mx.maximum(max_abs, mx.array(eps, dtype=max_abs.dtype))
    scale = max_abs / mx.array(127.0, dtype=max_abs.dtype)
    q_signed = mx.clip(mx.round(values / scale), -127, 127)
    q_unsigned = (q_signed + 128).astype(mx.uint8)
    scale_out = scale.squeeze(-1)
    return q_unsigned, scale_out.astype(mx.float16)


def _dequant_group_symmetric_8(q_unsigned: mx.array, scales: mx.array, group_size: int, D: int) -> mx.array:
    q_signed = q_unsigned.astype(mx.float32) - 128.0
    groups_per_head = (D + group_size - 1) // group_size
    scale_expanded = mx.repeat(scales.astype(mx.float32), group_size, axis=-1)
    if scale_expanded.shape[-1] > D:
        scale_expanded = scale_expanded[..., :D]
    elif scale_expanded.shape[-1] < D:
        pad = mx.zeros((*scale_expanded.shape[:-1], D - scale_expanded.shape[-1]), dtype=scale_expanded.dtype)
        scale_expanded = mx.concatenate([scale_expanded, pad], axis=-1)
    return q_signed * mx.abs(scale_expanded)


def _quantize_group_symmetric_4(values: mx.array, eps: float = 1e-8) -> tuple[mx.array, mx.array]:
    max_abs = mx.max(mx.abs(values), axis=-1, keepdims=True)
    max_abs = mx.maximum(max_abs, mx.array(eps, dtype=max_abs.dtype))
    scale = max_abs / mx.array(7.0, dtype=max_abs.dtype)
    q_signed = mx.clip(mx.round(values / scale), -7, 7)
    q_unsigned = (q_signed + 8).astype(mx.uint8)
    scale_out = scale.squeeze(-1)
    return q_unsigned, scale_out.astype(mx.float16)


def _dequant_group_symmetric_4(q_unsigned: mx.array, scales: mx.array, group_size: int, D: int) -> mx.array:
    q_signed = q_unsigned.astype(mx.float32) - 8.0
    scale_expanded = mx.repeat(scales.astype(mx.float32), group_size, axis=-1)
    if scale_expanded.shape[-1] > D:
        scale_expanded = scale_expanded[..., :D]
    elif scale_expanded.shape[-1] < D:
        pad = mx.zeros((*scale_expanded.shape[:-1], D - scale_expanded.shape[-1]), dtype=scale_expanded.dtype)
        scale_expanded = mx.concatenate([scale_expanded, pad], axis=-1)
    return q_signed * mx.abs(scale_expanded)


# ---------------------------------------------------------------------------
# Quantize / dequantize KV pages
# ---------------------------------------------------------------------------


def quantize_kv_pages(
    K_pages: mx.array,
    V_pages: mx.array,
    block_table: mx.array,
    lengths: mx.array,
    config: PagedQuantizedKVConfig,
) -> PagedQuantizedKVCache:
    config = config.validate()
    if K_pages.shape != V_pages.shape:
        raise ValueError(f"K_pages and V_pages must match, got {K_pages.shape} != {V_pages.shape}")
    if K_pages.ndim != 4:
        raise ValueError(f"K_pages must be 4-D [NUM_PAGES,PAGE_SIZE,Hkv,D], got {K_pages.shape}")

    NUM_PAGES, PAGE_SIZE, Hkv, D = K_pages.shape
    groups_per_head = (D + config.group_size - 1) // config.group_size
    B, MAX_BLOCKS = block_table.shape

    if config.bits == 8:
        k_pages_q = mx.zeros((NUM_PAGES, PAGE_SIZE, Hkv, D), dtype=mx.uint8)
        v_pages_q = mx.zeros((NUM_PAGES, PAGE_SIZE, Hkv, D), dtype=mx.uint8)
        k_scales = mx.zeros((NUM_PAGES, PAGE_SIZE, Hkv, groups_per_head), dtype=mx.float16)
        v_scales = mx.zeros((NUM_PAGES, PAGE_SIZE, Hkv, groups_per_head), dtype=mx.float16)

        Kf = K_pages.astype(mx.float32)
        Vf = V_pages.astype(mx.float32)
        for p in range(NUM_PAGES):
            for s in range(PAGE_SIZE):
                for h in range(Hkv):
                    chunk_k = Kf[p, s, h, :]
                    for g in range(groups_per_head):
                        g_start = g * config.group_size
                        g_end = min(g_start + config.group_size, D)
                        group_vals = chunk_k[g_start:g_end]
                        q_u, sc = _quantize_group_symmetric_8(group_vals.reshape(1, -1))
                        k_pages_q[p, s, h, g_start:g_end] = q_u.reshape(-1)
                        k_scales[p, s, h, g] = sc.reshape(())
                    chunk_v = Vf[p, s, h, :]
                    for g in range(groups_per_head):
                        g_start = g * config.group_size
                        g_end = min(g_start + config.group_size, D)
                        group_vals = chunk_v[g_start:g_end]
                        q_u, sc = _quantize_group_symmetric_8(group_vals.reshape(1, -1))
                        v_pages_q[p, s, h, g_start:g_end] = q_u.reshape(-1)
                        v_scales[p, s, h, g] = sc.reshape(())

        return PagedQuantizedKVCache(
            k_pages_q=k_pages_q, v_pages_q=v_pages_q,
            k_scales=k_scales, v_scales=v_scales,
            block_table=block_table, lengths=lengths,
            bits=8, page_size=config.page_size, group_size=config.group_size,
            original_page_shape=K_pages.shape,
        )

    elif config.bits == 4:
        D_packed = (D + 1) // 2
        k_pages_q = mx.zeros((NUM_PAGES, PAGE_SIZE, Hkv, D_packed), dtype=mx.uint8)
        v_pages_q = mx.zeros((NUM_PAGES, PAGE_SIZE, Hkv, D_packed), dtype=mx.uint8)
        k_scales = mx.zeros((NUM_PAGES, PAGE_SIZE, Hkv, groups_per_head), dtype=mx.float16)
        v_scales = mx.zeros((NUM_PAGES, PAGE_SIZE, Hkv, groups_per_head), dtype=mx.float16)

        Kf = K_pages.astype(mx.float32)
        Vf = V_pages.astype(mx.float32)
        for p in range(NUM_PAGES):
            for s in range(PAGE_SIZE):
                for h in range(Hkv):
                    chunk_k = Kf[p, s, h, :]
                    group_q = mx.zeros(D, dtype=mx.uint8)
                    for g in range(groups_per_head):
                        g_start = g * config.group_size
                        g_end = min(g_start + config.group_size, D)
                        group_vals = chunk_k[g_start:g_end]
                        q_u, sc = _quantize_group_symmetric_4(group_vals.reshape(1, -1))
                        group_q[g_start:g_end] = q_u.reshape(-1)
                        k_scales[p, s, h, g] = sc.reshape(())
                    packed = pack_q4(group_q.reshape(1, D))
                    k_pages_q[p, s, h, :] = packed.reshape(-1)

                    chunk_v = Vf[p, s, h, :]
                    group_q = mx.zeros(D, dtype=mx.uint8)
                    for g in range(groups_per_head):
                        g_start = g * config.group_size
                        g_end = min(g_start + config.group_size, D)
                        group_vals = chunk_v[g_start:g_end]
                        q_u, sc = _quantize_group_symmetric_4(group_vals.reshape(1, -1))
                        group_q[g_start:g_end] = q_u.reshape(-1)
                        v_scales[p, s, h, g] = sc.reshape(())
                    packed = pack_q4(group_q.reshape(1, D))
                    v_pages_q[p, s, h, :] = packed.reshape(-1)

        return PagedQuantizedKVCache(
            k_pages_q=k_pages_q, v_pages_q=v_pages_q,
            k_scales=k_scales, v_scales=v_scales,
            block_table=block_table, lengths=lengths,
            bits=4, page_size=config.page_size, group_size=config.group_size,
            original_page_shape=K_pages.shape,
        )

    else:
        raise ValueError(f"bits must be 4 or 8, got {config.bits}")


def dequantize_kv_pages(paged_qkv: PagedQuantizedKVCache) -> tuple[mx.array, mx.array]:
    paged_qkv = paged_qkv.validate()
    orig = paged_qkv.original_page_shape
    if orig is None:
        raise ValueError("original_page_shape is required for dequantization")
    NUM_PAGES, PAGE_SIZE, Hkv, D = orig
    groups_per_head = (D + paged_qkv.group_size - 1) // paged_qkv.group_size

    K_deq = mx.zeros((NUM_PAGES, PAGE_SIZE, Hkv, D), dtype=mx.float16)
    V_deq = mx.zeros((NUM_PAGES, PAGE_SIZE, Hkv, D), dtype=mx.float16)

    if paged_qkv.bits == 8:
        for p in range(NUM_PAGES):
            for s in range(PAGE_SIZE):
                for h in range(Hkv):
                    k_chunk = paged_qkv.k_pages_q[p, s, h, :]
                    k_sc = paged_qkv.k_scales[p, s, h, :]
                    K_deq[p, s, h, :] = _dequant_group_symmetric_8(
                        k_chunk.reshape(1, 1, -1), k_sc.reshape(1, 1, -1),
                        paged_qkv.group_size, D,
                    ).reshape(-1).astype(mx.float16)
                    v_chunk = paged_qkv.v_pages_q[p, s, h, :]
                    v_sc = paged_qkv.v_scales[p, s, h, :]
                    V_deq[p, s, h, :] = _dequant_group_symmetric_8(
                        v_chunk.reshape(1, 1, -1), v_sc.reshape(1, 1, -1),
                        paged_qkv.group_size, D,
                    ).reshape(-1).astype(mx.float16)
    elif paged_qkv.bits == 4:
        for p in range(NUM_PAGES):
            for s in range(PAGE_SIZE):
                for h in range(Hkv):
                    packed_k = paged_qkv.k_pages_q[p, s, h, :].reshape(1, -1)
                    unpacked_k = unpack_q4_reference(packed_k, K=D).reshape(-1)
                    k_sc = paged_qkv.k_scales[p, s, h, :]
                    K_deq[p, s, h, :] = _dequant_group_symmetric_4(
                        unpacked_k.reshape(1, 1, -1), k_sc.reshape(1, 1, -1),
                        paged_qkv.group_size, D,
                    ).reshape(-1).astype(mx.float16)
                    packed_v = paged_qkv.v_pages_q[p, s, h, :].reshape(1, -1)
                    unpacked_v = unpack_q4_reference(packed_v, K=D).reshape(-1)
                    v_sc = paged_qkv.v_scales[p, s, h, :]
                    V_deq[p, s, h, :] = _dequant_group_symmetric_4(
                        unpacked_v.reshape(1, 1, -1), v_sc.reshape(1, 1, -1),
                        paged_qkv.group_size, D,
                    ).reshape(-1).astype(mx.float16)

    return K_deq, V_deq


def paged_quantized_kv_error(
    K_pages: mx.array,
    V_pages: mx.array,
    paged_qkv: PagedQuantizedKVCache,
) -> dict[str, float]:
    K_deq, V_deq = dequantize_kv_pages(paged_qkv)
    K_f = K_pages.astype(mx.float32)
    V_f = V_pages.astype(mx.float32)
    K_deq_f = K_deq.astype(mx.float32)
    V_deq_f = V_deq.astype(mx.float32)

    k_diff = K_f - K_deq_f
    v_diff = V_f - V_deq_f
    cr = paged_qkv.compression_ratio(fp_bytes_per_value=2)

    return {
        "k_max_abs_error": float(mx.max(mx.abs(k_diff)).item()),
        "k_mean_abs_error": float(mx.mean(mx.abs(k_diff)).item()),
        "k_rmse": float(mx.sqrt(mx.mean(k_diff ** 2)).item()),
        "v_max_abs_error": float(mx.max(mx.abs(v_diff)).item()),
        "v_mean_abs_error": float(mx.mean(mx.abs(v_diff)).item()),
        "v_rmse": float(mx.sqrt(mx.mean(v_diff ** 2)).item()),
        "compression_ratio": float(cr) if cr is not None else 0.0,
    }


# ---------------------------------------------------------------------------
# Conversion helpers (contiguous <-> pages, for tests/demos)
# ---------------------------------------------------------------------------


def contiguous_kv_to_pages(
    K_cache: mx.array,
    V_cache: mx.array,
    lengths: list[int] | mx.array,
    *,
    page_size: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    if K_cache.ndim != 4 or V_cache.ndim != 4:
        raise ValueError(f"K_cache/V_cache must be 4-D [B,MAX_S,Hkv,D], got {K_cache.shape}, {V_cache.shape}")
    B, MAX_S, Hkv, D = K_cache.shape
    max_blocks = (MAX_S + page_size - 1) // page_size
    num_pages = B * max_blocks
    K_pages = mx.zeros((num_pages, page_size, Hkv, D), dtype=K_cache.dtype)
    V_pages = mx.zeros((num_pages, page_size, Hkv, D), dtype=V_cache.dtype)
    block_table = mx.full((B, max_blocks), -1, dtype=mx.int32)

    length_arr = [int(l) for l in (lengths.tolist() if hasattr(lengths, 'tolist') else lengths)]
    for b in range(B):
        valid = length_arr[b]
        for pos in range(valid):
            block_idx = pos // page_size
            offset = pos % page_size
            page_id = b * max_blocks + block_idx
            K_pages[page_id, offset, :, :] = K_cache[b, pos, :, :]
            V_pages[page_id, offset, :, :] = V_cache[b, pos, :, :]
            block_table[b, block_idx] = page_id

    lengths_arr = mx.array(length_arr, dtype=mx.int32)
    return K_pages, V_pages, block_table, lengths_arr


def pages_to_contiguous_kv(
    K_pages: mx.array,
    V_pages: mx.array,
    block_table: mx.array,
    lengths: mx.array,
    *,
    max_seq_len: int | None = None,
) -> tuple[mx.array, mx.array]:
    B, MAX_BLOCKS = block_table.shape
    NUM_PAGES, PAGE_SIZE, Hkv, D = K_pages.shape
    if max_seq_len is None:
        max_seq_len = MAX_BLOCKS * PAGE_SIZE
    K_cache = mx.zeros((B, max_seq_len, Hkv, D), dtype=K_pages.dtype)
    V_cache = mx.zeros((B, max_seq_len, Hkv, D), dtype=V_pages.dtype)

    for b in range(B):
        valid = int(lengths[b].item())
        for pos in range(valid):
            block_idx = pos // PAGE_SIZE
            offset = pos % PAGE_SIZE
            page_id = int(block_table[b, block_idx].item())
            if page_id < 0:
                continue
            K_cache[b, pos, :, :] = K_pages[page_id, offset, :, :]
            V_cache[b, pos, :, :] = V_pages[page_id, offset, :, :]

    return K_cache, V_cache


# ---------------------------------------------------------------------------
# Reference paged quantized GQA decode attention
# ---------------------------------------------------------------------------


def _validate_paged_quantized_decode_inputs(
    q: mx.array,
    paged_qkv: PagedQuantizedKVCache,
) -> tuple[mx.array, int, int, int, int, int, int, int]:
    if q.ndim != 4:
        raise ValueError(f"q must be 4-D [B,1,Hq,D], got {q.shape}")
    if q.shape[1] != 1:
        raise ValueError(f"q must have shape [B,1,Hq,D], got {q.shape}")
    B, _, Hq, D = q.shape
    orig = paged_qkv.original_page_shape
    if orig is None:
        raise ValueError("original_page_shape is required for decode attention")
    cache_NUM_PAGES, cache_PAGE_SIZE, Hkv, D_cache = orig
    cache_B = paged_qkv.block_table.shape[0]

    if B != cache_B:
        raise ValueError(f"batch mismatch: q batch={B}, cache batch={cache_B}")
    if D != D_cache:
        raise ValueError(f"head_dim mismatch: q D={D}, cache D={D_cache}")
    validate_gqa_heads(Hq, Hkv)

    MAX_BLOCKS = paged_qkv.block_table.shape[1]
    MAX_S = MAX_BLOCKS * cache_PAGE_SIZE
    lengths_arr = paged_qkv.lengths.reshape(B)
    lengths_arr = mx.minimum(lengths_arr, mx.array(MAX_S, dtype=mx.int32))

    return lengths_arr, B, MAX_S, cache_PAGE_SIZE, MAX_BLOCKS, Hq, Hkv, D


def reference_paged_quantized_kv_gqa_decode_attention(
    q: mx.array,
    paged_qkv: PagedQuantizedKVCache,
    *,
    scale: float | None = None,
) -> mx.array:
    lengths_arr, B, MAX_S, PAGE_SIZE, MAX_BLOCKS, Hq, Hkv, D = _validate_paged_quantized_decode_inputs(q, paged_qkv)
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    K_deq, V_deq = dequantize_kv_pages(paged_qkv)
    Kf = K_deq.astype(mx.float32)
    Vf = V_deq.astype(mx.float32)
    qf = q.astype(mx.float32)

    outputs = []
    for hq in range(Hq):
        hkv = q_head_to_kv_head(hq, Hq, Hkv)
        q_head = qf[:, :, hq:hq + 1, :]
        gathered_K = []
        gathered_V = []
        for b in range(B):
            rows_k = []
            rows_v = []
            valid = int(lengths_arr[b].item())
            for pos in range(valid):
                block_idx = pos // PAGE_SIZE
                offset = pos % PAGE_SIZE
                page_id = int(paged_qkv.block_table[b, block_idx].item())
                rows_k.append(Kf[page_id:page_id + 1, offset:offset + 1, hkv:hkv + 1, :])
                rows_v.append(Vf[page_id:page_id + 1, offset:offset + 1, hkv:hkv + 1, :])
            k_b = mx.concatenate(rows_k, axis=1) if rows_k else mx.zeros((1, 0, 1, D), dtype=mx.float32)
            v_b = mx.concatenate(rows_v, axis=1) if rows_v else mx.zeros((1, 0, 1, D), dtype=mx.float32)
            gathered_K.append(k_b)
            gathered_V.append(v_b)
        K_head = mx.concatenate(gathered_K, axis=0)
        V_head = mx.concatenate(gathered_V, axis=0)

        scores = mx.matmul(q_head.transpose(0, 2, 1, 3), K_head.transpose(0, 2, 3, 1)) * float(scale)
        positions = mx.arange(MAX_S).reshape(1, 1, 1, MAX_S)
        valid_mask = positions < lengths_arr.reshape(B, 1, 1, 1)
        scores = mx.where(valid_mask, scores, mx.array(-1.0e9, dtype=scores.dtype))
        probs = mx.softmax(scores, axis=-1)
        out_head = mx.matmul(probs, V_head.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        outputs.append(out_head)

    return mx.concatenate(outputs, axis=2).astype(q.dtype)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _make_header(dtype: mx.Dtype) -> str:
    return make_metal_header(dtype, MAX_HEAD_DIM=_MAX_HEAD_DIM)


@lru_cache(maxsize=8)
def _get_q8_paged_gqa_decode_kernel(kernel_name: str, dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name=kernel_name,
        input_names=["q", "K_pages_q", "V_pages_q", "K_scales", "V_scales", "block_table", "lengths", "meta", "scale"],
        output_names=["out"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=8)
def _get_q4_paged_gqa_decode_kernel(kernel_name: str, dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name=kernel_name,
        input_names=["q", "K_pages_q", "V_pages_q", "K_scales", "V_scales", "block_table", "lengths", "meta", "scale"],
        output_names=["out"],
        source=source,
        header=header,
    )


def _metal_q8_paged_gqa_decode_attention(
    q: mx.array,
    paged_qkv: PagedQuantizedKVCache,
    lengths_arr: mx.array,
    *,
    scale: float,
) -> mx.array:
    B = q.shape[0]
    Hq = q.shape[2]
    D = q.shape[3]
    orig = paged_qkv.original_page_shape
    NUM_PAGES, PAGE_SIZE, Hkv, D_cache = orig
    MAX_BLOCKS = paged_qkv.block_table.shape[1]
    groups_per_head = (D + paged_qkv.group_size - 1) // paged_qkv.group_size
    meta = mx.array(
        [B, NUM_PAGES, PAGE_SIZE, MAX_BLOCKS, Hq, Hkv, D, paged_qkv.group_size, groups_per_head],
        dtype=mx.int32,
    )
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    source = load_metal_source(_Q8_PAGED_GQA_DECODE_KERNEL)
    header = _make_header(q.dtype)
    kernel = _get_q8_paged_gqa_decode_kernel("q8_paged_gqa_decode_attention_forward", str(q.dtype), source, header)
    return kernel(
        inputs=[
            q.astype(q.dtype),
            paged_qkv.k_pages_q.astype(mx.uint8),
            paged_qkv.v_pages_q.astype(mx.uint8),
            paged_qkv.k_scales.astype(mx.float32),
            paged_qkv.v_scales.astype(mx.float32),
            paged_qkv.block_table.astype(mx.int32),
            lengths_arr.astype(mx.int32),
            meta,
            scale_arr,
        ],
        output_shapes=[(B, 1, Hq, D)],
        output_dtypes=[q.dtype],
        grid=(B * Hq, 1, 1),
        threadgroup=(1, 1, 1),
    )[0]


def _metal_q4_paged_gqa_decode_attention(
    q: mx.array,
    paged_qkv: PagedQuantizedKVCache,
    lengths_arr: mx.array,
    *,
    scale: float,
) -> mx.array:
    B = q.shape[0]
    Hq = q.shape[2]
    D = q.shape[3]
    orig = paged_qkv.original_page_shape
    NUM_PAGES, PAGE_SIZE, Hkv, D_cache = orig
    MAX_BLOCKS = paged_qkv.block_table.shape[1]
    D_packed = (D + 1) // 2
    groups_per_head = (D + paged_qkv.group_size - 1) // paged_qkv.group_size
    meta = mx.array(
        [B, NUM_PAGES, PAGE_SIZE, MAX_BLOCKS, Hq, Hkv, D, D_packed, paged_qkv.group_size, groups_per_head],
        dtype=mx.int32,
    )
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    source = load_metal_source(_Q4_PAGED_GQA_DECODE_KERNEL)
    header = _make_header(q.dtype)
    kernel = _get_q4_paged_gqa_decode_kernel("q4_paged_gqa_decode_attention_forward", str(q.dtype), source, header)
    return kernel(
        inputs=[
            q.astype(q.dtype),
            paged_qkv.k_pages_q.astype(mx.uint8),
            paged_qkv.v_pages_q.astype(mx.uint8),
            paged_qkv.k_scales.astype(mx.float32),
            paged_qkv.v_scales.astype(mx.float32),
            paged_qkv.block_table.astype(mx.int32),
            lengths_arr.astype(mx.int32),
            meta,
            scale_arr,
        ],
        output_shapes=[(B, 1, Hq, D)],
        output_dtypes=[q.dtype],
        grid=(B * Hq, 1, 1),
        threadgroup=(1, 1, 1),
    )[0]


def paged_quantized_kv_gqa_decode_attention(
    q: mx.array,
    paged_qkv: PagedQuantizedKVCache,
    *,
    scale: float | None = None,
    backend: str = "reference",
) -> mx.array:
    lengths_arr, B, MAX_S, PAGE_SIZE, MAX_BLOCKS, Hq, Hkv, D = _validate_paged_quantized_decode_inputs(q, paged_qkv)
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    backend_name = backend.lower()
    if backend_name == "reference":
        return reference_paged_quantized_kv_gqa_decode_attention(q, paged_qkv, scale=scale)

    if backend_name == "metal_q8":
        if paged_qkv.bits != 8:
            raise ValueError(f"metal_q8 backend requires bits=8, got bits={paged_qkv.bits}")
        if D > _MAX_HEAD_DIM:
            raise ValueError(f"metal_q8 backend supports D <= {_MAX_HEAD_DIM}, got D={D}")
        return _metal_q8_paged_gqa_decode_attention(q, paged_qkv, lengths_arr, scale=scale)

    if backend_name == "metal_q4":
        if paged_qkv.bits != 4:
            raise ValueError(f"metal_q4 backend requires bits=4, got bits={paged_qkv.bits}")
        if D > _MAX_HEAD_DIM:
            raise ValueError(f"metal_q4 backend supports D <= {_MAX_HEAD_DIM}, got D={D}")
        return _metal_q4_paged_gqa_decode_attention(q, paged_qkv, lengths_arr, scale=scale)

    raise ValueError(f"backend must be one of 'reference', 'metal_q8', 'metal_q4', got {backend_name!r}")


# ---------------------------------------------------------------------------
# Sparse paged quantized reference (scaffold)
# ---------------------------------------------------------------------------


def reference_sparse_paged_quantized_kv_gqa_decode_attention(
    q: mx.array,
    paged_qkv: PagedQuantizedKVCache,
    pattern: Any,
    *,
    scale: float | None = None,
) -> mx.array:
    from .sparse_attention_ops import build_sparse_attention_mask

    lengths_arr, B, MAX_S, PAGE_SIZE, MAX_BLOCKS, Hq, Hkv, D = _validate_paged_quantized_decode_inputs(q, paged_qkv)
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    K_deq, V_deq = dequantize_kv_pages(paged_qkv)
    Kf = K_deq.astype(mx.float32)
    Vf = V_deq.astype(mx.float32)
    qf = q.astype(mx.float32)

    outputs = []
    for b in range(B):
        valid_len = int(lengths_arr[b].item())
        if valid_len <= 0:
            raise ValueError("sparse decode attention requires lengths >= 1")
        mask = build_sparse_attention_mask(1, valid_len, pattern, start_position=valid_len - 1)
        q_b = qf[b:b + 1]
        head_outputs = []
        for hq in range(Hq):
            hkv = q_head_to_kv_head(hq, Hq, Hkv)
            q_head = q_b[:, :, hq:hq + 1, :]
            rows_k = []
            rows_v = []
            for pos in range(valid_len):
                block_idx = pos // PAGE_SIZE
                offset = pos % PAGE_SIZE
                page_id = int(paged_qkv.block_table[b, block_idx].item())
                rows_k.append(Kf[page_id:page_id + 1, offset:offset + 1, hkv:hkv + 1, :])
                rows_v.append(Vf[page_id:page_id + 1, offset:offset + 1, hkv:hkv + 1, :])
            k_head = mx.concatenate(rows_k, axis=1)
            v_head = mx.concatenate(rows_v, axis=1)
            scores = mx.matmul(q_head.transpose(0, 2, 1, 3), k_head.transpose(0, 2, 3, 1)) * float(scale)
            row_mask = mx.array(mask[0], dtype=mx.bool_).reshape(1, 1, 1, valid_len)
            scores = mx.where(row_mask, scores, mx.array(-1.0e9, dtype=scores.dtype))
            probs = mx.softmax(scores, axis=-1)
            head_outputs.append(mx.matmul(probs, v_head.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3))
        outputs.append(mx.concatenate(head_outputs, axis=2))
    return mx.concatenate(outputs, axis=0).astype(q.dtype)
