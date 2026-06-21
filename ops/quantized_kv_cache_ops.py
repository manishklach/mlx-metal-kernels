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
from .sparse_attention_ops import SparseAttentionPattern, build_sparse_attention_mask

_Q8_GQA_DECODE_KERNEL = KERNEL_DIR / "q8_gqa_decode_attention.metal"
_Q4_GQA_DECODE_KERNEL = KERNEL_DIR / "q4_gqa_decode_attention.metal"
_MAX_HEAD_DIM = 128


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class QuantizedKVCacheConfig:
    bits: int = 8
    group_size: int = 32
    symmetric: bool = True
    with_zeros: bool = False
    scale_dtype: str = "float16"
    layout: str = "contiguous"

    def validate(self) -> QuantizedKVCacheConfig:
        if self.bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {self.bits}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {self.group_size}")
        if not self.symmetric:
            raise NotImplementedError("asymmetric quantization is not implemented yet")
        if self.with_zeros:
            raise NotImplementedError("zero-point quantization is not implemented yet")
        if self.layout != "contiguous":
            raise NotImplementedError(f"layout={self.layout!r} is not implemented yet; use 'contiguous'")
        return self


# ---------------------------------------------------------------------------
# Quantized KV-cache dataclass
# ---------------------------------------------------------------------------


@dataclass
class QuantizedKVCache:
    k_q: Any
    v_q: Any
    k_scales: Any
    v_scales: Any
    k_zeros: Any | None = None
    v_zeros: Any | None = None
    bits: int = 8
    group_size: int = 32
    original_shape: tuple[int, ...] | None = None
    layout: str = "contiguous"
    metadata: dict[str, Any] = field(default_factory=dict)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "k_q": tuple(self.k_q.shape),
            "v_q": tuple(self.v_q.shape),
            "k_scales": tuple(self.k_scales.shape),
            "v_scales": tuple(self.v_scales.shape),
            "original": None if self.original_shape is None else tuple(self.original_shape),
        }

    def validate(self) -> QuantizedKVCache:
        if self.bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {self.bits}")
        if self.k_q.shape != self.v_q.shape:
            raise ValueError(f"k_q and v_q shapes must match, got {self.k_q.shape} != {self.v_q.shape}")
        if self.k_scales.shape != self.v_scales.shape:
            raise ValueError(f"k_scales and v_scales shapes must match, got {self.k_scales.shape} != {self.v_scales.shape}")
        if self.layout != "contiguous":
            raise NotImplementedError(f"layout={self.layout!r} is not supported")
        if self.original_shape is not None and len(self.original_shape) != 4:
            raise ValueError(f"original_shape must be 4-D [B,MAX_S,Hkv,D], got {self.original_shape}")
        return self

    def memory_bytes(self) -> int:
        total = 0
        for arr in (self.k_q, self.v_q, self.k_scales, self.v_scales):
            total += arr.size * arr.dtype.itemsize
        if self.k_zeros is not None:
            total += self.k_zeros.size * self.k_zeros.dtype.itemsize
        if self.v_zeros is not None:
            total += self.v_zeros.size * self.v_zeros.dtype.itemsize
        return total

    def compression_ratio(self, fp_bytes_per_value: int = 2) -> float | None:
        if self.original_shape is None:
            return None
        num_values = 2 * self.original_shape[0] * self.original_shape[1] * self.original_shape[2] * self.original_shape[3]
        fp_bytes = num_values * fp_bytes_per_value
        q_bytes = self.memory_bytes()
        if q_bytes == 0:
            return None
        return fp_bytes / q_bytes


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def _quantize_group_symmetric_8(values: mx.array, eps: float = 1e-8) -> tuple[mx.array, mx.array]:
    max_abs = mx.max(mx.abs(values), axis=-1, keepdims=True)
    max_abs = mx.maximum(max_abs, mx.array(eps, dtype=max_abs.dtype))
    scale = max_abs / mx.array(127.0, dtype=max_abs.dtype)
    q_signed = mx.clip(mx.round(values / scale), -127, 127)
    q_unsigned = (q_signed + 128).astype(mx.uint8)
    scale_out = scale.squeeze(-1)
    return q_unsigned, scale_out


def _dequant_group_symmetric_8(q_unsigned: mx.array, scales: mx.array, num_groups: int, group_size: int, D: int, eps: float = 1e-8) -> mx.array:
    q_signed = q_unsigned.astype(mx.float32) - 128.0
    scale_expanded = mx.repeat(scales.astype(mx.float32), group_size, axis=-1)
    if scale_expanded.shape[-1] > D:
        scale_expanded = scale_expanded[..., :D]
    elif scale_expanded.shape[-1] < D:
        pad = mx.ones((*scale_expanded.shape[:-1], D - scale_expanded.shape[-1]), dtype=scale_expanded.dtype) * eps
        scale_expanded = mx.concatenate([scale_expanded, pad], axis=-1)
    return q_signed * mx.abs(scale_expanded)


def _quantize_group_symmetric_4(values: mx.array, eps: float = 1e-8) -> tuple[mx.array, mx.array]:
    max_abs = mx.max(mx.abs(values), axis=-1, keepdims=True)
    max_abs = mx.maximum(max_abs, mx.array(eps, dtype=max_abs.dtype))
    scale = max_abs / mx.array(7.0, dtype=max_abs.dtype)
    q_signed = mx.clip(mx.round(values / scale), -7, 7)
    q_unsigned = (q_signed + 8).astype(mx.uint8)
    scale_out = scale.squeeze(-1)
    return q_unsigned, scale_out


def _dequant_group_symmetric_4(q_unsigned: mx.array, scales: mx.array, num_groups: int, group_size: int, D: int, eps: float = 1e-8) -> mx.array:
    q_signed = q_unsigned.astype(mx.float32) - 8.0
    scale_expanded = mx.repeat(scales.astype(mx.float32), group_size, axis=-1)
    if scale_expanded.shape[-1] > D:
        scale_expanded = scale_expanded[..., :D]
    elif scale_expanded.shape[-1] < D:
        pad = mx.ones((*scale_expanded.shape[:-1], D - scale_expanded.shape[-1]), dtype=scale_expanded.dtype) * eps
        scale_expanded = mx.concatenate([scale_expanded, pad], axis=-1)
    return q_signed * mx.abs(scale_expanded)


def quantize_kv_cache(
    K_cache: mx.array,
    V_cache: mx.array,
    config: QuantizedKVCacheConfig,
    *,
    lengths=None,
) -> QuantizedKVCache:
    config = config.validate()
    if K_cache.shape != V_cache.shape:
        raise ValueError(f"K_cache and V_cache must match, got {K_cache.shape} != {V_cache.shape}")
    if K_cache.ndim != 4:
        raise ValueError(f"K_cache must be 4-D [B,MAX_S,Hkv,D], got {K_cache.shape}")

    B, MAX_S, Hkv, D = K_cache.shape
    groups_per_head = (D + config.group_size - 1) // config.group_size
    num_groups = Hkv * groups_per_head

    if lengths is not None:
        lengths_arr = normalize_positions(lengths, B, MAX_S + 1)
    else:
        lengths_arr = mx.full((B,), MAX_S, dtype=mx.int32)

    if config.bits == 8:
        Kf = K_cache.astype(mx.float32)
        Vf = V_cache.astype(mx.float32)
        k_q_flat = mx.zeros((B, MAX_S, Hkv, D), dtype=mx.uint8)
        v_q_flat = mx.zeros((B, MAX_S, Hkv, D), dtype=mx.uint8)
        k_scales = mx.zeros((B, MAX_S, Hkv, groups_per_head), dtype=mx.float16)
        v_scales = mx.zeros((B, MAX_S, Hkv, groups_per_head), dtype=mx.float16)

        for b in range(B):
            valid = int(lengths_arr[b].item())
            for h in range(Hkv):
                for s in range(valid):
                    chunk = Kf[b, s, h, :]
                    for g in range(groups_per_head):
                        g_start = g * config.group_size
                        g_end = min(g_start + config.group_size, D)
                        group_vals = chunk[g_start:g_end]
                        q_u, sc = _quantize_group_symmetric_8(group_vals.reshape(1, -1))
                        k_q_flat[b, s, h, g_start:g_end] = q_u.reshape(-1)
                        k_scales[b, s, h, g] = sc.reshape(-1).astype(mx.float16)

                    chunk_v = Vf[b, s, h, :]
                    for g in range(groups_per_head):
                        g_start = g * config.group_size
                        g_end = min(g_start + config.group_size, D)
                        group_vals = chunk_v[g_start:g_end]
                        q_u, sc = _quantize_group_symmetric_8(group_vals.reshape(1, -1))
                        v_q_flat[b, s, h, g_start:g_end] = q_u.reshape(-1)
                        v_scales[b, s, h, g] = sc.reshape(-1).astype(mx.float16)

        return QuantizedKVCache(
            k_q=k_q_flat, v_q=v_q_flat,
            k_scales=k_scales, v_scales=v_scales,
            bits=8, group_size=config.group_size,
            original_shape=K_cache.shape,
            layout=config.layout,
        )
    elif config.bits == 4:
        D_packed = (D + 1) // 2
        Kf = K_cache.astype(mx.float32)
        Vf = V_cache.astype(mx.float32)
        k_q_flat = mx.zeros((B, MAX_S, Hkv, D_packed), dtype=mx.uint8)
        v_q_flat = mx.zeros((B, MAX_S, Hkv, D_packed), dtype=mx.uint8)
        k_scales = mx.zeros((B, MAX_S, Hkv, groups_per_head), dtype=mx.float16)
        v_scales = mx.zeros((B, MAX_S, Hkv, groups_per_head), dtype=mx.float16)

        for b in range(B):
            valid = int(lengths_arr[b].item())
            for h in range(Hkv):
                for s in range(valid):
                    chunk = Kf[b, s, h, :]
                    group_q = mx.zeros(D, dtype=mx.uint8)
                    for g in range(groups_per_head):
                        g_start = g * config.group_size
                        g_end = min(g_start + config.group_size, D)
                        group_vals = chunk[g_start:g_end]
                        q_u, sc = _quantize_group_symmetric_4(group_vals.reshape(1, -1))
                        group_q[g_start:g_end] = q_u.reshape(-1)
                        k_scales[b, s, h, g] = sc.reshape(-1).astype(mx.float16)
                    packed = pack_q4(group_q.reshape(1, D))
                    k_q_flat[b, s, h, :] = packed.reshape(-1)

                    chunk_v = Vf[b, s, h, :]
                    group_q = mx.zeros(D, dtype=mx.uint8)
                    for g in range(groups_per_head):
                        g_start = g * config.group_size
                        g_end = min(g_start + config.group_size, D)
                        group_vals = chunk_v[g_start:g_end]
                        q_u, sc = _quantize_group_symmetric_4(group_vals.reshape(1, -1))
                        group_q[g_start:g_end] = q_u.reshape(-1)
                        v_scales[b, s, h, g] = sc.reshape(-1).astype(mx.float16)
                    packed = pack_q4(group_q.reshape(1, D))
                    v_q_flat[b, s, h, :] = packed.reshape(-1)

        return QuantizedKVCache(
            k_q=k_q_flat, v_q=v_q_flat,
            k_scales=k_scales, v_scales=v_scales,
            bits=4, group_size=config.group_size,
            original_shape=K_cache.shape,
            layout=config.layout,
        )
    else:
        raise ValueError(f"bits must be 4 or 8, got {config.bits}")


def dequantize_kv_cache(qkv_cache: QuantizedKVCache) -> tuple[mx.array, mx.array]:
    qkv_cache = qkv_cache.validate()
    orig = qkv_cache.original_shape
    B, MAX_S, Hkv, D = orig if orig is not None else (qkv_cache.k_q.shape[0], qkv_cache.k_q.shape[1], qkv_cache.k_q.shape[2], qkv_cache.k_q.shape[3] if qkv_cache.bits == 8 else qkv_cache.k_q.shape[3] * 2)
    groups_per_head = (D + qkv_cache.group_size - 1) // qkv_cache.group_size

    K_deq = mx.zeros((B, MAX_S, Hkv, D), dtype=mx.float16)
    V_deq = mx.zeros((B, MAX_S, Hkv, D), dtype=mx.float16)

    if qkv_cache.bits == 8:
        for b in range(B):
            for h in range(Hkv):
                for s in range(MAX_S):
                    k_chunk = qkv_cache.k_q[b, s, h, :]
                    k_sc = qkv_cache.k_scales[b, s, h, :]
                    K_deq[b, s, h, :] = _dequant_group_symmetric_8(k_chunk.reshape(1, 1, -1), k_sc.reshape(1, 1, -1), groups_per_head, qkv_cache.group_size, D).reshape(-1).astype(mx.float16)
                    v_chunk = qkv_cache.v_q[b, s, h, :]
                    v_sc = qkv_cache.v_scales[b, s, h, :]
                    V_deq[b, s, h, :] = _dequant_group_symmetric_8(v_chunk.reshape(1, 1, -1), v_sc.reshape(1, 1, -1), groups_per_head, qkv_cache.group_size, D).reshape(-1).astype(mx.float16)
    elif qkv_cache.bits == 4:
        D_full = D
        for b in range(B):
            for h in range(Hkv):
                for s in range(MAX_S):
                    packed_k = qkv_cache.k_q[b, s, h, :].reshape(1, -1)
                    unpacked_k = unpack_q4_reference(packed_k, K=D_full).reshape(-1)
                    k_sc = qkv_cache.k_scales[b, s, h, :]
                    K_deq[b, s, h, :] = _dequant_group_symmetric_4(unpacked_k.reshape(1, 1, -1), k_sc.reshape(1, 1, -1), groups_per_head, qkv_cache.group_size, D).reshape(-1).astype(mx.float16)
                    packed_v = qkv_cache.v_q[b, s, h, :].reshape(1, -1)
                    unpacked_v = unpack_q4_reference(packed_v, K=D_full).reshape(-1)
                    v_sc = qkv_cache.v_scales[b, s, h, :]
                    V_deq[b, s, h, :] = _dequant_group_symmetric_4(unpacked_v.reshape(1, 1, -1), v_sc.reshape(1, 1, -1), groups_per_head, qkv_cache.group_size, D).reshape(-1).astype(mx.float16)

    return K_deq, V_deq


def quantized_kv_error(
    K_cache: mx.array,
    V_cache: mx.array,
    qkv_cache: QuantizedKVCache,
) -> dict[str, float]:
    K_deq, V_deq = dequantize_kv_cache(qkv_cache)
    K_cache_f = K_cache.astype(mx.float32)
    V_cache_f = V_cache.astype(mx.float32)
    K_deq_f = K_deq.astype(mx.float32)
    V_deq_f = V_deq.astype(mx.float32)

    k_diff = K_cache_f - K_deq_f
    v_diff = V_cache_f - V_deq_f

    cr = qkv_cache.compression_ratio(fp_bytes_per_value=2)

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
# Reference quantized KV decode attention
# ---------------------------------------------------------------------------


def _validate_quantized_decode_inputs(
    q: mx.array,
    qkv_cache: QuantizedKVCache,
    lengths,
) -> tuple[mx.array, int, int, int, int, int]:
    if q.ndim != 4:
        raise ValueError(f"q must be 4-D [B,1,Hq,D], got {q.shape}")
    if q.shape[1] != 1:
        raise ValueError(f"q must have shape [B,1,Hq,D], got {q.shape}")
    B, _, Hq, D = q.shape
    orig = qkv_cache.original_shape
    if orig is not None:
        cache_B, MAX_S, Hkv, D_cache = orig
    else:
        cache_B, MAX_S = qkv_cache.k_q.shape[0], qkv_cache.k_q.shape[1]
        Hkv = qkv_cache.k_q.shape[2]
        D_cache = qkv_cache.k_q.shape[3] if qkv_cache.bits == 8 else qkv_cache.k_q.shape[3] * 2

    if B != cache_B:
        raise ValueError(f"batch mismatch: q batch={B}, cache batch={cache_B}")
    if D != D_cache:
        raise ValueError(f"head_dim mismatch: q D={D}, cache D={D_cache}")
    validate_gqa_heads(Hq, Hkv)

    lengths_arr = normalize_positions(lengths if lengths is not None else MAX_S, B, MAX_S + 1)
    lengths_arr = mx.minimum(lengths_arr, mx.array(MAX_S, dtype=mx.int32))
    return lengths_arr, B, MAX_S, Hq, Hkv, D


def reference_quantized_kv_gqa_decode_attention(
    q: mx.array,
    qkv_cache: QuantizedKVCache,
    lengths=None,
    *,
    scale=None,
) -> mx.array:
    lengths_arr, B, MAX_S, Hq, Hkv, D = _validate_quantized_decode_inputs(q, qkv_cache, lengths)
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    K_deq, V_deq = dequantize_kv_cache(qkv_cache)
    Kf = K_deq.astype(mx.float32)
    Vf = V_deq.astype(mx.float32)
    qf = q.astype(mx.float32)

    outputs = []
    positions = mx.arange(MAX_S).reshape(1, MAX_S)
    for hq in range(Hq):
        hkv = q_head_to_kv_head(hq, Hq, Hkv)
        q_head = qf[:, :, hq:hq + 1, :]
        k_head = Kf[:, :, hkv:hkv + 1, :]
        v_head = Vf[:, :, hkv:hkv + 1, :]
        scores = mx.matmul(q_head.transpose(0, 2, 1, 3), k_head.transpose(0, 2, 3, 1)) * float(scale)
        valid_mask = positions.reshape(1, 1, 1, MAX_S) < lengths_arr.reshape(B, 1, 1, 1)
        scores = mx.where(valid_mask, scores, mx.array(-1.0e9, dtype=scores.dtype))
        probs = mx.softmax(scores, axis=-1)
        out_head = mx.matmul(probs, v_head.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        outputs.append(out_head)
    return mx.concatenate(outputs, axis=2).astype(q.dtype)


def _make_header(dtype: mx.Dtype) -> str:
    return make_metal_header(dtype, MAX_HEAD_DIM=_MAX_HEAD_DIM)


@lru_cache(maxsize=8)
def _get_q8_gqa_decode_kernel(kernel_name: str, dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name=kernel_name,
        input_names=["q", "K_q", "V_q", "K_scales", "V_scales", "lengths", "meta", "scale"],
        output_names=["out"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=8)
def _get_q4_gqa_decode_kernel(kernel_name: str, dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name=kernel_name,
        input_names=["q", "K_q", "V_q", "K_scales", "V_scales", "lengths", "meta", "scale"],
        output_names=["out"],
        source=source,
        header=header,
    )


def _metal_q8_gqa_decode_attention(
    q: mx.array,
    qkv_cache: QuantizedKVCache,
    lengths_arr: mx.array,
    *,
    scale: float,
) -> mx.array:
    B, MAX_S, Hq, Hkv, D = q.shape[0], qkv_cache.original_shape[1], q.shape[2], qkv_cache.original_shape[2], q.shape[3]
    groups_per_head = (D + qkv_cache.group_size - 1) // qkv_cache.group_size
    meta = mx.array(
        [B, MAX_S, Hq, Hkv, D, qkv_cache.group_size, groups_per_head],
        dtype=mx.int32,
    )
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    source = load_metal_source(_Q8_GQA_DECODE_KERNEL)
    header = _make_header(q.dtype)
    kernel = _get_q8_gqa_decode_kernel("q8_gqa_decode_attention_forward", str(q.dtype), source, header)
    return kernel(
        inputs=[
            q.astype(q.dtype),
            qkv_cache.k_q.astype(mx.uint8),
            qkv_cache.v_q.astype(mx.uint8),
            qkv_cache.k_scales.astype(mx.float32),
            qkv_cache.v_scales.astype(mx.float32),
            lengths_arr.astype(mx.int32),
            meta,
            scale_arr,
        ],
        output_shapes=[(B, 1, Hq, D)],
        output_dtypes=[q.dtype],
        grid=(B * Hq, 1, 1),
        threadgroup=(1, 1, 1),
    )[0]


def _metal_q4_gqa_decode_attention(
    q: mx.array,
    qkv_cache: QuantizedKVCache,
    lengths_arr: mx.array,
    *,
    scale: float,
) -> mx.array:
    B, MAX_S, Hq, Hkv, D = q.shape[0], qkv_cache.original_shape[1], q.shape[2], qkv_cache.original_shape[2], q.shape[3]
    D_packed = (D + 1) // 2
    groups_per_head = (D + qkv_cache.group_size - 1) // qkv_cache.group_size
    meta = mx.array(
        [B, MAX_S, Hq, Hkv, D, D_packed, qkv_cache.group_size, groups_per_head],
        dtype=mx.int32,
    )
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    source = load_metal_source(_Q4_GQA_DECODE_KERNEL)
    header = _make_header(q.dtype)
    kernel = _get_q4_gqa_decode_kernel("q4_gqa_decode_attention_forward", str(q.dtype), source, header)
    return kernel(
        inputs=[
            q.astype(q.dtype),
            qkv_cache.k_q.astype(mx.uint8),
            qkv_cache.v_q.astype(mx.uint8),
            qkv_cache.k_scales.astype(mx.float32),
            qkv_cache.v_scales.astype(mx.float32),
            lengths_arr.astype(mx.int32),
            meta,
            scale_arr,
        ],
        output_shapes=[(B, 1, Hq, D)],
        output_dtypes=[q.dtype],
        grid=(B * Hq, 1, 1),
        threadgroup=(1, 1, 1),
    )[0]


def quantized_kv_gqa_decode_attention(
    q: mx.array,
    qkv_cache: QuantizedKVCache,
    lengths=None,
    *,
    scale=None,
    backend: str = "reference",
) -> mx.array:
    lengths_arr, B, MAX_S, Hq, Hkv, D = _validate_quantized_decode_inputs(q, qkv_cache, lengths)
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    backend_name = backend.lower()
    if backend_name == "reference":
        return reference_quantized_kv_gqa_decode_attention(q, qkv_cache, lengths=lengths_arr, scale=scale)

    if backend_name == "metal_q8":
        if qkv_cache.bits != 8:
            raise ValueError(f"metal_q8 backend requires bits=8, got bits={qkv_cache.bits}")
        if D > _MAX_HEAD_DIM:
            raise ValueError(f"metal_q8 backend supports D <= {_MAX_HEAD_DIM}, got D={D}")
        return _metal_q8_gqa_decode_attention(q, qkv_cache, lengths_arr, scale=scale)

    if backend_name == "metal_q4":
        if qkv_cache.bits != 4:
            raise ValueError(f"metal_q4 backend requires bits=4, got bits={qkv_cache.bits}")
        if D > _MAX_HEAD_DIM:
            raise ValueError(f"metal_q4 backend supports D <= {_MAX_HEAD_DIM}, got D={D}")
        return _metal_q4_gqa_decode_attention(q, qkv_cache, lengths_arr, scale=scale)

    raise ValueError(f"backend must be one of 'reference', 'metal_q8', 'metal_q4', got {backend_name!r}")


# ---------------------------------------------------------------------------
# Sparse quantized KV decode attention (reference only for PR40)
# ---------------------------------------------------------------------------


def reference_quantized_kv_sparse_gqa_decode_attention(
    q: mx.array,
    qkv_cache: QuantizedKVCache,
    lengths,
    pattern: SparseAttentionPattern,
    *,
    scale=None,
) -> mx.array:
    lengths_arr, B, MAX_S, Hq, Hkv, D = _validate_quantized_decode_inputs(q, qkv_cache, lengths)
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    K_deq, V_deq = dequantize_kv_cache(qkv_cache)
    group = Hq // Hkv
    outputs = []
    for b in range(B):
        valid_len = int(lengths_arr[b].item())
        if valid_len <= 0:
            raise ValueError("decode sparse attention requires lengths >= 1")
        mask = build_sparse_attention_mask(1, valid_len, pattern, start_position=valid_len - 1)
        q_b = q[b : b + 1].astype(mx.float32)
        K_b = K_deq[b : b + 1, :valid_len].astype(mx.float32)
        V_b = V_deq[b : b + 1, :valid_len].astype(mx.float32)
        head_outputs = []
        for hq in range(Hq):
            hkv = hq // group
            q_head = q_b[:, :, hq:hq + 1, :]
            k_head = K_b[:, :, hkv:hkv + 1, :]
            v_head = V_b[:, :, hkv:hkv + 1, :]
            scores = mx.matmul(q_head.transpose(0, 2, 1, 3), k_head.transpose(0, 2, 3, 1)) * float(scale)
            if hasattr(mask, "tolist"):
                mask_rows = mask.tolist()
            else:
                mask_rows = mask
            row_mask = mx.array(mask_rows[0], dtype=mx.bool_).reshape(1, 1, 1, valid_len)
            scores = mx.where(row_mask, scores, mx.array(-1.0e9, dtype=scores.dtype))
            probs = mx.softmax(scores, axis=-1)
            head_outputs.append(mx.matmul(probs, v_head.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3))
        outputs.append(mx.concatenate(head_outputs, axis=2))
    return mx.concatenate(outputs, axis=0).astype(q.dtype)


def quantized_kv_sparse_gqa_decode_attention(
    q: mx.array,
    qkv_cache: QuantizedKVCache,
    lengths,
    pattern: SparseAttentionPattern,
    *,
    scale=None,
    backend: str = "reference",
) -> mx.array:
    if not isinstance(pattern, SparseAttentionPattern):
        raise TypeError("pattern must be a SparseAttentionPattern")
    backend_name = backend.lower()
    if backend_name == "reference":
        return reference_quantized_kv_sparse_gqa_decode_attention(q, qkv_cache, lengths, pattern, scale=scale)
    raise NotImplementedError(
        "Metal sparse+quantized backend is not implemented yet. "
        "Use backend='reference' or see sparse+gqa decode attention for fp16."
    )
