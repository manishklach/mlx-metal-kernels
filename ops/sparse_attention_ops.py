from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mlx.core as mx

from .decode_ops import normalize_positions
from .gqa_ops import validate_gqa_heads
from .kernel_utils import KERNEL_DIR, make_metal_header, load_metal_source

_KERNEL_DIR = KERNEL_DIR
_SPARSE_GQA_ATTENTION_KERNEL = _KERNEL_DIR / "sliding_window_gqa_attention.metal"
_SPARSE_GQA_DECODE_KERNEL = _KERNEL_DIR / "sliding_window_gqa_decode_attention.metal"
_MAX_HEAD_DIM = 128


@dataclass
class SparseAttentionPattern:
    pattern: str
    window_size: int | None = None
    sink_tokens: int = 0
    block_size: int | None = None
    block_mask: Any | None = None
    causal: bool = True

    def validate(self) -> "SparseAttentionPattern":
        if self.pattern not in ("dense", "sliding_window", "sliding_window_sink", "block_sparse"):
            raise ValueError(
                "pattern must be one of 'dense', 'sliding_window', 'sliding_window_sink', 'block_sparse'"
            )
        if self.sink_tokens < 0:
            raise ValueError(f"sink_tokens must be >= 0, got {self.sink_tokens}")
        if self.pattern == "sliding_window":
            if self.window_size is None or self.window_size <= 0:
                raise ValueError("sliding_window requires window_size > 0")
        if self.pattern == "sliding_window_sink":
            if self.window_size is None or self.window_size <= 0:
                raise ValueError("sliding_window_sink requires window_size > 0")
        if self.pattern == "block_sparse":
            if self.block_size is None or self.block_size <= 0:
                raise ValueError("block_sparse requires block_size > 0")
            if self.block_mask is None:
                raise ValueError("block_sparse requires block_mask")
        return self


def _make_header(dtype: mx.Dtype) -> str:
    return make_metal_header(dtype, MAX_HEAD_DIM=_MAX_HEAD_DIM)


@lru_cache(maxsize=8)
def _get_sparse_gqa_attention_kernel(kernel_name: str, dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name=kernel_name,
        input_names=["Q", "K", "V", "meta", "scale"],
        output_names=["O"],
        source=source,
        header=header,
    )


@lru_cache(maxsize=8)
def _get_sparse_gqa_decode_kernel(kernel_name: str, dtype_name: str, source: str, header: str):
    return mx.fast.metal_kernel(
        name=kernel_name,
        input_names=["q", "K_cache", "V_cache", "lengths", "meta", "scale"],
        output_names=["out"],
        source=source,
        header=header,
    )


def _bool_mask_to_rows(mask: mx.array) -> list[list[bool]]:
    mask_list = mask.tolist()
    return [[bool(v) for v in row] for row in mask_list]


def build_sparse_attention_mask(
    Sq,
    Sk,
    pattern: SparseAttentionPattern,
    *,
    start_position=0,
):
    pattern = pattern.validate()
    if Sq <= 0 or Sk <= 0:
        raise ValueError(f"Sq and Sk must be positive, got {Sq}, {Sk}")
    if start_position < 0:
        raise ValueError(f"start_position must be >= 0, got {start_position}")
    rows: list[list[bool]] = []
    for i in range(Sq):
        q_abs = start_position + i
        row: list[bool] = []
        for j in range(Sk):
            visible = False
            if pattern.pattern == "dense":
                visible = True if not pattern.causal else (j <= q_abs)
            elif pattern.pattern == "sliding_window":
                local_start = max(0, q_abs - int(pattern.window_size) + 1)
                visible = local_start <= j <= q_abs
            elif pattern.pattern == "sliding_window_sink":
                local_start = max(0, q_abs - int(pattern.window_size) + 1)
                visible = ((j < pattern.sink_tokens) and (j <= q_abs)) or (local_start <= j <= q_abs)
            elif pattern.pattern == "block_sparse":
                block_size = int(pattern.block_size)
                q_block = q_abs // block_size
                k_block = j // block_size
                block_mask = pattern.block_mask
                if hasattr(block_mask, "tolist"):
                    block_mask = block_mask.tolist()
                visible = bool(block_mask[q_block][k_block])
                if pattern.causal:
                    visible = visible and (j <= q_abs)
            row.append(bool(visible))
        rows.append(row)
    return mx.array(rows, dtype=mx.bool_)


def _validate_prefill_inputs(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    pattern: SparseAttentionPattern,
    *,
    require_metal_supported: bool = False,
) -> tuple[int, int, int, int, int, int]:
    pattern = pattern.validate()
    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        raise ValueError(f"Q, K, and V must be rank-4 [B,S,H,D], got {Q.shape}, {K.shape}, {V.shape}")
    if K.shape != V.shape:
        raise ValueError(f"K and V must have identical shapes, got {K.shape}, {V.shape}")
    B, Sq, Hq, D = Q.shape
    Bk, Sk, Hkv, Dk = K.shape
    if B != Bk or D != Dk:
        raise ValueError(f"Q, K, and V must agree on batch and head_dim, got {Q.shape}, {K.shape}, {V.shape}")
    validate_gqa_heads(Hq, Hkv)
    if require_metal_supported and D > _MAX_HEAD_DIM:
        raise ValueError(f"Metal sparse attention currently supports D <= {_MAX_HEAD_DIM}, got {D}")
    if Q.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"Q dtype must be float16 or bfloat16, got {Q.dtype}")
    if K.dtype not in (mx.float16, mx.bfloat16) or V.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"K/V dtype must be float16 or bfloat16, got {K.dtype}, {V.dtype}")
    if pattern.sink_tokens > Sk:
        raise ValueError(f"sink_tokens must be <= key sequence length, got {pattern.sink_tokens} > {Sk}")
    if pattern.pattern.startswith("sliding_window") and pattern.causal is False:
        # Non-causal support can be added later; keep current sparse kernels explicit.
        raise NotImplementedError("non-causal sparse sliding-window attention is not implemented yet")
    return B, Sq, Sk, Hq, Hkv, D


def _validate_decode_inputs(
    q: mx.array,
    K_cache: mx.array,
    V_cache: mx.array,
    lengths,
    pattern: SparseAttentionPattern,
    *,
    require_metal_supported: bool = False,
) -> tuple[mx.array, int, int, int, int, int]:
    pattern = pattern.validate()
    if q.ndim != 4 or K_cache.ndim != 4 or V_cache.ndim != 4:
        raise ValueError(f"q, K_cache, and V_cache must be rank-4, got {q.shape}, {K_cache.shape}, {V_cache.shape}")
    if q.shape[1] != 1:
        raise ValueError(f"q must have shape [B,1,Hq,D], got {q.shape}")
    if K_cache.shape != V_cache.shape:
        raise ValueError(f"K_cache and V_cache must have identical shapes, got {K_cache.shape}, {V_cache.shape}")
    B, _, Hq, D = q.shape
    Bk, MAX_S, Hkv, Dk = K_cache.shape
    if B != Bk or D != Dk:
        raise ValueError(f"q and caches must agree on batch/head_dim, got {q.shape}, {K_cache.shape}, {V_cache.shape}")
    validate_gqa_heads(Hq, Hkv)
    if require_metal_supported and D > _MAX_HEAD_DIM:
        raise ValueError(f"Metal sparse decode attention currently supports D <= {_MAX_HEAD_DIM}, got {D}")
    if q.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"q dtype must be float16 or bfloat16, got {q.dtype}")
    if K_cache.dtype not in (mx.float16, mx.bfloat16) or V_cache.dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(f"K_cache/V_cache dtype must be float16 or bfloat16, got {K_cache.dtype}, {V_cache.dtype}")
    lengths_arr = normalize_positions(lengths if lengths is not None else MAX_S, B, MAX_S + 1)
    lengths_arr = mx.minimum(lengths_arr, mx.array(MAX_S, dtype=mx.int32))
    return lengths_arr, B, MAX_S, Hq, Hkv, D


def _dense_masked_reference_attention(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    mask: mx.array,
    *,
    scale: float,
) -> mx.array:
    B, Sq, Hq, D = Q.shape
    _, Sk, Hkv, _ = K.shape
    group = Hq // Hkv
    mask_rows = _bool_mask_to_rows(mask)
    if any(not any(row) for row in mask_rows):
        raise ValueError("sparse attention mask produced a query row with no visible keys")
    outputs = []
    for hq in range(Hq):
        hkv = hq // group
        q_head = Q[:, :, hq:hq + 1, :].astype(mx.float32)
        k_head = K[:, :, hkv:hkv + 1, :].astype(mx.float32)
        v_head = V[:, :, hkv:hkv + 1, :].astype(mx.float32)
        row_outputs = []
        for i, row in enumerate(mask_rows):
            visible = [j for j, flag in enumerate(row) if flag]
            q_row = q_head[:, i : i + 1, :, :]
            k_visible = k_head[:, visible, :, :]
            v_visible = v_head[:, visible, :, :]
            scores = mx.matmul(q_row.transpose(0, 2, 1, 3), k_visible.transpose(0, 2, 3, 1)) * float(scale)
            probs = mx.softmax(scores, axis=-1)
            row_out = mx.matmul(probs, v_visible.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
            row_outputs.append(row_out)
        outputs.append(mx.concatenate(row_outputs, axis=1))
    return mx.concatenate(outputs, axis=2).astype(Q.dtype)


def reference_sparse_gqa_attention(
    Q,
    K,
    V,
    pattern: SparseAttentionPattern,
    *,
    scale=None,
    start_position=0,
):
    _, Sq, Sk, _, _, D = _validate_prefill_inputs(Q, K, V, pattern)
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    mask = build_sparse_attention_mask(Sq, Sk, pattern, start_position=start_position)
    return _dense_masked_reference_attention(Q, K, V, mask, scale=float(scale))


def reference_sparse_gqa_decode_attention(
    q,
    K_cache,
    V_cache,
    lengths,
    pattern: SparseAttentionPattern,
    *,
    scale=None,
):
    lengths_arr, B, MAX_S, Hq, Hkv, D = _validate_decode_inputs(q, K_cache, V_cache, lengths, pattern)
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    group = Hq // Hkv
    outputs = []
    for b in range(B):
        valid_len = int(lengths_arr[b].item())
        if valid_len <= 0:
            raise ValueError("decode sparse attention requires lengths >= 1")
        if pattern.sink_tokens > valid_len:
            raise ValueError(f"sink_tokens must be <= sequence length at runtime, got {pattern.sink_tokens} > {valid_len}")
        mask = build_sparse_attention_mask(1, valid_len, pattern, start_position=valid_len - 1)
        mask_rows = _bool_mask_to_rows(mask)
        if not any(mask_rows[0]):
            raise ValueError("sparse decode attention mask produced no visible keys")
        q_b = q[b : b + 1].astype(mx.float32)
        K_b = K_cache[b : b + 1, :valid_len].astype(mx.float32)
        V_b = V_cache[b : b + 1, :valid_len].astype(mx.float32)
        head_outputs = []
        for hq in range(Hq):
            hkv = hq // group
            q_head = q_b[:, :, hq:hq + 1, :]
            k_head = K_b[:, :, hkv:hkv + 1, :]
            v_head = V_b[:, :, hkv:hkv + 1, :]
            scores = mx.matmul(q_head.transpose(0, 2, 1, 3), k_head.transpose(0, 2, 3, 1)) * float(scale)
            row_mask = mx.array(mask_rows[0], dtype=mx.bool_).reshape(1, 1, 1, valid_len)
            scores = mx.where(row_mask, scores, mx.array(-1.0e9, dtype=scores.dtype))
            probs = mx.softmax(scores, axis=-1)
            head_outputs.append(mx.matmul(probs, v_head.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3))
        outputs.append(mx.concatenate(head_outputs, axis=2))
    return mx.concatenate(outputs, axis=0).astype(q.dtype)


def sparse_gqa_attention(
    Q,
    K,
    V,
    pattern,
    *,
    scale=None,
    start_position=0,
    backend="reference",
):
    if not isinstance(pattern, SparseAttentionPattern):
        raise TypeError("pattern must be a SparseAttentionPattern")
    backend_name = backend.lower()
    B, Sq, Sk, Hq, Hkv, D = _validate_prefill_inputs(Q, K, V, pattern, require_metal_supported=backend_name != "reference")
    if pattern.sink_tokens > Sk:
        raise ValueError(f"sink_tokens must be <= sequence length at runtime, got {pattern.sink_tokens} > {Sk}")
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if backend_name == "reference":
        return reference_sparse_gqa_attention(Q, K, V, pattern, scale=scale, start_position=start_position)
    if backend_name == "metal_block_sparse":
        raise NotImplementedError("metal_block_sparse is not implemented yet; use backend='reference'")
    if backend_name not in ("metal_sliding_window", "metal_sliding_window_sink"):
        raise ValueError(
            "backend must be one of 'reference', 'metal_sliding_window', 'metal_sliding_window_sink', 'metal_block_sparse'"
        )
    if pattern.pattern == "block_sparse":
        raise NotImplementedError("block_sparse Metal backend is not implemented yet")
    if pattern.pattern == "dense":
        raise ValueError("dense sparse attention should use backend='reference' or the existing dense attention APIs")
    if backend_name == "metal_sliding_window" and pattern.pattern != "sliding_window":
        raise ValueError("backend='metal_sliding_window' requires pattern='sliding_window'")
    if backend_name == "metal_sliding_window_sink" and pattern.pattern not in ("sliding_window", "sliding_window_sink"):
        raise ValueError("backend='metal_sliding_window_sink' requires a sliding-window pattern")

    dtype = Q.dtype
    meta = mx.array(
        [B, Sq, Sk, Hq, Hkv, D, int(pattern.window_size or 0), int(pattern.sink_tokens), int(start_position)],
        dtype=mx.int32,
    )
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    source = load_metal_source(_SPARSE_GQA_ATTENTION_KERNEL)
    header = _make_header(dtype)
    kernel = _get_sparse_gqa_attention_kernel("sliding_window_gqa_attention_forward", str(dtype), source, header)
    return kernel(
        inputs=[Q.astype(dtype), K.astype(dtype), V.astype(dtype), meta, scale_arr],
        output_shapes=[(B, Sq, Hq, D)],
        output_dtypes=[dtype],
        grid=(B * Sq * Hq, 1, 1),
        threadgroup=(1, 1, 1),
    )[0]


def sparse_gqa_decode_attention(
    q,
    K_cache,
    V_cache,
    lengths,
    pattern,
    *,
    scale=None,
    backend="reference",
):
    if not isinstance(pattern, SparseAttentionPattern):
        raise TypeError("pattern must be a SparseAttentionPattern")
    backend_name = backend.lower()
    lengths_arr, B, MAX_S, Hq, Hkv, D = _validate_decode_inputs(
        q,
        K_cache,
        V_cache,
        lengths,
        pattern,
        require_metal_supported=backend_name != "reference",
    )
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if backend_name == "reference":
        return reference_sparse_gqa_decode_attention(q, K_cache, V_cache, lengths_arr, pattern, scale=scale)
    if backend_name not in ("metal_sliding_window", "metal_sliding_window_sink"):
        raise ValueError("backend must be one of 'reference', 'metal_sliding_window', 'metal_sliding_window_sink'")
    if pattern.pattern == "block_sparse":
        raise NotImplementedError("block_sparse Metal decode backend is not implemented yet")
    if pattern.pattern == "dense":
        raise ValueError("dense sparse decode attention should use backend='reference' or the existing dense decode APIs")
    if backend_name == "metal_sliding_window" and pattern.pattern != "sliding_window":
        raise ValueError("backend='metal_sliding_window' requires pattern='sliding_window'")
    if backend_name == "metal_sliding_window_sink" and pattern.pattern not in ("sliding_window", "sliding_window_sink"):
        raise ValueError("backend='metal_sliding_window_sink' requires a sliding-window pattern")

    for b in range(B):
        valid_len = int(lengths_arr[b].item())
        if valid_len <= 0:
            raise ValueError("decode sparse attention requires lengths >= 1")
        if pattern.sink_tokens > valid_len:
            raise ValueError(f"sink_tokens must be <= sequence length at runtime, got {pattern.sink_tokens} > {valid_len}")

    dtype = q.dtype
    meta = mx.array([B, MAX_S, Hq, Hkv, D, int(pattern.window_size or 0), int(pattern.sink_tokens)], dtype=mx.int32)
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    source = load_metal_source(_SPARSE_GQA_DECODE_KERNEL)
    header = _make_header(dtype)
    kernel = _get_sparse_gqa_decode_kernel("sliding_window_gqa_decode_attention_forward", str(dtype), source, header)
    return kernel(
        inputs=[q.astype(dtype), K_cache.astype(dtype), V_cache.astype(dtype), lengths_arr.astype(mx.int32), meta, scale_arr],
        output_shapes=[(B, 1, Hq, D)],
        output_dtypes=[dtype],
        grid=(B * Hq, 1, 1),
        threadgroup=(1, 1, 1),
    )[0]
