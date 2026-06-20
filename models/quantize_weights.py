from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - optional MLX path
    mx = None

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - numpy is expected for this scaffold
    raise RuntimeError("numpy is required for quantization helpers") from exc


def _is_mlx_array(value: Any) -> bool:
    return mx is not None and type(value).__module__.startswith("mlx")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except Exception:  # noqa: BLE001
        pass
    if hasattr(value, "tolist"):
        return np.asarray(value.tolist())
    raise TypeError(f"Unsupported tensor type for quantization: {type(value)!r}")


def _cast_back(value: np.ndarray, template: Any):
    if _is_mlx_array(template):
        return mx.array(value)
    return value


def _scale_dtype(dtype_name: str):
    if dtype_name == "float16":
        return np.float16
    if dtype_name == "float32":
        return np.float32
    if dtype_name == "bfloat16":
        try:
            return np.dtype("bfloat16")
        except TypeError:  # pragma: no cover - older numpy
            return np.float32
    raise ValueError(f"dtype must be one of 'float16', 'bfloat16', 'float32', got {dtype_name!r}")


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _pack_q4_numpy(q_unpacked: np.ndarray) -> np.ndarray:
    if q_unpacked.ndim != 2:
        raise ValueError(f"q_unpacked must be 2-D [M,K], got {q_unpacked.shape}")
    q_int = q_unpacked.astype(np.uint8, copy=False)
    if q_int.shape[1] % 2 != 0:
        q_int = np.pad(q_int, ((0, 0), (0, 1)), mode="constant")
    low = q_int[:, 0::2]
    high = q_int[:, 1::2]
    return (low + (high << 4)).astype(np.uint8, copy=False)


def _unpack_q4_numpy(packed: np.ndarray, k_dim: int) -> np.ndarray:
    packed_u8 = packed.astype(np.uint8, copy=False)
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    unpacked = np.stack([low, high], axis=-1).reshape(packed.shape[0], packed.shape[1] * 2)
    return unpacked[:, :k_dim]


def _symmetric_zero_point(bits: int) -> float:
    if bits == 4:
        return 8.0
    if bits == 8:
        return 128.0
    raise ValueError(f"bits must be 4 or 8, got {bits}")


def _materialize_symmetric_zeros(scales: Any, bits: int):
    zeros = np.full(_shape_tuple(scales), _symmetric_zero_point(bits), dtype=np.float32)
    return _cast_back(zeros, scales)


@dataclass
class QuantizationConfig:
    bits: int = 4
    group_size: int = 32
    symmetric: bool = True
    with_zeros: bool = False
    dtype: str = "float16"
    eps: float = 1e-8

    def validate(self) -> "QuantizationConfig":
        if self.bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {self.bits}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        _scale_dtype(self.dtype)
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        return self


@dataclass
class QuantizedWeight:
    name: str | None
    bits: int
    group_size: int
    packed_weight: Any
    scales: Any
    zeros: Any | None = None
    original_shape: tuple[int, int] | None = None
    packed_shape: tuple[int, ...] | None = None
    scale_shape: tuple[int, ...] | None = None
    symmetric: bool = True

    def shapes(self) -> dict[str, tuple[int, ...] | None]:
        return {
            "original_shape": self.original_shape,
            "packed_shape": self.packed_shape or _shape_tuple(self.packed_weight),
            "scale_shape": self.scale_shape or _shape_tuple(self.scales),
            "zeros_shape": _shape_tuple(self.zeros),
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bits": self.bits,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            **self.shapes(),
        }

    def validate(self) -> "QuantizedWeight":
        if self.bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {self.bits}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if self.original_shape is None:
            raise ValueError("original_shape must be provided")
        out_dim, in_dim = self.original_shape
        expected_packed = (out_dim, math.ceil(in_dim / 2)) if self.bits == 4 else (out_dim, in_dim)
        packed_shape = self.packed_shape or _shape_tuple(self.packed_weight)
        scale_shape = self.scale_shape or _shape_tuple(self.scales)
        if packed_shape != expected_packed:
            raise ValueError(f"packed_weight must have shape {expected_packed}, got {packed_shape}")
        expected_groups = math.ceil(in_dim / self.group_size)
        if scale_shape != (out_dim, expected_groups):
            raise ValueError(f"scales must have shape {(out_dim, expected_groups)}, got {scale_shape}")
        zeros_shape = _shape_tuple(self.zeros)
        if zeros_shape is not None and zeros_shape != (out_dim, expected_groups):
            raise ValueError(f"zeros must have shape {(out_dim, expected_groups)}, got {zeros_shape}")
        return self


def quantize_weight_groupwise(weight, config: QuantizationConfig) -> QuantizedWeight:
    config = config.validate()
    weight_np = _to_numpy(weight).astype(np.float32, copy=False)
    if weight_np.ndim != 2:
        raise ValueError(f"weight must be rank-2 [OUT_DIM, IN_DIM], got {weight_np.shape}")
    out_dim, in_dim = weight_np.shape
    groups = math.ceil(in_dim / config.group_size)
    q = np.zeros((out_dim, in_dim), dtype=np.uint8)
    scales = np.zeros((out_dim, groups), dtype=np.float32)
    if not config.symmetric:
        raise NotImplementedError(
            "Asymmetric quantization is not implemented in this scaffold because the existing kernel path is defined around symmetric packaging."
        )
    signed_limit = 7 if config.bits == 4 else 127
    zero_point = int(_symmetric_zero_point(config.bits))
    for row in range(out_dim):
        for group_idx in range(groups):
            start = group_idx * config.group_size
            end = min(start + config.group_size, in_dim)
            group = weight_np[row, start:end]
            max_abs = float(np.max(np.abs(group))) if group.size else 0.0
            scale = max(max_abs / float(signed_limit), config.eps)
            q_signed = np.rint(group / scale)
            q_signed = np.clip(q_signed, -signed_limit, signed_limit)
            q[row, start:end] = (q_signed.astype(np.int32) + zero_point).astype(np.uint8)
            scales[row, group_idx] = scale
    packed = _pack_q4_numpy(q) if config.bits == 4 else q
    scales = scales.astype(_scale_dtype(config.dtype), copy=False)
    return QuantizedWeight(
        name=None,
        bits=config.bits,
        group_size=config.group_size,
        packed_weight=_cast_back(packed, weight),
        scales=_cast_back(scales, weight),
        zeros=None,
        original_shape=(out_dim, in_dim),
        packed_shape=tuple(int(dim) for dim in packed.shape),
        scale_shape=tuple(int(dim) for dim in scales.shape),
        symmetric=config.symmetric,
    ).validate()


def dequantize_quantized_weight(q_weight, scales, zeros=None, *, bits=4, group_size=32):
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    q_shape = _shape_tuple(q_weight)
    scale_shape = _shape_tuple(scales)
    if q_shape is None or scale_shape is None:
        raise TypeError("q_weight and scales must expose shape")
    groups = scale_shape[1]
    zeros_for_kernel = zeros if zeros is not None else _materialize_symmetric_zeros(scales, bits)
    if mx is not None and _is_mlx_array(q_weight):
        from ops.quant_ops import dequant_q4, dequant_q8

        if bits == 4:
            return dequant_q4(q_weight, scales, zeros_for_kernel, group_size=group_size, out_dtype=mx.float16, backend="reference")
        return dequant_q8(q_weight, scales, zeros_for_kernel, group_size=group_size, out_dtype=mx.float16, backend="reference")
    q_np = _to_numpy(q_weight)
    scales_np = _to_numpy(scales).astype(np.float32, copy=False)
    zeros_np = _to_numpy(zeros_for_kernel).astype(np.float32, copy=False)
    if bits == 4:
        q_unpacked = _unpack_q4_numpy(q_np, groups * group_size).astype(np.float32, copy=False)
        zero_full = np.repeat(zeros_np, group_size, axis=1)[:, :q_unpacked.shape[1]]
        scale_full = np.repeat(scales_np, group_size, axis=1)[:, :q_unpacked.shape[1]]
        return (q_unpacked - zero_full) * scale_full
    zero_full = np.repeat(zeros_np, group_size, axis=1)[:, :q_np.shape[1]]
    scale_full = np.repeat(scales_np, group_size, axis=1)[:, :q_np.shape[1]]
    return (q_np.astype(np.float32, copy=False) - zero_full) * scale_full


def quantization_error(original, dequantized) -> dict[str, float]:
    original_np = _to_numpy(original).astype(np.float32, copy=False)
    dequantized_np = _to_numpy(dequantized).astype(np.float32, copy=False)
    diff = dequantized_np - original_np
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom = float(np.sqrt(np.mean(original_np * original_np)))
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "rmse": rmse,
        "relative_rmse": float(rmse / max(denom, 1e-12)),
    }
