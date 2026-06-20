from __future__ import annotations

import math
from typing import Any

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - optional path
    mx = None

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for sampling helpers") from exc


def _is_mlx_array(value: Any) -> bool:
    return mx is not None and type(value).__module__.startswith("mlx")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except Exception:  # noqa: BLE001
        if hasattr(value, "tolist"):
            return np.asarray(value.tolist())
        raise


def _cast_like(value: np.ndarray, template: Any):
    if _is_mlx_array(template):
        return mx.array(value)
    return value


def _validate_temperature(temperature: float) -> None:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")


def _validate_top_k(top_k: int | None) -> None:
    if top_k is not None and top_k <= 0:
        raise ValueError(f"top_k must be positive when provided, got {top_k}")


def _validate_top_p(top_p: float | None) -> None:
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")


def softmax(logits, axis: int = -1):
    logits_np = _to_numpy(logits).astype(np.float64, copy=False)
    shifted = logits_np - np.max(logits_np, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    out = exp / np.sum(exp, axis=axis, keepdims=True)
    return _cast_like(out.astype(np.float32, copy=False), logits)


def top_k_filter(logits, k: int):
    _validate_top_k(k)
    logits_np = _to_numpy(logits).astype(np.float64, copy=True)
    if logits_np.ndim == 1:
        threshold = np.partition(logits_np, -k)[-k]
        logits_np[logits_np < threshold] = -np.inf
        return _cast_like(logits_np.astype(np.float32, copy=False), logits)
    filtered = np.full_like(logits_np, -np.inf)
    indices = np.argpartition(logits_np, -k, axis=-1)[..., -k:]
    np.put_along_axis(filtered, indices, np.take_along_axis(logits_np, indices, axis=-1), axis=-1)
    return _cast_like(filtered.astype(np.float32, copy=False), logits)


def top_p_filter(logits, p: float):
    _validate_top_p(p)
    logits_np = _to_numpy(logits).astype(np.float64, copy=True)
    original_shape = logits_np.shape
    if logits_np.ndim == 1:
        logits_np = logits_np.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    filtered = np.full_like(logits_np, -np.inf)
    probs = _to_numpy(softmax(logits_np, axis=-1)).astype(np.float64, copy=False)
    for row_idx in range(logits_np.shape[0]):
        sort_idx = np.argsort(logits_np[row_idx])[::-1]
        sorted_probs = probs[row_idx, sort_idx]
        cumulative = np.cumsum(sorted_probs)
        keep_count = int(np.searchsorted(cumulative, p, side="left")) + 1
        keep_idx = sort_idx[:keep_count]
        filtered[row_idx, keep_idx] = logits_np[row_idx, keep_idx]
    if squeeze:
        filtered = filtered.reshape(original_shape)
    return _cast_like(filtered.astype(np.float32, copy=False), logits)


def apply_repetition_penalty(logits, generated_ids, penalty: float = 1.0):
    if penalty < 1.0:
        raise ValueError(f"repetition_penalty must be >= 1.0, got {penalty}")
    logits_np = _to_numpy(logits).astype(np.float64, copy=True)
    if penalty == 1.0:
        return _cast_like(logits_np.astype(np.float32, copy=False), logits)
    if logits_np.ndim == 1:
        repeated = set(int(token_id) for token_id in generated_ids)
        for token_id in repeated:
            if 0 <= token_id < logits_np.shape[0]:
                logits_np[token_id] = logits_np[token_id] / penalty if logits_np[token_id] >= 0 else logits_np[token_id] * penalty
        return _cast_like(logits_np.astype(np.float32, copy=False), logits)
    if not generated_ids:
        return _cast_like(logits_np.astype(np.float32, copy=False), logits)
    batch_ids = generated_ids if isinstance(generated_ids[0], (list, tuple)) else [generated_ids] * logits_np.shape[0]
    for row_idx, row_ids in enumerate(batch_ids):
        for token_id in set(int(item) for item in row_ids):
            if 0 <= token_id < logits_np.shape[1]:
                value = logits_np[row_idx, token_id]
                logits_np[row_idx, token_id] = value / penalty if value >= 0 else value * penalty
    return _cast_like(logits_np.astype(np.float32, copy=False), logits)


def greedy_sample(logits):
    logits_np = _to_numpy(logits)
    if logits_np.ndim == 1:
        return int(np.argmax(logits_np))
    return [int(item) for item in np.argmax(logits_np, axis=-1).tolist()]


def sample_logits(
    logits,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
):
    _validate_temperature(temperature)
    _validate_top_k(top_k)
    _validate_top_p(top_p)
    logits_np = _to_numpy(logits).astype(np.float64, copy=False)
    if logits_np.ndim not in (1, 2):
        raise ValueError(f"logits must have shape [vocab] or [B,vocab], got {logits_np.shape}")
    working = logits_np / float(temperature)
    if top_k is not None:
        working = _to_numpy(top_k_filter(working, top_k)).astype(np.float64, copy=False)
    if top_p is not None:
        working = _to_numpy(top_p_filter(working, top_p)).astype(np.float64, copy=False)
    probs = _to_numpy(softmax(working, axis=-1)).astype(np.float64, copy=False)
    rng = np.random.default_rng(seed)
    if probs.ndim == 1:
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        total = probs.sum()
        if total <= 0:
            return int(np.argmax(working))
        probs = probs / total
        return int(rng.choice(probs.shape[0], p=probs))
    outputs: list[int] = []
    for row_idx, row in enumerate(probs):
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        total = row.sum()
        if total <= 0:
            outputs.append(int(np.argmax(working[row_idx])))
            continue
        row = row / total
        outputs.append(int(rng.choice(probs.shape[1], p=row)))
    return outputs
