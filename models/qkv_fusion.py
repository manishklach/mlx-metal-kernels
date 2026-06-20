from __future__ import annotations

from .checkpoint_manifest import CheckpointManifest, TensorInfo
from .checkpoint_mapping import llama_layer_tensor_names
from .llama_config import LlamaLikeConfig


def fused_qkv_shape(config: LlamaLikeConfig) -> tuple[int, int]:
    return (config.q_output_dim() + 2 * config.kv_output_dim(), config.hidden_size)


def fuse_qkv_shapes(q_shape, k_shape, v_shape):
    q_shape = tuple(q_shape)
    k_shape = tuple(k_shape)
    v_shape = tuple(v_shape)
    if len(q_shape) != 2 or len(k_shape) != 2 or len(v_shape) != 2:
        raise ValueError("q_shape, k_shape, and v_shape must all be rank-2")
    if q_shape[1] != k_shape[1] or q_shape[1] != v_shape[1]:
        raise ValueError(f"q/k/v input dims must match, got {q_shape}, {k_shape}, {v_shape}")
    return (q_shape[0] + k_shape[0] + v_shape[0], q_shape[1])


def split_fused_qkv_shape(fused_shape, config: LlamaLikeConfig):
    fused_shape = tuple(fused_shape)
    expected = fused_qkv_shape(config)
    if fused_shape != expected:
        raise ValueError(f"fused_shape must be {expected}, got {fused_shape}")
    return (
        (config.q_output_dim(), config.hidden_size),
        (config.kv_output_dim(), config.hidden_size),
        (config.kv_output_dim(), config.hidden_size),
    )


def _concat(values, axis):
    try:
        import mlx.core as mx

        return mx.concatenate(values, axis=axis)
    except Exception:  # noqa: BLE001
        try:
            import numpy as np

            return np.concatenate(values, axis=axis)
        except Exception:  # noqa: BLE001
            pass
        if axis != 0:
            raise ValueError("shape-only fallback supports only axis=0")
        out = []
        for value in values:
            out.extend(value)
        return out


def fuse_qkv_weights(q_w, k_w, v_w, *, axis=0):
    q_shape = tuple(q_w.shape)
    k_shape = tuple(k_w.shape)
    v_shape = tuple(v_w.shape)
    fuse_qkv_shapes(q_shape, k_shape, v_shape)
    if axis != 0:
        raise ValueError("fuse_qkv_weights currently supports only axis=0")
    return _concat([q_w, k_w, v_w], axis=axis)


def split_fused_qkv_weight(fused_w, config: LlamaLikeConfig, *, axis=0):
    if axis != 0:
        raise ValueError("split_fused_qkv_weight currently supports only axis=0")
    q_shape, k_shape, v_shape = split_fused_qkv_shape(tuple(fused_w.shape), config)
    q_rows = q_shape[0]
    k_rows = k_shape[0]
    return (
        fused_w[:q_rows, :],
        fused_w[q_rows:q_rows + k_rows, :],
        fused_w[q_rows + k_rows:, :],
    )


def build_fused_qkv_manifest_entries(manifest: CheckpointManifest, config: LlamaLikeConfig, *, layer_idx) -> TensorInfo:
    names = llama_layer_tensor_names(layer_idx)
    q_info = manifest.require(names.q_proj)
    k_info = manifest.require(names.k_proj)
    v_info = manifest.require(names.v_proj)
    dtypes = {q_info.dtype, k_info.dtype, v_info.dtype}
    dtype = q_info.dtype if len(dtypes) == 1 else "mixed"
    return TensorInfo(
        name=f"model.layers.{layer_idx}.self_attn.qkv_proj.fused_weight",
        shape=fused_qkv_shape(config),
        dtype=dtype,
        source="derived:fused_qkv",
        metadata={"source_tensors": [q_info.name, k_info.name, v_info.name]},
    )


def create_fused_qkv_manifest(manifest: CheckpointManifest, config: LlamaLikeConfig) -> CheckpointManifest:
    tensors = dict(manifest.tensors)
    for layer_idx in range(config.num_hidden_layers):
        fused = build_fused_qkv_manifest_entries(manifest, config, layer_idx=layer_idx)
        tensors[fused.name] = fused
    return CheckpointManifest(model_type=manifest.model_type, tensors=tensors, metadata=dict(manifest.metadata))
