from __future__ import annotations

import math
import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - optional path
    mx = None

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for the prefill scaffold") from exc

from models.llama_config import LlamaLikeConfig
from models.quantize_weights import dequantize_quantized_weight

if mx is not None:  # pragma: no branch
    from .gqa_ops import gqa_attention, reference_gqa_attention, reference_gqa_qkv_split_rope
    from .llama_layer_ops import LlamaLayerKernelWeights
    from .mlp_block_ops import quantized_linear, quantized_mlp_block, reference_quantized_mlp_block
    from .norm_ops import reference_rms_norm, rms_norm


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


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _build_rope_tables_numpy(config: LlamaLikeConfig, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    if config.head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {config.head_dim}")
    positions = np.arange(seq_len, dtype=np.float32)
    inv_freq = 1.0 / (float(config.rope_theta) ** (np.arange(0, config.head_dim, 2, dtype=np.float32) / float(config.head_dim)))
    freqs = positions[:, None] * inv_freq[None, :]
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def _rms_norm_numpy(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    return (x / rms) * weight.reshape((1,) * (x.ndim - 1) + (-1,))


def _silu_numpy(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _apply_rope_numpy(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, *, position_offset: int = 0) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"x must have shape [B,S,H,D], got {x.shape}")
    seq_len = x.shape[1]
    cos_slice = cos[position_offset:position_offset + seq_len].astype(np.float32, copy=False)
    sin_slice = sin[position_offset:position_offset + seq_len].astype(np.float32, copy=False)
    if cos_slice.shape[0] != seq_len or sin_slice.shape[0] != seq_len:
        raise ValueError("cos/sin do not cover the requested positions")
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    cos_b = cos_slice[None, :, None, :]
    sin_b = sin_slice[None, :, None, :]
    out = np.empty_like(x, dtype=np.float32)
    out[..., ::2] = x_even * cos_b - x_odd * sin_b
    out[..., 1::2] = x_even * sin_b + x_odd * cos_b
    return out


def _repeat_kv_heads_numpy(kv: np.ndarray, num_attention_heads: int) -> np.ndarray:
    group = num_attention_heads // kv.shape[2]
    return np.repeat(kv, group, axis=2)


def _causal_attention_numpy(q: np.ndarray, k: np.ndarray, v: np.ndarray, *, scale: float) -> np.ndarray:
    scores = np.einsum("bthd,bshd->bhts", q, k).astype(np.float32, copy=False) * float(scale)
    seq_q = q.shape[1]
    seq_k = k.shape[1]
    causal_mask = np.triu(np.ones((seq_q, seq_k), dtype=bool), k=1)
    scores = np.where(causal_mask[None, None, :, :], -1.0e9, scores)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return np.einsum("bhts,bshd->bthd", probs, v).astype(np.float32, copy=False)


def _contiguous_cache_update_numpy(cache, k: np.ndarray, v: np.ndarray, *, start_position: int):
    k_cache, v_cache = cache
    updated_k = np.array(k_cache, copy=True)
    updated_v = np.array(v_cache, copy=True)
    seq_len = k.shape[1]
    updated_k[:, start_position:start_position + seq_len, :, :] = k
    updated_v[:, start_position:start_position + seq_len, :, :] = v
    return updated_k, updated_v


def _contiguous_cache_update_mlx(cache, k, v, *, start_position: int):
    k_cache, v_cache = cache
    seq_len = int(k.shape[1])
    end_position = start_position + seq_len
    if start_position == 0:
        updated_k = mx.concatenate([k, k_cache[:, end_position:, :, :]], axis=1)
        updated_v = mx.concatenate([v, v_cache[:, end_position:, :, :]], axis=1)
        return updated_k, updated_v
    raise NotImplementedError("Continuation prefill with start_position > 0 is not implemented yet for contiguous MLX cache updates.")


def _zero_like_hidden(x):
    if _is_mlx_array(x):
        return mx.zeros_like(x)
    return np.zeros_like(_to_numpy(x), dtype=np.float32)


def _optional_stack_module():
    try:
        import ops.llama_stack_ops as stack_module

        return stack_module
    except Exception:  # noqa: BLE001
        pass
    stack_path = Path(__file__).resolve().parent / "llama_stack_ops.py"
    spec = importlib.util.spec_from_file_location("llama_stack_ops_prefill_fallback", stack_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load llama_stack_ops for prefill scaffold.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_prefill_input(x, config: LlamaLikeConfig):
    shape = _shape_tuple(x)
    if shape is None or len(shape) != 3 or shape[2] != config.hidden_size:
        raise ValueError(f"x must have shape [B,S,{config.hidden_size}], got {shape}")
    return shape


def _validate_contiguous_cache(cache, config: LlamaLikeConfig):
    if not isinstance(cache, tuple) or len(cache) != 2:
        raise ValueError("contiguous cache must be a tuple (K_cache, V_cache)")
    k_cache, v_cache = cache
    if _shape_tuple(k_cache) != _shape_tuple(v_cache):
        raise ValueError(f"K_cache and V_cache must match, got {_shape_tuple(k_cache)}, {_shape_tuple(v_cache)}")
    shape = _shape_tuple(k_cache)
    if shape is None or len(shape) != 4:
        raise ValueError(f"cache must have shape [B,MAX_S,Hkv,D], got {shape}")
    if shape[2] != config.num_key_value_heads or shape[3] != config.head_dim:
        raise ValueError(f"cache must have Hkv/head_dim {(config.num_key_value_heads, config.head_dim)}, got {shape}")
    return k_cache, v_cache


def _dequant_layer_weights_numpy(weights) -> dict[str, np.ndarray]:
    return {
        "input_ln": _to_numpy(weights.input_layernorm_weight).astype(np.float32, copy=False),
        "post_ln": _to_numpy(weights.post_attention_layernorm_weight).astype(np.float32, copy=False),
        "qkv": _to_numpy(dequantize_quantized_weight(weights.qkv_w, weights.qkv_scales, weights.qkv_zeros, bits=weights.bits, group_size=weights.group_size)).astype(np.float32, copy=False),
        "o": _to_numpy(dequantize_quantized_weight(weights.o_w, weights.o_scales, weights.o_zeros, bits=weights.bits, group_size=weights.group_size)).astype(np.float32, copy=False),
        "gate": _to_numpy(dequantize_quantized_weight(weights.gate_w, weights.gate_scales, weights.gate_zeros, bits=weights.bits, group_size=weights.group_size)).astype(np.float32, copy=False),
        "up": _to_numpy(dequantize_quantized_weight(weights.up_w, weights.up_scales, weights.up_zeros, bits=weights.bits, group_size=weights.group_size)).astype(np.float32, copy=False),
        "down": _to_numpy(dequantize_quantized_weight(weights.down_w, weights.down_scales, weights.down_zeros, bits=weights.bits, group_size=weights.group_size)).astype(np.float32, copy=False),
    }


def _numpy_qkv_split_rope(qkv: np.ndarray, config: LlamaLikeConfig, cos, sin, *, position_offset: int):
    q_rows = config.q_output_dim()
    kv_rows = config.kv_output_dim()
    q = qkv[..., :q_rows].reshape(qkv.shape[0], qkv.shape[1], config.num_attention_heads, config.head_dim)
    k = qkv[..., q_rows:q_rows + kv_rows].reshape(qkv.shape[0], qkv.shape[1], config.num_key_value_heads, config.head_dim)
    v = qkv[..., q_rows + kv_rows:].reshape(qkv.shape[0], qkv.shape[1], config.num_key_value_heads, config.head_dim)
    q_rope = _apply_rope_numpy(q.astype(np.float32, copy=False), _to_numpy(cos), _to_numpy(sin), position_offset=position_offset)
    k_rope = _apply_rope_numpy(k.astype(np.float32, copy=False), _to_numpy(cos), _to_numpy(sin), position_offset=position_offset)
    return q_rope, k_rope, v.astype(np.float32, copy=False)


@dataclass
class LlamaPrefillBackendConfig:
    norm_backend: str = "metal"
    qkv_matvec_backend: str = "metal_tiled"
    attention_backend: str = "metal_gqa_threadgroup"
    out_matvec_backend: str = "metal_tiled"
    mlp_backend_preset: str = "fused_experimental"
    cache_backend: str = "metal"
    cache_layout: str = "contiguous"
    use_autotune: bool = False

    def validate(self) -> "LlamaPrefillBackendConfig":
        if self.cache_layout not in ("contiguous", "paged"):
            raise ValueError("cache_layout must be one of ('contiguous', 'paged')")
        if self.cache_layout == "paged":
            raise NotImplementedError("Paged prefill is not wired through this scaffold yet.")
        return self


def reference_prefill_backend_config() -> LlamaPrefillBackendConfig:
    return LlamaPrefillBackendConfig(
        norm_backend="reference",
        qkv_matvec_backend="reference",
        attention_backend="reference",
        out_matvec_backend="reference",
        mlp_backend_preset="reference",
        cache_backend="reference",
        cache_layout="contiguous",
        use_autotune=False,
    )


def metal_prefill_backend_config() -> LlamaPrefillBackendConfig:
    return LlamaPrefillBackendConfig(
        norm_backend="metal",
        qkv_matvec_backend="metal",
        attention_backend="metal_gqa",
        out_matvec_backend="metal",
        mlp_backend_preset="metal",
        cache_backend="metal",
        cache_layout="contiguous",
        use_autotune=False,
    )


def tiled_prefill_backend_config() -> LlamaPrefillBackendConfig:
    return LlamaPrefillBackendConfig(
        norm_backend="metal",
        qkv_matvec_backend="metal_tiled",
        attention_backend="metal_gqa_threadgroup",
        out_matvec_backend="metal_tiled",
        mlp_backend_preset="tiled",
        cache_backend="metal",
        cache_layout="contiguous",
        use_autotune=False,
    )


def fused_experimental_prefill_backend_config() -> LlamaPrefillBackendConfig:
    return LlamaPrefillBackendConfig(
        norm_backend="metal",
        qkv_matvec_backend="metal_tiled",
        attention_backend="metal_gqa_threadgroup",
        out_matvec_backend="metal_tiled",
        mlp_backend_preset="fused_experimental",
        cache_backend="metal",
        cache_layout="contiguous",
        use_autotune=False,
    )


def _prefill_backend_from_preset(name: str) -> LlamaPrefillBackendConfig:
    mapping = {
        "reference": reference_prefill_backend_config,
        "metal": metal_prefill_backend_config,
        "tiled": tiled_prefill_backend_config,
        "fused_experimental": fused_experimental_prefill_backend_config,
    }
    if name not in mapping:
        raise ValueError(f"backend_preset must be one of {tuple(mapping)}, got {name}")
    return mapping[name]()


def _reference_llama_layer_prefill_numpy(
    x,
    weights,
    cache,
    cos,
    sin,
    config: LlamaLikeConfig,
    *,
    start_position: int = 0,
    return_intermediates: bool = False,
):
    if start_position > 0:
        raise NotImplementedError("Continuation prefill with start_position > 0 is not implemented yet.")
    x_np = _to_numpy(x).astype(np.float32, copy=False)
    _validate_prefill_input(x_np, config)
    _validate_contiguous_cache(cache, config)
    weights_np = _dequant_layer_weights_numpy(weights)
    x_norm = _rms_norm_numpy(x_np, weights_np["input_ln"], config.rms_norm_eps)
    qkv = np.einsum("bsh,oh->bso", x_norm, weights_np["qkv"]).astype(np.float32, copy=False)
    q, k, v = _numpy_qkv_split_rope(qkv, config, cos, sin, position_offset=start_position)
    updated_cache = _contiguous_cache_update_numpy(cache, k, v, start_position=start_position)
    seen_k = updated_cache[0][:, :x_np.shape[1], :, :]
    seen_v = updated_cache[1][:, :x_np.shape[1], :, :]
    seen_k = _repeat_kv_heads_numpy(seen_k, config.num_attention_heads)
    seen_v = _repeat_kv_heads_numpy(seen_v, config.num_attention_heads)
    attn = _causal_attention_numpy(q, seen_k, seen_v, scale=1.0 / math.sqrt(float(config.head_dim)))
    attn_flat = attn.reshape(x_np.shape[0], x_np.shape[1], config.hidden_size)
    attn_proj = np.einsum("bsh,oh->bso", attn_flat, weights_np["o"]).astype(np.float32, copy=False)
    h1 = x_np + attn_proj
    mlp_norm = _rms_norm_numpy(h1, weights_np["post_ln"], config.rms_norm_eps)
    gate = np.einsum("bsh,oh->bso", mlp_norm, weights_np["gate"]).astype(np.float32, copy=False)
    up = np.einsum("bsh,oh->bso", mlp_norm, weights_np["up"]).astype(np.float32, copy=False)
    mlp = _silu_numpy(gate) * up
    down = np.einsum("bsi,oi->bso", mlp, weights_np["down"]).astype(np.float32, copy=False)
    out = h1 + down
    if not return_intermediates:
        return out, updated_cache
    return out, updated_cache, {
        "x_norm": x_norm,
        "qkv": qkv,
        "q": q,
        "k": k,
        "v": v,
        "attn": attn,
        "attn_proj": attn_proj,
        "h1": h1,
        "mlp_norm": mlp_norm,
        "gate": gate,
        "up": up,
        "mlp": mlp,
        "down": down,
    }


def reference_llama_layer_prefill(
    x,
    weights,
    cache,
    cos,
    sin,
    config,
    *,
    start_position: int = 0,
    return_intermediates: bool = False,
):
    config = config.validate()
    if start_position > 0:
        raise NotImplementedError("Continuation prefill with start_position > 0 is not implemented yet.")
    if mx is None or not _is_mlx_array(x):
        return _reference_llama_layer_prefill_numpy(
            x,
            weights,
            cache,
            cos,
            sin,
            config,
            start_position=start_position,
            return_intermediates=return_intermediates,
        )
    weights.validate(config)
    _validate_prefill_input(x, config)
    k_cache, v_cache = _validate_contiguous_cache(cache, config)
    x_norm = reference_rms_norm(x, weights.input_layernorm_weight, eps=config.rms_norm_eps)
    qkv = quantized_linear(
        x_norm,
        weights.qkv_w,
        weights.qkv_scales,
        weights.qkv_zeros,
        bits=weights.bits,
        group_size=weights.group_size,
        backend="reference",
    )
    q, k, v = reference_gqa_qkv_split_rope(
        qkv,
        cos,
        sin,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        position_offset=start_position,
    )
    updated_cache = _contiguous_cache_update_mlx((k_cache, v_cache), k, v, start_position=start_position)
    attn = reference_gqa_attention(q, k, v, causal=True)
    attn_flat = attn.reshape(attn.shape[0], attn.shape[1], config.hidden_size)
    attn_proj = quantized_linear(
        attn_flat,
        weights.o_w,
        weights.o_scales,
        weights.o_zeros,
        bits=weights.bits,
        group_size=weights.group_size,
        backend="reference",
    )
    h1 = x + attn_proj
    out, mlp_intermediates = reference_quantized_mlp_block(
        h1,
        _zero_like_hidden(h1),
        weights.post_attention_layernorm_weight,
        weights.gate_w,
        weights.gate_scales,
        weights.up_w,
        weights.up_scales,
        weights.down_w,
        weights.down_scales,
        gate_zeros=weights.gate_zeros,
        up_zeros=weights.up_zeros,
        down_zeros=weights.down_zeros,
        bits=weights.bits,
        group_size=weights.group_size,
        eps=config.rms_norm_eps,
        return_intermediates=True,
    )
    if not return_intermediates:
        return out, updated_cache
    return out, updated_cache, {
        "x_norm": x_norm,
        "qkv": qkv,
        "q": q,
        "k": k,
        "v": v,
        "attn": attn,
        "attn_proj": attn_proj,
        "h1": h1,
        "mlp_norm": mlp_intermediates["normed"],
        "gate": mlp_intermediates["gate"],
        "up": mlp_intermediates["up"],
        "mlp": mlp_intermediates["mlp"],
        "down": mlp_intermediates["down"],
    }


def llama_layer_prefill(
    x,
    weights,
    cache,
    cos,
    sin,
    config,
    *,
    backend_config: LlamaPrefillBackendConfig | None = None,
    start_position: int = 0,
    return_intermediates: bool = False,
):
    config = config.validate()
    backend_config = (backend_config or fused_experimental_prefill_backend_config()).validate()
    if start_position > 0:
        raise NotImplementedError("Continuation prefill with start_position > 0 is not implemented yet.")
    if mx is None or not _is_mlx_array(x):
        return _reference_llama_layer_prefill_numpy(
            x,
            weights,
            cache,
            cos,
            sin,
            config,
            start_position=start_position,
            return_intermediates=return_intermediates,
        )
    weights.validate(config)
    _validate_prefill_input(x, config)
    k_cache, v_cache = _validate_contiguous_cache(cache, config)
    x_norm = reference_rms_norm(x, weights.input_layernorm_weight, eps=config.rms_norm_eps) if backend_config.norm_backend == "reference" else rms_norm(
        x, weights.input_layernorm_weight, eps=config.rms_norm_eps, backend=backend_config.norm_backend
    )
    qkv = quantized_linear(
        x_norm,
        weights.qkv_w,
        weights.qkv_scales,
        weights.qkv_zeros,
        bits=weights.bits,
        group_size=weights.group_size,
        backend=backend_config.qkv_matvec_backend,
    )
    q, k, v = reference_gqa_qkv_split_rope(
        qkv,
        cos,
        sin,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        position_offset=start_position,
    )
    updated_cache = _contiguous_cache_update_mlx((k_cache, v_cache), k, v, start_position=start_position)
    attn = gqa_attention(q, k, v, causal=True, backend=backend_config.attention_backend)
    attn_flat = attn.reshape(attn.shape[0], attn.shape[1], config.hidden_size)
    attn_proj = quantized_linear(
        attn_flat,
        weights.o_w,
        weights.o_scales,
        weights.o_zeros,
        bits=weights.bits,
        group_size=weights.group_size,
        backend=backend_config.out_matvec_backend,
    )
    h1 = x + attn_proj
    out, mlp_intermediates = quantized_mlp_block(
        h1,
        _zero_like_hidden(h1),
        weights.post_attention_layernorm_weight,
        weights.gate_w,
        weights.gate_scales,
        weights.up_w,
        weights.up_scales,
        weights.down_w,
        weights.down_scales,
        gate_zeros=weights.gate_zeros,
        up_zeros=weights.up_zeros,
        down_zeros=weights.down_zeros,
        bits=weights.bits,
        group_size=weights.group_size,
        eps=config.rms_norm_eps,
        backend_preset=backend_config.mlp_backend_preset,
        return_intermediates=True,
    )
    if not return_intermediates:
        return out, updated_cache
    return out, updated_cache, {
        "x_norm": x_norm,
        "qkv": qkv,
        "q": q,
        "k": k,
        "v": v,
        "attn": attn,
        "attn_proj": attn_proj,
        "h1": h1,
        "mlp_norm": mlp_intermediates["normed"],
        "gate": mlp_intermediates["gate"],
        "up": mlp_intermediates["up"],
        "mlp": mlp_intermediates["mlp"],
        "down": mlp_intermediates["down"],
    }


def reference_llama_stack_prefill(
    x,
    stack_weights,
    stack_cache,
    cos,
    sin,
    config,
    *,
    start_position: int = 0,
    return_intermediates: bool = False,
):
    stack_module = _optional_stack_module()
    LlamaStackCache = stack_module.LlamaStackCache
    logits_from_hidden = stack_module.logits_from_hidden

    config = config.validate()
    stack_weights = stack_weights.validate(config)
    if stack_cache.cache_layout != "contiguous":
        raise NotImplementedError("Only contiguous cache layout is currently supported in stack prefill.")
    hidden = x
    updated_layer_caches = []
    intermediates = {"layers": []}
    for layer_idx, layer_weights in enumerate(stack_weights.layers):
        result = reference_llama_layer_prefill(
            hidden,
            layer_weights,
            stack_cache.layer_caches[layer_idx],
            cos,
            sin,
            config,
            start_position=start_position,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            hidden, updated_cache, layer_intermediates = result
            intermediates["layers"].append({"layer_idx": layer_idx, **layer_intermediates})
        else:
            hidden, updated_cache = result
        updated_layer_caches.append(updated_cache)
    if mx is not None and _is_mlx_array(hidden):
        final_hidden = reference_rms_norm(hidden, stack_weights.final_norm_weight, eps=config.rms_norm_eps)
    else:
        final_hidden = _rms_norm_numpy(_to_numpy(hidden).astype(np.float32, copy=False), _to_numpy(stack_weights.final_norm_weight).astype(np.float32, copy=False), config.rms_norm_eps)
        final_hidden = _cast_like(final_hidden.astype(np.float32, copy=False), hidden)
    updated_stack_cache = LlamaStackCache(updated_layer_caches, stack_cache.cache_layout, stack_cache.max_seq_len, stack_cache.page_size)
    if stack_weights.lm_head is None:
        if return_intermediates:
            intermediates["final_hidden"] = final_hidden
            return final_hidden, updated_stack_cache, intermediates
        return final_hidden, updated_stack_cache
    logits = logits_from_hidden(final_hidden, stack_weights.lm_head)
    if return_intermediates:
        intermediates["final_hidden"] = final_hidden
        intermediates["logits"] = logits
        return logits, final_hidden, updated_stack_cache, intermediates
    return logits, final_hidden, updated_stack_cache


def llama_stack_prefill(
    x,
    stack_weights,
    stack_cache,
    cos,
    sin,
    config,
    *,
    backend_config: LlamaPrefillBackendConfig | None = None,
    start_position: int = 0,
    return_logits: bool = True,
    return_intermediates: bool = False,
):
    stack_module = _optional_stack_module()
    LlamaStackCache = stack_module.LlamaStackCache
    logits_from_hidden = stack_module.logits_from_hidden

    config = config.validate()
    stack_weights = stack_weights.validate(config)
    backend_config = (backend_config or fused_experimental_prefill_backend_config()).validate()
    if stack_cache.cache_layout != "contiguous":
        raise NotImplementedError("Only contiguous cache layout is currently supported in stack prefill.")
    hidden = x
    updated_layer_caches = []
    intermediates = {"layers": []}
    for layer_idx, layer_weights in enumerate(stack_weights.layers):
        result = llama_layer_prefill(
            hidden,
            layer_weights,
            stack_cache.layer_caches[layer_idx],
            cos,
            sin,
            config,
            backend_config=backend_config,
            start_position=start_position,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            hidden, updated_cache, layer_intermediates = result
            intermediates["layers"].append({"layer_idx": layer_idx, **layer_intermediates})
        else:
            hidden, updated_cache = result
        updated_layer_caches.append(updated_cache)
    if mx is not None and _is_mlx_array(hidden):
        final_hidden = reference_rms_norm(hidden, stack_weights.final_norm_weight, eps=config.rms_norm_eps) if backend_config.norm_backend == "reference" else rms_norm(
            hidden, stack_weights.final_norm_weight, eps=config.rms_norm_eps, backend=backend_config.norm_backend
        )
    else:
        final_hidden = _rms_norm_numpy(_to_numpy(hidden).astype(np.float32, copy=False), _to_numpy(stack_weights.final_norm_weight).astype(np.float32, copy=False), config.rms_norm_eps)
        final_hidden = _cast_like(final_hidden.astype(np.float32, copy=False), hidden)
    updated_stack_cache = LlamaStackCache(updated_layer_caches, stack_cache.cache_layout, stack_cache.max_seq_len, stack_cache.page_size)
    if stack_weights.lm_head is None or not return_logits:
        if return_intermediates:
            intermediates["final_hidden"] = final_hidden
            return final_hidden, updated_stack_cache, intermediates
        return final_hidden, updated_stack_cache
    logits = logits_from_hidden(final_hidden, stack_weights.lm_head)
    if return_intermediates:
        intermediates["final_hidden"] = final_hidden
        intermediates["logits"] = logits
        return logits, final_hidden, updated_stack_cache, intermediates
    return logits, final_hidden, updated_stack_cache


def prefill_token_ids(
    token_ids,
    stack_weights,
    stack_cache,
    cos,
    sin,
    config,
    *,
    embedding,
    backend_config: LlamaPrefillBackendConfig | None = None,
    start_position: int = 0,
    return_logits: bool = True,
):
    embed_token_ids = _optional_stack_module().embed_token_ids

    token_ids_np = np.asarray(token_ids, dtype=np.int64)
    if token_ids_np.ndim == 1:
        token_ids_np = token_ids_np.reshape(1, -1)
    if token_ids_np.ndim != 2:
        raise ValueError(f"token_ids must have shape [B,S] or [S], got {token_ids_np.shape}")
    embedded = embed_token_ids(token_ids_np, embedding)
    return llama_stack_prefill(
        embedded,
        stack_weights,
        stack_cache,
        cos,
        sin,
        config,
        backend_config=backend_config,
        start_position=start_position,
        return_logits=return_logits,
        return_intermediates=False,
    )
