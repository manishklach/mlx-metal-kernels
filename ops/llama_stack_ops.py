from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - optional path
    mx = None

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for the multi-layer decode scaffold") from exc

from models.llama_config import LlamaLikeConfig
from models.quantize_weights import QuantizationConfig, dequantize_quantized_weight, quantize_weight_groupwise
from models.quantized_layer_package import QuantizedLinearPackage, QuantizedLlamaLayerPackage


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


def _optional_single_layer_ops():
    try:
        from ops.llama_layer_ops import (
            LlamaLayerKernelWeights,
            create_random_quantized_llama_layer_weights,
            fused_experimental_backend_config,
            init_llama_layer_cache,
            llama_layer_decode_step,
            metal_backend_config,
            reference_backend_config,
            reference_llama_layer_decode_step,
            tiled_backend_config,
        )
        from ops.norm_ops import reference_rms_norm, rms_norm

        return {
            "LlamaLayerKernelWeights": LlamaLayerKernelWeights,
            "create_random_quantized_llama_layer_weights": create_random_quantized_llama_layer_weights,
            "fused_experimental_backend_config": fused_experimental_backend_config,
            "init_llama_layer_cache": init_llama_layer_cache,
            "llama_layer_decode_step": llama_layer_decode_step,
            "metal_backend_config": metal_backend_config,
            "reference_backend_config": reference_backend_config,
            "reference_llama_layer_decode_step": reference_llama_layer_decode_step,
            "reference_rms_norm": reference_rms_norm,
            "rms_norm": rms_norm,
            "tiled_backend_config": tiled_backend_config,
        }
    except Exception:  # noqa: BLE001
        return None


def _backend_config_from_preset(ops, preset: str):
    if ops is None:
        return None
    mapping = {
        "reference": ops["reference_backend_config"],
        "metal": ops["metal_backend_config"],
        "tiled": ops["tiled_backend_config"],
        "fused_experimental": ops["fused_experimental_backend_config"],
    }
    if preset not in mapping:
        raise ValueError(f"layer_backend_preset must be one of {tuple(mapping)}, got {preset}")
    return mapping[preset]()


def _build_rope_tables_numpy(config: LlamaLikeConfig, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    if config.head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {config.head_dim}")
    positions = np.arange(seq_len, dtype=np.float32)
    inv_freq = 1.0 / (float(config.rope_theta) ** (np.arange(0, config.head_dim, 2, dtype=np.float32) / float(config.head_dim)))
    freqs = positions[:, None] * inv_freq[None, :]
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def _rms_norm_numpy(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def _silu_numpy(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


@dataclass
class LlamaStackBackendConfig:
    layer_backend_preset: str = "fused_experimental"
    norm_backend: str = "metal"
    cache_layout: str = "contiguous"
    use_autotune: bool = False

    def validate(self) -> "LlamaStackBackendConfig":
        if self.layer_backend_preset not in ("reference", "metal", "tiled", "fused_experimental"):
            raise ValueError(
                "layer_backend_preset must be one of ('reference', 'metal', 'tiled', 'fused_experimental')"
            )
        if self.cache_layout not in ("contiguous", "paged"):
            raise ValueError("cache_layout must be one of ('contiguous', 'paged')")
        if self.cache_layout == "paged":
            raise NotImplementedError("Paged multi-layer stack cache is not yet wired through this scaffold.")
        return self


@dataclass
class LlamaStackWeights:
    layers: list[Any]
    final_norm_weight: Any
    lm_head: Any | None = None
    embedding: Any | None = None

    def num_layers(self) -> int:
        return len(self.layers)

    def validate(self, config) -> "LlamaStackWeights":
        config = config.validate()
        if len(self.layers) != config.num_hidden_layers:
            raise ValueError(f"layers must have length {config.num_hidden_layers}, got {len(self.layers)}")
        if _shape_tuple(self.final_norm_weight) != (config.hidden_size,):
            raise ValueError(f"final_norm_weight must have shape {(config.hidden_size,)}, got {_shape_tuple(self.final_norm_weight)}")
        if self.embedding is not None and _shape_tuple(self.embedding)[1] != config.hidden_size:
            raise ValueError(f"embedding must have shape [vocab_size,{config.hidden_size}], got {_shape_tuple(self.embedding)}")
        if self.lm_head is not None and _shape_tuple(self.lm_head)[1] != config.hidden_size:
            raise ValueError(
                f"lm_head must use the documented [vocab_size, hidden_size] convention with hidden size {config.hidden_size}, got {_shape_tuple(self.lm_head)}"
            )
        for layer in self.layers:
            validate = getattr(layer, "validate", None)
            if callable(validate):
                validate(config)
        return self

    def shapes(self) -> dict[str, Any]:
        return {
            "layers": [getattr(layer, "shapes", lambda: _shape_tuple(layer))() for layer in self.layers],
            "final_norm_weight": _shape_tuple(self.final_norm_weight),
            "lm_head": _shape_tuple(self.lm_head),
            "embedding": _shape_tuple(self.embedding),
        }


@dataclass
class LlamaStackCache:
    layer_caches: list[Any]
    cache_layout: str
    max_seq_len: int
    page_size: int | None = None

    def num_layers(self) -> int:
        return len(self.layer_caches)

    def shapes(self) -> dict[str, Any]:
        def _cache_shape(cache):
            if isinstance(cache, tuple):
                return tuple(_shape_tuple(item) for item in cache)
            return _shape_tuple(cache)

        return {
            "cache_layout": self.cache_layout,
            "max_seq_len": self.max_seq_len,
            "page_size": self.page_size,
            "layer_caches": [_cache_shape(cache) for cache in self.layer_caches],
        }


def init_llama_stack_cache(
    config,
    B,
    max_seq_len,
    *,
    cache_layout="contiguous",
    dtype=None,
    page_size=16,
):
    config = config.validate()
    if cache_layout == "paged":
        raise NotImplementedError("Paged multi-layer stack cache is not yet wired through this scaffold.")
    ops = _optional_single_layer_ops()
    if ops is not None and mx is not None and dtype is not None:
        layer_caches = [
            ops["init_llama_layer_cache"](config, B, max_seq_len, cache_layout=cache_layout, dtype=dtype, page_size=page_size)
            for _ in range(config.num_hidden_layers)
        ]
    else:
        layer_caches = [
            (
                np.zeros((B, max_seq_len, config.num_key_value_heads, config.head_dim), dtype=np.float32),
                np.zeros((B, max_seq_len, config.num_key_value_heads, config.head_dim), dtype=np.float32),
            )
            for _ in range(config.num_hidden_layers)
        ]
    return LlamaStackCache(layer_caches=layer_caches, cache_layout=cache_layout, max_seq_len=max_seq_len, page_size=page_size if cache_layout == "paged" else None)


def embed_token_ids(token_ids, embedding):
    token_ids_np = np.asarray(token_ids, dtype=np.int64)
    if token_ids_np.ndim == 1:
        token_ids_np = token_ids_np.reshape(-1, 1)
    if token_ids_np.ndim != 2:
        raise ValueError(f"token_ids must have shape [B] or [B,1], got {token_ids_np.shape}")
    gathered = _to_numpy(embedding)[token_ids_np.reshape(-1)]
    out = gathered.reshape(token_ids_np.shape[0], token_ids_np.shape[1], -1)
    return _cast_like(out.astype(np.float32, copy=False), embedding)


def logits_from_hidden(hidden, lm_head):
    hidden_np = _to_numpy(hidden).astype(np.float32, copy=False)
    if hidden_np.ndim != 3:
        raise ValueError(f"hidden must have shape [B,1,H] or [B,T,H], got {hidden_np.shape}")
    logits = hidden_np @ _to_numpy(lm_head).astype(np.float32, copy=False).T
    return _cast_like(logits.astype(np.float32, copy=False), lm_head)


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


def _fallback_layer_decode(x, weights, cache, position, config):
    x_np = _to_numpy(x).astype(np.float32, copy=False)
    if x_np.shape[0] != 1:
        raise NotImplementedError("The numpy fallback multi-layer stack currently supports only B=1")
    weights_np = _dequant_layer_weights_numpy(weights)
    x_row = x_np[0, 0]
    x_norm = _rms_norm_numpy(x_row, weights_np["input_ln"], config.rms_norm_eps)
    qkv = x_norm @ weights_np["qkv"].T
    q_rows = config.q_output_dim()
    kv_rows = config.kv_output_dim()
    q = qkv[:q_rows].reshape(config.num_attention_heads, config.head_dim)
    k = qkv[q_rows:q_rows + kv_rows].reshape(config.num_key_value_heads, config.head_dim)
    v = qkv[q_rows + kv_rows:].reshape(config.num_key_value_heads, config.head_dim)
    k_cache, v_cache = cache
    if position >= k_cache.shape[1]:
        raise ValueError("Layer cache is full; increase max_seq_len")
    k_cache = np.array(k_cache, copy=True)
    v_cache = np.array(v_cache, copy=True)
    k_cache[0, position] = k
    v_cache[0, position] = v
    seen_k = k_cache[0, :position + 1]
    seen_v = v_cache[0, :position + 1]
    repeat_factor = config.num_attention_heads // config.num_key_value_heads
    seen_k = np.repeat(seen_k, repeat_factor, axis=1)
    seen_v = np.repeat(seen_v, repeat_factor, axis=1)
    scores = np.einsum("hd,shd->hs", q, seen_k) / math.sqrt(float(config.head_dim))
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    attn = np.einsum("hs,shd->hd", probs, seen_v).reshape(config.hidden_size)
    attn_proj = attn @ weights_np["o"].T
    h = x_row + attn_proj
    mlp_norm = _rms_norm_numpy(h, weights_np["post_ln"], config.rms_norm_eps)
    gate = mlp_norm @ weights_np["gate"].T
    up = mlp_norm @ weights_np["up"].T
    mlp = _silu_numpy(gate) * up
    down = mlp @ weights_np["down"].T
    out = h + down
    return out.reshape(1, 1, config.hidden_size).astype(np.float32, copy=False), (k_cache, v_cache)


def reference_llama_stack_decode_step(
    x,
    stack_weights,
    stack_cache,
    cos,
    sin,
    position,
    config,
    *,
    return_intermediates=False,
):
    config = config.validate()
    stack_weights = stack_weights.validate(config)
    if stack_cache.cache_layout != "contiguous":
        raise NotImplementedError("Only contiguous cache layout is currently supported in the multi-layer stack.")
    ops = _optional_single_layer_ops()
    hidden = x
    updated_layer_caches = []
    intermediates = {"layers": []}
    for layer_idx, layer_weights in enumerate(stack_weights.layers):
        if ops is not None and mx is not None and _is_mlx_array(hidden):
            hidden, updated_cache = ops["reference_llama_layer_decode_step"](
                hidden,
                layer_weights,
                stack_cache.layer_caches[layer_idx],
                cos,
                sin,
                position,
                config,
                cache_layout=stack_cache.cache_layout,
                return_intermediates=False,
            )
        else:
            hidden, updated_cache = _fallback_layer_decode(hidden, layer_weights, stack_cache.layer_caches[layer_idx], position, config)
        updated_layer_caches.append(updated_cache)
        if return_intermediates:
            intermediates["layers"].append({"layer_idx": layer_idx, "hidden": hidden})
    if ops is not None and mx is not None and _is_mlx_array(hidden):
        final_hidden = ops["reference_rms_norm"](hidden, stack_weights.final_norm_weight, eps=config.rms_norm_eps)
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


def llama_stack_decode_step(
    x,
    stack_weights,
    stack_cache,
    cos,
    sin,
    position,
    config,
    *,
    backend_config=None,
    return_intermediates=False,
):
    config = config.validate()
    stack_weights = stack_weights.validate(config)
    backend_config = (backend_config or LlamaStackBackendConfig()).validate()
    ops = _optional_single_layer_ops()
    hidden = x
    updated_layer_caches = []
    intermediates = {"layers": []}
    for layer_idx, layer_weights in enumerate(stack_weights.layers):
        if ops is not None and mx is not None and _is_mlx_array(hidden):
            layer_backend = _backend_config_from_preset(ops, backend_config.layer_backend_preset)
            hidden, updated_cache = ops["llama_layer_decode_step"](
                hidden,
                layer_weights,
                stack_cache.layer_caches[layer_idx],
                cos,
                sin,
                position,
                config,
                backend_config=layer_backend,
                cache_layout=backend_config.cache_layout,
                return_intermediates=False,
            )
        else:
            hidden, updated_cache = _fallback_layer_decode(hidden, layer_weights, stack_cache.layer_caches[layer_idx], position, config)
        updated_layer_caches.append(updated_cache)
        if return_intermediates:
            intermediates["layers"].append({"layer_idx": layer_idx, "hidden": hidden})
    if ops is not None and mx is not None and _is_mlx_array(hidden):
        if backend_config.norm_backend == "reference":
            final_hidden = ops["reference_rms_norm"](hidden, stack_weights.final_norm_weight, eps=config.rms_norm_eps)
        else:
            final_hidden = ops["rms_norm"](hidden, stack_weights.final_norm_weight, eps=config.rms_norm_eps, backend=backend_config.norm_backend)
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


def reference_llama_stack_decode_loop(
    inputs,
    stack_weights,
    stack_cache,
    cos,
    sin,
    config,
    *,
    cache_layout="contiguous",
    T=None,
    return_logits=True,
):
    return llama_stack_decode_loop(
        inputs,
        stack_weights,
        stack_cache,
        cos,
        sin,
        config,
        backend_preset="reference",
        cache_layout=cache_layout,
        T=T,
        return_logits=return_logits,
    )


def llama_stack_decode_loop(
    inputs,
    stack_weights,
    stack_cache,
    cos,
    sin,
    config,
    *,
    backend_preset="fused_experimental",
    cache_layout="contiguous",
    T=None,
    return_logits=True,
):
    inputs_shape = _shape_tuple(inputs)
    if inputs_shape is None:
        raise ValueError("inputs must expose shape")
    inputs_np = _to_numpy(inputs) if len(inputs_shape) == 4 else None
    if len(inputs_shape) == 4:
        if inputs_shape[2] != 1:
            raise ValueError(f"inputs with rank 4 must have shape [B,T,1,H], got {inputs_shape}")
        if _is_mlx_array(inputs):
            inputs = inputs[:, :, 0, :]
        else:
            inputs = inputs_np[:, :, 0, :]
        inputs_shape = _shape_tuple(inputs)
    if len(inputs_shape) != 3:
        raise ValueError(f"inputs must have shape [B,T,H] or [B,T,1,H], got {inputs_shape}")
    T = inputs_shape[1] if T is None else T
    backend = LlamaStackBackendConfig(layer_backend_preset=backend_preset, cache_layout=cache_layout).validate()
    outputs = []
    running_cache = stack_cache
    for t in range(T):
        x_t = inputs[:, t:t + 1, :]
        result = llama_stack_decode_step(
            x_t,
            stack_weights,
            running_cache,
            cos,
            sin,
            t,
            config,
            backend_config=backend,
            return_intermediates=False,
        )
        if stack_weights.lm_head is None:
            out_t, running_cache = result
        else:
            logits_t, final_hidden_t, running_cache = result
            out_t = logits_t if return_logits else final_hidden_t
        outputs.append(out_t)
    if len(outputs) == 1:
        out = outputs[0]
    elif _is_mlx_array(outputs[0]):
        out = mx.concatenate(outputs, axis=1)
    else:
        out = np.concatenate([_to_numpy(item) for item in outputs], axis=1)
    return out, running_cache


def create_random_quantized_llama_stack_weights(
    config,
    *,
    vocab_size=None,
    bits=4,
    group_size=32,
    dtype=None,
    seed=0,
    include_embedding=True,
    include_lm_head=True,
):
    config = config.validate()
    vocab_size = 128 if vocab_size is None and config.vocab_size is None else (config.vocab_size if vocab_size is None else vocab_size)
    ops = _optional_single_layer_ops()
    if ops is not None and mx is not None and dtype is not None:
        layers = [
            ops["create_random_quantized_llama_layer_weights"](config, bits=bits, group_size=group_size, dtype=dtype, seed=seed + layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        mx.random.seed(seed + config.num_hidden_layers + 1)
        final_norm_weight = mx.ones((config.hidden_size,), dtype=dtype)
        embedding = mx.random.normal((vocab_size, config.hidden_size)).astype(dtype) if include_embedding else None
        lm_head = mx.random.normal((vocab_size, config.hidden_size)).astype(dtype) if include_lm_head else None
    else:
        rng = np.random.default_rng(seed)
        quant_config = QuantizationConfig(bits=bits, group_size=group_size)

        def _linear_package(name: str, shape: tuple[int, int]) -> QuantizedLinearPackage:
            weight = rng.normal(size=shape).astype(np.float32)
            quantized = quantize_weight_groupwise(weight, quant_config)
            return QuantizedLinearPackage(
                name=name,
                weight=quantized.packed_weight,
                scales=quantized.scales,
                zeros=quantized.zeros,
                bits=bits,
                group_size=group_size,
                original_shape=shape,
            )

        layers = []
        for layer_idx in range(config.num_hidden_layers):
            qkv_rows = config.q_output_dim() + 2 * config.kv_output_dim()
            package = QuantizedLlamaLayerPackage(
                layer_idx=layer_idx,
                input_layernorm_weight=np.ones((config.hidden_size,), dtype=np.float32),
                post_attention_layernorm_weight=np.ones((config.hidden_size,), dtype=np.float32),
                qkv=_linear_package(f"layer_{layer_idx}.qkv_fused", (qkv_rows, config.hidden_size)),
                o_proj=_linear_package(f"layer_{layer_idx}.o_proj", (config.hidden_size, config.hidden_size)),
                gate_proj=_linear_package(f"layer_{layer_idx}.gate_proj", (config.intermediate_size, config.hidden_size)),
                up_proj=_linear_package(f"layer_{layer_idx}.up_proj", (config.intermediate_size, config.hidden_size)),
                down_proj=_linear_package(f"layer_{layer_idx}.down_proj", (config.hidden_size, config.intermediate_size)),
            )
            layers.append(package.to_kernel_weights(config))
        final_norm_weight = np.ones((config.hidden_size,), dtype=np.float32)
        embedding = rng.normal(size=(vocab_size, config.hidden_size)).astype(np.float32) if include_embedding else None
        lm_head = rng.normal(size=(vocab_size, config.hidden_size)).astype(np.float32) if include_lm_head else None
    return LlamaStackWeights(layers=layers, final_norm_weight=final_norm_weight, lm_head=lm_head, embedding=embedding).validate(config)


def reference_llama_stack_prefill(
    x,
    stack_weights,
    stack_cache,
    cos,
    sin,
    config,
    *,
    start_position=0,
    return_intermediates=False,
):
    from .llama_prefill_ops import reference_llama_stack_prefill as _reference_llama_stack_prefill

    return _reference_llama_stack_prefill(
        x,
        stack_weights,
        stack_cache,
        cos,
        sin,
        config,
        start_position=start_position,
        return_intermediates=return_intermediates,
    )


def llama_stack_prefill(
    x,
    stack_weights,
    stack_cache,
    cos,
    sin,
    config,
    *,
    backend_config=None,
    start_position=0,
    return_logits=True,
    return_intermediates=False,
):
    from .llama_prefill_ops import llama_stack_prefill as _llama_stack_prefill

    return _llama_stack_prefill(
        x,
        stack_weights,
        stack_cache,
        cos,
        sin,
        config,
        backend_config=backend_config,
        start_position=start_position,
        return_logits=return_logits,
        return_intermediates=return_intermediates,
    )
