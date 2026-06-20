from __future__ import annotations

import math
from dataclasses import dataclass

from .llama_config import LlamaLikeConfig


@dataclass
class LinearWeightSpec:
    name: str
    out_dim: int
    in_dim: int
    quantized: bool = False
    bits: int | None = None
    group_size: int | None = None

    def expected_shape(self) -> tuple[int, int]:
        if not self.quantized:
            return (self.out_dim, self.in_dim)
        if self.bits == 4:
            return (self.out_dim, math.ceil(self.in_dim / 2))
        if self.bits == 8:
            return (self.out_dim, self.in_dim)
        raise ValueError(f"Unsupported bits for {self.name}: {self.bits}")

    def expected_scales_shape(self) -> tuple[int, int] | None:
        if not self.quantized:
            return None
        if self.group_size is None or self.group_size <= 0:
            raise ValueError(f"group_size must be positive for quantized weight {self.name}")
        return (self.out_dim, math.ceil(self.in_dim / self.group_size))


@dataclass
class LayerWeightSpec:
    layer_idx: int
    q_proj: LinearWeightSpec
    k_proj: LinearWeightSpec
    v_proj: LinearWeightSpec
    o_proj: LinearWeightSpec
    gate_proj: LinearWeightSpec
    up_proj: LinearWeightSpec
    down_proj: LinearWeightSpec
    input_layernorm: tuple[str, int]
    post_attention_layernorm: tuple[str, int]


def _linear(name: str, out_dim: int, in_dim: int, *, quantized: bool, bits: int | None, group_size: int | None) -> LinearWeightSpec:
    return LinearWeightSpec(
        name=name,
        out_dim=out_dim,
        in_dim=in_dim,
        quantized=quantized,
        bits=bits if quantized else None,
        group_size=group_size if quantized else None,
    )


def llama_layer_weight_specs(config: LlamaLikeConfig, *, quantized: bool = False, bits: int = 4, group_size: int = 32) -> list[LayerWeightSpec]:
    config.validate()
    specs = []
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layers.{layer_idx}"
        specs.append(
            LayerWeightSpec(
                layer_idx=layer_idx,
                q_proj=_linear(f"{prefix}.self_attn.q_proj.weight", config.q_output_dim(), config.hidden_size, quantized=quantized, bits=bits, group_size=group_size),
                k_proj=_linear(f"{prefix}.self_attn.k_proj.weight", config.kv_output_dim(), config.hidden_size, quantized=quantized, bits=bits, group_size=group_size),
                v_proj=_linear(f"{prefix}.self_attn.v_proj.weight", config.kv_output_dim(), config.hidden_size, quantized=quantized, bits=bits, group_size=group_size),
                o_proj=_linear(f"{prefix}.self_attn.o_proj.weight", config.hidden_size, config.hidden_size, quantized=quantized, bits=bits, group_size=group_size),
                gate_proj=_linear(f"{prefix}.mlp.gate_proj.weight", config.intermediate_size, config.hidden_size, quantized=quantized, bits=bits, group_size=group_size),
                up_proj=_linear(f"{prefix}.mlp.up_proj.weight", config.intermediate_size, config.hidden_size, quantized=quantized, bits=bits, group_size=group_size),
                down_proj=_linear(f"{prefix}.mlp.down_proj.weight", config.hidden_size, config.intermediate_size, quantized=quantized, bits=bits, group_size=group_size),
                input_layernorm=(f"{prefix}.input_layernorm.weight", config.hidden_size),
                post_attention_layernorm=(f"{prefix}.post_attention_layernorm.weight", config.hidden_size),
            )
        )
    return specs


def fused_qkv_spec(config: LlamaLikeConfig, *, quantized: bool = False, bits: int = 4, group_size: int = 32) -> LinearWeightSpec:
    config.validate()
    return _linear(
        "fused_qkv.weight",
        config.qkv_output_dim(),
        config.hidden_size,
        quantized=quantized,
        bits=bits,
        group_size=group_size,
    )


def validate_weight_shapes(specs, weights: dict) -> None:
    if isinstance(specs, LayerWeightSpec):
        specs = [specs]
    if isinstance(specs, LinearWeightSpec):
        specs = [specs]
    for spec in specs:
        if isinstance(spec, LinearWeightSpec):
            _validate_linear_spec(spec, weights)
            continue
        for linear_name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            _validate_linear_spec(getattr(spec, linear_name), weights)
        for norm_name, dim in (spec.input_layernorm, spec.post_attention_layernorm):
            if norm_name not in weights:
                raise ValueError(f"Missing weight: {norm_name}")
            if tuple(weights[norm_name].shape) != (dim,):
                raise ValueError(f"{norm_name} must have shape {(dim,)}, got {weights[norm_name].shape}")


def _validate_linear_spec(spec: LinearWeightSpec, weights: dict) -> None:
    if spec.name not in weights:
        raise ValueError(f"Missing weight: {spec.name}")
    expected = spec.expected_shape()
    if tuple(weights[spec.name].shape) != expected:
        raise ValueError(f"{spec.name} must have shape {expected}, got {weights[spec.name].shape}")
    if spec.quantized:
        scales_name = f"{spec.name}.scales"
        if scales_name not in weights:
            raise ValueError(f"Missing quantization scales: {scales_name}")
        expected_scales = spec.expected_scales_shape()
        if tuple(weights[scales_name].shape) != expected_scales:
            raise ValueError(f"{scales_name} must have shape {expected_scales}, got {weights[scales_name].shape}")
