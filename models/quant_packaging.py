from __future__ import annotations

import math
from dataclasses import dataclass

from .checkpoint_mapping import llama_layer_tensor_names
from .llama_config import LlamaLikeConfig


@dataclass
class QuantizedTensorSpec:
    name: str
    bits: int
    original_shape: tuple[int, int]
    packed_shape: tuple[int, ...]
    scales_shape: tuple[int, ...]
    zeros_shape: tuple[int, ...] | None
    group_size: int
    layout: str


def q4_packed_shape(weight_shape) -> tuple[int, int]:
    out_dim, in_dim = tuple(weight_shape)
    return (out_dim, math.ceil(in_dim / 2))


def q8_packed_shape(weight_shape) -> tuple[int, int]:
    out_dim, in_dim = tuple(weight_shape)
    return (out_dim, in_dim)


def quantized_linear_spec(name, weight_shape, *, bits=4, group_size=32, with_zeros=False) -> QuantizedTensorSpec:
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    original_shape = tuple(weight_shape)
    out_dim, in_dim = original_shape
    packed_shape = q4_packed_shape(original_shape) if bits == 4 else q8_packed_shape(original_shape)
    scales_shape = (out_dim, math.ceil(in_dim / group_size))
    zeros_shape = scales_shape if with_zeros else None
    return QuantizedTensorSpec(
        name=name,
        bits=bits,
        original_shape=original_shape,
        packed_shape=packed_shape,
        scales_shape=scales_shape,
        zeros_shape=zeros_shape,
        group_size=group_size,
        layout="groupwise_q4" if bits == 4 else "groupwise_q8",
    )


def llama_quantized_layer_specs(
    config: LlamaLikeConfig,
    *,
    layer_idx,
    bits=4,
    group_size=32,
    with_zeros=False,
) -> dict[str, QuantizedTensorSpec]:
    names = llama_layer_tensor_names(layer_idx)
    return {
        "q_proj": quantized_linear_spec(names.q_proj, (config.q_output_dim(), config.hidden_size), bits=bits, group_size=group_size, with_zeros=with_zeros),
        "k_proj": quantized_linear_spec(names.k_proj, (config.kv_output_dim(), config.hidden_size), bits=bits, group_size=group_size, with_zeros=with_zeros),
        "v_proj": quantized_linear_spec(names.v_proj, (config.kv_output_dim(), config.hidden_size), bits=bits, group_size=group_size, with_zeros=with_zeros),
        "o_proj": quantized_linear_spec(names.o_proj, (config.hidden_size, config.q_output_dim()), bits=bits, group_size=group_size, with_zeros=with_zeros),
        "gate_proj": quantized_linear_spec(names.gate_proj, (config.intermediate_size, config.hidden_size), bits=bits, group_size=group_size, with_zeros=with_zeros),
        "up_proj": quantized_linear_spec(names.up_proj, (config.intermediate_size, config.hidden_size), bits=bits, group_size=group_size, with_zeros=with_zeros),
        "down_proj": quantized_linear_spec(names.down_proj, (config.hidden_size, config.intermediate_size), bits=bits, group_size=group_size, with_zeros=with_zeros),
    }
