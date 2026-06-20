from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .checkpoint_adapter import CheckpointAdapter


@dataclass
class LayerWeights:
    layer_idx: int
    q_proj: Any | None = None
    k_proj: Any | None = None
    v_proj: Any | None = None
    qkv_fused: Any | None = None
    o_proj: Any | None = None
    gate_proj: Any | None = None
    up_proj: Any | None = None
    down_proj: Any | None = None
    input_layernorm: Any | None = None
    post_attention_layernorm: Any | None = None

    def has_fused_qkv(self) -> bool:
        return self.qkv_fused is not None

    def available_names(self) -> list[str]:
        return sorted(name for name, value in self.__dict__.items() if name != "layer_idx" and value is not None)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        out = {}
        for name, value in self.__dict__.items():
            if name == "layer_idx" or value is None:
                continue
            shape = getattr(value, "shape", None)
            if shape is not None:
                out[name] = tuple(int(dim) for dim in shape)
        return out


class LayerWeightAdapter:
    def __init__(self, checkpoint_adapter: CheckpointAdapter):
        self.checkpoint_adapter = checkpoint_adapter

    def required_tensor_names(self, layer_idx: int) -> dict[str, str]:
        return self.checkpoint_adapter.layer_names(layer_idx).as_dict()

    def layer_shape_summary(self, layer_idx: int) -> dict[str, tuple[int, ...] | None]:
        return self.checkpoint_adapter.layer_shapes(layer_idx)

    def load_layer(self, layer_idx: int, *, fuse_qkv: bool = True, load_tensors: bool = True) -> LayerWeights:
        names = self.required_tensor_names(layer_idx)
        if not load_tensors:
            return LayerWeights(layer_idx=layer_idx)
        weights = LayerWeights(
            layer_idx=layer_idx,
            q_proj=self.checkpoint_adapter.load_tensor(names["q_proj"]),
            k_proj=self.checkpoint_adapter.load_tensor(names["k_proj"]),
            v_proj=self.checkpoint_adapter.load_tensor(names["v_proj"]),
            o_proj=self.checkpoint_adapter.load_tensor(names["o_proj"]),
            gate_proj=self.checkpoint_adapter.load_tensor(names["gate_proj"]),
            up_proj=self.checkpoint_adapter.load_tensor(names["up_proj"]),
            down_proj=self.checkpoint_adapter.load_tensor(names["down_proj"]),
            input_layernorm=self.checkpoint_adapter.load_tensor(names["input_layernorm"]),
            post_attention_layernorm=self.checkpoint_adapter.load_tensor(names["post_attention_layernorm"]),
        )
        if fuse_qkv:
            weights.qkv_fused = self.checkpoint_adapter.fuse_qkv_for_layer(layer_idx)
        return weights

    def to_kernel_layer_weights(self, layer_idx: int):
        _ = self.load_layer(layer_idx, fuse_qkv=True, load_tensors=True)
        raise NotImplementedError(
            "This adapter currently expects quantized packed weights for kernel execution."
        )
