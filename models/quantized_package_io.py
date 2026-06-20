from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .llama_config import LlamaLikeConfig
from .quantized_layer_package import QuantizedLlamaLayerPackage

_REQUIRED_LAYER_TENSOR_KEYS = [
    "input_layernorm",
    "post_attention_layernorm",
    "qkv",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

_VALID_ROLES = [
    "qkv",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "embedding",
    "lm_head",
    "norm",
    "final_norm",
    "other",
]

_FORMAT_VERSION = "0.1.0"


@dataclass
class QuantizedTensorMetadata:
    name: str
    role: str
    bits: int
    group_size: int
    original_shape: tuple[int, ...]
    packed_shape: tuple[int, ...]
    scales_shape: tuple[int, ...]
    zeros_shape: tuple[int, ...] | None = None
    dtype: str = "float16"
    data_file: str | None = None
    scales_file: str | None = None
    zeros_file: str | None = None
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["original_shape"] = list(self.original_shape)
        d["packed_shape"] = list(self.packed_shape)
        d["scales_shape"] = list(self.scales_shape)
        if self.zeros_shape is not None:
            d["zeros_shape"] = list(self.zeros_shape)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantizedTensorMetadata:
        return cls(
            name=data["name"],
            role=data["role"],
            bits=data["bits"],
            group_size=data["group_size"],
            original_shape=tuple(data["original_shape"]),
            packed_shape=tuple(data["packed_shape"]),
            scales_shape=tuple(data["scales_shape"]),
            zeros_shape=tuple(data["zeros_shape"]) if data.get("zeros_shape") is not None else None,
            dtype=data.get("dtype", "float16"),
            data_file=data.get("data_file"),
            scales_file=data.get("scales_file"),
            zeros_file=data.get("zeros_file"),
            checksum=data.get("checksum"),
        )


@dataclass
class QuantizedLayerMetadata:
    layer_idx: int
    tensors: dict[str, QuantizedTensorMetadata]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "tensors": {key: tensor.to_dict() for key, tensor in self.tensors.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantizedLayerMetadata:
        tensors = {}
        for key, tensor_data in data.get("tensors", {}).items():
            tensors[key] = QuantizedTensorMetadata.from_dict(tensor_data)
        return cls(layer_idx=data["layer_idx"], tensors=tensors)


@dataclass
class QuantizedCheckpointPackage:
    format_version: str = _FORMAT_VERSION
    model_type: str = "llama_like"
    config: dict[str, Any] = field(default_factory=dict)
    quantization: dict[str, Any] = field(default_factory=dict)
    layers: list[QuantizedLayerMetadata] = field(default_factory=list)
    global_tensors: dict[str, QuantizedTensorMetadata] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "model_type": self.model_type,
            "config": dict(self.config),
            "quantization": dict(self.quantization),
            "layers": [layer.to_dict() for layer in self.layers],
            "global_tensors": {key: tensor.to_dict() for key, tensor in self.global_tensors.items()},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantizedCheckpointPackage:
        layers_data = data.get("layers", [])
        layers = [QuantizedLayerMetadata.from_dict(ld) for ld in layers_data]
        global_tensors_data = data.get("global_tensors", {})
        global_tensors = {}
        for key, tensor_data in global_tensors_data.items():
            global_tensors[key] = QuantizedTensorMetadata.from_dict(tensor_data)
        return cls(
            format_version=data.get("format_version", _FORMAT_VERSION),
            model_type=data.get("model_type", "llama_like"),
            config=data.get("config", {}),
            quantization=data.get("quantization", {}),
            layers=layers,
            global_tensors=global_tensors,
            metadata=data.get("metadata", {}),
        )

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> QuantizedCheckpointPackage:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def validate(self, *, allow_partial: bool = False) -> QuantizedCheckpointPackage:
        if not self.format_version:
            raise ValueError("format_version must be non-empty")
        if not self.model_type:
            raise ValueError("model_type must be non-empty")
        if not self.layers and not allow_partial:
            raise ValueError("package must contain at least one layer unless allow_partial=True")
        seen_indices: set[int] = set()
        for layer in self.layers:
            if layer.layer_idx < 0:
                raise ValueError(f"layer_idx must be non-negative, got {layer.layer_idx}")
            if layer.layer_idx in seen_indices:
                raise ValueError(f"duplicate layer_idx: {layer.layer_idx}")
            seen_indices.add(layer.layer_idx)
            for tensor_key in _REQUIRED_LAYER_TENSOR_KEYS:
                if tensor_key not in layer.tensors and not allow_partial:
                    raise ValueError(
                        f"layer {layer.layer_idx} missing required tensor key: {tensor_key!r}"
                    )
            for key, tensor in layer.tensors.items():
                if any(dim <= 0 for dim in tensor.original_shape):
                    raise ValueError(
                        f"tensor {tensor.name!r} original_shape contains non-positive dims: {tensor.original_shape}"
                    )
                if tensor.role not in _VALID_ROLES:
                    raise ValueError(
                        f"tensor {tensor.name!r} has invalid role {tensor.role!r}; "
                        f"expected one of {_VALID_ROLES}"
                    )
                if tensor.bits not in (4, 8):
                    raise ValueError(
                        f"tensor {tensor.name!r} bits must be 4 or 8, got {tensor.bits}"
                    )
                if tensor.group_size <= 0:
                    raise ValueError(
                        f"tensor {tensor.name!r} group_size must be positive, got {tensor.group_size}"
                    )
        if not allow_partial and len(seen_indices) > 0:
            min_idx = min(seen_indices)
            max_idx = max(seen_indices)
            expected = set(range(min_idx, max_idx + 1))
            if seen_indices != expected:
                missing = sorted(expected - seen_indices)
                raise ValueError(
                    f"layer indices must be contiguous from {min_idx} to {max_idx}, "
                    f"missing indices: {missing}"
                )
        return self

    def num_layers(self) -> int:
        return len(self.layers)

    def tensor_count(self) -> int:
        count = len(self.global_tensors)
        for layer in self.layers:
            count += len(layer.tensors)
        return count

    def summary(self) -> dict[str, Any]:
        per_layer = {}
        for layer in self.layers:
            per_layer[str(layer.layer_idx)] = {
                name: list(t.original_shape) for name, t in layer.tensors.items()
            }
        return {
            "format_version": self.format_version,
            "model_type": self.model_type,
            "num_layers": self.num_layers(),
            "tensor_count": self.tensor_count(),
            "bits": self.quantization.get("bits"),
            "group_size": self.quantization.get("group_size"),
            "config_keys": list(self.config.keys()),
            "per_layer": per_layer,
            "global_tensors": list(self.global_tensors.keys()),
        }


def _quantized_tensor_metadata_from_linear(
    name: str,
    role: str,
    qkv_fused_q_output_dim: int | None,
    qkv_fused_kv_output_dim: int | None,
    package: QuantizedLlamaLayerPackage,
) -> QuantizedTensorMetadata:
    if role == "qkv":
        linear = package.qkv
        original_rows = qkv_fused_q_output_dim + 2 * qkv_fused_kv_output_dim
        shape = (original_rows, package.qkv.original_shape[1])
    elif role == "o_proj":
        linear = package.o_proj
        shape = linear.original_shape
    elif role == "gate_proj":
        linear = package.gate_proj
        shape = linear.original_shape
    elif role == "up_proj":
        linear = package.up_proj
        shape = linear.original_shape
    elif role == "down_proj":
        linear = package.down_proj
        shape = linear.original_shape
    else:
        raise ValueError(f"unsupported role for linear tensor: {role!r}")
    return QuantizedTensorMetadata(
        name=name,
        role=role,
        bits=linear.bits,
        group_size=linear.group_size,
        original_shape=shape,
        packed_shape=tuple(int(dim) for dim in linear.weight.shape),
        scales_shape=tuple(int(dim) for dim in linear.scales.shape),
        zeros_shape=tuple(int(dim) for dim in linear.zeros.shape) if linear.zeros is not None else None,
        dtype="float16",
    )


def _norm_tensor_metadata(name: str, weight) -> QuantizedTensorMetadata:
    shape = tuple(int(dim) for dim in weight.shape) if hasattr(weight, "shape") else (0,)
    return QuantizedTensorMetadata(
        name=name,
        role="norm",
        bits=0,
        group_size=0,
        original_shape=shape,
        packed_shape=shape,
        scales_shape=(0,),
        dtype="float16",
    )


def package_from_quantized_layers(
    config: LlamaLikeConfig,
    quantized_layers: list[QuantizedLlamaLayerPackage],
    *,
    bits: int = 4,
    group_size: int = 32,
    model_type: str = "llama_like",
    symmetric: bool = True,
    with_zeros: bool = False,
    metadata: dict[str, Any] | None = None,
) -> QuantizedCheckpointPackage:
    config.validate()
    layers: list[QuantizedLayerMetadata] = []
    q_output_dim = config.q_output_dim()
    kv_output_dim = config.kv_output_dim()
    for pkg in quantized_layers:
        tensors: dict[str, QuantizedTensorMetadata] = {}
        tensors["qkv"] = _quantized_tensor_metadata_from_linear(
            f"layers.{pkg.layer_idx}.qkv",
            "qkv",
            q_output_dim,
            kv_output_dim,
            pkg,
        )
        tensors["o_proj"] = _quantized_tensor_metadata_from_linear(
            f"layers.{pkg.layer_idx}.o_proj", "o_proj", None, None, pkg
        )
        tensors["gate_proj"] = _quantized_tensor_metadata_from_linear(
            f"layers.{pkg.layer_idx}.gate_proj", "gate_proj", None, None, pkg
        )
        tensors["up_proj"] = _quantized_tensor_metadata_from_linear(
            f"layers.{pkg.layer_idx}.up_proj", "up_proj", None, None, pkg
        )
        tensors["down_proj"] = _quantized_tensor_metadata_from_linear(
            f"layers.{pkg.layer_idx}.down_proj", "down_proj", None, None, pkg
        )
        if pkg.input_layernorm_weight is not None:
            tensors["input_layernorm"] = _norm_tensor_metadata(
                f"layers.{pkg.layer_idx}.input_layernorm", pkg.input_layernorm_weight
            )
        if pkg.post_attention_layernorm_weight is not None:
            tensors["post_attention_layernorm"] = _norm_tensor_metadata(
                f"layers.{pkg.layer_idx}.post_attention_layernorm", pkg.post_attention_layernorm_weight
            )
        layers.append(QuantizedLayerMetadata(layer_idx=pkg.layer_idx, tensors=tensors))
    quantization = {
        "bits": bits,
        "group_size": group_size,
        "symmetric": symmetric,
        "with_zeros": with_zeros,
    }
    return QuantizedCheckpointPackage(
        format_version=_FORMAT_VERSION,
        model_type=model_type,
        config=config.to_dict(),
        quantization=quantization,
        layers=layers,
        metadata=metadata or {},
    )
