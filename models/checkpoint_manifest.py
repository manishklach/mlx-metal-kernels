from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    source: str | None = None
    offset: int | None = None
    nbytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("TensorInfo.name must be a non-empty string")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("TensorInfo.dtype must be a non-empty string")
        normalized_shape = tuple(int(dim) for dim in self.shape)
        if not normalized_shape or any(dim <= 0 for dim in normalized_shape):
            raise ValueError(f"TensorInfo.shape must contain positive integers, got {self.shape}")
        self.shape = normalized_shape

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TensorInfo":
        return cls(
            name=data["name"],
            shape=tuple(data["shape"]),
            dtype=data["dtype"],
            source=data.get("source"),
            offset=data.get("offset"),
            nbytes=data.get("nbytes"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CheckpointManifest:
    model_type: str
    tensors: dict[str, TensorInfo]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_type, str) or not self.model_type:
            raise ValueError("CheckpointManifest.model_type must be a non-empty string")
        if not isinstance(self.tensors, dict):
            raise ValueError("CheckpointManifest.tensors must be a dict[str, TensorInfo]")
        normalized: dict[str, TensorInfo] = {}
        for name, info in self.tensors.items():
            tensor = info if isinstance(info, TensorInfo) else TensorInfo.from_dict(info)
            if name != tensor.name:
                raise ValueError(f"Tensor dictionary key {name!r} does not match TensorInfo.name {tensor.name!r}")
            if name in normalized:
                raise ValueError(f"Duplicate tensor name in manifest: {name}")
            normalized[name] = tensor
        self.tensors = normalized

    def tensor_names(self) -> list[str]:
        return sorted(self.tensors)

    def get(self, name: str) -> TensorInfo | None:
        return self.tensors.get(name)

    def require(self, name: str) -> TensorInfo:
        if name not in self.tensors:
            raise KeyError(f"Missing tensor: {name}")
        return self.tensors[name]

    def has(self, name: str) -> bool:
        return name in self.tensors

    def find(self, prefix: str | None = None, suffix: str | None = None, contains: str | None = None) -> list[TensorInfo]:
        out = []
        for name in self.tensor_names():
            if prefix is not None and not name.startswith(prefix):
                continue
            if suffix is not None and not name.endswith(suffix):
                continue
            if contains is not None and contains not in name:
                continue
            out.append(self.tensors[name])
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "metadata": self.metadata,
            "tensors": {name: tensor.to_dict() for name, tensor in self.tensors.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointManifest":
        model_type = data["model_type"]
        metadata = data.get("metadata", {})
        tensors_data = data.get("tensors", {})
        if isinstance(tensors_data, list):
            tensors: dict[str, TensorInfo] = {}
            for item in tensors_data:
                tensor = TensorInfo.from_dict(item)
                if tensor.name in tensors:
                    raise ValueError(f"Duplicate tensor name in manifest list: {tensor.name}")
                tensors[tensor.name] = tensor
            return cls(model_type=model_type, tensors=tensors, metadata=metadata)
        if isinstance(tensors_data, dict):
            tensors = {}
            for name, item in tensors_data.items():
                payload = dict(item)
                payload.setdefault("name", name)
                tensor = TensorInfo.from_dict(payload)
                if tensor.name in tensors:
                    raise ValueError(f"Duplicate tensor name in manifest dict: {tensor.name}")
                tensors[tensor.name] = tensor
            return cls(model_type=model_type, tensors=tensors, metadata=metadata)
        raise ValueError("Manifest tensors must be either a dict or a list")

    @classmethod
    def load_json(cls, path) -> "CheckpointManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def save_json(self, path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
