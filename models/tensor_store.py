from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpoint_manifest import CheckpointManifest


def _shape_of(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"Tensor object must expose a shape attribute, got {type(value)!r}")
    return tuple(int(dim) for dim in shape)


def _dtype_of(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise TypeError(f"Tensor object must expose a dtype attribute, got {type(value)!r}")
    name = getattr(dtype, "__name__", None)
    if name:
        return str(name)
    return str(dtype)


class TensorStore:
    def keys(self) -> list[str]:
        raise NotImplementedError

    def has(self, name: str) -> bool:
        return name in self.keys()

    def get_shape(self, name: str) -> tuple[int, ...]:
        raise NotImplementedError

    def get_dtype(self, name: str) -> str:
        raise NotImplementedError

    def load(self, name: str):
        raise NotImplementedError


class InMemoryTensorStore(TensorStore):
    def __init__(self, tensors: dict[str, Any]):
        self._tensors = dict(tensors)

    def keys(self) -> list[str]:
        return sorted(self._tensors)

    def has(self, name: str) -> bool:
        return name in self._tensors

    def _require(self, name: str) -> Any:
        if name not in self._tensors:
            raise KeyError(f"Missing tensor: {name}")
        return self._tensors[name]

    def get_shape(self, name: str) -> tuple[int, ...]:
        return _shape_of(self._require(name))

    def get_dtype(self, name: str) -> str:
        return _dtype_of(self._require(name))

    def load(self, name: str):
        return self._require(name)


class ManifestTensorStore(TensorStore):
    def __init__(self, manifest: CheckpointManifest):
        self.manifest = manifest

    def keys(self) -> list[str]:
        return self.manifest.tensor_names()

    def has(self, name: str) -> bool:
        return self.manifest.has(name)

    def _require(self, name: str):
        return self.manifest.require(name)

    def get_shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._require(name).shape)

    def get_dtype(self, name: str) -> str:
        return self._require(name).dtype

    def load(self, name: str):
        self._require(name)
        raise NotImplementedError("ManifestTensorStore is shape-only and cannot load tensor data")


class SafeTensorsTensorStore(TensorStore):
    def __init__(self, path):
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError("safetensors is optional. Install safetensors to use SafeTensorsTensorStore.") from exc
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"SafeTensors file not found: {self.path}")
        self._safe_open = safe_open
        self._framework = "numpy"
        with self._safe_open(str(self.path), framework=self._framework) as handle:
            self._keys = sorted(handle.keys())
            self._shapes = {name: tuple(int(dim) for dim in handle.get_slice(name).get_shape()) for name in self._keys}
            self._dtypes = {name: str(handle.get_slice(name).dtype) for name in self._keys}

    def keys(self) -> list[str]:
        return list(self._keys)

    def has(self, name: str) -> bool:
        return name in self._shapes

    def _require(self, name: str) -> str:
        if name not in self._shapes:
            raise KeyError(f"Missing tensor: {name}")
        return name

    def get_shape(self, name: str) -> tuple[int, ...]:
        return self._shapes[self._require(name)]

    def get_dtype(self, name: str) -> str:
        return self._dtypes[self._require(name)]

    def load(self, name: str):
        self._require(name)
        with self._safe_open(str(self.path), framework=self._framework) as handle:
            return handle.get_tensor(name)
