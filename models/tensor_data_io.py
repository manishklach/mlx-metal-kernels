from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TensorDataInfo:
    file_path: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    checksum: str


def _import_numpy():
    import numpy as np
    return np


def compute_file_checksum(file_path: str | Path, algorithm: str = "sha256") -> str:
    path = Path(file_path)
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_tensor_npy(tensor: Any, file_path: str | Path) -> TensorDataInfo:
    np = _import_numpy()
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(tensor)
    with path.open("wb") as f:
        np.save(f, arr)
    checksum = compute_file_checksum(path)
    return TensorDataInfo(
        file_path=str(path),
        shape=tuple(int(dim) for dim in arr.shape),
        dtype=str(arr.dtype),
        nbytes=int(arr.nbytes),
        checksum=checksum,
    )


def load_tensor_npy(file_path: str | Path) -> Any:
    np = _import_numpy()
    with Path(file_path).open("rb") as f:
        return np.load(f)


def tensor_shape(file_path: str | Path) -> tuple[int, ...]:
    np = _import_numpy()
    arr = np.load(str(file_path), mmap_mode="r")
    return tuple(int(dim) for dim in arr.shape)


def tensor_dtype(file_path: str | Path) -> str:
    np = _import_numpy()
    arr = np.load(str(file_path), mmap_mode="r")
    return str(arr.dtype)


def tensor_nbytes(file_path: str | Path) -> int:
    np = _import_numpy()
    arr = np.load(str(file_path), mmap_mode="r")
    return int(arr.nbytes)


def validate_tensor_file(
    file_path: str | Path,
    *,
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: str | None = None,
    expected_checksum: str | None = None,
) -> list[str]:
    issues: list[str] = []
    path = Path(file_path)
    if not path.exists():
        issues.append(f"file not found: {path}")
        return issues
    try:
        shape = tensor_shape(path)
    except Exception as exc:
        issues.append(f"failed to read shape from {path}: {exc}")
        return issues
    if expected_shape is not None and shape != expected_shape:
        issues.append(
            f"shape mismatch for {path}: expected {expected_shape}, got {shape}"
        )
    try:
        dtype = tensor_dtype(path)
    except Exception as exc:
        issues.append(f"failed to read dtype from {path}: {exc}")
        return issues
    if expected_dtype is not None and dtype != expected_dtype:
        issues.append(
            f"dtype mismatch for {path}: expected {expected_dtype}, got {dtype}"
        )
    if expected_checksum is not None:
        try:
            actual = compute_file_checksum(path)
        except Exception as exc:
            issues.append(f"failed to compute checksum for {path}: {exc}")
            return issues
        if actual != expected_checksum:
            issues.append(
                f"checksum mismatch for {path}: expected {expected_checksum}, got {actual}"
            )
    return issues
