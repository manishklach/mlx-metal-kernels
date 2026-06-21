from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError("numpy is required for kv_offload_store") from exc


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _maybe_mlx_to_numpy(value: Any) -> np.ndarray:
    try:
        import mlx.core as mx
        if isinstance(value, mx.array):
            return np.asarray(value)
    except ImportError:
        pass
    return _as_numpy(value)


def _compute_checksum(arr: np.ndarray) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Offload store protocol
# ---------------------------------------------------------------------------

class KVOffloadStore:
    def put_block(self, block_id, k_block, v_block) -> str:
        raise NotImplementedError

    def get_block(self, block_id):
        raise NotImplementedError

    def has_block(self, block_id) -> bool:
        raise NotImplementedError

    def delete_block(self, block_id):
        raise NotImplementedError

    def stats(self) -> dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# In-memory offload store
# ---------------------------------------------------------------------------

class InMemoryKVOffloadStore(KVOffloadStore):
    def __init__(self):
        self._blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._puts = 0
        self._gets = 0
        self._deletes = 0

    def put_block(self, block_id, k_block, v_block) -> str:
        key = block_id.to_string() if hasattr(block_id, "to_string") else str(block_id)
        self._blocks[key] = (_maybe_mlx_to_numpy(k_block), _maybe_mlx_to_numpy(v_block))
        self._puts += 1
        return f"memory://{key}"

    def get_block(self, block_id):
        key = block_id.to_string() if hasattr(block_id, "to_string") else str(block_id)
        if key not in self._blocks:
            raise KeyError(f"Block not found in in-memory store: {key}")
        self._gets += 1
        return self._blocks[key]

    def has_block(self, block_id) -> bool:
        key = block_id.to_string() if hasattr(block_id, "to_string") else str(block_id)
        return key in self._blocks

    def delete_block(self, block_id):
        key = block_id.to_string() if hasattr(block_id, "to_string") else str(block_id)
        if key not in self._blocks:
            raise KeyError(f"Block not found in in-memory store: {key}")
        del self._blocks[key]
        self._deletes += 1

    def stats(self) -> dict[str, Any]:
        total_bytes = sum(k.nbytes + v.nbytes for k, v in self._blocks.values())
        return {
            "blocks": len(self._blocks),
            "bytes": total_bytes,
            "puts": self._puts,
            "gets": self._gets,
            "deletes": self._deletes,
        }

    def clear(self):
        self._blocks.clear()
        self._puts = 0
        self._gets = 0
        self._deletes = 0


# ---------------------------------------------------------------------------
# File-backed offload store
# ---------------------------------------------------------------------------

class FileKVOffloadStore(KVOffloadStore):
    def __init__(self, root_dir: str | Path, *, format: str = "npy", overwrite: bool = False):
        self._root = Path(root_dir)
        self._format = format
        self._overwrite = overwrite
        self._root.mkdir(parents=True, exist_ok=True)
        self._puts = 0
        self._gets = 0
        self._deletes = 0

    def _block_dir(self, block_id) -> Path:
        key = block_id.to_string() if hasattr(block_id, "to_string") else str(block_id)
        safe_key = key.replace(":", "_").replace("/", "_")
        return self._root / safe_key

    def put_block(self, block_id, k_block, v_block) -> str:
        block_dir = self._block_dir(block_id)
        block_dir.mkdir(parents=True, exist_ok=True)
        k_arr = _maybe_mlx_to_numpy(k_block)
        v_arr = _maybe_mlx_to_numpy(v_block)
        k_path = block_dir / "k.npy"
        v_path = block_dir / "v.npy"
        if k_path.exists() and not self._overwrite:
            raise FileExistsError(f"Block K file already exists: {k_path}")
        with k_path.open("wb") as f:
            np.save(f, k_arr)
        with v_path.open("wb") as f:
            np.save(f, v_arr)
        meta = {
            "k_shape": list(k_arr.shape),
            "k_dtype": str(k_arr.dtype),
            "v_shape": list(v_arr.shape),
            "v_dtype": str(v_arr.dtype),
            "checksum_k": _compute_checksum(k_arr),
            "checksum_v": _compute_checksum(v_arr),
        }
        meta_path = block_dir / "meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f)
        self._puts += 1
        return str(block_dir)

    def get_block(self, block_id):
        block_dir = self._block_dir(block_id)
        k_path = block_dir / "k.npy"
        v_path = block_dir / "v.npy"
        if not k_path.exists() or not v_path.exists():
            raise FileNotFoundError(f"Block file(s) not found for {block_id.to_string() if hasattr(block_id, 'to_string') else block_id}")
        with k_path.open("rb") as f:
            k_arr = np.load(f)
        with v_path.open("rb") as f:
            v_arr = np.load(f)
        self._gets += 1
        return (k_arr, v_arr)

    def has_block(self, block_id) -> bool:
        block_dir = self._block_dir(block_id)
        k_path = block_dir / "k.npy"
        v_path = block_dir / "v.npy"
        return k_path.exists() and v_path.exists()

    def delete_block(self, block_id):
        block_dir = self._block_dir(block_id)
        if not block_dir.exists():
            raise KeyError(f"Block directory not found: {block_dir}")
        shutil.rmtree(block_dir)
        self._deletes += 1

    def stats(self) -> dict[str, Any]:
        block_dirs = [d for d in self._root.iterdir() if d.is_dir()]
        total_bytes = 0
        for d in block_dirs:
            for f in d.iterdir():
                if f.is_file():
                    total_bytes += f.stat().st_size
        return {
            "blocks": len(block_dirs),
            "bytes": total_bytes,
            "puts": self._puts,
            "gets": self._gets,
            "deletes": self._deletes,
            "root_dir": str(self._root),
        }
