from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError("numpy is required for kv_offload") from exc


# ---------------------------------------------------------------------------
# KV block ID
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KVBlockId:
    layer_idx: int
    batch_idx: int
    block_idx: int
    kv_head_start: int | None = None
    kv_head_end: int | None = None

    def to_string(self) -> str:
        parts = [f"L{self.layer_idx}", f"B{self.batch_idx}", f"BLK{self.block_idx}"]
        if self.kv_head_start is not None and self.kv_head_end is not None:
            parts.append(f"H{self.kv_head_start}-{self.kv_head_end}")
        return "_".join(parts)

    @staticmethod
    def from_string(s: str) -> KVBlockId:
        parts = s.split("_")
        layer_idx = int(parts[0][1:])
        batch_idx = int(parts[1][1:])
        block_idx = int(parts[2][3:])
        kv_head_start = None
        kv_head_end = None
        if len(parts) >= 4:
            h_part = parts[3]
            if h_part.startswith("H"):
                h_range = h_part[1:]
                if "-" in h_range:
                    start_s, end_s = h_range.split("-", 1)
                    kv_head_start = int(start_s)
                    kv_head_end = int(end_s)
        return KVBlockId(
            layer_idx=layer_idx,
            batch_idx=batch_idx,
            block_idx=block_idx,
            kv_head_start=kv_head_start,
            kv_head_end=kv_head_end,
        )


# ---------------------------------------------------------------------------
# KV block metadata
# ---------------------------------------------------------------------------

@dataclass
class KVBlockMetadata:
    block_id: KVBlockId
    start_token: int
    end_token: int
    num_tokens: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    resident: bool = True
    offloaded: bool = False
    dirty: bool = False
    store_uri: str | None = None
    checksum: str | None = None
    last_access_step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def contains_token(self, pos: int) -> bool:
        return self.start_token <= pos < self.end_token

    def overlaps(self, start: int, end: int) -> bool:
        return self.start_token < end and self.end_token > start

    def shape(self) -> tuple[int, int, int, int]:
        return (1, self.num_tokens, self.num_kv_heads, self.head_dim)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id.to_string(),
            "start_token": self.start_token,
            "end_token": self.end_token,
            "num_tokens": self.num_tokens,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "dtype": self.dtype,
            "resident": self.resident,
            "offloaded": self.offloaded,
            "dirty": self.dirty,
            "store_uri": self.store_uri,
            "checksum": self.checksum,
            "last_access_step": self.last_access_step,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> KVBlockMetadata:
        block_id = KVBlockId.from_string(d["block_id"])
        return KVBlockMetadata(
            block_id=block_id,
            start_token=int(d["start_token"]),
            end_token=int(d["end_token"]),
            num_tokens=int(d["num_tokens"]),
            num_kv_heads=int(d["num_kv_heads"]),
            head_dim=int(d["head_dim"]),
            dtype=str(d["dtype"]),
            resident=bool(d.get("resident", True)),
            offloaded=bool(d.get("offloaded", False)),
            dirty=bool(d.get("dirty", False)),
            store_uri=d.get("store_uri"),
            checksum=d.get("checksum"),
            last_access_step=int(d.get("last_access_step", 0)),
            metadata=dict(d.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Block shape convention
# ---------------------------------------------------------------------------
# K/V block shape: [1, num_tokens, num_kv_heads, head_dim]
# This preserves the batch dimension for consistency with MLX cache shapes.
# The leading 1 is the batch dimension (B=1 for now).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Residency map
# ---------------------------------------------------------------------------

def _block_key(block_id: KVBlockId) -> str:
    return block_id.to_string()


@dataclass
class KVResidencyMap:
    blocks: dict[str, KVBlockMetadata] = field(default_factory=dict)

    def add_block(self, meta: KVBlockMetadata) -> None:
        key = _block_key(meta.block_id)
        self.blocks[key] = meta

    def get(self, block_id: KVBlockId) -> KVBlockMetadata | None:
        return self.blocks.get(_block_key(block_id))

    def mark_resident(self, block_id: KVBlockId) -> None:
        key = _block_key(block_id)
        if key in self.blocks:
            self.blocks[key].resident = True
            self.blocks[key].offloaded = False

    def mark_offloaded(self, block_id: KVBlockId, *, store_uri: str | None = None, checksum: str | None = None) -> None:
        key = _block_key(block_id)
        if key in self.blocks:
            self.blocks[key].resident = False
            self.blocks[key].offloaded = True
            if store_uri is not None:
                self.blocks[key].store_uri = store_uri
            if checksum is not None:
                self.blocks[key].checksum = checksum

    def resident_blocks(self) -> list[KVBlockMetadata]:
        return [b for b in self.blocks.values() if b.resident]

    def offloaded_blocks(self) -> list[KVBlockMetadata]:
        return [b for b in self.blocks.values() if b.offloaded]

    def blocks_for_token_range(self, layer_idx: int, batch_idx: int, start: int, end: int) -> list[KVBlockMetadata]:
        result: list[KVBlockMetadata] = []
        for meta in self.blocks.values():
            if meta.block_id.layer_idx == layer_idx and meta.block_id.batch_idx == batch_idx:
                if meta.overlaps(start, end):
                    result.append(meta)
        result.sort(key=lambda m: m.block_id.block_idx)
        return result

    def blocks_for_sparse_positions(self, layer_idx: int, batch_idx: int, positions: list[int]) -> list[KVBlockMetadata]:
        seen: set[str] = set()
        result: list[KVBlockMetadata] = []
        for pos in sorted(set(positions)):
            for meta in self.blocks.values():
                if meta.block_id.layer_idx == layer_idx and meta.block_id.batch_idx == batch_idx:
                    if meta.contains_token(pos):
                        key = _block_key(meta.block_id)
                        if key not in seen:
                            seen.add(key)
                            result.append(meta)
        result.sort(key=lambda m: m.block_id.block_idx)
        return result

    def summary(self) -> dict[str, Any]:
        resident = self.resident_blocks()
        offloaded = self.offloaded_blocks()
        return {
            "total_blocks": len(self.blocks),
            "resident_blocks": len(resident),
            "offloaded_blocks": len(offloaded),
            "resident_tokens": sum(b.num_tokens for b in resident),
            "offloaded_tokens": sum(b.num_tokens for b in offloaded),
            "layers": sorted(set(b.block_id.layer_idx for b in self.blocks.values())),
        }


# ---------------------------------------------------------------------------
# Partition helpers
# ---------------------------------------------------------------------------

def partition_sequence_into_blocks(
    *,
    layer_idx: int,
    batch_idx: int,
    seq_len: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: str = "float16",
) -> list[KVBlockMetadata]:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    blocks: list[KVBlockMetadata] = []
    start = 0
    block_idx = 0
    while start < seq_len:
        end = min(start + block_size, seq_len)
        num_tokens = end - start
        block_id = KVBlockId(
            layer_idx=layer_idx,
            batch_idx=batch_idx,
            block_idx=block_idx,
        )
        meta = KVBlockMetadata(
            block_id=block_id,
            start_token=start,
            end_token=end,
            num_tokens=num_tokens,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            resident=True,
            offloaded=False,
        )
        blocks.append(meta)
        start = end
        block_idx += 1
    return blocks


def token_positions_to_block_ids(
    positions: list[int],
    *,
    layer_idx: int,
    batch_idx: int,
    block_size: int,
) -> list[KVBlockId]:
    seen: set[tuple[int, int, int]] = set()
    result: list[KVBlockId] = []
    for pos in sorted(set(positions)):
        block_idx = pos // block_size
        key = (layer_idx, batch_idx, block_idx)
        if key not in seen:
            seen.add(key)
            result.append(
                KVBlockId(
                    layer_idx=layer_idx,
                    batch_idx=batch_idx,
                    block_idx=block_idx,
                )
            )
    return result
