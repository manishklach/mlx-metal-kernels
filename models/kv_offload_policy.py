from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .kv_offload import KVBlockId, KVBlockMetadata, KVResidencyMap, _block_key


# ---------------------------------------------------------------------------
# Offload policy config
# ---------------------------------------------------------------------------

@dataclass
class KVOffloadPolicyConfig:
    block_size: int = 128
    keep_recent_blocks: int = 4
    keep_sink_blocks: int = 1
    max_resident_blocks: int | None = None
    offload_enabled: bool = True
    simulated_latency_ms: float = 0.0

    def validate(self) -> KVOffloadPolicyConfig:
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.keep_recent_blocks < 0:
            raise ValueError(f"keep_recent_blocks must be >= 0, got {self.keep_recent_blocks}")
        if self.keep_sink_blocks < 0:
            raise ValueError(f"keep_sink_blocks must be >= 0, got {self.keep_sink_blocks}")
        if self.max_resident_blocks is not None and self.max_resident_blocks <= 0:
            raise ValueError(f"max_resident_blocks must be positive when set, got {self.max_resident_blocks}")
        if self.simulated_latency_ms < 0:
            raise ValueError(f"simulated_latency_ms must be >= 0, got {self.simulated_latency_ms}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "keep_recent_blocks": self.keep_recent_blocks,
            "keep_sink_blocks": self.keep_sink_blocks,
            "max_resident_blocks": self.max_resident_blocks,
            "offload_enabled": self.offload_enabled,
            "simulated_latency_ms": self.simulated_latency_ms,
        }


# ---------------------------------------------------------------------------
# Offload plan
# ---------------------------------------------------------------------------

@dataclass
class KVOffloadPlan:
    keep_resident: list[KVBlockId] = field(default_factory=list)
    offload: list[KVBlockId] = field(default_factory=list)
    prefetch: list[KVBlockId] = field(default_factory=list)
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keep_resident": [bid.to_string() for bid in self.keep_resident],
            "offload": [bid.to_string() for bid in self.offload],
            "prefetch": [bid.to_string() for bid in self.prefetch],
            "reason": self.reason,
            "num_keep": len(self.keep_resident),
            "num_offload": len(self.offload),
            "num_prefetch": len(self.prefetch),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Plan offload blocks
# ---------------------------------------------------------------------------

def _sink_block_ids(blocks: list[KVBlockMetadata], keep_sink: int) -> set[str]:
    from .kv_offload import _block_key
    seen: set[str] = set()
    for meta in blocks:
        if meta.block_id.block_idx < keep_sink:
            seen.add(_block_key(meta.block_id))
    return seen


def _recent_block_ids(blocks: list[KVBlockMetadata], current_position: int, block_size: int, keep_recent: int) -> set[str]:
    from .kv_offload import _block_key
    if keep_recent <= 0:
        return set()
    current_block = current_position // block_size
    start_block = max(0, current_block - keep_recent + 1)
    seen: set[str] = set()
    for meta in blocks:
        if start_block <= meta.block_id.block_idx <= current_block:
            seen.add(_block_key(meta.block_id))
    return seen


def plan_offload_blocks(
    residency_map: KVResidencyMap,
    *,
    current_position: int,
    policy_config: KVOffloadPolicyConfig,
    layer_idx: int | None = None,
    batch_idx: int = 0,
) -> KVOffloadPlan:
    config = policy_config.validate()
    if not config.offload_enabled:
        return KVOffloadPlan(reason="offload_disabled")

    total_blocks = list(residency_map.blocks.values())
    if layer_idx is not None:
        total_blocks = [b for b in total_blocks if b.block_id.layer_idx == layer_idx]
    total_blocks = [b for b in total_blocks if b.block_id.batch_idx == batch_idx]
    total_blocks.sort(key=lambda m: m.block_id.block_idx)

    sink_ids = _sink_block_ids(total_blocks, config.keep_sink_blocks) if config.keep_sink_blocks > 0 else set()
    recent_ids = _recent_block_ids(total_blocks, current_position, config.block_size, config.keep_recent_blocks)

    from .kv_offload import _block_key
    keep_ids: set[str] = sink_ids | recent_ids
    keep_ids_only_resident = {k for k in keep_ids if k in {_block_key(b.block_id) for b in total_blocks if b.resident}}
    offload_candidates = [b for b in total_blocks if _block_key(b.block_id) not in keep_ids and b.resident]

    if config.max_resident_blocks is not None:
        resident_blocks = [b for b in total_blocks if b.resident]
        num_must_keep = len(keep_ids_only_resident)
        remaining_budget = max(0, config.max_resident_blocks - num_must_keep)
        non_sink_non_recent = [b for b in resident_blocks if _block_key(b.block_id) not in keep_ids]
        if len(non_sink_non_recent) > remaining_budget:
            offload_count = len(non_sink_non_recent) - remaining_budget
            offload_candidates = offload_candidates[:offload_count]

    offload_ids = [b.block_id for b in offload_candidates]
    keep_resident_ids = [b.block_id for b in total_blocks if b.resident and _block_key(b.block_id) in keep_ids]

    return KVOffloadPlan(
        keep_resident=keep_resident_ids,
        offload=offload_ids,
        reason="offload_old_blocks" if offload_ids else "no_blocks_to_offload",
        metadata={
            "current_position": current_position,
            "total_blocks": len(total_blocks),
            "sink_blocks": len(sink_ids),
            "recent_blocks": len(recent_ids),
        },
    )


# ---------------------------------------------------------------------------
# Plan prefetch for sparse attention
# ---------------------------------------------------------------------------

def plan_prefetch_for_sparse_attention(
    residency_map: KVResidencyMap,
    *,
    layer_idx: int,
    batch_idx: int,
    needed_positions: list[int],
    block_size: int,
) -> KVOffloadPlan:
    from .kv_offload import _block_key, token_positions_to_block_ids

    needed_block_ids = token_positions_to_block_ids(
        needed_positions,
        layer_idx=layer_idx,
        batch_idx=batch_idx,
        block_size=block_size,
    )

    keep_resident: list[KVBlockId] = []
    prefetch: list[KVBlockId] = []
    for bid in needed_block_ids:
        meta = residency_map.get(bid)
        if meta is None:
            continue
        if meta.offloaded:
            prefetch.append(bid)
        elif meta.resident:
            keep_resident.append(bid)

    return KVOffloadPlan(
        keep_resident=keep_resident,
        prefetch=prefetch,
        reason="sparse_attention_needed_blocks",
        metadata={
            "layer_idx": layer_idx,
            "batch_idx": batch_idx,
            "needed_positions_count": len(needed_positions),
            "needed_blocks_count": len(needed_block_ids),
            "offloaded_needed": len(prefetch),
            "resident_needed": len(keep_resident),
        },
    )
