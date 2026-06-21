from __future__ import annotations

from typing import Any

try:
    import mlx.core as mx
except ImportError:
    mx = None

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError("numpy is required for long_context_ops") from exc


def _is_dict_pattern(pattern: Any) -> bool:
    return isinstance(pattern, dict)


def _ensure_pattern(pattern: Any):
    if not _is_dict_pattern(pattern):
        return pattern
    return pattern


def needed_positions_for_sparse_decode(
    *,
    length: int,
    sparse_pattern: Any,
) -> list[int]:
    if _is_dict_pattern(sparse_pattern):
        pattern = sparse_pattern
    else:
        try:
            from ops.sparse_attention_ops import SparseAttentionPattern
            if isinstance(sparse_pattern, SparseAttentionPattern):
                pattern = {
                    "pattern": sparse_pattern.pattern,
                    "window_size": sparse_pattern.window_size,
                    "sink_tokens": sparse_pattern.sink_tokens,
                    "causal": sparse_pattern.causal,
                }
            else:
                raise TypeError("sparse_pattern must be a SparseAttentionPattern or dict")
        except ImportError:
            raise RuntimeError("SparseAttentionPattern requires mlx which is not installed") from None

    q_pos = length - 1
    if q_pos < 0:
        return []

    positions: set[int] = set()
    p = pattern.get("pattern", "sliding_window")
    window = int(pattern.get("window_size", 0) or 0)
    sink = int(pattern.get("sink_tokens", 0) or 0)

    if p == "dense":
        positions.update(range(length))
    elif p in ("sliding_window", "sliding_window_sink"):
        local_start = max(0, q_pos - window + 1)
        for pos in range(local_start, length):
            positions.add(pos)
        if p == "sliding_window_sink":
            sink_count = min(sink, length)
            for pos in range(sink_count):
                positions.add(pos)
    else:
        positions.update(range(length))

    return sorted(positions)


def needed_blocks_for_positions(
    positions: list[int],
    *,
    layer_idx: int,
    batch_idx: int,
    block_size: int,
) -> list[Any]:
    from models.kv_offload import KVBlockId

    seen: set[tuple[int, int, int]] = set()
    result: list[Any] = []
    for pos in sorted(set(positions)):
        if pos < 0:
            raise ValueError(f"token positions must be >= 0, got {pos}")
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


def ensure_blocks_ready_for_attention(
    *,
    layer_idx: int,
    batch_idx: int,
    needed_positions: list[int],
    residency_map: Any,
    offload_store: Any,
    layer_cache: Any,
    block_size: int,
    report: Any = None,
):
    from models.kv_offload import KVBlockId, token_positions_to_block_ids
    from ops.kv_offload_ops import prefetch_kv_block

    needed_block_ids = token_positions_to_block_ids(
        needed_positions,
        layer_idx=layer_idx,
        batch_idx=batch_idx,
        block_size=block_size,
    )

    missing: list[KVBlockId] = []
    for bid in needed_block_ids:
        meta = residency_map.get(bid)
        if meta is None:
            missing.append(bid)
            continue
        if not meta.resident and meta.offloaded:
            updated_cache, result = prefetch_kv_block(layer_cache, meta, offload_store)
            layer_cache = updated_cache
            meta.resident = True
            meta.offloaded = False
            if report is not None:
                report.events.append(
                    LongContextEvent(kind="prefetch", message=f"Prefetched block {bid.to_string()}", metadata={"block_id": bid.to_string()})
                )
                report.blocks_prefetched += 1
            continue
        if not meta.resident and not meta.offloaded:
            raise RuntimeError(
                f"Block {bid.to_string()} is neither resident nor offloaded. "
                "Cannot proceed with attention."
            )

    if missing:
        raise RuntimeError(
            f"Sparse attention requires {len(missing)} block(s) that have no residency metadata: "
            f"{[b.to_string() for b in missing]}. "
            "Initialize the residency map before running sparse decode."
        )

    return layer_cache


from dataclasses import dataclass, field
from typing import Any as _Any

try:
    from models.long_context_runtime import LongContextEvent
except ImportError:
    @dataclass
    class LongContextEvent:  # type: ignore
        kind: str = ""
        message: str = ""
        metadata: dict[str, _Any] = field(default_factory=dict)
