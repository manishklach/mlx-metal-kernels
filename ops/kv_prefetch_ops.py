from __future__ import annotations

from typing import Any

try:
    import mlx.core as mx
except ImportError:
    mx = None

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError("numpy is required for kv_prefetch_ops") from exc


def prefetch_blocks_for_sparse_decode(
    *,
    scheduler: Any,
    residency_map: Any,
    layer_idx: int,
    batch_idx: int,
    current_length: int,
    sparse_pattern: Any,
    block_size: int,
    lookahead_tokens: int = 1,
    priority: int = 0,
) -> list[Any]:
    from ops.long_context_ops import needed_positions_for_sparse_decode
    from ops.long_context_ops import needed_blocks_for_positions

    if lookahead_tokens <= 0:
        lookahead_tokens = 1

    needed: set[int] = set()
    for offset in range(lookahead_tokens):
        length = current_length + offset + 1
        if length <= 0:
            continue
        positions = needed_positions_for_sparse_decode(
            length=length,
            sparse_pattern=sparse_pattern,
        )
        needed.update(positions)

    if not needed:
        return []

    block_ids = needed_blocks_for_positions(
        sorted(needed),
        layer_idx=layer_idx,
        batch_idx=batch_idx,
        block_size=block_size,
    )

    requests: list[Any] = []
    for bid in block_ids:
        meta = residency_map.get(bid)
        if meta is not None and not meta.resident and meta.offloaded:
            req = scheduler.submit(bid, priority=priority, reason="sparse_decode_lookahead")
            requests.append(req)
        elif meta is None:
            pass

    return requests


def prefetch_blocks_for_speculative_draft(
    *,
    scheduler: Any,
    residency_map: Any,
    layer_idx: int,
    batch_idx: int,
    current_length: int,
    draft_length: int,
    sparse_pattern: Any,
    block_size: int,
    priority: int = 0,
) -> list[Any]:
    from ops.long_context_ops import needed_positions_for_sparse_decode
    from ops.long_context_ops import needed_blocks_for_positions

    if draft_length <= 0:
        return []

    needed: set[int] = set()
    for offset in range(1, draft_length + 1):
        length = current_length + offset
        if length <= 0:
            continue
        positions = needed_positions_for_sparse_decode(
            length=length,
            sparse_pattern=sparse_pattern,
        )
        needed.update(positions)

    if not needed:
        return []

    block_ids = needed_blocks_for_positions(
        sorted(needed),
        layer_idx=layer_idx,
        batch_idx=batch_idx,
        block_size=block_size,
    )

    requests: list[Any] = []
    for bid in block_ids:
        meta = residency_map.get(bid)
        if meta is not None and not meta.resident and meta.offloaded:
            req = scheduler.submit(bid, priority=priority, reason="speculative_draft_lookahead")
            requests.append(req)
        elif meta is None:
            pass

    return requests


def ensure_prefetched_before_attention(
    *,
    scheduler: Any,
    block_ids: list[Any],
    layer_caches: list[tuple[Any, Any]] | None = None,
    max_steps: int | None = None,
) -> list[Any]:
    if not block_ids:
        return []

    needing_prefetch: list[Any] = []
    for bid in block_ids:
        meta = getattr(scheduler, "residency_map", None).get(bid) if hasattr(scheduler, "residency_map") else None
        if meta is not None and meta.resident:
            continue
        if scheduler._is_block_pending(bid):
            needing_prefetch.append(bid)
        else:
            scheduler.submit(bid, priority=0, reason="ensure_prefetched")
            needing_prefetch.append(bid)

    if not needing_prefetch:
        return _make_all_resident_results(block_ids)

    results = scheduler.wait_for(needing_prefetch, layer_caches=layer_caches, max_steps=max_steps)

    failed = [r for r in results if not r.ok]
    if failed:
        failed_ids = [r.block_id.to_string() for r in failed]
        raise RuntimeError(
            f"ensure_prefetched_before_attention: {len(failed)} required block(s) failed: "
            f"{failed_ids}. Cannot proceed with attention."
        )

    return results


def _make_all_resident_results(block_ids: list[Any]) -> list[Any]:
    from models.kv_prefetch_scheduler import KVPrefetchResult, KVPrefetchRequestId

    results: list[Any] = []
    for bid in block_ids:
        rid = KVPrefetchRequestId(value=f"resident_{bid.layer_idx}_{bid.batch_idx}_{bid.block_idx}")
        results.append(KVPrefetchResult(
            request_id=rid,
            block_id=bid,
            ok=True,
            status="already_resident",
            message="Block was already resident; no prefetch needed.",
        ))
    return results
