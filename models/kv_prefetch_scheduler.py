from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import mlx.core as mx
except ImportError:
    mx = None

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError("numpy is required for kv_prefetch_scheduler") from exc


# ---------------------------------------------------------------------------
# Request ID
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KVPrefetchRequestId:
    value: str

    @staticmethod
    def build(*, layer_idx: int, batch_idx: int, block_idx: int, seq: int = 0) -> KVPrefetchRequestId:
        return KVPrefetchRequestId(value=f"L{layer_idx}_B{batch_idx}_BLK{block_idx}_SEQ{seq}")

    def to_string(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass
class KVPrefetchRequest:
    request_id: KVPrefetchRequestId
    block_id: Any
    priority: int = 0
    issue_step: int = 0
    ready_step: int | None = None
    status: str = "queued"
    reason: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_done(self) -> bool:
        return self.status in ("complete", "failed", "cancelled")

    def is_ready(self, current_step: int) -> bool:
        if self.status != "in_flight":
            return False
        if self.ready_step is None:
            return False
        return current_step >= self.ready_step

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id.to_string(),
            "block_id": self.block_id.to_string() if hasattr(self.block_id, "to_string") else str(self.block_id),
            "priority": self.priority,
            "issue_step": self.issue_step,
            "ready_step": self.ready_step,
            "status": self.status,
            "reason": self.reason,
            "is_done": self.is_done(),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class KVPrefetchResult:
    request_id: KVPrefetchRequestId
    block_id: Any
    ok: bool
    status: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class KVPrefetchSchedulerStats:
    queued: int = 0
    in_flight: int = 0
    complete: int = 0
    failed: int = 0
    cancelled: int = 0
    issued: int = 0
    polled: int = 0
    blocks_loaded: int = 0
    bytes_loaded: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "queued": self.queued,
            "in_flight": self.in_flight,
            "complete": self.complete,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "issued": self.issued,
            "polled": self.polled,
            "blocks_loaded": self.blocks_loaded,
            "bytes_loaded": self.bytes_loaded,
        }


# ---------------------------------------------------------------------------
# Scheduler config
# ---------------------------------------------------------------------------


@dataclass
class KVPrefetchSchedulerConfig:
    mode: str = "step"
    max_in_flight: int = 4
    simulated_latency_steps: int = 1
    simulated_latency_ms: float = 0.0
    prioritize_recent: bool = True
    deduplicate_requests: bool = True
    fail_on_missing: bool = True

    def validate(self) -> KVPrefetchSchedulerConfig:
        valid_modes = ("step", "blocking")
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {self.mode!r}")
        if self.max_in_flight <= 0:
            raise ValueError(f"max_in_flight must be positive, got {self.max_in_flight}")
        if self.simulated_latency_steps < 0:
            raise ValueError(f"simulated_latency_steps must be >= 0, got {self.simulated_latency_steps}")
        if self.simulated_latency_ms < 0:
            raise ValueError(f"simulated_latency_ms must be >= 0, got {self.simulated_latency_ms}")
        return self


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class KVPrefetchScheduler:
    def __init__(self, store: Any, residency_map: Any, config: KVPrefetchSchedulerConfig | None = None):
        self.store = store
        self.residency_map = residency_map
        self.config = (config or KVPrefetchSchedulerConfig()).validate()
        self.current_step: int = 0
        self._queue: list[KVPrefetchRequest] = []
        self._in_flight: dict[str, KVPrefetchRequest] = {}
        self._completed: dict[str, KVPrefetchRequest] = {}
        self._failed: dict[str, KVPrefetchRequest] = {}
        self._req_seq: int = 0
        self.stats = KVPrefetchSchedulerStats()

    def _next_seq(self) -> int:
        seq = self._req_seq
        self._req_seq += 1
        return seq

    def _make_request_id(self, block_id: Any) -> KVPrefetchRequestId:
        return KVPrefetchRequestId.build(
            layer_idx=getattr(block_id, "layer_idx", 0),
            batch_idx=getattr(block_id, "batch_idx", 0),
            block_idx=getattr(block_id, "block_idx", 0),
            seq=self._next_seq(),
        )

    def _block_key(self, block_id: Any) -> str:
        if hasattr(block_id, "to_string"):
            return block_id.to_string()
        return str(block_id)

    def _is_block_pending(self, block_id: Any) -> bool:
        key = self._block_key(block_id)
        if key in self._completed or key in self._failed:
            return False
        for req in self._queue:
            if self._block_key(req.block_id) == key:
                return True
        if key in self._in_flight:
            return True
        return False

    def _find_existing(self, block_id: Any) -> KVPrefetchRequest | None:
        key = self._block_key(block_id)
        for req in self._queue:
            if self._block_key(req.block_id) == key:
                return req
        req = self._in_flight.get(key)
        if req is not None:
            return req
        req = self._completed.get(key)
        if req is not None:
            return req
        req = self._failed.get(key)
        if req is not None:
            return req
        return None

    def submit(self, block_id: Any, *, priority: int = 0, reason: str = "unknown", metadata: dict[str, Any] | None = None) -> KVPrefetchRequest:
        meta = self.residency_map.get(block_id)
        if meta is not None and meta.resident:
            rid = self._make_request_id(block_id)
            req = KVPrefetchRequest(
                request_id=rid,
                block_id=block_id,
                priority=priority,
                issue_step=self.current_step,
                ready_step=self.current_step,
                status="complete",
                reason=reason,
                metadata={"resident_already": True, **(metadata or {})},
            )
            self._completed[self._block_key(block_id)] = req
            self.stats.complete += 1
            return req

        if self.config.deduplicate_requests:
            existing = self._find_existing(block_id)
            if existing is not None:
                return existing

        rid = self._make_request_id(block_id)
        req = KVPrefetchRequest(
            request_id=rid,
            block_id=block_id,
            priority=priority,
            issue_step=self.current_step,
            status="queued",
            reason=reason,
            metadata=dict(metadata or {}),
        )
        self._queue.append(req)
        self.stats.queued += 1
        return req

    def submit_many(self, block_ids: list[Any], *, priority: int = 0, reason: str = "unknown") -> list[KVPrefetchRequest]:
        return [self.submit(bid, priority=priority, reason=reason) for bid in block_ids]

    def advance_step(self, layer_caches: list[tuple[Any, Any]] | None = None) -> None:
        self._issue_queued()
        self._complete_ready(layer_caches=layer_caches)
        self.current_step += 1
        self.stats.issued += 1

    def _issue_queued(self) -> None:
        slots = self.config.max_in_flight - len(self._in_flight)
        if slots <= 0:
            return

        if self.config.prioritize_recent:
            candidates = sorted(self._queue, key=lambda r: (-r.priority, r.issue_step))
        else:
            candidates = list(self._queue)

        to_issue = candidates[:slots]
        self._queue = [r for r in self._queue if r not in to_issue]

        for req in to_issue:
            req.status = "in_flight"
            req.issue_step = self.current_step
            req.ready_step = self.current_step + self.config.simulated_latency_steps
            self._in_flight[self._block_key(req.block_id)] = req
            self.stats.in_flight += 1
            self.stats.queued -= 1

    def _complete_ready(self, layer_caches: list[tuple[Any, Any]] | None = None) -> None:
        ready_keys = list(self._in_flight.keys())
        for key in ready_keys:
            req = self._in_flight[key]
            if req.is_ready(self.current_step):
                ok, message = self._do_prefetch(req, layer_caches=layer_caches)
                if ok:
                    req.status = "complete"
                    self._completed[key] = req
                    self.stats.complete += 1
                else:
                    req.status = "failed"
                    self._failed[key] = req
                    self.stats.failed += 1
                self.stats.in_flight -= 1
                del self._in_flight[key]

    def _do_prefetch(self, req: KVPrefetchRequest, layer_caches: list[tuple[Any, Any]] | None = None) -> tuple[bool, str]:
        from ops.kv_offload_ops import prefetch_kv_block

        bid = req.block_id
        meta = self.residency_map.get(bid)
        if meta is None:
            if self.config.fail_on_missing:
                return False, f"Block {self._block_key(bid)} not found in residency map"
            return True, f"Block {self._block_key(bid)} missing but fail_on_missing=False"
        if meta.resident:
            return True, f"Block {self._block_key(bid)} already resident"
        if not meta.offloaded:
            if self.config.fail_on_missing:
                return False, f"Block {self._block_key(bid)} is not offloaded (cannot prefetch)"
            return True, f"Block {self._block_key(bid)} not offloaded but fail_on_missing=False"
        if not self.store.has_block(bid):
            if self.config.fail_on_missing:
                return False, f"Block {self._block_key(bid)} not found in offload store"
            return True, f"Block {self._block_key(bid)} not in store but fail_on_missing=False"

        if layer_caches is not None:
            layer_idx = getattr(bid, "layer_idx", 0)
            if layer_idx < len(layer_caches):
                layer_cache = layer_caches[layer_idx]
                updated_cache, prefetch_result = prefetch_kv_block(layer_cache, meta, self.store)
                if isinstance(layer_caches, list):
                    layer_caches[layer_idx] = updated_cache
                self.stats.blocks_loaded += 1
                block_size = meta.end_token - meta.start_token
                num_kv_heads = getattr(meta, "num_kv_heads", 1)
                head_dim = getattr(meta, "head_dim", 64)
                bytes_per_element = 2
                self.stats.bytes_loaded += block_size * num_kv_heads * head_dim * bytes_per_element
                return True, f"Prefetched block {self._block_key(bid)} from store into layer {layer_idx}"
            return False, f"Layer index {layer_idx} out of range for {len(layer_caches)} caches"

        return True, f"Block {self._block_key(bid)} ready (layer_caches not provided, metadata updated)"

    def poll_ready(self, layer_caches: list[tuple[Any, Any]] | None = None) -> list[KVPrefetchResult]:
        self._complete_ready(layer_caches=layer_caches)
        self.stats.polled += 1

        results: list[KVPrefetchResult] = []
        for key, req in list(self._completed.items()):
            ok = True
            msg = "Prefetch completed"
            if hasattr(req.metadata, "get"):
                msg = req.metadata.get("message", msg)
            results.append(KVPrefetchResult(
                request_id=req.request_id,
                block_id=req.block_id,
                ok=ok,
                status=req.status,
                message=msg,
            ))
        for key, req in list(self._failed.items()):
            results.append(KVPrefetchResult(
                request_id=req.request_id,
                block_id=req.block_id,
                ok=False,
                status=req.status,
                message=f"Prefetch failed for block {self._block_key(req.block_id)}",
            ))
        return results

    def wait_for(
        self,
        block_ids: list[Any],
        layer_caches: list[tuple[Any, Any]] | None = None,
        max_steps: int | None = None,
    ) -> list[KVPrefetchResult]:
        if max_steps is None:
            max_steps = max(10, self.config.simulated_latency_steps * 2 + 1)

        pending_keys = set(self._block_key(bid) for bid in block_ids)
        steps_waited = 0

        while pending_keys and steps_waited < max_steps:
            self.advance_step(layer_caches=layer_caches)
            steps_waited += 1
            for key in list(pending_keys):
                if key in self._completed:
                    pending_keys.discard(key)
                elif key in self._failed:
                    pending_keys.discard(key)

        results: list[KVPrefetchResult] = []
        for bid in block_ids:
            key = self._block_key(bid)
            if key in self._completed:
                req = self._completed[key]
                results.append(KVPrefetchResult(
                    request_id=req.request_id,
                    block_id=bid,
                    ok=True,
                    status="complete",
                    message=f"Prefetch completed after {steps_waited} steps",
                ))
            elif key in self._failed:
                req = self._failed[key]
                results.append(KVPrefetchResult(
                    request_id=req.request_id,
                    block_id=bid,
                    ok=False,
                    status="failed",
                    message=f"Prefetch failed for block {key}",
                ))
            else:
                results.append(KVPrefetchResult(
                    request_id=self._make_request_id(bid),
                    block_id=bid,
                    ok=False,
                    status="timeout",
                    message=f"Block {key} not ready after {max_steps} steps (timeout)",
                ))

        return results

    def cancel(self, request_id: KVPrefetchRequestId) -> bool:
        key = request_id.to_string()
        for i, req in enumerate(self._queue):
            if req.request_id == request_id:
                req.status = "cancelled"
                self._queue.pop(i)
                self.stats.queued -= 1
                self.stats.cancelled += 1
                return True
        req = self._in_flight.get(key)
        if req is not None:
            req.status = "cancelled"
            del self._in_flight[key]
            self.stats.in_flight -= 1
            self.stats.cancelled += 1
            return True
        return False

    def clear_completed(self) -> int:
        count = len(self._completed) + len(self._failed)
        self._completed.clear()
        self._failed.clear()
        self.stats.complete = 0
        self.stats.failed = 0
        return count

    def pending_block_ids(self) -> list[Any]:
        ids: list[Any] = []
        for req in self._queue:
            ids.append(req.block_id)
        for req in self._in_flight.values():
            ids.append(req.block_id)
        return ids

    def stats_dict(self) -> dict[str, Any]:
        d = self.stats.to_dict()
        d["current_step"] = self.current_step
        d["mode"] = self.config.mode
        d["max_in_flight"] = self.config.max_in_flight
        d["simulated_latency_steps"] = self.config.simulated_latency_steps
        d["queue_length"] = len(self._queue)
        d["in_flight_count"] = len(self._in_flight)
        d["completed_count"] = len(self._completed)
        d["failed_count"] = len(self._failed)
        return d

    def describe(self) -> str:
        lines = [
            f"KVPrefetchScheduler (mode={self.config.mode}, step={self.current_step})",
            f"  Config: max_in_flight={self.config.max_in_flight}, "
            f"simulated_latency_steps={self.config.simulated_latency_steps}, "
            f"simulated_latency_ms={self.config.simulated_latency_ms}",
            f"  Queue: {len(self._queue)} queued, {len(self._in_flight)} in-flight, "
            f"{len(self._completed)} completed, {len(self._failed)} failed",
            f"  Stats: issued={self.stats.issued}, polled={self.stats.polled}, "
            f"blocks_loaded={self.stats.blocks_loaded}, bytes_loaded={self.stats.bytes_loaded}",
        ]
        return "\n".join(lines)
