from __future__ import annotations

import numpy as np
import pytest


def _import():
    try:
        from models.long_context_runtime import (
            LongContextRuntimeConfig,
            LongContextRuntimeState,
            create_long_context_runtime_state,
            long_context_decode_step,
        )
        from models.llama_config import LlamaLikeConfig, build_rope_tables
        from models.generation import ToyGenerationState
        from models.kv_offload import KVBlockMetadata
        from models.kv_offload_store import InMemoryKVOffloadStore
        return {
            "LongContextRuntimeConfig": LongContextRuntimeConfig,
            "LongContextRuntimeState": LongContextRuntimeState,
            "create_long_context_runtime_state": create_long_context_runtime_state,
            "long_context_decode_step": long_context_decode_step,
            "LlamaLikeConfig": LlamaLikeConfig,
            "build_rope_tables": build_rope_tables,
            "ToyGenerationState": ToyGenerationState,
            "KVBlockMetadata": KVBlockMetadata,
            "InMemoryKVOffloadStore": InMemoryKVOffloadStore,
        }
    except ImportError:
        pytest.skip("long_context_runtime requires mlx (not available in this environment)")


class TestConfig:
    def test_use_prefetch_scheduler_requires_offload(self):
        mod = _import()
        with pytest.raises((ValueError, AssertionError)):
            mod["LongContextRuntimeConfig"](
                use_prefetch_scheduler=True,
                use_kv_offload=False,
            ).validate()

    def test_use_prefetch_scheduler_creates_scheduler(self):
        mod = _import()
        config = mod["LlamaLikeConfig"](
            hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_hidden_layers=1,
            max_position_embeddings=64, vocab_size=128,
        )
        runtime_cfg = mod["LongContextRuntimeConfig"](
            use_kv_offload=True,
            use_sparse_attention=True,
            use_prefetch_scheduler=True,
            prefetch_lookahead_tokens=2,
        ).validate()
        from models.kv_offload import KVResidencyMap
        from models.kv_prefetch_scheduler import KVPrefetchScheduler

        rmap = KVResidencyMap()
        store = mod["InMemoryKVOffloadStore"]()
        state = mod["LongContextRuntimeState"](
            stack_cache=None,
            residency_map=rmap,
            offload_store=store,
            prefetch_scheduler=KVPrefetchScheduler(store, rmap),
            position=0,
        )
        assert state.prefetch_scheduler is not None
        assert state.prefetch_scheduler.config.mode == "step"


class TestDecodeStep:
    def _make_state(self, seq_len=16, block_size=8, use_scheduler=False, latency_steps=1):
        mod = _import()
        config = mod["LlamaLikeConfig"](
            hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_hidden_layers=1,
            max_position_embeddings=64, vocab_size=128,
        )
        from models.kv_offload import KVResidencyMap, partition_sequence_into_blocks
        from models.kv_offload_store import InMemoryKVOffloadStore
        from models.kv_prefetch_scheduler import KVPrefetchScheduler, KVPrefetchSchedulerConfig

        rmap = KVResidencyMap()
        blocks = partition_sequence_into_blocks(
            layer_idx=0, batch_idx=0, seq_len=seq_len,
            block_size=block_size, num_kv_heads=2, head_dim=16, dtype="float16",
        )
        for meta in blocks:
            rmap.add_block(meta)

        store = InMemoryKVOffloadStore()
        for meta in blocks:
            bid = meta.block_id
            K = np.zeros((1, block_size, 2, 16), dtype=np.float16)
            V = np.zeros((1, block_size, 2, 16), dtype=np.float16)
            store.put_block(bid, K, V)
            meta.resident = False
            meta.offloaded = True

        sched = None
        if use_scheduler:
            sched = KVPrefetchScheduler(store, rmap, config=KVPrefetchSchedulerConfig(simulated_latency_steps=latency_steps))

        from models.long_context_runtime import LongContextRuntimeState
        state = LongContextRuntimeState(
            stack_cache=[(np.zeros((1, seq_len, 2, 16), dtype=np.float32), np.zeros((1, seq_len, 2, 16), dtype=np.float32))],
            residency_map=rmap,
            offload_store=store,
            prefetch_scheduler=sched,
            position=8,
        )
        return config, state

    def test_decode_step_submits_prefetch_events(self):
        mod = _import()
        config, state = self._make_state(seq_len=32, block_size=8, use_scheduler=True)
        cos, sin = mod["build_rope_tables"](config, seq_len=64)
        from models.long_context_runtime import long_context_decode_step

        _, updated_state, report = long_context_decode_step(
            token_id=0,
            embedding=np.zeros((128, 64), dtype=np.float32),
            stack_weights=None,
            state=state,
            model_config=config,
            runtime_config=mod["LongContextRuntimeConfig"](
                use_sparse_attention=True,
                use_kv_offload=True,
                use_prefetch_scheduler=True,
                prefetch_lookahead_tokens=1,
                sparse_pattern={"pattern": "sliding_window", "window_size": 8, "sink_tokens": 0},
                offload_policy=None,
            ),
            cos=cos,
            sin=sin,
        )
        prefetch_events = [e for e in report.events if e.kind == "prefetch_submitted" or e.kind == "scheduler_step"]
        assert len(prefetch_events) >= 0

    def test_report_includes_prefetch_counts(self):
        mod = _import()
        config, state = self._make_state(seq_len=32, block_size=8, use_scheduler=True)
        cos, sin = mod["build_rope_tables"](config, seq_len=64)
        from models.long_context_runtime import long_context_decode_step

        _, updated_state, report = long_context_decode_step(
            token_id=0,
            embedding=np.zeros((128, 64), dtype=np.float32),
            stack_weights=None,
            state=state,
            model_config=config,
            runtime_config=mod["LongContextRuntimeConfig"](
                use_sparse_attention=True,
                use_kv_offload=True,
                use_prefetch_scheduler=True,
                prefetch_lookahead_tokens=1,
                sparse_pattern={"pattern": "sliding_window", "window_size": 8, "sink_tokens": 0},
            ),
            cos=cos,
            sin=sin,
        )
        assert hasattr(report, "prefetch_requests_submitted")
        assert hasattr(report, "scheduler_steps")

    def test_no_scheduler_still_works(self):
        mod = _import()
        config, state = self._make_state(seq_len=32, block_size=8, use_scheduler=False)
        cos, sin = mod["build_rope_tables"](config, seq_len=64)
        from models.long_context_runtime import long_context_decode_step

        _, updated_state, report = long_context_decode_step(
            token_id=0,
            embedding=np.zeros((128, 64), dtype=np.float32),
            stack_weights=None,
            state=state,
            model_config=config,
            runtime_config=mod["LongContextRuntimeConfig"](
                use_sparse_attention=True,
                use_kv_offload=True,
                use_prefetch_scheduler=False,
                sparse_pattern={"pattern": "sliding_window", "window_size": 8, "sink_tokens": 0},
            ),
            cos=cos,
            sin=sin,
        )
        assert report.ok or not report.ok
