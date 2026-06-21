from __future__ import annotations

import pytest

from models.kv_offload import (
    KVBlockId,
    KVBlockMetadata,
    KVResidencyMap,
    partition_sequence_into_blocks,
)
from models.kv_offload_policy import (
    KVOffloadPolicyConfig,
    KVOffloadPlan,
    plan_offload_blocks,
    plan_prefetch_for_sparse_attention,
)


def _make_residency_map(seq_len=512, block_size=128, num_layers=1) -> KVResidencyMap:
    rmap = KVResidencyMap()
    for layer in range(num_layers):
        blocks = partition_sequence_into_blocks(
            layer_idx=layer,
            batch_idx=0,
            seq_len=seq_len,
            block_size=block_size,
            num_kv_heads=4,
            head_dim=64,
            dtype="float16",
        )
        for b in blocks:
            rmap.add_block(b)
    return rmap


# ---------------------------------------------------------------------------
# KVOffloadPolicyConfig
# ---------------------------------------------------------------------------

class TestKVOffloadPolicyConfig:
    def test_default(self):
        cfg = KVOffloadPolicyConfig()
        assert cfg.validate() is cfg

    def test_block_size_zero(self):
        with pytest.raises(ValueError, match="block_size"):
            KVOffloadPolicyConfig(block_size=0).validate()

    def test_keep_recent_negative(self):
        with pytest.raises(ValueError, match="keep_recent_blocks"):
            KVOffloadPolicyConfig(keep_recent_blocks=-1).validate()

    def test_keep_sink_negative(self):
        with pytest.raises(ValueError, match="keep_sink_blocks"):
            KVOffloadPolicyConfig(keep_sink_blocks=-1).validate()

    def test_max_resident_zero(self):
        with pytest.raises(ValueError, match="max_resident_blocks"):
            KVOffloadPolicyConfig(max_resident_blocks=0).validate()

    def test_latency_negative(self):
        with pytest.raises(ValueError, match="simulated_latency_ms"):
            KVOffloadPolicyConfig(simulated_latency_ms=-1).validate()

    def test_to_dict(self):
        d = KVOffloadPolicyConfig(block_size=64).to_dict()
        assert d["block_size"] == 64
        assert d["offload_enabled"] is True


# ---------------------------------------------------------------------------
# plan_offload_blocks
# ---------------------------------------------------------------------------

class TestPlanOffloadBlocks:
    def test_disabled_when_not_enabled(self):
        rmap = _make_residency_map(seq_len=256, block_size=128)
        cfg = KVOffloadPolicyConfig(offload_enabled=False)
        plan = plan_offload_blocks(rmap, current_position=128, policy_config=cfg)
        assert plan.reason == "offload_disabled"
        assert len(plan.offload) == 0

    def test_keeps_sink_blocks(self):
        rmap = _make_residency_map(seq_len=512, block_size=128)
        cfg = KVOffloadPolicyConfig(
            keep_sink_blocks=1, keep_recent_blocks=0,
            max_resident_blocks=None,
        )
        plan = plan_offload_blocks(rmap, current_position=256, policy_config=cfg)
        sink_ids = [bid for bid in plan.keep_resident if bid.block_idx == 0]
        assert len(sink_ids) >= 1

    def test_keeps_recent_blocks(self):
        rmap = _make_residency_map(seq_len=512, block_size=128)
        cfg = KVOffloadPolicyConfig(
            keep_sink_blocks=0, keep_recent_blocks=2,
            max_resident_blocks=None,
        )
        plan = plan_offload_blocks(rmap, current_position=256, policy_config=cfg)
        recent_block_idxs = {bid.block_idx for bid in plan.keep_resident}
        assert 2 in recent_block_idxs or 1 in recent_block_idxs
        assert 3 not in recent_block_idxs

    def test_max_resident_triggers_offload(self):
        rmap = _make_residency_map(seq_len=512, block_size=128)
        cfg = KVOffloadPolicyConfig(
            keep_sink_blocks=1, keep_recent_blocks=1,
            max_resident_blocks=2,
        )
        plan = plan_offload_blocks(rmap, current_position=256, policy_config=cfg)
        total_blocks = len(rmap.blocks)
        assert len(plan.offload) > 0
        assert len(plan.offload) <= total_blocks - 2

    def test_no_offload_when_all_kept(self):
        rmap = _make_residency_map(seq_len=256, block_size=128)
        cfg = KVOffloadPolicyConfig(
            keep_sink_blocks=10, keep_recent_blocks=10,
            max_resident_blocks=None,
        )
        plan = plan_offload_blocks(rmap, current_position=128, policy_config=cfg)
        assert len(plan.offload) == 0

    def test_reason_set(self):
        rmap = _make_residency_map(seq_len=512, block_size=128)
        cfg = KVOffloadPolicyConfig(keep_sink_blocks=0, keep_recent_blocks=0)
        plan = plan_offload_blocks(rmap, current_position=256, policy_config=cfg)
        assert isinstance(plan.reason, str)

    def test_offload_returns_block_ids(self):
        rmap = _make_residency_map(seq_len=256, block_size=128)
        cfg = KVOffloadPolicyConfig(
            keep_sink_blocks=0, keep_recent_blocks=0,
            max_resident_blocks=1,
        )
        plan = plan_offload_blocks(rmap, current_position=128, policy_config=cfg)
        for bid in plan.offload:
            assert isinstance(bid, KVBlockId)

    def test_plan_to_dict(self):
        rmap = _make_residency_map(seq_len=128, block_size=64)
        cfg = KVOffloadPolicyConfig(keep_sink_blocks=0, keep_recent_blocks=0)
        plan = plan_offload_blocks(rmap, current_position=64, policy_config=cfg)
        d = plan.to_dict()
        assert isinstance(d, dict)
        assert "reason" in d
        assert "num_keep" in d


# ---------------------------------------------------------------------------
# plan_prefetch_for_sparse_attention
# ---------------------------------------------------------------------------

class TestPlanPrefetchForSparseAttention:
    def test_identifies_offloaded_needed_blocks(self):
        rmap = _make_residency_map(seq_len=512, block_size=128)
        for key, meta in rmap.blocks.items():
            if meta.block_id.block_idx == 2:
                meta.resident = False
                meta.offloaded = True
                meta.store_uri = "memory://test"
        plan = plan_prefetch_for_sparse_attention(
            rmap,
            layer_idx=0, batch_idx=0,
            needed_positions=[250, 300],
            block_size=128,
        )
        assert len(plan.prefetch) >= 1
        prefetched_idxs = {bid.block_idx for bid in plan.prefetch}
        assert 2 in prefetched_idxs

    def test_resident_blocks_not_in_prefetch(self):
        rmap = _make_residency_map(seq_len=512, block_size=128)
        plan = plan_prefetch_for_sparse_attention(
            rmap,
            layer_idx=0, batch_idx=0,
            needed_positions=[0, 1, 2],
            block_size=128,
        )
        assert len(plan.prefetch) == 0
        assert len(plan.keep_resident) >= 1

    def test_reason_sparse_attention(self):
        rmap = _make_residency_map(seq_len=256, block_size=128)
        plan = plan_prefetch_for_sparse_attention(
            rmap, layer_idx=0, batch_idx=0,
            needed_positions=[10], block_size=128,
        )
        assert plan.reason == "sparse_attention_needed_blocks"

    def test_empty_positions(self):
        rmap = _make_residency_map(seq_len=256, block_size=128)
        plan = plan_prefetch_for_sparse_attention(
            rmap, layer_idx=0, batch_idx=0,
            needed_positions=[], block_size=128,
        )
        assert len(plan.prefetch) == 0
        assert len(plan.keep_resident) == 0
