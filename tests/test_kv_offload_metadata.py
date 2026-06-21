from __future__ import annotations

import pytest

from models.kv_offload import (
    KVBlockId,
    KVBlockMetadata,
    KVResidencyMap,
    partition_sequence_into_blocks,
    token_positions_to_block_ids,
)


# ---------------------------------------------------------------------------
# KVBlockId
# ---------------------------------------------------------------------------

class TestKVBlockId:
    def test_to_string(self):
        bid = KVBlockId(layer_idx=2, batch_idx=0, block_idx=5)
        assert bid.to_string() == "L2_B0_BLK5"

    def test_to_string_with_kv_heads(self):
        bid = KVBlockId(layer_idx=0, batch_idx=1, block_idx=3, kv_head_start=0, kv_head_end=4)
        assert bid.to_string() == "L0_B1_BLK3_H0-4"

    def test_from_string(self):
        bid = KVBlockId.from_string("L2_B0_BLK5")
        assert bid.layer_idx == 2
        assert bid.batch_idx == 0
        assert bid.block_idx == 5
        assert bid.kv_head_start is None

    def test_from_string_with_kv_heads(self):
        bid = KVBlockId.from_string("L0_B1_BLK3_H0-4")
        assert bid.layer_idx == 0
        assert bid.batch_idx == 1
        assert bid.block_idx == 3
        assert bid.kv_head_start == 0
        assert bid.kv_head_end == 4

    def test_roundtrip(self):
        original = KVBlockId(layer_idx=1, batch_idx=0, block_idx=7)
        restored = KVBlockId.from_string(original.to_string())
        assert original == restored

    def test_frozen(self):
        bid = KVBlockId(layer_idx=0, batch_idx=0, block_idx=0)
        with pytest.raises(AttributeError):
            bid.layer_idx = 1


# ---------------------------------------------------------------------------
# KVBlockMetadata
# ---------------------------------------------------------------------------

class TestKVBlockMetadata:
    def test_contains_token(self):
        meta = KVBlockMetadata(
            block_id=KVBlockId(0, 0, 0),
            start_token=0, end_token=128, num_tokens=128,
            num_kv_heads=8, head_dim=128, dtype="float16",
        )
        assert meta.contains_token(0)
        assert meta.contains_token(127)
        assert not meta.contains_token(128)

    def test_overlaps(self):
        meta = KVBlockMetadata(
            block_id=KVBlockId(0, 0, 0),
            start_token=64, end_token=128, num_tokens=64,
            num_kv_heads=8, head_dim=128, dtype="float16",
        )
        assert meta.overlaps(0, 100)
        assert meta.overlaps(64, 128)
        assert not meta.overlaps(0, 64)
        assert not meta.overlaps(128, 256)

    def test_shape(self):
        meta = KVBlockMetadata(
            block_id=KVBlockId(0, 0, 0),
            start_token=0, end_token=64, num_tokens=64,
            num_kv_heads=8, head_dim=128, dtype="float16",
        )
        assert meta.shape() == (1, 64, 8, 128)

    def test_to_dict_roundtrip(self):
        meta = KVBlockMetadata(
            block_id=KVBlockId(1, 0, 3, kv_head_start=0, kv_head_end=4),
            start_token=256, end_token=384, num_tokens=128,
            num_kv_heads=4, head_dim=64, dtype="float32",
            resident=False, offloaded=True, store_uri="/tmp/block",
            checksum="abc123", last_access_step=42,
            metadata={"custom": "value"},
        )
        d = meta.to_dict()
        restored = KVBlockMetadata.from_dict(d)
        assert restored.block_id == meta.block_id
        assert restored.start_token == meta.start_token
        assert restored.end_token == meta.end_token
        assert restored.num_tokens == meta.num_tokens
        assert restored.num_kv_heads == meta.num_kv_heads
        assert restored.head_dim == meta.head_dim
        assert restored.dtype == meta.dtype
        assert restored.resident == meta.resident
        assert restored.offloaded == meta.offloaded
        assert restored.store_uri == meta.store_uri
        assert restored.checksum == meta.checksum
        assert restored.last_access_step == meta.last_access_step
        assert restored.metadata["custom"] == "value"

    def test_default_resident(self):
        meta = KVBlockMetadata(
            block_id=KVBlockId(0, 0, 0),
            start_token=0, end_token=64, num_tokens=64,
            num_kv_heads=2, head_dim=16, dtype="float16",
        )
        assert meta.resident is True
        assert meta.offloaded is False


# ---------------------------------------------------------------------------
# partition_sequence_into_blocks
# ---------------------------------------------------------------------------

class TestPartitionSequenceIntoBlocks:
    def test_exact_blocks(self):
        blocks = partition_sequence_into_blocks(
            layer_idx=0, batch_idx=0, seq_len=256,
            block_size=128, num_kv_heads=8, head_dim=64, dtype="float16",
        )
        assert len(blocks) == 2
        assert blocks[0].start_token == 0
        assert blocks[0].end_token == 128
        assert blocks[1].start_token == 128
        assert blocks[1].end_token == 256

    def test_last_block_shorter(self):
        blocks = partition_sequence_into_blocks(
            layer_idx=0, batch_idx=0, seq_len=300,
            block_size=128, num_kv_heads=4, head_dim=32, dtype="float16",
        )
        assert len(blocks) == 3
        assert blocks[2].num_tokens == 44
        assert blocks[2].end_token == 300

    def test_single_block(self):
        blocks = partition_sequence_into_blocks(
            layer_idx=1, batch_idx=0, seq_len=64,
            block_size=128, num_kv_heads=2, head_dim=16, dtype="float16",
        )
        assert len(blocks) == 1
        assert blocks[0].block_id.block_idx == 0
        assert blocks[0].num_tokens == 64

    def test_all_resident_by_default(self):
        blocks = partition_sequence_into_blocks(
            layer_idx=0, batch_idx=0, seq_len=128,
            block_size=64, num_kv_heads=2, head_dim=16, dtype="float16",
        )
        for b in blocks:
            assert b.resident is True
            assert b.offloaded is False

    def test_invalid_seq_len(self):
        with pytest.raises(ValueError, match="seq_len"):
            partition_sequence_into_blocks(
                layer_idx=0, batch_idx=0, seq_len=0,
                block_size=64, num_kv_heads=2, head_dim=16, dtype="float16",
            )

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            partition_sequence_into_blocks(
                layer_idx=0, batch_idx=0, seq_len=128,
                block_size=0, num_kv_heads=2, head_dim=16, dtype="float16",
            )


# ---------------------------------------------------------------------------
# token_positions_to_block_ids
# ---------------------------------------------------------------------------

class TestTokenPositionsToBlockIds:
    def test_basic(self):
        ids = token_positions_to_block_ids(
            [0, 63, 64, 127, 128],
            layer_idx=0, batch_idx=0, block_size=64,
        )
        assert len(ids) == 3
        assert ids[0].block_idx == 0
        assert ids[1].block_idx == 1
        assert ids[2].block_idx == 2

    def test_deduplicates(self):
        ids = token_positions_to_block_ids(
            [10, 20, 30, 40],
            layer_idx=0, batch_idx=0, block_size=64,
        )
        assert len(ids) == 1
        assert ids[0].block_idx == 0

    def test_sorted(self):
        ids = token_positions_to_block_ids(
            [200, 10, 0, 150],
            layer_idx=0, batch_idx=0, block_size=64,
        )
        block_idxs = [bid.block_idx for bid in ids]
        assert block_idxs == sorted(block_idxs)

    def test_negative_position_raises(self):
        with pytest.raises(ValueError, match="token positions"):
            token_positions_to_block_ids(
                [-1, 0],
                layer_idx=0, batch_idx=0, block_size=64,
            )

    def test_non_positive_block_size_raises(self):
        with pytest.raises(ValueError, match="block_size"):
            token_positions_to_block_ids(
                [0, 1],
                layer_idx=0, batch_idx=0, block_size=0,
            )


# ---------------------------------------------------------------------------
# KVResidencyMap
# ---------------------------------------------------------------------------

class TestKVResidencyMap:
    def test_add_and_get(self):
        meta = KVBlockMetadata(
            block_id=KVBlockId(0, 0, 0),
            start_token=0, end_token=64, num_tokens=64,
            num_kv_heads=2, head_dim=16, dtype="float16",
        )
        rmap = KVResidencyMap()
        rmap.add_block(meta)
        assert rmap.get(KVBlockId(0, 0, 0)) is meta

    def test_get_missing(self):
        rmap = KVResidencyMap()
        assert rmap.get(KVBlockId(9, 0, 0)) is None

    def test_mark_resident(self):
        meta = KVBlockMetadata(
            block_id=KVBlockId(0, 0, 0),
            start_token=0, end_token=64, num_tokens=64,
            num_kv_heads=2, head_dim=16, dtype="float16",
            resident=False, offloaded=True,
        )
        rmap = KVResidencyMap()
        rmap.add_block(meta)
        rmap.mark_resident(KVBlockId(0, 0, 0))
        assert rmap.get(KVBlockId(0, 0, 0)).resident is True
        assert rmap.get(KVBlockId(0, 0, 0)).offloaded is False

    def test_mark_offloaded(self):
        meta = KVBlockMetadata(
            block_id=KVBlockId(0, 0, 0),
            start_token=0, end_token=64, num_tokens=64,
            num_kv_heads=2, head_dim=16, dtype="float16",
        )
        rmap = KVResidencyMap()
        rmap.add_block(meta)
        rmap.mark_offloaded(KVBlockId(0, 0, 0), store_uri="memory://test", checksum="abc")
        m = rmap.get(KVBlockId(0, 0, 0))
        assert m.resident is False
        assert m.offloaded is True
        assert m.store_uri == "memory://test"
        assert m.checksum == "abc"

    def test_resident_blocks(self):
        rmap = KVResidencyMap()
        for i in range(4):
            rmap.add_block(KVBlockMetadata(
                block_id=KVBlockId(0, 0, i),
                start_token=i * 64, end_token=(i + 1) * 64, num_tokens=64,
                num_kv_heads=2, head_dim=16, dtype="float16",
                resident=i < 2,
                offloaded=i >= 2,
            ))
        assert len(rmap.resident_blocks()) == 2
        assert len(rmap.offloaded_blocks()) == 2

    def test_blocks_for_token_range(self):
        rmap = KVResidencyMap()
        for i in range(4):
            rmap.add_block(KVBlockMetadata(
                block_id=KVBlockId(0, 0, i),
                start_token=i * 64, end_token=(i + 1) * 64, num_tokens=64,
                num_kv_heads=2, head_dim=16, dtype="float16",
            ))
        result = rmap.blocks_for_token_range(0, 0, 64, 192)
        assert len(result) == 2
        assert result[0].block_id.block_idx == 1
        assert result[1].block_id.block_idx == 2

    def test_blocks_for_sparse_positions(self):
        rmap = KVResidencyMap()
        for i in range(4):
            rmap.add_block(KVBlockMetadata(
                block_id=KVBlockId(0, 0, i),
                start_token=i * 64, end_token=(i + 1) * 64, num_tokens=64,
                num_kv_heads=2, head_dim=16, dtype="float16",
            ))
        result = rmap.blocks_for_sparse_positions(0, 0, [0, 1, 128, 200])
        assert len(result) == 3
        assert [b.block_id.block_idx for b in result] == [0, 2, 3]

    def test_summary(self):
        rmap = KVResidencyMap()
        for i in range(3):
            rmap.add_block(KVBlockMetadata(
                block_id=KVBlockId(0, 0, i),
                start_token=i * 64, end_token=(i + 1) * 64, num_tokens=64,
                num_kv_heads=2, head_dim=16, dtype="float16",
                resident=i > 0,
                offloaded=i == 0,
            ))
        s = rmap.summary()
        assert s["total_blocks"] == 3
        assert s["resident_blocks"] == 2
        assert s["offloaded_blocks"] == 1
