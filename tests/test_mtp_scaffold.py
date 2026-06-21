from __future__ import annotations

import numpy as np
import pytest

from models.mtp import MTPConfig, SyntheticMTPHead, mtp_propose_tokens


# ---------------------------------------------------------------------------
# MTPConfig
# ---------------------------------------------------------------------------

class TestMTPConfig:
    def test_default_config(self):
        cfg = MTPConfig()
        assert cfg.num_draft_tokens == 4
        assert cfg.hidden_size == 64

    def test_validate_ok(self):
        cfg = MTPConfig(num_draft_tokens=5, hidden_size=128)
        assert cfg.validate() is cfg

    def test_validate_num_draft_tokens_zero(self):
        with pytest.raises(ValueError, match="num_draft_tokens"):
            MTPConfig(num_draft_tokens=0).validate()

    def test_validate_hidden_size_zero(self):
        with pytest.raises(ValueError, match="hidden_size"):
            MTPConfig(hidden_size=0).validate()

    def test_validate_num_layers_zero(self):
        with pytest.raises(ValueError, match="num_layers"):
            MTPConfig(num_layers=0).validate()

    def test_validate_max_seq_len_zero(self):
        with pytest.raises(ValueError, match="max_seq_len"):
            MTPConfig(max_seq_len=0).validate()

    def test_to_dict(self):
        d = MTPConfig(seed=7).to_dict()
        assert d["seed"] == 7
        assert d["num_draft_tokens"] == 4


# ---------------------------------------------------------------------------
# SyntheticMTPHead
# ---------------------------------------------------------------------------

class TestSyntheticMTPHead:
    def test_init(self):
        cfg = MTPConfig(hidden_size=16, num_draft_tokens=3, seed=42)
        head = SyntheticMTPHead(cfg, vocab_size=32)
        assert head.vocab_size == 32
        assert head._bias.shape == (32,)
        assert head._scale.shape == (16, 32)

    def test_forward_numpy(self):
        cfg = MTPConfig(hidden_size=8, num_draft_tokens=2, seed=1)
        head = SyntheticMTPHead(cfg, vocab_size=16)
        hidden = np.random.default_rng(0).normal(0, 1, (1, 5, 8)).astype(np.float32)
        logits = head.forward(hidden)
        assert logits.shape == (1, 5, 16)
        assert not np.all(np.isnan(logits))

    def test_forward_deterministic(self):
        cfg = MTPConfig(hidden_size=8, num_draft_tokens=2, seed=42)
        head = SyntheticMTPHead(cfg, vocab_size=16)
        hidden = np.ones((1, 3, 8), dtype=np.float32)
        logits_a = head.forward(hidden)
        logits_b = head.forward(hidden)
        np.testing.assert_array_almost_equal(logits_a, logits_b)

    def test_forward_batch(self):
        cfg = MTPConfig(hidden_size=4, seed=0)
        head = SyntheticMTPHead(cfg, vocab_size=8)
        hidden = np.random.default_rng(0).normal(0, 1, (2, 3, 4)).astype(np.float32)
        logits = head.forward(hidden)
        assert logits.shape == (2, 3, 8)

    def test_propose_numpy(self):
        cfg = MTPConfig(hidden_size=8, num_draft_tokens=3, seed=42)
        head = SyntheticMTPHead(cfg, vocab_size=16)
        hidden = np.ones((1, 2, 8), dtype=np.float32)
        logits = head.propose(hidden, 3)
        assert logits.shape == (1, 3, 16)

    def test_propose_zero_tokens(self):
        cfg = MTPConfig(hidden_size=8, seed=42)
        head = SyntheticMTPHead(cfg, vocab_size=16)
        hidden = np.ones((1, 2, 8), dtype=np.float32)
        logits = head.propose(hidden, 0)
        assert logits.shape == (1, 0, 16)

    def test_forward_preserves_input(self):
        cfg = MTPConfig(hidden_size=4, seed=0)
        head = SyntheticMTPHead(cfg, vocab_size=8)
        hidden = np.random.default_rng(1).normal(0, 1, (1, 2, 4)).astype(np.float32)
        hidden_copy = hidden.copy()
        head.forward(hidden)
        np.testing.assert_array_equal(hidden, hidden_copy)


# ---------------------------------------------------------------------------
# mtp_propose_tokens
# ---------------------------------------------------------------------------

class TestMTPProposeTokens:
    def test_propose_with_sampling(self):
        cfg = MTPConfig(hidden_size=8, num_draft_tokens=3, seed=42)
        head = SyntheticMTPHead(cfg, vocab_size=16)
        hidden = np.ones((1, 2, 8), dtype=np.float32)
        token_ids, logits = mtp_propose_tokens(head, hidden, 3, seed=0)
        assert len(token_ids) == 3
        assert all(0 <= tid < 16 for tid in token_ids)
        assert logits.shape == (1, 3, 16)

    def test_propose_zero_tokens(self):
        cfg = MTPConfig(hidden_size=8, seed=42)
        head = SyntheticMTPHead(cfg, vocab_size=16)
        hidden = np.ones((1, 2, 8), dtype=np.float32)
        token_ids, logits = mtp_propose_tokens(head, hidden, 0)
        assert token_ids == []

    def test_deterministic_with_seed(self):
        cfg = MTPConfig(hidden_size=8, num_draft_tokens=2, seed=42)
        head = SyntheticMTPHead(cfg, vocab_size=16)
        hidden = np.ones((1, 2, 8), dtype=np.float32)
        ids_a, _ = mtp_propose_tokens(head, hidden, 2, seed=42)
        ids_b, _ = mtp_propose_tokens(head, hidden, 2, seed=42)
        assert ids_a == ids_b
