from __future__ import annotations

import numpy as np
import pytest


def _import_verify_ops():
    try:
        from ops.speculative_verify_ops import (
            ParallelVerificationConfig,
            ParallelVerificationPassResult,
            embed_proposed_tokens,
            parallel_verify_tokens,
            target_tokens_from_verification_logits,
        )
        return {
            "ParallelVerificationConfig": ParallelVerificationConfig,
            "ParallelVerificationPassResult": ParallelVerificationPassResult,
            "embed_proposed_tokens": embed_proposed_tokens,
            "parallel_verify_tokens": parallel_verify_tokens,
            "target_tokens_from_verification_logits": target_tokens_from_verification_logits,
        }
    except ImportError:
        pytest.skip("speculative_verify_ops require mlx (not available in this environment)")


class TestParallelVerificationConfig:
    def test_defaults(self):
        mod = _import_verify_ops()
        cfg = mod["ParallelVerificationConfig"]()
        assert cfg.draft_length == 4
        assert cfg.mode == "greedy_exact"
        assert cfg.cache_layout == "contiguous"
        cfg.validate()

    def test_invalid_draft_length(self):
        mod = _import_verify_ops()
        with pytest.raises((ValueError, AssertionError)):
            mod["ParallelVerificationConfig"](draft_length=0).validate()
        with pytest.raises((ValueError, AssertionError)):
            mod["ParallelVerificationConfig"](draft_length=-1).validate()

    def test_invalid_mode(self):
        mod = _import_verify_ops()
        with pytest.raises((ValueError, AssertionError)):
            mod["ParallelVerificationConfig"](mode="invalid").validate()

    def test_paged_unsupported(self):
        mod = _import_verify_ops()
        with pytest.raises((NotImplementedError, ValueError)):
            mod["ParallelVerificationConfig"](cache_layout="paged").validate()


class TestParallelVerificationPassResult:
    def test_create(self):
        mod = _import_verify_ops()
        result = mod["ParallelVerificationPassResult"](
            proposed_token_ids=[1, 2, 3],
            target_token_ids=[1, 2, 4],
            accept_mask=[True, True, False],
            accepted_count=2,
            replacement_token_id=4,
            target_logits=[[0.1, 0.2, 0.3]],
            staged_cache="dummy",
            metadata={"key": "value"},
        )
        assert result.accepted_count == 2
        assert result.replacement_token_id == 4
        assert result.accepted_tokens() == [1, 2]

    def test_to_verification_result(self):
        mod = _import_verify_ops()
        result = mod["ParallelVerificationPassResult"](
            proposed_token_ids=[1, 2, 3],
            target_token_ids=[1, 2, 4],
            accept_mask=[True, True, False],
            accepted_count=2,
            replacement_token_id=4,
            target_logits=None,
            staged_cache=None,
        )
        vr = result.to_verification_result()
        assert vr.accepted_count == 2
        assert vr.replacement_token_id == 4

    def test_empty_proposal(self):
        mod = _import_verify_ops()
        result = mod["ParallelVerificationPassResult"].empty()
        assert result.accepted_count == 0
        assert result.replacement_token_id is None


class TestEmbedProposedTokens:
    def test_requires_numpy(self):
        mod = _import_verify_ops()
        result = mod["embed_proposed_tokens"]([1, 2, 3], embed_dim=8)
        assert result.shape == (3, 8)

    def test_empty_raises(self):
        mod = _import_verify_ops()
        with pytest.raises((ValueError, AssertionError)):
            mod["embed_proposed_tokens"]([], embed_dim=8)


class TestTargetTokensFromLogits:
    def test_greedy_argmax(self):
        mod = _import_verify_ops()
        logits = np.array([[0.1, 0.9, 0.2], [0.8, 0.1, 0.1]], dtype=np.float32)
        tokens = mod["target_tokens_from_verification_logits"](logits)
        assert tokens == [1, 0]

    def test_empty_logits(self):
        mod = _import_verify_ops()
        logits = np.empty((0, 3), dtype=np.float32)
        tokens = mod["target_tokens_from_verification_logits"](logits)
        assert tokens == []


class TestParallelVerifyTokens:
    def test_empty_proposal(self):
        mod = _import_verify_ops()
        with pytest.raises((ValueError, AssertionError)):
            mod["parallel_verify_tokens"](
                proposed_token_ids=[],
                stack_cache="dummy",
                position=0,
                decode_step_fn=lambda *a: None,
                model=None,
            )

    def test_requires_stack_cache(self):
        mod = _import_verify_ops()
        with pytest.raises((ValueError, AssertionError, TypeError)):
            mod["parallel_verify_tokens"](
                proposed_token_ids=[1],
                stack_cache=None,
                position=0,
                decode_step_fn=lambda *a: None,
                model=None,
            )

    def test_requires_position(self):
        mod = _import_verify_ops()
        with pytest.raises((ValueError, AssertionError, TypeError)):
            mod["parallel_verify_tokens"](
                proposed_token_ids=[1],
                stack_cache="dummy",
                position=None,
                decode_step_fn=lambda *a: None,
                model=None,
            )


class TestVerifyIntegration:
    def test_to_verification_result_roundtrip(self):
        from models.speculative_decoding import verify_draft_tokens

        proposed = [1, 2, 3]
        target = [1, 2, 4]
        vr = verify_draft_tokens(proposed, target)
        assert vr.accepted_count == 2
        assert vr.replacement_token_id == 4
        assert vr.accept_mask == [True, True, False]
