from __future__ import annotations

import numpy as np
import pytest

from models.speculative_decoding import (
    FixedDraftProposer,
    PipelineTargetVerifier,
    RandomDraftProposer,
    SpeculativeConfig,
    SpeculativeGenerationResult,
    SpeculativeGenerator,
    SpeculativeStepResult,
    VerificationResult,
    accepted_prefix_length,
    compute_accept_mask,
    verify_draft_tokens,
)


def _get_greedy_self_draft_proposer():
    from models.speculative_decoding import GreedySelfDraftProposer
    return GreedySelfDraftProposer


class _DummyPipeline:
    def __init__(self, vocab_size=128):
        self.vocab_size = vocab_size
        self.config = _DummyConfig()
        self.generation_config = _DummyGenConfig()

    def generate_ids(self, input_ids, generation_config=None, **kwargs):
        last = input_ids[-1] if input_ids else 0
        n = (generation_config.max_new_tokens if generation_config else 1)
        return list(input_ids) + [(last + 1 + i) % self.vocab_size for i in range(n)]

    def encode(self, prompt):
        return [10, 20, 30]

    def decode(self, token_ids):
        return " ".join(str(t) for t in token_ids)


class _DummyConfig:
    hidden_size = 64
    intermediate_size = 128
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = 16
    num_hidden_layers = 2
    max_position_embeddings = 128
    vocab_size = 128
    bits = 4
    group_size = 32
    dtype = "float16"
    backend_preset = "reference"
    cache_layout = "contiguous"
    use_prefill = True
    use_prefix_cache = False


class _DummyGenConfig:
    max_new_tokens = 16
    temperature = 1.0
    top_k = None
    top_p = None
    eos_token_id = None
    seed = 0
    backend_preset = "reference"
    repetition_penalty = 1.0

    def validate(self):
        return self


# ---------------------------------------------------------------------------
# SpeculativeConfig
# ---------------------------------------------------------------------------

class TestSpeculativeConfig:
    def test_default_config(self):
        cfg = SpeculativeConfig()
        assert cfg.draft_length == 4
        assert cfg.max_new_tokens == 16
        assert cfg.temperature == 1.0

    def test_validate_ok(self):
        cfg = SpeculativeConfig(draft_length=3, max_new_tokens=32)
        assert cfg.validate() is cfg

    def test_validate_draft_length_zero(self):
        with pytest.raises(ValueError, match="draft_length"):
            SpeculativeConfig(draft_length=0).validate()

    def test_validate_draft_length_negative(self):
        with pytest.raises(ValueError, match="draft_length"):
            SpeculativeConfig(draft_length=-1).validate()

    def test_validate_max_new_tokens_zero(self):
        with pytest.raises(ValueError, match="max_new_tokens"):
            SpeculativeConfig(max_new_tokens=0).validate()

    def test_validate_temperature_zero(self):
        with pytest.raises(ValueError, match="temperature"):
            SpeculativeConfig(temperature=0).validate()

    def test_validate_top_k_zero(self):
        with pytest.raises(ValueError, match="top_k"):
            SpeculativeConfig(top_k=0).validate()

    def test_validate_top_p_invalid(self):
        with pytest.raises(ValueError, match="top_p"):
            SpeculativeConfig(top_p=-1).validate()

    def test_validate_cache_layout_paged(self):
        cfg = SpeculativeConfig(cache_layout="paged")
        assert cfg.validate() is cfg

    def test_validate_cache_layout_invalid(self):
        with pytest.raises(ValueError, match="cache_layout"):
            SpeculativeConfig(cache_layout="unknown").validate()

    def test_to_dict(self):
        d = SpeculativeConfig().to_dict()
        assert isinstance(d, dict)
        assert d["draft_length"] == 4


# ---------------------------------------------------------------------------
# DraftProposal
# ---------------------------------------------------------------------------

class TestDraftProposal:
    def test_length(self):
        p = _mk_proposal([1, 2, 3])
        assert p.length() == 3

    def test_empty(self):
        p = _mk_proposal([])
        assert p.length() == 0


def _mk_proposal(ids, metadata=None):
    from models.speculative_decoding import DraftProposal
    return DraftProposal(token_ids=list(ids), metadata=metadata or {})


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

class TestVerificationResult:
    def test_accepted_tokens(self):
        vr = VerificationResult(
            proposed_token_ids=[1, 2, 3],
            target_token_ids=[1, 2, 3],
            accept_mask=[True, True, True],
            accepted_count=3,
            rejected_count=0,
        )
        assert vr.accepted_tokens() == [1, 2, 3]
        assert vr.rejected_tokens() == []

    def test_rejected_tokens(self):
        vr = VerificationResult(
            proposed_token_ids=[1, 2, 3],
            target_token_ids=[1, 99, 3],
            accept_mask=[True, False, False],
            accepted_count=1,
            rejected_count=2,
            replacement_token_id=99,
        )
        assert vr.accepted_tokens() == [1]
        assert vr.rejected_tokens() == [2, 3]

    def test_all_committed_tokens_with_replacement(self):
        vr = VerificationResult(
            proposed_token_ids=[1, 2, 3],
            target_token_ids=[1, 99, 3],
            accept_mask=[True, False, False],
            accepted_count=1,
            rejected_count=2,
            replacement_token_id=99,
        )
        assert vr.all_committed_tokens() == [1, 99]

    def test_all_committed_tokens_all_accepted(self):
        vr = VerificationResult(
            proposed_token_ids=[1, 2, 3],
            target_token_ids=[1, 2, 3],
            accept_mask=[True, True, True],
            accepted_count=3,
            rejected_count=0,
        )
        assert vr.all_committed_tokens() == [1, 2, 3]


# ---------------------------------------------------------------------------
# SpeculativeStepResult
# ---------------------------------------------------------------------------

class TestSpeculativeStepResult:
    def test_creation(self):
        prop = _mk_proposal([1, 2])
        ver = VerificationResult(
            proposed_token_ids=[1, 2],
            target_token_ids=[1, 99],
            accept_mask=[True, False],
            accepted_count=1,
            rejected_count=1,
            replacement_token_id=99,
        )
        sr = SpeculativeStepResult(
            proposal=prop,
            verification=ver,
            committed_token_ids=[1, 99],
            accepted_count=1,
            cache_committed=False,
        )
        assert sr.accepted_count == 1
        assert sr.committed_token_ids == [1, 99]


# ---------------------------------------------------------------------------
# SpeculativeGenerationResult
# ---------------------------------------------------------------------------

class TestSpeculativeGenerationResult:
    def test_acceptance_rate_all_accepted(self):
        res = _mk_gen_result(
            steps=[
                _mk_step_result(3, [1, 2, 3]),
                _mk_step_result(2, [4, 5]),
            ],
        )
        assert res.acceptance_rate() == 1.0

    def test_acceptance_rate_half(self):
        res = _mk_gen_result(
            steps=[
                _mk_step_result(1, [1], proposed_len=2),
                _mk_step_result(2, [4, 5], proposed_len=3),
            ],
        )
        assert res.acceptance_rate() == 0.6

    def test_acceptance_rate_no_proposals(self):
        res = _mk_gen_result(steps=[])
        assert res.acceptance_rate() == 0.0

    def test_acceptance_rate_no_proposed_tokens(self):
        s = _mk_step_result(0, [])
        s.proposal = _mk_proposal([])
        res = _mk_gen_result(steps=[s])
        assert res.acceptance_rate() == 0.0

    def test_tokens_per_step(self):
        res = _mk_gen_result(
            steps=[
                _mk_step_result(1, [1, 99]),
                _mk_step_result(2, [4, 5]),
            ],
        )
        assert res.tokens_per_step() == 2.0

    def test_tokens_per_step_no_steps(self):
        res = _mk_gen_result(steps=[])
        assert res.tokens_per_step() == 0.0

    def test_to_dict(self):
        res = _mk_gen_result(steps=[_mk_step_result(1, [1])])
        d = res.to_dict()
        assert isinstance(d, dict)
        assert "acceptance_rate" in d
        assert "tokens_per_step" in d
        assert d["num_steps"] == 1


def _mk_step_result(accepted=1, committed=None, proposed_len=None):
    proposed_len = proposed_len if proposed_len is not None else max(accepted, 2)
    committed = committed or list(range(accepted))
    return SpeculativeStepResult(
        proposal=_mk_proposal(list(range(proposed_len))),
        verification=VerificationResult(
            proposed_token_ids=list(range(proposed_len)),
            target_token_ids=list(range(accepted)),
            accept_mask=[True] * accepted + [False] * (proposed_len - accepted),
            accepted_count=accepted,
            rejected_count=proposed_len - accepted,
        ),
        committed_token_ids=list(committed),
        accepted_count=accepted,
        cache_committed=False,
    )


def _mk_gen_result(steps=None, proposals=None):
    steps = steps or []
    return SpeculativeGenerationResult(
        prompt=None,
        prompt_ids=[1, 2, 3],
        generated_ids=[4, 5],
        all_ids=[1, 2, 3, 4, 5],
        text=None,
        steps=steps,
    )


# ---------------------------------------------------------------------------
# compute_accept_mask
# ---------------------------------------------------------------------------

class TestComputeAcceptMask:
    def test_all_match(self):
        mask = compute_accept_mask([1, 2, 3], [1, 2, 3])
        assert mask == [True, True, True]

    def test_first_mismatch(self):
        mask = compute_accept_mask([1, 2, 3], [1, 99, 3])
        assert mask == [True, False, False]

    def test_empty_proposal(self):
        mask = compute_accept_mask([], [1, 2, 3])
        assert mask == []

    def test_shorter_target(self):
        mask = compute_accept_mask([1, 2, 3, 4], [1, 2])
        assert mask == [True, True, False, False]

    def test_no_exact_match(self):
        mask = compute_accept_mask([1, 2, 3], [1, 2, 3], require_exact_match=False)
        assert mask == [True, True, True]

    def test_no_exact_match_partial(self):
        mask = compute_accept_mask([1, 2, 3], [1, 99, 3], require_exact_match=False)
        assert mask == [True, False, True]

    def test_all_mismatch(self):
        mask = compute_accept_mask([1, 2], [99, 100])
        assert mask == [False, False]


# ---------------------------------------------------------------------------
# accepted_prefix_length
# ---------------------------------------------------------------------------

class TestAcceptedPrefixLength:
    def test_all_true(self):
        assert accepted_prefix_length([True, True, True]) == 3

    def test_stops_at_first_false(self):
        assert accepted_prefix_length([True, False, True]) == 1

    def test_empty(self):
        assert accepted_prefix_length([]) == 0


# ---------------------------------------------------------------------------
# verify_draft_tokens
# ---------------------------------------------------------------------------

class TestVerifyDraftTokens:
    def test_all_accepted(self):
        vr = verify_draft_tokens([1, 2], [1, 2])
        assert vr.accepted_count == 2
        assert vr.rejected_count == 0
        assert vr.accept_mask == [True, True]
        assert vr.replacement_token_id is None

    def test_first_rejected(self):
        vr = verify_draft_tokens([1, 2], [99, 2])
        assert vr.accepted_count == 0
        assert vr.rejected_count == 2
        assert vr.replacement_token_id == 99

    def test_partial_accept(self):
        vr = verify_draft_tokens([1, 2, 3], [1, 99, 3])
        assert vr.accepted_count == 1
        assert vr.rejected_count == 2
        assert vr.replacement_token_id == 99

    def test_empty_proposal(self):
        vr = verify_draft_tokens([], [1, 2])
        assert vr.accepted_count == 0
        assert vr.rejected_count == 0
        assert vr.proposed_token_ids == []

    def test_custom_replacement(self):
        vr = verify_draft_tokens([1, 2], [1, 99], replacement_token_id=77)
        assert vr.replacement_token_id == 77

    def test_require_exact_match_false(self):
        vr = verify_draft_tokens([1, 2, 3], [1, 99, 3], require_exact_match=False)
        assert vr.accepted_count == 1
        assert vr.accept_mask[2] is True

    def test_proposed_longer_than_target(self):
        vr = verify_draft_tokens([1, 2, 3, 4], [1, 2])
        assert vr.accepted_count == 2
        assert vr.rejected_count == 2
        assert vr.accept_mask == [True, True, False, False]


# ---------------------------------------------------------------------------
# FixedDraftProposer
# ---------------------------------------------------------------------------

class TestFixedDraftProposer:
    def test_propose_fixed_ids(self):
        p = FixedDraftProposer([10, 20, 30])
        prop = p.propose([1, 2, 3], max_tokens=2)
        assert prop.token_ids == [10, 20]
        assert prop.length() == 2

    def test_propose_exact_max(self):
        p = FixedDraftProposer([10, 20, 30])
        prop = p.propose([1], max_tokens=3)
        assert prop.token_ids == [10, 20, 30]

    def test_propose_zero_max(self):
        p = FixedDraftProposer([10, 20])
        prop = p.propose([1], max_tokens=0)
        assert prop.token_ids == []

    def test_propose_ignores_context(self):
        p = FixedDraftProposer([5, 6])
        prop = p.propose([100, 200], max_tokens=2)
        assert prop.token_ids == [5, 6]

    def test_metadata(self):
        p = FixedDraftProposer([7, 8])
        prop = p.propose([], max_tokens=1)
        assert prop.metadata["draft_mode"] == "fixed"


# ---------------------------------------------------------------------------
# RandomDraftProposer
# ---------------------------------------------------------------------------

class TestRandomDraftProposer:
    def test_propose_returns_correct_length(self):
        p = RandomDraftProposer(128, seed=42)
        prop = p.propose([1, 2], max_tokens=5)
        assert prop.length() == 5

    def test_propose_zero_max(self):
        p = RandomDraftProposer(128)
        prop = p.propose([1], max_tokens=0)
        assert prop.token_ids == []

    def test_deterministic_seed(self):
        p = RandomDraftProposer(128, seed=0)
        prop_a = p.propose([1], max_tokens=3, seed=0)
        prop_b = p.propose([1], max_tokens=3, seed=0)
        assert prop_a.token_ids == prop_b.token_ids

    def test_metadata(self):
        p = RandomDraftProposer(128, seed=1)
        prop = p.propose([], max_tokens=2)
        assert prop.metadata["draft_mode"] == "random"


# ---------------------------------------------------------------------------
# GreedySelfDraftProposer
# ---------------------------------------------------------------------------

class TestGreedySelfDraftProposer:
    def test_propose_returns_tokens(self):
        cls = _get_greedy_self_draft_proposer()
        pipeline = _DummyPipeline()
        proposer = cls(pipeline)
        prop = proposer.propose([1, 2], max_tokens=3)
        assert prop.length() == 3
        assert all(isinstance(t, int) for t in prop.token_ids)

    def test_zero_max(self):
        cls = _get_greedy_self_draft_proposer()
        proposer = cls(_DummyPipeline())
        prop = proposer.propose([1], max_tokens=0)
        assert prop.token_ids == []

    def test_metadata(self):
        cls = _get_greedy_self_draft_proposer()
        proposer = cls(_DummyPipeline())
        prop = proposer.propose([1], max_tokens=2)
        assert prop.metadata["draft_mode"] == "self"


# ---------------------------------------------------------------------------
# PipelineTargetVerifier
# ---------------------------------------------------------------------------

class TestPipelineTargetVerifier:
    def test_verify_all_match(self):
        pipeline = _DummyPipeline()
        verifier = PipelineTargetVerifier(pipeline)
        vr = verifier.verify([1, 2], [3], config=SpeculativeConfig())
        assert vr.accepted_count == 1
        assert vr.replacement_token_id is None

    def test_verify_no_proposal(self):
        verifier = PipelineTargetVerifier(_DummyPipeline())
        vr = verifier.verify([1, 2], [], config=SpeculativeConfig())
        assert vr.accepted_count == 0

    def test_verify_custom_config(self):
        verifier = PipelineTargetVerifier(_DummyPipeline())
        cfg = SpeculativeConfig(draft_length=2, max_new_tokens=4)
        vr = verifier.verify([1], [42], config=cfg)
        assert isinstance(vr, VerificationResult)


# ---------------------------------------------------------------------------
# SpeculativeGenerator
# ---------------------------------------------------------------------------

class TestSpeculativeGenerator:
    def test_generate_ids_fixed_proposer(self):
        pipeline = _DummyPipeline()
        proposer = FixedDraftProposer([42, 43, 44, 45])
        gen = SpeculativeGenerator(pipeline, draft_proposer=proposer)
        result = gen.generate_ids([1, 2], speculative_config=SpeculativeConfig(max_new_tokens=4))
        assert len(result.generated_ids) == 4
        assert isinstance(result, SpeculativeGenerationResult)
        assert result.metadata["acceptance_rate"] >= 0

    def test_generate_ids_random_proposer(self):
        pipeline = _DummyPipeline()
        proposer = RandomDraftProposer(128, seed=0)
        gen = SpeculativeGenerator(pipeline, draft_proposer=proposer)
        result = gen.generate_ids([1], speculative_config=SpeculativeConfig(max_new_tokens=2))
        assert len(result.generated_ids) > 0
        assert result.metadata["num_steps"] >= 1

    def test_generate_text(self):
        pipeline = _DummyPipeline()
        proposer = FixedDraftProposer([42])
        gen = SpeculativeGenerator(pipeline, draft_proposer=proposer)
        result = gen.generate_text("hello")
        assert result.prompt == "hello"
        assert len(result.generated_ids) > 0

    def test_max_new_tokens_respected(self):
        pipeline = _DummyPipeline()
        proposer = FixedDraftProposer(list(range(100)))
        cfg = SpeculativeConfig(draft_length=8, max_new_tokens=5)
        gen = SpeculativeGenerator(pipeline, draft_proposer=proposer, config=cfg)
        result = gen.generate_ids([1, 2, 3])
        assert len(result.generated_ids) == 5

    def test_empty_input_raises(self):
        pipeline = _DummyPipeline()
        gen = SpeculativeGenerator(pipeline)
        with pytest.raises(ValueError, match="input_ids"):
            gen.generate_ids([])

    def test_zero_max_new_tokens(self):
        pipeline = _DummyPipeline()
        proposer = FixedDraftProposer([42])
        cfg = SpeculativeConfig(draft_length=4, max_new_tokens=1)
        gen = SpeculativeGenerator(pipeline, draft_proposer=proposer, config=cfg)
        result = gen.generate_ids([1])
        assert len(result.generated_ids) == 1

    def test_steps_metadata(self):
        pipeline = _DummyPipeline()
        proposer = FixedDraftProposer([42])
        cfg = SpeculativeConfig(draft_length=4, max_new_tokens=3)
        gen = SpeculativeGenerator(pipeline, draft_proposer=proposer, config=cfg)
        result = gen.generate_ids([1, 2])
        assert len(result.steps) > 0
        for step in result.steps:
            assert step.metadata["step_index"] >= 0

    def test_generate_ids_does_not_mutate_config_seed(self):
        pipeline = _DummyPipeline()
        proposer = RandomDraftProposer(128, seed=7)
        cfg = SpeculativeConfig(draft_length=2, max_new_tokens=4, seed=11)
        gen = SpeculativeGenerator(pipeline, draft_proposer=proposer, config=cfg)
        _ = gen.generate_ids([1, 2], speculative_config=cfg)
        assert cfg.seed == 11

    def test_metadata_avg_tokens_per_step_matches_result(self):
        pipeline = _DummyPipeline()
        proposer = FixedDraftProposer([42, 43, 44, 45])
        gen = SpeculativeGenerator(pipeline, draft_proposer=proposer)
        result = gen.generate_ids([1, 2], speculative_config=SpeculativeConfig(max_new_tokens=4))
        assert result.metadata["avg_tokens_per_step"] == result.tokens_per_step()
