import numpy as np
import pytest

from models import CharTokenizer, GenerationConfig, LlamaLikeConfig, ToyGenerationState, create_synthetic_stack_generation_model


def _tiny_stack_config():
    return LlamaLikeConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=16,
        vocab_size=64,
        model_type="stack_generation_test",
    ).validate()


def test_stack_generation_model_init_state_has_one_cache_per_layer():
    model = create_synthetic_stack_generation_model(config=_tiny_stack_config(), vocab_size=64, seed=1)
    state = model.init_state()
    assert state.position == 0
    assert state.cache.num_layers() == model.config.num_hidden_layers


def test_stack_generation_decode_step_increments_position():
    model = create_synthetic_stack_generation_model(config=_tiny_stack_config(), vocab_size=64, seed=2)
    state = model.init_state()
    logits, state = model.decode_step(1, state)
    assert np.asarray(logits).shape == (model.vocab_size,)
    assert state.position == 1


def test_stack_generation_generate_token_ids_length():
    model = create_synthetic_stack_generation_model(config=_tiny_stack_config(), vocab_size=64, seed=3)
    output_ids = model.generate_token_ids([1, 2], GenerationConfig(max_new_tokens=4, top_k=5, seed=4))
    assert len(output_ids) == 6


def test_stack_generation_generate_text_works():
    tokenizer = CharTokenizer()
    model = create_synthetic_stack_generation_model(config=_tiny_stack_config(), tokenizer=tokenizer, seed=5)
    text = model.generate_text("Hello", GenerationConfig(max_new_tokens=4, top_k=5, seed=6))
    assert isinstance(text, str)
    assert len(text) >= len("Hello")


def test_stack_generation_greedy_is_deterministic():
    tokenizer = CharTokenizer()
    cfg = _tiny_stack_config()
    model_a = create_synthetic_stack_generation_model(config=cfg, tokenizer=tokenizer, seed=7)
    model_b = create_synthetic_stack_generation_model(config=cfg, tokenizer=tokenizer, seed=7)
    gen_cfg = GenerationConfig(max_new_tokens=4)
    out_a = model_a.generate_token_ids(tokenizer.encode("Hi"), gen_cfg)
    out_b = model_b.generate_token_ids(tokenizer.encode("Hi"), gen_cfg)
    assert out_a == out_b


def test_stack_generation_b_greater_than_one_raises():
    model = create_synthetic_stack_generation_model(config=_tiny_stack_config(), vocab_size=64, seed=8)
    with pytest.raises(NotImplementedError, match="B=1"):
        model.init_state(B=2)


def test_stack_generation_eos_stops_generation():
    model = create_synthetic_stack_generation_model(config=_tiny_stack_config(), vocab_size=8, seed=9)
    eos_token_id = 3

    def fake_decode_step(token_id, state: ToyGenerationState, generation_config=None):
        logits = np.full((model.vocab_size,), -10.0, dtype=np.float32)
        logits[eos_token_id] = 10.0
        state.position += 1
        state.generated_ids.append(int(token_id))
        return logits, state

    model.decode_step = fake_decode_step
    output_ids = model.generate_token_ids([1], GenerationConfig(max_new_tokens=6, eos_token_id=eos_token_id))
    assert output_ids[-1] == eos_token_id
    assert len(output_ids) == 2
