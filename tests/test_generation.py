import numpy as np
import pytest

from models import CharTokenizer, GenerationConfig, LlamaLikeConfig, create_synthetic_generation_model


def _tiny_config():
    return LlamaLikeConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=1,
        max_position_embeddings=16,
        vocab_size=128,
        model_type="toy_generation_test",
    ).validate()


def test_generation_config_validation():
    assert GenerationConfig(max_new_tokens=4, temperature=0.8, top_k=4, top_p=0.9, repetition_penalty=1.1).validate().max_new_tokens == 4
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        GenerationConfig(max_new_tokens=0).validate()


def test_create_synthetic_generation_model_returns_model():
    model = create_synthetic_generation_model(config=_tiny_config(), vocab_size=80, seed=3)
    assert model.vocab_size >= 80


def test_init_state_creates_cache_and_position_zero():
    model = create_synthetic_generation_model(config=_tiny_config(), vocab_size=80, seed=4)
    state = model.init_state()
    assert state.position == 0
    assert state.cache[0].shape[1] >= model.config.max_position_embeddings


def test_embed_token_ids_returns_documented_shape():
    model = create_synthetic_generation_model(config=_tiny_config(), vocab_size=80, seed=5)
    embedded = np.asarray(model.embed_token_ids([1]))
    assert embedded.shape == (1, 1, model.config.hidden_size)


def test_logits_from_hidden_returns_vocab_sized_logits():
    model = create_synthetic_generation_model(config=_tiny_config(), vocab_size=80, seed=6)
    hidden = model.embed_token_ids([1])
    logits = np.asarray(model.logits_from_hidden(hidden))
    assert logits.shape == (model.vocab_size,)


def test_decode_step_updates_state_position():
    model = create_synthetic_generation_model(config=_tiny_config(), vocab_size=80, seed=7)
    state = model.init_state()
    logits, state = model.decode_step(1, state)
    assert np.asarray(logits).shape == (model.vocab_size,)
    assert state.position == 1
    assert state.generated_ids == [1]


def test_generate_token_ids_returns_prompt_plus_new_tokens():
    model = create_synthetic_generation_model(config=_tiny_config(), vocab_size=80, seed=8)
    output_ids = model.generate_token_ids([1, 2], GenerationConfig(max_new_tokens=4, seed=9, top_k=5))
    assert len(output_ids) == 6
    assert output_ids[:2] == [1, 2]


def test_generate_text_works_with_char_tokenizer():
    tokenizer = CharTokenizer()
    model = create_synthetic_generation_model(config=_tiny_config(), tokenizer=tokenizer, seed=10)
    text = model.generate_text("Hello", GenerationConfig(max_new_tokens=4, seed=2, top_k=5))
    assert isinstance(text, str)
    assert len(text) >= len("Hello")


def test_b_greater_than_one_raises():
    model = create_synthetic_generation_model(config=_tiny_config(), vocab_size=80, seed=11)
    with pytest.raises(NotImplementedError, match="B=1"):
        model.init_state(B=2)


def test_seeded_sampling_is_repeatable():
    tokenizer = CharTokenizer()
    model_a = create_synthetic_generation_model(config=_tiny_config(), tokenizer=tokenizer, seed=12)
    model_b = create_synthetic_generation_model(config=_tiny_config(), tokenizer=tokenizer, seed=12)
    gen_cfg = GenerationConfig(max_new_tokens=4, seed=21, top_k=6, temperature=0.9)
    out_a = model_a.generate_token_ids(tokenizer.encode("Hi"), gen_cfg)
    out_b = model_b.generate_token_ids(tokenizer.encode("Hi"), gen_cfg)
    assert out_a == out_b
