import mlx.core as mx
import pytest

from models.llama_config import LlamaLikeConfig, build_rope_tables, tiny_debug_config


def test_tiny_debug_config_validates():
    cfg = tiny_debug_config()
    assert cfg.hidden_size == 64
    assert cfg.validate() is cfg


def test_hidden_size_mismatch_raises():
    cfg = LlamaLikeConfig(
        hidden_size=96,
        intermediate_size=192,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=64,
    )
    with pytest.raises(ValueError, match="hidden_size must equal"):
        cfg.validate()


def test_attention_groups_and_qkv_output_dim():
    cfg = tiny_debug_config()
    assert cfg.attention_groups() == 1
    assert cfg.qkv_output_dim() == 3 * cfg.hidden_size


def test_to_dict_from_dict_roundtrip():
    cfg = tiny_debug_config()
    restored = LlamaLikeConfig.from_dict(cfg.to_dict())
    assert restored.to_dict() == cfg.to_dict()


def test_build_rope_tables_shape():
    cfg = tiny_debug_config()
    cos, sin = build_rope_tables(cfg, seq_len=17, dtype=mx.float32)
    assert cos.shape == (17, cfg.head_dim // 2)
    assert sin.shape == (17, cfg.head_dim // 2)
