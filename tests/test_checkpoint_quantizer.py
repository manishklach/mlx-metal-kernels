import pytest

from models import CheckpointAdapter, CheckpointQuantizer, InMemoryTensorStore, QuantizationConfig, tiny_debug_config, tiny_gqa_debug_config

np = pytest.importorskip("numpy")


def _all_tensors(config, seed=0):
    rng = np.random.default_rng(seed)
    tensors = {}
    for layer_idx in range(config.num_hidden_layers):
        stem = f"model.layers.{layer_idx}"
        tensors.update(
            {
                f"{stem}.self_attn.q_proj.weight": rng.normal(size=(config.q_output_dim(), config.hidden_size)).astype(np.float16),
                f"{stem}.self_attn.k_proj.weight": rng.normal(size=(config.kv_output_dim(), config.hidden_size)).astype(np.float16),
                f"{stem}.self_attn.v_proj.weight": rng.normal(size=(config.kv_output_dim(), config.hidden_size)).astype(np.float16),
                f"{stem}.self_attn.o_proj.weight": rng.normal(size=(config.hidden_size, config.q_output_dim())).astype(np.float16),
                f"{stem}.mlp.gate_proj.weight": rng.normal(size=(config.intermediate_size, config.hidden_size)).astype(np.float16),
                f"{stem}.mlp.up_proj.weight": rng.normal(size=(config.intermediate_size, config.hidden_size)).astype(np.float16),
                f"{stem}.mlp.down_proj.weight": rng.normal(size=(config.hidden_size, config.intermediate_size)).astype(np.float16),
                f"{stem}.input_layernorm.weight": rng.normal(size=(config.hidden_size,)).astype(np.float16),
                f"{stem}.post_attention_layernorm.weight": rng.normal(size=(config.hidden_size,)).astype(np.float16),
            }
        )
    return tensors


def test_checkpoint_quantizer_quantizes_gqa_layer_bits_4():
    cfg = tiny_gqa_debug_config()
    adapter = CheckpointAdapter(cfg, InMemoryTensorStore(_all_tensors(cfg, seed=20)))
    quantizer = CheckpointQuantizer(adapter, QuantizationConfig(bits=4, group_size=32))
    package = quantizer.quantize_layer(0)
    report = quantizer.report()
    assert package.qkv.original_shape == (cfg.q_output_dim() + 2 * cfg.kv_output_dim(), cfg.hidden_size)
    assert package.o_proj.bits == 4
    assert package.gate_proj.group_size == 32
    assert "model.layers.0.self_attn.qkv_proj.fused_weight" in report.quantized_tensors
    assert report.ok is True


def test_checkpoint_quantizer_bits_8_works():
    cfg = tiny_debug_config()
    adapter = CheckpointAdapter(cfg, InMemoryTensorStore(_all_tensors(cfg, seed=21)))
    quantizer = CheckpointQuantizer(adapter, QuantizationConfig(bits=8, group_size=32))
    package = quantizer.quantize_layer(0)
    assert package.qkv.bits == 8
    assert package.qkv.shapes()["weight"] == (cfg.fused_qkv_output_dim(), cfg.hidden_size)


def test_checkpoint_quantizer_missing_tensor_raises_and_records_error():
    cfg = tiny_debug_config()
    tensors = _all_tensors(cfg, seed=22)
    del tensors["model.layers.0.mlp.up_proj.weight"]
    adapter = CheckpointAdapter(cfg, InMemoryTensorStore(tensors))
    quantizer = CheckpointQuantizer(adapter, QuantizationConfig(bits=4, group_size=32))
    with pytest.raises(KeyError, match="up_proj"):
        quantizer.quantize_layer(0)
    assert quantizer.report().errors
