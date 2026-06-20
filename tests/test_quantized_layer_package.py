import pytest

from models import (
    CheckpointAdapter,
    CheckpointQuantizer,
    InMemoryTensorStore,
    QuantizationConfig,
    tiny_gqa_debug_config,
)

np = pytest.importorskip("numpy")


def _tensors(config, seed=30):
    rng = np.random.default_rng(seed)
    stem = "model.layers.0"
    return {
        f"{stem}.self_attn.q_proj.weight": rng.normal(size=(config.q_output_dim(), config.hidden_size)).astype(np.float16),
        f"{stem}.self_attn.k_proj.weight": rng.normal(size=(config.kv_output_dim(), config.hidden_size)).astype(np.float16),
        f"{stem}.self_attn.v_proj.weight": rng.normal(size=(config.kv_output_dim(), config.hidden_size)).astype(np.float16),
        f"{stem}.self_attn.o_proj.weight": rng.normal(size=(config.hidden_size, config.q_output_dim())).astype(np.float16),
        f"{stem}.mlp.gate_proj.weight": rng.normal(size=(config.intermediate_size, config.hidden_size)).astype(np.float16),
        f"{stem}.mlp.up_proj.weight": rng.normal(size=(config.intermediate_size, config.hidden_size)).astype(np.float16),
        f"{stem}.mlp.down_proj.weight": rng.normal(size=(config.hidden_size, config.intermediate_size)).astype(np.float16),
        f"{stem}.input_layernorm.weight": np.ones((config.hidden_size,), dtype=np.float16),
        f"{stem}.post_attention_layernorm.weight": np.ones((config.hidden_size,), dtype=np.float16),
    }


def test_quantized_linear_package_shapes_and_to_kernel_weights():
    cfg = tiny_gqa_debug_config()
    adapter = CheckpointAdapter(cfg, InMemoryTensorStore(_tensors(cfg)))
    package = CheckpointQuantizer(adapter, QuantizationConfig(bits=4, group_size=32)).quantize_layer(0)
    qkv_shapes = package.qkv.shapes()
    assert qkv_shapes["original_shape"] == (cfg.q_output_dim() + 2 * cfg.kv_output_dim(), cfg.hidden_size)
    kernel_weights = package.to_kernel_weights(cfg)
    assert kernel_weights.bits == 4
    assert kernel_weights.group_size == 32
    assert kernel_weights.shapes()["qkv_w"] == package.qkv.shapes()["weight"]


def test_quantized_layer_package_shapes():
    cfg = tiny_gqa_debug_config()
    adapter = CheckpointAdapter(cfg, InMemoryTensorStore(_tensors(cfg, seed=31)))
    package = CheckpointQuantizer(adapter, QuantizationConfig(bits=8, group_size=32)).quantize_layer(0)
    shapes = package.shapes()
    assert shapes["input_layernorm_weight"] == (cfg.hidden_size,)
    assert shapes["qkv"]["original_shape"] == (cfg.q_output_dim() + 2 * cfg.kv_output_dim(), cfg.hidden_size)


def test_quantized_layer_package_kernel_weights_validate():
    cfg = tiny_gqa_debug_config()
    adapter = CheckpointAdapter(cfg, InMemoryTensorStore(_tensors(cfg, seed=32)))
    package = CheckpointQuantizer(adapter, QuantizationConfig(bits=4, group_size=32)).quantize_layer(0)
    kernel_weights = package.to_kernel_weights(cfg)
    validated = kernel_weights.validate(cfg)
    assert validated.qkv_w.shape[0] == cfg.q_output_dim() + 2 * cfg.kv_output_dim()


def test_quantized_layer_package_reference_decode_step_if_mlx_available():
    mx = pytest.importorskip("mlx.core")
    from models import build_rope_tables
    from ops.llama_layer_ops import init_llama_layer_cache, reference_llama_layer_decode_step

    cfg = tiny_gqa_debug_config()
    rng = np.random.default_rng(33)
    stem = "model.layers.0"
    tensors = {
        f"{stem}.self_attn.q_proj.weight": mx.array(rng.normal(size=(cfg.q_output_dim(), cfg.hidden_size)).astype(np.float16)),
        f"{stem}.self_attn.k_proj.weight": mx.array(rng.normal(size=(cfg.kv_output_dim(), cfg.hidden_size)).astype(np.float16)),
        f"{stem}.self_attn.v_proj.weight": mx.array(rng.normal(size=(cfg.kv_output_dim(), cfg.hidden_size)).astype(np.float16)),
        f"{stem}.self_attn.o_proj.weight": mx.array(rng.normal(size=(cfg.hidden_size, cfg.q_output_dim())).astype(np.float16)),
        f"{stem}.mlp.gate_proj.weight": mx.array(rng.normal(size=(cfg.intermediate_size, cfg.hidden_size)).astype(np.float16)),
        f"{stem}.mlp.up_proj.weight": mx.array(rng.normal(size=(cfg.intermediate_size, cfg.hidden_size)).astype(np.float16)),
        f"{stem}.mlp.down_proj.weight": mx.array(rng.normal(size=(cfg.hidden_size, cfg.intermediate_size)).astype(np.float16)),
        f"{stem}.input_layernorm.weight": mx.ones((cfg.hidden_size,), dtype=mx.float16),
        f"{stem}.post_attention_layernorm.weight": mx.ones((cfg.hidden_size,), dtype=mx.float16),
    }
    adapter = CheckpointAdapter(cfg, InMemoryTensorStore(tensors))
    package = CheckpointQuantizer(adapter, QuantizationConfig(bits=4, group_size=32)).quantize_layer(0)
    weights = package.to_kernel_weights(cfg)
    cos, sin = build_rope_tables(cfg, seq_len=cfg.max_position_embeddings + 1, dtype=mx.float32)
    cache = init_llama_layer_cache(cfg, 1, cfg.max_position_embeddings, dtype=mx.float16)
    x = mx.zeros((1, 1, cfg.hidden_size), dtype=mx.float16)
    out, _ = reference_llama_layer_decode_step(x, weights, cache, cos, sin, 0, cfg)
    mx.eval(out)
    assert out.shape == (1, 1, cfg.hidden_size)
