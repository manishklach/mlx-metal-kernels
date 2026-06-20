import mlx.core as mx

from models.llama_config import LlamaLikeConfig, tiny_debug_config, tiny_gqa_debug_config
from models.model_adapter import KernelBackendConfig, LlamaLikeKernelAdapter


def test_adapter_describe_returns_useful_dict():
    adapter = LlamaLikeKernelAdapter(tiny_debug_config())
    desc = adapter.describe()
    assert desc["config"]["hidden_size"] == 64
    assert desc["cache_layout"] == "contiguous"
    assert "fused_qkv_shape" in desc
    assert desc["mlp_shapes"]["gate_proj"] == (128, 64)


def test_init_cache_contiguous_shapes():
    cfg = tiny_debug_config()
    adapter = LlamaLikeKernelAdapter(cfg, cache_layout="contiguous")
    states = adapter.init_cache(2, dtype=mx.float16)
    assert len(states) == cfg.num_hidden_layers
    assert states[0].K_cache.shape == (2, cfg.max_position_embeddings, cfg.num_attention_heads, cfg.head_dim)
    assert states[0].V_cache.shape == (2, cfg.max_position_embeddings, cfg.num_attention_heads, cfg.head_dim)


def test_init_cache_paged_shapes():
    cfg = tiny_debug_config()
    adapter = LlamaLikeKernelAdapter(cfg, cache_layout="paged")
    states = adapter.init_cache(2, dtype=mx.float16)
    pages = 2 * ((cfg.max_position_embeddings + 15) // 16)
    assert len(states) == cfg.num_hidden_layers
    assert states[0].K_pages.shape == (pages, 16, cfg.num_key_value_heads, cfg.head_dim)
    assert states[0].block_table.shape[0] == 2


def test_validate_supported_allows_gqa():
    adapter = LlamaLikeKernelAdapter(tiny_gqa_debug_config())
    adapter.validate_supported()


def test_choose_backend_returns_configured_or_fallback_default():
    cfg = tiny_debug_config()
    adapter = LlamaLikeKernelAdapter(cfg, KernelBackendConfig(matvec_backend="metal_parallel", use_autotune=False))
    assert adapter.choose_backend("q4_matvec_decode", {"B": 1, "K": 64, "N": 64, "group_size": 32}, "float16") == "metal_parallel"

    tuned = LlamaLikeKernelAdapter(cfg, KernelBackendConfig(matvec_backend="metal_parallel", use_autotune=True))
    backend = tuned.choose_backend("q4_matvec_decode", {"B": 1, "K": 64, "N": 64, "group_size": 32}, "float16")
    assert backend == "metal_parallel"


def test_run_quantized_mlp_block_shape():
    cfg = tiny_debug_config()
    adapter = LlamaLikeKernelAdapter(cfg, KernelBackendConfig(matvec_backend="metal_parallel", use_autotune=False))
    weights = adapter.make_demo_quantized_weights(bits=4, group_size=32)
    x = mx.random.normal((1, 1, cfg.hidden_size)).astype(mx.float16)
    residual = mx.random.normal((1, 1, cfg.hidden_size)).astype(mx.float16)
    out = adapter.run_quantized_mlp_block(
        x,
        residual,
        weights["ffn_norm_weight"].astype(mx.float16),
        weights["gate_w"],
        weights["gate_scales"],
        weights["up_w"],
        weights["up_scales"],
        weights["down_w"],
        weights["down_scales"],
        bits=4,
        group_size=32,
    )
    mx.eval(out)
    assert out.shape == x.shape


def test_init_cache_gqa_uses_kv_head_count():
    cfg = tiny_gqa_debug_config()
    adapter = LlamaLikeKernelAdapter(cfg, cache_layout="contiguous")
    states = adapter.init_cache(1, dtype=mx.float16)
    assert states[0].K_cache.shape == (1, cfg.max_position_embeddings, cfg.num_key_value_heads, cfg.head_dim)
