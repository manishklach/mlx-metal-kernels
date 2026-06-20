from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import CheckpointAdapter, InMemoryTensorStore, tiny_gqa_debug_config


def _make_array(shape, dtype="float16"):
    try:
        import mlx.core as mx

        return mx.random.normal(shape).astype(getattr(mx, dtype))
    except Exception:  # noqa: BLE001
        import numpy as np

        return np.zeros(shape, dtype=dtype)


def _mock_tensors(config):
    tensors = {}
    for layer_idx in range(config.num_hidden_layers):
        stem = f"model.layers.{layer_idx}"
        tensors.update(
            {
                f"{stem}.self_attn.q_proj.weight": _make_array((config.q_output_dim(), config.hidden_size)),
                f"{stem}.self_attn.k_proj.weight": _make_array((config.kv_output_dim(), config.hidden_size)),
                f"{stem}.self_attn.v_proj.weight": _make_array((config.kv_output_dim(), config.hidden_size)),
                f"{stem}.self_attn.o_proj.weight": _make_array((config.hidden_size, config.q_output_dim())),
                f"{stem}.mlp.gate_proj.weight": _make_array((config.intermediate_size, config.hidden_size)),
                f"{stem}.mlp.up_proj.weight": _make_array((config.intermediate_size, config.hidden_size)),
                f"{stem}.mlp.down_proj.weight": _make_array((config.hidden_size, config.intermediate_size)),
                f"{stem}.input_layernorm.weight": _make_array((config.hidden_size,)),
                f"{stem}.post_attention_layernorm.weight": _make_array((config.hidden_size,)),
            }
        )
    return tensors


def main():
    config = tiny_gqa_debug_config()
    store = InMemoryTensorStore(_mock_tensors(config))
    adapter = CheckpointAdapter(config, store)
    report = adapter.validate()
    print(f"description={adapter.describe()}")
    print(f"validation_ok={report.ok}")
    print(f"layer_0_shapes={adapter.layer_shapes(0)}")
    print(f"fused_qkv_shape={adapter.get_fused_qkv_shape(0)}")
    print(f"quantized_specs={list(adapter.quantized_specs_for_layer(0))}")


if __name__ == "__main__":
    main()
