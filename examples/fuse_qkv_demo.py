from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    try:
        import mlx.core as mx
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError("MLX is required for fuse_qkv_demo.py") from exc

    from models import fuse_qkv_weights, split_fused_qkv_weight, tiny_gqa_debug_config

    config = tiny_gqa_debug_config()
    q_w = mx.random.normal((config.q_output_dim(), config.hidden_size)).astype(mx.float16)
    k_w = mx.random.normal((config.kv_output_dim(), config.hidden_size)).astype(mx.float16)
    v_w = mx.random.normal((config.kv_output_dim(), config.hidden_size)).astype(mx.float16)

    fused = fuse_qkv_weights(q_w, k_w, v_w)
    q_split, k_split, v_split = split_fused_qkv_weight(fused, config)
    mx.eval(fused, q_split, k_split, v_split)

    print(f"q_shape={q_w.shape}")
    print(f"k_shape={k_w.shape}")
    print(f"v_shape={v_w.shape}")
    print(f"fused_shape={fused.shape}")
    assert q_split.shape == q_w.shape
    assert k_split.shape == k_w.shape
    assert v_split.shape == v_w.shape
    print("split shapes match original q/k/v shapes")


if __name__ == "__main__":
    main()
