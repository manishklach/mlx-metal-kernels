from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import (
    CheckpointManifest,
    build_fused_qkv_manifest_entries,
    create_fused_qkv_manifest,
    llama_quantized_layer_specs,
    missing_required_tensors,
    tiny_gqa_debug_config,
    validate_llama_checkpoint_shapes,
)


def main():
    base_config = tiny_gqa_debug_config()
    config = base_config.from_dict({**base_config.to_dict(), "num_hidden_layers": 1})
    manifest_path = Path(__file__).with_name("mock_llama_manifest.json")
    manifest = CheckpointManifest.load_json(manifest_path)
    report = validate_llama_checkpoint_shapes(manifest, config, require_all_layers=False)
    fused_info = build_fused_qkv_manifest_entries(manifest, config, layer_idx=0)
    packaged = llama_quantized_layer_specs(config, layer_idx=0, bits=4, group_size=32)
    derived = create_fused_qkv_manifest(manifest, config)

    print(f"model_type={manifest.model_type}")
    print(f"num_tensors={len(manifest.tensors)}")
    print(f"missing_tensors={missing_required_tensors(manifest, config)}")
    print(f"validation_ok={report.ok}")
    print(f"validation_issues={[issue.message for issue in report.issues]}")
    print(f"fused_qkv_name={fused_info.name}")
    print(f"fused_qkv_shape={fused_info.shape}")
    print(f"derived_manifest_tensors={len(derived.tensors)}")
    print(f"q_proj_quant_spec={packaged['q_proj']}")


if __name__ == "__main__":
    main()
