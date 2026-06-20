from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import (
    CharTokenizer,
    QuantizedCheckpointPackage,
    SmokeTestConfig,
    TinyGenerationPipelineConfig,
    run_local_smoke_test,
)


def _metadata_only_package() -> QuantizedCheckpointPackage:
    config = TinyGenerationPipelineConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=2,
        max_position_embeddings=32,
        vocab_size=CharTokenizer().vocab_size,
        backend_preset="reference",
    ).to_llama_config()
    return QuantizedCheckpointPackage(
        model_type=config.model_type,
        config=config.to_dict(),
        quantization={"bits": 4, "group_size": 32},
        layers=[],
        metadata={
            "demo": "metadata-only package",
            "note": "Synthetic fallback uses random weights and does not produce meaningful language.",
        },
    )


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    package_path = out_dir / "smoke_test_metadata_package.json"
    _metadata_only_package().save_json(package_path)

    dry_run_report = run_local_smoke_test(
        SmokeTestConfig(
            package_path=str(package_path),
            dry_run=True,
            synthetic_fallback=True,
            validate_alignment=True,
        )
    )
    print("Dry-run metadata-only report:")
    print(dry_run_report.pretty_print())
    print()

    fallback_report = run_local_smoke_test(
        SmokeTestConfig(
            package_path=str(package_path),
            dry_run=False,
            synthetic_fallback=True,
            prompt="Hello",
            max_new_tokens=4,
            backend_preset="reference",
            validate_alignment=True,
        )
    )
    print("Synthetic fallback generation report:")
    print(fallback_report.pretty_print())
    print()
    print("This is a local smoke-test scaffold. Synthetic fallback uses random weights and does not produce meaningful language.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
