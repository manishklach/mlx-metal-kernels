from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import (
    QuantizedCheckpointPackage,
    TinyGenerationPipeline,
    TinyGenerationPipelineConfig,
    create_pipeline_from_quantized_package,
)


def _synthetic_metadata_only_package() -> QuantizedCheckpointPackage:
    config = TinyGenerationPipelineConfig().to_llama_config().to_dict()
    return QuantizedCheckpointPackage(
        config=config,
        quantization={"bits": 4, "group_size": 32},
        layers=[],
        metadata={
            "demo": "metadata-only package",
            "explanation": "Current package format stores tensor metadata but not tensor payloads.",
        },
    )


def main() -> int:
    package = _synthetic_metadata_only_package()
    print("Loaded metadata-only package summary:")
    print(json.dumps(package.summary(), indent=2, sort_keys=True))
    try:
        pipeline = create_pipeline_from_quantized_package(package)
    except NotImplementedError as exc:
        print(f"Package fallback: {exc}")
        print("Falling back to a synthetic tiny generation pipeline for the end-to-end demo.")
        pipeline = TinyGenerationPipeline(config=TinyGenerationPipelineConfig())
    result = pipeline.generate("Hello", max_new_tokens=4, greedy=True)
    print("This demo uses synthetic random weights. Output text is not meaningful language generation.")
    print("generated ids:", result.generated_ids)
    print("decoded text:", result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
