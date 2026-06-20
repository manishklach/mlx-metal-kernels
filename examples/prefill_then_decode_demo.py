from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import GenerationConfig, TinyGenerationPipeline, TinyGenerationPipelineConfig


def main() -> int:
    config = TinyGenerationPipelineConfig(use_prefill=True, backend_preset="reference").validate()
    pipeline = TinyGenerationPipeline(config=config, generation_config=GenerationConfig(max_new_tokens=4, eos_token_id=-1, backend_preset="reference"))
    prompt = "Hello world"
    prompt_ids = pipeline.encode(prompt)
    prefill = pipeline.prefill_prompt(prompt_ids)
    result = pipeline.generate(prompt, max_new_tokens=4, greedy=True)
    cache_layers = len(prefill.cache.layer_caches) if hasattr(prefill.cache, "layer_caches") else 0
    print("Uses synthetic random weights; output is not meaningful language generation.")
    print("prompt:", prompt)
    print("prompt length:", len(prompt_ids))
    print("prefill backend:", config.backend_preset)
    print("decode backend:", config.backend_preset)
    print("cache layers:", cache_layers)
    print("next decode position:", prefill.next_position)
    print("generated ids:", result.generated_ids)
    print("decoded text:", result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
