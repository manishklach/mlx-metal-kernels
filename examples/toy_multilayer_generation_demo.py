from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import CharTokenizer, GenerationConfig, create_synthetic_stack_generation_model


def main():
    tokenizer = CharTokenizer(add_bos=True, add_eos=False)
    model = create_synthetic_stack_generation_model(tokenizer=tokenizer, seed=13)
    prompt = "Hello"
    generation_config = GenerationConfig(max_new_tokens=16, top_k=8, temperature=0.9, seed=5)
    output_ids = model.generate_token_ids(tokenizer.encode(prompt), generation_config)
    decoded = tokenizer.decode(output_ids, stop_at_eos=True)
    print("This uses synthetic random weights and is not meaningful language generation.")
    print("prompt:", prompt)
    print("generated_token_ids:", output_ids)
    print("decoded_text:", decoded)


if __name__ == "__main__":
    main()
