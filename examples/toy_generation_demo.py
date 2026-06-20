from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import CharTokenizer, GenerationConfig, create_synthetic_generation_model


def main():
    tokenizer = CharTokenizer(add_bos=True, add_eos=False)
    model = create_synthetic_generation_model(tokenizer=tokenizer, seed=11)
    prompt = "Hello"
    generation_config = GenerationConfig(max_new_tokens=16, top_k=8, temperature=0.9, seed=5)
    input_ids = tokenizer.encode(prompt)
    output_ids = model.generate_token_ids(input_ids, generation_config)
    decoded = tokenizer.decode(output_ids, stop_at_eos=True)
    print("This demo uses synthetic random weights and is not meaningful language generation.")
    print("prompt:", prompt)
    print("prompt_token_ids:", input_ids)
    print("generated_ids:", output_ids)
    print("decoded_output:", decoded)


if __name__ == "__main__":
    main()
