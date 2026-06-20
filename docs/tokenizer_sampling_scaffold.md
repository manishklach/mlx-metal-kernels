# Tokenizer and sampling scaffold

## Purpose

This scaffold adds a lightweight tokenizer, sampling, and generation loop around the repo's Llama-like single-layer decode experiments.

It is intentionally small and educational. It is not a production tokenizer stack or model runtime.

## Tokenizer abstraction

The tokenizer layer is expressed through a tiny `TokenizerProtocol` with:

- `encode(text) -> list[int]`
- `decode(token_ids) -> str`
- `vocab_size`

This keeps future real-tokenizer integration possible without pulling in heavy dependencies today.

## CharTokenizer

`CharTokenizer` is a deterministic character-level tokenizer with:

- a built-in ASCII-ish vocabulary
- special tokens for pad, bos, eos, and unk
- optional BOS and EOS insertion

It is useful for tiny demos and unit tests where tokenization quality is not the goal.

## Sampling utilities

The scaffold includes:

- softmax
- greedy sampling
- top-k filtering
- top-p filtering
- repetition penalty
- seeded stochastic sampling

These helpers are designed for correctness and readability rather than production serving claims.

## GenerationConfig

`GenerationConfig` controls:

- `max_new_tokens`
- `temperature`
- `top_k`
- `top_p`
- `repetition_penalty`
- optional `eos_token_id`
- optional deterministic `seed`

## ToyLlamaGenerationModel

`ToyLlamaGenerationModel` is a single-layer generation scaffold.

Current flow:

1. embed token id
2. run a single-layer decode step
3. project hidden state to logits
4. sample the next token
5. update cache and position

When MLX and the decode-layer stack are available, it can use that path directly. In dependency-light environments, it falls back to a small NumPy reference-style path so plumbing can still be tested.

## Synthetic generation demo

The demo uses:

- random synthetic embeddings
- random synthetic quantized layer weights
- random synthetic lm head
- toy tokenization

The output is not meaningful language. The point is to validate wiring and interfaces.

## What this does not claim

- production tokenizer quality
- SentencePiece or BPE support
- Hugging Face tokenizer compatibility
- meaningful text quality from random weights
- multi-layer runtime completeness
- production inference performance

## Future work

- real tokenizer adapter
- optional SentencePiece or BPE integration
- prompt prefill via GQA prefill attention
- multi-layer decode stack
- checkpoint-derived logits from real model weights
- sampling loop with KV-cache across multiple layers
