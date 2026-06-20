# Tokenizer/checkpoint package alignment

## Purpose

This repo now includes a structured alignment layer for checking whether tokenizer metadata, `LlamaLikeConfig`, quantized checkpoint package metadata, embeddings, `lm_head`, and generation-scaffold settings agree before a generation or smoke-test path runs.

## Why alignment validation matters

The repo is moving toward local real-model smoke testing, but it is still intentionally conservative about what it claims. Before any tokenizer-driven generation path runs, we want a clear metadata boundary that answers:

- does `tokenizer.vocab_size` match the embedding and `lm_head` vocab dimension?
- do BOS/EOS/PAD/UNK ids fit inside the vocab?
- does the package config match the runtime config?
- do layer counts and expected tensor shapes line up?
- do q4/q8 metadata and packed shapes match the repo's quantization conventions?

These checks catch integration mistakes early and produce structured reports instead of ad-hoc runtime failures.

## Tokenizer checks

`models.alignment.tokenizer_alignment_info()` extracts:

- `vocab_size`
- `bos_token_id`
- `eos_token_id`
- `pad_token_id`
- `unk_token_id`
- `tokenizer_type`

The alignment layer does not require optional tokenizer dependencies. It works with the built-in toy tokenizers and with optional adapters when they are installed locally.

## Config/package checks

`validate_config_against_package()` checks:

- `hidden_size`
- `intermediate_size`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `num_hidden_layers`
- `max_position_embeddings`
- `model_type`
- package format presence
- layer metadata counts
- expected fused QKV / O / gate / up / down shapes

## Embedding/lm_head checks

The repo currently documents `embedding` and `lm_head` with the `[vocab_size, hidden_size]` convention.

Alignment validation checks:

- vocab dimension against tokenizer/config expectations
- hidden dimension against `config.hidden_size`
- whether `embedding` or `lm_head` is missing from the runtime context

Missing `embedding` or `lm_head` is reported structurally so callers can decide whether that is acceptable for a given scaffold.

## Quantization metadata checks

`validate_quantization_alignment()` checks:

- package-level `bits` and `group_size`
- per-tensor `bits` and `group_size`
- q4 packed shape expectation `[out_dim, ceil(in_dim / 2)]`
- q8 packed shape expectation `[out_dim, in_dim]`
- scales group dimension `ceil(in_dim / group_size)`
- optional zeros shape consistency

## Generation pipeline validation

`TinyGenerationPipeline.validate_alignment()` calls `validate_generation_alignment()` and combines:

- tokenizer vs config
- tokenizer vs package
- config vs package
- quantization metadata
- stack weight layer counts
- embedding / `lm_head` shape checks

`generate()` and `generate_ids()` now accept `validate_alignment=True` by default. Errors stop the generation path early. Warnings stay visible in the report but do not block execution.

## CLI package inspection integration

`scripts/inspect_quantized_package.py` supports:

- `--validate-alignment`
- `--tokenizer`
- `--tokenizer-kind`
- `--bits`
- `--group-size`

This keeps package inspection and alignment reporting available without network access or heavy required dependencies.

## Examples

```bash
python examples/alignment_demo.py
python scripts/inspect_quantized_package.py /tmp/mlx_quant_package.json --validate-alignment
```

## Current limitations

- validates metadata and shapes
- does not guarantee trained model quality
- does not download model/tokenizer
- does not run production inference
- optional tokenizer adapters remain optional

## Future work

- tokenizer metadata stored inside quantized package
- chat template validation
- real checkpoint package tensor-data validation
- local real-model smoke test
