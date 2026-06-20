# Local real-model smoke test scaffold

## Purpose

This repo now includes a conservative local-only smoke-test scaffold for checking whether a local tokenizer, local quantized package metadata, model config, optional tensor-data references, and runtime settings are aligned well enough to prepare for a future real-model run.

## What the smoke test checks

The smoke test answers:

- can a local package JSON be loaded and summarized?
- can a local tokenizer be loaded without network access?
- do tokenizer/package/config alignment checks pass?
- does the package appear executable, or is it metadata-only?
- if execution is requested, should the run stop clearly before generation?
- if synthetic fallback is explicitly enabled, can a tiny prefill/decode generation smoke run?

## Local-only design

- no downloads
- no model hub integration
- no required Hugging Face dependencies
- no required `tokenizers`, `sentencepiece`, or `safetensors`
- no claim of production Llama or Mistral runtime support

## Dry-run mode

Dry-run is the default mode.

It loads local metadata, inspects executability, optionally loads a tokenizer, runs alignment validation, and returns a structured report without attempting generation.

## Synthetic fallback mode

Synthetic fallback must be explicit.

If `--synthetic-fallback` is set, the smoke test may run a tiny synthetic generation path using random weights and the existing `TinyGenerationPipeline`. This is only a plumbing smoke test. It does not produce meaningful language.

## Metadata-only package limitation

The current package format is still metadata-first. A metadata-only package must not pretend to be executable.

The smoke test reports:

- whether tensor data paths are present
- which tensor-data references are missing
- whether the package is executable

## Tensor-data requirement

If `require_tensor_data=True`, the smoke test fails clearly when the package is metadata-only or has missing tensor-data files.

Even if data files exist, this repo may still stop before generation because a package tensor-data loader is not implemented yet.

## Tokenizer loading

Tokenizer loading is local-only:

- built-in `CharTokenizer`
- built-in `WhitespaceTokenizer`
- optional local `tokenizers` JSON adapter
- optional local SentencePiece adapter

Missing optional dependencies are reported explicitly.

## Alignment validation

When enabled, the smoke test reuses the tokenizer/checkpoint alignment layer to validate:

- tokenizer metadata
- package metadata
- config/package shape assumptions
- quantization metadata

## CLI examples

```bash
python scripts/smoke_test_local_model.py --package /tmp/mlx_quant_package.json --dry-run
python scripts/smoke_test_local_model.py --package /tmp/mlx_quant_package.json --tokenizer /path/to/tokenizer.json --tokenizer-kind hf-tokenizers --dry-run
python scripts/smoke_test_local_model.py --synthetic-fallback --no-dry-run --prompt "Hello" --max-new-tokens 4
python examples/local_smoke_test_demo.py
```

## Current limitations

- no downloads
- no production inference
- metadata-only package cannot generate
- tensor-data package loading may not exist yet
- synthetic fallback uses random weights
- optional tokenizer dependencies remain optional

## Future work

- quantized package tensor-data writer
- package tensor-data loader
- local real checkpoint smoke path
- real tokenizer/package alignment metadata
- chat template scaffold
