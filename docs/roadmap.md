# Roadmap

This repo is evolving as an experimental Apple Silicon MLX/Metal kernel lab for LLM inference. The roadmap is organized around correctness-first primitives, explicit experimental backends, and local-machine validation rather than broad performance claims.

## Snapshot

- [x] Baseline MLX custom Metal attention
- [x] Reference correctness path
- [x] Row-parallel and tiled-K/V attention experiments
- [x] RMSNorm, RoPE, SwiGLU, and residual helpers
- [x] KV-cache update and decode attention
- [x] Paged KV-cache and paged decode attention
- [x] Fused decode block helpers
- [x] q4/q8 dequantization and decode matvec
- [x] Parallel and tiled q4/q8 matvec
- [x] Quantized decode block
- [x] Threadgroup attention v2
- [x] Simdgroup attention experiments
- [x] Unified benchmark and report suite
- [x] Chip-specific autotuning
- [x] Toy transformer-layer decode benchmark
- [x] Llama-like model integration scaffold
- [x] Quantized MLP block
- [x] GQA/MQA reference and composed decode support
- [x] Checkpoint layout loader scaffold
- [x] Optimized GQA Metal decode attention
- [x] Fused q4/q8 MLP kernel experiments
- [x] Real checkpoint adapter scaffold
- [x] GQA/MQA prefill attention
- [x] Full fused Llama-like decode layer experiment
- [x] Checkpoint-to-quantized packaging
- [x] Tokenizer and sampling demo
- [x] Multi-layer decode stack
- [x] Production checkpoint converter
- [x] Real tokenizer adapter
- [x] Full tiny-model generation demo
- [x] Optimized prefill stack
- [ ] Tokenizer/checkpoint package alignment
- [ ] Local real-model smoke test
- [ ] Quantized package tensor-data writer
- [ ] Paged prefill

## Development pattern

The intended workflow for each new primitive is:

1. implement a pure MLX reference path
2. add a correctness-first Metal or composed backend
3. compare outputs against reference
4. benchmark locally on Apple Silicon
5. keep optimized paths explicit until validation is repeatable

## Completed phases

### Attention foundation

- baseline streaming attention kernel
- reference attention implementation
- row-parallel and tiled-K/V experiments
- shape-specialized D=64 and D=128 attention and decode kernels
- threadgroup attention v2
- experimental `simdgroup_d64` attention

### Decode and cache foundation

- contiguous KV-cache update
- decode attention
- paged KV-cache allocation and update
- paged decode attention
- decode-loop helpers
- fused decode helpers from packed QKV

### Transformer primitives

- RMSNorm
- RoPE
- SwiGLU
- residual add
- RMSNorm plus residual
- QKV split and split-plus-RoPE helpers
- fused QKV plus RoPE plus cache-update path

### Quantization and block composition

- q4 and q8 dequantization
- q4 and q8 decode matvec
- parallel q4/q8 matvec
- multi-output tiled q4/q8 matvec
- quantized QKV projection and output projection
- quantized decode block
- quantized MLP block

### Model-level scaffolding

- toy transformer-layer decode benchmark
- Llama-like config objects
- weight-layout mapping helpers
- model-adapter scaffold
- GQA/MQA decode utilities and routing
- checkpoint manifest and layout validation scaffold

### Benchmarking and autotuning

- unified benchmark runner
- JSON and CSV benchmark output
- local report generation flow
- backend registry
- local autotune cache keyed to the machine

## Near-term roadmap

### v0.20 checkpoint layout loader scaffold

- add a lightweight checkpoint-layout bridge for real Llama-like weight mapping
- keep tokenizer integration, full checkpoint ingestion, and serving concerns out of scope

### v0.21 optimized GQA Metal decode attention

- add optional Metal GQA decode kernels once the reference and composed paths are well covered
- keep equal-head kernels unchanged and keep GQA optimized paths explicit until locally benchmarked

### v0.22 fused q4/q8 MLP kernel experiments

- add explicit-only fused gate/up plus SwiGLU experiments on top of the stable composition-first MLP path
- prefer narrow, benchmarkable fusion steps over sweeping backend-default changes

### v0.23 real checkpoint adapter

- extend model-level scaffolding toward a more practical checkpoint adapter
- still avoid overclaiming production inference support until end-to-end validation exists

### v0.24 GQA prefill attention

- extend the GQA work from decode to prefill attention
- keep optimized prefill kernels explicit until correctness and benchmark coverage are in place

### v0.25 full fused Llama-like decode layer experiment

- explore a larger decode-layer fusion experiment once attention, quantized block, and MLP components are individually validated
- keep production claims out of scope until the composed path is benchmarked and debuggable

### v0.26 checkpoint-to-quantized packaging

- connect validated checkpoint layouts to deterministic q4/q8 packaging helpers
- keep quantization and calibration concerns out of the main runtime path until they are independently testable

### v0.27 tokenizer and sampling demo

- add a minimal tokenizer and sampling demo once checkpoint and runtime scaffolding are better aligned
- keep it separate from any production-serving claim

### v0.28 multi-layer decode stack

- extend the single-layer experiment into an explicit multi-layer decode stack
- keep it benchmarkable and debuggable before treating it like a general runtime

### v0.29 production checkpoint converter

- [x] JSON package metadata format
- [x] synthetic/in-memory conversion path
- [x] manifest dry-run planning
- [x] fused QKV packaging
- [x] q4/q8 layer package metadata
- [x] package inspection CLI
- [x] per-layer tensor metadata
- [x] converter tests and CLI tests
- extend the local packaging scaffold into a more practical checkpoint conversion flow
- keep dependency growth and model-format assumptions explicit

### v0.30 real tokenizer adapter

- [x] TokenizerProtocol extended with add_special_tokens, skip_special_tokens, and special token id properties
- [x] HFTokenizerAdapter for local tokenizer JSON (optional `tokenizers` package)
- [x] SentencePieceTokenizerAdapter for local .model files (optional `sentencepiece` package)
- [x] TokenizerAdapterFactory with auto-detection by file extension
- [x] describe_tokenizer helper
- [x] load_tokenizer_for_generation helper
- [x] tokenizer adapter demo
- [x] tests without optional dependencies
- add an optional adapter layer for real tokenizer integration without forcing heavy defaults
- keep production tokenizer claims out of scope until checkpoint and runtime wiring are more complete

### v0.31 full tiny-model generation demo

- [x] `TinyGenerationPipelineConfig` and `TinyGenerationPipeline`
- [x] end-to-end tokenizer -> embeddings -> stack -> logits -> sampling -> decode demo
- [x] synthetic q4/q8 stack generation path
- [x] package-based demo with metadata-only fallback
- [x] benchmark scaffold and tests
- extend the single-layer generation scaffold toward a slightly more complete tiny-model demo
- keep it explicit that this remains an experimental local-kernel lab, not a production runtime

### v0.32 optimized prefill stack

- [x] layer prefill scaffold
- [x] multi-layer stack prefill
- [x] token-id prefill helper
- [x] tiny pipeline `use_prefill=True`
- [x] prefill-then-decode demo
- [x] benchmark scaffold
- extend the stack scaffold with a clearer prefill path on top of the GQA prefill building blocks
- keep correctness and cache visibility ahead of optimization claims

### v0.33 tokenizer/checkpoint package alignment

- verify that tokenizer metadata, vocab assumptions, and checkpoint package metadata line up cleanly
- keep optional tokenizer dependencies and local-file-only behavior explicit

### v0.34 local real-model smoke test

- add a narrow local smoke-test path for a small real checkpoint once tensor loading exists
- avoid broad quality or performance claims until local verification is repeatable

### v0.35 quantized package tensor-data writer

- extend the metadata-only package format toward optional tensor payload support
- keep the initial scope focused on deterministic local packaging and loading

### v0.36 paged prefill

- extend prefill from contiguous-only cache filling toward paged KV-cache support
- keep continuation semantics and validation explicit before broadening defaults

## Long-term goal

The longer-term goal is a practical collection of Apple Silicon MLX/Metal inference primitives for local transformer experimentation:

- attention
- decode
- KV-cache and paged KV-cache
- quantized matvec
- MLP and normalization blocks
- model-layout helpers
- backend autotuning

The repo remains intentionally experimental. Stable defaults should stay conservative, and every optimized backend should earn its place through reference validation and local Apple Silicon benchmarks.
