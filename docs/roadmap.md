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
- [ ] Tokenizer and sampling demo
- [ ] Multi-layer decode stack
- [ ] Production checkpoint converter
- [ ] Calibration-aware quantization

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

- extend the local packaging scaffold into a more practical checkpoint conversion flow
- keep dependency growth and model-format assumptions explicit

### v0.30 calibration-aware quantization

- add validation-friendly hooks for richer quantization workflows
- keep advanced quantization claims out of scope until they are benchmarked and compared carefully

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
