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
- [ ] Optimized GQA Metal decode attention
- [ ] Fused q4 MLP kernel
- [ ] Real checkpoint adapter
- [ ] Tokenizer and sampling demo

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

### v0.22 fused q4 MLP kernel

- explore a dedicated fused q4 MLP kernel after the composition-first MLP path is validated
- prefer narrow, benchmarkable kernel experiments over sweeping backend-default changes

### v0.23 real checkpoint adapter

- extend model-level scaffolding toward a more practical checkpoint adapter
- still avoid overclaiming production inference support until end-to-end validation exists

### v0.24 tokenizer and sampling demo

- add a minimal tokenizer plus sampling-loop demonstration once checkpoint and adapter contracts stabilize
- keep it clearly separate from any production serving claim

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
