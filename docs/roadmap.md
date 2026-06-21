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
- [x] Tokenizer/checkpoint package alignment
- [x] Local real-model smoke test
- [ ] Quantized package tensor-data writer
- [x] Sparse and sliding-window GQA attention kernels
- [x] Prefix KV-cache reuse and cache matching
- [x] Speculative decoding / MTP scaffold
- [ ] Flash/NAND KV offload tier
- [ ] Quantized KV-cache attention

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

- [x] structured alignment reports
- [x] tokenizer/config/package validation helpers
- [x] q4/q8 metadata validation
- [x] tiny pipeline pre-generation alignment checks
- [x] package inspector alignment mode
- verify that tokenizer metadata, vocab assumptions, and checkpoint package metadata line up cleanly
- keep optional tokenizer dependencies and local-file-only behavior explicit

### v0.34 local real-model smoke test

- [x] structured smoke-test report
- [x] package executability inspection
- [x] local tokenizer loading helper
- [x] dry-run metadata validation path
- [x] explicit synthetic fallback generation path
- [x] CLI smoke-test script
- add a narrow local smoke-test path for a small real checkpoint once tensor loading exists
- avoid broad quality or performance claims until local verification is repeatable

### v0.35 quantized package tensor-data writer

- [x] `tensor_data_io.py` — numpy `.npy` save/load, shape/dtype/checksum helpers
- [x] `quantized_package_writer.py` — `QuantizedPackageWriter` with config, checksums, dry-run mode
- [x] `QuantizedCheckpointPackage` — `has_tensor_data()`, `tensor_files()`, `validate_tensor_files()`
- [x] `CheckpointConverterConfig.save_tensor_data=True` — wires the writer through the existing converter
- [x] `SmokeTestConfig.inspect_package_executability` — checksum validation support
- [x] `scripts/write_quantized_package.py` — CLI with `--synthetic` and `--dry-run` modes
- [x] `scripts/inspect_quantized_package.py` — `--check-tensor-files`, `--check-checksums`, `--package-root`
- [x] Tests for all new modules and CLIs
- extend the metadata-only package format toward optional tensor payload support
- keep the initial scope focused on deterministic local packaging and loading

### v0.36 sparse and sliding-window GQA attention kernels

- [x] reference sparse attention masks
- [x] sparse GQA/MQA/MHA reference attention
- [x] sliding-window prefill Metal kernel
- [x] sliding-window decode Metal kernel
- [x] sink-token sparse attention path
- [x] sparse benchmarks and documentation

### v0.37 prefix KV-cache reuse and cache matching

- [x] `ops/kv_cache_reuse_ops.py` — clone, slice, copy, cache_prefix_equal ops
- [x] `models/prefix_cache.py` — fingerprint, InMemoryPrefixCache, prefill_with_prefix_reuse
- [x] pipeline integration via `use_prefix_cache` flag
- [x] fingerprint-based safety (avoids cross-model/config reuse)
- [x] LRU eviction at capacity
- [x] test coverage for cache ops, prefix matching, eviction, and end-to-end reuse
- [x] benchmark scaffold
- [x] example and documentation

### v0.38 speculative decoding / MTP scaffold

- [x] `SpeculativeConfig`, `DraftProposal`, `VerificationResult`, `SpeculativeStepResult`, `SpeculativeGenerationResult`
- [x] `FixedDraftProposer`, `RandomDraftProposer`, `GreedySelfDraftProposer` — three draft proposer implementations
- [x] `PipelineTargetVerifier` — greedy target model verifier
- [x] `SpeculativeGenerator` — generate_ids + generate_text with accept/reject loop
- [x] `compute_accept_mask`, `accepted_prefix_length`, `verify_draft_tokens` — accept/reject helpers
- [x] `SpeculativeGenerationResult.acceptance_rate()` and `tokens_per_step()` — metadata queries
- [x] `ops/speculative_cache_ops.py` — `commit_accepted_cache`, `discard_suffix`
- [x] `models/mtp.py` — `MTPConfig`, `SyntheticMTPHead`, `mtp_propose_tokens`
- [x] Pipeline integration via `TinyGenerationPipeline.generate_speculative()`
- [x] `generate_speculative` is opt-in; existing `generate()` unchanged
- [x] B=1, contiguous cache first; paged speculative cache raises `NotImplementedError`
- [x] Test coverage: accept/reject, proposers, verifier, generator, cache ops, MTP scaffold, pipeline speculative mode
- [x] Benchmark scaffold and examples
- [x] Documentation and roadmap update
- keep model-quality and acceptance-rate claims out of scope until local validation exists

### v0.39 Flash/NAND KV offload tier scaffold

- [x] KVBlockId, KVBlockMetadata, KVResidencyMap — block metadata and tracking
- [x] `partition_sequence_into_blocks`, `token_positions_to_block_ids` — partitioning helpers
- [x] `InMemoryKVOffloadStore`, `FileKVOffloadStore` (npy-backed) — offload stores
- [x] `KVOffloadPolicyConfig`, `KVOffloadPlan` — offload policy types
- [x] `plan_offload_blocks` — sink/recent/max-resident policy
- [x] `plan_prefetch_for_sparse_attention` — prefetch planning for sparse patterns
- [x] `extract_kv_block`, `insert_kv_block`, `offload_kv_block`, `prefetch_kv_block` — offload operations
- [x] `apply_offload_plan` — batch offload/prefetch execution
- [x] `ensure_sparse_blocks_resident` — hard guard against attending to offloaded blocks
- [x] `sparse_positions_for_decode` — determine needed token positions for sparse decode
- [x] `clone_residency_map` — prefix-cache integration helper
- [x] Test coverage: metadata, store, policy, ops, sparse integration (5 test files)
- [x] Benchmark scaffold and examples
- [x] Documentation and roadmap update
- scaffold only; no real async IO, DMA, or automatic runtime offload

### v0.40 quantized KV-cache attention

- extend sparse and decode experiments toward quantized KV-cache reads
- keep accuracy, layout, and bandwidth tradeoffs explicit

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
