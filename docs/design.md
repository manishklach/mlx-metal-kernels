# MLX Metal Kernels Design

## Project Goal

MLX makes Apple Silicon a serious local machine learning platform, but many high-performance model operations still benefit from custom fused kernels. This project investigates how far MLX custom Metal kernels can be pushed for transformer inference workloads on Mac.

The initial focus is attention. Standard attention materializes or conceptually computes the full `QK^T` score matrix before applying softmax and multiplying by `V`. FlashAttention-style kernels avoid materializing the full attention matrix by streaming over keys and values while maintaining online softmax statistics. This reduces memory traffic and creates a better foundation for long-context inference.

This repository starts with a correctness-first implementation and then adds progressively more optimized backends:

1. Pure MLX reference implementation
2. Baseline custom Metal streaming attention kernel
3. Row-parallel attention kernel
4. Tiled K/V attention kernel
5. Specialized D=64 and D=128 kernels
6. Decode attention for single-token inference
7. KV-cache and paged-KV kernels
8. Additional MLX custom kernels for transformer inference

The design philosophy is simple:

- correctness first
- every optimized backend must match the reference implementation
- no performance claims without benchmarks
- keep experimental kernels behind explicit backend flags
- make Apple Silicon GPU behavior visible and measurable
