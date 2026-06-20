# Fused Quantized MLP Experiments

## Why MLP fusion matters

Llama-like feed-forward blocks spend substantial time in the sequence:

`RMSNorm -> gate projection -> up projection -> SwiGLU -> down projection -> residual add`

The stable path in this repo already composes those operations from validated MLX and Metal pieces. PR #22 adds experimental fusion points that reduce launches and reuse the normalized input across multiple quantized projections.

## Llama-like MLP path

The current quantized MLP block follows:

1. residual add
2. RMSNorm
3. quantized gate projection
4. quantized up projection
5. SwiGLU
6. quantized down projection
7. residual add

## Stable composition path from PR18

The default and reference-oriented path remains composition-first:

- `reference_quantized_mlp_block`
- `quantized_mlp_block(...)` with existing `reference`, `metal`, `parallel`, or `tiled` backend choices

This remains the stability baseline for all experimental work.

## Experimental fused path from PR22

This PR adds an explicit-only preset:

- `backend_preset="fused_experimental"`

It maps to:

- `norm_backend="metal"`
- `matvec_backend="metal_gate_up_tiled"`
- `activation_backend="metal_fused"`
- `down_backend="metal_tiled"`

`auto` does not route here.

## Gate/up combined q4/q8 matvec

The new `q4_gate_up_matvec_tiled` and `q8_gate_up_matvec_tiled` kernels compute both gate and up projections from the same normalized input tile in one launch.

Python entry points:

- `ops.quant_ops.q4_gate_up_matvec_tiled`
- `ops.quant_ops.q8_gate_up_matvec_tiled`
- `ops.mlp_block_ops.quantized_gate_up_projection(..., backend="metal_gate_up_tiled")`

These kernels are intended to be correctness-first reusable fusion steps, not blanket replacements for the existing single-output tiled matvec kernels.

## Fused SwiGLU

The repo now includes `fused_swiglu` for flattened row-major MLP activation shapes:

- `ops.activation_ops.fused_swiglu(..., backend="metal_fused")`

This keeps the activation step explicit and separately testable.

## What remains unfused

- down projection still uses the existing tiled q4/q8 matvec kernels
- residual add still happens after the down projection
- there is no monolithic all-in-one quantized MLP kernel in this PR

## Benchmark commands

```bash
python benchmarks/bench_fused_mlp_kernels.py --bits 4 --B 1 --S 1 --hidden-size 4096 --intermediate-size 11008 --dtype float16 --backend-preset all --validate
python benchmarks/bench_fused_mlp_kernels.py --bits 8 --B 1 --S 1 --hidden-size 4096 --intermediate-size 11008 --dtype float16 --backend-preset all --validate
python benchmarks/run_all_benchmarks.py --quick
```

## Current limitations

- experimental
- explicit backend only
- no default auto routing
- no performance claims without local Apple Silicon benchmark data
- validation is still required against `reference_quantized_mlp_block`

## Future

- fuse gate/up projection directly into SwiGLU
- explore fused SwiGLU plus down projection
- add specialized hidden/intermediate size kernels
- autotune tile sizes per Apple chip family
