# Quantization Kernels

This PR adds correctness-first quantization helpers and decode matvec kernels.

## Q4 packing

Q4 stores two 4-bit values inside one `uint8` byte:

- low nibble: even element
- high nibble: odd element

## Groupwise scales and zero-points

The current helpers use groupwise quantization:

- one scale per `group_size` values
- optional one zero-point per `group_size` values

## Kernels

- `dequant_q4`: unpack q4 and dequantize
- `dequant_q8`: dequantize q8 values
- `q4_matvec_decode`: dequantize q4 weights on the fly during decode matvec
- `q8_matvec_decode`: dequantize q8 weights on the fly during decode matvec

## Status

This v1 path is correctness-first and not yet heavily optimized.

## Future direction

- parallel reduction over K
- blockwise or tiled matvec
- q4 GEMV for decode
- q4 GEMM for prefill
- fused dequant + matvec + bias
- tighter integration with the decode block
