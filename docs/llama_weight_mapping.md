# Llama Weight Mapping

This document explains how common Llama-style names relate to the kernel-oriented layouts used in this repo.

## Common Weight Names

- `self_attn.q_proj.weight`
- `self_attn.k_proj.weight`
- `self_attn.v_proj.weight`
- `self_attn.o_proj.weight`
- `mlp.gate_proj.weight`
- `mlp.up_proj.weight`
- `mlp.down_proj.weight`
- `input_layernorm.weight`
- `post_attention_layernorm.weight`

## Mapping To Existing Kernels

The repo already has decode paths that prefer a fused QKV representation for the projected attention input.

That means there are two conceptual integration paths:

1. Keep separate `q_proj`, `k_proj`, and `v_proj`, then concatenate projected outputs into a fused QKV activation.
2. Pre-fuse weights into a single QKV projection layout and target the fused decode helpers directly.

## Fused QKV Format

For the current MHA-only scaffold, fused QKV is stacked as:

```text
[q_proj; k_proj; v_proj]
```

with shape:

```text
[q_out + k_out + v_out, hidden_size]
```

For the common MHA case in this scaffold:

```text
[3 * hidden_size, hidden_size]
```

## Quantized Layout Expected By q4/q8 Decode Matvec

The decode matvec kernels expect logical weights shaped as:

```text
[OUT_DIM, IN_DIM]
```

For q4:

```text
packed weights: [OUT_DIM, ceil(IN_DIM / 2)]
scales:         [OUT_DIM, ceil(IN_DIM / group_size)]
```

For q8:

```text
weights: [OUT_DIM, IN_DIM]
scales:  [OUT_DIM, ceil(IN_DIM / group_size)]
```

## GQA/MQA Note

Standard grouped-query variants use:

- `q_proj` output based on `num_attention_heads * head_dim`
- `k_proj` / `v_proj` output based on `num_key_value_heads * head_dim`

This PR records those shapes in the specs, but the runtime adapter keeps the first supported path limited to:

```text
num_key_value_heads == num_attention_heads
```

until the decode/cache path grows explicit GQA support.
