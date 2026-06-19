// Body-only MLX custom Metal kernel source.
// MLX generates the `kernel void fast_attention_forward(...)` signature from
// input_names=["Q", "K", "V", "meta", "scale"] and output_names=["O"].
//
// Inputs available in this body:
//   Q, K, V    : device const ELEM_TYPE*  [B, S, H, D], BSHD layout
//   meta       : device const int*        [B, S, H, D, causal]
//   scale      : device const float*      [1]
//   O          : device ELEM_TYPE*        [B, S, H, D]
//   thread_position_in_grid : uint3
//
// Correctness-first implementation:
//   - one Metal thread computes one full output row O[b, q, h, :]
//   - three streaming passes over K/V: max, denominator, weighted value sum
//   - avoids materializing the [S, S] attention matrix
//   - supports D <= MAX_HEAD_DIM, fp16/bf16 via ELEM_TYPE
//
// This is intentionally simple and repo-safe.  The optimized tiled/simdgroup
// kernel should be added later as a separate experimental kernel.

uint row_id = thread_position_in_grid.x;

const int B      = meta[0];
const int S      = meta[1];
const int H      = meta[2];
const int D      = meta[3];
const bool causal = (meta[4] != 0);

if (D > MAX_HEAD_DIM) {
    return;
}

const uint total_rows = uint(B * S * H);
if (row_id >= total_rows) {
    return;
}

const int h = int(row_id % uint(H));
const int q = int((row_id / uint(H)) % uint(S));
const int b = int(row_id / uint(S * H));

const int stride_b = S * H * D;
const int stride_s = H * D;
const int stride_h = D;

const int q_base = b * stride_b + q * stride_s + h * stride_h;
const float scale_val = scale[0];

// Pass 1: row max for numerical stability.
float row_max = -INFINITY;
for (int k_idx = 0; k_idx < S; ++k_idx) {
    if (causal && k_idx > q) {
        continue;
    }

    const int k_base = b * stride_b + k_idx * stride_s + h * stride_h;
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
        dot += float(Q[q_base + d]) * float(K[k_base + d]);
    }

    const float score = dot * scale_val;
    row_max = max(row_max, score);
}

// Pass 2: denominator.
float denom = 0.0f;
for (int k_idx = 0; k_idx < S; ++k_idx) {
    if (causal && k_idx > q) {
        continue;
    }

    const int k_base = b * stride_b + k_idx * stride_s + h * stride_h;
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
        dot += float(Q[q_base + d]) * float(K[k_base + d]);
    }

    const float score = dot * scale_val;
    denom += exp(score - row_max);
}

// Pass 3: weighted V accumulation.
float acc[MAX_HEAD_DIM];
for (int d = 0; d < MAX_HEAD_DIM; ++d) {
    acc[d] = 0.0f;
}

for (int k_idx = 0; k_idx < S; ++k_idx) {
    if (causal && k_idx > q) {
        continue;
    }

    const int k_base = b * stride_b + k_idx * stride_s + h * stride_h;
    const int v_base = k_base;

    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
        dot += float(Q[q_base + d]) * float(K[k_base + d]);
    }

    const float score = dot * scale_val;
    const float p = exp(score - row_max) / max(denom, 1.0e-20f);

    for (int d = 0; d < D; ++d) {
        acc[d] += p * float(V[v_base + d]);
    }
}

for (int d = 0; d < D; ++d) {
    O[q_base + d] = ELEM_TYPE(acc[d]);
}
