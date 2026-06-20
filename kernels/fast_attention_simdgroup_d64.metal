// Experimental body-only MLX custom Metal kernel source.
//
// This kernel is intentionally narrow:
//   - backend name: simdgroup_d64
//   - head_dim fixed at 64
//   - fp16 only (validated in Python)
//   - one simdgroup cooperates on one [b, q, h] attention row
//
// The goal for PR #15 is to establish a correctness-first simdgroup attention
// experiment without changing stable/default backend behavior. This version
// uses simdgroup reductions and lane-partitioned V accumulation as a practical
// precursor to more aggressive simdgroup_matrix tiling.

const uint lane = thread_position_in_threadgroup.x;
const uint row_id = threadgroup_position_in_grid.x;

const int B = meta[0];
const int S = meta[1];
const int H = meta[2];
const int D = meta[3];
const bool causal = (meta[4] != 0);

if (D != 64) {
    return;
}
if (threads_per_threadgroup.x != 32) {
    return;
}

const uint total_rows = uint(B * S * H);
if (row_id >= total_rows) {
    return;
}

const int h = int(row_id % uint(H));
const int q_idx = int((row_id / uint(H)) % uint(S));
const int b = int(row_id / uint(S * H));

const int stride_b = S * H * D;
const int stride_s = H * D;
const int stride_h = D;
const int q_base = b * stride_b + q_idx * stride_s + h * stride_h;
const float scale_val = scale[0];

const int d0 = int(lane);
const int d1 = d0 + 32;
const float q0 = float(Q[q_base + d0]);
const float q1 = float(Q[q_base + d1]);

float row_m = -INFINITY;
float row_l = 0.0f;
float acc0 = 0.0f;
float acc1 = 0.0f;

for (int k_idx = 0; k_idx < S; ++k_idx) {
    if (causal && k_idx > q_idx) {
        continue;
    }

    const int kv_base = b * stride_b + k_idx * stride_s + h * stride_h;
    float partial = q0 * float(K[kv_base + d0]) + q1 * float(K[kv_base + d1]);
    const float score = simd_sum(partial) * scale_val;

    const float new_m = max(row_m, score);
    const float alpha = isfinite(row_m) ? exp(row_m - new_m) : 0.0f;
    const float beta = exp(score - new_m);

    acc0 = acc0 * alpha + beta * float(V[kv_base + d0]);
    acc1 = acc1 * alpha + beta * float(V[kv_base + d1]);
    row_l = row_l * alpha + beta;
    row_m = new_m;
}

const float inv_l = 1.0f / max(row_l, 1.0e-20f);
O[q_base + d0] = ELEM_TYPE(acc0 * inv_l);
O[q_base + d1] = ELEM_TYPE(acc1 * inv_l);
