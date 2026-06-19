uint row_id = thread_position_in_grid.x;

const int B = meta[0];
const int S = meta[1];
const int H = meta[2];
const int D = meta[3];

if (D > MAX_HEAD_DIM) {
    return;
}

const uint total_rows = uint(B * H);
if (row_id >= total_rows) {
    return;
}

const int h = int(row_id % uint(H));
const int b = int(row_id / uint(H));

const int q_stride_b = H * D;
const int q_stride_h = D;
const int kv_stride_b = S * H * D;
const int kv_stride_s = H * D;
const int kv_stride_h = D;

const int q_base = b * q_stride_b + h * q_stride_h;
const float scale_val = scale[0];

float row_max = -INFINITY;
float row_sum = 0.0f;
float acc[MAX_HEAD_DIM];
for (int d = 0; d < MAX_HEAD_DIM; ++d) {
    acc[d] = 0.0f;
}

for (int k_idx = 0; k_idx < S; ++k_idx) {
    const int kv_base = b * kv_stride_b + k_idx * kv_stride_s + h * kv_stride_h;

    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
        dot += float(q[q_base + d]) * float(K_cache[kv_base + d]);
    }

    const float score = dot * scale_val;
    const float new_max = max(row_max, score);
    const float alpha = isfinite(row_max) ? exp(row_max - new_max) : 0.0f;
    const float beta = exp(score - new_max);

    for (int d = 0; d < D; ++d) {
        acc[d] = acc[d] * alpha + beta * float(V_cache[kv_base + d]);
    }

    row_sum = row_sum * alpha + beta;
    row_max = new_max;
}

const int out_base = b * q_stride_b + h * q_stride_h;
const float inv_sum = 1.0f / max(row_sum, 1.0e-20f);
for (int d = 0; d < D; ++d) {
    out[out_base + d] = ELEM_TYPE(acc[d] * inv_sum);
}
