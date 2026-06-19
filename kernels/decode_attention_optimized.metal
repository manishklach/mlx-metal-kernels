uint row_id = thread_position_in_grid.x;

const int B = meta[0];
const int MAX_S = meta[1];
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
const int kv_stride_b = MAX_S * H * D;
const int kv_stride_s = H * D;
const int kv_stride_h = D;
const int valid_len = lengths[b];
const int q_base = b * q_stride_b + h * q_stride_h;
const float scale_val = scale[0];

float m = -INFINITY;
float l = 0.0f;
float acc[MAX_HEAD_DIM];
for (int d = 0; d < MAX_HEAD_DIM; ++d) {
    acc[d] = 0.0f;
}

for (int j = 0; j < valid_len; ++j) {
    const int kv_base = b * kv_stride_b + j * kv_stride_s + h * kv_stride_h;

    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
        dot += float(q[q_base + d]) * float(K_cache[kv_base + d]);
    }

    const float score = dot * scale_val;
    const float m_new = max(m, score);
    const float alpha = isfinite(m) ? exp(m - m_new) : 0.0f;
    const float beta = exp(score - m_new);

    for (int d = 0; d < D; ++d) {
        acc[d] = acc[d] * alpha + beta * float(V_cache[kv_base + d]);
    }

    l = l * alpha + beta;
    m = m_new;
}

const int out_base = b * q_stride_b + h * q_stride_h;
const float inv_l = 1.0f / max(l, 1.0e-20f);
for (int d = 0; d < D; ++d) {
    out[out_base + d] = ELEM_TYPE(acc[d] * inv_l);
}
