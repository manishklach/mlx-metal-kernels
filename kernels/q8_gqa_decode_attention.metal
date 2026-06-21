uint row_id = thread_position_in_grid.x;

const int B = meta[0];
const int MAX_S = meta[1];
const int Hq = meta[2];
const int Hkv = meta[3];
const int D = meta[4];
const int group_size = meta[5];
const int groups_per_head = meta[6];

if (D > MAX_HEAD_DIM) {
    return;
}

const uint total_rows = uint(B * Hq);
if (row_id >= total_rows) {
    return;
}

const int hq = int(row_id % uint(Hq));
const int b = int(row_id / uint(Hq));
const int group = Hq / Hkv;
const int hkv = hq / group;

const int q_stride_b = Hq * D;
const int q_stride_h = D;
const int kv_stride_b = MAX_S * Hkv * D;
const int kv_stride_s = Hkv * D;
const int kv_stride_h = D;
const int scale_stride_b = MAX_S * Hkv * groups_per_head;
const int scale_stride_s = Hkv * groups_per_head;
const int scale_stride_h = groups_per_head;

const int valid_len = lengths[b];
if (valid_len <= 0) {
    return;
}

const int q_base = b * q_stride_b + hq * q_stride_h;
const float scale_val = scale[0];

float row_m = -INFINITY;
float row_l = 0.0f;
float acc[MAX_HEAD_DIM];
for (int d = 0; d < MAX_HEAD_DIM; ++d) {
    acc[d] = 0.0f;
}

for (int j = 0; j < valid_len; ++j) {
    const int kv_base = b * kv_stride_b + j * kv_stride_s + hkv * kv_stride_h;
    const int sc_base = b * scale_stride_b + j * scale_stride_s + hkv * scale_stride_h;

    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
        const int g = d / group_size;
        const float q_val = float(q[q_base + d]);
        const float k_val = (float(K_q[kv_base + d]) - 128.0f) * K_scales[sc_base + g];
        dot += q_val * k_val;
    }
    const float score = dot * scale_val;
    const float m_new = max(row_m, score);
    const float alpha = isfinite(row_m) ? exp(row_m - m_new) : 0.0f;
    const float beta = exp(score - m_new);
    row_l = row_l * alpha + beta;
    row_m = m_new;

    for (int d = 0; d < D; ++d) {
        const int g = d / group_size;
        const float v_val = (float(V_q[kv_base + d]) - 128.0f) * V_scales[sc_base + g];
        acc[d] = acc[d] * alpha + beta * v_val;
    }
}

const float inv_l = 1.0f / max(row_l, 1.0e-20f);
for (int d = 0; d < D; ++d) {
    out[q_base + d] = ELEM_TYPE(acc[d] * inv_l);
}
