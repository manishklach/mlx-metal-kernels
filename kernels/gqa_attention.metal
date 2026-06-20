uint row_id = thread_position_in_grid.x;

const int B = meta[0];
const int Sq = meta[1];
const int Sk = meta[2];
const int Hq = meta[3];
const int Hkv = meta[4];
const int D = meta[5];
const bool causal = (meta[6] != 0);

if (D > MAX_HEAD_DIM) {
    return;
}

const uint total_rows = uint(B * Sq * Hq);
if (row_id >= total_rows) {
    return;
}

const int hq = int(row_id % uint(Hq));
const int q_idx = int((row_id / uint(Hq)) % uint(Sq));
const int b = int(row_id / uint(Sq * Hq));
const int group = Hq / Hkv;
const int hkv = hq / group;

const int q_stride_b = Sq * Hq * D;
const int q_stride_s = Hq * D;
const int q_stride_h = D;
const int kv_stride_b = Sk * Hkv * D;
const int kv_stride_s = Hkv * D;
const int kv_stride_h = D;

const int q_base = b * q_stride_b + q_idx * q_stride_s + hq * q_stride_h;
const int j_max = causal ? (q_idx + 1) : Sk;
const float scale_val = scale[0];

float row_m = -INFINITY;
float row_l = 0.0f;
float acc[MAX_HEAD_DIM];
for (int d = 0; d < MAX_HEAD_DIM; ++d) {
    acc[d] = 0.0f;
}

for (int k_idx = 0; k_idx < j_max; ++k_idx) {
    const int kv_base = b * kv_stride_b + k_idx * kv_stride_s + hkv * kv_stride_h;

    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
        dot += float(Q[q_base + d]) * float(K[kv_base + d]);
    }

    const float score = dot * scale_val;
    const float m_new = max(row_m, score);
    const float alpha = isfinite(row_m) ? exp(row_m - m_new) : 0.0f;
    const float beta = exp(score - m_new);
    row_l = row_l * alpha + beta;
    row_m = m_new;

    for (int d = 0; d < D; ++d) {
        acc[d] = acc[d] * alpha + beta * float(V[kv_base + d]);
    }
}

for (int d = 0; d < D; ++d) {
    O[q_base + d] = ELEM_TYPE(acc[d] / max(row_l, 1.0e-20f));
}
