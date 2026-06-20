uint row_id = thread_position_in_grid.x;

const int B = meta[0];
const int S = meta[1];
const int H = meta[2];
const int D = meta[3];
const bool causal = (meta[4] != 0);

if (D != HEAD_DIM) {
    return;
}

const uint total_rows = uint(B * S * H);
if (row_id >= total_rows) {
    return;
}

const int h = int(row_id % uint(H));
const int q_idx = int((row_id / uint(H)) % uint(S));
const int b = int(row_id / uint(S * H));
const int out_base = ((b * S + q_idx) * H + h) * D;
const float scale_val = scale[0];

float qv[HEAD_DIM];
float acc[HEAD_DIM];
for (int d = 0; d < HEAD_DIM; ++d) {
    qv[d] = float(Q[out_base + d]);
    acc[d] = 0.0f;
}

float m = -INFINITY;
float l = 0.0f;
for (int k_idx = 0; k_idx < S; ++k_idx) {
    if (causal && k_idx > q_idx) {
        continue;
    }
    const int kv_base = ((b * S + k_idx) * H + h) * D;
    float dot = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
        dot += qv[d] * float(K[kv_base + d]);
    }
    const float score = dot * scale_val;
    const float m_new = max(m, score);
    const float alpha = isfinite(m) ? exp(m - m_new) : 0.0f;
    const float beta = exp(score - m_new);
    for (int d = 0; d < HEAD_DIM; ++d) {
        acc[d] = acc[d] * alpha + beta * float(V[kv_base + d]);
    }
    l = l * alpha + beta;
    m = m_new;
}

const float inv_l = 1.0f / max(l, 1.0e-20f);
for (int d = 0; d < HEAD_DIM; ++d) {
    O[out_base + d] = ELEM_TYPE(acc[d] * inv_l);
}
