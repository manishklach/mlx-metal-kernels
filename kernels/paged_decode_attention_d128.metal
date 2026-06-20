uint row_id = thread_position_in_grid.x;

const int B = meta[0];
const int NUM_PAGES = meta[1];
const int PAGE_SIZE = meta[2];
const int H = meta[3];
const int D = meta[4];
const int MAX_BLOCKS = meta[5];

if (D != HEAD_DIM) {
    return;
}

const uint total_rows = uint(B * H);
if (row_id >= total_rows) {
    return;
}

const int h = int(row_id % uint(H));
const int b = int(row_id / uint(H));
const int valid_len = lengths[b];
const int q_base = (b * H + h) * D;
const float scale_val = scale[0];

float qv[HEAD_DIM];
float acc[HEAD_DIM];
for (int d = 0; d < HEAD_DIM; ++d) {
    qv[d] = float(q[q_base + d]);
    acc[d] = 0.0f;
}

float m = -INFINITY;
float l = 0.0f;
for (int j = 0; j < valid_len; ++j) {
    const int block_id = j / PAGE_SIZE;
    const int offset = j % PAGE_SIZE;
    const int page_id = block_table[b * MAX_BLOCKS + block_id];
    const int kv_base = ((page_id * PAGE_SIZE + offset) * H + h) * D;
    float dot = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
        dot += qv[d] * float(K_pages[kv_base + d]);
    }
    const float score = dot * scale_val;
    const float m_new = max(m, score);
    const float alpha = isfinite(m) ? exp(m - m_new) : 0.0f;
    const float beta = exp(score - m_new);
    for (int d = 0; d < HEAD_DIM; ++d) {
        acc[d] = acc[d] * alpha + beta * float(V_pages[kv_base + d]);
    }
    l = l * alpha + beta;
    m = m_new;
}

const float inv_l = 1.0f / max(l, 1.0e-20f);
for (int d = 0; d < HEAD_DIM; ++d) {
    out[q_base + d] = ELEM_TYPE(acc[d] * inv_l);
}
